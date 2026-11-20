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

# conftest.py
import gc
import os
import sys
import time

# --- JAX persistent compilation cache: one directory PER XDIST WORKER --------
# MUST run before ``import jax`` below: JAX reads JAX_COMPILATION_CACHE_DIR when
# the backend initializes, so a later assignment is ignored.
#
# The CI host exports a single JAX_COMPILATION_CACHE_DIR per runner instance
# (e.g. .../<runner>/_work/_temp/jax-cache). That directory is isolated
# BETWEEN jobs -- each job gets its own runner -- but every xdist worker inside
# one job shares it, and 8 workers writing the same cache entry can abort the
# interpreter mid-write:
#
#   Fatal Python error: Aborted
#     jax/_src/compilation_cache.py in put_executable...
#     jax/_src/compiler.py in _cache_write
#
# when two workers write the same cache directory (a race, not reproducible
# on rerun). This mirrors the per-job venv isolation a few lines below in the
# workflow: same shared-directory problem, same fix, applied at the granularity
# that actually collides.
#
# Deterministic (the worker id, not a random name) so each worker keeps reusing
# its own cache instead of repopulating a fresh directory every run. No-op when
# the host sets no cache dir, or when running without xdist -- local runs are
# unaffected.
_cache_base = os.environ.get("JAX_COMPILATION_CACHE_DIR")
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _cache_base and _xdist_worker:
    os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(_cache_base, _xdist_worker)


# --- NaN is never agreement -----------------------------------------------------
# Installed before jax/jqmc and before pytest imports any test module, so that
# ``from numpy.testing import assert_allclose`` in a test module binds the guard.
# See tests/_nan_guard.py; pytest_collection_finish below rebinds any stale copy.
from tests import _nan_guard

_nan_guard.install()

import jax

# --- jaxlib 0.9.1/0.9.2: a persistent-cache HIT can overflow a worker stack
# Runs right after ``import jax`` and before anything is compiled: jax opens the
# persistent cache lazily at the first compile, so ``jax.config.update`` here is in
# time for this process, and the environment variable carries the switch to every
# child process (xdist workers, CLI subprocesses) before their own ``import jax``.
#
# On these jaxlib releases ``PyClient::DeserializeExecutable`` hands the load
# of a cached CPU executable to a tsl thread-pool worker with a 512 KiB stack.
# ``CpuAotLoader::LoadAotCompilationResult`` -> ``ObjectLoader::LookupSymbols``
# then links the cached object files through ORC, which materializes each
# pending object file recursively inside the previous one's link (about eight
# frames per object file).  A large executable -- the curated analytic-vs-
# autograd ``e_L`` cases on the dense-AGP + MPC composite fixture -- recurses
# about 1980 frames deep and runs into the guard page:
#
#   Thread "python" received signal SIGSEGV
#   #0    llvm::object::ELFFile<...>::sections() const
#   ...   [RTDyldObjectLinkingLayer::emit -> jitLinkForORC -> lookup ->
#          dispatchOutstandingMUs -> materialize -> emit] x ~245
#   #1965 xla::cpu::ObjectLoader::LookupSymbols
#   #1968 xla::cpu::CpuAotLoader::LoadAotCompilationResult
#   #1972 xla::ifrt::PjRtCompiler::DeserializeLoadedExecutable
#   #1975 Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop
#   (fault address = rsp - 8, 521904 bytes below the thread's stack top)
#
# The overflow happens on a thread without faulthandler's alternate signal
# stack, so the worker dies with no Python traceback ("node down: Not properly
# terminated" under xdist).  The crash needs a cache HIT, i.e. the same large
# executable compiled earlier in the same worker; the first compile and a run
# without the persistent cache both pass.  Measured with the same test and a
# shared cache directory: jaxlib 0.9.1 and 0.9.2 crash on the hit; 0.7.2,
# 0.8.0, 0.9.0, 0.10.0, 0.10.1 and 0.10.2 load the same entries and pass.
# The persistent cache is only a compile-time saving, so it is switched off for
# the affected releases; every test still runs unchanged.
#
# The affected-release table is jqmc's own (``jqmc._setting``, which also drives
# the user-facing runtime warning): one source of truth.
from jqmc._setting import installed_jaxlib_version, jaxlib_persistent_cache_load_overflows

_jaxlib_version = installed_jaxlib_version()
if jaxlib_persistent_cache_load_overflows(_jaxlib_version):
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    jax.config.update("jax_enable_compilation_cache", False)

import numpy as np
import pytest

from jqmc.structure import Structure_data


@pytest.fixture(scope="session", autouse=True)
def _jax_distributed_init():
    """Initialize ``jax.distributed`` once per pytest session under multi-rank MPI.

    The device branch of ``run_optimize`` (``use_device_optimizer=True``)
    relies on ``jax.lax.psum`` / ``jax.lax.all_gather`` to aggregate across
    MPI ranks. Without ``jax.distributed.initialize`` each rank's JAX sees
    only its own local devices (``jax.process_count() == 1``) and the
    collectives degenerate to no-ops, silently producing wrong results.

    Mirrors the production CLI setup in ``jqmc_cli.py``: strip HTTP proxy
    environment variables before calling ``initialize`` so the JAX gRPC
    coordination service doesn't try to route the local-host connection
    through a proxy (which is the typical cause of "hangs forever" on
    macOS / corporate networks).

    The failure mode this fixture exists to prevent is EXACTLY the one the
    docstring above names -- collectives degenerating to no-ops and *silently
    producing wrong results* -- so ``initialize`` failures are not swallowed: a
    broken init would otherwise make the downstream multi-rank tests see
    ``jax.process_count() < 2`` and skip, and the multi-rank verification would
    vanish without a single red mark.

    So: the only benign case ("already initialized in THIS process") is detected
    up front and short-circuits the call; every other failure propagates. A hard
    post-condition then asserts JAX actually sees all the MPI ranks, which is the
    property the collectives depend on (init can also "succeed" while auto-detection
    picks up the wrong topology).
    """
    import os

    from mpi4py import MPI

    global _MPI_HARNESS_COMM

    size = MPI.COMM_WORLD.Get_size()
    if size > 1:
        # Duplicate COMM_WORLD up front, while every rank is still in step. The
        # duplicate carries the harness's own boundary rendezvous, which must
        # not alias the drivers' collectives (see _mpi_boundary_rendezvous).
        # Creating it lazily would be a collective posted from inside the very
        # deadlock it is meant to detect.
        _MPI_HARNESS_COMM = MPI.COMM_WORLD.Dup()
        # Same proxy-strip workaround as jqmc_cli.py.
        for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(_proxy_var, None)
        from jax._src import distributed as _jax_distributed

        from jqmc._driver_helpers import _initialize_jax_distributed

        # The one benign case: a previous session/fixture in the SAME process already
        # brought the coordination client up. Re-calling would raise "should only be
        # called once", which is not a real failure -- but it is the ONLY exception we
        # are willing to absorb, and we absorb it by not making the call at all.
        if getattr(_jax_distributed.global_state, "client", None) is None:
            _initialize_jax_distributed()

        n_proc = jax.process_count()
        if n_proc != size:
            raise RuntimeError(
                f"jax.distributed did not pick up the MPI world: jax.process_count()={n_proc} "
                f"but MPI.COMM_WORLD.Get_size()={size}. jax.lax.psum / all_gather would "
                "degenerate to per-rank no-ops and silently produce wrong results, and the "
                "multi-rank agreement tests would report nothing. Do NOT downgrade this to a "
                "skip -- fix the init (proxy env vars, "
                "network sandboxing, mpi4py/JAX version skew)."
            )
    try:
        yield
    finally:
        if size > 1:
            # Teardown is an ordered collective protocol, not two unrelated
            # Python atexit callbacks.  First keep every rank inside the live
            # MPI world until all pytest work has ended; then stop the JAX
            # coordination clients, synchronize once more, and explicitly
            # finalize MPI.  Relying on mpi4py's automatic finalizer still
            # allowed rank 0 to leave without MPI_Finalize after all tests had
            # passed (Open MPI reports "exiting improperly").
            MPI.COMM_WORLD.Barrier()
            jax.distributed.shutdown()
            MPI.COMM_WORLD.Barrier()
            if _MPI_HARNESS_COMM is not None:
                _MPI_HARNESS_COMM.Free()
                _MPI_HARNESS_COMM = None
            if not MPI.Is_finalized():
                MPI.Finalize()


def pytest_addoption(parser):
    """Add options for pytests."""
    parser.addoption("--disable-jit", action="store_true", default=False, help="Disable jax.jit for pytests")
    parser.addoption("--skip-heavy", action="store_true", default=False, help="Skip heavy calculations for pytests")
    parser.addoption(
        "--precision-mode",
        default="full",
        choices=["full", "mixed"],
        help="Precision mode for tests (default: full)",
    )
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Fail the session when any selected test is skipped or xfailed (CI lanes): a test that did not run "
        "is red. A skip_mixed_precision test is deselected in mixed precision instead of skipped.",
    )


#: Node ids of tests that were skipped or xfailed in this session (``--fail-on-skip``).
_NOT_RUN: list[str] = []


def pytest_runtest_logreport(report):
    """Record every test that did not run: a skip in any phase, or an xfail."""
    if report.skipped:
        _NOT_RUN.append(report.nodeid)


def pytest_collectreport(report):
    """Record a collection-time skip (``pytest.skip(allow_module_level=True)``, ``importorskip``).

    Its tests never become items, so no runtest report exists for them. Under
    xdist every worker collects, and the controller receives each skipped
    collect report through this same hook, so serial and xdist sessions agree.
    """
    if report.skipped:
        _NOT_RUN.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    """``--fail-on-skip``: any skipped or xfailed test fails the session."""
    if not session.config.getoption("--fail-on-skip") or not _NOT_RUN:
        return
    listing = "\n".join(f"  - {nodeid}" for nodeid in sorted(set(_NOT_RUN))[:50])
    print(
        f"\nERROR: --fail-on-skip: {len(set(_NOT_RUN))} selected test(s) or module(s) did not run (skipped or xfailed):\n{listing}",
        file=sys.stderr,
        flush=True,
    )
    if session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def _strict_complex_cast_guard():
    """Systematic guard: promote ``ComplexWarning``
    to a test failure session-wide so any silent
    ``jnp.asarray(complex_value, dtype=real_dtype)`` cast on the complex
    Bloch path fails at the cast site, not as a downstream energy
    discrepancy.

    Why session-wide (autouse): real-path tests never trigger
    ``ComplexWarning`` (no complex arrays anywhere), so the guard is
    silent on them; complex-path tests (``lithium_k025``, ``complex_pgto``,
    ...) trip on any dtype hardcast that drops ``Im``. We get near-zero-
    cost detection coverage from EVERY complex-touching test in the
    suite, not only from dedicated tests.

    Why this guard exists at all: a Wirtinger ``grad(|c|^2) = 2 conj(c)``
    test plus an ``|Im<E_L>|`` runtime warning is INSUFFICIENT, because a
    kinetic-chain hardcast produces
    ``Im e_L = 0`` walker-by-walker (the ``Im`` was dropped at the cast),
    so the ``|Im<E_L>|`` guard reports ``OK`` while the energy
    chain is silently wrong. Promoting ``ComplexWarning`` to error
    catches the cast itself.

    What the trap catches under the pinned JAX 0.10.2 / numpy 2.x: a
    complex -> real cast raises ``ComplexWarning`` through
    ``jnp.asarray(z, dtype=real)`` / ``jnp.array(z, dtype=real)`` (eager
    and inside ``jax.jit``), ``lax.convert_element_type(z, real)`` (eager
    and jit), ``.at[...].set(z)`` into a real array, ``jnp.float64(z)``,
    and every numpy cast (``np.asarray(z, dtype=real)``,
    ``ndarray.astype(real)``, ``float(z)``). These warn whenever the source
    dtype is complex, including ``Im == 0``.

    Known limits: (i) ``jax.Array.astype(real)`` on a complex array (eager
    and jit) emits NO warning, so ``z.astype(real)`` dropping ``Im`` is not
    caught. (ii) The transpose (VJP) of a real -> complex promotion, i.e.
    the real projection of a complex cotangent onto a real primal, emits no
    warning. (iii) An explicit ``jnp.real(z)`` is silent by design, so an
    unintended ``jnp.real`` is not caught either. (iv) Covers only code
    paths exercised by the test; new paths add coverage automatically.
    Real-projection defects that keep the complex dtype are therefore
    caught only by value tests (e.g. the independent twist value oracle).
    """
    import warnings as _warnings

    from numpy.exceptions import ComplexWarning as _ComplexWarning

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", _ComplexWarning)
        yield


@pytest.fixture(autouse=True)
def enable_jit(request):
    """Fixture to enable/disable jax.jit for pytests."""
    if request.config.getoption("--disable-jit"):
        # Disable jax.jit (jax_disable_jit=True means JIT is disabled)
        jax.config.update("jax_disable_jit", True)
    else:
        # Enable jax.jit (default)
        jax.config.update("jax_disable_jit", False)
    yield
    # Reset to default after tests
    jax.config.update("jax_disable_jit", False)


@pytest.fixture(scope="module", autouse=True)
def _initialize_module_execution_context(request):
    """Establish the requested lane before module-scoped model fixtures run."""
    from jqmc._precision import configure

    configure(request.config.getoption("--precision-mode"))
    node_path = getattr(request.node, "path", None)
    if node_path is not None and "test_rs_kernel" in node_path.parts:
        yield
        return
    with pytest.MonkeyPatch.context() as context:
        context.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
        yield


@pytest.fixture(autouse=True)
def configure_precision(request):
    """Configure precision mode before each test."""
    from jqmc._precision import configure

    mode = request.config.getoption("--precision-mode")
    configure(mode)


@pytest.fixture(autouse=True)
def _force_xla_backend_for_non_rust_tests(request, monkeypatch):
    """Keep ordinary tests on XLA, independent of host/backend defaults.

    ``flag_rust_on_CPU`` defaults to true, so a full-precision CPU test can
    otherwise import whichever ``jqmc_rs`` binary happens to be installed and
    silently exercise Rust instead of XLA.  All tests involving Rust belong in
    the dedicated ``test_rs_kernel`` suite, which is the only opt-out.
    """
    node_path = getattr(request.node, "path", None)
    is_rust_suite = node_path is not None and "test_rs_kernel" in node_path.parts
    if is_rust_suite:
        yield
        return

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    yield


def _read_max_map_count() -> int | None:
    """The kernel's per-process VMA ceiling, or ``None`` off Linux."""
    try:
        with open("/proc/sys/vm/max_map_count") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _count_vmas() -> int | None:
    """This process's current VMA (``/proc/self/maps`` line) count."""
    try:
        with open("/proc/self/maps") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def _vma_reclaim_needed(current: int, worst_test_cost: int, limit: int) -> bool:
    """Should the next test start from a cleared JAX executable cache?

    Two clauses, neither of which carries a tuned constant:

    (a) *history* -- a repeat of the most expensive test measured so far would
        no longer fit.

    (b) *equal room* -- the process is holding more than it could still
        acquire (``current > limit - current``). History alone is not enough:
        one test can cost 1.5x the peak cost of every test before it (32655
        against 21969 VMAs measured in ``test_twist_decomposition_mpi.py``),
        so clause (a) stays quiet at the boundary that matters. Clause (b)
        makes no assumption about the next test at all -- it simply
        guarantees every test starts with at least half the table free.

    What neither clause can do is rescue a *single* test that needs more than
    half the table on its own (32655 of 65530 is 49.8%). Such a test releases
    its own executables at a phase boundary IN THE TEST: the two GFMC rows of
    ``tests/test_jqmc_kernel/test_twist_decomposition_mpi.py`` release the
    standalone chain's executables once the comparison data is NumPy, because
    the driver phase rebuilds every ``run()``-local jit from scratch anyway
    and so shares nothing with them (see the phase-boundary note there); the
    worst row costs ~17k (~26%). ``tests/test_jqmc_kernel/test_jax_cache_isolation.py``
    pins the predicate at the 32655 / 21969 numbers the clauses are designed
    against.
    """
    return (worst_test_cost > 0 and current + worst_test_cost > limit) or 2 * current > limit


_VMA_LIMIT = _read_max_map_count()
_VMA_WORST_TEST_COST = 0


# --- Multi-rank deadlock backstop ------------------------------------------
#
# A rank-local failure inside a test that also calls MPI collectives strands
# the other ranks in a collective the failing rank will never reach. Nothing
# crashes and nothing is reported -- the job just stops making progress until
# its wall-clock cap, with ZERO reported failures: one marginal tolerance miss
# inside an ``if mpi_rank == 0:`` block leaves rank 1 waiting in ``get_E``'s
# ``allreduce``.
#
# ``tests/_mpi_collective_check.collective_check`` removes the *known* sites of
# that divergence by making the failure collective. This is the backstop for
# the ones nobody anticipated: at every test boundary the ranks rendezvous, and
# a rank that does not arrive within the deadline gets the whole world
# ``MPI_Abort``ed with a diagnostic naming the test. A job that dies in minutes
# pointing at the offending test is strictly better than one that burns the CI
# cap and reports nothing.
#
# The deadline is per-test wall clock, not per-session: the ranks run the SAME
# test, so their only legitimate skew is compile-cache asymmetry (one rank hits
# the persistent JAX cache, the other compiles). The heaviest single test in
# this repo compiles for a few minutes, so 30 minutes is ~10x headroom and
# still 16x tighter than the CI cap it replaces. Override with
# ``JQMC_MPI_BOUNDARY_TIMEOUT_S`` (0 disables the abort and restores the plain
# blocking rendezvous).
MPI_BOUNDARY_TIMEOUT_S = float(os.environ.get("JQMC_MPI_BOUNDARY_TIMEOUT_S", 1800.0))
MPI_BOUNDARY_POLL_S = 0.05

# A PRIVATE duplicate of COMM_WORLD for the harness's own collectives.
#
# Mandatory, not hygiene: MPI matches collectives positionally per
# communicator. If the harness posted its rendezvous on COMM_WORLD while a
# stuck rank sat in the driver's ``allreduce`` on COMM_WORLD, the two would be
# matched against each other -- erroneous by the standard, and in practice it
# would either "succeed" with garbage or hang anyway, i.e. the watchdog could
# not observe the very condition it exists to detect. On a duplicate the
# harness's traffic can never alias the drivers'.
_MPI_HARNESS_COMM = None


def _mpi_boundary_rendezvous(label: str, capture_manager=None) -> None:
    """Synchronise all ranks at a test boundary, or ``MPI_Abort`` the world.

    Args:
        label: Node id (or other identifier) of the test just finished,
            reported in the abort diagnostic.
        capture_manager: pytest's ``capturemanager`` plugin, if available.
            Its global capture is suspended before the diagnostic is written:
            ``MPI_Abort`` kills the process before pytest ever gets to replay
            captured output, so a diagnostic left inside the capture buffer is
            a diagnostic nobody reads -- which is the exact failure mode this
            watchdog exists to end.
    """
    comm = _MPI_HARNESS_COMM
    if comm is None:
        return

    barrier = comm.Ibarrier()
    if MPI_BOUNDARY_TIMEOUT_S <= 0:
        barrier.Wait()
        return

    deadline = time.monotonic() + MPI_BOUNDARY_TIMEOUT_S
    while not barrier.Test():
        if time.monotonic() >= deadline:
            from mpi4py import MPI

            if capture_manager is not None:
                try:
                    capture_manager.suspend_global_capture(in_=False)
                except Exception:  # pragma: no cover - diagnostics must never mask the abort
                    pass
            sys.stderr.write(
                f"\n[jqmc-mpi-watchdog] rank {comm.Get_rank()} of {comm.Get_size()} waited "
                f"{MPI_BOUNDARY_TIMEOUT_S:.0f}s at the test boundary after {label!r} and at "
                "least one other rank never arrived.\n"
                "[jqmc-mpi-watchdog] That means the ranks diverged INSIDE that test: a "
                "rank-local raise (assert / skip / exception) let this rank leave while the "
                "others are still blocked in an MPI collective (get_E / get_aF / branching).\n"
                "[jqmc-mpi-watchdog] Guard the rank-divergent block with "
                "tests/_mpi_collective_check.collective_check so the failure is raised on "
                "every rank. Aborting the world now instead of hanging until the CI cap.\n"
            )
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
        time.sleep(MPI_BOUNDARY_POLL_S)


@pytest.fixture(autouse=True)
def _bound_xla_executable_address_space(request):
    """Keep the process below ``vm.max_map_count`` by dropping XLA executables.

    Every XLA-CPU executable this process ever compiles (or deserializes from
    the persistent compilation cache) is kept alive forever by JAX's internal
    compiled-function caches, and each *kernel* inside it costs three VMAs --
    an ``r-xp`` code page, an ``r--p`` rodata page and an ``rw-p`` data page
    carved out of LLVM's section allocator.  Measured on this venv (jax
    0.7.2): a 12-kernel graph costs 36 VMAs, and one of the LRDMC
    force+stress graphs costs several thousand.  Nothing ever gives them
    back, so a long single-process session walks monotonically toward the
    kernel's ``vm.max_map_count`` ceiling (65530, the Linux default).

    At the ceiling every ``mmap`` -- and therefore every ``malloc`` that needs
    fresh address space -- fails.  XLA notices and returns a non-OK status, so
    ``xla::ValueOrThrow`` throws...  and libgcc's unwinder itself needs to
    allocate in ``_Unwind_Find_FDE``, gets ``NULL``, and dereferences it.  The
    interpreter dies with

        Fatal Python error: Segmentation fault
          jax/_src/compilation_cache.py:238 in get_executable_and_time
          jax/_src/compiler.py:794 in _cache_read

    which is the crash of a long MPI session of
    ``test_twist_decomposition_mpi`` in mixed precision: the *cache read*
    frame is where the process
    happened to run out of address space, not a corrupt cache entry.  The same
    exhaustion inside ``_compile_and_write_cache`` instead reports the clean
    ``LLVM ERROR: Unable to allocate section memory! / Fatal Python error:
    Aborted``. A single process was also observed to SIGSEGV after about 313
    tests; xdist hid the same memory ceiling by spreading the tests over
    twelve short-lived processes.  Sessions with no xdist to hide behind (the
    MPI runs) therefore die, and they die sooner the bigger the graphs get.

    ``jax.clear_caches()`` releases the executables and the allocator
    ``munmap``s their sections: measured 7222 VMAs accumulated over 200
    compiles, 7183 returned by one call, with live Python references to the
    jitted callables still held.  So the leak is bounded here, at the test
    boundary, by reclaiming whenever the remaining headroom has shrunk below
    the worst per-test cost observed so far.  "Cost" is the NET growth
    between the two boundary samples (count after the test minus count before
    it), not an intra-test high water.

    Net cost is enough because clause (b) of ``_vma_reclaim_needed`` does not
    depend on it: every test starts with at least half of ``vm.max_map_count``
    free, whatever the history.  A test can peak far above where it settles
    (measured on the GFMC twist test: 3126 -> 18358 -> 10077, peak cost 15232
    vs net cost 6951), but the most expensive test in the suite now peaks at
    ~17k of 65530 VMAs (~26%; see ``_vma_reclaim_needed``), comfortably inside
    the guaranteed half.  The residual risk is a single future test whose own
    peak exceeds half the table; neither the boundary samples nor an
    intra-test sampler could rescue that one, because the reclaim only ever
    runs between tests.

    There is deliberately no intra-test sampling thread.  A sampler running
    as a ``threading.Thread`` is traced by coverage.py under ``pytest --cov``.  When a coverage-traced second
    thread meets a new file after ``sys.path`` changed, coverage re-matches
    its paths inside ``coverage.misc.sys_modules_saved()``.  That context
    manager deletes every module imported meanwhile, by any thread, including
    the main thread's in-flight lazy ``jaxlib.mlir.dialects.mhlo`` /
    ``_mhlo_ops_gen`` import at a worker's first lowering on jaxlib 0.7.x.
    The failure signature is ``KeyError: 'jaxlib.mlir.dialects._mhlo_ops_gen'``,
    then ``Dialect namespace 'mhlo' is already registered``.
    The harness therefore runs no Python thread of its own.

    Under MPI the decision must be unanimous: some jitted callables are
    multi-process collectives (that is what ``_jax_distributed_init`` above
    exists for), so a rank that recompiles alone would block in a compilation
    the other ranks never enter.  The predicate is therefore OR-reduced before
    anybody clears.  That reduction assumes every rank executes the same test
    list, which holds for an ``mpirun -np 2 pytest`` session run with neither
    ``-x`` nor ``--maxfail`` (a rank-local failure then does not truncate that
    rank's session).

    This fixture is also the harness's per-test MPI rendezvous point: the
    ``_mpi_boundary_rendezvous`` call in the ``finally`` block is what turns a
    mid-test rank divergence into a prompt ``MPI_Abort`` with a diagnostic
    instead of a silent hang until the CI wall-clock cap.  Both collectives run
    on the PRIVATE duplicate of ``COMM_WORLD`` (``_MPI_HARNESS_COMM``) so the
    harness's traffic can never be matched against a driver's ``allreduce``.
    """
    global _VMA_WORST_TEST_COST

    before = _count_vmas() if _VMA_LIMIT is not None else None
    try:
        yield
    finally:
        # Deadlock backstop FIRST, and on the private harness communicator: it
        # is the one collective guaranteed to run at every test boundary, so it
        # is where a rank that never left the test body is detected. The
        # ``reclaim`` reduction below is itself a collective and would hang
        # forever if a rank were still stuck -- rendezvous before it, never
        # after.
        _mpi_boundary_rendezvous(
            request.node.nodeid,
            request.config.pluginmanager.getplugin("capturemanager"),
        )

        after = _count_vmas() if before is not None else None
        if after is not None:
            # Net cost from the two boundary samples (see the docstring for why it suffices).
            _VMA_WORST_TEST_COST = max(_VMA_WORST_TEST_COST, after - before)
            reclaim = _vma_reclaim_needed(after, _VMA_WORST_TEST_COST, _VMA_LIMIT)

            from mpi4py import MPI

            if _MPI_HARNESS_COMM is not None and not MPI.Is_finalized():
                reclaim = _MPI_HARNESS_COMM.allreduce(reclaim, op=MPI.LOR)

            if reclaim:
                jax.clear_caches()
                gc.collect()


@pytest.fixture(autouse=True)
def _isolate_cwd(request, tmp_path):
    """Run every test from a private temporary working directory.

    Several production routines write side-effect files to the *current
    working directory* with **fixed** names:

    - ``MCMC.run_optimize`` -> ``hamiltonian_data_opt_step_*.h5``
    - ``MCMC.run`` / ``GFMC_*.run`` -> ``external_control_*.toml`` stop flags
    - restart checkpoints (``restart.chk`` etc.)

    Under ``pytest-xdist`` (the CI runs ``-n 8``) parametrized variants and
    sibling workers execute concurrently in the *same* repo directory, so
    they race on those fixed filenames. h5py then raises
    ``BlockingIOError: [Errno 11] Resource temporarily unavailable`` (file
    lock) and the TOML stop-flag poller can hit ``FileNotFoundError``.

    Isolating the CWD per test removes the whole class of races at one
    stroke, without polluting production code with PID/MPI-aware filenames.
    Test inputs are read via ``os.path.dirname(__file__)`` absolute paths, so
    moving the CWD does not affect input discovery.

    Every test must declare its CWD policy with exactly one of
    ``@pytest.mark.with_chdir`` (autouse runs; the common case) or
    ``@pytest.mark.without_chdir`` (test manages its own working
    directory, e.g. ``test_integration_cli.py`` which juggles ``os.chdir``
    between the repo root and ``tmp_path`` itself). The mandatory-XOR
    enforcement lives in ``pytest_collection_modifyitems``; this fixture
    simply honours ``without_chdir`` as a skip signal.
    """
    if request.node.get_closest_marker("without_chdir") is not None:
        yield
        return
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def pytest_collection_finish(session):
    """Re-assert the NaN-agreement guard after every test module is imported.

    A test module or helper must not undo the guard: the public names are put back
    and any module global that still holds an unguarded original (a
    ``from numpy.testing import ...`` executed before ``install()``) is rebound.
    """
    _nan_guard.install()
    _nan_guard.rebind_stale_references()


def pytest_itemcollected(item):
    """Show reason for obsolete tests."""
    obsolete_marker = item.get_closest_marker("obsolete")
    if obsolete_marker:
        reason = obsolete_marker.kwargs.get("reasons", "")
        item._nodeid += f" [OBSOLETE: {reason}]"


# Custom marker for conditional skip
def pytest_configure(config):
    """Pytest configuration."""
    from pathlib import Path

    from gdd.engine import derivative_registry
    from gdd.engine.derived_cache import prepare_derived
    from gdd.engine.execution_records import register

    from tests import _test_data_cache

    # The QE example tree is read through a per-process mirror whose
    # pseudopotentials come from the host-local cache (tests/_test_data_cache.py).
    # Built here, before any test module computes its fixture paths at import
    # time; every xdist worker and MPI rank builds its own.
    mirror = _test_data_cache.create_session_mirror()
    config.add_cleanup(lambda: _test_data_cache.remove_session_mirror(mirror))

    prepare_derived(Path(__file__).resolve().parents[1])
    register(config)
    # Issue 789: every case's emitted derivative receipts must equal its registered keys.
    derivative_registry.register(config)
    config.pluginmanager.register(_TestShardPlugin(), "jqmc-test-shard")
    config.addinivalue_line("markers", "activate_if_disable_jit: activate test if --disable-jit is set")
    config.addinivalue_line("markers", "heavy: skip test if --skip-heavy is set")
    config.addinivalue_line(
        "markers",
        "mpi_only: file-level marker for tests whose fixtures require an MPI world; "
        "run such a file under mpirun, not in a serial session.",
    )
    config.addinivalue_line(
        "markers",
        "light: explicit marker stating the test is fast enough for the "
        "default CI tier (under the 60s/test budget under JIT warm-up). Every "
        "collected item MUST carry EXACTLY one of ``light`` / ``heavy`` "
        "(see ``pytest_collection_modifyitems``); apply it explicitly per test "
        "or per ``pytest.param`` row -- there is no module-level default tier.",
    )
    config.addinivalue_line("markers", "obsolete: tests that are obsolete and should be removed in the future")
    config.addinivalue_line(
        "markers",
        "with_chdir: declare that the test runs under the autouse ``_isolate_cwd`` "
        "fixture (chdir into a private ``tmp_path``). The common case; usually applied "
        "via a module-level ``pytestmark`` list.",
    )
    config.addinivalue_line(
        "markers",
        "without_chdir: opt out of the autouse per-test CWD isolation (test manages its own "
        "working directory). Every collected item MUST carry either ``with_chdir`` or "
        "``without_chdir`` (mandatory XOR, enforced in ``pytest_collection_modifyitems``).",
    )
    for name, (help_text, _) in _SESSION_MARKERS.items():
        # Outside its session a marked test is deselected under --fail-on-skip
        # and skipped otherwise (pytest_collection_modifyitems).
        config.addinivalue_line(
            "markers",
            f"{help_text} Outside that session it is deselected under --fail-on-skip and skipped otherwise; "
            "the reason kwarg is required and is the skip message.",
        )


#: Environment variables that select one execution shard of the collected test
#: set. Generic on purpose -- ANY lane may shard, so the names are NOT tied to
#: one test matrix.
SHARD_INDEX_ENV = "JQMC_TEST_SHARD_INDEX"
SHARD_COUNT_ENV = "JQMC_TEST_SHARD_COUNT"
#: Optional split strategy for a sharded session: empty (the default) keeps the
#: stride over sorted node ids; a lane key of ``scripts/ci_shard_durations.json``
#: selects the duration-weighted LPT partition over that lane's measured rows
#: (``scripts/ci_shard_partition.py``).
SHARD_WEIGHTS_ENV = "JQMC_TEST_SHARD_WEIGHTS"


def _shard_usage_error(message: str) -> pytest.UsageError:
    """Print the diagnosis to stderr, then build the ``UsageError`` to raise.

    The print is NOT redundant. Under ``pytest-xdist`` this hook runs inside the
    WORKER processes, and a ``UsageError`` raised there kills the worker during
    collection; the controller then reports a bare
    ``INTERNALERROR ... assert not crashitem`` with the exception message
    nowhere in the log. A plain ``sys.stderr`` write from the worker DOES reach
    the job log (verified with xdist 3.8.0), so the operator gets the reason in
    every execution mode -- serial, xdist and ``mpirun`` -- while the raise still
    makes the run red.
    """
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    return pytest.UsageError(message)


def _read_shard_selection() -> tuple[int, int] | None:
    """Parse ``(index, count)`` from the environment, or ``None`` for "no shard".

    Safety contract:

    * COUNT unset / empty / ``1`` -> ``None``: the session behaves EXACTLY as it
      did before sharding existed (no deselection at all). This is the local and
      the un-sharded-CI path, so an operator who never sets the variables can
      never be silently running a fraction of the suite.
    * Anything else that does not parse as ``0 <= index < count`` with
      ``count >= 1`` is a ``UsageError`` (pytest exit code 4), not a
      best-effort fallback: a typo'd shard spec must not quietly run the whole
      file N times (the duplicated-work failure mode the empty
      ``HEAVY_FILE_SHARD_COUNTS`` dict would have produced) nor quietly run
      nothing.
    """
    raw_count = os.environ.get(SHARD_COUNT_ENV, "").strip()
    raw_index = os.environ.get(SHARD_INDEX_ENV, "").strip()

    if not raw_count:
        # An index without a count is a half-configured lane -- loud, because the
        # operator clearly *meant* to shard and would otherwise get a full run
        # under a job name that claims to be one shard of several.
        if raw_index not in ("", "0"):
            raise _shard_usage_error(
                f"{SHARD_INDEX_ENV}={raw_index!r} is set but {SHARD_COUNT_ENV} is not: "
                "the shard spec is incomplete, so this session would silently run the "
                "FULL collection under a shard-labelled job. Set both or neither."
            )
        return None

    try:
        count = int(raw_count)
    except ValueError:
        raise _shard_usage_error(f"{SHARD_COUNT_ENV}={raw_count!r} is not an integer") from None
    if count < 1:
        raise _shard_usage_error(f"{SHARD_COUNT_ENV}={count} must be >= 1")
    if count == 1:
        if raw_index not in ("", "0"):
            raise _shard_usage_error(f"{SHARD_COUNT_ENV}=1 requires {SHARD_INDEX_ENV}=0, got {raw_index!r}")
        return None

    try:
        index = int(raw_index or "0")
    except ValueError:
        raise _shard_usage_error(f"{SHARD_INDEX_ENV}={raw_index!r} is not an integer") from None
    if not 0 <= index < count:
        raise _shard_usage_error(f"{SHARD_INDEX_ENV}={index} is out of range for {SHARD_COUNT_ENV}={count}")
    return index, count


class _TestShardPlugin:
    """Applies the execution shard AFTER every other collection hook.

    It is a separate plugin, not another branch of this module's
    ``pytest_collection_modifyitems``, because the two must run at OPPOSITE
    ends of the hook chain:

    * The classification gates in ``pytest_collection_modifyitems`` below must
      see the collection BEFORE pytest's own ``-m`` deselection, so a
      both-marked or unclassified item fails the gate no matter which lane's
      marker expression is in force.
    * The shard split must run AFTER that deselection. Splitting first and
      filtering second is not sharding at all: shard ``k`` would be
      ``sorted(ALL items)[k::n] AND -m <expr>``, whose sizes follow how the
      lane's marked items happen to fall across the stride classes. Measured
      on ``test_gfmc_force_analytic``'s 64 heavy items over 5 shards that gave
      9/12/16/16/11 instead of 13/13/13/13/12, and on ``test_jqmc_gfmc``'s 13
      ``-m "not heavy"`` items over 9 shards it produced an EMPTY shard.
      ``trylast`` puts this after ``_pytest.mark``'s deselection (and after any
      reordering plugin), so it splits exactly the set the lane will run.
    """

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        _apply_test_shard(config, items)


def _apply_test_shard(config, items) -> None:
    """Deselect every item that does not belong to this execution shard.

    Split strategy: **round-robin over the node ids sorted lexicographically**,
    i.e. shard ``k`` of ``n`` keeps ``sorted(items)[k::n]``.

    * *Sorted* (not "as collected"): the split must be a pure function of the
      collected SET, never of its order. pytest-xdist collects independently in
      every worker, the two ``mpirun -np 2`` ranks collect independently, and a
      reordering plugin (``pytest-randomly``, ``-p no:randomly`` forgotten) may
      shuffle ``items`` -- all of those must agree on the same shard or the job
      runs a different subset per worker/rank and the union stops covering the
      file. Sorting makes them agree by construction.
    * *Round-robin*, not contiguous blocks: cost is strongly correlated with
      neighbouring node ids, because adjacent ids are usually ``pytest.param``
      rows of the SAME test function and the expensive rows cluster by fixture
      (all ``lithium*`` rows, all ``*_k025`` rows). A contiguous block split
      would put a whole expensive cluster in one shard and leave another shard
      nearly empty -- exactly the imbalance sharding exists to remove. A stride
      spreads any such run evenly. It costs nothing in determinism: index
      ``i % count`` needs no cost model and no state.

    ``items`` is filtered in place but NOT reordered: the surviving items keep
    their original relative order, so any collection-order-sensitive behaviour
    is unchanged.

    **Duration-weighted variant** (``JQMC_TEST_SHARD_WEIGHTS=<lane>``):
    the stride cannot keep two ~2-hour rows apart once a file has more rows than
    shards, so for the duration-weighted files of the CI matrix
    shard ``k`` keeps the positions that ``scripts/ci_shard_partition.weighted_partition``
    assigns to ``k`` -- a greedy LPT over the committed per-test minutes. It keeps
    every property above that matters: it is a pure function of the sorted node
    ids and a committed file (all workers/ranks agree), every position gets
    exactly one shard (union = selection, no duplicates), and every shard is
    non-empty whenever ``count <= len(items)``. A row absent from the table is
    weighted with its file's median, so it is still selected by exactly one shard.
    """
    selection = _read_shard_selection()
    if selection is None:
        return
    index, count = selection
    if not items:
        # Nothing was collected at all (e.g. this lane's marker filter selected
        # nothing). That is the workflow's pre-existing rc=5 case with its own
        # diagnosis; do not convert it into a different error here.
        return

    # ``id()`` rather than the node id itself: duplicate node ids are legal
    # (the same file reachable under two rootdir-relative paths) and set
    # membership on the string would then keep or drop both copies.
    ordered = sorted(items, key=lambda item: item.nodeid)
    weights_lane = os.environ.get(SHARD_WEIGHTS_ENV, "").strip()
    # One pure function of the sorted node ids: every worker and rank agrees.
    chosen = [ordered[position] for position in _shard_positions([item.nodeid for item in ordered], index, count, weights_lane)]
    keep = {id(item) for item in chosen}
    selected = [item for item in items if id(item) in keep]
    deselected = [item for item in items if id(item) not in keep]

    if not selected:
        # A shard that runs nothing is a green job that proves nothing -- the
        # same failure mode the workflow's "collects ZERO tests" guard exists to
        # prevent. Only reachable when count > len(items), i.e. the configured
        # shard count exceeds the number of tests in this lane's selection.
        raise _shard_usage_error(
            f"{SHARD_COUNT_ENV}={count} / {SHARD_INDEX_ENV}={index} selects ZERO of the "
            f"{len(items)} collected item(s): the shard count exceeds the number of tests in "
            "this lane's selection, so this job would go green having run nothing. Lower the "
            "shard count for this file in the CI matrix."
        )

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def _shard_positions(node_ids: list[str], index: int, count: int, lane: str) -> list[int]:
    """Positions this shard keeps, from scripts/ci_shard_partition.select_shard (fail closed)."""
    from scripts.ci_shard_partition import DURATIONS_PATH, select_shard

    try:
        return select_shard(node_ids, index, count, lane)
    except (OSError, ValueError, KeyError) as exc:
        raise _shard_usage_error(
            f"{SHARD_WEIGHTS_ENV}={lane!r} cannot be applied: {exc}. The duration-weighted split reads "
            f"{DURATIONS_PATH}; regenerate the durations table or clear "
            f"{SHARD_WEIGHTS_ENV} to fall back to the stride."
        ) from None


def pytest_collection_modifyitems(config, items):
    """Skip tests based on CLI options + enforce the light / heavy
    classification at collection time.

    Every test item must carry **exactly one** of ``light`` / ``heavy``:
    an item with neither fails collection, and an item with *both* also
    fails collection. There is no module-level default-light tier -- label
    each test (or each ``pytest.param`` row) explicitly so its cost tier is
    readable at the test site without scrolling to the file top. For a
    parametrized test whose rows differ in cost, mark each row:
    ``pytest.param(..., marks=pytest.mark.light)`` /
    ``pytest.param(..., marks=pytest.mark.heavy)`` -- do **not** put a
    method-level ``@pytest.mark.light`` and override only the heavy rows,
    which produces ambiguous both-marked items (``-m light`` would then
    select a heavy test).
    """
    # 1. Heavy / light classification enforcement: exactly one marker.
    unclassified = [item for item in items if not item.get_closest_marker("heavy") and not item.get_closest_marker("light")]
    if unclassified:
        listing = "\n".join(f"  - {item.nodeid}" for item in unclassified[:20])
        suffix = f"\n  ... and {len(unclassified) - 20} more" if len(unclassified) > 20 else ""
        raise pytest.UsageError(
            "heavy/light classification is mandatory: every collected test must "
            "carry either ``light`` or ``heavy`` (explicitly, per test or per "
            "``pytest.param`` row -- there is no module-level default tier). "
            f"{len(unclassified)} item(s) carry neither:\n{listing}{suffix}"
        )

    both_marked = [item for item in items if item.get_closest_marker("heavy") and item.get_closest_marker("light")]
    if both_marked:
        listing = "\n".join(f"  - {item.nodeid}" for item in both_marked[:20])
        suffix = f"\n  ... and {len(both_marked) - 20} more" if len(both_marked) > 20 else ""
        raise pytest.UsageError(
            "a test must carry EXACTLY one of ``light`` / ``heavy``, not both: an "
            "ambiguous tier cannot be read at the test site and ``-m light`` would "
            "select a heavy test. Mark each parametrize row explicitly instead of a "
            "method-level ``light`` plus per-row ``heavy``. "
            f"{len(both_marked)} item(s) carry both:\n{listing}{suffix}"
        )

    # 2. with_chdir / without_chdir CWD-policy classification enforcement.
    unclassified_chdir = [
        item for item in items if not item.get_closest_marker("with_chdir") and not item.get_closest_marker("without_chdir")
    ]
    if unclassified_chdir:
        listing = "\n".join(f"  - {item.nodeid}" for item in unclassified_chdir[:20])
        suffix = f"\n  ... and {len(unclassified_chdir) - 20} more" if len(unclassified_chdir) > 20 else ""
        raise pytest.UsageError(
            "with_chdir/without_chdir classification is mandatory: every collected test must "
            "carry either ``with_chdir`` (autouse ``_isolate_cwd`` runs) or ``without_chdir`` "
            "(test manages its own CWD). Typically applied via a module-level "
            "``pytestmark`` list. "
            f"{len(unclassified_chdir)} item(s) carry neither:\n{listing}{suffix}"
        )

    if config.getoption("--skip-heavy"):
        skip_marker = pytest.mark.skip(reason="skipped by --skip-heavy")
        for item in items:
            if item.get_closest_marker("heavy"):
                item.add_marker(skip_marker)

    # Session-eligibility markers (_SESSION_MARKERS): a marked test outside its
    # precision mode or MPI world is skipped with its reason; with --fail-on-skip
    # (every CI lane) it leaves the session before the shard split instead, so
    # every kept item must run and any other skip still fails the session.
    excluded_by = [name for name, (_, excludes) in _SESSION_MARKERS.items() if excludes(config)]
    ineligible = []
    for item in items:
        for name in _SESSION_MARKERS:
            marker = item.get_closest_marker(name)
            if marker is not None and not marker.kwargs.get("reason"):
                raise pytest.UsageError(
                    f"{item.nodeid}: @pytest.mark.{name} requires a non-empty "
                    "``reason=`` kwarg (the reason is surfaced as the skip message)."
                )
        name = next((name for name in excluded_by if item.get_closest_marker(name) is not None), None)
        if name is not None:
            ineligible.append(item)
            item.add_marker(pytest.mark.skip(reason=f"{name}: {item.get_closest_marker(name).kwargs['reason']}"))
    # Read by the CI matrix collection plugin.
    config.jqmc_ineligible = [item.nodeid for item in ineligible]
    if ineligible and config.getoption("--fail-on-skip"):
        config.hook.pytest_deselected(items=ineligible)
        dropped = {id(item) for item in ineligible}
        items[:] = [item for item in items if id(item) not in dropped]


def _mpi_size() -> int:
    """The MPI world size of this process, fixed at startup (1 without mpirun)."""
    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_size()


#: The session-eligibility markers: name -> (marker help, whether this session
#: excludes a test that carries it). Every one needs a non-empty ``reason``.
_SESSION_MARKERS = {
    "skip_mixed_precision": (
        "skip_mixed_precision(reason, alternative_guarantee_ref=None): the test is valid only in "
        "--precision-mode=full. "
        "A governed FD/numerical-oracle exclusion must also declare the blocking mixed-precision "
        "``alternative_guarantee_ref`` that closes the same production capability. "
        "Typical reasons: 'FD / numerical quadrature comparison dominated by float32 round-off', "
        "'external reference (e.g. TurboRVB) validated only in --precision-mode=full', "
        "'GFMC_t tau-projection: stochastic dtau inherits fp32 cancellation noise from "
        "J_xp - J_x, amplifying trajectory drift through autodiff force gradients beyond "
        "strict tolerance'.",
        lambda config: config.getoption("--precision-mode") == "mixed" and os.environ.get("JQMC_GDD_AUDIT_MIXED_SKIPS") != "1",
    ),
    "skip_full_precision": (
        "skip_full_precision(reason): the test is valid only in --precision-mode=mixed.",
        lambda config: config.getoption("--precision-mode") != "mixed",
    ),
    "skip_serial": (
        "skip_serial(reason): the test is valid only under MPI (world size > 1; CI runs mpirun -np 2). "
        "A test that needs a particular size checks it itself and fails on a mismatch.",
        lambda config: _mpi_size() == 1,
    ),
    "skip_mpi": (
        "skip_mpi(reason): the test is valid only in a single process (world size 1).",
        lambda config: _mpi_size() > 1,
    ),
}


# ---------------------------------------------------------------------------
# Shared PBC test fixtures.
# Imported as ``from tests.conftest import CELL_FIXTURES, ...`` by the
# periodic test modules (e.g. ``test_structure.py``, ``test_atomic_orbital.py``, ``test_jastrow.py``).
# ---------------------------------------------------------------------------


#: Reference cells covering the periodic validation surface:
#:   - ``cubic_10bohr``     -- orthorhombic, n_max = 2
#:   - ``primitive_fcc_si`` -- primitive FCC (Si lattice constant 10.26 Bohr),
#:                             oblique, n_max = 3
#:   - ``triclinic_skewed`` -- triclinic with non-orthogonal off-diagonals,
#:                             n_max = 3
CELL_FIXTURES: dict[str, np.ndarray] = {
    "cubic_10bohr": np.diag([10.0, 10.0, 10.0]).astype(np.float64),
    "primitive_fcc_si": 0.5
    * 10.26
    * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    ),
    "triclinic_skewed": np.array(
        [[5.0, 0.0, 0.0], [2.5, 4.330, 0.0], [2.5, 1.443, 4.082]],
        dtype=np.float64,
    ),
}

#: ``Structure_data._n_max`` value expected after ``Structure_data.build``
#: computes it on each fixture using the formula ``ceil(half_diag / L_min) + 1``
#: with one cell of safety margin.
EXPECTED_N_MAX: dict[str, int] = {
    "cubic_10bohr": 2,
    "primitive_fcc_si": 3,
    "triclinic_skewed": 3,
}


def make_pbc_structure(
    cell: np.ndarray,
    positions: np.ndarray | None = None,
    atomic_numbers: tuple[int, ...] | None = None,
    element_symbols: tuple[str, ...] | None = None,
    atomic_labels: tuple[str, ...] | None = None,
) -> Structure_data:
    """Build a minimal ``Structure_data(pbc_flag=True)`` for testing.

    Takes the cell as an explicit argument so the same builder serves every
    entry of :data:`CELL_FIXTURES`. Defaults to a single hydrogen atom at
    the origin when no atomic info is supplied --- enough to exercise lattice
    helpers that ignore per-atom data.
    """
    if positions is None:
        positions = np.zeros((1, 3), dtype=np.float64)
        atomic_numbers = (1,)
        element_symbols = ("H",)
        atomic_labels = ("H1",)
    return Structure_data.build(
        pbc_flag=True,
        positions=np.asarray(positions, dtype=np.float64),
        vec_a=tuple(float(x) for x in cell[0]),
        vec_b=tuple(float(x) for x in cell[1]),
        vec_c=tuple(float(x) for x in cell[2]),
        atomic_numbers=atomic_numbers,
        element_symbols=element_symbols,
        atomic_labels=atomic_labels,
    )


def brute_force_mic(
    diff: np.ndarray,
    cell: np.ndarray,
    n_range: int = 5,
) -> np.ndarray:
    """Reference: enumerate +-n_range integer lattice images, return the closest.

    Independent of the production / debug helpers in ``jqmc.structure`` so it
    serves as an external cross-check. Always fp64.
    """
    diff = np.asarray(diff, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    best_d2, best_v = np.inf, diff
    for na in range(-n_range, n_range + 1):
        for nb in range(-n_range, n_range + 1):
            for nc in range(-n_range, n_range + 1):
                T = na * cell[0] + nb * cell[1] + nc * cell[2]
                v = diff + T
                d2 = float(np.dot(v, v))
                if d2 < best_d2:
                    best_d2, best_v = d2, v
    return best_v


def check_symmetric_force(force_mean, force_std, atol, rtol):
    """Validate Newton's third law for a two-atom system.

    For a diatomic molecule the two atomic forces must be equal and opposite
    (``F[0] == -F[1]``) and carry the same statistical error
    (``std[0] == std[1]``). This is the appropriate force check for the
    two-atom H2 fixtures.

    Args:
        force_mean: Per-atom mean force, shape ``(num_atoms, 3)`` with ``num_atoms == 2``.
        force_std: Per-atom force std, shape ``(num_atoms, 3)``.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """
    force_mean = np.asarray(force_mean)
    force_std = np.asarray(force_std)
    assert not np.any(np.isnan(force_mean)), "NaN detected in force_mean"
    assert not np.any(np.isnan(force_std)), "NaN detected in force_std"
    np.testing.assert_allclose(force_mean[0], -1.0 * force_mean[1], atol=atol, rtol=rtol)
    np.testing.assert_allclose(force_std[0], force_std[1], atol=atol, rtol=rtol)


def force_sum_rule_observation(force_mean, atol):
    """Return the deterministic quantities needed to assert the force sum rule.

    Because the SWCT weights satisfy the partition of unity
    (``sum_I omega_I(r) == 1``) and the local energy / log-WF are invariant
    under a rigid translation of nuclei and electrons together, the per-sample
    atomic forces sum to zero exactly: ``sum_I F_I == 0``. The residual is
    deterministic, set by the weakest precision zone on the e_L autodiff path
    (fp64 round-off in full mode, fp32 round-off in mixed mode), scaled by the
    magnitude of the cancelling terms.

    Because ``desired == 0``, ``rtol`` contributes nothing to
    ``assert_allclose`` and ``atol`` alone bounds the residual. The supplied
    ``atol`` from ``get_tolerance_min`` is calibrated for O(1) values; for
    sum-rule tests we scale it by ``max(|force_mean|)`` so that the effective
    tolerance tracks the size of the cancelling terms (matching what ``rtol``
    would have provided against a non-zero target).

    Args:
        force_mean: Per-atom mean force, shape ``(num_atoms, 3)``.
        atol: Absolute tolerance (calibrated for O(1) values; scaled here).
    Returns:
        ``(force_values, force_sum, zero, atol_eff)``.  The caller owns the
        assertions so a governed pytest node records the physical verdict
        directly instead of hiding it inside an imported helper.
    """
    force_mean = np.asarray(force_mean)
    force_sum = np.sum(force_mean, axis=0)
    force_scale = max(np.max(np.abs(force_mean)), 1.0)
    atol_eff = atol * force_scale
    return force_mean, force_sum, np.zeros_like(force_sum), atol_eff


def check_sum_rule_force(force_mean, atol, rtol):
    """Compatibility assertion wrapper; governed tests assert the returned observation directly."""
    force_values, force_sum, zero, atol_eff = force_sum_rule_observation(force_mean, atol)
    assert np.all(np.isfinite(force_values)), "non-finite force_mean"
    np.testing.assert_allclose(force_sum, zero, atol=atol_eff, rtol=rtol)
