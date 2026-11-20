"""Process-level guards that keep JAX from killing a pytest session.

Two independent failure modes, both fatal to a whole test session, both mitigated in
``tests/conftest.py``:

1. **Shared persistent compilation cache.** When several xdist workers (8, for
   example) write one host-provided cache directory, one can die inside
   ``compilation_cache.put_executable`` -> ``Fatal Python error: Aborted``.
   conftest gives each worker its own subdirectory.

2. **Unbounded XLA executable address space.** Every XLA-CPU executable the
   process ever compiles or deserializes is pinned by JAX's compiled-function
   caches, and each kernel in it costs three VMAs. A long single-process
   session therefore climbs monotonically to ``vm.max_map_count`` (65530),
   after which every mmap fails, XLA's error path throws, and libgcc's
   unwinder segfaults on the failing allocation -- reported as
   ``Fatal Python error: Segmentation fault`` at
   ``compilation_cache.py:238 in get_executable_and_time``. conftest bounds it
   with ``jax.clear_caches()`` at the test boundary.

3. **Persistent-cache hit on jaxlib 0.9.1/0.9.2.** Those releases load a cached
   CPU executable on a 512 KiB thread-pool stack and link its object files
   recursively; a large executable overflows the stack and the worker dies
   with no Python traceback. conftest switches the persistent cache
   off for exactly those releases.

4. **Harness threads.** An intra-test VMA sampler running as a
   ``threading.Thread`` is traced by coverage under ``pytest --cov``, and
   coverage's ``sys.modules`` save/restore on that thread deletes the main
   thread's in-flight ``_mhlo_ops_gen`` import. The harness
   runs no Python thread of its own; the VMA bound uses boundary samples only.
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

import importlib.util
import os

import pytest

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.light
def test_jax_cache_dir_is_per_worker():
    """Under xdist with a host-provided cache dir, the effective dir ends in the worker id."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")

    if not cache_dir:
        pytest.skip("no host-provided JAX_COMPILATION_CACHE_DIR (local run)")
    if not worker:
        pytest.skip("not running under pytest-xdist")

    assert os.path.basename(cache_dir) == worker, (
        f"JAX compilation cache is shared across workers: {cache_dir!r} does not end in {worker!r}. "
        "Concurrent writes to one cache entry can abort the interpreter mid-write."
    )


@pytest.mark.light
def test_jax_cache_isolation_is_applied_before_jax_import():
    """The rewrite is useless if it lands after JAX has read the variable.

    conftest.py must perform the assignment above its own ``import jax``.
    """
    conftest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    with open(conftest) as fh:
        src = fh.read()

    assign = src.index('os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(')
    jax_import = src.index("\nimport jax")
    assert assign < jax_import, (
        "tests/conftest.py assigns JAX_COMPILATION_CACHE_DIR after 'import jax'; "
        "JAX reads the variable at backend init, so the per-worker split would be ignored."
    )


@pytest.mark.light
def test_vma_reclaim_policy_keeps_room_for_the_worst_test():
    """The reclaim predicate must fire while a repeat of the worst test still fits.

    Firing only once the table is already full is useless: the allocation that
    trips ``vm.max_map_count`` is the one inside XLA, and it segfaults rather
    than returning an error we could react to.
    """
    from tests.conftest import _vma_reclaim_needed

    limit = 65530

    # Plenty of headroom -> do not pay for a recompile.
    assert not _vma_reclaim_needed(current=1_000, worst_test_cost=5_000, limit=limit)
    # Headroom is smaller than the worst observed test -> reclaim now. Chosen
    # below limit/2 so it is the history clause, not equal room, that fires.
    assert _vma_reclaim_needed(current=30_000, worst_test_cost=40_000, limit=limit)
    # Exactly enough room for one more worst-case test -> still no need.
    assert not _vma_reclaim_needed(current=25_530, worst_test_cost=40_000, limit=limit)
    # Nothing measured yet (first test still running) -> never reclaim.
    assert not _vma_reclaim_needed(current=1_000, worst_test_cost=0, limit=limit)


@pytest.mark.light
def test_vma_reclaim_policy_never_holds_more_than_it_could_still_acquire():
    """History alone is not enough; the equal-room clause is what keeps a two-rank session alive.

    The numbers below are a recorded measurement on
    ``test_twist_decomposition_mpi.py`` under ``mpirun -np 2
    --precision-mode=mixed``: the session entered its last test holding 36059
    VMAs, and that last test then cost 32655 -- 1.5x the 21969 peak cost of
    the most expensive test before it. A predicate that only reserves the
    worst cost seen so far stays quiet at exactly that boundary, and the run
    dies inside the last test (SIGSEGV on rank 0).

    They are kept as literals on purpose: they are a worked example of a
    session where clause (a) is provably insufficient -- which is the property
    under test -- whatever that file's rows cost in a current run. This is
    a test of the PREDICATE, not a pin on any file's present cost.
    """
    from tests.conftest import _vma_reclaim_needed

    limit = 65530

    # The measured boundary: the history clause is quiet, equal room is not.
    measured_entry, measured_worst_before = 36_059, 21_969
    assert limit >= measured_entry + measured_worst_before
    assert _vma_reclaim_needed(current=measured_entry, worst_test_cost=measured_worst_before, limit=limit)
    # Holding less than it could still acquire, with a cheap history -> quiet.
    assert not _vma_reclaim_needed(current=32_765, worst_test_cost=21_969, limit=limit)
    # A light suite never approaches either clause.
    assert not _vma_reclaim_needed(current=1_300, worst_test_cost=523, limit=limit)


@pytest.mark.light
def test_vma_bound_is_installed_as_an_autouse_fixture():
    """The bound is worthless unless every test runs through it.

    Also pins the MPI OR-reduction: some jitted callables are multi-process
    collectives, so a rank that clears its caches alone would recompile inside
    a collective the other ranks never enter, and the MPI run would hang.

    Also pins that the harness runs no Python thread of its own: the
    bound uses boundary samples only, because a ``threading.Thread`` sampler was
    coverage-traced under ``pytest --cov`` and coverage's ``sys.modules``
    save/restore on it deleted the main thread's in-flight ``_mhlo_ops_gen``
    import.
    """
    # Inspect the fixture OBJECT, not the text of conftest.py: locating the
    # fixture with ``str.index`` on the source breaks (``ValueError: substring
    # not found``) as soon as the signature gains a parameter -- reporting a
    # harness edit as a failure instead of checking the property it cares about.
    import inspect

    conftest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    spec = importlib.util.spec_from_file_location("_jqmc_root_conftest", conftest_path)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    fixture = conftest._bound_xla_executable_address_space
    # pytest keeps the marker under different names across majors
    # (``_pytestfixturefunction`` up to 8.x, ``_fixture_function_marker`` in 9),
    # so accept either rather than pinning one private name.
    marker = getattr(fixture, "_fixture_function_marker", None) or getattr(fixture, "_pytestfixturefunction", None)
    assert marker is not None and marker.autouse, (
        "_bound_xla_executable_address_space is not an autouse fixture; tests that skip it "
        "keep leaking XLA executable VMAs until the process segfaults."
    )
    body = inspect.getsource(getattr(fixture, "__wrapped__", fixture))
    assert "MPI.LOR" in body, (
        "the reclaim decision is not OR-reduced over COMM_WORLD; a single rank clearing "
        "its compiled-function cache alone can deadlock a multi-rank run."
    )

    # The harness runs no Python thread of its own. Under ``pytest --cov``
    # every ``threading.Thread`` is coverage-traced, and coverage's
    # ``sys_modules_saved()`` on such a thread deletes the main thread's in-flight
    # imports (the ``KeyError: 'jaxlib.mlir.dialects._mhlo_ops_gen'`` / "Dialect
    # namespace 'mhlo' is already registered" pair). Two complementary checks:
    # (1) conftest.py contains no thread-creation call at all;
    import ast
    import threading

    with open(conftest_path) as fh:
        tree = ast.parse(fh.read())
    thread_factories = {"Thread", "Timer", "start_new_thread", "start_joinable_thread", "ThreadPoolExecutor"}
    creations = sorted(
        f"line {node.lineno}: {name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [getattr(node.func, "attr", None) or getattr(node.func, "id", None)]
        if name in thread_factories
    )
    assert not creations, f"tests/conftest.py starts a thread ({creations})"

    # (2) this test has just run through every autouse fixture's setup, so a
    # thread started by the harness would be alive now. The probe identifies a
    # thread by the source file of the code it runs.
    def harness_threads() -> list[str]:
        found = []
        for thread in threading.enumerate():
            if thread is threading.main_thread():
                continue
            target = getattr(thread, "_target", None) or type(thread).run
            code = getattr(getattr(target, "__func__", target), "__code__", None)
            if code is not None and os.path.exists(code.co_filename) and os.path.samefile(code.co_filename, conftest_path):
                found.append(thread.name)
        return found

    assert harness_threads() == [], f"the test harness runs Python threads {harness_threads()}"

    # Paired evidence: the same probe does see a live thread whose code comes from
    # conftest.py, so the empty result above is not a probe that cannot see.
    release = threading.Event()
    namespace: dict = {}
    exec(compile("def park(event):\n    event.wait()\n", conftest_path, "exec"), namespace)
    control = threading.Thread(target=namespace["park"], args=(release,), name="w133-probe-control")
    control.start()
    try:
        assert harness_threads() == ["w133-probe-control"], harness_threads()
    finally:
        release.set()
        control.join()


@pytest.mark.light
def test_persistent_cache_is_off_where_a_cache_hit_overflows_a_worker_stack():
    """jaxlib 0.9.1/0.9.2 load a cached CPU executable on a 512 KiB pool thread.

    ORC links the cached object files recursively, so a large executable (the
    curated dense-AGP + MPC ``e_L`` cases) overflows that stack on a cache HIT
    and the worker dies with no Python traceback. conftest switches the
    persistent cache off for exactly those releases: ``jax.config`` for this
    process (before anything compiles) and ``JAX_ENABLE_COMPILATION_CACHE`` for
    every child process (xdist workers, CLI subprocesses).

    The release table is the measured one (same test, shared cache dir): 0.9.1
    and 0.9.2 crash on the hit; 0.7.2, 0.8.0, 0.9.0, 0.10.0, 0.10.1 and 0.10.2
    pass. On an affected jaxlib this session must actually run with the cache
    off; elsewhere the setting is left to the host. The table is jqmc's own
    (``jqmc._setting``), shared with the runtime warning.
    """
    import jax

    from jqmc._setting import installed_jaxlib_version as _installed_jaxlib_version
    from jqmc._setting import jaxlib_persistent_cache_load_overflows as _persistent_cache_load_overflows

    for release in ("0.9.1", "0.9.2", "0.9.2.dev20260901"):
        assert _persistent_cache_load_overflows(release), release
    for release in ("0.7.0", "0.7.2", "0.8.0", "0.9.0", "0.10.0", "0.10.1", "0.10.2", "unknown"):
        assert not _persistent_cache_load_overflows(release), release

    import jaxlib

    installed = _installed_jaxlib_version()
    assert installed == jaxlib.__version__, (installed, jaxlib.__version__)
    # This very session: cache off on an affected release, jax's default (on) otherwise.
    cache_enabled = bool(jax.config.jax_enable_compilation_cache)
    assert cache_enabled != _persistent_cache_load_overflows(installed), (
        f"jaxlib {installed}: persistent compilation cache enabled={cache_enabled}; conftest must switch it off "
        "exactly on the releases whose cache load can overflow a 512 KiB worker stack, and leave it on elsewhere."
    )

    # Child processes (xdist workers, CLI subprocesses) inherit the switch through the environment.
    child_switch = os.environ.get("JAX_ENABLE_COMPILATION_CACHE")
    assert child_switch == "false" or not _persistent_cache_load_overflows(installed), (
        f"jaxlib {installed}: JAX_ENABLE_COMPILATION_CACHE={child_switch!r}; child processes of this session "
        "would still load cached executables."
    )
