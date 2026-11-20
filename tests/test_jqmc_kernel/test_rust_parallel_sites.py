"""Static parallel-boundary inventory and deterministic detector controls.

No native import, numerical execution, thread-value rerun, or tolerance is
involved. The repository node is the obligation; scratch controls test the
guard's diagnostics, including source-level import aliases and re-exports.
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

from pathlib import Path

import pytest

from scripts.check_rust_parallel_sites import CONFIGURE_BODY, SOURCE, validation_errors


pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.design_pin]
ROOT = Path(__file__).resolve().parents[2]
WALKER = """
use rayon::prelude::*;
pub fn for_each_walker<T: Send>(py: Python<'_>, nw: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    py.allow_threads(|| (0..nw).into_par_iter().map(f).collect())
}
"""
CONFIGURATION = (
    "fn py_configure_threads() {" + CONFIGURE_BODY + "}\nfn py_current_num_threads() -> usize { rayon::current_num_threads() }"
)
SERIAL = """
fn py_native_vmc() {
    hamiltonian_data.py().allow_threads(|| {
        run(&terms, seed, dt, epsilon, moves_per_block, n_warmup, n_blocks,)
    });
}
fn py_native_vmc_complex() {
    hamiltonian_data.py().allow_threads(|| {
        run_complex(&terms, seed, dt, epsilon, moves_per_block, n_warmup, n_blocks,)
    });
}
"""


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / SOURCE
    source.mkdir(parents=True)
    for name, text in (("walker_batch.rs", WALKER), ("lib.rs", CONFIGURATION), ("native_driver.rs", SERIAL)):
        (source / name).write_text(text, encoding="utf-8")
    assert validation_errors(tmp_path) == []
    return tmp_path


def test_rust_production_parallel_source_inventory():
    """Keep production dispatch at the reviewed walker boundary and serial GIL sites."""
    assert validation_errors(ROOT) == []


@pytest.mark.parametrize(
    "code",
    [
        "fn bad() { (0..n).into_par_iter().map(f).sum::<f64>(); }",
        "fn bad() { values.par_iter().sum::<f64>(); }",
        "fn bad() { values.par_iter_mut().for_each(f); }",
        "fn bad() { values.par_bridge().for_each(f); }",
        "fn bad() { values.par_chunks(2).for_each(f); }",
        "fn bad() { rayon::join(a, b); }",
        "fn bad() { rayon::scope(|s| s.spawn(f)); }",
        "fn bad() { rayon::scope_fifo(|s| s.spawn_fifo(f)); }",
        "fn bad() { rayon::spawn(f); }",
        "use rayon as work; fn bad() { work::join(a, b); }",
        "use rayon::{join as fork, scope as region}; fn bad() { fork(a, b); }",
        "use rayon::{self as work}; fn bad() { work::join(a, b); }",
        "pub use rayon::join as fork;",
        "pub use ::rayon::{self as work, join as fork};",
        "extern crate rayon as work; fn bad() { work::join(a, b); }",
        "use rayon::*; fn bad() { join(a, b); }",
        "fn bad() { let fork = rayon::join; fork(a, b); }",
        "fn bad() { IntoParallelIterator::into_par_iter(values); }",
        "fn bad() { <T as IntoParallelIterator>::into_par_iter(values); }",
        "fn bad() { values.r#par_iter().sum::<f64>(); }",
        "use r#rayon as work; fn bad() { work::join(a, b); }",
        "fn bad() { rayon_core::join(a, b); }",
        "macro_rules! bad { () => { rayon::join(a, b) }; }",
        "#[cfg(not(test))] fn bad() { values.par_iter(); }",
        '#[cfg(feature = "future")] fn bad() { values.par_iter(); }',
        "mod tests { fn still_production() { values.par_iter(); } }",
        "#[cfg(test)] mod tests { fn unit() {} } fn bad() { values.par_iter(); }",
    ],
)
def test_rust_parallel_inventory_rejects_new_sites_and_aliases(source_tree, code):
    """Reject added parallel entry tokens and Rayon imports, aliases, or re-exports."""
    # A nested file is deliberate: the obligation is src/**/*.rs, not *.rs.
    path = source_tree / SOURCE / "new_kernel" / "gfmc.rs"
    path.parent.mkdir()
    path.write_text(code, encoding="utf-8")
    errors = validation_errors(source_tree)
    assert any(error.startswith("RAYON-SITE: new_kernel/gfmc.rs:") for error in errors), errors
    assert not any(error.startswith("RAYON-SOURCE-SYNTAX:") for error in errors), errors


@pytest.mark.parametrize(
    "code",
    [
        "// rayon::join(a, b); values.par_iter();\nfn ordinary() {}",
        "/* rayon::scope /* nested */ values.par_iter() */ fn ordinary() {}",
        'fn ordinary() { let text = "rayon::join(a, b) \\" par_iter"; }',
        'fn ordinary() { let text = r###"rayon::join("a", b); par_iter"###; }',
        'fn ordinary() { let text = br##"rayon::scope par_bridge"##; }',
        "fn ordinary<'a>(x: &'a str) { let a = '}'; let b = '\\''; let c = '\\u{7b}'; }",
        "#[cfg(test)] mod tests { use rayon::prelude::*; fn unit() { values.par_iter(); } }",
        "#[cfg(test)] #[allow(dead_code)] fn unit() { rayon::join(a, b); }",
        "impl Example { #[cfg(test)] fn unit(&self) { rayon::join(a, b); } }",
    ],
)
def test_rust_parallel_inventory_ignores_nonproduction_text(source_tree, code):
    """Exclude comments, literals, and directly test-gated items from production inventory."""
    (source_tree / SOURCE / "ordinary.rs").write_text(code, encoding="utf-8")
    assert validation_errors(source_tree) == []


@pytest.mark.parametrize("filename", ["gfmc.rs", "walker_batch.rs", "native_driver.rs", "lib.rs"])
def test_rust_parallel_inventory_rejects_extra_gil_release(source_tree, filename):
    """Reject GIL-release sites outside the three reviewed functions, including known files."""
    path = source_tree / SOURCE / filename
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(old + "\nfn unreviewed() { py.allow_threads(|| run()); }\n", encoding="utf-8")
    errors = validation_errors(source_tree)
    assert any(error.startswith(f"GIL-RELEASE-SITE: {filename}:") for error in errors), errors


@pytest.mark.parametrize(
    "replacement",
    [
        "py.allow_threads(|| (0..nw).into_par_iter().map(f).sum())",
        "py.allow_threads(|| { rayon::join(a, b); (0..nw).into_par_iter().map(f).collect() })",
        "py.allow_threads(|| (0..nw).into_par_iter().map(|w| f(w)).collect())",
    ],
)
def test_rust_parallel_inventory_pins_ordered_walker_body(source_tree, replacement):
    """Reject changes to the reviewed indexed walker map-and-collect expression."""
    path = source_tree / SOURCE / "walker_batch.rs"
    path.write_text(
        WALKER.replace("py.allow_threads(|| (0..nw).into_par_iter().map(f).collect())", replacement), encoding="utf-8"
    )
    errors = validation_errors(source_tree)
    assert "RAYON-WALKER-BODY: walker_batch.rs::for_each_walker ordered map/collect body changed" in errors


def test_rust_parallel_inventory_rejects_second_site_in_allowed_file(source_tree):
    """Reject an additional parallel site even inside the walker helper's source file."""
    path = source_tree / SOURCE / "walker_batch.rs"
    path.write_text(WALKER + "fn another() { values.into_par_iter().sum::<f64>(); }", encoding="utf-8")
    errors = validation_errors(source_tree)
    assert any("::another: unreviewed parallel entry into_par_iter" in error for error in errors), errors


def test_rust_parallel_inventory_pins_serial_gil_release_body(source_tree):
    """Require the native-driver GIL closures to remain direct serial run calls."""
    path = source_tree / SOURCE / "native_driver.rs"
    path.write_text(SERIAL.replace("run(&terms", "changed_run(&terms"), encoding="utf-8")
    errors = validation_errors(source_tree)
    assert any(error.startswith("GIL-RELEASE-BODY:") and "direct serial run closure" in error for error in errors), errors


def test_rust_parallel_inventory_requires_expected_sites(source_tree):
    """Detect missing reviewed sites instead of accepting a vacuously empty inventory."""
    (source_tree / SOURCE / "walker_batch.rs").write_text("fn for_each_walker() {}", encoding="utf-8")
    errors = validation_errors(source_tree)
    assert any(error.startswith("RAYON-INVENTORY: missing=") and "into_par_iter" in error for error in errors), errors


@pytest.mark.parametrize(
    "method",
    ["par_chars", "par_bytes", "par_lines", "par_drain", "par_extend", "par_array_windows", "par_chunk_by", "par_future_api"],
)
def test_rust_parallel_inventory_rejects_prelude_dispatch(source_tree, method):
    """Reject additional par-prefixed dispatch in the file that already imports Rayon's prelude."""
    path = source_tree / SOURCE / "walker_batch.rs"
    path.write_text(WALKER + f"fn another() {{ values.{method}().count(); }}", encoding="utf-8")
    errors = validation_errors(source_tree)
    assert any(f"::another: unreviewed parallel entry {method}" in error for error in errors), errors
    assert not any(error.startswith("RAYON-SOURCE-SYNTAX:") for error in errors), errors


@pytest.mark.parametrize("method", ["spawn", "install", "scope", "scope_fifo", "broadcast"])
def test_rust_parallel_inventory_rejects_local_pool_dispatch(source_tree, method):
    """Reject turning approved global-pool setup into a local pool with work dispatch."""
    path = source_tree / SOURCE / "lib.rs"
    mutation = f"let pool = builder.build().unwrap(); pool.{method}(|| work()); match Ok(())"
    path.write_text(CONFIGURATION.replace("match builder.build_global()", mutation), encoding="utf-8")
    errors = validation_errors(source_tree)
    assert "RAYON-CONFIGURE-BODY: lib.rs::py_configure_threads global-pool configuration changed" in errors
    assert not any(error.startswith("RAYON-SOURCE-SYNTAX:") for error in errors), errors


@pytest.mark.parametrize(
    "setup",
    [
        'if env_enabled("JQMC_RS_NUMA_INTERLEAVE") { enable_numa_interleave().unwrap(); }',
        "builder = builder.start_handler(|_| pin_current_thread(0));",
    ],
    ids=["main-numa-policy", "worker-cpu-pinning"],
)
def test_rust_parallel_inventory_rejects_internal_placement(source_tree, setup):
    """Keep CPU and NUMA placement out of the thread-count configuration."""
    path = source_tree / SOURCE / "lib.rs"
    path.write_text(
        CONFIGURATION.replace("match builder.build_global()", f"{setup}\nmatch builder.build_global()"),
        encoding="utf-8",
    )
    errors = validation_errors(source_tree)
    assert errors == ["RAYON-CONFIGURE-BODY: lib.rs::py_configure_threads global-pool configuration changed"]
