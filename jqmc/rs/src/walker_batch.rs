// Copyright (C) 2024- Kosuke Nakano
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//! The single Rayon entry point for the walker axis.
//!
//! Every batch PyO3 entry that fans a computation over walkers goes through
//! [`for_each_walker`]; no other site opens its own Rayon parallel iterator.
//! The two serial native VMC drivers separately release the GIL around their
//! complete serial runs. Centralizing walker dispatch here preserves its GIL
//! release and walker order; the source inventory checks these boundaries.

use pyo3::Python;
use rayon::prelude::*;

/// Run `f(w)` for every walker index `w` in `0..nw` on the rayon global pool,
/// releasing the GIL for the duration, and collect the per-walker results in
/// **walker order** (`out[w]` is always walker `w`'s result).
///
/// # Determinism contract
/// The returned `Vec<T>` is **bit-identical to a serial `(0..nw).map(f).collect()`
/// regardless of the thread count**, provided `f(w)` reads only per-walker inputs
/// (its own threefry key `keys[w]`, its own `r_up`/`r_dn`/`geminal[w]` slice) and
/// never another walker's state. Two facts guarantee this:
///   * `IndexedParallelIterator::map(..).collect()` writes each result to its own
///     index, so thread scheduling can never reorder the output -- `out[w]` is
///     walker `w` on 1 thread and on N threads alike.
///   * The per-walker threefry key makes each walker's RNG stream walker-local, so
///     a batch of `nw` walkers produces the same draws no matter how the walkers
///     are distributed across threads.
///     If the threads=1 and threads=N outputs differ, shared mutable state has
///     leaked into `f`; fix the sharing rather than loosening the comparison.
///
/// # GIL rule
/// `f` runs while the GIL is **released** (`py.allow_threads`). Callers MUST extract
/// every `PyReadonlyArray` into an owned array / ndarray view (which are `Send`)
/// **before** calling `for_each_walker` -- `PyReadonlyArray` is not `Send` and must
/// not be captured by `f`. Big read-only statics belong in `NativeObsCtx`, which is
/// `Arc`-held and `Sync` (enforced at compile time by `_assert_obs_ctx_sync` in
/// `lib.rs`); `f` may borrow `&NativeObsCtx` freely across the worker threads. `f`
/// itself must be `Sync + Send` so rayon can share `&f` with the workers.
pub fn for_each_walker<T: Send>(
    py: Python<'_>,
    nw: usize,
    f: impl Fn(usize) -> T + Sync + Send,
) -> Vec<T> {
    py.allow_threads(|| (0..nw).into_par_iter().map(f).collect())
}
