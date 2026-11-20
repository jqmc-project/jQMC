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

//! Lightweight hot-path phase profiler (env `JQMC_RS_PROFILE=1`). Thread-local
//! nanosecond accumulators keyed by a `&'static str`; `dump()` prints the
//! breakdown to stderr. Zero overhead when off -- a single relaxed atomic load
//! guards each `timed!` block. Instrumentation only (never affects results).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

pub static PROFILE_ON: AtomicBool = AtomicBool::new(false);

thread_local! {
    static PROF: RefCell<Vec<(&'static str, u128, u64)>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub fn on() -> bool {
    PROFILE_ON.load(Ordering::Relaxed)
}

/// Enable/disable from the env once per run entry point.
pub fn set_from_env() {
    PROFILE_ON.store(std::env::var("JQMC_RS_PROFILE").is_ok(), Ordering::Relaxed);
}

#[inline]
pub fn add(name: &'static str, nanos: u128) {
    PROF.with(|p| {
        let mut v = p.borrow_mut();
        if let Some(e) = v.iter_mut().find(|e| e.0 == name) {
            e.1 += nanos;
            e.2 += 1;
        } else {
            v.push((name, nanos, 1));
        }
    });
}

/// Print the accumulated breakdown (descending) and clear it.
pub fn dump(tag: &str) {
    PROF.with(|p| {
        let mut v = p.borrow().clone();
        v.sort_by_key(|b| std::cmp::Reverse(b.1));
        eprintln!("--- JQMC_RS profile [{tag}] (thread-local, ms) ---");
        for (name, nanos, count) in &v {
            eprintln!(
                "  {:<20} {:>10.1} ms  ({} calls, {:.4} ms/call)",
                name,
                *nanos as f64 / 1e6,
                count,
                *nanos as f64 / 1e6 / *count as f64
            );
        }
        p.borrow_mut().clear();
    });
}

/// Time `$body`, accumulating under `$name` when profiling is on. Returns the
/// body's value either way.
#[macro_export]
macro_rules! timed {
    ($name:expr, $body:expr) => {{
        if $crate::prof::on() {
            let __t = std::time::Instant::now();
            let __r = $body;
            $crate::prof::add($name, __t.elapsed().as_nanos());
            __r
        } else {
            $body
        }
    }};
}
