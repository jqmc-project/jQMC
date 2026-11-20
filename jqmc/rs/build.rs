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

// Build-identity guard.
//
// A stale ``jqmc_rs.abi3.so`` (older than the Rust source it is supposed to
// embody) silently passes every parity test, because ``import jqmc_rs`` succeeds
// and nothing checks that the loaded binary was built from the current source.
// This computes a dependency-free FNV-1a 64-bit digest over the crate's Rust
// source tree (``src/**/*.rs``), Cargo/build/generator inputs and ../_setting.py,
// then bakes it into the
// extension as the compile-time env var ``JQMC_RS_SOURCE_FNV64`` (exposed as
// ``jqmc_rs.__source_fnv64__``). ``tests/test_rs_kernel/test_build_identity.py``
// recomputes the SAME digest from the working tree and asserts they match, so a
// stale binary fails loudly instead of being tested silently.
//
// FNV-1a is used (not a crypto hash) purely as a change detector: no security
// property is needed, and it is trivial to reproduce byte-for-byte in Python
// without adding any build dependency. The digest scheme MUST stay in lock-step
// with ``_source_fnv64`` in the test: for each (relpath, content) in the file
// set sorted by relpath, fold ``relpath`` bytes, a NUL, the raw file bytes, and
// a NUL.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn collect_rs(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
    for entry in fs::read_dir(dir).expect("read_dir src") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, root, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let rel = path
                .strip_prefix(root)
                .expect("strip crate root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}

fn main() {
    pyo3_build_config::add_extension_module_link_args();

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // Parse the authored literals without importing jqmc (or requiring JAX in
    // the wheel-build environment). An explicitly selected interpreter failing
    // is an error, not a reason to silently use another environment.
    for variable in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let python = std::env::var_os("PYO3_PYTHON")
        .or_else(|| std::env::var_os("PYTHON_SYS_EXECUTABLE"))
        .or_else(|| pyo3_build_config::get().executable.as_ref().map(Into::into))
        .unwrap_or_else(|| "python3".into());
    let output = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let generated = Command::new(&python)
        .arg(root.join("generate_setting_consts.py"))
        .arg(root.join("../_setting.py"))
        .arg(output.join("setting_consts.rs"))
        .output()
        .unwrap_or_else(|e| panic!("run central-settings generator with {python:?}: {e}"));
    assert!(
        generated.status.success(),
        "central-settings generator failed: {}",
        String::from_utf8_lossy(&generated.stderr)
    );

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_rs(&root.join("src"), root, &mut files);
    files.push(("Cargo.toml".to_string(), root.join("Cargo.toml")));
    files.push(("build.rs".to_string(), root.join("build.rs")));
    files.push((
        "generate_setting_consts.py".to_string(),
        root.join("generate_setting_consts.py"),
    ));
    files.push(("../_setting.py".to_string(), root.join("../_setting.py")));
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut digest = FNV_OFFSET;
    for (rel, path) in &files {
        let content = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        digest = fnv_update(digest, rel.as_bytes());
        digest = fnv_update(digest, b"\0");
        digest = fnv_update(digest, &content);
        digest = fnv_update(digest, b"\0");
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=JQMC_RS_SOURCE_FNV64={digest:016x}");
}
