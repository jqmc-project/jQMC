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

//! Host (CPU/MPI) optimizer linear algebra on the Rayon global pool (issue #815).
//!
//! One-to-one Rust implementations of the BLAS-heavy host functions of
//! `jqmc/jqmc_optimizer_kernel.py` / `jqmc/jqmc_optimizer.py`:
//!
//! * `solve_wide_sr_real_cg` / `solve_wide_sr_real_direct`
//! * `solve_tall_sr_real_cg` / `solve_tall_sr_real_direct`
//! * `_build_lm_real_pencil_from_coordinate_arrays` (the LM pencil core)
//!
//! The Python functions keep their signatures, validation and logging and
//! dispatch here under the production CPU rule (`JQMC_FLAG_RUST_ON_CPU` /
//! `flag_rust_on_CPU` on the JAX cpu backend). MPI stays in Python: every
//! cross-rank sum is the caller's `reduce_sum`, passed in as a Python callable
//! and invoked with the GIL held at exactly the points where the NumPy
//! implementation reduces. This module never links MPI.
//!
//! # Parallelism and determinism
//! Every parallel section goes through the reviewed ordered boundary
//! [`for_each_walker`] (GIL released, results collected in index order). The
//! work is cut into tiles whose sizes depend on the matrix shape only, never on
//! the thread count; each tile is computed sequentially and partial results are
//! combined in a fixed order. Matrix-vector products, Gram matrices and the
//! pencil contractions are therefore bit-identical for any
//! `JQMC_RS_NUM_THREADS`. The partial-pivot LU of the direct solves is faer's,
//! run with faer's global parallelism (the same Rayon global pool) on one task
//! of that boundary so the GIL is released; its roundoff may depend on the
//! thread count.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::{Accum, Mat, MatRef, Par};
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::walker_batch::for_each_walker;

/// Minimum rows of one row block (shape-only tiling).
const MIN_BLOCK_ROWS: usize = 256;
/// Upper bound of the number of row blocks before blocks grow.
const MAX_ROW_BLOCKS: usize = 256;
/// Column-strip width of the matrix-vector tiles.
const STRIP: usize = 2048;
/// Output tile edge of the Gram / weighted cross-product tiles.
const GRAM_TILE: usize = 256;

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b.max(1))
}

/// Rows per block for a matrix with `nrows` rows: a function of the shape only,
/// and a multiple of `GRAM_TILE` so a Gram tile never straddles two blocks.
#[inline]
fn block_rows(nrows: usize) -> usize {
    GRAM_TILE
        * ceil_div(
            MIN_BLOCK_ROWS.max(ceil_div(nrows, MAX_ROW_BLOCKS)),
            GRAM_TILE,
        )
}

/// Run `f` once on the Rayon global pool with the GIL released.
fn release<T: Send>(py: Python<'_>, f: impl Fn() -> T + Sync + Send) -> T {
    for_each_walker(py, 1, |_| f()).pop().expect("one task")
}

/// Deterministic sequential dot product (fixed 8-lane accumulation pattern,
/// independent of the thread count; vectorizable without reassociation).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let mut a8 = a.chunks_exact(8);
    let mut b8 = b.chunks_exact(8);
    for (x, y) in (&mut a8).zip(&mut b8) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0;
    for (x, y) in a8.remainder().iter().zip(b8.remainder()) {
        tail += x * y;
    }
    (((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))) + tail
}

/// Upper bound of the doubles held by the partial results of one fused pass.
const FUSED_PARTIAL_BUDGET: usize = 1 << 22;
/// Minimum rows per task of the fused pass.
const FUSED_MIN_ROWS: usize = 64;

/// A row-major matrix stored as consecutive row blocks of `br` rows (the last
/// block may be shorter). Blocks either borrow a NumPy buffer or are owned.
struct Blocks<'a> {
    blocks: Vec<&'a [f64]>,
    br: usize,
    nrows: usize,
    ncols: usize,
}

impl<'a> Blocks<'a> {
    fn from_slice(data: &'a [f64], nrows: usize, ncols: usize) -> Self {
        let br = block_rows(nrows);
        let blocks = if nrows == 0 || ncols == 0 {
            Vec::new()
        } else {
            data[..nrows * ncols].chunks(br * ncols).collect()
        };
        Blocks {
            blocks,
            br,
            nrows,
            ncols,
        }
    }

    fn from_owned(owned: &'a [Vec<f64>], br: usize, nrows: usize, ncols: usize) -> Self {
        Blocks {
            blocks: owned.iter().map(|b| b.as_slice()).collect(),
            br,
            nrows,
            ncols,
        }
    }

    fn row(&self, r: usize) -> &[f64] {
        let off = (r % self.br) * self.ncols;
        &self.blocks[r / self.br][off..off + self.ncols]
    }

    fn rows_in(&self, b: usize) -> usize {
        self.blocks[b].len() / self.ncols.max(1)
    }

    /// Row-major faer view of block `b`.
    fn view(&self, b: usize) -> MatRef<'_, f64> {
        MatRef::from_row_major_slice(self.blocks[b], self.rows_in(b), self.ncols)
    }

    /// `out = A x` (length `nrows`).
    fn av(&self, py: Python<'_>, x: &[f64]) -> Vec<f64> {
        if self.blocks.is_empty() {
            return vec![0.0; self.nrows];
        }
        MATRIX_SWEEPS.fetch_add(1, Ordering::Relaxed);
        let nstrip = ceil_div(self.ncols, STRIP);
        let ncols = self.ncols;
        let parts = for_each_walker(py, self.blocks.len(), |b| {
            let block = self.blocks[b];
            let rows = block.len() / ncols.max(1);
            let mut out = vec![0.0; rows];
            for (r, slot) in out.iter_mut().enumerate() {
                let row = &block[r * ncols..(r + 1) * ncols];
                let mut total = 0.0;
                for s in 0..nstrip {
                    let c0 = s * STRIP;
                    let c1 = (c0 + STRIP).min(ncols);
                    total += dot(&row[c0..c1], &x[c0..c1]);
                }
                *slot = total;
            }
            out
        });
        let mut out = Vec::with_capacity(self.nrows);
        for part in parts {
            out.extend_from_slice(&part);
        }
        out
    }

    /// `out = A^T y` (length `ncols`).
    fn atv(&self, py: Python<'_>, y: &[f64]) -> Vec<f64> {
        if self.blocks.is_empty() {
            return vec![0.0; self.ncols];
        }
        MATRIX_SWEEPS.fetch_add(1, Ordering::Relaxed);
        let ncols = self.ncols;
        let nblock = self.blocks.len();
        let nstrip = ceil_div(ncols, STRIP);
        let br = self.br;
        // One task per (row block, column strip); combined over blocks in order.
        let parts = for_each_walker(py, nblock * nstrip, |task| {
            let b = task / nstrip;
            let s = task % nstrip;
            let c0 = s * STRIP;
            let c1 = (c0 + STRIP).min(ncols);
            let block = self.blocks[b];
            let rows = block.len() / ncols.max(1);
            let mut acc = vec![0.0; c1 - c0];
            for r in 0..rows {
                let yr = y[b * br + r];
                let row = &block[r * ncols + c0..r * ncols + c1];
                for (a, v) in acc.iter_mut().zip(row) {
                    *a += v * yr;
                }
            }
            acc
        });
        let mut out = vec![0.0; ncols];
        for b in 0..nblock {
            for s in 0..nstrip {
                let c0 = s * STRIP;
                for (o, v) in out[c0..].iter_mut().zip(&parts[b * nstrip + s]) {
                    *o += v;
                }
            }
        }
        out
    }

    /// `out = A^T (A v)` in one streaming pass over the rows of `A`: each row
    /// is read once and used for its dot product and its scaled accumulation.
    /// Fixed row chunks (a function of the shape) produce partial vectors that
    /// are summed in chunk order, so the result does not depend on the threads.
    fn fused_ata(&self, py: Python<'_>, v: &[f64]) -> Vec<f64> {
        let ncols = self.ncols;
        if self.blocks.is_empty() {
            return vec![0.0; ncols];
        }
        MATRIX_SWEEPS.fetch_add(1, Ordering::Relaxed);
        let max_parts = 8usize.max((FUSED_PARTIAL_BUDGET / ncols.max(1)).min(512));
        let rows_per_task = FUSED_MIN_ROWS.max(ceil_div(self.nrows, max_parts));
        let ntask = ceil_div(self.nrows, rows_per_task);
        let parts = for_each_walker(py, ntask, |t| {
            let r0 = t * rows_per_task;
            let r1 = ((t + 1) * rows_per_task).min(self.nrows);
            let mut acc = vec![0.0; ncols];
            for r in r0..r1 {
                let row = self.row(r);
                let scale = dot(row, v);
                for (a, x) in acc.iter_mut().zip(row) {
                    *a += scale * x;
                }
            }
            acc
        });
        let nstrip = ceil_div(ncols, STRIP);
        let strips = for_each_walker(py, nstrip, |s| {
            let c0 = s * STRIP;
            let c1 = (c0 + STRIP).min(ncols);
            let mut out = vec![0.0; c1 - c0];
            for part in &parts {
                for (o, p) in out.iter_mut().zip(&part[c0..c1]) {
                    *o += p;
                }
            }
            out
        });
        strips.concat()
    }

    /// Row Gram matrix `A A^T` (nrows x nrows), lower tiles mirrored.
    fn row_gram(&self, py: Python<'_>) -> Mat<f64> {
        let n = self.nrows;
        let nt = ceil_div(n, GRAM_TILE);
        let pairs: Vec<(usize, usize)> =
            (0..nt).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
        let tiles = for_each_walker(py, pairs.len(), |t| {
            let (i, j) = pairs[t];
            let (r0, r1) = (i * GRAM_TILE, ((i + 1) * GRAM_TILE).min(n));
            let (c0, c1) = (j * GRAM_TILE, ((j + 1) * GRAM_TILE).min(n));
            let mut out = Mat::<f64>::zeros(r1 - r0, c1 - c0);
            if self.ncols > 0 {
                let lhs = self.view(r0 / self.br).subrows(r0 % self.br, r1 - r0);
                let rhs = self.view(c0 / self.br).subrows(c0 % self.br, c1 - c0);
                faer::linalg::matmul::matmul(
                    out.as_mut(),
                    Accum::Replace,
                    lhs,
                    rhs.transpose(),
                    1.0,
                    Par::Seq,
                );
            }
            out
        });
        let mut gram = Mat::<f64>::zeros(n, n);
        for (t, tile) in tiles.iter().enumerate() {
            let (i, j) = pairs[t];
            for a in 0..tile.nrows() {
                for b in 0..tile.ncols() {
                    let (r, c) = (i * GRAM_TILE + a, j * GRAM_TILE + b);
                    gram[(r, c)] = tile[(a, b)];
                    gram[(c, r)] = tile[(a, b)];
                }
            }
        }
        gram
    }

    /// Column Gram matrix `A^T A` (ncols x ncols), lower tiles mirrored.
    fn col_gram(&self, py: Python<'_>) -> Mat<f64> {
        let n = self.ncols;
        let nt = ceil_div(n, GRAM_TILE);
        let pairs: Vec<(usize, usize)> =
            (0..nt).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
        let tiles = for_each_walker(py, pairs.len(), |t| {
            let (i, j) = pairs[t];
            let (r0, r1) = (i * GRAM_TILE, ((i + 1) * GRAM_TILE).min(n));
            let (c0, c1) = (j * GRAM_TILE, ((j + 1) * GRAM_TILE).min(n));
            let mut out = Mat::<f64>::zeros(r1 - r0, c1 - c0);
            for b in 0..self.blocks.len() {
                let view = self.view(b);
                faer::linalg::matmul::matmul(
                    out.as_mut(),
                    Accum::Add,
                    view.subcols(r0, r1 - r0).transpose(),
                    view.subcols(c0, c1 - c0),
                    1.0,
                    Par::Seq,
                );
            }
            out
        });
        let mut gram = Mat::<f64>::zeros(n, n);
        for (t, tile) in tiles.iter().enumerate() {
            let (i, j) = pairs[t];
            for a in 0..tile.nrows() {
                for b in 0..tile.ncols() {
                    let (r, c) = (i * GRAM_TILE + a, j * GRAM_TILE + b);
                    gram[(r, c)] = tile[(a, b)];
                    gram[(c, r)] = tile[(a, b)];
                }
            }
        }
        gram
    }
}

/// Weighted cross product `A^T diag(c) B` for row-blocked A (n x qa), B (n x qb)
/// with identical row blocking. Full (unsymmetrized) result.
fn weighted_cross(py: Python<'_>, a: &Blocks<'_>, b: &Blocks<'_>, c: &[f64]) -> Mat<f64> {
    let (qa, qb) = (a.ncols, b.ncols);
    let (ta, tb) = (ceil_div(qa, GRAM_TILE), ceil_div(qb, GRAM_TILE));
    let br = a.br;
    let tiles = for_each_walker(py, ta * tb, |t| {
        let (i, j) = (t / tb, t % tb);
        let (r0, r1) = (i * GRAM_TILE, ((i + 1) * GRAM_TILE).min(qa));
        let (c0, c1) = (j * GRAM_TILE, ((j + 1) * GRAM_TILE).min(qb));
        let mut out = Mat::<f64>::zeros(r1 - r0, c1 - c0);
        for blk in 0..a.blocks.len() {
            let rows = a.rows_in(blk);
            let bview = b.view(blk);
            let scaled = Mat::from_fn(rows, c1 - c0, |r, k| c[blk * br + r] * bview[(r, c0 + k)]);
            faer::linalg::matmul::matmul(
                out.as_mut(),
                Accum::Add,
                a.view(blk).subcols(r0, r1 - r0).transpose(),
                scaled.as_ref(),
                1.0,
                Par::Seq,
            );
        }
        out
    });
    let mut result = Mat::<f64>::zeros(qa, qb);
    for (t, tile) in tiles.iter().enumerate() {
        let (i, j) = (t / tb, t % tb);
        for r in 0..tile.nrows() {
            for k in 0..tile.ncols() {
                result[(i * GRAM_TILE + r, j * GRAM_TILE + k)] = tile[(r, k)];
            }
        }
    }
    result
}

/// True if any imaginary part of an interleaved complex matrix is non-zero.
fn any_imag(py: Python<'_>, data: &[f64], nrows: usize, ncols_complex: usize) -> bool {
    let src = Blocks::from_slice(data, nrows, 2 * ncols_complex);
    for_each_walker(py, src.blocks.len(), |b| {
        src.blocks[b].iter().skip(1).step_by(2).any(|v| *v != 0.0)
    })
    .into_iter()
    .any(|x| x)
}

/// Owned real matrix built from an interleaved complex (nrows x ncomplex)
/// matrix: its real part, or with `stack_imag` the real part followed by the
/// imaginary part as stacked rows (`2 * nrows` rows).
fn realify(
    py: Python<'_>,
    data: &[f64],
    nrows: usize,
    ncomplex: usize,
    stack_imag: bool,
) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let src = Blocks::from_slice(data, nrows, 2 * ncomplex);
    let (out_rows, out_cols) = (if stack_imag { 2 * nrows } else { nrows }, ncomplex);
    let br = block_rows(out_rows);
    let nblocks = if out_rows == 0 || out_cols == 0 {
        0
    } else {
        ceil_div(out_rows, br)
    };
    let blocks = for_each_walker(py, nblocks, |b| {
        let r0 = b * br;
        let r1 = ((b + 1) * br).min(out_rows);
        let mut out = Vec::with_capacity((r1 - r0) * out_cols);
        for r in r0..r1 {
            let (sr, part) = if r < nrows { (r, 0) } else { (r - nrows, 1) };
            out.extend(src.row(sr).iter().skip(part).step_by(2));
        }
        out
    });
    (blocks, br, out_rows, out_cols)
}

/// Which index of the stored matrix the SR operator acts on.
#[derive(Clone, Copy, PartialEq)]
enum Space {
    /// operator `A A^T` on the row index
    Rows,
    /// operator `A^T A` on the column index
    Cols,
}

/// SR operator applications and full passes over an operator matrix (one
/// matrix-vector product of [`Blocks`] or one fused pass), counted since the
/// process started; see [`py_optimizer_sr_operator_counts`].
static OPERATOR_APPLICATIONS: AtomicUsize = AtomicUsize::new(0);
static MATRIX_SWEEPS: AtomicUsize = AtomicUsize::new(0);

/// `(applications, sweeps)` of the SR operators so far in this process: a
/// one-pass operator adds one sweep per application, `A (A^T v)` two.
#[pyfunction]
#[pyo3(name = "optimizer_sr_operator_counts")]
pub fn py_optimizer_sr_operator_counts() -> (usize, usize) {
    (
        OPERATOR_APPLICATIONS.load(Ordering::Relaxed),
        MATRIX_SWEEPS.load(Ordering::Relaxed),
    )
}

fn gram_apply(py: Python<'_>, a: &Blocks<'_>, space: Space, v: &[f64]) -> Vec<f64> {
    OPERATOR_APPLICATIONS.fetch_add(1, Ordering::Relaxed);
    match space {
        Space::Rows => {
            let t = a.atv(py, v);
            a.av(py, &t)
        }
        Space::Cols => a.fused_ata(py, v),
    }
}

fn gram_matrix(py: Python<'_>, a: &Blocks<'_>, space: Space) -> Mat<f64> {
    match space {
        Space::Rows => a.row_gram(py),
        Space::Cols => a.col_gram(py),
    }
}

/// Call the Python reduction wrapper on a float64 vector and return its
/// float64 result, which must keep the length.
fn reduce_vector(
    py: Python<'_>,
    reduce: &Bound<'_, PyAny>,
    values: Vec<f64>,
    what: &str,
) -> PyResult<Vec<f64>> {
    let n = values.len();
    let out = reduce.call1((values.into_pyarray(py),))?;
    let arr: PyReadonlyArrayDyn<'_, f64> = out.extract()?;
    if arr.len() != n || arr.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "{what}: reduction must return a matching float64 vector."
        )));
    }
    Ok(arr.as_array().iter().copied().collect())
}

/// The plain CG iteration of the NumPy reference (break on breakdown).
///
/// `tol` is relative (issue #859): the loop stops once
/// `||r||^2 <= (tol * tol) * ||b||^2`, the rule of the NumPy and XLA CGs.
/// `b` is the full right-hand side of the solved system on every caller
/// (coordinate-replicated for the wide CG, the gathered dual `h` for the
/// tall CG), so `||b||` is the global norm. `b = 0` starts from its exact
/// solution `0`, whatever `x`, so no iteration runs. The returned residual
/// is the absolute `||r||`. The last value is `true` when the loop stopped on a
/// breakdown (a non-finite or `|p^T A p| <= tiny` denominator); the broken-down
/// iteration is not counted, the convention of the NumPy and XLA CGs.
#[allow(clippy::too_many_arguments)]
fn conjugate_gradient(
    b: &[f64],
    mut x: Vec<f64>,
    max_iter: i64,
    tol: f64,
    tiny: f64,
    mut apply: impl FnMut(&[f64]) -> PyResult<Vec<f64>>,
) -> PyResult<(Vec<f64>, f64, i64, bool)> {
    let rhs_sq = dot(b, b);
    if rhs_sq == 0.0 {
        x.iter_mut().for_each(|xi| *xi = 0.0);
    }
    let ax = apply(&x)?;
    let mut residual: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
    let mut direction = residual.clone();
    let mut residual_sq = dot(&residual, &residual);
    let tolerance_sq = (tol * tol) * rhs_sq;
    let mut iterations: i64 = 0;
    let mut breakdown = false;
    while iterations < max_iter && residual_sq > tolerance_sq {
        let applied = apply(&direction)?;
        let denominator = dot(&direction, &applied);
        if !denominator.is_finite() || denominator.abs() <= tiny {
            breakdown = true;
            break;
        }
        let alpha = residual_sq / denominator;
        for (xi, di) in x.iter_mut().zip(&direction) {
            *xi += alpha * di;
        }
        for (ri, ai) in residual.iter_mut().zip(&applied) {
            *ri -= alpha * ai;
        }
        let next_residual_sq = dot(&residual, &residual);
        let beta = if residual_sq > 0.0 {
            next_residual_sq / residual_sq
        } else {
            0.0
        };
        for (di, ri) in direction.iter_mut().zip(&residual) {
            *di = ri + beta * *di;
        }
        residual_sq = next_residual_sq;
        iterations += 1;
    }
    Ok((x, residual_sq.sqrt(), iterations, breakdown))
}

/// LU factor of `gram + epsilon I` and the refined solve against `apply`.
fn refined_lu_solve(
    py: Python<'_>,
    mut operator: Mat<f64>,
    epsilon: f64,
    rhs: &[f64],
    refinement_steps: usize,
    mut apply: impl FnMut(&[f64]) -> PyResult<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let n = operator.nrows();
    for i in 0..n {
        operator[(i, i)] += epsilon;
    }
    let factor = release(py, || PartialPivLu::new(operator.as_ref()));
    let solve = |v: &[f64]| -> Vec<f64> {
        let mut col = Mat::from_fn(n, 1, |i, _| v[i]);
        factor.solve_in_place(col.as_mut());
        (0..n).map(|i| col[(i, 0)]).collect()
    };
    let mut solution = release(py, || solve(rhs));
    for _ in 0..refinement_steps {
        let applied = apply(&solution)?;
        let correction_rhs: Vec<f64> = rhs.iter().zip(&applied).map(|(r, a)| r - a).collect();
        let correction = release(py, || solve(&correction_rhs));
        for (s, c) in solution.iter_mut().zip(&correction) {
            *s += c;
        }
    }
    Ok(solution)
}

/// Stored wide/tall SR matrix: borrowed real data or owned realified blocks.
struct SrMatrix<'a> {
    owned: Option<(Vec<Vec<f64>>, usize)>,
    data: &'a [f64],
    nrows: usize,
    ncols: usize,
}

impl<'a> SrMatrix<'a> {
    fn blocks(&self) -> Blocks<'_> {
        match &self.owned {
            Some((owned, br)) => Blocks::from_owned(owned, *br, self.nrows, self.ncols),
            None => Blocks::from_slice(self.data, self.nrows, self.ncols),
        }
    }
}

/// The real wide-SR matrix of the NumPy reference: `Y = Re X` when `X` is real
/// or has an identically zero imaginary part, else `Y = [Re X | Im X]`.
/// `stored` is the row-major array handed over by Python: `X` itself
/// (`transposed=false`, rows are coordinates) or `X^T` (`transposed=true`,
/// rows are samples), viewed as float64 (interleaved when `complex`).
///
/// A real `X` is used in place. A complex `X` is realified once into owned
/// sample-major rows `Y^T` (one row per sample, real parts then imaginary
/// parts), so the CG operator `Y (Y^T v)` runs as one fused streaming pass.
fn wide_matrix<'a>(
    py: Python<'_>,
    stored: &'a [f64],
    nrows: usize,
    ncols: usize,
    transposed: bool,
    complex: bool,
) -> (SrMatrix<'a>, Space) {
    if !complex {
        let space = if transposed { Space::Cols } else { Space::Rows };
        return (
            SrMatrix {
                owned: None,
                data: stored,
                nrows,
                ncols,
            },
            space,
        );
    }
    let ncomplex = ncols / 2;
    let with_imag = any_imag(py, stored, nrows, ncomplex);
    let (owned, br, r, c) = if transposed {
        realify(py, stored, nrows, ncomplex, with_imag)
    } else {
        transpose_realify(py, stored, nrows, ncomplex, with_imag)
    };
    (
        SrMatrix {
            owned: Some((owned, br)),
            data: &[],
            nrows: r,
            ncols: c,
        },
        Space::Cols,
    )
}

/// Owned sample-major rows of a coordinate-major interleaved complex matrix
/// (`nrows` coordinates x `ncomplex` samples): row `n` holds `Re X[:, n]`, and
/// with `with_imag` rows `ncomplex + n` hold `Im X[:, n]`.
fn transpose_realify(
    py: Python<'_>,
    data: &[f64],
    nrows: usize,
    ncomplex: usize,
    with_imag: bool,
) -> (Vec<Vec<f64>>, usize, usize, usize) {
    let out_rows = if with_imag { 2 * ncomplex } else { ncomplex };
    let out_cols = nrows;
    let br = block_rows(out_rows);
    let nblocks = if out_rows == 0 || out_cols == 0 {
        0
    } else {
        ceil_div(out_rows, br)
    };
    const COLUMN_TILE: usize = 64;
    let blocks = for_each_walker(py, nblocks, |b| {
        let r0 = b * br;
        let r1 = ((b + 1) * br).min(out_rows);
        let mut out = vec![0.0; (r1 - r0) * out_cols];
        for c0 in (0..out_cols).step_by(COLUMN_TILE) {
            let c1 = (c0 + COLUMN_TILE).min(out_cols);
            for r in r0..r1 {
                let (sample, part) = (r % ncomplex, r / ncomplex);
                let dst = &mut out[(r - r0) * out_cols + c0..(r - r0) * out_cols + c1];
                for (k, slot) in dst.iter_mut().enumerate() {
                    *slot = data[(c0 + k) * 2 * ncomplex + 2 * sample + part];
                }
            }
        }
        out
    });
    (blocks, br, out_rows, out_cols)
}

#[pyfunction]
#[pyo3(name = "optimizer_wide_sr_real_cg", signature = (stored, transposed, complex, b, epsilon, max_iter, tol, tiny, x0, reduce))]
#[allow(clippy::too_many_arguments)]
pub fn py_optimizer_wide_sr_real_cg<'py>(
    py: Python<'py>,
    stored: PyReadonlyArray2<'py, f64>,
    transposed: bool,
    complex: bool,
    b: PyReadonlyArray1<'py, f64>,
    epsilon: f64,
    max_iter: i64,
    tol: f64,
    tiny: f64,
    x0: PyReadonlyArray1<'py, f64>,
    reduce: Option<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, f64, i64, bool)> {
    let (nrows, ncols) = (stored.shape()[0], stored.shape()[1]);
    let data = stored.as_slice()?;
    let (matrix, space) = wide_matrix(py, data, nrows, ncols, transposed, complex);
    let blocks = matrix.blocks();
    let b = b.as_slice()?.to_vec();
    let x0 = x0.as_slice()?.to_vec();
    let apply = |v: &[f64]| -> PyResult<Vec<f64>> {
        let local = gram_apply(py, &blocks, space, v);
        let global = match &reduce {
            None => local,
            Some(r) => reduce_vector(py, r, local, "wide real SR CG")?,
        };
        Ok(global
            .iter()
            .zip(v)
            .map(|(g, vi)| g + epsilon * vi)
            .collect())
    };
    let (x, residual, iterations, breakdown) =
        conjugate_gradient(&b, x0, max_iter, tol, tiny, apply)?;
    Ok((x.into_pyarray(py), residual, iterations, breakdown))
}

#[pyfunction]
#[pyo3(name = "optimizer_wide_sr_real_direct", signature = (stored, transposed, complex, b, epsilon, refinement_steps, reduce_metric, reduce))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
pub fn py_optimizer_wide_sr_real_direct<'py>(
    py: Python<'py>,
    stored: PyReadonlyArray2<'py, f64>,
    transposed: bool,
    complex: bool,
    b: PyReadonlyArray1<'py, f64>,
    epsilon: f64,
    refinement_steps: usize,
    reduce_metric: Option<Bound<'py, PyAny>>,
    reduce: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let (nrows, ncols) = (stored.shape()[0], stored.shape()[1]);
    let data = stored.as_slice()?;
    let (matrix, space) = wide_matrix(py, data, nrows, ncols, transposed, complex);
    let blocks = matrix.blocks();
    let b = b.as_slice()?.to_vec();
    let n = b.len();
    let local_metric = gram_matrix(py, &blocks, space);
    let metric = match &reduce_metric {
        None => local_metric,
        Some(r) => {
            let flat: Vec<f64> = (0..n * n).map(|k| local_metric[(k / n, k % n)]).collect();
            let arr = numpy::PyArray1::from_vec(py, flat).reshape([n, n])?;
            let out = r.call1((arr,))?;
            let reduced: PyReadonlyArray2<'_, f64> = out.extract()?;
            if reduced.shape() != [n, n] {
                return Err(PyValueError::new_err(
                    "wide real SR reduction must return a matching real metric.",
                ));
            }
            let view = reduced.as_array();
            Mat::from_fn(n, n, |i, j| view[[i, j]])
        }
    };
    let apply = |v: &[f64]| -> PyResult<Vec<f64>> {
        let local = gram_apply(py, &blocks, space, v);
        let global = match &reduce {
            None => local,
            Some(r) => reduce_vector(py, r, local, "wide real SR direct")?,
        };
        Ok(global
            .iter()
            .zip(v)
            .map(|(g, vi)| g + epsilon * vi)
            .collect())
    };
    let theta = refined_lu_solve(py, metric, epsilon, &b, refinement_steps, apply)?;
    Ok(theta.into_pyarray(py))
}

/// The preconditioned matrix of a wide-only SR system, sample-major: row `n`
/// holds `X_q[active[i], n] * scale[i]` for every `i` (interleaved real and
/// imaginary parts when `complex`). `stored` is `X_q` (`transposed=false`) or
/// `X_q^T` (`transposed=true`) as float64, interleaved for complex128.
///
/// This parallelises the single-threaded NumPy pass
/// `np.take(X_q.T, active, axis=1) * scale` over fixed row blocks on the Rayon
/// pool. Each element is the same product as NumPy's (for complex, NumPy's
/// complex product with `scale + 0j`), so the result is bit-identical. The
/// caller allocates `out` (samples x width * active, C order) with NumPy, so
/// the workers fill an untouched buffer instead of one the allocator has
/// zero-filled serially on the calling thread.
#[pyfunction]
#[pyo3(name = "optimizer_sr_precondition_sample_major")]
pub fn py_optimizer_sr_precondition_sample_major<'py>(
    py: Python<'py>,
    stored: PyReadonlyArray2<'py, f64>,
    transposed: bool,
    complex: bool,
    active: PyReadonlyArray1<'py, i64>,
    scale: PyReadonlyArray1<'py, f64>,
    mut out: numpy::PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let (srows, scols) = (stored.shape()[0], stored.shape()[1]);
    let data = stored.as_slice()?;
    let active = active.as_slice()?;
    let scale = scale.as_slice()?;
    let width = if complex { 2 } else { 1 };
    let (coordinates, samples) = if transposed {
        (scols / width, srows)
    } else {
        (srows, scols / width)
    };
    if active.len() != scale.len() || active.iter().any(|&q| q < 0 || q as usize >= coordinates) {
        return Err(PyValueError::new_err(
            "SR preconditioning requires matching in-range active indices and scales.",
        ));
    }
    let ncols = width * active.len();
    if out.shape() != [samples, ncols] {
        return Err(PyValueError::new_err(
            "SR preconditioning output must be (samples, width * active).",
        ));
    }
    let out = out.as_slice_mut()?;
    if ncols > 0 && samples > 0 {
        let br = block_rows(samples);
        let blocks: Vec<Mutex<&mut [f64]>> = out.chunks_mut(br * ncols).map(Mutex::new).collect();
        for_each_walker(py, blocks.len(), |b| {
            let mut block = blocks[b].lock().expect("row block lock");
            for (k, row) in block.chunks_mut(ncols).enumerate() {
                let n = b * br + k;
                for (i, (&q, &f)) in active.iter().zip(scale).enumerate() {
                    let src = if transposed {
                        n * scols + width * q as usize
                    } else {
                        q as usize * scols + width * n
                    };
                    if complex {
                        let (re, im) = (data[src], data[src + 1]);
                        row[2 * i] = re * f - im * 0.0;
                        row[2 * i + 1] = re * 0.0 + im * f;
                    } else {
                        row[i] = data[src] * f;
                    }
                }
            }
        });
    }
    Ok(())
}

/// Owned row blocks of `A^T` for a row-major `A` (nrows x ncols).
fn transpose_copy(
    py: Python<'_>,
    data: &[f64],
    nrows: usize,
    ncols: usize,
) -> (Vec<Vec<f64>>, usize) {
    let br = block_rows(ncols);
    let nblocks = if nrows == 0 || ncols == 0 {
        0
    } else {
        ceil_div(ncols, br)
    };
    const ROW_TILE: usize = 64;
    let blocks = for_each_walker(py, nblocks, |b| {
        let c0 = b * br;
        let c1 = ((b + 1) * br).min(ncols);
        let width = c1 - c0;
        let mut out = vec![0.0; width * nrows];
        for r0 in (0..nrows).step_by(ROW_TILE) {
            let r1 = (r0 + ROW_TILE).min(nrows);
            for c in 0..width {
                let dst = &mut out[c * nrows + r0..c * nrows + r1];
                for (k, slot) in dst.iter_mut().enumerate() {
                    *slot = data[(r0 + k) * ncols + c0 + c];
                }
            }
        }
        out
    });
    (blocks, br)
}

/// Tall SR: `stored` is `Z` (rows are coordinates) or `Z^T` (rows are samples).
fn tall_parts<'a>(
    stored: &'a [f64],
    nrows: usize,
    ncols: usize,
    transposed: bool,
) -> (Blocks<'a>, Space) {
    // The dual operator Z^T Z acts on the sample index.
    (
        Blocks::from_slice(stored, nrows, ncols),
        if transposed { Space::Rows } else { Space::Cols },
    )
}

/// `q = Z dual` from the dual (sample-space) vector.
fn tall_primal(py: Python<'_>, a: &Blocks<'_>, space: Space, dual: &[f64]) -> Vec<f64> {
    match space {
        Space::Cols => a.av(py, dual),
        Space::Rows => a.atv(py, dual),
    }
}

#[pyfunction]
#[pyo3(name = "optimizer_tall_sr_real_cg")]
#[allow(clippy::too_many_arguments)]
pub fn py_optimizer_tall_sr_real_cg<'py>(
    py: Python<'py>,
    stored: PyReadonlyArray2<'py, f64>,
    transposed: bool,
    h: PyReadonlyArray1<'py, f64>,
    epsilon: f64,
    max_iter: i64,
    tol: f64,
    tiny: f64,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, f64, i64, bool)> {
    let (nrows, ncols) = (stored.shape()[0], stored.shape()[1]);
    let data = stored.as_slice()?;
    // The CG operator Z^T Z runs as one fused streaming pass over the rows of
    // a row-major Z; a sample-major (Fortran) Z is transposed once into owned
    // row blocks first.
    let owned = if transposed {
        Some(transpose_copy(py, data, nrows, ncols))
    } else {
        None
    };
    let blocks = match &owned {
        Some((rows, br)) => Blocks::from_owned(rows, *br, ncols, nrows),
        None => Blocks::from_slice(data, nrows, ncols),
    };
    let space = Space::Cols;
    let h = h.as_slice()?.to_vec();
    let apply = |v: &[f64]| -> PyResult<Vec<f64>> {
        let local = gram_apply(py, &blocks, space, v);
        Ok(local
            .iter()
            .zip(v)
            .map(|(g, vi)| g + epsilon * vi)
            .collect())
    };
    let dual0 = vec![0.0; h.len()];
    let (dual, residual, iterations, breakdown) =
        conjugate_gradient(&h, dual0, max_iter, tol, tiny, apply)?;
    Ok((
        tall_primal(py, &blocks, space, &dual).into_pyarray(py),
        residual,
        iterations,
        breakdown,
    ))
}

#[pyfunction]
#[pyo3(name = "optimizer_tall_sr_real_direct")]
pub fn py_optimizer_tall_sr_real_direct<'py>(
    py: Python<'py>,
    stored: PyReadonlyArray2<'py, f64>,
    transposed: bool,
    h: PyReadonlyArray1<'py, f64>,
    epsilon: f64,
    refinement_steps: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let (nrows, ncols) = (stored.shape()[0], stored.shape()[1]);
    let (blocks, space) = tall_parts(stored.as_slice()?, nrows, ncols, transposed);
    let h = h.as_slice()?.to_vec();
    let gram = gram_matrix(py, &blocks, space);
    let apply = |v: &[f64]| -> PyResult<Vec<f64>> {
        let local = gram_apply(py, &blocks, space, v);
        Ok(local
            .iter()
            .zip(v)
            .map(|(g, vi)| g + epsilon * vi)
            .collect())
    };
    let dual = refined_lu_solve(py, gram, epsilon, &h, refinement_steps, apply)?;
    Ok(tall_primal(py, &blocks, space, &dual).into_pyarray(py))
}

/// Centered planar copies of an (n x q) real or interleaved-complex matrix:
/// `(Re(X - mean), Im(X - mean))` as row blocks of the shared blocking.
fn center_planar(
    py: Python<'_>,
    data: &[f64],
    n: usize,
    q: usize,
    complex: bool,
    mean: &[f64],
) -> (Vec<Vec<f64>>, Option<Vec<Vec<f64>>>) {
    let width = if complex { 2 * q } else { q };
    let src = Blocks::from_slice(data, n, width);
    let parts = for_each_walker(py, src.blocks.len(), |b| {
        let block = src.blocks[b];
        let rows = block.len() / width.max(1);
        let mut re = Vec::with_capacity(rows * q);
        let mut im = if complex {
            Vec::with_capacity(rows * q)
        } else {
            Vec::new()
        };
        for r in 0..rows {
            let row = &block[r * width..(r + 1) * width];
            if complex {
                for a in 0..q {
                    re.push(row[2 * a] - mean[2 * a]);
                    im.push(row[2 * a + 1] - mean[2 * a + 1]);
                }
            } else {
                for a in 0..q {
                    re.push(row[a] - mean[a]);
                }
            }
        }
        (re, im)
    });
    let mut re_blocks = Vec::with_capacity(parts.len());
    let mut im_blocks = Vec::with_capacity(parts.len());
    for (re, im) in parts {
        re_blocks.push(re);
        im_blocks.push(im);
    }
    (re_blocks, if complex { Some(im_blocks) } else { None })
}

fn reduce_or_identity(
    py: Python<'_>,
    reduce: &Option<Bound<'_, PyAny>>,
    values: Vec<f64>,
    what: &str,
) -> PyResult<Vec<f64>> {
    match reduce {
        None => Ok(values),
        Some(r) => reduce_vector(py, r, values, what),
    }
}

fn mat_to_vec(m: &Mat<f64>) -> Vec<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    (0..r * c).map(|k| m[(k / c, k % c)]).collect()
}

fn vec_to_mat(v: &[f64], r: usize, c: usize) -> Mat<f64> {
    Mat::from_fn(r, c, |i, j| v[i * c + j])
}

type PencilOut<'py> = (
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    f64,
    Bound<'py, numpy::PyArray2<f64>>,
);

/// The centered real LM pencil of `_build_lm_real_pencil_from_coordinate_arrays`.
///
/// `o`/`d` are (n x q) row-major, interleaved (n x 2q) when complex; `e` is
/// (n) or interleaved (2n). Every reduction happens at the same point and in
/// the same order as in the NumPy implementation; the complex moments enter
/// the real pencil only through their real parts, so those are reduced.
#[pyfunction]
#[pyo3(name = "optimizer_lm_pencil_core", signature = (o, o_complex, d, d_complex, weights, energies, e_complex, q, reduce_total, reduce))]
#[allow(clippy::too_many_arguments)]
pub fn py_optimizer_lm_pencil_core<'py>(
    py: Python<'py>,
    o: PyReadonlyArray2<'py, f64>,
    o_complex: bool,
    d: PyReadonlyArray2<'py, f64>,
    d_complex: bool,
    weights: PyReadonlyArray1<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    e_complex: bool,
    q: usize,
    reduce_total: Option<Bound<'py, PyAny>>,
    reduce: Option<Bound<'py, PyAny>>,
) -> PyResult<PencilOut<'py>> {
    let w = weights.as_slice()?;
    let n = w.len();
    let o_data = o.as_slice()?;
    let d_data = d.as_slice()?;
    let e_data = energies.as_slice()?;
    let (er, ei): (Vec<f64>, Option<Vec<f64>>) = if e_complex {
        (
            e_data.iter().step_by(2).copied().collect(),
            Some(e_data.iter().skip(1).step_by(2).copied().collect()),
        )
    } else {
        (e_data.to_vec(), None)
    };
    let o_width = if o_complex { 2 * q } else { q };
    let d_width = if d_complex { 2 * q } else { q };

    let mut local_total = 0.0;
    for wi in w {
        local_total += wi;
    }
    let total_weight = match &reduce_total {
        None => local_total,
        Some(r) => r.call1((local_total,))?.extract::<f64>()?,
    };
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(PyValueError::new_err(
            "LM total weight must be a positive finite scalar.",
        ));
    }
    let o_blocks = Blocks::from_slice(o_data, n, o_width);
    let d_blocks = Blocks::from_slice(d_data, n, d_width);
    let weighted_o = reduce_or_identity(py, &reduce, o_blocks.atv(py, w), "LM weighted O")?;
    let weighted_d = reduce_or_identity(py, &reduce, d_blocks.atv(py, w), "LM weighted D")?;
    let mut we = vec![dot(w, &er)];
    if let Some(ei) = &ei {
        we.push(dot(w, ei));
    }
    let weighted_e = reduce_or_identity(py, &reduce, we, "LM weighted e")?;
    let o_bar: Vec<f64> = weighted_o.iter().map(|v| v / total_weight).collect();
    let d_bar: Vec<f64> = weighted_d.iter().map(|v| v / total_weight).collect();
    let e_bar_re = weighted_e[0] / total_weight;
    let e_bar_im = if e_complex {
        weighted_e[1] / total_weight
    } else {
        0.0
    };

    let (dor_owned, doi_owned) = center_planar(py, o_data, n, q, o_complex, &o_bar);
    let (ddr_owned, ddi_owned) = center_planar(py, d_data, n, q, d_complex, &d_bar);
    let br = block_rows(n);
    let dor = Blocks::from_owned(&dor_owned, br, n, q);
    let doi = doi_owned.as_ref().map(|b| Blocks::from_owned(b, br, n, q));
    let ddr = Blocks::from_owned(&ddr_owned, br, n, q);
    let ddi = ddi_owned.as_ref().map(|b| Blocks::from_owned(b, br, n, q));
    let de_r: Vec<f64> = er.iter().map(|v| v - e_bar_re).collect();
    let de_i: Option<Vec<f64>> = ei
        .as_ref()
        .map(|ei| ei.iter().map(|v| v - e_bar_im).collect());

    // S_c = <conj(dO) dO^T>
    let mut s_c = weighted_cross(py, &dor, &dor, w);
    if let Some(doi) = &doi {
        s_c += weighted_cross(py, doi, doi, w);
    }
    let s_c = reduce_or_identity(py, &reduce, mat_to_vec(&s_c), "LM S")?;
    // f_c = -2 <conj(dO) de>, f_bra = -2 <dO de>
    let w_der: Vec<f64> = w.iter().zip(&de_r).map(|(a, b)| a * b).collect();
    let cov_re = dor.atv(py, &w_der);
    let cov_im = match (&doi, &de_i) {
        (Some(doi), Some(de_i)) => {
            let w_dei: Vec<f64> = w.iter().zip(de_i).map(|(a, b)| a * b).collect();
            Some(doi.atv(py, &w_dei))
        }
        _ => None,
    };
    let f_c_local: Vec<f64> = (0..q)
        .map(|a| cov_re[a] + cov_im.as_ref().map_or(0.0, |c| c[a]))
        .collect();
    let f_c: Vec<f64> = reduce_or_identity(py, &reduce, f_c_local, "LM force")?
        .iter()
        .map(|v| -2.0 * v / total_weight)
        .collect();
    // K_c = <e conj(dO) dO^T>
    let w_er: Vec<f64> = w.iter().zip(&er).map(|(a, b)| a * b).collect();
    let mut k_c = weighted_cross(py, &dor, &dor, &w_er);
    if let Some(doi) = &doi {
        k_c += weighted_cross(py, doi, doi, &w_er);
        if let Some(ei) = &ei {
            let w_ei: Vec<f64> = w.iter().zip(ei).map(|(a, b)| a * b).collect();
            k_c = k_c - weighted_cross(py, &dor, doi, &w_ei) + weighted_cross(py, doi, &dor, &w_ei);
        }
    }
    let k_c = reduce_or_identity(py, &reduce, mat_to_vec(&k_c), "LM K")?;
    // B_c = <conj(dD) dO^T>
    let mut b_c = weighted_cross(py, &ddr, &dor, w);
    if let (Some(ddi), Some(doi)) = (&ddi, &doi) {
        b_c += weighted_cross(py, ddi, doi, w);
    }
    let b_c = reduce_or_identity(py, &reduce, mat_to_vec(&b_c), "LM B")?;
    let f_bra_local: Vec<f64> = (0..q)
        .map(|a| cov_re[a] - cov_im.as_ref().map_or(0.0, |c| c[a]))
        .collect();
    let f_bra: Vec<f64> = reduce_or_identity(py, &reduce, f_bra_local, "LM bra force")?
        .iter()
        .map(|v| -2.0 * v / total_weight)
        .collect();

    let s_c = vec_to_mat(&s_c, q, q);
    let k_c = vec_to_mat(&k_c, q, q);
    let b_c = vec_to_mat(&b_c, q, q);
    let h_0 = e_bar_re;
    let size = q + 1;
    let d_bar_re = |a: usize| if d_complex { d_bar[2 * a] } else { d_bar[a] };
    let h = Mat::from_fn(size, size, |i, j| match (i, j) {
        (0, 0) => h_0,
        (i, 0) => -0.5 * f_c[i - 1],
        (0, j) => -0.5 * f_bra[j - 1] + d_bar_re(j - 1),
        (i, j) => k_c[(i - 1, j - 1)] / total_weight + b_c[(j - 1, i - 1)] / total_weight,
    });
    let s_div = Mat::from_fn(q, q, |i, j| s_c[(i, j)] / total_weight);
    let s = Mat::from_fn(size, size, |i, j| match (i, j) {
        (0, 0) => 1.0,
        (0, _) | (_, 0) => 0.0,
        (i, j) => 0.5 * (s_div[(i - 1, j - 1)] + s_div[(j - 1, i - 1)]),
    });
    let metric = Mat::from_fn(q, q, |i, j| 0.5 * (s_div[(i, j)] + s_div[(j, i)]));
    let to_py = |m: &Mat<f64>| -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        numpy::PyArray1::from_vec(py, mat_to_vec(m)).reshape([m.nrows(), m.ncols()])
    };
    Ok((to_py(&h)?, to_py(&s)?, h_0, to_py(&metric)?))
}
