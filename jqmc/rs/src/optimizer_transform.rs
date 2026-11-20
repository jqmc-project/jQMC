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

//! Host optimizer coordinate transform and coordinate-force jackknife (issue #841).
//!
//! Rust ports of two host (CPU/NumPy) functions of `jqmc/jqmc_optimizer.py`:
//!
//! * `OptimizationCoordinateMap.transform_derivatives` -> [`py_optimizer_transform_derivatives`]:
//!   exact derivative sources (`direct`, or Wirtinger `d_p`/`d_pbar`) are mapped to the
//!   directional columns of every real optimization coordinate, through the
//!   Euclidean field adjoint, the linear-tie (J3 / tied lambda) COO adjoint, and the
//!   S-Stiefel row projection adjoint. The JAX twin is `transform_derivatives_device`.
//! * `estimate_coordinate_force_jackknife` -> [`py_optimizer_coordinate_force_jackknife`]:
//!   the real-coordinate force and its delete-one-bin jackknife standard error. The JAX
//!   twin is `estimate_coordinate_force_jackknife_device`.
//!
//! MPI stays in Python: the jackknife receives the scalar/array SUM reductions as Python
//! callables and calls them (with the GIL) at the same collective points, in the same
//! order, as the NumPy implementation. Rows (samples / bins) are processed in fixed-size
//! chunks through [`for_each_walker`], so results are bit-identical for any Rayon width.
//!
//! Real wave functions: a block whose field and metric data are real ("real structure")
//! never forms the imaginary covector branch. For such a block the exact columns are
//! `Adj(Re p) + i Adj(Im p)` with the real-linear adjoint `Adj` (the S-Stiefel adjoint
//! runs in f64). When every block is real structure with a real source and no `d_pbar`,
//! the Python caller selects the real route: the output is float64 and `Adj(Im p)` is
//! never computed. Otherwise the output is complex128 and a real source of a
//! real-structure block writes an exactly zero imaginary part.

use crate::walker_batch::for_each_walker;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use numpy::{
    Complex64 as C64, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Mutex;

type PyVecPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Fixed sample rows per parallel task. Independent of the Rayon width, so the
/// per-row arithmetic (including the faer GEMM shapes) never depends on it.
const ROW_CHUNK: usize = 32;
/// Fixed coordinate columns per parallel jackknife task.
const COLUMN_CHUNK: usize = 256;

const ZERO: C64 = C64 { re: 0.0, im: 0.0 };

/// Scalar field of the S-Stiefel kernels: real (`f64`) or complex (`C64`).
trait Field:
    faer::traits::ComplexField
    + Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn cj(self) -> Self;
    fn re_part(self) -> f64;
    fn from_real(x: f64) -> Self;
}

impl Field for f64 {
    #[inline]
    fn cj(self) -> Self {
        self
    }
    #[inline]
    fn re_part(self) -> f64 {
        self
    }
    #[inline]
    fn from_real(x: f64) -> Self {
        x
    }
}

impl Field for C64 {
    #[inline]
    fn cj(self) -> Self {
        self.conj()
    }
    #[inline]
    fn re_part(self) -> f64 {
        self.re
    }
    #[inline]
    fn from_real(x: f64) -> Self {
        C64::new(x, 0.0)
    }
}

/// `dst = lhs @ rhs` (or `dst += alpha * lhs @ rhs`), sequential faer GEMM.
fn gemm<T, L, R>(dst: MatMut<'_, T>, accum: Accum, lhs: MatRef<'_, L>, rhs: MatRef<'_, R>, alpha: T)
where
    T: Field,
    L: faer::traits::Conjugate<Canonical = T>,
    R: faer::traits::Conjugate<Canonical = T>,
{
    matmul(dst, accum, lhs, rhs, alpha, Par::Seq);
}

/// Per-block constants of the S-Stiefel row adjoint
/// `u -> P_W(u @ inv_sqrt^T) @ sqrt^T` with the whitened rows `W = C @ sqrt^T`.
struct Stiefel<T: Field> {
    rows: usize,
    cols: usize,
    /// whitened coefficients W (rows x cols), row-major
    whitened: Vec<T>,
    /// sqrt (cols x cols), row-major; used transposed
    sqrt: Vec<T>,
    /// inv_sqrt (cols x cols), row-major; used transposed
    inv_sqrt: Vec<T>,
    /// eigenvectors V of the Hermitian row Gram (rows x rows), row-major
    eigenvectors: Vec<T>,
    eigenvalues: Vec<f64>,
}

impl<T: Field> Stiefel<T> {
    /// Pure Rust: the `#[test]`s call it, and their executable has no libpython to
    /// link (issue #901). A failure is a message that [`parse_blocks`] raises as
    /// `FloatingPointError`.
    fn build(
        values: &[T],
        sqrt: &[T],
        inv_sqrt: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let one = T::from_real(1.0);
        let mut whitened = vec![T::from_real(0.0); rows * cols];
        gemm(
            MatMut::from_row_major_slice_mut(&mut whitened, rows, cols),
            Accum::Replace,
            MatRef::from_row_major_slice(values, rows, cols),
            MatRef::from_row_major_slice(sqrt, cols, cols).transpose(),
            one,
        );
        let mut gram = vec![T::from_real(0.0); rows * rows];
        let w = MatRef::from_row_major_slice(&whitened, rows, cols);
        gemm(
            MatMut::from_row_major_slice_mut(&mut gram, rows, rows),
            Accum::Replace,
            w,
            w.adjoint(),
            one,
        );
        // 0.5 * (G + G^H), as the NumPy implementation symmetrizes before eigh.
        let mut sym = vec![T::from_real(0.0); rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                sym[i * rows + j] =
                    (gram[i * rows + j] + gram[j * rows + i].cj()) * T::from_real(0.5);
            }
        }
        let mut eigenvalues_t = vec![T::from_real(0.0); rows];
        let mut eigenvectors = vec![T::from_real(0.0); rows * rows];
        {
            let par = Par::Seq;
            let mut buffer =
                faer::dyn_stack::MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<T>(
                    rows,
                    faer::linalg::evd::ComputeEigenvectors::Yes,
                    par,
                    Default::default(),
                ));
            faer::linalg::evd::self_adjoint_evd(
                MatRef::from_row_major_slice(&sym, rows, rows),
                faer::diag::DiagMut::from_slice_mut(&mut eigenvalues_t),
                Some(MatMut::from_row_major_slice_mut(
                    &mut eigenvectors,
                    rows,
                    rows,
                )),
                par,
                faer::dyn_stack::MemStack::new(&mut buffer),
                Default::default(),
            )
            .map_err(|_| "S-Stiefel row Gram eigendecomposition did not converge.".to_string())?;
        }
        let eigenvalues: Vec<f64> = eigenvalues_t.iter().map(|value| value.re_part()).collect();
        if eigenvalues.iter().any(|&value| value <= 0.0) {
            return Err("S-Stiefel row Gram must be positive definite.".to_string());
        }
        Ok(Self {
            rows,
            cols,
            whitened,
            sqrt: sqrt.to_vec(),
            inv_sqrt: inv_sqrt.to_vec(),
            eigenvectors,
            eigenvalues,
        })
    }

    /// Apply the ambient adjoint to `samples` stacked samples `x` (samples*rows x cols,
    /// row-major) and write `samples*rows x cols` row-major values into `out`.
    fn adjoint(&self, x: &[T], samples: usize, out: &mut [T]) {
        let (p, n) = (self.rows, self.cols);
        let stacked = samples * p;
        let one = T::from_real(1.0);
        let zero = T::from_real(0.0);
        let half = T::from_real(0.5);
        let mut z = vec![zero; stacked * n];
        gemm(
            MatMut::from_row_major_slice_mut(&mut z, stacked, n),
            Accum::Replace,
            MatRef::from_row_major_slice(x, stacked, n),
            MatRef::from_row_major_slice(&self.inv_sqrt, n, n).transpose(),
            one,
        );
        let w = MatRef::from_row_major_slice(&self.whitened, p, n);
        let mut h = vec![zero; stacked * p];
        gemm(
            MatMut::from_row_major_slice_mut(&mut h, stacked, p),
            Accum::Replace,
            MatRef::from_row_major_slice(&z, stacked, n),
            w.adjoint(),
            one,
        );
        let v = MatRef::from_row_major_slice(&self.eigenvectors, p, p);
        let mut rhs = vec![zero; p * p];
        let mut tmp = vec![zero; p * p];
        let mut rotated = vec![zero; p * p];
        let mut multiplier = vec![zero; p * p];
        let mut hermitian = vec![zero; p * p];
        for k in 0..samples {
            let hk = &h[k * p * p..(k + 1) * p * p];
            // rhs = Z W^H + W Z^H, already Hermitian (its 0.5 (rhs + rhs^H) is exact).
            for i in 0..p {
                for j in 0..p {
                    rhs[i * p + j] = hk[i * p + j] + hk[j * p + i].cj();
                }
            }
            gemm(
                MatMut::from_row_major_slice_mut(&mut tmp, p, p),
                Accum::Replace,
                v.adjoint(),
                MatRef::from_row_major_slice(&rhs, p, p),
                one,
            );
            gemm(
                MatMut::from_row_major_slice_mut(&mut rotated, p, p),
                Accum::Replace,
                MatRef::from_row_major_slice(&tmp, p, p),
                v,
                one,
            );
            for i in 0..p {
                for j in 0..p {
                    rotated[i * p + j] = rotated[i * p + j]
                        / T::from_real(self.eigenvalues[i] + self.eigenvalues[j]);
                }
            }
            gemm(
                MatMut::from_row_major_slice_mut(&mut tmp, p, p),
                Accum::Replace,
                v,
                MatRef::from_row_major_slice(&rotated, p, p),
                one,
            );
            gemm(
                MatMut::from_row_major_slice_mut(&mut multiplier, p, p),
                Accum::Replace,
                MatRef::from_row_major_slice(&tmp, p, p),
                v.adjoint(),
                one,
            );
            for i in 0..p {
                for j in 0..p {
                    hermitian[i * p + j] =
                        (multiplier[i * p + j] + multiplier[j * p + i].cj()) * half;
                }
            }
            // Z_k <- Z_k - M W
            gemm(
                MatMut::from_row_major_slice_mut(&mut z[k * p * n..(k + 1) * p * n], p, n),
                Accum::Add,
                MatRef::from_row_major_slice(&hermitian, p, p),
                w,
                -one,
            );
        }
        gemm(
            MatMut::from_row_major_slice_mut(out, stacked, n),
            Accum::Replace,
            MatRef::from_row_major_slice(&z, stacked, n),
            MatRef::from_row_major_slice(&self.sqrt, n, n).transpose(),
            one,
        );
    }
}

/// Owned adjoint data of one block kind.
enum Kind {
    Euclidean,
    /// Linear tie as CSR by coordinate: `offsets[q]..offsets[q + 1]` index
    /// `outputs`/`coefficients` in published COO order.
    Tie {
        offsets: Vec<usize>,
        outputs: Vec<usize>,
        coefficients: Vec<C64>,
    },
    StiefelReal(Stiefel<f64>),
    StiefelComplex(Stiefel<C64>),
}

/// Borrowed row-major `(samples, size)` source component.
#[derive(Clone, Copy)]
enum Rows<'a> {
    Real(&'a [f64]),
    Complex(&'a [C64]),
}

impl Rows<'_> {
    /// Element `index` as seen by the transform (`conj` applies the source conjugation).
    #[inline]
    fn get(&self, index: usize, conj: bool) -> C64 {
        match self {
            Rows::Real(values) => C64::new(values[index], 0.0),
            Rows::Complex(values) => {
                if conj {
                    values[index].conj()
                } else {
                    values[index]
                }
            }
        }
    }
}

struct Block<'a> {
    kind: Kind,
    field_complex: bool,
    coordinate_start: usize,
    coordinate_size: usize,
    size: usize,
    primary: Rows<'a>,
    secondary: Option<&'a [C64]>,
}

impl Block<'_> {
    /// Real-structure blocks: real field and real metric data. Their adjoint is
    /// real-linear with real coefficients, so the imaginary covector branch is
    /// `Adj(Im p)` and never needs the complex covectors.
    fn real_structure(&self) -> bool {
        !self.field_complex && !matches!(self.kind, Kind::StiefelComplex(_))
    }
}

/// Covectors of one sample: `u_r = conj(p) [+ s]`, `u_i = i (conj(p) [- s])`.
#[inline]
fn covectors(primary: C64, secondary: Option<C64>) -> (C64, C64) {
    let conj_primary = primary.conj();
    let i = C64::new(0.0, 1.0);
    match secondary {
        None => (conj_primary, i * conj_primary),
        Some(s) => (conj_primary + s, i * (conj_primary - s)),
    }
}

/// Compute rows `start..start + count` of a real-structure block with a real source
/// `values` into `out` (row stride `width`). The columns are `Adj(p)` of the real-linear
/// adjoint; the imaginary covector branch is never formed. `T = f64` is the real route
/// of [`py_optimizer_transform_derivatives`]; `T = C64` writes the same values with a
/// zero imaginary part on the complex route.
fn real_block_rows<T: Field>(
    block: &Block<'_>,
    values: &[f64],
    start: usize,
    count: usize,
    out: &mut [T],
    width: usize,
) {
    let size = block.size;
    let c0 = block.coordinate_start;
    match &block.kind {
        Kind::Euclidean => {
            for r in 0..count {
                let p = &values[(start + r) * size..(start + r + 1) * size];
                let row = &mut out[r * width..(r + 1) * width];
                for k in 0..size {
                    row[c0 + k] = T::from_real(p[k]);
                }
            }
        }
        Kind::Tie {
            offsets,
            outputs,
            coefficients,
        } => {
            for r in 0..count {
                let p = &values[(start + r) * size..(start + r + 1) * size];
                let row = &mut out[r * width..(r + 1) * width];
                for q in 0..block.coordinate_size {
                    let mut acc = 0.0f64;
                    for e in offsets[q]..offsets[q + 1] {
                        acc += coefficients[e].re * p[outputs[e]];
                    }
                    row[c0 + q] = T::from_real(acc);
                }
            }
        }
        Kind::StiefelReal(stiefel) => {
            let mut adjoint = vec![0.0f64; count * size];
            stiefel.adjoint(
                &values[start * size..(start + count) * size],
                count,
                &mut adjoint,
            );
            for r in 0..count {
                let row = &mut out[r * width..(r + 1) * width];
                for k in 0..size {
                    row[c0 + k] = T::from_real(adjoint[r * size + k]);
                }
            }
        }
        Kind::StiefelComplex(_) => {
            unreachable!("a complex-metric S-Stiefel block is never real structure")
        }
    }
}

/// Compute rows `start..start + count` of one block into `out` (row stride `width`).
fn transform_block_rows(
    block: &Block<'_>,
    conj: bool,
    start: usize,
    count: usize,
    out: &mut [C64],
    width: usize,
) {
    if let (true, Rows::Real(values)) = (block.real_structure(), block.primary) {
        real_block_rows(block, values, start, count, out, width);
        return;
    }
    let size = block.size;
    let c0 = block.coordinate_start;
    let secondary_at = |index: usize| -> Option<C64> {
        block.secondary.map(|values| {
            if conj {
                values[index].conj()
            } else {
                values[index]
            }
        })
    };
    match &block.kind {
        Kind::Euclidean => {
            for r in 0..count {
                let base = (start + r) * size;
                let row = &mut out[r * width..(r + 1) * width];
                if block.real_structure() {
                    for k in 0..size {
                        row[c0 + k] = block.primary.get(base + k, conj);
                    }
                } else {
                    for k in 0..size {
                        let (ur, ui) =
                            covectors(block.primary.get(base + k, conj), secondary_at(base + k));
                        row[c0 + 2 * k] = C64::new(ur.re, ui.re);
                        row[c0 + 2 * k + 1] = C64::new(ur.im, ui.im);
                    }
                }
            }
        }
        Kind::Tie {
            offsets,
            outputs,
            coefficients,
        } => {
            let real = block.real_structure();
            let mut ur = vec![ZERO; if real { 0 } else { size }];
            let mut ui = vec![ZERO; if real { 0 } else { size }];
            for r in 0..count {
                let base = (start + r) * size;
                let row = &mut out[r * width..(r + 1) * width];
                if real {
                    // Complex source on a real-structure tie: Adj(Re p) + i Adj(Im p).
                    for q in 0..block.coordinate_size {
                        let (mut acc_re, mut acc_im) = (0.0f64, 0.0f64);
                        for e in offsets[q]..offsets[q + 1] {
                            let value = block.primary.get(base + outputs[e], conj);
                            acc_re += coefficients[e].re * value.re;
                            acc_im += coefficients[e].re * value.im;
                        }
                        row[c0 + q] = C64::new(acc_re, acc_im);
                    }
                } else {
                    for k in 0..size {
                        let (a, b) =
                            covectors(block.primary.get(base + k, conj), secondary_at(base + k));
                        ur[k] = a;
                        ui[k] = b;
                    }
                    for q in 0..block.coordinate_size {
                        let (mut acc_re, mut acc_im) = (0.0f64, 0.0f64);
                        for e in offsets[q]..offsets[q + 1] {
                            let c = coefficients[e].conj();
                            acc_re += (c * ur[outputs[e]]).re;
                            acc_im += (c * ui[outputs[e]]).re;
                        }
                        row[c0 + q] = C64::new(acc_re, acc_im);
                    }
                }
            }
        }
        Kind::StiefelReal(stiefel) => {
            // Real structure with a complex source: columns = Adj(Re p) + i Adj(Im p).
            let n_values = count * size;
            let mut part_re = vec![0.0f64; n_values];
            let mut part_im = vec![0.0f64; n_values];
            for index in 0..n_values {
                let value = block.primary.get(start * size + index, conj);
                part_re[index] = value.re;
                part_im[index] = value.im;
            }
            let mut adj_re = vec![0.0f64; n_values];
            let mut adj_im = vec![0.0f64; n_values];
            stiefel.adjoint(&part_re, count, &mut adj_re);
            stiefel.adjoint(&part_im, count, &mut adj_im);
            for r in 0..count {
                let row = &mut out[r * width..(r + 1) * width];
                for k in 0..size {
                    row[c0 + k] = C64::new(adj_re[r * size + k], adj_im[r * size + k]);
                }
            }
        }
        Kind::StiefelComplex(stiefel) => {
            let n_values = count * size;
            let mut ur = vec![ZERO; n_values];
            let mut ui = vec![ZERO; n_values];
            for index in 0..n_values {
                let global = start * size + index;
                let (a, b) = covectors(block.primary.get(global, conj), secondary_at(global));
                ur[index] = a;
                ui[index] = b;
            }
            let mut ar = vec![ZERO; n_values];
            let mut ai = vec![ZERO; n_values];
            stiefel.adjoint(&ur, count, &mut ar);
            stiefel.adjoint(&ui, count, &mut ai);
            for r in 0..count {
                let row = &mut out[r * width..(r + 1) * width];
                for k in 0..size {
                    let (a, b) = (ar[r * size + k], ai[r * size + k]);
                    if block.field_complex {
                        row[c0 + 2 * k] = C64::new(a.re, b.re);
                        row[c0 + 2 * k + 1] = C64::new(a.im, b.im);
                    } else {
                        row[c0 + k] = C64::new(a.re, b.re);
                    }
                }
            }
        }
    }
}

/// Run `rows(start, count, chunk)` over fixed `ROW_CHUNK`-row chunks of the row-major
/// `(samples, width)` output in parallel.
fn for_each_row_chunk<T: Send>(
    py: Python<'_>,
    samples: usize,
    width: usize,
    out: &mut [T],
    rows: impl Fn(usize, usize, &mut [T]) + Sync + Send,
) {
    if samples == 0 || width == 0 {
        return;
    }
    let chunks: Vec<Mutex<&mut [T]>> = out.chunks_mut(ROW_CHUNK * width).map(Mutex::new).collect();
    let n_chunks = chunks.len();
    for_each_walker(py, n_chunks, |chunk| {
        let mut guard = chunks[chunk].lock().expect("row chunk lock");
        let start = chunk * ROW_CHUNK;
        rows(start, ROW_CHUNK.min(samples - start), &mut guard[..]);
    });
}

/// Transform all blocks for `samples` rows into the row-major complex `(samples, width)` output.
fn transform_into(
    py: Python<'_>,
    blocks: &[Block<'_>],
    conj: bool,
    samples: usize,
    width: usize,
    out: &mut [C64],
) {
    for_each_row_chunk(py, samples, width, out, |start, count, rows| {
        for block in blocks {
            transform_block_rows(block, conj, start, count, rows, width);
        }
    });
}

/// Real route: transform real-structure blocks with real sources into the row-major
/// float64 `(samples, width)` output. Any other block is an error, never a conversion.
fn transform_real_into(
    py: Python<'_>,
    blocks: &[Block<'_>],
    samples: usize,
    width: usize,
    out: &mut [f64],
) -> PyResult<()> {
    let mut sources = Vec::with_capacity(blocks.len());
    for block in blocks {
        match (block.real_structure(), block.primary, block.secondary) {
            (true, Rows::Real(values), None) => sources.push(values),
            _ => {
                return Err(PyValueError::new_err(
                    "the real optimizer transform requires real-structure blocks with real sources and no d_pbar.",
                ))
            }
        }
    }
    for_each_row_chunk(py, samples, width, out, |start, count, rows| {
        for (block, values) in blocks.iter().zip(&sources) {
            real_block_rows(block, values, start, count, rows, width);
        }
    });
    Ok(())
}

/// Keeps the NumPy readonly borrows alive while their slices are in use.
enum Held<'py> {
    Real(PyReadonlyArray2<'py, f64>),
    Complex(PyReadonlyArray2<'py, C64>),
}

fn rows_from<'py>(
    array: &Bound<'py, PyAny>,
    held: &mut Vec<Held<'py>>,
    what: &str,
) -> PyResult<()> {
    if let Ok(real) = array.downcast::<PyArray2<f64>>() {
        held.push(Held::Real(real.readonly()));
    } else if let Ok(complex) = array.downcast::<PyArray2<C64>>() {
        held.push(Held::Complex(complex.readonly()));
    } else {
        return Err(PyValueError::new_err(format!(
            "{what} must be a two-dimensional float64 or complex128 array."
        )));
    }
    Ok(())
}

fn held_rows<'a>(held: &'a Held<'_>, expected: usize, what: &str) -> PyResult<Rows<'a>> {
    let (rows, len) = match held {
        Held::Real(array) => {
            let slice = array
                .as_slice()
                .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous.")))?;
            (Rows::Real(slice), slice.len())
        }
        Held::Complex(array) => {
            let slice = array
                .as_slice()
                .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous.")))?;
            (Rows::Complex(slice), slice.len())
        }
    };
    if len != expected {
        return Err(PyValueError::new_err(format!(
            "{what} has {len} values; expected {expected}."
        )));
    }
    Ok(rows)
}

fn contiguous<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray2<'_, T>,
    what: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous.")))
}

/// Parsed block descriptors plus the readonly borrows their slices point into.
struct Parsed<'py> {
    held: Vec<Held<'py>>,
    secondary: Vec<Option<PyReadonlyArray2<'py, C64>>>,
    kinds: Vec<Kind>,
    meta: Vec<(bool, usize, usize, usize)>,
}

/// Parse `(kind, field_complex, coordinate_start, coordinate_size, size, primary,
/// secondary, tie, stiefel)` descriptors produced by `jqmc_optimizer._rust_block_specs`.
fn parse_blocks<'py>(specs: &[Bound<'py, PyAny>], width: usize) -> PyResult<Parsed<'py>> {
    let mut parsed = Parsed {
        held: Vec::new(),
        secondary: Vec::new(),
        kinds: Vec::new(),
        meta: Vec::new(),
    };
    for spec in specs {
        let tuple = spec.downcast::<PyTuple>()?;
        if tuple.len() != 9 {
            return Err(PyValueError::new_err(
                "optimizer block descriptor must have nine fields.",
            ));
        }
        let kind: String = tuple.get_item(0)?.extract()?;
        let field_complex: bool = tuple.get_item(1)?.extract()?;
        let coordinate_start: usize = tuple.get_item(2)?.extract()?;
        let coordinate_size: usize = tuple.get_item(3)?.extract()?;
        let size: usize = tuple.get_item(4)?.extract()?;
        if coordinate_start + coordinate_size > width {
            return Err(PyValueError::new_err(
                "optimizer block coordinates exceed the coordinate size.",
            ));
        }
        rows_from(&tuple.get_item(5)?, &mut parsed.held, "derivative source")?;
        let secondary_item = tuple.get_item(6)?;
        parsed.secondary.push(if secondary_item.is_none() {
            None
        } else {
            Some(secondary_item.downcast::<PyArray2<C64>>()?.readonly())
        });
        let tie_item = tuple.get_item(7)?;
        let stiefel_item = tuple.get_item(8)?;
        let block_kind = match kind.as_str() {
            "euclidean" => {
                let expected = if field_complex { 2 * size } else { size };
                if coordinate_size != expected {
                    return Err(PyValueError::new_err(
                        "Euclidean block coordinate size mismatch.",
                    ));
                }
                Kind::Euclidean
            }
            "linear_tie" => {
                let (outputs, coordinates, coefficients): (
                    PyReadonlyArray1<'py, i64>,
                    PyReadonlyArray1<'py, i64>,
                    PyReadonlyArray1<'py, C64>,
                ) = tie_item.extract()?;
                let outputs = outputs.as_array();
                let coordinates = coordinates.as_array();
                let coefficients = coefficients.as_array();
                let nnz = outputs.len();
                if coordinates.len() != nnz || coefficients.len() != nnz {
                    return Err(PyValueError::new_err(
                        "linear-tie COO arrays must have identical lengths.",
                    ));
                }
                let mut counts = vec![0usize; coordinate_size + 1];
                for e in 0..nnz {
                    let (o, q) = (outputs[e], coordinates[e]);
                    if o < 0 || o as usize >= size || q < 0 || q as usize >= coordinate_size {
                        return Err(PyValueError::new_err("linear-tie COO index out of range."));
                    }
                    counts[q as usize + 1] += 1;
                }
                for q in 0..coordinate_size {
                    counts[q + 1] += counts[q];
                }
                if !field_complex && coefficients.iter().any(|c| c.im != 0.0) {
                    return Err(PyValueError::new_err(
                        "a real-field linear tie must not introduce imaginary storage directions.",
                    ));
                }
                let offsets = counts.clone();
                let mut cursor = counts;
                let mut csr_outputs = vec![0usize; nnz];
                let mut csr_coefficients = vec![ZERO; nnz];
                // Stable by published COO order within each coordinate (np.add.at order).
                for e in 0..nnz {
                    let q = coordinates[e] as usize;
                    csr_outputs[cursor[q]] = outputs[e] as usize;
                    csr_coefficients[cursor[q]] = coefficients[e];
                    cursor[q] += 1;
                }
                Kind::Tie {
                    offsets,
                    outputs: csr_outputs,
                    coefficients: csr_coefficients,
                }
            }
            "s_stiefel_rows" => {
                let stiefel = stiefel_item.downcast::<PyTuple>()?;
                let values = stiefel.get_item(0)?;
                let (rows, cols) = {
                    let shape = values.downcast::<numpy::PyUntypedArray>()?.shape().to_vec();
                    if shape.len() != 2 || shape[0] * shape[1] != size {
                        return Err(PyValueError::new_err(
                            "S-Stiefel coefficients have the wrong shape.",
                        ));
                    }
                    (shape[0], shape[1])
                };
                let expected = if field_complex { 2 * size } else { size };
                if coordinate_size != expected {
                    return Err(PyValueError::new_err(
                        "S-Stiefel block coordinate size mismatch.",
                    ));
                }
                if let Ok(real) = values.downcast::<PyArray2<f64>>() {
                    let sqrt: PyReadonlyArray2<'py, f64> = stiefel.get_item(1)?.extract()?;
                    let inv_sqrt: PyReadonlyArray2<'py, f64> = stiefel.get_item(2)?.extract()?;
                    let values = real.readonly();
                    Kind::StiefelReal(
                        Stiefel::build(
                            contiguous(&values, "S-Stiefel coefficients")?,
                            contiguous(&sqrt, "S-Stiefel metric sqrt")?,
                            contiguous(&inv_sqrt, "S-Stiefel metric inv_sqrt")?,
                            rows,
                            cols,
                        )
                        .map_err(PyFloatingPointError::new_err)?,
                    )
                } else {
                    let values: PyReadonlyArray2<'py, C64> = values.extract()?;
                    let sqrt: PyReadonlyArray2<'py, C64> = stiefel.get_item(1)?.extract()?;
                    let inv_sqrt: PyReadonlyArray2<'py, C64> = stiefel.get_item(2)?.extract()?;
                    Kind::StiefelComplex(
                        Stiefel::build(
                            contiguous(&values, "S-Stiefel coefficients")?,
                            contiguous(&sqrt, "S-Stiefel metric sqrt")?,
                            contiguous(&inv_sqrt, "S-Stiefel metric inv_sqrt")?,
                            rows,
                            cols,
                        )
                        .map_err(PyFloatingPointError::new_err)?,
                    )
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown optimizer tangent kind {other:?}."
                )))
            }
        };
        if let Kind::Tie { .. } = block_kind {
            if field_complex != parsed.secondary.last().unwrap().is_some() {
                return Err(PyValueError::new_err(
                    "a linear-tie block publishes d_pbar exactly when its field is complex.",
                ));
            }
        }
        if let Kind::StiefelReal(_) = block_kind {
            if field_complex {
                return Err(PyValueError::new_err(
                    "a complex-field S-Stiefel block requires complex metric data.",
                ));
            }
        }
        if !field_complex && parsed.secondary.last().unwrap().is_some() {
            return Err(PyValueError::new_err(
                "a real-field block has no d_pbar component.",
            ));
        }
        parsed.kinds.push(block_kind);
        parsed
            .meta
            .push((field_complex, coordinate_start, coordinate_size, size));
    }
    Ok(parsed)
}

fn borrow_blocks<'a>(parsed: &'a mut Parsed<'_>, samples: usize) -> PyResult<Vec<Block<'a>>> {
    let kinds = std::mem::take(&mut parsed.kinds);
    let mut blocks = Vec::with_capacity(kinds.len());
    for (index, kind) in kinds.into_iter().enumerate() {
        let (field_complex, coordinate_start, coordinate_size, size) = parsed.meta[index];
        let primary = held_rows(&parsed.held[index], samples * size, "derivative source")?;
        if field_complex && matches!(primary, Rows::Real(_)) {
            return Err(PyValueError::new_err(
                "a complex-field derivative source must be complex128.",
            ));
        }
        let secondary = match &parsed.secondary[index] {
            None => None,
            Some(array) => {
                let slice = contiguous(array, "d_pbar source")?;
                if slice.len() != samples * size {
                    return Err(PyValueError::new_err(
                        "d_pbar source has the wrong number of values.",
                    ));
                }
                Some(slice)
            }
        };
        blocks.push(Block {
            kind,
            field_complex,
            coordinate_start,
            coordinate_size,
            size,
            primary,
            secondary,
        });
    }
    Ok(blocks)
}

/// Rust port of `OptimizationCoordinateMap.transform_derivatives` (host).
///
/// `blocks` are the per-block descriptors of `jqmc_optimizer._rust_block_specs`
/// (sources flattened to `(samples, block.size)`). Returns the `(samples, coordinate_size)`
/// directional columns in canonical real-q order: float64 when `real_output` (the real
/// route, selected by `jqmc_optimizer._transform_derivatives_dtype`; every block must be
/// real structure with a real source, otherwise this raises), complex128 otherwise.
#[pyfunction]
#[pyo3(name = "optimizer_transform_derivatives", signature = (blocks, samples, coordinate_size, real_output))]
pub fn py_optimizer_transform_derivatives<'py>(
    py: Python<'py>,
    blocks: Vec<Bound<'py, PyAny>>,
    samples: usize,
    coordinate_size: usize,
    real_output: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let mut parsed = parse_blocks(&blocks, coordinate_size)?;
    let borrowed = borrow_blocks(&mut parsed, samples)?;
    if real_output {
        let output = PyArray2::<f64>::zeros(py, [samples, coordinate_size], false);
        {
            let mut writable = output.readwrite();
            let slice = writable
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("transform output must be C-contiguous."))?;
            transform_real_into(py, &borrowed, samples, coordinate_size, slice)?;
        }
        return Ok(output.into_any());
    }
    let output = PyArray2::<C64>::zeros(py, [samples, coordinate_size], false);
    {
        let mut writable = output.readwrite();
        let slice = writable
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("transform output must be C-contiguous."))?;
        transform_into(py, &borrowed, false, samples, coordinate_size, slice);
    }
    Ok(output.into_any())
}

/// Column sums over the leading (bin) axis, sequential per column.
fn column_sums(py: Python<'_>, values: &[C64], rows: usize, width: usize) -> Vec<C64> {
    let n_chunks = width.div_ceil(COLUMN_CHUNK);
    let parts = for_each_walker(py, n_chunks, |chunk| {
        let c0 = chunk * COLUMN_CHUNK;
        let c1 = (c0 + COLUMN_CHUNK).min(width);
        let mut sums = vec![ZERO; c1 - c0];
        for r in 0..rows {
            let row = &values[r * width..(r + 1) * width];
            for (q, sum) in (c0..c1).zip(sums.iter_mut()) {
                *sum += row[q];
            }
        }
        sums
    });
    parts.concat()
}

/// `-2 Re((OE - conj(O) * (E / w)) / w)`.
#[inline]
fn force(weight: f64, energy: C64, o: C64, oe: C64) -> f64 {
    let ratio = energy / weight;
    let centered = oe.re - (o.re * ratio.re + o.im * ratio.im);
    -2.0 * (centered / weight)
}

fn reduce_array_c64<'py>(
    py: Python<'py>,
    reduce: &Bound<'py, PyAny>,
    values: Vec<C64>,
) -> PyResult<Vec<C64>> {
    let reduced = reduce.call1((PyArray1::from_vec(py, values),))?;
    let reduced: PyReadonlyArray1<'py, C64> = reduced.extract()?;
    Ok(reduced.as_array().to_vec())
}

fn reduce_array_f64<'py>(
    py: Python<'py>,
    reduce: &Bound<'py, PyAny>,
    values: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let reduced = reduce.call1((PyArray1::from_vec(py, values),))?;
    let reduced: PyReadonlyArray1<'py, f64> = reduced.extract()?;
    Ok(reduced.as_array().to_vec())
}

/// Rust port of `estimate_coordinate_force_jackknife` (host).
///
/// `source_sum_blocks` / `conjugate_energy_blocks` are the descriptors of the per-bin
/// `source_sum` and `conjugate_source_energy_sum` moments (`bins` rows each). The
/// conjugate-energy columns are `conj(transform(conj(sources)))`, formed by conjugating
/// on load. `reduce_scalar` / `reduce_array` are the Python SUM reductions (identity
/// without MPI). Returns `(force, standard_error)` as float64 arrays.
#[pyfunction]
#[pyo3(
    name = "optimizer_coordinate_force_jackknife",
    signature = (source_sum_blocks, conjugate_energy_blocks, coordinate_size, bin_weight_sums, bin_weighted_energy_sums, reduce_scalar, reduce_array)
)]
#[allow(clippy::too_many_arguments)]
pub fn py_optimizer_coordinate_force_jackknife<'py>(
    py: Python<'py>,
    source_sum_blocks: Vec<Bound<'py, PyAny>>,
    conjugate_energy_blocks: Vec<Bound<'py, PyAny>>,
    coordinate_size: usize,
    bin_weight_sums: PyReadonlyArray1<'py, f64>,
    bin_weighted_energy_sums: PyReadonlyArray1<'py, C64>,
    reduce_scalar: Bound<'py, PyAny>,
    reduce_array: Bound<'py, PyAny>,
) -> PyResult<PyVecPair<'py>> {
    let weights = bin_weight_sums.as_array().to_vec();
    let energies = bin_weighted_energy_sums.as_array().to_vec();
    let bins = weights.len();
    if energies.len() != bins {
        return Err(PyValueError::new_err(
            "jackknife weight and energy sums must have the same length.",
        ));
    }
    let width = coordinate_size;
    let mut o_local = vec![ZERO; bins * width];
    let mut oe_local = vec![ZERO; bins * width];
    {
        let mut parsed = parse_blocks(&source_sum_blocks, width)?;
        let borrowed = borrow_blocks(&mut parsed, bins)?;
        transform_into(py, &borrowed, false, bins, width, &mut o_local);
    }
    {
        let mut parsed = parse_blocks(&conjugate_energy_blocks, width)?;
        let borrowed = borrow_blocks(&mut parsed, bins)?;
        transform_into(py, &borrowed, true, bins, width, &mut oe_local);
    }
    for value in oe_local.iter_mut() {
        *value = value.conj();
    }

    let total_bin_count: i64 = reduce_scalar.call1((bins as i64,))?.extract()?;
    if total_bin_count < 2 {
        return Err(PyValueError::new_err(
            "coordinate-force jackknife requires at least two global bins.",
        ));
    }
    let local_weight: f64 = weights.iter().sum();
    let local_energy: C64 = energies.iter().fold(ZERO, |acc, &value| acc + value);
    let total_weight: f64 = reduce_scalar.call1((local_weight,))?.extract()?;
    let total_energy = {
        let local = pyo3::types::PyComplex::from_doubles(py, local_energy.re, local_energy.im);
        let reduced = reduce_scalar.call1((local,))?;
        C64::new(
            reduced.getattr("real")?.extract()?,
            reduced.getattr("imag")?.extract()?,
        )
    };
    let total_o = reduce_array_c64(py, &reduce_array, column_sums(py, &o_local, bins, width))?;
    let total_oe = reduce_array_c64(py, &reduce_array, column_sums(py, &oe_local, bins, width))?;
    if total_o.len() != width || total_oe.len() != width {
        return Err(PyValueError::new_err(
            "jackknife array reduction changed the coordinate length.",
        ));
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(PyValueError::new_err(
            "coordinate-force jackknife requires positive finite total weight.",
        ));
    }
    let full: Vec<f64> = (0..width)
        .map(|q| force(total_weight, total_energy, total_o[q], total_oe[q]))
        .collect();
    let leave_weights: Vec<f64> = weights.iter().map(|&w| total_weight - w).collect();
    if leave_weights.iter().any(|&w| !w.is_finite() || w <= 0.0) {
        return Err(PyValueError::new_err(
            "coordinate-force jackknife leave-one-bin weight must remain positive.",
        ));
    }
    let leave_energies: Vec<C64> = energies.iter().map(|&e| total_energy - e).collect();
    let n_chunks = width.div_ceil(COLUMN_CHUNK);
    let replica = |b: usize, q: usize| -> f64 {
        force(
            leave_weights[b],
            leave_energies[b],
            total_o[q] - o_local[b * width + q],
            total_oe[q] - oe_local[b * width + q],
        )
    };
    let replica_sums = for_each_walker(py, n_chunks, |chunk| {
        let c0 = chunk * COLUMN_CHUNK;
        let c1 = (c0 + COLUMN_CHUNK).min(width);
        let mut sums = vec![0.0f64; c1 - c0];
        for b in 0..bins {
            for (q, sum) in (c0..c1).zip(sums.iter_mut()) {
                *sum += replica(b, q);
            }
        }
        sums
    })
    .concat();
    let global_replica_sum = reduce_array_f64(py, &reduce_array, replica_sums)?;
    let count = total_bin_count as f64;
    let mean: Vec<f64> = global_replica_sum
        .iter()
        .map(|&value| value / count)
        .collect();
    let centered_sumsq = for_each_walker(py, n_chunks, |chunk| {
        let c0 = chunk * COLUMN_CHUNK;
        let c1 = (c0 + COLUMN_CHUNK).min(width);
        let mut sums = vec![0.0f64; c1 - c0];
        for b in 0..bins {
            for (q, sum) in (c0..c1).zip(sums.iter_mut()) {
                let centered = replica(b, q) - mean[q];
                *sum += centered * centered;
            }
        }
        sums
    })
    .concat();
    let global_centered_sumsq = reduce_array_f64(py, &reduce_array, centered_sumsq)?;
    let std: Vec<f64> = global_centered_sumsq
        .iter()
        .map(|&value| ((count - 1.0) / count * value).max(0.0).sqrt())
        .collect();
    Ok((PyArray1::from_vec(py, full), PyArray1::from_vec(py, std)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    /// Independent dense reference `rows @ T_dense` of a real COO tie.
    fn dense_tie_adjoint(
        rows: &[f64],
        samples: usize,
        size: usize,
        coo: &[(usize, usize, f64)],
        width: usize,
    ) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((size, width));
        for &(o, q, c) in coo {
            dense[[o, q]] += c;
        }
        let mut out = Array2::<f64>::zeros((samples, width));
        for s in 0..samples {
            for q in 0..width {
                out[[s, q]] = (0..size).map(|o| rows[s * size + o] * dense[[o, q]]).sum();
            }
        }
        out
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    #[test]
    fn real_linear_tie_matches_dense_matrix_product() {
        // Repeated coordinates and outputs, non-unit coefficients (1/sqrt(2)).
        let coo = [
            (0usize, 0usize, 1.0f64),
            (1, 1, std::f64::consts::FRAC_1_SQRT_2),
            (2, 1, std::f64::consts::FRAC_1_SQRT_2),
            (3, 2, 1.0),
            (4, 0, -0.5),
            (5, 2, 2.0),
        ];
        let (samples, size, width) = (5usize, 6usize, 3usize);
        let mut seed = 7u64;
        let rows: Vec<f64> = (0..samples * size).map(|_| lcg(&mut seed)).collect();
        let mut offsets = vec![0usize; width + 1];
        for &(_, q, _) in &coo {
            offsets[q + 1] += 1;
        }
        for q in 0..width {
            offsets[q + 1] += offsets[q];
        }
        let mut cursor = offsets.clone();
        let mut outputs = vec![0usize; coo.len()];
        let mut coefficients = vec![ZERO; coo.len()];
        for &(o, q, c) in &coo {
            outputs[cursor[q]] = o;
            coefficients[cursor[q]] = C64::new(c, 0.0);
            cursor[q] += 1;
        }
        let block = Block {
            kind: Kind::Tie {
                offsets,
                outputs,
                coefficients,
            },
            field_complex: false,
            coordinate_start: 0,
            coordinate_size: width,
            size,
            primary: Rows::Real(&rows),
            secondary: None,
        };
        let mut out = vec![ZERO; samples * width];
        transform_block_rows(&block, false, 0, samples, &mut out, width);
        let expected = dense_tie_adjoint(&rows, samples, size, &coo, width);
        for s in 0..samples {
            for q in 0..width {
                let value = out[s * width + q];
                assert!(
                    (value.re - expected[[s, q]]).abs()
                        <= crate::setting_consts::ATOL_PYTHON_FP64_LOOSE
                            * (1.0 + expected[[s, q]].abs())
                );
                assert_eq!(value.im, 0.0);
            }
        }
    }

    #[test]
    fn stiefel_adjoint_output_is_a_whitened_tangent() {
        // With sqrt = inv_sqrt = I the adjoint output Y satisfies Y W^T + W Y^T = 0.
        let (p, n) = (2usize, 4usize);
        // Orthonormal rows.
        let values = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.8, 0.0];
        let mut identity = vec![0.0f64; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let stiefel =
            Stiefel::<f64>::build(&values, &identity, &identity, p, n).expect("positive Gram");
        let mut seed = 11u64;
        let samples = 3usize;
        let x: Vec<f64> = (0..samples * p * n).map(|_| lcg(&mut seed)).collect();
        let mut y = vec![0.0f64; samples * p * n];
        stiefel.adjoint(&x, samples, &mut y);
        for k in 0..samples {
            let yk = &y[k * p * n..(k + 1) * p * n];
            for i in 0..p {
                for j in 0..p {
                    let mut constraint = 0.0;
                    for c in 0..n {
                        constraint +=
                            yk[i * n + c] * values[j * n + c] + values[i * n + c] * yk[j * n + c];
                    }
                    assert!(
                        constraint.abs() <= crate::setting_consts::ATOL_PYTHON_FP64_LOOSE,
                        "tangent constraint {constraint}"
                    );
                }
            }
        }
    }
}
