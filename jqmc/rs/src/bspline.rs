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

//! B-spline orbital basis (`AOs_bspline_data`, periodic real-Gamma). Tensor-product 3D
//! cardinal (uniform) B-spline on a regular grid -- the jQMC einspline analog. Each
//! band is one spline AO; the MO layer is the identity (AO==MO). 1:1 port of
//! `atomic_orbital.py::AOs_bspline_data._eval_core` (value / Cartesian grad / Laplacian).
//!
//! Eval per electron: `frac = r*inv_cell` (mod 1), `t = frac*N`, `i=floor(t)`, `f=t-i`;
//! per-axis cardinal weights `w/dw/d2w` (cubic/6 / quintic/120); gather the local
//! `K=order+1` control points per axis with MODULAR wrap; separable contraction ->
//! value, t-gradient and t-Hessian; map to Cartesian via `inv_cell` / the metric
//! `M=inv_cellT*inv_cell` (`A[a,b]=N_a N_b M[a,b]`). Real Gamma only (`_complex=False`).

use std::sync::Arc;

use numpy::ndarray::{Array2, ArrayView2};
use numpy::Complex64;
use pyo3::prelude::*;

use crate::ewald::inv3;

#[derive(Clone)]
pub struct AosBsplineData {
    pub num_ao: usize,
    pub shape: [usize; 3],
    pub order: usize,
    /// Grid-major `(N0*N1*N2, num_ao)` flat layout:
    /// `coeffs[g*num_ao + b]` is band `b` at grid point `g`. Keeping only this
    /// band-contiguous layout avoids a second full control-point table while preserving
    /// unit-stride band accumulation in the hot value/VGL kernels. Immutable and
    /// `Arc`-held so clones share the same rank-private allocation.
    pub coeffs: Arc<Vec<f64>>,
    /// `frac[a] = Sum_c r[c]*inv_cell[c][a]`.
    pub inv_cell: [[f64; 3]; 3],
    /// Lattice vectors as rows (`r = frac*cell`), retained by tiled adapters.
    pub cell: [[f64; 3]; 3],
    /// `A[a][b] = N_a N_b M[a][b]`, `M = inv_cellT*inv_cell` (t-Hessian -> Cart Laplacian).
    pub a_mat: [[f64; 3]; 3],
    pub n_f: [f64; 3],
    /// Construction-time arithmetic policy (see `AosGtoData::precision_policy`).
    pub(crate) precision_policy: crate::PrecisionPolicy,
}

/// Centred uniform B-spline value/1st/2nd-deriv weights (length `order+1`) at offset
/// `fin[0,1)`. 1:1 `_bspline_weights_jax` (cubic /6, quintic /120). Returns fixed-size
/// length-6 arrays; only the first `order+1` entries are valid.
#[inline]
fn bspline_weights(f: f64, order: usize) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let mut w = [0.0; 6];
    let mut dw = [0.0; 6];
    let mut d2 = [0.0; 6];
    if order == 3 {
        let g = 1.0 / 6.0;
        let om = 1.0 - f;
        w[0] = om * om * om * g;
        w[1] = (3.0 * f.powi(3) - 6.0 * f * f + 4.0) * g;
        w[2] = (-3.0 * f.powi(3) + 3.0 * f * f + 3.0 * f + 1.0) * g;
        w[3] = f.powi(3) * g;
        dw[0] = -3.0 * om * om * g;
        dw[1] = (9.0 * f * f - 12.0 * f) * g;
        dw[2] = (-9.0 * f * f + 6.0 * f + 3.0) * g;
        dw[3] = 3.0 * f * f * g;
        d2[0] = 6.0 * om * g;
        d2[1] = (18.0 * f - 12.0) * g;
        d2[2] = (-18.0 * f + 6.0) * g;
        d2[3] = 6.0 * f * g;
    } else if order == 5 {
        let g = 1.0 / 120.0;
        let om = 1.0 - f;
        let (f2, f3, f4, f5) = (f * f, f.powi(3), f.powi(4), f.powi(5));
        w[0] = om.powi(5) * g;
        w[1] = (26.0 - 50.0 * f + 20.0 * f2 + 20.0 * f3 - 20.0 * f4 + 5.0 * f5) * g;
        w[2] = (66.0 - 60.0 * f2 + 30.0 * f4 - 10.0 * f5) * g;
        w[3] = (26.0 + 50.0 * f + 20.0 * f2 - 20.0 * f3 - 20.0 * f4 + 10.0 * f5) * g;
        w[4] = (1.0 + 5.0 * f + 10.0 * f2 + 10.0 * f3 + 5.0 * f4 - 5.0 * f5) * g;
        w[5] = f5 * g;
        dw[0] = -5.0 * om.powi(4) * g;
        dw[1] = (-50.0 + 40.0 * f + 60.0 * f2 - 80.0 * f3 + 25.0 * f4) * g;
        dw[2] = (-120.0 * f + 120.0 * f3 - 50.0 * f4) * g;
        dw[3] = (50.0 + 40.0 * f - 60.0 * f2 - 80.0 * f3 + 50.0 * f4) * g;
        dw[4] = (5.0 + 20.0 * f + 30.0 * f2 + 20.0 * f3 - 25.0 * f4) * g;
        dw[5] = 5.0 * f4 * g;
        d2[0] = 20.0 * om.powi(3) * g;
        d2[1] = (40.0 + 120.0 * f - 240.0 * f2 + 100.0 * f3) * g;
        d2[2] = (-120.0 + 360.0 * f2 - 200.0 * f3) * g;
        d2[3] = (40.0 - 120.0 * f - 240.0 * f2 + 200.0 * f3) * g;
        d2[4] = (20.0 + 60.0 * f + 60.0 * f2 - 100.0 * f3) * g;
        d2[5] = 20.0 * f3 * g;
    } else {
        panic!("Unsupported B-spline order {order} (use 3 or 5)");
    }
    (w, dw, d2)
}

#[inline]
fn bspline_value_weights_f32(f: f32, order: usize) -> [f32; 6] {
    let mut w = [0.0_f32; 6];
    if order == 3 {
        let g = 1.0_f32 / 6.0;
        let om = 1.0 - f;
        w[0] = om * om * om * g;
        w[1] = (3.0 * f.powi(3) - 6.0 * f * f + 4.0) * g;
        w[2] = (-3.0 * f.powi(3) + 3.0 * f * f + 3.0 * f + 1.0) * g;
        w[3] = f.powi(3) * g;
    } else if order == 5 {
        let g = 1.0_f32 / 120.0;
        let om = 1.0 - f;
        let (f2, f3, f4, f5) = (f * f, f.powi(3), f.powi(4), f.powi(5));
        w[0] = om.powi(5) * g;
        w[1] = (26.0 - 50.0 * f + 20.0 * f2 + 20.0 * f3 - 20.0 * f4 + 5.0 * f5) * g;
        w[2] = (66.0 - 60.0 * f2 + 30.0 * f4 - 10.0 * f5) * g;
        w[3] = (26.0 + 50.0 * f + 20.0 * f2 - 20.0 * f3 - 20.0 * f4 + 10.0 * f5) * g;
        w[4] = (1.0 + 5.0 * f + 10.0 * f2 + 10.0 * f3 + 5.0 * f4 - 5.0 * f5) * g;
        w[5] = f5 * g;
    } else {
        panic!("Unsupported B-spline order {order} (use 3 or 5)");
    }
    w
}

/// Third f-derivative of the cardinal B-spline weight vector (length order+1). 1:1
/// `derivatives/ao.py::_bspline_d3w`. Needed by the Laplacian's cell (stress) derivative.
#[inline]
fn bspline_d3w(f: f64, order: usize) -> [f64; 6] {
    let mut d3 = [0.0; 6];
    if order == 3 {
        d3[0] = -1.0;
        d3[1] = 3.0;
        d3[2] = -3.0;
        d3[3] = 1.0;
    } else if order == 5 {
        let g = 1.0 / 120.0;
        let om = 1.0 - f;
        let f2 = f * f;
        d3[0] = -60.0 * om * om * g;
        d3[1] = (120.0 - 480.0 * f + 300.0 * f2) * g;
        d3[2] = (720.0 * f - 600.0 * f2) * g;
        d3[3] = (-120.0 - 480.0 * f + 600.0 * f2) * g;
        d3[4] = (60.0 + 120.0 * f - 300.0 * f2) * g;
        d3[5] = 60.0 * f2 * g;
    } else {
        panic!("Unsupported B-spline order {order} (use 3 or 5)");
    }
    d3
}

impl AosBsplineData {
    /// Directional derivative of (value, grad, lap) along a cell perturbation `dcell`
    /// (3x3), each `(num_ao, N_e)`. 1:1 `derivatives/ao.py::bspline_value_grad_lap_dcell`
    /// -- the B-spline STRESS building block (force == 0: the spline has no nuclear-position
    /// dependence; only the cell channel via inv_cell/metric survives). `dinv=-inv*dcell*inv`.
    pub fn value_grad_lap_dcell(
        &self,
        r: &ArrayView2<f64>,
        dcell: &[[f64; 3]; 3],
    ) -> crate::atomic_orbital::ValueGradLap {
        let n_e = r.nrows();
        let nb = self.num_ao;
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let inv = self.inv_cell;
        let coeffs = self.coeffs.as_slice();
        // dinv = -inv * dcell * inv
        let mut idc = [[0.0_f64; 3]; 3]; // inv*dcell
        for a in 0..3 {
            for b in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += inv[a][c] * dcell[c][b];
                }
                idc[a][b] = s;
            }
        }
        let mut dinv = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += idc[a][c] * inv[c][b];
                }
                dinv[a][b] = -s;
            }
        }
        // dmetric = dinvT*inv + invT*dinv ; da_mat = (ns(x)ns)*dmetric
        let mut da_mat = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut dm = 0.0;
                for c in 0..3 {
                    dm += dinv[c][a] * inv[c][b] + inv[c][a] * dinv[c][b];
                }
                da_mat[a][b] = self.n_f[a] * self.n_f[b] * dm;
            }
        }
        let a_mat = self.a_mat;
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut dval = Array2::<f64>::zeros((nb, n_e));
        let mut dgx = Array2::<f64>::zeros((nb, n_e));
        let mut dgy = Array2::<f64>::zeros((nb, n_e));
        let mut dgz = Array2::<f64>::zeros((nb, n_e));
        let mut dlap = Array2::<f64>::zeros((nb, n_e));
        for e in 0..n_e {
            let rc = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            let mut frac = [0.0_f64; 3];
            for a in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += rc[c] * inv[c][a];
                }
                frac[a] = s - s.floor();
            }
            // df = (r * dinv) * ns
            let mut df = [0.0_f64; 3];
            for a in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += rc[c] * dinv[c][a];
                }
                df[a] = s * self.n_f[a];
            }
            // per-axis weight stacks W[axis][deriv 0..3][p]
            let mut wv = [[[0.0_f64; 6]; 4]; 3];
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac[a] * self.n_f[a];
                let ifl = t.floor();
                let f = t - ifl;
                let (w, dw, d2) = bspline_weights(f, self.order);
                wv[a][0] = w;
                wv[a][1] = dw;
                wv[a][2] = d2;
                wv[a][3] = bspline_d3w(f, self.order);
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, self.shape[a]);
                }
            }
            for b in 0..nb {
                // E(a,b,c) over the K^3 block for this band.
                let ee = |da: usize, db: usize, dc: usize| -> f64 {
                    let mut acc = 0.0;
                    for p0 in 0..k {
                        let w0 = wv[0][da][p0];
                        if w0 == 0.0 {
                            continue;
                        }
                        let base0 = idx[0][p0] * n1 * n2;
                        for p1 in 0..k {
                            let w01 = w0 * wv[1][db][p1];
                            let base1 = base0 + idx[1][p1] * n2;
                            for p2 in 0..k {
                                let g = base1 + idx[2][p2];
                                acc += coeffs[g * nb + b] * w01 * wv[2][dc][p2];
                            }
                        }
                    }
                    acc
                };
                let gt = [ee(1, 0, 0), ee(0, 1, 0), ee(0, 0, 1)];
                let (h00, h11, h22) = (ee(2, 0, 0), ee(0, 2, 0), ee(0, 0, 2));
                let (h01, h02, h12) = (ee(1, 1, 0), ee(1, 0, 1), ee(0, 1, 1));
                // dval = grad_t * df
                dval[[b, e]] = gt[0] * df[0] + gt[1] * df[1] + gt[2] * df[2];
                // Hdf = H @ df
                let hdf = [
                    h00 * df[0] + h01 * df[1] + h02 * df[2],
                    h01 * df[0] + h11 * df[1] + h12 * df[2],
                    h02 * df[0] + h12 * df[1] + h22 * df[2],
                ];
                // dgrad_r = (Hdf*ns)@invT + (grad_t*ns)@dinvT ; (x@MT)[c]=Sum_a x[a]M[c][a]
                let mut dg = [0.0_f64; 3];
                for c in 0..3 {
                    let mut s = 0.0;
                    for a in 0..3 {
                        s += hdf[a] * self.n_f[a] * inv[c][a] + gt[a] * self.n_f[a] * dinv[c][a];
                    }
                    dg[c] = s;
                }
                dgx[[b, e]] = dg[0];
                dgy[[b, e]] = dg[1];
                dgz[[b, e]] = dg[2];
                // dH[i,j] = Sum_m (raise m) df[m]
                let dh00 = ee(3, 0, 0) * df[0] + ee(2, 1, 0) * df[1] + ee(2, 0, 1) * df[2];
                let dh11 = ee(1, 2, 0) * df[0] + ee(0, 3, 0) * df[1] + ee(0, 2, 1) * df[2];
                let dh22 = ee(1, 0, 2) * df[0] + ee(0, 1, 2) * df[1] + ee(0, 0, 3) * df[2];
                let dh01 = ee(2, 1, 0) * df[0] + ee(1, 2, 0) * df[1] + ee(1, 1, 1) * df[2];
                let dh02 = ee(2, 0, 1) * df[0] + ee(1, 1, 1) * df[1] + ee(1, 0, 2) * df[2];
                let dh12 = ee(1, 1, 1) * df[0] + ee(0, 2, 1) * df[1] + ee(0, 1, 2) * df[2];
                let contract = |a: &[[f64; 3]; 3],
                                c00: f64,
                                c11: f64,
                                c22: f64,
                                c01: f64,
                                c02: f64,
                                c12: f64| {
                    a[0][0] * c00
                        + a[1][1] * c11
                        + a[2][2] * c22
                        + 2.0 * (a[0][1] * c01 + a[0][2] * c02 + a[1][2] * c12)
                };
                dlap[[b, e]] = contract(&da_mat, h00, h11, h22, h01, h02, h12)
                    + contract(&a_mat, dh00, dh11, dh22, dh01, dh02, dh12);
            }
        }
        (dval, dgx, dgy, dgz, dlap)
    }

    pub fn from_py(aos_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // `AOs_bspline_data` carries the static `_complex` flag; the twist class
        // `AOs_bspline_complex_data` has no such attribute (it is complex by type).
        // Read the flag only when present so a complex leaf gets the explicit
        // NotImplementedError below instead of an AttributeError (issue #88).
        let is_complex: bool = if aos_data.hasattr("_complex")? {
            aos_data.getattr("_complex")?.extract()?
        } else {
            aos_data.get_type().name()?.to_string().contains("complex")
        };
        if is_complex {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "complex (twist) B-spline not supported in Rust",
            ));
        }
        let np = aos_data.py().import("numpy")?;
        let coeffs_arr: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (aos_data.getattr("spline_coefficients")?,))?
            .extract()?;
        let coeffs_view = coeffs_arr.as_array();
        let num_ao = coeffs_view.nrows();
        let shape_t: Vec<usize> = aos_data.getattr("spline_shape")?.extract()?;
        let shape = [shape_t[0], shape_t[1], shape_t[2]];
        let order: usize = aos_data.getattr("spline_order")?.extract()?;
        // cell (rows = lattice vectors): frac = r @ inv_cell.
        let sd = aos_data.getattr("structure_data")?;
        let cell_arr: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (sd.getattr("cell")?,))?
            .extract()?;
        let cell = cell_arr.as_array();
        let inv_cell = inv3(&cell);
        let cell_rows = [
            [cell[[0, 0]], cell[[0, 1]], cell[[0, 2]]],
            [cell[[1, 0]], cell[[1, 1]], cell[[1, 2]]],
            [cell[[2, 0]], cell[[2, 1]], cell[[2, 2]]],
        ];
        // metric M = inv_cellT*inv_cell ; A[a][b] = N_a N_b M[a][b].
        let n_f = [shape[0] as f64, shape[1] as f64, shape[2] as f64];
        let mut a_mat = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut m = 0.0;
                for inv_cell_k in &inv_cell {
                    m += inv_cell_k[a] * inv_cell_k[b];
                }
                a_mat[a][b] = n_f[a] * n_f[b] * m;
            }
        }
        // Build only the grid-major layout used by the Rust kernels. Transpose directly
        // from the Python/JAX band-major view into the final allocation, without first
        // materializing a second owned band-major Vec.
        let grid = shape[0] * shape[1] * shape[2];
        let mut coeffs = vec![0.0_f64; grid * num_ao];
        for b in 0..num_ao {
            for g in 0..grid {
                coeffs[g * num_ao + b] = coeffs_view[[b, g]];
            }
        }
        Ok(AosBsplineData {
            num_ao,
            shape,
            order,
            coeffs: Arc::new(coeffs),
            inv_cell,
            cell: cell_rows,
            a_mat,
            n_f,
            precision_policy: crate::construction_precision_policy(),
        })
    }

    /// t-space spline derivatives for the B-spline kinetic STRESS (affine-invariance route):
    /// `gt[e,b,k] = dAO_b/dt_k` and `Ht[e,b,k,l] = d^2AO_b/dt_kdt_l` per electron, in the
    /// fractional-grid t-space. frac=r*inv_cell is affine-strain-invariant, so gt/Ht are
    /// strain-invariant -- only the metric (inv, N) responds (applied in stress_pbc). 1:1
    /// `derivatives/ao.py::bspline_t_fields`. Returns (gt (N,nb,3), Ht (N,nb,3,3)).
    pub fn t_fields(
        &self,
        r: &ArrayView2<f64>,
    ) -> (numpy::ndarray::Array3<f64>, numpy::ndarray::Array4<f64>) {
        let n_e = r.nrows();
        let nb = self.num_ao;
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let coeffs = self.coeffs.as_slice();
        let mut gt = numpy::ndarray::Array3::<f64>::zeros((n_e, nb, 3));
        let mut ht = numpy::ndarray::Array4::<f64>::zeros((n_e, nb, 3, 3));
        for e in 0..n_e {
            let rc = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            let mut frac = [0.0_f64; 3];
            for (a, frac_a) in frac.iter_mut().enumerate() {
                let mut s = 0.0;
                for (c, rc_c) in rc.iter().enumerate() {
                    s += *rc_c * self.inv_cell[c][a];
                }
                *frac_a = s - s.floor();
            }
            // per-axis weights: deriv 0 (w), 1 (dw), 2 (d2)
            let mut wv = [[[0.0_f64; 6]; 3]; 3]; // [axis][deriv][p]
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac[a] * self.n_f[a];
                let ifl = t.floor();
                let (w, dw, d2) = bspline_weights(t - ifl, self.order);
                wv[a][0] = w;
                wv[a][1] = dw;
                wv[a][2] = d2;
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, self.shape[a]);
                }
            }
            for b in 0..nb {
                let ee = |da: usize, db: usize, dc: usize| -> f64 {
                    let mut acc = 0.0;
                    for p0 in 0..k {
                        let w0 = wv[0][da][p0];
                        if w0 == 0.0 {
                            continue;
                        }
                        let base0 = idx[0][p0] * n1 * n2;
                        for p1 in 0..k {
                            let w01 = w0 * wv[1][db][p1];
                            let base1 = base0 + idx[1][p1] * n2;
                            for p2 in 0..k {
                                let g = base1 + idx[2][p2];
                                acc += coeffs[g * nb + b] * w01 * wv[2][dc][p2];
                            }
                        }
                    }
                    acc
                };
                gt[[e, b, 0]] = ee(1, 0, 0);
                gt[[e, b, 1]] = ee(0, 1, 0);
                gt[[e, b, 2]] = ee(0, 0, 1);
                let (h00, h11, h22) = (ee(2, 0, 0), ee(0, 2, 0), ee(0, 0, 2));
                let (h01, h02, h12) = (ee(1, 1, 0), ee(1, 0, 1), ee(0, 1, 1));
                ht[[e, b, 0, 0]] = h00;
                ht[[e, b, 1, 1]] = h11;
                ht[[e, b, 2, 2]] = h22;
                ht[[e, b, 0, 1]] = h01;
                ht[[e, b, 1, 0]] = h01;
                ht[[e, b, 0, 2]] = h02;
                ht[[e, b, 2, 0]] = h02;
                ht[[e, b, 1, 2]] = h12;
                ht[[e, b, 2, 1]] = h12;
            }
        }
        (gt, ht)
    }

    /// Cartesian electron-axis kinetic fields for the periodic det-kinetic force, per band
    /// per electron: `ga[d]` = dAO/dr_d, `gl[d]` = d(lap AO)/dr_d, `h[d][e]` = d2AO/dr_d dr_e,
    /// each `(num_ao, N_e)`. The B-spline analog of `pgto_grad`/`pgto_gradlap`/`pgto_hessian`
    /// -- consumed by `forces_pbc::bspline_det_egrad_side`. There is NO nuclear-force analog:
    /// dAO/dR = 0 (the control points sit on a cell-fractional grid, R-independent).
    ///
    /// t-space -> Cartesian maps (frac_a = Sum_c r_c inv[c][a], t_a = N_a frac_a, so
    /// `J[c][a] = dt_a/dr_c = N_a inv[c][a]`): grad_r[c]=Sum_a gt[a] J[c][a];
    /// Hess_r[c][d]=Sum_ab Ht[a][b] J[c][a] J[d][b]; lap=Sum_ab a_mat[a][b] Ht[a][b] so
    /// gradlap_r[c]=Sum_g (Sum_ab a_mat[a][b] T3[a][b][g]) J[c][g] with T3 the 3rd t-deriv.
    #[allow(clippy::type_complexity)]
    pub fn det_egrad_fields(
        &self,
        r: &ArrayView2<f64>,
    ) -> ([Array2<f64>; 3], [Array2<f64>; 3], [[Array2<f64>; 3]; 3]) {
        let n_e = r.nrows();
        let nb = self.num_ao;
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let inv = self.inv_cell;
        let a_mat = self.a_mat;
        let coeffs = self.coeffs.as_slice();
        // J[c][a] = dt_a/dr_c = N_a inv[c][a].
        let mut jmat = [[0.0_f64; 3]; 3];
        for c in 0..3 {
            for a in 0..3 {
                jmat[c][a] = self.n_f[a] * inv[c][a];
            }
        }
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut ga = [
            Array2::<f64>::zeros((nb, n_e)),
            Array2::<f64>::zeros((nb, n_e)),
            Array2::<f64>::zeros((nb, n_e)),
        ];
        let mut gl = [
            Array2::<f64>::zeros((nb, n_e)),
            Array2::<f64>::zeros((nb, n_e)),
            Array2::<f64>::zeros((nb, n_e)),
        ];
        let mut h = [
            [
                Array2::<f64>::zeros((nb, n_e)),
                Array2::<f64>::zeros((nb, n_e)),
                Array2::<f64>::zeros((nb, n_e)),
            ],
            [
                Array2::<f64>::zeros((nb, n_e)),
                Array2::<f64>::zeros((nb, n_e)),
                Array2::<f64>::zeros((nb, n_e)),
            ],
            [
                Array2::<f64>::zeros((nb, n_e)),
                Array2::<f64>::zeros((nb, n_e)),
                Array2::<f64>::zeros((nb, n_e)),
            ],
        ];
        for e in 0..n_e {
            let rc = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            let mut frac = [0.0_f64; 3];
            for a in 0..3 {
                let mut s = 0.0;
                for c in 0..3 {
                    s += rc[c] * inv[c][a];
                }
                frac[a] = s - s.floor();
            }
            // per-axis weight stacks W[axis][deriv 0..3][p] (value/dw/d2/d3).
            let mut wv = [[[0.0_f64; 6]; 4]; 3];
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac[a] * self.n_f[a];
                let ifl = t.floor();
                let f = t - ifl;
                let (w, dw, d2) = bspline_weights(f, self.order);
                wv[a][0] = w;
                wv[a][1] = dw;
                wv[a][2] = d2;
                wv[a][3] = bspline_d3w(f, self.order);
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, self.shape[a]);
                }
            }
            for b in 0..nb {
                let ee = |da: usize, db: usize, dc: usize| -> f64 {
                    let mut acc = 0.0;
                    for p0 in 0..k {
                        let w0 = wv[0][da][p0];
                        if w0 == 0.0 {
                            continue;
                        }
                        let base0 = idx[0][p0] * n1 * n2;
                        for p1 in 0..k {
                            let w01 = w0 * wv[1][db][p1];
                            let base1 = base0 + idx[1][p1] * n2;
                            for p2 in 0..k {
                                let g = base1 + idx[2][p2];
                                acc += coeffs[g * nb + b] * w01 * wv[2][dc][p2];
                            }
                        }
                    }
                    acc
                };
                let gt = [ee(1, 0, 0), ee(0, 1, 0), ee(0, 0, 1)];
                let (h00, h11, h22) = (ee(2, 0, 0), ee(0, 2, 0), ee(0, 0, 2));
                let (h01, h02, h12) = (ee(1, 1, 0), ee(1, 0, 1), ee(0, 1, 1));
                let htm = [[h00, h01, h02], [h01, h11, h12], [h02, h12, h22]];
                // gradlap in t-space: glt[g] = Sum_ab a_mat[a][b] * d^3AO/dt_a dt_b dt_g.
                let glt = [
                    a_mat[0][0] * ee(3, 0, 0)
                        + a_mat[1][1] * ee(1, 2, 0)
                        + a_mat[2][2] * ee(1, 0, 2)
                        + 2.0
                            * (a_mat[0][1] * ee(2, 1, 0)
                                + a_mat[0][2] * ee(2, 0, 1)
                                + a_mat[1][2] * ee(1, 1, 1)),
                    a_mat[0][0] * ee(2, 1, 0)
                        + a_mat[1][1] * ee(0, 3, 0)
                        + a_mat[2][2] * ee(0, 1, 2)
                        + 2.0
                            * (a_mat[0][1] * ee(1, 2, 0)
                                + a_mat[0][2] * ee(1, 1, 1)
                                + a_mat[1][2] * ee(0, 2, 1)),
                    a_mat[0][0] * ee(2, 0, 1)
                        + a_mat[1][1] * ee(0, 2, 1)
                        + a_mat[2][2] * ee(0, 0, 3)
                        + 2.0
                            * (a_mat[0][1] * ee(1, 1, 1)
                                + a_mat[0][2] * ee(1, 0, 2)
                                + a_mat[1][2] * ee(0, 1, 2)),
                ];
                for c in 0..3 {
                    // grad_r[c] = Sum_a gt[a] J[c][a]
                    let mut grc = 0.0;
                    let mut glc = 0.0;
                    for a in 0..3 {
                        grc += gt[a] * jmat[c][a];
                        glc += glt[a] * jmat[c][a];
                    }
                    ga[c][[b, e]] = grc;
                    gl[c][[b, e]] = glc;
                    // Hess_r[c][d] = Sum_ab Ht[a][b] J[c][a] J[d][b]
                    for d in 0..3 {
                        let mut hcd = 0.0;
                        for a in 0..3 {
                            let mut ja_htm = 0.0;
                            for bb in 0..3 {
                                ja_htm += htm[a][bb] * jmat[d][bb];
                            }
                            hcd += jmat[c][a] * ja_htm;
                        }
                        h[c][d][[b, e]] = hcd;
                    }
                }
            }
        }
        (ga, gl, h)
    }

    /// Value-only spline eval at ONE position into `out` (length `num_ao`). The
    /// per-move (MCMC sweep) hot path: drops all grad/Hessian work of `value_grad_lap`.
    pub fn value_single(&self, pos: [f64; 3], out: &mut [f64]) {
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut frac = [0.0_f64; 3];
        for (a, frac_a) in frac.iter_mut().enumerate() {
            let mut s = 0.0;
            for (c, pos_c) in pos.iter().enumerate() {
                s += *pos_c * self.inv_cell[c][a];
            }
            *frac_a = s - s.floor();
        }
        let mut tw = [[0.0_f64; 6]; 3];
        let mut idx = [[0usize; 6]; 3];
        for a in 0..3 {
            let t = frac[a] * self.n_f[a];
            let ifl = t.floor();
            let (w, _dw, _d2) = bspline_weights(t - ifl, self.order);
            tw[a] = w;
            let base = ifl as i64;
            for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                *idx_ap = wrap(base + start + p as i64, self.shape[a]);
            }
        }
        // Band-contiguous accumulation into `out` (reads `coeffs` rows with UNIT band
        // stride). BIT-IDENTICAL to the band-major nested sum: the per-element expression
        // stays `(c * w01) * w2` and the p0/p1/p2 accumulation order is unchanged, so the
        // accept/reject sweep chain is preserved exactly (`bspline_value_single_bitexact`).
        let nb = self.num_ao;
        let coeffs = self.coeffs.as_slice();
        for o in out.iter_mut().take(nb) {
            *o = 0.0;
        }
        for p0 in 0..k {
            let w0 = tw[0][p0];
            let i0 = idx[0][p0] * n1 * n2;
            for p1 in 0..k {
                let w01 = w0 * tw[1][p1];
                let i01 = i0 + idx[1][p1] * n2;
                for p2 in 0..k {
                    let w2 = tw[2][p2];
                    let g = i01 + idx[2][p2];
                    let row = &coeffs[g * nb..g * nb + nb];
                    for (b, o) in out.iter_mut().enumerate().take(nb) {
                        *o += row[b] * w01 * w2;
                    }
                }
            }
        }
    }

    /// [`Self::value_single`] for two coefficient sets on the SAME grid (the re/im halves of a
    /// complex spline): one stencil walk, weights and indices computed once, both rows read
    /// per stencil point. Per element the expression `(c * w01) * w2` and the p0/p1/p2
    /// order are those of `value_single`, so each output is bitwise identical to a separate
    /// `value_single` call on that coefficient set.
    pub(crate) fn value_single_pair(
        &self,
        other: &[f64],
        pos: [f64; 3],
        out_a: &mut [f64],
        out_b: &mut [f64],
    ) {
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut frac = [0.0_f64; 3];
        for (a, frac_a) in frac.iter_mut().enumerate() {
            let mut s = 0.0;
            for (c, pos_c) in pos.iter().enumerate() {
                s += *pos_c * self.inv_cell[c][a];
            }
            *frac_a = s - s.floor();
        }
        let mut tw = [[0.0_f64; 6]; 3];
        let mut idx = [[0usize; 6]; 3];
        for a in 0..3 {
            let t = frac[a] * self.n_f[a];
            let ifl = t.floor();
            let (w, _dw, _d2) = bspline_weights(t - ifl, self.order);
            tw[a] = w;
            let base = ifl as i64;
            for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                *idx_ap = wrap(base + start + p as i64, self.shape[a]);
            }
        }
        let nb = self.num_ao;
        let ca = self.coeffs.as_slice();
        let cb = &other[..ca.len()];
        let (out_a, out_b) = (&mut out_a[..nb], &mut out_b[..nb]);
        out_a.fill(0.0);
        out_b.fill(0.0);
        for p0 in 0..k {
            let w0 = tw[0][p0];
            let i0 = idx[0][p0] * n1 * n2;
            for p1 in 0..k {
                let w01 = w0 * tw[1][p1];
                let i01 = i0 + idx[1][p1] * n2;
                for p2 in 0..k {
                    let w2 = tw[2][p2];
                    let g = i01 + idx[2][p2];
                    let ra = &ca[g * nb..g * nb + nb];
                    let rb = &cb[g * nb..g * nb + nb];
                    for b in 0..nb {
                        out_a[b] += ra[b] * w01 * w2;
                        out_b[b] += rb[b] * w01 * w2;
                    }
                }
            }
        }
    }

    /// Batch VALUE-only spline eval `(num_ao, N_e)`. Band-contiguous: loops the K^3
    /// stencil outermost and accumulates over bands with UNIT stride (`coeffs` rows),
    /// so the innermost loop is a vectorizable AXPY -- vs the band-major strided gather
    /// (bands `N0*N1*N2` doubles apart). e_L observable (FLOOR); the bit-exact per-move
    /// sweep keeps `value_single`.
    pub fn value(&self, r: &ArrayView2<f64>) -> Array2<f64> {
        let n_e = r.nrows();
        let nb = self.num_ao;
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let mut val = Array2::<f64>::zeros((nb, n_e));
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut acc = vec![0.0_f64; nb];
        let coeffs = self.coeffs.as_slice();
        for e in 0..n_e {
            let mut frac = [0.0_f64; 3];
            for (a, frac_a) in frac.iter_mut().enumerate() {
                let mut s = 0.0;
                for c in 0..3 {
                    s += r[[e, c]] * self.inv_cell[c][a];
                }
                *frac_a = s - s.floor();
            }
            let mut tw = [[0.0_f64; 6]; 3];
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac[a] * self.n_f[a];
                let ifl = t.floor();
                let (w, _dw, _d2) = bspline_weights(t - ifl, self.order);
                tw[a] = w;
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, self.shape[a]);
                }
            }
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            for p0 in 0..k {
                let w0 = tw[0][p0];
                let i0 = idx[0][p0] * n1 * n2;
                for p1 in 0..k {
                    let w01 = w0 * tw[1][p1];
                    let i01 = i0 + idx[1][p1] * n2;
                    for p2 in 0..k {
                        let w = w01 * tw[2][p2];
                        let g = i01 + idx[2][p2];
                        let row = &coeffs[g * nb..g * nb + nb];
                        for b in 0..nb {
                            acc[b] += w * row[b];
                        }
                    }
                }
            }
            for b in 0..nb {
                val[[b, e]] = acc[b];
            }
        }
        val
    }

    pub(crate) fn value_with_policy(
        &self,
        r: &ArrayView2<f64>,
        f32_arithmetic: bool,
    ) -> Array2<f64> {
        if !f32_arithmetic {
            return self.value(r);
        }
        let n_e = r.nrows();
        let nb = self.num_ao;
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut val = Array2::<f64>::zeros((nb, n_e));
        let coeffs = self.coeffs.as_slice();
        for e in 0..n_e {
            // Preserve coordinate precision through wrapping and grid indexing.
            // Cast only the local offset to f32: a Cartesian home-cell round trip
            // or a rounded fractional coordinate can displace a spline node.
            let mut frac64 = [0.0_f64; 3];
            for (a, frac64_a) in frac64.iter_mut().enumerate() {
                let s = r[[e, 0]] * self.inv_cell[0][a]
                    + r[[e, 1]] * self.inv_cell[1][a]
                    + r[[e, 2]] * self.inv_cell[2][a];
                *frac64_a = s - s.floor();
            }
            let mut tw = [[0.0_f32; 6]; 3];
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac64[a] * self.n_f[a];
                let ifl = t.floor();
                tw[a] = bspline_value_weights_f32((t - ifl) as f32, self.order);
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, self.shape[a]);
                }
            }
            let mut acc = vec![0.0_f32; nb];
            for p0 in 0..k {
                let i0 = idx[0][p0] * n1 * n2;
                for p1 in 0..k {
                    let w01 = tw[0][p0] * tw[1][p1];
                    let i01 = i0 + idx[1][p1] * n2;
                    for p2 in 0..k {
                        let w = w01 * tw[2][p2];
                        let g = i01 + idx[2][p2];
                        let row = &coeffs[g * nb..g * nb + nb];
                        for b in 0..nb {
                            acc[b] += w * row[b] as f32;
                        }
                    }
                }
            }
            for b in 0..nb {
                val[[b, e]] = acc[b] as f64;
            }
        }
        val
    }

    /// `(val, gx, gy, gz, lap)`, each `(num_ao, N_e)`.
    pub fn value_grad_lap(&self, r: &ArrayView2<f64>) -> crate::atomic_orbital::ValueGradLap {
        let n_e = r.nrows();
        let nb = self.num_ao;
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let mut val = Array2::<f64>::zeros((nb, n_e));
        let mut gx = Array2::<f64>::zeros((nb, n_e));
        let mut gy = Array2::<f64>::zeros((nb, n_e));
        let mut gz = Array2::<f64>::zeros((nb, n_e));
        let mut lap = Array2::<f64>::zeros((nb, n_e));
        // per-band stencil accumulators (value, t-grad x3, t-Hessian x6), reused per electron.
        let (mut av, mut adt0, mut adt1, mut adt2) = (
            vec![0.0_f64; nb],
            vec![0.0_f64; nb],
            vec![0.0_f64; nb],
            vec![0.0_f64; nb],
        );
        let (mut ah00, mut ah11, mut ah22) =
            (vec![0.0_f64; nb], vec![0.0_f64; nb], vec![0.0_f64; nb]);
        let (mut ah01, mut ah02, mut ah12) =
            (vec![0.0_f64; nb], vec![0.0_f64; nb], vec![0.0_f64; nb]);
        let coeffs = self.coeffs.as_slice();
        // per-axis wrapped grid indices for the local block
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let value_is_f32 = self.precision_policy.is_f32("ao_eval");
        for e in 0..n_e {
            let rc = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            // frac = r @ inv_cell, then into [0,1)
            let mut frac = [0.0_f64; 3];
            for (a, frac_a) in frac.iter_mut().enumerate() {
                let mut s = 0.0;
                for (c, rc_c) in rc.iter().enumerate() {
                    s += *rc_c * self.inv_cell[c][a];
                }
                *frac_a = s - s.floor();
            }
            let mut tw = [[0.0_f64; 6]; 3]; // value weights per axis
            let mut tdw = [[0.0_f64; 6]; 3];
            let mut td2 = [[0.0_f64; 6]; 3];
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac[a] * self.n_f[a];
                let ifl = t.floor();
                let f = t - ifl;
                let (w, dw, d2) = bspline_weights(f, self.order);
                tw[a] = w;
                tdw[a] = dw;
                td2[a] = d2;
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, self.shape[a]);
                }
            }
            // Band-contiguous: stencil outermost, 10 unit-stride AXPYs over bands sharing
            // the single `coeffs` row load per stencil point (no per-band strided gather).
            for x in [
                &mut av, &mut adt0, &mut adt1, &mut adt2, &mut ah00, &mut ah11, &mut ah22,
                &mut ah01, &mut ah02, &mut ah12,
            ] {
                for v in x.iter_mut() {
                    *v = 0.0;
                }
            }
            for p0 in 0..k {
                let (w0, dw0, d20) = (tw[0][p0], tdw[0][p0], td2[0][p0]);
                let i0 = idx[0][p0] * n1 * n2;
                for p1 in 0..k {
                    let (w1, dw1, d21) = (tw[1][p1], tdw[1][p1], td2[1][p1]);
                    let i01 = i0 + idx[1][p1] * n2;
                    for p2 in 0..k {
                        let (w2, dw2, d22) = (tw[2][p2], tdw[2][p2], td2[2][p2]);
                        let g = i01 + idx[2][p2];
                        let row = &coeffs[g * nb..g * nb + nb];
                        let (wv, wdt0, wdt1, wdt2) =
                            (w0 * w1 * w2, dw0 * w1 * w2, w0 * dw1 * w2, w0 * w1 * dw2);
                        let (wh00, wh11, wh22) = (d20 * w1 * w2, w0 * d21 * w2, w0 * w1 * d22);
                        let (wh01, wh02, wh12) = (dw0 * dw1 * w2, dw0 * w1 * dw2, w0 * dw1 * dw2);
                        for b in 0..nb {
                            let c = row[b];
                            av[b] += c * wv;
                            adt0[b] += c * wdt0;
                            adt1[b] += c * wdt1;
                            adt2[b] += c * wdt2;
                            ah00[b] += c * wh00;
                            ah11[b] += c * wh11;
                            ah22[b] += c * wh22;
                            ah01[b] += c * wh01;
                            ah02[b] += c * wh02;
                            ah12[b] += c * wh12;
                        }
                    }
                }
            }
            let amat = &self.a_mat;
            for b in 0..nb {
                let gt = [
                    adt0[b] * self.n_f[0],
                    adt1[b] * self.n_f[1],
                    adt2[b] * self.n_f[2],
                ];
                let mut gr = [0.0_f64; 3];
                for (cc, gr_cc) in gr.iter_mut().enumerate() {
                    for (a, gt_a) in gt.iter().enumerate() {
                        *gr_cc += *gt_a * self.inv_cell[cc][a];
                    }
                }
                let l = ah00[b] * amat[0][0]
                    + ah11[b] * amat[1][1]
                    + ah22[b] * amat[2][2]
                    + 2.0 * (ah01[b] * amat[0][1] + ah02[b] * amat[0][2] + ah12[b] * amat[1][2]);
                val[[b, e]] = if value_is_f32 {
                    (av[b] as f32) as f64
                } else {
                    av[b]
                };
                gx[[b, e]] = gr[0];
                gy[[b, e]] = gr[1];
                gz[[b, e]] = gr[2];
                lap[[b, e]] = l;
            }
        }
        (val, gx, gy, gz, lap)
    }

    /// Candidate AO VALUE rows at `positions`, written row-major into `out`
    /// (`(npoint, num_ao)`, `out.len() == npoint * num_ao`) -- the B-spline
    /// counterpart of gfmc's `molecular_aos_candidate_batch`. Thin loop over
    /// [`Self::value_single`], i.e. the per-move sweep BIT-CHAIN convention
    /// (`(c * w01) * w2` element order), NOT the batch [`Self::value`] fp order
    /// (`(w01 * w2) * c`); a candidate table seeded by this kernel must also be
    /// refreshed by it (or by `value_single`) to stay bit-consistent. The
    /// spline wraps positions in fractional space internally, so no home-cell
    /// pre-wrap contract is needed.
    // Used by the GFMC fixed-mesh candidate and derivative routes.
    pub fn candidate_batch_into(&self, positions: &[[f64; 3]], out: &mut [f64]) {
        debug_assert_eq!(out.len(), positions.len() * self.num_ao);
        let nb = self.num_ao;
        for (v, &pos) in positions.iter().enumerate() {
            self.value_single(pos, &mut out[v * nb..(v + 1) * nb]);
        }
    }

    /// Single-position sibling of [`Self::value_grad_lap`]: spline value + grad
    /// (x,y,z) + laplacian at ONE position into caller buffers (each `num_ao`)
    /// -- the B-spline entry for the per-hop derivative-table column update
    /// (`update_det_grad_lap_cache_and_value` analogue). BIT-CONSISTENT with the
    /// batch kernel: same stencil traversal (p0/p1/p2 outermost), same
    /// accumulator ORDER per band (av, dt0, dt1, dt2, h00, h11, h22, h01, h02,
    /// h12), same `c * w` element products and the same metric transform, so a
    /// batch-initialized derivative table and this single-point refresh agree
    /// bitwise. The five output buffers double as the value / t-gradient / h00
    /// accumulators; `scratch` is resized to `5 * num_ao` for the remaining
    /// Hessian accumulators (allocation-free in steady state).
    // Used by the GFMC fixed-mesh candidate and derivative routes.
    #[allow(
        clippy::too_many_arguments,
        reason = "caller-owned value/grad/lap output buffers"
    )]
    pub fn value_grad_lap_single_into(
        &self,
        pos: [f64; 3],
        scratch: &mut Vec<f64>,
        value: &mut [f64],
        gx: &mut [f64],
        gy: &mut [f64],
        gz: &mut [f64],
        lap: &mut [f64],
    ) {
        let nb = self.num_ao;
        debug_assert_eq!(value.len(), nb);
        debug_assert_eq!(gx.len(), nb);
        debug_assert_eq!(gy.len(), nb);
        debug_assert_eq!(gz.len(), nb);
        debug_assert_eq!(lap.len(), nb);
        let k = self.order + 1;
        let start = -(((self.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (self.shape[0], self.shape[1], self.shape[2]);
        let coeffs = self.coeffs.as_slice();
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let value_is_f32 = self.precision_policy.is_f32("ao_eval");
        // frac = pos @ inv_cell, then into [0,1) -- identical to the batch body.
        let mut frac = [0.0_f64; 3];
        for (a, frac_a) in frac.iter_mut().enumerate() {
            let mut s = 0.0;
            for (c, pos_c) in pos.iter().enumerate() {
                s += *pos_c * self.inv_cell[c][a];
            }
            *frac_a = s - s.floor();
        }
        let mut tw = [[0.0_f64; 6]; 3];
        let mut tdw = [[0.0_f64; 6]; 3];
        let mut td2 = [[0.0_f64; 6]; 3];
        let mut idx = [[0usize; 6]; 3];
        for a in 0..3 {
            let t = frac[a] * self.n_f[a];
            let ifl = t.floor();
            let f = t - ifl;
            let (w, dw, d2) = bspline_weights(f, self.order);
            tw[a] = w;
            tdw[a] = dw;
            td2[a] = d2;
            let base = ifl as i64;
            for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                *idx_ap = wrap(base + start + p as i64, self.shape[a]);
            }
        }
        // Accumulators: value=av, gx=dt0, gy=dt1, gz=dt2, lap=h00; scratch
        // carries h11/h22/h01/h02/h12.
        scratch.clear();
        scratch.resize(5 * nb, 0.0);
        let (ah11, rest) = scratch.split_at_mut(nb);
        let (ah22, rest) = rest.split_at_mut(nb);
        let (ah01, rest) = rest.split_at_mut(nb);
        let (ah02, ah12) = rest.split_at_mut(nb);
        for buf in [&mut *value, &mut *gx, &mut *gy, &mut *gz, &mut *lap] {
            for v in buf.iter_mut() {
                *v = 0.0;
            }
        }
        for p0 in 0..k {
            let (w0, dw0, d20) = (tw[0][p0], tdw[0][p0], td2[0][p0]);
            let i0 = idx[0][p0] * n1 * n2;
            for p1 in 0..k {
                let (w1, dw1, d21) = (tw[1][p1], tdw[1][p1], td2[1][p1]);
                let i01 = i0 + idx[1][p1] * n2;
                for p2 in 0..k {
                    let (w2, dw2, d22) = (tw[2][p2], tdw[2][p2], td2[2][p2]);
                    let g = i01 + idx[2][p2];
                    let row = &coeffs[g * nb..g * nb + nb];
                    let (wv, wdt0, wdt1, wdt2) =
                        (w0 * w1 * w2, dw0 * w1 * w2, w0 * dw1 * w2, w0 * w1 * dw2);
                    let (wh00, wh11, wh22) = (d20 * w1 * w2, w0 * d21 * w2, w0 * w1 * d22);
                    let (wh01, wh02, wh12) = (dw0 * dw1 * w2, dw0 * w1 * dw2, w0 * dw1 * dw2);
                    for b in 0..nb {
                        let c = row[b];
                        value[b] += c * wv;
                        gx[b] += c * wdt0;
                        gy[b] += c * wdt1;
                        gz[b] += c * wdt2;
                        lap[b] += c * wh00;
                        ah11[b] += c * wh11;
                        ah22[b] += c * wh22;
                        ah01[b] += c * wh01;
                        ah02[b] += c * wh02;
                        ah12[b] += c * wh12;
                    }
                }
            }
        }
        let amat = &self.a_mat;
        for b in 0..nb {
            let gt = [
                gx[b] * self.n_f[0],
                gy[b] * self.n_f[1],
                gz[b] * self.n_f[2],
            ];
            let mut gr = [0.0_f64; 3];
            for (cc, gr_cc) in gr.iter_mut().enumerate() {
                for (a, gt_a) in gt.iter().enumerate() {
                    *gr_cc += *gt_a * self.inv_cell[cc][a];
                }
            }
            let l = lap[b] * amat[0][0]
                + ah11[b] * amat[1][1]
                + ah22[b] * amat[2][2]
                + 2.0 * (ah01[b] * amat[0][1] + ah02[b] * amat[0][2] + ah12[b] * amat[1][2]);
            let av = value[b];
            value[b] = if value_is_f32 { (av as f32) as f64 } else { av };
            gx[b] = gr[0];
            gy[b] = gr[1];
            gz[b] = gr[2];
            lap[b] = l;
        }
    }
}

/// Complex (twist k != Gamma) B-spline AO leaf (`AOs_bspline_complex_data`). Each AO is
/// one whole-cell Bloch band `psi_k(r) = e^{i k . r} u_k(r)`, where the cell-periodic part
/// `u_k` is a periodic B-spline of COMPLEX control points. jqmc splits the complex control
/// points into real/imag halves and runs the identical REAL separable spline einsum on each
/// (`cg_re`/`cg_im`), then combines `u = u_re + i u_im`; this port does the same by holding
/// two real [`AosBsplineData`] (shared grid + metric, re/im coefficients) and reusing their
/// value/value_grad_lap kernels verbatim -- so `u`, `grad u`, `lap u` are bit-identical to
/// jqmc's `_eval_u_core`. The Bloch phase `e^{i k . r}` (with the FULL, unwrapped `r`; `u`
/// is cell-periodic so the spline eval wraps internally) is layered on top with the product
/// rule (1:1 `_compute_AOs{,_value_grad_lap}`):
///   psi      = u e^{ik.r}
///   grad psi = (grad u + i k u) e^{ik.r}
///   lap psi  = (lap u + 2 i k . grad u - |k|^2 u) e^{ik.r}
#[derive(Clone, Copy)]
struct BsplineGeometry {
    shape: [usize; 3],
    order: usize,
    inv_cell: [[f64; 3]; 3],
    cell: [[f64; 3]; 3],
    a_mat: [[f64; 3]; 3],
    n_f: [f64; 3],
}

impl BsplineGeometry {
    fn from_py(np: &Bound<'_, PyAny>, aos_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape_t: Vec<usize> = aos_data.getattr("spline_shape")?.extract()?;
        if shape_t.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "B-spline shape must have length 3, got {}",
                shape_t.len()
            )));
        }
        let shape = [shape_t[0], shape_t[1], shape_t[2]];
        let order: usize = aos_data.getattr("spline_order")?.extract()?;
        let cell_source = if aos_data.hasattr("primitive_cell_np")? {
            aos_data.getattr("primitive_cell_np")?
        } else {
            aos_data.getattr("structure_data")?.getattr("cell")?
        };
        let cell_arr: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (cell_source, np.getattr("float64")?))?
            .extract()?;
        let cell_view = cell_arr.as_array();
        let inv_cell = inv3(&cell_view);
        let cell = [
            [cell_view[[0, 0]], cell_view[[0, 1]], cell_view[[0, 2]]],
            [cell_view[[1, 0]], cell_view[[1, 1]], cell_view[[1, 2]]],
            [cell_view[[2, 0]], cell_view[[2, 1]], cell_view[[2, 2]]],
        ];
        let n_f = [shape[0] as f64, shape[1] as f64, shape[2] as f64];
        let mut a_mat = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut m = 0.0;
                for inv_cell_k in &inv_cell {
                    m += inv_cell_k[a] * inv_cell_k[b];
                }
                a_mat[a][b] = n_f[a] * n_f[b] * m;
            }
        }
        Ok(Self {
            shape,
            order,
            inv_cell,
            cell,
            a_mat,
            n_f,
        })
    }

    fn with_coeffs(self, num_ao: usize, coeffs: Arc<Vec<f64>>) -> AosBsplineData {
        AosBsplineData {
            num_ao,
            shape: self.shape,
            order: self.order,
            coeffs,
            inv_cell: self.inv_cell,
            cell: self.cell,
            a_mat: self.a_mat,
            n_f: self.n_f,
            precision_policy: crate::construction_precision_policy(),
        }
    }

    fn k_cart(self, np: &Bound<'_, PyAny>, aos_data: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
        let k_frac_arr: numpy::PyReadonlyArray1<f64> = np
            .call_method1(
                "asarray",
                (aos_data.getattr("k_frac_np")?, np.getattr("float64")?),
            )?
            .extract()?;
        let kf = k_frac_arr.as_array();
        let twopi = 2.0 * std::f64::consts::PI;
        let mut k_cart = [0.0_f64; 3];
        for (b, k_cart_b) in k_cart.iter_mut().enumerate() {
            let mut s = 0.0;
            for a in 0..3 {
                s += kf[a] * self.inv_cell[b][a];
            }
            *k_cart_b = twopi * s;
        }
        Ok(k_cart)
    }
}

pub struct AosBsplineComplexData {
    pub num_ao: usize,
    /// Cartesian Bloch vector per AO. Ordinary leaves repeat one k; tiled
    /// leaves provide one primitive-cell k for every unfolded band.
    pub k_cart: Vec<[f64; 3]>,
    /// Real part of the complex control points (shares grid + metric with `im`).
    re: AosBsplineData,
    /// Imaginary part of the complex control points.
    im: AosBsplineData,
    /// `true` for a time-reversal sibling whose logical coefficients are the
    /// conjugate of the shared `re`/`im` backing (`im` is consumed with sign -1).
    conjugate_coefficients: bool,
}

impl AosBsplineComplexData {
    fn orbital_k_cart(
        np: &Bound<'_, PyAny>,
        aos_data: &Bound<'_, PyAny>,
        geometry: BsplineGeometry,
        num_ao: usize,
    ) -> PyResult<Vec<[f64; 3]>> {
        if aos_data.hasattr("orbital_k_cart_np")? {
            let values: numpy::PyReadonlyArray2<f64> = np
                .call_method1(
                    "asarray",
                    (
                        aos_data.getattr("orbital_k_cart_np")?,
                        np.getattr("float64")?,
                    ),
                )?
                .extract()?;
            let view = values.as_array();
            if view.nrows() != num_ao || view.ncols() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "per-orbital B-spline k table has shape ({}, {}), expected ({num_ao}, 3)",
                    view.nrows(),
                    view.ncols()
                )));
            }
            return Ok((0..num_ao)
                .map(|row| [view[[row, 0]], view[[row, 1]], view[[row, 2]]])
                .collect());
        }
        Ok(vec![geometry.k_cart(np, aos_data)?; num_ao])
    }

    pub fn from_py(aos_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let np = aos_data.py().import("numpy")?;
        // `np.asarray(..., complex128)` is a no-copy view when the Python leaf already has
        // the production dtype. Unlike `astype(complex128)`, it does not force a temporary
        // grid-sized copy before Rust builds its final grid-major SoA backing.
        let coeffs_c: numpy::PyReadonlyArray2<Complex64> = np
            .call_method1(
                "asarray",
                (
                    aos_data.getattr("spline_coefficients")?,
                    np.getattr("complex128")?,
                ),
            )?
            .extract()?;
        let cview = coeffs_c.as_array();
        let num_ao = cview.nrows();
        let geometry = BsplineGeometry::from_py(&np, aos_data)?;
        let grid = geometry.shape[0] * geometry.shape[1] * geometry.shape[2];
        if cview.ncols() != grid {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "complex B-spline coefficients have {} grid columns, expected {grid}",
                cview.ncols()
            )));
        }
        // Split directly into the two final grid-major arrays. This keeps the arithmetic
        // structure-of-arrays representation without materializing band-major re/im Vecs.
        let mut re_coeffs = vec![0.0_f64; grid * num_ao];
        let mut im_coeffs = vec![0.0_f64; grid * num_ao];
        for b in 0..num_ao {
            for g in 0..grid {
                let z = cview[[b, g]];
                re_coeffs[g * num_ao + b] = z.re;
                im_coeffs[g * num_ao + b] = z.im;
            }
        }
        let k_cart = Self::orbital_k_cart(&np, aos_data, geometry, num_ao)?;
        let conjugate_coefficients = aos_data
            .getattr("conjugate_coefficients")
            .and_then(|value| value.extract::<bool>())
            .unwrap_or(false);
        Ok(AosBsplineComplexData {
            num_ao,
            k_cart,
            re: geometry.with_coeffs(num_ao, Arc::new(re_coeffs)),
            im: geometry.with_coeffs(num_ao, Arc::new(im_coeffs)),
            conjugate_coefficients,
        })
    }

    /// Build the time-reversal sibling without reading or copying its conjugated Python
    /// control-point table. Geometry and `k` come from the down-spin leaf, while the two
    /// grid-sized coefficient allocations are shared with `source` and consumed conjugated.
    pub(crate) fn from_py_with_conjugated_coefficients(
        aos_data: &Bound<'_, PyAny>,
        source: &Self,
    ) -> PyResult<Self> {
        let np = aos_data.py().import("numpy")?;
        let geometry = BsplineGeometry::from_py(&np, aos_data)?;
        let num_ao: usize = aos_data.getattr("num_ao")?.extract()?;
        if num_ao != source.num_ao
            || geometry.shape != source.re.shape
            || geometry.order != source.re.order
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time-reversal B-spline siblings must have identical band count, grid, and order",
            ));
        }
        Ok(Self {
            num_ao,
            k_cart: Self::orbital_k_cart(&np, aos_data, geometry, num_ao)?,
            re: geometry.with_coeffs(num_ao, source.re.coeffs.clone()),
            im: geometry.with_coeffs(num_ao, source.im.coeffs.clone()),
            conjugate_coefficients: !source.conjugate_coefficients,
        })
    }

    #[inline]
    fn imag_sign(&self) -> f64 {
        if self.conjugate_coefficients {
            -1.0
        } else {
            1.0
        }
    }

    pub(crate) fn coefficient_storage_ptrs(&self) -> [usize; 2] {
        [
            Arc::as_ptr(&self.re.coeffs) as usize,
            Arc::as_ptr(&self.im.coeffs) as usize,
        ]
    }

    /// Bloch phase `e^{i k . r}` at one (full, unwrapped) position.
    #[inline]
    fn phase(&self, band: usize, pos: [f64; 3]) -> Complex64 {
        let k = self.k_cart[band];
        let theta = k[0] * pos[0] + k[1] * pos[1] + k[2] * pos[2];
        Complex64::new(0.0, theta).exp()
    }

    /// Value-only complex AO at ONE position: `psi = u e^{ik.r}` (the sweep hot path).
    pub fn value_single(&self, pos: [f64; 3]) -> Vec<Complex64> {
        let mut vr = vec![0.0_f64; self.num_ao];
        let mut vi = vec![0.0_f64; self.num_ao];
        self.re
            .value_single_pair(self.im.coeffs.as_slice(), pos, &mut vr, &mut vi);
        let imag_sign = self.imag_sign();
        (0..self.num_ao)
            .map(|b| Complex64::new(vr[b], imag_sign * vi[b]) * self.phase(b, pos))
            .collect()
    }

    /// Weighted value contraction at ONE position: `Sum_b w_b psi_b(pos)` with
    /// `w_b = w_re[b] + i w_im[b]`, without forming the complex AO vector. The
    /// spline halves come from the same `value_single_pair` stencil as
    /// [`Self::value_single`]. When every band shares one k (ordinary leaves),
    /// the Bloch phase multiplies the contracted sum once instead of every band
    /// (fp-reassociated, FAST ECP mesh route only); tiled leaves with per-band k
    /// apply each band's phase.
    /// `scratch_re` / `scratch_im` must hold `num_ao` values.
    pub(crate) fn wdot_single(
        &self,
        pos: [f64; 3],
        w_re: &[f64],
        w_im: &[f64],
        scratch_re: &mut [f64],
        scratch_im: &mut [f64],
    ) -> Complex64 {
        debug_assert_eq!(w_re.len(), self.num_ao);
        debug_assert_eq!(w_im.len(), self.num_ao);
        self.re
            .value_single_pair(self.im.coeffs.as_slice(), pos, scratch_re, scratch_im);
        let s = self.imag_sign();
        let k0 = self.k_cart[0];
        if self.k_cart.iter().all(|k| *k == k0) {
            let (mut acc_re, mut acc_im) = (0.0_f64, 0.0_f64);
            for b in 0..self.num_ao {
                let (ur, ui) = (scratch_re[b], s * scratch_im[b]);
                acc_re += w_re[b] * ur - w_im[b] * ui;
                acc_im += w_re[b] * ui + w_im[b] * ur;
            }
            return Complex64::new(acc_re, acc_im) * self.phase(0, pos);
        }
        let mut acc = Complex64::new(0.0, 0.0);
        for b in 0..self.num_ao {
            let psi = Complex64::new(scratch_re[b], s * scratch_im[b]) * self.phase(b, pos);
            acc += Complex64::new(w_re[b], w_im[b]) * psi;
        }
        acc
    }

    /// Complex AO values `(num_ao, N_e)`.
    pub fn value(&self, r: &ArrayView2<f64>) -> Array2<Complex64> {
        self.value_with_policy(r, self.re.precision_policy.is_f32("ao_eval"))
    }

    pub(crate) fn value_with_policy(
        &self,
        r: &ArrayView2<f64>,
        f32_arithmetic: bool,
    ) -> Array2<Complex64> {
        if f32_arithmetic {
            let ur = self.re.value_with_policy(r, true);
            let ui = self.im.value_with_policy(r, true);
            let imag_sign = self.imag_sign() as f32;
            let mut out = Array2::<Complex64>::zeros((self.num_ao, r.nrows()));
            for e in 0..r.nrows() {
                // k.r grows with the distance from the home cell: form it in f64 and
                // reduce it modulo 2 pi before the f32 cast (jqmc `_phase_in_zone`).
                for b in 0..self.num_ao {
                    let k = self.k_cart[b];
                    let theta64 = k[0] * r[[e, 0]] + k[1] * r[[e, 1]] + k[2] * r[[e, 2]];
                    let tau = std::f64::consts::TAU;
                    let theta = (theta64 - tau * (theta64 / tau).round()) as f32;
                    let (pr, pi) = (theta.cos(), theta.sin());
                    let (ar, ai) = (ur[[b, e]] as f32, imag_sign * ui[[b, e]] as f32);
                    out[[b, e]] =
                        Complex64::new((ar * pr - ai * pi) as f64, (ar * pi + ai * pr) as f64);
                }
            }
            return out;
        }
        let n = r.nrows();
        let mut out = Array2::<Complex64>::zeros((self.num_ao, n));
        for e in 0..n {
            let col = self.value_single([r[[e, 0]], r[[e, 1]], r[[e, 2]]]);
            for b in 0..self.num_ao {
                out[[b, e]] = col[b];
            }
        }
        out
    }

    /// Complex AO value + Cartesian grad `[dx,dy,dz]` + Laplacian, each `(num_ao, N_e)`.
    /// Reuses the REAL value_grad_lap on the re/im coefficient halves for
    /// `u`/`grad u`/`lap u`, then applies the Bloch-phase product rule.
    #[allow(clippy::type_complexity)]
    pub fn value_grad_lap(
        &self,
        r: &ArrayView2<f64>,
    ) -> (Array2<Complex64>, [Array2<Complex64>; 3], Array2<Complex64>) {
        let (ur, gxr, gyr, gzr, lr) = self.re.value_grad_lap(r);
        let (ui, gxi, gyi, gzi, li) = self.im.value_grad_lap(r);
        let imag_sign = self.imag_sign();
        let n = r.nrows();
        let mk = || Array2::<Complex64>::zeros((self.num_ao, n));
        let (mut val, mut gx, mut gy, mut gz, mut lap) = (mk(), mk(), mk(), mk(), mk());
        let ii = Complex64::new(0.0, 1.0);
        let value_is_f32 = self.re.precision_policy.is_f32("ao_eval");
        for e in 0..n {
            for b in 0..self.num_ao {
                let position = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
                let ph = self.phase(b, position);
                let k = self.k_cart[b];
                let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                let u = Complex64::new(ur[[b, e]], imag_sign * ui[[b, e]]);
                let gu = [
                    Complex64::new(gxr[[b, e]], imag_sign * gxi[[b, e]]),
                    Complex64::new(gyr[[b, e]], imag_sign * gyi[[b, e]]),
                    Complex64::new(gzr[[b, e]], imag_sign * gzi[[b, e]]),
                ];
                let lu = Complex64::new(lr[[b, e]], imag_sign * li[[b, e]]);
                let value = u * ph;
                val[[b, e]] = if value_is_f32 {
                    Complex64::new((value.re as f32) as f64, (value.im as f32) as f64)
                } else {
                    value
                };
                gx[[b, e]] = (gu[0] + ii * k[0] * u) * ph;
                gy[[b, e]] = (gu[1] + ii * k[1] * u) * ph;
                gz[[b, e]] = (gu[2] + ii * k[2] * u) * ph;
                let kdotg = k[0] * gu[0] + k[1] * gu[1] + k[2] * gu[2];
                lap[[b, e]] = (lu + Complex64::new(0.0, 2.0) * kdotg - k2 * u) * ph;
            }
        }
        (val, [gx, gy, gz], lap)
    }

    /// Complex AO value and Cartesian gradient `(num_ao, N_e)`: the first two outputs of
    /// [`Self::value_grad_lap`] (bitwise: the same stencil weights, accumulation order and
    /// Bloch-phase product rule), with both coefficient halves read in ONE stencil walk, no
    /// Laplacian / Hessian accumulators, and the Bloch phase formed once per point when every
    /// band shares one k.
    #[allow(clippy::type_complexity)]
    pub(crate) fn value_grad(
        &self,
        r: &ArrayView2<f64>,
    ) -> (Array2<Complex64>, [Array2<Complex64>; 3]) {
        let d = &self.re;
        let n_e = r.nrows();
        let nb = d.num_ao;
        let k = d.order + 1;
        let start = -(((d.order - 1) / 2) as i64);
        let (n1, n2) = (d.shape[1], d.shape[2]);
        let ca = d.coeffs.as_slice();
        let cb = &self.im.coeffs.as_slice()[..ca.len()];
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let value_is_f32 = d.precision_policy.is_f32("ao_eval");
        let imag_sign = self.imag_sign();
        let k0 = self.k_cart[0];
        let one_k = self.k_cart.iter().all(|kk| *kk == k0);
        let ii = Complex64::new(0.0, 1.0);
        let mk = || Array2::<Complex64>::zeros((nb, n_e));
        let (mut val, mut gx, mut gy, mut gz) = (mk(), mk(), mk(), mk());
        // per-band accumulators (value, t-grad x3) of the real and imaginary halves
        let mut acc = vec![0.0_f64; 8 * nb];
        for e in 0..n_e {
            let rc = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            let mut frac = [0.0_f64; 3];
            for (a, frac_a) in frac.iter_mut().enumerate() {
                let mut s = 0.0;
                for (c, rc_c) in rc.iter().enumerate() {
                    s += *rc_c * d.inv_cell[c][a];
                }
                *frac_a = s - s.floor();
            }
            let mut tw = [[0.0_f64; 6]; 3];
            let mut tdw = [[0.0_f64; 6]; 3];
            let mut idx = [[0usize; 6]; 3];
            for a in 0..3 {
                let t = frac[a] * d.n_f[a];
                let ifl = t.floor();
                let (w, dw, _d2) = bspline_weights(t - ifl, d.order);
                tw[a] = w;
                tdw[a] = dw;
                let base = ifl as i64;
                for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                    *idx_ap = wrap(base + start + p as i64, d.shape[a]);
                }
            }
            acc.fill(0.0);
            let (av_r, rest) = acc.split_at_mut(nb);
            let (a0_r, rest) = rest.split_at_mut(nb);
            let (a1_r, rest) = rest.split_at_mut(nb);
            let (a2_r, rest) = rest.split_at_mut(nb);
            let (av_i, rest) = rest.split_at_mut(nb);
            let (a0_i, rest) = rest.split_at_mut(nb);
            let (a1_i, a2_i) = rest.split_at_mut(nb);
            for p0 in 0..k {
                let (w0, dw0) = (tw[0][p0], tdw[0][p0]);
                let i0 = idx[0][p0] * n1 * n2;
                for p1 in 0..k {
                    let (w1, dw1) = (tw[1][p1], tdw[1][p1]);
                    let i01 = i0 + idx[1][p1] * n2;
                    for p2 in 0..k {
                        let (w2, dw2) = (tw[2][p2], tdw[2][p2]);
                        let g = i01 + idx[2][p2];
                        let ra = &ca[g * nb..g * nb + nb];
                        let rb = &cb[g * nb..g * nb + nb];
                        let (wv, wdt0, wdt1, wdt2) =
                            (w0 * w1 * w2, dw0 * w1 * w2, w0 * dw1 * w2, w0 * w1 * dw2);
                        for b in 0..nb {
                            let (c, ci) = (ra[b], rb[b]);
                            av_r[b] += c * wv;
                            a0_r[b] += c * wdt0;
                            a1_r[b] += c * wdt1;
                            a2_r[b] += c * wdt2;
                            av_i[b] += ci * wv;
                            a0_i[b] += ci * wdt0;
                            a1_i[b] += ci * wdt1;
                            a2_i[b] += ci * wdt2;
                        }
                    }
                }
            }
            let cart = |t0: f64, t1: f64, t2: f64| -> [f64; 3] {
                let gt = [t0 * d.n_f[0], t1 * d.n_f[1], t2 * d.n_f[2]];
                let mut gr = [0.0_f64; 3];
                for (cc, gr_cc) in gr.iter_mut().enumerate() {
                    for (a, gt_a) in gt.iter().enumerate() {
                        *gr_cc += *gt_a * d.inv_cell[cc][a];
                    }
                }
                gr
            };
            let ph0 = self.phase(0, rc);
            for b in 0..nb {
                let ph = if one_k { ph0 } else { self.phase(b, rc) };
                let kb = self.k_cart[b];
                let ur = if value_is_f32 {
                    (av_r[b] as f32) as f64
                } else {
                    av_r[b]
                };
                let ui = if value_is_f32 {
                    (av_i[b] as f32) as f64
                } else {
                    av_i[b]
                };
                let u = Complex64::new(ur, imag_sign * ui);
                let gr = cart(a0_r[b], a1_r[b], a2_r[b]);
                let gi = cart(a0_i[b], a1_i[b], a2_i[b]);
                let gu = [
                    Complex64::new(gr[0], imag_sign * gi[0]),
                    Complex64::new(gr[1], imag_sign * gi[1]),
                    Complex64::new(gr[2], imag_sign * gi[2]),
                ];
                let value = u * ph;
                val[[b, e]] = if value_is_f32 {
                    Complex64::new((value.re as f32) as f64, (value.im as f32) as f64)
                } else {
                    value
                };
                gx[[b, e]] = (gu[0] + ii * kb[0] * u) * ph;
                gy[[b, e]] = (gu[1] + ii * kb[1] * u) * ph;
                gz[[b, e]] = (gu[2] + ii * kb[2] * u) * ph;
            }
        }
        (val, [gx, gy, gz])
    }

    /// Complex B-spline determinant electron-gradient FIELDS `(ga, gl, h)` for the kinetic
    /// FORCE (electron axis), each `(num_ao, N_e)` complex: `ga[d] = d(psi)/dr_d`,
    /// `gl[d] = d(grad^2 psi)/dr_d`, `h[d][e] = d(grad_e psi)/dr_d`. The complex analog of
    /// [`AosBsplineData::det_egrad_fields`]. `psi = u e^{ik.r}`: compose the real re/im
    /// halves' `value_grad_lap` (u, grad u, lap u) + `det_egrad_fields` (grad^2-grad,
    /// Hessian of u) with the Bloch-phase product rule (same rule as `value_grad_lap`).
    #[allow(clippy::type_complexity)]
    pub fn det_egrad_fields(
        &self,
        r: &ArrayView2<f64>,
    ) -> (
        [Array2<Complex64>; 3],
        [Array2<Complex64>; 3],
        [[Array2<Complex64>; 3]; 3],
    ) {
        let (ur, gxr, gyr, gzr, lr) = self.re.value_grad_lap(r);
        let (ui, gxi, gyi, gzi, li) = self.im.value_grad_lap(r);
        let (_gar, glr, hr) = self.re.det_egrad_fields(r);
        let (_gai, gli, hi) = self.im.det_egrad_fields(r);
        let imag_sign = self.imag_sign();
        let n = r.nrows();
        let nb = self.num_ao;
        let mk = || Array2::<Complex64>::zeros((nb, n));
        let mut ga = [mk(), mk(), mk()];
        let mut gl = [mk(), mk(), mk()];
        let mut h = [[mk(), mk(), mk()], [mk(), mk(), mk()], [mk(), mk(), mk()]];
        let ii = Complex64::new(0.0, 1.0);
        for e in 0..n {
            for b in 0..nb {
                let position = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
                let ph = self.phase(b, position);
                let k = self.k_cart[b];
                let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                let u = Complex64::new(ur[[b, e]], imag_sign * ui[[b, e]]);
                let gu = [
                    Complex64::new(gxr[[b, e]], imag_sign * gxi[[b, e]]),
                    Complex64::new(gyr[[b, e]], imag_sign * gyi[[b, e]]),
                    Complex64::new(gzr[[b, e]], imag_sign * gzi[[b, e]]),
                ];
                let lu = Complex64::new(lr[[b, e]], imag_sign * li[[b, e]]);
                // Hessian of u (complex) and grad(lap u) (complex).
                let mut hu = [[Complex64::new(0.0, 0.0); 3]; 3];
                let mut glu = [Complex64::new(0.0, 0.0); 3];
                for d in 0..3 {
                    glu[d] = Complex64::new(glr[d][[b, e]], imag_sign * gli[d][[b, e]]);
                    for f in 0..3 {
                        hu[d][f] = Complex64::new(hr[d][f][[b, e]], imag_sign * hi[d][f][[b, e]]);
                    }
                }
                // psi = u e^{ik.r}: grad_d psi = (grad_d u + i k_d u) ph.
                for d in 0..3 {
                    ga[d][[b, e]] = (gu[d] + ii * k[d] * u) * ph;
                }
                // Hessian: grad_d grad_e psi = (Hu_de + i k_d gu_e + i k_e gu_d - k_d k_e u) ph.
                for d in 0..3 {
                    for f in 0..3 {
                        h[d][f][[b, e]] = (hu[d][f] + ii * k[d] * gu[f] + ii * k[f] * gu[d]
                            - k[d] * k[f] * u)
                            * ph;
                    }
                }
                // lap psi = L ph, L = lu + 2i (k.gu) - k2 u; grad_d(lap psi) = (grad_d L + i k_d L) ph.
                let kdotg = k[0] * gu[0] + k[1] * gu[1] + k[2] * gu[2];
                let l_big = lu + Complex64::new(0.0, 2.0) * kdotg - k2 * u;
                for d in 0..3 {
                    let kdoth = k[0] * hu[0][d] + k[1] * hu[1][d] + k[2] * hu[2][d];
                    let grad_d_l = glu[d] + Complex64::new(0.0, 2.0) * kdoth - k2 * gu[d];
                    gl[d][[b, e]] = (grad_d_l + ii * k[d] * l_big) * ph;
                }
            }
        }
        (ga, gl, h)
    }
}

/// Synthetic complex B-spline leaf for crate unit tests (deterministic LCG
/// control points on a 5x6x7 cubic grid of an oblique cell, twist `k_cart`).
#[cfg(test)]
pub(crate) fn synthetic_complex_bspline_for_tests(
    num_ao: usize,
    seed: u64,
    k_cart: [f64; 3],
    conjugate: bool,
) -> AosBsplineComplexData {
    let shape = [5usize, 6, 7];
    let order = 3usize;
    let grid = shape[0] * shape[1] * shape[2];
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let cell = [[6.0, 0.3, 0.0], [0.1, 7.0, 0.2], [0.0, 0.4, 8.0]];
    let inv_cell = crate::ewald::inv3(&numpy::ndarray::arr2(&cell).view());
    let n_f = [shape[0] as f64, shape[1] as f64, shape[2] as f64];
    let mut a_mat = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut m = 0.0;
            for inv_cell_kk in &inv_cell {
                m += inv_cell_kk[a] * inv_cell_kk[b];
            }
            a_mat[a][b] = n_f[a] * n_f[b] * m;
        }
    }
    let mut mk = || {
        let coeffs: Vec<f64> = (0..grid * num_ao).map(|_| next()).collect();
        AosBsplineData {
            num_ao,
            shape,
            order,
            coeffs: Arc::new(coeffs),
            inv_cell,
            cell,
            a_mat,
            n_f,
            precision_policy: crate::PrecisionPolicy::full(),
        }
    };
    let re = mk();
    let im = mk();
    AosBsplineComplexData {
        num_ao,
        k_cart: vec![k_cart; num_ao],
        re,
        im,
        conjugate_coefficients: conjugate,
    }
}

#[cfg(test)]
mod bspline_layout_tests {
    use super::*;
    use numpy::ndarray::Array2;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn mk(rng: &mut Lcg, num_ao: usize, shape: [usize; 3], order: usize) -> AosBsplineData {
        let grid = shape[0] * shape[1] * shape[2];
        let band_major: Vec<f64> = (0..num_ao * grid).map(|_| rng.f()).collect();
        let mut coeffs = vec![0.0_f64; grid * num_ao];
        for b in 0..num_ao {
            for g in 0..grid {
                coeffs[g * num_ao + b] = band_major[b * grid + g];
            }
        }
        // simple invertible-ish cell (diagonal + small off-diag)
        let cell = [[6.0, 0.3, 0.0], [0.1, 7.0, 0.2], [0.0, 0.4, 8.0]];
        let inv_cell = crate::ewald::inv3(&numpy::ndarray::arr2(&cell).view());
        let n_f = [shape[0] as f64, shape[1] as f64, shape[2] as f64];
        let mut a_mat = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut m = 0.0;
                for inv_cell_kk in &inv_cell {
                    m += inv_cell_kk[a] * inv_cell_kk[b];
                }
                a_mat[a][b] = n_f[a] * n_f[b] * m;
            }
        }
        AosBsplineData {
            num_ao,
            shape,
            order,
            coeffs: Arc::new(coeffs),
            inv_cell,
            cell,
            a_mat,
            n_f,
            precision_policy: crate::PrecisionPolicy::full(),
        }
    }

    // Reference band-major value_single (the pre-transpose formula) -- must match bit-for-bit.
    fn ref_value_single(d: &AosBsplineData, pos: [f64; 3], out: &mut [f64]) {
        let k = d.order + 1;
        let start = -(((d.order - 1) / 2) as i64);
        let (_n0, n1, n2) = (d.shape[0], d.shape[1], d.shape[2]);
        let wrap =
            |i: i64, n: usize| -> usize { (((i % n as i64) + n as i64) % n as i64) as usize };
        let mut frac = [0.0_f64; 3];
        for (a, frac_a) in frac.iter_mut().enumerate() {
            let mut s = 0.0;
            for (c, pos_c) in pos.iter().enumerate() {
                s += *pos_c * d.inv_cell[c][a];
            }
            *frac_a = s - s.floor();
        }
        let mut tw = [[0.0_f64; 6]; 3];
        let mut idx = [[0usize; 6]; 3];
        for a in 0..3 {
            let t = frac[a] * d.n_f[a];
            let ifl = t.floor();
            let (w, _dw, _d2) = bspline_weights(t - ifl, d.order);
            tw[a] = w;
            let base = ifl as i64;
            for (p, idx_ap) in idx[a][..k].iter_mut().enumerate() {
                *idx_ap = wrap(base + start + p as i64, d.shape[a]);
            }
        }
        let dcoeffs = d.coeffs.as_slice();
        for (b, o) in out.iter_mut().enumerate().take(d.num_ao) {
            let mut v = 0.0;
            for p0 in 0..k {
                let w0 = tw[0][p0];
                let base0 = idx[0][p0] * n1 * n2;
                for p1 in 0..k {
                    let w01 = w0 * tw[1][p1];
                    let base1 = base0 + idx[1][p1] * n2;
                    for p2 in 0..k {
                        let g = base1 + idx[2][p2];
                        v += dcoeffs[g * d.num_ao + b] * w01 * tw[2][p2];
                    }
                }
            }
            *o = v;
        }
    }

    #[test]
    fn value_single_pair_matches_two_value_single_bitexact() {
        let mut rng = Lcg(0x5eed_1234_abcd_0001);
        for &order in &[3usize, 5] {
            let d = mk(&mut rng, 7, [5, 6, 7], order);
            let d2 = mk(&mut rng, 7, [5, 6, 7], order);
            let (mut a1, mut b1) = (vec![0.0; d.num_ao], vec![0.0; d.num_ao]);
            let (mut a2, mut b2) = (vec![1.0; d.num_ao], vec![1.0; d.num_ao]);
            for _ in 0..100 {
                let pos = [rng.f() * 20.0, rng.f() * 20.0, rng.f() * 20.0];
                d.value_single(pos, &mut a1);
                d2.value_single(pos, &mut b1);
                d.value_single_pair(d2.coeffs.as_slice(), pos, &mut a2, &mut b2);
                for i in 0..d.num_ao {
                    assert_eq!(a1[i].to_bits(), a2[i].to_bits(), "order {order} re {i}");
                    assert_eq!(b1[i].to_bits(), b2[i].to_bits(), "order {order} im {i}");
                }
            }
        }
    }

    #[test]
    fn grid_major_value_single_matches_band_outer_reference_bitexact() {
        let mut rng = Lcg(0xdeadbeef01234567);
        for &order in &[3usize, 5] {
            let d = mk(&mut rng, 7, [5, 6, 7], order);
            let mut a = vec![0.0; d.num_ao];
            let mut b = vec![0.0; d.num_ao];
            for _ in 0..200 {
                let pos = [rng.f() * 20.0, rng.f() * 20.0, rng.f() * 20.0];
                d.value_single(pos, &mut a);
                ref_value_single(&d, pos, &mut b);
                for i in 0..d.num_ao {
                    assert!(
                        a[i].to_bits() == b[i].to_bits(),
                        "order {order} band {i}: {} vs {} (not bit-equal)",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    #[test]
    fn batch_value_matches_value_single() {
        // The FLOOR-observable batch value() (precomputed w) vs the per-position sweep
        // value (kept for reference) -- within a tight tolerance (reorder only).
        let mut rng = Lcg(0x0123456789abcdef);
        let d = mk(&mut rng, 6, [6, 5, 7], 5);
        let n_e = 9;
        let mut r = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                r[[e, c]] = rng.f() * 15.0;
            }
        }
        let vb = d.value(&r.view());
        let (vgl, _, _, _, _) = d.value_grad_lap(&r.view());
        let mut single = vec![0.0; d.num_ao];
        for e in 0..n_e {
            d.value_single([r[[e, 0]], r[[e, 1]], r[[e, 2]]], &mut single);
            for b in 0..d.num_ao {
                assert!(
                    (vb[[b, e]] - single[b]).abs()
                        <= crate::setting_consts::ATOL_PYTHON_FP64_LOOSE * single[b].abs().max(1.0),
                    "value batch"
                );
                assert!(
                    (vgl[[b, e]] - single[b]).abs()
                        <= crate::setting_consts::ATOL_PYTHON_FP64_LOOSE * single[b].abs().max(1.0),
                    "vgl value"
                );
            }
        }
    }

    /// `value_grad` reproduces the value and gradient of `value_grad_lap` bitwise, for one
    /// shared k, per-band k, and conjugated coefficients, at points outside the home cell.
    #[test]
    fn complex_value_grad_matches_value_grad_lap_bitexact() {
        let mut rng = Lcg(0x1234_5678_9abc_def1);
        for (kc, conj) in [
            (vec![[0.17, -0.23, 0.31]; 6], false),
            (
                (0..6)
                    .map(|b| [0.1 * b as f64, -0.05, 0.2 - 0.03 * b as f64])
                    .collect(),
                false,
            ),
            (vec![[-0.11, 0.07, 0.29]; 6], true),
        ] {
            let re = mk(&mut rng, 6, [5, 6, 7], 3);
            let im = mk(&mut rng, 6, [5, 6, 7], 3);
            let d = AosBsplineComplexData {
                num_ao: 6,
                k_cart: kc,
                re,
                im,
                conjugate_coefficients: conj,
            };
            let r = Array2::from_shape_fn((9, 3), |_| rng.f() * 20.0 - 6.0);
            let (v0, g0, _) = d.value_grad_lap(&r.view());
            let (v1, g1) = d.value_grad(&r.view());
            assert_eq!(v0, v1);
            for c in 0..3 {
                assert_eq!(g0[c], g1[c]);
            }
        }
    }

    #[test]
    fn complex_time_reversal_sibling_shares_coefficients() {
        let mut rng = Lcg(0x778899aabbccddee);
        let re = mk(&mut rng, 5, [5, 6, 7], 3);
        let im = mk(&mut rng, 5, [5, 6, 7], 3);
        let source = AosBsplineComplexData {
            num_ao: 5,
            k_cart: vec![[0.17, -0.23, 0.31]; 5],
            re,
            im,
            conjugate_coefficients: false,
        };
        let sibling = AosBsplineComplexData {
            num_ao: source.num_ao,
            k_cart: source
                .k_cart
                .iter()
                .map(|k| [-k[0], -k[1], -k[2]])
                .collect(),
            re: source.re.clone(),
            im: source.im.clone(),
            conjugate_coefficients: true,
        };
        assert_eq!(
            source.coefficient_storage_ptrs(),
            sibling.coefficient_storage_ptrs()
        );
        for _ in 0..20 {
            let pos = [rng.f() * 8.0, rng.f() * 8.0, rng.f() * 8.0];
            let up = source.value_single(pos);
            let dn = sibling.value_single(pos);
            for b in 0..source.num_ao {
                assert!(
                    (dn[b] - up[b].conj()).norm()
                        <= crate::setting_consts::ATOL_RS_UNIT_TIME_REVERSAL
                );
            }
        }
    }
}
