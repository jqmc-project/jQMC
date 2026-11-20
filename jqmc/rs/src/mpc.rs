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

//! MPC (Model Periodic Coulomb) finite-size correction -- per-configuration e-e VALUE kernels.
//!
//! Canonical, energy-only (`physical = no`): Ewald is the propagated Hamiltonian and
//! MPC contributes only the passive observable `dV = V_ee^MPC - V_ee^Ewald` accumulated
//! over the SAME Ewald-propagated walkers. There are NO MPC derivatives (force / stress /
//! parameter / position / strain); Rust implements only the two per-configuration value
//! kernels, consuming a pre-built `MPC_data` (+ `Ewald_data` for the correction).
//!
//! ```text
//! V_ee^MPC(R) = 0.5 Sum_{i != j} 1/|r_ij^MI|   (minimum image, bare 1/r; no lattice sum, no screening)
//!             + Sum_i V_LR(r_i)                 (periodic tricubic B-spline of v_lr_coef)
//!             + v_const.
//! ```
//!
//! 1:1 with `jqmc.mpc.compute_mpc_potential_ee` (QMCPACK `MPC::evaluate`). `build_mpc_data`
//! (the trial-density FFT table build) stays in Python/JAX.

use numpy::ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

use crate::ewald::{compute_ewald_potential_ee, det3, inv3, min_image};

/// 1:1 with `jqmc.mpc.MPC_data`. Built once in Python by `build_mpc_data` and read here
/// as owned arrays -- only the fields the per-configuration value kernel needs.
pub struct MpcData {
    /// Real-space cell (rows = lattice vectors), `mpc_data.cell` `(3, 3)` Bohr.
    pub cell: Array2<f64>,
    /// Grid origin (nuclear centroid), `mpc_data.origin` `(3,)` Bohr.
    pub origin: [f64; 3],
    /// Min-image half-width `mpc_data.n_max`.
    pub n_max: i32,
    /// Static FFT grid shape `(n1, n2, n3)` (= `mpc_data.grid_shape`).
    pub grid_shape: (usize, usize, usize),
    /// Periodic cubic B-spline coefficients of the long-range one-body potential,
    /// `mpc_data.v_lr_coef` `(N1, N2, N3)` Ha. The node values `v_lr_grid` are NOT read:
    /// the evaluator interpolates the coefficients (1:1 `jqmc.mpc._tricubic_bspline_interp`).
    pub v_lr_coef: Array3<f64>,
    /// Configuration-independent constant `mpc_data.v_const`, Ha.
    pub v_const: f64,
}

impl MpcData {
    /// Read the Python `MPC_data` dataclass. Fail-closed: any missing field, wrong dtype
    /// or a `grid_shape` that disagrees with `v_lr_coef.shape` is a `PyResult` error.
    pub fn from_py(md: &Bound<'_, PyAny>) -> PyResult<Self> {
        let np = md.py().import("numpy")?;
        let as_arr2 = |o: Bound<'_, PyAny>| -> PyResult<Array2<f64>> {
            let a: numpy::PyReadonlyArray2<f64> = np.call_method1("asarray", (o,))?.extract()?;
            Ok(a.as_array().to_owned())
        };
        let cell = as_arr2(md.getattr("cell")?)?;
        if cell.shape() != [3, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "MPC_data.cell must have shape (3, 3), got {:?}",
                cell.shape()
            )));
        }
        let origin_a: numpy::PyReadonlyArray1<f64> = np
            .call_method1("asarray", (md.getattr("origin")?,))?
            .extract()?;
        let ov = origin_a.as_array();
        if ov.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "MPC_data.origin must have length 3, got {}",
                ov.len()
            )));
        }
        let origin = [ov[0], ov[1], ov[2]];
        let n_max: i32 = md.getattr("n_max")?.extract()?;
        let vlr: numpy::PyReadonlyArray3<f64> = np
            .call_method1("asarray", (md.getattr("v_lr_coef")?,))?
            .extract()?;
        let v_lr_coef = vlr.as_array().to_owned();
        let arr_shape = (
            v_lr_coef.shape()[0],
            v_lr_coef.shape()[1],
            v_lr_coef.shape()[2],
        );
        let grid_shape: (usize, usize, usize) = md.getattr("grid_shape")?.extract()?;
        if grid_shape.0 == 0 || grid_shape.1 == 0 || grid_shape.2 == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "MPC_data.grid_shape dimensions must be positive, got {grid_shape:?}"
            )));
        }
        if grid_shape != arr_shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "MPC_data.grid_shape {grid_shape:?} disagrees with v_lr_coef.shape {arr_shape:?}"
            )));
        }
        let v_const: f64 = np
            .call_method1("asarray", (md.getattr("v_const")?,))?
            .call_method0("item")?
            .extract()?;
        Ok(MpcData {
            cell,
            origin,
            n_max,
            grid_shape,
            v_lr_coef,
            v_const,
        })
    }
}

/// The four uniform cubic B-spline basis weights at fractional offset `t in [0, 1)`, for
/// the nodes `i0 - 1 .. i0 + 2` in that order (1:1 `jqmc.mpc._cubic_bspline_weights`;
/// same operation order so both backends agree to round-off).
#[inline]
fn cubic_bspline_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    [
        omt * omt * omt / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Periodic tricubic B-spline evaluation of a `(n1,n2,n3)` coefficient grid at Cartesian
/// point `re` (1:1 `jqmc.mpc._tricubic_bspline_interp`). `origin` = nuclear centroid;
/// frac = `((r-origin) @ inv_cell) mod 1`; the 4x4x4 stencil around the enclosing node
/// is blended with the cubic B-spline weights, wrapping periodically. `coef` must be the
/// B-spline coefficients built in Python (`_periodic_cubic_bspline_coefficients`), not
/// the node values.
fn tricubic_bspline(
    coef: &ArrayView3<f64>,
    inv_cell: &[[f64; 3]; 3],
    origin: &[f64; 3],
    re: [f64; 3],
) -> f64 {
    let (n1, n2, n3) = (coef.shape()[0], coef.shape()[1], coef.shape()[2]);
    let d = [re[0] - origin[0], re[1] - origin[1], re[2] - origin[2]];
    let mut frac = [0.0_f64; 3];
    for k in 0..3 {
        frac[k] = d[0] * inv_cell[0][k] + d[1] * inv_cell[1][k] + d[2] * inv_cell[2][k];
        frac[k] -= frac[k].floor();
    }
    let gx = frac[0] * n1 as f64;
    let gy = frac[1] * n2 as f64;
    let gz = frac[2] * n3 as f64;
    let i0 = gx.floor() as i64;
    let j0 = gy.floor() as i64;
    let k0 = gz.floor() as i64;
    let wx = cubic_bspline_weights(gx - i0 as f64);
    let wy = cubic_bspline_weights(gy - j0 as f64);
    let wz = cubic_bspline_weights(gz - k0 as f64);
    let wrap = |a: i64, n: usize| -> usize { (((a % n as i64) + n as i64) % n as i64) as usize };
    let mut ii = [0usize; 4];
    let mut jj = [0usize; 4];
    let mut kk = [0usize; 4];
    for a in 0..4 {
        let o = a as i64 - 1;
        ii[a] = wrap(i0 + o, n1);
        jj[a] = wrap(j0 + o, n2);
        kk[a] = wrap(k0 + o, n3);
    }
    // Contract z, then y, then x (matches the einsum nesting of the JAX twin).
    let mut sx = 0.0_f64;
    for a in 0..4 {
        let mut sy = 0.0_f64;
        for b in 0..4 {
            let mut sz = 0.0_f64;
            for c in 0..4 {
                sz += wz[c] * coef[[ii[a], jj[b], kk[c]]];
            }
            sy += wy[b] * sz;
        }
        sx += wx[a] * sy;
    }
    sx
}

/// Canonical MPC e-e potential (Ha) for one configuration, 1:1
/// `jqmc.mpc.compute_mpc_potential_ee`. term1 = `0.5 Sum_{i!=j} 1/|r_ij^MI|` (single nearest
/// image, no lattice sum, no screening); term2 = `Sum_i V_LR(r_i)` (periodic tricubic
/// B-spline of `v_lr_coef`); term3 = `v_const`.
#[allow(clippy::too_many_arguments)]
pub fn compute_mpc_potential_ee(
    cell: &ArrayView2<f64>,
    origin: &[f64; 3],
    n_max: i32,
    v_lr_coef: &ArrayView3<f64>,
    v_const: f64,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let inv = inv3(cell);
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let r = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };

    // Term 1: minimum-image bare-Coulomb pair sum. The i==j diagonal (and any exactly
    // coincident pair) has zero displacement and is masked out, exactly like the JAX
    // `jnp.where(r2 > 0, 1/sqrt(r2), 0)`.
    let mut term1 = 0.0_f64;
    for i in 0..n_e {
        let ri = r(i);
        for j in 0..n_e {
            if i == j {
                continue;
            }
            let rj = r(j);
            let mic = min_image(
                [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]],
                cell,
                &inv,
                n_max,
            );
            let d2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
            if d2 > 0.0 {
                term1 += 1.0 / d2.sqrt();
            }
        }
    }
    term1 *= 0.5;

    // Term 2: one-body long-range potential (tricubic B-spline; origin = nuclear centroid).
    let mut term2 = 0.0_f64;
    for i in 0..n_e {
        term2 += tricubic_bspline(v_lr_coef, &inv, origin, r(i));
    }

    term1 + term2 + v_const
}

/// The Ewald e-e Cartesian tables the MPC correction needs, rebuilt ONCE from the
/// `Ewald_data` integer index sets + the (possibly strained) cell, exactly as the
/// scalar `_compute_mpc_correction_ee` did inline. Shared across every walker of a
/// batch by reference: `MpcEwaldTables` is plain owned data (`Sync`), so the rayon
/// walker closure borrows it read-only. `T_n = idx @ cell`,
/// `G_m = g_idx @ (2 pi inv(cell).T)`, `g_factors = (4 pi / V) exp(-G^2/4 eta^2)/G^2`.
pub struct MpcEwaldTables {
    pub eta: f64,
    pub cell: Array2<f64>,
    pub n_max: i32,
    pub real_images: Array2<f64>,
    pub g_vectors: Array2<f64>,
    pub g_factors: Vec<f64>,
}

/// Rebuild [`MpcEwaldTables`] from the Python `Ewald_data`. Factored out of the scalar
/// `_compute_mpc_correction_ee` so the scalar entry and the batched entry parse the
/// Ewald statics through the SAME path (byte-identical tables). It is also the table
/// source of [`crate::ewald::EwaldData::from_py`], so every native Ewald consumer
/// follows the current `ewald_data.structure_data` (issue #373).
pub fn mpc_ewald_tables_from_py(ewald_data: &Bound<'_, PyAny>) -> PyResult<MpcEwaldTables> {
    let np = ewald_data.py().import("numpy")?;
    // Coerce to float64: the integer index sets (`_real_image_indices_jnp` /
    // `_g_indices_jnp`) are int32, so extract fails without the cast.
    let arr2 = |o: Bound<'_, PyAny>| -> PyResult<Array2<f64>> {
        let a = np
            .call_method1("asarray", (o,))?
            .call_method1("astype", ("float64",))?;
        Ok(a.extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned())
    };
    let eta: f64 = np
        .call_method1("asarray", (ewald_data.getattr("eta")?,))?
        .call_method0("item")?
        .extract()?;
    let esd = ewald_data.getattr("structure_data")?;
    let cell = arr2(esd.getattr("cell")?)?;
    let n_max: i32 = esd.getattr("_n_max")?.extract()?;
    let real_idx = arr2(ewald_data.getattr("_real_image_indices_jnp")?)?;
    let g_idx = arr2(ewald_data.getattr("_g_indices_jnp")?)?;
    let cv = cell.view();
    // real images T_n = idx @ cell
    let n_img = real_idx.nrows();
    let mut real_images = Array2::<f64>::zeros((n_img, 3));
    for n in 0..n_img {
        for d in 0..3 {
            real_images[[n, d]] = real_idx[[n, 0]] * cv[[0, d]]
                + real_idx[[n, 1]] * cv[[1, d]]
                + real_idx[[n, 2]] * cv[[2, d]];
        }
    }
    // reciprocal rows: recip = 2 pi * inv(cell).T  =>  G_m = g_idx @ recip
    let inv = inv3(&cv);
    let two_pi = 2.0 * std::f64::consts::PI;
    let volume = det3(&cv).abs();
    let n_g = g_idx.nrows();
    let mut g_vectors = Array2::<f64>::zeros((n_g, 3));
    let mut g_factors = vec![0.0_f64; n_g];
    for m in 0..n_g {
        let mut gm = [0.0_f64; 3];
        for d in 0..3 {
            // recip[k][d] = 2 pi * inv(cell).T[k][d] = 2 pi * inv[d][k]
            gm[d] = two_pi
                * (g_idx[[m, 0]] * inv[d][0]
                    + g_idx[[m, 1]] * inv[d][1]
                    + g_idx[[m, 2]] * inv[d][2]);
            g_vectors[[m, d]] = gm[d];
        }
        let g2 = gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2];
        g_factors[m] = if g2 > 0.0 {
            (4.0 * std::f64::consts::PI / volume) * (-g2 / (4.0 * eta * eta)).exp() / g2
        } else {
            0.0
        };
    }
    Ok(MpcEwaldTables {
        eta,
        cell,
        n_max,
        real_images,
        g_vectors,
        g_factors,
    })
}

/// Canonical MPC finite-size correction `dV = V_ee^MPC - V_ee^Ewald` (Ha) for ONE
/// configuration, given the pre-parsed [`MpcData`] and [`MpcEwaldTables`]. The
/// per-walker closure of both the scalar and the batched PyO3 entry. Reuses the shared
/// Rust Ewald e-e value kernel; no Ewald reimplementation, no MPC derivative.
pub fn mpc_correction_value(
    m: &MpcData,
    t: &MpcEwaldTables,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let v_mpc = compute_mpc_potential_ee(
        &m.cell.view(),
        &m.origin,
        m.n_max,
        &m.v_lr_coef.view(),
        m.v_const,
        r_up,
        r_dn,
    );
    let v_ewald = compute_ewald_potential_ee(
        r_up,
        r_dn,
        t.eta,
        &t.real_images.view(),
        &t.g_vectors.view(),
        &t.g_factors,
        &t.cell.view(),
        t.n_max,
        None,
    );
    v_mpc - v_ewald
}
