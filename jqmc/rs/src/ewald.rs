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

//! Ewald lattice Coulomb (periodic V), 1:1 with `compute_ewald_potential_*`
//! (coulomb_potential.py). Real-space erfc sum over the precomputed
//! `real_image_list`, reciprocal structure-factor sum over `g_vectors`/`g_factors`,
//! plus Gaussian self-energy and the jellium background. The Ewald index sets
//! and `eta` are chosen once in Python (`build_ewald_data`); the Cartesian tables
//! and the ion-ion constant consumed by the production kernels are derived in
//! [`EwaldData::from_py`] from those index sets and the CURRENT
//! `ewald_data.structure_data`, exactly as the XLA kernels rebuild them in-trace
//! (issue #373: the cached `real_image_list` / `g_vectors` / `g_factors` /
//! `ii_energy` describe the structure at build time and go stale when positions or
//! the cell are changed afterwards).

use numpy::ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// Mirrors `jqmc.coulomb_potential.Ewald_data`, owning the
/// precomputed periodic-Coulomb tables plus the cell / min-image count the
/// kernels need. Built once from a periodic `Coulomb_potential_data` via
/// [`EwaldData::from_py`]; value, force, and stress kernels borrow a lightweight
/// [`EwaldRef`].
///
/// It is the single owned parse of these fields; the VMC, GFMC, force and
/// stress drivers all read the Ewald tables through it.
pub struct EwaldData {
    /// Ewald split parameter eta (`ewald_data.eta`).
    pub eta: f64,
    /// Real-space image offsets `T_n = real_image_indices @ cell`, shape `(n_img,3)`.
    pub real_images: Array2<f64>,
    /// Reciprocal vectors `G_m = g_indices @ (2 pi inv(cell).T)`, shape `(n_g,3)`.
    pub g_vectors: Array2<f64>,
    /// Reciprocal weights `(4 pi / V) exp(-G^2 / 4 eta^2) / G^2`, length `n_g`.
    pub g_factors: Vec<f64>,
    /// Ion-ion energy constant (config-independent), computed by
    /// [`compute_ewald_potential_ii`] from the tables above and the current
    /// `ewald_data.structure_data` positions (never the cached `ii_energy`).
    pub ii: f64,
    /// Real-space cell (`ewald_data.structure_data.cell`), rows = lattice vectors.
    pub cell: Array2<f64>,
    /// Min-image replication count `ewald_data.structure_data._n_max`.
    pub n_max: i32,
}

impl EwaldData {
    /// Build the periodic tables for a `Coulomb_potential_data`. Returns `None`
    /// when molecular (`structure_data.pbc_flag == False`).
    ///
    /// The tables are a function of the CURRENT structure, like the XLA kernels
    /// (`_ewald_tables_from_cell` / `compute_ewald_potential_ii`): `T_n`, `G_m` and
    /// `g_factors` are rebuilt from the static integer index sets and
    /// `ewald_data.structure_data.cell` (the same rebuild the MPC correction uses,
    /// [`crate::mpc::mpc_ewald_tables_from_py`]), and `ii` is evaluated from those
    /// tables at `ewald_data.structure_data.positions`. The setup-time caches
    /// `real_image_list` / `g_vectors` / `g_factors` / `ii_energy` are not read: they
    /// describe the structure at `build_ewald_data` time and go stale when ions are
    /// moved or the cell is strained afterwards (issue #373).
    pub fn from_py(cp: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let sd = cp.getattr("structure_data")?;
        if !sd.getattr("pbc_flag")?.extract::<bool>()? {
            return Ok(None);
        }
        let ed = cp.getattr("ewald_data")?;
        let np = cp.py().import("numpy")?;
        let t = crate::mpc::mpc_ewald_tables_from_py(&ed)?;
        let charges: Vec<f64> = np
            .call_method1("asarray", (cp.getattr("_effective_charges")?, "float64"))?
            .call_method0("tolist")?
            .extract()?;
        let esd_positions = ed.getattr("structure_data")?.getattr("positions")?;
        let positions: Array2<f64> = np
            .call_method1("asarray", (esd_positions, "float64"))?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let ii = compute_ewald_potential_ii(
            &charges,
            &positions.view(),
            t.eta,
            &t.real_images.view(),
            &t.g_vectors.view(),
            &t.g_factors,
            &t.cell.view(),
        );
        Ok(Some(EwaldData {
            eta: t.eta,
            real_images: t.real_images,
            g_vectors: t.g_vectors,
            g_factors: t.g_factors,
            ii,
            cell: t.cell,
            n_max: t.n_max,
        }))
    }

    /// Borrow a view-context for the kernels.
    pub fn as_ref(&self) -> EwaldRef<'_> {
        EwaldRef {
            eta: self.eta,
            real_images: self.real_images.view(),
            g_vectors: self.g_vectors.view(),
            g_factors: &self.g_factors,
            ii: self.ii,
            cell: self.cell.view(),
            n_max: self.n_max,
        }
    }
}

/// Borrowed view of an [`EwaldData`] for the value, force, and stress kernels.
/// The GFMC and GFMC-force kernels share this one view type.
pub struct EwaldRef<'a> {
    pub eta: f64,
    pub real_images: ArrayView2<'a, f64>,
    pub g_vectors: ArrayView2<'a, f64>,
    pub g_factors: &'a [f64],
    pub ii: f64,
    pub cell: ArrayView2<'a, f64>,
    pub n_max: i32,
}

/// 3x3 determinant.
#[inline]
pub(crate) fn det3(c: &ArrayView2<f64>) -> f64 {
    c[[0, 0]] * (c[[1, 1]] * c[[2, 2]] - c[[1, 2]] * c[[2, 1]])
        - c[[0, 1]] * (c[[1, 0]] * c[[2, 2]] - c[[1, 2]] * c[[2, 0]])
        + c[[0, 2]] * (c[[1, 0]] * c[[2, 1]] - c[[1, 1]] * c[[2, 0]])
}

/// 3x3 inverse (rows of `cell` are the lattice vectors).
pub(crate) fn inv3(c: &ArrayView2<f64>) -> [[f64; 3]; 3] {
    let d = det3(c);
    let inv_d = 1.0 / d;
    let mut m = [[0.0_f64; 3]; 3];
    m[0][0] = (c[[1, 1]] * c[[2, 2]] - c[[1, 2]] * c[[2, 1]]) * inv_d;
    m[0][1] = (c[[0, 2]] * c[[2, 1]] - c[[0, 1]] * c[[2, 2]]) * inv_d;
    m[0][2] = (c[[0, 1]] * c[[1, 2]] - c[[0, 2]] * c[[1, 1]]) * inv_d;
    m[1][0] = (c[[1, 2]] * c[[2, 0]] - c[[1, 0]] * c[[2, 2]]) * inv_d;
    m[1][1] = (c[[0, 0]] * c[[2, 2]] - c[[0, 2]] * c[[2, 0]]) * inv_d;
    m[1][2] = (c[[0, 2]] * c[[1, 0]] - c[[0, 0]] * c[[1, 2]]) * inv_d;
    m[2][0] = (c[[1, 0]] * c[[2, 1]] - c[[1, 1]] * c[[2, 0]]) * inv_d;
    m[2][1] = (c[[0, 1]] * c[[2, 0]] - c[[0, 0]] * c[[2, 1]]) * inv_d;
    m[2][2] = (c[[0, 0]] * c[[1, 1]] - c[[0, 1]] * c[[1, 0]]) * inv_d;
    m
}

/// Parallelepiped-centred fold `d - round(d @ inv_cell) @ cell` of an ion-ion
/// pair displacement (ties to even, like `numpy.round` / `jnp.round`).
///
/// The real-space image list is complete only for `|R_I - R_J| <= sum_i |a_i|`
/// (`_enumerate_real_images`); the fold guarantees `|d| <= sum_i |a_i| / 2` for
/// any input, so a nucleus given several cells outside the home cell keeps all
/// of its real-space images (issue #52). Mirrors the XLA / numpy ii kernels.
#[inline]
pub(crate) fn fold_parallelepiped(
    diff: [f64; 3],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
) -> [f64; 3] {
    let mut n = [0.0_f64; 3];
    for k in 0..3 {
        let mut s = 0.0;
        for j in 0..3 {
            s += diff[j] * inv[j][k];
        }
        n[k] = s.round_ties_even();
    }
    let mut out = diff;
    for k in 0..3 {
        out[k] -= n[0] * cell[[0, k]] + n[1] * cell[[1, k]] + n[2] * cell[[2, k]];
    }
    out
}

/// Image list `(N,3)` as three contiguous columns plus the image norms, reordered by
/// ascending `|T|` (stable, so `T_0` stays first) for the vectorized sums: the
/// sorted norms let `fastmath::ewald_image_sum` stop at the radius beyond which
/// every term is an exact zero.
pub(crate) fn image_columns(images: &ArrayView2<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = images.nrows();
    let norm = |i: usize| -> f64 {
        let (x, y, z) = (images[[i, 0]], images[[i, 1]], images[[i, 2]]);
        (x * x + y * y + z * z).sqrt()
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| norm(a).total_cmp(&norm(b)));
    (
        order.iter().map(|&i| images[[i, 0]]).collect(),
        order.iter().map(|&i| images[[i, 1]]).collect(),
        order.iter().map(|&i| images[[i, 2]]).collect(),
        order.iter().map(|&i| norm(i)).collect(),
    )
}

/// Wigner-Seitz minimum-image displacement (`_minimum_image_displacement`):
/// parallelepiped-wrap then pick the closest of the `(2n_max+1)^3` translations.
#[inline]
pub(crate) fn min_image(
    diff: [f64; 3],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
) -> [f64; 3] {
    // diff_frac = diff @ inv_cell, wrapped to (-1/2, 1/2]
    let mut frac = [0.0_f64; 3];
    for k in 0..3 {
        let mut s = 0.0;
        for j in 0..3 {
            s += diff[j] * inv[j][k];
        }
        frac[k] = s - s.round();
    }
    // diff_wrapped = frac @ cell
    let mut dw = [0.0_f64; 3];
    for k in 0..3 {
        let mut s = 0.0;
        for j in 0..3 {
            s += frac[j] * cell[[j, k]];
        }
        dw[k] = s;
    }
    let (mut best_d2, mut best) = (f64::INFINITY, dw);
    for na in -n_max..=n_max {
        for nb in -n_max..=n_max {
            for nc in -n_max..=n_max {
                let nf = [na as f64, nb as f64, nc as f64];
                let mut cand = [0.0_f64; 3];
                let mut d2 = 0.0;
                for k in 0..3 {
                    let t = nf[0] * cell[[0, k]] + nf[1] * cell[[1, k]] + nf[2] * cell[[2, k]];
                    cand[k] = dw[k] + t;
                    d2 += cand[k] * cand[k];
                }
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = cand;
                }
            }
        }
    }
    best
}

/// Electron reciprocal structure factor halves `SR_e[g] = Sum_i cos(G_g*r_i)`,
/// `SI_e[g] = Sum_i sin(G_g*r_i)` over all electrons (up then dn). The ee AND ei
/// reciprocal sums both reduce to this same electron SF, so a driver computes it
/// ONCE per e_L and feeds it to both -- turning the ei reciprocal from
/// `O(n_e*n_g)` (per-electron sin/cos) into `O(n_g)`, and sharing the ee sin/cos.
pub fn compute_electron_structure_factor(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_vectors: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    // SEPARABLE-PHASE structure factor. Each G on the reciprocal lattice is
    // `G = Sum_a h_a b_a` with integer Miller indices `h_a` (recovered as
    // `round(G . a_a / 2pi)`, `a_a` = cell row a, since `b_a . a_c = 2pi delta`),
    // so `G . r = 2pi Sum_a h_a frac_a` (`frac = r . inv_cell`) FACTORIZES over
    // the three axes: `e^{iG.r} = e^{ih0 th0} e^{ih1 th1} e^{ih2 th2}`, `th_a = 2pi frac_a`.
    // Build per-electron 1D phase tables `E_a[m] = e^{im th_a}` (one `sin_cos` per
    // (electron, axis, index) -- `O(n_e * Sum_a span_a)` transcendentals instead
    // of the brute-force `O(n_e * n_g)`), then each G's phase is ONE complex triple
    // product (no transcendental). Matches the direct `Sum_i (G.r_i).sin_cos()` to
    // fp rounding (~1e-13); the reciprocal sum it feeds is an energy compared at
    // `_setting` tolerance, and the chunk-vs-unfused e_L stays byte-identical (both
    // sides route through this one function).
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_g = g_vectors.nrows();
    let mut sr_e = vec![0.0_f64; n_g];
    let mut si_e = vec![0.0_f64; n_g];
    if n_g == 0 || n_e == 0 {
        return (sr_e, si_e);
    }
    let mut sep = SeparablePhase::new(g_vectors, cell);
    for i in 0..n_e {
        let r = if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let k = i - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        };
        sep.load(r);
        sep.accumulate(1.0, &mut sr_e, &mut si_e);
    }
    (sr_e, si_e)
}

/// Separable plane-wave phase `e^{iG.r}` over a reciprocal-lattice G list (see
/// [`compute_electron_structure_factor`]): the integer Miller indices are recovered
/// once; per position, [`Self::load`] fills the three 1D phase tables (`Sum_a span_a`
/// `sin_cos` calls) and [`Self::accumulate`] adds `sign * e^{iG.r}` for every G as one
/// complex triple product, in the same operation order the structure factor uses.
#[derive(Clone)]
pub(crate) struct SeparablePhase {
    gi: [Vec<usize>; 3],
    imin: [i32; 3],
    inv: [[f64; 3]; 3],
    ec: [Vec<f64>; 3],
    es: [Vec<f64>; 3],
}

impl SeparablePhase {
    pub(crate) fn new(g_vectors: &ArrayView2<f64>, cell: &ArrayView2<f64>) -> Self {
        let n_g = g_vectors.nrows();
        let inv_twopi = 1.0 / (2.0 * std::f64::consts::PI);
        // Integer Miller indices (constant across electrons) + per-axis index range.
        let mut idx_raw = vec![[0i32; 3]; n_g];
        let (mut imin, mut imax) = ([i32::MAX; 3], [i32::MIN; 3]);
        for g in 0..n_g {
            let gv = [g_vectors[[g, 0]], g_vectors[[g, 1]], g_vectors[[g, 2]]];
            for a in 0..3 {
                let dot = gv[0] * cell[[a, 0]] + gv[1] * cell[[a, 1]] + gv[2] * cell[[a, 2]];
                let m = (dot * inv_twopi).round() as i32;
                idx_raw[g][a] = m;
                imin[a] = imin[a].min(m);
                imax[a] = imax[a].max(m);
            }
        }
        if n_g == 0 {
            imin = [0; 3];
            imax = [-1; 3];
        }
        let span = [
            (imax[0] - imin[0] + 1) as usize,
            (imax[1] - imin[1] + 1) as usize,
            (imax[2] - imin[2] + 1) as usize,
        ];
        // Flatten offset indices for a tight, gather-friendly inner G-loop.
        let mut gi = [vec![0usize; n_g], vec![0usize; n_g], vec![0usize; n_g]];
        for g in 0..n_g {
            for a in 0..3 {
                gi[a][g] = (idx_raw[g][a] - imin[a]) as usize;
            }
        }
        Self {
            gi,
            imin,
            inv: inv3(cell),
            ec: [vec![0.0; span[0]], vec![0.0; span[1]], vec![0.0; span[2]]],
            es: [vec![0.0; span[0]], vec![0.0; span[1]], vec![0.0; span[2]]],
        }
    }

    /// Fill the 1D phase tables `E_a[m] = e^{i m th_a}` at position `r`.
    #[inline]
    pub(crate) fn load(&mut self, r: [f64; 3]) {
        let twopi = 2.0 * std::f64::consts::PI;
        let inv = &self.inv;
        let (x, y, z) = (r[0], r[1], r[2]);
        let theta = [
            twopi * (x * inv[0][0] + y * inv[1][0] + z * inv[2][0]),
            twopi * (x * inv[0][1] + y * inv[1][1] + z * inv[2][1]),
            twopi * (x * inv[0][2] + y * inv[1][2] + z * inv[2][2]),
        ];
        for (a, theta_a) in theta.iter().enumerate() {
            let (ec, es) = (&mut self.ec[a], &mut self.es[a]);
            for j in 0..ec.len() {
                let m = self.imin[a] + j as i32;
                let (s, c) = (m as f64 * *theta_a).sin_cos();
                ec[j] = c;
                es[j] = s;
            }
        }
    }

    /// `Sum_g w[g] * (cos(G.r) * a_re[g] + sin(G.r) * a_im[g])` at the loaded
    /// position, accumulated in G order (the reciprocal el-ion row form).
    #[inline]
    pub(crate) fn dot(&self, w: &[f64], a_re: &[f64], a_im: &[f64]) -> f64 {
        let n_g = self.gi[0].len();
        let (gi0, gi1, gi2) = (&self.gi[0][..n_g], &self.gi[1][..n_g], &self.gi[2][..n_g]);
        let (e0c, e0s) = (&self.ec[0], &self.es[0]);
        let (e1c, e1s) = (&self.ec[1], &self.es[1]);
        let (e2c, e2s) = (&self.ec[2], &self.es[2]);
        let (w, a_re, a_im) = (&w[..n_g], &a_re[..n_g], &a_im[..n_g]);
        let mut acc = 0.0_f64;
        for g in 0..n_g {
            let (c0, s0) = (e0c[gi0[g]], e0s[gi0[g]]);
            let (c1, s1) = (e1c[gi1[g]], e1s[gi1[g]]);
            let (c2, s2) = (e2c[gi2[g]], e2s[gi2[g]]);
            let b_re = c0 * c1 - s0 * s1;
            let b_im = c0 * s1 + s0 * c1;
            let c = b_re * c2 - b_im * s2;
            let s = b_im * c2 + b_re * s2;
            acc += w[g] * (c * a_re[g] + s * a_im[g]);
        }
        acc
    }

    /// Three [`Self::dot`] sums sharing one phase evaluation per G:
    /// `out[c] = Sum_g w[c][g] * (cos(G.r) * a_re[g] + sin(G.r) * a_im[g])`.
    #[inline]
    pub(crate) fn dot3(&self, w: [&[f64]; 3], a_re: &[f64], a_im: &[f64]) -> [f64; 3] {
        let n_g = self.gi[0].len();
        let (gi0, gi1, gi2) = (&self.gi[0][..n_g], &self.gi[1][..n_g], &self.gi[2][..n_g]);
        let (e0c, e0s) = (&self.ec[0], &self.es[0]);
        let (e1c, e1s) = (&self.ec[1], &self.es[1]);
        let (e2c, e2s) = (&self.ec[2], &self.es[2]);
        let (w0, w1, w2) = (&w[0][..n_g], &w[1][..n_g], &w[2][..n_g]);
        let (a_re, a_im) = (&a_re[..n_g], &a_im[..n_g]);
        let mut acc = [0.0_f64; 3];
        for g in 0..n_g {
            let (c0, s0) = (e0c[gi0[g]], e0s[gi0[g]]);
            let (c1, s1) = (e1c[gi1[g]], e1s[gi1[g]]);
            let (c2, s2) = (e2c[gi2[g]], e2s[gi2[g]]);
            let b_re = c0 * c1 - s0 * s1;
            let b_im = c0 * s1 + s0 * c1;
            let c = b_re * c2 - b_im * s2;
            let s = b_im * c2 + b_re * s2;
            let t = c * a_re[g] + s * a_im[g];
            acc[0] += w0[g] * t;
            acc[1] += w1[g] * t;
            acc[2] += w2[g] * t;
        }
        acc
    }

    /// `out_re[g] += Re e^{iG.r}`, `out_im[g] += Im e^{iG.r}` at the loaded position
    /// (`sign = 1.0`), or `-=` for `sign = -1.0` (the sign multiplies the finished
    /// phase, so `+1.0` is the structure factor's exact arithmetic).
    #[inline]
    pub(crate) fn accumulate(&self, sign: f64, out_re: &mut [f64], out_im: &mut [f64]) {
        let n_g = self.gi[0].len();
        let (gi0, gi1, gi2) = (&self.gi[0][..n_g], &self.gi[1][..n_g], &self.gi[2][..n_g]);
        let (e0c, e0s) = (&self.ec[0], &self.es[0]);
        let (e1c, e1s) = (&self.ec[1], &self.es[1]);
        let (e2c, e2s) = (&self.ec[2], &self.es[2]);
        let (out_re, out_im) = (&mut out_re[..n_g], &mut out_im[..n_g]);
        macro_rules! sweep {
            ($op:tt) => {
                for g in 0..n_g {
                    let (c0, s0) = (e0c[gi0[g]], e0s[gi0[g]]);
                    let (c1, s1) = (e1c[gi1[g]], e1s[gi1[g]]);
                    let (c2, s2) = (e2c[gi2[g]], e2s[gi2[g]]);
                    let a_re = c0 * c1 - s0 * s1;
                    let a_im = c0 * s1 + s0 * c1;
                    out_re[g] $op a_re * c2 - a_im * s2;
                    out_im[g] $op a_im * c2 + a_re * s2;
                }
            };
        }
        if sign == 1.0 {
            sweep!(+=);
        } else {
            sweep!(-=);
        }
    }
}

/// Electron-electron Ewald energy (`compute_ewald_potential_ee`, tin-foil BC).
/// `real_images`/`g_vectors` are `(N,3)`, `g_factors` is `(N_g,)`, `cell` rows
/// are lattice vectors. All electrons carry charge -1. `esf` = precomputed
/// electron structure factor ([`compute_electron_structure_factor`]); `None`
/// computes it inline (the bit-exact test path).
#[allow(clippy::too_many_arguments)]
pub fn compute_ewald_potential_ee(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    real_images: &ArrayView2<f64>,
    g_vectors: &ArrayView2<f64>,
    g_factors: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
    esf: Option<(&[f64], &[f64])>,
) -> f64 {
    let inv = inv3(cell);
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let r = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let n_img = real_images.nrows();

    // Real-space: 0.5 Sum_{i,j,n} erfc(eta d)/d, skip the (i=j, n=0) self pair.
    let mut e_real = crate::timed!("ewald_ee.real", {
        // (i,j) and (j,i) are equal: mic is odd and the image list inversion symmetric.
        let (ix, iy, iz, inorm) = image_columns(real_images);
        let mut work = vec![0.0_f64; n_img];
        let mut e_real = 0.0_f64;
        for i in 0..n_e {
            let ri = r(i);
            for j in i..n_e {
                let rj = r(j);
                let mic = min_image(
                    [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]],
                    cell,
                    &inv,
                    n_max,
                );
                let weight = if j == i { 1.0 } else { 2.0 };
                e_real += weight
                    * crate::fastmath::ewald_image_sum(
                        mic, &ix, &iy, &iz, &inorm, eta, None, &mut work,
                    );
            }
        }
        e_real
    });
    e_real *= 0.5;

    // Reciprocal-space: 0.5 Sum_G g_factor[G] |S(G)|^2, S(G) = Sum_i e^{-iG*r_i}.
    // Use the driver's precomputed electron structure factor (shared with ei), or
    // compute it inline for the exact path.
    let owned_esf = match esf {
        Some(_) => None,
        None => Some(crate::timed!(
            "ewald_ee.recip_sf",
            compute_electron_structure_factor(r_up, r_dn, g_vectors, cell)
        )),
    };
    let (sr_e, si_e): (&[f64], &[f64]) = match (esf, &owned_esf) {
        (Some((sr, si)), _) => (sr, si),
        (None, Some((sr, si))) => (sr, si),
        (None, None) => unreachable!(),
    };
    let n_g = g_factors.len();
    let mut e_recip = crate::timed!("ewald_ee.recip", {
        let mut e_recip = 0.0_f64;
        for g in 0..n_g {
            e_recip += g_factors[g] * (sr_e[g] * sr_e[g] + si_e[g] * si_e[g]);
        }
        e_recip
    });
    e_recip *= 0.5;

    let sqrt_pi = std::f64::consts::PI.sqrt();
    let e_self = -(eta / sqrt_pi) * n_e as f64;
    let volume = det3(cell).abs();
    let ne = n_e as f64;
    let e_background = -(std::f64::consts::PI / (2.0 * eta * eta * volume)) * ne * ne;

    e_real + e_recip + e_self + e_background
}

/// Ion reciprocal structure factor halves `SI_re[g] = Sum_I Z_I cos(G_g*R_I)`,
/// `SI_im[g] = Sum_I Z_I sin(G_g*R_I)`. Depends only on the (fixed) ion positions,
/// charges and G-vectors -- CONSTANT across a VMC/GFMC run, so a driver computes
/// it ONCE and feeds it to [`compute_ewald_potential_ei`] instead of paying the
/// `O(n_g*n_atom)` sin/cos every e_L (the dominant `ewald_ei` cost for many-atom
/// cells, e.g. 54% of ei on a 128-atom hydrogen cell).
pub fn compute_ion_structure_factor(
    g_vectors: &ArrayView2<f64>,
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n_g = g_vectors.nrows();
    let n_atom = r_ion.nrows();
    let mut si_re = vec![0.0_f64; n_g];
    let mut si_im = vec![0.0_f64; n_g];
    for g in 0..n_g {
        let (gx, gy, gz) = (g_vectors[[g, 0]], g_vectors[[g, 1]], g_vectors[[g, 2]]);
        let (mut sr, mut sii) = (0.0_f64, 0.0_f64);
        for ia in 0..n_atom {
            let gr = gx * r_ion[[ia, 0]] + gy * r_ion[[ia, 1]] + gz * r_ion[[ia, 2]];
            let (s, c) = gr.sin_cos();
            sr += z_ion[ia] * c;
            sii += z_ion[ia] * s;
        }
        si_re[g] = sr;
        si_im[g] = sii;
    }
    (si_re, si_im)
}

/// Electron-ion Ewald energy (`compute_ewald_potential_ei`), summed over
/// electrons. `E_ei = Sum_i [ -Sum_I Z_I Sum_n erfc(eta d)/d - Sum_G gf[G]*(cos(G*r_i)*SI_re
/// + sin(G*r_i)*SI_im) + pi/(eta^2V)*Sum_I Z_I ]`, with `SI = Sum_I Z_I e^{-iG*R_I}`.
/// No self-energy (cross-species). `z_ion`/`r_ion` are the effective ion charges.
#[allow(clippy::too_many_arguments)]
pub fn compute_ewald_potential_ei(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    real_images: &ArrayView2<f64>,
    g_vectors: &ArrayView2<f64>,
    g_factors: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    // Precomputed ion structure factor (`compute_ion_structure_factor`); `None`
    // => compute inline (the bit-exact test path). A driver caches it once/run.
    si: Option<(&[f64], &[f64])>,
    // Precomputed electron structure factor (`compute_electron_structure_factor`,
    // shared with ee, recomputed each e_L); `Some` => O(n_g) reciprocal.
    esf: Option<(&[f64], &[f64])>,
) -> f64 {
    let inv = inv3(cell);
    let volume = det3(cell).abs();
    let q_ion: f64 = z_ion.iter().sum();
    let bg_per_e = (std::f64::consts::PI / (eta * eta * volume)) * q_ion;
    let n_atom = r_ion.nrows();
    let n_img = real_images.nrows();
    let n_g = g_factors.len();

    // Flatten G columns once (avoids 2D indexing in the per-electron loop).
    let mut gvx = vec![0.0_f64; n_g];
    let mut gvy = vec![0.0_f64; n_g];
    let mut gvz = vec![0.0_f64; n_g];
    for g in 0..n_g {
        gvx[g] = g_vectors[[g, 0]];
        gvy[g] = g_vectors[[g, 1]];
        gvz[g] = g_vectors[[g, 2]];
    }

    // Ion structure-factor halves: use the driver's precomputed cache (ions are
    // fixed -> constant across e_L calls), or compute inline for the exact path.
    let owned_si = match si {
        Some(_) => None,
        None => Some(crate::timed!(
            "ewald_ei.si",
            compute_ion_structure_factor(g_vectors, z_ion, r_ion)
        )),
    };
    let (si_re, si_im): (&[f64], &[f64]) = match (si, &owned_si) {
        (Some((sr, sii)), _) => (sr, sii),
        (None, Some((sr, sii))) => (sr, sii),
        (None, None) => unreachable!(),
    };

    // When the driver supplies the shared electron structure factor, the ei
    // reciprocal `Sum_i Sum_g gf*(cos*SI_re + sin*SI_im)` collapses to the O(n_g) form
    // `Sum_g gf*(SI_re*SR_e + SI_im*SI_e)` (SR_e/SI_e = electron SF) -- no per-electron
    // sin/cos. Otherwise fall back to the per-electron reciprocal (exact path).
    let shared_recip = esf.map(|(sr_e, si_e)| {
        let mut acc = 0.0_f64;
        for g in 0..n_g {
            acc += g_factors[g] * (si_re[g] * sr_e[g] + si_im[g] * si_e[g]);
        }
        acc
    });

    let (ix, iy, iz, inorm) = image_columns(real_images);
    let mut work_buf = vec![0.0_f64; n_img];
    let mut per_e = |ri: [f64; 3]| -> f64 {
        let work = &mut work_buf[..];
        let mut e_real = 0.0_f64;
        for ia in 0..n_atom {
            let mic = min_image(
                [
                    ri[0] - r_ion[[ia, 0]],
                    ri[1] - r_ion[[ia, 1]],
                    ri[2] - r_ion[[ia, 2]],
                ],
                cell,
                &inv,
                n_max,
            );
            e_real += z_ion[ia]
                * crate::fastmath::ewald_image_sum(mic, &ix, &iy, &iz, &inorm, eta, None, work);
        }
        // per-electron reciprocal only when not using the shared electron SF.
        let mut e_recip = 0.0_f64;
        if shared_recip.is_none() {
            for g in 0..n_g {
                let (s, c) = (gvx[g] * ri[0] + gvy[g] * ri[1] + gvz[g] * ri[2]).sin_cos();
                e_recip += g_factors[g] * (c * si_re[g] + s * si_im[g]);
            }
        }
        -e_real - e_recip + bg_per_e
    };

    crate::timed!("ewald_ei.pere", {
        let mut total = 0.0_f64;
        for i in 0..r_up.nrows() {
            total += per_e([r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]);
        }
        for i in 0..r_dn.nrows() {
            total += per_e([r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]);
        }
        // subtract the shared-form reciprocal total (= Sum_i e_recip_i) once.
        total - shared_recip.unwrap_or(0.0)
    })
}

/// Ion-ion Ewald energy from the given tables.
///
/// The production Rust path evaluates it once per context build in
/// [`EwaldData::from_py`] (from tables rebuilt for the current structure) and the
/// e_L kernels add the result as a config-independent constant (`ee + ei + ew.ii`);
/// it is also exposed as a parity binding. It exercises the
/// same generic table consumption (real-space erfc image sum over the ~400-image
/// slab list + reciprocal structure factor + Gaussian self-energy + charged-
/// sublattice jellium background) that the shipped ee/ei kernels use.
///
/// 1:1 with `coulomb_potential.py::_compute_ewald_ii_energy`: the pair
/// displacement `R_I - R_J` is folded to the parallelepiped-centred image
/// ([`fold_parallelepiped`]) before the image sum, so nuclei given outside the
/// home cell are handled (issue #52); the `(I=J, n=0)` self pair excluded, and
/// `charges` are the effective ion charges.
pub fn compute_ewald_potential_ii(
    charges: &[f64],
    positions: &ArrayView2<f64>,
    eta: f64,
    real_images: &ArrayView2<f64>,
    g_vectors: &ArrayView2<f64>,
    g_factors: &[f64],
    cell: &ArrayView2<f64>,
) -> f64 {
    let n = positions.nrows();
    let n_img = real_images.nrows();
    let n_g = g_factors.len();

    // Real-space: 0.5 Sum_{I,J,n} Z_I Z_J erfc(eta d)/d, skip the (I=J, n=0) self pair.
    let inv = inv3(cell);
    let mut e_real = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let [dx, dy, dz] = fold_parallelepiped(
                [
                    positions[[i, 0]] - positions[[j, 0]],
                    positions[[i, 1]] - positions[[j, 1]],
                    positions[[i, 2]] - positions[[j, 2]],
                ],
                cell,
                &inv,
            );
            let zz = charges[i] * charges[j];
            for m in 0..n_img {
                let vx = dx + real_images[[m, 0]];
                let vy = dy + real_images[[m, 1]];
                let vz = dz + real_images[[m, 2]];
                let d2 = vx * vx + vy * vy + vz * vz;
                if d2 > 0.0 {
                    let d = d2.sqrt();
                    e_real += zz * libm::erfc(eta * d) / d;
                }
            }
        }
    }
    e_real *= 0.5;

    // Reciprocal-space: 0.5 Sum_G gf[G] |S(G)|^2, S(G) = Sum_I Z_I e^{-iG*R_I}.
    let mut e_recip = 0.0_f64;
    for g in 0..n_g {
        let (gx, gy, gz) = (g_vectors[[g, 0]], g_vectors[[g, 1]], g_vectors[[g, 2]]);
        let (mut sr, mut si) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let gr = gx * positions[[i, 0]] + gy * positions[[i, 1]] + gz * positions[[i, 2]];
            let (s, c) = gr.sin_cos();
            sr += charges[i] * c;
            si += charges[i] * s;
        }
        e_recip += g_factors[g] * (sr * sr + si * si);
    }
    e_recip *= 0.5;

    // Self-energy and charged-sublattice jellium background (eta-dependent
    // constants; recomputed here, identical closed forms to Python).
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let q2: f64 = charges.iter().map(|z| z * z).sum();
    let e_self = -(eta / sqrt_pi) * q2;
    let volume = det3(cell).abs();
    let q_tot: f64 = charges.iter().sum();
    let e_background = -(std::f64::consts::PI / (2.0 * eta * eta * volume)) * q_tot * q_tot;

    e_real + e_recip + e_self + e_background
}

/// PER-ELECTRON Ewald electron-ion potential (up then dn), for the LRDMC ZV
/// regularization. `alat = None` -> bare (`compute_ewald_potential_ei` decomposed per
/// electron); `alat = Some(a)` -> the discretized variant that floors ONLY the real-space
/// erfc/r term at `max(r, a)` (reciprocal + background pass through unchanged), the periodic
/// analogue of molecular `-Z/max(|r-R|, alat)` (coulomb_potential.py).
#[allow(clippy::too_many_arguments)]
pub fn compute_ewald_potential_ei_per_electron(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    real_images: &ArrayView2<f64>,
    g_vectors: &ArrayView2<f64>,
    g_factors: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    alat: Option<f64>,
) -> Vec<f64> {
    let volume = det3(cell).abs();
    let q_ion: f64 = z_ion.iter().sum();
    let bg_per_e = (std::f64::consts::PI / (eta * eta * volume)) * q_ion;
    let n_atom = r_ion.nrows();
    let n_g = g_factors.len();
    let mut si_re = vec![0.0_f64; n_g];
    let mut si_im = vec![0.0_f64; n_g];
    for g in 0..n_g {
        let (gx, gy, gz) = (g_vectors[[g, 0]], g_vectors[[g, 1]], g_vectors[[g, 2]]);
        let (mut sr, mut sii) = (0.0_f64, 0.0_f64);
        for ia in 0..n_atom {
            let gr = gx * r_ion[[ia, 0]] + gy * r_ion[[ia, 1]] + gz * r_ion[[ia, 2]];
            let (s, c) = gr.sin_cos();
            sr += z_ion[ia] * c;
            sii += z_ion[ia] * s;
        }
        si_re[g] = sr;
        si_im[g] = sii;
    }
    let mut row = EiRowKernel::new(real_images, g_vectors, cell);
    let mut per_e = |ri: [f64; 3]| -> f64 {
        row.eval(
            ri, eta, g_factors, cell, n_max, z_ion, r_ion, &si_re, &si_im, alat, bg_per_e,
        )
    };
    let mut out = Vec::with_capacity(r_up.nrows() + r_dn.nrows());
    for i in 0..r_up.nrows() {
        out.push(per_e([r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]));
    }
    for i in 0..r_dn.nrows() {
        out.push(per_e([r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]));
    }
    out
}

// ===========================================================================
// Incremental single-electron-move Ewald updates for one GFMC/LRDMC
// projection interval.
//
// A projection hop moves EXACTLY ONE electron by one lattice vector (or to one
// ECP T-move grid point). The measured per-hop budget (a 24-electron periodic
// B-spline cell) is ~76% the two Ewald sums, both recomputed in full each hop. These
// helpers turn each into a one-electron delta:
//   * e-i is trivially exact: `compute_ewald_potential_ei_per_electron` returns a
//     per-electron array whose row `per_e(r_i)` depends ONLY on electron `i`, so a
//     hop recomputes ONLY the moved row -- O(N_ion*n_img + n_g) vs O(N * ...).
//   * e-e carries the electron structure factor S(G) and the running real-space
//     half-sum; a hop applies a pairwise delta against the (N-1) unmoved electrons
//     plus an O(n_g) S(G) update -- O(N*n_img + n_g) vs O(N^2*n_img).
//
// The incremental value equals the full recompute modulo fp reordering; the
// twin tests below pin |incremental - full| within a tight absolute tolerance.
// These pure additive kernels update only the carried interval state.
// ===========================================================================

/// Single-electron el-ion Ewald contribution `per_e(ri)` -- the exact body of the
/// `compute_ewald_potential_ei_per_electron` per-electron closure, exposed so a
/// projection hop can recompute ONLY the moved electron's row. `si_re`/`si_im`
/// are the (constant, ion) reciprocal structure-factor halves
/// (`compute_ion_structure_factor`); pass them once per interval. `alat=Some(a)`
/// selects the discretized (floored) form, `None` the bare form -- matching the
/// two calls the ZV composition makes.
#[allow(clippy::too_many_arguments)]
pub fn ewald_ei_single_electron(
    ri: [f64; 3],
    eta: f64,
    real_images: &ArrayView2<f64>,
    g_vectors: &ArrayView2<f64>,
    g_factors: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    si_re: &[f64],
    si_im: &[f64],
    alat: Option<f64>,
) -> f64 {
    let mut row = EiRowKernel::new(real_images, g_vectors, cell);
    ewald_ei_single_electron_with(
        &mut row, ri, eta, g_factors, cell, n_max, z_ion, r_ion, si_re, si_im, alat,
    )
}

/// [`ewald_ei_single_electron`] on caller-held interval-constant tables (the image
/// columns, the cell inverse and the separable-phase Miller tables), so a projection
/// hop does not rebuild them. Same arithmetic, same value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ewald_ei_single_electron_with(
    row: &mut EiRowKernel,
    ri: [f64; 3],
    eta: f64,
    g_factors: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    si_re: &[f64],
    si_im: &[f64],
    alat: Option<f64>,
) -> f64 {
    let volume = det3(cell).abs();
    let q_ion: f64 = z_ion.iter().sum();
    let bg_per_e = (std::f64::consts::PI / (eta * eta * volume)) * q_ion;
    row.eval(
        ri, eta, g_factors, cell, n_max, z_ion, r_ion, si_re, si_im, alat, bg_per_e,
    )
}

/// One electron's el-ion Ewald row `per_e(ri)`: real-space image sums against every
/// ion (`fastmath::ewald_image_sum`) and the reciprocal sum against the ion structure
/// factor through the separable phase of [`compute_electron_structure_factor`]. The
/// ONE row kernel of both the full per-electron array and the single-electron hop
/// update, so a recomputed row is bit-identical to the full kernel's.
#[derive(Clone)]
pub(crate) struct EiRowKernel {
    ix: Vec<f64>,
    iy: Vec<f64>,
    iz: Vec<f64>,
    inorm: Vec<f64>,
    work: Vec<f64>,
    inv: [[f64; 3]; 3],
    sep: SeparablePhase,
}

impl EiRowKernel {
    pub(crate) fn new(
        real_images: &ArrayView2<f64>,
        g_vectors: &ArrayView2<f64>,
        cell: &ArrayView2<f64>,
    ) -> Self {
        let (ix, iy, iz, inorm) = image_columns(real_images);
        let work = vec![0.0_f64; ix.len()];
        Self {
            ix,
            iy,
            iz,
            inorm,
            work,
            inv: inv3(cell),
            sep: SeparablePhase::new(g_vectors, cell),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval(
        &mut self,
        ri: [f64; 3],
        eta: f64,
        g_factors: &[f64],
        cell: &ArrayView2<f64>,
        n_max: i32,
        z_ion: &[f64],
        r_ion: &ArrayView2<f64>,
        si_re: &[f64],
        si_im: &[f64],
        alat: Option<f64>,
        bg_per_e: f64,
    ) -> f64 {
        let mut e_real = 0.0_f64;
        for ia in 0..r_ion.nrows() {
            let mic = min_image(
                [
                    ri[0] - r_ion[[ia, 0]],
                    ri[1] - r_ion[[ia, 1]],
                    ri[2] - r_ion[[ia, 2]],
                ],
                cell,
                &self.inv,
                n_max,
            );
            e_real += z_ion[ia]
                * crate::fastmath::ewald_image_sum(
                    mic,
                    &self.ix,
                    &self.iy,
                    &self.iz,
                    &self.inorm,
                    eta,
                    alat,
                    &mut self.work,
                );
        }
        debug_assert_eq!(g_factors.len(), self.sep.gi[0].len());
        self.sep.load(ri);
        let e_recip = self.sep.dot(g_factors, si_re, si_im);
        -e_real - e_recip + bg_per_e
    }
}

/// Carried electron-electron Ewald state for one projection interval: the
/// running real-space half-sum and the electron structure factor S(G). Built
/// once via [`EwaldEeState::build`]; advanced per hop via [`EwaldEeState::advance`]
/// at O(N*n_img + n_g); the interval energy is [`EwaldEeState::energy`].
#[derive(Clone)]
pub struct EwaldEeState {
    /// `0.5 * Sum_{i,j,n != self} erfc(eta d)/d`.
    e_real_half: f64,
    /// `SR[g] = Sum_i cos(G_g . r_i)` (electron charge sign folds out of |S|^2).
    s_re: Vec<f64>,
    /// `SI[g] = Sum_i sin(G_g . r_i)`.
    s_im: Vec<f64>,
    /// Position-independent Gaussian self-energy `-(eta/sqrt(pi)) N`.
    e_self: f64,
    /// Position-independent jellium background `-(pi/(2 eta^2 V)) N^2`.
    e_bg: f64,
    /// Interval-constant tables for [`EwaldEeState::advance`]: the real-space image
    /// columns (the `fastmath::ewald_image_sum` layout the build uses), its scratch,
    /// the cell inverse, and the separable-phase Miller tables of the G list.
    ix: Vec<f64>,
    iy: Vec<f64>,
    iz: Vec<f64>,
    inorm: Vec<f64>,
    work: Vec<f64>,
    inv: [[f64; 3]; 3],
    sep: SeparablePhase,
    /// Pair image sums `phi[i * n_e + j] = Sum_n erfc(eta d)/d` of the current
    /// configuration (symmetric, diagonal = self-image sum), kept current by
    /// [`EwaldEeState::advance`] so a hop reads the mover's OLD row instead of
    /// re-evaluating it.
    phi: Vec<f64>,
    n_e: usize,
}

impl EwaldEeState {
    /// Full build from the current electron configuration (the interval start and
    /// every periodic drift-refresh). Real-space uses the same ordered double loop
    /// as [`compute_ewald_potential_ee`]; S(G) uses the SAME
    /// [`compute_electron_structure_factor`] the reference kernel calls, so the
    /// built state's [`EwaldEeState::energy`] is BYTE-IDENTICAL to the full kernel
    /// at the build configuration (drift enters only through later `advance`s).
    pub fn build(
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        eta: f64,
        real_images: &ArrayView2<f64>,
        g_vectors: &ArrayView2<f64>,
        cell: &ArrayView2<f64>,
        n_max: i32,
    ) -> Self {
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
        // Same accumulation as the real-space sum of `compute_ewald_potential_ee`,
        // so the built value is byte-identical to it at this configuration.
        let n_img = real_images.nrows();
        // (i,j) and (j,i) are equal: mic is odd and the image list inversion symmetric.
        let (ix, iy, iz, inorm) = image_columns(real_images);
        let mut work = vec![0.0_f64; n_img];
        let mut e_real = 0.0_f64;
        let mut phi = vec![0.0_f64; n_e * n_e];
        for i in 0..n_e {
            let ri = r(i);
            for j in i..n_e {
                let rj = r(j);
                let mic = min_image(
                    [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]],
                    cell,
                    &inv,
                    n_max,
                );
                let weight = if j == i { 1.0 } else { 2.0 };
                let pair = crate::fastmath::ewald_image_sum(
                    mic, &ix, &iy, &iz, &inorm, eta, None, &mut work,
                );
                phi[i * n_e + j] = pair;
                phi[j * n_e + i] = pair;
                e_real += weight * pair;
            }
        }
        let (s_re, s_im) = compute_electron_structure_factor(r_up, r_dn, g_vectors, cell);
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let e_self = -(eta / sqrt_pi) * n_e as f64;
        let volume = det3(cell).abs();
        let ne = n_e as f64;
        let e_bg = -(std::f64::consts::PI / (2.0 * eta * eta * volume)) * ne * ne;
        Self {
            e_real_half: 0.5 * e_real,
            s_re,
            s_im,
            e_self,
            e_bg,
            ix,
            iy,
            iz,
            inorm,
            work,
            inv,
            sep: SeparablePhase::new(g_vectors, cell),
            phi,
            n_e,
        }
    }

    /// Total ee Ewald energy from the carried state (`0.5 E_real + 0.5 E_recip +
    /// E_self + E_bg`), 1:1 with [`compute_ewald_potential_ee`].
    pub fn energy(&self, g_factors: &[f64]) -> f64 {
        let mut e_recip = 0.0_f64;
        for (g, g_factors_g) in g_factors.iter().enumerate() {
            e_recip += *g_factors_g * (self.s_re[g] * self.s_re[g] + self.s_im[g] * self.s_im[g]);
        }
        self.e_real_half + 0.5 * e_recip + self.e_self + self.e_bg
    }

    /// Apply a single-electron move of global electron `mover` (`old` -> `new`; up
    /// electrons first, then down) given the CURRENT positions of all OTHER electrons
    /// (`others`, length N-1, in global order with `mover` skipped). Updates the
    /// real-space half-sum, the carried pair table and S(G) in place.
    /// O(|others|*n_img + n_g).
    ///
    /// Both halves use the kernels [`EwaldEeState::build`] uses: the real-space pair
    /// image sum is `fastmath::ewald_image_sum` (the build's and the full ee kernel's
    /// erfc, masking the `d = 0` self image), and the S(G) delta is the separable
    /// phase of [`compute_electron_structure_factor`] at `new` minus that at `old`
    /// (`Sum_a span_a` transcendentals per position instead of one `sin_cos` per G).
    /// The carried state therefore tracks the full recompute to fp reordering only.
    #[allow(clippy::too_many_arguments)]
    pub fn advance(
        &mut self,
        mover: usize,
        others: &[[f64; 3]],
        old: [f64; 3],
        new: [f64; 3],
        eta: f64,
        real_images: &ArrayView2<f64>,
        g_vectors: &ArrayView2<f64>,
        cell: &ArrayView2<f64>,
        n_max: i32,
    ) {
        debug_assert_eq!(real_images.nrows(), self.ix.len());
        debug_assert_eq!(g_vectors.nrows(), self.s_re.len());
        let inv = self.inv;
        // Real-space: the (i,j) double loop counts both (p,j) and (j,p); pair sum is
        // symmetric, so the net change to 0.5*Sum is Sum_{j != p}[phi(new,j)-phi(old,j)].
        // The mover's OLD pair values are the carried table row: each entry was
        // evaluated at the current positions of both electrons (neither has moved
        // since), so only the NEW row is evaluated here.
        let n_e = self.n_e;
        debug_assert_eq!(others.len() + 1, n_e);
        let mut d_real = 0.0_f64;
        for (idx, &rj) in others.iter().enumerate() {
            let j = if idx < mover { idx } else { idx + 1 };
            let mic_new = min_image(
                [new[0] - rj[0], new[1] - rj[1], new[2] - rj[2]],
                cell,
                &inv,
                n_max,
            );
            let phi_new = crate::fastmath::ewald_image_sum(
                mic_new,
                &self.ix,
                &self.iy,
                &self.iz,
                &self.inorm,
                eta,
                None,
                &mut self.work,
            );
            let phi_old = self.phi[mover * n_e + j];
            d_real += phi_new - phi_old;
            self.phi[mover * n_e + j] = phi_new;
            self.phi[j * n_e + mover] = phi_new;
        }
        self.e_real_half += d_real;
        // Reciprocal: S(G) += (phase(new) - phase(old)).
        self.sep.load(new);
        self.sep.accumulate(1.0, &mut self.s_re, &mut self.s_im);
        self.sep.load(old);
        self.sep.accumulate(-1.0, &mut self.s_re, &mut self.s_im);
    }
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use numpy::ndarray::arr2;

    // Deterministic LCG so the test needs no rng dep.
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    // Small cubic-cell Ewald fixture: the delta==full identity holds over ANY
    // (cell, image, g) tables, so unconverged toy tables are fine here.
    fn fixture() -> (
        Array2<f64>, // cell
        Array2<f64>, // real_images
        Array2<f64>, // g_vectors
        Vec<f64>,    // g_factors
        f64,         // eta
        i32,         // n_max
    ) {
        let l = 6.0_f64;
        let cell = arr2(&[[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);
        // A few real-space images (home + +/- along each axis).
        let mut imgs = vec![[0.0, 0.0, 0.0]];
        for s in [-l, l] {
            imgs.push([s, 0.0, 0.0]);
            imgs.push([0.0, s, 0.0]);
            imgs.push([0.0, 0.0, s]);
        }
        let real_images = Array2::from_shape_vec(
            (imgs.len(), 3),
            imgs.iter().flat_map(|v| v.iter().copied()).collect(),
        )
        .unwrap();
        // A handful of reciprocal vectors b = 2pi/l along axes + a diagonal.
        let b = 2.0 * std::f64::consts::PI / l;
        let gv = [
            [b, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, b],
            [b, b, 0.0],
            [b, 0.0, b],
            [2.0 * b, 0.0, 0.0],
        ];
        let g_vectors = Array2::from_shape_vec(
            (gv.len(), 3),
            gv.iter().flat_map(|v| v.iter().copied()).collect(),
        )
        .unwrap();
        let eta = 0.8_f64;
        let volume = l * l * l;
        // gf[g] = (4 pi / V) exp(-|G|^2/(4 eta^2)) / |G|^2  (standard Ewald recip weight).
        let g_factors: Vec<f64> = gv
            .iter()
            .map(|g| {
                let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                (4.0 * std::f64::consts::PI / volume) * (-g2 / (4.0 * eta * eta)).exp() / g2
            })
            .collect();
        (cell, real_images, g_vectors, g_factors, eta, 0)
    }

    fn split(positions: &[[f64; 3]], n_up: usize) -> (Array2<f64>, Array2<f64>) {
        let up = Array2::from_shape_vec(
            (n_up, 3),
            positions[..n_up]
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect(),
        )
        .unwrap();
        let dn = Array2::from_shape_vec(
            (positions.len() - n_up, 3),
            positions[n_up..]
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect(),
        )
        .unwrap();
        (up, dn)
    }

    #[test]
    fn ee_incremental_matches_full_recompute() {
        let (cell, real_images, g_vectors, g_factors, eta, n_max) = fixture();
        let n_up = 3usize;
        let n = 6usize;
        let mut rng = Lcg(0x1234_5678);
        let mut pos: Vec<[f64; 3]> = (0..n)
            .map(|_| [rng.f() * 6.0, rng.f() * 6.0, rng.f() * 6.0])
            .collect();

        // Baseline: incremental state == production full kernel at config 0.
        let (up0, dn0) = split(&pos, n_up);
        let full0 = compute_ewald_potential_ee(
            &up0.view(),
            &dn0.view(),
            eta,
            &real_images.view(),
            &g_vectors.view(),
            &g_factors,
            &cell.view(),
            n_max,
            None,
        );
        let mut st = EwaldEeState::build(
            &up0.view(),
            &dn0.view(),
            eta,
            &real_images.view(),
            &g_vectors.view(),
            &cell.view(),
            n_max,
        );
        assert!(
            (st.energy(&g_factors) - full0).abs()
                < crate::setting_consts::ATOL_GFMC_INCREMENTAL_TWIN,
            "ee build vs full: {} vs {}",
            st.energy(&g_factors),
            full0
        );

        // Ten random single-electron hops; after each, incremental == full recompute.
        for hop in 0..10 {
            let p = (rng.f() * n as f64) as usize % n;
            let old = pos[p];
            let new = [
                old[0] + (rng.f() - 0.5),
                old[1] + (rng.f() - 0.5),
                old[2] + (rng.f() - 0.5),
            ];
            let others: Vec<[f64; 3]> = pos
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != p)
                .map(|(_, r)| *r)
                .collect();
            st.advance(
                p,
                &others,
                old,
                new,
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &cell.view(),
                n_max,
            );
            pos[p] = new;
            let (up, dn) = split(&pos, n_up);
            let full = compute_ewald_potential_ee(
                &up.view(),
                &dn.view(),
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &g_factors,
                &cell.view(),
                n_max,
                None,
            );
            let inc = st.energy(&g_factors);
            assert!(
                (inc - full).abs() < crate::setting_consts::ATOL_GFMC_INCREMENTAL_TWIN,
                "hop {hop}: ee inc {inc} vs full {full} (dabs {})",
                (inc - full).abs()
            );
        }
    }

    #[test]
    fn ei_incremental_only_moved_row_changes() {
        let (cell, real_images, g_vectors, g_factors, eta, n_max) = fixture();
        let n_up = 3usize;
        let n = 6usize;
        let z_ion = vec![4.0_f64, 1.0, 1.0]; // toy effective charges
        let r_ion = arr2(&[[1.0, 1.0, 1.0], [3.0, 3.0, 1.0], [5.0, 1.0, 4.0]]);
        let (si_re, si_im) = compute_ion_structure_factor(&g_vectors.view(), &z_ion, &r_ion.view());

        let mut rng = Lcg(0xC0FFEE);
        let mut pos: Vec<[f64; 3]> = (0..n)
            .map(|_| [rng.f() * 6.0, rng.f() * 6.0, rng.f() * 6.0])
            .collect();

        for alat in [None, Some(0.3_f64)] {
            let (up0, dn0) = split(&pos, n_up);
            let row0 = compute_ewald_potential_ei_per_electron(
                &up0.view(),
                &dn0.view(),
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &g_factors,
                &cell.view(),
                n_max,
                &z_ion,
                &r_ion.view(),
                alat,
            );
            // Single-electron helper reproduces every row exactly (bit-identical body).
            for (i, ri) in pos.iter().enumerate() {
                let single = ewald_ei_single_electron(
                    *ri,
                    eta,
                    &real_images.view(),
                    &g_vectors.view(),
                    &g_factors,
                    &cell.view(),
                    n_max,
                    &z_ion,
                    &r_ion.view(),
                    &si_re,
                    &si_im,
                    alat,
                );
                assert_eq!(
                    single, row0[i],
                    "ei single vs full row {i} (alat {:?})",
                    alat
                );
            }
            // Move electron p; only row p changes, and it equals the helper on r_new.
            let p = 2usize;
            let old = pos[p];
            let new = [old[0] + 0.4, old[1] - 0.3, old[2] + 0.2];
            let mut pos2 = pos.clone();
            pos2[p] = new;
            let (up1, dn1) = split(&pos2, n_up);
            let row1 = compute_ewald_potential_ei_per_electron(
                &up1.view(),
                &dn1.view(),
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &g_factors,
                &cell.view(),
                n_max,
                &z_ion,
                &r_ion.view(),
                alat,
            );
            for i in 0..n {
                if i == p {
                    let single = ewald_ei_single_electron(
                        new,
                        eta,
                        &real_images.view(),
                        &g_vectors.view(),
                        &g_factors,
                        &cell.view(),
                        n_max,
                        &z_ion,
                        &r_ion.view(),
                        &si_re,
                        &si_im,
                        alat,
                    );
                    assert_eq!(single, row1[i], "moved row (alat {:?})", alat);
                } else {
                    assert_eq!(
                        row0[i], row1[i],
                        "unmoved row {i} changed (alat {:?})",
                        alat
                    );
                }
            }
        }
        let _ = pos.pop();
    }

    // Representative-size per-hop micro-benchmark (24 e-, 19 ions, slab-like image
    // list, ~1000 G): full recompute vs one-electron incremental for ee + ei.
    // `cargo test --release -- --ignored --nocapture ewald_incremental_bench`.
    #[test]
    #[ignore]
    fn ewald_incremental_bench() {
        use std::time::Instant;
        let l = 8.0_f64;
        let cell = arr2(&[[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]);
        let eta = 0.7_f64;
        let n_max = 0;
        // ~400 real-space images (7x7x7 minus corners ~ slab-list scale).
        let mut imgs = Vec::new();
        for a in -3..=3 {
            for b in -3..=3 {
                for c in -3..=3 {
                    imgs.push([a as f64 * l, b as f64 * l, c as f64 * l]);
                }
            }
        }
        let real_images = Array2::from_shape_vec(
            (imgs.len(), 3),
            imgs.iter().flat_map(|v| v.iter().copied()).collect(),
        )
        .unwrap();
        // ~1000 reciprocal vectors.
        let b0 = 2.0 * std::f64::consts::PI / l;
        let mut gv = Vec::new();
        for a in -5..=5 {
            for b in -5..=5 {
                for c in -5..=5 {
                    if a == 0 && b == 0 && c == 0 {
                        continue;
                    }
                    gv.push([a as f64 * b0, b as f64 * b0, c as f64 * b0]);
                }
            }
        }
        let g_vectors = Array2::from_shape_vec(
            (gv.len(), 3),
            gv.iter().flat_map(|v| v.iter().copied()).collect(),
        )
        .unwrap();
        let volume = l * l * l;
        let g_factors: Vec<f64> = gv
            .iter()
            .map(|g| {
                let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                (4.0 * std::f64::consts::PI / volume) * (-g2 / (4.0 * eta * eta)).exp() / g2
            })
            .collect();
        let n = 24usize;
        let n_up = 12usize;
        let mut rng = Lcg(0xBEEF);
        let mut pos: Vec<[f64; 3]> = (0..n)
            .map(|_| [rng.f() * l, rng.f() * l, rng.f() * l])
            .collect();
        let z_ion = vec![3.0_f64; 19];
        let r_ion = Array2::from_shape_fn((19, 3), |(_, _)| rng.f() * l);
        let (si_re, si_im) = compute_ion_structure_factor(&g_vectors.view(), &z_ion, &r_ion.view());

        let iters = 200;
        // --- ee full recompute per hop ---
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..iters {
            let (up, dn) = split(&pos, n_up);
            acc += compute_ewald_potential_ee(
                &up.view(),
                &dn.view(),
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &g_factors,
                &cell.view(),
                n_max,
                None,
            );
        }
        let ee_full = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        // --- ee incremental per hop ---
        let (bup, bdn) = split(&pos, n_up);
        let mut st = EwaldEeState::build(
            &bup.view(),
            &bdn.view(),
            eta,
            &real_images.view(),
            &g_vectors.view(),
            &cell.view(),
            n_max,
        );
        let t0 = Instant::now();
        for h in 0..iters {
            let p = h % n;
            let old = pos[p];
            let new = [old[0] + 0.1, old[1] - 0.1, old[2] + 0.05];
            let others: Vec<[f64; 3]> = pos
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != p)
                .map(|(_, r)| *r)
                .collect();
            st.advance(
                p,
                &others,
                old,
                new,
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &cell.view(),
                n_max,
            );
            pos[p] = new;
            acc += st.energy(&g_factors);
        }
        let ee_inc = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        // --- ei full (all rows) vs single row ---
        let t0 = Instant::now();
        for _ in 0..iters {
            let (up, dn) = split(&pos, n_up);
            let v = compute_ewald_potential_ei_per_electron(
                &up.view(),
                &dn.view(),
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &g_factors,
                &cell.view(),
                n_max,
                &z_ion,
                &r_ion.view(),
                None,
            );
            acc += v[0];
        }
        let ei_full = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        let t0 = Instant::now();
        for h in 0..iters {
            acc += ewald_ei_single_electron(
                pos[h % n],
                eta,
                &real_images.view(),
                &g_vectors.view(),
                &g_factors,
                &cell.view(),
                n_max,
                &z_ion,
                &r_ion.view(),
                &si_re,
                &si_im,
                None,
            );
        }
        let ei_inc = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        println!(
            "EWALD_BENCH n_e={n} n_ion=19 n_img={} n_g={} (acc {acc:.3e})",
            imgs.len(),
            gv.len()
        );
        println!(
            "  ee : full {ee_full:.4} ms/hop  incr {ee_inc:.4} ms/hop  speedup {:.1}x",
            ee_full / ee_inc
        );
        println!(
            "  ei : full {ei_full:.4} ms/hop  incr {ei_inc:.4} ms/hop  speedup {:.1}x",
            ei_full / ei_inc
        );
        println!(
            "  ee+ei combined per-hop: full {:.4} -> incr {:.4} ms  ({:.1}x)",
            ee_full + ei_full,
            ee_inc + ei_inc,
            (ee_full + ei_full) / (ee_inc + ei_inc)
        );
    }
}
