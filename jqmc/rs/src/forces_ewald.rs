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

//! Periodic Ewald nuclear/electron forces. The
//! periodic analog of the molecular bare-Coulomb force. Autograd-free. Tables
//! (eta, real images `T_n`, G-vectors, g-factors) are built in Python and passed
//! in (same contract as `ewald::compute_ewald_potential_*`).
//!
//! Real-space derivative `g'(d) = -2eta/(sqrtpi d)e^{-eta^2d^2} - erfc(etad)/d^2`.

use numpy::ndarray::{Array2, ArrayView2};

use crate::ewald::{fold_parallelepiped, inv3, min_image};

/// Sorted real-space image columns (see `ewald::image_columns`) plus a scratch row.
struct Images {
    ix: Vec<f64>,
    iy: Vec<f64>,
    iz: Vec<f64>,
    inorm: Vec<f64>,
    work: Vec<f64>,
    eta: f64,
}

impl Images {
    fn new(t_n: &ArrayView2<f64>, eta: f64) -> Self {
        let (ix, iy, iz, inorm) = crate::ewald::image_columns(t_n);
        let work = vec![0.0; ix.len()];
        Images {
            ix,
            iy,
            iz,
            inorm,
            work,
            eta,
        }
    }

    /// `(Sum_n g'(d_n)/d_n v_n, Sum_n g'(d_n)/d_n v_n (x) v_n)` over `v_n = mic + T_n`,
    /// `g(d) = erfc(eta d)/d`, leaving out `d_n < NATIVE_DISTANCE_SKIP_THRESHOLD`.
    fn grad(&mut self, mic: [f64; 3], strain: bool) -> ([f64; 3], [[f64; 3]; 3]) {
        let (g, u) = crate::fastmath::ewald_image_grad_sum(
            mic,
            &self.ix,
            &self.iy,
            &self.iz,
            &self.inorm,
            self.eta,
            crate::setting_consts::NATIVE_DISTANCE_SKIP_THRESHOLD,
            strain,
            &mut self.work,
        );
        let vv = [[u[0], u[1], u[2]], [u[1], u[3], u[4]], [u[2], u[4], u[5]]];
        (g, vv)
    }
}

/// `w[c][g] = g_fac[g] * G_g[c]`, the reciprocal force weights.
fn recip_weights(g_m: &ArrayView2<f64>, g_fac: &[f64]) -> [Vec<f64>; 3] {
    let n_g = g_fac.len();
    let col = |c: usize| {
        (0..n_g)
            .map(|g| g_fac[g] * g_m[[g, c]])
            .collect::<Vec<f64>>()
    };
    [col(0), col(1), col(2)]
}

fn add_bracket(out: &mut Array2<f64>, g_m: &ArrayView2<f64>, eta: f64, wg: &[f64]) {
    for (g, &w) in wg.iter().enumerate() {
        let b = bracket([g_m[[g, 0]], g_m[[g, 1]], g_m[[g, 2]]], eta);
        for c in 0..3 {
            for e in 0..3 {
                out[[c, e]] += w * b[c][e];
            }
        }
    }
}

fn collect(r_up: &ArrayView2<f64>, r_dn: &ArrayView2<f64>) -> Vec<[f64; 3]> {
    let mut v = Vec::with_capacity(r_up.nrows() + r_dn.nrows());
    for i in 0..r_up.nrows() {
        v.push([r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]);
    }
    for i in 0..r_dn.nrows() {
        v.push([r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]);
    }
    v
}

/// `dE_ee/dr_i` per electron `(Ne,3)`.
///
/// Real space: each unordered pair once (`mic` is odd and the image set is inversion
/// symmetric, so the `(j,i)` term is minus the `(i,j)` term; the `i == j` image sum
/// vanishes by the same symmetry), over the sorted image prefix inside the `erfc` cut.
/// Reciprocal: the separable-phase electron structure factor of the e_L, then one
/// phase-table sweep per electron.
#[allow(clippy::too_many_arguments)]
pub fn grad_eee_wrt_r(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    t_n: &ArrayView2<f64>,
    g_m: &ArrayView2<f64>,
    g_fac: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
) -> Array2<f64> {
    let r = collect(r_up, r_dn);
    let ne = r.len();
    let inv = inv3(cell);
    let mut out = Array2::<f64>::zeros((ne, 3));
    let mut img = Images::new(t_n, eta);
    for i in 0..ne {
        for j in i + 1..ne {
            let mic = min_image(
                [r[i][0] - r[j][0], r[i][1] - r[j][1], r[i][2] - r[j][2]],
                cell,
                &inv,
                n_max,
            );
            let (gr, _) = img.grad(mic, false);
            for c in 0..3 {
                out[[i, c]] += gr[c];
                out[[j, c]] -= gr[c];
            }
        }
    }
    if !g_fac.is_empty() {
        let (sr, si) = crate::ewald::compute_electron_structure_factor(r_up, r_dn, g_m, cell);
        let msr: Vec<f64> = sr.iter().map(|x| -x).collect();
        let w = recip_weights(g_m, g_fac);
        let mut sep = crate::ewald::SeparablePhase::new(g_m, cell);
        for i in 0..ne {
            sep.load(r[i]);
            let f = sep.dot3([&w[0], &w[1], &w[2]], &si, &msr);
            for c in 0..3 {
                out[[i, c]] += f[c];
            }
        }
    }
    out
}

/// `(dE_ei/dr (Ne,3), dE_ei/dR (NI,3))`.
///
/// Real space: `w = min_image(r_i - R_I) + T_n`. The MIN-IMAGE of the electron-ion
/// displacement matches the ei ENERGY (`compute_ewald_potential_ei`) -- without it the
/// truncated image sum diverges once a walker roams a lattice vector out of the home cell
/// (raw `r_i - R_I - T_n` misses the dominant near image). `d(min_image)/dr = I` a.e.,
/// so the continuous ei force is exact + wrap-invariant. Reciprocal:
/// `Sum_I Z_I sin(G.(r_i - R_I))` factorizes over the ion and electron structure factors.
#[allow(clippy::too_many_arguments)]
pub fn grad_eei(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    t_n: &ArrayView2<f64>,
    g_m: &ArrayView2<f64>,
    g_fac: &[f64],
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    n_max: i32,
) -> (Array2<f64>, Array2<f64>) {
    let r = collect(r_up, r_dn);
    let ne = r.len();
    let ni = r_ion.nrows();
    let inv = crate::ewald::inv3(cell);
    let mut dr = Array2::<f64>::zeros((ne, 3));
    let mut dr_ion = Array2::<f64>::zeros((ni, 3));
    let mut img = Images::new(t_n, eta);
    for i in 0..ne {
        for ii in 0..ni {
            let z = z_ion[ii];
            let mic = min_image(
                [
                    r[i][0] - r_ion[[ii, 0]],
                    r[i][1] - r_ion[[ii, 1]],
                    r[i][2] - r_ion[[ii, 2]],
                ],
                cell,
                &inv,
                n_max,
            );
            let (gr, _) = img.grad(mic, false);
            for c in 0..3 {
                dr[[i, c]] += -z * gr[c]; // -Sum_I Z g' w/d
                dr_ion[[ii, c]] += z * gr[c]; // +Z Sum_{i,n} g' w/d
            }
        }
    }
    if !g_fac.is_empty() {
        // dr_i = Sum_G g G [sin(G.r_i) SI_re - cos(G.r_i) SI_im],
        // dR_I = -Z_I Sum_G g G [SI_e cos(G.R_I) - SR_e sin(G.R_I)].
        let (si_re, si_im) = crate::ewald::compute_ion_structure_factor(g_m, z_ion, r_ion);
        let (sr_e, si_e) = crate::ewald::compute_electron_structure_factor(r_up, r_dn, g_m, cell);
        let m_si_im: Vec<f64> = si_im.iter().map(|x| -x).collect();
        let m_sr_e: Vec<f64> = sr_e.iter().map(|x| -x).collect();
        let w = recip_weights(g_m, g_fac);
        let mut sep = crate::ewald::SeparablePhase::new(g_m, cell);
        for i in 0..ne {
            sep.load(r[i]);
            let f = sep.dot3([&w[0], &w[1], &w[2]], &m_si_im, &si_re);
            for c in 0..3 {
                dr[[i, c]] += f[c];
            }
        }
        for ii in 0..ni {
            sep.load([r_ion[[ii, 0]], r_ion[[ii, 1]], r_ion[[ii, 2]]]);
            let f = sep.dot3([&w[0], &w[1], &w[2]], &si_e, &m_sr_e);
            for c in 0..3 {
                dr_ion[[ii, c]] -= z_ion[ii] * f[c];
            }
        }
    }
    (dr, dr_ion)
}

/// `dE_ii/dR_I` per ion `(NI,3)`. The pair displacement `R_I - R_J` is folded
/// to the parallelepiped-centred image (`ewald::fold_parallelepiped`) before the
/// image sum, as in the forward ii energy, so nuclei given outside the home cell
/// keep all real-space images (issue #52).
#[allow(clippy::too_many_arguments)]
pub fn grad_eii_wrt_r(
    eta: f64,
    t_n: &ArrayView2<f64>,
    g_m: &ArrayView2<f64>,
    g_fac: &[f64],
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> Array2<f64> {
    let inv = inv3(cell);
    let ni = r_ion.nrows();
    let pos = |i: usize| -> [f64; 3] { [r_ion[[i, 0]], r_ion[[i, 1]], r_ion[[i, 2]]] };
    let mut out = Array2::<f64>::zeros((ni, 3));
    let mut img = Images::new(t_n, eta);
    // real: Sum_{J,n} Z_I Z_J g'(d) v/d, v = fold(R_I - R_J) + T_n (the J == I image sum
    // vanishes by the inversion symmetry of the image set)
    for i in 0..ni {
        for j in 0..ni {
            if j == i {
                continue;
            }
            let zz = z_ion[i] * z_ion[j];
            let (ri, rj) = (pos(i), pos(j));
            let dij =
                fold_parallelepiped([ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]], cell, &inv);
            let (gr, _) = img.grad(dij, false);
            for c in 0..3 {
                out[[i, c]] += zz * gr[c];
            }
        }
    }
    if !g_fac.is_empty() {
        // Z_I Sum_G g G [SI_im cos(G.R_I) - SI_re sin(G.R_I)]
        let (si_re, si_im) = crate::ewald::compute_ion_structure_factor(g_m, z_ion, r_ion);
        let m_si_re: Vec<f64> = si_re.iter().map(|x| -x).collect();
        let w = recip_weights(g_m, g_fac);
        let mut sep = crate::ewald::SeparablePhase::new(g_m, cell);
        for i in 0..ni {
            sep.load(pos(i));
            let f = sep.dot3([&w[0], &w[1], &w[2]], &si_im, &m_si_re);
            for c in 0..3 {
                out[[i, c]] += z_ion[i] * f[c];
            }
        }
    }
    out
}

use crate::ewald::det3;

/// Reciprocal bracket `-I + G(x)G/(2eta^2) + 2 G(x)G/|G|^2` (3x3) for one G-vector.
#[inline]
fn bracket(g: [f64; 3], eta: f64) -> [[f64; 3]; 3] {
    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let mut b = [[0.0; 3]; 3];
    for c in 0..3 {
        for d in 0..3 {
            let gg = g[c] * g[d];
            b[c][d] = gg / (2.0 * eta * eta) + 2.0 * gg / g2 - if c == d { 1.0 } else { 0.0 };
        }
    }
    b
}

/// `dE_ee/deps` (3x3).
#[allow(clippy::too_many_arguments)]
pub fn grad_eee_strain(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    t_n: &ArrayView2<f64>,
    g_m: &ArrayView2<f64>,
    g_fac: &[f64],
    cell: &ArrayView2<f64>,
    n_max: i32,
) -> Array2<f64> {
    let r = collect(r_up, r_dn);
    let ne = r.len();
    let inv = inv3(cell);
    let v = det3(cell);
    let mut out = Array2::<f64>::zeros((3, 3));
    let mut img = Images::new(t_n, eta);
    // real: 0.5 Sum_{i,j} (g'/d) v(x)v. The (i,j) and (j,i) terms are equal (the summand is
    // even in the displacement and the image set is symmetric), so each unordered pair is
    // evaluated once with weight 1 and the i == j self-image terms with weight 0.5.
    for i in 0..ne {
        for j in i..ne {
            let pair_w = if i == j { 0.5 } else { 1.0 };
            let mic = min_image(
                [r[i][0] - r[j][0], r[i][1] - r[j][1], r[i][2] - r[j][2]],
                cell,
                &inv,
                n_max,
            );
            let (_, vv) = img.grad(mic, true);
            for c in 0..3 {
                for e in 0..3 {
                    out[[c, e]] += pair_w * vv[c][e];
                }
            }
        }
    }
    // recip: 0.5 Sum_G g_fac |S|^2 bracket
    let (sr, si) = crate::ewald::compute_electron_structure_factor(r_up, r_dn, g_m, cell);
    let wg: Vec<f64> = (0..g_fac.len())
        .map(|g| 0.5 * g_fac[g] * (sr[g] * sr[g] + si[g] * si[g]))
        .collect();
    add_bracket(&mut out, g_m, eta, &wg);
    // background: (pi/(2eta^2V)) N^2 I
    let bg = std::f64::consts::PI / (2.0 * eta * eta * v) * (ne as f64).powi(2);
    for c in 0..3 {
        out[[c, c]] += bg;
    }
    out
}

/// `dE_ei/deps` (3x3).
///
/// Real space: `-Sum_I Z_I Sum_{i,n} (g'/d) w(x)w`, `w = MIN_IMAGE(r_i - R_I) + T_n`. The
/// min-image matches the ei ENERGY (`compute_ewald_potential_ei`) and the JAX strain oracle
/// (`derivatives.ewald.grad_Eei_wrt_strain`) -- without it the truncated image sum diverges
/// once a walker roams a lattice vector out of the home cell (raw `r_i - R_I - T_n` misses
/// the dominant near image; measured 4.0 Ha HF-stress error on a lithium MCMC walker at
/// frac +1.7). Under the affine strain the min-image displacement deforms as `u -> F u`
/// (round index locally constant), so the strain virial of the min-imaged `w` is exact --
/// the identical convention `grad_eei` (force) and `grad_eee_strain` already use.
#[allow(clippy::too_many_arguments)]
pub fn grad_eei_strain(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    eta: f64,
    t_n: &ArrayView2<f64>,
    g_m: &ArrayView2<f64>,
    g_fac: &[f64],
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    n_max: i32,
) -> Array2<f64> {
    let r = collect(r_up, r_dn);
    let ne = r.len();
    let ni = r_ion.nrows();
    let v = det3(cell);
    let inv = inv3(cell);
    let mut out = Array2::<f64>::zeros((3, 3));
    let mut img = Images::new(t_n, eta);
    for r_i in r[..ne].iter() {
        for ii in 0..ni {
            let z = z_ion[ii];
            let mic = min_image(
                [
                    r_i[0] - r_ion[[ii, 0]],
                    r_i[1] - r_ion[[ii, 1]],
                    r_i[2] - r_ion[[ii, 2]],
                ],
                cell,
                &inv,
                n_max,
            );
            let (_, vv) = img.grad(mic, true);
            for c in 0..3 {
                for e in 0..3 {
                    out[[c, e]] += -z * vv[c][e];
                }
            }
        }
    }
    // recip: -Sum_G [Sum_i Sum_I Z_I cos(G*(r_i-R_I))] g_fac bracket  (factored over i,I)
    let z_total: f64 = z_ion.iter().sum();
    let (sec, ses) = crate::ewald::compute_electron_structure_factor(r_up, r_dn, g_m, cell);
    let (sic, sis) = crate::ewald::compute_ion_structure_factor(g_m, z_ion, r_ion);
    let wg: Vec<f64> = (0..g_fac.len())
        .map(|g| -g_fac[g] * (sec[g] * sic[g] + ses[g] * sis[g]))
        .collect();
    add_bracket(&mut out, g_m, eta, &wg);
    // background: -E_bg I, E_bg = (pi/(eta^2V)) Ne SumZ
    let bg = -(std::f64::consts::PI / (eta * eta * v)) * (ne as f64) * z_total;
    for c in 0..3 {
        out[[c, c]] += bg;
    }
    out
}

/// `dE_ii/deps` (3x3).
#[allow(clippy::too_many_arguments)]
pub fn grad_eii_strain(
    eta: f64,
    t_n: &ArrayView2<f64>,
    g_m: &ArrayView2<f64>,
    g_fac: &[f64],
    z_ion: &[f64],
    r_ion: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> Array2<f64> {
    let ni = r_ion.nrows();
    let v = det3(cell);
    let inv = inv3(cell);
    let pos = |i: usize| -> [f64; 3] { [r_ion[[i, 0]], r_ion[[i, 1]], r_ion[[i, 2]]] };
    let mut out = Array2::<f64>::zeros((3, 3));
    let mut img = Images::new(t_n, eta);
    for i in 0..ni {
        for j in 0..ni {
            let zz = z_ion[i] * z_ion[j];
            let (ri, rj) = (pos(i), pos(j));
            // pair displacement folded first, as in the forward ii energy (issue #52)
            let dij =
                fold_parallelepiped([ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]], cell, &inv);
            let (_, vv) = img.grad(dij, true);
            for c in 0..3 {
                for e in 0..3 {
                    out[[c, e]] += 0.5 * zz * vv[c][e];
                }
            }
        }
    }
    let z_total: f64 = z_ion.iter().sum();
    let (sr, si) = crate::ewald::compute_ion_structure_factor(g_m, z_ion, r_ion);
    let wg: Vec<f64> = (0..g_fac.len())
        .map(|g| 0.5 * g_fac[g] * (sr[g] * sr[g] + si[g] * si[g]))
        .collect();
    add_bracket(&mut out, g_m, eta, &wg);
    let bg = std::f64::consts::PI / (2.0 * eta * eta * v) * z_total * z_total;
    for c in 0..3 {
        out[[c, c]] += bg;
    }
    out
}
