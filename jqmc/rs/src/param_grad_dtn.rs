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

//! DTN parameter gradients `O_k = dJ_DTN/d{alpha,beta,gamma}_indep`. J_DTN is linear in the inflated coeffs
//! (`full = M*indep + b`), so `O_k = (channel feature sums) @ M`. Autograd-free.

use numpy::ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

use crate::data::JastrowData;
use crate::determinant::{GeminalData, OrbData};
use crate::dtn::{cutoff_derivs, JastrowDtnData};
use crate::jastrow_factor::{min_image_dist, CellInfo};
use crate::param_grad_el::{det_grad_field, jastrow_grads};

/// DTN data plus the inflation matrices `M_{alpha,beta,gamma}` and polynomial orders.
pub struct DtnOkData {
    pub base: JastrowDtnData,
    pub n_order_u: usize,
    pub n_order_chi: usize,
    pub m_alpha: Vec<f64>, // (2, Nu+1, na)
    pub na: usize,
    pub m_beta: Vec<f64>, // (n_sp, Nc+1, nb)
    pub nb: usize,
    pub m_gamma: Vec<f64>, // (n_sp, NeN+1, NeN+1, Nee+1, ng)
    pub ng: usize,
}

fn ravel_f64(o: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let py = o.py();
    let np = py.import("numpy")?;
    np.call_method1("asarray", (o,))?
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()
}

fn last_dim(o: &Bound<'_, PyAny>) -> PyResult<usize> {
    // jqmc's DTN data holds M_alpha/M_beta/M_gamma as TUPLES of arrays; np.asarray
    // stacks them into one leading-dim array (order/last-dim unchanged), matching
    // `ravel_f64` which already coerces. So coerce here too instead of `.shape` on a tuple.
    let np = o.py().import("numpy")?;
    let arr = np.call_method1("asarray", (o,))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    Ok(*shape.last().unwrap())
}

impl DtnOkData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let base = JastrowDtnData::from_py(obj)?;
        let n_order_u: usize = obj.getattr("n_order_u")?.extract()?;
        let n_order_chi: usize = obj.getattr("n_order_chi")?.extract()?;
        let (m_alpha, na) = if base.use_u {
            let ma = obj.getattr("M_alpha")?;
            (ravel_f64(&ma)?, last_dim(&ma)?)
        } else {
            (vec![], 0)
        };
        let (m_beta, nb) = if base.use_chi {
            let mb = obj.getattr("M_beta")?;
            (ravel_f64(&mb)?, last_dim(&mb)?)
        } else {
            (vec![], 0)
        };
        let (m_gamma, ng) = if base.use_f {
            let mg = obj.getattr("M_gamma")?;
            (ravel_f64(&mg)?, last_dim(&mg)?)
        } else {
            (vec![], 0)
        };
        Ok(DtnOkData {
            base,
            n_order_u,
            n_order_chi,
            m_alpha,
            na,
            m_beta,
            nb,
            m_gamma,
            ng,
        })
    }
}

#[inline]
fn cutoff(r: f64, l: f64, c: i32) -> f64 {
    if l > r {
        (l - r).powi(c)
    } else {
        0.0
    }
}

/// DTN O_k triple `(alpha_indep (2,na), beta_indep (n_sp,nb), gamma_indep (n_sp,ng))`.
pub type DtnGrads<T = f64> = (Option<Array2<T>>, Option<Array2<T>>, Option<Array2<T>>);

/// Channel feature sums before the `@ M` projection: alpha `(2, Nu+1)`, beta
/// `(n_sp, Nc+1)`, gamma `(n_sp, NeN+1, NeN+1, Nee+1)`; empty for a disabled channel.
#[derive(Clone)]
pub(crate) struct DtnFeatures {
    alpha: Vec<f64>,
    beta: Vec<f64>,
    gamma: Vec<f64>,
}

impl DtnFeatures {
    /// `self - old + new`, element-wise (one electron's terms swapped).
    fn swapped(&self, old: &Self, new: &Self) -> Self {
        let f = |a: &[f64], o: &[f64], n: &[f64]| -> Vec<f64> {
            a.iter()
                .zip(o)
                .zip(n)
                .map(|((a, o), n)| a - o + n)
                .collect()
        };
        DtnFeatures {
            alpha: f(&self.alpha, &old.alpha, &new.alpha),
            beta: f(&self.beta, &old.beta, &new.beta),
            gamma: f(&self.gamma, &old.gamma, &new.gamma),
        }
    }
}

/// Visit the electron pairs `(i, j)`, `i < j`, of `n_e` stacked electrons: all of
/// them (`only = None`, row-major order), or only the pairs containing `g`.
fn for_pairs(n_e: usize, only: Option<usize>, mut f: impl FnMut(usize, usize)) {
    match only {
        None => {
            for i in 0..n_e {
                for j in (i + 1)..n_e {
                    f(i, j);
                }
            }
        }
        Some(g) => {
            for j in (0..n_e).filter(|&j| j != g) {
                f(g.min(j), g.max(j));
            }
        }
    }
}

/// DTN feature sums over every term of the configuration (`only = None`), or over the
/// terms containing stacked electron `g` (up then dn) placed at `pos` with the other
/// electrons fixed (`only = Some((g, pos))`). The features are sums over terms, so the
/// features of a configuration with electron `g` moved are
/// `all - only(g, old) + only(g, new)`.
fn dtn_features(
    d: &DtnOkData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    only: Option<(usize, [f64; 3])>,
) -> DtnFeatures {
    let b = &d.base;
    let c = b.c;
    let ci = b.cell.as_ref().map(|cl| CellInfo::new(&cl.view()));
    let pos = b.positions.view();
    let n_atom = pos.nrows();
    let n_sp = b.beta.nrows();
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let only_g = only.map(|(g, _)| g);
    // all electrons stacked (up then dn); electron `g` sits at `pos` when restricted
    let elec = |g: usize| -> [f64; 3] {
        if let Some((og, p)) = only {
            if og == g {
                return p;
            }
        }
        if g < n_up {
            [r_up[[g, 0]], r_up[[g, 1]], r_up[[g, 2]]]
        } else {
            let k = g - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    // DTN uses the CASINO hard cutoff on the minimum-image EUCLIDEAN distance
    // (matching the DTN VALUE `dtn.rs::dtn_dist`), NOT the Williamson smooth metric.
    let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
        match &ci {
            Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci).0,
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
        }
    };
    let atom = |i: usize| -> [f64; 3] { [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]] };

    // ---- alpha (u, e-e): feat (2, Nu+1), row s = (same spin = 0, opposite = 1) ----
    let mut alpha = Vec::new();
    if b.use_u {
        let nu = d.n_order_u;
        let lu = b.l_u;
        alpha = vec![0.0_f64; 2 * (nu + 1)];
        // Stacked i < j visits the same-spin pairs of each spin set and the up x dn
        // pairs in the same per-row order as separate loops over the three sets.
        for_pairs(n_e, only_g, |i, j| {
            let row = usize::from((i < n_up) != (j < n_up));
            let r = dist(elec(i), elec(j));
            let phi = cutoff(r, lu, c);
            if phi == 0.0 {
                return;
            }
            let base_i = row * (nu + 1);
            let mut rk = 1.0_f64;
            for k in 0..=nu {
                alpha[base_i + k] += phi * rk;
                rk *= r;
            }
        });
    }

    // ---- beta (chi, e-n): feat (n_sp, Nc+1) ----
    let mut beta = Vec::new();
    if b.use_chi {
        let nc = d.n_order_chi;
        beta = vec![0.0_f64; n_sp * (nc + 1)];
        let electrons: Vec<usize> = match only_g {
            None => (0..n_e).collect(),
            Some(g) => vec![g],
        };
        for ii in 0..n_atom {
            let sp = b.species_index[ii];
            let l = b.l_chi[sp];
            let ai = atom(ii);
            for &g in &electrons {
                let r = dist(elec(g), ai);
                let phi = cutoff(r, l, c);
                if phi == 0.0 {
                    continue;
                }
                let base_i = sp * (nc + 1);
                let mut rk = 1.0_f64;
                for k in 0..=nc {
                    beta[base_i + k] += phi * rk;
                    rk *= r;
                }
            }
        }
    }

    // ---- gamma (f, e-e-n): feat (n_sp, NeN+1, NeN+1, Nee+1) ----
    let mut gamma = Vec::new();
    if b.use_f {
        let nl = b.n_en; // NeN
        let nn = b.n_ee; // Nee
        let dl = nl + 1;
        let dn = nn + 1;
        let block = dl * dl * dn; // per-species feature size
        gamma = vec![0.0_f64; n_sp * block];
        for_pairs(n_e, only_g, |i, j| {
            let ei = elec(i);
            let ej = elec(j);
            let r_ab = dist(ei, ej);
            // powers of r_ab
            let mut pab = vec![0.0_f64; dn];
            let mut t = 1.0;
            for pab_n in pab[..dn].iter_mut() {
                *pab_n = t;
                t *= r_ab;
            }
            for ai in 0..n_atom {
                let sp = b.species_index[ai];
                let lf = b.l_f[sp];
                let r_a = dist(ei, atom(ai));
                let r_b = dist(ej, atom(ai));
                let ca = cutoff(r_a, lf, c);
                let cb = cutoff(r_b, lf, c);
                let w = ca * cb;
                if w == 0.0 {
                    continue;
                }
                // powers of r_a (l), r_b (m)
                let mut pa = vec![0.0_f64; dl];
                let mut pb = vec![0.0_f64; dl];
                let (mut ta, mut tb) = (1.0, 1.0);
                for l in 0..dl {
                    pa[l] = ta;
                    pb[l] = tb;
                    ta *= r_a;
                    tb *= r_b;
                }
                let sbase = sp * block;
                for (l, pa_l) in pa[..dl].iter().enumerate() {
                    for (m, pb_m) in pb[..dl].iter().enumerate() {
                        let wlm = w * *pa_l * *pb_m;
                        let lm_base = sbase + (l * dl + m) * dn;
                        for n in 0..dn {
                            gamma[lm_base + n] += wlm * pab[n];
                        }
                    }
                }
            }
        });
    }
    DtnFeatures { alpha, beta, gamma }
}

/// Project the feature sums onto the independent coefficients (`feat @ M` per channel).
fn project_dtn_features(d: &DtnOkData, f: &DtnFeatures) -> DtnGrads {
    let b = &d.base;
    let n_sp = b.beta.nrows();
    // einsum("sk,ski->si")
    let alpha = b.use_u.then(|| {
        let nu = d.n_order_u;
        let mut out = Array2::<f64>::zeros((2, d.na));
        for s in 0..2 {
            for i in 0..d.na {
                let mut acc = 0.0_f64;
                for k in 0..=nu {
                    acc += f.alpha[s * (nu + 1) + k] * d.m_alpha[(s * (nu + 1) + k) * d.na + i];
                }
                out[[s, i]] = acc;
            }
        }
        out
    });
    let beta = b.use_chi.then(|| {
        let nc = d.n_order_chi;
        let mut out = Array2::<f64>::zeros((n_sp, d.nb));
        for s in 0..n_sp {
            for i in 0..d.nb {
                let mut acc = 0.0_f64;
                for k in 0..=nc {
                    acc += f.beta[s * (nc + 1) + k] * d.m_beta[(s * (nc + 1) + k) * d.nb + i];
                }
                out[[s, i]] = acc;
            }
        }
        out
    });
    // einsum("slmn,slmni->si")
    let gamma = b.use_f.then(|| {
        let dl = b.n_en + 1;
        let block = dl * dl * (b.n_ee + 1);
        let mut out = Array2::<f64>::zeros((n_sp, d.ng));
        for s in 0..n_sp {
            for idx in 0..block {
                let fv = f.gamma[s * block + idx];
                if fv == 0.0 {
                    continue;
                }
                let mbase = (s * block + idx) * d.ng;
                for i in 0..d.ng {
                    out[[s, i]] += fv * d.m_gamma[mbase + i];
                }
            }
        }
        out
    });
    (alpha, beta, gamma)
}

/// Returns `(alpha_indep (2,na), beta_indep (n_sp,nb), gamma_indep (n_sp,ng))`,
/// each `None` if its channel is disabled.
pub fn compute_dtn_param_grads(
    d: &DtnOkData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DtnGrads {
    project_dtn_features(d, &dtn_features(d, r_up, r_dn, None))
}

/// DTN O_k at configurations that differ from a fixed base configuration by ONE
/// electron (the ECP non-local mesh of the dE_L/dc ECP term). Each moved O_k swaps
/// only the moved electron's terms in the base feature sums -- O(N_e N_atom) instead
/// of the O(N_e^2 N_atom) full recomputation per mesh point -- and equals
/// [`compute_dtn_param_grads`] on the moved configuration up to fp reassociation.
pub(crate) struct DtnMovedOk<'a> {
    d: &'a DtnOkData,
    r_up: ArrayView2<'a, f64>,
    r_dn: ArrayView2<'a, f64>,
    base: DtnFeatures,
    /// Base-configuration terms of each stacked electron, built on first use.
    own: Vec<Option<DtnFeatures>>,
}

impl<'a> DtnMovedOk<'a> {
    pub(crate) fn new(
        d: &'a DtnOkData,
        r_up: &'a ArrayView2<'_, f64>,
        r_dn: &'a ArrayView2<'_, f64>,
    ) -> Self {
        let n_e = r_up.nrows() + r_dn.nrows();
        DtnMovedOk {
            d,
            r_up: r_up.view(),
            r_dn: r_dn.view(),
            base: dtn_features(d, r_up, r_dn, None),
            own: vec![None; n_e],
        }
    }

    /// O_k of the base configuration (= `compute_dtn_param_grads(d, r_up, r_dn)`).
    pub(crate) fn base_grads(&self) -> DtnGrads {
        project_dtn_features(self.d, &self.base)
    }

    /// O_k with electron `idx` of spin `is_up` moved to `pos`.
    pub(crate) fn moved_grads(&mut self, is_up: bool, idx: usize, pos: [f64; 3]) -> DtnGrads {
        let (d, r_up, r_dn) = (self.d, &self.r_up, &self.r_dn);
        let (g, cur) = if is_up {
            (idx, [r_up[[idx, 0]], r_up[[idx, 1]], r_up[[idx, 2]]])
        } else {
            (
                r_up.nrows() + idx,
                [r_dn[[idx, 0]], r_dn[[idx, 1]], r_dn[[idx, 2]]],
            )
        };
        let old = self.own[g].get_or_insert_with(|| dtn_features(d, r_up, r_dn, Some((g, cur))));
        let new = dtn_features(d, r_up, r_dn, Some((g, pos)));
        project_dtn_features(d, &self.base.swapped(old, &new))
    }
}

// ============================================================================
// DTN kinetic dT/dc. Reduction -1/2Sumgrad^2f_c - Sum(gJ+gD)*gradf_c.
// ============================================================================

/// `g[i]=Sum_k M[k,i]r^k`, `g'`, `g''` for one (Nk+1, n_ind) M block at scalar r.
fn poly_derivs(r: f64, nk: usize, m: &[f64], n_ind: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut g, mut gp, mut gpp) = (vec![0.0; n_ind], vec![0.0; n_ind], vec![0.0; n_ind]);
    // powers r^k, k r^{k-1}, k(k-1) r^{k-2}
    for k in 0..=nk {
        let rk = r.powi(k as i32);
        let rk1 = if k >= 1 {
            (k as f64) * r.powi(k as i32 - 1)
        } else {
            0.0
        };
        let rk2 = if k >= 2 {
            (k as f64) * ((k - 1) as f64) * r.powi(k as i32 - 2)
        } else {
            0.0
        };
        for i in 0..n_ind {
            let c = m[k * n_ind + i];
            g[i] += c * rk;
            gp[i] += c * rk1;
            gpp[i] += c * rk2;
        }
    }
    (g, gp, gpp)
}

/// `A_l=phir^l`, `A'_l`, `A''_l` for l=0..nk at scalar r (f-channel radial leg).
fn radial_phi_pow(l_cut: f64, r: f64, nk: usize, c: i32) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (phi, dphi, ddphi) = cutoff_derivs(r, l_cut, c);
    let (mut a, mut ap, mut app) = (vec![0.0; nk + 1], vec![0.0; nk + 1], vec![0.0; nk + 1]);
    for l in 0..=nk {
        let rl = r.powi(l as i32);
        let rl1 = if l >= 1 {
            (l as f64) * r.powi(l as i32 - 1)
        } else {
            0.0
        };
        let rl2 = if l >= 2 {
            (l as f64) * ((l - 1) as f64) * r.powi(l as i32 - 2)
        } else {
            0.0
        };
        a[l] = phi * rl;
        ap[l] = dphi * rl + phi * rl1;
        app[l] = ddphi * rl + 2.0 * dphi * rl1 + phi * rl2;
    }
    (a, ap, app)
}

/// `out[p] = Sum_{l,m,n} M[l,m,n,p]*X[l]*Y[m]*Z[n]`; `swap` => use `M[m,l,n,p]`.
/// `m_block` is the per-species gamma tensor flat `(dl, dl, dn, ng)`.
#[allow(
    clippy::too_many_arguments,
    reason = "operands and block shape of the gamma contraction"
)]
fn einsum_gamma(
    m_block: &[f64],
    swap: bool,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    dl: usize,
    dn: usize,
    ng: usize,
    out: &mut [f64],
) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for (l, x_l) in x[..dl].iter().enumerate() {
        for (m, y_m) in y[..dl].iter().enumerate() {
            let xy = *x_l * *y_m;
            if xy == 0.0 {
                continue;
            }
            // index into M for (l,m) or (m,l) when swapped
            let (li, mi) = if swap { (m, l) } else { (l, m) };
            for (n, z_n) in z[..dn].iter().enumerate() {
                let coef = xy * *z_n;
                if coef == 0.0 {
                    continue;
                }
                let base = ((li * dl + mi) * dn + n) * ng;
                for p in 0..ng {
                    out[p] += m_block[base + p] * coef;
                }
            }
        }
    }
}

/// `dT/d{alpha,beta,gamma}_indep` (u/chi/f channels). `(alpha (2,na), beta (n_sp,nb),
/// gamma (n_sp,ng))`, each `None` if its channel is disabled.
#[allow(clippy::too_many_arguments)]
pub fn compute_dtn_kinetic_grads(
    d: &DtnOkData,
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DtnGrads {
    // Real combined cotangent drift gJ + gD (per electron).
    let (gj_up, gj_dn) = jastrow_grads(Some(jastrow), j3_orb, j3_jm, r_up, r_dn);
    let (gd_up, gd_dn) = det_grad_field(gem, r_up, r_dn);
    let drift_up = &gj_up + &gd_up;
    let drift_dn = &gj_dn + &gd_dn;
    dtn_kinetic_reduce(d, r_up, r_dn, &drift_up, &drift_dn)
}

/// DTN kinetic `dT/dc` (u/chi/f) given the precomputed combined cotangent drift
/// `(gJ+gD)` per electron (`drift_up (n_up,3)`, `drift_dn (n_dn,3)`). The DTN field
/// math is basis/geminal-INDEPENDENT (min-image distances only) -> the complex-twist
/// path reuses this verbatim, feeding `gJ + Re(gD_complex)` as the drift.
pub(crate) fn dtn_kinetic_reduce(
    d: &DtnOkData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    drift_up: &Array2<f64>,
    drift_dn: &Array2<f64>,
) -> DtnGrads {
    let b = &d.base;
    let c = b.c;
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_sp = b.beta.nrows();
    let pos = b.positions.view();
    let n_atom = pos.nrows();
    let elec = |g: usize| -> [f64; 3] {
        if g < n_up {
            [r_up[[g, 0]], r_up[[g, 1]], r_up[[g, 2]]]
        } else {
            let k = g - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let spin = |g: usize| -> usize {
        if g < n_up {
            0
        } else {
            1
        }
    };
    let atom = |i: usize| -> [f64; 3] { [pos[[i, 0]], pos[[i, 1]], pos[[i, 2]]] };
    // DTN uses the hard (L-r)^C cutoff on the minimum-image EUCLIDEAN
    // distance (not the Williamson smooth metric); inside the cutoff the image is
    // locally constant so the grad/lap collapse to the plain Euclidean forms on the
    // min-image displacement (|grad d|^2 = 1, grad^2 d = 2/d) -- identical algebra to
    // the molecular reduction, only the displacement changes. `None` => molecular.
    let ci = b.cell.as_ref().map(|cl| CellInfo::new(&cl.view()));

    // Combined cotangent field gJ + gD (Ne, 3), supplied by the caller.
    let gdc = |e: usize| -> [f64; 3] {
        if e < n_up {
            [drift_up[[e, 0]], drift_up[[e, 1]], drift_up[[e, 2]]]
        } else {
            let l = e - n_up;
            [drift_dn[[l, 0]], drift_dn[[l, 1]], drift_dn[[l, 2]]]
        }
    };
    let dist_uvec = |a: [f64; 3], b: [f64; 3]| -> (f64, [f64; 3]) {
        let x = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        match &ci {
            Some(ci) => {
                let (r, u, _, _) = min_image_dist(x, ci);
                (r, u)
            }
            None => {
                let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                (r, [x[0] / r, x[1] / r, x[2] / r])
            }
        }
    };

    // ---- alpha (u, e-e) ----
    let alpha = if b.use_u {
        let nu = d.n_order_u;
        let na = d.na;
        let mut out = Array2::<f64>::zeros((2, na));
        for e in 0..n_e {
            let re = elec(e);
            for s in 0..2 {
                let target = if s == 0 { spin(e) } else { 1 - spin(e) };
                let mut grad_f = vec![[0.0_f64; 3]; na];
                let mut lap_f = vec![0.0_f64; na];
                let mblock = &d.m_alpha[s * (nu + 1) * na..(s + 1) * (nu + 1) * na];
                for ep in 0..n_e {
                    if ep == e || spin(ep) != target {
                        continue;
                    }
                    let (r, u) = dist_uvec(re, elec(ep));
                    let (phi, dphi, ddphi) = cutoff_derivs(r, b.l_u, c);
                    if phi == 0.0 && dphi == 0.0 && ddphi == 0.0 {
                        continue;
                    }
                    let (g, gp, gpp) = poly_derivs(r, nu, mblock, na);
                    for i in 0..na {
                        let hp = dphi * g[i] + phi * gp[i];
                        let hpp = ddphi * g[i] + 2.0 * dphi * gp[i] + phi * gpp[i];
                        grad_f[i][0] += hp * u[0];
                        grad_f[i][1] += hp * u[1];
                        grad_f[i][2] += hp * u[2];
                        lap_f[i] += hpp + 2.0 * hp / r;
                    }
                }
                let gde = gdc(e);
                for i in 0..na {
                    out[[s, i]] += -0.5 * lap_f[i]
                        - (grad_f[i][0] * gde[0] + grad_f[i][1] * gde[1] + grad_f[i][2] * gde[2]);
                }
            }
        }
        Some(out)
    } else {
        None
    };

    // ---- beta (chi, e-n) ----
    let beta = if b.use_chi {
        let nc = d.n_order_chi;
        let nb = d.nb;
        let mut out = Array2::<f64>::zeros((n_sp, nb));
        for e in 0..n_e {
            let re = elec(e);
            let gde = gdc(e);
            for ii in 0..n_atom {
                let sp = b.species_index[ii];
                let (r, u) = dist_uvec(re, atom(ii));
                let (phi, dphi, ddphi) = cutoff_derivs(r, b.l_chi[sp], c);
                if phi == 0.0 && dphi == 0.0 && ddphi == 0.0 {
                    continue;
                }
                let mblock = &d.m_beta[sp * (nc + 1) * nb..(sp + 1) * (nc + 1) * nb];
                let (g, gp, gpp) = poly_derivs(r, nc, mblock, nb);
                for i in 0..nb {
                    let hp = dphi * g[i] + phi * gp[i];
                    let hpp = ddphi * g[i] + 2.0 * dphi * gp[i] + phi * gpp[i];
                    let gf = [hp * u[0], hp * u[1], hp * u[2]];
                    let lap = hpp + 2.0 * hp / r;
                    out[[sp, i]] += -0.5 * lap - (gf[0] * gde[0] + gf[1] * gde[1] + gf[2] * gde[2]);
                }
            }
        }
        Some(out)
    } else {
        None
    };

    // ---- gamma (f, e-e-n) ----
    let gamma = if b.use_f {
        let nl = b.n_en; // NeN
        let nn = b.n_ee; // Nee
        let dl = nl + 1;
        let dn = nn + 1;
        let ng = d.ng;
        let block = dl * dl * dn * ng;
        let mut out = Array2::<f64>::zeros((n_sp, ng));
        let mut tmp = vec![0.0_f64; ng];
        for e in 0..n_e {
            let re = elec(e);
            let gde = gdc(e);
            for ep in 0..n_e {
                if ep == e {
                    continue;
                }
                let rep = elec(ep);
                let (r_ee, u_ee) = dist_uvec(re, rep);
                // power factors of r_ee: C_n, C'_n, C''_n
                let (mut cn, mut cp, mut cpp) = (vec![0.0; dn], vec![0.0; dn], vec![0.0; dn]);
                for n in 0..dn {
                    cn[n] = r_ee.powi(n as i32);
                    cp[n] = if n >= 1 {
                        (n as f64) * r_ee.powi(n as i32 - 1)
                    } else {
                        0.0
                    };
                    cpp[n] = if n >= 2 {
                        (n as f64) * ((n - 1) as f64) * r_ee.powi(n as i32 - 2)
                    } else {
                        0.0
                    };
                }
                for ii in 0..n_atom {
                    let sp = b.species_index[ii];
                    let (r_si, u_si) = dist_uvec(re, atom(ii));
                    let (r_oi, _) = dist_uvec(rep, atom(ii));
                    let (a, ap, app) = radial_phi_pow(b.l_f[sp], r_si, nl, c);
                    let (bb, _, _) = radial_phi_pow(b.l_f[sp], r_oi, nl, c);
                    let swap = e >= ep;
                    let mblock = &d.m_gamma[sp * block..(sp + 1) * block];
                    let cos_i = u_si[0] * u_ee[0] + u_si[1] * u_ee[1] + u_si[2] * u_ee[2];
                    // coefI = einsum(M, A', B, Cn)
                    einsum_gamma(mblock, swap, &ap, &bb, &cn, dl, dn, ng, &mut tmp);
                    let coef_i: Vec<f64> = tmp.clone();
                    // coefEE = einsum(M, A, B, C')   (grad_f_dot along u_ee)
                    einsum_gamma(mblock, swap, &a, &bb, &cp, dl, dn, ng, &mut tmp);
                    let coef_ee: Vec<f64> = tmp.clone();
                    // lap = einsum(M, A''+2A'/r_si, B, Cn) + 2cos*einsum(M, A', B, C')
                    //       + einsum(M, A, B, C''+2C'/r_ee)
                    let a_lap: Vec<f64> = (0..dl).map(|l| app[l] + 2.0 * ap[l] / r_si).collect();
                    einsum_gamma(mblock, swap, &a_lap, &bb, &cn, dl, dn, ng, &mut tmp);
                    let mut lap: Vec<f64> = tmp.clone();
                    // cross term: 2cos*einsum(M, A', B, C')  (note A', not A)
                    einsum_gamma(mblock, swap, &ap, &bb, &cp, dl, dn, ng, &mut tmp);
                    for p in 0..ng {
                        lap[p] += 2.0 * cos_i * tmp[p];
                    }
                    let c_lap: Vec<f64> = (0..dn).map(|n| cpp[n] + 2.0 * cp[n] / r_ee).collect();
                    einsum_gamma(mblock, swap, &a, &bb, &c_lap, dl, dn, ng, &mut tmp);
                    for p in 0..ng {
                        lap[p] += tmp[p];
                    }
                    let dot_si = gde[0] * u_si[0] + gde[1] * u_si[1] + gde[2] * u_si[2];
                    let dot_ee = gde[0] * u_ee[0] + gde[1] * u_ee[1] + gde[2] * u_ee[2];
                    for p in 0..ng {
                        let grad_f_dot = coef_i[p] * dot_si + coef_ee[p] * dot_ee;
                        out[[sp, p]] += -0.5 * lap[p] - grad_f_dot;
                    }
                }
            }
        }
        Some(out)
    } else {
        None
    };

    (alpha, beta, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setting_consts::{ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER};
    use numpy::ndarray::{array, Array2};

    /// Synthetic two-species DTN with every channel on and deterministic inflation
    /// matrices (`na = 3`, `nb = 3`, `ng = 4`); `cell = Some` makes it periodic.
    fn ok_data(cell: Option<Array2<f64>>) -> DtnOkData {
        let (n_sp, n_en, n_ee, n_u, n_chi) = (2usize, 2usize, 2usize, 3usize, 3usize);
        let gamma_len = n_sp * (n_en + 1) * (n_en + 1) * (n_ee + 1);
        let fill = |n: usize, s: f64| -> Vec<f64> {
            (0..n)
                .map(|i| (((i * 37 + 11) % 23) as f64 - 11.0) * s)
                .collect()
        };
        let (na, nb, ng) = (3usize, 3usize, 4usize);
        DtnOkData {
            base: JastrowDtnData {
                use_u: true,
                use_chi: true,
                use_f: true,
                c: 3,
                n_en,
                n_ee,
                l_u: 4.0,
                l_chi: vec![3.4, 3.8],
                l_f: vec![3.1, 3.6],
                alpha: Array2::zeros((2, n_u + 1)),
                beta: Array2::zeros((n_sp, n_chi + 1)),
                gamma: vec![0.0; gamma_len],
                alpha_policy: Array2::zeros((2, n_u + 1)),
                beta_policy: Array2::zeros((n_sp, n_chi + 1)),
                gamma_policy: vec![0.0; gamma_len],
                species_index: vec![0, 1],
                positions: array![[0.1, -0.2, 0.3], [1.4, 0.7, -0.5]],
                cell,
                precision_policy: crate::PrecisionPolicy::full(),
                zone_coefficients: None,
                coefficients_f32: false,
            },
            n_order_u: n_u,
            n_order_chi: n_chi,
            m_alpha: fill(2 * (n_u + 1) * na, 0.013),
            na,
            m_beta: fill(n_sp * (n_chi + 1) * nb, 0.017),
            nb,
            m_gamma: fill(gamma_len * ng, 0.011),
            ng,
        }
    }

    fn assert_triple_close(got: &DtnGrads, want: &DtnGrads) -> f64 {
        let mut max_abs = 0.0_f64;
        for (g, w) in [(&got.0, &want.0), (&got.1, &want.1), (&got.2, &want.2)] {
            let (g, w) = (
                g.as_ref().expect("channel on"),
                w.as_ref().expect("channel on"),
            );
            assert_eq!(g.dim(), w.dim());
            for (a, b) in g.iter().zip(w.iter()) {
                assert!(
                    (a - b).abs() <= ATOL_REDUCTION_ORDER + RTOL_REDUCTION_ORDER * b.abs(),
                    "moved O_k {a:.17e} != full recompute {b:.17e}"
                );
                max_abs = max_abs.max(b.abs());
            }
        }
        max_abs
    }

    #[test]
    fn moved_ok_matches_full_recompute() {
        // The ECP mesh route: O_k with one electron moved, from the base feature sums with
        // that electron's terms swapped, equals compute_dtn_param_grads on the moved
        // configuration (every electron of both spins, two moves each, the base O_k too),
        // molecular and periodic (minimum image across the cell).
        let r_up = array![[0.3, 0.1, 0.2], [1.1, 0.4, -0.3], [-0.6, 0.5, 0.9]];
        let r_dn = array![[0.9, -0.4, 0.1], [0.2, 1.3, -0.7]];
        let moves = [[0.35, -0.25, 0.4], [1.9, 2.4, -1.6]];
        for cell in [
            None,
            Some(array![[4.5, 0.0, 0.0], [0.8, 4.2, 0.0], [0.0, 0.3, 4.8]]),
        ] {
            let d = ok_data(cell);
            let (ru, rd) = (r_up.view(), r_dn.view());
            let mut ok = DtnMovedOk::new(&d, &ru, &rd);
            assert!(
                assert_triple_close(&ok.base_grads(), &compute_dtn_param_grads(&d, &ru, &rd)) > 0.0
            );
            for (is_up, n) in [(true, r_up.nrows()), (false, r_dn.nrows())] {
                for idx in 0..n {
                    for pos in moves {
                        let mut moved = if is_up { r_up.clone() } else { r_dn.clone() };
                        moved.row_mut(idx).assign(&array![pos[0], pos[1], pos[2]]);
                        let want = if is_up {
                            compute_dtn_param_grads(&d, &moved.view(), &rd)
                        } else {
                            compute_dtn_param_grads(&d, &ru, &moved.view())
                        };
                        let scale = assert_triple_close(&ok.moved_grads(is_up, idx, pos), &want);
                        assert!(scale > 0.0, "vacuous case: O_k identically zero");
                    }
                }
            }
        }
    }
}
