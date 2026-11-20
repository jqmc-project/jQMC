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

//! PBC kinetic nuclear force `dT_det/dR` without Jastrow. The determinant-VJP cotangents
//! (`bar_orb`/`bar_lap`) are geometry-agnostic (reuse `geminal_reverse` on the
//! periodic geminal -- `compute_aos*` fold the image axis). Only the final
//! translation-identity step is periodic: pgto AOs use the folded `pgto_grad` /
//! `pgto_gradlap`, scattered by the molecular-rank home atom. PW children give
//! zero force (not atom-centered). REAL Gamma-point (Rust all-f64).

use numpy::ndarray::{Array2, ArrayView2};

use crate::ao_cart_deriv::pgto_derivs;
use crate::atomic_orbital::{compute_pw, compute_pw_value_grad_lap, AosGtoData, AosPwData};
use crate::bspline::AosBsplineData;
use crate::data::JastrowOneBodyData;

/// Composite basis `(value, grad [3], laplacian, Hessian [3][3], grad-laplacian [3])`.
type BasisTables = (
    Array2<f64>,
    [Array2<f64>; 3],
    Array2<f64>,
    [[Array2<f64>; 3]; 3],
    [Array2<f64>; 3],
);

/// The non-atom-centred child of a REAL composite J3 orbital: the plane-wave child of a
/// pGTO(+)PW composite or the B-spline child of a pGTO(+)B-spline composite (the two
/// composites the QE converter writes at the Gamma point). The periodic cart J3 kernels
/// stack its rows next to the pGTO block in `gto_first` order: they enter every J3
/// cotangent through the (AO-rank, lowered) `j_matrix`, give zero nuclear force rows
/// (dAO/dR = 0) and are strain-invariant in value (fractional coordinates are fixed under
/// the affine strain), so only their gradient / Laplacian fields respond to a strain.
#[derive(Clone, Copy)]
pub enum J3Extra<'a> {
    Pw(&'a AosPwData),
    Bspline(&'a AosBsplineData),
}

/// Owned [`J3Extra`] (the per-context storage); [`J3ExtraOwned::view`] borrows it.
#[derive(Clone)]
pub enum J3ExtraOwned {
    Pw(AosPwData),
    Bspline(AosBsplineData),
}

impl J3ExtraOwned {
    pub fn view(&self) -> J3Extra<'_> {
        match self {
            J3ExtraOwned::Pw(p) => J3Extra::Pw(p),
            J3ExtraOwned::Bspline(b) => J3Extra::Bspline(b),
        }
    }
}

impl J3Extra<'_> {
    /// Number of AO rows of the child.
    pub fn num_ao(&self) -> usize {
        match self {
            J3Extra::Pw(p) => p.num_ao,
            J3Extra::Bspline(b) => b.num_ao,
        }
    }

    /// AO values `(num_ao, N)`.
    pub fn value(&self, r: &ArrayView2<f64>) -> Array2<f64> {
        match self {
            J3Extra::Pw(p) => compute_pw(p, r),
            J3Extra::Bspline(b) => b.value_grad_lap(r).0,
        }
    }

    /// `(value, grad_x, grad_y, grad_z, laplacian)`, each `(num_ao, N)`.
    #[allow(clippy::type_complexity)]
    pub fn vgl(
        &self,
        r: &ArrayView2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
    ) {
        match self {
            J3Extra::Pw(p) => compute_pw_value_grad_lap(p, r),
            J3Extra::Bspline(b) => b.value_grad_lap(r),
        }
    }

    /// Electron-axis second/third derivative fields `(H[c][d] = d2AO/dr_c dr_d,
    /// GL[c] = d(lap AO)/dr_c)`, each `(num_ao, N)`. `val`/`grad` are this child's
    /// [`J3Extra::vgl`] value and gradient at the same `r` (the plane wave reuses them:
    /// `H = -kappa_c kappa_d val`, `GL = -|kappa|^2 grad`).
    #[allow(clippy::type_complexity)]
    pub fn hess_gradlap(
        &self,
        r: &ArrayView2<f64>,
        val: &Array2<f64>,
        grad: &[Array2<f64>; 3],
    ) -> ([[Array2<f64>; 3]; 3], [Array2<f64>; 3]) {
        match self {
            J3Extra::Pw(pw) => {
                let mut ph: [[Array2<f64>; 3]; 3] = std::array::from_fn(|_| {
                    std::array::from_fn(|_| Array2::<f64>::zeros(val.raw_dim()))
                });
                let mut pgl: [Array2<f64>; 3] =
                    std::array::from_fn(|_| Array2::<f64>::zeros(val.raw_dim()));
                for aidx in 0..pw.num_ao {
                    let kap = [pw.gx[aidx], pw.gy[aidx], pw.gz[aidx]];
                    for e in 0..val.ncols() {
                        for c in 0..3 {
                            pgl[c][[aidx, e]] = -pw.g2[aidx] * grad[c][[aidx, e]];
                            for d in 0..3 {
                                ph[c][d][[aidx, e]] = -kap[c] * kap[d] * val[[aidx, e]];
                            }
                        }
                    }
                }
                (ph, pgl)
            }
            J3Extra::Bspline(bs) => {
                let (_ga, gl, h) = bs.det_egrad_fields(r);
                (h, gl)
            }
        }
    }
}

/// Force contribution of one cart-pgto leaf: `-(CM_orb*gradAO + CM_lap*gradgrad^2AO)`
/// scattered by `home_atom`. `c_cart` maps orbital->cart-AO (`orb = C_cart*AO`);
/// `bar_orb`/`bar_lap` are orbital-rank `(m, N)` cotangents.
#[allow(clippy::too_many_arguments)]
pub fn pgto_det_force_side(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    n_atom: usize,
    out: &mut Array2<f64>,
) {
    let _ = n_atom;
    pgto_det_force_egrad(
        cart_aos,
        c_cart,
        bar_orb,
        Some(bar_lap),
        bar_gou,
        r,
        cell,
        home_atom,
        Some(out),
        None,
        crate::ao_cart_deriv::DerivScreens::EXACT,
    );
}

/// Electron-axis periodic det-kinetic gradient `dT_det/dr_e` of one cart-pgto leaf,
/// accumulated per electron `(N,3)` (NOT scattered to atoms). Mirror of
/// `pgto_det_force_side` with `dphi/dr_e = +gradphi` (vs `dphi/dR = -gradphi`). `bar_gou` adds the
/// with-Jastrow Hessian cross term. Accumulates into `out` `(N,3)`.
#[allow(clippy::too_many_arguments)]
pub fn pgto_det_egrad_side(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    pgto_det_force_egrad(
        cart_aos,
        c_cart,
        bar_orb,
        Some(bar_lap),
        bar_gou,
        r,
        cell,
        &[],
        None,
        Some(out),
        crate::ao_cart_deriv::DerivScreens::EXACT,
    );
}

/// [`pgto_det_force_side`] (`out_r`, nuclear axis) and/or [`pgto_det_egrad_side`]
/// (`out_e`, electron axis) of one cart-pgto leaf from ONE fused periodic derivative
/// sweep ([`pgto_derivs`]) and one set of `C^T bar` projections, instead of a separate
/// gradient / grad-of-Laplacian / Hessian sweep per axis. `bar_lap = None` is a zero
/// Laplacian cotangent (the Pulay / drift contractions), which skips the third-order
/// tables. `home_atom` is only read when `out_r` is given.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pgto_det_force_egrad(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: Option<&Array2<f64>>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    out_r: Option<&mut Array2<f64>>,
    out_e: Option<&mut Array2<f64>>,
    scr: crate::ao_cart_deriv::DerivScreens,
) {
    // CM = C_cartT * bar  (num_cart_ao, N)
    let cm_orb = c_cart.t().dot(bar_orb);
    let cm_lap = bar_lap.map(|b| c_cart.t().dot(b));
    let cm_gou = bar_gou.map(|bg| {
        [
            c_cart.t().dot(&bg[0]),
            c_cart.t().dot(&bg[1]),
            c_cart.t().dot(&bg[2]),
        ]
    });
    let num_ao = cm_orb.nrows();
    let n = cm_orb.ncols();
    // Fused contraction (no derivative tables) when the leaf is covered.
    // The shell geometry is only needed for the atom-resolved (nuclear) output.
    let geom = if out_r.is_some() {
        crate::mesh_vgs::ShellGeom::new(cart_aos, home_atom, None)
    } else {
        None
    };
    if std::env::var("JQMC_RS_MESH_TABLES").is_err()
        && crate::mesh_vgs::supported(cart_aos)
        && (out_r.is_none() || geom.is_some())
    {
        {
            let r_w = crate::mcmc::wrap_rows(r, cell, &crate::ewald::inv3(cell));
            let std_c = |a: &Array2<f64>| a.as_standard_layout().to_owned();
            let co = std_c(&cm_orb);
            let cl = cm_lap.as_ref().map(std_c);
            let cg = cm_gou
                .as_ref()
                .map(|g| [std_c(&g[0]), std_c(&g[1]), std_c(&g[2])]);
            let pts: Vec<[f64; 3]> = (0..n)
                .map(|k| [r_w[[k, 0]], r_w[[k, 1]], r_w[[k, 2]]])
                .collect();
            let w = crate::mesh_vgs::KinWeights {
                orb: co.as_slice().unwrap(),
                lap: cl.as_ref().map(|c| c.as_slice().unwrap()),
                gou: cg.as_ref().map(|g| {
                    [
                        g[0].as_slice().unwrap(),
                        g[1].as_slice().unwrap(),
                        g[2].as_slice().unwrap(),
                    ]
                }),
                kscale: None,
            };
            let n_atom = out_r.as_ref().map_or(0, |o| o.nrows());
            let mut o = crate::mesh_vgs::KinOut {
                elec: vec![[0.0; 3]; n],
                atom: vec![[0.0; 3]; n_atom],
                strain: None,
            };
            crate::mesh_vgs::kin_contract(
                cart_aos,
                &pts,
                &w,
                scr.value,
                scr.high,
                geom.as_ref(),
                &mut o,
            );
            if let Some(out) = out_r {
                for a in 0..n_atom {
                    for d in 0..3 {
                        out[[a, d]] -= o.atom[a][d];
                    }
                }
            }
            if let Some(out) = out_e {
                for k in 0..n {
                    for d in 0..3 {
                        out[[k, d]] += o.elec[k][d];
                    }
                }
            }
            return;
        }
    }
    let dv = crate::ao_cart_deriv::pgto_derivs_screened(
        cart_aos,
        r,
        cell,
        false,
        cm_gou.is_some(),
        cm_lap.is_some(),
        scr,
    );
    let ga = &dv.grad;
    // per-(AO, electron, axis) field: CM_orb*gA + CM_lap*gL (+ CM_gou . H)
    let field = |c: usize, k: usize, d: usize| -> f64 {
        let mut v = cm_orb[[c, k]] * ga[d][[c, k]];
        if let (Some(cl), Some(gl)) = (cm_lap.as_ref(), dv.gradlap.as_ref()) {
            v += cl[[c, k]] * gl[d][[c, k]];
        }
        v
    };
    if let Some(out) = out_r {
        // s[d,c] = -(Sum_k CM_orb*gA + CM_lap*gL)
        for (c, home_atom_c) in home_atom[..num_ao].iter().enumerate() {
            let ia = *home_atom_c;
            for d in 0..3 {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += field(c, k, d);
                }
                out[[ia, d]] -= acc;
            }
        }
        // with-Jastrow cross term: bar_gou * Hessian
        if let (Some(cg), Some(h)) = (cm_gou.as_ref(), dv.hess.as_ref()) {
            for c in 0..num_ao {
                let ia = home_atom[c];
                for d in 0..3 {
                    let mut acc = 0.0;
                    for e in 0..3 {
                        for k in 0..n {
                            acc += cg[e][[c, k]] * h[d][e][[c, k]];
                        }
                    }
                    out[[ia, d]] -= acc;
                }
            }
        }
    }
    if let Some(out) = out_e {
        for k in 0..n {
            for d in 0..3 {
                let mut acc = 0.0;
                for c in 0..num_ao {
                    acc += field(c, k, d);
                }
                out[[k, d]] += acc;
            }
        }
        if let (Some(cg), Some(h)) = (cm_gou.as_ref(), dv.hess.as_ref()) {
            for k in 0..n {
                for d in 0..3 {
                    let mut acc = 0.0;
                    for c in 0..num_ao {
                        for e in 0..3 {
                            acc += cg[e][[c, k]] * h[d][e][[c, k]];
                        }
                    }
                    out[[k, d]] += acc;
                }
            }
        }
    }
}

/// Electron-axis periodic det-kinetic gradient `dT_det/dr_e (N,3)` of a B-spline geminal
/// leaf, accumulated per electron. The B-spline analog of `pgto_det_egrad_side`: the fixed
/// fractional grid gives dAO/dR = 0 (no nuclear force), but the electron move drives the AO
/// grad/lap, so the electron-axis kinetic force is nonzero. `mo` = `(num_mo, num_ao)` MO map
/// (`None` => raw AO); `bar_gou` adds the with-Jastrow Hessian cross term.
pub fn bspline_det_egrad_side(
    bs: &crate::bspline::AosBsplineData,
    mo: Option<&Array2<f64>>,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    let (ga, gl, h) = bs.det_egrad_fields(r);
    // Project MO-rank cotangents to AO rank: cm = C^T bar (or identity for raw AO).
    let proj = |bar: &Array2<f64>| -> Array2<f64> {
        match mo {
            Some(c) => c.t().dot(bar),
            None => bar.clone(),
        }
    };
    let cm_orb = proj(bar_orb);
    let cm_lap = proj(bar_lap);
    let num_ao = cm_orb.nrows();
    let n = cm_orb.ncols();
    for k in 0..n {
        for d in 0..3 {
            let mut acc = 0.0;
            for c in 0..num_ao {
                acc += cm_orb[[c, k]] * ga[d][[c, k]] + cm_lap[[c, k]] * gl[d][[c, k]];
            }
            out[[k, d]] += acc;
        }
    }
    if let Some(bg) = bar_gou {
        let cm_gou = [proj(&bg[0]), proj(&bg[1]), proj(&bg[2])];
        for k in 0..n {
            for d in 0..3 {
                let mut acc = 0.0;
                for c in 0..num_ao {
                    for e in 0..3 {
                        acc += cm_gou[e][[c, k]] * h[d][e][[c, k]];
                    }
                }
                out[[k, d]] += acc;
            }
        }
    }
}

/// Positive full-Gram periodic distance and its closed coordinate derivatives.
pub(crate) fn smooth_dist_derivs(
    disp: [f64; 3],
    ci: &crate::jastrow_factor::CellInfo,
) -> (f64, [f64; 3], [[f64; 3]; 3], f64, [f64; 3]) {
    let raw = crate::jastrow_factor::positive_geometry(disp, ci);
    // Preserve the force route's unclamped denominator convention.
    let d = raw.square.sqrt();
    let (gd, _q, lap_d, hd, glap_d) = crate::jastrow_factor::positive_hessian(&raw, ci, d);
    (d, gd, hd, lap_d, glap_d)
}

/// EUCLIDEAN distance derivative fields `(d, gradd, Hess_d, grad^2d, grad(grad^2d))`
/// for a displacement -- the molecular-metric sibling of [`smooth_dist_derivs`].
/// `gradd = u`, `Hess = (I - u(x)u)/d`, `grad^2d = 2/d`, `grad(grad^2d) = -2u/d^2`.
///
/// Needed because jqmc keys EACH Jastrow term's metric on ITS OWN
/// ``structure_data.pbc_flag`` (NOT the geminal's periodicity): a cell-less J1/J2
/// inside a periodic determinant is evaluated with the plain Euclidean distance
/// (``compute_Jastrow_{one,two}_body`` molecular branch), and its kinetic force must
/// differentiate the SAME function. The native value/drift kernels already key on the
/// data's own cell (``j2_pair``/``j1_grad_lap_one_spin``); the force kinetic used the
/// geminal cell unconditionally -- measured 3.5e-2 electron-axis kinetic error on a
/// 2-electron periodic cell whose J2 carried no structure_data (H2_pbc dispatch ham).
pub(crate) fn euclid_dist_derivs(disp: [f64; 3]) -> (f64, [f64; 3], [[f64; 3]; 3], f64, [f64; 3]) {
    let d = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
    let u = [disp[0] / d, disp[1] / d, disp[2] / d];
    let mut hd = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            hd[i][j] = (((i == j) as u8 as f64) - u[i] * u[j]) / d;
        }
    }
    let lap_d = 2.0 / d;
    let glap_d = [
        -2.0 * u[0] / (d * d),
        -2.0 * u[1] / (d * d),
        -2.0 * u[2] / (d * d),
    ];
    (d, u, hd, lap_d, glap_d)
}

/// Distance derivative fields under a Jastrow term's OWN metric: `Some(context)`
/// => positive full-Gram periodic; `None` => Euclidean (see [`euclid_dist_derivs`]).
pub(crate) fn jastrow_dist_derivs(
    disp: [f64; 3],
    cm: Option<&crate::jastrow_factor::CellInfo>,
) -> (f64, [f64; 3], [[f64; 3]; 3], f64, [f64; 3]) {
    match cm {
        Some(ci) => smooth_dist_derivs(disp, ci),
        None => euclid_dist_derivs(disp),
    }
}

/// PBC J1 Jastrow-side kinetic force.
pub fn j1_pbc_side(
    j1: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bar_gj: &Array2<f64>, // (Ne,3) = -(gJ+gD)
    out: &mut Array2<f64>,
) {
    // Metric = the J1 data's OWN convention (Williamson iff it carries a cell; a
    // cell-less J1 inside a periodic determinant is Euclidean -- jqmc keys the J1
    // metric on j1.structure_data.pbc_flag, never on the geminal's periodicity).
    let cm = j1
        .structure_data
        .cell
        .as_ref()
        .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
    let r_atoms = j1.structure_data.positions.view();
    let n_atom = j1.structure_data.n_atoms();
    let p = j1.jastrow_1b_param;
    let is_exp = j1.jastrow_1b_type == "exp";
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let bar_lap_j = -0.5;
    for e in 0..n_e {
        let re = elec(e);
        let bg = [bar_gj[[e, 0]], bar_gj[[e, 1]], bar_gj[[e, 2]]];
        for ia in 0..n_atom {
            let z = j1.structure_data.atomic_numbers[ia] - j1.core_electrons[ia];
            let c = (2.0 * z).powf(0.25);
            let pref = if is_exp {
                -(2.0 * z).powf(0.75)
            } else {
                -2.0 * z
            };
            let disp = [
                re[0] - r_atoms[[ia, 0]],
                re[1] - r_atoms[[ia, 1]],
                re[2] - r_atoms[[ia, 2]],
            ];
            let (d, gd, hd, lap_d, glap_d) = jastrow_dist_derivs(disp, cm.as_ref());
            let (fp, fpp, fppp) = if is_exp {
                let a = p * c;
                let ee = (-a * d).exp();
                (
                    pref * c / 2.0 * ee,
                    -pref * (p * c * c) / 2.0 * ee,
                    pref * (p * p * c * c * c) / 2.0 * ee,
                )
            } else {
                let b = p * c;
                let u = 1.0 + b * d;
                (
                    pref / (2.0 * u * u),
                    -pref * b / u.powi(3),
                    3.0 * pref * b * b / u.powi(4),
                )
            };
            let q = gd[0] * gd[0] + gd[1] * gd[1] + gd[2] * gd[2];
            // M[i][j] = -(f''*gd_i gd_j + f'*Hd_ij) ; dlap_i
            let mut hd_gd = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    hd_gd[i] += hd[i][j] * gd[j];
                }
            }
            for i in 0..3 {
                // bar_gJ * M (column i)
                let mut bm = 0.0;
                for jj in 0..3 {
                    let m_ji = -(fpp * gd[jj] * gd[i] + fp * hd[jj][i]);
                    bm += bg[jj] * m_ji;
                }
                let dlap = -(fppp * q * gd[i]
                    + fpp * 2.0 * hd_gd[i]
                    + fpp * lap_d * gd[i]
                    + fp * glap_d[i]);
                out[[ia, i]] += bm + bar_lap_j * dlap;
            }
        }
    }
}

use numpy::ndarray::Axis;

pub(crate) fn k_strict_lower(n: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            m[[i, j]] = 1.0;
        }
    }
    m
}

/// Vertically stack the GTO block `gto` over the PW block `pw` in `gto_first` order
/// (both `(n, n_elec)`) -> the concatenated composite AO field.
fn stack_rows(gto: &Array2<f64>, pw: &Array2<f64>, gto_first: bool) -> Array2<f64> {
    let (top, bot) = if gto_first { (gto, pw) } else { (pw, gto) };
    numpy::ndarray::concatenate(Axis(0), &[top.view(), bot.view()]).unwrap()
}

/// PBC J3 Jastrow-side kinetic force.
/// J3 basis = bare pgto leaf (sphe->cart via `t_mol`, `sphe=T*cart`), then the
/// molecular dJ3/dA cotangent machinery + translation id + home-atom scatter.
/// `bar_lapJ = -0.5` (const); `bg_*` are `-(gJ+gD)` per spin `(N,3)`.
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_side(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    out: &mut Array2<f64>,
) {
    j3_pbc_side_w(
        cart_j3, t_mol, j_matrix, r_up, r_dn, bg_up, bg_dn, cell, home_atom, None, None, true, out,
    )
}

/// `j3_pbc_side` with a per-electron kinetic weight `w` (global index; S' mask).
/// `None` => all-1 (the full dT_J3/dR).
///
/// COMPOSITE (pgto(+)pw) J3 orbital: `pw = Some` appends the plane-wave child's
/// value/grad/laplacian to the AO basis in `gto_first` order (the coefA/side/bar_a
/// cotangent machinery couples every basis row through the symmetric j_matrix, so the
/// PW rows DO enter the pGTO rows' cotangent). But the PW child is NOT atom-centered
/// (dAO/dR = 0), so it contributes ZERO to the nuclear force -> only the pGTO rows are
/// scattered (translation identity), exactly as `pgto_force_leaves` drops the PW child
/// for the determinant side. `pw = None` => byte-identical single-leaf pgto J3. The same
/// holds for the B-spline child of a pGTO(+)B-spline composite (`pw = Some(J3Extra::Bspline)`,
/// a fixed cell-fractional grid: dAO/dR = 0); every J3 kernel below takes either child.
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_side_w(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    w: Option<&[f64]>,
    pw: Option<J3Extra<'_>>,
    gto_first: bool,
    out: &mut Array2<f64>,
) {
    // GTO-block sphe rank; PW child count; global offset of the gto block.
    let num_gto = t_mol.nrows();
    let num_pw = pw.map_or(0, |p| p.num_ao());
    let gto0 = if gto_first { 0 } else { num_pw };
    // Full composite basis value/grad/lap (orb = num_gto+num_pw rows, gto_first order);
    // Hessian/gradlap kept GTO-ONLY (num_gto rows) -- only the pGTO rows are scattered.
    let basis = |r: &ArrayView2<f64>| -> BasisTables {
        let dv = pgto_derivs(cart_j3, r, cell, true, true, true);
        let ag = t_mol.dot(dv.val.as_ref().unwrap());
        let g = &dv.grad;
        let gog = [t_mol.dot(&g[0]), t_mol.dot(&g[1]), t_mol.dot(&g[2])];
        let log = t_mol.dot(dv.lap.as_ref().unwrap());
        let hh = dv.hess.as_ref().unwrap();
        let h = [
            [
                t_mol.dot(&hh[0][0]),
                t_mol.dot(&hh[0][1]),
                t_mol.dot(&hh[0][2]),
            ],
            [
                t_mol.dot(&hh[1][0]),
                t_mol.dot(&hh[1][1]),
                t_mol.dot(&hh[1][2]),
            ],
            [
                t_mol.dot(&hh[2][0]),
                t_mol.dot(&hh[2][1]),
                t_mol.dot(&hh[2][2]),
            ],
        ];
        let gll = dv.gradlap.as_ref().unwrap();
        let gl = [t_mol.dot(&gll[0]), t_mol.dot(&gll[1]), t_mol.dot(&gll[2])];
        match pw {
            None => (ag, gog, log, h, gl),
            Some(pw) => {
                let (pv, pgx, pgy, pgz, pl) = pw.vgl(r);
                let a = stack_rows(&ag, &pv, gto_first);
                let go = [
                    stack_rows(&gog[0], &pgx, gto_first),
                    stack_rows(&gog[1], &pgy, gto_first),
                    stack_rows(&gog[2], &pgz, gto_first),
                ];
                let lo = stack_rows(&log, &pl, gto_first);
                (a, go, lo, h, gl)
            }
        }
    };
    let (a_up, go_u, lo_u, hu, glu) = basis(r_up);
    let (a_dn, go_d, lo_d, hd, gld) = basis(r_dn);
    let orb = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb]).to_owned();
    let j1col = j_matrix.column(orb).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let coef =
        |a: &Array2<f64>, k: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
            let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
            let nc = a.ncols();
            for p in 0..orb {
                for c in 0..nc {
                    g[[p, c]] += j1col[p] + cross[p];
                }
            }
            g
        };
    let coef_up = coef(&a_up, &ku, &m_s_dn);
    let coef_dn = coef(&a_dn, &kd, &mt_s_up);

    // side_cot -> (bar_coefA, bar_gO[3], bar_lO); per-column k = per-electron kc_k -> weight w[off+k]
    let side = |coef: &Array2<f64>,
                bg: &Array2<f64>,
                go: &[Array2<f64>; 3],
                lo: &Array2<f64>,
                off: usize|
     -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        let nc = coef.ncols();
        let mut bcoef = Array2::<f64>::zeros((orb, nc));
        let mut bgo = [
            Array2::<f64>::zeros((orb, nc)),
            Array2::<f64>::zeros((orb, nc)),
            Array2::<f64>::zeros((orb, nc)),
        ];
        let mut blo = Array2::<f64>::zeros((orb, nc));
        for a in 0..orb {
            for k in 0..nc {
                let wk = w.map_or(1.0, |w| w[off + k]);
                let mut acc = 0.0;
                for d in 0..3 {
                    acc += bg[[k, d]] * go[d][[a, k]];
                    bgo[d][[a, k]] = wk * bg[[k, d]] * coef[[a, k]];
                }
                bcoef[[a, k]] = wk * (acc + (-0.5) * lo[[a, k]]);
                blo[[a, k]] = -0.5 * wk * coef[[a, k]];
            }
        }
        (bcoef, bgo, blo)
    };
    let (bcoef_u, bgo_u, blo_u) = side(&coef_up, bg_up, &go_u, &lo_u, 0);
    let (bcoef_d, bgo_d, blo_d) = side(&coef_dn, bg_dn, &go_d, &lo_d, n_up);
    let bcoef_d_sum = bcoef_d.sum_axis(Axis(1));
    let bcoef_u_sum = bcoef_u.sum_axis(Axis(1));
    let bar_a = |bc: &Array2<f64>,
                 k: &Array2<f64>,
                 add: &numpy::ndarray::Array1<f64>,
                 n: usize|
     -> Array2<f64> {
        let mut g = m.t().dot(bc).dot(&k.t()) + m.dot(bc).dot(k);
        for a in 0..orb {
            for c in 0..n {
                g[[a, c]] += add[a];
            }
        }
        g
    };
    let bar_a_up = bar_a(&bcoef_u, &ku, &m.dot(&bcoef_d_sum), n_up);
    let bar_a_dn = bar_a(&bcoef_d, &kd, &m.t().dot(&bcoef_u_sum), n_dn);

    // force_side: s[d,a] = -(bar_A*gO + bar_gO*H + bar_lO*GL); scatter by home atom.
    // Only the pGTO rows (local `ag`, global `gto0+ag`) are atom-centered -> scattered;
    // cotangents (bar_a/bgo/blo/go) index the FULL composite row, while H/GL are GTO-only.
    let force_side = |bar_a: &Array2<f64>,
                      bgo: &[Array2<f64>; 3],
                      blo: &Array2<f64>,
                      go: &[Array2<f64>; 3],
                      h: &[[Array2<f64>; 3]; 3],
                      gl: &[Array2<f64>; 3],
                      out: &mut Array2<f64>| {
        let nc = bar_a.ncols();
        for ag in 0..num_gto {
            let g = gto0 + ag;
            let ia = home_atom[ag];
            for d in 0..3 {
                let mut acc = 0.0;
                for k in 0..nc {
                    acc += bar_a[[g, k]] * go[d][[g, k]];
                    for e in 0..3 {
                        acc += bgo[e][[g, k]] * h[d][e][[ag, k]];
                    }
                    acc += blo[[g, k]] * gl[d][[ag, k]];
                }
                out[[ia, d]] -= acc;
            }
        }
    };
    force_side(&bar_a_up, &bgo_u, &blo_u, &go_u, &hu, &glu, out);
    force_side(&bar_a_dn, &bgo_d, &blo_d, &go_d, &hd, &gld, out);
}

/// Periodic J3 Pulay `dJ3/dR` `(n_atom,3)` (value-level, for the GFMC Pulay force).
/// Mirror of the molecular `forces::grad_j3_wrt_r` with the periodic sphe basis
/// (`a = T*pgto`, `gO = T*pgto_grad`), scattered by the per-sphe-orb home atom.
#[allow(clippy::too_many_arguments)]
pub fn grad_j3_wrt_r_pbc(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    home_atom: &[usize],
    n_atom: usize,
    cell: &ArrayView2<f64>,
    pw: Option<J3Extra<'_>>,
    gto_first: bool,
) -> Array2<f64> {
    let num_gto = t_mol.nrows();
    let num_pw = pw.map_or(0, |p| p.num_ao());
    let gto0 = if gto_first { 0 } else { num_pw };
    let basis = |r: &ArrayView2<f64>| {
        let dv = pgto_derivs(cart_j3, r, cell, true, false, false);
        let a = t_mol.dot(dv.val.as_ref().unwrap());
        let g = &dv.grad;
        let go = [t_mol.dot(&g[0]), t_mol.dot(&g[1]), t_mol.dot(&g[2])];
        match pw {
            None => (a, go),
            Some(pw) => {
                let (pv, pgx, pgy, pgz, _pl) = pw.vgl(r);
                (
                    stack_rows(&a, &pv, gto_first),
                    [
                        stack_rows(&go[0], &pgx, gto_first),
                        stack_rows(&go[1], &pgy, gto_first),
                        stack_rows(&go[2], &pgz, gto_first),
                    ],
                )
            }
        }
    };
    let (a_up, go_up) = basis(r_up);
    let (a_dn, go_dn) = basis(r_dn);
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let mk =
        |a: &Array2<f64>, k: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
            let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
            let nc = a.ncols();
            for p in 0..orb_n {
                for c in 0..nc {
                    g[[p, c]] += j1col[p] + cross[p];
                }
            }
            g
        };
    let coef_up = mk(&a_up, &ku, &m_s_dn);
    let coef_dn = mk(&a_dn, &kd, &mt_s_up);
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    let mut do_side = |coef: &Array2<f64>, go: &[Array2<f64>; 3]| {
        let nc = coef.ncols();
        for (ag, home_atom_ag) in home_atom[..num_gto].iter().enumerate() {
            let c = gto0 + ag;
            let ia = *home_atom_ag;
            for d in 0..3 {
                let mut acc = 0.0;
                for k in 0..nc {
                    acc += coef[[c, k]] * go[d][[c, k]];
                }
                out[[ia, d]] -= acc;
            }
        }
    };
    do_side(&coef_up, &go_up);
    do_side(&coef_dn, &go_dn);
    out
}

/// Periodic J3 electron-axis drift `dln(J3)/dr` (`(N_up,3)`, `(N_dn,3)`) -- the
/// electron-axis MIRROR of [`grad_j3_wrt_r_pbc`], using the SAME `coefA` cotangent AND the
/// SAME full-image `pgto_grad` orbital gradient, only scattered per ELECTRON (`+`) instead
/// of per home-atom (`-`).
///
/// Used by the periodic FORCE drift ([`crate::gfmc_forces`]'s `pbc_drift`) INSTEAD of the
/// batch `compute_grads_and_laplacian_jastrow_three_body` (which routes the J3 orbital grad
/// through `orb_value_grad_lap` -> `compute_aos_grad_lap`, applying the dynamic LEPSBAS
/// per-image cutoff the cart J3 leaf's `pgto_grad` does NOT). Sharing the identical cotangent
/// + full-image grad on both channels closes the per-config translation sum rule
///   `sum_I dln(J3)/dR_I = -sum_i dln(J3)/dr_i` to fp64 (otherwise the ~1e-9 single-vs-batch
///   image gap, Gaussian-curvature-amplified, is the residual). The batch path is kept for e_L.
#[allow(
    clippy::too_many_arguments,
    reason = "periodic J3 leaf, cell and composite-child inputs"
)]
pub fn grad_j3_wrt_r_e_pbc(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    pw: Option<J3Extra<'_>>,
    gto_first: bool,
) -> (Array2<f64>, Array2<f64>) {
    let basis = |r: &ArrayView2<f64>| {
        let dv = pgto_derivs(cart_j3, r, cell, true, false, false);
        let a = t_mol.dot(dv.val.as_ref().unwrap());
        let g = &dv.grad;
        let go = [t_mol.dot(&g[0]), t_mol.dot(&g[1]), t_mol.dot(&g[2])];
        match pw {
            None => (a, go),
            Some(pw) => {
                let (pv, pgx, pgy, pgz, _pl) = pw.vgl(r);
                (
                    stack_rows(&a, &pv, gto_first),
                    [
                        stack_rows(&go[0], &pgx, gto_first),
                        stack_rows(&go[1], &pgy, gto_first),
                        stack_rows(&go[2], &pgz, gto_first),
                    ],
                )
            }
        }
    };
    let (a_up, go_up) = basis(r_up);
    let (a_dn, go_dn) = basis(r_dn);
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let mk =
        |a: &Array2<f64>, k: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
            let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
            let nc = a.ncols();
            for p in 0..orb_n {
                for c in 0..nc {
                    g[[p, c]] += j1col[p] + cross[p];
                }
            }
            g
        };
    let coef_up = mk(&a_up, &ku, &m_s_dn);
    let coef_dn = mk(&a_dn, &kd, &mt_s_up);
    let e_side = |coef: &Array2<f64>, go: &[Array2<f64>; 3], ne: usize| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((ne, 3));
        for k in 0..ne {
            for d in 0..3 {
                let mut acc = 0.0;
                for c in 0..orb_n {
                    acc += coef[[c, k]] * go[d][[c, k]];
                }
                out[[k, d]] = acc;
            }
        }
        out
    };
    let e_up = e_side(&coef_up, &go_up, n_up);
    let e_dn = e_side(&coef_dn, &go_dn, n_dn);
    (e_up, e_dn)
}

/// Periodic J3 electron-axis kinetic `dT_J3/dr` (accumulate into `dt (Ne,3)`),
/// per-electron weighted `w` (S' mask). Mirror of the molecular
/// `stress::j3_grad_wrt_r` with the periodic sphe basis (T*pgto derivs).
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_egrad_side(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cot: &Array2<f64>,
    w: Option<&[f64]>,
    cell: &ArrayView2<f64>,
    pw: Option<J3Extra<'_>>,
    gto_first: bool,
    dt: &mut Array2<f64>,
) {
    let basis = |r: &ArrayView2<f64>| -> BasisTables {
        let dv = pgto_derivs(cart_j3, r, cell, true, true, true);
        let a = t_mol.dot(dv.val.as_ref().unwrap());
        let g = &dv.grad;
        let go = [t_mol.dot(&g[0]), t_mol.dot(&g[1]), t_mol.dot(&g[2])];
        let lo = t_mol.dot(dv.lap.as_ref().unwrap());
        let hh = dv.hess.as_ref().unwrap();
        let h = [
            [
                t_mol.dot(&hh[0][0]),
                t_mol.dot(&hh[0][1]),
                t_mol.dot(&hh[0][2]),
            ],
            [
                t_mol.dot(&hh[1][0]),
                t_mol.dot(&hh[1][1]),
                t_mol.dot(&hh[1][2]),
            ],
            [
                t_mol.dot(&hh[2][0]),
                t_mol.dot(&hh[2][1]),
                t_mol.dot(&hh[2][2]),
            ],
        ];
        let gll = dv.gradlap.as_ref().unwrap();
        let gl = [t_mol.dot(&gll[0]), t_mol.dot(&gll[1]), t_mol.dot(&gll[2])];
        match pw {
            None => (a, go, lo, h, gl),
            Some(pw) => {
                let (pv, pgx, pgy, pgz, pl) = pw.vgl(r);
                let pg = [pgx, pgy, pgz];
                let (ph, pgl) = pw.hess_gradlap(r, &pv, &pg);
                (
                    stack_rows(&a, &pv, gto_first),
                    std::array::from_fn(|c| stack_rows(&go[c], &pg[c], gto_first)),
                    stack_rows(&lo, &pl, gto_first),
                    std::array::from_fn(|c| {
                        std::array::from_fn(|d| stack_rows(&h[c][d], &ph[c][d], gto_first))
                    }),
                    std::array::from_fn(|c| stack_rows(&gl[c], &pgl[c], gto_first)),
                )
            }
        }
    };
    let (a_up, go_u, lo_u, hu, glu) = basis(r_up);
    let (a_dn, go_d, lo_d, hd, gld) = basis(r_dn);
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let mk =
        |a: &Array2<f64>, k: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
            let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
            let nc = a.ncols();
            for p in 0..orb_n {
                for c in 0..nc {
                    g[[p, c]] += j1col[p] + cross[p];
                }
            }
            g
        };
    let coef_up = mk(&a_up, &ku, &m_s_dn);
    let coef_dn = mk(&a_dn, &kd, &mt_s_up);
    let side = |coef: &Array2<f64>,
                go: &[Array2<f64>; 3],
                lo: &Array2<f64>,
                off: usize|
     -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        let nc = coef.ncols();
        let mut bcoef = Array2::<f64>::zeros((orb_n, nc));
        let mut bgo = [
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
        ];
        let mut blo = Array2::<f64>::zeros((orb_n, nc));
        for a in 0..orb_n {
            for k in 0..nc {
                let wk = w.map_or(1.0, |w| w[off + k]);
                let mut acc = 0.0;
                for d in 0..3 {
                    acc += cot[[off + k, d]] * go[d][[a, k]];
                    bgo[d][[a, k]] = -wk * cot[[off + k, d]] * coef[[a, k]];
                }
                bcoef[[a, k]] = wk * (-0.5 * lo[[a, k]] - acc);
                blo[[a, k]] = -0.5 * wk * coef[[a, k]];
            }
        }
        (bcoef, bgo, blo)
    };
    let (bcoef_u, bgo_u, blo_u) = side(&coef_up, &go_u, &lo_u, 0);
    let (bcoef_d, bgo_d, blo_d) = side(&coef_dn, &go_d, &lo_d, n_up);
    let de_side = |bcoef: &Array2<f64>,
                   bgo: &[Array2<f64>; 3],
                   blo: &Array2<f64>,
                   go: &[Array2<f64>; 3],
                   h: &[[Array2<f64>; 3]; 3],
                   gl: &[Array2<f64>; 3],
                   k: &Array2<f64>,
                   off: usize,
                   dt: &mut Array2<f64>| {
        let nc = bcoef.ncols();
        let bar_a = m.t().dot(bcoef).dot(&k.t()) + m.dot(bcoef).dot(k);
        for e in 0..nc {
            for c in 0..3 {
                let mut acc = 0.0;
                for a in 0..orb_n {
                    acc += bar_a[[a, e]] * go[c][[a, e]];
                    for d in 0..3 {
                        acc += bgo[d][[a, e]] * h[c][d][[a, e]];
                    }
                    acc += blo[[a, e]] * gl[c][[a, e]];
                }
                dt[[off + e, c]] += acc;
            }
        }
    };
    de_side(&bcoef_u, &bgo_u, &blo_u, &go_u, &hu, &glu, &ku, 0, dt);
    de_side(&bcoef_d, &bgo_d, &blo_d, &go_d, &hd, &gld, &kd, n_up, dt);
    let bar_s_dn = m.t().dot(&bcoef_u.sum_axis(Axis(1)));
    let bar_s_up = m.dot(&bcoef_d.sum_axis(Axis(1)));
    for e in 0..n_dn {
        for c in 0..3 {
            let mut acc = 0.0;
            for a in 0..orb_n {
                acc += bar_s_dn[a] * go_d[c][[a, e]];
            }
            dt[[n_up + e, c]] += acc;
        }
    }
    for e in 0..n_up {
        for c in 0..3 {
            let mut acc = 0.0;
            for a in 0..orb_n {
                acc += bar_s_up[a] * go_u[c][[a, e]];
            }
            dt[[e, c]] += acc;
        }
    }
}
