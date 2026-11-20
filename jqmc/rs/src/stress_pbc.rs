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

//! PBC strain derivatives `dlnPsi/deps`. Affine strain F=I+eps deforms cell, atoms,
//! electrons AND image translations T_n=n*cell. PW AOs are strain-invariant
//! (G*r invariant) -> 0. For a pgto AO `dAO/deps = (gradAO)(x)(r_w-R_I) - Sum_n w_n
//! (gradphi_n)(x)T_n` (affine virial + image-T correction). Real Gamma (Rust all-f64).

use numpy::ndarray::{Array2, ArrayView2, Axis};

use crate::ao_cart_deriv::{pgto_deriv, pgto_derivs, pgto_strain_tables, DerivScreens};
use crate::atomic_orbital::{compute_pw, compute_pw_value_grad_lap, AosGtoData, AosPwData};
use crate::data::{JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{compute_geminal_all_elements, det_sign_lnabs, GeminalData};
use crate::ewald::{inv3, min_image};
use crate::forces_pbc::J3Extra;
use crate::jastrow_factor::{min_image_disp, CellInfo};
use crate::mcmc::wrap_rows;
use crate::param_grad_el::{det_grad_field, gem_forward};

/// One pgto stress leaf: cart AO leaf, cart coeff block, home-atom map, kernel-atom map.
pub type StressLeaf = (AosGtoData, Array2<f64>, Vec<usize>, Vec<usize>);

/// PBC det Pulay stress `dln|detG|/deps` (3x3) at a config, jnone. Cotangent =
/// `gem_forward` wg/gq projected over the pgto leaves (1:1 grad_lnPsi_pbc det part).
fn pbc_pulay_det(
    gem: &GeminalData,
    leaves_up: &[StressLeaf],
    leaves_dn: &[StressLeaf],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
) -> Array2<f64> {
    let f = gem_forward(gem, r_up, r_dn);
    let bar_up = f.wg.clone();
    let bar_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
    let mut out = Array2::<f64>::zeros((3, 3));
    for (cart_aos, c_cart, home, kernel_atom) in leaves_up {
        pgto_det_stress_side(
            cart_aos,
            c_cart,
            &bar_up,
            r_up,
            cell,
            home,
            kernel_atom,
            positions,
            &mut out,
        );
    }
    for (cart_aos, c_cart, home, kernel_atom) in leaves_dn {
        pgto_det_stress_side(
            cart_aos,
            c_cart,
            &bar_dn,
            r_dn,
            cell,
            home,
            kernel_atom,
            positions,
            &mut out,
        );
    }
    out
}

/// Legendre `P_l(x)` and `P_l'(x)`.
fn leg_d(l: i64, x: f64) -> (f64, f64) {
    crate::forces_ecp::leg_and_deriv(l, x)
}

/// `dV_ecp(non-local)/deps` (3x3) for a PERIODIC ECP system (jnone). Per ECP term
/// (atom I, l, electron e) and mesh point g: explicit `dA/deps` (min-image V_l/P_l/costheta)
/// + wf-ratio `A*[pulay_mesh + mesh-motion - pulay_base]`, weighted by the det ratio.
///   Derivation verified vs jax.grad (RT-fixed quadrature). 1:1 the Python reference.
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_nonlocal_strain(
    gem: &GeminalData,
    leaves_up: &[StressLeaf],
    leaves_dn: &[StressLeaf],
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    nmax: i32,
    dtn: Option<&crate::dtn::JastrowDtnData>,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
) -> Array2<f64> {
    use crate::dtn::{compute_dtn_value_in_zone, compute_grads_and_laplacian_dtn};
    use crate::jastrow_factor::{
        compute_jastrow_one_body_in_zone, compute_jastrow_two_body_in_zone, j1_grad_lap_one_spin,
        j2_grad_lap,
    };
    // The mesh wf ratio `exp(J(R') - J(R))` is formed from two whole values; both are
    // evaluated in the `jastrow_ratio` zone that owns the ratio, never in `jastrow_eval`
    // (float32 in mixed), whose rounding of `|J|` the difference would keep (issue #781).
    const RATIO_ZONE: &str = "jastrow_ratio";
    let compute_dtn_value =
        |d: &crate::dtn::JastrowDtnData, ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
            compute_dtn_value_in_zone(d, RATIO_ZONE, ru, rd)
        };
    let compute_jastrow_one_body =
        |o: &JastrowOneBodyData, ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
            compute_jastrow_one_body_in_zone(o, RATIO_ZONE, ru, rd)
        };
    let compute_jastrow_two_body =
        |t: &JastrowTwoBodyData, ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
            compute_jastrow_two_body_in_zone(t, RATIO_ZONE, ru, rd)
        };
    let inv = inv3(cell);
    let nv = weights.len();
    let n_up = r_up.nrows();
    let (s0, l0) = det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view());
    let mut pulay_base = pbc_pulay_det(gem, leaves_up, leaves_dn, r_up, r_dn, cell, positions);
    // DTN (u+chi) contribution to the wf ratio: J_DTN value ratio + Pulay + drift.
    let jval_base = dtn.map(|d| compute_dtn_value(d, r_up, r_dn));
    if let Some(d) = dtn {
        pulay_base += &dtn_value_strain(d, r_up, r_dn);
    }
    // J1 (e-N) + J2 (e-e) contribution to the wf ratio: J value ratio + Pulay + drift.
    // Both periodic Williamson metrics strain with the physical lattice.
    let j12_val_base = j1.map_or(0.0, |o| compute_jastrow_one_body(o, r_up, r_dn))
        + j2.map_or(0.0, |t| compute_jastrow_two_body(t, r_up, r_dn));
    if j1.is_some() || j2.is_some() {
        jastrow_strain(j1, j2, r_up, r_dn, &mut pulay_base);
    }
    let mut out = Array2::<f64>::zeros((3, 3));
    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let l = ang_mom[t];
        let (be, ce, pe) = (exponents[t], coefficients[t], powers[t]);
        for is_up in [true, false] {
            let nr = if is_up { n_up } else { r_dn.nrows() };
            for i in 0..nr {
                let re = if is_up {
                    [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
                } else {
                    [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
                };
                // rel = min-image(R_I - r_e)
                let rel0 = [
                    positions[[ia, 0]] - re[0],
                    positions[[ia, 1]] - re[1],
                    positions[[ia, 2]] - re[2],
                ];
                let rel = min_image(rel0, cell, &inv, nmax);
                let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                let vl = ce * nrm.powf(pe - 2.0) * (-be * nrm * nrm).exp();
                let vlp = ce
                    * ((pe - 2.0) * nrm.powf(pe - 3.0) - 2.0 * be * nrm.powf(pe - 1.0))
                    * (-be * nrm * nrm).exp();
                for v in 0..nv {
                    let vg = [
                        grid_points[[v, 0]],
                        grid_points[[v, 1]],
                        grid_points[[v, 2]],
                    ];
                    let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                    let cth = -(rel[0] * vg[0] + rel[1] * vg[1] + rel[2] * vg[2]) / (nrm * vgn);
                    let (pl, dpl) = leg_d(l, cth);
                    // mesh config: electron i -> r' = r_e + rel + nrm*v_g
                    let rp = [
                        re[0] + rel[0] + nrm * vg[0],
                        re[1] + rel[1] + nrm * vg[1],
                        re[2] + rel[2] + nrm * vg[2],
                    ];
                    let (mu, md) = if is_up {
                        let mut m = r_up.to_owned();
                        m[[i, 0]] = rp[0];
                        m[[i, 1]] = rp[1];
                        m[[i, 2]] = rp[2];
                        (m, r_dn.to_owned())
                    } else {
                        let mut m = r_dn.to_owned();
                        m[[i, 0]] = rp[0];
                        m[[i, 1]] = rp[1];
                        m[[i, 2]] = rp[2];
                        (r_up.to_owned(), m)
                    };
                    let (sm, lm) = det_sign_lnabs(
                        &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                    );
                    let mut ratio = sm * s0 * (lm - l0).exp();
                    let (gdu, gdd) = det_grad_field(gem, &mu.view(), &md.view());
                    let mut drift = if is_up {
                        [gdu[[i, 0]], gdu[[i, 1]], gdu[[i, 2]]]
                    } else {
                        [gdd[[i, 0]], gdd[[i, 1]], gdd[[i, 2]]]
                    };
                    let mut pulay_m = pbc_pulay_det(
                        gem,
                        leaves_up,
                        leaves_dn,
                        &mu.view(),
                        &md.view(),
                        cell,
                        positions,
                    );
                    if let Some(d) = dtn {
                        ratio *= (compute_dtn_value(d, &mu.view(), &md.view())
                            - jval_base.unwrap())
                        .exp();
                        let (jgu, jgd, _, _) =
                            compute_grads_and_laplacian_dtn(d, &mu.view(), &md.view());
                        for c in 0..3 {
                            drift[c] += if is_up { jgu[[i, c]] } else { jgd[[i, c]] };
                        }
                        pulay_m += &dtn_value_strain(d, &mu.view(), &md.view());
                    }
                    if j1.is_some() || j2.is_some() {
                        let j12_val_m = j1
                            .map_or(0.0, |o| compute_jastrow_one_body(o, &mu.view(), &md.view()))
                            + j2.map_or(0.0, |t| {
                                compute_jastrow_two_body(t, &mu.view(), &md.view())
                            });
                        ratio *= (j12_val_m - j12_val_base).exp();
                        if let Some(o) = j1 {
                            let mv = mu.view();
                            let dv = md.view();
                            let (jg, _) = j1_grad_lap_one_spin(o, if is_up { &mv } else { &dv });
                            for c in 0..3 {
                                drift[c] += jg[[i, c]];
                            }
                        }
                        if let Some(t) = j2 {
                            let (jgu, jgd, _, _) = j2_grad_lap(t, &mu.view(), &md.view());
                            for c in 0..3 {
                                drift[c] += if is_up { jgu[[i, c]] } else { jgd[[i, c]] };
                            }
                        }
                        jastrow_strain(j1, j2, &mu.view(), &md.view(), &mut pulay_m);
                    }
                    let drift_vg = drift[0] * vg[0] + drift[1] * vg[1] + drift[2] * vg[2];
                    let a = vl * ((2 * l + 1) as f64) * pl * weights[v];
                    for c in 0..3 {
                        for d in 0..3 {
                            // explicit dA/deps
                            let dcth = -(vg[c] * rel[d]) / (nrm * vgn)
                                - cth * rel[c] * rel[d] / (nrm * nrm);
                            let da = ((2 * l + 1) as f64)
                                * weights[v]
                                * (vlp * pl * rel[c] * rel[d] / nrm + vl * dpl * dcth);
                            // wf-ratio dln(ratio)/deps = mesh_wf - pulay_base
                            let mesh_wf = pulay_m[[c, d]] - drift[c] * nrm * vg[d]
                                + (rel[c] * rel[d] / nrm) * drift_vg;
                            let dln_ratio = mesh_wf - pulay_base[[c, d]];
                            out[[c, d]] += ratio * (da + a * dln_ratio);
                        }
                    }
                }
            }
        }
    }
    out
}

/// `Sum_{mu,k} CM*dfield/deps` (3x3) for one pgto leaf: term1 affine virial (folded
/// field-grad `gfold`, WRAPPED electron) + term2 image-T (kernel field-grad
/// `gkern` x image translation `T_kappa`). 1:1 g6_pbc_kinetic_stress._field_stress
/// (real Gamma: phase=restore=1). `cm` is the AO-rank cotangent over this leaf.
#[allow(clippy::too_many_arguments)]
pub(crate) fn field_stress(
    cart_aos: &AosGtoData,
    cm: &Array2<f64>,
    gfold: &[Array2<f64>; 3],
    gkern: &[Array2<f64>; 3],
    r_w: &Array2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    let num_ao = cm.nrows();
    let n = cm.ncols();
    for mu in 0..num_ao {
        let ra = home_atom[mu];
        for k in 0..n {
            let cmk = cm[[mu, k]];
            if cmk == 0.0 {
                continue;
            }
            for c in 0..3 {
                let g = cmk * gfold[c][[mu, k]];
                for d in 0..3 {
                    out[[c, d]] += g * (r_w[[k, d]] - positions[[ra, d]]);
                }
            }
        }
    }
    for kap in 0..cart_aos.num_ao_kernel {
        let home = cart_aos.kernel_to_home[kap];
        let at = kernel_atom[kap];
        let t = [
            cart_aos.centers[[kap, 0]] - positions[[at, 0]],
            cart_aos.centers[[kap, 1]] - positions[[at, 1]],
            cart_aos.centers[[kap, 2]] - positions[[at, 2]],
        ];
        for k in 0..n {
            let cmk = cm[[home, k]];
            if cmk == 0.0 {
                continue;
            }
            for c in 0..3 {
                let g = gkern[c][[kap, k]] * cmk;
                for d in 0..3 {
                    out[[c, d]] -= g * t[d];
                }
            }
        }
    }
}

/// Det-side `dln|det|/deps` (3x3) from one Cartesian pGTO leaf.
#[allow(clippy::too_many_arguments)]
pub fn pgto_det_stress_side(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    pgto_det_stress_side_screened(
        cart_aos,
        c_cart,
        bar_orb,
        r,
        cell,
        home_atom,
        kernel_atom,
        positions,
        DerivScreens::EXACT,
        out,
    );
}

/// [`pgto_det_stress_side`] with the AO tables under the e_L screens `scr`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pgto_det_stress_side_screened(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    scr: DerivScreens,
    out: &mut Array2<f64>,
) {
    let cm = c_cart.t().dot(bar_orb);
    let r_w = wrap_rows(r, cell, &inv3(cell));
    // Fused contraction (no kernel-rank gradient tables) when the leaf is covered.
    if std::env::var("JQMC_RS_MESH_TABLES").is_err() {
        if let Some(geom) =
            crate::mesh_vgs::ShellGeom::new(cart_aos, home_atom, Some((kernel_atom, positions)))
        {
            let cm = cm.as_standard_layout().to_owned();
            let pts: Vec<[f64; 3]> = (0..r_w.nrows())
                .map(|k| [r_w[[k, 0]], r_w[[k, 1]], r_w[[k, 2]]])
                .collect();
            let w = crate::mesh_vgs::KinWeights {
                orb: cm.as_slice().unwrap(),
                lap: None,
                gou: None,
                kscale: None,
            };
            let mut o = crate::mesh_vgs::KinOut {
                elec: vec![[0.0; 3]; pts.len()],
                atom: Vec::new(),
                strain: Some([[0.0; 3]; 3]),
            };
            crate::mesh_vgs::kin_contract(
                cart_aos,
                &pts,
                &w,
                scr.value,
                scr.high,
                Some(&geom),
                &mut o,
            );
            let st = o.strain.unwrap();
            for c in 0..3 {
                for d in 0..3 {
                    out[[c, d]] += st[c][d];
                }
            }
            return;
        }
    }
    // one fused sweep: kernel-rank grad + its fold (== pgto_grad at the raw electrons)
    let tab = pgto_strain_tables(cart_aos, &r_w.view(), false, false, scr);
    field_stress(
        cart_aos,
        &cm,
        &tab.gf,
        &tab.gk,
        &r_w,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
}

/// Det-side kinetic `dT_det/deps` (3x3) from one cart-pgto leaf: value-field
/// (bar_orb, grad) plus Laplacian-field (bar_lap, gradlap) strain.
#[allow(clippy::too_many_arguments)]
pub fn pgto_kstress_side(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    pgto_kstress_side_screened(
        cart_aos,
        c_cart,
        bar_orb,
        bar_lap,
        bar_gou,
        r,
        cell,
        home_atom,
        kernel_atom,
        positions,
        DerivScreens::EXACT,
        out,
    );
}

/// [`pgto_kstress_side`] with the AO tables under the e_L screens `scr`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pgto_kstress_side_screened(
    cart_aos: &AosGtoData,
    c_cart: &Array2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    scr: DerivScreens,
    out: &mut Array2<f64>,
) {
    let r_w = wrap_rows(r, cell, &inv3(cell));
    // Fused contraction (no kernel-rank derivative tables) when the leaf is covered.
    if std::env::var("JQMC_RS_MESH_TABLES").is_err() {
        if let Some(geom) =
            crate::mesh_vgs::ShellGeom::new(cart_aos, home_atom, Some((kernel_atom, positions)))
        {
            let std_c = |a: Array2<f64>| a.as_standard_layout().to_owned();
            let cm_orb = std_c(c_cart.t().dot(bar_orb));
            let cm_lap = std_c(c_cart.t().dot(bar_lap));
            let cm_gou = bar_gou.map(|bg| {
                [
                    std_c(c_cart.t().dot(&bg[0])),
                    std_c(c_cart.t().dot(&bg[1])),
                    std_c(c_cart.t().dot(&bg[2])),
                ]
            });
            let pts: Vec<[f64; 3]> = (0..r_w.nrows())
                .map(|k| [r_w[[k, 0]], r_w[[k, 1]], r_w[[k, 2]]])
                .collect();
            let w = crate::mesh_vgs::KinWeights {
                orb: cm_orb.as_slice().unwrap(),
                lap: Some(cm_lap.as_slice().unwrap()),
                gou: cm_gou.as_ref().map(|g| {
                    [
                        g[0].as_slice().unwrap(),
                        g[1].as_slice().unwrap(),
                        g[2].as_slice().unwrap(),
                    ]
                }),
                kscale: None,
            };
            let mut o = crate::mesh_vgs::KinOut {
                elec: vec![[0.0; 3]; pts.len()],
                atom: Vec::new(),
                strain: Some([[0.0; 3]; 3]),
            };
            crate::mesh_vgs::kin_contract(
                cart_aos,
                &pts,
                &w,
                scr.value,
                scr.high,
                Some(&geom),
                &mut o,
            );
            let st = o.strain.unwrap();
            for c in 0..3 {
                for d in 0..3 {
                    out[[c, d]] += st[c][d];
                }
            }
            return;
        }
    }
    // ONE fused sweep for every field below (kernel rank + folds).
    let tab = pgto_strain_tables(cart_aos, &r_w.view(), bar_gou.is_some(), true, scr);
    // value field (bar_orb, grad)
    let cm_orb = c_cart.t().dot(bar_orb);
    field_stress(
        cart_aos,
        &cm_orb,
        &tab.gf,
        &tab.gk,
        &r_w,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
    // Laplacian field (bar_lap, gradlap)
    let cm_lap = c_cart.t().dot(bar_lap);
    field_stress(
        cart_aos,
        &cm_lap,
        tab.glf.as_ref().unwrap(),
        tab.glk.as_ref().unwrap(),
        &r_w,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
    // gradient field (bar_gou, Hessian): grad(grad_d AO) = Hessian[:,d] (with-Jastrow only)
    if let Some(bg) = bar_gou {
        let hf = tab.hf.as_ref().unwrap(); // [3c][3d] (num_ao,N)
        let hk = tab.hk.as_ref().unwrap();
        for d in 0..3 {
            let cm_gou_d = c_cart.t().dot(&bg[d]);
            let gfold_d = [hf[0][d].clone(), hf[1][d].clone(), hf[2][d].clone()];
            let gkern_d = [hk[0][d].clone(), hk[1][d].clone(), hk[2][d].clone()];
            field_stress(
                cart_aos,
                &cm_gou_d,
                &gfold_d,
                &gkern_d,
                &r_w,
                home_atom,
                kernel_atom,
                positions,
                out,
            );
        }
    }
}

/// PW gradient-field stress: `d(grad_d AO)/deps = -grad_c AO*delta_{d,c'}` (kappa->kappa*F^-^1) ->
/// `out[c][c'] = -Sum_k cm_gou[c'][mu,k]*grad_c[mu,k]`. `cm_gou` = [3] (num_pw, N).
pub fn pw_gou_stress(
    pw: &AosPwData,
    cm_gou_pw: &[Array2<f64>; 3],
    r: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    let (_v, gx, gy, gz, _l) = compute_pw_value_grad_lap(pw, r);
    let grad = [gx, gy, gz];
    let n = grad[0].ncols();
    for c in 0..3 {
        for cp in 0..3 {
            let mut acc = 0.0;
            for a in 0..pw.num_ao {
                for k in 0..n {
                    acc += cm_gou_pw[cp][[a, k]] * grad[c][[a, k]];
                }
            }
            out[[c, cp]] -= acc;
        }
    }
}

/// Plane-wave kinetic stress: PW value strain-invariant but `lap=-|kappa|^2val` and
/// `|kappa|^2(cell)` varies -> `dlap/deps = 2kappa_c kappa_d val`. `cm_lap_pw` = (num_pw, N).
pub fn pw_lap_stress(
    pw: &AosPwData,
    cm_lap_pw: &Array2<f64>,
    r: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    let val = compute_pw(pw, r); // (num_pw, N)
    let n = val.ncols();
    for a in 0..pw.num_ao {
        let mut w = 0.0;
        for k in 0..n {
            w += cm_lap_pw[[a, k]] * val[[a, k]];
        }
        let kap = [pw.gx[a], pw.gy[a], pw.gz[a]];
        for c in 0..3 {
            for d in 0..3 {
                out[[c, d]] += 2.0 * w * kap[c] * kap[d];
            }
        }
    }
}

/// Williamson smooth distance `d` plus its gradient field `gd=gradd`, `q=gd*gd`,
/// Laplacian `L=grad^2d` and all their strain derivatives in closed form. Fractional `s` is strain-
/// invariant; eps enters only via `a^2_k=|cell_k|^2/pi^2` and `B[:,k]=inv_cell[:,k]`.
struct SmoothStrain {
    d: f64,
    gd: [f64; 3],
    q: f64,
    l: f64,
    dd: [[f64; 3]; 3],
    dgd: [[[f64; 3]; 3]; 3], // (i,a,b)
    dq: [[f64; 3]; 3],
    dl: [[f64; 3]; 3],
}

fn smooth_strain(disp: [f64; 3], ci: &crate::jastrow_factor::CellInfo) -> SmoothStrain {
    use crate::jastrow_factor::positive_geometry;
    use std::f64::consts::PI;
    let raw = positive_geometry(disp, ci);
    let d = raw.square.sqrt();
    let gd = raw.numerator.map(|v| v / d);
    let q = gd.iter().map(|v| v * v).sum::<f64>();
    let l = (raw.trace - q) / d;
    let off_h: [f64; 3] = std::array::from_fn(|i| {
        (0..3)
            .filter(|&j| i != j)
            .map(|j| raw.h[j] * ci.gram[i][j])
            .sum()
    });
    let mut dd = [[0.0; 3]; 3];
    let mut dgd = [[[0.0; 3]; 3]; 3];
    let mut dq = [[0.0; 3]; 3];
    let mut dl = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut tensor = raw.cart[a] * raw.cart[b];
            let mut d_off_h = [0.0; 3];
            let mut dtrace = 0.0;
            for i in 0..3 {
                tensor += raw.z[i] * raw.z[i] * ci.cell[i][a] * ci.cell[i][b] / (PI * PI);
                let d_diagonal = 2.0 * ci.cell[i][a] * ci.cell[i][b];
                let d_inverse_diagonal = -2.0 * ci.inv[a][i] * ci.inv[b][i];
                dtrace += 2.0
                    * (d_diagonal * ci.inverse_gram[i][i] + ci.gram[i][i] * d_inverse_diagonal)
                    * raw.z[i];
                for j in 0..3 {
                    if i != j {
                        let d_off = ci.cell[i][a] * ci.cell[j][b] + ci.cell[i][b] * ci.cell[j][a];
                        let d_inverse = -ci.inv[a][i] * ci.inv[b][j] - ci.inv[b][i] * ci.inv[a][j];
                        d_off_h[i] += raw.h[j] * d_off;
                        dtrace += 4.0
                            * raw.z[i]
                            * raw.z[j]
                            * (d_off * ci.inverse_gram[i][j] + ci.gram[i][j] * d_inverse);
                    }
                }
                dtrace -= 4.0
                    * raw.h[i]
                    * (d_inverse_diagonal * off_h[i] + ci.inverse_gram[i][i] * d_off_h[i]);
            }
            dd[a][b] = tensor / d;
            for i in 0..3 {
                let mut dm = -(2.0 / PI)
                    * (0..3)
                        .map(|k| raw.z[k] * d_off_h[k] * ci.inv[i][k])
                        .sum::<f64>();
                if i == a {
                    dm += raw.cart[b];
                }
                if i == b {
                    dm += raw.cart[a] - raw.numerator[a];
                }
                dgd[i][a][b] = (dm - gd[i] * dd[a][b]) / d;
                dq[a][b] += 2.0 * gd[i] * dgd[i][a][b];
            }
            dl[a][b] = (dtrace - dq[a][b] - l * dd[a][b]) / d;
        }
    }
    SmoothStrain {
        d,
        gd,
        q,
        l,
        dd,
        dgd,
        dq,
        dl,
    }
}

/// DTN minimum-image **Euclidean** affine strain (used instead of `smooth_strain` for the
/// DTN channels, as jqmc's `_dtn_scalar_dist_with_derivs` uses the min-image distance).
/// Under `F=I+eps` the min-image displacement `u->F*u` (round index locally
/// constant), so on `u=u_min`, `d=|u|`: `gd_i=u_i/d`, `q=1`, `l=2/d`, and
/// `dd[a][b]=u_a u_b/d`, `dgd[i][a][b]=delta_ia u_b/d - u_i u_a u_b/d^3`, `dq=0`,
/// `dl[a][b]=-2 u_a u_b/d^3`. (eps is the full asymmetric matrix, matching jax.grad.)
fn min_image_strain(disp: [f64; 3], ci: &CellInfo) -> SmoothStrain {
    // min-image displacement u: the shared DTN rule (wrap plus image shell,
    // exact below the Wigner-Seitz inscribed radius on any cell, issue #68).
    let u = min_image_disp(disp, ci);
    let d = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let gd = [u[0] / d, u[1] / d, u[2] / d];
    let q = 1.0;
    let l = 2.0 / d;
    let mut dd = [[0.0; 3]; 3];
    let mut dgd = [[[0.0; 3]; 3]; 3];
    let dq = [[0.0; 3]; 3];
    let mut dl = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            dd[a][b] = u[a] * u[b] / d;
            dl[a][b] = -2.0 * u[a] * u[b] / d.powi(3);
            for i in 0..3 {
                let kron = if i == a { 1.0 } else { 0.0 };
                dgd[i][a][b] = kron * u[b] / d - u[i] * u[a] * u[b] / d.powi(3);
            }
        }
    }
    SmoothStrain {
        d,
        gd,
        q,
        l,
        dd,
        dgd,
        dq,
        dl,
    }
}

/// No-cell **Euclidean** affine strain for a **cell-less J2** (the Jastrow carries no
/// cell, so the pair distance is plain Euclidean `|disp|`, not the Williamson smooth
/// metric and not min-image). Closed form (same as [`min_image_strain`] on the raw
/// displacement, no wrapping): `d=|u|`, `gd=u/d`, `q=1`, `l=2/d`, `dd[a][b]=u_a u_b/d`,
/// `dgd[i][a][b]=delta_ia u_b/d - u_i u_a u_b/d^3`, `dq=0`, `dl[a][b]=-2 u_a u_b/d^3`.
/// For a cell-less J2 there is NO lattice metric to strain, so the cell-straining
/// `jastrow_kstress`/`jastrow_strain` J2 block reduces to this position virial -- the
/// same position-virial result used by the cell-less Python path.
fn euclidean_strain(disp: [f64; 3]) -> SmoothStrain {
    let u = disp;
    let d = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let gd = [u[0] / d, u[1] / d, u[2] / d];
    let mut dd = [[0.0; 3]; 3];
    let mut dgd = [[[0.0; 3]; 3]; 3];
    let dq = [[0.0; 3]; 3];
    let mut dl = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            dd[a][b] = u[a] * u[b] / d;
            dl[a][b] = -2.0 * u[a] * u[b] / d.powi(3);
            for i in 0..3 {
                let kron = if i == a { 1.0 } else { 0.0 };
                dgd[i][a][b] = kron * u[b] / d - u[i] * u[a] * u[b] / d.powi(3);
            }
        }
    }
    SmoothStrain {
        d,
        gd,
        q: 1.0,
        l: 2.0 / d,
        dd,
        dgd,
        dq,
        dl,
    }
}

/// `dJ_DTN/deps` (3x3, the DTN Pulay/value strain) for u + chi + f channels.
/// Periodic DTN uses the minimum-image metric; molecular DTN uses the ordinary
/// Euclidean affine metric.  This distinction matters for the molecular ECP GFMC
/// stress path, whose DTN data intentionally has no cell.
#[inline]
fn dtn_policy_radial(
    l: f64,
    coeff: &[f64],
    coeff_policy: &[f64],
    r: f64,
    c: i32,
) -> (f64, f64, f64) {
    let direct = crate::dtn::radial_kernel_derivs(r, coeff_policy, l, c);
    let cutoff = crate::dtn::radial_kernel_derivs(r, coeff, l, c - 1);
    let scale = (c as f64) * l;
    (
        direct.0 + scale * cutoff.0,
        direct.1 + scale * cutoff.1,
        direct.2 + scale * cutoff.2,
    )
}

/// The scalar cutoff-policy field O_L=dJ_DTN/dln(ell_V), its electron
/// gradient and its electron Laplacian.  The policy exists only for PBC.
fn dtn_cutoff_policy_fields(
    d: &crate::dtn::JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (f64, Array2<f64>, Vec<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let mut grad = Array2::<f64>::zeros((n_e, 3));
    let mut lap = vec![0.0; n_e];
    if d.cell.is_none() {
        return (0.0, grad, lap);
    }
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let cell = d.cell.as_ref().expect("PBC DTN cell").view();
    let ci = CellInfo::new(&cell);
    let metric = |disp: [f64; 3]| min_image_strain(disp, &ci);
    let mut value = 0.0;

    if d.use_u {
        for i in 0..n_e {
            for j in (i + 1)..n_e {
                let ch = if (i < n_up) == (j < n_up) { 0 } else { 1 };
                let sm = metric([
                    elec(i)[0] - elec(j)[0],
                    elec(i)[1] - elec(j)[1],
                    elec(i)[2] - elec(j)[2],
                ]);
                let coeff = d.alpha.row(ch).to_vec();
                let coeff_policy = d.alpha_policy.row(ch).to_vec();
                let (h, h1, h2) = dtn_policy_radial(d.l_u, &coeff, &coeff_policy, sm.d, d.c);
                value += h;
                let lap_pair = h2 + h1 * sm.l;
                for axis in 0..3 {
                    let g = h1 * sm.gd[axis];
                    grad[[i, axis]] += g;
                    grad[[j, axis]] -= g;
                }
                lap[i] += lap_pair;
                lap[j] += lap_pair;
            }
        }
    }
    if d.use_chi {
        for e in 0..n_e {
            let re = elec(e);
            for ia in 0..d.positions.nrows() {
                let sp = d.species_index[ia];
                let sm = metric([
                    re[0] - d.positions[[ia, 0]],
                    re[1] - d.positions[[ia, 1]],
                    re[2] - d.positions[[ia, 2]],
                ]);
                let coeff = d.beta.row(sp).to_vec();
                let coeff_policy = d.beta_policy.row(sp).to_vec();
                let (h, h1, h2) = dtn_policy_radial(d.l_chi[sp], &coeff, &coeff_policy, sm.d, d.c);
                value += h;
                for axis in 0..3 {
                    grad[[e, axis]] += h1 * sm.gd[axis];
                }
                lap[e] += h2 + h1 * sm.l;
            }
        }
    }
    if d.use_f {
        // e-N geometry once per (electron, atom); triples outside the cutoff add exact zeros.
        let n_atom = d.positions.nrows();
        let en: Vec<SmoothStrain> = (0..n_e)
            .flat_map(|e| {
                let re = elec(e);
                (0..n_atom)
                    .map(|ia| {
                        metric([
                            re[0] - d.positions[[ia, 0]],
                            re[1] - d.positions[[ia, 1]],
                            re[2] - d.positions[[ia, 2]],
                        ])
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for i in 0..n_e {
            let ri = elec(i);
            for j in (i + 1)..n_e {
                let rj = elec(j);
                let sz = metric([ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]]);
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let (sx, sy) = (&en[i * n_atom + ia], &en[j * n_atom + ia]);
                    if sx.d >= d.l_f[sp] || sy.d >= d.l_f[sp] {
                        continue;
                    }
                    let (f, fx, fy, fz, fxx, fyy, fzz, fxz, fyz) =
                        crate::dtn::f_atom_policy_value_grad_hess(
                            d, sp, sx.d, sy.d, sz.d, d.l_f[sp],
                        );
                    value += f;
                    let sxz: f64 = (0..3).map(|axis| sx.gd[axis] * sz.gd[axis]).sum();
                    let syz: f64 = (0..3).map(|axis| sy.gd[axis] * sz.gd[axis]).sum();
                    for axis in 0..3 {
                        grad[[i, axis]] += fx * sx.gd[axis] + fz * sz.gd[axis];
                        grad[[j, axis]] += fy * sy.gd[axis] - fz * sz.gd[axis];
                    }
                    lap[i] += fxx + fzz + 2.0 * fxz * sxz + fx * sx.l + fz * sz.l;
                    lap[j] += fyy + fzz - 2.0 * fyz * syz + fy * sy.l + fz * sz.l;
                }
            }
        }
    }
    (value, grad, lap)
}

pub fn dtn_value_strain(
    d: &crate::dtn::JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    use crate::forces_dtn::h_derivs;
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = d.positions.nrows();
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let ci = d.cell.as_ref().map(|cell| CellInfo::new(&cell.view()));
    // Affine strain of a min-image (cell) / Euclidean (no cell) distance: `d = |u|`,
    // `dd/deps_ab = u_a u_b / d` (the `dd` of `min_image_strain` / `euclidean_strain`).
    // Each electron-atom and electron-pair geometry is formed once and reused by every
    // channel and triple.
    let geo = |disp: [f64; 3]| -> (f64, [[f64; 3]; 3]) {
        let u = match ci.as_ref() {
            Some(ci) => min_image_disp(disp, ci),
            None => disp,
        };
        let dist = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let mut dd = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                dd[a][b] = u[a] * u[b] / dist;
            }
        }
        (dist, dd)
    };
    let policy_on = d.cell.is_some();
    let mut policy = 0.0_f64;
    let mut out = Array2::<f64>::zeros((3, 3));
    let add = |out: &mut Array2<f64>, w: f64, dd: &[[f64; 3]; 3]| {
        for a in 0..3 {
            for b in 0..3 {
                out[[a, b]] += w * dd[a][b];
            }
        }
    };
    let need_en = d.use_chi || d.use_f;
    let en: Vec<(f64, [[f64; 3]; 3])> = if need_en {
        (0..n_e)
            .flat_map(|e| {
                let re = elec(e);
                (0..n_atom)
                    .map(|ia| {
                        geo([
                            re[0] - d.positions[[ia, 0]],
                            re[1] - d.positions[[ia, 1]],
                            re[2] - d.positions[[ia, 2]],
                        ])
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        Vec::new()
    };
    if d.use_chi {
        for e in 0..n_e {
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let coef = d.beta.row(sp).to_vec();
                let (dist, dd) = &en[e * n_atom + ia];
                let (h1, _, _) = h_derivs(d.l_chi[sp], &coef, *dist, d.c);
                add(&mut out, h1, dd);
                if policy_on {
                    let cp = d.beta_policy.row(sp).to_vec();
                    policy += dtn_policy_radial(d.l_chi[sp], &coef, &cp, *dist, d.c).0;
                }
            }
        }
    }
    if d.use_u || d.use_f {
        let alpha: Vec<Vec<f64>> = (0..d.alpha.nrows())
            .map(|c| d.alpha.row(c).to_vec())
            .collect();
        let alpha_p: Option<Vec<Vec<f64>>> = (policy_on && d.use_u).then(|| {
            (0..d.alpha_policy.nrows())
                .map(|c| d.alpha_policy.row(c).to_vec())
                .collect()
        });
        for i in 0..n_e {
            let ri = elec(i);
            for j in (i + 1)..n_e {
                let rj = elec(j);
                let (dij, ddij) = geo([ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]]);
                if d.use_u {
                    let ch = if (i < n_up) == (j < n_up) { 0 } else { 1 };
                    let (h1, _, _) = h_derivs(d.l_u, &alpha[ch], dij, d.c);
                    add(&mut out, h1, &ddij);
                    if let Some(ap) = alpha_p.as_ref() {
                        policy += dtn_policy_radial(d.l_u, &alpha[ch], &ap[ch], dij, d.c).0;
                    }
                }
                // f-channel (e-e-N 3-body): f = cut(r_iI)*cut(r_jI)*Sum gamma r_iI^l r_jI^m
                // r_ij^n. df/deps = f_x*dr_iI/deps + f_y*dr_jI/deps + f_z*dr_ij/deps. The cutoff
                // `(l - r)^c` and every derivative vanish for `r_iI >= l` or `r_jI >= l`, so such
                // triples add exact zeros and are skipped.
                if d.use_f {
                    for ia in 0..n_atom {
                        let sp = d.species_index[ia];
                        let lf = d.l_f[sp];
                        let (di, ddi) = &en[i * n_atom + ia];
                        let (dj, ddj) = &en[j * n_atom + ia];
                        if *di >= lf || *dj >= lf {
                            continue;
                        }
                        let (fx, fy, fz, ..) =
                            crate::dtn::f_atom_grad_hess(d, sp, *di, *dj, dij, lf);
                        for a in 0..3 {
                            for b in 0..3 {
                                out[[a, b]] += fx * ddi[a][b] + fy * ddj[a][b] + fz * ddij[a][b];
                            }
                        }
                        if policy_on {
                            policy +=
                                crate::dtn::f_atom_policy_value_grad_hess(d, sp, *di, *dj, dij, lf)
                                    .0;
                        }
                    }
                }
            }
        }
    }
    if policy_on {
        for axis in 0..3 {
            out[[axis, axis]] += policy / 3.0;
        }
    }
    out
}

/// Policy-field terms that contain global electron `g_idx` at `src`.
/// Their new-minus-old difference is the rank-one comoving-cutoff addition to
/// [`dtn_value_strain_move_delta`].
fn dtn_cutoff_policy_one_electron_value(
    d: &crate::dtn::JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_idx: usize,
    src: [f64; 3],
) -> f64 {
    if d.cell.is_none() {
        return 0.0;
    }
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
    let cell = d.cell.as_ref().expect("PBC DTN cell").view();
    let ci = CellInfo::new(&cell);
    let dist =
        |a: [f64; 3], b: [f64; 3]| min_image_strain([a[0] - b[0], a[1] - b[1], a[2] - b[2]], &ci).d;
    let mut value = 0.0;
    if d.use_u {
        for k in 0..n_e {
            if k == g_idx {
                continue;
            }
            let ch = if (g_idx < n_up) == (k < n_up) { 0 } else { 1 };
            let coeff = d.alpha.row(ch).to_vec();
            let coeff_policy = d.alpha_policy.row(ch).to_vec();
            value += dtn_policy_radial(d.l_u, &coeff, &coeff_policy, dist(src, elec(k)), d.c).0;
        }
    }
    if d.use_chi {
        for ia in 0..d.positions.nrows() {
            let sp = d.species_index[ia];
            let atom = [
                d.positions[[ia, 0]],
                d.positions[[ia, 1]],
                d.positions[[ia, 2]],
            ];
            let coeff = d.beta.row(sp).to_vec();
            let coeff_policy = d.beta_policy.row(sp).to_vec();
            value += dtn_policy_radial(d.l_chi[sp], &coeff, &coeff_policy, dist(src, atom), d.c).0;
        }
    }
    if d.use_f {
        for k in 0..n_e {
            if k == g_idx {
                continue;
            }
            let rk = elec(k);
            let (a, b) = if g_idx < k { (src, rk) } else { (rk, src) };
            let rab = dist(a, b);
            for ia in 0..d.positions.nrows() {
                let sp = d.species_index[ia];
                let atom = [
                    d.positions[[ia, 0]],
                    d.positions[[ia, 1]],
                    d.positions[[ia, 2]],
                ];
                value += crate::dtn::f_atom_policy_value_grad_hess(
                    d,
                    sp,
                    dist(a, atom),
                    dist(b, atom),
                    rab,
                    d.l_f[sp],
                )
                .0;
            }
        }
    }
    value
}

/// DTN value strain (`dJ/deps`, u+chi+f) DELTA for a single-electron hop of electron
/// `li` (spin `is_up`) to `new_pos`: `strain(moved) - strain(base)`, restricted to the
/// terms that touch the moved electron `g`. Per-term formulas byte-identical to
/// [`dtn_value_strain`], so `base + delta` reproduces the full moved-config value strain
/// to fp noise. O(Ne + Ne*Natom) per hop, NOT the full O(Ne^2 + Ne^2*Natom).
pub fn dtn_value_strain_move_delta(
    d: &crate::dtn::JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_pos: [f64; 3],
) -> Array2<f64> {
    use crate::forces_dtn::h_derivs;
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
    let ci = d.cell.as_ref().map(|cell| CellInfo::new(&cell.view()));
    let metric = |disp: [f64; 3]| match ci.as_ref() {
        Some(ci) => min_image_strain(disp, ci),
        _ => euclidean_strain(disp),
    };
    let g_idx = if is_up { li } else { n_up + li };
    let r_old = elec(g_idx);
    let spin = |e: usize| -> usize {
        if e < n_up {
            0
        } else {
            1
        }
    };
    let mut out = Array2::<f64>::zeros((3, 3));
    let accumulate = |g_pos: [f64; 3], sign: f64, out: &mut Array2<f64>| {
        // chi: only the moved electron's e-N legs.
        if d.use_chi {
            for ia in 0..d.positions.nrows() {
                let sp = d.species_index[ia];
                let coef: Vec<f64> = d.beta.row(sp).to_vec();
                let sm = metric([
                    g_pos[0] - d.positions[[ia, 0]],
                    g_pos[1] - d.positions[[ia, 1]],
                    g_pos[2] - d.positions[[ia, 2]],
                ]);
                let (h1, _, _) = h_derivs(d.l_chi[sp], &coef, sm.d, d.c);
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += sign * h1 * sm.dd[a][b];
                    }
                }
            }
        }
        // u: pairs (g,k) (dd is symmetric under displacement negation -> order-free).
        if d.use_u {
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let ch = if spin(g_idx) == spin(k) { 0 } else { 1 };
                let coef: Vec<f64> = d.alpha.row(ch).to_vec();
                let rk = elec(k);
                let sm = metric([g_pos[0] - rk[0], g_pos[1] - rk[1], g_pos[2] - rk[2]]);
                let (h1, _, _) = h_derivs(d.l_u, &coef, sm.d, d.c);
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += sign * h1 * sm.dd[a][b];
                    }
                }
            }
        }
        // f: pairs (g,k) x atoms, respecting the global i<j ordering (fx = first leg = a).
        if d.use_f {
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let (a_pos, b_pos) = if g_idx < k {
                    (g_pos, elec(k))
                } else {
                    (elec(k), g_pos)
                };
                let sm_ij = metric([
                    a_pos[0] - b_pos[0],
                    a_pos[1] - b_pos[1],
                    a_pos[2] - b_pos[2],
                ]);
                for ia in 0..d.positions.nrows() {
                    let sp = d.species_index[ia];
                    let pia = [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ];
                    let sm_i = metric([a_pos[0] - pia[0], a_pos[1] - pia[1], a_pos[2] - pia[2]]);
                    let sm_j = metric([b_pos[0] - pia[0], b_pos[1] - pia[1], b_pos[2] - pia[2]]);
                    let (fx, fy, fz, ..) =
                        crate::dtn::f_atom_grad_hess(d, sp, sm_i.d, sm_j.d, sm_ij.d, d.l_f[sp]);
                    for a in 0..3 {
                        for b in 0..3 {
                            out[[a, b]] += sign
                                * (fx * sm_i.dd[a][b] + fy * sm_j.dd[a][b] + fz * sm_ij.dd[a][b]);
                        }
                    }
                }
            }
        }
    };
    accumulate(new_pos, 1.0, &mut out);
    accumulate(r_old, -1.0, &mut out);
    if d.cell.is_some() {
        let delta_policy = dtn_cutoff_policy_one_electron_value(d, r_up, r_dn, g_idx, new_pos)
            - dtn_cutoff_policy_one_electron_value(d, r_up, r_dn, g_idx, r_old);
        for axis in 0..3 {
            out[[axis, axis]] += delta_policy / 3.0;
        }
    }
    out
}

/// One f-channel `(other electron k, atom I)` term of [`dtn_value_strain_move_delta_batch`]
/// with the k-I leg fixed: the `gamma` (and `gamma_policy`) sums over the fixed leg's power
/// index contracted at its distance `r_f`, so each hop of the moved electron sums only over
/// the moving e-N and e-e power indices. `g_first`: the moved electron is the lower global
/// index, i.e. the `x` (`r_iI`) leg of [`crate::dtn::f_atom_grad_hess`]; otherwise the `y` leg.
struct FixedLeg {
    k: usize,
    ia: usize,
    sp: usize,
    g_first: bool,
    /// fixed-leg cutoff value and derivative `(l-r_f)^c`, `d/dr_f`, and `(l-r_f)^(c-1)`.
    cf: (f64, f64, f64),
    /// `[moving power][e-e power]` contractions: `t0 = Sum_f gamma r_f^f`,
    /// `t1 = Sum_f gamma f r_f^(f-1)`, `tp = Sum_f gamma_policy r_f^f`.
    t0: Vec<f64>,
    t1: Vec<f64>,
    tp: Vec<f64>,
}

impl FixedLeg {
    fn new(
        d: &crate::dtn::JastrowDtnData,
        sp: usize,
        k: usize,
        ia: usize,
        rf: f64,
        g_first: bool,
    ) -> Self {
        let (ne, nz) = (d.n_en + 1, d.n_ee + 1);
        let stride_n = nz;
        let stride_m = ne * stride_n;
        let stride_l = ne * stride_m;
        let base = sp * stride_l;
        let mut pw = vec![0.0; ne];
        let mut dpw = vec![0.0; ne];
        let (mut rk, mut rkm1) = (1.0_f64, 0.0_f64);
        for f in 0..ne {
            pw[f] = rk;
            dpw[f] = f as f64 * rkm1;
            rkm1 = rk;
            rk *= rf;
        }
        let (mut t0, mut t1, mut tp) = (vec![0.0; ne * nz], vec![0.0; ne * nz], vec![0.0; ne * nz]);
        for mv in 0..ne {
            for f in 0..ne {
                // gamma[l][m][n]: the moving leg is `l` when the moved electron is first.
                let off = if g_first {
                    base + mv * stride_m + f * stride_n
                } else {
                    base + f * stride_m + mv * stride_n
                };
                for n in 0..nz {
                    let g = d.gamma[off + n];
                    let gp = d.gamma_policy[off + n];
                    t0[mv * nz + n] += g * pw[f];
                    t1[mv * nz + n] += g * dpw[f];
                    tp[mv * nz + n] += gp * pw[f];
                }
            }
        }
        let l_i = d.l_f[sp];
        let (c0, c1, _) = crate::dtn::cutoff_derivs(rf, l_i, d.c);
        let cm = crate::dtn::cutoff_derivs(rf, l_i, d.c - 1).0;
        FixedLeg {
            k,
            ia,
            sp,
            g_first,
            cf: (c0, c1, cm),
            t0,
            t1,
            tp,
        }
    }

    /// `(f_moving, f_fixed, f_ee, policy)` at moving e-N distance `rm` and e-e distance `rz`:
    /// the gradient of [`crate::dtn::f_atom_grad_hess`] split by leg, and the value of
    /// [`crate::dtn::f_atom_policy_value_grad_hess`] (the cutoff-policy f field).
    fn eval(
        &self,
        d: &crate::dtn::JastrowDtnData,
        rm: f64,
        rz: f64,
        l_i: f64,
    ) -> (f64, f64, f64, f64) {
        let (ne, nz) = (d.n_en + 1, d.n_ee + 1);
        let (gm, gpm, _) = crate::dtn::cutoff_derivs(rm, l_i, d.c);
        let gmm = crate::dtn::cutoff_derivs(rm, l_i, d.c - 1).0;
        let (gf, gpf, gfm) = self.cf;
        let (mut s0, mut sm, mut sf, mut sz, mut spol) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut xm = 1.0_f64;
        let mut dxm = 0.0_f64;
        for mv in 0..ne {
            let mut zp = 1.0_f64;
            let mut dzp = 0.0_f64;
            for n in 0..nz {
                let (a0, a1, ap) = (
                    self.t0[mv * nz + n],
                    self.t1[mv * nz + n],
                    self.tp[mv * nz + n],
                );
                s0 += a0 * xm * zp;
                sm += a0 * dxm * zp;
                sf += a1 * xm * zp;
                sz += a0 * xm * dzp;
                spol += ap * xm * zp;
                dzp = (n + 1) as f64 * zp;
                zp *= rz;
            }
            dxm = (mv + 1) as f64 * xm;
            xm *= rm;
        }
        let gh = gm * gf;
        let f_mov = gpm * gf * s0 + gh * sm;
        let f_fix = gm * gpf * s0 + gh * sf;
        let f_z = gh * sz;
        let cl = (d.c as f64) * l_i;
        let policy = gh * spol + cl * (gmm * gf * s0 + gm * gfm * s0);
        (f_mov, f_fix, f_z, policy)
    }
}

/// Distance and its affine-strain derivative `dd[a][b] = u_a u_b / d` (the `d` / `dd`
/// entries of [`min_image_strain`] / [`euclidean_strain`]).
struct DistStrain {
    d: f64,
    dd: [[f64; 3]; 3],
}

/// The DTN drift (electron gradient) of global electron `g_idx` alone, with that electron at
/// `g_pos` and every other electron at its configuration position: the moved row of
/// [`crate::dtn::dtn_drift_move_delta`] is `dtn_drift_row(new) - dtn_drift_row(old)` (the
/// same u / chi / f terms that touch `g_idx`, in the same per-term arithmetic), without the
/// other electrons' rows. fp64 lane (the caller keeps the full delta under an f32 policy).
pub(crate) fn dtn_drift_row(
    d: &crate::dtn::JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_idx: usize,
    g_pos: [f64; 3],
) -> [f64; 3] {
    use crate::dtn::{cutoff_derivs, f_atom_grad_hess, radial_kernel_derivs};
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dgl = |a: [f64; 3], b: [f64; 3]| -> (f64, [f64; 3]) {
        match &ci {
            Some(ci) => {
                let (r, gv, _, _) = crate::jastrow_factor::min_image_dist(
                    [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
                    ci,
                );
                (r, gv)
            }
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                let r = (dx * dx + dy * dy + dz * dz)
                    .sqrt()
                    .max(crate::setting_consts::EPS_safe_distance);
                (r, [dx / r, dy / r, dz / r])
            }
        }
    };
    let n_up = r_up.nrows();
    let n_tot = n_up + r_dn.nrows();
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let pos = |ia: usize| -> [f64; 3] {
        [
            d.positions[[ia, 0]],
            d.positions[[ia, 1]],
            d.positions[[ia, 2]],
        ]
    };
    let n_atom = d.positions.nrows();
    let is_up = g_idx < n_up;
    let mut out = [0.0_f64; 3];
    if d.use_u {
        let a_par = d.alpha.row(0);
        let a_anti = d.alpha.row(d.alpha.nrows() - 1);
        for k in 0..n_tot {
            if k == g_idx {
                continue;
            }
            let same = (k < n_up) == is_up;
            let a = if same {
                a_par.as_slice().unwrap()
            } else {
                a_anti.as_slice().unwrap()
            };
            let (r, gv) = dgl(g_pos, rget(k));
            let (_f, fp, _fpp) = radial_kernel_derivs(r, a, d.l_u, d.c);
            for c in 0..3 {
                out[c] += fp * gv[c];
            }
        }
    }
    let en: Vec<(f64, [f64; 3])> = (0..n_atom).map(|ia| dgl(g_pos, pos(ia))).collect();
    if d.use_chi {
        for (ia, en_ia) in en[..n_atom].iter().enumerate() {
            let sp = d.species_index[ia];
            let (r, gv) = *en_ia;
            let (cf, _, _) = cutoff_derivs(r, d.l_chi[sp], d.c);
            if cf == 0.0 {
                continue;
            }
            let (_f, fp, _fpp) =
                radial_kernel_derivs(r, d.beta.row(sp).as_slice().unwrap(), d.l_chi[sp], d.c);
            for c in 0..3 {
                out[c] += fp * gv[c];
            }
        }
    }
    if d.use_f {
        for k in 0..n_tot {
            if k == g_idx {
                continue;
            }
            let rk = rget(k);
            // global i<j ordering: the r_ij leg points from the lower to the higher index
            let g_first = g_idx < k;
            let (r_ij, g_ij_a) = if g_first {
                dgl(g_pos, rk)
            } else {
                dgl(rk, g_pos)
            };
            let g_ij_g = if g_first {
                g_ij_a
            } else {
                [-g_ij_a[0], -g_ij_a[1], -g_ij_a[2]]
            };
            for (ia, en_ia) in en[..n_atom].iter().enumerate() {
                let sp = d.species_index[ia];
                let l_i = d.l_f[sp];
                let (r_g, g_g) = *en_ia;
                if l_i - r_g <= 0.0 {
                    continue;
                }
                let (r_k, _) = dgl(rk, pos(ia));
                if l_i - r_k <= 0.0 {
                    continue;
                }
                let (r_ii, r_ji) = if g_first { (r_g, r_k) } else { (r_k, r_g) };
                let (f_x, f_y, f_z, ..) = f_atom_grad_hess(d, sp, r_ii, r_ji, r_ij, l_i);
                let f_g = if g_first { f_x } else { f_y };
                for c in 0..3 {
                    out[c] += f_g * g_g[c] + f_z * g_ij_g[c];
                }
            }
        }
    }
    out
}

/// [`dtn_value_strain_move_delta`] for the `nv` hops of ONE electron (one ECP mesh group):
/// the base-position terms (`- strain(base)` and the cutoff-policy base value) are evaluated
/// once for the group instead of once per hop, and the metric of every other electron to each
/// atom (the non-moving e-N leg of the f channel) is tabulated once per group instead of per
/// (pair, atom). In the f channel, a term with either e-N leg at or past the cutoff is exactly
/// zero and is skipped, and the sum over the fixed (non-moving) leg's power index is
/// contracted once per group ([`FixedLeg`]), so each hop sums only over the moving e-N and e-e
/// indices, value and gradient together. Same per-term formulas; agrees with the per-hop
/// function up to floating-point reassociation.
pub fn dtn_value_strain_move_delta_batch(
    d: &crate::dtn::JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_positions: &[[f64; 3]],
) -> Vec<Array2<f64>> {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = d.positions.nrows();
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let ci = d.cell.as_ref().map(|cell| CellInfo::new(&cell.view()));
    // Only the distance and its strain derivative `dd` enter the value strain.
    let metric = |disp: [f64; 3]| -> DistStrain {
        let u = match ci.as_ref() {
            Some(ci) => min_image_disp(disp, ci),
            None => disp,
        };
        let d = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let mut dd = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                dd[a][b] = u[a] * u[b] / d;
            }
        }
        DistStrain { d, dd }
    };
    let atom = |ia: usize| {
        [
            d.positions[[ia, 0]],
            d.positions[[ia, 1]],
            d.positions[[ia, 2]],
        ]
    };
    let g_idx = if is_up { li } else { n_up + li };
    let r_old = elec(g_idx);
    let same_spin = |e: usize| (e < n_up) == (g_idx < n_up);
    // radial coefficient rows, copied once per group (u: same / opposite spin; chi: species)
    let rows = |m: &Array2<f64>| -> Vec<Vec<f64>> { m.outer_iter().map(|r| r.to_vec()).collect() };
    let (alpha, alpha_policy) = (rows(&d.alpha), rows(&d.alpha_policy));
    let (beta, beta_policy) = (rows(&d.beta), rows(&d.beta_policy));
    // e-N metric of every other electron (the f channel's non-moving leg), once per group.
    let en: Vec<Vec<DistStrain>> = if d.use_f {
        (0..n_e)
            .map(|k| {
                if k == g_idx {
                    Vec::new()
                } else {
                    let rk = elec(k);
                    (0..n_atom)
                        .map(|ia| {
                            let p = atom(ia);
                            metric([rk[0] - p[0], rk[1] - p[1], rk[2] - p[2]])
                        })
                        .collect()
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    // f channel: every (other electron k, atom I) term with the k-I leg inside the cutoff,
    // with the polynomial sum over the FIXED leg's power index contracted once per group
    // (`FixedLeg`); a term with either e-N leg at or past `l_f` is exactly zero.
    let legs: Vec<FixedLeg> = if d.use_f {
        let mut v = Vec::new();
        for (k, en_k) in en[..n_e].iter().enumerate() {
            if k == g_idx {
                continue;
            }
            for (ia, en_k_ia) in en_k[..n_atom].iter().enumerate() {
                let sp = d.species_index[ia];
                let rf = en_k_ia.d;
                if rf >= d.l_f[sp] {
                    continue;
                }
                v.push(FixedLeg::new(d, sp, k, ia, rf, g_idx < k));
            }
        }
        v
    } else {
        Vec::new()
    };
    // f-channel strain terms (sign-weighted into `out`) and the cutoff-policy f value at `g_pos`.
    let f_terms = |g_pos: [f64; 3], gm: &[DistStrain], sign: f64, out: &mut Array2<f64>| -> f64 {
        let mut policy = 0.0;
        for leg in &legs {
            let sm_g = &gm[leg.ia];
            let l_i = d.l_f[leg.sp];
            if sm_g.d >= l_i {
                continue;
            }
            let rk = elec(leg.k);
            let disp = if leg.g_first {
                [g_pos[0] - rk[0], g_pos[1] - rk[1], g_pos[2] - rk[2]]
            } else {
                [rk[0] - g_pos[0], rk[1] - g_pos[1], rk[2] - g_pos[2]]
            };
            let sm_ij = metric(disp);
            let sm_k = &en[leg.k][leg.ia];
            let (fg, ff, fz, pv) = leg.eval(d, sm_g.d, sm_ij.d, l_i);
            // (fx, fy) in the global (lower, higher) electron order.
            let (fx, fy, sm_i, sm_j) = if leg.g_first {
                (fg, ff, sm_g, sm_k)
            } else {
                (ff, fg, sm_k, sm_g)
            };
            for a in 0..3 {
                for b in 0..3 {
                    out[[a, b]] +=
                        sign * (fx * sm_i.dd[a][b] + fy * sm_j.dd[a][b] + fz * sm_ij.dd[a][b]);
                }
            }
            policy += pv;
        }
        policy
    };
    let accumulate = |g_pos: [f64; 3], sign: f64, out: &mut Array2<f64>| -> f64 {
        let gm: Vec<DistStrain> = (0..n_atom)
            .map(|ia| {
                let p = atom(ia);
                metric([g_pos[0] - p[0], g_pos[1] - p[1], g_pos[2] - p[2]])
            })
            .collect();
        if d.use_chi {
            for (ia, gm_ia) in gm[..n_atom].iter().enumerate() {
                let sp = d.species_index[ia];
                let sm = gm_ia;
                let (h1, _, _) = crate::forces_dtn::h_derivs(d.l_chi[sp], &beta[sp], sm.d, d.c);
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += sign * h1 * sm.dd[a][b];
                    }
                }
            }
        }
        if d.use_u {
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let ch = if same_spin(k) { 0 } else { 1 };
                let rk = elec(k);
                let sm = metric([g_pos[0] - rk[0], g_pos[1] - rk[1], g_pos[2] - rk[2]]);
                let (h1, _, _) = crate::forces_dtn::h_derivs(d.l_u, &alpha[ch], sm.d, d.c);
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += sign * h1 * sm.dd[a][b];
                    }
                }
            }
        }
        f_terms(g_pos, &gm, sign, out)
    };
    // `dtn_cutoff_policy_one_electron_value` (same terms), reusing the e-N distance tables.
    let policy = |g_pos: [f64; 3]| -> f64 {
        let gm: Vec<f64> = (0..n_atom)
            .map(|ia| {
                let p = atom(ia);
                metric([g_pos[0] - p[0], g_pos[1] - p[1], g_pos[2] - p[2]]).d
            })
            .collect();
        let mut value = 0.0;
        if d.use_u {
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let ch = if same_spin(k) { 0 } else { 1 };
                let rk = elec(k);
                let r = metric([g_pos[0] - rk[0], g_pos[1] - rk[1], g_pos[2] - rk[2]]).d;
                value += dtn_policy_radial(d.l_u, &alpha[ch], &alpha_policy[ch], r, d.c).0;
            }
        }
        if d.use_chi {
            for (ia, &r) in gm.iter().enumerate() {
                let sp = d.species_index[ia];
                value += dtn_policy_radial(d.l_chi[sp], &beta[sp], &beta_policy[sp], r, d.c).0;
            }
        }
        value
    };
    let mut old = Array2::<f64>::zeros((3, 3));
    let f_policy_old = accumulate(r_old, -1.0, &mut old);
    let policy_old = if d.cell.is_some() {
        policy(r_old) + f_policy_old
    } else {
        0.0
    };
    new_positions
        .iter()
        .map(|&p| {
            let mut out = Array2::<f64>::zeros((3, 3));
            let f_policy = accumulate(p, 1.0, &mut out);
            out += &old;
            if d.cell.is_some() {
                let delta_policy = policy(p) + f_policy - policy_old;
                for axis in 0..3 {
                    out[[axis, axis]] += delta_policy / 3.0;
                }
            }
            out
        })
        .collect()
}

/// `dT_DTN/deps` (3x3) for the DTN u (e-e) + chi (e-N) channels over the Williamson
/// SMOOTH periodic distance, kinetic cotangent `cot=gJ+gD`. DTN analog of
/// `jastrow_kstress`, radial `h=phi*Sumc r^k` via `forces_dtn::h_derivs`. The f-channel
/// (e-e-N 3-body) is included when `d.use_f` (see `dtn_kstress_w`).
pub fn dtn_kstress(
    d: &crate::dtn::JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    dtn_kstress_w(d, cot, r_up, r_dn, None, out)
}

/// Per-electron weighted `dtn_kstress` (S' masked DTN kinetic; u+chi channels).
/// Same masking rule as [`jastrow_kstress_w`]: chi (e-N) by `w_e`, u (e-e) shared
/// lap by `(w_i+w_j)/2` and each drift by its own weight. The f-channel (e-e-N)
/// is implemented below under `if d.use_f`, with the same per-electron drift weights.
#[allow(clippy::too_many_arguments)]
pub fn dtn_kstress_w(
    d: &crate::dtn::JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    use crate::forces_dtn::h_derivs;
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let wgt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let cell = d.cell.as_ref().expect("PBC DTN cell").view();
    let ci = CellInfo::new(&cell);
    let dot = |e: usize, g: &[f64; 3]| cot[[e, 0]] * g[0] + cot[[e, 1]] * g[1] + cot[[e, 2]] * g[2];
    // chi (e-N): like J1
    if d.use_chi {
        let n_atom = d.positions.nrows();
        for e in 0..n_e {
            let re = elec(e);
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let lchi = d.l_chi[sp];
                let coef: Vec<f64> = d.beta.row(sp).to_vec();
                let disp = [
                    re[0] - d.positions[[ia, 0]],
                    re[1] - d.positions[[ia, 1]],
                    re[2] - d.positions[[ia, 2]],
                ];
                let sm = min_image_strain(disp, &ci);
                let (h1, h2, h3) = h_derivs(lchi, &coef, sm.d, d.c);
                let cdg = dot(e, &sm.gd);
                let we = wgt(e);
                for a in 0..3 {
                    for b in 0..3 {
                        let mut dgj = h2 * sm.dd[a][b] * cdg;
                        for i in 0..3 {
                            dgj += h1 * cot[[e, i]] * sm.dgd[i][a][b];
                        }
                        let dlap = h3 * sm.dd[a][b] * sm.q
                            + h2 * sm.dq[a][b]
                            + h2 * sm.dd[a][b] * sm.l
                            + h1 * sm.dl[a][b];
                        out[[a, b]] += we * (-0.5 * dlap - dgj);
                    }
                }
            }
        }
    }
    // u (e-e): like J2, spin-channel ch
    if d.use_u {
        let lu = d.l_u;
        let spin = |e: usize| -> usize {
            if e < n_up {
                0
            } else {
                1
            }
        };
        for i in 0..n_e {
            for j in (i + 1)..n_e {
                let ch = if spin(i) == spin(j) { 0 } else { 1 };
                let coef: Vec<f64> = d.alpha.row(ch).to_vec();
                let ri = elec(i);
                let rj = elec(j);
                let sm = min_image_strain([ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]], &ci);
                let (h1, h2, h3) = h_derivs(lu, &coef, sm.d, d.c);
                let cdi = dot(i, &sm.gd);
                let cdj = dot(j, &sm.gd);
                let (wi, wj) = (wgt(i), wgt(j));
                for a in 0..3 {
                    for b in 0..3 {
                        let mut dgi = h2 * sm.dd[a][b] * cdi;
                        let mut dgj = h2 * sm.dd[a][b] * cdj;
                        for k in 0..3 {
                            dgi += h1 * cot[[i, k]] * sm.dgd[k][a][b];
                            dgj += h1 * cot[[j, k]] * sm.dgd[k][a][b];
                        }
                        let dlap = h3 * sm.dd[a][b] * sm.q
                            + h2 * sm.dq[a][b]
                            + h2 * sm.dd[a][b] * sm.l
                            + h1 * sm.dl[a][b];
                        out[[a, b]] += -0.5 * (wi + wj) * dlap - wi * dgi + wj * dgj;
                    }
                }
            }
        }
    }
    // f-channel (e-e-N 3-body): kinetic stress dT_f/deps (per pair i<j x atom I). The e-N
    // strain geometry is formed once per (electron, atom); a triple with `r_iI >= l` or
    // `r_jI >= l` has every cutoff derivative zero (`(l - r)^c` factor) and is skipped.
    if d.use_f {
        let n_atom = d.positions.nrows();
        let en: Vec<SmoothStrain> = (0..n_e)
            .flat_map(|e| {
                let re = elec(e);
                (0..n_atom)
                    .map(|ia| {
                        min_image_strain(
                            [
                                re[0] - d.positions[[ia, 0]],
                                re[1] - d.positions[[ia, 1]],
                                re[2] - d.positions[[ia, 2]],
                            ],
                            &ci,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for i in 0..n_e {
            let ri = elec(i);
            for j in (i + 1)..n_e {
                let rj = elec(j);
                let sz = min_image_strain([ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]], &ci);
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let (sx, sy) = (&en[i * n_atom + ia], &en[j * n_atom + ia]);
                    if sx.d >= d.l_f[sp] || sy.d >= d.l_f[sp] {
                        continue;
                    }
                    let fd = crate::dtn::f_atom_derivs3(d, sp, sx.d, sy.d, sz.d, d.l_f[sp]);
                    // geometric scalars (g_hat_z is electron-i's gradr_ij; electron-j's is -g_hat_z)
                    let sxz: f64 = (0..3).map(|c| sx.gd[c] * sz.gd[c]).sum();
                    let syz: f64 = (0..3).map(|c| sy.gd[c] * sz.gd[c]).sum();
                    let cix: f64 = (0..3).map(|c| cot[[i, c]] * sx.gd[c]).sum();
                    let ciz: f64 = (0..3).map(|c| cot[[i, c]] * sz.gd[c]).sum();
                    let cjy: f64 = (0..3).map(|c| cot[[j, c]] * sy.gd[c]).sum();
                    let cjz: f64 = (0..3).map(|c| cot[[j, c]] * sz.gd[c]).sum();
                    let (wi, wj) = (wgt(i), wgt(j));
                    for a in 0..3 {
                        for b in 0..3 {
                            let (dx, dy, dz) = (sx.dd[a][b], sy.dd[a][b], sz.dd[a][b]);
                            // strained f-partials (chain rule over the three distances)
                            let dfx = fd.fxx * dx + fd.fxy * dy + fd.fxz * dz;
                            let dfy = fd.fxy * dx + fd.fyy * dy + fd.fyz * dz;
                            let dfz = fd.fxz * dx + fd.fyz * dy + fd.fzz * dz;
                            let dfxx = fd.fxxx * dx + fd.fxxy * dy + fd.fxxz * dz;
                            let dfyy = fd.fxyy * dx + fd.fyyy * dy + fd.fyyz * dz;
                            let dfzz = fd.fxzz * dx + fd.fyzz * dy + fd.fzzz * dz;
                            let dfxz = fd.fxxz * dx + fd.fxyz * dy + fd.fxzz * dz;
                            let dfyz = fd.fxyz * dx + fd.fyyz * dy + fd.fyzz * dz;
                            // strained geometric scalars (q const -> dq=0)
                            let dsxz: f64 = (0..3)
                                .map(|c| sx.dgd[c][a][b] * sz.gd[c] + sx.gd[c] * sz.dgd[c][a][b])
                                .sum();
                            let dsyz: f64 = (0..3)
                                .map(|c| sy.dgd[c][a][b] * sz.gd[c] + sy.gd[c] * sz.dgd[c][a][b])
                                .sum();
                            let dcix: f64 = (0..3).map(|c| cot[[i, c]] * sx.dgd[c][a][b]).sum();
                            let dciz: f64 = (0..3).map(|c| cot[[i, c]] * sz.dgd[c][a][b]).sum();
                            let dcjy: f64 = (0..3).map(|c| cot[[j, c]] * sy.dgd[c][a][b]).sum();
                            let dcjz: f64 = (0..3).map(|c| cot[[j, c]] * sz.dgd[c][a][b]).sum();
                            // d(grad^2_i f), d(grad^2_j f)  (q_x=q_y=q_z=1, dq=0)
                            let dlap_i = dfxx
                                + dfzz
                                + 2.0 * (dfxz * sxz + fd.fxz * dsxz)
                                + dfx * sx.l
                                + fd.fx * sx.dl[a][b]
                                + dfz * sz.l
                                + fd.fz * sz.dl[a][b];
                            let dlap_j = dfyy + dfzz - 2.0 * (dfyz * syz + fd.fyz * dsyz)
                                + dfy * sy.l
                                + fd.fy * sy.dl[a][b]
                                + dfz * sz.l
                                + fd.fz * sz.dl[a][b];
                            // d(drift_i*grad_i f) and d(drift_j*grad_j f)  (grad_j z = -g_hat_z)
                            let dcross_i = dfx * cix + fd.fx * dcix + dfz * ciz + fd.fz * dciz;
                            let dcross_j = dfy * cjy + fd.fy * dcjy - dfz * cjz - fd.fz * dcjz;
                            out[[a, b]] +=
                                -0.5 * (wi * dlap_i + wj * dlap_j) - wi * dcross_i - wj * dcross_j;
                        }
                    }
                }
            }
        }
    }
    if d.cell.is_some() {
        let (_, policy_grad, policy_lap) = dtn_cutoff_policy_fields(d, r_up, r_dn);
        let mut policy_kinetic = 0.0;
        for e in 0..n_e {
            let we = wgt(e);
            policy_kinetic -= 0.5 * we * policy_lap[e];
            for axis in 0..3 {
                policy_kinetic -= we * cot[[e, axis]] * policy_grad[[e, axis]];
            }
        }
        for axis in 0..3 {
            out[[axis, axis]] += policy_kinetic / 3.0;
        }
    }
}

/// `dT_J/deps` (3x3) for J1 + J2: smooth-distance lattice strain with the kinetic
/// cotangent `cot=gJ+gD` for J1/J2.
#[allow(clippy::too_many_arguments)]
pub fn jastrow_kstress(
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    jastrow_kstress_w(j1, j2, cot, r_up, r_dn, None, out)
}

/// Per-electron weighted `jastrow_kstress` (S' masked kinetic): the single-electron
/// (J1) contribution scales by `w_e`, and the J2 pair `(i,j)` splits -- the shared
/// laplacian by `(w_i+w_j)/2` (each electron owns half of grad^2J2) and each drift term
/// by its own electron's weight. `w=None` => all ones (= [`jastrow_kstress`]).
#[allow(clippy::too_many_arguments)]
pub fn jastrow_kstress_w(
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let wgt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let dot = |e: usize, g: &[f64; 3]| cot[[e, 0]] * g[0] + cot[[e, 1]] * g[1] + cot[[e, 2]] * g[2];
    if let Some(j1) = j1 {
        let cell = j1.structure_data.cell.as_ref().expect("PBC J1 cell").view();
        let metric = crate::jastrow_factor::CellInfo::new(&cell);
        let r_atoms = j1.structure_data.positions.view();
        let n_atom = j1.structure_data.n_atoms();
        let p = j1.jastrow_1b_param;
        let is_exp = j1.jastrow_1b_type == "exp";
        for e in 0..n_e {
            let re = elec(e);
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
                let sm = smooth_strain(disp, &metric);
                let (fp, fpp, fppp) = if is_exp {
                    let a = p * c;
                    let ex = (-a * sm.d).exp();
                    (
                        pref * c / 2.0 * ex,
                        -pref * p * c * c / 2.0 * ex,
                        pref * p * p * c.powi(3) / 2.0 * ex,
                    )
                } else {
                    let bb = p * c;
                    let u = 1.0 + bb * sm.d;
                    (
                        pref / (2.0 * u * u),
                        -pref * bb / u.powi(3),
                        3.0 * pref * bb * bb / u.powi(4),
                    )
                };
                let cdg = dot(e, &sm.gd);
                let we = wgt(e);
                for a in 0..3 {
                    for b in 0..3 {
                        let mut dgj = fpp * sm.dd[a][b] * cdg;
                        for i in 0..3 {
                            dgj += fp * cot[[e, i]] * sm.dgd[i][a][b];
                        }
                        let dlap = fppp * sm.dd[a][b] * sm.q
                            + fpp * sm.dq[a][b]
                            + fpp * sm.dd[a][b] * sm.l
                            + fp * sm.dl[a][b];
                        out[[a, b]] += we * (-0.5 * dlap - dgj);
                    }
                }
            }
        }
    }
    if let Some(j2) = j2 {
        // Own-cell metric: Williamson smooth strain when the J2 carries a cell; a
        // cell-less J2 is Euclidean (no lattice metric to strain, so this block reduces
        // to the position virial) instead of panicking on the absent cell.
        let cm = j2
            .cell
            .as_ref()
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
        let p = j2.jastrow_2b_param;
        let is_exp = j2.jastrow_2b_type == "exp";
        for i in 0..n_e {
            for jj in (i + 1)..n_e {
                let ri = elec(i);
                let rj = elec(jj);
                let disp = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
                let sm = match &cm {
                    Some(metric) => smooth_strain(disp, metric),
                    None => euclidean_strain(disp),
                };
                let (hp, hpp, hppp) = if is_exp {
                    let ex = (-p * sm.d).exp();
                    (0.5 * ex, -0.5 * p * ex, 0.5 * p * p * ex)
                } else {
                    let u = 1.0 + p * sm.d;
                    (1.0 / (2.0 * u * u), -p / u.powi(3), 3.0 * p * p / u.powi(4))
                };
                let cdg_i = dot(i, &sm.gd);
                let cdg_j = dot(jj, &sm.gd);
                let (wi, wj) = (wgt(i), wgt(jj));
                for a in 0..3 {
                    for b in 0..3 {
                        let mut dgj_i = hpp * sm.dd[a][b] * cdg_i;
                        let mut dgj_j = hpp * sm.dd[a][b] * cdg_j;
                        for k in 0..3 {
                            dgj_i += hp * cot[[i, k]] * sm.dgd[k][a][b];
                            dgj_j += hp * cot[[jj, k]] * sm.dgd[k][a][b];
                        }
                        let dlap = hppp * sm.dd[a][b] * sm.q
                            + hpp * sm.dq[a][b]
                            + hpp * sm.dd[a][b] * sm.l
                            + hp * sm.dl[a][b];
                        // lapJ2[i]+lapJ2[j] = 2*dlap -> -1/2*2 = -1; the j-electron's
                        // gd has opposite sign so its dgJ enters with +. S' mask:
                        // each electron owns half the shared lap (-> (wi+wj)/2) and
                        // its own drift term.
                        out[[a, b]] += -0.5 * (wi + wj) * dlap - wi * dgj_i + wj * dgj_j;
                    }
                }
            }
        }
    }
}

pub(crate) fn k_strict_lower(n: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            m[[i, j]] = 1.0;
        }
    }
    m
}

/// PBC J3 Jastrow-side kinetic stress.
/// J3 cotangents (bar_A value, bar_gO gradient, bar_lO=-1/2 Laplacian) on the
/// cart-pgto J3 basis (sphe=`t_mol`*cart), fed through the value+lap+gou field
/// strain. `bg_*` are `-(gJ+gD)` per spin `(N,3)`.
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_kstress(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    j3_pbc_kstress_w(
        cart_j3,
        t_mol,
        j_matrix,
        r_up,
        r_dn,
        bg_up,
        bg_dn,
        None,
        cell,
        home_atom,
        kernel_atom,
        positions,
        None,
        true,
        out,
    )
}

/// Vertically stack the GTO block over the PW block in `gto_first` order.
fn stack_rows(gto: &Array2<f64>, pw: &Array2<f64>, gto_first: bool) -> Array2<f64> {
    let (top, bot) = if gto_first { (gto, pw) } else { (pw, gto) };
    numpy::ndarray::concatenate(Axis(0), &[top.view(), bot.view()]).unwrap()
}

/// Per-electron **weighted** [`j3_pbc_kstress`] (S' masked J3 kinetic). J3 is
/// bilinear in the AO value `A`, so electron `k`'s kinetic localizes to AO
/// column `k`; weighting the `side()` cotangents (bcoef / bgo / blo) by `w_k`
/// and propagating through the M-coupled `bar_a` yields `d(Sum_k w_k*kc_k^J3)/deps`.
/// `w=None` => all ones. `w` is spin-concatenated `[up.. , dn..]`.
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_kstress_w(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    w: Option<&[f64]>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    pw: Option<J3Extra<'_>>,
    gto_first: bool,
    out: &mut Array2<f64>,
) {
    let n_up_w = r_up.nrows();
    let w_up: Option<Vec<f64>> = w.map(|w| w[0..n_up_w].to_vec());
    let w_dn: Option<Vec<f64>> = w.map(|w| w[n_up_w..].to_vec());
    // COMPOSITE (pgto(+)pw) J3: append the PW child's value/grad/lap to the AO basis in
    // gto_first order (j_matrix couples all rows through the cotangents). PW value is
    // strain-INVARIANT (fractional coords fixed under affine strain), so only its lap
    // (`pw_lap_stress`) + gradient (`pw_gou_stress`) fields respond; the pGTO block goes
    // through `pgto_kstress_side` as before. `pw = None` => byte-identical single leaf. A
    // B-spline child (`J3Extra::Bspline`) is strain-invariant in value too (see below).
    let num_gto = t_mol.nrows();
    let num_pw = pw.map_or(0, |p| p.num_ao());
    let gto0 = if gto_first { 0 } else { num_pw };
    let pw0 = if gto_first { num_gto } else { 0 };
    let basis = |r: &ArrayView2<f64>| -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        let dv = pgto_derivs(cart_j3, r, cell, true, true, false);
        let ag = t_mol.dot(dv.val.as_ref().unwrap());
        let g = &dv.grad;
        let gog = [t_mol.dot(&g[0]), t_mol.dot(&g[1]), t_mol.dot(&g[2])];
        let log = t_mol.dot(dv.lap.as_ref().unwrap());
        match pw {
            None => (ag, gog, log),
            Some(pw) => {
                let (pv, pgx, pgy, pgz, pl) = pw.vgl(r);
                (
                    stack_rows(&ag, &pv, gto_first),
                    [
                        stack_rows(&gog[0], &pgx, gto_first),
                        stack_rows(&gog[1], &pgy, gto_first),
                        stack_rows(&gog[2], &pgz, gto_first),
                    ],
                    stack_rows(&log, &pl, gto_first),
                )
            }
        }
    };
    let (a_up, go_u, lo_u) = basis(r_up);
    let (a_dn, go_d, lo_d) = basis(r_dn);
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
    // side_cot -> (bar_coefA, bar_gO[3], bar_lO). `wcol[k]` weights electron k's
    // kinetic seed (S' mask); applied to all three cotangents (bg-term, lap, gou).
    let side = |coef: &Array2<f64>,
                bg: &Array2<f64>,
                go: &[Array2<f64>; 3],
                lo: &Array2<f64>,
                wcol: &Option<Vec<f64>>|
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
                let wk = wcol.as_ref().map_or(1.0, |v| v[k]);
                let mut acc = 0.0;
                for d in 0..3 {
                    acc += bg[[k, d]] * go[d][[a, k]];
                    bgo[d][[a, k]] = wk * bg[[k, d]] * coef[[a, k]];
                }
                bcoef[[a, k]] = wk * (acc + (-0.5) * lo[[a, k]]);
                blo[[a, k]] = wk * (-0.5) * coef[[a, k]];
            }
        }
        (bcoef, bgo, blo)
    };
    let (bcoef_u, bgo_u, blo_u) = side(&coef_up, bg_up, &go_u, &lo_u, &w_up);
    let (bcoef_d, bgo_d, blo_d) = side(&coef_dn, bg_dn, &go_d, &lo_d, &w_dn);
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
    // GTO block: value (bar_A) + lap (bar_lO) + gou (bar_gO) field strain through the cart
    // J3 leaf. Slice the composite cotangents to the gto row range [gto0, gto0+num_gto).
    let g_rows = |x: &Array2<f64>| {
        x.slice(numpy::ndarray::s![gto0..gto0 + num_gto, ..])
            .to_owned()
    };
    let g_rows3 = |x: &[Array2<f64>; 3]| [g_rows(&x[0]), g_rows(&x[1]), g_rows(&x[2])];
    pgto_kstress_side(
        cart_j3,
        t_mol,
        &g_rows(&bar_a_up),
        &g_rows(&blo_u),
        Some(&g_rows3(&bgo_u)),
        r_up,
        cell,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
    pgto_kstress_side(
        cart_j3,
        t_mol,
        &g_rows(&bar_a_dn),
        &g_rows(&blo_d),
        Some(&g_rows3(&bgo_d)),
        r_dn,
        cell,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
    // PW block: value strain-invariant (skip bar_A); lap field `2*kappa_c*kappa_d*val` and
    // gradient field `d(grad_d)/deps` respond via the reciprocal-vector metric response.
    // A B-spline child (pGTO(+)B-spline) takes the same lap/grad field responses through its
    // t-space metric (`bspline_ao_kstress`, the AO-rank core of the determinant
    // `bspline_kstress_side`).
    if let Some(extra) = pw {
        let p_rows = |x: &Array2<f64>| {
            x.slice(numpy::ndarray::s![pw0..pw0 + num_pw, ..])
                .to_owned()
        };
        let sides = [(&blo_u, &bgo_u, r_up.view()), (&blo_d, &bgo_d, r_dn.view())];
        for (blo, bgo, r) in sides {
            let r = &r;
            let cm_gou = [p_rows(&bgo[0]), p_rows(&bgo[1]), p_rows(&bgo[2])];
            match extra {
                J3Extra::Pw(pw) => {
                    pw_lap_stress(pw, &p_rows(blo), r, out);
                    pw_gou_stress(pw, &cm_gou, r, out);
                }
                J3Extra::Bspline(bs) => {
                    bspline_ao_kstress(bs, &p_rows(blo), Some(&cm_gou), r, out);
                }
            }
        }
    }
}

/// PBC J3 **value** strain `d(lnJ3)/deps` (3x3). The J3 value cotangent on the AO
/// value `A` is `coef = dJ3/dA` (identical to the base term inside
/// [`j3_pbc_kstress`] before the grad/lap kinetic mixing), fed through the AO
/// **value** field-strain [`pgto_det_stress_side`]. Building block for the S'
/// strained-mesh sigma_PP and the ECP-nl mesh Pulay with J3.
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_value_strain(
    cart_j3: &AosGtoData,
    t_mol: &Array2<f64>,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    pw: Option<J3Extra<'_>>,
    gto_first: bool,
    out: &mut Array2<f64>,
) {
    let num_gto = t_mol.nrows();
    let num_pw = pw.map_or(0, |p| p.num_ao());
    let gto0 = if gto_first { 0 } else { num_pw };
    let basis = |r: &ArrayView2<f64>| {
        let gto = t_mol.dot(&pgto_deriv(cart_j3, r, cell, (0, 0, 0)));
        match pw {
            None => gto,
            Some(pw) => stack_rows(&gto, &pw.value(r), gto_first),
        }
    };
    let a_up = basis(r_up);
    let a_dn = basis(r_dn);
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
    // coef = dJ3/dA (value cotangent), 1:1 the `coef` closure in j3_pbc_kstress.
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
    let gto_rows = |x: &Array2<f64>| {
        x.slice(numpy::ndarray::s![gto0..gto0 + num_gto, ..])
            .to_owned()
    };
    // Plane-wave AO values are invariant under the affine strain convention, but their
    // rows still change the J3 cotangent on the atom-centred GTO rows through j_matrix.
    pgto_det_stress_side(
        cart_j3,
        t_mol,
        &gto_rows(&coef_up),
        r_up,
        cell,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
    pgto_det_stress_side(
        cart_j3,
        t_mol,
        &gto_rows(&coef_dn),
        r_dn,
        cell,
        home_atom,
        kernel_atom,
        positions,
        out,
    );
}

/// Positive full-Gram distance and all nine affine strain components.
fn dd_dstrain(disp: [f64; 3], ci: &crate::jastrow_factor::CellInfo) -> (f64, [[f64; 3]; 3]) {
    let raw = crate::jastrow_factor::positive_geometry(disp, ci);
    let d = raw.square.sqrt();
    let mut dd = [[0.0; 3]; 3];
    for (a, dd_a) in dd.iter_mut().enumerate() {
        for (b, dd_ab) in dd_a.iter_mut().enumerate() {
            let mut tensor = raw.cart[a] * raw.cart[b];
            for k in 0..3 {
                tensor += raw.z[k] * raw.z[k] * ci.cell[k][a] * ci.cell[k][b]
                    / std::f64::consts::PI.powi(2);
            }
            *dd_ab = tensor / d;
        }
    }
    (d, dd)
}

/// `d(J1+J2)/deps` (3x3) PBC smooth-distance lattice-length stress. J3 is not part
/// of this function: callers add it through `j3_pbc_value_strain`.
pub fn jastrow_strain(
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
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
    if let Some(j1) = j1 {
        let cell = j1.structure_data.cell.as_ref().expect("PBC J1 cell").view();
        let metric = crate::jastrow_factor::CellInfo::new(&cell);
        let r_atoms = j1.structure_data.positions.view();
        let n_atom = j1.structure_data.n_atoms();
        let p = j1.jastrow_1b_param;
        let is_exp = j1.jastrow_1b_type == "exp";
        for e in 0..n_e {
            let re = elec(e);
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
                let (d, dd) = dd_dstrain(disp, &metric);
                let fp = if is_exp {
                    pref * c / 2.0 * (-p * c * d).exp()
                } else {
                    pref / (2.0 * (1.0 + p * c * d).powi(2))
                };
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += fp * dd[a][b];
                    }
                }
            }
        }
    }
    if let Some(j2) = j2 {
        // Own-cell metric: Williamson smooth strain when the J2 carries a cell; a
        // cell-less J2 is Euclidean (no lattice metric to strain, so this is the
        // position virial) instead of panicking on the absent cell.
        let cm = j2
            .cell
            .as_ref()
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
        let p = j2.jastrow_2b_param;
        let is_exp = j2.jastrow_2b_type == "exp";
        for i in 0..n_e {
            for j in (i + 1)..n_e {
                let ri = elec(i);
                let rj = elec(j);
                let disp = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
                let (d, dd) = match &cm {
                    Some(metric) => dd_dstrain(disp, metric),
                    None => {
                        let s = euclidean_strain(disp);
                        (s.d, s.dd)
                    }
                };
                let hp = if is_exp {
                    0.5 * (-p * d).exp()
                } else {
                    1.0 / (2.0 * (1.0 + p * d).powi(2))
                };
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += hp * dd[a][b];
                    }
                }
            }
        }
    }
}

/// [`jastrow_strain`] (J1 + J2 value strain) DELTA for the `nv` single-electron hops of
/// electron `li` (spin `is_up`) to `new_positions`: `strain(moved) - strain(base)`, restricted
/// to the terms that touch the moved electron (its J1 electron-nucleus terms and its J2 pairs,
/// each with the pair ordering and metric of [`jastrow_strain`]). The base-position terms are
/// evaluated once for the group. Same per-term formulas, so `base + delta` reproduces the
/// moved-config value strain up to floating-point reassociation, in O(N_atom + N_e) per hop.
pub fn jastrow_strain_move_delta_batch(
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_positions: &[[f64; 3]],
) -> Vec<Array2<f64>> {
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
    let g_idx = if is_up { li } else { n_up + li };
    let j1m = j1.map(|j1| {
        let cell = j1.structure_data.cell.as_ref().expect("PBC J1 cell").view();
        crate::jastrow_factor::CellInfo::new(&cell)
    });
    let j2m = j2.and_then(|j2| {
        j2.cell
            .as_ref()
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()))
    });
    // All J1/J2 terms of electron `g` at position `re`, with weight `sign`.
    let accumulate = |re: [f64; 3], sign: f64, out: &mut Array2<f64>| {
        if let (Some(j1), Some(metric)) = (j1, j1m.as_ref()) {
            let r_atoms = j1.structure_data.positions.view();
            let p = j1.jastrow_1b_param;
            let is_exp = j1.jastrow_1b_type == "exp";
            for ia in 0..j1.structure_data.n_atoms() {
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
                let (d, dd) = dd_dstrain(disp, metric);
                let fp = if is_exp {
                    pref * c / 2.0 * (-p * c * d).exp()
                } else {
                    pref / (2.0 * (1.0 + p * c * d).powi(2))
                };
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += sign * fp * dd[a][b];
                    }
                }
            }
        }
        if let Some(j2) = j2 {
            let p = j2.jastrow_2b_param;
            let is_exp = j2.jastrow_2b_type == "exp";
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let rk = elec(k);
                // `jastrow_strain` pair order: displacement of the lower electron index.
                let disp = if g_idx < k {
                    [re[0] - rk[0], re[1] - rk[1], re[2] - rk[2]]
                } else {
                    [rk[0] - re[0], rk[1] - re[1], rk[2] - re[2]]
                };
                let (d, dd) = match &j2m {
                    Some(metric) => dd_dstrain(disp, metric),
                    None => {
                        let s = euclidean_strain(disp);
                        (s.d, s.dd)
                    }
                };
                let hp = if is_exp {
                    0.5 * (-p * d).exp()
                } else {
                    1.0 / (2.0 * (1.0 + p * d).powi(2))
                };
                for a in 0..3 {
                    for b in 0..3 {
                        out[[a, b]] += sign * hp * dd[a][b];
                    }
                }
            }
        }
    };
    let mut old = Array2::<f64>::zeros((3, 3));
    accumulate(elec(g_idx), -1.0, &mut old);
    new_positions
        .iter()
        .map(|&p| {
            let mut out = old.clone();
            accumulate(p, 1.0, &mut out);
            out
        })
        .collect()
}

/// B-spline determinant KINETIC stress `dT_det/deps` (full 3x3, affine strain cell+electrons).
/// AFFINE-INVARIANCE route (1:1 `pbc_stress._bspline_kinetic_stress_contrib`): frac=r*inv_cell
/// is strain-invariant => t-space gt/Ht are invariant and the VALUE side is 0; only the metric
/// (inv, N) responds. `SL[k,l]=Sum_bi CM_lap[b,i]*Ht[i,b,k,l]`, `SLn=SL*N_k N_l`,
/// `lap_term=-inv*(SLn+SLnT)*invT`; `SG2[d,k]=Sum_bi CM_gou[d,b,i]*gt[i,b,k]`,
/// `gou_term[a,b]=-Sum_k inv[a,k]*SG2[b,k]*N_k`. CM_* = CT*bar_* (geminal_reverse kinetic
/// cotangents -> AO rank; jnone bar_gou=0). This is the FULL dT_det/deps (NOT the cell-only
/// `grad_t_bspline_dcell`). The Jastrow side (DTN/J1/J2 smooth-distance strain) is added separately.
#[allow(clippy::too_many_arguments)]
pub fn bspline_kinetic_stress_det(
    gem: &GeminalData,
    jastrow: Option<&crate::data::JastrowData>,
    j3_orb: Option<&crate::determinant::OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    // Standalone (py-wrapper) entry: builds its own unit-weight kinetic reverse (fresh LU). The
    // VMC aggregate shares its g_inv-reused reverse cotangents by calling `bspline_kstress_side`
    // directly (see `gfmc_forces::vmc_grad_e_l_eps_pbc::kin_stress`).
    let (gj_up, gj_dn) = crate::param_grad_el::jastrow_grads(jastrow, j3_orb, j3_jm, r_up, r_dn);
    let rev = crate::param_grad_el::geminal_reverse(gem, r_up, r_dn, &gj_up, &gj_dn);
    let mut out = Array2::<f64>::zeros((3, 3));
    bspline_kstress_side(
        &gem.orb_up,
        &rev.bar_lap_up,
        Some(&rev.bar_gou),
        r_up,
        &mut out,
    );
    bspline_kstress_side(
        &gem.orb_dn,
        &rev.bar_lap_dn,
        Some(&rev.bar_godn),
        r_dn,
        &mut out,
    );
    out
}

/// One-side B-spline determinant KINETIC stress accumulation (mirrors `pgto_kstress_side` /
/// `pw_lap_stress`+`pw_gou_stress`): consumes the already-computed kinetic reverse cotangents
/// `bar_lap` / `bar_gou` (so the VMC aggregate can share the g_inv-reused reverse pass) and
/// adds this orbital's `dT_det/deps` metric response to `out`. `bar_gou=None` => jnone (skip the
/// gradient field). 1:1 `pbc_stress._bspline_kinetic_stress_contrib`.
pub fn bspline_kstress_side(
    orb: &crate::determinant::OrbData,
    bar_lap: &Array2<f64>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    use crate::determinant::matmul;
    let bs = orb
        .bspline
        .as_ref()
        .expect("bspline_kstress_side: not a B-spline leaf");
    // Cotangent -> B-spline AO rank via the B-spline MO block (the composite-safe slice; == the
    // full C for a pure B-spline). `None` MO => raw-AO passthrough.
    let block = orb.bspline_mo_block();
    let proj = |bar: &Array2<f64>| -> Array2<f64> {
        match &block {
            Some(c) => matmul(&c.t(), &bar.view()),
            None => bar.clone(),
        }
    };
    let cm_lap = proj(bar_lap);
    let has_gou = bar_gou.is_some_and(|bg| bg.iter().any(|m| m.iter().any(|&x| x != 0.0)));
    let cm_gou = if has_gou {
        let bg = bar_gou.unwrap();
        Some([proj(&bg[0]), proj(&bg[1]), proj(&bg[2])])
    } else {
        None
    };
    bspline_ao_kstress(bs, &cm_lap, cm_gou.as_ref(), r, out);
}

/// AO-rank B-spline KINETIC stress accumulation: the metric (inv, N) response of the
/// B-spline Laplacian and gradient fields for AO-rank cotangents `cm_lap` `(nb, N)` and
/// `cm_gou` `[3] (nb, N)` (`None` => no gradient field). frac = r*inv_cell is invariant
/// under the affine strain, so the t-space `gt`/`Ht` are invariant and the VALUE field has
/// no response. `SL[k,l]=Sum_bi CM_lap[b,i]*Ht[i,b,k,l]`, `SLn=SL*N_k N_l`,
/// `lap_term=-inv*(SLn+SLnT)*invT`; `SG2[d,k]=Sum_bi CM_gou[d,b,i]*gt[i,b,k]`,
/// `gou_term[a,b]=-Sum_k inv[a,k]*SG2[b,k]*N_k`. Shared by the determinant
/// [`bspline_kstress_side`] and the B-spline rows of a pGTO(+)B-spline J3 in
/// [`j3_pbc_kstress_w`].
pub fn bspline_ao_kstress(
    bs: &crate::bspline::AosBsplineData,
    cm_lap: &Array2<f64>,
    cm_gou: Option<&[Array2<f64>; 3]>,
    r: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    let (gt, ht) = bs.t_fields(r); // (N,nb,3),(N,nb,3,3)
    let inv = bs.inv_cell;
    let nf = bs.n_f;
    let nb = cm_lap.nrows();
    let ne = cm_lap.ncols();
    // SL[k,l] = Sum_{b,i} CM_lap[b,i]*Ht[i,b,k,l]; SLn = SL*N_k N_l
    let mut sln = [[0.0_f64; 3]; 3];
    for kk in 0..3 {
        for ll in 0..3 {
            let mut s = 0.0;
            for b in 0..nb {
                for i in 0..ne {
                    s += cm_lap[[b, i]] * ht[[i, b, kk, ll]];
                }
            }
            sln[kk][ll] = s * nf[kk] * nf[ll];
        }
    }
    // lap_term = -inv*(SLn+SLnT)*invT : [a,b] = -Sum_kl inv[a][k](SLn+SLnT)[k][l]inv[b][l]
    for a in 0..3 {
        for bb in 0..3 {
            let mut s = 0.0;
            for kk in 0..3 {
                for ll in 0..3 {
                    s += inv[a][kk] * (sln[kk][ll] + sln[ll][kk]) * inv[bb][ll];
                }
            }
            out[[a, bb]] -= s;
        }
    }
    if let Some(cm_gou) = cm_gou {
        // SG2[d,k] = Sum_{b,i} CM_gou[d][b,i]*gt[i,b,k]
        let mut sg2 = [[0.0_f64; 3]; 3];
        for d in 0..3 {
            for kk in 0..3 {
                let mut s = 0.0;
                for b in 0..nb {
                    for i in 0..ne {
                        s += cm_gou[d][[b, i]] * gt[[i, b, kk]];
                    }
                }
                sg2[d][kk] = s;
            }
        }
        // gou_term[a,b] = -Sum_k inv[a][k]*SG2[b][k]*N_k
        for a in 0..3 {
            for bb in 0..3 {
                let mut s = 0.0;
                for kk in 0..3 {
                    s += inv[a][kk] * sg2[bb][kk] * nf[kk];
                }
                out[[a, bb]] -= s;
            }
        }
    }
}

/// B-spline kinetic CELL-derivative `dT_det/dcell` along `dcell` (3x3), **electrons FIXED**.
/// `Sum_e [cm_orb*dval/dcell + cm_lap*dlap/dcell + Sum_d cm_gou_d*d(grad_d orb)/dcell]`, cm_* = CT*bar_*
/// (geminal_reverse kinetic cotangents -> AO rank; jnone bar_gou=0). Verified vs central FD of
/// compute_kinetic_energy with the cell perturbed and electrons fixed.
/// **NOTE -- this is ONE PART of the physical kinetic stress, not the full dT/deps.** The full
/// PBC stress strains the cell AND the electrons (r->Fr); dT/deps = this cell-field +
/// the electron virial Sum_e (dT/dr_e)(x)r_e. For B-spline the VALUE (ln|det|) is frac-invariant
/// under the simultaneous strain (dln|det|/deps~=0, FD 7e-7), so the cell-field and electron
/// virial of the value cancel -- but the kinetic (metric-dependent grad/grad^2) does NOT cancel. The
/// full dT_det/deps is [`bspline_kinetic_stress_det`] (the affine-invariance route); this
/// function alone is NOT the full kinetic stress.
#[allow(clippy::too_many_arguments)]
pub fn grad_t_bspline_dcell(
    gem: &GeminalData,
    jastrow: Option<&crate::data::JastrowData>,
    j3_orb: Option<&crate::determinant::OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    dcell: &[[f64; 3]; 3],
) -> f64 {
    use crate::determinant::matmul;
    let (gj_up, gj_dn) = crate::param_grad_el::jastrow_grads(jastrow, j3_orb, j3_jm, r_up, r_dn);
    let rev = crate::param_grad_el::geminal_reverse(gem, r_up, r_dn, &gj_up, &gj_dn);
    // Cotangent -> AO rank: cm = CT*bar (identity passthrough when no MO).
    let proj = |orb: &crate::determinant::OrbData, bar: &Array2<f64>| -> Array2<f64> {
        match &orb.mo_coefficients {
            Some(c) => matmul(&c.t(), &bar.view()),
            None => bar.clone(),
        }
    };
    let dot = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };
    let side = |orb: &crate::determinant::OrbData,
                bar_orb: &Array2<f64>,
                bar_lap: &Array2<f64>,
                bar_g: &[Array2<f64>; 3],
                r: &ArrayView2<f64>|
     -> f64 {
        let bs = orb
            .bspline
            .as_ref()
            .expect("grad_t_bspline_dcell: orbital is not a B-spline leaf");
        let (dv, dgx, dgy, dgz, dl) = bs.value_grad_lap_dcell(r, dcell);
        let cm_orb = proj(orb, bar_orb);
        let cm_lap = proj(orb, bar_lap);
        let mut s = dot(&cm_orb, &dv) + dot(&cm_lap, &dl);
        if bar_g.iter().any(|m| m.iter().any(|&x| x != 0.0)) {
            let cmg = [
                proj(orb, &bar_g[0]),
                proj(orb, &bar_g[1]),
                proj(orb, &bar_g[2]),
            ];
            s += dot(&cmg[0], &dgx) + dot(&cmg[1], &dgy) + dot(&cmg[2], &dgz);
        }
        s
    };
    side(
        &gem.orb_up,
        &rev.bar_orb_up,
        &rev.bar_lap_up,
        &rev.bar_gou,
        r_up,
    ) + side(
        &gem.orb_dn,
        &rev.bar_orb_dn,
        &rev.bar_lap_dn,
        &rev.bar_godn,
        r_dn,
    )
}

/// GFMC variant of `grad_v_ecp_nonlocal_strain`: NN=nn nearest-atom mesh (LRDMC
/// tmove). The quadrature directions stay fixed under the strain (each strained point
/// lies on the sphere of the strained electron-nucleus distance), so the rotation
/// average is the exact strain derivative of the angular integral; no grid-stretch
/// term. jnone (dtn=None). Returns the periodic NN ECP non-local stress
/// `d(Sum V_nl)/deps (3x3)`.
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_nonlocal_strain_gfmc(
    gem: &GeminalData,
    leaves_up: &[StressLeaf],
    leaves_dn: &[StressLeaf],
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    nmax: i32,
    nn: usize,
) -> Array2<f64> {
    let inv = inv3(cell);
    let nv = weights.len();
    let n_up = r_up.nrows();
    let (s0, l0) = det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view());
    let pulay_base = pbc_pulay_det(gem, leaves_up, leaves_dn, r_up, r_dn, cell, positions);
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    &inv,
                    nmax,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let mut out = Array2::<f64>::zeros((3, 3));
    for is_up in [true, false] {
        let nr = if is_up { n_up } else { r_dn.nrows() };
        for i in 0..nr {
            let re = if is_up {
                [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
            } else {
                [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
            };
            let near = nearest_set(re);
            for t in 0..nucleus_index.len() {
                let ia = nucleus_index[t] as usize;
                if !near.contains(&ia) {
                    continue;
                }
                let l = ang_mom[t];
                let (be, ce, pe) = (exponents[t], coefficients[t], powers[t]);
                let rel0 = [
                    positions[[ia, 0]] - re[0],
                    positions[[ia, 1]] - re[1],
                    positions[[ia, 2]] - re[2],
                ];
                let rel = min_image(rel0, cell, &inv, nmax);
                let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                let vl = ce * nrm.powf(pe - 2.0) * (-be * nrm * nrm).exp();
                let vlp = ce
                    * ((pe - 2.0) * nrm.powf(pe - 3.0) - 2.0 * be * nrm.powf(pe - 1.0))
                    * (-be * nrm * nrm).exp();
                for v in 0..nv {
                    let vg = [
                        grid_points[[v, 0]],
                        grid_points[[v, 1]],
                        grid_points[[v, 2]],
                    ];
                    let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                    let cth = -(rel[0] * vg[0] + rel[1] * vg[1] + rel[2] * vg[2]) / (nrm * vgn);
                    let (pl, dpl) = leg_d(l, cth);
                    let rp = [
                        re[0] + rel[0] + nrm * vg[0],
                        re[1] + rel[1] + nrm * vg[1],
                        re[2] + rel[2] + nrm * vg[2],
                    ];
                    let (mu, md) = if is_up {
                        let mut m = r_up.to_owned();
                        m[[i, 0]] = rp[0];
                        m[[i, 1]] = rp[1];
                        m[[i, 2]] = rp[2];
                        (m, r_dn.to_owned())
                    } else {
                        let mut m = r_dn.to_owned();
                        m[[i, 0]] = rp[0];
                        m[[i, 1]] = rp[1];
                        m[[i, 2]] = rp[2];
                        (r_up.to_owned(), m)
                    };
                    let (sm, lm) = det_sign_lnabs(
                        &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                    );
                    let ratio = sm * s0 * (lm - l0).exp();
                    let (gdu, gdd) = det_grad_field(gem, &mu.view(), &md.view());
                    let drift = if is_up {
                        [gdu[[i, 0]], gdu[[i, 1]], gdu[[i, 2]]]
                    } else {
                        [gdd[[i, 0]], gdd[[i, 1]], gdd[[i, 2]]]
                    };
                    let pulay_m = pbc_pulay_det(
                        gem,
                        leaves_up,
                        leaves_dn,
                        &mu.view(),
                        &md.view(),
                        cell,
                        positions,
                    );
                    let drift_vg = drift[0] * vg[0] + drift[1] * vg[1] + drift[2] * vg[2];
                    let a = vl * ((2 * l + 1) as f64) * pl * weights[v];
                    for c in 0..3 {
                        for d in 0..3 {
                            let dcth = -(vg[c] * rel[d]) / (nrm * vgn)
                                - cth * rel[c] * rel[d] / (nrm * nrm);
                            let da = ((2 * l + 1) as f64)
                                * weights[v]
                                * (vlp * pl * rel[c] * rel[d] / nrm + vl * dpl * dcth);
                            let mesh_wf = pulay_m[[c, d]] - drift[c] * nrm * vg[d]
                                + (rel[c] * rel[d] / nrm) * drift_vg;
                            let dln_ratio = mesh_wf - pulay_base[[c, d]];
                            out[[c, d]] += ratio * (da + a * dln_ratio);
                        }
                    }
                }
            }
        }
    }
    out
}
