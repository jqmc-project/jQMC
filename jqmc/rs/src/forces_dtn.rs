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

//! DTN Jastrow kinetic gradients `dT_DTN/dr` (electron) and `dT_DTN/dR` (atom),
//! Channels u (e-e, mirrors J2), chi
//! (e-N, mirrors J1) and f (e-e-n, 3-body `_dS_dself`). cot = gJ+gD held fixed.
//! Needs one cutoff/poly derivative order beyond the value (phi''',g''').
//!
//! Every displacement is min-imaged
//! when the DTN carries a cell (`d.cell = Some`, periodic), exactly as the DTN
//! VALUE/drift kernels (`dtn.rs::min_image_dist`) and jqmc's
//! `_dtn_scalar_dist_with_derivs` -- the hard `(L-r)^C` cutoff acts
//! on the TRUE min-image distance, and inside the cutoff the image index is
//! locally constant, so all derivative formulas keep their Euclidean form on the
//! min-imaged displacement. The RAW displacement agrees with jqmc only in-cell:
//! for a walker roamed > half a cell out it makes the native DTN force differ
//! from jqmc by O(0.1-1) (measured 7.3e-1 in the far-config jdtn HF force,
//! 2.9e-1 in the Pulay dR). `min_image_disp` is
//! bit-exact identity in-cell and `cell=None` (molecular) is untouched.

use numpy::ndarray::{Array2, ArrayView2};

use crate::dtn::JastrowDtnData;
use crate::jastrow_factor::{min_image_disp, CellInfo};

/// Displacement `a - b` (min-imaged under PBC) + its Euclidean norm.
#[inline]
fn dtn_disp(a: [f64; 3], b: [f64; 3], ci: Option<&CellInfo>) -> ([f64; 3], f64) {
    let x = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let u = match ci {
        Some(ci) => min_image_disp(x, ci),
        None => x,
    };
    (u, (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
}

/// `phi=max(L-r,0)^C` and `phi',phi'',phi'''` at scalar r (0 outside the cutoff).
fn phi_derivs(l: f64, r: f64, c: i32) -> (f64, f64, f64, f64) {
    let s = (l - r).max(0.0);
    if s <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let cf = c as f64;
    let phi = s.powi(c);
    let d1 = -cf * s.powi(c - 1);
    let d2 = cf * (cf - 1.0) * s.powi(c - 2);
    let d3 = -cf * (cf - 1.0) * (cf - 2.0) * s.powi(c - 3);
    (phi, d1, d2, d3)
}

/// `g=Sum_k coef[k] r^k` and `g',g'',g'''` at scalar r.
fn poly_derivs(coef: &[f64], r: f64) -> (f64, f64, f64, f64) {
    let (mut g, mut g1, mut g2, mut g3) = (0.0, 0.0, 0.0, 0.0);
    for (k, &a) in coef.iter().enumerate() {
        let kf = k as f64;
        g += a * r.powi(k as i32);
        if k >= 1 {
            g1 += a * kf * r.powi(k as i32 - 1);
        }
        if k >= 2 {
            g2 += a * kf * (kf - 1.0) * r.powi(k as i32 - 2);
        }
        if k >= 3 {
            g3 += a * kf * (kf - 1.0) * (kf - 2.0) * r.powi(k as i32 - 3);
        }
    }
    (g, g1, g2, g3)
}

/// Radial `h=phig` and `h',h'',h'''` at scalar r.
pub(crate) fn h_derivs(l: f64, coef: &[f64], r: f64, c: i32) -> (f64, f64, f64) {
    let (phi, p1, p2, p3) = phi_derivs(l, r, c);
    let (g, g1, g2, g3) = poly_derivs(coef, r);
    let h1 = p1 * g + phi * g1;
    let h2 = p2 * g + 2.0 * p1 * g1 + phi * g2;
    let h3 = p3 * g + 3.0 * p2 * g1 + 3.0 * p1 * g2 + phi * g3;
    (h1, h2, h3)
}

/// `P_k=phi(r)r^k` and `P',P'',P'''` for k=0..=nmax at scalar r.
fn p_arrays(l: f64, r: f64, nmax: usize, c: i32) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (phi, p1, p2, p3) = phi_derivs(l, r, c);
    let (mut pp, mut pp1, mut pp2, mut pp3) = (
        vec![0.0; nmax + 1],
        vec![0.0; nmax + 1],
        vec![0.0; nmax + 1],
        vec![0.0; nmax + 1],
    );
    for k in 0..=nmax {
        let kf = k as f64;
        let rk = r.powi(k as i32);
        let rk1 = if k >= 1 {
            kf * r.powi(k as i32 - 1)
        } else {
            0.0
        };
        let rk2 = if k >= 2 {
            kf * (kf - 1.0) * r.powi(k as i32 - 2)
        } else {
            0.0
        };
        let rk3 = if k >= 3 {
            kf * (kf - 1.0) * (kf - 2.0) * r.powi(k as i32 - 3)
        } else {
            0.0
        };
        pp[k] = phi * rk;
        pp1[k] = p1 * rk + phi * rk1;
        pp2[k] = p2 * rk + 2.0 * p1 * rk1 + phi * rk2;
        pp3[k] = p3 * rk + 3.0 * p2 * rk1 + 3.0 * p1 * rk2 + phi * rk3;
    }
    (pp, pp1, pp2, pp3)
}

/// `C_n=r^n` and `C',C'',C'''` for n=0..=nmax at scalar r.
fn c_arrays(r: f64, nmax: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut c0, mut c1, mut c2, mut c3) = (
        vec![0.0; nmax + 1],
        vec![0.0; nmax + 1],
        vec![0.0; nmax + 1],
        vec![0.0; nmax + 1],
    );
    for n in 0..=nmax {
        let nf = n as f64;
        c0[n] = r.powi(n as i32);
        c1[n] = if n >= 1 {
            nf * r.powi(n as i32 - 1)
        } else {
            0.0
        };
        c2[n] = if n >= 2 {
            nf * (nf - 1.0) * r.powi(n as i32 - 2)
        } else {
            0.0
        };
        c3[n] = if n >= 3 {
            nf * (nf - 1.0) * (nf - 2.0) * r.powi(n as i32 - 3)
        } else {
            0.0
        };
    }
    (c0, c1, c2, c3)
}

#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `(du_hat/dr_self)T v = (v - (u_hat*v)u_hat)/r`.
#[inline]
fn du(u: &[f64; 3], v: &[f64; 3], r: f64) -> [f64; 3] {
    let uv = dot3(u, v);
    [
        (v[0] - uv * u[0]) / r,
        (v[1] - uv * u[1]) / r,
        (v[2] - uv * u[2]) / r,
    ]
}

/// `dS_triple/dr_self` for one (self, other, atom I) f-channel triple.
/// `gam` = species gamma slice (l,m,n) C-flat dims (n_en+1, n_en+1, n_ee+1);
/// `tr` swaps the (l,m) axes (role transpose).
#[allow(clippy::too_many_arguments)]
fn ds_dself(
    self_r: [f64; 3],
    other_r: [f64; 3],
    r_atom: [f64; 3],
    cot_a: [f64; 3],
    cot_b: [f64; 3],
    gam: &[f64],
    tr: bool,
    l: f64,
    c: i32,
    n_en: usize,
    n_ee: usize,
    w_self: f64,
    w_other: f64,
    ci: Option<&CellInfo>,
) -> [f64; 3] {
    let (ai, r_ai) = dtn_disp(self_r, r_atom, ci);
    let (bi, r_bi) = dtn_disp(other_r, r_atom, ci);
    if r_ai >= l || r_bi >= l {
        return [0.0; 3];
    }
    let (ab, r_ab) = dtn_disp(self_r, other_r, ci);
    let uai = [ai[0] / r_ai, ai[1] / r_ai, ai[2] / r_ai];
    let ubi = [bi[0] / r_bi, bi[1] / r_bi, bi[2] / r_bi];
    let uab = [ab[0] / r_ab, ab[1] / r_ab, ab[2] / r_ab];
    let (pa, pa1, pa2, pa3) = p_arrays(l, r_ai, n_en, c);
    let (pb, pb1, pb2, _pb3) = p_arrays(l, r_bi, n_en, c);
    let (cc, cc1, cc2, cc3) = c_arrays(r_ab, n_ee);
    let d2 = n_en + 1;
    let d3 = n_ee + 1;
    // G(i,j,n) = Sum_{l,m,p} gam[l,m,p] i[l] j[m] n[p]  (tr -> gam[m,l,p])
    let g = |i: &[f64], j: &[f64], nn: &[f64]| -> f64 {
        let mut s = 0.0;
        for (li, i_li) in i[..n_en + 1].iter().enumerate() {
            for (mi, j_mi) in j[..n_en + 1].iter().enumerate() {
                let off = if tr {
                    mi * d2 * d3 + li * d3
                } else {
                    li * d2 * d3 + mi * d3
                };
                let im = *i_li * *j_mi;
                if im == 0.0 {
                    continue;
                }
                for p in 0..=n_ee {
                    s += gam[off + p] * im * nn[p];
                }
            }
        }
        s
    };
    let cos_i = dot3(&uai, &uab);
    let cos_j = -dot3(&ubi, &uab);
    let dcos_i = [
        (uab[0] - cos_i * uai[0]) / r_ai + (uai[0] - cos_i * uab[0]) / r_ab,
        (uab[1] - cos_i * uai[1]) / r_ai + (uai[1] - cos_i * uab[1]) / r_ab,
        (uab[2] - cos_i * uai[2]) / r_ai + (uai[2] - cos_i * uab[2]) / r_ab,
    ];
    let dcos_j = [
        -(ubi[0] + cos_j * uab[0]) / r_ab,
        -(ubi[1] + cos_j * uab[1]) / r_ab,
        -(ubi[2] + cos_j * uab[2]) / r_ab,
    ];
    let mut out = [0.0; 3];
    let axpy = |o: &mut [f64; 3], s: f64, v: &[f64; 3]| {
        for d in 0..3 {
            o[d] += s * v[d];
        }
    };
    // ---- (1) -d(cot_a*gf_a)/dr_self ----
    let coef_i = g(&pa1, &pb, &cc);
    let coef_ee = g(&pa, &pb, &cc1);
    let s_ia = dot3(&cot_a, &uai);
    let s_ea = dot3(&cot_a, &uab);
    let g_pa2_pb_cc = g(&pa2, &pb, &cc);
    let g_pa1_pb_cc1 = g(&pa1, &pb, &cc1);
    let g_pa_pb_cc2 = g(&pa, &pb, &cc2);
    let dcoef_i = [
        g_pa2_pb_cc * uai[0] + g_pa1_pb_cc1 * uab[0],
        g_pa2_pb_cc * uai[1] + g_pa1_pb_cc1 * uab[1],
        g_pa2_pb_cc * uai[2] + g_pa1_pb_cc1 * uab[2],
    ];
    let dcoef_ee = [
        g_pa1_pb_cc1 * uai[0] + g_pa_pb_cc2 * uab[0],
        g_pa1_pb_cc1 * uai[1] + g_pa_pb_cc2 * uab[1],
        g_pa1_pb_cc1 * uai[2] + g_pa_pb_cc2 * uab[2],
    ];
    let du_uai_a = du(&uai, &cot_a, r_ai);
    let du_uab_a = du(&uab, &cot_a, r_ab);
    let mut t1 = [0.0; 3];
    axpy(&mut t1, s_ia, &dcoef_i);
    axpy(&mut t1, coef_i, &du_uai_a);
    axpy(&mut t1, s_ea, &dcoef_ee);
    axpy(&mut t1, coef_ee, &du_uab_a);
    for d in 0..3 {
        out[d] -= w_self * t1[d]; // (1) belongs to kc_a (self)
    }
    // ---- (2) -d(cot_b*gf_b)/dr_self ----  gf_b = coefJ ubI - coefEE uab
    let s_jb = dot3(&cot_b, &ubi);
    let s_eb = dot3(&cot_b, &uab);
    let g_pa1_pb1_cc = g(&pa1, &pb1, &cc);
    let g_pa_pb1_cc1 = g(&pa, &pb1, &cc1);
    let dcoef_j = [
        g_pa1_pb1_cc * uai[0] + g_pa_pb1_cc1 * uab[0],
        g_pa1_pb1_cc * uai[1] + g_pa_pb1_cc1 * uab[1],
        g_pa1_pb1_cc * uai[2] + g_pa_pb1_cc1 * uab[2],
    ];
    let du_uab_b = du(&uab, &cot_b, r_ab);
    let mut t2 = [0.0; 3];
    axpy(&mut t2, s_jb, &dcoef_j);
    axpy(&mut t2, -s_eb, &dcoef_ee);
    axpy(&mut t2, -coef_ee, &du_uab_b);
    for d in 0..3 {
        out[d] -= w_other * t2[d]; // (2) belongs to kc_b (other)
    }
    // ---- (3) -1/2 dlap_a/dr_self ----
    let g_pa3_pb_cc = g(&pa3, &pb, &cc);
    let g_pa1_pb_cc = g(&pa1, &pb, &cc);
    let g_pa2_pb_cc1 = g(&pa2, &pb, &cc1);
    let g_pa_pb_cc3 = g(&pa, &pb, &cc3);
    let g_pa_pb_cc1 = g(&pa, &pb, &cc1);
    let g_pa1_pb_cc2 = g(&pa1, &pb, &cc2);
    let ai_radial = g_pa3_pb_cc + 2.0 * g_pa2_pb_cc / r_ai - 2.0 * g_pa1_pb_cc / (r_ai * r_ai);
    let ai_cross = g_pa2_pb_cc1 + 2.0 * g_pa1_pb_cc1 / r_ai;
    let ab_radial1 = g_pa1_pb_cc2 + 2.0 * g_pa1_pb_cc1 / r_ab;
    let ab_radial2 = g_pa_pb_cc3 + 2.0 * g_pa_pb_cc2 / r_ab - 2.0 * g_pa_pb_cc1 / (r_ab * r_ab);
    let dla1 = [
        ai_radial * uai[0] + ai_cross * uab[0],
        ai_radial * uai[1] + ai_cross * uab[1],
        ai_radial * uai[2] + ai_cross * uab[2],
    ];
    let dla2 = [
        ab_radial1 * uai[0] + ab_radial2 * uab[0],
        ab_radial1 * uai[1] + ab_radial2 * uab[1],
        ab_radial1 * uai[2] + ab_radial2 * uab[2],
    ];
    let kk = g_pa1_pb_cc1;
    let dk = [
        g_pa2_pb_cc1 * uai[0] + g_pa1_pb_cc2 * uab[0],
        g_pa2_pb_cc1 * uai[1] + g_pa1_pb_cc2 * uab[1],
        g_pa2_pb_cc1 * uai[2] + g_pa1_pb_cc2 * uab[2],
    ];
    let mut dla3 = [0.0; 3];
    axpy(&mut dla3, 2.0 * kk, &dcos_i);
    axpy(&mut dla3, 2.0 * cos_i, &dk);
    for d in 0..3 {
        out[d] -= w_self * 0.5 * (dla1[d] + dla2[d] + dla3[d]); // (3) lap_a -> kc_a (self)
    }
    // ---- (4) -1/2 dlap_b/dr_self ----  lap_b = Lb1 + La2 + Lb3
    let g_pa1_pb2_cc = g(&pa1, &pb2, &cc);
    let g_pa_pb2_cc1 = g(&pa, &pb2, &cc1);
    let g_pa1_pb1_cc1 = g(&pa1, &pb1, &cc1);
    let g_pa_pb1_cc2 = g(&pa, &pb1, &cc2);
    let bi_radial = g_pa1_pb2_cc + 2.0 * g_pa1_pb1_cc / r_bi;
    let bi_cross = g_pa_pb2_cc1 + 2.0 * g_pa_pb1_cc1 / r_bi;
    let dlb1 = [
        bi_radial * uai[0] + bi_cross * uab[0],
        bi_radial * uai[1] + bi_cross * uab[1],
        bi_radial * uai[2] + bi_cross * uab[2],
    ];
    let kb = g_pa_pb1_cc1;
    let dkb = [
        g_pa1_pb1_cc1 * uai[0] + g_pa_pb1_cc2 * uab[0],
        g_pa1_pb1_cc1 * uai[1] + g_pa_pb1_cc2 * uab[1],
        g_pa1_pb1_cc1 * uai[2] + g_pa_pb1_cc2 * uab[2],
    ];
    let mut dlb3 = [0.0; 3];
    axpy(&mut dlb3, 2.0 * kb, &dcos_j);
    axpy(&mut dlb3, 2.0 * cos_j, &dkb);
    for d in 0..3 {
        out[d] -= w_other * 0.5 * (dlb1[d] + dla2[d] + dlb3[d]); // (4) lap_b -> kc_b (other)
    }
    out
}

/// `Sum_{l,m,p} gam[l,m,p] i[l] j[m] n[p]` for the f-channel gamma contraction.
fn g_contract(gam: &[f64], n_en: usize, n_ee: usize, i: &[f64], j: &[f64], nn: &[f64]) -> f64 {
    let d2 = n_en + 1;
    let d3 = n_ee + 1;
    let mut s = 0.0;
    for (li, i_li) in i[..n_en + 1].iter().enumerate() {
        for (mi, j_mi) in j[..n_en + 1].iter().enumerate() {
            let im = *i_li * *j_mi;
            if im == 0.0 {
                continue;
            }
            let off = li * d2 * d3 + mi * d3;
            for p in 0..=n_ee {
                s += gam[off + p] * im * nn[p];
            }
        }
    }
    s
}

/// `dJ_DTN/dR` per atom (Pulay force; chi + f channels, u is e-e only). cot-free
/// (value gradient). Translation identity: `dr_eI/dR_I=-u_hat_eI`.
pub fn grad_j_dtn_wrt_r_atom(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let all_e = all_elec(r_up, r_dn);
    let n_e = all_e.len();
    let n_atom = d.positions.nrows();
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    // chi: d/dR_I = -Sum_e h'(r_eI) u_hat_eI ,  h = phi*Sumbeta r^k
    if d.use_chi {
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let lchi = d.l_chi[sp];
            let coef: Vec<f64> = d.beta.row(sp).to_vec();
            for all_e_e in all_e[..n_e].iter() {
                let (vec, r) = dtn_disp(
                    *all_e_e,
                    [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ],
                    ci.as_ref(),
                );
                if r < crate::setting_consts::NATIVE_DISTANCE_SKIP_THRESHOLD || r >= lchi {
                    continue;
                }
                let (h1, _h2, _h3) = h_derivs(lchi, &coef, r, d.c);
                for k in 0..3 {
                    out[[ia, k]] -= h1 * vec[k] / r;
                }
            }
        }
    }
    // f: d/dR_I = -Sum_{a<b} [coef_i u_hat_iI + coef_j u_hat_jI]
    if d.use_f {
        let stride = (d.n_en + 1) * (d.n_en + 1) * (d.n_ee + 1);
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let gam = &d.gamma[sp * stride..(sp + 1) * stride];
            let lf = d.l_f[sp];
            let r_atom = [
                d.positions[[ia, 0]],
                d.positions[[ia, 1]],
                d.positions[[ia, 2]],
            ];
            for a in 0..n_e {
                for b in (a + 1)..n_e {
                    let (ai, r_ai) = dtn_disp(all_e[a], r_atom, ci.as_ref());
                    let (bi, r_bi) = dtn_disp(all_e[b], r_atom, ci.as_ref());
                    if r_ai >= lf || r_bi >= lf {
                        continue;
                    }
                    let (_ab, r_ab) = dtn_disp(all_e[a], all_e[b], ci.as_ref());
                    let (pa, pa1, _, _) = p_arrays(lf, r_ai, d.n_en, d.c);
                    let (pb, pb1, _, _) = p_arrays(lf, r_bi, d.n_en, d.c);
                    let (cn, _, _, _) = c_arrays(r_ab, d.n_ee);
                    let coef_i = g_contract(gam, d.n_en, d.n_ee, &pa1, &pb, &cn);
                    let coef_j = g_contract(gam, d.n_en, d.n_ee, &pa, &pb1, &cn);
                    for k in 0..3 {
                        out[[ia, k]] -= coef_i * ai[k] / r_ai + coef_j * bi[k] / r_bi;
                    }
                }
            }
        }
    }
    out
}

/// DTN Pulay (`dJ/dR`, chi + f channels) DELTA for a single-electron hop of electron
/// `li` (spin `is_up`) to `new_pos`: `pulay(moved) - pulay(base)`, restricted to the
/// terms that touch the moved electron `g`. Per-term formulas byte-identical to
/// [`grad_j_dtn_wrt_r_atom`] (u is e-e only -> no Pulay), so `base + delta` reproduces
/// the full moved-config Pulay to fp noise. O(Natom + Ne*Natom) per hop (moved
/// electron's chi legs + its f pairs), NOT the full O(Ne*Natom + Ne^2*Natom).
pub fn grad_j_dtn_wrt_r_atom_move_delta(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_pos: [f64; 3],
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let all_e = all_elec(r_up, r_dn);
    let n_e = all_e.len();
    let n_atom = d.positions.nrows();
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let g_idx = if is_up { li } else { n_up + li };
    let r_old = all_e[g_idx];
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    let accumulate = |g_pos: [f64; 3], sign: f64, out: &mut Array2<f64>| {
        // chi: dJ/dR_I = -h'(r_gI) u_hat_gI (only the moved electron's e-N legs).
        if d.use_chi {
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let lchi = d.l_chi[sp];
                let coef: Vec<f64> = d.beta.row(sp).to_vec();
                let (vec, r) = dtn_disp(
                    g_pos,
                    [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ],
                    ci.as_ref(),
                );
                if r < crate::setting_consts::NATIVE_DISTANCE_SKIP_THRESHOLD || r >= lchi {
                    continue;
                }
                let (h1, _h2, _h3) = h_derivs(lchi, &coef, r, d.c);
                for k in 0..3 {
                    out[[ia, k]] -= sign * h1 * vec[k] / r;
                }
            }
        }
        // f: pairs (g,k) x atoms, respecting the global a<b ordering (coef_i on a's leg).
        if d.use_f {
            let stride = (d.n_en + 1) * (d.n_en + 1) * (d.n_ee + 1);
            for (kk, all_e_kk) in all_e[..n_e].iter().enumerate() {
                if kk == g_idx {
                    continue;
                }
                let (a_pos, b_pos) = if g_idx < kk {
                    (g_pos, *all_e_kk)
                } else {
                    (*all_e_kk, g_pos)
                };
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let gam = &d.gamma[sp * stride..(sp + 1) * stride];
                    let lf = d.l_f[sp];
                    let r_atom = [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ];
                    let (ai, r_ai) = dtn_disp(a_pos, r_atom, ci.as_ref());
                    let (bi, r_bi) = dtn_disp(b_pos, r_atom, ci.as_ref());
                    if r_ai >= lf || r_bi >= lf {
                        continue;
                    }
                    let (_ab, r_ab) = dtn_disp(a_pos, b_pos, ci.as_ref());
                    let (pa, pa1, _, _) = p_arrays(lf, r_ai, d.n_en, d.c);
                    let (pb, pb1, _, _) = p_arrays(lf, r_bi, d.n_en, d.c);
                    let (cn, _, _, _) = c_arrays(r_ab, d.n_ee);
                    let coef_i = g_contract(gam, d.n_en, d.n_ee, &pa1, &pb, &cn);
                    let coef_j = g_contract(gam, d.n_en, d.n_ee, &pa, &pb1, &cn);
                    for k in 0..3 {
                        out[[ia, k]] -= sign * (coef_i * ai[k] / r_ai + coef_j * bi[k] / r_bi);
                    }
                }
            }
        }
    };
    accumulate(new_pos, 1.0, &mut out);
    accumulate(r_old, -1.0, &mut out);
    out
}

fn all_elec(r_up: &ArrayView2<f64>, r_dn: &ArrayView2<f64>) -> Vec<[f64; 3]> {
    let mut v = Vec::with_capacity(r_up.nrows() + r_dn.nrows());
    for e in 0..r_up.nrows() {
        v.push([r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]);
    }
    for e in 0..r_dn.nrows() {
        v.push([r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]]);
    }
    v
}

/// `dT_DTN_f/dr` per electron (f-channel e-e-n). cot=gJ+gD `(Ne,3)`. -> `(Ne,3)`.
fn grad_t_dtn_f_wrt_r(
    d: &JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
) -> Array2<f64> {
    let all_e = all_elec(r_up, r_dn);
    let n_e = all_e.len();
    let n_atom = d.positions.nrows();
    let stride = (d.n_en + 1) * (d.n_en + 1) * (d.n_ee + 1);
    let wt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mut dt = Array2::<f64>::zeros((n_e, 3));
    for a in 0..n_e {
        let ca = [cot[[a, 0]], cot[[a, 1]], cot[[a, 2]]];
        for b in (a + 1)..n_e {
            let cb = [cot[[b, 0]], cot[[b, 1]], cot[[b, 2]]];
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let gam = &d.gamma[sp * stride..(sp + 1) * stride];
                let lf = d.l_f[sp];
                let r_atom = [
                    d.positions[[ia, 0]],
                    d.positions[[ia, 1]],
                    d.positions[[ia, 2]],
                ];
                // ds_dself MIXES kc_self and kc_other (parts 1,3 = self; 2,4 = other), so the
                // per-kc mask weights go INSIDE (w_self, w_other), not as an outer factor.
                let da = ds_dself(
                    all_e[a],
                    all_e[b],
                    r_atom,
                    ca,
                    cb,
                    gam,
                    false,
                    lf,
                    d.c,
                    d.n_en,
                    d.n_ee,
                    wt(a),
                    wt(b),
                    ci.as_ref(),
                );
                let db = ds_dself(
                    all_e[b],
                    all_e[a],
                    r_atom,
                    cb,
                    ca,
                    gam,
                    true,
                    lf,
                    d.c,
                    d.n_en,
                    d.n_ee,
                    wt(b),
                    wt(a),
                    ci.as_ref(),
                );
                for k in 0..3 {
                    dt[[a, k]] += da[k];
                    dt[[b, k]] += db[k];
                }
            }
        }
    }
    dt
}

/// `dT_DTN/dr` per electron (u + chi + f). cot=gJ+gD `(Ne,3)`. -> `(Ne,3)`.
pub fn grad_t_dtn_wrt_r(
    d: &JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    grad_t_dtn_wrt_r_w(d, cot, r_up, r_dn, None)
}

/// `grad_t_dtn_wrt_r` with per-electron kinetic-element weights `w` (`d(Sum_e w_e*kc_e^DTN)/dr`).
/// Each pair/triple term is weighted by the electron whose `kc` it belongs to. `None` = all-1.
pub fn grad_t_dtn_wrt_r_w(
    d: &JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
) -> Array2<f64> {
    let all_e = all_elec(r_up, r_dn);
    let n_up = r_up.nrows();
    let n_e = all_e.len();
    let spin = |e: usize| -> usize {
        if e < n_up {
            0
        } else {
            1
        }
    };
    let wt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mut dt = Array2::<f64>::zeros((n_e, 3));
    // hess*v helper: Hess = h2 u_hat(x)u_hat + (h1/r)(I - u_hat(x)u_hat); returns Hess*v
    let hess_dot = |h1: f64, h2: f64, r: f64, u: &[f64; 3], v: &[f64; 3]| -> [f64; 3] {
        let uv = dot3(u, v);
        [
            h2 * u[0] * uv + (h1 / r) * (v[0] - uv * u[0]),
            h2 * u[1] * uv + (h1 / r) * (v[1] - uv * u[1]),
            h2 * u[2] * uv + (h1 / r) * (v[2] - uv * u[2]),
        ]
    };
    // ---- u-channel (e-e) ----
    if d.use_u {
        let lu = d.l_u;
        for e in 0..n_e {
            let ce = [cot[[e, 0]], cot[[e, 1]], cot[[e, 2]]];
            for k in 0..n_e {
                if k == e {
                    continue;
                }
                let ch = if spin(e) == spin(k) { 0 } else { 1 };
                let (vec, r) = dtn_disp(all_e[k], all_e[e], ci.as_ref());
                if r < crate::setting_consts::NATIVE_DISTANCE_SKIP_THRESHOLD || r >= lu {
                    continue;
                }
                let u = [vec[0] / r, vec[1] / r, vec[2] / r];
                let coef: Vec<f64> = d.alpha.row(ch).to_vec();
                let (h1, h2, h3) = h_derivs(lu, &coef, r, d.c);
                let lp = h3 + 2.0 * h2 / r - 2.0 * h1 / (r * r);
                let ck = [cot[[k, 0]], cot[[k, 1]], cot[[k, 2]]];
                // partner (k!=e): -0.5(-Lp u_hat) - cot[k]*(-Hess) = +0.5 Lp u_hat + Hess*cot[k]
                let hck = hess_dot(h1, h2, r, &u, &ck);
                let hce = hess_dot(h1, h2, r, &u, &ce);
                for dd in 0..3 {
                    // partner term -> d(kc_k)/dr_e (weight w[k]); self -> d(kc_e)/dr_e (weight w[e])
                    dt[[e, dd]] += wt(k) * (0.5 * lp * u[dd] + hck[dd]);
                    dt[[e, dd]] += wt(e) * (0.5 * lp * u[dd] - hce[dd]);
                }
            }
        }
    }
    // ---- chi-channel (e-N) ----
    if d.use_chi {
        let n_atom = d.positions.nrows();
        for e in 0..n_e {
            let ce = [cot[[e, 0]], cot[[e, 1]], cot[[e, 2]]];
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let lchi = d.l_chi[sp];
                let (vec, r) = dtn_disp(
                    all_e[e],
                    [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ],
                    ci.as_ref(),
                );
                if r < crate::setting_consts::NATIVE_DISTANCE_SKIP_THRESHOLD || r >= lchi {
                    continue;
                }
                let u = [vec[0] / r, vec[1] / r, vec[2] / r];
                let coef: Vec<f64> = d.beta.row(sp).to_vec();
                let (h1, h2, h3) = h_derivs(lchi, &coef, r, d.c);
                let lp = h3 + 2.0 * h2 / r - 2.0 * h1 / (r * r);
                let hce = hess_dot(h1, h2, r, &u, &ce);
                for dd in 0..3 {
                    dt[[e, dd]] += wt(e) * (-0.5 * lp * u[dd] - hce[dd]);
                }
            }
        }
    }
    // ---- f-channel (e-e-n) ----
    if d.use_f {
        let f = grad_t_dtn_f_wrt_r(d, cot, r_up, r_dn, w);
        dt += &f;
    }
    dt
}

/// `dT_DTN/dR` per atom (chi + f; u has no R-dep). cot=gJ+gD `(Ne,3)`. ->
/// `(n_atom,3)`. Translation id: `dS/dR_I = -Sum_e dS/dr_e` scattered to atom I.
pub fn grad_t_dtn_wrt_r_atom(
    d: &JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    grad_t_dtn_wrt_r_atom_w(d, cot, r_up, r_dn, None)
}

/// `grad_t_dtn_wrt_r_atom` with per-electron kinetic weights `w`. `None` = all-1.
pub fn grad_t_dtn_wrt_r_atom_w(
    d: &JastrowDtnData,
    cot: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
) -> Array2<f64> {
    let all_e = all_elec(r_up, r_dn);
    let n_e = all_e.len();
    let n_atom = d.positions.nrows();
    let wt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mut dr = Array2::<f64>::zeros((n_atom, 3));
    let hess_dot = |h1: f64, h2: f64, r: f64, u: &[f64; 3], v: &[f64; 3]| -> [f64; 3] {
        let uv = dot3(u, v);
        [
            h2 * u[0] * uv + (h1 / r) * (v[0] - uv * u[0]),
            h2 * u[1] * uv + (h1 / r) * (v[1] - uv * u[1]),
            h2 * u[2] * uv + (h1 / r) * (v[2] - uv * u[2]),
        ]
    };
    // chi-channel: dT_chi/dR_I = -Sum_e (chi dT/dr_e for atom I) = +0.5 dlap + Hess*cot[e]
    if d.use_chi {
        for e in 0..n_e {
            let ce = [cot[[e, 0]], cot[[e, 1]], cot[[e, 2]]];
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let lchi = d.l_chi[sp];
                let (vec, r) = dtn_disp(
                    all_e[e],
                    [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ],
                    ci.as_ref(),
                );
                if r < crate::setting_consts::NATIVE_DISTANCE_SKIP_THRESHOLD || r >= lchi {
                    continue;
                }
                let u = [vec[0] / r, vec[1] / r, vec[2] / r];
                let coef: Vec<f64> = d.beta.row(sp).to_vec();
                let (h1, h2, h3) = h_derivs(lchi, &coef, r, d.c);
                let lp = h3 + 2.0 * h2 / r - 2.0 * h1 / (r * r);
                let hce = hess_dot(h1, h2, r, &u, &ce);
                for dd in 0..3 {
                    // -(-0.5 lp u_hat - Hess*cot) = +0.5 lp u_hat + Hess*cot (kc_e -> weight w[e])
                    dr[[ia, dd]] += wt(e) * (0.5 * lp * u[dd] + hce[dd]);
                }
            }
        }
    }
    // f-channel: dT_f/dR_I = -Sum_{a<b}(dS/dr_a + dS/dr_b) for triples at atom I
    if d.use_f {
        let stride = (d.n_en + 1) * (d.n_en + 1) * (d.n_ee + 1);
        for a in 0..n_e {
            let ca = [cot[[a, 0]], cot[[a, 1]], cot[[a, 2]]];
            for b in (a + 1)..n_e {
                let cb = [cot[[b, 0]], cot[[b, 1]], cot[[b, 2]]];
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let gam = &d.gamma[sp * stride..(sp + 1) * stride];
                    let lf = d.l_f[sp];
                    let r_atom = [
                        d.positions[[ia, 0]],
                        d.positions[[ia, 1]],
                        d.positions[[ia, 2]],
                    ];
                    let da = ds_dself(
                        all_e[a],
                        all_e[b],
                        r_atom,
                        ca,
                        cb,
                        gam,
                        false,
                        lf,
                        d.c,
                        d.n_en,
                        d.n_ee,
                        wt(a),
                        wt(b),
                        ci.as_ref(),
                    );
                    let db = ds_dself(
                        all_e[b],
                        all_e[a],
                        r_atom,
                        cb,
                        ca,
                        gam,
                        true,
                        lf,
                        d.c,
                        d.n_en,
                        d.n_ee,
                        wt(b),
                        wt(a),
                        ci.as_ref(),
                    );
                    for k in 0..3 {
                        dr[[ia, k]] -= da[k] + db[k];
                    }
                }
            }
        }
    }
    dr
}
