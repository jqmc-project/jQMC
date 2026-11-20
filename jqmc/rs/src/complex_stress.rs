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

//! Complex Bloch PBC stress (twist k!=Gamma): `dln|det|/deps` (Pulay) and `dT/deps`
//! (kinetic), the complex analog of `stress_pbc.rs`. Affine strain virial (folded
//! field-grad x (r_w-R)) + image-T term (kernel field-grad x phase x T_kappa), in the
//! sphe home/kernel rank with the complex image phase. `ln|det|`/`T` real -> Re.

use crate::complex_det::det_ratio_complex;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::Complex64;

use crate::atomic_orbital::{
    compute_aos_gradlap_kernel_sphe, compute_aos_hessian_kernel_sphe, compute_aos_vgl_kernel,
    AosGtoData,
};
use crate::complex_ao::{
    compute_aos_vgl_complex, compute_pw_complex, compute_pw_vgl_complex, fold_complex,
    ComplexPwData,
};
use crate::complex_det::{
    complex_gem_forward, complex_geminal_reverse, compute_geminal_complex, det_phase_lnabs,
    grads_and_laplacian_ln_det_complex, ComplexGeminal, ComplexOrb,
};
use crate::coulomb_potential::legendre;
use crate::ewald::{inv3, min_image};
#[allow(unused_imports)]
use crate::mcmc::wrap_rows;

/// `Sum CM*dfield/deps` (3x3 complex) for one complex sphe leaf: affine virial (folded
/// grad `gfold`, wrapped electron) + image-T (kernel grad `gkern_real`xphase x T_kappa).
/// `cm` = AO/sphe-home-rank complex cotangent. ``restore[k]`` = the per-electron Bloch
/// wrap-restore phase `e^{2pii n0.k_frac}` (jqmc `_ComplexBlochPGTOMixin`): the fields are
/// evaluated at the WRAPPED coordinate and multiplied by `restore` so they equal the exact
/// raw-coordinate Bloch value -- 1:1 the JAX ``pbc_stress._field_stress`` convention (its
/// `gfold` carries the restore inside ``pgto_deriv``; its `term2` weight is
/// `gkern(r_w) * image_phase * restore`). All-ones at Gamma / inside the home cell.
#[allow(clippy::too_many_arguments)]
pub(crate) fn complex_field_stress(
    aos: &AosGtoData,
    cm: &Array2<Complex64>,
    gfold: &[Array2<Complex64>; 3],
    gkern_real: &[Array2<f64>; 3],
    phase: &[Complex64],
    restore: &[Complex64],
    r_w: &Array2<f64>,
    home_atom: &[usize],
    kernel_atom: &[usize],
    positions: &ArrayView2<f64>,
    out: &mut Array2<Complex64>,
) {
    let num_ao = cm.nrows();
    let n = cm.ncols();
    // term1: affine virial (WRAPPED electron weight, restore-phased folded field)
    for mu in 0..num_ao {
        let ra = home_atom[mu];
        for k in 0..n {
            let cmk = cm[[mu, k]] * restore[k];
            for c in 0..3 {
                let g = cmk * gfold[c][[mu, k]];
                for d in 0..3 {
                    out[[c, d]] += g * (r_w[[k, d]] - positions[[ra, d]]);
                }
            }
        }
    }
    // term2: image-T (kernel grad at the WRAPPED coordinate x image phase x restore)
    for kap in 0..aos.num_ao_kernel {
        let home = aos.kernel_to_home[kap];
        let at = kernel_atom[kap];
        let t = [
            aos.centers[[kap, 0]] - positions[[at, 0]],
            aos.centers[[kap, 1]] - positions[[at, 1]],
            aos.centers[[kap, 2]] - positions[[at, 2]],
        ];
        let ph = phase[kap];
        for k in 0..n {
            let cmk = cm[[home, k]] * restore[k];
            for c in 0..3 {
                let g = ph * gkern_real[c][[kap, k]] * cmk;
                for d in 0..3 {
                    out[[c, d]] -= g * t[d];
                }
            }
        }
    }
}

/// Home-cell wrap + Bloch restore phase for a complex det/J3 orbital leaf: the stress
/// field kernels evaluate at the WRAPPED coordinate (the static image tiling underflows
/// once a walker roams past ~half a cell -- the far-image AO class) and re-apply
/// `restore = e^{2pii n0.k_frac}` so the fields equal the exact raw Bloch value. This is
/// jqmc's `_ComplexBlochPGTOMixin` convention (`compute_AOs` wraps) -- the SAME identity
/// the complex e_L value kernels use (`ComplexOrb.wrap`, WrapCtx). `None` wrap (molecular
/// / Gamma-real leaf) = identity.
pub(crate) fn wrap_restore(orb: &ComplexOrb, r: &ArrayView2<f64>) -> (Array2<f64>, Vec<Complex64>) {
    match &orb.wrap {
        Some(w) => w.wrap(r),
        None => (r.to_owned(), vec![Complex64::new(1.0, 0.0); r.nrows()]),
    }
}

/// Per-side complex grad fields: folded (home, complex) + kernel (real). Evaluated at the
/// caller-WRAPPED `r_w` (the caller applies the Bloch restore phase via `complex_field_stress`).
pub(crate) fn grad_fields(
    orb: &ComplexOrb,
    r_w: &Array2<f64>,
) -> ([Array2<Complex64>; 3], [Array2<f64>; 3]) {
    let (_, gx, gy, gz, _) = compute_aos_vgl_kernel(&orb.aos, &r_w.view(), false);
    let gk = [gx, gy, gz];
    let gf = [
        fold_complex(&orb.aos, &gk[0], &orb.image_phase),
        fold_complex(&orb.aos, &gk[1], &orb.image_phase),
        fold_complex(&orb.aos, &gk[2], &orb.image_phase),
    ];
    (gf, gk)
}

fn gradlap_fields(
    orb: &ComplexOrb,
    r_w: &Array2<f64>,
) -> ([Array2<Complex64>; 3], [Array2<f64>; 3]) {
    let gk = compute_aos_gradlap_kernel_sphe(&orb.aos, &r_w.view(), false);
    let gf = [
        fold_complex(&orb.aos, &gk[0], &orb.image_phase),
        fold_complex(&orb.aos, &gk[1], &orb.image_phase),
        fold_complex(&orb.aos, &gk[2], &orb.image_phase),
    ];
    (gf, gk)
}

/// Per-side complex Hessian fields: folded (home, complex) 3x3 + kernel (real) 3x3.
/// `[a][b]` = `d_ad_b AO`. For the withJ gou stress (field = grad_e AO, its spatial
/// grad is the e-th Hessian column `H[:][e]`). Caller-wrapped `r_w` (sphe leaf).
#[allow(clippy::type_complexity)]
fn hessian_fields(
    orb: &ComplexOrb,
    r_w: &Array2<f64>,
) -> ([[Array2<Complex64>; 3]; 3], [[Array2<f64>; 3]; 3]) {
    let hk = compute_aos_hessian_kernel_sphe(&orb.aos, &r_w.view(), false);
    let hf = [
        [
            fold_complex(&orb.aos, &hk[0][0], &orb.image_phase),
            fold_complex(&orb.aos, &hk[0][1], &orb.image_phase),
            fold_complex(&orb.aos, &hk[0][2], &orb.image_phase),
        ],
        [
            fold_complex(&orb.aos, &hk[1][0], &orb.image_phase),
            fold_complex(&orb.aos, &hk[1][1], &orb.image_phase),
            fold_complex(&orb.aos, &hk[1][2], &orb.image_phase),
        ],
        [
            fold_complex(&orb.aos, &hk[2][0], &orb.image_phase),
            fold_complex(&orb.aos, &hk[2][1], &orb.image_phase),
            fold_complex(&orb.aos, &hk[2][2], &orb.image_phase),
        ],
    ];
    (hf, hk)
}

/// Complex PW kinetic stress `d(grad^2)/deps` contribution for a composite twist
/// determinant: PW value is strain-invariant (frac-invariant) so only lap
/// contributes -- `lap=-|gk|^2*val`, `dlap/deps_cd = 2*gk_c*gk_d*val`. Re taken by
/// the caller. `cm_lap_pw = (num_pw, N)` complex.
fn pw_lap_stress_complex(
    pw: &ComplexPwData,
    cm_lap_pw: &Array2<Complex64>,
    r: &ArrayView2<f64>,
    out: &mut Array2<Complex64>,
) {
    let val = compute_pw_complex(pw, r);
    let n = val.ncols();
    for a in 0..pw.num_ao {
        let mut w = Complex64::new(0.0, 0.0);
        for k in 0..n {
            w += cm_lap_pw[[a, k]] * val[[a, k]];
        }
        let gk = pw.gk[a];
        for c in 0..3 {
            for d in 0..3 {
                out[[c, d]] += Complex64::new(2.0 * gk[c] * gk[d], 0.0) * w;
            }
        }
    }
}

/// Complex PW gou (drift-outer) kinetic stress: `d(grad_c AO)/deps`, cotangent
/// `cm_gou_pw[cp]`. Uses the true PW gradient (twist `i*gk_c*val`, Gamma real cos/sin
/// `-sin(G.r - is_sin*pi/2)*g_c`); mirrors `stress_pbc::pw_gou_stress`.
fn pw_gou_stress_complex(
    pw: &ComplexPwData,
    cm_gou_pw: &[Array2<Complex64>; 3],
    r: &ArrayView2<f64>,
    out: &mut Array2<Complex64>,
) {
    let (_val, grad, _lap) = compute_pw_vgl_complex(pw, r);
    let n = grad[0].ncols();
    for c in 0..3 {
        for cp in 0..3 {
            let mut acc = Complex64::new(0.0, 0.0);
            for a in 0..pw.num_ao {
                for k in 0..n {
                    acc += cm_gou_pw[cp][[a, k]] * grad[c][[a, k]];
                }
            }
            out[[c, cp]] -= acc;
        }
    }
}

/// Complex-twist B-spline determinant KINETIC stress `dT_det/deps` contribution (added complex to
/// `out`, caller takes Re). Mirrors the real `stress_pbc::bspline_kstress_side` but uses the
/// Cartesian Bloch-field form of JAX `ao.bspline_complex_kinetic_stress_contrib`: frac=r.inv(cell)
/// is affine-invariant while `k_cart -> F^-T k_cart` (as `G -> F^-T G` for PW), so the Bloch AO
/// kinetic fields transform as covariant tensors and
///     d(grad psi_p)/deps_pq = -delta_{aq}(grad psi)_p  ->  gou term = -Sum CM_gou[q]*grad[p]
///     d(lap psi)/deps_pq    = -2 (Hess psi)[p,q]        ->  lap term = -2 Sum CM_lap*Hess[p,q]
/// (value side 0). `grad`/`Hess` are the complex Bloch cart fields from `det_egrad_fields`
/// (`psi = u e^{ik.r}`), so at k=0 this reduces to the real `bspline_kstress_side` bit-for-bit.
/// `cm_lap_full` / `cm_gou_full` are the FULL (gto (+) bspline) AO-rank cotangents; the bspline
/// rows are sliced here (mirrors the composite slice of the complex geminal reverse).
fn bspline_kstress_side_complex(
    orb: &ComplexOrb,
    cm_lap_full: &Array2<Complex64>,
    cm_gou_full: &[Array2<Complex64>; 3],
    r: &ArrayView2<f64>,
    out: &mut Array2<Complex64>,
) {
    let bs = match &orb.bspline {
        Some(b) => b,
        None => return,
    };
    let num_bs = bs.num_ao;
    let total = cm_lap_full.nrows();
    let num_gto = total - num_bs;
    let bs0 = if orb.gto_first { num_gto } else { 0 };
    let sl = |cm: &Array2<Complex64>| {
        cm.slice(numpy::ndarray::s![bs0..bs0 + num_bs, ..])
            .to_owned()
    };
    let (ga, _gl, h) = bs.det_egrad_fields(r); // ga[p], h[p][q] cart Bloch fields (num_bs, N)
    let cm_lap = sl(cm_lap_full);
    let n = cm_lap.ncols();
    // lap term: out[p,q] += -2 Sum_{b,i} cm_lap[b,i] * Hess[p][q][b,i]
    for p in 0..3 {
        for q in 0..3 {
            let mut acc = Complex64::new(0.0, 0.0);
            for b in 0..num_bs {
                for i in 0..n {
                    acc += cm_lap[[b, i]] * h[p][q][[b, i]];
                }
            }
            out[[p, q]] += Complex64::new(-2.0, 0.0) * acc;
        }
    }
    // gou term (withJ / nonzero drift only): out[p,q] += -Sum_{b,i} cm_gou[q][b,i] * grad[p][b,i]
    let has_gou = cm_gou_full
        .iter()
        .any(|m| m.iter().any(|x| *x != Complex64::new(0.0, 0.0)));
    if has_gou {
        let cm_gou = [
            sl(&cm_gou_full[0]),
            sl(&cm_gou_full[1]),
            sl(&cm_gou_full[2]),
        ];
        for p in 0..3 {
            for q in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for b in 0..num_bs {
                    for i in 0..n {
                        acc += cm_gou[q][[b, i]] * ga[p][[b, i]];
                    }
                }
                out[[p, q]] -= acc;
            }
        }
    }
}

fn cm_project(orb: &ComplexOrb, bar: &Array2<Complex64>) -> Array2<Complex64> {
    match &orb.mo {
        Some(c) => c.t().dot(bar),
        None => bar.clone(),
    }
}

/// Complex det Pulay stress `dln det/deps` (COMPLEX 3x3) at a config (the det
/// cotangent field strain, before taking the real part, and the ECP non-local ratio stress.
#[allow(clippy::too_many_arguments)]
pub fn pbc_det_pulay_stress_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    gi: Option<&ArrayView2<Complex64>>,
) -> Array2<Complex64> {
    // Pure plane-wave / B-spline determinant (no GTO rows, no tiled leaf): the det VALUE
    // strain-Pulay is identically zero (fractional coordinates are strain-invariant), so skip
    // the geminal forward that would only feed empty field sweeps.
    let no_gto = |o: &ComplexOrb| o.tiled.is_none() && o.aos.num_ao == 0;
    if no_gto(&gem.orb_up) && no_gto(&gem.orb_dn) {
        return Array2::<Complex64>::zeros((3, 3));
    }
    let f = complex_gem_forward(gem, r_up, r_dn, gi);
    let bar_orb_up = f.wg.clone();
    let bar_orb_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    det_strain_side_complex(
        &gem.orb_up,
        r_up,
        &bar_orb_up,
        home_up,
        kernel_up,
        positions,
        &mut acc,
    );
    det_strain_side_complex(
        &gem.orb_dn,
        r_dn,
        &bar_orb_dn,
        home_dn,
        kernel_dn,
        positions,
        &mut acc,
    );
    acc
}

/// One spin side of [`pbc_det_pulay_stress_complex`]: the det VALUE field strain of the
/// orbital-rank cotangent `bar (M, N)` at the electrons `r (N, 3)`, accumulated into `acc`.
/// LINEAR in `bar` and SEPARABLE over the electron columns (every term of a column `k` reads
/// only electron `k`'s fields, wrap and restore phase), which the rank-1 mesh-stress path
/// uses to sum weighted candidate bars before ONE evaluation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn det_strain_side_complex(
    orb: &ComplexOrb,
    r: &ArrayView2<f64>,
    bar: &Array2<Complex64>,
    home: &[usize],
    kern: &[usize],
    positions: &ArrayView2<f64>,
    acc: &mut Array2<Complex64>,
) {
    if let Some(tiled) = &orb.tiled {
        *acc += &tiled.field_vjp(
            r,
            bar,
            None,
            None,
            crate::orbital_tiling::TiledCoordinate::Strain,
        );
        return;
    }
    let (r_w, restore) = wrap_restore(orb, r); // wrap + Bloch restore (jqmc compute_AOs convention)
    let cm = cm_project(orb, bar);
    // Composite / pure non-pgto: slice the GTO rows only. The PW / B-spline det VALUE Pulay is
    // strain-invariant (frac-invariant), so the extra rows contribute nothing; for a pure
    // PW / B-spline leaf num_gto=0 -> an empty cm and a no-op gto field stress.
    let num_extra = orb
        .pw
        .as_ref()
        .map(|p| p.num_ao)
        .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
        .unwrap_or(0);
    let cm = if num_extra == 0 {
        cm
    } else {
        let g0 = if orb.gto_first { 0 } else { num_extra };
        cm.slice(numpy::ndarray::s![g0..g0 + (cm.nrows() - num_extra), ..])
            .to_owned()
    };
    if let Some((re, im)) = fused_complex_field_stress(
        orb, &r_w, &restore, &cm, None, None, home, kern, positions, true,
    ) {
        let im = im.unwrap();
        for c in 0..3 {
            for d in 0..3 {
                acc[[c, d]] += Complex64::new(re[[c, d]], im[[c, d]]);
            }
        }
        return;
    }
    let (gf, gk) = grad_fields(orb, &r_w);
    complex_field_stress(
        &orb.aos,
        &cm,
        &gf,
        &gk,
        &orb.image_phase,
        &restore,
        &r_w,
        home,
        kern,
        positions,
        acc,
    );
}

/// The REAL parts of [`det_strain_side_complex`] for two orbital-rank cotangents `bars` at the
/// same electrons, from one fused [`crate::mesh_vgs::kin_contract_multi`] pass over the gto block
/// (four real weight sets: the real part of each phased bar), for callers that only read the
/// real part. Falls back to two [`det_strain_side_complex`] calls when the block is not covered.
pub(crate) fn det_strain_side_complex_re_pair(
    orb: &ComplexOrb,
    r: &ArrayView2<f64>,
    bars: [&Array2<Complex64>; 2],
    home: &[usize],
    kern: &[usize],
    positions: &ArrayView2<f64>,
) -> [Array2<f64>; 2] {
    let fallback = || {
        bars.map(|b| {
            let mut acc = Array2::<Complex64>::zeros((3, 3));
            det_strain_side_complex(orb, r, b, home, kern, positions, &mut acc);
            acc.mapv(|x| x.re)
        })
    };
    let aos = &orb.aos;
    if orb.tiled.is_some()
        || aos.num_ao == 0
        || std::env::var("JQMC_RS_MESH_TABLES").is_ok()
        || !crate::mesh_vgs::supported(aos)
        || orb.image_phase.len() != aos.num_ao_kernel
    {
        return if orb.tiled.is_some() || aos.num_ao > 0 {
            fallback()
        } else {
            [Array2::zeros((3, 3)), Array2::zeros((3, 3))]
        };
    }
    let Some(geom) = crate::mesh_vgs::ShellGeom::new(aos, home, Some((kern, positions))) else {
        return fallback();
    };
    let (r_w, restore) = wrap_restore(orb, r);
    let n = r_w.nrows();
    let g0 = orb.gto_ao_offset();
    let split = |bar: &Array2<Complex64>| -> (Vec<f64>, Vec<f64>) {
        let cm = cm_project(orb, bar);
        let (mut re, mut im) = (vec![0.0; aos.num_ao * n], vec![0.0; aos.num_ao * n]);
        for mu in 0..aos.num_ao {
            for k in 0..n {
                let x = cm[[g0 + mu, k]] * restore[k];
                re[mu * n + k] = x.re;
                im[mu * n + k] = x.im;
            }
        }
        (re, im)
    };
    let (a, b) = (split(bars[0]), split(bars[1]));
    let ph_re: Vec<f64> = orb.image_phase.iter().map(|z| z.re).collect();
    let ph_im: Vec<f64> = orb.image_phase.iter().map(|z| z.im).collect();
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|k| [r_w[[k, 0]], r_w[[k, 1]], r_w[[k, 2]]])
        .collect();
    fn kw<'w>(w: &'w [f64], ph: &'w [f64]) -> crate::mesh_vgs::KinWeights<'w> {
        crate::mesh_vgs::KinWeights {
            orb: w,
            lap: None,
            gou: None,
            kscale: Some(ph),
        }
    }
    // real part of each bar: (W_re, ph_re) - (W_im, ph_im)
    let ws = [
        kw(&a.0, &ph_re),
        kw(&a.1, &ph_im),
        kw(&b.0, &ph_re),
        kw(&b.1, &ph_im),
    ];
    let mut outs: Vec<crate::mesh_vgs::KinOut> = (0..4)
        .map(|_| crate::mesh_vgs::KinOut {
            elec: vec![[0.0; 3]; n],
            atom: Vec::new(),
            strain: Some([[0.0; 3]; 3]),
        })
        .collect();
    crate::mesh_vgs::kin_contract_multi(
        aos,
        &pts,
        &ws,
        crate::ao_cart_deriv::AoScreen::None,
        crate::ao_cart_deriv::AoScreen::None,
        Some(&geom),
        &mut outs,
    );
    let st: Vec<[[f64; 3]; 3]> = outs.iter().map(|o| o.strain.unwrap()).collect();
    let to = |x: [[f64; 3]; 3], y: [[f64; 3]; 3]| {
        Array2::from_shape_fn((3, 3), |(c, d)| x[c][d] - y[c][d])
    };
    [to(st[0], st[1]), to(st[2], st[3])]
}

/// [`det_strain_side_complex`] of a group whose bar columns are all multiples of ONE column,
/// `bar[:, v] = s[v] * col`, at the points `pts` (the moved electron's rank-1 candidate bar
/// columns are its base column over the det ratio): `Sum_v s[v] FS(col, point v)` from two
/// fused real contractions (real and imaginary part of the Bloch-phased kernel weights
/// `w_kap = (C^T col)[home(kap)] * image_phase[kap]`, [`crate::mesh_vgs`]) instead of the
/// kernel-rank complex AO gradient tables, under the e_L's ECP mesh value screen (shell and
/// primitive `pz_val`, as the real-Gamma moved electron; a dropped image contributes below
/// `FAST_SCREEN_ABS_BOUND`). `None` when the leaf is not covered (tiled orbitals, no
/// Cartesian gto block, or a shell without one home atom).
#[allow(clippy::too_many_arguments)]
pub(crate) fn det_strain_side_complex_col(
    orb: &ComplexOrb,
    pts: &ArrayView2<f64>,
    col: &numpy::ndarray::Array1<Complex64>,
    s: &[Complex64],
    home: &[usize],
    kern: &[usize],
    positions: &ArrayView2<f64>,
) -> Option<Array2<Complex64>> {
    if orb.tiled.is_some()
        || orb.aos.num_ao == 0
        || std::env::var("JQMC_RS_MESH_TABLES").is_ok()
        || !crate::mesh_vgs::supported(&orb.aos)
    {
        return None;
    }
    let geom = crate::mesh_vgs::ShellGeom::new(&orb.aos, home, Some((kern, positions)))?;
    let (r_w, restore) = wrap_restore(orb, pts);
    let cm = cm_project(orb, &col.clone().insert_axis(numpy::ndarray::Axis(1)));
    let num_extra = orb
        .pw
        .as_ref()
        .map(|p| p.num_ao)
        .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
        .unwrap_or(0);
    let g0 = if num_extra == 0 || orb.gto_first {
        0
    } else {
        num_extra
    };
    let aos = &orb.aos;
    let (mut w_re, mut w_im) = (vec![0.0; aos.num_ao_kernel], vec![0.0; aos.num_ao_kernel]);
    for kap in 0..aos.num_ao_kernel {
        let u = cm[[g0 + aos.kernel_to_home[kap], 0]] * orb.image_phase[kap];
        w_re[kap] = u.re;
        w_im[kap] = u.im;
    }
    let np = r_w.nrows();
    let p: Vec<[f64; 3]> = (0..np)
        .map(|v| [r_w[[v, 0]], r_w[[v, 1]], r_w[[v, 2]]])
        .collect();
    let run = |w: &[f64]| {
        let mut o = crate::mesh_vgs::MeshVgs::new(np, None, true, None);
        crate::mesh_vgs::mesh_vgs(
            aos,
            &p,
            crate::mesh_vgs::Weights::Kernel(w),
            crate::ao_cart_deriv::AoScreen::MeshValue,
            Some(&geom),
            &mut o,
        );
        o.strain
    };
    let (sr, si) = (run(&w_re), run(&w_im));
    let mut out = Array2::<Complex64>::zeros((3, 3));
    for v in 0..np {
        let f = s[v] * restore[v];
        for c in 0..3 {
            for d in 0..3 {
                out[[c, d]] += f * Complex64::new(sr[v][c][d], si[v][c][d]);
            }
        }
    }
    Some(out)
}

/// `Sum_a col[a] grad mo_a(r_v)` per point (the orbital gradient contracted with one MO-rank
/// column), the gradient part of [`ComplexOrb::value_grad_lap`] without forming the orbital
/// gradient tables: the gto block from two fused real contractions with the Bloch-phased
/// kernel weights (wrap + restore; the e_L's ECP mesh value screen, as the real-Gamma moved
/// electron), the plane-wave /
/// B-spline child from its own gradient tables. `None` when the leaf is not covered.
pub(crate) fn contracted_grad_complex(
    orb: &ComplexOrb,
    r: &ArrayView2<f64>,
    col: &numpy::ndarray::Array1<Complex64>,
) -> Option<Vec<[Complex64; 3]>> {
    if orb.tiled.is_some()
        || std::env::var("JQMC_RS_MESH_TABLES").is_ok()
        || (orb.aos.num_ao > 0 && !crate::mesh_vgs::supported(&orb.aos))
    {
        return None;
    }
    let u = match &orb.mo {
        Some(c) => c.t().dot(col),
        None => col.clone(),
    };
    let n_gto = orb.aos.num_ao;
    let n_extra = orb.num_extra_ao();
    if u.len() != n_gto + n_extra {
        return None;
    }
    let (g0, e0) = if orb.gto_first {
        (0, n_gto)
    } else {
        (n_extra, 0)
    };
    let np = r.nrows();
    let mut out = vec![[Complex64::new(0.0, 0.0); 3]; np];
    if n_gto > 0 {
        let aos = &orb.aos;
        let (r_w, restore) = wrap_restore(orb, r);
        let (mut w_re, mut w_im) = (vec![0.0; aos.num_ao_kernel], vec![0.0; aos.num_ao_kernel]);
        for kap in 0..aos.num_ao_kernel {
            let x = u[g0 + aos.kernel_to_home[kap]] * orb.image_phase[kap];
            w_re[kap] = x.re;
            w_im[kap] = x.im;
        }
        let p: Vec<[f64; 3]> = (0..np)
            .map(|v| [r_w[[v, 0]], r_w[[v, 1]], r_w[[v, 2]]])
            .collect();
        let run = |w: &[f64]| {
            let mut o = crate::mesh_vgs::MeshVgs::new(np, None, false, None);
            crate::mesh_vgs::mesh_vgs(
                aos,
                &p,
                crate::mesh_vgs::Weights::Kernel(w),
                crate::ao_cart_deriv::AoScreen::MeshValue,
                None,
                &mut o,
            );
            o.grad
        };
        let (gr, gi) = (run(&w_re), run(&w_im));
        for v in 0..np {
            for d in 0..3 {
                out[v][d] += restore[v] * Complex64::new(gr[v][d], gi[v][d]);
            }
        }
    }
    if n_extra > 0 {
        let grads: [Array2<Complex64>; 3] = if let Some(pw) = &orb.pw {
            crate::complex_ao::compute_pw_vgl_complex(pw, r).1
        } else if let Some(bs) = &orb.bspline {
            bs.value_grad_lap(r).1
        } else {
            return None;
        };
        for v in 0..np {
            for d in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for a in 0..n_extra {
                    acc += u[e0 + a] * grads[d][[a, v]];
                }
                out[v][d] += acc;
            }
        }
    }
    Some(out)
}

/// Leaves the fused complex moved-electron contraction [`moved_group_complex`] covers: no tiled
/// orbitals, a gto block (when non-empty) evaluated in an fp64 value zone (`aos.value_zone`; the
/// Bloch `ao_bloch_eval` zone is fp64 in both precision modes; an empty gto block of a pure
/// plane-wave / B-spline leaf is not evaluated, so its zone is not checked, as in
/// [`ComplexOrb::mesh_wdot_ok`]), at most one plane-wave / B-spline
/// child, a gto block (when present) of the shell-table Cartesian form with one Bloch phase per
/// kernel AO, and an MO layer (when present) of AO rank. An fp32 `ao_eval` zone is covered: the
/// plane-wave / B-spline child then takes the value convention of the per-candidate fallback
/// (`ComplexOrb::value_grad_lap`: fp64 body, value rounded to fp32, gradient fp64).
pub(crate) fn moved_group_ok(orb: &ComplexOrb) -> bool {
    orb.tiled.is_none()
        && std::env::var("JQMC_RS_MESH_TABLES").is_err()
        && (orb.aos.num_ao == 0 || !orb.precision_policy.is_f32(orb.aos.value_zone))
        && !(orb.pw.is_some() && orb.bspline.is_some())
        && (orb.aos.num_ao == 0
            || (crate::mesh_vgs::supported(&orb.aos)
                && orb.image_phase.len() == orb.aos.num_ao_kernel))
        && orb
            .mo
            .as_ref()
            .is_none_or(|c| c.ncols() == orb.num_ao_total())
}

/// Moved-electron fields of one complex mesh group ([`moved_group_complex`]), per point `v`.
pub(crate) struct MovedGroupC {
    /// `Sum_a col[a] mo_a(r_v)`.
    pub val: Vec<Complex64>,
    /// `Sum_a col[a] grad mo_a(r_v)`.
    pub grad: Vec<[Complex64; 3]>,
    /// gto-block part of `grad` restricted to the shells of each home atom (`np * n_atom`,
    /// point-major), when requested.
    pub atom: Vec<[Complex64; 3]>,
    /// gto-block strain virial of `grad` (the per-point [`det_strain_side_complex_col`] field),
    /// when requested.
    pub strain: Vec<[[Complex64; 3]; 3]>,
    /// Pre-MO complex AO values `(num_ao_total, np)` in the composite (`gto_first`) order.
    pub aotab: Array2<Complex64>,
}

/// One fused pass over the moved electron's candidate points `pts` (RAW positions): with the
/// AO-rank weights `u = C^T col` of one MO-rank column `col`, the contracted value and gradient
/// (`val`, `grad`), optionally the atom-resolved gradient (`n_atom > 0`, needs `geom` with the
/// home atoms) and the strain virial (`want_strain`, needs `geom` with the kernel offsets), and
/// the AO value table. The gto block is evaluated at the home-wrapped points times the wrap
/// restore phase, with the Bloch-phased kernel weights split into real and imaginary weight
/// sets of one [`crate::mesh_vgs::mesh_vgs_multi`] pass (the folded value table carries the
/// phase components), under `screen`; the plane-wave / B-spline child from its own tables at
/// the raw points. `None` when [`moved_group_ok`] declines the leaf.
pub(crate) fn moved_group_complex(
    orb: &ComplexOrb,
    pts: &[[f64; 3]],
    col: &numpy::ndarray::Array1<Complex64>,
    geom: Option<&crate::mesh_vgs::ShellGeom>,
    n_atom: usize,
    want_strain: bool,
    screen: crate::ao_cart_deriv::AoScreen,
) -> Option<MovedGroupC> {
    if !moved_group_ok(orb) {
        return None;
    }
    let u = match &orb.mo {
        Some(c) => c.t().dot(col),
        None => col.clone(),
    };
    let n_gto = orb.aos.num_ao;
    let n_extra = orb.num_extra_ao();
    if u.len() != n_gto + n_extra {
        return None;
    }
    let (g0, e0) = if orb.gto_first {
        (0, n_gto)
    } else {
        (n_extra, 0)
    };
    let np = pts.len();
    let z = Complex64::new(0.0, 0.0);
    let mut out = MovedGroupC {
        val: vec![z; np],
        grad: vec![[z; 3]; np],
        atom: vec![[z; 3]; if n_gto > 0 { np * n_atom } else { 0 }],
        strain: if want_strain && n_gto > 0 {
            vec![[[z; 3]; 3]; np]
        } else {
            Vec::new()
        },
        aotab: Array2::<Complex64>::zeros((n_gto + n_extra, np)),
    };
    let rmat = Array2::from_shape_fn((np, 3), |(v, c)| pts[v][c]);
    if n_gto > 0 {
        if (n_atom > 0 || want_strain) && geom.is_none() {
            return None;
        }
        let aos = &orb.aos;
        let (r_w, restore) = wrap_restore(orb, &rmat.view());
        let nk = aos.num_ao_kernel;
        let (mut w_re, mut w_im) = (vec![0.0; nk], vec![0.0; nk]);
        let (mut ph_re, mut ph_im) = (vec![0.0; nk], vec![0.0; nk]);
        for kap in 0..nk {
            let ph = orb.image_phase[kap];
            let x = u[g0 + aos.kernel_to_home[kap]] * ph;
            w_re[kap] = x.re;
            w_im[kap] = x.im;
            ph_re[kap] = ph.re;
            ph_im[kap] = ph.im;
        }
        let p: Vec<[f64; 3]> = (0..np)
            .map(|v| [r_w[[v, 0]], r_w[[v, 1]], r_w[[v, 2]]])
            .collect();
        let na = (n_atom > 0).then_some(n_atom);
        let mut outs = [
            crate::mesh_vgs::MeshVgs::new(np, na, want_strain, Some(n_gto)),
            crate::mesh_vgs::MeshVgs::new(np, na, want_strain, Some(n_gto)),
        ];
        crate::mesh_vgs::mesh_vgs_multi(
            aos,
            &p,
            &[
                crate::mesh_vgs::Weights::Kernel(&w_re),
                crate::mesh_vgs::Weights::Kernel(&w_im),
            ],
            &[Some(&ph_re), Some(&ph_im)],
            screen,
            geom,
            &mut outs,
        );
        let [o_re, o_im] = &outs;
        for (v, restore_v) in restore[..np].iter().enumerate() {
            let rs = *restore_v;
            out.val[v] = rs * Complex64::new(o_re.val[v], o_im.val[v]);
            for d in 0..3 {
                out.grad[v][d] = rs * Complex64::new(o_re.grad[v][d], o_im.grad[v][d]);
            }
            for a in 0..n_atom {
                let i = v * n_atom + a;
                for d in 0..3 {
                    out.atom[i][d] = rs * Complex64::new(o_re.atom[i][d], o_im.atom[i][d]);
                }
            }
            if want_strain {
                for c in 0..3 {
                    for d in 0..3 {
                        out.strain[v][c][d] =
                            rs * Complex64::new(o_re.strain[v][c][d], o_im.strain[v][c][d]);
                    }
                }
            }
            for h in 0..n_gto {
                out.aotab[[g0 + h, v]] =
                    rs * Complex64::new(o_re.vtab[h * np + v], o_im.vtab[h * np + v]);
            }
        }
    }
    if n_extra > 0 {
        let (vals, grads): (Array2<Complex64>, [Array2<Complex64>; 3]) = if let Some(pw) = &orb.pw {
            if pw.phase_is_sin.is_none() {
                // Bloch plane waves e^{i gk.r}: the vectorized value kernel, grad = i gk value.
                let mut v =
                    crate::complex_ao::compute_pw_complex_with_policy(pw, &rmat.view(), false);
                let ii = Complex64::new(0.0, 1.0);
                let g = [0, 1, 2].map(|d| {
                    Array2::from_shape_fn(v.raw_dim(), |(a, e)| ii * pw.gk[a][d] * v[[a, e]])
                });
                // fp32 `ao_eval`: round the completed value as `compute_pw_vgl_complex` does.
                if pw.precision_policy.is_f32("ao_eval") {
                    v.mapv_inplace(|x| Complex64::new((x.re as f32) as f64, (x.im as f32) as f64));
                }
                (v, g)
            } else {
                let (v, g, _) = crate::complex_ao::compute_pw_vgl_complex(pw, &rmat.view());
                (v, g)
            }
        } else if let Some(bs) = &orb.bspline {
            bs.value_grad(&rmat.view())
        } else {
            return None;
        };
        for v in 0..np {
            let mut acc = z;
            let mut ga = [z; 3];
            for a in 0..n_extra {
                let ua = u[e0 + a];
                acc += ua * vals[[a, v]];
                for d in 0..3 {
                    ga[d] += ua * grads[d][[a, v]];
                }
                out.aotab[[e0 + a, v]] = vals[[a, v]];
            }
            out.val[v] += acc;
            for (d, ga_d) in ga.iter().enumerate() {
                out.grad[v][d] += *ga_d;
            }
        }
    }
    Some(out)
}

/// Fused complex det kinetic / Pulay contraction of a Cartesian Bloch gto block at the RAW
/// electrons `r` (wrapped inside, times the wrap restore): with the AO-rank (gto rows) complex
/// cotangents of the orbital value `cm_o`, optionally of the Laplacian `cm_l` and of the three
/// gradient components `cm_g`, returns the REAL parts of
/// `elec[k] = Sum_kap restore_k cm[home(kap),k] phase_kap [grad phi + ...]_kap(r_w,k)` and of
/// its home-atom sums `atom[a]` (`geom` carries the home atoms), from one
/// [`crate::mesh_vgs::kin_contract_multi`] pass (two weight sets: the real part of the phased
/// weights) under the unscreened AO model of [`ComplexOrb::ao_grad`]. `None` when the gto block
/// is not covered.
pub(crate) fn fused_complex_kin_force(
    orb: &ComplexOrb,
    r: &ArrayView2<f64>,
    cm_o: &Array2<Complex64>,
    cm_l: Option<&Array2<Complex64>>,
    cm_g: Option<[&Array2<Complex64>; 3]>,
    geom: &crate::mesh_vgs::ShellGeom,
    n_atom: usize,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let aos = &orb.aos;
    if orb.tiled.is_some()
        || aos.num_ao == 0
        || std::env::var("JQMC_RS_MESH_TABLES").is_ok()
        || !crate::mesh_vgs::supported(aos)
        || cm_o.nrows() != aos.num_ao
        || orb.image_phase.len() != aos.num_ao_kernel
    {
        return None;
    }
    let (r_w, restore) = wrap_restore(orb, r);
    let n = r_w.nrows();
    let split = |cm: &Array2<Complex64>| -> (Vec<f64>, Vec<f64>) {
        let (mut re, mut im) = (vec![0.0; cm.nrows() * n], vec![0.0; cm.nrows() * n]);
        for mu in 0..cm.nrows() {
            for k in 0..n {
                let x = cm[[mu, k]] * restore[k];
                re[mu * n + k] = x.re;
                im[mu * n + k] = x.im;
            }
        }
        (re, im)
    };
    let o = split(cm_o);
    let l = cm_l.map(split);
    let g = cm_g.map(|g| [split(g[0]), split(g[1]), split(g[2])]);
    let ph_re: Vec<f64> = orb.image_phase.iter().map(|z| z.re).collect();
    let ph_im: Vec<f64> = orb.image_phase.iter().map(|z| z.im).collect();
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|k| [r_w[[k, 0]], r_w[[k, 1]], r_w[[k, 2]]])
        .collect();
    fn sel(x: &(Vec<f64>, Vec<f64>), im_w: bool) -> &[f64] {
        if im_w {
            &x.1
        } else {
            &x.0
        }
    }
    // real part: (W_re, ph_re) - (W_im, ph_im)
    let ws: Vec<crate::mesh_vgs::KinWeights> = [(false, &ph_re), (true, &ph_im)]
        .iter()
        .map(|&(im_w, ph)| crate::mesh_vgs::KinWeights {
            orb: sel(&o, im_w),
            lap: l.as_ref().map(|x| sel(x, im_w)),
            gou: g
                .as_ref()
                .map(|g| [sel(&g[0], im_w), sel(&g[1], im_w), sel(&g[2], im_w)]),
            kscale: Some(ph.as_slice()),
        })
        .collect();
    if geom.atom.iter().any(|&a| a >= n_atom) {
        return None;
    }
    let mut outs: Vec<crate::mesh_vgs::KinOut> = (0..2)
        .map(|_| crate::mesh_vgs::KinOut {
            elec: vec![[0.0; 3]; n],
            atom: vec![[0.0; 3]; n_atom],
            strain: None,
        })
        .collect();
    crate::mesh_vgs::kin_contract_multi(
        aos,
        &pts,
        &ws,
        crate::ao_cart_deriv::AoScreen::None,
        crate::ao_cart_deriv::AoScreen::None,
        Some(geom),
        &mut outs,
    );
    let elec = Array2::from_shape_fn((n, 3), |(k, c)| outs[0].elec[k][c] - outs[1].elec[k][c]);
    let atom = Array2::from_shape_fn((n_atom, 3), |(a, c)| {
        outs[0].atom[a][c] - outs[1].atom[a][c]
    });
    Some((elec, atom))
}

/// Fused form of [`complex_field_stress`] for the value (`cm_o`, AO gradient), Laplacian
/// (`cm_l`, gradient of the Laplacian) and gradient (`cm_g`, Hessian columns) cotangents of a
/// Cartesian Bloch gto block at the WRAPPED electrons `r_w` with restore phases `restore`:
/// `Sum_{k,kap} restore_k cm[home(kap),k] phase_kap o_{kap,k} (x) (r_w,k - C_kap + R_at - R_home)`
/// from the real fused kinetic contraction ([`crate::mesh_vgs::kin_contract`]; two runs for the
/// real part, four with the imaginary part) instead of the kernel-rank AO derivative tables.
/// The AO model is the unscreened one of [`grad_fields`] / [`gradlap_fields`] /
/// [`hessian_fields`]. `None` when the gto block is not covered.
#[allow(clippy::too_many_arguments)]
fn fused_complex_field_stress(
    orb: &ComplexOrb,
    r_w: &Array2<f64>,
    restore: &[Complex64],
    cm_o: &Array2<Complex64>,
    cm_l: Option<&Array2<Complex64>>,
    cm_g: Option<[&Array2<Complex64>; 3]>,
    home: &[usize],
    kern: &[usize],
    positions: &ArrayView2<f64>,
    want_im: bool,
) -> Option<(Array2<f64>, Option<Array2<f64>>)> {
    let aos = &orb.aos;
    if orb.tiled.is_some()
        || aos.num_ao == 0
        || std::env::var("JQMC_RS_MESH_TABLES").is_ok()
        || !crate::mesh_vgs::supported(aos)
        || cm_o.nrows() != aos.num_ao
    {
        return None;
    }
    let geom = crate::mesh_vgs::ShellGeom::new(aos, home, Some((kern, positions)))?;
    let n = r_w.nrows();
    let split = |cm: &Array2<Complex64>| -> (Vec<f64>, Vec<f64>) {
        let (mut re, mut im) = (vec![0.0; cm.nrows() * n], vec![0.0; cm.nrows() * n]);
        for mu in 0..cm.nrows() {
            for k in 0..n {
                let x = cm[[mu, k]] * restore[k];
                re[mu * n + k] = x.re;
                im[mu * n + k] = x.im;
            }
        }
        (re, im)
    };
    let o = split(cm_o);
    let l = cm_l.map(split);
    let g = cm_g.map(|g| [split(g[0]), split(g[1]), split(g[2])]);
    let ph_re: Vec<f64> = orb.image_phase.iter().map(|z| z.re).collect();
    let ph_im: Vec<f64> = orb.image_phase.iter().map(|z| z.im).collect();
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|k| [r_w[[k, 0]], r_w[[k, 1]], r_w[[k, 2]]])
        .collect();
    fn sel(x: &(Vec<f64>, Vec<f64>), im_w: bool) -> &[f64] {
        if im_w {
            &x.1
        } else {
            &x.0
        }
    }
    // real part: (W_re, ph_re) - (W_im, ph_im); imaginary part: (W_re, ph_im) + (W_im, ph_re)
    let mut specs: Vec<(bool, &[f64])> = vec![(false, &ph_re), (true, &ph_im)];
    if want_im {
        specs.push((false, &ph_im));
        specs.push((true, &ph_re));
    }
    let ws: Vec<crate::mesh_vgs::KinWeights> = specs
        .iter()
        .map(|&(im_w, ph)| crate::mesh_vgs::KinWeights {
            orb: sel(&o, im_w),
            lap: l.as_ref().map(|x| sel(x, im_w)),
            gou: g
                .as_ref()
                .map(|g| [sel(&g[0], im_w), sel(&g[1], im_w), sel(&g[2], im_w)]),
            kscale: Some(ph),
        })
        .collect();
    let mut outs: Vec<crate::mesh_vgs::KinOut> = (0..ws.len())
        .map(|_| crate::mesh_vgs::KinOut {
            elec: vec![[0.0; 3]; n],
            atom: Vec::new(),
            strain: Some([[0.0; 3]; 3]),
        })
        .collect();
    crate::mesh_vgs::kin_contract_multi(
        aos,
        &pts,
        &ws,
        crate::ao_cart_deriv::AoScreen::None,
        crate::ao_cart_deriv::AoScreen::None,
        Some(&geom),
        &mut outs,
    );
    let st: Vec<[[f64; 3]; 3]> = outs.iter().map(|o| o.strain.unwrap()).collect();
    let to = |a: [[f64; 3]; 3], b: [[f64; 3]; 3], sgn: f64| {
        Array2::from_shape_fn((3, 3), |(c, d)| a[c][d] + sgn * b[c][d])
    };
    let re = to(st[0], st[1], -1.0);
    let im = want_im.then(|| to(st[2], st[3], 1.0));
    Some((re, im))
}

/// Complex det electron drift `(dln det/dr_up (N_up,3), dln det/dr_dn (N_dn,3))` from a complex
/// geminal forward (orbital gradients contracted with the forward's `W G^-1` / `G^-1 Q` bars).
fn det_drift_from_fwd_c(
    f: &crate::complex_det::ComplexGemFwd,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let n_up = f.ou.ncols();
    let n_dn = f.n_dn;
    let mut gu = Array2::<Complex64>::zeros((n_up, 3));
    let mut gd = Array2::<Complex64>::zeros((n_dn, 3));
    for c in 0..3 {
        for k in 0..n_up {
            let mut acc = Complex64::new(0.0, 0.0);
            for a in 0..f.gou[c].nrows() {
                acc += f.gou[c][[a, k]] * f.wg[[a, k]];
            }
            gu[[k, c]] = acc;
        }
        for l in 0..n_dn {
            let mut acc = Complex64::new(0.0, 0.0);
            for b in 0..f.godn[c].nrows() {
                acc += f.godn[c][[b, l]] * f.gq[[l, b]];
            }
            gd[[l, c]] = acc;
        }
    }
    (gu, gd)
}

/// Complex Pulay stress `dln|Psi|/deps` (real 3x3) for the twisted geminal.
/// `jas = None` -> the pure determinant Pulay stress (jnone). WITH a Jastrow the periodic
/// VALUE strain of every Jastrow metric is added, exactly as the real
/// `stress_pbc::stress_lnPsi_pbc` does and as `_apply_strain` deforms them: J1/J2 via the
/// cell-straining Williamson kernels, DTN via `dtn_value_strain`, and the complex-Bloch J3
/// via `j3_value_strain_complex` (`A = Re(complex sphe AO)`). The Jastrow value is REAL, so
/// it enters the (still holomorphic) determinant stress as a real diagonal before the final
/// `Re` -- the `Re(A)Re(B) != Re(A*B)` trap does not arise (no product of two complex factors).
#[allow(clippy::too_many_arguments)]
pub fn grad_lnpsi_wrt_strain_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    jas: Option<&crate::data::JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3_jm: Option<&ArrayView2<f64>>,
    j3_home: Option<&[usize]>,
    j3_kernel: Option<&[usize]>,
) -> Array2<f64> {
    let _ = cell;
    let mut out = pbc_det_pulay_stress_complex(
        gem, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, None,
    )
    .mapv(|x| x.re);
    if let Some(j) = jas {
        crate::stress_pbc::jastrow_strain(
            j.one_body.as_ref(),
            j.two_body.as_ref(),
            r_up,
            r_dn,
            &mut out,
        );
        if let Some(d) = &j.dtn {
            out += &crate::stress_pbc::dtn_value_strain(d, r_up, r_dn);
        }
    }
    if let (Some(o), Some(jm), Some(h), Some(k)) = (j3c, j3_jm, j3_home, j3_kernel) {
        out += &j3_value_strain_complex(o, jm, r_up, r_dn, positions, h, k);
    }
    out
}

/// Complex kinetic stress `dT/deps` (real 3x3) for the twisted geminal without Jastrow.
#[allow(clippy::too_many_arguments)]
pub fn grad_t_wrt_strain_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
) -> Array2<f64> {
    let zu = Array2::<f64>::zeros((r_up.nrows(), 3));
    let zd = Array2::<f64>::zeros((r_dn.nrows(), 3));
    let b = complex_geminal_reverse(gem, r_up, r_dn, &zu, &zd, None);
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    let do_side = |orb: &ComplexOrb,
                   r: &ArrayView2<f64>,
                   bar_orb: &Array2<Complex64>,
                   bar_lap: &Array2<Complex64>,
                   home: &[usize],
                   kern: &[usize],
                   acc: &mut Array2<Complex64>| {
        if let Some(tiled) = &orb.tiled {
            *acc += &tiled.field_vjp(
                r,
                bar_orb,
                None,
                Some(bar_lap),
                crate::orbital_tiling::TiledCoordinate::Strain,
            );
            return;
        }
        let (r_w, restore) = wrap_restore(orb, r); // wrap + Bloch restore (jqmc compute_AOs convention)
        let cm_o = cm_project(orb, bar_orb);
        let (gf, gk) = grad_fields(orb, &r_w);
        complex_field_stress(
            &orb.aos,
            &cm_o,
            &gf,
            &gk,
            &orb.image_phase,
            &restore,
            &r_w,
            home,
            kern,
            positions,
            acc,
        );
        let cm_l = cm_project(orb, bar_lap);
        let (gfl, gkl) = gradlap_fields(orb, &r_w);
        complex_field_stress(
            &orb.aos,
            &cm_l,
            &gfl,
            &gkl,
            &orb.image_phase,
            &restore,
            &r_w,
            home,
            kern,
            positions,
            acc,
        );
    };
    do_side(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        home_up,
        kernel_up,
        &mut acc,
    );
    do_side(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        home_dn,
        kernel_dn,
        &mut acc,
    );
    acc.mapv(|x| x.re)
}

/// Complex withJ kinetic stress `dT_withJ/deps` (real 3x3). Det side = bar_orb +
/// bar_lap + **bar_gou** field strains (gou = the gradient cotangent contracted with
/// `d(grad_e AO)/deps`, i.e. the e-th Hessian column field stress; needs the sphe AO
/// Hessian). complex_geminal_reverse is fed the REAL Jastrow drift `gj` so bar_orb/
/// bar_lap/bar_gou carry the det-Jastrow coupling. J1 and J2 use their cell-straining
/// Williamson kernels with cot = gJ + Re(gD).
/// Mirrors real `grad_T_pbc_wrt_strain`. J3 (complex Bloch sphe) not handled here.
#[allow(clippy::too_many_arguments)]
pub fn grad_t_wrt_strain_complex_withj(
    gem: &ComplexGeminal,
    j1: Option<&crate::data::JastrowOneBodyData>,
    j2: Option<&crate::data::JastrowTwoBodyData>,
    jastrow: Option<&crate::data::JastrowData>,
    j3_orb: Option<&crate::determinant::OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    j3c: Option<&ComplexOrb>,
    j3_home: Option<&[usize]>,
    j3_kernel: Option<&[usize]>,
) -> Array2<f64> {
    let _ = (j3_orb, j3_jm); // J3 drift is the complex-sphe path below, not the real cart one
                             // J1/J2 jastrow drift (real); the COMPLEX J3 drift is added separately (the real
                             // three-body would evaluate the complex Bloch sphe leaf as a bare real AO -> wrong).
    let (mut gj_up, mut gj_dn) =
        crate::param_grad_el::jastrow_grads(jastrow, None, None, r_up, r_dn);
    if let (Some(j3o), Some(jm)) = (j3c, j3_jm) {
        let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(j3o, jm, r_up, r_dn);
        gj_up += &g3u;
        gj_dn += &g3d;
    }
    let b = complex_geminal_reverse(gem, r_up, r_dn, &gj_up, &gj_dn, None);
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    let do_side = |orb: &ComplexOrb,
                   r: &ArrayView2<f64>,
                   bar_orb: &Array2<Complex64>,
                   bar_lap: &Array2<Complex64>,
                   bar_gou: &[Array2<Complex64>; 3],
                   home: &[usize],
                   kern: &[usize],
                   acc: &mut Array2<Complex64>| {
        if let Some(tiled) = &orb.tiled {
            *acc += &tiled.field_vjp(
                r,
                bar_orb,
                Some(bar_gou),
                Some(bar_lap),
                crate::orbital_tiling::TiledCoordinate::Strain,
            );
            return;
        }
        let (r_w, restore) = wrap_restore(orb, r); // wrap + Bloch restore (jqmc compute_AOs convention)
                                                   // Composite (pgto(+)pw): cm_project spans the full AO rank = GTO (image-
                                                   // tiled) + PW (untiled). Slice the GTO rows for the field-strain (its
                                                   // `home` map is the tiled GTO count) and the PW rows for the PW block.
                                                   // Non-composite (pw=None): full cm, no slice, no PW (backward compatible).
                                                   // Slice the GTO rows out of the full AO-rank cotangent. The "extra" block is EITHER a PW
                                                   // child OR a B-spline child (mutually exclusive); for a pure PW / B-spline leaf num_gto=0
                                                   // so this yields a 0-row array and the gto field stress is a no-op (mirrors the complex
                                                   // geminal reverse's `num_extra` slice). Bare pgto (no extra) => full cm passthrough.
        let num_extra = orb
            .pw
            .as_ref()
            .map(|p| p.num_ao)
            .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
            .unwrap_or(0);
        let gsl = |cm: &Array2<Complex64>| -> Array2<Complex64> {
            if num_extra == 0 {
                return cm.clone();
            }
            let total = cm.nrows();
            let g0 = if orb.gto_first { 0 } else { num_extra };
            cm.slice(numpy::ndarray::s![g0..g0 + (total - num_extra), ..])
                .to_owned()
        };
        let psl = |cm: &Array2<Complex64>, pw: &ComplexPwData| -> Array2<Complex64> {
            let (total, npw) = (cm.nrows(), pw.num_ao);
            let p0 = if orb.gto_first { total - npw } else { 0 };
            cm.slice(numpy::ndarray::s![p0..p0 + npw, ..]).to_owned()
        };
        // bar_orb * dAO/deps  (PW value is strain-invariant -> no PW value term)
        let cm_o = cm_project(orb, bar_orb);
        let (gf, gk) = grad_fields(orb, &r_w);
        complex_field_stress(
            &orb.aos,
            &gsl(&cm_o),
            &gf,
            &gk,
            &orb.image_phase,
            &restore,
            &r_w,
            home,
            kern,
            positions,
            acc,
        );
        // bar_lap * d(grad^2AO)/deps
        let cm_l = cm_project(orb, bar_lap);
        let (gfl, gkl) = gradlap_fields(orb, &r_w);
        complex_field_stress(
            &orb.aos,
            &gsl(&cm_l),
            &gfl,
            &gkl,
            &orb.image_phase,
            &restore,
            &r_w,
            home,
            kern,
            positions,
            acc,
        );
        // bar_gou[e] * d(grad_e AO)/deps  (e-th Hessian column field stress)
        let (hf, hk) = hessian_fields(orb, &r_w);
        let mut cm_gs: Vec<Array2<Complex64>> = Vec::with_capacity(3);
        for e in 0..3 {
            let cm_g = cm_project(orb, &bar_gou[e]);
            let gfe = [hf[0][e].clone(), hf[1][e].clone(), hf[2][e].clone()];
            let gke = [hk[0][e].clone(), hk[1][e].clone(), hk[2][e].clone()];
            complex_field_stress(
                &orb.aos,
                &gsl(&cm_g),
                &gfe,
                &gke,
                &orb.image_phase,
                &restore,
                &r_w,
                home,
                kern,
                positions,
                acc,
            );
            cm_gs.push(cm_g);
        }
        // PW kinetic stress (composite): lap + gou. PW uses the RAW (unwrapped) r --
        // a plane wave is exact at any coordinate (no image tiling), matching the JAX
        // ``_pw_kinetic_stress_contrib`` raw-r convention.
        if let Some(pw) = &orb.pw {
            pw_lap_stress_complex(pw, &psl(&cm_l, pw), r, acc);
            let cm_gou_pw = [psl(&cm_gs[0], pw), psl(&cm_gs[1], pw), psl(&cm_gs[2], pw)];
            pw_gou_stress_complex(pw, &cm_gou_pw, r, acc);
        }
        // B-spline kinetic stress (pure-B-spline det or pgto(+)bspline composite): metric response
        // via the complex Bloch cart fields (value frac-invariant), sharing `cm_l`/`cm_gs`.
        if orb.bspline.is_some() {
            let cm_gou_full = [cm_gs[0].clone(), cm_gs[1].clone(), cm_gs[2].clone()];
            bspline_kstress_side_complex(orb, &cm_l, &cm_gou_full, r, acc);
        }
    };
    do_side(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        &b.bar_gou,
        home_up,
        kernel_up,
        &mut acc,
    );
    do_side(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        &b.bar_godn,
        home_dn,
        kernel_dn,
        &mut acc,
    );
    let mut out = acc.mapv(|x| x.re);
    // Jastrow-side stress: cot = gJ_total + Re(gD) (total drift). gj_up/gj_dn already
    // include J1/J2 + the complex J3 drift.
    let dtn = jastrow.and_then(|j| j.dtn.as_ref());
    if j1.is_some() || j2.is_some() || j3c.is_some() || dtn.is_some() {
        let (gd_up, gd_dn, _, _) = grads_and_laplacian_ln_det_complex(gem, r_up, r_dn);
        let n_up = r_up.nrows();
        let n_e = n_up + r_dn.nrows();
        let mut cot = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                cot[[e, c]] = if e < n_up {
                    gj_up[[e, c]] + gd_up[[e, c]].re
                } else {
                    let l = e - n_up;
                    gj_dn[[l, c]] + gd_dn[[l, c]].re
                };
            }
        }
        crate::stress_pbc::jastrow_kstress(j1, j2, &cot, r_up, r_dn, &mut out);
        // DTN kinetic stress (u+chi+f) -- the DTN's own laplacian/drift-cross term
        // (its drift is already in `cot`; this adds the dT_DTN/deps contribution).
        if let Some(d) = dtn {
            crate::stress_pbc::dtn_kstress(d, &cot, r_up, r_dn, &mut out);
        }
        // J3 (complex Bloch sphe): bg = -cot.
        if let (Some(j3o), Some(jm), Some(hm), Some(km)) = (j3c, j3_jm, j3_home, j3_kernel) {
            let bg_up = cot.slice(numpy::ndarray::s![0..n_up, ..]).mapv(|x| -x);
            let bg_dn = cot.slice(numpy::ndarray::s![n_up.., ..]).mapv(|x| -x);
            j3_pbc_kstress_complex(
                j3o, jm, r_up, r_dn, &bg_up, &bg_dn, None, positions, hm, km, &mut out,
            );
        }
    }
    out
}

/// Complex-twist withJ periodic kinetic STRESS `dT/deps` (real 3x3), per-electron S'-weighted
/// (`w=None` => full). Det side = complex weighted geminal reverse -> bar_orb/bar_lap/bar_gou
/// field strains (Re); J1/J2 via `jastrow_kstress_w`; DTN via `dtn_kstress_w`
/// (cot = gJ + Re(gD), UNWEIGHTED; the mask `w` is
/// applied INSIDE the weighted kernels). J3 (complex Bloch sphe) via
/// `j3_pbc_kstress_complex` (masked). The masked-kinetic core of the complex GFMC stress.
/// Mirrors `grad_t_wrt_strain_complex_withj`.
#[allow(clippy::too_many_arguments)]
pub fn complex_kinetic_stress_w(
    gem: &ComplexGeminal,
    jas: &crate::data::JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    j3c: Option<&ComplexOrb>,
    j3_jm: Option<&ArrayView2<f64>>,
    j3_home: Option<&[usize]>,
    j3_kernel: Option<&[usize]>,
    w: Option<&[f64]>,
    gi: Option<&ArrayView2<Complex64>>,
    // The base complex geminal forward when the caller already has it (at `gi`).
    fwd: Option<&crate::complex_det::ComplexGemFwd>,
) -> Array2<f64> {
    let _ = cell;
    let n_up = r_up.nrows();
    let (mut gj_up, mut gj_dn) =
        crate::param_grad_el::jastrow_grads(Some(jas), None, None, r_up, r_dn);
    if let (Some(j3o), Some(jm)) = (j3c, j3_jm) {
        let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(j3o, jm, r_up, r_dn);
        gj_up += &g3u;
        gj_dn += &g3d;
    }
    let (wu, wd) = match w {
        None => (None, None),
        Some(w) => (Some(w[0..n_up].to_vec()), Some(w[n_up..].to_vec())),
    };
    let f_own;
    let f = match fwd {
        Some(f) => f,
        None => {
            f_own = crate::complex_det::complex_gem_forward(gem, r_up, r_dn, gi);
            &f_own
        }
    };
    let b = crate::complex_det::complex_geminal_reverse_from_fwd(
        gem,
        f,
        &gj_up,
        &gj_dn,
        wu.as_deref(),
        wd.as_deref(),
    );
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    let do_side = |orb: &ComplexOrb,
                   r: &ArrayView2<f64>,
                   bar_orb: &Array2<Complex64>,
                   bar_lap: &Array2<Complex64>,
                   bar_gou: &[Array2<Complex64>; 3],
                   home: &[usize],
                   kern: &[usize],
                   acc: &mut Array2<Complex64>| {
        if let Some(tiled) = &orb.tiled {
            *acc += &tiled.field_vjp(
                r,
                bar_orb,
                Some(bar_gou),
                Some(bar_lap),
                crate::orbital_tiling::TiledCoordinate::Strain,
            );
            return;
        }
        let (r_w, restore) = wrap_restore(orb, r); // wrap + Bloch restore (jqmc compute_AOs convention)
                                                   // Slice the GTO rows out of the full AO-rank cotangent. The "extra" block is EITHER a PW
                                                   // child OR a B-spline child (mutually exclusive); for a pure PW / B-spline leaf num_gto=0
                                                   // so this yields a 0-row array and the gto field stress is a no-op (mirrors the complex
                                                   // geminal reverse's `num_extra` slice). Bare pgto (no extra) => full cm passthrough.
        let num_extra = orb
            .pw
            .as_ref()
            .map(|p| p.num_ao)
            .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
            .unwrap_or(0);
        let gsl = |cm: &Array2<Complex64>| -> Array2<Complex64> {
            if num_extra == 0 {
                return cm.clone();
            }
            let total = cm.nrows();
            let g0 = if orb.gto_first { 0 } else { num_extra };
            cm.slice(numpy::ndarray::s![g0..g0 + (total - num_extra), ..])
                .to_owned()
        };
        let psl = |cm: &Array2<Complex64>, pw: &ComplexPwData| -> Array2<Complex64> {
            let (total, npw) = (cm.nrows(), pw.num_ao);
            let p0 = if orb.gto_first { total - npw } else { 0 };
            cm.slice(numpy::ndarray::s![p0..p0 + npw, ..]).to_owned()
        };
        let cm_o = cm_project(orb, bar_orb);
        let cm_l = cm_project(orb, bar_lap);
        let cm_gs: Vec<Array2<Complex64>> = (0..3).map(|e| cm_project(orb, &bar_gou[e])).collect();
        // Fused gto block (only the real part of `acc` is read below).
        let fused = fused_complex_field_stress(
            orb,
            &r_w,
            &restore,
            &gsl(&cm_o),
            Some(&gsl(&cm_l)),
            Some([&gsl(&cm_gs[0]), &gsl(&cm_gs[1]), &gsl(&cm_gs[2])]),
            home,
            kern,
            positions,
            false,
        );
        if let Some((re, _)) = fused {
            for c in 0..3 {
                for d in 0..3 {
                    acc[[c, d]] += Complex64::new(re[[c, d]], 0.0);
                }
            }
        } else {
            let (gf, gk) = grad_fields(orb, &r_w);
            complex_field_stress(
                &orb.aos,
                &gsl(&cm_o),
                &gf,
                &gk,
                &orb.image_phase,
                &restore,
                &r_w,
                home,
                kern,
                positions,
                acc,
            );
            let (gfl, gkl) = gradlap_fields(orb, &r_w);
            complex_field_stress(
                &orb.aos,
                &gsl(&cm_l),
                &gfl,
                &gkl,
                &orb.image_phase,
                &restore,
                &r_w,
                home,
                kern,
                positions,
                acc,
            );
            let (hf, hk) = hessian_fields(orb, &r_w);
            for e in 0..3 {
                let cm_g = &cm_gs[e];
                let gfe = [hf[0][e].clone(), hf[1][e].clone(), hf[2][e].clone()];
                let gke = [hk[0][e].clone(), hk[1][e].clone(), hk[2][e].clone()];
                complex_field_stress(
                    &orb.aos,
                    &gsl(cm_g),
                    &gfe,
                    &gke,
                    &orb.image_phase,
                    &restore,
                    &r_w,
                    home,
                    kern,
                    positions,
                    acc,
                );
            }
        }
        if let Some(pw) = &orb.pw {
            pw_lap_stress_complex(pw, &psl(&cm_l, pw), r, acc);
            let cm_gou_pw = [psl(&cm_gs[0], pw), psl(&cm_gs[1], pw), psl(&cm_gs[2], pw)];
            pw_gou_stress_complex(pw, &cm_gou_pw, r, acc);
        }
        // B-spline kinetic stress (pure-B-spline det or pgto(+)bspline composite): the metric
        // response via the complex Bloch cart fields (value frac-invariant). Shares the same
        // g_inv-reused reverse cotangents (`cm_l`/`cm_gs`, full AO rank).
        if orb.bspline.is_some() {
            let cm_gou_full = [cm_gs[0].clone(), cm_gs[1].clone(), cm_gs[2].clone()];
            bspline_kstress_side_complex(orb, &cm_l, &cm_gou_full, r, acc);
        }
    };
    do_side(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        &b.bar_gou,
        home_up,
        kernel_up,
        &mut acc,
    );
    do_side(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        &b.bar_godn,
        home_dn,
        kernel_dn,
        &mut acc,
    );
    let mut out = acc.mapv(|x| x.re);
    // Jastrow-side stress: cot = gJ + Re(gD) (UNWEIGHTED); w applied inside the *_w kernels.
    let j1 = jas.one_body.as_ref();
    let j2 = jas.two_body.as_ref();
    let dtn = jas.dtn.as_ref();
    if j1.is_some() || j2.is_some() || dtn.is_some() || j3c.is_some() {
        // det drift from the same forward: gD_up[k] = Sum_a gou[a,k] (W G^-1)[a,k],
        // gD_dn[l] = Sum_b godn[b,l] (G^-1 Q)[l,b].
        let (gd_up, gd_dn) = det_drift_from_fwd_c(f);
        let n_e = n_up + r_dn.nrows();
        let mut cot = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                cot[[e, c]] = if e < n_up {
                    gj_up[[e, c]] + gd_up[[e, c]].re
                } else {
                    let l = e - n_up;
                    gj_dn[[l, c]] + gd_dn[[l, c]].re
                };
            }
        }
        // J1/J2 kinetic stress via their cell-straining Williamson kernels.
        crate::stress_pbc::jastrow_kstress_w(j1, j2, &cot, r_up, r_dn, w, &mut out);
        if let Some(d) = dtn {
            crate::stress_pbc::dtn_kstress_w(d, &cot, r_up, r_dn, w, &mut out);
        }
        // J3 (complex Bloch sphe): bg = -cot; masked via `w`. The J3 cell IS strained by the
        // twist oracle (`_apply_strain`), so use the cell-straining complex J3 kinetic stress.
        if let (Some(j3o), Some(jm)) = (j3c, j3_jm) {
            let bg_up = cot.slice(numpy::ndarray::s![0..n_up, ..]).mapv(|x| -x);
            let bg_dn = cot.slice(numpy::ndarray::s![n_up.., ..]).mapv(|x| -x);
            // A PURE (MO-wrapped) plane-wave / B-spline complex J3 leaf (home/kernel None) has
            // orbital rank != AO rank, which the AO-rank-sliced `j3_pbc_kstress_complex` cannot
            // address; its kinetic SELF-strain is served by the leaf-owning
            // `j3_leaf_kstress_side_complex` (no atom-centered home/kernel needed).
            if j3o.aos.num_ao == 0 && (j3o.bspline.is_some() || j3o.pw.is_some()) {
                j3_leaf_kstress_side_complex(j3o, jm, r_up, r_dn, &bg_up, &bg_dn, w, &mut out);
            } else if let (Some(hm), Some(km)) = (j3_home, j3_kernel) {
                j3_pbc_kstress_complex(
                    j3o, jm, r_up, r_dn, &bg_up, &bg_dn, w, positions, hm, km, &mut out,
                );
            }
        }
    }
    out
}

/// Complex-twist ln|Psi| STRAIN (Pulay stress) `d ln Psi / d eps` (real 3x3): complex determinant
/// Pulay stress + real Jastrow value strain (J1/J2 cell-strain + DTN).
/// No complex J3 (deferred).
/// The strained-mesh sigma_PP of the complex GFMC stress S' term. `det_only` drops the Jastrow.
#[allow(clippy::too_many_arguments)]
pub fn sigma_pp_complex(
    gem: &ComplexGeminal,
    jas: &crate::data::JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    det_only: bool,
) -> Array2<f64> {
    let mut out = pbc_det_pulay_stress_complex(
        gem, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, None,
    )
    .mapv(|x| x.re);
    if !det_only {
        crate::stress_pbc::jastrow_strain(
            jas.one_body.as_ref(),
            jas.two_body.as_ref(),
            r_up,
            r_dn,
            &mut out,
        );
        if let Some(d) = &jas.dtn {
            out += &crate::stress_pbc::dtn_value_strain(d, r_up, r_dn);
        }
    }
    out
}

/// Complex Bloch-sphe J3 kinetic STRESS `dT_J3/deps` (added Re to `out`). The J3 AO
/// basis is `A=Re(complex sphe AO)` (the Jastrow is real); cotangents bar_A/bar_gO/
/// bar_lO from the j_matrix (real arithmetic, 1:1 `stress_pbc::j3_pbc_kstress`); the
/// field strain uses the COMPLEX sphe AO grad/gradlap/Hessian field stress, Re'd. No
/// sphe->cart conversion needed (complex_field_stress works at the sphe rank).
#[allow(clippy::too_many_arguments)]
pub fn j3_pbc_kstress_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    w: Option<&[f64]>,
    positions: &ArrayView2<f64>,
    home: &[usize],
    kernel: &[usize],
    out: &mut Array2<f64>,
) {
    use numpy::ndarray::{s, Axis};
    // The shared J3 basis has no Bloch phase. Its forward values and reverse
    // strain fields must use the SAME home-cell coordinates. Under an affine
    // cell/electron strain the integer image label is fixed, so straining the
    // wrapped coordinate is exactly straining r - image * cell, not holding
    // the subtracted lattice translation constant.
    let wrapped_up = j3.wrap.as_ref().map(|ctx| ctx.wrap(r_up).0);
    let wrapped_dn = j3.wrap.as_ref().map(|ctx| ctx.wrap(r_dn).0);
    let r_up = &wrapped_up
        .as_ref()
        .map_or_else(|| r_up.view(), |r| r.view());
    let r_dn = &wrapped_dn
        .as_ref()
        .map_or_else(|| r_dn.view(), |r| r.view());
    let n_up_w = r_up.nrows();
    let w_up: Option<Vec<f64>> = w.map(|w| w[0..n_up_w].to_vec());
    let w_dn: Option<Vec<f64>> = w.map(|w| w[n_up_w..].to_vec());
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    // ORBITAL-rank path (any MO layer and/or a B-spline child): cotangents at orbital rank,
    // MO-projected to AO rank (`C^T bar`) before the per-child field strains (issue #89).
    let orbital_rank = j3.j3_orbital_rank();
    // COMPOSITE pgto(+)pw J3: append the PW child Re(value/grad/lap) in gto_first order.
    // PW value is strain-INVARIANT; only its lap/gradient fields respond (pw_lap/gou_stress
    // _complex). `pw = None` => byte-identical single leaf.
    let num_pw = j3.pw.as_ref().map_or(0, |p| p.num_ao);
    let gto0 = if j3.gto_first { 0 } else { num_pw };
    let vs = |gto: &Array2<f64>, pw: &Array2<f64>| -> Array2<f64> {
        let (top, bot) = if j3.gto_first { (gto, pw) } else { (pw, gto) };
        numpy::ndarray::concatenate(Axis(0), &[top.view(), bot.view()]).unwrap()
    };
    let basis = |r: &ArrayView2<f64>| -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        if orbital_rank {
            let (v, g, l) = j3.value_grad_lap(r);
            return (re(&v), [re(&g[0]), re(&g[1]), re(&g[2])], re(&l));
        }
        let (v, gx, gy, gz, l) = compute_aos_vgl_complex(&j3.aos, &j3.image_phase, r);
        let (vg, gxg, gyg, gzg, lg) = (re(&v), re(&gx), re(&gy), re(&gz), re(&l));
        match &j3.pw {
            None => (vg, [gxg, gyg, gzg], lg),
            Some(pw) => {
                let (pv, pg, pl) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
                (
                    vs(&vg, &re(&pv)),
                    [
                        vs(&gxg, &re(&pg[0])),
                        vs(&gyg, &re(&pg[1])),
                        vs(&gzg, &re(&pg[2])),
                    ],
                    vs(&lg, &re(&pl)),
                )
            }
        }
    };
    let (a_up, go_u, lo_u) = basis(r_up);
    let (a_dn, go_d, lo_d) = basis(r_dn);
    let orb = a_up.nrows();
    let num_gto = if orbital_rank {
        j3.aos.num_ao
    } else {
        orb - num_pw
    };
    // AO-rank row offsets (orbital rank: of the MO-projected cotangent rows).
    let (gto0, pw0) = if orbital_rank {
        (
            j3.gto_ao_offset(),
            if j3.gto_first { j3.aos.num_ao } else { 0 },
        )
    } else {
        (gto0, if j3.gto_first { num_gto } else { 0 })
    };
    let (n_up, n_dn) = (a_up.ncols(), a_dn.ncols());
    let ku = crate::stress_pbc::k_strict_lower(n_up);
    let kd = crate::stress_pbc::k_strict_lower(n_dn);
    let m = j_matrix.slice(s![.., 0..orb]).to_owned();
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
                bcoef[[a, k]] = wk * (acc - 0.5 * lo[[a, k]]);
                blo[[a, k]] = -0.5 * wk * coef[[a, k]];
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
    // field strain (value bar_A + lap bar_lO + gou bar_gO) via the complex sphe AO, Re'd.
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    let do_side = |ba: &Array2<f64>,
                   blo: &Array2<f64>,
                   bgo: &[Array2<f64>; 3],
                   r: &ArrayView2<f64>,
                   acc: &mut Array2<Complex64>| {
        // wrap + Bloch restore (jqmc compute_AOs convention; the J3 cotangents are REAL, so
        // Re(cm*restore*gfold) = cm*Re(restore*gfold) -- the Re-per-orbital J3 form survives).
        let (r_w, restore) = wrap_restore(j3, r);
        // GTO block: complex sphe/cart AO field strain over the gto rows [gto0, gto0+num_gto).
        let czg = |a: &Array2<f64>| {
            if orbital_rank {
                j3.mo_project_t(a)
                    .slice(s![gto0..gto0 + num_gto, ..])
                    .to_owned()
            } else {
                a.slice(s![gto0..gto0 + num_gto, ..])
                    .mapv(|x| Complex64::new(x, 0.0))
            }
        };
        let (gf, gk) = grad_fields(j3, &r_w);
        complex_field_stress(
            &j3.aos,
            &czg(ba),
            &gf,
            &gk,
            &j3.image_phase,
            &restore,
            &r_w,
            home,
            kernel,
            positions,
            acc,
        );
        let (gfl, gkl) = gradlap_fields(j3, &r_w);
        complex_field_stress(
            &j3.aos,
            &czg(blo),
            &gfl,
            &gkl,
            &j3.image_phase,
            &restore,
            &r_w,
            home,
            kernel,
            positions,
            acc,
        );
        let (hf, hk) = hessian_fields(j3, &r_w);
        for e in 0..3 {
            let gfe = [hf[0][e].clone(), hf[1][e].clone(), hf[2][e].clone()];
            let gke = [hk[0][e].clone(), hk[1][e].clone(), hk[2][e].clone()];
            complex_field_stress(
                &j3.aos,
                &czg(&bgo[e]),
                &gfe,
                &gke,
                &j3.image_phase,
                &restore,
                &r_w,
                home,
                kernel,
                positions,
                acc,
            );
        }
        // PW block: value strain-invariant; lap + gradient fields via the reciprocal metric.
        if let Some(pw) = &j3.pw {
            let czp = |a: &Array2<f64>| {
                if orbital_rank {
                    j3.mo_project_t(a)
                        .slice(s![pw0..pw0 + num_pw, ..])
                        .to_owned()
                } else {
                    a.slice(s![pw0..pw0 + num_pw, ..])
                        .mapv(|x| Complex64::new(x, 0.0))
                }
            };
            pw_lap_stress_complex(pw, &czp(blo), r, acc);
            let cm_gou_pw = [czp(&bgo[0]), czp(&bgo[1]), czp(&bgo[2])];
            pw_gou_stress_complex(pw, &cm_gou_pw, r, acc);
        }
        // B-spline block (orbital rank only): value strain-invariant (fractional coordinates);
        // lap + gradient fields via the complex B-spline kinetic strain, AO-rank cotangents.
        if j3.bspline.is_some() {
            let cm_lap = j3.mo_project_t(blo);
            let cm_gou = [
                j3.mo_project_t(&bgo[0]),
                j3.mo_project_t(&bgo[1]),
                j3.mo_project_t(&bgo[2]),
            ];
            bspline_kstress_side_complex(j3, &cm_lap, &cm_gou, r, acc);
        }
    };
    do_side(&bar_a_up, &blo_u, &bgo_u, r_up, &mut acc);
    do_side(&bar_a_dn, &blo_d, &bgo_d, r_dn, &mut acc);
    *out = &*out + &acc.mapv(|x| x.re);
}

/// Complex-twist J3 kinetic STRAIN `dT_J3/deps` (3x3, added Re to `out`) for a PURE (MO-wrapped)
/// plane-wave / B-spline complex J3 leaf -- the complex twin of the real
/// [`crate::gfmc_forces::j3_leaf_kstress_side`], and the leaf-owning analog of
/// [`j3_pbc_kstress_complex`] (which slices `j_matrix` at AO rank and so cannot see a MO-wrapped
/// leaf whose orbital rank differs from its AO rank).
///
/// The J3 kinetic-strain cotangents (`bgo` = gou/gradient-field cotangent, `blo` = lap-field
/// cotangent) are LEAF-INDEPENDENT: `coef = dJ3/dA` from the Re-projected ORBITAL VALUE
/// ([`ComplexOrb::value`], MO-wrapping aware) and the drift `bg = -(gJ + Re(gD))`, exactly as
/// `j3_pbc_kstress_complex`. The **value** field-strain (`bar_a`) is 0 for these leaves (the
/// B-spline / PW value is affine-strain-invariant -- fractional coords / reciprocal metric fixed),
/// so ONLY the lap + gou metric responses survive; those are MO-projected to AO rank and supplied
/// by the SAME [`bspline_kstress_side_complex`] / [`pw_lap_stress_complex`]+[`pw_gou_stress_complex`]
/// the determinant kinetic stress uses (no new strain math). `w` = S' mask (None => 1).
#[allow(clippy::too_many_arguments)]
pub fn j3_leaf_kstress_side_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    use numpy::ndarray::{s, Array1, Axis};
    let n_up_w = r_up.nrows();
    let w_up: Option<Vec<f64>> = w.map(|w| w[0..n_up_w].to_vec());
    let w_dn: Option<Vec<f64>> = w.map(|w| w[n_up_w..].to_vec());
    // ORBITAL-rank value (Re), MO-wrapping aware -- 1:1 the value leg of `j3_grads_lap_complex`.
    let av = |r: &ArrayView2<f64>| j3.value(r).mapv(|x| x.re);
    let a_up = av(r_up);
    let a_dn = av(r_dn);
    let orb_n = a_up.nrows();
    let (n_up, n_dn) = (a_up.ncols(), a_dn.ncols());
    let ku = crate::stress_pbc::k_strict_lower(n_up);
    let kd = crate::stress_pbc::k_strict_lower(n_dn);
    let m = j_matrix.slice(s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let coef = |a: &Array2<f64>, k: &Array2<f64>, cross: &Array1<f64>| -> Array2<f64> {
        let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
        let nc = a.ncols();
        for p in 0..orb_n {
            for c in 0..nc {
                g[[p, c]] += j1col[p] + cross[p];
            }
        }
        g
    };
    let coef_up = coef(&a_up, &ku, &m_s_dn);
    let coef_dn = coef(&a_dn, &kd, &mt_s_up);
    // bgo = w*bg*coef (gou field), blo = -0.5*w*coef (lap field); value bar_a = 0 (affine-inv).
    let side = |coef: &Array2<f64>,
                bg: &Array2<f64>,
                wcol: &Option<Vec<f64>>|
     -> ([Array2<f64>; 3], Array2<f64>) {
        let nc = coef.ncols();
        let mut bgo = [
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
        ];
        let mut blo = Array2::<f64>::zeros((orb_n, nc));
        for a in 0..orb_n {
            for k in 0..nc {
                let wk = wcol.as_ref().map_or(1.0, |v| v[k]);
                for d in 0..3 {
                    bgo[d][[a, k]] = wk * bg[[k, d]] * coef[[a, k]];
                }
                blo[[a, k]] = wk * (-0.5) * coef[[a, k]];
            }
        }
        (bgo, blo)
    };
    let (bgo_u, blo_u) = side(&coef_up, bg_up, &w_up);
    let (bgo_d, blo_d) = side(&coef_dn, bg_dn, &w_dn);
    // MO-project the (real) orbital-rank cotangents to AO rank (complex), delegate to the complex
    // leaf kinetic-stress kernels (Re last). `mo == None` => AO rank == orbital rank.
    let cmz = |bar: &Array2<f64>| -> Array2<Complex64> {
        let barc = bar.mapv(|x| Complex64::new(x, 0.0));
        match &j3.mo {
            Some(c) => c.t().dot(&barc),
            None => barc,
        }
    };
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    let do_side = |blo: &Array2<f64>,
                   bgo: &[Array2<f64>; 3],
                   r: &ArrayView2<f64>,
                   acc: &mut Array2<Complex64>| {
        let cm_lap = cmz(blo);
        let cm_gou = [cmz(&bgo[0]), cmz(&bgo[1]), cmz(&bgo[2])];
        if j3.bspline.is_some() {
            bspline_kstress_side_complex(j3, &cm_lap, &cm_gou, r, acc);
        }
        if let Some(pw) = &j3.pw {
            pw_lap_stress_complex(pw, &cm_lap, r, acc);
            pw_gou_stress_complex(pw, &cm_gou, r, acc);
        }
    };
    do_side(&blo_u, &bgo_u, r_up, &mut acc);
    do_side(&blo_d, &bgo_d, r_dn, &mut acc);
    *out = &*out + &acc.mapv(|x| x.re);
}

/// J3 coefficient `coefA[o,n] = dJ3/dA_o(r_n)` per spin (REAL), with `A=Re(complex
/// sphe AO value)`. 1:1 the `coef` closure of `j3_pbc_kstress_complex`.
fn j3_coefa_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    use numpy::ndarray::{s, Axis};
    // Orbital rank, with an optional complex MO projection applied BEFORE Re.
    let av = |r: &ArrayView2<f64>| -> Array2<f64> { j3.value(r).mapv(|x| x.re) };
    let a_up = av(r_up);
    let a_dn = av(r_dn);
    let orb = a_up.nrows();
    let (n_up, n_dn) = (a_up.ncols(), a_dn.ncols());
    let ku = crate::stress_pbc::k_strict_lower(n_up);
    let kd = crate::stress_pbc::k_strict_lower(n_dn);
    let m = j_matrix.slice(s![.., 0..orb]).to_owned();
    let j1col = j_matrix.column(orb).to_owned();
    let m_s_dn = m.dot(&a_dn.sum_axis(Axis(1)));
    let mt_s_up = m.t().dot(&a_up.sum_axis(Axis(1)));
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
    (
        coef(&a_up, &ku, &m_s_dn),
        coef(&a_dn, &kd, &mt_s_up),
        a_up,
        a_dn,
    )
}

/// Complex Bloch-sphe J3 VALUE strain `dJ3/deps` (real 3x3) = Sum_e coefA[*,e]*dA_e/deps
/// (the value-field stress contracted with coefA; A=Re(complex sphe AO)). For the
/// ECP-with-Jastrow nonlocal stress Pulay term.
#[allow(clippy::too_many_arguments)]
pub fn j3_value_strain_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home: &[usize],
    kernel: &[usize],
) -> Array2<f64> {
    use numpy::ndarray::s;
    let (coef_up, coef_dn, _, _) = j3_coefa_complex(j3, j_matrix, r_up, r_dn);
    // Composite: only the GTO rows of coefA drive the J3 VALUE strain; the PW value is
    // strain-invariant (cos(G.r) invariant under a proper strain), so its rows contribute
    // nothing. Slice the gto rows [gto0, gto0+num_gto) matching the complex sphe AO block.
    let num_pw = j3.pw.as_ref().map_or(0, |p| p.num_ao);
    let orb = coef_up.nrows();
    // ORBITAL-rank J3 (any MO layer and/or a B-spline child): MO-project coefA to AO rank and take
    // the gto rows at their AO offset (issue #89). Otherwise the AO-rank rows ARE the AO rows.
    let orbital_rank = j3.j3_orbital_rank();
    let num_gto = if orbital_rank {
        j3.aos.num_ao
    } else {
        orb - num_pw
    };
    let gto0 = if orbital_rank {
        j3.gto_ao_offset()
    } else if j3.gto_first {
        0
    } else {
        num_pw
    };
    let cz = |a: &Array2<f64>| {
        if orbital_rank {
            j3.mo_project_t(a)
                .slice(s![gto0..gto0 + num_gto, ..])
                .to_owned()
        } else {
            a.slice(s![gto0..gto0 + num_gto, ..])
                .mapv(|x| Complex64::new(x, 0.0))
        }
    };
    let mut acc = Array2::<Complex64>::zeros((3, 3));
    // wrap + Bloch restore (jqmc compute_AOs convention; real J3 cotangent commutes with Re).
    for (coef, r) in [(&coef_up, r_up.to_owned()), (&coef_dn, r_dn.to_owned())] {
        let (r_w, restore) = wrap_restore(j3, &r.view());
        let (gf, gk) = grad_fields(j3, &r_w);
        complex_field_stress(
            &j3.aos,
            &cz(coef),
            &gf,
            &gk,
            &j3.image_phase,
            &restore,
            &r_w,
            home,
            kernel,
            positions,
            &mut acc,
        );
    }
    acc.mapv(|x| x.re)
}

/// Complex Bloch-sphe J3 VALUE force `dJ3/dR` (real n_atomx3) = -Sum_e coefA[*,e]*dA_e/dr
/// scattered to the AO home atom (dA/dR=-dA/dr, translation). A=Re(complex sphe AO).
/// Mirror of `forces::grad_j3_wrt_r` with complex sphe AOs. For the ECP-with-Jastrow
/// nonlocal force Pulay term.
#[allow(clippy::too_many_arguments)]
pub fn j3_value_force_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc: &[usize],
    n_atom: usize,
) -> Array2<f64> {
    let (coef_up, coef_dn, _, _) = j3_coefa_complex(j3, j_matrix, r_up, r_dn);
    // Composite: coefA runs over the full basis (gto (+) pw), but only the GTO AOs carry a
    // nuclear force (the PW child is R-independent -> dA_pw/dR = 0). Scatter only the gto
    // rows [gto0, gto0+num_gto) of coefA against the gto grad (`nuc` maps gto AOs -> atoms).
    let num_pw = j3.pw.as_ref().map_or(0, |p| p.num_ao);
    let orb = coef_up.nrows();
    // ORBITAL-rank J3 (any MO layer and/or a B-spline child): MO-project coefA to AO rank and take
    // the gto rows at their AO offset (issue #89). Otherwise the AO-rank rows ARE the AO rows.
    let orbital_rank = j3.j3_orbital_rank();
    let num_gto = if orbital_rank {
        j3.aos.num_ao
    } else {
        orb - num_pw
    };
    let gto0 = if orbital_rank {
        j3.gto_ao_offset()
    } else if j3.gto_first {
        0
    } else {
        num_pw
    };
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    let mut side = |coef: &Array2<f64>, r: &ArrayView2<f64>| {
        let ga = j3.ao_grad(r);
        let nc = coef.ncols();
        let pcoef = if orbital_rank {
            j3.mo_project_t(coef)
        } else {
            coef.mapv(|x| Complex64::new(x, 0.0))
        };
        let mut s = Array2::<f64>::zeros((3, num_gto));
        for d in 0..3 {
            for c in 0..num_gto {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..nc {
                    acc += pcoef[[gto0 + c, k]] * ga[d][[c, k]];
                }
                s[[d, c]] = -acc.re;
            }
        }
        for c in 0..num_gto {
            let ia = nuc[c];
            for d in 0..3 {
                out[[ia, d]] += s[[d, c]];
            }
        }
    };
    side(&coef_up, r_up);
    side(&coef_dn, r_dn);
    out
}

fn leg_d(l: i64, x: f64) -> (f64, f64) {
    crate::forces_ecp::leg_and_deriv(l, x)
}

/// Complex ECP non-local stress `dV_ecp_nl/deps` (real 3x3) for a twisted ECP geminal,
/// jnone. Complex analog of `stress_pbc::grad_v_ecp_nonlocal_strain`: complex det
/// ratio + complex Pulay stress + complex drift at mesh configs; explicit A is real;
/// Re at the end (the ECP non-local energy is Re of the complex ratio sum).
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_nonlocal_strain_complex(
    gem: &ComplexGeminal,
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
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    jas: Option<&crate::data::JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3_jm: Option<&ArrayView2<f64>>,
    j3_home: Option<&[usize]>,
    j3_kernel: Option<&[usize]>,
) -> Array2<f64> {
    use crate::complex_det::j3_grads_lap_complex;
    let inv = inv3(cell);
    let nv = weights.len();
    let n_up = r_up.nrows();
    let j1 = jas.and_then(|j| j.one_body.as_ref());
    let j2 = jas.and_then(|j| j.two_body.as_ref());
    let dtn = jas.and_then(|j| j.dtn.as_ref());
    let with_j = jas.is_some();
    // Jastrow value strain dJ/deps: J1/J2 cell-strain + complex J3 + DTN.
    let jvs = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> Array2<f64> {
        let mut o = Array2::<f64>::zeros((3, 3));
        crate::stress_pbc::jastrow_strain(j1, j2, ru, rd, &mut o);
        if let (Some(j3), Some(jm), Some(hm), Some(km)) = (j3c, j3_jm, j3_home, j3_kernel) {
            o += &j3_value_strain_complex(j3, jm, ru, rd, positions, hm, km);
        }
        if let Some(d) = dtn {
            o += &crate::stress_pbc::dtn_value_strain(d, ru, rd);
        }
        o
    };
    // Jastrow drift gradJ (n,3) per spin (J1/J2 + complex J3).
    let jdrift = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> (Array2<f64>, Array2<f64>) {
        let (mut gu, mut gd) = crate::param_grad_el::jastrow_grads(jas, None, None, ru, rd);
        if let (Some(j3), Some(jm)) = (j3c, j3_jm) {
            let (g3u, g3d, _, _) = j3_grads_lap_complex(j3, jm, ru, rd);
            gu += &g3u;
            gd += &g3d;
        }
        (gu, gd)
    };
    // Mesh Jastrow ratio `exp(J(r') - J(r))` in the `jastrow_ratio` zone (float64 in both
    // modes), the single-move factor of the twist ECP energy / parameter meshes; the
    // difference of two whole `jastrow_value_complex` values rounds in `jastrow_eval`
    // (float32 in mixed), issue #757.
    let jas_ratio = jas.map(|jd| {
        (
            jd,
            crate::gfmc::ComplexJasRatioState::new(jd, j3c, j3_jm, r_up, r_dn),
        )
    });
    let det_base = det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn));
    let mut pulay_base = pbc_det_pulay_stress_complex(
        gem, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, None,
    );
    if with_j {
        pulay_base += &jvs(r_up, r_dn).mapv(|x| Complex64::new(x, 0.0));
    }
    let mut out = Array2::<Complex64>::zeros((3, 3));
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
                    let mut ratio = det_ratio_complex(
                        det_phase_lnabs(&compute_geminal_complex(gem, &mu.view(), &md.view())),
                        det_base,
                    );
                    let (gdu, gdd, _, _) =
                        grads_and_laplacian_ln_det_complex(gem, &mu.view(), &md.view());
                    let mut drift = if is_up {
                        [gdu[[i, 0]], gdu[[i, 1]], gdu[[i, 2]]]
                    } else {
                        [gdd[[i, 0]], gdd[[i, 1]], gdd[[i, 2]]]
                    };
                    let mut pulay_m = pbc_det_pulay_stress_complex(
                        gem,
                        &mu.view(),
                        &md.view(),
                        positions,
                        home_up,
                        kernel_up,
                        home_dn,
                        kernel_dn,
                        None,
                    );
                    if let Some((jd, st)) = &jas_ratio {
                        ratio *= st.ratio(jd, j3c, r_up, r_dn, is_up, i, rp);
                        let (jgu, jgd) = jdrift(&mu.view(), &md.view());
                        let jg = if is_up { &jgu } else { &jgd };
                        for c in 0..3 {
                            drift[c] += Complex64::new(jg[[i, c]], 0.0);
                        }
                        pulay_m += &jvs(&mu.view(), &md.view()).mapv(|x| Complex64::new(x, 0.0));
                    }
                    let drift_vg = drift[0] * vg[0] + drift[1] * vg[1] + drift[2] * vg[2];
                    let a = vl * ((2 * l + 1) as f64) * pl * weights[v];
                    for c in 0..3 {
                        for d in 0..3 {
                            let dcth = -(vg[c] * rel[d]) / (nrm * vgn)
                                - cth * rel[c] * rel[d] / (nrm * nrm);
                            let da = ((2 * l + 1) as f64)
                                * weights[v]
                                * (vlp * pl * rel[c] * rel[d] / nrm + vl * dpl * dcth);
                            let mesh_wf = pulay_m[[c, d]] - drift[c] * (nrm * vg[d])
                                + (rel[c] * rel[d] / nrm) * drift_vg;
                            let dln_ratio = mesh_wf - pulay_base[[c, d]];
                            out[[c, d]] += ratio
                                * (Complex64::new(da, 0.0) + Complex64::new(a, 0.0) * dln_ratio);
                        }
                    }
                }
            }
        }
    }
    out.mapv(|x| x.re)
}

/// Complex ECP non-local force `dV_ecp_nl/dR` (real n_atomx3) for a twisted ECP
/// geminal, jnone. Per term: explicit dA/dR_ia (V_l'/P_l'/dcostheta) + A*dln(ratio)/dR
/// (O_R(mesh)-O_R(base) [complex Pulay force] + mesh-motion at ia). Re at end.
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_nonlocal_force_complex(
    gem: &ComplexGeminal,
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
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
    jas: Option<&crate::data::JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3_jm: Option<&ArrayView2<f64>>,
    j3_nuc: Option<&[usize]>,
) -> Array2<f64> {
    use crate::complex_det::{j3_grads_lap_complex, pbc_det_pulay_force_complex};
    let inv = inv3(cell);
    let nv = weights.len();
    let n_up = r_up.nrows();
    let j1 = jas.and_then(|j| j.one_body.as_ref());
    let dtn = jas.and_then(|j| j.dtn.as_ref());
    let with_j = jas.is_some();
    // Jastrow value-force dJ/dR (n_atom,3): J1 Williamson + complex J3 + DTN (J2 e-e: no nuclear force).
    let jforce = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> Array2<f64> {
        let mut o = Array2::<f64>::zeros((n_atom, 3));
        if let Some(j1d) = j1 {
            o += &crate::forces::grad_j1_wrt_r_pbc(j1d, ru, rd);
        }
        if let (Some(j3), Some(jm), Some(jn)) = (j3c, j3_jm, j3_nuc) {
            o += &j3_value_force_complex(j3, jm, ru, rd, jn, n_atom);
        }
        if let Some(d) = dtn {
            o += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, ru, rd);
        }
        o
    };
    // Jastrow drift gradJ (n,3) per spin (J1/J2 + complex J3).
    let jdrift = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> (Array2<f64>, Array2<f64>) {
        let (mut gu, mut gd) = crate::param_grad_el::jastrow_grads(jas, None, None, ru, rd);
        if let (Some(j3), Some(jm)) = (j3c, j3_jm) {
            let (g3u, g3d, _, _) = j3_grads_lap_complex(j3, jm, ru, rd);
            gu += &g3u;
            gd += &g3d;
        }
        (gu, gd)
    };
    // Mesh Jastrow ratio in the `jastrow_ratio` zone, as in the strain kernel above (issue #757).
    let jas_ratio = jas.map(|jd| {
        (
            jd,
            crate::gfmc::ComplexJasRatioState::new(jd, j3c, j3_jm, r_up, r_dn),
        )
    });
    let det_base = det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn));
    let mut or_base = pbc_det_pulay_force_complex(gem, r_up, r_dn, nuc_up, nuc_dn, n_atom, None);
    if with_j {
        or_base += &jforce(r_up, r_dn).mapv(|x| Complex64::new(x, 0.0));
    }
    let mut out = Array2::<Complex64>::zeros((n_atom, 3));
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
                let rel0 = [
                    positions[[ia, 0]] - re[0],
                    positions[[ia, 1]] - re[1],
                    positions[[ia, 2]] - re[2],
                ];
                let rel = min_image(rel0, cell, &inv, nmax);
                let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                let uei = [rel[0] / nrm, rel[1] / nrm, rel[2] / nrm];
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
                    let mut ratio = det_ratio_complex(
                        det_phase_lnabs(&compute_geminal_complex(gem, &mu.view(), &md.view())),
                        det_base,
                    );
                    let mut or_mesh = pbc_det_pulay_force_complex(
                        gem,
                        &mu.view(),
                        &md.view(),
                        nuc_up,
                        nuc_dn,
                        n_atom,
                        None,
                    );
                    let (gdu, gdd, _, _) =
                        grads_and_laplacian_ln_det_complex(gem, &mu.view(), &md.view());
                    let mut drift = if is_up {
                        [gdu[[i, 0]], gdu[[i, 1]], gdu[[i, 2]]]
                    } else {
                        [gdd[[i, 0]], gdd[[i, 1]], gdd[[i, 2]]]
                    };
                    if let Some((jd, st)) = &jas_ratio {
                        ratio *= st.ratio(jd, j3c, r_up, r_dn, is_up, i, rp);
                        or_mesh += &jforce(&mu.view(), &md.view()).mapv(|x| Complex64::new(x, 0.0));
                        let (jgu, jgd) = jdrift(&mu.view(), &md.view());
                        let jg = if is_up { &jgu } else { &jgd };
                        for d in 0..3 {
                            drift[d] += Complex64::new(jg[[i, d]], 0.0);
                        }
                    }
                    let drift_vg = drift[0] * vg[0] + drift[1] * vg[1] + drift[2] * vg[2];
                    let a = vl * ((2 * l + 1) as f64) * pl * weights[v];
                    for big_i in 0..n_atom {
                        for d in 0..3 {
                            let mut term = ratio * a * (or_mesh[[big_i, d]] - or_base[[big_i, d]]);
                            if big_i == ia {
                                // explicit dA/dR_ia + mesh-motion
                                let dcth = -vg[d] / (nrm * vgn) - cth * uei[d] / nrm;
                                let da = ((2 * l + 1) as f64)
                                    * weights[v]
                                    * (vlp * pl * uei[d] + vl * dpl * dcth);
                                let motion = drift[d] + uei[d] * drift_vg;
                                term += ratio
                                    * (Complex64::new(da, 0.0) + Complex64::new(a, 0.0) * motion);
                            }
                            out[[big_i, d]] += term;
                        }
                    }
                }
            }
        }
    }
    out.mapv(|x| x.re)
}

/// Near-singular fallback threshold on the single-move complex det ratio `d`. Mirror
/// of the REAL `param_grad_ecp::sm_singular_guard`: below `sqrt(f64::EPSILON)` (the
/// classical half-mantissa rank-1-update conditioning bound) the moved geminal is
/// within fp resolution of singular, so THAT candidate defers to the fresh per-point
/// O_k rebuild. Derived from `f64::EPSILON`, not a tuned constant. ECP mesh moves land
/// near the ECP shell (well away from geminal nodes) so `|d| = O(1)` in practice.
#[inline]
fn sm_singular_guard_c() -> f64 {
    f64::EPSILON.sqrt()
}

/// Effective guard honoring the test-only `JQMC_ECP_OK_FORCE_REBUILD` hook (mirror of
/// the REAL `param_grad_ecp::sm_effective_guard`): when set, `+inf` forces EVERY mesh
/// candidate through the fresh per-point rebuild so an A/B test can compare the SM-
/// incremental fast path against the per-point reference on identical inputs. Read once
/// per call (never per mesh point), so it is off the hot path.
#[inline]
fn sm_effective_guard_c() -> f64 {
    if std::env::var_os("JQMC_ECP_OK_FORCE_REBUILD").is_some() {
        f64::INFINITY
    } else {
        sm_singular_guard_c()
    }
}

/// Complex rank-1 outer product `a b^T` -> (a.len(), b.len()) folding the SM reduced
/// vectors into the O_k accumulator (mirror of `param_grad_ecp::outer`, complex).
#[inline]
fn outer_c(
    a: &numpy::ndarray::Array1<Complex64>,
    b: &numpy::ndarray::Array1<Complex64>,
) -> Array2<Complex64> {
    let (na, nb) = (a.len(), b.len());
    let mut o = Array2::<Complex64>::zeros((na, nb));
    let zero = Complex64::new(0.0, 0.0);
    for i in 0..na {
        let ai = a[i];
        if ai == zero {
            continue;
        }
        for j in 0..nb {
            o[[i, j]] = ai * b[j];
        }
    }
    o
}

/// Complex ECP non-local parameter gradient `dV_ecp_nl/dMO` (up, dn complex) for a
/// twisted ECP geminal. `dV_nl/dMO = Sum_t A_t*ratio_t*[O_k(mesh_t)-O_k(base)]`,
/// A=V_l*(2l+1)*P_l*w (real), O_k = dln det/dMO (holomorphic; take Re downstream).
///
/// Each mesh point moves ONE electron => the geminal `G` shifts by a rank-1 update, so
/// both the det ratio `d = det(G')/det(G)` AND `O_k(mesh) - O_k(base)` come from a
/// Sherman-Morrison update OFF THE SINGLE BASE `G^{-1}` -- no per-grid-point O(N^3)
/// geminal rebuild + fresh LU + fresh inverse (`compute_geminal_complex` +
/// `det_phase_lnabs` + `grad_ln_det_mo_complex` per point). Besides the speed, this avoids
/// an fp-reassociation residual: a per-point rebuild factors the SAME `G(r')` TWICE
/// (LU for the det, Gauss-Jordan for the inverse) and, at a far-roamed / ill-conditioned
/// mesh config, those two fresh factorizations disagree with each other and with XLA's
/// single-inverse contraction; the SM update produces ONE consistent updated inverse,
/// matching XLA's order. Byte-consistent 1:1 with the REAL `param_grad_ecp::grad_ecp_mo`
/// SM decomposition, in complex holomorphic arithmetic (plain transpose, never conj).
/// Near-singular candidates defer to the fresh per-point rebuild (`sm_singular_guard_c`).
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_nonlocal_mo_complex(
    gem: &ComplexGeminal,
    jastrow: Option<&crate::data::JastrowData>,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
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
) -> (Array2<Complex64>, Array2<Complex64>) {
    use crate::complex_det::{grad_ln_det_mo_complex, inverse_complex};
    use numpy::ndarray::{s, Array1};
    let inv = inv3(cell);
    let nv = weights.len();
    let n_up_e = r_up.nrows();
    let n_dn_e = r_dn.nrows();
    // Mesh Jastrow ratio `exp(J(r') - J(r))` in the `jastrow_ratio` zone (float64 in both
    // modes): the single-move J1/J2 delta, the rank-1 J3 ratio and the DTN streaming
    // ratio of the shared complex mesh state, i.e. the SAME factor the twist J1/J2 ECP
    // parameter blocks (`twist_param_mesh`) and XLA's `_compute_ratio_Jastrow_part_split_spin`
    // use. The difference of two whole `jastrow_value_complex` values would
    // round in `jastrow_eval` (float32 in mixed), issue #717.
    let j_matrix_c = j_matrix.map(|m| m.as_standard_layout().to_owned());
    let jm_view = j_matrix_c.as_ref().map(|m| m.view());
    let jas_mesh = jastrow
        .filter(|jd| {
            jd.one_body.is_some() || jd.two_body.is_some() || jd.dtn.is_some() || j3c.is_some()
        })
        .map(|jd| {
            let st =
                crate::gfmc::mesh_fast_state_complex(gem, jd, j3c, jm_view.as_ref(), r_up, r_dn)
                    .expect("standard-layout j_matrix always seeds the complex mesh Jastrow state");
            (jd, st)
        });
    let guard = sm_effective_guard_c();
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // ---- Base fields (computed ONCE; the SM rank-1 rides this single base G^{-1}). ----
    // Mirror `grad_ln_det_mo_complex` so `ok_*_base` is bit-identical to the reference
    // base, then keep `ginv`/`W`/`Q`/`GOU`/`GQ` for the per-electron SM. Plain `.t()`
    // everywhere (HOLOMORPHIC; conj-transpose would break the twist Im.Im cross term).
    let ao_up0 = gem.orb_up.ao(r_up); // (num_ao_up, N_up)
    let ao_dn0 = gem.orb_dn.ao(r_dn); // (num_ao_dn, N_dn)
    let orb_up0 = gem.orb_up.value(r_up); // (m_up, N_up)  = C_up ao_up0
    let orb_dn0 = gem.orb_dn.value(r_dn); // (m_dn, N_dn)  = C_dn ao_dn0
    let m_up = orb_up0.nrows();
    let m_dn = gem.orb_num_dn;
    let n_up = ao_up0.ncols();
    let n_dn = ao_dn0.ncols();
    let num_ao_up = ao_up0.nrows();
    let num_ao_dn = ao_dn0.nrows();
    // FULL complex lambda: the strain (eps) O_k / kinetic base G, G^-1, W, Q must ride the
    // complex lambda so `Im(lambda)` reaches dE_L/deps and the Pulay-stress on a complex-
    // `lambda` AGP twist (real-pairing twist: Im == 0 -> byte-identical).
    let w = gem.w_from_dn(&orb_dn0, n_up);
    let g = orb_up0.t().dot(&w);
    let ginv = inverse_complex(&g);
    let q = gem.up_t_times_paired(&orb_up0); // (N_up, m_dn)
    let gou = w.dot(&ginv); // (m_up, N_up)  = bar_orb_up
    let gq = ginv.dot(&q); // (N_up, m_dn)
    let ok_up_base = gou.dot(&ao_up0.t()); // (m_up, num_ao_up)
    let ok_dn_base = gq.slice(s![0..n_dn, ..]).t().dot(&ao_dn0.t()); // (m_dn, num_ao_dn)
    let mut g_up = Array2::<Complex64>::zeros(ok_up_base.dim());
    let mut g_dn = Array2::<Complex64>::zeros(ok_dn_base.dim());

    // ---- Per-electron nearest-`NN`-nucleus set (min-image aware), matching the
    // production non-local e_L (`compute_ecp_non_local_parts_nearest_neighbors`,
    // NN_default=1) and the REAL `param_grad_ecp::collect_mesh`. The reference c-axis
    // derivative (`assemble._dV_ecp_dc` -> `..._nearest_neighbors_holomorphic`) reduces
    // over THIS NN mesh, so summing all atom pairs would add the far nucleus's
    // exponentially-suppressed `V_l = c*|r|^{p-2} e^{-b|r|^2}` term (the ~1e-7 residual
    // This is a far-nucleus configuration, not a far-roamed electron configuration.
    const NN: usize = 1;
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |rx: f64, ry: f64, rz: f64| -> Vec<usize> {
        if NN == 0 || NN >= uniq.len() {
            return uniq.clone();
        }
        let mut d: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d0 = [
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                ];
                let dd = min_image(d0, cell, &inv, nmax);
                (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2], ia)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(NN).map(|&(_, ia)| ia).collect()
    };
    let nearest_up: Vec<Vec<usize>> = (0..n_up_e)
        .map(|i| nearest_set(r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]))
        .collect();
    let nearest_dn: Vec<Vec<usize>> = (0..n_dn_e)
        .map(|i| nearest_set(r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]))
        .collect();

    // ---- Collect the NN-truncated ECP mesh: per moved electron, its grid positions +
    // the real weight A_t and the (real, exp) Jastrow ratio. The O_k are evaluated by
    // SM (below); the mesh SET is the production NN mesh. ----
    let mut pos_e: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_up_e + n_dn_e];
    let mut wr_e: Vec<Vec<f64>> = vec![Vec::new(); n_up_e + n_dn_e];
    let mut jf_e: Vec<Vec<f64>> = vec![Vec::new(); n_up_e + n_dn_e];
    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let l = ang_mom[t];
        let (be, ce, pe) = (exponents[t], coefficients[t], powers[t]);
        for is_up in [true, false] {
            let nr = if is_up { n_up_e } else { n_dn_e };
            let nearest = if is_up { &nearest_up } else { &nearest_dn };
            for i in 0..nr {
                if !nearest[i].contains(&ia) {
                    continue;
                }
                let re = if is_up {
                    [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
                } else {
                    [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
                };
                let rel0 = [
                    positions[[ia, 0]] - re[0],
                    positions[[ia, 1]] - re[1],
                    positions[[ia, 2]] - re[2],
                ];
                let rel = min_image(rel0, cell, &inv, nmax);
                let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                let vl = ce * nrm.powf(pe - 2.0) * (-be * nrm * nrm).exp();
                let e = if is_up { i } else { n_up_e + i };
                for v in 0..nv {
                    let vg = [
                        grid_points[[v, 0]],
                        grid_points[[v, 1]],
                        grid_points[[v, 2]],
                    ];
                    let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                    let cth = -(rel[0] * vg[0] + rel[1] * vg[1] + rel[2] * vg[2]) / (nrm * vgn);
                    let pl = legendre(l, cth);
                    let rp = [
                        re[0] + rel[0] + nrm * vg[0],
                        re[1] + rel[1] + nrm * vg[1],
                        re[2] + rel[2] + nrm * vg[2],
                    ];
                    let wr = vl * ((2 * l + 1) as f64) * pl * weights[v];
                    let jf = match &jas_mesh {
                        Some((jd, st)) => {
                            st.jas_ratio(jd, j3c, jm_view.as_ref(), r_up, r_dn, is_up, i, rp)
                        }
                        None => 1.0,
                    };
                    pos_e[e].push(rp);
                    wr_e[e].push(wr);
                    jf_e[e].push(jf);
                }
            }
        }
    }

    // ---- Per-electron batched Sherman-Morrison O_k reduce. `ratio = d * jf`, `A = wr`,
    // and the SM rank-1 carries a `1/d`, so the O_k coefficient `inv = A*ratio/d = wr*jf`
    // (the det ratio cancels the SM denominator EXACTLY -- the same algebra the REAL
    // `grad_ecp_mo` uses via `vgc/dj`). `d` is REUSED for the mesh det ratio (no separate
    // `det_phase_lnabs`), which is the speed win. ----
    for e in 0..(n_up_e + n_dn_e) {
        let k = pos_e[e].len();
        if k == 0 {
            continue;
        }
        let is_up = e < n_up_e;
        let idx = if is_up { e } else { e - n_up_e };
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        let mut ptmat = Array2::<f64>::zeros((k, 3));
        for (j, p) in pos_e[e].iter().enumerate() {
            ptmat[[j, 0]] = p[0];
            ptmat[[j, 1]] = p[1];
            ptmat[[j, 2]] = p[2];
        }
        // AO of the moved electron ONLY, batched over its k grid points (wrap internal);
        // do_mesh = C ao_mesh (identity if no MO). No full-geminal AO eval.
        let ao_mesh = orb.ao(&ptmat.view()); // (num_ao, k)
        let do_mesh = match &orb.mo {
            Some(c) => c.dot(&ao_mesh),
            None => ao_mesh.clone(),
        }; // (m, k)
        let wr = &wr_e[e];
        let jf = &jf_e[e];
        if is_up {
            let mut dao = ao_mesh.clone(); // (num_ao_up, k)
            for r in 0..num_ao_up {
                let old = ao_up0[[r, idx]];
                for x in 0..k {
                    dao[[r, x]] -= old;
                }
            }
            let mut d_o = do_mesh.clone(); // (m_up, k)  = C_up dao
            for r in 0..m_up {
                let old = orb_up0[[r, idx]];
                for x in 0..k {
                    d_o[[r, x]] -= old;
                }
            }
            let v = w.t().dot(&d_o); // (N_up, k)
            let g_row = ginv.t().dot(&v); // (N_up, k)  = G^{-T} v = z
            let s = ao_up0.dot(&g_row); // (num_ao_up, k)
            let dq = gem.paired_t_times(&d_o); // (m_dn, k)
            let tt = gq.t().dot(&v); // (m_dn, k)
            let g_col = ginv.column(idx).to_owned(); // (N_up)
            let gou_i = gou.column(idx).to_owned(); // (m_up), per-electron
            let mut r_vec = Array1::<Complex64>::zeros(num_ao_dn); // ao_dn0 @ g_col[:n_dn]
            for jj in 0..num_ao_dn {
                let mut su = zero;
                for nn in 0..n_dn {
                    su += ao_dn0[[jj, nn]] * g_col[nn];
                }
                r_vec[jj] = su;
            }
            let mut cu = Array1::<Complex64>::zeros(num_ao_up);
            let mut cd = Array1::<Complex64>::zeros(m_dn);
            for x in 0..k {
                let mut dj = one;
                for al in 0..n_up {
                    dj += g_col[al] * v[[al, x]];
                }
                let a = Complex64::new(wr[x] * jf[x], 0.0) * dj; // = A * ratio
                if dj.re.is_finite() && dj.im.is_finite() && dj.norm() >= guard {
                    let inv = a / dj; // = wr*jf (det ratio cancels the SM denominator)
                    for r in 0..num_ao_up {
                        cu[r] += inv * (dao[[r, x]] - s[[r, x]]);
                    }
                    for bcol in 0..m_dn {
                        cd[bcol] += inv * (dq[[bcol, x]] - tt[[bcol, x]]);
                    }
                } else {
                    // near-singular candidate: fresh per-point O_k rebuild (pre-opt path).
                    let mut m = r_up.to_owned();
                    m[[idx, 0]] = pos_e[e][x][0];
                    m[[idx, 1]] = pos_e[e][x][1];
                    m[[idx, 2]] = pos_e[e][x][2];
                    let (fu, fd) = grad_ln_det_mo_complex(gem, &m.view(), r_dn);
                    g_up.scaled_add(a, &(&fu - &ok_up_base));
                    g_dn.scaled_add(a, &(&fd - &ok_dn_base));
                }
            }
            g_up += &outer_c(&gou_i, &cu);
            g_dn += &outer_c(&cd, &r_vec);
        } else {
            let mut dao = ao_mesh.clone(); // (num_ao_dn, k)
            for r in 0..num_ao_dn {
                let old = ao_dn0[[r, idx]];
                for x in 0..k {
                    dao[[r, x]] -= old;
                }
            }
            let mut d_o = do_mesh.clone(); // (m_dn, k)  = C_dn dao
            for r in 0..m_dn {
                let old = orb_dn0[[r, idx]];
                for x in 0..k {
                    d_o[[r, x]] -= old;
                }
            }
            let ww = q.dot(&d_o); // (N_up, k)
            let g_col2 = ginv.dot(&ww); // (N_up, k)
            let wg = w.dot(&g_col2); // (m_up, k)
            let dw = gem.paired_times(&d_o); // (m_up, k)
            let y2 = ao_dn0.dot(&g_col2.slice(s![0..n_dn, ..])); // (num_ao_dn, k)
            let g_row2 = ginv.row(idx).to_owned(); // (N_up)
            let tq = gq.row(idx).to_owned(); // (m_dn), per-electron
            let mut ru2 = Array1::<Complex64>::zeros(num_ao_up); // ao_up0 @ g_row2
            for jj in 0..num_ao_up {
                let mut su = zero;
                for nn in 0..n_up {
                    su += ao_up0[[jj, nn]] * g_row2[nn];
                }
                ru2[jj] = su;
            }
            let mut cu2 = Array1::<Complex64>::zeros(m_up);
            let mut cd2 = Array1::<Complex64>::zeros(num_ao_dn);
            for x in 0..k {
                let dj = one + g_col2[[idx, x]];
                let a = Complex64::new(wr[x] * jf[x], 0.0) * dj;
                if dj.re.is_finite() && dj.im.is_finite() && dj.norm() >= guard {
                    let inv = a / dj;
                    for al in 0..m_up {
                        cu2[al] += inv * (dw[[al, x]] - wg[[al, x]]);
                    }
                    for r in 0..num_ao_dn {
                        cd2[r] += inv * (dao[[r, x]] - y2[[r, x]]);
                    }
                } else {
                    let mut m = r_dn.to_owned();
                    m[[idx, 0]] = pos_e[e][x][0];
                    m[[idx, 1]] = pos_e[e][x][1];
                    m[[idx, 2]] = pos_e[e][x][2];
                    let (fu, fd) = grad_ln_det_mo_complex(gem, r_up, &m.view());
                    g_up.scaled_add(a, &(&fu - &ok_up_base));
                    g_dn.scaled_add(a, &(&fd - &ok_dn_base));
                }
            }
            g_up += &outer_c(&cu2, &ru2);
            g_dn += &outer_c(&tq, &cd2);
        }
    }
    // As in the real kernel, a square occupied-SD coefficient matrix changes
    // Psi only by a constant determinant factor, which cancels from ECP ratios.
    if gem.is_sd() {
        if g_up.nrows() == g_up.ncols() {
            g_up.fill(Complex64::new(0.0, 0.0));
        }
        if g_dn.nrows() == g_dn.ncols() {
            g_dn.fill(Complex64::new(0.0, 0.0));
        }
    }
    (g_up, g_dn)
}

#[cfg(test)]
mod moved_group_tests {
    use super::*;

    #[test]
    fn pure_bspline_moved_group_matches_per_point_reference_in_both_modes() {
        // A pure complex B-spline leaf (0-AO gto placeholder in the `ao_eval` zone) is covered by
        // the fused moved-electron contraction in both precision modes, and its contracted value
        // and gradient equal the per-point orbital route `col . value_grad_lap(r)` within the
        // reduction-order class (the fused kernel uses `value_grad`, the first two outputs of
        // `value_grad_lap`).
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let mixed = crate::PrecisionPolicy {
            dtypes: crate::PRECISION_ZONES.map(|zone| match zone {
                "ao_eval" | "jastrow_eval" | "jastrow_grad_lap" => crate::RealDType::F32,
                _ => crate::RealDType::F64,
            }),
        };
        fn leaf(policy: crate::PrecisionPolicy) -> ComplexOrb {
            let mut aos = crate::atomic_orbital::AosGtoData::empty();
            aos.precision_policy = policy;
            ComplexOrb {
                precision_policy: policy,
                aos,
                image_phase: Vec::new(),
                mo: None,
                pw: None,
                gto_first: true,
                wrap: None,
                bspline: Some(crate::bspline::synthetic_complex_bspline_for_tests(
                    6,
                    0x005e_ed91,
                    [0.21, -0.13, 0.08],
                    false,
                )),
                tiled: None,
            }
        }
        for (mode, orb) in [
            ("full", leaf(crate::PrecisionPolicy::full())),
            ("mixed", leaf(mixed)),
        ] {
            assert!(
                moved_group_ok(&orb),
                "{mode}: the pure B-spline leaf must be covered"
            );
            let n = orb.num_ao_total();
            let col = numpy::ndarray::Array1::from_shape_fn(n, |a| {
                Complex64::new(
                    0.3 * ((a % 5) as f64 - 2.0),
                    0.2 * (((a * 3) % 7) as f64 - 3.0),
                )
            });
            let pts: Vec<[f64; 3]> = (0..24)
                .map(|v| {
                    let t = 0.37 * v as f64;
                    [
                        0.4 + 1.3 * t.cos(),
                        -0.2 + 0.9 * t.sin(),
                        0.1 * v as f64 - 0.8,
                    ]
                })
                .collect();
            let got = moved_group_complex(
                &orb,
                &pts,
                &col,
                None,
                0,
                false,
                crate::ao_cart_deriv::AoScreen::None,
            )
            .expect("covered leaf");
            let rmat = Array2::from_shape_fn((pts.len(), 3), |(v, c)| pts[v][c]);
            let (vals, grads, _) = orb.value_grad_lap(&rmat.view());
            let mut max_val = 0.0_f64;
            for v in 0..pts.len() {
                let want: Complex64 = (0..n).map(|a| col[a] * vals[[a, v]]).sum();
                let scale: f64 = (0..n).map(|a| col[a].norm() * vals[[a, v]].norm()).sum();
                let tol = rtol * scale;
                assert!(
                    (got.val[v] - want).norm() <= tol,
                    "{mode} v={v}: value {} vs {want}",
                    got.val[v]
                );
                for (d, grads_d) in grads.iter().enumerate() {
                    let want_g: Complex64 = (0..n).map(|a| col[a] * grads_d[[a, v]]).sum();
                    let scale_g: f64 = (0..n).map(|a| col[a].norm() * grads_d[[a, v]].norm()).sum();
                    let tol_g = rtol * scale_g;
                    assert!(
                        (got.grad[v][d] - want_g).norm() <= tol_g,
                        "{mode} v={v} d={d}: grad {} vs {want_g}",
                        got.grad[v][d]
                    );
                }
                max_val = max_val.max(want.norm());
            }
            assert!(
                max_val > 1.0e-3,
                "{mode}: vacuous contraction (max {max_val})"
            );
        }
    }
}
