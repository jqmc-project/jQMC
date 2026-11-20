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

//! Analytical nuclear forces `de_L/dR` (Pulay plus Hellmann-Feynman).
//! Translation identity: R enters lnPsi/T only through
//! AO centers, so `d(orbital q)/dR_I = -grad_r q` scattered by `nucleus_index`.
//! CART molecular scope (sphe->cart conversion is a follow-up). Autograd-free.

use numpy::ndarray::{Array2, ArrayView2};

use crate::ao_cart_deriv::{ao_deriv_bundle, ao_grad, AoDerivBundle};
use crate::data::{JastrowData, JastrowOneBodyData};
use crate::determinant::{GeminalData, OrbData};

/// `(electron-axis force, AO derivative bundle, AO gradient, Pulay bar C^T WG)` of one
/// orbital side of the fused kinetic force.
type OrbSideTables = (
    Array2<f64>,
    Option<AoDerivBundle>,
    Option<[Array2<f64>; 3]>,
    Array2<f64>,
);

/// Fused/shared kinetic-force fields computed ONCE and threaded through both the
/// dR (`grad_t_wrt_r_w_gi`) and dr (`tfull_grad_wrt_r_w_gi`) kernels of a single
/// force step: `jastrow_grads`, the `geminal_reverse` cotangent bundle, the det
/// grad field, and the fused det/J3 AO-derivative bundles. Built for `w = None`
/// (the full `dT`); the masked GFMC ZV paths (`w != None`) build locally instead.
pub(crate) struct KinShared {
    pub gj_up: Array2<f64>,
    pub gj_dn: Array2<f64>,
    pub b: crate::param_grad_el::GemReverse,
    pub gd_up: Array2<f64>,
    pub gd_dn: Array2<f64>,
    /// Determinant AO derivative bundles. `None` for a spin whose orbital-side
    /// kinetic force was contracted in the fused sweep ([`kin_force_sweep`]); that
    /// spin keeps only its AO gradients (`det_grad_*`).
    pub bundle_up: Option<AoDerivBundle>,
    pub bundle_dn: Option<AoDerivBundle>,
    pub det_grad_up: Option<[Array2<f64>; 3]>,
    pub det_grad_dn: Option<[Array2<f64>; 3]>,
    pub j3: Option<KinSharedJ3>,
    /// Orbital-side (det + J3) kinetic force, contracted once for both the dR
    /// and the dr kernel (builds with a [`KinFuse`] only).
    pub fused: Option<KinFused>,
    /// The determinant forward (`gem_forward_gi_cut` of the forward orbitals).
    /// Its value-level fields (`ou`, `od`, `w`, `ginv`, `q`, `wg`, `gq`) do not
    /// depend on the FAST derivative screen.
    pub fwd: crate::param_grad_el::GemFwd,
}

/// Orbital-side kinetic force of the determinant and J3 AO tables: `dT/dR`
/// `(n_atom, 3)` and `dT/dr` per spin, from ONE contraction of the per-(AO,
/// electron) product that both derivatives share ([`kin_contract_bundle`]).
pub(crate) struct KinFused {
    pub d_r: Array2<f64>,
    pub d_re_up: Array2<f64>,
    pub d_re_dn: Array2<f64>,
    /// The J3 orbital side is included (else the kernels add it themselves).
    pub j3: bool,
    /// `(C^T WG, C^T GQ[:N_dn]^T)` when they equal the kinetic `cw` (see the build).
    pub pulay_bars: Option<(Array2<f64>, Array2<f64>)>,
}

/// Extra inputs of a [`KinShared`] build that also contracts the orbital-side
/// kinetic force ([`KinFused`]). Full kinetic only (`w = None`).
pub(crate) struct KinFuse<'a> {
    pub nuc_up: &'a [usize],
    pub nuc_dn: &'a [usize],
    pub nuc_j3: Option<&'a [usize]>,
    pub n_atom: usize,
}

pub(crate) fn take_grad(g: Option<[Array2<f64>; 3]>, b: Option<AoDerivBundle>) -> [Array2<f64>; 3] {
    match (g, b) {
        (Some(g), _) => g,
        (None, Some(b)) => b.grad,
        (None, None) => unreachable!("KinShared holds one AO gradient table per spin"),
    }
}

/// Shared J3-basis AO evaluations for the kinetic force (value + fused derivative
/// bundle), reused across the dR and dr J3 kernels.
pub(crate) struct KinSharedJ3 {
    pub a_up: Array2<f64>,
    pub a_dn: Array2<f64>,
    pub bundle_up: AoDerivBundle,
    pub bundle_dn: AoDerivBundle,
}

impl KinShared {
    /// Build all shared fields (`jastrow` is the active Jastrow, `Some`). `w` feeds
    /// only the `geminal_reverse` seed weights; all AO/field pieces are w-independent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        gem: &GeminalData,
        jastrow: Option<&JastrowData>,
        j3_orb: Option<&OrbData>,
        j3_jm: Option<&ArrayView2<f64>>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        w: Option<&[f64]>,
        g_inv: Option<&ArrayView2<f64>>,
    ) -> KinShared {
        Self::build_with_forward(
            gem,
            gem,
            jastrow,
            j3_orb,
            j3_jm,
            r_up,
            r_dn,
            w,
            g_inv,
            crate::gfmc_forces::FastScreen::default(),
            None,
            None,
        )
    }

    /// Build derivative fields on the Cartesian-lowered orbital leaves while
    /// obtaining determinant forward/cotangent fields from the stored orbital
    /// representation.  For a spherical MO leaf this preserves XLA's ordering:
    /// spherical AO values are rounded in `ao_eval` before the fp64 MO
    /// projection, whereas Cartesian lowering is used only for AO-center
    /// derivatives.
    ///
    /// `det_tables`: the FAST chunk's carried screened MO grad/Laplacian tables at
    /// `(r_up, r_dn)`, used instead of re-evaluating them (`screen.det` only).
    ///
    /// `fuse = Some` (requires `w = None`) also contracts the orbital-side
    /// kinetic force once for both the dR and the dr kernel ([`KinFused`]).
    /// For a determinant leaf that [`kin_sweep_ok`] accepts, the AO derivatives
    /// are then contracted inside the AO sweep ([`kin_force_sweep`]) and only the
    /// AO gradients are stored.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_with_forward(
        gem: &GeminalData,
        forward_gem: &GeminalData,
        jastrow: Option<&JastrowData>,
        j3_orb: Option<&OrbData>,
        j3_jm: Option<&ArrayView2<f64>>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        w: Option<&[f64]>,
        g_inv: Option<&ArrayView2<f64>>,
        screen: crate::gfmc_forces::FastScreen,
        fuse: Option<KinFuse>,
        det_tables: Option<crate::param_grad_el::MoDerivTables>,
    ) -> KinShared {
        let fuse = fuse.filter(|_| w.is_none());
        let n_up_e = r_up.nrows();
        let bundle = |aos: &crate::atomic_orbital::AosGtoData, r: &ArrayView2<f64>, cut: bool| {
            if cut {
                crate::ao_cart_deriv::ao_deriv_bundle_fast_cut(aos, r)
            } else {
                ao_deriv_bundle(aos, r)
            }
        };
        // FAST (`screen.j3`): the J3 drift of the FAST e_L kinetic (derivative-screened).
        let (gj_up, gj_dn) = crate::timed!(
            "ks.jas_grads",
            crate::param_grad_el::jastrow_grads_cut(jastrow, j3_orb, j3_jm, r_up, r_dn, screen.j3)
        );
        let (w_up_vec, w_dn_vec) = match w {
            None => (None, None),
            Some(w) => (Some(w[0..n_up_e].to_vec()), Some(w[n_up_e..].to_vec())),
        };
        // One determinant forward serves the reverse bundle AND the drift field
        // (both need `gem_forward_gi(forward_gem, r, g_inv)`; it is evaluated once).
        let f = crate::timed!(
            "ks.gem_forward",
            crate::param_grad_el::gem_forward_gi_cut_tables(
                forward_gem,
                r_up,
                r_dn,
                g_inv,
                screen.det,
                det_tables.filter(|_| screen.det)
            )
        );
        let b = crate::timed!(
            "ks.gem_reverse",
            crate::param_grad_el::geminal_reverse_from_fwd(
                &f,
                &gj_up,
                &gj_dn,
                w_up_vec.as_deref(),
                w_dn_vec.as_deref(),
            )
        );
        let (gd_up, gd_dn) = crate::param_grad_el::det_grad_field_from_fwd(&f);
        let _t_j3 = crate::prof::on().then(std::time::Instant::now);
        let j3 = j3_orb.map(|orb| KinSharedJ3 {
            a_up: crate::atomic_orbital::compute_aos(&orb.aos, r_up),
            a_dn: crate::atomic_orbital::compute_aos(&orb.aos, r_dn),
            bundle_up: bundle(&orb.aos, r_up, screen.j3),
            bundle_dn: bundle(&orb.aos, r_dn, screen.j3),
        });
        if let Some(t) = _t_j3 {
            crate::prof::add("ks.j3_bundle", t.elapsed().as_nanos());
        }
        let Some(fz) = fuse else {
            let bundle_up =
                crate::timed!("ks.ao_bundle", bundle(&gem.orb_up.aos, r_up, screen.det));
            let bundle_dn =
                crate::timed!("ks.ao_bundle", bundle(&gem.orb_dn.aos, r_dn, screen.det));
            return KinShared {
                gj_up,
                gj_dn,
                b,
                gd_up,
                gd_dn,
                bundle_up: Some(bundle_up),
                bundle_dn: Some(bundle_dn),
                det_grad_up: None,
                det_grad_dn: None,
                j3,
                fused: None,
                fwd: f,
            };
        };
        // ---- fused orbital-side kinetic force (w = None) ----
        // geminal_reverse seeds: bar_lap = -1/2 WG_f, bar_gou[d] = -gJ[:,d] WG_f
        // (up; dn: GQ_f[:N_dn]^T), so the two derivative cotangents are column
        // scalings of ONE AO-level matrix `cw = C^T WG_f` (see `kin_contract_bundle`).
        let n_dn_e = r_dn.nrows();
        let wg_f = f.w_fused.dot(&f.ginv);
        let gq_f_t = f
            .ginv
            .dot(&f.q_fused)
            .slice(numpy::ndarray::s![0..n_dn_e, ..])
            .t()
            .to_owned();
        let mut d_r = Array2::<f64>::zeros((fz.n_atom, 3));
        let mut side = |orb: &OrbData,
                        r: &ArrayView2<f64>,
                        bar_orb: &Array2<f64>,
                        wgt: &Array2<f64>,
                        gj: &Array2<f64>,
                        nuc: &[usize]|
         -> OrbSideTables {
            let ca = pull_to_ao(orb, bar_orb);
            let cw = pull_to_ao(orb, wgt);
            let (s, e, bundle_out, grad_out) = if kin_sweep_ok(orb) {
                let (g, s, e) = crate::timed!(
                    "ks.kin_sweep",
                    kin_force_sweep(&orb.aos, r, &ca, &cw, &gj.view(), screen.det)
                );
                (s, e, None, Some(g))
            } else {
                let bnd = crate::timed!("ks.ao_bundle", bundle(&orb.aos, r, screen.det));
                let (s, e) = kin_contract_bundle(&bnd, &ca, &cw, &gj.view());
                (s, e, Some(bnd), None)
            };
            scatter_rows_neg(&s, nuc, &mut d_r);
            (e, bundle_out, grad_out, cw)
        };
        let (d_re_up, bundle_up, det_grad_up, cw_up) =
            side(&gem.orb_up, r_up, &b.bar_orb_up, &wg_f, &gj_up, fz.nuc_up);
        let (d_re_dn, bundle_dn, det_grad_dn, cw_dn) =
            side(&gem.orb_dn, r_dn, &b.bar_orb_dn, &gq_f_t, &gj_dn, fz.nuc_dn);
        // `cw = C^T WG_f` IS the Pulay bar `C^T WG` when the kinetic and the
        // standalone value tables coincide (the fp64 route): keep it for reuse.
        let same_values = f.w == f.w_fused && f.q == f.q_fused;
        let mut fused = KinFused {
            d_r,
            d_re_up,
            d_re_dn,
            j3: false,
            pulay_bars: same_values.then_some((cw_up, cw_dn)),
        };
        if let (Some(jm), Some(nuc), Some(sj3), Some(_)) = (j3_jm, fz.nuc_j3, j3.as_ref(), jastrow)
        {
            // J3 kinetic cotangent cot = gJ + gD (the `bar_gJ = -cot` of `j3_side`).
            let cot_up = &gj_up + &gd_up;
            let cot_dn = &gj_dn + &gd_dn;
            crate::timed!(
                "ks.j3_fused",
                j3_kin_force_merged(jm, sj3, &cot_up, &cot_dn, nuc, &mut fused)
            );
        }
        KinShared {
            gj_up,
            gj_dn,
            b,
            gd_up,
            gd_dn,
            bundle_up,
            bundle_dn,
            det_grad_up,
            det_grad_dn,
            j3,
            fused: Some(fused),
            fwd: f,
        }
    }
}

/// `out[nuc[c], d] -= s[c, d]` (`s` is `(num_ao, 3)`).
pub(crate) fn scatter_rows_neg(s: &Array2<f64>, nuc: &[usize], out: &mut Array2<f64>) {
    for c in 0..s.nrows() {
        for d in 0..3 {
            out[[nuc[c], d]] -= s[[c, d]];
        }
    }
}

/// Value-cotangent contraction `t_d[c,k] = cm[c,k] G_d[c,k]` for both
/// derivatives: `(s = Sum_k t (num_ao, 3), e = Sum_c t (N, 3))`. With
/// `cm = C^T WG` this is the determinant Pulay (`-s` scattered by atom) and the
/// determinant drift `e` (`Sum_a gou WG = Sum_c (C^T WG) G`).
pub(crate) fn val_contract(cm: &Array2<f64>, g: &[Array2<f64>; 3]) -> (Array2<f64>, Array2<f64>) {
    let (num_ao, n) = cm.dim();
    let mut s = Array2::<f64>::zeros((num_ao, 3));
    let mut e = [vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]];
    let cm = cm.as_standard_layout();
    let gs: [_; 3] = std::array::from_fn(|d| g[d].as_standard_layout());
    let (cms, gsl) = (
        cm.as_slice().unwrap(),
        [
            gs[0].as_slice().unwrap(),
            gs[1].as_slice().unwrap(),
            gs[2].as_slice().unwrap(),
        ],
    );
    for c in 0..num_ao {
        let row = &cms[c * n..(c + 1) * n];
        for d in 0..3 {
            let gr = &gsl[d][c * n..(c + 1) * n];
            let ed = &mut e[d];
            let mut acc = 0.0;
            for k in 0..n {
                let t = row[k] * gr[k];
                acc += t;
                ed[k] += t;
            }
            s[[c, d]] = acc;
        }
    }
    (s, Array2::from_shape_fn((n, 3), |(k, d)| e[d][k]))
}

/// The orbital-side kinetic force of one AO leaf, contracted once for both
/// derivatives. With AO-level cotangents `ca` (AO value) and `cw`, and the
/// per-electron cotangent `cot (N, 3)`, the kinetic cotangents of the AO
/// gradient and Laplacian are `-cot[k,e] cw[c,k]` and `-1/2 cw[c,k]`
/// (the `geminal_reverse` / J3 seeds), so every `dT` is a sum of
///
/// `t_d[c,k] = ca G_d + cw (-1/2 GL_d - Sum_e cot[k,e] H_de)`
///
/// (`G` grad, `H` Hessian, `GL` grad-Laplacian of AO `c` at electron `k`).
/// By the translation identity `dT/dR_I = -Sum_{c in I} Sum_k t[c,k]` and
/// `dT/dr_k = Sum_c t[c,k]`: this returns `(s = Sum_k t (num_ao, 3),
/// e = Sum_c t (N, 3))`. Same sums as `orbital_side` + `orb_grad_wrt_r`
/// (det) or `j3_side` + `j3_grad_wrt_r` (J3), with the product formed once.
pub(crate) fn kin_contract_bundle(
    b: &AoDerivBundle,
    ca: &Array2<f64>,
    cw: &Array2<f64>,
    cot: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let num_ao = ca.nrows();
    let n = ca.ncols();
    let mut s = Array2::<f64>::zeros((num_ao, 3));
    let mut e = Array2::<f64>::zeros((n, 3));
    let (g, h, gl) = (&b.grad, &b.hessian, &b.gradlap);
    for c in 0..num_ao {
        for k in 0..n {
            let (a, w) = (ca[[c, k]], cw[[c, k]]);
            let ck = [cot[[k, 0]], cot[[k, 1]], cot[[k, 2]]];
            for d in 0..3 {
                let hc =
                    ck[0] * h[d][0][[c, k]] + ck[1] * h[d][1][[c, k]] + ck[2] * h[d][2][[c, k]];
                let t = a * g[d][[c, k]] + w * (-0.5 * gl[d][[c, k]] - hc);
                s[[c, d]] += t;
                e[[k, d]] += t;
            }
        }
    }
    (s, e)
}

/// Leaves [`kin_force_sweep`] handles: molecular Cartesian GTO leaves with a
/// finalized shell table (the [`crate::ao_cart_deriv::ao_deriv_bundle_fast_cut`]
/// domain).
pub(crate) fn kin_sweep_ok(orb: &OrbData) -> bool {
    let a = &orb.aos;
    orb.pw.is_none()
        && orb.bspline.is_none()
        && a.kind == crate::atomic_orbital::AoKind::Cart
        && a.n_image == 1
        && a.num_ao_kernel == a.num_ao
        && !a.shells.is_empty()
}

/// [`kin_contract_bundle`] without the bundle: ONE shell-major sweep forms each
/// AO's gradient, Hessian and grad-Laplacian at each electron in registers and
/// contracts them with `ca`, `cw`, `cot` on the spot. Returns the AO gradient
/// tables (kept for the Pulay / ECP contractions) and `(s, e)`.
///
/// `cut`: the FAST derivative screen of [`crate::ao_cart_deriv::ao_deriv_bundle_fast_cut`]
/// (`pz_deriv r^2 > LEPSBAS` drops the Hessian and grad-Laplacian of the shell at
/// that electron; the gradient is never screened). Same 1D Hermite factors and
/// power tables as that bundle, so the kept entries equal it to fp rounding.
pub(crate) fn kin_force_sweep(
    aos: &crate::atomic_orbital::AosGtoData,
    r: &ArrayView2<f64>,
    ca: &Array2<f64>,
    cw: &Array2<f64>,
    cot: &ArrayView2<f64>,
    cut: bool,
) -> ([Array2<f64>; 3], Array2<f64>, Array2<f64>) {
    let lepsbas = crate::atomic_orbital::LEPSBAS;
    let n_e = r.nrows();
    let num_ao = aos.num_ao;
    let ca = ca.as_standard_layout();
    let cw = cw.as_standard_layout();
    let (ca, cw) = (ca.as_slice().unwrap(), cw.as_slice().unwrap());
    let mut g = [
        vec![0.0_f64; num_ao * n_e],
        vec![0.0_f64; num_ao * n_e],
        vec![0.0_f64; num_ao * n_e],
    ];
    let mut s = vec![[0.0_f64; 3]; num_ao];
    let mut eo = vec![[0.0_f64; 3]; n_e];
    let lmax = aos
        .shells
        .iter()
        .map(|sh| sh.l.max(0) as usize)
        .max()
        .unwrap_or(0);
    let pmax = aos
        .shells
        .iter()
        .map(|sh| sh.prim_z.len())
        .max()
        .unwrap_or(0);
    // tab[(q * 3 + a) * stride + 4 n + k] = P_k(t_a; n, Z_q), k <= 3, n <= l.
    let stride = 4 * (lmax + 1);
    let mut tab = vec![0.0_f64; pmax * 3 * stride];
    let mut ex = vec![0.0_f64; pmax];
    let mut pw = vec![0.0_f64; lmax + 4];
    for sh in &aos.shells {
        let l = sh.l.max(0) as usize;
        let [cx, cy, cz] = sh.center;
        for e in 0..n_e {
            let t = [r[[e, 0]] - cx, r[[e, 1]] - cy, r[[e, 2]] - cz];
            let r2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
            let alive = !cut || sh.pz_deriv * r2 <= lepsbas;
            for (q, &z) in sh.prim_z.iter().enumerate() {
                // the FAST e_L's exp (`atomic_orbital::exp_pd_into` core)
                ex[q] = crate::fastmath::exp_neg(-z * r2);
                for (a, t_a) in t.iter().enumerate() {
                    pw[0] = 1.0;
                    for m in 1..l + 4 {
                        pw[m] = pw[m - 1] * *t_a;
                    }
                    let p = |m: usize, back: usize| if m < back { 0.0 } else { pw[m - back] };
                    let base = (q * 3 + a) * stride;
                    for n in 0..=l {
                        let nf = n as f64;
                        let tb = &mut tab[base + 4 * n..base + 4 * n + 4];
                        tb[0] = pw[n];
                        tb[1] = nf * p(n, 1) - 2.0 * z * pw[n + 1];
                        if alive {
                            tb[2] = nf * (nf - 1.0) * p(n, 2) - 2.0 * z * (2.0 * nf + 1.0) * pw[n]
                                + 4.0 * z * z * pw[n + 2];
                            tb[3] = nf * (nf - 1.0) * (nf - 2.0) * p(n, 3)
                                - 6.0 * z * nf * nf * p(n, 1)
                                + 12.0 * z * z * (nf + 1.0) * pw[n + 1]
                                - 8.0 * z * z * z * pw[n + 3];
                        }
                    }
                }
            }
            let ck = [cot[[e, 0]], cot[[e, 1]], cot[[e, 2]]];
            for &ao in &sh.aos {
                let off = aos.ao_prim_offset[ao];
                let (nx, ny, nz) = (
                    4 * aos.poly_x[ao] as usize,
                    4 * aos.poly_y[ao] as usize,
                    4 * aos.poly_z[ao] as usize,
                );
                let mut gr = [0.0_f64; 3];
                // h = [xx, yy, zz, xy, xz, yz]; gl = grad of the Laplacian.
                let mut h = [0.0_f64; 6];
                let mut gl = [0.0_f64; 3];
                for (q, ex_q) in ex[..sh.prim_z.len()].iter().enumerate() {
                    let rad = aos.flat_nc[off + q] * *ex_q;
                    let xb = q * 3 * stride + nx;
                    let yb = (q * 3 + 1) * stride + ny;
                    let zb = (q * 3 + 2) * stride + nz;
                    let (x0, x1) = (tab[xb], tab[xb + 1]);
                    let (y0, y1) = (tab[yb], tab[yb + 1]);
                    let (z0, z1) = (tab[zb], tab[zb + 1]);
                    gr[0] += rad * x1 * y0 * z0;
                    gr[1] += rad * x0 * y1 * z0;
                    gr[2] += rad * x0 * y0 * z1;
                    if alive {
                        let (x2, x3) = (tab[xb + 2], tab[xb + 3]);
                        let (y2, y3) = (tab[yb + 2], tab[yb + 3]);
                        let (z2, z3) = (tab[zb + 2], tab[zb + 3]);
                        h[0] += rad * x2 * y0 * z0;
                        h[1] += rad * x0 * y2 * z0;
                        h[2] += rad * x0 * y0 * z2;
                        h[3] += rad * x1 * y1 * z0;
                        h[4] += rad * x1 * y0 * z1;
                        h[5] += rad * x0 * y1 * z1;
                        gl[0] += rad * (x3 * y0 * z0 + x1 * y2 * z0 + x1 * y0 * z2);
                        gl[1] += rad * (x2 * y1 * z0 + x0 * y3 * z0 + x0 * y1 * z2);
                        gl[2] += rad * (x2 * y0 * z1 + x0 * y2 * z1 + x0 * y0 * z3);
                    }
                }
                let idx = ao * n_e + e;
                let a = ca[idx];
                let mut tv = [a * gr[0], a * gr[1], a * gr[2]];
                if alive {
                    let w = cw[idx];
                    let hc = [
                        ck[0] * h[0] + ck[1] * h[3] + ck[2] * h[4],
                        ck[0] * h[3] + ck[1] * h[1] + ck[2] * h[5],
                        ck[0] * h[4] + ck[1] * h[5] + ck[2] * h[2],
                    ];
                    for d in 0..3 {
                        tv[d] += w * (-0.5 * gl[d] - hc[d]);
                    }
                }
                for d in 0..3 {
                    g[d][idx] = gr[d];
                    s[ao][d] += tv[d];
                    eo[e][d] += tv[d];
                }
            }
        }
    }
    let [gx, gy, gz] = g;
    let to2 = |v: Vec<f64>| Array2::from_shape_vec((num_ao, n_e), v).unwrap();
    let s = Array2::from_shape_fn((num_ao, 3), |(c, d)| s[c][d]);
    let eo = Array2::from_shape_fn((n_e, 3), |(k, d)| eo[k][d]);
    ([to2(gx), to2(gy), to2(gz)], s, eo)
}

/// The J3 same-spin pair sum `M (X K) + M^T (X K^T)` (`K` the strict-lower
/// ones matrix) for a symmetric coupling block `M`: `X K + X K^T = row sums - X`,
/// so it is the single product `M (X 1 1^T - X)`. `M = M^T` is the J3 contract
/// (jqmc `Jastrow_three_body_data.sanity_check`, asserted at native-context build
/// by `assert_j3_block_symmetric`) that the e_L's own J3 kernels
/// (`compute_grads_and_laplacian_jastrow_three_body`, the ratio state) rely on.
pub(crate) fn j3_pair_sum(m: &ArrayView2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let rs = x.sum_axis(numpy::ndarray::Axis(1));
    let y = Array2::from_shape_fn(x.raw_dim(), |(p, k)| rs[p] - x[[p, k]]);
    m.dot(&y)
}

/// J3 kernel coefficient `coef_X[:, k] = dJ3/dA_X[:, k]` per spin (the value
/// cotangent of the J3 AO table): `M (A K) + M^T (A K^T) + j1col + cross-spin`
/// ([`j3_pair_sum`]; `grad_j3_wrt_r`'s `coeff_up` / `coeff_dn`).
pub(crate) fn j3_coef(
    j_matrix: &ArrayView2<f64>,
    a_up: &Array2<f64>,
    a_dn: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let orb_n = a_up.nrows();
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]);
    let j1col = j_matrix.column(orb_n);
    let s_up = a_up.sum_axis(numpy::ndarray::Axis(1));
    let s_dn = a_dn.sum_axis(numpy::ndarray::Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let coef = |a: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
        let mut g = j3_pair_sum(&m, a);
        for p in 0..orb_n {
            let add = j1col[p] + cross[p];
            for c in 0..a.ncols() {
                g[[p, c]] += add;
            }
        }
        g
    };
    (coef(a_up, &m_s_dn), coef(a_dn, &mt_s_up))
}

/// J3 orbital-side kinetic force for BOTH derivatives, added to `fz`: the merged
/// `j3_side` (dR) and `stress::j3_grad_wrt_r` (dr). Both contract the same
/// `t = bar_A G + coef (-1/2 GL - cot.H)` per (AO, electron) with the full
/// `bar_A` (pair VJP plus the cross-spin sum terms), so it is formed once
/// ([`kin_contract_bundle`]). The pair sums are [`j3_pair_sum`] products.
fn j3_kin_force_merged(
    j_matrix: &ArrayView2<f64>,
    sj3: &KinSharedJ3,
    cot_up: &Array2<f64>,
    cot_dn: &Array2<f64>,
    nuc: &[usize],
    fz: &mut KinFused,
) {
    let (bu, bd) = (&sj3.bundle_up, &sj3.bundle_dn);
    let orb_n = sj3.a_up.nrows();
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]);
    let (coef_up, coef_dn) = j3_coef(j_matrix, &sj3.a_up, &sj3.a_dn);
    // bar_coef[a,k] = -1/2 lO - Sum_d cot[k,d] gO[d][a,k]
    let bcoef = |go: &[Array2<f64>; 3], lo: &Array2<f64>, cot: &Array2<f64>| -> Array2<f64> {
        Array2::from_shape_fn(lo.raw_dim(), |(a, k)| {
            -0.5 * lo[[a, k]]
                - (cot[[k, 0]] * go[0][[a, k]]
                    + cot[[k, 1]] * go[1][[a, k]]
                    + cot[[k, 2]] * go[2][[a, k]])
        })
    };
    let bcoef_u = bcoef(&bu.grad, &bu.laplacian, cot_up);
    let bcoef_d = bcoef(&bd.grad, &bd.laplacian, cot_dn);
    // bar_A = M^T (bcoef K^T) + M (bcoef K) + cross-spin sum
    let bar_a = |bc: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
        let mut g = j3_pair_sum(&m, bc);
        for p in 0..orb_n {
            for c in 0..bc.ncols() {
                g[[p, c]] += cross[p];
            }
        }
        g
    };
    let bar_a_up = bar_a(&bcoef_u, &m.dot(&bcoef_d.sum_axis(numpy::ndarray::Axis(1))));
    let bar_a_dn = bar_a(
        &bcoef_d,
        &m.t().dot(&bcoef_u.sum_axis(numpy::ndarray::Axis(1))),
    );
    let (s_u, e_u) = kin_contract_bundle(bu, &bar_a_up, &coef_up, &cot_up.view());
    let (s_d, e_d) = kin_contract_bundle(bd, &bar_a_dn, &coef_dn, &cot_dn.view());
    scatter_rows_neg(&s_u, nuc, &mut fz.d_r);
    scatter_rows_neg(&s_d, nuc, &mut fz.d_r);
    fz.d_re_up += &e_u;
    fz.d_re_dn += &e_d;
    fz.j3 = true;
}

/// `CM = CT*bar` (num_ao, N), or `bar` itself when there is no MO transform.
pub(crate) fn pull_to_ao(orb: &OrbData, bar: &Array2<f64>) -> Array2<f64> {
    match &orb.mo_coefficients {
        None => bar.clone(),
        Some(c) => c.t().dot(bar), // (num_ao, m)*(m, N)
    }
}

/// Scatter `s` (3, num_ao) into `(n_atom, 3)` by per-AO atom index `nuc`.
fn scatter(s: &Array2<f64>, nuc: &[usize], out: &mut Array2<f64>) {
    let num_ao = s.ncols();
    for c in 0..num_ao {
        let ia = nuc[c];
        for d in 0..3 {
            out[[ia, d]] += s[[d, c]];
        }
    }
}

/// Orbital-side force: translation identity over value/lap/grad cotangents.
/// `s[d,c] = -(Sum_k CM_orb*dAO + CM_lap*dgrad^2AO + Sum_e CM_gou_e*d(d_e AO))`.
#[allow(clippy::too_many_arguments)]
fn orbital_side(
    orb: &OrbData,
    r: &ArrayView2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: Option<&Array2<f64>>,
    bar_gou: Option<&[Array2<f64>; 3]>,
    nuc: &[usize],
    out: &mut Array2<f64>,
    precomp: Option<&AoDerivBundle>,
) {
    let aos = &orb.aos;
    let cm_orb = pull_to_ao(orb, bar_orb);
    let num_ao = cm_orb.nrows();
    let n = cm_orb.ncols();
    // Kinetic case (lap and/or gou present) needs grad+hessian+gradlap -> fuse one
    // sweep. Grad-only case (det Pulay) keeps the cheap 3-order grad. `precomp`
    // threads a bundle shared with the dr kinetic path.
    let need_ho = bar_lap.is_some() || bar_gou.is_some();
    let owned_bundle: Option<AoDerivBundle> = if precomp.is_none() && need_ho {
        Some(ao_deriv_bundle(aos, r))
    } else {
        None
    };
    let bundle = precomp.or(owned_bundle.as_ref());
    let owned_grad = if bundle.is_none() {
        Some(ao_grad(aos, r))
    } else {
        None
    };
    let ga: &[Array2<f64>; 3] = match bundle {
        Some(b) => &b.grad,
        None => owned_grad.as_ref().unwrap(),
    };
    let mut s = Array2::<f64>::zeros((3, num_ao));
    for d in 0..3 {
        for c in 0..num_ao {
            let mut acc = 0.0;
            for k in 0..n {
                acc += cm_orb[[c, k]] * ga[d][[c, k]];
            }
            s[[d, c]] = -acc;
        }
    }
    if let Some(bl) = bar_lap {
        let cm_lap = pull_to_ao(orb, bl);
        let gl = &bundle.expect("kinetic path has a bundle").gradlap;
        for d in 0..3 {
            for c in 0..num_ao {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += cm_lap[[c, k]] * gl[d][[c, k]];
                }
                s[[d, c]] -= acc;
            }
        }
    }
    if let Some(bg) = bar_gou {
        let cm_gou = [
            pull_to_ao(orb, &bg[0]),
            pull_to_ao(orb, &bg[1]),
            pull_to_ao(orb, &bg[2]),
        ];
        let h = &bundle.expect("kinetic path has a bundle").hessian;
        for d in 0..3 {
            for c in 0..num_ao {
                let mut acc = 0.0;
                for e in 0..3 {
                    for k in 0..n {
                        acc += cm_gou[e][[c, k]] * h[d][e][[c, k]];
                    }
                }
                s[[d, c]] -= acc;
            }
        }
    }
    scatter(&s, nuc, out);
}

/// `dln|detG|/dR` (det Pulay), `(n_atom, 3)`.
pub fn grad_lndet_wrt_r(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
) -> Array2<f64> {
    grad_lndet_wrt_r_gi(gem, r_up, r_dn, nuc_up, nuc_dn, n_atom, None)
}

/// `grad_lndet_wrt_r` reusing a caller-supplied geminal inverse (`gem_forward_gi`).
#[allow(clippy::too_many_arguments)]
pub fn grad_lndet_wrt_r_gi(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    let f = crate::param_grad_el::gem_forward_gi(gem, r_up, r_dn, g_inv);
    // coeff on d orb_up = WG; on d orb_dn = (G^-^1Q)[:N_dn]T
    let m_up = f.wg.clone(); // (M_up, N_up)
    let m_dn_coeff = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned(); // (M_dn, N_dn)
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    orbital_side(&gem.orb_up, r_up, &m_up, None, None, nuc_up, &mut out, None);
    orbital_side(
        &gem.orb_dn,
        r_dn,
        &m_dn_coeff,
        None,
        None,
        nuc_dn,
        &mut out,
        None,
    );
    out
}

/// `dJ1/dR`, `(n_atom, 3)`.
pub fn grad_j1_wrt_r(
    j1: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
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
    let mut out = Array2::<f64>::zeros((n_atom, 3));
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
            // diff = R - r ;  unit = (R-r)/|R-r|
            let dx = r_atoms[[ia, 0]] - re[0];
            let dy = r_atoms[[ia, 1]] - re[1];
            let dz = r_atoms[[ia, 2]] - re[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let dterm_dr = if is_exp {
                pref * (c / 2.0) * (-p * c * dist).exp()
            } else {
                pref / (2.0 * (1.0 + p * c * dist).powi(2))
            };
            let inv = 1.0 / dist;
            out[[ia, 0]] += dterm_dr * dx * inv;
            out[[ia, 1]] += dterm_dr * dy * inv;
            out[[ia, 2]] += dterm_dr * dz * inv;
        }
    }
    out
}

/// `dJ1/dR` with the PERIODIC Williamson smooth distance, `(n_atom, 3)`. Mirrors
/// `grad_j1_wrt_r` but the bare distance/unit-vector is replaced by the Williamson
/// `smooth_dist` `(d, grad)` (grad = dd/dr_e => dd/dR = -grad). jnone-cell falls back
/// to the bare distance (identical to `grad_j1_wrt_r`).
pub fn grad_j1_wrt_r_pbc(
    j1: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    use crate::jastrow_factor::{smooth_dist, CellInfo};
    let r_atoms = j1.structure_data.positions.view();
    let n_atom = j1.structure_data.n_atoms();
    let p = j1.jastrow_1b_param;
    let is_exp = j1.jastrow_1b_type == "exp";
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let ci = j1
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let mut out = Array2::<f64>::zeros((n_atom, 3));
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
            let dxyz = [
                re[0] - r_atoms[[ia, 0]],
                re[1] - r_atoms[[ia, 1]],
                re[2] - r_atoms[[ia, 2]],
            ];
            // d = smooth distance, ge = dd/dr_e ;  dd/dR = -ge
            let (dist, ge) = match &ci {
                Some(ci) => {
                    let (d, g, _, _) = smooth_dist(dxyz, ci);
                    (d, g)
                }
                None => {
                    let d = (dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]).sqrt();
                    (d, [dxyz[0] / d, dxyz[1] / d, dxyz[2] / d])
                }
            };
            let dterm_dr = if is_exp {
                pref * (c / 2.0) * (-p * c * dist).exp()
            } else {
                pref / (2.0 * (1.0 + p * c * dist).powi(2))
            };
            for d in 0..3 {
                out[[ia, d]] += dterm_dr * (-ge[d]);
            }
        }
    }
    out
}

/// Strict-lower-triangular ones `(n, n)`.
fn k_strict_lower(n: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            m[[i, j]] = 1.0;
        }
    }
    m
}

/// `dJ3/dR`, `(n_atom, 3)`. `j3_orb` is a Cartesian leaf.
pub fn grad_j3_wrt_r(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc: &[usize],
    n_atom: usize,
) -> Array2<f64> {
    let a_up = j3_orb.compute(r_up); // (orb, N_up)
    let a_dn = j3_orb.compute(r_dn);
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned(); // (orb, orb)
    let j1col = j_matrix.column(orb_n).to_owned(); // (orb,)
    let s_up = a_up.sum_axis(numpy::ndarray::Axis(1)); // (orb,)
    let s_dn = a_dn.sum_axis(numpy::ndarray::Axis(1));
    // coeff_X = j1col + M*A*K + MT*A*KT + cross-spin
    let m_s_dn = m.dot(&s_dn); // (orb,)
    let mt_s_up = m.t().dot(&s_up);
    let coeff_up = {
        let mut g = m.dot(&a_up).dot(&ku) + m.t().dot(&a_up).dot(&ku.t());
        for a in 0..orb_n {
            for k in 0..n_up {
                g[[a, k]] += j1col[a] + m_s_dn[a];
            }
        }
        g
    };
    let coeff_dn = {
        let mut g = m.dot(&a_dn).dot(&kd) + m.t().dot(&a_dn).dot(&kd.t());
        for a in 0..orb_n {
            for k in 0..n_dn {
                g[[a, k]] += j1col[a] + mt_s_up[a];
            }
        }
        g
    };
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    let mut do_side = |coeff: &Array2<f64>, r: &ArrayView2<f64>| {
        let ga = ao_grad(&j3_orb.aos, r); // [3] (orb, N)
        let n = coeff.ncols();
        let mut s = Array2::<f64>::zeros((3, orb_n));
        for d in 0..3 {
            for c in 0..orb_n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += coeff[[c, k]] * ga[d][[c, k]];
                }
                s[[d, c]] = -acc;
            }
        }
        scatter(&s, nuc, &mut out);
    };
    do_side(&coeff_up, r_up);
    do_side(&coeff_dn, r_dn);
    out
}

/// `dV/dR` classical Hellmann-Feynman (bare Coulomb e-N + N-N), `(n_atom, 3)`.
pub fn grad_v_wrt_r(
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let n_atom = positions.nrows();
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
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    // e-N: -Z_I Sum_e (r_e - R_I)/|r_e-R_I|^3
    for ia in 0..n_atom {
        let (rx, ry, rz) = (positions[[ia, 0]], positions[[ia, 1]], positions[[ia, 2]]);
        for e in 0..n_e {
            let re = elec(e);
            let dx = re[0] - rx;
            let dy = re[1] - ry;
            let dz = re[2] - rz;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let inv3 = 1.0 / (d * d * d);
            out[[ia, 0]] += -z_eff[ia] * dx * inv3;
            out[[ia, 1]] += -z_eff[ia] * dy * inv3;
            out[[ia, 2]] += -z_eff[ia] * dz * inv3;
        }
    }
    // N-N: -Sum_{J!=I} Z_I Z_J (R_I-R_J)/|R_I-R_J|^3
    for i in 0..n_atom {
        for j in 0..n_atom {
            if i == j {
                continue;
            }
            let dx = positions[[i, 0]] - positions[[j, 0]];
            let dy = positions[[i, 1]] - positions[[j, 1]];
            let dz = positions[[i, 2]] - positions[[j, 2]];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let inv3 = 1.0 / (d * d * d);
            let zz = z_eff[i] * z_eff[j];
            out[[i, 0]] += -zz * dx * inv3;
            out[[i, 1]] += -zz * dy * inv3;
            out[[i, 2]] += -zz * dz * inv3;
        }
    }
    out
}

/// `dT/dR` kinetic force (with optional Jastrow). `gJ=0` => jnone (det only).
/// Orbital side plus J1 and Cartesian J3 contributions.
#[allow(clippy::too_many_arguments)]
pub fn grad_t_wrt_r(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
) -> Array2<f64> {
    grad_t_wrt_r_w(
        gem, jastrow, j1, j3_orb, j3_jm, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, None,
    )
}

/// `grad_t_wrt_r` with per-electron weights `w` (global index: up 0..N_up, dn after):
/// computes `d(Sum_e w_e*kc_e)/dR` -- the masked kinetic force the GFMC ZV branch
/// corrections need. `None` = all-1 (the full `dT/dR`).
#[allow(clippy::too_many_arguments)]
pub fn grad_t_wrt_r_w(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
    w: Option<&[f64]>,
) -> Array2<f64> {
    grad_t_wrt_r_w_gi(
        gem, jastrow, j1, j3_orb, j3_jm, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, w, None,
    )
}

/// `grad_t_wrt_r_w` reusing a caller-supplied geminal inverse (`gem_forward_gi`):
/// the MCMC observable VMC HF-force path threads the sweep SM inverse so the
/// kinetic dT/dR stays near-singular-stable. `None` == fresh (bit-identical).
#[allow(clippy::too_many_arguments)]
pub fn grad_t_wrt_r_w_gi(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
    w: Option<&[f64]>,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    grad_t_wrt_r_w_gi_shared(
        gem, jastrow, j1, j3_orb, j3_jm, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, w, g_inv, None,
    )
}

/// `grad_t_wrt_r_w_gi` reusing the shared per-step kinetic fields (`KinShared`).
/// `shared = None` builds them locally (still fused per call). A `shared` build is
/// valid only for `w = None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grad_t_wrt_r_w_gi_shared(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
    w: Option<&[f64]>,
    g_inv: Option<&ArrayView2<f64>>,
    shared: Option<&KinShared>,
) -> Array2<f64> {
    let local;
    let sh: &KinShared = match shared {
        Some(s) => s,
        None => {
            local = crate::timed!(
                "f.dR_shared",
                KinShared::build(gem, jastrow, j3_orb, j3_jm, r_up, r_dn, w, g_inv)
            );
            &local
        }
    };
    let b = &sh.b;
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    // orbital side: value + lap + grad cotangents (fused det AO bundle, shared with dr)
    if let Some(fz) = &sh.fused {
        out += &fz.d_r; // det (+ J3 when `fz.j3`) orbital side, contracted in the build
    } else {
        crate::timed!("f.dR_orbside", {
            orbital_side(
                &gem.orb_up,
                r_up,
                &b.bar_orb_up,
                Some(&b.bar_lap_up),
                Some(&b.bar_gou),
                nuc_up,
                &mut out,
                sh.bundle_up.as_ref(),
            );
            orbital_side(
                &gem.orb_dn,
                r_dn,
                &b.bar_orb_dn,
                Some(&b.bar_lap_dn),
                Some(&b.bar_godn),
                nuc_dn,
                &mut out,
                sh.bundle_dn.as_ref(),
            );
        });
    }
    // Jastrow side (only if Jastrow present)
    if jastrow.is_some() {
        let (gj_up, gj_dn) = (&sh.gj_up, &sh.gj_dn);
        let (gd_up, gd_dn) = (&sh.gd_up, &sh.gd_dn);
        let n_up = r_up.nrows();
        // bar_gJ = -(gJ+gD) (Ne,3); bar_lapJ = -0.5 (both weighted per element in the sides)
        let bar_gj = |e: usize, d: usize| -> f64 {
            if e < n_up {
                -(gj_up[[e, d]] + gd_up[[e, d]])
            } else {
                let l = e - n_up;
                -(gj_dn[[l, d]] + gd_dn[[l, d]])
            }
        };
        if let Some(j1) = j1 {
            j1_side(j1, r_up, r_dn, &bar_gj, w, &mut out);
        }
        let j3_fused = sh.fused.as_ref().is_some_and(|fz| fz.j3);
        if let (Some(orb), Some(jm), Some(nuc), false) = (j3_orb, j3_jm, nuc_j3, j3_fused) {
            crate::timed!(
                "f.dR_j3side",
                j3_side(
                    orb,
                    jm,
                    r_up,
                    r_dn,
                    &bar_gj,
                    nuc,
                    w,
                    &mut out,
                    sh.j3.as_ref()
                )
            );
        }
    }
    out
}

/// J1 Jastrow-side kinetic force.
pub(crate) fn j1_side(
    j1: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bar_gj: &dyn Fn(usize, usize) -> f64,
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let r_atoms = j1.structure_data.positions.view();
    let n_atom = j1.structure_data.n_atoms();
    let p = j1.jastrow_1b_param;
    let is_exp = j1.jastrow_1b_type == "exp";
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    for e in 0..n_e {
        let re = elec(e);
        let we = w.map_or(1.0, |w| w[e]);
        if we == 0.0 {
            continue;
        }
        let bg = [we * bar_gj(e, 0), we * bar_gj(e, 1), we * bar_gj(e, 2)];
        for ia in 0..n_atom {
            let z = j1.structure_data.atomic_numbers[ia] - j1.core_electrons[ia];
            let c = (2.0 * z).powf(0.25);
            let pref = if is_exp {
                -(2.0 * z).powf(0.75)
            } else {
                -2.0 * z
            };
            // diff = r - R ; unit = (r-R)/|r-R|
            let dx = re[0] - r_atoms[[ia, 0]];
            let dy = re[1] - r_atoms[[ia, 1]];
            let dz = re[2] - r_atoms[[ia, 2]];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let u = [dx / dist, dy / dist, dz / dist];
            // f', f'', f''' (per j1 type), h' = f'''+2f''/r-2f'/r^2
            let (fp, fpp, fppp) = if is_exp {
                let a = p * c;
                let ee = (-a * dist).exp();
                (
                    pref * c / 2.0 * ee,
                    -pref * (p * c * c) / 2.0 * ee,
                    pref * (p * p * c * c * c) / 2.0 * ee,
                )
            } else {
                let bb = p * c;
                let uu = 1.0 + bb * dist;
                (
                    pref / (2.0 * uu * uu),
                    -pref * bb / uu.powi(3),
                    3.0 * pref * bb * bb / uu.powi(4),
                )
            };
            let hp = fppp + 2.0 * fpp / dist - 2.0 * fp / (dist * dist);
            let bu = bg[0] * u[0] + bg[1] * u[1] + bg[2] * u[2]; // bar_gJ*u_hat
            let bar_lap_j = -0.5 * we;
            for d in 0..3 {
                // term_g = -[ f''(bu)u_d + (f'/r)(bg_d - bu*u_d) ]
                let term_g = -(fpp * bu * u[d] + (fp / dist) * (bg[d] - bu * u[d]));
                // term_l = -bar_lapJ*(h' u_d)
                let term_l = -bar_lap_j * (hp * u[d]);
                out[[ia, d]] += term_g + term_l;
            }
        }
    }
}

/// PERIODIC J1 Jastrow-side kinetic force over the **Williamson smooth** e-N
/// distance (jqmc PBC J1 uses Williamson -- unlike DTN's min-image). Generalises
/// `j1_side`: replaces the bare-distance geometry (u_hat, 1, 2/r, (I-u_hat(x)u_hat)/r, -2/r^2u_hat)
/// with `smooth_dist_hess` (g, q, L, H, gradq, gradL). Per (e,I):
/// `f[d] += -(f''*g_d*(bg*g) + f'*(bg*H)_d) + 1/2(f'''*g_d*q + f''*gradq_d + f''*g_d*L + f'*gradL_d)`,
/// `bg = bar_gJ = -(gJ+gD)`. Reduces to `j1_side` for the bare metric.
#[allow(clippy::too_many_arguments)]
pub(crate) fn j1_side_pbc(
    j1: &JastrowOneBodyData,
    ci: &crate::jastrow_factor::CellInfo,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bar_gj: &dyn Fn(usize, usize) -> f64,
    n_e: usize,
    n_up: usize,
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    let r_atoms = j1.structure_data.positions.view();
    let n_atom = j1.structure_data.n_atoms();
    let p = j1.jastrow_1b_param;
    let is_exp = j1.jastrow_1b_type == "exp";
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    for e in 0..n_e {
        let re = elec(e);
        let we = w.map_or(1.0, |w| w[e]);
        if we == 0.0 {
            continue;
        }
        let bg = [we * bar_gj(e, 0), we * bar_gj(e, 1), we * bar_gj(e, 2)];
        for ia in 0..n_atom {
            let z = j1.structure_data.atomic_numbers[ia] - j1.core_electrons[ia];
            let c = (2.0 * z).powf(0.25);
            let pref = if is_exp {
                -(2.0 * z).powf(0.75)
            } else {
                -2.0 * z
            };
            let x = [
                re[0] - r_atoms[[ia, 0]],
                re[1] - r_atoms[[ia, 1]],
                re[2] - r_atoms[[ia, 2]],
            ];
            let (d, g, q, lap, h, gl) = crate::jastrow_factor::smooth_dist_hess(x, ci);
            // J1 radial kernel u(d): f'(d), f''(d), f'''(d).
            let (fp, fpp, fppp) = if is_exp {
                let a = p * c;
                let ee = (-a * d).exp();
                (
                    pref * c / 2.0 * ee,
                    -pref * (p * c * c) / 2.0 * ee,
                    pref * (p * p * c * c * c) / 2.0 * ee,
                )
            } else {
                let bb = p * c;
                let uu = 1.0 + bb * d;
                (
                    pref / (2.0 * uu * uu),
                    -pref * bb / uu.powi(3),
                    3.0 * pref * bb * bb / uu.powi(4),
                )
            };
            let bg_g = bg[0] * g[0] + bg[1] * g[1] + bg[2] * g[2];
            // gradq = 2 H g (recompute the bg*H contraction directly)
            let gq = [
                2.0 * (h[0][0] * g[0] + h[0][1] * g[1] + h[0][2] * g[2]),
                2.0 * (h[1][0] * g[0] + h[1][1] * g[1] + h[1][2] * g[2]),
                2.0 * (h[2][0] * g[0] + h[2][1] * g[1] + h[2][2] * g[2]),
            ];
            for d_ in 0..3 {
                let bg_h = bg[0] * h[0][d_] + bg[1] * h[1][d_] + bg[2] * h[2][d_];
                let cross = -(fpp * g[d_] * bg_g + fp * bg_h);
                let lap_term =
                    we * 0.5 * (fppp * g[d_] * q + fpp * gq[d_] + fpp * g[d_] * lap + fp * gl[d_]);
                out[[ia, d_]] += cross + lap_term;
            }
        }
    }
}

/// Cartesian J3 Jastrow-side kinetic force.
#[allow(clippy::too_many_arguments)]
pub(crate) fn j3_side(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bar_gj: &dyn Fn(usize, usize) -> f64,
    nuc: &[usize],
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
    precomp: Option<&KinSharedJ3>,
) {
    let aos = &j3_orb.aos;
    // value + fused (grad + lap + hessian + gradlap) of the J3 cart basis, shared
    // with the dr J3 kernel when `precomp` is threaded.
    let owned: Option<(Array2<f64>, Array2<f64>, AoDerivBundle, AoDerivBundle)> =
        if precomp.is_none() {
            Some((
                crate::atomic_orbital::compute_aos(aos, r_up),
                crate::atomic_orbital::compute_aos(aos, r_dn),
                ao_deriv_bundle(aos, r_up),
                ao_deriv_bundle(aos, r_dn),
            ))
        } else {
            None
        };
    let (a_up, a_dn, bu, bd) = match precomp {
        Some(j3) => (&j3.a_up, &j3.a_dn, &j3.bundle_up, &j3.bundle_dn),
        None => {
            let o = owned.as_ref().unwrap();
            (&o.0, &o.1, &o.2, &o.3)
        }
    };
    let go_u = &bu.grad; // [3] (orb, N_up)
    let go_d = &bd.grad;
    let lo_u = &bu.laplacian; // (orb, N_up)
    let lo_d = &bd.laplacian;
    let hu = &bu.hessian;
    let hd = &bd.hessian;
    let glu = &bu.gradlap;
    let gld = &bd.gradlap;

    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(numpy::ndarray::Axis(1));
    let s_dn = a_dn.sum_axis(numpy::ndarray::Axis(1));
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
    let coef_up = mk(a_up, &ku, &m_s_dn); // (orb, N_up)
    let coef_dn = mk(a_dn, &kd, &mt_s_up);

    // side: bar_gO[d,a,k]=bg[k,d]*coef[a,k]; bar_lO[a,k]=bl*coef; bar_coefA[a,k]=Sum_d bg[k,d]*gO[d,a,k]+bl*lO
    // then bar_A via VJP, and force_side contracts bar_A*gO + bar_gO*H + bar_lO*GL.
    let process = |coef: &Array2<f64>,
                   go: &[Array2<f64>; 3],
                   lo: &Array2<f64>,
                   spin_up: bool|
     -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        let nc = coef.ncols();
        let off = if spin_up { 0 } else { n_up };
        // bar_coefA (orb, N) -- per-electron weighted (kc_k seeds live in column k)
        let mut bcoef = Array2::<f64>::zeros((orb_n, nc));
        for a in 0..orb_n {
            for k in 0..nc {
                let wk = w.map_or(1.0, |w| w[off + k]);
                let mut acc = 0.0;
                for (d, go_d) in go[..3].iter().enumerate() {
                    acc += bar_gj(off + k, d) * go_d[[a, k]];
                }
                bcoef[[a, k]] = wk * (acc + (-0.5) * lo[[a, k]]);
            }
        }
        // bar_gO[d] (orb,N), bar_lO (orb,N)
        let mut bgo = [
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
        ];
        let mut blo = Array2::<f64>::zeros((orb_n, nc));
        for a in 0..orb_n {
            for k in 0..nc {
                let wk = w.map_or(1.0, |w| w[off + k]);
                for (d, bgo_d) in bgo.iter_mut().enumerate() {
                    bgo_d[[a, k]] = wk * bar_gj(off + k, d) * coef[[a, k]];
                }
                blo[[a, k]] = -0.5 * wk * coef[[a, k]];
            }
        }
        (bcoef, bgo, blo)
    };
    let (bcoef_u, bgo_u, blo_u) = process(&coef_up, go_u, lo_u, true);
    let (bcoef_d, bgo_d, blo_d) = process(&coef_dn, go_d, lo_d, false);

    // bar_A via VJP of coefA = M*A*K + MT*A*KT + cross-spin sum
    let bcoef_d_sum = bcoef_d.sum_axis(numpy::ndarray::Axis(1)); // (orb,)
    let bcoef_u_sum = bcoef_u.sum_axis(numpy::ndarray::Axis(1));
    let bar_a_up = {
        let mut g = m.t().dot(&bcoef_u).dot(&ku.t()) + m.dot(&bcoef_u).dot(&ku);
        let add = m.dot(&bcoef_d_sum); // (orb,)
        for a in 0..orb_n {
            for k in 0..n_up {
                g[[a, k]] += add[a];
            }
        }
        g
    };
    let bar_a_dn = {
        let mut g = m.t().dot(&bcoef_d).dot(&kd.t()) + m.dot(&bcoef_d).dot(&kd);
        let add = m.t().dot(&bcoef_u_sum);
        for a in 0..orb_n {
            for k in 0..n_dn {
                g[[a, k]] += add[a];
            }
        }
        g
    };

    // force_side: s[d,a] = -(bar_A*dA + bar_gO*dgO[H] + bar_lO*dlO[GL])
    let force_side = |bar_a: &Array2<f64>,
                      bgo: &[Array2<f64>; 3],
                      blo: &Array2<f64>,
                      go: &[Array2<f64>; 3],
                      h: &[[Array2<f64>; 3]; 3],
                      gl: &[Array2<f64>; 3],
                      out: &mut Array2<f64>| {
        let nc = bar_a.ncols();
        let mut s = Array2::<f64>::zeros((3, orb_n));
        for d in 0..3 {
            for a in 0..orb_n {
                let mut acc = 0.0;
                for k in 0..nc {
                    acc += bar_a[[a, k]] * go[d][[a, k]];
                    for e in 0..3 {
                        acc += bgo[e][[a, k]] * h[d][e][[a, k]];
                    }
                    acc += blo[[a, k]] * gl[d][[a, k]];
                }
                s[[d, a]] = -acc;
            }
        }
        scatter(&s, nuc, out);
    };
    force_side(&bar_a_up, &bgo_u, &blo_u, go_u, hu, glu, out);
    force_side(&bar_a_dn, &bgo_d, &blo_d, go_d, hd, gld, out);
}

#[cfg(test)]
mod fused_kin_tests {
    use super::*;
    use crate::atomic_orbital::AoKind;

    /// `|x - y| <= atol + rtol * scale`; `scale` = the sum of the magnitudes of
    /// the summed terms (the reduction-order bound of a sum with cancellation).
    fn close_scaled(x: f64, y: f64, scale: f64, atol: f64, rtol: f64) -> bool {
        (x - y).abs() <= atol + rtol * scale
    }

    /// The fused sweep contracts the same kinetic product as the separate-table
    /// route. Independent reference: the prim-outer derivative bundles
    /// ([`ao_deriv_bundle`], [`crate::ao_cart_deriv::ao_deriv_bundle_fast_cut`])
    /// contracted with the `geminal_reverse`-form cotangents `cm_lap = -1/2 cw`,
    /// `cm_gou[e] = -cot[:, e] cw` (the `orbital_side` / `orb_grad_wrt_r` sums),
    /// with and without the FAST derivative screen; points span radii where
    /// shells are dropped and kept.
    #[test]
    fn kin_sweep_matches_bundle_contraction() {
        let d = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 1, 0.0, true);
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let mut pv = Vec::new();
        for &rad in &[0.2, 0.7, 1.3, 2.1, 3.4, 5.0, 7.5] {
            for u in [[0.6, 0.8, 0.0], [0.0, -0.6, 0.8], [0.48, 0.6, -0.64]] {
                pv.push([0.1 + rad * u[0], -0.2 + rad * u[1], 0.3 + rad * u[2]]);
            }
        }
        let n = pv.len();
        let r = Array2::from_shape_vec((n, 3), pv.concat()).unwrap();
        let na = d.num_ao;
        let ca = Array2::from_shape_fn((na, n), |(c, k)| ((c * 7 + k * 3) % 11) as f64 / 5.0 - 1.0);
        let cw = Array2::from_shape_fn((na, n), |(c, k)| ((c * 5 + k * 13) % 9) as f64 / 4.0 - 1.0);
        let cot = Array2::from_shape_fn((n, 3), |(k, e)| ((k * 3 + e * 7) % 5) as f64 / 2.0 - 1.0);
        let mut n_dead = 0usize;
        for cut in [false, true] {
            let b = if cut {
                crate::ao_cart_deriv::ao_deriv_bundle_fast_cut(&d, &r.view())
            } else {
                ao_deriv_bundle(&d, &r.view())
            };
            let mut s_ref = Array2::<f64>::zeros((na, 3));
            let mut e_ref = Array2::<f64>::zeros((n, 3));
            let mut s_mag = Array2::<f64>::zeros((na, 3));
            let mut e_mag = Array2::<f64>::zeros((n, 3));
            for c in 0..na {
                for k in 0..n {
                    let cm_lap = -0.5 * cw[[c, k]];
                    for dd in 0..3 {
                        let mut t =
                            ca[[c, k]] * b.grad[dd][[c, k]] + cm_lap * b.gradlap[dd][[c, k]];
                        for e in 0..3 {
                            t += -cot[[k, e]] * cw[[c, k]] * b.hessian[dd][e][[c, k]];
                        }
                        s_ref[[c, dd]] += t;
                        e_ref[[k, dd]] += t;
                        s_mag[[c, dd]] += t.abs();
                        e_mag[[k, dd]] += t.abs();
                    }
                }
            }
            let (g, s, e) = kin_force_sweep(&d, &r.view(), &ca, &cw, &cot.view(), cut);
            for dd in 0..3 {
                for c in 0..na {
                    for k in 0..n {
                        let (x, y) = (g[dd][[c, k]], b.grad[dd][[c, k]]);
                        assert!(
                            (x - y).abs() <= atol + rtol * y.abs(),
                            "grad cut={cut} {dd} {c} {k}: {x} vs {y}"
                        );
                    }
                    let (x, y) = (s[[c, dd]], s_ref[[c, dd]]);
                    assert!(
                        close_scaled(x, y, s_mag[[c, dd]], atol, rtol),
                        "s cut={cut} {c} {dd}: {x} vs {y}"
                    );
                }
                for k in 0..n {
                    let (x, y) = (e[[k, dd]], e_ref[[k, dd]]);
                    assert!(
                        close_scaled(x, y, e_mag[[k, dd]], atol, rtol),
                        "e cut={cut} {k} {dd}: {x} vs {y}"
                    );
                }
            }
            if cut {
                for sh in &d.shells {
                    for k in 0..n {
                        let r2: f64 = (0..3).map(|a| (r[[k, a]] - sh.center[a]).powi(2)).sum();
                        if sh.pz_deriv * r2 > crate::atomic_orbital::LEPSBAS {
                            n_dead += 1;
                        }
                    }
                }
            }
        }
        assert!(n_dead > 0, "derivative screen not exercised");
    }
}
