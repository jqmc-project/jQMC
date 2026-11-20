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

//! Basis-parameter `O_k`: `dln|detG|/dradial.{exponents,coefficients}` for
//! Cartesian GTO, spherical GTO, and periodic-pGTO geminals. The Gaussian
//! exponents/coefficients are
//! per-SHELL `radial` DOF; the full per-AO-prim arrays are
//! `expand(radial, ao_prim_to_shell_prim)`. So the gradient is a segment_sum
//! scatter of the per-AO-prim sensitivities over that map.
//!
//! The Z-power of the primitive normalization is identical for cart and sphe
//! (`N_p ~ Z_p^{(2l+3)/4}`), so the radial sensitivity is angular-kind agnostic:
//! only the angular factor `P` (cart monomial vs real solid harmonic) differs.
//! For periodic pGTO the per-primitive AOs live at kernel rank (image-shifted
//! centers) and fold back to the home AO via `kernel_to_home` -- the radial DOF is
//! shared across images, so every image's sensitivity scatters to the SAME
//! shell-prim, contracted with the home-AO cotangent.

use numpy::ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::Complex64;

use crate::atomic_orbital::{sphe_angular_value_grad, AoKind, AosGtoData};
use crate::complex_det::{
    complex_gem_forward_clam, complex_geminal_reverse_clam, ComplexGeminal, ComplexOrb,
};
use crate::data::JastrowData;
use crate::determinant::{orb_value_grad_lap, GeminalData, OrbData};
use crate::forces::pull_to_ao;
use crate::param_grad_el::{gem_forward, geminal_reverse, jastrow_grads, kinetic_drift, wrap_both};

/// Per-spin radial basis gradients `((d/dexp_up, d/dcoeff_up), (d/dexp_dn, d/dcoeff_dn))`.
pub type RadialGrads<T = f64> = ((Vec<T>, Vec<T>), (Vec<T>, Vec<T>));

/// Wrap electron positions into the home cell `[0,1)^3` (jqmc
/// `_wrap_into_cell`: `frac = r*inv_cell; frac -= floor(frac); r = frac*cell`).
/// For periodic pGTO the per-image radial attribution depends on WHICH image is
/// nearest the wrapped electron -- the folded value/cotangent is wrap-invariant
/// (so gem_forward is unaffected) but `d/d(radial[image])` is not, so the
/// sensitivity loop must see the SAME wrapped `r` jqmc's AO eval uses. Molecular
/// (no `cell`) returns `r` unchanged.
pub(crate) fn wrap_into_cell(r: &ArrayView2<f64>, cell: Option<&ArrayView2<f64>>) -> Array2<f64> {
    let c = match cell {
        None => return r.to_owned(),
        Some(c) => c,
    };
    let inv = crate::ewald::inv3(c);
    let n = r.nrows();
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let mut fl = [0.0f64; 3];
        for (a, f) in fl.iter_mut().enumerate() {
            let frac = r[[i, 0]] * inv[0][a] + r[[i, 1]] * inv[1][a] + r[[i, 2]] * inv[2][a];
            *f = frac.floor();
        }
        for c2 in 0..3 {
            out[[i, c2]] =
                r[[i, c2]] - (fl[0] * c[[0, c2]] + fl[1] * c[[1, c2]] + fl[2] * c[[2, c2]]);
        }
    }
    out
}

/// Slice the GTO child's home-AO rows out of a cotangent that spans the whole
/// composite AO axis. `bar_ao` is `(ao_count, N)` = `(num_gto_ao + num_extra_ao, N)`
/// in child order; the GTO block starts at `orb.gto_ao_offset()` and is
/// `orb.aos.num_ao` rows long. A pure GTO leaf has offset 0 and `num_ao` == the
/// full row count, so this is a no-op (returns an owned copy).
#[inline]
fn pull_gto_rows(orb: &OrbData, bar_ao: &Array2<f64>) -> Array2<f64> {
    let off = orb.gto_ao_offset();
    let ng = orb.aos.num_ao;
    if off == 0 && ng == bar_ao.nrows() {
        return bar_ao.clone();
    }
    bar_ao
        .slice(numpy::ndarray::s![off..off + ng, ..])
        .to_owned()
}

/// Angular factor `P` of AO `ao` at `(dx,dy,dz)` (`r2 = dx^2+dy^2+dz^2`): the
/// cart monomial `x^nx y^ny z^nz` or the sphe real solid harmonic.
#[inline]
fn angular_value(aos: &AosGtoData, ao: usize, dx: f64, dy: f64, dz: f64, r2: f64) -> f64 {
    match aos.kind {
        AoKind::Cart => {
            dx.powi(aos.poly_x[ao] as i32)
                * dy.powi(aos.poly_y[ao] as i32)
                * dz.powi(aos.poly_z[ao] as i32)
        }
        AoKind::Sphe => aos.sphe_angular[ao].eval(dx, dy, dz, r2),
    }
}

/// Per-AO-prim sensitivities of one AO leaf, contracted with the det cotangent
/// `bar_ao = dln|detG|/dAO` (home-AO rank) over electrons, scattered to the
/// `n_shell` shell-prims via `a2s = ao_prim_to_shell_prim`. Returns
/// `(d/dexponents, d/dcoefficients)` each length `n_shell`. Serves cart/sphe and
/// molecular/periodic (kernel AO -> home AO via `kernel_to_home`).
///
/// For the unit-coefficient primitive `U_p(r) = N_p P exp(-Z_p|r-R|^2)`,
/// `dAO/dZ_p = c_p [(2l_p+3)/(4Z_p) - |r-R|^2] U_p` and `dAO/dc_p = U_p`.
fn grad_radial_side(
    aos: &AosGtoData,
    a2s: &[usize],
    n_shell: usize,
    bar_ao: &Array2<f64>,
    r: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n = r.nrows();
    let p_count = aos.orbital_indices.len();
    let num_home = bar_ao.nrows();
    let mut gz = vec![0.0f64; n_shell];
    let mut gc = vec![0.0f64; n_shell];
    for pp in 0..p_count {
        let ao = aos.orbital_indices[pp];
        let home = aos.kernel_to_home[ao]; // == ao for molecular
        debug_assert!(home < num_home && a2s[pp] < n_shell);
        let z = aos.exponents[pp];
        let c = aos.coefficients[pp];
        let nrm = aos.normalizations[pp];
        let l = aos.angular_momentums[ao] as f64;
        let cx = aos.centers[[ao, 0]];
        let cy = aos.centers[[ao, 1]];
        let cz = aos.centers[[ao, 2]];
        let mut acc_z = 0.0;
        let mut acc_c = 0.0;
        for k in 0..n {
            let dx = r[[k, 0]] - cx;
            let dy = r[[k, 1]] - cy;
            let dz = r[[k, 2]] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let ang = angular_value(aos, ao, dx, dy, dz, r2);
            let sens_c = nrm * ang * (-z * r2).exp();
            let sens_z = ((2.0 * l + 3.0) / (4.0 * z) - r2) * (c * sens_c);
            let b = bar_ao[[home, k]];
            acc_z += b * sens_z;
            acc_c += b * sens_c;
        }
        gz[a2s[pp]] += acc_z;
        gc[a2s[pp]] += acc_c * aos.coefficient_scale(ao);
    }
    (gz, gc)
}

/// `dln|detG|/dradial.{exp,coeff}` for up/dn cart AO leaves. Returns
/// `((gZ_up, gc_up), (gZ_dn, gc_dn))`, each component length = that leaf's
/// `radial.exponents`. AGP (orb=AO) and SD/AGPn (orb=C*AO) via `pull_to_ao`.
#[allow(clippy::too_many_arguments)]
pub fn grad_radial(
    gem: &GeminalData,
    a2s_up: &[usize],
    n_shell_up: usize,
    a2s_dn: &[usize],
    n_shell_dn: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: Option<&ArrayView2<f64>>,
) -> RadialGrads {
    // Wrap into the home cell for periodic leaves so the per-image radial
    // attribution matches jqmc's AO eval (folded value/cotangent is wrap-invariant).
    let ru = wrap_into_cell(r_up, cell);
    let rd = wrap_into_cell(r_dn, cell);
    let (r_up, r_dn) = (&ru.view(), &rd.view());
    let f = gem_forward(gem, r_up, r_dn);
    let bar_orb_up = f.wg.clone(); // (M_up, N_up) = W G^-^1
    let bar_orb_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned(); // (M_dn, N_dn)
                                                                                   // For a composite (pgto(+)pw / pgto(+)bspline) the pulled cotangent spans the
                                                                                   // whole composite AO axis; the GTO child's home rows sit at `gto_ao_offset`.
                                                                                   // Slice them out so `grad_radial_side`'s home index (0..num_gto_ao) is aligned
                                                                                   // (a no-op for a pure GTO leaf: offset 0, num_ao == full row count). The non-GTO
                                                                                   // rows carry no radial DOF, so this exactly matches Python's `_concat_gto_radial`
                                                                                   // (GTO children only, in child order).
    let bar_ao_up = pull_gto_rows(&gem.orb_up, &pull_to_ao(&gem.orb_up, &bar_orb_up)); // (num_gto_ao_up, N_up)
    let bar_ao_dn = pull_gto_rows(&gem.orb_dn, &pull_to_ao(&gem.orb_dn, &bar_orb_dn));
    let up = grad_radial_side(&gem.orb_up.aos, a2s_up, n_shell_up, &bar_ao_up, r_up);
    let dn = grad_radial_side(&gem.orb_dn.aos, a2s_dn, n_shell_dn, &bar_ao_dn, r_dn);
    (up, dn)
}

/// 1D Hermite-Gaussian derivative factor `P(t,n,Z,a) = d^a_t(t^n e^{-Zt^2})/e^{-Zt^2}`
/// for a<=2 (the orders the Laplacian basis sensitivity needs).
#[inline]
fn p1(t: f64, n: i32, z: f64, a: u32) -> f64 {
    let nf = n as f64;
    match a {
        0 => t.powi(n),
        1 => {
            let t0 = if n >= 1 { nf * t.powi(n - 1) } else { 0.0 };
            t0 - 2.0 * z * t.powi(n + 1)
        }
        _ => {
            let t0 = if n >= 2 {
                nf * (nf - 1.0) * t.powi(n - 2)
            } else {
                0.0
            };
            t0 - 2.0 * z * (2.0 * nf + 1.0) * t.powi(n) + 4.0 * z * z * t.powi(n + 2)
        }
    }
}

/// `(prim, lap, u.grad, grad[3])` of ONE Gaussian primitive of AO `ao` at the
/// displacement `u = (dx,dy,dz)`, `r2 = |u|^2`, with the radial prefactor
/// `rad = N_p e^{-Z_p r2}` already formed. Cart uses the 1D Hermite factors,
/// sphe the solid-harmonic value/grad. Shared by the real and the complex-Bloch
/// kinetic radial sides so the two can never drift apart.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "scalar inputs of an inlined hot primitive helper"
)]
fn prim_vgl(
    aos: &AosGtoData,
    ao: usize,
    z: f64,
    rad: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    r2: f64,
) -> (f64, f64, f64, [f64; 3]) {
    match aos.kind {
        AoKind::Cart => {
            let nx = aos.poly_x[ao] as i32;
            let ny = aos.poly_y[ao] as i32;
            let nz = aos.poly_z[ao] as i32;
            let px0 = p1(dx, nx, z, 0);
            let py0 = p1(dy, ny, z, 0);
            let pz0 = p1(dz, nz, z, 0);
            let prim = rad * px0 * py0 * pz0;
            // grad^2prim = D(2,0,0)+D(0,2,0)+D(0,0,2)
            let lap = rad
                * (p1(dx, nx, z, 2) * py0 * pz0
                    + px0 * p1(dy, ny, z, 2) * pz0
                    + px0 * py0 * p1(dz, nz, z, 2));
            // gradprim_d = D(1,0,0)/D(0,1,0)/D(0,0,1)
            let gx = rad * p1(dx, nx, z, 1) * py0 * pz0;
            let gy = rad * px0 * p1(dy, ny, z, 1) * pz0;
            let gz_p = rad * px0 * py0 * p1(dz, nz, z, 1);
            (prim, lap, dx * gx + dy * gy + dz * gz_p, [gx, gy, gz_p])
        }
        AoKind::Sphe => {
            // prim = rad*S; gradprim_d = rad*dS_d - 2Z u_d prim;
            // lap = rad[(4Z^2 r^2 - 6Z)S - 4Z (u.dS)]; dgrad = u.gradprim.
            let (pp_val, gpx, gpy, gpz) = sphe_angular_value_grad(aos, ao, dx, dy, dz, r2);
            let r_dot_gp = dx * gpx + dy * gpy + dz * gpz;
            let prim = rad * pp_val;
            let lap = rad * ((4.0 * z * z * r2 - 6.0 * z) * pp_val - 4.0 * z * r_dot_gp);
            let grad = [
                rad * gpx - 2.0 * z * dx * prim,
                rad * gpy - 2.0 * z * dy * prim,
                rad * gpz - 2.0 * z * dz * prim,
            ];
            let dgrad = dx * grad[0] + dy * grad[1] + dz * grad[2];
            (prim, lap, dgrad, grad)
        }
    }
}

/// Kinetic `dT_det/dradial` for one AO leaf: value cotangent `bar_o` on
/// `dAO/d{Z,c}` + Laplacian cotangent `bar_l` on `d(grad^2AO)/d{Z,c}` + orbital-
/// GRADIENT cotangent `bar_g` (3 comps) on `d(gradAO)/d{Z,c}` -- the last is the
/// `-Sum gJ.gD` Jastrow cross term (zero for jnone, where `bar_g==0`). Per-AO-prim,
/// scattered to `n_shell` shells. Serves cart (Hermite factors) and sphe (solid-
/// harmonic moments), molecular and periodic (kernel AO -> home AO via
/// `kernel_to_home`). 1:1 basis.py `_prim_sens` / `_basis_contract`.
///
/// Gradient sensitivity (angular-agnostic, since the angular factor is Z/c-free):
/// For a unit-coefficient primitive U, `sZ_grad[d] = c*((a-r2)*grad(U)[d] - 2*u_d*U)`,
/// `sc_grad[d] = grad(U)[d]`.
fn grad_radial_kinetic_side(
    aos: &AosGtoData,
    a2s: &[usize],
    n_shell: usize,
    bar_o: &Array2<f64>,
    bar_l: &Array2<f64>,
    bar_g: &[Array2<f64>; 3],
    r: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n = r.nrows();
    let p_count = aos.orbital_indices.len();
    let num_home = bar_o.nrows();
    let mut gz = vec![0.0f64; n_shell];
    let mut gc = vec![0.0f64; n_shell];
    for pp in 0..p_count {
        let ao = aos.orbital_indices[pp];
        let home = aos.kernel_to_home[ao]; // == ao for molecular
        debug_assert!(home < num_home && a2s[pp] < n_shell);
        let z = aos.exponents[pp];
        let c = aos.coefficients[pp];
        let nrm = aos.normalizations[pp];
        let l = aos.angular_momentums[ao] as f64;
        let cx = aos.centers[[ao, 0]];
        let cy = aos.centers[[ao, 1]];
        let cz = aos.centers[[ao, 2]];
        let a = (2.0 * l + 3.0) / (4.0 * z);
        let mut acc_z = 0.0;
        let mut acc_c = 0.0;
        for k in 0..n {
            let dx = r[[k, 0]] - cx;
            let dy = r[[k, 1]] - cy;
            let dz = r[[k, 2]] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let rad = nrm * (-z * r2).exp();
            let (sc_val, sc_lap, sc_dgrad, sc_grad) = prim_vgl(aos, ao, z, rad, dx, dy, dz, r2);
            let sz_val = c * (a - r2) * sc_val;
            let sz_lap = c * ((a - r2) * sc_lap - 4.0 * sc_dgrad - 6.0 * sc_val);
            let bo = bar_o[[home, k]];
            let bl = bar_l[[home, k]];
            acc_z += bo * sz_val + bl * sz_lap;
            acc_c += bo * sc_val + bl * sc_lap;
            let u = [dx, dy, dz];
            for d in 0..3 {
                let bg = bar_g[d][[home, k]];
                if bg != 0.0 {
                    let sz_grad = c * ((a - r2) * sc_grad[d] - 2.0 * u[d] * sc_val);
                    acc_z += bg * sz_grad;
                    acc_c += bg * sc_grad[d];
                }
            }
        }
        gz[a2s[pp]] += acc_z;
        gc[a2s[pp]] += acc_c * aos.coefficient_scale(ao);
    }
    (gz, gc)
}

/// `dT_det/dradial.{exp,coeff}` (jnone kinetic) for up/dn cart AO leaves.
/// Returns `((gZ_up,gc_up),(gZ_dn,gc_dn))`.
#[allow(clippy::too_many_arguments)]
pub fn grad_radial_kinetic(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    a2s_up: &[usize],
    n_shell_up: usize,
    a2s_dn: &[usize],
    n_shell_dn: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: Option<&ArrayView2<f64>>,
) -> RadialGrads {
    let ru = wrap_into_cell(r_up, cell);
    let rd = wrap_into_cell(r_dn, cell);
    let (r_up, r_dn) = (&ru.view(), &rd.view());
    // Jastrow drift gJ = grad ln J (0 for jnone) -> the geminal reverse carries the
    // `-Sum gJ.gD` cross term into `bar_gou`/`bar_godn`. The J drift is evaluated at
    // the SAME wrapped positions as the det reverse (periodic-safe).
    let (gj_up, gj_dn) = jastrow_grads(jastrow, j3_orb, j3_jm, r_up, r_dn);
    let b = geminal_reverse(gem, r_up, r_dn, &gj_up, &gj_dn);
    // Composite-aware: slice the GTO child's home rows from each pulled cotangent
    // (see `pull_gto_rows` / `grad_radial`); no-op for a pure GTO leaf.
    let pg = |orb: &OrbData, bar: &Array2<f64>| pull_gto_rows(orb, &pull_to_ao(orb, bar));
    let bar_o_up = pg(&gem.orb_up, &b.bar_orb_up);
    let bar_l_up = pg(&gem.orb_up, &b.bar_lap_up);
    let bar_o_dn = pg(&gem.orb_dn, &b.bar_orb_dn);
    let bar_l_dn = pg(&gem.orb_dn, &b.bar_lap_dn);
    let bar_g_up = [
        pg(&gem.orb_up, &b.bar_gou[0]),
        pg(&gem.orb_up, &b.bar_gou[1]),
        pg(&gem.orb_up, &b.bar_gou[2]),
    ];
    let bar_g_dn = [
        pg(&gem.orb_dn, &b.bar_godn[0]),
        pg(&gem.orb_dn, &b.bar_godn[1]),
        pg(&gem.orb_dn, &b.bar_godn[2]),
    ];
    let up = grad_radial_kinetic_side(
        &gem.orb_up.aos,
        a2s_up,
        n_shell_up,
        &bar_o_up,
        &bar_l_up,
        &bar_g_up,
        r_up,
    );
    let dn = grad_radial_kinetic_side(
        &gem.orb_dn.aos,
        a2s_dn,
        n_shell_dn,
        &bar_o_dn,
        &bar_l_dn,
        &bar_g_dn,
        r_dn,
    );
    (up, dn)
}

// ============================================================================
// Complex-twist (Bloch, k != Gamma) radial c-axis: the complex twins of
// `grad_radial` / `grad_radial_kinetic`.
//
// The Bloch AO is `AO[home,e] = restore[e] * Sum_{kappa->home} e^{ik.T_kappa}
// kernel_real[kappa, e](r_wrapped)` (jqmc `_ComplexBlochPGTOMixin`: wrap into the
// home cell, truncated image sum, re-apply the wrap-restore phase). The radial DOF
// is Z/c only, so BOTH phases are radial-INDEPENDENT and factor straight through:
//   dAO[home,e]/d{Z,c} = restore[e] * Sum_{kappa->home} e^{ik.T_kappa} * sens[kappa,e]
// with `sens` the SAME real per-primitive sensitivity the real path uses, evaluated
// at the WRAPPED electrons. Equivalently (the adjoint form jqmc's
// `basis._value_general_leaf` / `_kinetic_general_leaf` use), the home-AO cotangent
// is un-folded to kernel rank by `phase[kappa] * bar[home,e] * restore[e]`.
//
// HOLOMORPHIC seam (`Re(A)*Re(B) != Re(A*B)`): every link of the chain -- the
// geminal forward/reverse cotangents, the MO pull (`C^T`, NEVER conjugated), the
// image phase and the restore phase -- stays complex, and `Re` is taken ONLY on the
// finished shell-prim accumulator. The ln_Psi / e_L consumers want that `Re`; the
// ECP non-local `dV_nl/dc` (param_grad_twist::grad_ecp_radial_twist) needs the FULL
// holomorphic O_k because it multiplies by a complex per-grid weight first, so the
// value kernel is exposed in both forms.
// ============================================================================

/// GTO-child cotangent of a complex leaf at HOME-AO rank. Pulls through the MO
/// holomorphically (`C^T bar`, un-conjugated -- the MO is the complex variable the
/// determinant consumes) and slices the GTO block out of the composite AO axis in
/// `gto_first` order. A pure complex-pGTO leaf has no extra block -> whole array.
fn gto_cot_complex(orb: &ComplexOrb, bar: &Array2<Complex64>) -> Array2<Complex64> {
    let cm = match &orb.mo {
        Some(c) => c.t().dot(bar),
        None => bar.clone(),
    };
    let num_extra = orb
        .pw
        .as_ref()
        .map(|p| p.num_ao)
        .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
        .unwrap_or(0);
    let num_gto = cm.nrows() - num_extra;
    let gto0 = if orb.gto_first { 0 } else { num_extra };
    debug_assert_eq!(num_gto, orb.aos.num_ao);
    cm.slice(numpy::ndarray::s![gto0..gto0 + num_gto, ..])
        .to_owned()
}

/// `(r_wrapped, restore)` for the complex GTO block: exactly what `ComplexOrb`'s own
/// `gto_vgl` applies to the forward AO, so the sensitivity sees the SAME electrons and
/// the SAME per-electron phase. A wrap-less leaf (molecular / Gamma) is the identity.
fn wrap_restore_complex(orb: &ComplexOrb, r: &ArrayView2<f64>) -> (Array2<f64>, Vec<Complex64>) {
    match &orb.wrap {
        None => (r.to_owned(), vec![Complex64::new(1.0, 0.0); r.nrows()]),
        Some(w) => w.wrap(r),
    }
}

/// Complex twin of [`grad_radial_side`]: the det VALUE cotangent contracted with the
/// Bloch AO's radial sensitivity, HOLOMORPHIC (no `Re`). `bar_ao` is the GTO-child
/// home-AO cotangent, `r` the RAW electron block (wrapping happens here).
fn grad_radial_side_complex(
    orb: &ComplexOrb,
    a2s: &[usize],
    n_shell: usize,
    bar_ao: &Array2<Complex64>,
    r: &ArrayView2<f64>,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let aos = &orb.aos;
    let (rw, restore) = wrap_restore_complex(orb, r);
    let n = rw.nrows();
    let num_home = bar_ao.nrows();
    let zero = Complex64::new(0.0, 0.0);
    let mut gz = vec![zero; n_shell];
    let mut gc = vec![zero; n_shell];
    for pp in 0..aos.orbital_indices.len() {
        let ao = aos.orbital_indices[pp]; // KERNEL-rank AO (image-tiled centre)
        let home = aos.kernel_to_home[ao];
        debug_assert!(home < num_home && a2s[pp] < n_shell);
        let ph = orb.image_phase[ao];
        let z = aos.exponents[pp];
        let c = aos.coefficients[pp];
        let nrm = aos.normalizations[pp];
        let l = aos.angular_momentums[ao] as f64;
        let cx = aos.centers[[ao, 0]];
        let cy = aos.centers[[ao, 1]];
        let cz = aos.centers[[ao, 2]];
        let mut acc_z = zero;
        let mut acc_c = zero;
        for k in 0..n {
            let dx = rw[[k, 0]] - cx;
            let dy = rw[[k, 1]] - cy;
            let dz = rw[[k, 2]] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let ang = angular_value(aos, ao, dx, dy, dz, r2);
            let sens_c = nrm * ang * (-z * r2).exp();
            let sens_z = ((2.0 * l + 3.0) / (4.0 * z) - r2) * (c * sens_c);
            // Un-fold the home cotangent to this image: phase * bar * restore (all complex).
            let b = ph * bar_ao[[home, k]] * restore[k];
            acc_z += b * sens_z;
            acc_c += b * sens_c;
        }
        gz[a2s[pp]] += acc_z;
        gc[a2s[pp]] += acc_c * aos.coefficient_scale(ao);
    }
    (gz, gc)
}

/// `dln det/dradial.{exp,coeff}` for a COMPLEX-twist geminal, HOLOMORPHIC (complex).
/// `((gZ_up,gc_up),(gZ_dn,gc_dn))`. Mirror of [`grad_radial`] with the complex
/// forward: `bar_orb_up = W G^-1`, `bar_orb_dn = (G^-1 Q)[:N_dn]^T`. No electron wrap
/// here -- the complex leaf wraps internally (and carries the restore phase), so the
/// forward and the sensitivity share the same convention.
#[allow(clippy::too_many_arguments)]
pub fn grad_radial_complex(
    gem: &ComplexGeminal,
    lam: Option<&ArrayView2<Complex64>>,
    a2s_up: &[usize],
    n_shell_up: usize,
    a2s_dn: &[usize],
    n_shell_dn: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> RadialGrads<Complex64> {
    let f = match lam {
        Some(lam) => complex_gem_forward_clam(gem, lam, r_up, r_dn, None),
        None => crate::complex_det::complex_gem_forward(gem, r_up, r_dn, None),
    };
    let bar_orb_up = f.wg.clone(); // (M_up, N_up) = W G^-1
    let bar_orb_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned(); // (M_dn, N_dn)
    let bar_ao_up = gto_cot_complex(&gem.orb_up, &bar_orb_up);
    let bar_ao_dn = gto_cot_complex(&gem.orb_dn, &bar_orb_dn);
    (
        grad_radial_side_complex(&gem.orb_up, a2s_up, n_shell_up, &bar_ao_up, r_up),
        grad_radial_side_complex(&gem.orb_dn, a2s_dn, n_shell_dn, &bar_ao_dn, r_dn),
    )
}

/// `O_k = dln|det|/dradial = Re(dln det/dradial)` for a complex-twist geminal (the
/// radial params are REAL DOF, so the energy/log-amplitude gradient is the real part
/// -- taken LAST, on the finished holomorphic accumulator).
#[allow(clippy::too_many_arguments)]
pub fn grad_radial_complex_re(
    gem: &ComplexGeminal,
    lam: Option<&ArrayView2<Complex64>>,
    a2s_up: &[usize],
    n_shell_up: usize,
    a2s_dn: &[usize],
    n_shell_dn: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> RadialGrads {
    let (up, dn) =
        grad_radial_complex(gem, lam, a2s_up, n_shell_up, a2s_dn, n_shell_dn, r_up, r_dn);
    let re = |v: Vec<Complex64>| v.into_iter().map(|x| x.re).collect::<Vec<f64>>();
    ((re(up.0), re(up.1)), (re(dn.0), re(dn.1)))
}

/// Complex twin of [`grad_radial_kinetic_side`]: value + Laplacian + orbital-GRADIENT
/// cotangents on the Bloch AO's radial sensitivities, HOLOMORPHIC (no `Re`).
#[allow(clippy::too_many_arguments)]
fn grad_radial_kinetic_side_complex(
    orb: &ComplexOrb,
    a2s: &[usize],
    n_shell: usize,
    bar_o: &Array2<Complex64>,
    bar_l: &Array2<Complex64>,
    bar_g: &[Array2<Complex64>; 3],
    r: &ArrayView2<f64>,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let aos = &orb.aos;
    let (rw, restore) = wrap_restore_complex(orb, r);
    let n = rw.nrows();
    let num_home = bar_o.nrows();
    let zero = Complex64::new(0.0, 0.0);
    let mut gz = vec![zero; n_shell];
    let mut gc = vec![zero; n_shell];
    for pp in 0..aos.orbital_indices.len() {
        let ao = aos.orbital_indices[pp];
        let home = aos.kernel_to_home[ao];
        debug_assert!(home < num_home && a2s[pp] < n_shell);
        let ph = orb.image_phase[ao];
        let z = aos.exponents[pp];
        let c = aos.coefficients[pp];
        let nrm = aos.normalizations[pp];
        let l = aos.angular_momentums[ao] as f64;
        let cx = aos.centers[[ao, 0]];
        let cy = aos.centers[[ao, 1]];
        let cz = aos.centers[[ao, 2]];
        let a = (2.0 * l + 3.0) / (4.0 * z);
        let mut acc_z = zero;
        let mut acc_c = zero;
        for k in 0..n {
            let dx = rw[[k, 0]] - cx;
            let dy = rw[[k, 1]] - cy;
            let dz = rw[[k, 2]] - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let rad = nrm * (-z * r2).exp();
            let (sc_val, sc_lap, sc_dgrad, sc_grad) = prim_vgl(aos, ao, z, rad, dx, dy, dz, r2);
            let sz_val = c * (a - r2) * sc_val;
            let sz_lap = c * ((a - r2) * sc_lap - 4.0 * sc_dgrad - 6.0 * sc_val);
            let w = ph * restore[k]; // image + wrap-restore phase (radial-independent)
            let bo = w * bar_o[[home, k]];
            let bl = w * bar_l[[home, k]];
            acc_z += bo * sz_val + bl * sz_lap;
            acc_c += bo * sc_val + bl * sc_lap;
            let u = [dx, dy, dz];
            for d in 0..3 {
                let bg = w * bar_g[d][[home, k]];
                if bg != zero {
                    let sz_grad = c * ((a - r2) * sc_grad[d] - 2.0 * u[d] * sc_val);
                    acc_z += bg * sz_grad;
                    acc_c += bg * sc_grad[d];
                }
            }
        }
        gz[a2s[pp]] += acc_z;
        gc[a2s[pp]] += acc_c * aos.coefficient_scale(ao);
    }
    (gz, gc)
}

/// REAL Jastrow drift `gJ = grad ln J` for a twist: J1/J2/DTN through the shared real
/// kernel plus the complex-Bloch J3 drift (the three-body Jastrow consumes `Re(orb)`,
/// so `gJ` is real). Mirrors `param_grad_twist::twist_drift`'s Jastrow half; the
/// determinant drift is NOT added here -- `complex_geminal_reverse` carries the full
/// complex `gD` itself through the `-Sum gJ.gD` cross term.
fn jastrow_drift_complex(
    jastrow: Option<&JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (mut gu, mut gd) = match jastrow {
        None => (
            Array2::zeros((r_up.nrows(), 3)),
            Array2::zeros((r_dn.nrows(), 3)),
        ),
        Some(j) => {
            let (gu, gd, _, _) =
                crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(j, r_up, r_dn);
            (gu, gd)
        }
    };
    if let (Some(orb), Some(jm)) = (j3c, j3m) {
        let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(orb, jm, r_up, r_dn);
        gu = gu + g3u;
        gd = gd + g3d;
    }
    (gu, gd)
}

/// Exact per-walker `de_L/dradial = dT_det/dradial` for a COMPLEX-twist geminal, with Jastrow
/// (`jastrow=None` -> the jnone det kinetic). Mirror of [`grad_radial_kinetic`]:
/// the complex kinetic VJP cotangents (value / Laplacian / orbital-gradient, the last
/// carrying the `-Sum gJ.gD` cross term) contracted with the Bloch radial
/// sensitivities. The full complex shell accumulator is retained for the optimizer;
/// the legacy real-parameter diagnostic projection is performed by the Python driver.
#[allow(clippy::too_many_arguments)]
pub fn grad_radial_kinetic_complex(
    gem: &ComplexGeminal,
    lam: Option<&ArrayView2<Complex64>>,
    jastrow: Option<&JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    a2s_up: &[usize],
    n_shell_up: usize,
    a2s_dn: &[usize],
    n_shell_dn: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> RadialGrads<Complex64> {
    let (gj_up, gj_dn) = jastrow_drift_complex(jastrow, j3c, j3m, r_up, r_dn);
    let b = match lam {
        Some(lam) => complex_geminal_reverse_clam(gem, lam, r_up, r_dn, &gj_up, &gj_dn, None),
        None => crate::complex_det::complex_geminal_reverse(gem, r_up, r_dn, &gj_up, &gj_dn, None),
    };
    let cot = |orb: &ComplexOrb, m: &Array2<Complex64>| gto_cot_complex(orb, m);
    let bar_g_up = [
        cot(&gem.orb_up, &b.bar_gou[0]),
        cot(&gem.orb_up, &b.bar_gou[1]),
        cot(&gem.orb_up, &b.bar_gou[2]),
    ];
    let bar_g_dn = [
        cot(&gem.orb_dn, &b.bar_godn[0]),
        cot(&gem.orb_dn, &b.bar_godn[1]),
        cot(&gem.orb_dn, &b.bar_godn[2]),
    ];
    let up = grad_radial_kinetic_side_complex(
        &gem.orb_up,
        a2s_up,
        n_shell_up,
        &cot(&gem.orb_up, &b.bar_orb_up),
        &cot(&gem.orb_up, &b.bar_lap_up),
        &bar_g_up,
        r_up,
    );
    let dn = grad_radial_kinetic_side_complex(
        &gem.orb_dn,
        a2s_dn,
        n_shell_dn,
        &cot(&gem.orb_dn, &b.bar_orb_dn),
        &cot(&gem.orb_dn, &b.bar_lap_dn),
        &bar_g_dn,
        r_dn,
    );
    (up, dn)
}

// ===========================================================================
// J3 three-body-Jastrow ORBITAL radial basis-opt (`opt_J3_basis_exp` /
// `opt_J3_basis_coeff`): d/d(J3 orbital radial.{exponents,coefficients}).
// 1:1 with `jqmc/derivatives/basis.py` `grad_J3_wrt_basis` (O_k) and
// `grad_T_wrt_J3_basis` (dE_L/dc kinetic).
//
// The J3 orbital is SHARED across both spins (ONE leaf evaluated at r_up and at
// r_dn), so unlike the geminal `lambda_basis_*` family -- which concatenates a
// separate up and dn shell block -- each derivative here is the SUM of the up- and
// dn-electron contributions of that single leaf, and the published block has the
// leaf's own shell rank. The radial contraction itself reuses the SAME verified
// `grad_radial_side` / `grad_radial_kinetic_side` primitive-sensitivity kernels the
// geminal path uses; only the cotangent that seeds them is J3-specific.
// ===========================================================================

/// `X @ K` with `K = tril(ones, -1)` (strict LOWER): `out[a,j] = Sum_{i>j} X[a,i]`
/// -- a reverse cumulative sum along the electron axis. Python spells this
/// `... @ Ku`; the explicit prefix/suffix form is O(orb*N) instead of O(orb*N^2).
fn suffix_cols(x: &Array2<f64>) -> Array2<f64> {
    let (m, n) = x.dim();
    let mut out = Array2::<f64>::zeros((m, n));
    for a in 0..m {
        let mut acc = 0.0_f64;
        for j in (0..n).rev() {
            out[[a, j]] = acc;
            acc += x[[a, j]];
        }
    }
    out
}

/// `X @ K^T` with `K = tril(ones, -1)`: `out[a,j] = Sum_{i<j} X[a,i]` (forward
/// cumulative sum). Python spells this `... @ Ku.T`.
fn prefix_cols(x: &Array2<f64>) -> Array2<f64> {
    let (m, n) = x.dim();
    let mut out = Array2::<f64>::zeros((m, n));
    for a in 0..m {
        let mut acc = 0.0_f64;
        for j in 0..n {
            out[[a, j]] = acc;
            acc += x[[a, j]];
        }
    }
    out
}

/// `(M, j1col)` split of the J3 `j_matrix` `(orb, orb+1)`.
fn j3_split(jm: &ArrayView2<f64>, orb: usize) -> (Array2<f64>, Array1<f64>) {
    let m = jm.slice(numpy::ndarray::s![.., 0..orb]).to_owned();
    let j1 = jm.slice(numpy::ndarray::s![.., orb]).to_owned();
    (m, j1)
}

/// The J3 value-field cotangent `coefA = dJ3/dA` on the orbital value field
/// `A = Re(orb(r))`, given the already-evaluated per-spin orbital matrices. 1:1
/// `basis._j3_value_cotangents`:
/// `coefA_up = j1 + M A_up Ku + M^T A_up Ku^T + M s_dn`,
/// `coefA_dn = j1 + M A_dn Kd + M^T A_dn Kd^T + M^T s_up`
/// (the last term broadcast over the electron axis; note the up side contracts
/// `s_dn` with `M` and the dn side contracts `s_up` with `M^T` -- the outer block
/// `s_up^T M s_dn` is not symmetric per spin).
///
/// Also the coefficient the kinetic path's pair/outer VALUE derivative multiplies,
/// which is why `grad_j3_radial_kinetic` reuses it verbatim.
fn j3_value_cotangents(
    jm: &ArrayView2<f64>,
    a_up: &Array2<f64>,
    a_dn: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let orb = a_up.nrows();
    let (m, j1) = j3_split(jm, orb);
    let mt = m.t().to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let mut bar_up = suffix_cols(&m.dot(a_up)) + prefix_cols(&mt.dot(a_up));
    let mut bar_dn = suffix_cols(&m.dot(a_dn)) + prefix_cols(&mt.dot(a_dn));
    let cu = m.dot(&s_dn); // (orb,)
    let cd = mt.dot(&s_up);
    let n_up = bar_up.ncols();
    let n_dn = bar_dn.ncols();
    for p in 0..orb {
        let bu = j1[p] + cu[p];
        let bd = j1[p] + cd[p];
        for k in 0..n_up {
            bar_up[[p, k]] += bu;
        }
        for k in 0..n_dn {
            bar_dn[[p, k]] += bd;
        }
    }
    (bar_up, bar_dn)
}

/// Orbital-rank cotangent -> the GTO leaf's home-AO rows (MO layer folded in,
/// composite non-GTO children sliced off) -- the J3 analogue of the geminal path's
/// `pull_gto_rows(pull_to_ao(...))`.
#[inline]
fn pull_j3(orb: &OrbData, bar: &Array2<f64>) -> Array2<f64> {
    pull_gto_rows(orb, &pull_to_ao(orb, bar))
}

#[inline]
fn add_into(dst: &mut (Vec<f64>, Vec<f64>), src: &(Vec<f64>, Vec<f64>)) {
    for (d, v) in dst.0.iter_mut().zip(src.0.iter()) {
        *d += v;
    }
    for (d, v) in dst.1.iter_mut().zip(src.1.iter()) {
        *d += v;
    }
}

/// `O_k` block for the J3 orbital radial params:
/// `d ln|Psi| / d(J3 radial.{exponents,coefficients})`. The determinant is
/// J3-independent, so this is exactly `dJ3/d(radial)`. Returns
/// `(d/dexponents, d/dcoefficients)`, each of the J3 leaf's shell rank.
/// 1:1 `basis.grad_J3_wrt_basis`.
///
/// Home-cell wrap: the J3 orbital is an AO stack, whose truncated periodic image
/// sum is enumerated relative to the home cell (same contract as
/// `param_grad::grad_j3_jmatrix`). No-op molecular / pure-PW.
pub fn grad_j3_radial(
    j3_orb: &OrbData,
    j3_jm: &ArrayView2<f64>,
    a2s: &[usize],
    n_shell: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let ruw = j3_orb
        .wrap_electrons(r_up)
        .unwrap_or_else(|| r_up.to_owned());
    let rdw = j3_orb
        .wrap_electrons(r_dn)
        .unwrap_or_else(|| r_dn.to_owned());
    let (r_up, r_dn) = (&ruw.view(), &rdw.view());
    let a_up = j3_orb.compute(r_up); // (orb_num, N_up)
    let a_dn = j3_orb.compute(r_dn);
    let (bar_up, bar_dn) = j3_value_cotangents(j3_jm, &a_up, &a_dn);
    let bar_ao_up = pull_j3(j3_orb, &bar_up);
    let bar_ao_dn = pull_j3(j3_orb, &bar_dn);
    let mut out = grad_radial_side(&j3_orb.aos, a2s, n_shell, &bar_ao_up, r_up);
    let dn = grad_radial_side(&j3_orb.aos, a2s, n_shell, &bar_ao_dn, r_dn);
    add_into(&mut out, &dn);
    out
}

/// `dE_L/dc` (kinetic `dT/dc`) block for the J3 orbital radial params.
/// 1:1 `basis.grad_T_wrt_J3_basis`: keep the reverse cotangents on the J3 orbital
/// FIELDS (value / gradient / Laplacian) instead of collapsing them onto `r`, then
/// contract each with the radial value/grad/lap primitive sensitivities.
///
/// With `cot = grad ln|Psi| = gJ + gD` (the kinetic drift), `coefA` the J3 value
/// cotangent and `bcA = -0.5 lapA - cot.gradA`:
/// * `bar_lap  = -0.5 coefA`
/// * `bar_gou  = -cot (x) coefA`
/// * `bar_orb  = M^T bcA Ku^T + M bcA Ku + M bcA_other.sum(elec)`
///   summed over up + dn (the leaf is shared). Wrap policy is the one
///   `param_grad_el::grad_t_j3` pins: the Jastrow METRIC (J1/J2/DTN) on the raw
///   coordinate, the J3 orbital + determinant AO stack on the wrapped one.
#[allow(clippy::too_many_arguments)]
pub fn grad_j3_radial_kinetic(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3_orb: &OrbData,
    j3_jm: &ArrayView2<f64>,
    a2s: &[usize],
    n_shell: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let (rw_up, rw_dn) = wrap_both(gem, r_up, r_dn);
    let (rwu, rwd) = (&rw_up.view(), &rw_dn.view());
    let (drift_up, drift_dn) = kinetic_drift(
        gem,
        Some(jastrow),
        Some(j3_orb),
        Some(j3_jm),
        r_up,
        r_dn,
        rwu,
        rwd,
    );
    let (r_up, r_dn) = (rwu, rwd);
    let (a_up, gxu, gyu, gzu, la_up) = orb_value_grad_lap(j3_orb, r_up);
    let (a_dn, gxd, gyd, gzd, la_dn) = orb_value_grad_lap(j3_orb, r_dn);
    let ga_up = [gxu, gyu, gzu];
    let ga_dn = [gxd, gyd, gzd];
    let orb = a_up.nrows();
    let (m, _j1) = j3_split(j3_jm, orb);
    let mt = m.t().to_owned();

    let (coef_up, coef_dn) = j3_value_cotangents(j3_jm, &a_up, &a_dn);

    // bcA[a,k] = -0.5 lapA[a,k] - Sum_d cot[k,d] gradA[d][a,k]
    let bca = |la: &Array2<f64>, ga: &[Array2<f64>; 3], cot: &Array2<f64>| -> Array2<f64> {
        let n = la.ncols();
        let mut out = Array2::<f64>::zeros((orb, n));
        for a in 0..orb {
            for k in 0..n {
                let mut s = -0.5 * la[[a, k]];
                for d in 0..3 {
                    s -= cot[[k, d]] * ga[d][[a, k]];
                }
                out[[a, k]] = s;
            }
        }
        out
    };
    let bca_up = bca(&la_up, &ga_up, &drift_up);
    let bca_dn = bca(&la_dn, &ga_dn, &drift_dn);

    // bar_gou[d][a,k] = -cot[k,d] * coefA[a,k]
    let bar_g = |cot: &Array2<f64>, coef: &Array2<f64>| -> [Array2<f64>; 3] {
        let n = coef.ncols();
        let mut out = [
            Array2::<f64>::zeros((orb, n)),
            Array2::<f64>::zeros((orb, n)),
            Array2::<f64>::zeros((orb, n)),
        ];
        for (d, o) in out.iter_mut().enumerate() {
            for a in 0..orb {
                for k in 0..n {
                    o[[a, k]] = -cot[[k, d]] * coef[[a, k]];
                }
            }
        }
        out
    };
    let bar_g_up = bar_g(&drift_up, &coef_up);
    let bar_g_dn = bar_g(&drift_dn, &coef_dn);

    let bar_lap_up = -0.5 * &coef_up;
    let bar_lap_dn = -0.5 * &coef_dn;

    // bar_orb = M^T bcA Ku^T + M bcA Ku + (M_side bcA_other.sum(elec)) broadcast
    let bar_orb =
        |bc_self: &Array2<f64>, bc_other: &Array2<f64>, cross: &Array2<f64>| -> Array2<f64> {
            let mut o = prefix_cols(&mt.dot(bc_self)) + suffix_cols(&m.dot(bc_self));
            let v = cross.dot(&bc_other.sum_axis(Axis(1))); // (orb,)
            let n = o.ncols();
            for a in 0..orb {
                for k in 0..n {
                    o[[a, k]] += v[a];
                }
            }
            o
        };
    let bar_o_up = bar_orb(&bca_up, &bca_dn, &m);
    let bar_o_dn = bar_orb(&bca_dn, &bca_up, &mt);

    let side = |bar_o: &Array2<f64>,
                bar_l: &Array2<f64>,
                bar_g: &[Array2<f64>; 3],
                r: &ArrayView2<f64>| {
        let bo = pull_j3(j3_orb, bar_o);
        let bl = pull_j3(j3_orb, bar_l);
        let bg = [
            pull_j3(j3_orb, &bar_g[0]),
            pull_j3(j3_orb, &bar_g[1]),
            pull_j3(j3_orb, &bar_g[2]),
        ];
        grad_radial_kinetic_side(&j3_orb.aos, a2s, n_shell, &bo, &bl, &bg, r)
    };
    let mut out = side(&bar_o_up, &bar_lap_up, &bar_g_up, r_up);
    let dn = side(&bar_o_dn, &bar_lap_dn, &bar_g_dn, r_dn);
    add_into(&mut out, &dn);
    out
}

/// Complex-Bloch twin of [`grad_j3_radial`].  J3 is a real function of
/// `A = Re(orbital)`, while its radial parameters are real.  Keep the MO/image
/// chain complex through the GTO contraction and project to `Re` only on the
/// completed shell accumulator.
pub fn grad_j3_radial_complex(
    j3_orb: &ComplexOrb,
    j3_jm: &ArrayView2<f64>,
    a2s: &[usize],
    n_shell: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let re = |m: &Array2<Complex64>| m.mapv(|z| z.re);
    let a_up = re(&j3_orb.value(r_up));
    let a_dn = re(&j3_orb.value(r_dn));
    let (bar_up, bar_dn) = j3_value_cotangents(j3_jm, &a_up, &a_dn);
    let lift = |m: &Array2<f64>| m.mapv(|x| Complex64::new(x, 0.0));
    let bu = gto_cot_complex(j3_orb, &lift(&bar_up));
    let bd = gto_cot_complex(j3_orb, &lift(&bar_dn));
    let mut up = grad_radial_side_complex(j3_orb, a2s, n_shell, &bu, r_up);
    let dn = grad_radial_side_complex(j3_orb, a2s, n_shell, &bd, r_dn);
    for (d, v) in up.0.iter_mut().zip(dn.0.iter()) {
        *d += *v;
    }
    for (d, v) in up.1.iter_mut().zip(dn.1.iter()) {
        *d += *v;
    }
    (
        up.0.into_iter().map(|z| z.re).collect(),
        up.1.into_iter().map(|z| z.re).collect(),
    )
}

/// Complex-Bloch twin of [`grad_j3_radial_kinetic`].  The J3 value/gradient/
/// Laplacian fields are `Re(complex orbital)` and the kinetic cotangent is the
/// full real drift `gJ + Re(gD)`.  The radial AO sensitivity itself remains
/// holomorphic until the final real projection.
#[allow(clippy::too_many_arguments)]
pub fn grad_j3_radial_kinetic_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3_orb: &ComplexOrb,
    j3_jm: &ArrayView2<f64>,
    a2s: &[usize],
    n_shell: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let re = |m: &Array2<Complex64>| m.mapv(|z| z.re);
    let vgl_re = |r: &ArrayView2<f64>| {
        let (v, g, l) = j3_orb.value_grad_lap(r);
        (re(&v), [re(&g[0]), re(&g[1]), re(&g[2])], re(&l))
    };
    let (a_up, ga_up, la_up) = vgl_re(r_up);
    let (a_dn, ga_dn, la_dn) = vgl_re(r_dn);
    let orb = a_up.nrows();
    let (m, _j1) = j3_split(j3_jm, orb);
    let mt = m.t().to_owned();
    let (coef_up, coef_dn) = j3_value_cotangents(j3_jm, &a_up, &a_dn);
    let (drift_up, drift_dn) =
        crate::param_grad_twist::twist_drift(gem, jastrow, Some(j3_orb), Some(j3_jm), r_up, r_dn);

    let bca = |la: &Array2<f64>, ga: &[Array2<f64>; 3], cot: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(la.dim());
        for a in 0..la.nrows() {
            for k in 0..la.ncols() {
                out[[a, k]] =
                    -0.5 * la[[a, k]] - (0..3).map(|d| cot[[k, d]] * ga[d][[a, k]]).sum::<f64>();
            }
        }
        out
    };
    let bca_up = bca(&la_up, &ga_up, &drift_up);
    let bca_dn = bca(&la_dn, &ga_dn, &drift_dn);
    let bar_g = |cot: &Array2<f64>, coef: &Array2<f64>| -> [Array2<f64>; 3] {
        std::array::from_fn(|d| {
            let mut out = Array2::<f64>::zeros(coef.dim());
            for a in 0..coef.nrows() {
                for k in 0..coef.ncols() {
                    out[[a, k]] = -cot[[k, d]] * coef[[a, k]];
                }
            }
            out
        })
    };
    let bg_up = bar_g(&drift_up, &coef_up);
    let bg_dn = bar_g(&drift_dn, &coef_dn);
    let bl_up = -0.5 * &coef_up;
    let bl_dn = -0.5 * &coef_dn;
    let bar_orb = |bc_self: &Array2<f64>, bc_other: &Array2<f64>, cross: &Array2<f64>| {
        let mut out = prefix_cols(&mt.dot(bc_self)) + suffix_cols(&m.dot(bc_self));
        let v = cross.dot(&bc_other.sum_axis(Axis(1)));
        for a in 0..out.nrows() {
            for k in 0..out.ncols() {
                out[[a, k]] += v[a];
            }
        }
        out
    };
    let bo_up = bar_orb(&bca_up, &bca_dn, &m);
    let bo_dn = bar_orb(&bca_dn, &bca_up, &mt);

    let lift_pull = |x: &Array2<f64>| gto_cot_complex(j3_orb, &x.mapv(|v| Complex64::new(v, 0.0)));
    let side = |bo: &Array2<f64>, bl: &Array2<f64>, bg: &[Array2<f64>; 3], r: &ArrayView2<f64>| {
        let bgc = [lift_pull(&bg[0]), lift_pull(&bg[1]), lift_pull(&bg[2])];
        grad_radial_kinetic_side_complex(
            j3_orb,
            a2s,
            n_shell,
            &lift_pull(bo),
            &lift_pull(bl),
            &bgc,
            r,
        )
    };
    let mut up = side(&bo_up, &bl_up, &bg_up, r_up);
    let dn = side(&bo_dn, &bl_dn, &bg_dn, r_dn);
    for (d, v) in up.0.iter_mut().zip(dn.0.iter()) {
        *d += *v;
    }
    for (d, v) in up.1.iter_mut().zip(dn.1.iter()) {
        *d += *v;
    }
    (
        up.0.into_iter().map(|z| z.re).collect(),
        up.1.into_iter().map(|z| z.re).collect(),
    )
}
