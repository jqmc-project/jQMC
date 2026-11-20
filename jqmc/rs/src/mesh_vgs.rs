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

//! Fused value + gradient (+ strain) contraction of a periodic Cartesian pGTO leaf over a
//! small point set, in the loop order of the e_L's ECP mesh value kernel
//! (`atomic_orbital::dot_aos_multi_cutoff_cart_soa`): shell-major over the image-tiled
//! shell table, one bounding-sphere test per shell, then branch-free loops over the points.
//!
//! For weights `w` shared by all points (home rank, or kernel rank for Bloch-phased
//! weights) and wrapped points `r_v` it returns, per point,
//! * `val[v]  = Sum_kap w[home(kap)] phi_kap(r_v)`,
//! * `grad[v] = Sum_kap w[home(kap)] grad phi_kap(r_v)`,
//! * optionally the atom-resolved gradient `atom[v][a]` (the same sum restricted to the
//!   shells of home atom `a`) and the strain virial
//!   `strain[v] = Sum_kap w[home(kap)] grad phi_kap(r_v) (x) (r_v - C_kap + R_at(kap) - R_home(kap))`
//!   (the per-point form of `stress_pbc::field_stress`),
//! * optionally the folded value table `vtab[home][v] = Sum_{kap -> home} phi_kap(r_v)`.
//!
//! The AO model is the one of the derivative kernel `ao_cart_deriv::ao_deriv_kernel_multi_screened`
//! under the same [`AoScreen`]: `ShellValue` drops a shell image at a point where
//! `pz_val r^2 > LEPSBAS`, `MeshValue` additionally drops primitive `q` where
//! `prim_val[q] r^2 > LEPSBAS`, `None` keeps everything. The gradient is the analytic
//! derivative of that screened value. Only the summation order differs from the
//! table kernels (the value and gradient of every AO are formed from the same primitive
//! sums `Sum_q c_q exp(-z_q r^2)` and `Sum_q c_q z_q exp(-z_q r^2)`).

use crate::ao_cart_deriv::AoScreen;
use crate::atomic_orbital::{AosGtoData, LEPSBAS};

const NP: usize = 32;
const NA: usize = 16;
const NPOW: usize = 9;

/// Per-shell geometry of one leaf for the optional outputs: the home atom of each shell
/// (atom-resolved gradient) and the strain offset `R_at - R_home` of its AOs.
pub(crate) struct ShellGeom {
    pub atom: Vec<usize>,
    pub off: Vec<[f64; 3]>,
}

impl ShellGeom {
    /// `home_atom[mu]` = atom of folded AO `mu`; `kernel_atom = (atom of kernel AO kap,
    /// atom positions)` for the strain offsets (`None` = zero offsets). `None` when the AOs of a shell do not
    /// share one home atom and one offset, or the leaf is not a shell-table Cartesian leaf.
    pub(crate) fn new(
        aos: &AosGtoData,
        home_atom: &[usize],
        kernel_atom: Option<(&[usize], &numpy::ndarray::ArrayView2<f64>)>,
    ) -> Option<Self> {
        if !supported(aos) || home_atom.len() != aos.num_ao {
            return None;
        }
        if let Some((ka, _)) = kernel_atom {
            if ka.len() != aos.num_ao_kernel {
                return None;
            }
        }
        let mut atom = Vec::with_capacity(aos.shells.len());
        let mut off = Vec::with_capacity(aos.shells.len());
        for sh in &aos.shells {
            let first = *sh.aos.first()?;
            let a0 = home_atom[aos.kernel_to_home[first]];
            let o_of = |kap: usize| -> [f64; 3] {
                match kernel_atom {
                    None => [0.0; 3],
                    Some((ka, positions)) => {
                        let at = ka[kap];
                        let ra = home_atom[aos.kernel_to_home[kap]];
                        [
                            positions[[at, 0]] - positions[[ra, 0]],
                            positions[[at, 1]] - positions[[ra, 1]],
                            positions[[at, 2]] - positions[[ra, 2]],
                        ]
                    }
                }
            };
            let o0 = o_of(first);
            for &kap in &sh.aos {
                if home_atom[aos.kernel_to_home[kap]] != a0 || o_of(kap) != o0 {
                    return None;
                }
            }
            atom.push(a0);
            off.push(o0);
        }
        Some(ShellGeom { atom, off })
    }
}

/// Leaves the kernel covers: Cartesian, finalized shell table, folded rank, at most
/// [`NA`] AOs per shell and axis powers below [`NPOW`] - 1.
pub(crate) fn supported(aos: &AosGtoData) -> bool {
    aos.kind == crate::atomic_orbital::AoKind::Cart
        && !aos.shells.is_empty()
        && aos.kernel_to_home.len() == aos.num_ao_kernel
        && aos.shells.iter().all(|s| s.aos.len() <= NA)
        && aos
            .poly_x
            .iter()
            .chain(&aos.poly_y)
            .chain(&aos.poly_z)
            .all(|&n| (0..NPOW as i64 - 1).contains(&n))
}

/// Output of [`mesh_vgs`]; every field accumulates (`+=`) so several leaves add up.
pub(crate) struct MeshVgs {
    pub np: usize,
    pub val: Vec<f64>,
    pub grad: Vec<[f64; 3]>,
    /// `np * n_atom` (point-major) when requested.
    pub atom: Vec<[f64; 3]>,
    pub n_atom: usize,
    /// `np` when requested.
    pub strain: Vec<[[f64; 3]; 3]>,
    /// `num_ao * np` (home-major) folded value table when requested.
    pub vtab: Vec<f64>,
}

impl MeshVgs {
    pub(crate) fn new(
        np: usize,
        n_atom: Option<usize>,
        strain: bool,
        vtab_ao: Option<usize>,
    ) -> Self {
        MeshVgs {
            np,
            val: vec![0.0; np],
            grad: vec![[0.0; 3]; np],
            atom: vec![[0.0; 3]; np * n_atom.unwrap_or(0)],
            n_atom: n_atom.unwrap_or(0),
            strain: if strain {
                vec![[[0.0; 3]; 3]; np]
            } else {
                Vec::new()
            },
            vtab: vec![0.0; vtab_ao.unwrap_or(0) * np],
        }
    }
}

/// Weights of [`mesh_vgs`], shared by all points.
#[derive(Clone, Copy)]
pub(crate) enum Weights<'a> {
    /// One home-rank vector (`num_ao`).
    Shared(&'a [f64]),
    /// One KERNEL-rank vector (`num_ao_kernel`, indexed by the image-tiled AO) shared by all
    /// points, e.g. a home weight times a per-image Bloch phase component.
    Kernel(&'a [f64]),
}

/// Accumulate the contractions of one leaf at the WRAPPED points `pts` into `out`
/// (see the module doc). `geom` is required for the atom-resolved / strain outputs.
pub(crate) fn mesh_vgs(
    aos: &AosGtoData,
    pts: &[[f64; 3]],
    w: Weights<'_>,
    screen: AoScreen,
    geom: Option<&ShellGeom>,
    out: &mut MeshVgs,
) {
    mesh_vgs_multi(
        aos,
        pts,
        &[w],
        &[None],
        screen,
        geom,
        std::slice::from_mut(out),
    );
}

/// [`mesh_vgs`] for up to [`NS`] weight sets at once (`outs[s]` from `ws[s]`): the AO values
/// and gradients of each (shell image, point) are formed once and contracted with every set
/// (e.g. the real and imaginary parts of Bloch-phased complex weights). `vscale[s]`, when
/// given, is a per-KERNEL-AO factor of set `s`'s folded value table
/// (`vtab[home][v] = Sum_{kap -> home} vscale[kap] phi_kap(r_v)`, e.g. one component of the
/// image Bloch phase); `None` keeps the unweighted table.
pub(crate) fn mesh_vgs_multi(
    aos: &AosGtoData,
    pts: &[[f64; 3]],
    ws: &[Weights<'_>],
    vscale: &[Option<&[f64]>],
    screen: AoScreen,
    geom: Option<&ShellGeom>,
    outs: &mut [MeshVgs],
) {
    let np = pts.len();
    let ns = ws.len();
    assert!(
        ns <= NS && ns == outs.len() && ns == vscale.len(),
        "mesh_vgs_multi: weight sets"
    );
    for out in outs.iter() {
        assert_eq!(np, out.np);
        assert!(
            !(!out.atom.is_empty() || !out.strain.is_empty()) || geom.is_some(),
            "mesh_vgs: atom/strain need ShellGeom"
        );
    }
    let mut v0 = 0;
    while v0 < np {
        let n = (np - v0).min(NP);
        chunk(aos, &pts[v0..v0 + n], v0, ws, vscale, screen, geom, outs);
        v0 += n;
    }
}

/// Maximal number of weight sets of [`mesh_vgs_multi`].
const NS: usize = 2;

#[allow(clippy::too_many_arguments)]
fn chunk(
    aos: &AosGtoData,
    pts: &[[f64; 3]],
    v0: usize,
    ws: &[Weights<'_>],
    vscale: &[Option<&[f64]>],
    screen: AoScreen,
    geom: Option<&ShellGeom>,
    outs: &mut [MeshVgs],
) {
    let nv = pts.len();
    let ns = ws.len();
    let shell_screen = screen != AoScreen::None;
    let prim_screen = screen == AoScreen::MeshValue;
    let any_vtab = outs.iter().any(|o| !o.vtab.is_empty());
    let (mut px, mut py, mut pz) = ([0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP]);
    for (v, p) in pts.iter().enumerate() {
        px[v] = p[0];
        py[v] = p[1];
        pz[v] = p[2];
    }
    let (bm, brad) = bounding_sphere(&px[..nv], &py[..nv], &pz[..nv]);
    let mut xp = [[1.0_f64; NP]; NPOW];
    let mut yp = [[1.0_f64; NP]; NPOW];
    let mut zp = [[1.0_f64; NP]; NPOW];
    let mut r2 = [0.0_f64; NP];
    let mut mask = [0.0_f64; NP];
    let mut wv = [0.0_f64; NP];
    // per set and AO: w * angular value and w * its three axis derivatives (without the radial part)
    let mut a0 = [[[0.0_f64; NP]; NA]; NS];
    let mut ax = [[[0.0_f64; NP]; NA]; NS];
    let mut ay = [[[0.0_f64; NP]; NA]; NS];
    let mut az = [[[0.0_f64; NP]; NA]; NS];
    let mut ev = [0.0_f64; NP];
    let (mut ca, mut cx, mut cy, mut cz) =
        ([0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP]);
    let mut p0 = [[0.0_f64; NP]; NS];
    let mut p1 = [[0.0_f64; NP]; NS];
    let mut qx = [[0.0_f64; NP]; NS];
    let mut qy = [[0.0_f64; NP]; NS];
    let mut qz = [[0.0_f64; NP]; NS];
    // radial sums of the value table (unweighted), per AO
    let mut rv = [[0.0_f64; NP]; NA];
    for (si, sh) in aos.shells.iter().enumerate() {
        if sh.aos.is_empty() {
            continue;
        }
        let [cxs, cys, czs] = sh.center;
        if shell_screen && dead_on_mesh(sh.center, sh.pz_val, bm, brad) {
            continue;
        }
        // Unscreened: a shell whose slowest primitive underflows `exp_neg` at every point
        // contributes exact zeros.
        if !shell_screen && dead_exact(sh.center, sh.pz_min, bm, brad) {
            continue;
        }
        let mut r2min = f64::INFINITY;
        for v in 0..nv {
            let (dx, dy, dz) = (px[v] - cxs, py[v] - cys, pz[v] - czs);
            xp[1][v] = dx;
            yp[1][v] = dy;
            zp[1][v] = dz;
            let rr = dx * dx + dy * dy + dz * dz;
            r2[v] = rr;
            r2min = if rr < r2min { rr } else { r2min };
        }
        if shell_screen && sh.pz_val * r2min > LEPSBAS {
            continue;
        }
        for v in 0..nv {
            mask[v] = if !shell_screen || sh.pz_val * r2[v] <= LEPSBAS {
                1.0
            } else {
                0.0
            };
        }
        let lsh = sh
            .aos
            .iter()
            .map(|&ao| aos.poly_x[ao].max(aos.poly_y[ao]).max(aos.poly_z[ao]))
            .max()
            .unwrap_or(0) as usize;
        for k in 2..=lsh {
            for v in 0..nv {
                xp[k][v] = xp[k - 1][v] * xp[1][v];
                yp[k][v] = yp[k - 1][v] * yp[1][v];
                zp[k][v] = zp[k - 1][v] * zp[1][v];
            }
        }
        let na = sh.aos.len();
        for (j, &ao) in sh.aos.iter().enumerate() {
            let home = aos.kernel_to_home[ao];
            let (nx, ny, nz) = (
                aos.poly_x[ao] as usize,
                aos.poly_y[ao] as usize,
                aos.poly_z[ao] as usize,
            );
            let (fx, fy, fz) = (nx as f64, ny as f64, nz as f64);
            let (xa, ya, za) = (&xp[nx], &yp[ny], &zp[nz]);
            let (xd, yd, zd) = (
                &xp[nx.saturating_sub(1)],
                &yp[ny.saturating_sub(1)],
                &zp[nz.saturating_sub(1)],
            );
            for s in 0..ns {
                let wj = match ws[s] {
                    Weights::Shared(wh) => wh[home],
                    Weights::Kernel(wk) => wk[ao],
                };
                for v in 0..nv {
                    wv[v] = wj * mask[v];
                }
                for v in 0..nv {
                    let yz = ya[v] * za[v];
                    a0[s][j][v] = xa[v] * yz * wv[v];
                    ax[s][j][v] = fx * xd[v] * yz * wv[v];
                    ay[s][j][v] = fy * xa[v] * yd[v] * za[v] * wv[v];
                    az[s][j][v] = fz * xa[v] * ya[v] * zd[v] * wv[v];
                }
            }
            if any_vtab {
                rv[j][..nv].fill(0.0);
            }
        }
        for s in 0..ns {
            for v in 0..nv {
                p0[s][v] = 0.0;
                p1[s][v] = 0.0;
                qx[s][v] = 0.0;
                qy[s][v] = 0.0;
                qz[s][v] = 0.0;
            }
        }
        for (q, &zq) in sh.prim_z.iter().enumerate() {
            let pv = sh.prim_val.get(q).copied().unwrap_or(0.0);
            if prim_screen && pv * r2min > LEPSBAS {
                continue;
            }
            if zq * r2min > -crate::fastmath::EXP_NEG_MIN_ARG {
                continue; // exp_neg underflows to exactly 0 at every point
            }
            for v in 0..nv {
                let e = crate::fastmath::exp_neg(-zq * r2[v]);
                ev[v] = if !prim_screen || pv * r2[v] <= LEPSBAS {
                    e
                } else {
                    0.0
                };
            }
            if any_vtab {
                for (j, &ao) in sh.aos.iter().enumerate().take(na) {
                    let c = aos.flat_nc[aos.ao_prim_offset[ao] + q];
                    for v in 0..nv {
                        rv[j][v] += c * ev[v];
                    }
                }
            }
            for s in 0..ns {
                for v in 0..nv {
                    ca[v] = 0.0;
                    cx[v] = 0.0;
                    cy[v] = 0.0;
                    cz[v] = 0.0;
                }
                for (j, &ao) in sh.aos.iter().enumerate().take(na) {
                    let c = aos.flat_nc[aos.ao_prim_offset[ao] + q];
                    for v in 0..nv {
                        ca[v] += c * a0[s][j][v];
                        cx[v] += c * ax[s][j][v];
                        cy[v] += c * ay[s][j][v];
                        cz[v] += c * az[s][j][v];
                    }
                }
                for v in 0..nv {
                    let e = ev[v];
                    p0[s][v] += e * ca[v];
                    p1[s][v] += zq * e * ca[v];
                    qx[s][v] += e * cx[v];
                    qy[s][v] += e * cy[v];
                    qz[s][v] += e * cz[v];
                }
            }
        }
        for s in 0..ns {
            let out = &mut outs[s];
            let n_atom = out.n_atom;
            let want_atom = !out.atom.is_empty();
            let want_strain = !out.strain.is_empty();
            for v in 0..nv {
                let gx = qx[s][v] - 2.0 * xp[1][v] * p1[s][v];
                let gy = qy[s][v] - 2.0 * yp[1][v] * p1[s][v];
                let gz = qz[s][v] - 2.0 * zp[1][v] * p1[s][v];
                let vo = v0 + v;
                out.val[vo] += p0[s][v];
                out.grad[vo][0] += gx;
                out.grad[vo][1] += gy;
                out.grad[vo][2] += gz;
                if want_atom {
                    let a = &mut out.atom[vo * n_atom + geom.unwrap().atom[si]];
                    a[0] += gx;
                    a[1] += gy;
                    a[2] += gz;
                }
                if want_strain {
                    let o = geom.unwrap().off[si];
                    let dsp = [xp[1][v] + o[0], yp[1][v] + o[1], zp[1][v] + o[2]];
                    let g = [gx, gy, gz];
                    let st = &mut out.strain[vo];
                    for c in 0..3 {
                        for d in 0..3 {
                            st[c][d] += g[c] * dsp[d];
                        }
                    }
                }
            }
            if !out.vtab.is_empty() {
                let np_all = out.np;
                for (j, &ao) in sh.aos.iter().enumerate() {
                    let (nx, ny, nz) = (
                        aos.poly_x[ao] as usize,
                        aos.poly_y[ao] as usize,
                        aos.poly_z[ao] as usize,
                    );
                    let home = aos.kernel_to_home[ao];
                    let row = &mut out.vtab[home * np_all + v0..home * np_all + v0 + nv];
                    match vscale[s] {
                        None => {
                            for v in 0..nv {
                                row[v] += xp[nx][v] * yp[ny][v] * zp[nz][v] * mask[v] * rv[j][v];
                            }
                        }
                        Some(f) => {
                            let fk = f[ao];
                            for v in 0..nv {
                                row[v] +=
                                    xp[nx][v] * yp[ny][v] * zp[nz][v] * mask[v] * rv[j][v] * fk;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Centroid and padded radius of a point set (the e_L mesh bound).
fn bounding_sphere(px: &[f64], py: &[f64], pz: &[f64]) -> ([f64; 3], f64) {
    let n = px.len() as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for v in 0..px.len() {
        mx += px[v];
        my += py[v];
        mz += pz[v];
    }
    let (mx, my, mz) = (mx / n, my / n, mz / n);
    let mut r2 = 0.0_f64;
    for v in 0..px.len() {
        let d = (px[v] - mx).powi(2) + (py[v] - my).powi(2) + (pz[v] - mz).powi(2);
        r2 = r2.max(d);
    }
    ([mx, my, mz], r2.sqrt() * (1.0 + 1e-12) + 1e-12)
}

/// `true` when `exp_neg(-pz_min r^2)` is exactly 0 at every point of the bounding sphere.
#[inline]
fn dead_exact(c: [f64; 3], pz_min: f64, m: [f64; 3], rad: f64) -> bool {
    let d = ((c[0] - m[0]).powi(2) + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt() - rad;
    d > 0.0 && pz_min * d * d > -crate::fastmath::EXP_NEG_MIN_ARG
}

/// `true` when `pz_val r^2 > LEPSBAS` at every point of the bounding sphere `(m, rad)`.
#[inline]
fn dead_on_mesh(c: [f64; 3], pz_val: f64, m: [f64; 3], rad: f64) -> bool {
    let d = ((c[0] - m[0]).powi(2) + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt() - rad;
    d > 0.0 && pz_val * d * d > LEPSBAS
}

/// `kappa(n,k,m)` of `d^k/dt^k [t^n exp(-z t^2)] = exp(-z t^2) Sum_m kappa(n,k,m) z^m
/// t^(n-k+2m)` for `k, m <= 3` (the recursion of `ao_cart_deriv::hermite_kappa`).
fn kappa(n: i64) -> [[f64; 4]; 4] {
    let mut kap = [[0.0_f64; 4]; 4];
    kap[0][0] = 1.0;
    for k in 0..3 {
        for m in 0..=k + 1 {
            let keep = if m <= k {
                kap[k][m] * (n - k as i64 + 2 * m as i64) as f64
            } else {
                0.0
            };
            let raise = if m >= 1 { -2.0 * kap[k][m - 1] } else { 0.0 };
            kap[k + 1][m] = keep + raise;
        }
    }
    kap
}

/// Per-electron weights of [`kin_contract`]: home-major `(num_ao, np)` tables (point `k` of
/// AO `mu` at `mu * np + k`) of the cotangents of the AO gradient-contracted value (`orb`),
/// the Laplacian (`lap`) and the three gradient components (`gou`, optional).
pub(crate) struct KinWeights<'a> {
    pub orb: &'a [f64],
    pub lap: Option<&'a [f64]>,
    pub gou: Option<[&'a [f64]; 3]>,
    /// Optional per-KERNEL-AO factor (`num_ao_kernel`) on every weight, e.g. one component of
    /// the per-image Bloch phase.
    pub kscale: Option<&'a [f64]>,
}

/// Output of [`kin_contract`] (accumulated).
pub(crate) struct KinOut {
    /// per point `o_k = Sum_kap [W_orb grad phi + W_lap grad lap phi + Sum_e W_gou_e grad d_e phi]`
    pub elec: Vec<[f64; 3]>,
    /// `Sum_k o_k` restricted to the shells of each home atom (`n_atom`), when requested.
    pub atom: Vec<[f64; 3]>,
    /// `Sum_k Sum_kap o_{k,kap} (x) (r_k - C_kap + R_at - R_home)` when requested.
    pub strain: Option<[[f64; 3]; 3]>,
}

/// Fused kinetic derivative contraction of one periodic Cartesian pGTO leaf at the WRAPPED
/// electrons `pts`: the gradient, Hessian and gradient-of-Laplacian of every image-tiled AO
/// are formed in registers from the shell radial moments `Sum_q c_q z_q^m exp(-z_q r^2)`
/// (`m <= 3`) and the per-axis Hermite factors, and contracted at once with the per-electron
/// weights `w` -- the kernel-rank / folded derivative tables of
/// `ao_cart_deriv::ao_deriv_kernel_multi_screened` are never formed. The gradient (`orb`
/// term) follows `scr_val`, the Hessian / gradient-of-Laplacian (`gou`, `lap` terms) follow
/// `scr_high` (the e_L screens of the value and of its derivatives, as the table kernel).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kin_contract(
    aos: &AosGtoData,
    pts: &[[f64; 3]],
    w: &KinWeights<'_>,
    scr_val: AoScreen,
    scr_high: AoScreen,
    geom: Option<&ShellGeom>,
    out: &mut KinOut,
) {
    kin_contract_multi(
        aos,
        pts,
        std::slice::from_ref(w),
        scr_val,
        scr_high,
        geom,
        std::slice::from_mut(out),
    );
}

/// [`kin_contract`] for several weight sets at once (`outs[s]` from `ws[s]`): the AO
/// derivatives of each (shell image, electron) are formed once and contracted with every set
/// (e.g. the real and imaginary parts of Bloch-phased complex weights).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kin_contract_multi(
    aos: &AosGtoData,
    pts: &[[f64; 3]],
    ws: &[KinWeights<'_>],
    scr_val: AoScreen,
    scr_high: AoScreen,
    geom: Option<&ShellGeom>,
    outs: &mut [KinOut],
) {
    let np = pts.len();
    let ns = ws.len();
    assert_eq!(outs.len(), ns);
    for o in outs.iter() {
        assert_eq!(o.elec.len(), np);
        assert!(
            !(!o.atom.is_empty() || o.strain.is_some()) || geom.is_some(),
            "kin_contract: atom/strain need ShellGeom"
        );
    }
    let n_poly = aos
        .poly_x
        .iter()
        .chain(&aos.poly_y)
        .chain(&aos.poly_z)
        .copied()
        .max()
        .unwrap_or(0)
        .max(0);
    let kap: Vec<[[f64; 4]; 4]> = (0..=n_poly).map(kappa).collect();
    let mut strain = vec![[[0.0_f64; 3]; 3]; ns];
    let mut ez = [0.0_f64; 64];
    // h[axis][n][k][m]
    let mut h = [[[[0.0_f64; 4]; 4]; NPOW]; 3];
    let alive = |sc: AoScreen, sh: &crate::atomic_orbital::Shell, r2: f64| -> bool {
        match sc {
            AoScreen::None => true,
            AoScreen::ShellValue | AoScreen::MeshValue => sh.pz_val * r2 <= LEPSBAS,
            AoScreen::Deriv => sh.pz_deriv * r2 <= LEPSBAS,
        }
    };
    let want_lap = ws.iter().any(|w| w.lap.is_some());
    let want_gou = ws.iter().any(|w| w.gou.is_some());
    // highest radial moment needed: grad 1, Hessian 2, grad-of-Laplacian 3
    let mmax = if want_lap {
        3
    } else if want_gou {
        2
    } else {
        1
    };
    let mut o = vec![[0.0_f64; 3]; ns];
    for (si, sh) in aos.shells.iter().enumerate() {
        if sh.aos.is_empty() {
            continue;
        }
        let [cx, cy, cz] = sh.center;
        let nq = sh.prim_z.len();
        assert!(nq <= ez.len(), "kin_contract: {nq} primitives in a shell");
        let lsh = sh
            .aos
            .iter()
            .map(|&ao| aos.poly_x[ao].max(aos.poly_y[ao]).max(aos.poly_z[ao]))
            .max()
            .unwrap_or(0) as usize;
        for (k, p) in pts.iter().enumerate() {
            let t = [p[0] - cx, p[1] - cy, p[2] - cz];
            let r2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
            if sh.pz_min * r2 > -crate::fastmath::EXP_NEG_MIN_ARG {
                continue; // every primitive underflows to exactly 0
            }
            let on_v = alive(scr_val, sh, r2);
            let on_h = alive(scr_high, sh, r2);
            if !on_v && (!on_h || mmax == 1) {
                continue;
            }
            let mut any = false;
            for (q, ez_q) in ez[..nq].iter_mut().enumerate() {
                *ez_q = crate::fastmath::exp_neg(-sh.prim_z[q] * r2);
                any |= *ez_q != 0.0;
            }
            if !any {
                continue;
            }
            // Hermite factors h[a][n][kd][m] = kappa(n,kd,m) t_a^(n-kd+2m), n <= lsh
            for a in 0..3 {
                let mut pw = [0.0_f64; NPOW + 3];
                let mut v = 1.0;
                for slot in pw.iter_mut().take(lsh + 4) {
                    *slot = v;
                    v *= t[a];
                }
                for n in 0..=lsh {
                    let kn = &kap[n];
                    for kd in 0..=mmax {
                        for m in 0..=kd {
                            let e = n as i64 - kd as i64 + 2 * m as i64;
                            h[a][n][kd][m] = if e < 0 {
                                0.0
                            } else {
                                kn[kd][m] * pw[e as usize]
                            };
                        }
                    }
                }
            }
            for x in o.iter_mut() {
                *x = [0.0; 3];
            }
            for &ao in &sh.aos {
                let home = aos.kernel_to_home[ao];
                let wi = home * np + k;
                // per-set effective weights (zero sets skip the derivative work entirely)
                let mut wset = [[0.0_f64; 5]; 4];
                assert!(
                    ns <= wset.len(),
                    "kin_contract_multi: at most 4 weight sets"
                );
                let mut live = false;
                for (s, w) in ws.iter().enumerate() {
                    let f = w.kscale.map_or(1.0, |ks| ks[ao]);
                    let wo = if on_v { w.orb[wi] * f } else { 0.0 };
                    let wl = match w.lap {
                        Some(l) if on_h => l[wi] * f,
                        _ => 0.0,
                    };
                    let wg = match w.gou {
                        Some(g) if on_h => [g[0][wi] * f, g[1][wi] * f, g[2][wi] * f],
                        _ => [0.0; 3],
                    };
                    wset[s] = [wo, wl, wg[0], wg[1], wg[2]];
                    live |= wset[s] != [0.0; 5];
                }
                if !live {
                    continue;
                }
                let off = aos.ao_prim_offset[ao];
                let mut sm = [0.0_f64; 4];
                for (q, ez_q) in ez[..nq].iter().enumerate() {
                    let z = sh.prim_z[q];
                    let mut tq = aos.flat_nc[off + q] * *ez_q;
                    for v in sm.iter_mut().take(mmax + 1) {
                        *v += tq;
                        tq *= z;
                    }
                }
                let (hx, hy, hz) = (
                    &h[0][aos.poly_x[ao] as usize],
                    &h[1][aos.poly_y[ao] as usize],
                    &h[2][aos.poly_z[ao] as usize],
                );
                // D(i,j,l) = Sum hx[i][mx] hy[j][my] hz[l][mz] S[mx+my+mz]
                let d = |i: usize, j: usize, l: usize| -> f64 {
                    let mut acc = 0.0;
                    for mx in 0..=i {
                        for my in 0..=j {
                            let cxy = hx[i][mx] * hy[j][my];
                            for mz in 0..=l {
                                acc += cxy * hz[l][mz] * sm[mx + my + mz];
                            }
                        }
                    }
                    acc
                };
                let g = if on_v {
                    [d(1, 0, 0), d(0, 1, 0), d(0, 0, 1)]
                } else {
                    [0.0; 3]
                };
                let hs = if on_h && want_gou {
                    [
                        d(2, 0, 0),
                        d(0, 2, 0),
                        d(0, 0, 2),
                        d(1, 1, 0),
                        d(1, 0, 1),
                        d(0, 1, 1),
                    ]
                } else {
                    [0.0; 6]
                };
                let gl = if on_h && want_lap {
                    [
                        d(3, 0, 0) + d(1, 2, 0) + d(1, 0, 2),
                        d(2, 1, 0) + d(0, 3, 0) + d(0, 1, 2),
                        d(2, 0, 1) + d(0, 2, 1) + d(0, 0, 3),
                    ]
                } else {
                    [0.0; 3]
                };
                let [hxx, hyy, hzz, hxy, hxz, hyz] = hs;
                for s in 0..ns {
                    let [wo, wl, g0, g1, g2] = wset[s];
                    let os = &mut o[s];
                    os[0] += wo * g[0] + wl * gl[0] + g0 * hxx + g1 * hxy + g2 * hxz;
                    os[1] += wo * g[1] + wl * gl[1] + g0 * hxy + g1 * hyy + g2 * hyz;
                    os[2] += wo * g[2] + wl * gl[2] + g0 * hxz + g1 * hyz + g2 * hzz;
                }
            }
            for s in 0..ns {
                let os = o[s];
                let out = &mut outs[s];
                for (c, os_c) in os.iter().enumerate() {
                    out.elec[k][c] += *os_c;
                }
                if !out.atom.is_empty() {
                    let a = &mut out.atom[geom.unwrap().atom[si]];
                    for c in 0..3 {
                        a[c] += os[c];
                    }
                }
                if out.strain.is_some() {
                    let of = geom.unwrap().off[si];
                    for c in 0..3 {
                        for dd in 0..3 {
                            strain[s][c][dd] += os[c] * (t[dd] + of[dd]);
                        }
                    }
                }
            }
        }
    }
    for s in 0..ns {
        if let Some(st) = outs[s].strain.as_mut() {
            for c in 0..3 {
                for dd in 0..3 {
                    st[c][dd] += strain[s][c][dd];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ao_cart_deriv::ao_deriv_kernel_multi_screened;
    use crate::atomic_orbital::AoKind;
    use numpy::ndarray::Array2;

    /// Reference: the derivative kernel's kernel-rank value / gradient tables (a separate
    /// Hermite-moment implementation), contracted by hand, for every screen, both weight
    /// forms (home-rank and kernel-rank), the atom-resolved gradient, the strain virial and the folded value table.
    /// Points span radii where tight shells and primitives switch off.
    #[test]
    fn mesh_vgs_matches_derivative_kernel_tables() {
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let aos = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 3, 2.5, true);
        assert!(supported(&aos));
        let close = |a: f64, b: f64| (a - b).abs() <= atol + rtol * b.abs();
        let pts: Vec<[f64; 3]> = (0..37)
            .map(|v| {
                let t = v as f64;
                let rad = 0.05 + 0.19 * t;
                [
                    0.1 + rad * (0.7 * t).cos(),
                    -0.2 + rad * (0.7 * t).sin(),
                    0.3 + 0.4 * (0.3 * t).cos(),
                ]
            })
            .collect();
        let np = pts.len();
        let pm = Array2::from_shape_fn((np, 3), |(v, c)| pts[v][c]);
        let half = aos.num_ao / 2;
        let home_atom: Vec<usize> = (0..aos.num_ao).map(|mu| usize::from(mu >= half)).collect();
        let geom = ShellGeom::new(&aos, &home_atom, None).unwrap();
        let w: Vec<f64> = (0..aos.num_ao)
            .map(|c| ((c * 37 % 11) as f64 - 5.0) / 7.0)
            .collect();
        let wk: Vec<f64> = (0..aos.num_ao_kernel)
            .map(|i| ((i * 13 % 17) as f64 - 8.0) / 9.0)
            .collect();
        let orders = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)];
        for screen in [AoScreen::None, AoScreen::ShellValue, AoScreen::MeshValue] {
            let t = ao_deriv_kernel_multi_screened(&aos, &pm.view(), &orders, &[screen; 4], false);
            for kernel_w in [false, true] {
                let wts = if kernel_w {
                    Weights::Kernel(&wk)
                } else {
                    Weights::Shared(&w)
                };
                let mut out = MeshVgs::new(np, Some(2), true, Some(aos.num_ao));
                mesh_vgs(&aos, &pts, wts, screen, Some(&geom), &mut out);
                for v in 0..np {
                    let (mut val, mut g, mut s) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
                    let mut ga = [[0.0; 3]; 2];
                    let mut vt = vec![0.0; aos.num_ao];
                    for kap in 0..aos.num_ao_kernel {
                        let h = aos.kernel_to_home[kap];
                        let wq = if kernel_w { wk[kap] } else { w[h] };
                        val += wq * t[0][[kap, v]];
                        vt[h] += t[0][[kap, v]];
                        for c in 0..3 {
                            let gk = wq * t[1 + c][[kap, v]];
                            g[c] += gk;
                            ga[home_atom[h]][c] += gk;
                            for d in 0..3 {
                                s[c][d] += gk * (pts[v][d] - aos.centers[[kap, d]]);
                            }
                        }
                    }
                    assert!(close(out.val[v], val), "val {screen:?} {kernel_w} v={v}");
                    for (h, &vt_h) in vt.iter().enumerate() {
                        assert!(close(out.vtab[h * np + v], vt_h), "vtab {screen:?} v={v}");
                    }
                    for c in 0..3 {
                        assert!(close(out.grad[v][c], g[c]), "grad {screen:?} v={v}");
                        for (a, ga_a) in ga.iter().enumerate() {
                            assert!(close(out.atom[v * 2 + a][c], ga_a[c]), "atom {screen:?}");
                        }
                        for (d, &s_cd) in s[c].iter().enumerate() {
                            assert!(close(out.strain[v][c][d], s_cd), "strain {screen:?}");
                        }
                    }
                }
            }
        }
    }

    /// [`mesh_vgs_multi`] with two kernel-rank weight sets and a per-kernel-AO value-table
    /// scale on one of them equals the derivative kernel's tables contracted by hand per set
    /// (the scaled folded table `Sum_{kap -> home} f_kap phi_kap`).
    #[test]
    fn mesh_vgs_multi_matches_derivative_kernel_tables() {
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let aos = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 3, 2.5, true);
        let close = |a: f64, b: f64| (a - b).abs() <= atol + rtol * b.abs();
        let pts: Vec<[f64; 3]> = (0..41)
            .map(|v| {
                let t = v as f64;
                let rad = 0.05 + 0.17 * t;
                [
                    0.2 + rad * (0.6 * t).cos(),
                    -0.1 + rad * (0.6 * t).sin(),
                    0.1 + 0.5 * (0.2 * t).cos(),
                ]
            })
            .collect();
        let np = pts.len();
        let pm = Array2::from_shape_fn((np, 3), |(v, c)| pts[v][c]);
        let half = aos.num_ao / 2;
        let home_atom: Vec<usize> = (0..aos.num_ao).map(|mu| usize::from(mu >= half)).collect();
        let geom = ShellGeom::new(&aos, &home_atom, None).unwrap();
        let nk = aos.num_ao_kernel;
        let w1: Vec<f64> = (0..nk)
            .map(|i| ((i * 13 % 17) as f64 - 8.0) / 9.0)
            .collect();
        let w2: Vec<f64> = (0..nk).map(|i| ((i * 7 % 11) as f64 - 5.0) / 6.0).collect();
        let f2: Vec<f64> = (0..nk).map(|i| (0.3 * i as f64).cos()).collect();
        let orders = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)];
        for screen in [AoScreen::None, AoScreen::MeshValue] {
            let t = ao_deriv_kernel_multi_screened(&aos, &pm.view(), &orders, &[screen; 4], false);
            let mut outs = [
                MeshVgs::new(np, Some(2), true, Some(aos.num_ao)),
                MeshVgs::new(np, Some(2), true, Some(aos.num_ao)),
            ];
            mesh_vgs_multi(
                &aos,
                &pts,
                &[Weights::Kernel(&w1), Weights::Kernel(&w2)],
                &[None, Some(&f2)],
                screen,
                Some(&geom),
                &mut outs,
            );
            for (s, (wk, fk)) in [(&w1, None), (&w2, Some(&f2))].into_iter().enumerate() {
                let out = &outs[s];
                for v in 0..np {
                    let (mut val, mut g, mut st) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
                    let mut ga = [[0.0; 3]; 2];
                    let mut vt = vec![0.0; aos.num_ao];
                    for kap in 0..nk {
                        let h = aos.kernel_to_home[kap];
                        val += wk[kap] * t[0][[kap, v]];
                        vt[h] += fk.map_or(1.0, |f: &Vec<f64>| f[kap]) * t[0][[kap, v]];
                        for c in 0..3 {
                            let gk = wk[kap] * t[1 + c][[kap, v]];
                            g[c] += gk;
                            ga[home_atom[h]][c] += gk;
                            for d in 0..3 {
                                st[c][d] += gk * (pts[v][d] - aos.centers[[kap, d]]);
                            }
                        }
                    }
                    assert!(close(out.val[v], val), "val set {s} {screen:?} v={v}");
                    for (h, &vt_h) in vt.iter().enumerate() {
                        assert!(
                            close(out.vtab[h * np + v], vt_h),
                            "vtab set {s} {screen:?} v={v}"
                        );
                    }
                    for c in 0..3 {
                        assert!(close(out.grad[v][c], g[c]), "grad set {s} {screen:?} v={v}");
                        for (a, ga_a) in ga.iter().enumerate() {
                            assert!(
                                close(out.atom[v * 2 + a][c], ga_a[c]),
                                "atom set {s} {screen:?}"
                            );
                        }
                        for (d, &st_cd) in st[c].iter().enumerate() {
                            assert!(
                                close(out.strain[v][c][d], st_cd),
                                "strain set {s} {screen:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Reference for [`kin_contract`]: the derivative kernel's kernel-rank gradient, Hessian
    /// and gradient-of-Laplacian tables contracted by hand with the per-electron weights, for
    /// the value / derivative screen pairs the callers use.
    #[test]
    fn kin_contract_matches_derivative_kernel_tables() {
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let aos = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 3, 2.5, true);
        let close = |a: f64, b: f64| (a - b).abs() <= atol + rtol * b.abs();
        let pts: Vec<[f64; 3]> = (0..9)
            .map(|v| {
                let t = v as f64;
                [
                    0.1 + 0.37 * t - 1.0,
                    -0.2 + 0.5 * (0.9 * t).sin(),
                    0.3 + 0.6 * (0.4 * t).cos(),
                ]
            })
            .collect();
        let np = pts.len();
        let pm = Array2::from_shape_fn((np, 3), |(v, c)| pts[v][c]);
        let half = aos.num_ao / 2;
        let home_atom: Vec<usize> = (0..aos.num_ao).map(|mu| usize::from(mu >= half)).collect();
        let geom = ShellGeom::new(&aos, &home_atom, None).unwrap();
        let tab = |f: usize| -> Vec<f64> {
            (0..aos.num_ao * np)
                .map(|i| (((i * f) % 23) as f64 - 11.0) / 13.0)
                .collect()
        };
        let (wo, wl, g0, g1, g2) = (tab(7), tab(11), tab(5), tab(3), tab(17));
        // orders: grad (3), Hessian xx yy zz xy xz yz (6), third orders for grad-lap (9)
        let orders: Vec<(u32, u32, u32)> = vec![
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (3, 0, 0),
            (1, 2, 0),
            (1, 0, 2),
            (2, 1, 0),
            (0, 3, 0),
            (0, 1, 2),
            (2, 0, 1),
            (0, 2, 1),
            (0, 0, 3),
        ];
        for (sv, sh) in [
            (AoScreen::None, AoScreen::None),
            (AoScreen::ShellValue, AoScreen::Deriv),
        ] {
            let mut scr = vec![sv; 3];
            scr.extend(vec![sh; 15]);
            let t = ao_deriv_kernel_multi_screened(&aos, &pm.view(), &orders, &scr, false);
            let w = KinWeights {
                orb: &wo,
                lap: Some(&wl),
                gou: Some([&g0, &g1, &g2]),
                kscale: None,
            };
            let mut out = KinOut {
                elec: vec![[0.0; 3]; np],
                atom: vec![[0.0; 3]; 2],
                strain: Some([[0.0; 3]; 3]),
            };
            kin_contract(&aos, &pts, &w, sv, sh, Some(&geom), &mut out);
            let (mut atom, mut strain) = ([[0.0; 3]; 2], [[0.0; 3]; 3]);
            for k in 0..np {
                let mut o = [0.0; 3];
                for kap in 0..aos.num_ao_kernel {
                    let h = aos.kernel_to_home[kap];
                    let i = h * np + k;
                    let hs = |a: usize, b: usize| -> f64 {
                        let idx = match (a.min(b), a.max(b)) {
                            (0, 0) => 3,
                            (1, 1) => 4,
                            (2, 2) => 5,
                            (0, 1) => 6,
                            (0, 2) => 7,
                            _ => 8,
                        };
                        t[idx][[kap, k]]
                    };
                    let gl = [
                        t[9][[kap, k]] + t[10][[kap, k]] + t[11][[kap, k]],
                        t[12][[kap, k]] + t[13][[kap, k]] + t[14][[kap, k]],
                        t[15][[kap, k]] + t[16][[kap, k]] + t[17][[kap, k]],
                    ];
                    let mut ok = [0.0; 3];
                    for c in 0..3 {
                        ok[c] = wo[i] * t[c][[kap, k]]
                            + wl[i] * gl[c]
                            + g0[i] * hs(c, 0)
                            + g1[i] * hs(c, 1)
                            + g2[i] * hs(c, 2);
                        o[c] += ok[c];
                        atom[home_atom[h]][c] += ok[c];
                        for d in 0..3 {
                            strain[c][d] += ok[c] * (pts[k][d] - aos.centers[[kap, d]]);
                        }
                    }
                }
                for (c, o_c) in o.iter().enumerate() {
                    assert!(close(out.elec[k][c], *o_c), "elec {sv:?} k={k}");
                }
            }
            let st = out.strain.unwrap();
            for c in 0..3 {
                for (a, atom_a) in atom.iter().enumerate() {
                    assert!(close(out.atom[a][c], atom_a[c]), "atom {sv:?}");
                }
                for d in 0..3 {
                    assert!(close(st[c][d], strain[c][d]), "strain {sv:?}");
                }
            }
        }
    }
}
