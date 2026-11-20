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

//! Wavefunction-level value functions (`wavefunction.py`): the local kinetic
//! energy assembled from the analytic grad/Laplacian of `ln|det G|` and the
//! Jastrow part. Pure value function (autodiff-free), 1:1 with
//! `compute_kinetic_energy` (wavefunction.py).

use crate::data::JastrowData;
use crate::determinant::{
    compute_grads_and_laplacian_ln_det, compute_grads_and_laplacian_ln_det_with_inv, GeminalData,
};
use crate::jastrow_factor::{
    compute_grads_and_laplacian_jastrow_part, compute_grads_and_laplacian_jastrow_three_body,
    compute_grads_and_laplacian_jastrow_three_body_fast,
};
use numpy::ndarray::{Array1, Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// Mirrors `jqmc.wavefunction.Wavefunction_data`: the trial
/// wavefunction = Jastrow x Geminal. Real (Gamma / molecular) path; the complex
/// twist uses `ComplexGeminal` directly (its geminal leaf is complex).
pub struct WavefunctionData {
    pub jastrow: JastrowData,
    pub geminal: GeminalData,
}

impl WavefunctionData {
    pub fn from_py(wf: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(WavefunctionData {
            jastrow: JastrowData::from_py(&wf.getattr("jastrow_data")?)?,
            geminal: GeminalData::from_py(&wf.getattr("geminal_data")?)?,
        })
    }
}

/// Local kinetic energy
/// `T_L = 1/2[ -(Sum grad^2lnPsi) - Sum_i |grad_i lnPsi|^2 ]`, with `lnPsi = ln|det G| + J`
/// (`gradlnPsi = gradln|det G| + gradJ`, `J = J1+J2+J3`). 1:1 with `compute_kinetic_energy`.
/// `j3` supplies the three-body `(orb, j_matrix)` when present (None => J1+J2 only).
pub fn compute_kinetic_energy(
    geminal_data: &GeminalData,
    jastrow_data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let (mut gj_up, mut gj_dn, mut lj_up, mut lj_dn) = crate::timed!(
        "kin.j12",
        compute_grads_and_laplacian_jastrow_part(jastrow_data, r_up, r_dn)
    );
    // J3 (three-body) lives on `jastrow_data.three_body`, read
    // here exactly as jqmc's `compute_grads_and_laplacian_Jastrow_part` folds in
    // `jastrow_three_body_data`. The orb encodes periodicity (image-tiled AOs),
    // so this is correct for both molecular and Gamma-point PBC.
    if let Some(j3) = &jastrow_data.three_body {
        let (g3u, g3d, l3u, l3d) = crate::timed!(
            "kin.j3",
            compute_grads_and_laplacian_jastrow_three_body(
                &j3.orb,
                &j3.j_matrix.view(),
                r_up,
                r_dn
            )
        );
        if jastrow_data.precision_policy.is_f32("jastrow_grad_lap") {
            gj_up = (gj_up.mapv(|x| x as f32) + g3u.mapv(|x| x as f32)).mapv(f64::from);
            gj_dn = (gj_dn.mapv(|x| x as f32) + g3d.mapv(|x| x as f32)).mapv(f64::from);
            lj_up = (lj_up.mapv(|x| x as f32) + l3u.mapv(|x| x as f32)).mapv(f64::from);
            lj_dn = (lj_dn.mapv(|x| x as f32) + l3d.mapv(|x| x as f32)).mapv(f64::from);
        } else {
            gj_up = gj_up + g3u;
            gj_dn = gj_dn + g3d;
            lj_up = lj_up + l3u;
            lj_dn = lj_dn + l3d;
        }
    }
    let (gd_up, gd_dn, ld_up, ld_dn) = crate::timed!(
        "kin.det",
        compute_grads_and_laplacian_ln_det(geminal_data, r_up, r_dn)
    );
    assemble_kinetic(
        r_up, r_dn, &gj_up, &gj_dn, &lj_up, &lj_dn, &gd_up, &gd_dn, &ld_up, &ld_dn,
    )
}

/// Local kinetic energy reusing a caller-supplied geminal inverse. Identical to
/// `compute_kinetic_energy` except the determinant
/// grad/laplacian is evaluated with the Sherman-Morrison running inverse the
/// MCMC sweep emits (`compute_grads_and_laplacian_ln_det_with_inv`), matching
/// jqmc's `compute_kinetic_energy_all_elements_fast_update` -- avoids the
/// fresh-LU NaN at a near-singular geminal (jqmc `compute_local_energy_fast`).
///
/// The reused `g_inv` is the sweep's SM inverse, seeded/streamed with the per-move
/// SINGLE AO path (`orb_matrix_via_single`), so the determinant grad/laplacian is
/// taken with the matching SINGLE AO convention (`via_single = true`) -- else the
/// ~1e-9 single-vs-batch periodic-image gap is amplified by the Gaussian curvature
/// into a ~0.06 Ha kinetic error. Molecular, pure PW, and B-spline leaves
/// delegate to the batch path, preserving their existing evaluation order.
pub fn compute_kinetic_energy_with_inv(
    geminal_data: &GeminalData,
    jastrow_data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: &ArrayView2<f64>,
) -> f64 {
    // J1/J2 stay on the raw coordinate and use min-image distances. The J3
    // three-body term is orbital-based:
    // jqmc wraps its AO internally; the native J3 grad/laplacian kernel
    // home-cell-wraps internally too, with one wrap shared by local-energy,
    // LRDMC, and force consumers. No wrap is needed here.
    let (mut gj_up, mut gj_dn, mut lj_up, mut lj_dn) = crate::timed!(
        "kin.j12",
        compute_grads_and_laplacian_jastrow_part(jastrow_data, r_up, r_dn)
    );
    if let Some(j3) = &jastrow_data.three_body {
        let (g3u, g3d, l3u, l3d) = crate::timed!(
            "kin.j3",
            compute_grads_and_laplacian_jastrow_three_body(
                &j3.orb,
                &j3.j_matrix.view(),
                r_up,
                r_dn,
            )
        );
        if jastrow_data.precision_policy.is_f32("jastrow_grad_lap") {
            gj_up = (gj_up.mapv(|x| x as f32) + g3u.mapv(|x| x as f32)).mapv(f64::from);
            gj_dn = (gj_dn.mapv(|x| x as f32) + g3d.mapv(|x| x as f32)).mapv(f64::from);
            lj_up = (lj_up.mapv(|x| x as f32) + l3u.mapv(|x| x as f32)).mapv(f64::from);
            lj_dn = (lj_dn.mapv(|x| x as f32) + l3d.mapv(|x| x as f32)).mapv(f64::from);
        } else {
            gj_up = gj_up + g3u;
            gj_dn = gj_dn + g3d;
            lj_up = lj_up + l3u;
            lj_dn = lj_dn + l3d;
        }
    }
    let (gd_up, gd_dn, ld_up, ld_dn) = crate::timed!(
        "kin.det",
        compute_grads_and_laplacian_ln_det_with_inv(geminal_data, r_up, r_dn, g_inv, true)
    );
    assemble_kinetic(
        r_up, r_dn, &gj_up, &gj_dn, &lj_up, &lj_dn, &gd_up, &gd_dn, &ld_up, &ld_dn,
    )
}

/// FAST-mode kinetic (production chunk only, floor-compared): as
/// [`compute_kinetic_energy_with_inv`] but the det part reuses the sweep's
/// streaming state for the orbital VALUE matrices + paired product and applies
/// the molecular LEPSBAS cutoff to the AO grad/lap sweep
/// (`compute_grads_and_laplacian_ln_det_fast`). Molecular MO-GTO only
/// (caller-gated). J1/J2/J3 identical to the exact path.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit wavefunction, inverse and carried-cache inputs"
)]
pub fn compute_kinetic_energy_fast(
    geminal_data: &GeminalData,
    jastrow_data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: &ArrayView2<f64>,
    st: &crate::determinant::DetRatioState,
    // Chunk-carried MO derivative tables at exactly (r_up, r_dn) (molecular
    // FAST chunk: only moved electrons' columns refreshed). `None` builds them.
    det_gl_cache: Option<&crate::determinant::DetGradLapCache>,
    // Chunk-carried raw-AO J3 derivative table + the sweep's J3 ratio state.
    j3_gl: Option<(
        &crate::jastrow_factor::J3GradLapCache,
        &crate::jastrow_factor::J3RatioState,
    )>,
) -> f64 {
    let (mut gj_up, mut gj_dn, mut lj_up, mut lj_dn) = crate::timed!(
        "kin.j12",
        compute_grads_and_laplacian_jastrow_part(jastrow_data, r_up, r_dn)
    );
    if let Some(j3) = &jastrow_data.three_body {
        let (g3u, g3d, l3u, l3d) = crate::timed!(
            "kin.j3",
            match j3_gl {
                Some((cache, st3)) => {
                    crate::jastrow_factor::contract_j3_grad_lap_cache(
                        &j3.j_matrix.view(),
                        st3,
                        cache,
                    )
                }
                None => compute_grads_and_laplacian_jastrow_three_body_fast(
                    &j3.orb,
                    &j3.j_matrix.view(),
                    r_up,
                    r_dn,
                ),
            }
        );
        if jastrow_data.precision_policy.is_f32("jastrow_grad_lap") {
            gj_up = (gj_up.mapv(|x| x as f32) + g3u.mapv(|x| x as f32)).mapv(f64::from);
            gj_dn = (gj_dn.mapv(|x| x as f32) + g3d.mapv(|x| x as f32)).mapv(f64::from);
            lj_up = (lj_up.mapv(|x| x as f32) + l3u.mapv(|x| x as f32)).mapv(f64::from);
            lj_dn = (lj_dn.mapv(|x| x as f32) + l3d.mapv(|x| x as f32)).mapv(f64::from);
        } else {
            gj_up = gj_up + g3u;
            gj_dn = gj_dn + g3d;
            lj_up = lj_up + l3u;
            lj_dn = lj_dn + l3d;
        }
    }
    let (gd_up, gd_dn, ld_up, ld_dn) = crate::timed!(
        "kin.det",
        match det_gl_cache {
            Some(cache) =>
                crate::determinant::contract_det_grad_lap_cache(geminal_data, g_inv, st, cache),
            None => crate::determinant::compute_grads_and_laplacian_ln_det_fast(
                geminal_data,
                r_up,
                r_dn,
                g_inv,
                st
            ),
        }
    );
    assemble_kinetic(
        r_up, r_dn, &gj_up, &gj_dn, &lj_up, &lj_dn, &gd_up, &gd_dn, &ld_up, &ld_dn,
    )
}

/// Shared kinetic assembly `0.5 * (-sum_lap - sum |grad J + grad ln|det||^2)`
/// from the Jastrow + determinant grad/laplacian pieces (identical accumulation
/// order for the fresh-inverse and reused-inverse kinetic entries).
#[allow(clippy::too_many_arguments)]
fn assemble_kinetic(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    lj_up: &Array1<f64>,
    lj_dn: &Array1<f64>,
    gd_up: &Array2<f64>,
    gd_dn: &Array2<f64>,
    ld_up: &Array1<f64>,
    ld_dn: &Array1<f64>,
) -> f64 {
    let sum_lap: f64 = lj_up.sum() + lj_dn.sum() + ld_up.sum() + ld_dn.sum();

    let mut sum_grad2 = 0.0_f64;
    for i in 0..r_up.nrows() {
        for c in 0..3 {
            let v = gj_up[[i, c]] + gd_up[[i, c]];
            sum_grad2 += v * v;
        }
    }
    for i in 0..r_dn.nrows() {
        for c in 0..3 {
            let v = gj_dn[[i, c]] + gd_dn[[i, c]];
            sum_grad2 += v * v;
        }
    }

    0.5 * (-sum_lap - sum_grad2)
}
