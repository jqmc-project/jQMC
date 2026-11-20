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

//! Complex Bloch geminal / determinant (PBC twist k!=Gamma). Mirrors the real
//! `determinant.rs` + `param_grad_el::gem_forward` but with complex orbitals
//! (MO*AO both complex) and a real lambda. The wavefunction amplitude used by MCMC is
//! `ln|det G|` (real) -- the det phase drops out of |Psi|^2. Plain transpose `ouT*W`
//! (no conjugation: the down-spin leaf uses `-k`, and its MO coefficients
//! already contain the conjugate of the up-spin `+k` coefficients).

use numpy::ndarray::{Array2, ArrayView2};
use numpy::Complex64;
use pyo3::prelude::*;

use crate::atomic_orbital::AosGtoData;
use crate::complex_ao::compute_aos_vgl_complex;

/// One complex orbital leaf: real Gaussian AO kernel + per-image Bloch phase +
/// (optional) complex MO coefficients `(num_mo, num_ao)`.
pub struct ComplexOrb {
    pub(crate) precision_policy: crate::PrecisionPolicy,
    pub aos: AosGtoData,
    pub image_phase: Vec<Complex64>,
    pub mo: Option<Array2<Complex64>>,
    /// Plane-wave child of a pgto(+)pw composite (`None` for pure-pGTO). `gto_first`
    /// = the gto AO block precedes the pw block in the (pre-MO) AO ordering.
    pub pw: Option<crate::complex_ao::ComplexPwData>,
    pub gto_first: bool,
    /// Periodic wrap+restore for the gto block (`None` = molecular / Gamma; the
    /// unwrapped sum already suffices there). PW is never wrapped.
    pub wrap: Option<crate::complex_ao::WrapCtx>,
    /// Complex B-spline child of a pgto(+)bspline composite, or a pure complex
    /// B-spline leaf (then `aos` is the 0-AO placeholder). Mutually exclusive with
    /// `pw`; concatenates with the gto block in `gto_first` order (the Bloch phase
    /// is applied inside the spline kernel, so no `wrap` is needed for it).
    pub bspline: Option<crate::bspline::AosBsplineComplexData>,
    pub tiled: Option<Box<crate::orbital_tiling::TiledOrbitalSet>>,
}

/// Inverse of a 3x3 (rows = lattice vectors) via the adjugate / determinant.
pub(crate) fn inv_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Keep the cofactor construction, determinant association and element-wise
    // division identical to structure._inv_3x3.  A reciprocal followed by a
    // multiply differs by an fp64 ulp on oblique cells; at a periodic seam that
    // can change the wrapped fp32 AO coordinate by several ulps.
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    [
        [c00 / det, c10 / det, c20 / det],
        [c01 / det, c11 / det, c21 / det],
        [c02 / det, c12 / det, c22 / det],
    ]
}

/// Stack two `(m,N)` complex AO blocks `a`(top)/`b`(bottom) -> `(m_a+m_b, N)`.
fn vstack_c(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (ma, mb, n) = (a.nrows(), b.nrows(), a.ncols());
    let mut out = Array2::<Complex64>::zeros((ma + mb, n));
    out.slice_mut(numpy::ndarray::s![0..ma, ..]).assign(a);
    out.slice_mut(numpy::ndarray::s![ma.., ..]).assign(b);
    out
}

impl ComplexOrb {
    /// Read a complex pgto orbital leaf (`MOs_data` over a complex AO leaf, or a
    /// bare complex AO leaf) from Python.
    pub fn from_py(orb_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_py_with_tr_source(orb_data, None)
    }

    /// Parse an orbital, optionally reusing a time-reversal partner's complex B-spline
    /// coefficient backing. Other (small) AO/MO metadata remains independently owned.
    fn from_py_with_tr_source(
        orb_data: &Bound<'_, PyAny>,
        tr_source: Option<&ComplexOrb>,
    ) -> PyResult<Self> {
        let py = orb_data.py();
        let np = py.import("numpy")?;
        if orb_data.get_type().name()? == "TiledOrbitalSet_data" {
            return Ok(Self {
                precision_policy: crate::construction_precision_policy(),
                aos: AosGtoData::empty(),
                image_phase: Vec::new(),
                mo: None,
                pw: None,
                gto_first: true,
                wrap: None,
                bspline: None,
                tiled: Some(Box::new(
                    crate::orbital_tiling::TiledOrbitalSet::from_py_with_source(
                        orb_data,
                        tr_source.and_then(|orb| orb.tiled.as_deref()),
                    )?,
                )),
            });
        }
        let (aos_py, mo) = if orb_data.hasattr("aos_data")? {
            let aos_py = orb_data.getattr("aos_data")?;
            let identity = orb_data
                .getattr("identity")
                .and_then(|value| value.extract::<bool>())
                .unwrap_or(false);
            if identity {
                (aos_py, None)
            } else {
                // Coerce to complex128: the twist geminal MO is genuinely complex, but an
                // identity-MO-wrapped J3 composite carries a REAL (float64) eye -> astype
                // makes both paths read uniformly as Complex64.
                let c: numpy::PyReadonlyArray2<Complex64> = np
                    .call_method1("asarray", (orb_data.getattr("mo_coefficients")?,))?
                    .call_method1("astype", (np.getattr("complex128")?,))?
                    .extract()?;
                (aos_py, Some(c.as_array().to_owned()))
            }
        } else {
            (orb_data.clone(), None)
        };
        // Composite pgto(+){pw,bspline}: `aos_data.children` = (gto_leaf, extra_leaf)
        // in some order. The gto child carries `_image_phase_factors`; the extra child
        // is either a complex plane-wave (`AOs_plane_wave_complex_data`) or a complex
        // B-spline (`AOs_bspline_complex_data`, has `spline_coefficients`). `pw` and
        // `bspline` are mutually exclusive. A bare complex pgto leaf has no children; a
        // pure complex B-spline leaf has `spline_coefficients` and no gto block.
        let mut pw: Option<crate::complex_ao::ComplexPwData> = None;
        let mut bspline: Option<crate::bspline::AosBsplineComplexData> = None;
        let source_bspline = tr_source.and_then(|orb| orb.bspline.as_ref());
        let mut gto_first = true;
        let is_pw = |c: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(c.get_type().name()?.to_string().contains("plane_wave"))
        };
        let gto_py: Option<Bound<'_, PyAny>> = if aos_py.hasattr("children")? {
            let children = aos_py.getattr("children")?;
            // Only children 0 and 1 are read below, as one gto + one extra block. Refuse
            // every other composite (issue #165) instead of dropping children or parsing
            // a gto child as a plane wave.
            let n_children: usize = children.len()?;
            if n_children != 2 {
                return Err(crate::determinant::unsupported_composite_err(
                    "complex",
                    &format!("{n_children} children"),
                ));
            }
            let c0 = children.get_item(0)?;
            let c1 = children.get_item(1)?;
            // The gto child is the one that is neither a B-spline nor a plane-wave.
            let c0_extra = c0.hasattr("spline_coefficients")? || is_pw(&c0)?;
            let c1_extra = c1.hasattr("spline_coefficients")? || is_pw(&c1)?;
            if c0_extra == c1_extra {
                return Err(crate::determinant::unsupported_composite_err(
                    "complex",
                    if c0_extra {
                        "two plane-wave/B-spline children"
                    } else {
                        "two GTO/pGTO children"
                    },
                ));
            }
            let (gto_c, extra_c, gf) = if c0_extra {
                (c1, c0, false)
            } else {
                (c0, c1, true)
            };
            gto_first = gf;
            if extra_c.hasattr("spline_coefficients")? {
                bspline = Some(match source_bspline {
                    Some(source) => {
                        crate::bspline::AosBsplineComplexData::from_py_with_conjugated_coefficients(
                            &extra_c, source,
                        )?
                    }
                    None => crate::bspline::AosBsplineComplexData::from_py(&extra_c)?,
                });
            } else {
                pw = Some(crate::complex_ao::ComplexPwData::from_py(&extra_c)?);
            }
            Some(gto_c)
        } else if aos_py.hasattr("spline_coefficients")? {
            bspline = Some(match source_bspline {
                Some(source) => {
                    crate::bspline::AosBsplineComplexData::from_py_with_conjugated_coefficients(
                        &aos_py, source,
                    )?
                }
                None => crate::bspline::AosBsplineComplexData::from_py(&aos_py)?,
            });
            None
        } else if is_pw(&aos_py)? {
            // Pure complex plane-wave leaf (`AOs_plane_wave_complex_data`): no gto
            // primitives. Route to the complex PW kernel with a 0-AO gto placeholder --
            // the mirror of the REAL `OrbData::from_py` pure `AOs_plane_wave_data` path
            // (and of the complex PW *child* branch above). Skips `_num_ao_kernel`, which
            // this leaf does not carry (it exposes `_gk_vectors_jnp` / `_g_vectors_jnp`).
            pw = Some(crate::complex_ao::ComplexPwData::from_py(&aos_py)?);
            None
        } else {
            Some(aos_py.clone())
        };
        // gto block (`None` for a pure complex B-spline / plane-wave leaf -> 0-AO placeholder).
        let (aos, image_phase, wrap) = match &gto_py {
            None => (AosGtoData::empty(), Vec::new(), None),
            Some(gto_py) => {
                let aos = AosGtoData::from_py(gto_py)?;
                // Twist (`_ComplexBlochPGTOMixin`) leaf carries `_image_phase_factors` =
                // e^{ik*T} per kernel-AO. A REAL-Gamma pgto child (k = 0) lacks it; at
                // k = 0 every phase is 1, so fall back to ones and the complex fold
                // reduces to the plain real sum.
                let image_phase: Vec<Complex64> = match gto_py.getattr("_image_phase_factors") {
                    Ok(a) => {
                        let ipf: numpy::PyReadonlyArray1<Complex64> =
                            np.call_method1("asarray", (a,))?.extract()?;
                        ipf.as_array().to_vec()
                    }
                    Err(_) => vec![Complex64::new(1.0, 0.0); aos.num_ao_kernel],
                };
                let wrap = Self::wrap_from_py(&np, gto_py, &aos)?;
                (aos, image_phase, wrap)
            }
        };
        let extra_rows = pw
            .as_ref()
            .map(|p| p.num_ao)
            .or_else(|| bspline.as_ref().map(|b| b.num_ao))
            .unwrap_or(0);
        crate::determinant::check_native_ao_rank(
            "complex",
            &aos_py,
            aos.num_ao + extra_rows,
            0,
            mo.as_ref().map(|c| c.ncols()),
        )?;
        Ok(ComplexOrb {
            precision_policy: crate::construction_precision_policy(),
            aos,
            image_phase,
            mo,
            pw,
            gto_first,
            wrap,
            bspline,
            tiled: None,
        })
    }

    /// Periodic wrap+restore context for the gto block (every periodic leaf,
    /// including a one-image one: its single home image only covers home-cell
    /// coordinates, so a translated electron must be wrapped and phase-restored).
    fn wrap_from_py(
        np: &Bound<'_, PyAny>,
        gto_py: &Bound<'_, PyAny>,
        aos: &AosGtoData,
    ) -> PyResult<Option<crate::complex_ao::WrapCtx>> {
        if aos.n_image > 1 || crate::atomic_orbital::leaf_pbc_flag(gto_py)? {
            let sd = gto_py.getattr("structure_data")?;
            let cell_a: numpy::PyReadonlyArray2<f64> = np
                .call_method1("asarray", (sd.getattr("_cell_jnp")?,))?
                .extract()?;
            // Twist leaf exposes `k_frac_np`; a REAL-Gamma child has no twist, so
            // k_frac = 0 (restore phase e^{2pii n0*k_frac} == 1 -- wrap is a pure fold).
            let k_frac: [f64; 3] = match gto_py.getattr("k_frac_np") {
                Ok(kf_obj) => {
                    let kf: numpy::PyReadonlyArray1<f64> =
                        np.call_method1("asarray", (kf_obj,))?.extract()?;
                    let kfa = kf.as_array();
                    [kfa[0], kfa[1], kfa[2]]
                }
                Err(_) => [0.0, 0.0, 0.0],
            };
            let c = cell_a.as_array();
            let cell = [
                [c[[0, 0]], c[[0, 1]], c[[0, 2]]],
                [c[[1, 0]], c[[1, 1]], c[[1, 2]]],
                [c[[2, 0]], c[[2, 1]], c[[2, 2]]],
            ];
            let inv = inv_3x3(&cell);
            Ok(Some(crate::complex_ao::WrapCtx {
                cell,
                inv_cell: inv,
                k_frac,
            }))
        } else {
            Ok(None)
        }
    }

    /// gto AO values with periodic wrap+restore (jqmc convention). `None` wrap ->
    /// evaluate at raw r (molecular / Gamma).
    fn gto_value(&self, r: &ArrayView2<f64>) -> Array2<Complex64> {
        let f32_arithmetic = self.precision_policy.is_f32(self.aos.value_zone);
        match &self.wrap {
            None => crate::complex_ao::compute_aos_complex_with_policy(
                &self.aos,
                &self.image_phase,
                r,
                f32_arithmetic,
            ),
            Some(w) => {
                let (rw, restore) = w.wrap(r);
                let mut v = crate::complex_ao::compute_aos_complex_with_policy(
                    &self.aos,
                    &self.image_phase,
                    &rw.view(),
                    f32_arithmetic,
                );
                if f32_arithmetic {
                    crate::complex_ao::apply_restore_f32(&mut v, &restore);
                } else {
                    crate::complex_ao::apply_restore(&mut v, &restore);
                }
                v
            }
        }
    }

    /// gto value+grad+laplacian (5-tuple) with wrap+restore (restore is a uniform
    /// per-electron factor -- locally constant in r, so it multiplies val/grad/lap).
    #[allow(clippy::type_complexity)]
    fn gto_vgl(
        &self,
        r: &ArrayView2<f64>,
    ) -> (
        Array2<Complex64>,
        Array2<Complex64>,
        Array2<Complex64>,
        Array2<Complex64>,
        Array2<Complex64>,
    ) {
        let value_f32 = self.precision_policy.is_f32(self.aos.value_zone);
        match &self.wrap {
            None => crate::complex_ao::compute_aos_vgl_complex_with_policy(
                &self.aos,
                &self.image_phase,
                r,
                value_f32,
            ),
            Some(w) => {
                let (rw, restore) = w.wrap(r);
                let (mut v, mut gx, mut gy, mut gz, mut l) =
                    crate::complex_ao::compute_aos_vgl_complex_with_policy(
                        &self.aos,
                        &self.image_phase,
                        &rw.view(),
                        value_f32,
                    );
                if value_f32 {
                    crate::complex_ao::apply_restore_f32(&mut v, &restore);
                } else {
                    crate::complex_ao::apply_restore(&mut v, &restore);
                }
                for a in [&mut gx, &mut gy, &mut gz, &mut l] {
                    crate::complex_ao::apply_restore(a, &restore);
                }
                (v, gx, gy, gz, l)
            }
        }
    }

    /// Full complex AO block (gto (+) pw or gto (+) bspline in the composite
    /// ordering), pre-MO. The B-spline block carries its own Bloch phase.
    fn ao_full(&self, r: &ArrayView2<f64>) -> Array2<Complex64> {
        if let Some(tiled) = &self.tiled {
            return tiled.value(r);
        }
        let ao_f32 = self.precision_policy.is_f32("ao_eval");
        if let Some(bs) = &self.bspline {
            let bv = crate::timed!("corb.ao_bspline", bs.value_with_policy(r, ao_f32));
            if self.aos.num_ao == 0 {
                return bv;
            }
            let gto = crate::timed!("corb.ao_gto", self.gto_value(r));
            return if self.gto_first {
                vstack_c(&gto, &bv)
            } else {
                vstack_c(&bv, &gto)
            };
        }
        let gto = crate::timed!("corb.ao_gto", self.gto_value(r));
        match &self.pw {
            None => gto,
            Some(pw) => {
                let pwv = crate::timed!(
                    "corb.ao_pw",
                    crate::complex_ao::compute_pw_complex_with_policy(pw, r, ao_f32)
                );
                if self.gto_first {
                    vstack_c(&gto, &pwv)
                } else {
                    vstack_c(&pwv, &gto)
                }
            }
        }
    }

    /// Complex AO values `(num_ao, N)` (before the MO transform; gto(+)pw composite).
    pub fn ao(&self, r: &ArrayView2<f64>) -> Array2<Complex64> {
        self.ao_full(r)
    }

    /// Complex AO gradient `[dx,dy,dz]` at num_ao rank (before MO; gto wrap+restore).
    pub fn ao_grad(&self, r: &ArrayView2<f64>) -> [Array2<Complex64>; 3] {
        let (_, gx, gy, gz, _) = self.gto_vgl(r);
        [gx, gy, gz]
    }

    /// Complex AO `grad(grad^2AO)` `[dx,dy,dz]` at num_ao rank (sphe leaf; gto wrap+restore).
    pub fn ao_gradlap(&self, r: &ArrayView2<f64>) -> [Array2<Complex64>; 3] {
        match &self.wrap {
            None => crate::complex_ao::compute_aos_gradlap_complex(&self.aos, &self.image_phase, r),
            Some(w) => {
                let (rw, restore) = w.wrap(r);
                let mut g = crate::complex_ao::compute_aos_gradlap_complex(
                    &self.aos,
                    &self.image_phase,
                    &rw.view(),
                );
                for a in g.iter_mut() {
                    crate::complex_ao::apply_restore(a, &restore);
                }
                g
            }
        }
    }

    /// Complex AO Hessian `grad_agrad_b AO` (3x3) at num_ao rank (sphe leaf; wrap+restore).
    pub fn ao_hessian(&self, r: &ArrayView2<f64>) -> [[Array2<Complex64>; 3]; 3] {
        match &self.wrap {
            None => crate::complex_ao::compute_aos_hessian_complex(&self.aos, &self.image_phase, r),
            Some(w) => {
                let (rw, restore) = w.wrap(r);
                let mut h = crate::complex_ao::compute_aos_hessian_complex(
                    &self.aos,
                    &self.image_phase,
                    &rw.view(),
                );
                for row in h.iter_mut() {
                    for a in row.iter_mut() {
                        crate::complex_ao::apply_restore(a, &restore);
                    }
                }
                h
            }
        }
    }

    /// Complex orbital values `(num_orb, N)`.
    pub fn value(&self, r: &ArrayView2<f64>) -> Array2<Complex64> {
        let ao = self.ao(r);
        match &self.mo {
            Some(c) => crate::timed!("corb.mo_zgemm", c.dot(&ao)),
            None => ao,
        }
    }

    /// Complex orbital value + grad `[dx,dy,dz]` + Laplacian, each `(num_orb, N)`.
    #[allow(clippy::type_complexity)]
    pub fn value_grad_lap(
        &self,
        r: &ArrayView2<f64>,
    ) -> (Array2<Complex64>, [Array2<Complex64>; 3], Array2<Complex64>) {
        if let Some(tiled) = &self.tiled {
            let [v, gx, gy, gz, lap] = tiled.value_grad_lap(r);
            return (v, [gx, gy, gz], lap);
        }
        let cat = |a: &Array2<Complex64>, b: &Array2<Complex64>| {
            if self.gto_first {
                vstack_c(a, b)
            } else {
                vstack_c(b, a)
            }
        };
        let (v, gx, gy, gz, l) = if let Some(bs) = &self.bspline {
            let (bv, bg, bl) = bs.value_grad_lap(r);
            if self.aos.num_ao == 0 {
                (bv, bg[0].clone(), bg[1].clone(), bg[2].clone(), bl)
            } else {
                let (gv, ggx, ggy, ggz, gl) = self.gto_vgl(r);
                (
                    cat(&gv, &bv),
                    cat(&ggx, &bg[0]),
                    cat(&ggy, &bg[1]),
                    cat(&ggz, &bg[2]),
                    cat(&gl, &bl),
                )
            }
        } else {
            let (gv, ggx, ggy, ggz, gl) = self.gto_vgl(r);
            match &self.pw {
                None => (gv, ggx, ggy, ggz, gl),
                Some(pw) => {
                    let (pv, pg, pl) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
                    (
                        cat(&gv, &pv),
                        cat(&ggx, &pg[0]),
                        cat(&ggy, &pg[1]),
                        cat(&ggz, &pg[2]),
                        cat(&gl, &pl),
                    )
                }
            }
        };
        match &self.mo {
            Some(c) => (c.dot(&v), [c.dot(&gx), c.dot(&gy), c.dot(&gz)], c.dot(&l)),
            None => (v, [gx, gy, gz], l),
        }
    }

    /// Full pre-MO AO rank (gto block + plane-wave / B-spline child).
    pub(crate) fn num_ao_total(&self) -> usize {
        self.aos.num_ao + self.num_extra_ao()
    }

    /// Gate of the FAST ECP mesh AO-space contraction [`Self::mesh_wdot_into`]:
    /// a gto block evaluated in an fp64 value zone (`aos.value_zone`: the Bloch
    /// `ao_bloch_eval` zone is fp64 in both precision modes; an fp32 gto zone would
    /// round AO values the fp64 Bloch dot never forms), a non-empty AO block, a gto
    /// block that the screened Bloch dot covers (`atomic_orbital::bloch_mesh_dot_ok`),
    /// at most one extra child, and an MO layer (when present) of AO rank. An fp32
    /// `ao_eval` zone is covered: the plane-wave / B-spline child is then contracted
    /// from its zone-local fp32 values (see [`Self::mesh_wdot_into`]). Decline keeps
    /// the orbital-rank candidate table (`complex_candidate_batch`).
    pub(crate) fn mesh_wdot_ok(&self) -> bool {
        (self.aos.num_ao == 0 || !self.precision_policy.is_f32(self.aos.value_zone))
            && self.num_ao_total() > 0
            && !(self.pw.is_some() && self.bspline.is_some())
            && (self.aos.num_ao == 0
                || (crate::atomic_orbital::bloch_mesh_dot_ok(&self.aos)
                    && self.image_phase.len() == self.aos.num_ao_kernel))
            && self
                .mo
                .as_ref()
                .is_none_or(|c| c.ncols() == self.num_ao_total())
    }

    /// FAST ECP mesh AO-space contraction: `out[v] = Sum_mu w_mu AO_mu(pts[v])` over
    /// the full pre-MO complex AO block (gto (+) pw / B-spline in `gto_first` order),
    /// `w = w_re + i w_im` of length [`Self::num_ao_total`]. The orbital value is
    /// `C AO`, so with `w = q C` this equals `q . value(pts[v])` -- the candidate
    /// table's affine det-ratio term -- without forming the AO or MO mesh. Same
    /// conventions as [`Self::ao`]: the gto block is evaluated at the home-cell
    /// wrapped point times the wrap-restore phase (`WrapCtx`), as the screened
    /// Bloch dot `atomic_orbital::dot_aos_multi_cutoff_bloch` (FAST value screens,
    /// absolute bound 1e-12 per dropped AO value); the plane waves and the B-spline
    /// child at the RAW point. Twist plane waves use the same vectorized `sincos`
    /// as `compute_pw_complex_with_policy`; the Gamma real cos/sin child the
    /// scalar `cos` of its value kernel. Under an fp32 `ao_eval` zone the child is
    /// contracted from its zone-local fp32 values instead (the batch
    /// `compute_pw_complex_with_policy(.., true)` / B-spline `value_with_policy(.., true)`
    /// of the candidate route's [`Self::ao`]), in fp64. fp-reassociated relative to
    /// `q . value(pts)`: FAST route only, caller-gated by [`Self::mesh_wdot_ok`].
    pub(crate) fn mesh_wdot_into(
        &self,
        w_re: &[f64],
        w_im: &[f64],
        pts: &[[f64; 3]],
        scratch: &mut MeshWdotScratch,
        out: &mut [Complex64],
    ) {
        let nv = pts.len();
        debug_assert_eq!(w_re.len(), self.num_ao_total());
        debug_assert_eq!(w_im.len(), self.num_ao_total());
        debug_assert_eq!(out.len(), nv);
        let n_gto = self.aos.num_ao;
        let n_extra = self.num_extra_ao();
        let (gto_off, extra_off) = if self.gto_first {
            (0, n_gto)
        } else {
            (n_extra, 0)
        };
        let zero = Complex64::new(0.0, 0.0);
        out.fill(zero);
        if n_gto > 0 {
            let sc = &mut *scratch;
            sc.pts.clear();
            sc.restore.clear();
            match &self.wrap {
                None => sc.pts.extend_from_slice(pts),
                Some(w) => {
                    for &p in pts {
                        let (rw, ph) = w.wrap_point(p);
                        sc.pts.push(rw);
                        sc.restore.push(ph);
                    }
                }
            }
            sc.g_re.resize(nv, 0.0);
            sc.g_im.resize(nv, 0.0);
            crate::atomic_orbital::dot_aos_multi_cutoff_bloch(
                &self.aos,
                &sc.pts,
                &w_re[gto_off..gto_off + n_gto],
                &w_im[gto_off..gto_off + n_gto],
                &self.image_phase,
                &mut sc.g_re[..nv],
                &mut sc.g_im[..nv],
            );
            for (v, out_v) in out[..nv].iter_mut().enumerate() {
                let g = Complex64::new(sc.g_re[v], sc.g_im[v]);
                *out_v = if self.wrap.is_some() {
                    g * sc.restore[v]
                } else {
                    g
                };
            }
        }
        self.extra_wdot_add(
            &w_re[extra_off..extra_off + n_extra],
            &w_im[extra_off..extra_off + n_extra],
            pts,
            scratch,
            out,
        );
    }

    /// Plane-wave / B-spline child part of [`Self::mesh_wdot_into`], added to `out`.
    fn extra_wdot_add(
        &self,
        w_re: &[f64],
        w_im: &[f64],
        pts: &[[f64; 3]],
        sc: &mut MeshWdotScratch,
        out: &mut [Complex64],
    ) {
        if self.precision_policy.is_f32("ao_eval") && (self.pw.is_some() || self.bspline.is_some())
        {
            // fp32 `ao_eval` zone: the child's zone-local fp32 values, exactly as the
            // candidate route's `ao_full` evaluates them, contracted with `w` in fp64.
            let r = Array2::from_shape_fn((pts.len(), 3), |(v, c)| pts[v][c]);
            let vals = if let Some(pw) = &self.pw {
                crate::complex_ao::compute_pw_complex_with_policy(pw, &r.view(), true)
            } else if let Some(bs) = &self.bspline {
                bs.value_with_policy(&r.view(), true)
            } else {
                return;
            };
            for (v, o) in out.iter_mut().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..vals.nrows() {
                    acc += Complex64::new(w_re[k], w_im[k]) * vals[[k, v]];
                }
                *o += acc;
            }
            return;
        }
        if let Some(pw) = &self.pw {
            let n = pw.num_ao;
            sc.th.resize(n, 0.0);
            sc.sn.resize(n, 0.0);
            sc.cs.resize(n, 0.0);
            for (v, p) in pts.iter().enumerate() {
                for k in 0..n {
                    let g = pw.gk[k];
                    sc.th[k] = g[0] * p[0] + g[1] * p[1] + g[2] * p[2];
                }
                let (mut acc_re, mut acc_im) = (0.0_f64, 0.0_f64);
                match &pw.phase_is_sin {
                    None => {
                        crate::atomic_orbital::sincos_pd_into(
                            &sc.th[..n],
                            &mut sc.sn[..n],
                            &mut sc.cs[..n],
                        );
                        for k in 0..n {
                            acc_re += w_re[k] * sc.cs[k] - w_im[k] * sc.sn[k];
                            acc_im += w_re[k] * sc.sn[k] + w_im[k] * sc.cs[k];
                        }
                    }
                    Some(s) => {
                        for k in 0..n {
                            let c = (sc.th[k] - s[k] * std::f64::consts::FRAC_PI_2).cos();
                            acc_re += w_re[k] * c;
                            acc_im += w_im[k] * c;
                        }
                    }
                }
                out[v] += Complex64::new(acc_re, acc_im);
            }
        } else if let Some(bs) = &self.bspline {
            sc.b_re.resize(bs.num_ao, 0.0);
            sc.b_im.resize(bs.num_ao, 0.0);
            for (v, &p) in pts.iter().enumerate() {
                out[v] += bs.wdot_single(p, w_re, w_im, &mut sc.b_re, &mut sc.b_im);
            }
        }
    }

    /// Orbital rank: MO rows, or the full (gto + extra child) AO count without an MO layer.
    pub(crate) fn num_orb(&self) -> usize {
        if let Some(tiled) = &self.tiled {
            return tiled.num_orb();
        }
        match &self.mo {
            Some(c) => c.nrows(),
            None => self.aos.num_ao + self.num_extra_ao(),
        }
    }

    /// Reject a J3 `j_matrix` whose shape is not `(n_orb, n_orb + 1)` for this orbital
    /// with a Python error, before any kernel slices it (a mismatch would otherwise
    /// abort the process with an ndarray panic).
    pub(crate) fn check_j3_matrix_shape(&self, nrows: usize, ncols: usize) -> PyResult<()> {
        let n = self.num_orb();
        if nrows != n || ncols != n + 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "J3 j_matrix shape ({nrows}, {ncols}) does not match the J3 orbital rank {n}; expected ({n}, {})",
                n + 1
            )));
        }
        Ok(())
    }

    /// AO count of the non-gto child (plane-wave or B-spline), 0 for a single leaf.
    pub(crate) fn num_extra_ao(&self) -> usize {
        self.pw
            .as_ref()
            .map(|p| p.num_ao)
            .or_else(|| self.bspline.as_ref().map(|b| b.num_ao))
            .unwrap_or(0)
    }

    /// Row offset of the gto block in the (pre-MO) AO ordering.
    pub(crate) fn gto_ao_offset(&self) -> usize {
        if self.gto_first {
            0
        } else {
            self.num_extra_ao()
        }
    }

    /// True when the J3 kernels must work at ORBITAL rank (`value_grad_lap`, MO
    /// applied before `Re`) and MO-project their real cotangents back to AO rank
    /// (`mo_project_t`), instead of the AO-rank shortcut that equates orbital and
    /// AO rows. The MO criterion is the one the forward orbital evaluators
    /// (`value` / `value_grad_lap`) use: an MO layer is present. The shortcut is
    /// valid only for a bare (MO-free) gto / gto(+)PW basis; deciding by the MO
    /// SHAPE would send a square non-identity MO to it and silently drop `C`
    /// (issue #119). An identity MO takes the orbital-rank path too, where `C = I`
    /// reproduces the AO-rank fields. A B-spline child's rows are never carried by
    /// the AO-rank shortcut either (issues #88/#89).
    pub(crate) fn j3_orbital_rank(&self) -> bool {
        self.bspline.is_some() || self.mo.is_some()
    }

    /// `C^T bar` (AO rank, complex) for a real orbital-rank cotangent `bar`; `bar`
    /// itself when there is no MO layer.
    pub(crate) fn mo_project_t(&self, bar: &Array2<f64>) -> Array2<Complex64> {
        let barc = bar.mapv(|x| Complex64::new(x, 0.0));
        match &self.mo {
            Some(c) => c.t().dot(&barc),
            None => barc,
        }
    }
}

/// Complex geminal shared by dense-lambda AGP/AGPn and occupied-orbital SD.
pub struct ComplexGeminal {
    pub orb_up: ComplexOrb,
    pub orb_dn: ComplexOrb,
    pub lambda: Option<Array2<f64>>,
    /// FULL complex `lambda_matrix` (Re + Im) for AGP/AGPn. Canonical SD carries
    /// neither this field nor `lambda`; its occupied-orbital selector is implicit.
    /// Dense-lambda consumers preserve the imaginary part until the final real
    /// projection.
    pub lambda_c: Option<Array2<Complex64>>,
    pub orb_num_up: usize,
    pub orb_num_dn: usize,
}

impl ComplexGeminal {
    pub fn from_py(geminal_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = geminal_data.py();
        let np = py.import("numpy")?;
        let orb_up = ComplexOrb::from_py(&geminal_data.getattr("orb_data_up_spin")?)?;
        let time_reversal: bool = geminal_data.getattr("time_reversal")?.extract()?;
        let orb_dn_py = geminal_data.getattr("orb_data_dn_spin")?;
        let orb_dn = if time_reversal {
            ComplexOrb::from_py_with_tr_source(&orb_dn_py, Some(&orb_up))?
        } else {
            ComplexOrb::from_py(&orb_dn_py)?
        };
        // lambda_matrix arrives as complex128 (zero imaginary part) at a twist: the
        // fully-complex SR/LM convention promotes its dtype even though the pairing
        // coefficients stay real for these WFs. ComplexGeminal stores lambda as real
        // (geminal_w casts it to complex downstream), so read the real part -- np.real is
        // a no-op on an already-real f64 lambda (molecular / real-Gamma AGP) and drops the
        // zero imaginary part on the promoted twist lambda.
        let (lambda, lambda_c) = if geminal_data.hasattr("lambda_matrix")? {
            let lam_arr = np.call_method1("asarray", (geminal_data.getattr("lambda_matrix")?,))?;
            let lam: numpy::PyReadonlyArray2<f64> =
                np.call_method1("real", (&lam_arr,))?.extract()?;
            // FULL complex lambda (Re + Im). `astype(complex128)` is a no-op on the already-
            // complex twist lambda and promotes a real molecular lambda with a zero Im part.
            let lam_c: numpy::PyReadonlyArray2<numpy::Complex64> = lam_arr
                .call_method1("astype", (np.getattr("complex128")?,))?
                .extract()?;
            (
                Some(lam.as_array().to_owned()),
                Some(lam_c.as_array().to_owned()),
            )
        } else {
            (None, None)
        };
        let orb_num_up: usize = geminal_data.getattr("orb_num_up")?.extract()?;
        let orb_num_dn: usize = geminal_data.getattr("orb_num_dn")?.extract()?;
        Ok(ComplexGeminal {
            orb_up,
            orb_dn,
            lambda,
            lambda_c,
            orb_num_up,
            orb_num_dn,
        })
    }

    /// Number of unique grid-sized B-spline coefficient allocations owned by the spin
    /// pair. A time-reversal pair has two (re/im); independent spins have four.
    pub(crate) fn bspline_coefficient_storage_count(&self) -> usize {
        let mut ptrs = std::collections::HashSet::<usize>::new();
        for orb in [&self.orb_up, &self.orb_dn] {
            if let Some(bspline) = orb.bspline.as_ref() {
                ptrs.extend(bspline.coefficient_storage_ptrs());
            }
        }
        ptrs.len()
    }

    #[inline]
    pub fn is_sd(&self) -> bool {
        self.lambda_c.is_none()
    }

    #[inline]
    pub fn orb_num_up(&self) -> usize {
        self.orb_num_up
    }

    #[inline]
    pub fn n_unpaired(&self) -> usize {
        self.lambda_c
            .as_ref()
            .map_or(self.orb_num_up() - self.orb_num_dn, |l| {
                l.ncols() - self.orb_num_dn
            })
    }

    #[inline]
    pub fn lambda_c_value(&self, row: usize, col: usize) -> Complex64 {
        self.lambda_c.as_ref().map_or_else(
            || Complex64::new(if row == col { 1.0 } else { 0.0 }, 0.0),
            |l| l[[row, col]],
        )
    }

    pub fn paired_times(&self, od: &Array2<Complex64>) -> Array2<Complex64> {
        if let Some(lam) = &self.lambda_c {
            return lam
                .slice(numpy::ndarray::s![.., 0..self.orb_num_dn])
                .dot(od);
        }
        let mut out = Array2::<Complex64>::zeros((self.orb_num_up(), od.ncols()));
        out.slice_mut(numpy::ndarray::s![0..self.orb_num_dn, ..])
            .assign(od);
        out
    }

    pub fn paired_t_times(&self, x: &Array2<Complex64>) -> Array2<Complex64> {
        if let Some(lam) = &self.lambda_c {
            return lam
                .slice(numpy::ndarray::s![.., 0..self.orb_num_dn])
                .t()
                .dot(x);
        }
        x.slice(numpy::ndarray::s![0..self.orb_num_dn, ..])
            .to_owned()
    }

    pub fn up_t_times_paired(&self, ou: &Array2<Complex64>) -> Array2<Complex64> {
        if let Some(lam) = &self.lambda_c {
            return ou
                .t()
                .dot(&lam.slice(numpy::ndarray::s![.., 0..self.orb_num_dn]));
        }
        ou.slice(numpy::ndarray::s![0..self.orb_num_dn, ..])
            .t()
            .as_standard_layout()
            .to_owned()
    }

    pub fn w_from_dn(&self, od: &Array2<Complex64>, n_up: usize) -> Array2<Complex64> {
        let mut w = Array2::<Complex64>::zeros((self.orb_num_up(), n_up));
        w.slice_mut(numpy::ndarray::s![.., 0..od.ncols()])
            .assign(&self.paired_times(od));
        for a in 0..self.orb_num_up() {
            for j in 0..self.n_unpaired() {
                w[[a, od.ncols() + j]] = self.lambda_c_value(a, self.orb_num_dn + j);
            }
        }
        w
    }
}

/// `W = [lambda_p*od | lambda_unp]` (M_up, N_up). `lambda` is the FULL complex
/// `lambda_matrix` (callers pass `gem.lambda_c`): a real-pairing twist has `Im == 0` so
/// this is byte-identical to a `Re(lambda)+0i` cast, while a genuine complex-`lambda`
/// AGP twist threads its imaginary part into `W` (and thus into `G`, `G^-1`, the drift
/// and the Pulay/stress/kinetic chains that ride it).
pub fn geminal_w(
    od: &Array2<Complex64>,
    lambda: &Array2<Complex64>,
    orb_num_dn: usize,
    n_up: usize,
) -> Array2<Complex64> {
    let m_up = lambda.nrows();
    let n_dn = od.ncols();
    let lam_c = lambda;
    let lam_p = lam_c.slice(numpy::ndarray::s![.., 0..orb_num_dn]);
    let lp_od = lam_p.dot(od); // (M_up, N_dn)
    let mut w = Array2::<Complex64>::zeros((m_up, n_up));
    w.slice_mut(numpy::ndarray::s![.., 0..n_dn]).assign(&lp_od);
    w.slice_mut(numpy::ndarray::s![.., n_dn..])
        .assign(&lam_c.slice(numpy::ndarray::s![.., orb_num_dn..]));
    w
}

/// Complex geminal matrix `G = ouT*W` (N_up, N_up).
pub fn compute_geminal_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<Complex64> {
    let ou = gem.orb_up.value(r_up);
    let od = gem.orb_dn.value(r_dn);
    let w = gem.w_from_dn(&od, ou.ncols());
    ou.t().dot(&w)
}

/// `ln|det A|` (real) of a complex matrix via LU with partial pivoting.
pub fn ln_abs_det(a_in: &Array2<Complex64>) -> f64 {
    let n = a_in.nrows();
    let mut a = a_in.to_owned();
    let mut ln = 0.0_f64;
    for k in 0..n {
        // partial pivot on |a[i,k]|
        let mut piv = k;
        let mut best = a[[k, k]].norm();
        for i in (k + 1)..n {
            let v = a[[i, k]].norm();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if piv != k {
            for j in 0..n {
                let t = a[[k, j]];
                a[[k, j]] = a[[piv, j]];
                a[[piv, j]] = t;
            }
        }
        let d = a[[k, k]];
        ln += d.norm().ln();
        for i in (k + 1)..n {
            let f = crate::complex_div::div_range_safe(a[[i, k]], d);
            for j in (k + 1)..n {
                let akj = a[[k, j]];
                a[[i, j]] -= f * akj;
            }
        }
    }
    ln
}

/// `ln|det G|` (real, the |Psi|^2 amplitude) for the complex Bloch geminal.
pub fn compute_ln_det_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    ln_abs_det(&compute_geminal_complex(gem, r_up, r_dn))
}

/// Complex Pulay force `dln|det|/dR` (real, `(n_atom,3)`) for the twisted geminal.
/// Translation id `dAO/dR_I=-gradAO` (Bloch phase preserved: all images shift with R_I)
/// -> `-Re(Sum_{a:nuc=I} cm_orb[a,k]*gradAO[a,k])`, cm_orb = MOT*bar_orb (AO-rank det cotangent).
/// Complex determinant Pulay force `dln det/dR` (complex n_atomx3, before taking
/// the real part) and the ECP non-local ratio force. `nuc_*` = home-AO atom maps.
/// Slice a full-AO-rank cotangent `cm` to the GTO (image-tiled) rows of a
/// composite pgto(+)pw orb (the PW has no atom force / value Pulay). No-op for a
/// pure-pgto orb. `ao_grad`/`ao_gradlap`/`ao_hessian` are GTO-only, so cm (which
/// spans gto+pw after `moT*bar`) must be trimmed to match.
pub(crate) fn gto_slice_cm(orb: &ComplexOrb, cm: Array2<Complex64>) -> Array2<Complex64> {
    // Keep only the atom-centered GTO rows (the extra PW / B-spline block carries no
    // nuclear force -- dAO/dR = 0). Pure PW / pure B-spline => 0 GTO rows.
    let num_extra = orb
        .pw
        .as_ref()
        .map(|p| p.num_ao)
        .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
        .unwrap_or(0);
    if num_extra == 0 {
        return cm;
    }
    let g0 = if orb.gto_first { 0 } else { num_extra };
    cm.slice(numpy::ndarray::s![g0..g0 + (cm.nrows() - num_extra), ..])
        .to_owned()
}

pub fn pbc_det_pulay_force_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
    gi: Option<&ArrayView2<Complex64>>,
) -> Array2<Complex64> {
    let f = complex_gem_forward(gem, r_up, r_dn, gi);
    let bar_orb_up = f.wg.clone();
    let bar_orb_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
    let mut out = Array2::<Complex64>::zeros((n_atom, 3));
    let side = |orb: &ComplexOrb,
                r: &ArrayView2<f64>,
                bar_orb: &Array2<Complex64>,
                nuc: &[usize],
                out: &mut Array2<Complex64>| {
        if let Some(tiled) = &orb.tiled {
            *out += &tiled.field_vjp(
                r,
                bar_orb,
                None,
                None,
                crate::orbital_tiling::TiledCoordinate::Atoms,
            );
            return;
        }
        let cm = gto_slice_cm(
            orb,
            match &orb.mo {
                Some(c) => c.t().dot(bar_orb),
                None => bar_orb.clone(),
            },
        );
        let g = orb.ao_grad(r);
        let num_ao = cm.nrows();
        let n = cm.ncols();
        for a in 0..num_ao {
            let ia = nuc[a];
            for c in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    acc += cm[[a, k]] * g[c][[a, k]];
                }
                out[[ia, c]] -= acc;
            }
        }
    };
    side(&gem.orb_up, r_up, &bar_orb_up, nuc_up, &mut out);
    side(&gem.orb_dn, r_dn, &bar_orb_dn, nuc_dn, &mut out);
    out
}

pub fn grad_lndet_wrt_r_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
) -> Array2<f64> {
    pbc_det_pulay_force_complex(gem, r_up, r_dn, nuc_up, nuc_dn, n_atom, None).mapv(|x| x.re)
}

/// Complex kinetic force `dT/dR` (real, `(n_atom,3)`) for the twisted geminal, jnone.
/// `dT/dR_I = -Re Sum_{a:home=I,k} [cm_orb[a,k]*gradAO[a,k] + cm_lap[a,k]*gradgrad^2AO[a,k]]`
/// (translation id; bar_gou=0 for jnone). cm = MOT*bar from `complex_geminal_reverse`.
pub fn grad_t_wrt_r_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
) -> Array2<f64> {
    let zu = Array2::<f64>::zeros((r_up.nrows(), 3));
    let zd = Array2::<f64>::zeros((r_dn.nrows(), 3));
    let b = complex_geminal_reverse(gem, r_up, r_dn, &zu, &zd, None);
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    let side = |orb: &ComplexOrb,
                r: &ArrayView2<f64>,
                bar_orb: &Array2<Complex64>,
                bar_lap: &Array2<Complex64>,
                nuc: &[usize],
                out: &mut Array2<f64>| {
        if let Some(tiled) = &orb.tiled {
            *out += &tiled
                .field_vjp(
                    r,
                    bar_orb,
                    None,
                    Some(bar_lap),
                    crate::orbital_tiling::TiledCoordinate::Atoms,
                )
                .mapv(|v| v.re);
            return;
        }
        let cm_orb = gto_slice_cm(
            orb,
            match &orb.mo {
                Some(c) => c.t().dot(bar_orb),
                None => bar_orb.clone(),
            },
        );
        let cm_lap = gto_slice_cm(
            orb,
            match &orb.mo {
                Some(c) => c.t().dot(bar_lap),
                None => bar_lap.clone(),
            },
        );
        let g = orb.ao_grad(r); // [3] (num_ao,N)
        let gl = orb.ao_gradlap(r); // [3] (num_ao,N)
        let num_ao = cm_orb.nrows();
        let n = cm_orb.ncols();
        for a in 0..num_ao {
            let ia = nuc[a];
            for c in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    acc += cm_orb[[a, k]] * g[c][[a, k]] + cm_lap[[a, k]] * gl[c][[a, k]];
                }
                out[[ia, c]] -= acc.re;
            }
        }
    };
    side(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        nuc_up,
        &mut out,
    );
    side(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        nuc_dn,
        &mut out,
    );
    out
}

/// Complex kinetic force dT_withJ/dR (twist + Jastrow). Det side: bar_orb*gradAO +
/// bar_lap*gradgrad^2AO + **bar_gou*gradgradAO** (the gou cross term -- needs the sphe AO Hessian;
/// for jnone Ne1-1 it cancels, but with Jastrow gD*gJ does NOT cancel). Jastrow side:
/// since e_L=Re(*) and J1/J3 derivs are real, only Re(bar_gj)=-(gJ+Re gD) matters, so
/// the REAL `j1_side`/`j3_side` are reused. Mirrors real `forces::grad_t_wrt_r`.
#[allow(clippy::too_many_arguments)]
pub fn grad_t_wrt_r_complex_withj(
    gem: &ComplexGeminal,
    jastrow: Option<&crate::data::JastrowData>,
    j1: Option<&crate::data::JastrowOneBodyData>,
    j3_orb: Option<&crate::determinant::OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
    j3c: Option<&ComplexOrb>,
) -> Array2<f64> {
    let _ = j3_orb; // J3 drift uses the complex-sphe path (j3c), not the real cart one
    let (mut gj_up, mut gj_dn) =
        crate::param_grad_el::jastrow_grads(jastrow, None, None, r_up, r_dn);
    if let (Some(j3o), Some(jm)) = (j3c, j3_jm) {
        let (g3u, g3d, _, _) = j3_grads_lap_complex(j3o, jm, r_up, r_dn);
        gj_up += &g3u;
        gj_dn += &g3d;
    }
    let b = complex_geminal_reverse(gem, r_up, r_dn, &gj_up, &gj_dn, None);
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    // Det orbital side: bar_orb*gradAO + bar_lap*gradgrad^2AO + bar_gou*gradgradAO, scatter Re.
    let side = |orb: &ComplexOrb,
                r: &ArrayView2<f64>,
                bar_orb: &Array2<Complex64>,
                bar_lap: &Array2<Complex64>,
                bar_gou: &[Array2<Complex64>; 3],
                nuc: &[usize],
                out: &mut Array2<f64>| {
        if let Some(tiled) = &orb.tiled {
            *out += &tiled
                .field_vjp(
                    r,
                    bar_orb,
                    Some(bar_gou),
                    Some(bar_lap),
                    crate::orbital_tiling::TiledCoordinate::Atoms,
                )
                .mapv(|v| v.re);
            return;
        }
        let cm_orb0 = match &orb.mo {
            Some(c) => c.t().dot(bar_orb),
            None => bar_orb.clone(),
        };
        let cm_lap0 = match &orb.mo {
            Some(c) => c.t().dot(bar_lap),
            None => bar_lap.clone(),
        };
        let cm_gou0: Vec<Array2<Complex64>> = bar_gou
            .iter()
            .map(|bg| match &orb.mo {
                Some(c) => c.t().dot(bg),
                None => bg.clone(),
            })
            .collect();
        // Composite pgto(+)pw: cm spans gto (image-tiled) + pw; ao_grad/gradlap/
        // hessian are GTO-only, and the PW has no atom force (R-independent) ->
        // slice cm to the GTO rows.
        let (gto0, num_gto) = match &orb.pw {
            Some(pw) => (
                if orb.gto_first { 0 } else { pw.num_ao },
                cm_orb0.nrows() - pw.num_ao,
            ),
            None => (0, cm_orb0.nrows()),
        };
        let gsl = |cm: &Array2<Complex64>| {
            cm.slice(numpy::ndarray::s![gto0..gto0 + num_gto, ..])
                .to_owned()
        };
        let cm_orb = gsl(&cm_orb0);
        let cm_lap = gsl(&cm_lap0);
        let cm_gou: Vec<Array2<Complex64>> = cm_gou0.iter().map(gsl).collect();
        let g = orb.ao_grad(r);
        let gl = orb.ao_gradlap(r);
        let h = orb.ao_hessian(r);
        let num_ao = cm_orb.nrows();
        let n = cm_orb.ncols();
        for a in 0..num_ao {
            let ia = nuc[a];
            for c in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    acc += cm_orb[[a, k]] * g[c][[a, k]] + cm_lap[[a, k]] * gl[c][[a, k]];
                    for e in 0..3 {
                        acc += cm_gou[e][[a, k]] * h[c][e][[a, k]];
                    }
                }
                out[[ia, c]] -= acc.re;
            }
        }
    };
    side(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        &b.bar_gou,
        nuc_up,
        &mut out,
    );
    side(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        &b.bar_godn,
        nuc_dn,
        &mut out,
    );
    // Jastrow side: bar_gj = -(gJ + Re gD) (J1/J3 derivs real -> only Re(gD) survives).
    if jastrow.is_some() {
        let (gd_up, gd_dn, _, _) = grads_and_laplacian_ln_det_complex(gem, r_up, r_dn);
        let n_up = r_up.nrows();
        let n_e = n_up + r_dn.nrows();
        let bar_gj = |e: usize, d: usize| -> f64 {
            if e < n_up {
                -(gj_up[[e, d]] + gd_up[[e, d]].re)
            } else {
                let l = e - n_up;
                -(gj_dn[[l, d]] + gd_dn[[l, d]].re)
            }
        };
        // PBC twist: jqmc's J1 uses the **Williamson smooth periodic distance** -> use
        // j1_side_pbc (Williamson geometry). J3 = the periodic cart-AO three-body force.
        if let Some(j1) = j1 {
            match j1.structure_data.cell.as_ref() {
                Some(cell) => {
                    let ci = crate::jastrow_factor::CellInfo::new(&cell.view());
                    crate::forces::j1_side_pbc(
                        j1, &ci, r_up, r_dn, &bar_gj, n_e, n_up, None, &mut out,
                    );
                }
                None => crate::forces::j1_side(j1, r_up, r_dn, &bar_gj, None, &mut out),
            }
        }
        // J3 (complex Bloch sphe): the complex-sphe force path (Re of complex AO derivs).
        if let (Some(j3o), Some(jm), Some(nuc)) = (j3c, j3_jm, nuc_j3) {
            j3_side_complex(j3o, jm, r_up, r_dn, &bar_gj, nuc, None, &mut out);
        }
        // DTN kinetic ATOM force (u+chi+f); cot = gJ+Re(gD) = -bar_gj (the real
        // gfmc path uses +cot with grad_t_dtn_wrt_r_atom).
        if let Some(d) = jastrow.and_then(|j| j.dtn.as_ref()) {
            let mut cot = Array2::<f64>::zeros((n_e, 3));
            for e in 0..n_e {
                for c in 0..3 {
                    cot[[e, c]] = -bar_gj(e, c);
                }
            }
            out += &crate::forces_dtn::grad_t_dtn_wrt_r_atom(d, &cot, r_up, r_dn);
        }
    }
    out
}

/// Complex geminal forward (value+grad+lap orbitals + G/G^-^1/W/Q/WG/GQ), mirror of
/// `param_grad_el::gem_forward`, used by the complex kinetic VJP.
pub struct ComplexGemFwd {
    pub ou: Array2<Complex64>,
    pub od: Array2<Complex64>,
    pub lap_up: Array2<Complex64>,
    pub lap_dn: Array2<Complex64>,
    pub gou: [Array2<Complex64>; 3],
    pub godn: [Array2<Complex64>; 3],
    pub w: Array2<Complex64>,
    pub ginv: Array2<Complex64>,
    pub q: Array2<Complex64>,
    pub wg: Array2<Complex64>,
    pub gq: Array2<Complex64>,
    pub lam_p: Option<Array2<Complex64>>,
    pub n_dn: usize,
    pub m_dn: usize,
}

/// `gi = Some(G^-1)` REUSES a caller-supplied complex geminal inverse (the MCMC sweep's
/// running Sherman-Morrison inverse) instead of a fresh Gauss-Jordan `inverse_complex`.
/// This is the complex sibling of the real `gem_forward_gi`: a fresh per-config inverse
/// NaNs at a near-singular geminal, while the AS-regularized swept inverse is stable, so
/// every BASE-config complex derivative reuses it (mesh/candidate configs pass `None`).
pub fn complex_gem_forward(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> ComplexGemFwd {
    // FULL complex lambda (real-pairing twist: Im == 0 -> byte-identical to a
    // Re-only cast; complex-`lambda` AGP twist: threads Im into the kinetic VJP forward).
    if let Some(lam) = &gem.lambda_c {
        return complex_gem_forward_clam(gem, &lam.view(), r_up, r_dn, gi);
    }
    let (ou, gou, lap_up) = gem.orb_up.value_grad_lap(r_up);
    let (od, godn, lap_dn) = gem.orb_dn.value_grad_lap(r_dn);
    let n_dn = r_dn.nrows();
    let n_up = ou.ncols();
    let w = gem.w_from_dn(&od, n_up);
    let ginv = match gi {
        Some(v) => v.to_owned(),
        None => inverse_complex(&ou.t().dot(&w)),
    };
    let q = gem.up_t_times_paired(&ou);
    let wg = w.dot(&ginv);
    let gq = ginv.dot(&q);
    ComplexGemFwd {
        ou,
        od,
        lap_up,
        lap_dn,
        gou,
        godn,
        w,
        ginv,
        q,
        wg,
        gq,
        lam_p: None,
        n_dn,
        m_dn: gem.orb_num_dn,
    }
}

/// As [`complex_gem_forward`] but with an EXPLICIT complex `lambda`. A genuine
/// complex-twist `AGP_data` carries a `lambda_matrix` with a nonzero imaginary part,
/// which `ComplexGeminal::from_py` stores Re-only (right for the SD / real-pairing
/// twists, where Re is the identity). Any HOLOMORPHIC c-axis on such a geminal must
/// use the full complex lambda -- same seam as
/// [`grad_ln_det_lambda_complex_holo_clam`]. `gi` is the same optional reused
/// Sherman-Morrison inverse as in [`complex_gem_forward`].
pub fn complex_gem_forward_clam(
    gem: &ComplexGeminal,
    lam: &ArrayView2<Complex64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> ComplexGemFwd {
    let (ou, gou, lap_up) = gem.orb_up.value_grad_lap(r_up);
    let (od, godn, lap_dn) = gem.orb_dn.value_grad_lap(r_dn);
    let m_dn = gem.orb_num_dn;
    let n_dn = r_dn.nrows();
    let n_up = ou.ncols();
    let lam = lam.to_owned();
    let lam_p = lam.slice(numpy::ndarray::s![.., 0..m_dn]).to_owned();
    let lam_unp = lam.slice(numpy::ndarray::s![.., m_dn..]).to_owned();
    let lp_od = lam_p.dot(&od);
    let mut w = Array2::<Complex64>::zeros((ou.nrows(), n_up));
    w.slice_mut(numpy::ndarray::s![.., 0..n_dn]).assign(&lp_od);
    w.slice_mut(numpy::ndarray::s![.., n_dn..]).assign(&lam_unp);
    let g = ou.t().dot(&w);
    let ginv = match gi {
        Some(v) => v.to_owned(),
        None => inverse_complex(&g),
    };
    let q = ou.t().dot(&lam_p);
    let wg = w.dot(&ginv);
    let gq = ginv.dot(&q);
    ComplexGemFwd {
        ou,
        od,
        lap_up,
        lap_dn,
        gou,
        godn,
        w,
        ginv,
        q,
        wg,
        gq,
        lam_p: Some(lam_p),
        n_dn,
        m_dn,
    }
}

/// Complex kinetic VJP cotangents (mirror of `geminal_reverse`). `gj_*` = Jastrow
/// drift (real; zero for jnone). Returns bar_orb/lap/gou + bar_lambda.
pub struct GemReverseC {
    pub bar_orb_up: Array2<Complex64>,
    pub bar_orb_dn: Array2<Complex64>,
    pub bar_lap_up: Array2<Complex64>,
    pub bar_lap_dn: Array2<Complex64>,
    pub bar_gou: [Array2<Complex64>; 3],
    pub bar_godn: [Array2<Complex64>; 3],
    pub bar_lam_p: Array2<Complex64>,
    pub bar_lam_unp: Array2<Complex64>,
}

pub fn complex_geminal_reverse(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> GemReverseC {
    // FULL complex lambda (see [`complex_gem_forward`]): the kinetic VJP reverse -- which
    // feeds the c-axis `dT/dlambda`, the r/R Pulay reverse and the strain reduction --
    // must ride the complex lambda so `Im(lambda)` reaches `dE_L/dc` and the drift.
    if let Some(lam) = &gem.lambda_c {
        complex_geminal_reverse_clam(gem, &lam.view(), r_up, r_dn, gj_up, gj_dn, gi)
    } else {
        complex_geminal_reverse_w(gem, r_up, r_dn, gj_up, gj_dn, None, None, gi)
    }
}

/// As [`complex_geminal_reverse`] but with an EXPLICIT complex `lambda` (see
/// [`complex_gem_forward_clam`]): required for the holomorphic c-axis on a genuine
/// complex-lambda `AGP_data` twist. The reverse itself reads lambda only through the
/// forward's `lam_p`, so this is a pure pass-through. `gi` is the same optional reused
/// Sherman-Morrison inverse as in [`complex_geminal_reverse`].
pub fn complex_geminal_reverse_clam(
    gem: &ComplexGeminal,
    lam: &ArrayView2<Complex64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> GemReverseC {
    let f = complex_gem_forward_clam(gem, lam, r_up, r_dn, gi);
    let n_up = f.ou.ncols();
    let m_up = f.ou.nrows();
    let (n_dn, m_dn) = (f.n_dn, f.m_dn);
    let bar_lap_up = f.wg.mapv(|x| x * -0.5);
    let mut bar_lap_dn = Array2::<Complex64>::zeros((m_dn, n_dn));
    for l in 0..n_dn {
        for b in 0..m_dn {
            bar_lap_dn[[b, l]] = f.gq[[l, b]] * -0.5;
        }
    }
    let mut bar_wg = Array2::<Complex64>::zeros((m_up, n_up));
    for a in 0..m_up {
        for k in 0..n_up {
            let mut s = f.lap_up[[a, k]] * -0.5;
            for d in 0..3 {
                s -= f.gou[d][[a, k]] * gj_up[[k, d]];
            }
            bar_wg[[a, k]] = s;
        }
    }
    let mut bar_gou = [
        Array2::<Complex64>::zeros((m_up, n_up)),
        Array2::<Complex64>::zeros((m_up, n_up)),
        Array2::<Complex64>::zeros((m_up, n_up)),
    ];
    for d in 0..3 {
        for a in 0..m_up {
            for k in 0..n_up {
                bar_gou[d][[a, k]] = f.wg[[a, k]] * -gj_up[[k, d]];
            }
        }
    }
    let mut bar_gq = Array2::<Complex64>::zeros((n_up, m_dn));
    for l in 0..n_dn {
        for b in 0..m_dn {
            let mut s = f.lap_dn[[b, l]] * -0.5;
            for d in 0..3 {
                s -= f.godn[d][[b, l]] * gj_dn[[l, d]];
            }
            bar_gq[[l, b]] = s;
        }
    }
    let mut bar_godn = [
        Array2::<Complex64>::zeros((m_dn, n_dn)),
        Array2::<Complex64>::zeros((m_dn, n_dn)),
        Array2::<Complex64>::zeros((m_dn, n_dn)),
    ];
    for d in 0..3 {
        for b in 0..m_dn {
            for l in 0..n_dn {
                bar_godn[d][[b, l]] = f.gq[[l, b]] * -gj_dn[[l, d]];
            }
        }
    }
    let bar_ginv = f.w.t().dot(&bar_wg) + bar_gq.dot(&f.q.t());
    let mut bar_w = bar_wg.dot(&f.ginv.t());
    let bar_q = f.ginv.t().dot(&bar_gq);
    let bar_g = f.ginv.t().dot(&bar_ginv).dot(&f.ginv.t()).mapv(|x| -x);
    bar_w = bar_w + f.ou.dot(&bar_g);
    let bar_q_term = if let Some(lam_p) = &f.lam_p {
        lam_p.dot(&bar_q.t())
    } else {
        let mut out = Array2::<Complex64>::zeros((m_up, n_up));
        out.slice_mut(numpy::ndarray::s![0..m_dn, ..])
            .assign(&bar_q.t());
        out
    };
    let bar_orb_up = f.w.dot(&bar_g.t()) + bar_q_term;
    let bar_w_left = bar_w.slice(numpy::ndarray::s![.., 0..n_dn]).to_owned();
    let bar_orb_dn = if let Some(lam_p) = &f.lam_p {
        lam_p.t().dot(&bar_w_left)
    } else {
        bar_w_left.slice(numpy::ndarray::s![0..m_dn, ..]).to_owned()
    };
    let (bar_lam_p, bar_lam_unp) = if f.lam_p.is_some() {
        (
            bar_w_left.dot(&f.od.t()) + f.ou.dot(&bar_q),
            bar_w.slice(numpy::ndarray::s![.., n_dn..]).to_owned(),
        )
    } else if gem.orb_up.tiled.is_some() {
        // Frozen tiled SD has no global variational lambda operator. Force,
        // stress and Jastrow VJPs consume only the orbital cotangents above;
        // do not allocate unused supercell-sized coefficient-gradient buffers.
        (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
    } else {
        (
            Array2::<Complex64>::zeros((m_up, m_dn)),
            Array2::<Complex64>::zeros((m_up, n_up - n_dn)),
        )
    };
    GemReverseC {
        bar_orb_up,
        bar_orb_dn,
        bar_lap_up,
        bar_lap_dn,
        bar_gou,
        bar_godn,
        bar_lam_p,
        bar_lam_unp,
    }
}

/// Per-electron **weighted** complex kinetic VJP (mirror of the real `geminal_reverse_w`).
/// Every per-electron (column) seed of [`complex_geminal_reverse`] -- the lap/WG/gou/GQ/godn
/// cotangents -- is scaled by the S' membership weight `w_up[k]` / `w_dn[l]` BEFORE the linear
/// `G^-1` VJP propagation (which carries the weights unchanged). `w=None` reduces bit-identically
/// to the unweighted reverse. Used by the complex S' masked continuous-kinetic correction.
#[allow(
    clippy::too_many_arguments,
    reason = "per-spin drift and weight inputs of the geminal reverse pass"
)]
pub fn complex_geminal_reverse_w(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    w_up: Option<&[f64]>,
    w_dn: Option<&[f64]>,
    gi: Option<&ArrayView2<Complex64>>,
) -> GemReverseC {
    let f = complex_gem_forward(gem, r_up, r_dn, gi);
    complex_geminal_reverse_from_fwd(gem, &f, gj_up, gj_dn, w_up, w_dn)
}

/// [`complex_geminal_reverse_w`] on an already evaluated [`complex_gem_forward`] forward.
pub fn complex_geminal_reverse_from_fwd(
    gem: &ComplexGeminal,
    f: &ComplexGemFwd,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    w_up: Option<&[f64]>,
    w_dn: Option<&[f64]>,
) -> GemReverseC {
    let n_up = f.ou.ncols();
    let m_up = f.ou.nrows();
    let (n_dn, m_dn) = (f.n_dn, f.m_dn);
    let wu = |k: usize| -> f64 { w_up.map_or(1.0, |w| w[k]) };
    let wd = |l: usize| -> f64 { w_dn.map_or(1.0, |w| w[l]) };
    let bar_lap_up = {
        let mut b = f.wg.mapv(|x| x * -0.5);
        for a in 0..m_up {
            for k in 0..n_up {
                b[[a, k]] *= wu(k);
            }
        }
        b
    };
    let mut bar_lap_dn = Array2::<Complex64>::zeros((m_dn, n_dn));
    for l in 0..n_dn {
        for b in 0..m_dn {
            bar_lap_dn[[b, l]] = f.gq[[l, b]] * -0.5 * wd(l);
        }
    }
    let mut bar_wg = Array2::<Complex64>::zeros((m_up, n_up));
    for a in 0..m_up {
        for k in 0..n_up {
            let mut s = f.lap_up[[a, k]] * -0.5;
            for d in 0..3 {
                s -= f.gou[d][[a, k]] * gj_up[[k, d]];
            }
            bar_wg[[a, k]] = s * wu(k);
        }
    }
    let mut bar_gou = [
        Array2::<Complex64>::zeros((m_up, n_up)),
        Array2::<Complex64>::zeros((m_up, n_up)),
        Array2::<Complex64>::zeros((m_up, n_up)),
    ];
    for d in 0..3 {
        for a in 0..m_up {
            for k in 0..n_up {
                bar_gou[d][[a, k]] = f.wg[[a, k]] * -gj_up[[k, d]] * wu(k);
            }
        }
    }
    let mut bar_gq = Array2::<Complex64>::zeros((n_up, m_dn));
    for l in 0..n_dn {
        for b in 0..m_dn {
            let mut s = f.lap_dn[[b, l]] * -0.5;
            for d in 0..3 {
                s -= f.godn[d][[b, l]] * gj_dn[[l, d]];
            }
            bar_gq[[l, b]] = s * wd(l);
        }
    }
    let mut bar_godn = [
        Array2::<Complex64>::zeros((m_dn, n_dn)),
        Array2::<Complex64>::zeros((m_dn, n_dn)),
        Array2::<Complex64>::zeros((m_dn, n_dn)),
    ];
    for d in 0..3 {
        for b in 0..m_dn {
            for l in 0..n_dn {
                bar_godn[d][[b, l]] = f.gq[[l, b]] * -gj_dn[[l, d]] * wd(l);
            }
        }
    }
    let bar_ginv = f.w.t().dot(&bar_wg) + bar_gq.dot(&f.q.t());
    let mut bar_w = bar_wg.dot(&f.ginv.t());
    let bar_q = f.ginv.t().dot(&bar_gq);
    let bar_g = f.ginv.t().dot(&bar_ginv).dot(&f.ginv.t()).mapv(|x| -x);
    bar_w = bar_w + f.ou.dot(&bar_g);
    let bar_q_term = if let Some(lam_p) = &f.lam_p {
        lam_p.dot(&bar_q.t())
    } else {
        let mut out = Array2::<Complex64>::zeros((m_up, n_up));
        out.slice_mut(numpy::ndarray::s![0..m_dn, ..])
            .assign(&bar_q.t());
        out
    };
    let bar_orb_up = f.w.dot(&bar_g.t()) + bar_q_term;
    let bar_w_left = bar_w.slice(numpy::ndarray::s![.., 0..n_dn]).to_owned();
    let bar_orb_dn = if let Some(lam_p) = &f.lam_p {
        lam_p.t().dot(&bar_w_left)
    } else {
        bar_w_left.slice(numpy::ndarray::s![0..m_dn, ..]).to_owned()
    };
    let (bar_lam_p, bar_lam_unp) = if f.lam_p.is_some() {
        (
            bar_w_left.dot(&f.od.t()) + f.ou.dot(&bar_q),
            bar_w.slice(numpy::ndarray::s![.., n_dn..]).to_owned(),
        )
    } else if gem.orb_up.tiled.is_some() {
        // Frozen tiled SD has no global variational lambda operator. Force,
        // stress and Jastrow VJPs consume only the orbital cotangents above;
        // do not allocate unused supercell-sized coefficient-gradient buffers.
        (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
    } else {
        (
            Array2::<Complex64>::zeros((m_up, m_dn)),
            Array2::<Complex64>::zeros((m_up, n_up - n_dn)),
        )
    };
    GemReverseC {
        bar_orb_up,
        bar_orb_dn,
        bar_lap_up,
        bar_lap_dn,
        bar_gou,
        bar_godn,
        bar_lam_p,
        bar_lam_unp,
    }
}

/// Complex determinant kinetic force (BOTH axes), per-electron S'-weighted, Re-projected:
/// returns `(d_r (n_atom,3) atom-axis dT_det/dR, d_re (n_e,3) electron-axis dT_det/dr)`.
/// Det side of the withJ kinetic force (the Jastrow J1/J2/J3/DTN side is added separately by the
/// caller from `cot = gJ + Re(gD)`). `gj_*` seed the complex reverse's gou cross term (real
/// Jastrow drift; zero for jnone); `w=None` => full dT_det/dtheta. Uses the SAME complex AO
/// grad/gradlap/hessian as `grad_t_wrt_r_complex_withj`: atom-axis via the translation identity
/// `dAO/dR = -grad AO` (scatter -, sum over electrons) and electron-axis `dAO/dr_k = +grad AO`
/// (+, the electron's own column). GTO-only (composite-PW twist has no committed fixture).
#[allow(clippy::too_many_arguments)]
pub fn complex_det_kinetic_grads(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    w_up: Option<&[f64]>,
    w_dn: Option<&[f64]>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
    gi: Option<&ArrayView2<Complex64>>,
    // The base forward at `gi` when the caller already has it.
    fwd: Option<&ComplexGemFwd>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let b = match fwd {
        Some(f) => complex_geminal_reverse_from_fwd(gem, f, gj_up, gj_dn, w_up, w_dn),
        None => complex_geminal_reverse_w(gem, r_up, r_dn, gj_up, gj_dn, w_up, w_dn, gi),
    };
    let mut d_r = Array2::<f64>::zeros((n_atom, 3));
    let mut d_re = Array2::<f64>::zeros((n_up + n_dn, 3));
    let side = |orb: &ComplexOrb,
                r: &ArrayView2<f64>,
                bar_orb: &Array2<Complex64>,
                bar_lap: &Array2<Complex64>,
                bar_gou: &[Array2<Complex64>; 3],
                nuc: &[usize],
                e_off: usize,
                d_r: &mut Array2<f64>,
                d_re: &mut Array2<f64>| {
        if let Some(tiled) = &orb.tiled {
            *d_r += &tiled
                .field_vjp(
                    r,
                    bar_orb,
                    Some(bar_gou),
                    Some(bar_lap),
                    crate::orbital_tiling::TiledCoordinate::Atoms,
                )
                .mapv(|v| v.re);
            let electron = tiled
                .field_vjp(
                    r,
                    bar_orb,
                    Some(bar_gou),
                    Some(bar_lap),
                    crate::orbital_tiling::TiledCoordinate::Electrons,
                )
                .mapv(|v| v.re);
            let mut target = d_re.slice_mut(numpy::ndarray::s![e_off..e_off + r.nrows(), ..]);
            target += &electron;
            return;
        }
        let cm_orb0 = match &orb.mo {
            Some(c) => c.t().dot(bar_orb),
            None => bar_orb.clone(),
        };
        let cm_lap0 = match &orb.mo {
            Some(c) => c.t().dot(bar_lap),
            None => bar_lap.clone(),
        };
        let cm_gou0: Vec<Array2<Complex64>> = bar_gou
            .iter()
            .map(|bg| match &orb.mo {
                Some(c) => c.t().dot(bg),
                None => bg.clone(),
            })
            .collect();
        // AO-block layout of the (pre-MO) cotangent rows: `num_gto` atom-centered GTO rows
        // (BOTH axes -- nuclear force via the translation identity + electron kinetic force)
        // and `num_extra` PW/B-spline rows (electron axis ONLY -- dAO/dR = 0). `gto_first`
        // fixes the row order. Pure PW / pure B-spline => num_gto = 0 (empty gto block).
        let num_extra = orb
            .pw
            .as_ref()
            .map(|p| p.num_ao)
            .or_else(|| orb.bspline.as_ref().map(|b| b.num_ao))
            .unwrap_or(0);
        let total = cm_orb0.nrows();
        let num_gto = total - num_extra;
        let gto0 = if orb.gto_first { 0 } else { num_extra };
        let extra0 = if orb.gto_first { num_gto } else { 0 };
        let sl = |cm: &Array2<Complex64>, off: usize, len: usize| {
            cm.slice(numpy::ndarray::s![off..off + len, ..]).to_owned()
        };
        // --- GTO rows: both axes (nuclear force + electron kinetic force) ---
        if num_gto > 0 {
            let cm_orb = sl(&cm_orb0, gto0, num_gto);
            let cm_lap = sl(&cm_lap0, gto0, num_gto);
            let cm_gou: Vec<Array2<Complex64>> =
                cm_gou0.iter().map(|c| sl(c, gto0, num_gto)).collect();
            // Fused contraction (no kernel-rank derivative tables) when the leaf is covered.
            let fused = crate::mesh_vgs::ShellGeom::new(&orb.aos, nuc, None).and_then(|geom| {
                crate::complex_stress::fused_complex_kin_force(
                    orb,
                    r,
                    &cm_orb,
                    Some(&cm_lap),
                    Some([&cm_gou[0], &cm_gou[1], &cm_gou[2]]),
                    &geom,
                    d_r.nrows(),
                )
            });
            if let Some((elec, atom)) = fused {
                *d_r -= &atom;
                let mut target = d_re.slice_mut(numpy::ndarray::s![e_off..e_off + r.nrows(), ..]);
                target += &elec;
            } else {
                let g = orb.ao_grad(r);
                let gl = orb.ao_gradlap(r);
                let h = orb.ao_hessian(r);
                let n = cm_orb.ncols();
                for a in 0..num_gto {
                    let ia = nuc[a];
                    for k in 0..n {
                        for c in 0..3 {
                            let mut acc = Complex64::new(0.0, 0.0);
                            acc += cm_orb[[a, k]] * g[c][[a, k]] + cm_lap[[a, k]] * gl[c][[a, k]];
                            for e in 0..3 {
                                acc += cm_gou[e][[a, k]] * h[c][e][[a, k]];
                            }
                            // atom axis: dAO/dR = -grad AO (scatter to home atom, sum over electrons k)
                            d_r[[ia, c]] -= acc.re;
                            // electron axis: dAO/dr_k = +grad AO (electron k of this spin block)
                            d_re[[e_off + k, c]] += acc.re;
                        }
                    }
                }
            }
        }
        // --- PW rows: ELECTRON axis only (R-independent basis -> dAO/dR = 0). The complex
        // analog of the real `pw_det_egrad_side`: phi = e^{i gk.r} (twist) / cos(gk.r) (Gamma),
        // d(value)/dr_c = grad_c, d(lap)/dr_c = -g2*grad_c, d(grad_e)/dr_c = Hess[c][e]. ---
        if let Some(pw) = &orb.pw {
            let cm_orb = sl(&cm_orb0, extra0, num_extra);
            let cm_lap = sl(&cm_lap0, extra0, num_extra);
            let cm_gou: Vec<Array2<Complex64>> =
                cm_gou0.iter().map(|c| sl(c, extra0, num_extra)).collect();
            let (_v, grad, _lap) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
            let hpw = crate::complex_ao::compute_pw_hessian_complex(pw, r);
            let n = cm_orb.ncols();
            for a in 0..num_extra {
                let gk = pw.gk[a];
                let g2 = gk[0] * gk[0] + gk[1] * gk[1] + gk[2] * gk[2];
                for k in 0..n {
                    for c in 0..3 {
                        let gc = grad[c][[a, k]];
                        let mut acc = cm_orb[[a, k]] * gc + cm_lap[[a, k]] * (-g2 * gc);
                        for e in 0..3 {
                            acc += cm_gou[e][[a, k]] * hpw[c][e][[a, k]];
                        }
                        d_re[[e_off + k, c]] += acc.re;
                    }
                }
            }
        }
        // --- B-spline rows: ELECTRON axis only (fixed grid -> dAO/dR = 0). Complex analog of
        // the real `bspline_det_egrad_side`, via the complex `det_egrad_fields` (ga/gl/h). ---
        if let Some(bs) = &orb.bspline {
            let cm_orb = sl(&cm_orb0, extra0, num_extra);
            let cm_lap = sl(&cm_lap0, extra0, num_extra);
            let cm_gou: Vec<Array2<Complex64>> =
                cm_gou0.iter().map(|c| sl(c, extra0, num_extra)).collect();
            let (ga, gl, h) = bs.det_egrad_fields(r);
            let n = cm_orb.ncols();
            for a in 0..num_extra {
                for k in 0..n {
                    for c in 0..3 {
                        let mut acc =
                            cm_orb[[a, k]] * ga[c][[a, k]] + cm_lap[[a, k]] * gl[c][[a, k]];
                        for e in 0..3 {
                            acc += cm_gou[e][[a, k]] * h[c][e][[a, k]];
                        }
                        d_re[[e_off + k, c]] += acc.re;
                    }
                }
            }
        }
    };
    side(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        &b.bar_gou,
        nuc_up,
        0,
        &mut d_r,
        &mut d_re,
    );
    side(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        &b.bar_godn,
        nuc_dn,
        n_up,
        &mut d_r,
        &mut d_re,
    );
    (d_r, d_re)
}

/// Complex `dT/dMO` (up, dn) for the twisted geminal (jnone: gj=0). `bar_orb*AOT
/// + bar_lap*grad^2AOT + Sum_d bar_gou*grad_d AOT`. 1:1 the real `grad_t_mo`.
pub fn grad_t_mo_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let zu = Array2::<f64>::zeros((r_up.nrows(), 3));
    let zd = Array2::<f64>::zeros((r_dn.nrows(), 3));
    grad_t_mo_complex_gj(gem, r_up, r_dn, &zu, &zd)
}

/// `dT/dMO` with an externally-supplied real Jastrow drift `gj` (diagnostic /
/// withJ: exercises the gj!=0 path of `complex_geminal_reverse` incl. bar_gou*gradAO).
pub fn grad_t_mo_complex_gj(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
) -> (Array2<Complex64>, Array2<Complex64>) {
    grad_t_mo_complex_gj_gi(gem, r_up, r_dn, gj_up, gj_dn, None)
}

/// Complex `dT/dMO` with an optional caller-supplied MCMC geminal inverse.
/// `None` retains the standalone fresh-inverse contract.
pub fn grad_t_mo_complex_gj_gi(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    g_inv: Option<&ArrayView2<Complex64>>,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let b = complex_geminal_reverse(gem, r_up, r_dn, gj_up, gj_dn, g_inv);
    let to_c = |orb: &ComplexOrb,
                r: &ArrayView2<f64>,
                bar_orb: &Array2<Complex64>,
                bar_lap: &Array2<Complex64>,
                bar_g: &[Array2<Complex64>; 3]|
     -> Array2<Complex64> {
        let (v, g, l) = compute_aos_vgl_tuple(orb, r);
        bar_orb.dot(&v.t())
            + bar_lap.dot(&l.t())
            + bar_g[0].dot(&g[0].t())
            + bar_g[1].dot(&g[1].t())
            + bar_g[2].dot(&g[2].t())
    };
    let mut up = to_c(&gem.orb_up, r_up, &b.bar_orb_up, &b.bar_lap_up, &b.bar_gou);
    let mut dn = to_c(&gem.orb_dn, r_dn, &b.bar_orb_dn, &b.bar_lap_dn, &b.bar_godn);
    if gem.is_sd() {
        if up.nrows() == up.ncols() {
            up.fill(Complex64::new(0.0, 0.0));
        }
        if dn.nrows() == dn.ncols() {
            dn.fill(Complex64::new(0.0, 0.0));
        }
    }
    (up, dn)
}

/// Complex `dT/dlambda` (= Re for the energy gradient; lambda real) for the twisted geminal (jnone).
pub fn grad_t_lambda_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<Complex64> {
    let zu = Array2::<f64>::zeros((r_up.nrows(), 3));
    let zd = Array2::<f64>::zeros((r_dn.nrows(), 3));
    grad_t_lambda_complex_gj(gem, r_up, r_dn, &zu, &zd)
}

/// Complex `dT/dlambda` with an externally-supplied REAL Jastrow drift `gj` (the
/// with-Jastrow twin of [`grad_t_lambda_complex`], which is just `gj = 0`). `J(r)` is
/// real and lambda-INDEPENDENT, so the only Jastrow entry into `dT/dlambda` is the
/// cross term `-(gD + gJ) . d gD/dlambda`, i.e. exactly the `gj` seed
/// `complex_geminal_reverse` already threads. Stays HOLOMORPHIC (complex `bar_lambda`);
/// `Re` is taken by the caller, LAST.
pub fn grad_t_lambda_complex_gj(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
) -> Array2<Complex64> {
    let b = complex_geminal_reverse(gem, r_up, r_dn, gj_up, gj_dn, None);
    let m_up = b.bar_lam_p.nrows();
    let n_p = b.bar_lam_p.ncols();
    let mut out = Array2::<Complex64>::zeros((m_up, n_p + b.bar_lam_unp.ncols()));
    out.slice_mut(numpy::ndarray::s![.., 0..n_p])
        .assign(&b.bar_lam_p);
    out.slice_mut(numpy::ndarray::s![.., n_p..])
        .assign(&b.bar_lam_unp);
    out
}

/// AO value + grad + Laplacian of one complex orbital leaf (the MO-transformed AO).
fn compute_aos_vgl_tuple(
    orb: &ComplexOrb,
    r: &ArrayView2<f64>,
) -> (Array2<Complex64>, [Array2<Complex64>; 3], Array2<Complex64>) {
    // dT/dMO contracts with the raw AO (not MO-transformed): the MO IS the variable.
    // Mirror of `ComplexOrb::value_grad_lap` MINUS the MO transform: the raw AO stack
    // is (gto[+]{pw,bspline}) in `gto_first` child order. `pw` and `bspline` are
    // mutually exclusive; a pure complex plane_wave / bspline leaf has a 0-AO gto block.
    let cat = |a: &Array2<Complex64>, b: &Array2<Complex64>| {
        if orb.gto_first {
            vstack_c(a, b)
        } else {
            vstack_c(b, a)
        }
    };
    if let Some(bs) = &orb.bspline {
        let (bv, bg, bl) = bs.value_grad_lap(r);
        if orb.aos.num_ao == 0 {
            return (bv, [bg[0].clone(), bg[1].clone(), bg[2].clone()], bl);
        }
        let (gv, ggx, ggy, ggz, gl) = orb.gto_vgl(r);
        return (
            cat(&gv, &bv),
            [cat(&ggx, &bg[0]), cat(&ggy, &bg[1]), cat(&ggz, &bg[2])],
            cat(&gl, &bl),
        );
    }
    let (gv, ggx, ggy, ggz, gl) = orb.gto_vgl(r);
    // Composite pgto(+)pw: append the pw block in the orbital ordering.
    match &orb.pw {
        None => (gv, [ggx, ggy, ggz], gl),
        Some(pw) => {
            let (pv, pg, pl) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
            (
                cat(&gv, &pv),
                [cat(&ggx, &pg[0]), cat(&ggy, &pg[1]), cat(&ggz, &pg[2])],
                cat(&gl, &pl),
            )
        }
    }
}

/// Complex `O_k = dln det/dMO` for up/dn (holomorphic; jax.grad of ln|det| wrt the
/// complex MO returns exactly this -- verified). `(M_up,num_ao_up),(M_dn,num_ao_dn)`.
pub fn grad_ln_det_mo_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let n_dn = r_dn.nrows();
    let ao_up = gem.orb_up.ao(r_up);
    let ao_dn = gem.orb_dn.ao(r_dn);
    let ou = gem.orb_up.value(r_up);
    let od = gem.orb_dn.value(r_dn);
    let n_up = ou.ncols();
    let w = gem.w_from_dn(&od, n_up);
    let g = ou.t().dot(&w);
    let ginv = inverse_complex(&g);
    let bar_orb_up = w.dot(&ginv); // (M_up, N_up)
    let q = gem.up_t_times_paired(&ou); // (N_up, M_dn)
    let gq = ginv.dot(&q); // (N_up, M_dn)
    let bar_orb_dn = gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(); // (M_dn, N_dn)
    (bar_orb_up.dot(&ao_up.t()), bar_orb_dn.dot(&ao_dn.t()))
}

/// `O_k = dln det/dlambda` HOLOMORPHIC (un-conjugated, complex) for a twisted
/// geminal, `(M_up, lambda.ncols)`. The c-axis O_k that the
/// driver stores for a COMPLEX lambda (`AGP_data` twist) is the un-conjugated
/// holomorphic derivative -- complex, NOT Re-projected. Mirror of the real
/// `grad_ln_det_lambda` with a complex geminal (jax.grad of `Re ln det` wrt the
/// complex lambda returns exactly this). The real-projecting sibling below is for
/// the energy-gradient consumer where lambda is a real DOF.
pub fn grad_ln_det_lambda_complex_holo(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<Complex64> {
    let lam = gem
        .lambda
        .as_ref()
        .expect("lambda derivative is undefined for occupied-only SD")
        .mapv(|x| Complex64::new(x, 0.0));
    grad_ln_det_lambda_complex_holo_clam(gem, &lam.view(), r_up, r_dn)
}

/// As [`grad_ln_det_lambda_complex_holo`] but with an explicit complex `lambda`.
/// This is used by dense-lambda AGP/AGPn derivative consumers; occupied-only SD
/// has no lambda derivative.
pub fn grad_ln_det_lambda_complex_holo_clam(
    gem: &ComplexGeminal,
    lam: &ArrayView2<Complex64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<Complex64> {
    let n_dn = r_dn.nrows();
    let m_dn = gem.orb_num_dn;
    let a = gem.orb_up.value(r_up); // (M_up, N_up)
    let a = a.t().to_owned(); // (N_up, M_up)
    let b = gem.orb_dn.value(r_dn); // (M_dn, N_dn)
    let lam = lam.to_owned();
    let lam_p = lam.slice(numpy::ndarray::s![.., 0..m_dn]).to_owned();
    let lam_u = lam.slice(numpy::ndarray::s![.., m_dn..]).to_owned();
    let g_paired = a.dot(&lam_p).dot(&b); // (N_up, N_dn)
    let g_unpaired = a.dot(&lam_u);
    let n_up = a.nrows();
    let mut g = Array2::<Complex64>::zeros((n_up, n_up));
    g.slice_mut(numpy::ndarray::s![.., 0..n_dn])
        .assign(&g_paired);
    g.slice_mut(numpy::ndarray::s![.., n_dn..])
        .assign(&g_unpaired);
    let ginv = inverse_complex(&g);
    let p = ginv.dot(&a); // (N_up, M_up)
    let grad_paired = p.slice(numpy::ndarray::s![0..n_dn, ..]).t().dot(&b.t()); // (M_up, M_dn)
    let grad_unpaired = p.slice(numpy::ndarray::s![n_dn.., ..]).t().to_owned();
    let m_up = grad_paired.nrows();
    let mut out = Array2::<Complex64>::zeros((m_up, lam.ncols()));
    out.slice_mut(numpy::ndarray::s![.., 0..m_dn])
        .assign(&grad_paired);
    out.slice_mut(numpy::ndarray::s![.., m_dn..])
        .assign(&grad_unpaired);
    out
}

/// `O_k = dln|det|/dlambda = Re(dln det/dlambda)` (lambda is real), `(M_up, lambda.ncols)`. Mirror of
/// the real `grad_ln_det_lambda` with a complex geminal.
pub fn grad_ln_det_lambda_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    grad_ln_det_lambda_complex_holo(gem, r_up, r_dn).mapv(|z| z.re)
}

/// AGPn twist `O_k` on the natural-orbital leaves: `Re` of the paired-block
/// diagonal + the unpaired columns of the complex-twist dense-`lambda` gradient.
/// `lambda_diag` is real (occupations), so its gradient is the real part. The
/// unpaired columns are Re-projected here (valid where the block is empty, i.e. the
/// closed-shell AGPn geminals -- the only twist AGPn form in the fixture matrix).
pub fn grad_ln_det_lambda_diag_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (numpy::ndarray::Array1<f64>, Array2<f64>) {
    let dense = grad_ln_det_lambda_complex(gem, r_up, r_dn); // Array2<f64> (already Re)
    let m_dn = gem.orb_num_dn;
    let n = m_dn.min(dense.nrows());
    let diag = numpy::ndarray::Array1::from_iter((0..n).map(|k| dense[[k, k]]));
    let unpaired = dense.slice(numpy::ndarray::s![.., m_dn..]).to_owned();
    (diag, unpaired)
}

/// AGPn twist dE_L on the natural-orbital leaves: `Re` of the paired-block diagonal
/// + unpaired columns of `grad_t_lambda_complex` (Jastrow-free twist kinetic).
pub fn grad_t_lambda_diag_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (numpy::ndarray::Array1<f64>, Array2<f64>) {
    let dense = grad_t_lambda_complex(gem, r_up, r_dn); // Array2<Complex64>
    let m_dn = gem.orb_num_dn;
    let n = m_dn.min(dense.nrows());
    let diag = numpy::ndarray::Array1::from_iter((0..n).map(|k| dense[[k, k]].re));
    let unpaired = dense.slice(numpy::ndarray::s![.., m_dn..]).mapv(|z| z.re);
    (diag, unpaired)
}

/// Complex analogue of ``determinant::geminal_from_det_ratio_state``.
pub fn geminal_from_det_ratio_state_complex(
    gem: &ComplexGeminal,
    state: &DetRatioStateC,
) -> Array2<Complex64> {
    let n_up = state.ao_up.ncols();
    let n_orb_up = state.ao_up.nrows();
    let n_dn = state.paired_dn.ncols();
    let n_unpaired = gem.n_unpaired();
    let mut g = Array2::<Complex64>::zeros((n_up, n_dn + n_unpaired));
    for i in 0..n_up {
        for j in 0..n_dn {
            let mut s = Complex64::new(0.0, 0.0);
            for a in 0..state.paired_dn.nrows() {
                s += state.ao_up[[a, i]] * state.paired_dn[[a, j]];
            }
            g[[i, j]] = s;
        }
        for k in 0..n_unpaired {
            let mut s = Complex64::new(0.0, 0.0);
            for a in 0..n_orb_up {
                s += state.ao_up[[a, i]] * gem.lambda_c_value(a, gem.orb_num_dn + k);
            }
            g[[i, n_dn + k]] = s;
        }
    }
    g
}

/// Complex analogue of ``determinant::running_inverse_probe_residual``.
pub fn running_inverse_probe_residual_complex(
    g: &ArrayView2<Complex64>,
    gi: &ArrayView2<Complex64>,
) -> f64 {
    let n = gi.nrows();
    let mut worst = 0.0_f64;
    for probe in 0..2 {
        let p: Vec<Complex64> = (0..n)
            .map(|j| {
                Complex64::new(
                    if probe == 0 {
                        1.0
                    } else {
                        (1.0 + 0.7 * j as f64).cos()
                    },
                    0.0,
                )
            })
            .collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| (0..n).map(|j| gi[[i, j]] * p[j]).sum::<Complex64>())
            .collect();
        for i in 0..n {
            let r = (0..n).map(|j| g[[i, j]] * y[j]).sum::<Complex64>() - p[i];
            let a = r.norm();
            if a.is_nan() {
                return f64::NAN;
            }
            if a > worst {
                worst = a;
            }
        }
    }
    worst
}

/// Complex analogue of ``determinant::det_state_inverse_probe_residual``:
/// the probe residual of a carried inverse against the geminal the
/// streaming caches imply ([`geminal_from_det_ratio_state_complex`]), applied
/// in factored form (O(N^2 + N M), no ``G`` formed). Plain transpose, no
/// conjugation, exactly like the geminal assembly.
pub fn det_state_inverse_probe_residual_complex(
    gem: &ComplexGeminal,
    state: &DetRatioStateC,
    gi: &ArrayView2<Complex64>,
) -> f64 {
    let n = gi.nrows();
    let n_dn = state.paired_dn.ncols();
    let m_paired = state.paired_dn.nrows();
    let m_up = state.ao_up.nrows();
    let n_unpaired = gem.n_unpaired();
    // Both (real) probe columns in one pass over every operand, row-major
    // contiguous traversal: y = X P, z = paired_dn y_p (+ lambda_u y_u),
    // G y = ao_up^T z.
    let p1: Vec<f64> = (0..n).map(|j| (1.0 + 0.7 * j as f64).cos()).collect();
    let zero = Complex64::new(0.0, 0.0);
    let gi = gi.as_standard_layout();
    let gis = gi.as_slice().expect("standard layout");
    let mut y0 = vec![zero; n];
    let mut y1 = vec![zero; n];
    for i in 0..n {
        let row = &gis[i * n..(i + 1) * n];
        let (mut s0, mut s1) = (zero, zero);
        for (x, &q) in row.iter().zip(p1.iter()) {
            s0 += *x;
            s1 += *x * q;
        }
        y0[i] = s0;
        y1[i] = s1;
    }
    let pdn = state.paired_dn.as_standard_layout();
    let pdns = pdn.as_slice().expect("standard layout");
    let mut z0 = vec![zero; m_up];
    let mut z1 = vec![zero; m_up];
    for a in 0..m_paired {
        let row = &pdns[a * n_dn..(a + 1) * n_dn];
        let (mut s0, mut s1) = (zero, zero);
        for j in 0..n_dn {
            s0 += row[j] * y0[j];
            s1 += row[j] * y1[j];
        }
        z0[a] = s0;
        z1[a] = s1;
    }
    if n_unpaired > 0 {
        for a in 0..m_up {
            for k in 0..n_unpaired {
                let l = gem.lambda_c_value(a, gem.orb_num_dn + k);
                z0[a] += l * y0[n_dn + k];
                z1[a] += l * y1[n_dn + k];
            }
        }
    }
    let aup = state.ao_up.as_standard_layout();
    let aups = aup.as_slice().expect("standard layout");
    let mut g0 = vec![zero; n];
    let mut g1 = vec![zero; n];
    for a in 0..m_up {
        let row = &aups[a * n..(a + 1) * n];
        let (za, zb) = (z0[a], z1[a]);
        for i in 0..n {
            g0[i] += row[i] * za;
            g1[i] += row[i] * zb;
        }
    }
    let mut worst = 0.0_f64;
    for i in 0..n {
        for r in [(g0[i] - 1.0).norm(), (g1[i] - p1[i]).norm()] {
            if r.is_nan() {
                return f64::NAN;
            }
            if r > worst {
                worst = r;
            }
        }
    }
    worst
}

/// Cutoff of the geminal pseudo-inverse: singular values below
/// ``RCOND_SVD * s_max`` are dropped. The float64 value of the Python drivers'
/// ``jqmc._setting.get_eps("rcond_svd", float64)``.
pub use crate::setting_consts::EPS_rcond_SVD as RCOND_SVD;

/// Issue #605: a walker's equilibrated LU inverse ``Y = inv(A)``, ``A = R G C``,
/// is kept only if it is finite and its scale-free carried-inverse probe residual
/// ``max|A (Y P) - P|`` does not exceed this; otherwise the walker takes the
/// equilibrated pseudo-inverse. The Python drivers'
/// ``jqmc._setting.GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD``. Not a user knob.
pub use crate::setting_consts::GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD;

/// Issue #605: pinv equilibration exponent clip: keeps ``C A^+ R`` finite for
/// any finite ``G`` (``jqmc._setting.GEMINAL_EQUILIBRATION_MAX_EXPONENT``, the
/// ``|e|`` bound of the pseudo-inverse's power-of-two scales). The unscale
/// factor ``c_i r_k`` is then at most ``2**956`` and the thresholded
/// pseudo-inverse of the equilibrated matrix is bounded by
/// ``1 / (RCOND_SVD s_max) ~ 2**66``. The LU inverse keeps the full exponent range: there an
/// out-of-range exact inverse is caught by the finiteness guard instead.
pub const GEMINAL_EQUILIBRATION_MAX_EXPONENT: i32 =
    crate::setting_consts::GEMINAL_EQUILIBRATION_MAX_EXPONENT as i32;

/// Power-of-two scale ``2**-e`` with ``norm = m 2**e``, ``0.5 <= m < 1``
/// (``numpy.frexp``), the exponent clipped to ``[-max_exponent, max_exponent]``;
/// ``1`` for a zero or non-finite norm.
fn pow2_scale_clipped(norm: f64, max_exponent: i32) -> f64 {
    if !(norm.is_finite() && norm > 0.0) {
        return 1.0;
    }
    let (x, shift) = if norm < f64::MIN_POSITIVE {
        (norm * 2f64.powi(54), 54)
    } else {
        (norm, 0)
    };
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    let exponent = (biased - 1022 - shift).clamp(-max_exponent, max_exponent);
    f64::from_bits(((1023 - exponent) as u64) << 52)
}

/// [`pow2_scale_clipped`] over the full normal range (``|e| <= 1021``): the
/// range-safe division and the equilibration of the geminal LU inverse.
pub(crate) fn pow2_scale(norm: f64) -> f64 {
    pow2_scale_clipped(norm, 1021)
}

/// Equilibration scale of the geminal pseudo-inverse: [`pow2_scale_clipped`]
/// with ``|e| <= GEMINAL_EQUILIBRATION_MAX_EXPONENT``.
pub(crate) fn pinv_equilibration_scale(norm: f64) -> f64 {
    pow2_scale_clipped(norm, GEMINAL_EQUILIBRATION_MAX_EXPONENT)
}

/// Row and column equilibration of a complex geminal: ``(r, c, A)`` with
/// ``A = diag(r) G diag(c)``; ``r = scale(row max-norms of G)``,
/// ``c = scale(column max-norms of diag(r) G)``.
fn equilibrate_geminal_complex(
    g: &ArrayView2<Complex64>,
    scale: fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>, Array2<Complex64>) {
    let n = g.nrows();
    let row_scale: Vec<f64> = (0..n)
        .map(|i| scale((0..n).map(|j| g[[i, j]].norm()).fold(0.0, f64::max)))
        .collect();
    let mut w = Array2::<Complex64>::from_shape_fn((n, n), |(i, j)| g[[i, j]] * row_scale[i]);
    let col_scale: Vec<f64> = (0..n)
        .map(|j| scale((0..n).map(|i| w[[i, j]].norm()).fold(0.0, f64::max)))
        .collect();
    for i in 0..n {
        for j in 0..n {
            w[[i, j]] *= col_scale[j];
        }
    }
    (row_scale, col_scale, w)
}

/// Issue #605: the seed / carried-inverse refresh inverse of one complex geminal.
///
/// ``C inv(A) R`` from Gauss-Jordan elimination of the equilibrated
/// ``A = R G C`` (full exponent range, [`pow2_scale`]), kept when it is finite
/// and ``max|A (inv(A) P) - P|`` is at most
/// [`GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD`] (the probe columns ``P``,
/// [`running_inverse_probe_residual_complex`]); otherwise the equilibrated
/// pseudo-inverse [`geminal_pinv_from_matrix_complex`] (singular,
/// rank-deficient or non-finite ``G``, or an exact inverse outside the float64
/// range). The residual is scale-free, so a graded but invertible ``G`` keeps
/// its LU inverse; the guard is O(N^2).
pub fn geminal_inverse_guarded_complex(g: &ArrayView2<Complex64>) -> Array2<Complex64> {
    let n = g.nrows();
    if g.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
        let (row_scale, col_scale, a) = equilibrate_geminal_complex(g, pow2_scale);
        let y = inverse_complex(&a);
        let residual = running_inverse_probe_residual_complex(&a.view(), &y.view());
        if residual <= GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD {
            let x = Array2::from_shape_fn((n, n), |(i, k)| {
                unscale_pow2(y[[i, k]], col_scale[i], row_scale[k])
            });
            if x.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
                return x;
            }
        }
    }
    geminal_pinv_from_matrix_complex(g)
}

/// Row/column-equilibrated, thresholded SVD pseudo-inverse of one geminal --
/// the native mirror of ``jqmc_mcmc._geminal_pinv_from_matrix`` (issue #127).
///
/// Rows and then columns are scaled by exact powers of two into ``[0.5, 1)``
/// (exponents clipped at [`GEMINAL_EQUILIBRATION_MAX_EXPONENT`], so the result
/// is finite for every finite ``G``),
/// the SVD of the equilibrated matrix is taken by one-sided (Hestenes) Jacobi
/// rotations, singular values below ``RCOND_SVD * s_max`` are zeroed, and the
/// result is unscaled. A singular ``G`` (an exactly underflowed row, a node)
/// therefore gives a FINITE inverse whose probe residual stays large, as in
/// Python, instead of the ``inf`` of an unpivotable Gauss-Jordan elimination.
/// Non-finite input gives an all-NaN result (Python's SVD propagates it).
/// Real geminals use the same routine: every rotation then stays real.
pub fn geminal_pinv_from_matrix_complex(g: &ArrayView2<Complex64>) -> Array2<Complex64> {
    let n = g.nrows();
    let zero = Complex64::new(0.0, 0.0);
    if g.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
        return Array2::from_elem((n, n), Complex64::new(f64::NAN, f64::NAN));
    }
    let (row_scale, col_scale, mut w) = equilibrate_geminal_complex(g, pinv_equilibration_scale);
    // One-sided Jacobi: W <- W J, V <- V J until the columns of W are orthogonal;
    // then W = U diag(s), A = W V^H.
    let mut v = Array2::<Complex64>::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            Complex64::new(1.0, 0.0)
        } else {
            zero
        }
    });
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0_f64, 0.0_f64, zero);
                for i in 0..n {
                    alpha += w[[i, p]].norm_sqr();
                    beta += w[[i, q]].norm_sqr();
                    gamma += w[[i, p]].conj() * w[[i, q]];
                }
                let gabs = gamma.norm();
                let orthogonal = gabs.partial_cmp(&(f64::EPSILON * alpha.sqrt() * beta.sqrt()))
                    != Some(std::cmp::Ordering::Greater);
                if alpha == 0.0 || beta == 0.0 || orthogonal {
                    continue;
                }
                rotated = true;
                let phase_conj = (gamma / gabs).conj();
                let zeta = (beta - alpha) / (2.0 * gabs);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for m in [&mut w, &mut v] {
                    for i in 0..n {
                        let x = m[[i, p]];
                        let y = m[[i, q]] * phase_conj;
                        m[[i, p]] = x * c - y * s;
                        m[[i, q]] = x * s + y * c;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sv: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|i| w[[i, j]].norm_sqr()).sum::<f64>().sqrt())
        .collect();
    let s_max = sv.iter().cloned().fold(0.0, f64::max);
    // pinv(A) = V diag(1/s) U^H = sum_j v_j conj(w_j)^T / s_j^2 over kept j.
    let mut out = Array2::<Complex64>::from_elem((n, n), zero);
    for j in 0..n {
        if sv[j].partial_cmp(&(RCOND_SVD * s_max)) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let inv_s2 = 1.0 / (sv[j] * sv[j]);
        for i in 0..n {
            let vij = v[[i, j]] * inv_s2;
            for k in 0..n {
                out[[i, k]] += vij * w[[k, j]].conj();
            }
        }
    }
    // Unscale by C on the left and R on the right through the summed exponent
    // (see ``unscale_pow2``): ``col_scale[i] * row_scale[k]`` itself can
    // overflow while the inverse entry is finite (and gives NaN on a
    // structural zero), e.g. a row and a column both graded by 1e-160.
    for i in 0..n {
        for k in 0..n {
            out[[i, k]] = unscale_pow2(out[[i, k]], col_scale[i], row_scale[k]);
        }
    }
    out
}

/// Binary exponent of a normal power of two (every ``pow2_scale`` value).
pub(crate) fn pow2_exponent(scale: f64) -> i32 {
    ((scale.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// Real ``x * a * b`` for normal powers of two ``a``, ``b`` (see
/// [`unscale_pow2`]).
pub(crate) fn unscale_pow2_f64(x: f64, a: f64, b: f64) -> f64 {
    let e = pow2_exponent(a) + pow2_exponent(b);
    let pow2 = |k: i32| f64::from_bits(((1023 + k) as u64) << 52);
    if (-1022..=1023).contains(&e) {
        x * pow2(e)
    } else {
        let half = e / 2;
        x * pow2(half) * pow2(e - half)
    }
}

/// ``x * a * b`` for normal powers of two ``a``, ``b`` without forming an
/// out-of-range intermediate: the exponents are summed, and a sum outside the
/// normal range is applied as two same-sign power-of-two steps, so every
/// intermediate lies between ``|x|`` and the result (finite whenever the
/// result is; zero stays zero). In range this is one exact power-of-two
/// multiplication, bit-identical to ``x * (a * b)``.
fn unscale_pow2(x: Complex64, a: f64, b: f64) -> Complex64 {
    mul_pow2(x, pow2_exponent(a) + pow2_exponent(b))
}

/// ``x * 2**e`` without an out-of-range intermediate: one exact power-of-two
/// multiplication when ``e`` is a normal exponent, else two same-sign
/// power-of-two steps (each intermediate lies between ``|x|`` and the result).
fn mul_pow2(x: Complex64, e: i32) -> Complex64 {
    let pow2 = |k: i32| f64::from_bits(((1023 + k) as u64) << 52);
    if (-1022..=1023).contains(&e) {
        x * pow2(e)
    } else {
        let half = e / 2;
        x * pow2(half) * pow2(e - half)
    }
}

/// Range-safe complex division ``a / b`` (issue #127).
///
/// ``num_complex``'s ``a / b`` forms ``|b|**2`` and ``a * conj(b)`` unscaled:
/// both overflow once ``|b|`` (or ``|a| |b|``) exceeds ~1.3e154 and underflow
/// below ~1e-154, so a finite quotient such as ``ainv_u / R`` with
/// ``|R| ~ 5e161`` came out ``NaN``. Here ``a`` and ``b`` are first brought to
/// a max-component magnitude in ``[0.5, 1)`` by exact powers of two
/// (``pow2_scale``), the small quotient is formed with the same
/// ``num_complex`` arithmetic, and the exponent difference is restored by
/// [`mul_pow2`]. The result is finite whenever the exact quotient is
/// representable. When the unscaled operation stays in the normal range every
/// step is an exact power-of-two rescaling of it, so the result is
/// bit-identical to ``a / b``. A zero or non-finite operand falls back to
/// ``a / b`` (same IEEE ``inf`` / ``NaN`` semantics).
pub fn div_range_safe(a: Complex64, b: Complex64) -> Complex64 {
    let a_max = a.re.abs().max(a.im.abs());
    let b_max = b.re.abs().max(b.im.abs());
    if !(a_max.is_finite() && b_max.is_finite() && a_max > 0.0 && b_max > 0.0) {
        return a / b;
    }
    let (sa, sb) = (pow2_scale(a_max), pow2_scale(b_max));
    let q = (a * sa) / (b * sb);
    // a / b = (a sa / (b sb)) * sb / sa.
    mul_pow2(q, pow2_exponent(sb) - pow2_exponent(sa))
}

/// Complex analogue of ``determinant::refresh_running_inverse_if_needed``.
pub fn refresh_running_inverse_if_needed_complex(
    g: &Array2<Complex64>,
    gi: &mut Array2<Complex64>,
    threshold: f64,
) -> bool {
    if !threshold.is_finite() {
        return false;
    }
    if running_inverse_probe_residual_complex(&g.view(), &gi.view()) > threshold {
        // Issues #127 / #605: the guarded equilibrated LU inverse, falling back
        // per walker to the equilibrated pseudo-inverse (finite on a singular
        // carried G), the same rule as the Python carried-inverse refresh.
        *gi = geminal_inverse_guarded_complex(&g.view());
        true
    } else {
        false
    }
}

/// Complex matrix inverse via Gauss-Jordan with partial pivoting.
pub fn inverse_complex(a_in: &Array2<Complex64>) -> Array2<Complex64> {
    let n = a_in.nrows();
    let mut a = a_in.to_owned();
    let mut inv = Array2::<Complex64>::eye(n);
    for k in 0..n {
        let mut piv = k;
        let mut best = a[[k, k]].norm();
        for i in (k + 1)..n {
            let v = a[[i, k]].norm();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if piv != k {
            for j in 0..n {
                let ta = a[[k, j]];
                a[[k, j]] = a[[piv, j]];
                a[[piv, j]] = ta;
                let ti = inv[[k, j]];
                inv[[k, j]] = inv[[piv, j]];
                inv[[piv, j]] = ti;
            }
        }
        let d = a[[k, k]];
        // Issue #164: range-safe (bit-identical to `/` in range); `num_complex`
        // `/` forms `|d|^2` unscaled and gives NaN/inf for |d| < ~1.5e-154.
        for j in 0..n {
            a[[k, j]] = crate::complex_div::div_range_safe(a[[k, j]], d);
            inv[[k, j]] = crate::complex_div::div_range_safe(inv[[k, j]], d);
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let f = a[[i, k]];
            if f == Complex64::new(0.0, 0.0) {
                continue;
            }
            for j in 0..n {
                let akj = a[[k, j]];
                let ikj = inv[[k, j]];
                a[[i, j]] -= f * akj;
                inv[[i, j]] -= f * ikj;
            }
        }
    }
    inv
}

/// Complex determinant as a unit phase and a logarithmic magnitude.
///
/// Reference mesh ratios need the quotient even when both determinants lie
/// outside the f64 range. Keep the magnitude in log space until the ratio is
/// formed, just as the real `det_sign_lnabs` route does. LU elimination uses
/// range-safe complex division independently of the determinant magnitude.
pub fn det_phase_lnabs(a_in: &Array2<Complex64>) -> (Complex64, f64) {
    let n = a_in.nrows();
    let mut a = a_in.to_owned();
    let mut phase = Complex64::new(1.0, 0.0);
    let mut ln_abs = 0.0_f64;
    for k in 0..n {
        let mut piv = k;
        let mut best = a[[k, k]].norm();
        for i in (k + 1)..n {
            let value = a[[i, k]].norm();
            if value > best {
                best = value;
                piv = i;
            }
        }
        if piv != k {
            for j in 0..n {
                let value = a[[k, j]];
                a[[k, j]] = a[[piv, j]];
                a[[piv, j]] = value;
            }
            phase = -phase;
        }
        let d = a[[k, k]];
        let scale = d.re.abs().max(d.im.abs());
        if scale == 0.0 {
            return (Complex64::new(0.0, 0.0), f64::NEG_INFINITY);
        }
        let scaled = d / scale;
        let magnitude = scaled.norm();
        phase *= scaled / magnitude;
        ln_abs += scale.ln() + magnitude.ln();
        for i in (k + 1)..n {
            let factor = crate::complex_div::div_range_safe(a[[i, k]], d);
            for j in (k + 1)..n {
                let akj = a[[k, j]];
                a[[i, j]] -= factor * akj;
            }
        }
    }
    (phase, ln_abs)
}

/// Quotient of phase/log determinants, without materializing either amplitude.
pub fn det_ratio_complex(numerator: (Complex64, f64), denominator: (Complex64, f64)) -> Complex64 {
    crate::complex_div::div_range_safe(numerator.0, denominator.0)
        * (numerator.1 - denominator.1).exp()
}

/// Complex `(gradlnPsi, grad^2lnPsi)` det field per electron, mirror of the real
/// `compute_grads_and_laplacian_ln_det`. `grad^2lndet = tr - (gradlndet)^2` (complex
/// square). Returns `(grad_up, grad_dn, lap_up, lap_dn)`.
#[allow(clippy::type_complexity)]
pub fn grads_and_laplacian_ln_det_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
) {
    // Fresh-inverse entry (1:1 the real `compute_grads_and_laplacian_ln_det`):
    // build the complex G and invert it here, then delegate to the shared body.
    //
    // G comes from the FUSED-VALUE tables (`value_grad_lap`, whose gto_vgl body
    // runs in the fp64 `ao_grad_lap` zone and rounds only the completed value),
    // NOT from `ComplexOrb::value`, whose `ao_full` performs zone-local fp32
    // arithmetic under a mixed `ao_eval` policy. Same rule as the real entry:
    // a full-f32-arithmetic G fed into `inverse_complex` makes the mixed-precision
    // kinetic deviate from fp64 far more than XLA, whose kinetic builds G from
    // the fp64-bodied fused tables (determinant.py; the deviation bisects to the
    // ao_eval zone). Bit-for-bit neutral in full precision, where both value
    // kernels share one fp64 body. Grad tables computed here are discarded and
    // recomputed by the shared body; this fresh entry is off the per-move sweep.
    let (od, _gd, _ld) = gem.orb_dn.value_grad_lap(r_dn);
    let (ou, _gu, _lu) = gem.orb_up.value_grad_lap(r_up);
    let w = gem.w_from_dn(&od, r_up.nrows());
    let g = ou.t().dot(&w);
    let g_inv = inverse_complex(&g);
    grads_and_laplacian_ln_det_complex_with_inv(gem, r_up, r_dn, &g_inv.view())
}

/// Complex det grad/laplacian reusing a CALLER-supplied complex geminal inverse
/// `g_inv == G(r_up, r_dn)^-1` -- the complex sibling of the real
/// `compute_grads_and_laplacian_ln_det_with_inv`. Skips
/// the fresh LU so the complex-twist OBSERVABLE e_L reuses the running complex SM
/// inverse the MCMC path carries (the fresh `inverse_complex` NaNs at a
/// near-singular geminal; the AS-regularized reused inverse is stable). No
/// `via_single` flag is needed here (unlike the real periodic path): the complex
/// batch AO (`ComplexOrb::value` / `value_grad_lap`) already home-cell-wraps
/// INTERNALLY via its `WrapCtx`, so the AO VALUE (which the reused `g_inv` was
/// built from, JAX `compute_geminal_all_elements`) and the AO grad/laplacian are
/// the SAME full-image convention -- verified far-from-home vs JAX kinetic.
#[allow(clippy::type_complexity)]
pub fn grads_and_laplacian_ln_det_complex_with_inv(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: &ArrayView2<Complex64>,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
) {
    use numpy::ndarray::Array1;
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let (ou, gou, oul) = gem.orb_up.value_grad_lap(r_up); // (orb_up, n_up)
    let (od, god, odl) = gem.orb_dn.value_grad_lap(r_dn); // (orb_dn, n_dn)
    let g_inv = g_inv.to_owned();
    let orb_up = ou.nrows();
    // FULL complex lambda: the det drift/laplacian core is the primary kinetic-drift seam
    // for the complex-`lambda` AGP twist; `Re(lambda)+0i` here would silently drop `Im(lambda)`.
    let m = gem.w_from_dn(&od, n_up);
    let mg = m.dot(&g_inv); // (orb_up, n_up)
    let gu = ou.dot(&g_inv.t()); // (orb_up, n_up)
                                 // lambda_p * O_dn grad/lap (orb_up, n_dn)
    let pdn_g = [
        gem.paired_times(&god[0]),
        gem.paired_times(&god[1]),
        gem.paired_times(&god[2]),
    ];
    let pdn_lap = gem.paired_times(&odl);
    let zero = Complex64::new(0.0, 0.0);
    let mut grad_up = Array2::<Complex64>::zeros((n_up, 3));
    let mut lap_up = Array1::<Complex64>::zeros(n_up);
    for i in 0..n_up {
        let mut gc = [zero; 3];
        let mut tr = zero;
        for a in 0..orb_up {
            let mgi = mg[[a, i]];
            for c in 0..3 {
                gc[c] += gou[c][[a, i]] * mgi;
            }
            tr += oul[[a, i]] * mgi;
        }
        for c in 0..3 {
            grad_up[[i, c]] = gc[c];
        }
        lap_up[i] = tr - (gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]);
    }
    let mut grad_dn = Array2::<Complex64>::zeros((n_dn, 3));
    let mut lap_dn = Array1::<Complex64>::zeros(n_dn);
    for i in 0..n_dn {
        let mut gc = [zero; 3];
        let mut tr = zero;
        for a in 0..orb_up {
            let gui = gu[[a, i]];
            for c in 0..3 {
                gc[c] += pdn_g[c][[a, i]] * gui;
            }
            tr += pdn_lap[[a, i]] * gui;
        }
        for c in 0..3 {
            grad_dn[[i, c]] = gc[c];
        }
        lap_dn[i] = tr - (gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]);
    }
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Stack two `(m,N)` real AO blocks vertically (a on top of b) -> `(m_a+m_b, N)`.
fn vstack_r(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ma, mb, n) = (a.nrows(), b.nrows(), a.ncols());
    let mut out = Array2::<f64>::zeros((ma + mb, n));
    out.slice_mut(numpy::ndarray::s![0..ma, ..]).assign(a);
    out.slice_mut(numpy::ndarray::s![ma.., ..]).assign(b);
    out
}

/// Stack the GTO block over the PW block in `gto_first` order.
fn vstack_r_o(gto: &Array2<f64>, pw: &Array2<f64>, gto_first: bool) -> Array2<f64> {
    if gto_first {
        vstack_r(gto, pw)
    } else {
        vstack_r(pw, gto)
    }
}

/// `A = Re(complex composite J3 AO)` = `Re(gto) (+) Re(pw)` in `gto_first` order,
/// mirroring the verified determinant `ComplexOrb::ao_full`. `pw = None` (every
/// shipped J3 fixture, pure sphe-GTO) -> GTO-only.
/// The pw branch is structurally identical to the determinant PW concat (no
/// pgtopw+J3 fixture exists to exercise it directly).
pub(crate) fn j3_ao_re(j3: &ComplexOrb, r: &ArrayView2<f64>) -> Array2<f64> {
    // ORBITAL rank (`ComplexOrb::value`), then `Re` per element -- jqmc's
    // `jnp.real(compute_orb(orb_data, r))`. `value` owns every J3 leaf: single complex
    // pGTO, composite pGTO(+)PW / pGTO(+)B-spline, PURE complex plane-wave / B-spline
    // (0-AO gto placeholder), and the MO projection on top -- applied BEFORE `Re`, since
    // `Re(C)Re(AO) != Re(C*AO)`. An AO-rank form would drop the MO layer and the
    // B-spline block entirely, so a pure MO-wrapped complex B-spline J3 leaf would give a
    // 0-row `A` and the `(orb, orb+1)` j_matrix contraction would index out of bounds.
    j3.value(r).mapv(|x| x.re)
}

/// One column of [`j3_ao_re`] at a single position (n_orb_j3,). The J3 leaf convention
/// (NO MO transform, pw appended in `gto_first` order) is the one
/// [`jastrow_value_complex`] consumes, so the rank-1 J3 ratio built from these columns
/// reproduces `exp(J3(r') - J3(r))` of the reference exactly.
pub(crate) fn j3_ao_re_single(j3: &ComplexOrb, pos: [f64; 3]) -> Vec<f64> {
    let r = Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap();
    j3_ao_re(j3, &r.view()).column(0).to_vec()
}

/// Full real Jastrow value `J = J1 + J2 + J3` at a config for a complex-twist wf:
/// J1/J2 via the periodic-Williamson real value kernels; J3 via `A=Re(complex sphe
/// AO)` bilinear. Used by the complex ECP-with-Jastrow ratio `exp(J(r')-J(r))`.
pub fn jastrow_value_complex(
    j1: Option<&crate::data::JastrowOneBodyData>,
    j2: Option<&crate::data::JastrowTwoBodyData>,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    dtn: Option<&crate::dtn::JastrowDtnData>,
) -> f64 {
    let mut v = 0.0;
    let f32_arithmetic = j1
        .map(|j| j.precision_policy.is_f32("jastrow_eval"))
        .or_else(|| j2.map(|j| j.precision_policy.is_f32("jastrow_eval")))
        .or_else(|| j3c.map(|j| j.precision_policy.is_f32("jastrow_eval")))
        .or_else(|| dtn.map(|d| d.precision_policy.is_f32("jastrow_eval")))
        // No Jastrow term at all: nothing is accumulated, so the policy is moot.
        // The process-wide standalone setting is never consulted.
        .unwrap_or(false);
    let add = |acc: f64, x: f64| -> f64 {
        if f32_arithmetic {
            ((acc as f32) + (x as f32)) as f64
        } else {
            acc + x
        }
    };
    if let Some(j1) = j1 {
        v = add(
            v,
            crate::jastrow_factor::compute_jastrow_one_body(j1, r_up, r_dn),
        );
    }
    if let Some(j2) = j2 {
        v = add(
            v,
            crate::jastrow_factor::compute_jastrow_two_body(j2, r_up, r_dn),
        );
    }
    if let (Some(j3), Some(jm)) = (j3c, j_matrix) {
        // A = Re(complex composite AO); the pw child (if any) is appended in gto_first
        // order, mirroring the verified determinant `ComplexOrb::ao_full`. pw=None keeps
        // the GTO-only path byte-identical (all shipped J3 fixtures are pure sphe-GTO).
        let au = j3_ao_re(j3, r_up);
        let ad = j3_ao_re(j3, r_dn);
        v = add(
            v,
            crate::jastrow_factor::compute_jastrow_three_body_value_with_policy(
                jm,
                &au.view(),
                &ad.view(),
                f32_arithmetic,
            ),
        );
    }
    // DTN is a real distance-based Jastrow -- it factors out of the complex twist
    // exactly like J1/J2/J3 (real `exp(J)`), so it enters the value unchanged.
    if let Some(dtn) = dtn {
        v = add(v, crate::dtn::compute_dtn_value(dtn, r_up, r_dn));
    }
    v
}

/// Complex Bloch-sphe J3 kinetic FORCE (dT_J3/dR, scattered to J3 home atoms). 1:1
/// real `forces::j3_side` (== `forces_pbc::j3_pbc_side_w`) but every AO quantity is
/// `Re(complex sphe AO)` (value/grad/lap/Hessian/gradlap); the Jastrow is real so Re.
/// bar_gj = -(gJ+Re gD). Per-electron kinetic weight `w` (global index; the GFMC S'
/// mask): J3 is bilinear in `A`, so electron `k`'s kinetic localizes to AO column `k` ->
/// weight the `process()` cotangents (bcoef/bgo/blo) by `w_k`, propagated through the
/// M-coupled `bar_a`. `w=None` => all-1 (the full dT_J3/dR, the VMC force).
#[allow(clippy::too_many_arguments)]
pub fn j3_side_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bar_gj: &dyn Fn(usize, usize) -> f64,
    nuc: &[usize],
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    use numpy::ndarray::{s, Axis};
    // J3 uses a phase-free periodic reference, independently of the determinant
    // twist. Its finite image stencil is valid in the home cell only. Normalize
    // once before ALL value/gradient/Hessian fields; wrapping only the value
    // leaves nuclear forces inconsistent with the electron derivatives.
    let wrapped_up = j3.wrap.as_ref().map(|ctx| ctx.wrap(r_up).0);
    let wrapped_dn = j3.wrap.as_ref().map(|ctx| ctx.wrap(r_dn).0);
    let r_up = &wrapped_up
        .as_ref()
        .map_or_else(|| r_up.view(), |r| r.view());
    let r_dn = &wrapped_dn
        .as_ref()
        .map_or_else(|| r_dn.view(), |r| r.view());
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    let re3 = |m: &[Array2<Complex64>; 3]| [re(&m[0]), re(&m[1]), re(&m[2])];
    // ORBITAL-rank path (any MO layer and/or a B-spline child): every cotangent below is built at
    // orbital rank and MO-projected to the AO rank at the nuclear scatter.
    let orbital_rank = j3.j3_orbital_rank();
    // COMPOSITE pgto(+)pw J3: append the PW child's Re(value/grad/lap) to the AO basis in
    // gto_first order (j_matrix couples all rows), but the PW child is NOT atom-centered
    // -> zero nuclear force, so only the pGTO rows are scattered (translation identity).
    // Hessian/gradlap stay GTO-only (num_gto rows). `pw = None` => byte-identical single leaf.
    let num_pw = j3.pw.as_ref().map_or(0, |p| p.num_ao);
    let gto0 = if j3.gto_first { 0 } else { num_pw };
    let aos_re = |r: &ArrayView2<f64>| {
        if orbital_rank {
            let (v, g, l) = j3.value_grad_lap(r);
            return (re(&v), re3(&g), re(&l));
        }
        let (v, gx, gy, gz, l) = compute_aos_vgl_complex(&j3.aos, &j3.image_phase, r);
        let (vg, gxg, gyg, gzg, lg) = (re(&v), re(&gx), re(&gy), re(&gz), re(&l));
        match &j3.pw {
            None => (vg, [gxg, gyg, gzg], lg),
            Some(pw) => {
                let (pv, pg, pl) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
                let (pv, pgx, pgy, pgz, pl) =
                    (re(&pv), re(&pg[0]), re(&pg[1]), re(&pg[2]), re(&pl));
                let gf = j3.gto_first;
                (
                    vstack_r_o(&vg, &pv, gf),
                    [
                        vstack_r_o(&gxg, &pgx, gf),
                        vstack_r_o(&gyg, &pgy, gf),
                        vstack_r_o(&gzg, &pgz, gf),
                    ],
                    vstack_r_o(&lg, &pl, gf),
                )
            }
        }
    };
    let (a_up, go_u, lo_u) = aos_re(r_up);
    let (a_dn, go_d, lo_d) = aos_re(r_dn);
    // AO-rank path only (orbital rank scatters via `j3.aos.num_ao`); saturate so a dense MO
    // (n_orb < num_pw) cannot underflow.
    let num_gto = a_up.nrows().saturating_sub(num_pw);
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
    let coef =
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
    let coef_up = coef(&a_up, &ku, &m_s_dn);
    let coef_dn = coef(&a_dn, &kd, &mt_s_up);
    let process = |coef: &Array2<f64>,
                   go: &[Array2<f64>; 3],
                   lo: &Array2<f64>,
                   spin_up: bool|
     -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        let nc = coef.ncols();
        let off = if spin_up { 0 } else { n_up };
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
                    acc += bar_gj(off + k, d) * go[d][[a, k]];
                    bgo[d][[a, k]] = wk * bar_gj(off + k, d) * coef[[a, k]];
                }
                bcoef[[a, k]] = wk * (acc - 0.5 * lo[[a, k]]);
                blo[[a, k]] = -0.5 * wk * coef[[a, k]];
            }
        }
        (bcoef, bgo, blo)
    };
    let (bcoef_u, bgo_u, blo_u) = process(&coef_up, &go_u, &lo_u, true);
    let (bcoef_d, bgo_d, blo_d) = process(&coef_dn, &go_d, &lo_d, false);
    let bcoef_d_sum = bcoef_d.sum_axis(Axis(1));
    let bcoef_u_sum = bcoef_u.sum_axis(Axis(1));
    let bar_a = |bc: &Array2<f64>,
                 k: &Array2<f64>,
                 add: &numpy::ndarray::Array1<f64>,
                 n: usize|
     -> Array2<f64> {
        let mut g = m.t().dot(bc).dot(&k.t()) + m.dot(bc).dot(k);
        for a in 0..orb_n {
            for c in 0..n {
                g[[a, c]] += add[a];
            }
        }
        g
    };
    let bar_a_up = bar_a(&bcoef_u, &ku, &m.dot(&bcoef_d_sum), n_up);
    let bar_a_dn = bar_a(&bcoef_d, &kd, &m.t().dot(&bcoef_u_sum), n_dn);
    // Scatter only the pGTO rows (local `ag`, global `gto0+ag`): cotangents (ba/bgo/blo/go)
    // index the FULL composite row; H/GL are GTO-only. PW rows have dAO/dR = 0.
    let force_side = |ba: &Array2<f64>,
                      bgo: &[Array2<f64>; 3],
                      blo: &Array2<f64>,
                      go: &[Array2<f64>; 3],
                      h: &[[Array2<f64>; 3]; 3],
                      gl: &[Array2<f64>; 3],
                      out: &mut Array2<f64>| {
        let nc = ba.ncols();
        for ag in 0..num_gto {
            let g = gto0 + ag;
            let ia = nuc[ag];
            for d in 0..3 {
                let mut acc = 0.0;
                for k in 0..nc {
                    acc += ba[[g, k]] * go[d][[g, k]];
                    for e in 0..3 {
                        acc += bgo[e][[g, k]] * h[d][e][[ag, k]];
                    }
                    acc += blo[[g, k]] * gl[d][[ag, k]];
                }
                out[[ia, d]] += -acc;
            }
        }
    };
    if orbital_rank {
        // MO-project the real orbital-rank cotangents to AO rank (`C^T bar`, complex) and scatter
        // the gto rows [g0, g0+num_ao) against the complex gto fields, `Re` LAST (`A = Re(C AO)`
        // => `bar . dA/dR = Re((C^T bar) . dAO/dR)`). The PW / B-spline rows carry no R dependence.
        let g0 = j3.gto_ao_offset();
        let mut force_side_mo =
            |ba: &Array2<f64>, bgo: &[Array2<f64>; 3], blo: &Array2<f64>, r: &ArrayView2<f64>| {
                if j3.aos.num_ao == 0 {
                    return;
                }
                let (_v, gx, gy, gz, _l) = j3.gto_vgl(r);
                let go = [gx, gy, gz];
                let h = j3.ao_hessian(r);
                let gl = j3.ao_gradlap(r);
                let pba = j3.mo_project_t(ba);
                let pbgo = [
                    j3.mo_project_t(&bgo[0]),
                    j3.mo_project_t(&bgo[1]),
                    j3.mo_project_t(&bgo[2]),
                ];
                let pblo = j3.mo_project_t(blo);
                for a in 0..j3.aos.num_ao {
                    let row = g0 + a;
                    let ia = nuc[a];
                    for d in 0..3 {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for k in 0..ba.ncols() {
                            acc += pba[[row, k]] * go[d][[a, k]];
                            for e in 0..3 {
                                acc += pbgo[e][[row, k]] * h[d][e][[a, k]];
                            }
                            acc += pblo[[row, k]] * gl[d][[a, k]];
                        }
                        out[[ia, d]] -= acc.re;
                    }
                }
            };
        force_side_mo(&bar_a_up, &bgo_u, &blo_u, r_up);
        force_side_mo(&bar_a_dn, &bgo_d, &blo_d, r_dn);
    } else {
        let hu = {
            let h = crate::complex_ao::compute_aos_hessian_complex(&j3.aos, &j3.image_phase, r_up);
            [re3(&h[0]), re3(&h[1]), re3(&h[2])]
        };
        let hd = {
            let h = crate::complex_ao::compute_aos_hessian_complex(&j3.aos, &j3.image_phase, r_dn);
            [re3(&h[0]), re3(&h[1]), re3(&h[2])]
        };
        let glu = re3(&crate::complex_ao::compute_aos_gradlap_complex(
            &j3.aos,
            &j3.image_phase,
            r_up,
        ));
        let gld = re3(&crate::complex_ao::compute_aos_gradlap_complex(
            &j3.aos,
            &j3.image_phase,
            r_dn,
        ));
        force_side(&bar_a_up, &bgo_u, &blo_u, &go_u, &hu, &glu, out);
        force_side(&bar_a_dn, &bgo_d, &blo_d, &go_d, &hd, &gld, out);
    }
}

/// Complex Bloch-sphe J3 ELECTRON-axis kinetic force `dT_J3/dr` (accumulate into
/// `dt (Ne,3)`), per-electron weighted `w` (the GFMC S' mask; `w=None` => full). 1:1 the
/// real `forces_pbc::j3_pbc_egrad_side` but every AO quantity is `Re(complex sphe AO)`
/// (value/grad/lap/Hessian/gradlap) -- the J3 side of the complex-twist continuum kinetic
/// force on the electron axis (the piece VMC never needs, only the diffusion-MC force does).
/// `cot` is the FULL drift `gJ + Re(gD)` per electron.
#[allow(clippy::too_many_arguments)]
pub fn j3_egrad_side_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cot: &Array2<f64>,
    w: Option<&[f64]>,
    dt: &mut Array2<f64>,
) {
    use numpy::ndarray::{s, Axis};
    // ORBITAL-rank J3 (any MO layer and/or a B-spline child): the leaf-owning orbital-rank reverse
    // covers the gto block too (MO-projected cotangents, every child's fields, `Re` last).
    if j3.j3_orbital_rank() {
        j3_leaf_egrad_side_complex(j3, j_matrix, r_up, r_dn, cot, w, dt);
        return;
    }
    // Match the nuclear J3 reverse and the wrapped forward orbital evaluator.
    // The shared phase-free J3 basis has unit Bloch restore factors.
    let wrapped_up = j3.wrap.as_ref().map(|ctx| ctx.wrap(r_up).0);
    let wrapped_dn = j3.wrap.as_ref().map(|ctx| ctx.wrap(r_dn).0);
    let r_up = &wrapped_up
        .as_ref()
        .map_or_else(|| r_up.view(), |r| r.view());
    let r_dn = &wrapped_dn
        .as_ref()
        .map_or_else(|| r_dn.view(), |r| r.view());
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    let re3 = |m: &[Array2<Complex64>; 3]| [re(&m[0]), re(&m[1]), re(&m[2])];
    let re3x3 = |h: &[[Array2<Complex64>; 3]; 3]| [re3(&h[0]), re3(&h[1]), re3(&h[2])];
    // COMPOSITE pgto(+)pw J3 (twist): the AO value/grad/lap concatenate the PW child in
    // `gto_first` order (j_matrix couples ALL rows), mirroring `j3_side_complex`. `pw = None`
    // (single complex-pGTO leaf) => byte-identical GTO-only.
    let num_pw = j3.pw.as_ref().map_or(0, |p| p.num_ao);
    let aos_re = |r: &ArrayView2<f64>| {
        let (v, gx, gy, gz, l) = compute_aos_vgl_complex(&j3.aos, &j3.image_phase, r);
        let (vg, gxg, gyg, gzg, lg) = (re(&v), re(&gx), re(&gy), re(&gz), re(&l));
        match &j3.pw {
            None => (vg, [gxg, gyg, gzg], lg),
            Some(pw) => {
                let (pv, pg, pl) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
                let (pv, pgx, pgy, pgz, pl) =
                    (re(&pv), re(&pg[0]), re(&pg[1]), re(&pg[2]), re(&pl));
                let gf = j3.gto_first;
                (
                    vstack_r_o(&vg, &pv, gf),
                    [
                        vstack_r_o(&gxg, &pgx, gf),
                        vstack_r_o(&gyg, &pgy, gf),
                        vstack_r_o(&gzg, &pgz, gf),
                    ],
                    vstack_r_o(&lg, &pl, gf),
                )
            }
        }
    };
    let hess_gto = |r: &ArrayView2<f64>| {
        let h = j3.ao_hessian(r);
        [re3(&h[0]), re3(&h[1]), re3(&h[2])]
    };
    let gradlap_gto = |r: &ArrayView2<f64>| re3(&j3.ao_gradlap(r));
    let (a_up, go_u, lo_u) = aos_re(r_up);
    let (a_dn, go_d, lo_d) = aos_re(r_dn);
    let orb_n = a_up.nrows();
    let num_gto = orb_n - num_pw;
    let gto0 = if j3.gto_first { 0 } else { num_pw };
    let pw0 = if j3.gto_first { num_gto } else { 0 };
    // FULL-rank (orb_n) second-derivative fields: GTO rows carry the pGTO Hessian / gradlap; PW
    // rows carry the PW Hessian and the PW gradlap `d(grad^2 phi)/dr = -g2 * grad phi` (phi =
    // cos(G*r - phase)). The value-gradient (dAO/dr) is `go` for BOTH children (already concat).
    let build_fields = |hgto: &[[Array2<f64>; 3]; 3],
                        glgto: &[Array2<f64>; 3],
                        go: &[Array2<f64>; 3],
                        r: &ArrayView2<f64>|
     -> ([[Array2<f64>; 3]; 3], [Array2<f64>; 3]) {
        let nc = go[0].ncols();
        let z = || Array2::<f64>::zeros((orb_n, nc));
        let mut hf = [[z(), z(), z()], [z(), z(), z()], [z(), z(), z()]];
        let mut glf = [z(), z(), z()];
        for ag in 0..num_gto {
            let a = gto0 + ag;
            for c in 0..3 {
                for e in 0..nc {
                    glf[c][[a, e]] = glgto[c][[ag, e]];
                    for d in 0..3 {
                        hf[c][d][[a, e]] = hgto[c][d][[ag, e]];
                    }
                }
            }
        }
        if let Some(pw) = &j3.pw {
            let hpw = re3x3(&crate::complex_ao::compute_pw_hessian_complex(pw, r));
            for ap in 0..num_pw {
                let a = pw0 + ap;
                let gk = pw.gk[ap];
                let g2 = gk[0] * gk[0] + gk[1] * gk[1] + gk[2] * gk[2];
                for c in 0..3 {
                    for e in 0..nc {
                        glf[c][[a, e]] = -g2 * go[c][[a, e]];
                        for d in 0..3 {
                            hf[c][d][[a, e]] = hpw[c][d][[ap, e]];
                        }
                    }
                }
            }
        }
        (hf, glf)
    };
    let (hu, glu) = build_fields(&hess_gto(r_up), &gradlap_gto(r_up), &go_u, r_up);
    let (hd, gld) = build_fields(&hess_gto(r_dn), &gradlap_gto(r_dn), &go_d, r_dn);
    let (n_up, n_dn) = (a_up.ncols(), a_dn.ncols());
    let ku = crate::stress_pbc::k_strict_lower(n_up);
    let kd = crate::stress_pbc::k_strict_lower(n_dn);
    let m = j_matrix.slice(s![.., 0..orb_n]).to_owned();
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

/// Complex-twist J3 kinetic ELECTRON-force SELF term at ORBITAL rank: a PURE (MO-wrapped)
/// plane-wave / B-spline complex J3 leaf, and (via [`j3_egrad_side_complex`]) every J3 whose
/// orbital index is not its AO index (any MO layer or B-spline child, issues #89/#119), whose
/// gto block is contracted here too -- for a pure leaf -- the complex twin of the real
/// [`crate::gfmc_forces::j3_leaf_egrad_side`], and the leaf-owning analog of
/// [`j3_egrad_side_complex`] (which covers only the atom-centered GTO AO block `j3.aos`).
///
/// The J3 bilinear cotangents (`bcoef`=value / `bgo`=grad / `blo`=lap, plus the `mk`
/// M-contraction + cross-spin `bar_s`) are LEAF-INDEPENDENT: they are built from the Re-projected
/// ORBITAL-rank value/grad/lap ([`ComplexOrb::value_grad_lap`], the SAME evaluator the J3 drift
/// [`j3_grads_lap_complex`] uses -- MO-wrapping aware) and the full drift `cot = gJ + Re(gD)`,
/// arithmetic identical to `j3_egrad_side_complex`. The value/lap/grad orbital cotangents are then
/// MO-projected to AO rank (`mo^T`) and contracted with the leaf's complex second-derivative
/// fields (grad AO, grad(grad^2 AO), Hessian) supplied by the SAME `det_egrad_fields` /
/// complex-PW branch the complex determinant kinetic force ([`complex_det_kinetic_grads`]) uses --
/// `Re` taken LAST (the J3 cotangents are REAL, so `Re(cm * field) = Re(mo^T bar) . Re(field)`
/// holds and no Re(A)Re(B) cross term is dropped). The NUCLEAR (dR) side is 0 for these
/// atom-center-free leaves (dAO/dR = 0); the J3 dR contribution rides the J1/J2/det cross terms
/// via the full drift `cot` in the caller.
#[allow(clippy::too_many_arguments)]
pub fn j3_leaf_egrad_side_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cot: &Array2<f64>,
    w: Option<&[f64]>,
    dt: &mut Array2<f64>,
) {
    use numpy::ndarray::{s, Array1, Axis};
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    // ORBITAL-rank value/grad/lap (Re), MO-wrapping aware -- 1:1 `j3_grads_lap_complex`.
    let vgl_re = |r: &ArrayView2<f64>| {
        let (v, g, l) = j3.value_grad_lap(r);
        (re(&v), [re(&g[0]), re(&g[1]), re(&g[2])], re(&l))
    };
    let (a_up, go_u, lo_u) = vgl_re(r_up);
    let (a_dn, go_d, lo_d) = vgl_re(r_dn);
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
    let mk = |a: &Array2<f64>, k: &Array2<f64>, cross: &Array1<f64>| -> Array2<f64> {
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
    // bar_orb (value-cotangent, M-contracted) + cross-spin row-sum bar_s, both at ORBITAL rank.
    let mut bar_orb_u = m.t().dot(&bcoef_u).dot(&ku.t()) + m.dot(&bcoef_u).dot(&ku);
    let mut bar_orb_d = m.t().dot(&bcoef_d).dot(&kd.t()) + m.dot(&bcoef_d).dot(&kd);
    let bar_s_dn = m.t().dot(&bcoef_u.sum_axis(Axis(1))); // up bcoef -> dn electrons
    let bar_s_up = m.dot(&bcoef_d.sum_axis(Axis(1))); // dn bcoef -> up electrons
    for a in 0..orb_n {
        for k in 0..n_up {
            bar_orb_u[[a, k]] += bar_s_up[a];
        }
        for k in 0..n_dn {
            bar_orb_d[[a, k]] += bar_s_dn[a];
        }
    }
    // MO-project the (real) orbital-rank cotangents to AO rank (complex), then contract the leaf's
    // complex second-derivative fields (Re last). `mo == None` => AO rank == orbital rank.
    let cmz = |bar: &Array2<f64>| -> Array2<Complex64> { j3.mo_project_t(bar) };
    // AO-rank row offsets of the gto block and of the extra (PW / B-spline) child. A pure leaf
    // has a 0-AO gto block, so its extra child starts at row 0.
    let g0 = j3.gto_ao_offset();
    let e0 = if j3.gto_first { j3.aos.num_ao } else { 0 };
    let contract = |bar_orb: &Array2<f64>,
                    blo: &Array2<f64>,
                    bgo: &[Array2<f64>; 3],
                    r: &ArrayView2<f64>,
                    off: usize,
                    dt: &mut Array2<f64>| {
        let cm_orb = cmz(bar_orb);
        let cm_lap = cmz(blo);
        let cm_gou = [cmz(&bgo[0]), cmz(&bgo[1]), cmz(&bgo[2])];
        let nc = cm_orb.ncols();
        // gto block of an MO-wrapped or B-spline composite J3: the complex gto fields with the
        // same wrap+restore as the forward `value_grad_lap` (grad, grad(grad^2), Hessian).
        if j3.aos.num_ao > 0 {
            let (_v, gx, gy, gz, _l) = j3.gto_vgl(r);
            let ga = [gx, gy, gz];
            let gl = j3.ao_gradlap(r);
            let h = j3.ao_hessian(r);
            for a in 0..j3.aos.num_ao {
                let row = g0 + a;
                for k in 0..nc {
                    for c in 0..3 {
                        let mut acc =
                            cm_orb[[row, k]] * ga[c][[a, k]] + cm_lap[[row, k]] * gl[c][[a, k]];
                        for d in 0..3 {
                            acc += cm_gou[d][[row, k]] * h[c][d][[a, k]];
                        }
                        dt[[off + k, c]] += acc.re;
                    }
                }
            }
        }
        // B-spline child / leaf: fields via the complex Bloch `det_egrad_fields` (ga=grad,
        // gl=gradlap, h=Hessian), exactly as the determinant kinetic force's B-spline block.
        if let Some(bs) = &j3.bspline {
            let (ga, gl, h) = bs.det_egrad_fields(r);
            for a in 0..bs.num_ao {
                let row = e0 + a;
                for k in 0..nc {
                    for c in 0..3 {
                        let mut acc =
                            cm_orb[[row, k]] * ga[c][[a, k]] + cm_lap[[row, k]] * gl[c][[a, k]];
                        for d in 0..3 {
                            acc += cm_gou[d][[row, k]] * h[c][d][[a, k]];
                        }
                        dt[[off + k, c]] += acc.re;
                    }
                }
            }
        }
        // PW leaf: value grad = `grad`, lap-field = `-g2*grad`, grad-field = Hessian (the complex
        // analog the determinant kinetic force's PW block uses).
        if let Some(pw) = &j3.pw {
            let (_v, grad, _l) = crate::complex_ao::compute_pw_vgl_complex(pw, r);
            let hpw = crate::complex_ao::compute_pw_hessian_complex(pw, r);
            for a in 0..pw.num_ao {
                let row = e0 + a;
                let gk = pw.gk[a];
                let g2 = gk[0] * gk[0] + gk[1] * gk[1] + gk[2] * gk[2];
                for k in 0..nc {
                    for c in 0..3 {
                        let gc = grad[c][[a, k]];
                        let mut acc = cm_orb[[row, k]] * gc + cm_lap[[row, k]] * (-g2 * gc);
                        for d in 0..3 {
                            acc += cm_gou[d][[row, k]] * hpw[c][d][[a, k]];
                        }
                        dt[[off + k, c]] += acc.re;
                    }
                }
            }
        }
    };
    contract(&bar_orb_u, &blo_u, &bgo_u, r_up, 0, dt);
    contract(&bar_orb_d, &blo_d, &bgo_d, r_dn, n_up, dt);
}

/// Complex Bloch-sphe J3 drift/Laplacian (per electron, REAL). The J3 AO basis is
/// `A=Re(complex sphe AO)` (the Jastrow is real); coefA `g[o,n]` (j1 + triangular
/// suffix/prefix + cross-spin row-sum) is the SAME as the real three-body; drift =
/// Sum_o g*gradA, lap = Sum_o g*grad^2A (g-|r_n so no extra term). 1:1 real
/// `compute_grads_and_laplacian_jastrow_three_body` with Re(complex) AOs.
pub fn j3_grads_lap_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (
    Array2<f64>,
    Array2<f64>,
    numpy::ndarray::Array1<f64>,
    numpy::ndarray::Array1<f64>,
) {
    use numpy::ndarray::{s, Array1};
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    // ORBITAL-rank value/grad/laplacian, then `Re` per element -- exactly jqmc's
    // `map(jnp.real, compute_orb_vgl(orb_data, r))` (`jastrow_factor.py`), whose orbital
    // layer is `compute_MOs_*` when the J3 leaf is MO-wrapped. `ComplexOrb::value_grad_lap`
    // owns the whole leaf taxonomy in one place: single complex pGTO, the composite
    // pGTO(+)PW / pGTO(+)B-spline concatenation in `gto_first` order, a PURE complex
    // plane-wave or B-spline leaf (0-AO gto placeholder), and the MO projection on top.
    // Re is taken AFTER the (holomorphic) MO contraction, never before -- projecting the
    // AO block first would compute Re(C)Re(AO) != Re(C*AO) for a genuinely complex MO.
    // The gto block is home-cell wrapped (jqmc `_wrap_into_cell_if_pbc` inside `compute_AOs`).
    let vgl_re = |r: &ArrayView2<f64>| {
        let (v, g, l) = j3.value_grad_lap(r);
        (re(&v), re(&g[0]), re(&g[1]), re(&g[2]), re(&l))
    };
    let (au, aux, auy, auz, aul) = vgl_re(r_up);
    let (ad, adx, ady, adz, adl) = vgl_re(r_dn);
    let n_orb = au.nrows();
    let (n_up, n_dn) = (au.ncols(), ad.ncols());
    let j1: Vec<f64> = (0..n_orb).map(|o| j_matrix[[o, n_orb]]).collect();
    let jm = j_matrix.slice(s![.., 0..n_orb]).to_owned();
    let a_up = jm.dot(&au);
    let at_up = jm.t().dot(&au);
    let a_dn = jm.dot(&ad);
    let at_dn = jm.t().dot(&ad);
    let build_g = |a: &Array2<f64>,
                   at: &Array2<f64>,
                   cross: &Array2<f64>,
                   ne: usize,
                   ne_cross: usize|
     -> Array2<f64> {
        let mut g = Array2::<f64>::zeros((n_orb, ne));
        for o in 0..n_orb {
            let rs_cross: f64 = (0..ne_cross).map(|mm| cross[[o, mm]]).sum();
            let mut suffix = vec![0.0_f64; ne];
            let mut acc = 0.0;
            for n in (0..ne).rev() {
                suffix[n] = acc;
                acc += a[[o, n]];
            }
            let mut pref = 0.0;
            for n in 0..ne {
                g[[o, n]] = j1[o] + suffix[n] + pref + rs_cross;
                pref += at[[o, n]];
            }
        }
        g
    };
    let g_up = build_g(&a_up, &at_up, &a_dn, n_up, n_dn);
    let g_dn = build_g(&a_dn, &at_dn, &at_up, n_dn, n_up);
    let contract = |g: &Array2<f64>,
                    gx: &Array2<f64>,
                    gy: &Array2<f64>,
                    gz: &Array2<f64>,
                    lap: &Array2<f64>,
                    ne: usize|
     -> (Array2<f64>, Array1<f64>) {
        let mut grad = Array2::<f64>::zeros((ne, 3));
        let mut lp = Array1::<f64>::zeros(ne);
        for n in 0..ne {
            let (mut x, mut y, mut z, mut l) = (0.0, 0.0, 0.0, 0.0);
            for o in 0..n_orb {
                let gv = g[[o, n]];
                x += gv * gx[[o, n]];
                y += gv * gy[[o, n]];
                z += gv * gz[[o, n]];
                l += gv * lap[[o, n]];
            }
            grad[[n, 0]] = x;
            grad[[n, 1]] = y;
            grad[[n, 2]] = z;
            lp[n] = l;
        }
        (grad, lp)
    };
    let (gu, lu) = contract(&g_up, &aux, &auy, &auz, &aul, n_up);
    let (gd, ld) = contract(&g_dn, &adx, &ady, &adz, &adl, n_dn);
    (gu, gd, lu, ld)
}

/// Complex kinetic energy WITH Jastrow `T = 1/2(-Sum(grad^2lnPsi_det+grad^2J) - Sum(gradlnPsi_det+gradJ)^2)`
/// (complex). Det field complex (`grads_and_laplacian_ln_det_complex`), Jastrow grad/lap
/// REAL (electrons + params real). j3 optional. 1:1 the real `compute_kinetic_energy`.
pub fn compute_kinetic_withj_complex(
    gem: &ComplexGeminal,
    jastrow: &crate::data::JastrowData,
    j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Complex64 {
    let (gd_up, gd_dn, ld_up, ld_dn) = grads_and_laplacian_ln_det_complex(gem, r_up, r_dn);
    // Fresh-inverse reference path (FD-validated near configs): J3 at the RAW
    // coordinate (`wrap_j3 = false`) keeps it bit-identical to the verified kernel.
    kinetic_withj_complex_assemble(
        &gd_up, &gd_dn, &ld_up, &ld_dn, jastrow, j3, r_up, r_dn, false,
    )
}

/// Complex kinetic energy with Jastrow reusing a caller-supplied complex geminal
/// inverse -- the complex sibling of the real
/// `compute_kinetic_energy_with_inv`, feeding `compute_local_energy_fast_complex`.
/// The determinant grad/laplacian reuses `g_inv` (the MCMC path's running complex
/// inverse) instead of a fresh `inverse_complex` (which NaNs at a near-singular
/// geminal). J1/J2 stay on the RAW coordinate (min-image, jqmc's non-invariant
/// reference); the J3 three-body orbital is home-cell-wrapped (`wrap_j3 = true`)
/// so a far-wandered electron does not collapse the image tiling -- mirroring the
/// real path, and a no-op fold for a Gamma J3 leaf (restore phase == 1).
pub fn compute_kinetic_withj_complex_with_inv(
    gem: &ComplexGeminal,
    jastrow: &crate::data::JastrowData,
    j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: &ArrayView2<Complex64>,
) -> Complex64 {
    let (gd_up, gd_dn, ld_up, ld_dn) =
        grads_and_laplacian_ln_det_complex_with_inv(gem, r_up, r_dn, g_inv);
    kinetic_withj_complex_assemble(
        &gd_up, &gd_dn, &ld_up, &ld_dn, jastrow, j3, r_up, r_dn, true,
    )
}

/// Shared complex kinetic assembly `0.5*(-Sum lap - Sum (grad_det + grad_J)^2)`
/// (complex square) from the complex determinant grad/laplacian fields + the REAL
/// Jastrow grad/laplacian (J1/J2 + optional J3). `wrap_j3` home-cell-wraps the J3
/// coordinate for the OBSERVABLE (reused-inverse) path; the fresh-inverse path
/// passes it RAW so it stays bit-identical to the FD-validated kernel.
#[allow(clippy::too_many_arguments)]
fn kinetic_withj_complex_assemble(
    gd_up: &Array2<Complex64>,
    gd_dn: &Array2<Complex64>,
    ld_up: &numpy::ndarray::Array1<Complex64>,
    ld_dn: &numpy::ndarray::Array1<Complex64>,
    jastrow: &crate::data::JastrowData,
    j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    wrap_j3: bool,
) -> Complex64 {
    use crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part;
    let (mut gj_up, mut gj_dn, mut lj_up, mut lj_dn) =
        compute_grads_and_laplacian_jastrow_part(jastrow, r_up, r_dn);
    if let Some((orb, jm)) = j3 {
        let (rwu, rwd) = if wrap_j3 {
            (
                crate::atomic_orbital::wrap_r(&orb.aos, r_up),
                crate::atomic_orbital::wrap_r(&orb.aos, r_dn),
            )
        } else {
            (r_up.to_owned(), r_dn.to_owned())
        };
        let (g3u, g3d, l3u, l3d) = j3_grads_lap_complex(orb, jm, &rwu.view(), &rwd.view());
        gj_up = gj_up + g3u;
        gj_dn = gj_dn + g3d;
        lj_up = lj_up + l3u;
        lj_dn = lj_dn + l3d;
    }
    kinetic_complex_from_fields(gd_up, gd_dn, ld_up, ld_dn, &gj_up, &gj_dn, &lj_up, &lj_dn)
}

/// The summation tail of the complex kinetic assembly,
/// `0.5*(-Sum (lap_det + lap_J) - Sum (grad_det + grad_J)^2)` (complex square), from
/// the complex determinant fields and the REAL total Jastrow fields. Shared by
/// [`kinetic_withj_complex_assemble`] (fresh fields) and the FAST chunk's carried
/// kinetic tables (`gfmc::ComplexChunkKinCarry`), so both sum in one order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kinetic_complex_from_fields(
    gd_up: &Array2<Complex64>,
    gd_dn: &Array2<Complex64>,
    ld_up: &numpy::ndarray::Array1<Complex64>,
    ld_dn: &numpy::ndarray::Array1<Complex64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    lj_up: &numpy::ndarray::Array1<f64>,
    lj_dn: &numpy::ndarray::Array1<f64>,
) -> Complex64 {
    let (n_up, n_dn) = (gd_up.nrows(), gd_dn.nrows());
    let mut sum_lap = Complex64::new(0.0, 0.0);
    for i in 0..n_up {
        sum_lap += ld_up[i] + lj_up[i];
    }
    for i in 0..n_dn {
        sum_lap += ld_dn[i] + lj_dn[i];
    }
    let mut sum_g2 = Complex64::new(0.0, 0.0);
    for i in 0..n_up {
        for c in 0..3 {
            let v = gd_up[[i, c]] + gj_up[[i, c]];
            sum_g2 += v * v;
        }
    }
    for i in 0..n_dn {
        for c in 0..3 {
            let v = gd_dn[[i, c]] + gj_dn[[i, c]];
            sum_g2 += v * v;
        }
    }
    Complex64::new(0.5, 0.0) * (-sum_lap - sum_g2)
}

/// Complex determinant kinetic energy (jnone) `T = 1/2(-Sumgrad^2lnPsi - Sum(gradlnPsi)^2)`
/// (complex). 1:1 the real `compute_kinetic_energy` with gJ=0, complex square.
pub fn compute_kinetic_det_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Complex64 {
    let (gu, gd, lu, ld) = grads_and_laplacian_ln_det_complex(gem, r_up, r_dn);
    let mut sum_lap = Complex64::new(0.0, 0.0);
    for v in lu.iter().chain(ld.iter()) {
        sum_lap += *v;
    }
    let mut sum_g2 = Complex64::new(0.0, 0.0);
    for grad in [&gu, &gd] {
        for i in 0..grad.nrows() {
            for c in 0..3 {
                sum_g2 += grad[[i, c]] * grad[[i, c]];
            }
        }
    }
    Complex64::new(0.5, 0.0) * (-sum_lap - sum_g2)
}

// ============================================================================
// Complex Sherman-Morrison per-move primitives (complex-twist native sweep, 1a).
// Complex analog of determinant.rs DetRatioState + compute_v/u_move + sm_ratio.
// The streaming state mirrors the real path (ao_up/ao_dn/paired_dn/paired_up_lambda)
// but complex; lambda stays real (cast to complex on use). Verified vs the batch complex
// det ratio det_complex(G_moved)/det_complex(G_base).
// ============================================================================

/// Complex streaming det-ratio state (1:1 DetRatioState, Complex64).
#[derive(Clone)]
pub struct DetRatioStateC {
    pub ao_up: Array2<Complex64>,
    pub ao_dn: Array2<Complex64>,
    pub paired_dn: Array2<Complex64>,
    pub paired_up_lambda: Array2<Complex64>,
}

/// Single complex orbital column at one position (num_orb,). Reuses the verified
/// batch `ComplexOrb::value` on a 1-row input (fold + wrap/restore + PW + MO).
pub fn orb_single_complex(orb: &ComplexOrb, pos: [f64; 3]) -> Vec<Complex64> {
    let r = Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap();
    orb.value(&r.view()).column(0).to_vec()
}

/// Batched complex orbital candidate table `(npts, num_orb)` -- one row per
/// candidate position, orbital-minor. GFMC fixed-mesh callers concatenate
/// per-electron groups so the full table
/// is electron-major candidate-minor). One `ComplexOrb::value` call evaluates
/// the whole block (single real-kernel AVX exp stream + one Bloch fold + one MO
/// GEMM) instead of npts `orb_single_complex` calls. Same kernel family as
/// [`orb_single_complex`]: the per-point AO chain (kernel eval, fold,
/// wrap/restore, PW / B-spline children) is position-independent, and the MO
/// projection is ndarray's generic complex matmul (deterministic per-element
/// fold order, no BLAS dispatch for Complex64), so a table row carries the
/// per-point column's fp chain.
pub(crate) fn complex_candidate_batch(
    orb: &ComplexOrb,
    positions: &[[f64; 3]],
) -> Array2<Complex64> {
    let n = positions.len();
    let mut r = Array2::<f64>::zeros((n, 3));
    for (i, p) in positions.iter().enumerate() {
        r[[i, 0]] = p[0];
        r[[i, 1]] = p[1];
        r[[i, 2]] = p[2];
    }
    let v = orb.value(&r.view()); // (num_orb, npts)
    let mut out = Array2::<Complex64>::zeros((n, v.nrows()));
    for a in 0..v.nrows() {
        for s in 0..n {
            out[[s, a]] = v[[a, s]];
        }
    }
    out
}

/// Reusable scratch of [`ComplexOrb::mesh_wdot_into`] (allocation-free in steady state).
#[derive(Default)]
pub(crate) struct MeshWdotScratch {
    pts: Vec<[f64; 3]>,
    restore: Vec<Complex64>,
    g_re: Vec<f64>,
    g_im: Vec<f64>,
    th: Vec<f64>,
    sn: Vec<f64>,
    cs: Vec<f64>,
    b_re: Vec<f64>,
    b_im: Vec<f64>,
}

/// AO-space contraction weights of the FAST ECP mesh w-dot: rows `0..nrows` of
/// `q C` (`q` = the `G^-1`-folded orbital-space pairing rows, `(>= nrows, num_orb)`;
/// `C` = the leaf's MO coefficients `(num_orb, num_ao_total)`, identity without an
/// MO layer), split into real and imaginary `(nrows, num_ao_total)` row-major
/// tables. Formed in split real arithmetic (four real AXPY streams per pairing
/// row) so the inner loops vectorize.
pub(crate) fn mesh_wdot_weights(
    orb: &ComplexOrb,
    q: &ArrayView2<Complex64>,
    nrows: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n_ao = orb.num_ao_total();
    let mut w_re = Array2::<f64>::zeros((nrows, n_ao));
    let mut w_im = Array2::<f64>::zeros((nrows, n_ao));
    match &orb.mo {
        None => {
            debug_assert_eq!(q.ncols(), n_ao);
            for i in 0..nrows {
                for mu in 0..n_ao {
                    w_re[[i, mu]] = q[[i, mu]].re;
                    w_im[[i, mu]] = q[[i, mu]].im;
                }
            }
        }
        Some(c) => {
            let m = c.nrows();
            debug_assert_eq!(q.ncols(), m);
            // `iter` walks the logical (row-major) order for any memory layout.
            let c_re: Vec<f64> = c.iter().map(|z| z.re).collect();
            let c_im: Vec<f64> = c.iter().map(|z| z.im).collect();
            for i in 0..nrows {
                let mut wr = w_re.row_mut(i);
                let wr = wr.as_slice_mut().expect("weight row contiguous");
                let mut wi_row = w_im.row_mut(i);
                let wi = wi_row.as_slice_mut().expect("weight row contiguous");
                for o in 0..m {
                    let (qr, qi) = (q[[i, o]].re, q[[i, o]].im);
                    if qr == 0.0 && qi == 0.0 {
                        continue;
                    }
                    let (cr, ci) = (
                        &c_re[o * n_ao..(o + 1) * n_ao],
                        &c_im[o * n_ao..(o + 1) * n_ao],
                    );
                    for mu in 0..n_ao {
                        wr[mu] += qr * cr[mu] - qi * ci[mu];
                        wi[mu] += qr * ci[mu] + qi * cr[mu];
                    }
                }
            }
        }
    }
    (w_re, w_im)
}

/// Batched `Re(complex orbital)` J3 candidate table `(npts, n_orb_j3)` -- rows
/// are the exact [`j3_ao_re_single`] columns (same `j3_ao_re` chain on a batch),
/// so a rank-1 `j3_ratio` off a table row reproduces the per-point complex J3
/// ratio chain.
pub(crate) fn j3_re_candidate_batch(j3: &ComplexOrb, positions: &[[f64; 3]]) -> Array2<f64> {
    let n = positions.len();
    let mut r = Array2::<f64>::zeros((n, 3));
    for (i, p) in positions.iter().enumerate() {
        r[[i, 0]] = p[0];
        r[[i, 1]] = p[1];
        r[[i, 2]] = p[2];
    }
    let v = j3_ao_re(j3, &r.view()); // (n_orb_j3, npts)
    let mut out = Array2::<f64>::zeros((n, v.nrows()));
    for a in 0..v.nrows() {
        for s in 0..n {
            out[[s, a]] = v[[a, s]];
        }
    }
    out
}

/// `init_det_ratio_streaming_state` (complex). lambda_paired cast to complex.
pub fn init_det_ratio_state_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DetRatioStateC {
    let ao_up = gem.orb_up.value(r_up); // (orb_num_up, N_up)
    let ao_dn = gem.orb_dn.value(r_dn); // (orb_num_dn, N_dn)
    let paired_dn = if gem.is_sd() {
        ao_dn.clone()
    } else {
        gem.paired_times(&ao_dn)
    };
    let paired_up_lambda = gem.up_t_times_paired(&ao_up); // (N_up, orb_num_dn)
    DetRatioStateC {
        ao_up,
        ao_dn,
        paired_dn,
        paired_up_lambda,
    }
}

/// `_compute_v_up_move` (complex): row-delta v (N_up,) for an up-electron move.
pub fn compute_v_up_move_complex(
    gem: &ComplexGeminal,
    state: &DetRatioStateC,
    moved_index: usize,
    ao_new: &[Complex64],
) -> Vec<Complex64> {
    let n_orb_up = ao_new.len();
    let delta: Vec<Complex64> = (0..n_orb_up)
        .map(|a| ao_new[a] - state.ao_up[[a, moved_index]])
        .collect();
    let n_dn = state.paired_dn.ncols();
    let mut v = vec![Complex64::new(0.0, 0.0); n_dn];
    for (a, &d) in delta.iter().enumerate().take(state.paired_dn.nrows()) {
        for (j, v_j) in v[..n_dn].iter_mut().enumerate() {
            *v_j += d * state.paired_dn[[a, j]];
        }
    }
    let n_unpaired = gem.n_unpaired();
    for j in 0..n_unpaired {
        let mut s = Complex64::new(0.0, 0.0);
        for (a, &d) in delta.iter().enumerate() {
            s += d * gem.lambda_c_value(a, gem.orb_num_dn + j);
        }
        v.push(s);
    }
    v
}

/// `_compute_u_dn_move` (complex): col-delta u (N_up,) for a dn-electron move.
pub fn compute_u_dn_move_complex(
    state: &DetRatioStateC,
    moved_index: usize,
    ao_new: &[Complex64],
) -> Vec<Complex64> {
    let n_orb_dn = ao_new.len();
    let n_up = state.paired_up_lambda.nrows();
    let mut u = vec![Complex64::new(0.0, 0.0); n_up];
    for (i, u_i) in u[..n_up].iter_mut().enumerate() {
        let mut s = Complex64::new(0.0, 0.0);
        for (a, ao_new_a) in ao_new[..n_orb_dn].iter().enumerate() {
            s += state.paired_up_lambda[[i, a]] * (*ao_new_a - state.ao_dn[[a, moved_index]]);
        }
        *u_i = s;
    }
    u
}

/// Issue #134 (complex): proposed geminal row `G_new[k, :]` of an up move, built
/// directly from the new orbital column (`compute_v_up_move_complex` with
/// `ao_new` in place of the difference), for the AS factor of the proposal.
pub fn compute_up_row_complex(
    gem: &ComplexGeminal,
    state: &DetRatioStateC,
    ao_new: &[Complex64],
) -> Vec<Complex64> {
    let n_dn = state.paired_dn.ncols();
    let mut row = vec![Complex64::new(0.0, 0.0); n_dn];
    for (a, &x) in ao_new.iter().enumerate().take(state.paired_dn.nrows()) {
        for (j, row_j) in row[..n_dn].iter_mut().enumerate() {
            *row_j += x * state.paired_dn[[a, j]];
        }
    }
    let n_unpaired = gem.n_unpaired();
    for j in 0..n_unpaired {
        let mut s = Complex64::new(0.0, 0.0);
        for (a, &x) in ao_new.iter().enumerate() {
            s += x * gem.lambda_c_value(a, gem.orb_num_dn + j);
        }
        row.push(s);
    }
    row
}

/// Issue #134 (complex): proposed geminal column `G_new[:, k]` of a dn move,
/// built directly from the new orbital column.
pub fn compute_dn_col_complex(state: &DetRatioStateC, ao_new: &[Complex64]) -> Vec<Complex64> {
    let n_orb_dn = ao_new.len();
    let n_up = state.paired_up_lambda.nrows();
    let mut col = vec![Complex64::new(0.0, 0.0); n_up];
    for (i, col_i) in col[..n_up].iter_mut().enumerate() {
        let mut s = Complex64::new(0.0, 0.0);
        for (a, ao_new_a) in ao_new[..n_orb_dn].iter().enumerate() {
            s += state.paired_up_lambda[[i, a]] * *ao_new_a;
        }
        *col_i = s;
    }
    col
}

/// Complex Sherman-Morrison det ratio `1 + vTG^-^1u` + the rank-1 factors
/// `ainv_u = G^-^1u`, `vt_ainv = vTG^-^1` (for the in-place inverse update on accept).
pub fn sm_ratio_complex(
    geminal_inv: &ArrayView2<Complex64>,
    u: &[Complex64],
    v: &[Complex64],
    ainv_u: &mut [Complex64],
    vt_ainv: &mut [Complex64],
) -> Complex64 {
    let n = u.len();
    for x in vt_ainv.iter_mut() {
        *x = Complex64::new(0.0, 0.0);
    }
    for i in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for k in 0..n {
            s += geminal_inv[[i, k]] * u[k];
        }
        ainv_u[i] = s;
        let vi = v[i];
        for j in 0..n {
            vt_ainv[j] += vi * geminal_inv[[i, j]];
        }
    }
    let mut ratio = Complex64::new(1.0, 0.0);
    for k in 0..n {
        ratio += v[k] * ainv_u[k];
    }
    ratio
}

/// Apply the accepted complex rank-1 update IN PLACE (complex `sm_apply_inplace`).
#[allow(clippy::too_many_arguments)]
pub fn sm_apply_inplace_complex(
    geminal_inv: &mut Array2<Complex64>,
    geminal: &mut Array2<Complex64>,
    u: &[Complex64],
    v: &[Complex64],
    ainv_u: &[Complex64],
    vt_ainv: &[Complex64],
    ratio: Complex64,
    moved_spin_is_up: bool,
    moved_index: usize,
) {
    let n = u.len();
    // Issue #127: every division by the ratio is range-safe (``div_range_safe``;
    // bit-identical to ``/`` in range): a finite ``|R|`` above ~1e154 made the
    // ``num_complex`` quotient ``NaN``.
    for i in 0..n {
        let ai = div_range_safe(ainv_u[i], ratio);
        for j in 0..n {
            let cur = geminal_inv[[i, j]];
            geminal_inv[[i, j]] = cur - ai * vt_ainv[j];
        }
    }
    // Issue #127: exact moved column / row on a large ratio (complex twin of
    // ``determinant::sm_large_ratio_exact_slice``; no-op below the threshold).
    if ratio.norm() > crate::determinant::SM_LARGE_RATIO_THRESHOLD {
        if moved_spin_is_up {
            for i in 0..n {
                geminal_inv[[i, moved_index]] = div_range_safe(ainv_u[i], ratio);
            }
        } else {
            for j in 0..n {
                geminal_inv[[moved_index, j]] = div_range_safe(vt_ainv[j], ratio);
            }
        }
    }
    if moved_spin_is_up {
        for j in 0..n {
            geminal[[moved_index, j]] += v[j];
        }
    } else {
        for i in 0..n {
            geminal[[i, moved_index]] += u[i];
        }
    }
}

/// `_advance_det_ratio_streaming_state` (complex) after an accepted move: refresh the
/// moved electron's column in ao_up/ao_dn and the dependent paired table.
pub fn advance_det_ratio_state_complex(
    gem: &ComplexGeminal,
    state: &mut DetRatioStateC,
    moved_spin_is_up: bool,
    moved_index: usize,
    ao_col: &[Complex64],
) {
    if moved_spin_is_up {
        // refresh ao_up[:,k] and paired_up_lambda[k,:] = ao_up[:,k]T*lambda_p
        for (a, ao_col_a) in ao_col[..state.ao_up.nrows()].iter().enumerate() {
            state.ao_up[[a, moved_index]] = *ao_col_a;
        }
        for j in 0..gem.orb_num_dn {
            let mut s = Complex64::new(0.0, 0.0);
            for (a, ao_col_a) in ao_col.iter().enumerate() {
                s += *ao_col_a * gem.lambda_c_value(a, j);
            }
            state.paired_up_lambda[[moved_index, j]] = s;
        }
    } else {
        // refresh ao_dn[:,k] and paired_dn[:,k] = lambda_p*ao_dn[:,k]
        for (a, ao_col_a) in ao_col[..state.ao_dn.nrows()].iter().enumerate() {
            state.ao_dn[[a, moved_index]] = *ao_col_a;
        }
        for i in 0..state.paired_dn.nrows() {
            let mut s = Complex64::new(0.0, 0.0);
            for (a, ao_col_a) in ao_col.iter().enumerate() {
                s += gem.lambda_c_value(i, a) * *ao_col_a;
            }
            state.paired_dn[[i, moved_index]] = s;
        }
    }
}

// ============================================================================
// Complex continuum-kinetic derivative cache: the
// Complex64 twin of determinant.rs `DetGradLapCache`. Seeded ONCE per
// projection interval, advanced per hop (one fused single-point column + one
// rank-1 SM update), contracted per sub-step -- replacing the per-sub-step
// `grads_and_laplacian_ln_det_complex` full rebuild (fresh G + LU + two batch
// value_grad_lap sweeps) in the complex GFMC e_L kernels.
//
// SEEDING CONVENTION (mixed-precision policy -- see the doc
// comment in `grads_and_laplacian_ln_det_complex`): EVERY table in this cache
// (the derivative tables, the value tables of `kin_state`, and the G that
// `kin_ginv` inverts) comes from the fused fp64 `ComplexOrb::value_grad_lap`
// family, exactly the evaluation family the FRESH kinetic entry uses. It does
// NOT touch -- and must never be fed from -- the carried `MeshFastStateC`
// value/ginv chain, whose base is the `ao_eval`-zone `ComplexOrb::value`
// single-point path (fp32 zone arithmetic under a mixed policy). Contracting
// fp64-fused derivative tables against that value-chain inverse would reopen
// the exact mixed-lane kinetic seam this convention closes, so the cache
// carries its OWN `kin_ginv` (seeded by inverting the fused-value G, advanced
// by the same conj-free rank-1 SM algebra) and its own fused-value
// `kin_state`. In full precision both value kernels share one fp64 body, so
// the duplication costs memory, not parity.
// ============================================================================

/// Complex det derivative cache: fused-family orbital-rank grad/lap tables
/// (up raw, dn pre-folded through the paired lambda block -- the association
/// of the fresh `_with_inv` contraction), plus the kinetic-quality value state
/// and G-inverse described above.
pub struct DetGradLapCacheC {
    up_x: Array2<Complex64>,
    up_y: Array2<Complex64>,
    up_z: Array2<Complex64>,
    up_lap: Array2<Complex64>,
    /// `lambda_paired * grad/lap(O_dn)` -- the fresh path's `pdn_g` / `pdn_lap`.
    pdn_x: Array2<Complex64>,
    pdn_y: Array2<Complex64>,
    pdn_z: Array2<Complex64>,
    pdn_lap: Array2<Complex64>,
    /// Fused-family value tables (ao_up/ao_dn/paired products), advanced with
    /// the standard complex streaming-state machinery.
    kin_state: DetRatioStateC,
    /// Inverse of the fused-value geminal, SM rank-1 advanced per hop.
    kin_ginv: Array2<Complex64>,
    /// Up-spin contraction weights `m * kin_ginv`, `m = [lambda_paired*O_dn |
    /// lambda_unpaired]`. Seeded by the full product, then carried
    /// by the SAME rank-1 factors that advance `kin_ginv` (plus the moved dn
    /// column of `m`), so it stays in the fp64 kinetic family: O(M N) per hop
    /// instead of the O(M N^2) product per sub-step.
    mg: Array2<Complex64>,
    /// Dn-spin contraction weights `O_up * kin_ginv^T` (plain transpose,
    /// Convention B); carried like `mg` (plus the moved up column of `O_up`).
    gu: Array2<Complex64>,
}

/// `m = [lambda_paired*O_dn | lambda_unpaired]` rebuilt from the maintained
/// `paired_dn` columns -- the value set `w_from_dn` produces on the fresh path
/// (`paired_dn` IS `paired_times(ao_dn)` there, column-advanced here).
fn det_grad_lap_m_complex(gem: &ComplexGeminal, st: &DetRatioStateC) -> Array2<Complex64> {
    let zero = Complex64::new(0.0, 0.0);
    let n_up = st.ao_up.ncols();
    let n_dn = st.ao_dn.ncols();
    let mut m = Array2::<Complex64>::zeros((gem.orb_num_up(), n_up));
    for a in 0..gem.orb_num_up() {
        for j in 0..n_dn {
            m[[a, j]] = if a < st.paired_dn.nrows() {
                st.paired_dn[[a, j]]
            } else {
                zero
            };
        }
        for j in 0..gem.n_unpaired() {
            m[[a, n_dn + j]] = gem.lambda_c_value(a, gem.orb_num_dn + j);
        }
    }
    m
}

/// Build the complex derivative cache at the interval's base configuration.
/// Bit-path note: at build time the contraction
/// [`contract_det_grad_lap_cache_complex`] reproduces the fresh
/// `grads_and_laplacian_ln_det_complex` output exactly (same batch
/// `value_grad_lap` tables, same `w_from_dn`/`inverse_complex` G-inverse, same
/// contraction loops); later sub-steps differ from a fresh rebuild only through
/// the SM-advanced `kin_ginv` and the batch(1)-refreshed columns -- the
/// documented fast-chain fp-floor class.
pub fn init_det_grad_lap_cache_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DetGradLapCacheC {
    let (ou, [gux, guy, guz], oul) = gem.orb_up.value_grad_lap(r_up);
    let (od, [gdx, gdy, gdz], odl) = gem.orb_dn.value_grad_lap(r_dn);
    // G from the fused fp64 value tables (the mixed-precision seeding
    // convention), inverted once per interval.
    let w = gem.w_from_dn(&od, r_up.nrows());
    let g = ou.t().dot(&w);
    let kin_ginv = inverse_complex(&g);
    let pdn_x = gem.paired_times(&gdx);
    let pdn_y = gem.paired_times(&gdy);
    let pdn_z = gem.paired_times(&gdz);
    let pdn_lap = gem.paired_times(&odl);
    let paired_dn = if gem.is_sd() {
        od.clone()
    } else {
        gem.paired_times(&od)
    };
    let paired_up_lambda = gem.up_t_times_paired(&ou);
    let kin_state = DetRatioStateC {
        ao_up: ou,
        ao_dn: od,
        paired_dn,
        paired_up_lambda,
    };
    // The two contraction products, formed once per interval; every later
    // sub-step reads the rank-1-carried copies.
    let mg = det_grad_lap_m_complex(gem, &kin_state).dot(&kin_ginv);
    let gu = kin_state.ao_up.dot(&kin_ginv.t());
    DetGradLapCacheC {
        up_x: gux,
        up_y: guy,
        up_z: guz,
        up_lap: oul,
        pdn_x,
        pdn_y,
        pdn_z,
        pdn_lap,
        kin_state,
        kin_ginv,
        mg,
        gu,
    }
}

/// Advance the complex derivative cache after an accepted hop: ONE fused
/// single-point `value_grad_lap` column (batch-of-1 of the exact seeding
/// kernels -- the per-point AO chain is position-independent, so seed and
/// refresh share one fp chain), written into the moved electron's derivative
/// column, followed by the conj-free rank-1 SM update of `kin_ginv` and the
/// standard streaming-state column refresh. `pos` is RAW (the complex leaves
/// home-cell-wrap internally). NOTE: the value column here is deliberately NOT
/// shared with the carried value-chain advance (`orb_single_complex`) -- the
/// two chains are different fp families by the seeding policy above.
pub fn update_det_grad_lap_cache_complex(
    gem: &ComplexGeminal,
    cache: &mut DetGradLapCacheC,
    is_up: bool,
    electron: usize,
    pos: [f64; 3],
) {
    let r = Array2::from_shape_vec((1, 3), pos.to_vec()).expect("one derivative point");
    let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
    let (v, [gx, gy, gz], l) = orb.value_grad_lap(&r.view());
    if is_up {
        for a in 0..cache.up_x.nrows() {
            cache.up_x[[a, electron]] = gx[[a, 0]];
            cache.up_y[[a, electron]] = gy[[a, 0]];
            cache.up_z[[a, electron]] = gz[[a, 0]];
            cache.up_lap[[a, electron]] = l[[a, 0]];
        }
    } else {
        // Fold the single dn column through the SAME `paired_times` helper the
        // seed used on the batch (generic ndarray matmul -- column-independent
        // fold order, so the refreshed column carries the seed's fp chain).
        let px = gem.paired_times(&gx);
        let py = gem.paired_times(&gy);
        let pz = gem.paired_times(&gz);
        let pl = gem.paired_times(&l);
        for a in 0..cache.pdn_x.nrows() {
            cache.pdn_x[[a, electron]] = px[[a, 0]];
            cache.pdn_y[[a, electron]] = py[[a, 0]];
            cache.pdn_z[[a, electron]] = pz[[a, 0]];
            cache.pdn_lap[[a, electron]] = pl[[a, 0]];
        }
    }
    // Rank-1 SM advance of the kinetic-quality inverse with the fused VALUE
    // column (plain-transpose algebra because the down-spin MO already stores
    // the conjugated `+k` coefficients, term-for-term the value-chain
    // advance in `MeshFastStateC::advance_after_hop`).
    let ao_col: Vec<Complex64> = (0..v.nrows()).map(|a| v[[a, 0]]).collect();
    let zero = Complex64::new(0.0, 0.0);
    let n = cache.kin_ginv.nrows();
    let mut u = vec![zero; n];
    let mut vv = vec![zero; n];
    if is_up {
        u[electron] = Complex64::new(1.0, 0.0);
        let vx = compute_v_up_move_complex(gem, &cache.kin_state, electron, &ao_col);
        vv.copy_from_slice(&vx);
    } else {
        vv[electron] = Complex64::new(1.0, 0.0);
        let ux = compute_u_dn_move_complex(&cache.kin_state, electron, &ao_col);
        u.copy_from_slice(&ux);
    }
    let mut ainv_u = vec![zero; n];
    let mut vt_ainv = vec![zero; n];
    let ratio = sm_ratio_complex(&cache.kin_ginv.view(), &u, &vv, &mut ainv_u, &mut vt_ainv);
    // `kin_ginv' = kin_ginv - a b^T` with `a = kin_ginv u / ratio`,
    // `b^T = v^T kin_ginv`. The carried contraction weights follow with the
    // PRE-hop tables:
    //   mg' = mg - (m a) b^T          [+ dm_col (x) kin_ginv'[e, :] on a dn hop]
    //   gu' = gu - (O_up b) a^T       [+ do_col (x) kin_ginv'[:, e] on an up hop]
    // (`m` changes only in the moved dn column, `O_up` only in the moved up
    // column; both transposes are plain, Convention B.)
    let a_vec: Vec<Complex64> = ainv_u.iter().map(|x| *x / ratio).collect();
    let n_dn = cache.kin_state.ao_dn.ncols();
    let paired_rows = cache.kin_state.paired_dn.nrows();
    let mg_rows = cache.mg.nrows();
    let gu_rows = cache.gu.nrows();
    let mut m_a = vec![zero; mg_rows];
    for (row, out) in m_a.iter_mut().enumerate() {
        let mut acc = zero;
        if row < paired_rows {
            for (j, a_vec_j) in a_vec[..n_dn].iter().enumerate() {
                acc += cache.kin_state.paired_dn[[row, j]] * *a_vec_j;
            }
        }
        for j in 0..gem.n_unpaired() {
            acc += gem.lambda_c_value(row, gem.orb_num_dn + j) * a_vec[n_dn + j];
        }
        *out = acc;
    }
    let mut o_b = vec![zero; gu_rows];
    for (row, out) in o_b.iter_mut().enumerate() {
        let mut acc = zero;
        for (k, vt_ainv_k) in vt_ainv[..n].iter().enumerate() {
            acc += cache.kin_state.ao_up[[row, k]] * *vt_ainv_k;
        }
        *out = acc;
    }
    for (i, a_vec_i) in a_vec[..n].iter().enumerate() {
        let ai = *a_vec_i;
        for (k, vt_ainv_k) in vt_ainv[..n].iter().enumerate() {
            let cur = cache.kin_ginv[[i, k]];
            cache.kin_ginv[[i, k]] = cur - ai * *vt_ainv_k;
        }
    }
    for (row, m_a_row) in m_a[..mg_rows].iter().enumerate() {
        let ma = *m_a_row;
        for (k, vt_ainv_k) in vt_ainv[..n].iter().enumerate() {
            cache.mg[[row, k]] -= ma * *vt_ainv_k;
        }
    }
    for (row, o_b_row) in o_b[..gu_rows].iter().enumerate() {
        let ob = *o_b_row;
        for (k, a_vec_k) in a_vec[..n].iter().enumerate() {
            cache.gu[[row, k]] -= ob * *a_vec_k;
        }
    }
    // The moved electron's own column of `O_up` (up hop) or `m` (dn hop).
    let old_col: Vec<Complex64> = if is_up {
        (0..gu_rows)
            .map(|row| cache.kin_state.ao_up[[row, electron]])
            .collect()
    } else {
        (0..paired_rows)
            .map(|row| cache.kin_state.paired_dn[[row, electron]])
            .collect()
    };
    advance_det_ratio_state_complex(gem, &mut cache.kin_state, is_up, electron, &ao_col);
    if is_up {
        for (row, old_col_row) in old_col[..gu_rows].iter().enumerate() {
            let delta = cache.kin_state.ao_up[[row, electron]] - *old_col_row;
            for k in 0..n {
                cache.gu[[row, k]] += delta * cache.kin_ginv[[k, electron]];
            }
        }
    } else {
        for (row, old_col_row) in old_col[..paired_rows.min(mg_rows)].iter().enumerate() {
            let delta = cache.kin_state.paired_dn[[row, electron]] - *old_col_row;
            for k in 0..n {
                cache.mg[[row, k]] += delta * cache.kin_ginv[[electron, k]];
            }
        }
    }
}

/// Contract the cached derivative tables with the up/dn contraction weights
/// (`mg = m kin_ginv`, `gu = O_up kin_ginv^T`) -- loop-for-loop the body of
/// [`grads_and_laplacian_ln_det_complex_with_inv`].
#[allow(clippy::type_complexity)]
fn contract_det_grad_lap_weights_complex(
    cache: &DetGradLapCacheC,
    mg: &Array2<Complex64>,
    gu: &Array2<Complex64>,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
) {
    use numpy::ndarray::Array1;
    let zero = Complex64::new(0.0, 0.0);
    let n_up = cache.kin_state.ao_up.ncols();
    let n_dn = cache.kin_state.ao_dn.ncols();
    let orb_up = cache.kin_state.ao_up.nrows();
    let mut grad_up = Array2::<Complex64>::zeros((n_up, 3));
    let mut lap_up = Array1::<Complex64>::zeros(n_up);
    for i in 0..n_up {
        let mut gc = [zero; 3];
        let mut tr = zero;
        for a in 0..orb_up {
            let mgi = mg[[a, i]];
            gc[0] += cache.up_x[[a, i]] * mgi;
            gc[1] += cache.up_y[[a, i]] * mgi;
            gc[2] += cache.up_z[[a, i]] * mgi;
            tr += cache.up_lap[[a, i]] * mgi;
        }
        for c in 0..3 {
            grad_up[[i, c]] = gc[c];
        }
        lap_up[i] = tr - (gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]);
    }
    let mut grad_dn = Array2::<Complex64>::zeros((n_dn, 3));
    let mut lap_dn = Array1::<Complex64>::zeros(n_dn);
    for i in 0..n_dn {
        let mut gc = [zero; 3];
        let mut tr = zero;
        for a in 0..orb_up {
            let gui = gu[[a, i]];
            gc[0] += cache.pdn_x[[a, i]] * gui;
            gc[1] += cache.pdn_y[[a, i]] * gui;
            gc[2] += cache.pdn_z[[a, i]] * gui;
            tr += cache.pdn_lap[[a, i]] * gui;
        }
        for c in 0..3 {
            grad_dn[[i, c]] = gc[c];
        }
        lap_dn[i] = tr - (gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2]);
    }
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// carried-inverse probe residual of the cache's own kinetic-quality inverse `kin_ginv`
/// against the geminal its fused-value `kin_state` implies
/// ([`det_state_inverse_probe_residual_complex`], O(N^2 + N M), no `G` formed).
/// Lets a long-lived carrier (the FAST chunk) re-anchor the rank-1 chain.
pub fn det_grad_lap_cache_inverse_residual_complex(
    gem: &ComplexGeminal,
    cache: &DetGradLapCacheC,
) -> f64 {
    det_state_inverse_probe_residual_complex(gem, &cache.kin_state, &cache.kin_ginv.view())
}

/// Contract the cached tables into the per-electron complex det grad/lap with
/// the CARRIED contraction weights (`mg`, `gu`): O(M N) per sub-step, no
/// matrix product. [`contract_det_grad_lap_cache_complex_matmul`] is the
/// matrix-product reference for the same cache.
#[allow(clippy::type_complexity)]
pub fn contract_det_grad_lap_cache_complex(
    gem: &ComplexGeminal,
    cache: &DetGradLapCacheC,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
) {
    debug_assert_eq!(cache.pdn_x.nrows(), gem.orb_num_up());
    contract_det_grad_lap_weights_complex(cache, &cache.mg, &cache.gu)
}

/// Matrix-product reference of [`contract_det_grad_lap_cache_complex`]: forms
/// `m kin_ginv` and `O_up kin_ginv^T` from the cache's current value state and
/// inverse (the per-sub-step matrix-product route). Used by the carried-weight
/// test probe; not on the production path.
#[allow(clippy::type_complexity)]
pub fn contract_det_grad_lap_cache_complex_matmul(
    gem: &ComplexGeminal,
    cache: &DetGradLapCacheC,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
) {
    let mg = det_grad_lap_m_complex(gem, &cache.kin_state).dot(&cache.kin_ginv);
    let gu = cache.kin_state.ao_up.dot(&cache.kin_ginv.t());
    contract_det_grad_lap_weights_complex(cache, &mg, &gu)
}

// ============================================================================
// Complex J3 derivative cache: the `Re(complex AO)` twin
// of jastrow_factor.rs `J3GradLapCache` + `J3RatioState`. Same seeding policy
// as `DetGradLapCacheC`: every table (derivative rows AND the value tables the
// kinetic contraction weights are built from) comes from the fused fp64
// `ComplexOrb::value_grad_lap` family -- the exact family the fresh
// `j3_grads_lap_complex` uses (Re taken AFTER the holomorphic MO contraction).
// The carried `MeshFastStateC::j3_state` (the RATIO chain, seeded from the
// `ao_eval`-zone `j3_ao_re`) is untouched; this cache carries its OWN
// fp64-policy `J3RatioState` for the kinetic q-weights.
// ============================================================================

/// Fused-family J3 derivative rows (electron-major, real) + the kinetic-quality
/// value state whose w/cross tables build the contraction weights.
pub struct J3GradLapCacheC {
    cache: crate::jastrow_factor::J3GradLapCache,
    kin_state: crate::jastrow_factor::J3RatioState,
}

/// Seed the complex J3 derivative cache at the interval's base configuration.
/// The per-sub-step contraction ([`contract_j3_grad_lap_cache_complex`]) uses
/// the q-vector association of the real fast path
/// (`contract_j3_grad_lap_cache`), which differs from the fresh
/// `j3_grads_lap_complex` suffix/prefix association at the fp-floor class the
/// real chain already carries -- fast-chain tolerance contract, not bitwise.
pub fn init_j3_grad_lap_cache_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> J3GradLapCacheC {
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    let vgl = |r: &ArrayView2<f64>| {
        let (v, g, l) = j3.value_grad_lap(r);
        (re(&v), re(&g[0]), re(&g[1]), re(&g[2]), re(&l))
    };
    let (au, aux, auy, auz, aul) = vgl(r_up);
    let (ad, adx, ady, adz, adl) = vgl(r_dn);
    // fp64 policy (`false`): the kinetic weights mirror the fresh fused chain,
    // NOT the ratio chain's `jastrow_ratio` zone policy.
    let kin_state = crate::jastrow_factor::init_j3_ratio_state_with_policy(
        j_matrix,
        &au.view(),
        &ad.view(),
        false,
    );
    let t = |m: &Array2<f64>| m.t().as_standard_layout().to_owned();
    let cache = crate::jastrow_factor::j3_grad_lap_cache_from_tables(
        t(&aux),
        t(&auy),
        t(&auz),
        t(&aul),
        t(&adx),
        t(&ady),
        t(&adz),
        t(&adl),
    );
    J3GradLapCacheC { cache, kin_state }
}

/// Per-hop refresh: one fused single-point `value_grad_lap` column (batch-of-1
/// of the seeding kernels), Re'd after the MO contraction, written into the
/// moved electron's derivative row; the kinetic value state advances with the
/// same column's values through the standard rank-1 machinery.
pub fn update_j3_grad_lap_cache_complex(
    j3: &ComplexOrb,
    j_matrix: &ArrayView2<f64>,
    c: &mut J3GradLapCacheC,
    is_up: bool,
    electron: usize,
    pos: [f64; 3],
) {
    let r = Array2::from_shape_vec((1, 3), pos.to_vec()).expect("one J3 derivative point");
    let (v, [gx, gy, gz], l) = j3.value_grad_lap(&r.view());
    let col =
        |m: &Array2<Complex64>| -> Vec<f64> { (0..m.nrows()).map(|a| m[[a, 0]].re).collect() };
    crate::jastrow_factor::j3_grad_lap_cache_set_row(
        &mut c.cache,
        is_up,
        electron,
        &col(&gx),
        &col(&gy),
        &col(&gz),
        &col(&l),
    );
    let vcol = col(&v);
    crate::jastrow_factor::advance_j3_ratio_state(
        j_matrix,
        &mut c.kin_state,
        is_up,
        electron,
        &vcol,
    );
}

/// Contract the cached complex J3 tables (real result -- the J3 kinetic term is
/// the `Re(complex AO)` bilinear).
pub fn contract_j3_grad_lap_cache_complex(
    j_matrix: &ArrayView2<f64>,
    c: &J3GradLapCacheC,
) -> (
    Array2<f64>,
    Array2<f64>,
    numpy::ndarray::Array1<f64>,
    numpy::ndarray::Array1<f64>,
) {
    crate::jastrow_factor::contract_j3_grad_lap_cache(j_matrix, &c.kin_state, &c.cache)
}

/// AS regularization factor `(S*F)^(-3/8)` for a COMPLEX geminal (real result).
/// `F = ||G^-^1||^2_F = Sum|G^-^1_ij|^2`, `S = min(min row |*|^2, min col |*|^2)` of G -- the
/// complex analog (|x|^2 = x.norm_sqr()) of `determinant::compute_as_regularization_factor`.
pub fn compute_as_regularization_factor_complex(
    geminal: &ArrayView2<Complex64>,
    geminal_inv: &ArrayView2<Complex64>,
) -> f64 {
    let theta = 3.0 / 8.0;
    let n = geminal.nrows();
    let m = geminal.ncols();
    let mut f = 0.0_f64;
    for x in geminal_inv.iter() {
        f += x.norm_sqr();
    }
    let mut min_row = f64::INFINITY;
    let mut col = vec![0.0_f64; m];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..m {
            let c2 = geminal[[i, j]].norm_sqr();
            s += c2;
            col[j] += c2;
        }
        if s < min_row {
            min_row = s;
        }
    }
    let min_col = col.iter().cloned().fold(f64::INFINITY, f64::min);
    let s_fac = min_row.min(min_col);
    let sf = s_fac * f;
    // Issue #131: linear only where exact to rounding, else scaled logarithms.
    if s_fac >= crate::determinant::AS_LINEAR_MIN_SQ_NORM
        && f.is_finite()
        && sf.is_finite()
        && sf > 0.0
    {
        sf.powf(-theta)
    } else {
        let gi = geminal_inv;
        crate::determinant::as_factor_scaled(
            n,
            m,
            |i, j| geminal[[i, j]].norm(),
            gi.len(),
            |t| gi[[t / gi.ncols(), t % gi.ncols()]].norm(),
        )
    }
}

#[cfg(test)]
mod mesh_wdot_tests {
    use super::*;
    use crate::atomic_orbital::AoKind;

    fn unit(theta: f64) -> Complex64 {
        Complex64::new(0.0, theta).exp()
    }

    /// Synthetic complex leaf: optional image-tiled GTO block (with per-kernel
    /// Bloch phases and an optional wrap context), optional PW / B-spline child,
    /// optional complex MO layer of `n_mo` rows.
    fn leaf(
        gto: Option<(AoKind, usize)>,
        extra: &str,
        gto_first: bool,
        wrap_k: Option<[f64; 3]>,
        n_mo: Option<usize>,
    ) -> ComplexOrb {
        let aos = match gto {
            Some((kind, n_image)) => {
                crate::atomic_orbital::tests::vgl_test_data_l(kind, n_image, 9.0, true)
            }
            None => AosGtoData::empty(),
        };
        let image_phase: Vec<Complex64> = (0..aos.num_ao_kernel)
            .map(|k| unit(0.41 * k as f64 - 0.2))
            .collect();
        let wrap = wrap_k.map(|k_frac| {
            let cell = [[9.0, 0.0, 0.0], [0.4, 7.0, 0.0], [0.0, 0.3, 8.0]];
            crate::complex_ao::WrapCtx {
                cell,
                inv_cell: inv_3x3(&cell),
                k_frac,
            }
        });
        let (pw, bspline) = match extra {
            "pw-twist" | "pw-gamma" => {
                let gk: Vec<[f64; 3]> = (0..9)
                    .map(|g| {
                        let t = g as f64;
                        [
                            0.7 * (t - 4.0) + 0.05,
                            0.45 * ((g * 5 % 7) as f64 - 3.0),
                            0.3 * ((g * 3 % 5) as f64 - 2.0),
                        ]
                    })
                    .collect();
                let phase_is_sin =
                    (extra == "pw-gamma").then(|| (0..9).map(|g| (g % 2) as f64).collect());
                (
                    Some(crate::complex_ao::ComplexPwData {
                        num_ao: gk.len(),
                        gk,
                        phase_is_sin,
                        precision_policy: crate::PrecisionPolicy::full(),
                    }),
                    None,
                )
            }
            "bspline" => (
                None,
                Some(crate::bspline::synthetic_complex_bspline_for_tests(
                    6,
                    0x005e_ed77,
                    [0.21, -0.13, 0.08],
                    false,
                )),
            ),
            "bspline-conj" => (
                None,
                Some(crate::bspline::synthetic_complex_bspline_for_tests(
                    6,
                    0x005e_ed78,
                    [0.21, -0.13, 0.08],
                    true,
                )),
            ),
            _ => (None, None),
        };
        let mut orb = ComplexOrb {
            precision_policy: crate::PrecisionPolicy::full(),
            aos,
            image_phase,
            mo: None,
            pw,
            gto_first,
            wrap,
            bspline,
            tiled: None,
        };
        let n_ao = orb.num_ao_total();
        orb.mo = n_mo.map(|m| {
            Array2::from_shape_fn((m, n_ao), |(o, mu)| {
                let a = ((o * 31 + mu * 17) % 13) as f64 - 6.0;
                Complex64::new(0.11 * a, 0.07 * (((o * 7 + mu * 3) % 9) as f64 - 4.0))
            })
        });
        orb
    }

    #[test]
    fn mesh_wdot_matches_candidate_batch_ratio() {
        // The FAST AO-space w-dot affine term (q C).AO(r) must equal the candidate
        // table route q . value(r) (complex_candidate_batch) up to (a) the FAST
        // value screen -- every dropped kernel AO value is below
        // FAST_AO_SCREEN_ABS_BOUND, weighted by |(q C)_home| -- and (b) the fast
        // exp / reassociation, reduction-order class relative to the
        // absolute-term sum of the bilinear form.
        let bound = crate::central_setting_f64("FAST_AO_SCREEN_ABS_BOUND");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        // Mixed precision: fp32 `ao_eval` (plane-wave / B-spline child values) with the
        // Bloch gto block in its fp64 `ao_bloch_eval` zone, as the production complex
        // pGTO leaves are built. The candidate table then carries the zone-local fp32
        // child values and the w-dot must contract those same values.
        let mixed = crate::PrecisionPolicy {
            dtypes: crate::PRECISION_ZONES.map(|zone| match zone {
                "ao_eval" | "jastrow_eval" | "jastrow_grad_lap" => crate::RealDType::F32,
                _ => crate::RealDType::F64,
            }),
        };
        for (mode, policy) in [("full", crate::PrecisionPolicy::full()), ("mixed", mixed)] {
            let with_policy = |mut orb: ComplexOrb| {
                orb.precision_policy = policy;
                orb.aos.precision_policy = policy;
                if orb.aos.num_ao > 0 {
                    orb.aos.value_zone = "ao_bloch_eval";
                }
                if let Some(pw) = orb.pw.as_mut() {
                    pw.precision_policy = policy;
                }
                orb
            };
            let cases: Vec<(&str, ComplexOrb)> = vec![
                (
                    "sphe+pw-twist",
                    leaf(
                        Some((AoKind::Sphe, 6)),
                        "pw-twist",
                        true,
                        Some([0.25, 0.1, -0.2]),
                        Some(5),
                    ),
                ),
                (
                    "pw-gamma+cart",
                    leaf(
                        Some((AoKind::Cart, 6)),
                        "pw-gamma",
                        false,
                        Some([0.0, 0.0, 0.0]),
                        Some(5),
                    ),
                ),
                (
                    "cart-bare",
                    leaf(Some((AoKind::Cart, 1)), "", true, None, None),
                ),
                (
                    "sphe+bspline",
                    leaf(
                        Some((AoKind::Sphe, 6)),
                        "bspline",
                        true,
                        Some([0.25, 0.1, -0.2]),
                        Some(4),
                    ),
                ),
                (
                    "bspline-conj+cart",
                    leaf(
                        Some((AoKind::Cart, 6)),
                        "bspline-conj",
                        false,
                        Some([0.1, 0.0, 0.3]),
                        Some(4),
                    ),
                ),
                ("bspline-pure", leaf(None, "bspline", true, None, None)),
                ("pw-pure", leaf(None, "pw-twist", true, None, Some(3))),
            ]
            .into_iter()
            .map(|(name, orb)| (name, with_policy(orb)))
            .collect();
            for (name, orb) in &cases {
                assert!(
                    orb.mesh_wdot_ok(),
                    "{mode} {name}: w-dot route must cover the leaf"
                );
                let m = orb.num_orb();
                let n_rows = 3;
                let q = Array2::from_shape_fn((n_rows, m), |(i, a)| {
                    Complex64::new(
                        0.3 * (((i * 5 + a * 3) % 7) as f64 - 3.0),
                        0.2 * (((i + 2 * a) % 5) as f64 - 2.0),
                    )
                });
                let (w_re, w_im) = mesh_wdot_weights(orb, &q.view(), n_rows);
                // 40 points (two kernel blocks) on spheres around the atoms, some shifted
                // out of the home cell so the wrap-restore phase is exercised.
                let pts: Vec<[f64; 3]> = (0..40)
                    .map(|v| {
                        let (th, ph) = (0.3 + 0.17 * v as f64, 0.9 * v as f64);
                        let rad = 0.4 + 0.08 * (v % 11) as f64;
                        let shift = [
                            9.0 * ((v % 3) as f64 - 1.0),
                            0.0,
                            -8.0 * ((v % 4 == 0) as i32 as f64),
                        ];
                        let c = if v % 2 == 0 {
                            [0.1, -0.2, 0.3]
                        } else {
                            [1.4, 0.9, -0.7]
                        };
                        [
                            c[0] + rad * th.sin() * ph.cos() + shift[0],
                            c[1] + rad * th.sin() * ph.sin() + shift[1],
                            c[2] + rad * th.cos() + shift[2],
                        ]
                    })
                    .collect();
                let table = complex_candidate_batch(orb, &pts);
                let ra = Array2::from_shape_fn((pts.len(), 3), |(v, k)| pts[v][k]);
                let ao = orb.ao(&ra.view()); // (n_ao_total, npts), for the error scale
                let mut scratch = MeshWdotScratch::default();
                let mut got = vec![Complex64::new(0.0, 0.0); pts.len()];
                let (mut n_restored, mut n_dropped) = (0usize, 0usize);
                for i in 0..n_rows {
                    orb.mesh_wdot_into(
                        w_re.row(i).as_slice().unwrap(),
                        w_im.row(i).as_slice().unwrap(),
                        &pts,
                        &mut scratch,
                        &mut got,
                    );
                    let gto_off = orb.gto_ao_offset();
                    let n_gto = orb.aos.num_ao;
                    let wsum: f64 = (0..orb.aos.num_ao_kernel)
                        .map(|k| {
                            Complex64::new(
                                w_re[[i, gto_off + orb.aos.kernel_to_home[k]]],
                                w_im[[i, gto_off + orb.aos.kernel_to_home[k]]],
                            )
                            .norm()
                        })
                        .sum();
                    for (v, p) in pts.iter().enumerate() {
                        let want: Complex64 = (0..m).map(|a| q[[i, a]] * table[[v, a]]).sum();
                        let scale: f64 = (0..orb.num_ao_total())
                            .map(|mu| {
                                let cq: f64 = match &orb.mo {
                                    Some(mo_c) => (0..m)
                                        .map(|orb_ix| {
                                            q[[i, orb_ix]].norm() * mo_c[[orb_ix, mu]].norm()
                                        })
                                        .sum(),
                                    None => q[[i, mu]].norm(),
                                };
                                cq * ao[[mu, v]].norm()
                            })
                            .sum();
                        let tol = bound * wsum + rtol * scale;
                        assert!(
                            (got[v] - want).norm() <= tol,
                            "{mode} {name} row {i} v {v}: {} vs {want} tol {tol}",
                            got[v]
                        );
                        if n_gto > 0 {
                            let pw = match &orb.wrap {
                                Some(w) => {
                                    let (rw, ph) = w.wrap_point(*p);
                                    n_restored += usize::from(
                                        (ph - Complex64::new(1.0, 0.0)).norm() > 1.0e-3,
                                    );
                                    rw
                                }
                                None => *p,
                            };
                            for sh in &orb.aos.shells {
                                let r2: f64 = (0..3).map(|c| (pw[c] - sh.center[c]).powi(2)).sum();
                                n_dropped +=
                                    usize::from(sh.pz_val * r2 > crate::atomic_orbital::LEPSBAS);
                            }
                        }
                    }
                    // Non-vacuity: the affine term is not negligible.
                    let max = got.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
                    assert!(
                        max > 1.0e-2,
                        "{name} row {i}: vacuous contraction (max {max})"
                    );
                }
                if orb.aos.num_ao > 0 {
                    assert!(n_dropped > 0, "{name}: the value screen must drop shells");
                }
                if orb
                    .wrap
                    .as_ref()
                    .is_some_and(|w| w.k_frac.iter().any(|&k| k != 0.0))
                {
                    assert!(n_restored > 0, "{name}: points must leave the home cell");
                }
            }
        }
    }
}

#[cfg(test)]
mod geminal_guarded_inverse_tests {
    //! Issue #605: the seed / carried-inverse refresh inverse over the pathological-geminal
    //! table, through the production refresh functions (real and complex). A
    //! zero carried inverse makes the carried-inverse refresh trigger deterministically
    //! for every finite G.
    //! Every finite geminal must get a finite inverse; a healthy (invertible,
    //! representable-inverse) geminal must get exactly the equilibrated LU
    //! inverse ``C inv(A) R``; a singular, rank-deficient, non-finite or
    //! out-of-range one exactly the equilibrated pseudo-inverse.
    use super::*;
    use crate::determinant::{
        geminal_pinv_from_matrix, inverse, refresh_running_inverse_if_needed,
    };
    use crate::setting_consts::SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD;

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Want {
        Lu,
        Pinv,
        // Near the guard threshold (residual within ~10x of it): either rule is correct.
        Either,
    }

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// The pathological table of issue #605 (one geminal per case).
    fn table(n: usize, cplx: bool, seed: u64) -> Vec<(&'static str, Want, Array2<Complex64>)> {
        let mut r = rng(seed);
        let mut rnd = |rows: usize, cols: usize| {
            Array2::<Complex64>::from_shape_fn((rows, cols), |_| {
                let re = r();
                Complex64::new(re, if cplx { r() } else { 0.0 })
            })
        };
        let mut out = Vec::new();
        out.push(("control", Want::Lu, rnd(n, n)));
        let mut g = rnd(n, n);
        g.row_mut(3).fill(Complex64::new(0.0, 0.0));
        out.push(("zero row", Want::Pinv, g));
        let mut g = rnd(n, n);
        g.column_mut(5).fill(Complex64::new(0.0, 0.0));
        out.push(("zero column", Want::Pinv, g));
        let mut g = rnd(n, n);
        let row = g.row(2).to_owned();
        g.row_mut(4).assign(&row);
        out.push(("duplicate row", Want::Pinv, g));
        let mut g = rnd(n, n);
        let col = g.column(1).to_owned();
        g.column_mut(6).assign(&col);
        out.push(("duplicate column", Want::Pinv, g));
        let mut g = rnd(n, n);
        let row = g.row(2).mapv(|z| z * 0.3) - g.row(7).mapv(|z| z * 1.7);
        g.row_mut(4).assign(&row);
        out.push(("row = linear combination", Want::Pinv, g));
        let (a, b) = (rnd(n, n - 2), rnd(n - 2, n));
        out.push(("rank N-2 product", Want::Pinv, a.dot(&b)));
        let mut g = rnd(n, n);
        let row = g.row(2).mapv(|z| z * (1.0 + 1e-13));
        g.row_mut(4).assign(&row);
        out.push(("near-duplicate row 1e-13", Want::Pinv, g));
        let mut g = rnd(n, n);
        let noise = rnd(1, n);
        let row = g.row(2).to_owned() + noise.row(0).mapv(|z| z * 1e-9);
        g.row_mut(4).assign(&row);
        out.push(("near-duplicate row 1e-9", Want::Either, g));
        let mut g = rnd(n, n);
        g.row_mut(5).mapv_inplace(|z| z * 1e-200);
        out.push(("graded row 1e-200", Want::Lu, g));
        let mut g = rnd(n, n);
        g.row_mut(5).mapv_inplace(|z| z * 1e-310);
        out.push(("subnormal row 1e-310", Want::Pinv, g));
        // An up and a down electron both far out: the exact inverse entry
        // (9, 5) ~ 1e400 is not a float64, while G is finite.
        let mut g = rnd(n, n);
        g.row_mut(5).mapv_inplace(|z| z * 1e-200);
        g.column_mut(9).mapv_inplace(|z| z * 1e-200);
        out.push(("far up and far down 1e-200", Want::Pinv, g));
        let mut g = rnd(n, n);
        g[[2, 2]] = Complex64::new(f64::NAN, 0.0);
        out.push(("NaN entry", Want::Pinv, g));
        let mut g = rnd(n, n);
        g[[2, 2]] = Complex64::new(f64::INFINITY, 0.0);
        out.push(("inf entry", Want::Pinv, g));
        out.push(("all-zero", Want::Pinv, Array2::zeros((n, n))));
        out
    }

    fn same_bits(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// ``C inv(A) R`` by the production Gauss-Jordan routines, built here from
    /// the equilibration scales recomputed from their definition.
    fn equilibrated_lu_complex(g: &Array2<Complex64>) -> Array2<Complex64> {
        let n = g.nrows();
        let r: Vec<f64> = (0..n)
            .map(|i| pow2_scale(g.row(i).iter().map(|z| z.norm()).fold(0.0, f64::max)))
            .collect();
        let c: Vec<f64> = (0..n)
            .map(|j| {
                pow2_scale(
                    (0..n)
                        .map(|i| (g[[i, j]] * r[i]).norm())
                        .fold(0.0, f64::max),
                )
            })
            .collect();
        let a = Array2::from_shape_fn((n, n), |(i, j)| g[[i, j]] * r[i] * c[j]);
        let y = inverse_complex(&a);
        Array2::from_shape_fn((n, n), |(i, k)| unscale_pow2(y[[i, k]], c[i], r[k]))
    }

    fn equilibrated_lu_real(g: &Array2<f64>) -> Array2<f64> {
        let n = g.nrows();
        let r: Vec<f64> = (0..n)
            .map(|i| pow2_scale(g.row(i).iter().map(|x| x.abs()).fold(0.0, f64::max)))
            .collect();
        let c: Vec<f64> = (0..n)
            .map(|j| pow2_scale((0..n).map(|i| (g[[i, j]] * r[i]).abs()).fold(0.0, f64::max)))
            .collect();
        let a = Array2::from_shape_fn((n, n), |(i, j)| g[[i, j]] * r[i] * c[j]);
        let y = inverse(&a.view());
        Array2::from_shape_fn((n, n), |(i, k)| unscale_pow2_f64(y[[i, k]], c[i], r[k]))
    }

    fn check(
        label: &str,
        want: Want,
        finite_in: bool,
        got: &[(f64, f64)],
        lu: &[(f64, f64)],
        pinv: &[(f64, f64)],
    ) {
        let eq = |x: &[(f64, f64)]| {
            got.iter()
                .zip(x)
                .all(|(a, b)| same_bits(a.0, b.0) && same_bits(a.1, b.1))
        };
        if finite_in {
            assert!(
                got.iter().all(|z| z.0.is_finite() && z.1.is_finite()),
                "{label}: non-finite inverse of a finite G"
            );
        }
        match want {
            Want::Lu => assert!(
                eq(lu),
                "{label}: healthy walker must keep the equilibrated LU inverse"
            ),
            Want::Pinv => assert!(
                eq(pinv),
                "{label}: guarded walker must take the equilibrated pseudo-inverse"
            ),
            Want::Either => assert!(
                eq(lu) || eq(pinv),
                "{label}: neither the LU nor the pseudo-inverse"
            ),
        }
    }

    #[test]
    fn guarded_inverse_over_pathological_table() {
        for (n, cplx) in [(16, false), (80, false), (16, true), (64, true)] {
            for (name, want, g) in table(n, cplx, 0x9e37_79b9_7f4a_7c15 ^ n as u64) {
                let label = format!("N={n} {} {name}", if cplx { "complex" } else { "real" });
                let finite_in = g.iter().all(|z| z.re.is_finite() && z.im.is_finite());
                let flat_c =
                    |m: &Array2<Complex64>| m.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>();
                let flat_r = |m: &Array2<f64>| m.iter().map(|&x| (x, 0.0)).collect::<Vec<_>>();
                // A non-finite G gives a NaN probe residual, which never triggers the
                // carried-inverse refresh (existing convention); its inverse is checked directly.
                if cplx {
                    let mut gi = Array2::<Complex64>::zeros((n, n));
                    if finite_in {
                        let refreshed = refresh_running_inverse_if_needed_complex(
                            &g,
                            &mut gi,
                            SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD,
                        );
                        assert!(
                            refreshed,
                            "{label}: zero carried inverse must trigger the refresh"
                        );
                    } else {
                        gi = geminal_inverse_guarded_complex(&g.view());
                    }
                    let lu = equilibrated_lu_complex(&g);
                    let pinv = geminal_pinv_from_matrix_complex(&g.view());
                    check(
                        &label,
                        want,
                        finite_in,
                        &flat_c(&gi),
                        &flat_c(&lu),
                        &flat_c(&pinv),
                    );
                } else {
                    let gr = g.mapv(|z| z.re);
                    let mut gi = Array2::<f64>::zeros((n, n));
                    if finite_in {
                        let refreshed = refresh_running_inverse_if_needed(
                            &gr.view(),
                            &mut gi,
                            SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD,
                        );
                        assert!(
                            refreshed,
                            "{label}: zero carried inverse must trigger the refresh"
                        );
                    } else {
                        gi = crate::determinant::geminal_inverse_guarded(&gr.view());
                    }
                    let lu = equilibrated_lu_real(&gr);
                    let pinv = geminal_pinv_from_matrix(&gr.view());
                    check(
                        &label,
                        want,
                        finite_in,
                        &flat_r(&gi),
                        &flat_r(&lu),
                        &flat_r(&pinv),
                    );
                }
            }
        }
    }
}
