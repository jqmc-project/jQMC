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

//! Geminal matrix construction + ln|det| (hand-rolled, no BLAS), mirroring
//! `_compute_geminal_all_elements` (determinant.py) and
//! `compute_ln_det_geminal_all_elements` (:2891). Real fp64 path (Gamma).
//!
//!   orb_matrix_up = compute_orb(orb_up, r_up)            (orb_num_up, N_up)
//!   orb_matrix_dn = compute_orb(orb_dn, r_dn)            (orb_num_dn, N_dn)
//!   lambda split at orb_num_dn -> paired (*, orb_num_dn), unpaired (*, N_unpaired)
//!   G_paired   = ouT * (lambda_paired * od)                   (N_up, N_dn)
//!   G_unpaired = ouT * lambda_unpaired                        (N_up, N_unpaired)
//!   G = [G_paired | G_unpaired]                          (N_up, N_up)
//!   ln|det| = Sum ln|U_ii|  (LU with partial pivoting)
//!
//! Per-move ratio / Sherman-Morrison streaming state follows below.

use crate::atomic_orbital::AosGtoData;
use numpy::ndarray::{Array1, Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// An orbital basis used by a geminal spin block: either raw AOs, or MOs
/// (`mo_coefficients @ AO`). Mirrors `compute_orb_api` dispatch.
#[derive(Clone)]
pub struct OrbData {
    pub(crate) precision_policy: crate::PrecisionPolicy,
    pub aos: AosGtoData,
    /// Plane-wave child of a pgto(+)pw composite basis (`None` for pure-pGTO). The
    /// composite concatenates the children along the AO axis; `gto_first` records
    /// whether the gto block precedes the pw block (so the MO-coefficient columns
    /// line up).
    pub pw: Option<crate::atomic_orbital::AosPwData>,
    pub gto_first: bool,
    /// `(num_mo, num_ao)` if this is an MO basis; `None` for a raw AO basis.
    pub mo_coefficients: Option<Array2<f64>>,
    /// Transposed MO coefficients `(num_ao, num_mo)`, row-contiguous -- each AO's
    /// MO-coefficient vector is contiguous, enabling the sparse AXPY MO
    /// projection `out += ao[a]*C_T[a,:]` over only the AOs that survive the
    /// distance cutoff (the molecular analogue of the periodic image cutoff).
    pub mo_coefficients_t: Option<Array2<f64>>,
    /// B-spline AO leaf (`AOs_bspline_data`, periodic real-Gamma). Mutually exclusive with
    /// the GTO/PW leaves; when `Some`, value/grad/lap come from the spline (MO layer,
    /// if any, is applied on top exactly as for GTO -- for the SD bspline it is identity).
    pub bspline: Option<crate::bspline::AosBsplineData>,
}

/// MO projection `out = C[nmo,na] @ ao[na]` via faer (BLAS-class GEMV). C is
/// row-major `(nmo, na)`, which equals a column-major `(na, nmo)` view of `CT`;
/// `out = (CT)T * ao`. ~1.74x over the hand AVX-512 on the 80x1140 kernel.
#[inline]
pub fn project_faer(c: &[f64], ao: &[f64], na: usize, nmo: usize, out: &mut [f64]) {
    let ct = faer::MatRef::from_column_major_slice(c, na, nmo);
    let aov = faer::MatRef::from_column_major_slice(ao, na, 1);
    let mut res = faer::MatMut::from_column_major_slice_mut(out, nmo, 1);
    faer::linalg::matmul::matmul(
        res.as_mut(),
        faer::Accum::Replace,
        ct.transpose(),
        aov,
        1.0_f64,
        faer::Par::Seq,
    );
}

/// MO projection `out[0..nmo] = C[nmo,na] @ ao[na]` -- the real hot kernel.
/// Single portable implementation: per-MO row dot with a 32-wide independent
/// FMA accumulator block (4 x [f64; 8], hiding L2 load latency on the constant
/// C matrix), auto-vectorized under `target-cpu=native`. fp64 throughout;
/// reassociation only.
#[inline]
pub fn project_8acc(c: &[f64], ao: &[f64], na: usize, nmo: usize, out: &mut [f64]) {
    // See the SLP-vectorization rule above dot_product_fast in gfmc.rs: a
    // single chunks_exact loop over ONE flat accumulator, sequential fold,
    // plain scalar tail. 32 lanes -> 4 independent zmm FMA chains on AVX-512
    // (8 ymm on AVX2), hiding L2 load latency on the constant C matrix.
    // 64 lanes regress (ymm bundles on SPR, register spills on AVX2).
    let n32 = na & !31;
    for m in 0..nmo {
        let row = &c[m * na..m * na + na];
        let mut a = [0.0_f64; 32];
        for (rc, ac) in row[..n32].chunks_exact(32).zip(ao[..n32].chunks_exact(32)) {
            for l in 0..32 {
                a[l] = rc[l].mul_add(ac[l], a[l]);
            }
        }
        let mut s = 0.0;
        for a_l in &a {
            s += *a_l;
        }
        for h in n32..na {
            s += row[h] * ao[h];
        }
        out[m] = s;
    }
}

/// Project the value and four AO derivative vectors/// Project the value and four AO derivative vectors through the same MO
/// coefficient matrix.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "scalar inputs of an inlined hot projection helper"
)]
fn project5(
    c: &[f64],
    a0: &[f64],
    a1: &[f64],
    a2: &[f64],
    a3: &[f64],
    a4: &[f64],
    na: usize,
    nmo: usize,
    o0: &mut [f64],
    o1: &mut [f64],
    o2: &mut [f64],
    o3: &mut [f64],
    o4: &mut [f64],
) {
    project_8acc(c, a0, na, nmo, o0);
    project_8acc(c, a1, na, nmo, o1);
    project_8acc(c, a2, na, nmo, o2);
    project_8acc(c, a3, na, nmo, o3);
    project_8acc(c, a4, na, nmo, o4);
}

/// `NotImplementedError` for an `AOs_composite_data` whose children the native
/// orbital layout cannot hold (issue #165). The native real and complex orbitals keep
/// ONE GTO/pGTO block plus at most ONE plane-wave or B-spline block, so a composite
/// with two children of the same family (gto + gto, pw + pw, bspline + bspline) or
/// with a plane-wave AND a B-spline child would silently lose children. Such a
/// composite is refused at parse time with a Python exception raised before any
/// evaluation, instead of a Rust panic or a silently wrong value.
pub(crate) fn unsupported_composite_err(orbital: &str, detail: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "native {orbital} orbital: unsupported AOs_composite_data ({detail}); the native \
         backend supports one GTO/pGTO child plus at most one plane-wave or B-spline child. \
         Use the XLA backend (JQMC_FLAG_RUST_ON_CPU=0) for this AO basis."
    ))
}

/// Integer invariant checked after every native orbital parse (issue #165): the
/// native AO rank plus the rows of children skipped on purpose (complex children the
/// real orbital leaves to the complex path) must equal the Python `num_ao` of the AO
/// basis, and an MO coefficient matrix must have exactly `num_ao` columns. The
/// release-build kernels only `debug_assert` these dimensions, so a mismatch would
/// otherwise panic or contract the wrong AO rows.
pub(crate) fn check_native_ao_rank(
    orbital: &str,
    aos_obj: &Bound<'_, PyAny>,
    native_rows: usize,
    skipped_rows: usize,
    mo_cols: Option<usize>,
) -> PyResult<()> {
    let py_num_ao: usize = aos_obj.getattr("num_ao")?.extract()?;
    if native_rows + skipped_rows != py_num_ao || mo_cols.is_some_and(|c| c != py_num_ao) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "native {orbital} orbital: AO rank mismatch (native {native_rows} + skipped \
             {skipped_rows} rows, Python num_ao {py_num_ao}, MO coefficient columns {mo_cols:?})"
        )));
    }
    Ok(())
}

/// Extract a `(gto, pw, bspline, gto_first, skipped_rows)` tuple from an `aos_data`
/// that is either a single pGTO / B-spline leaf or an `AOs_composite_data` (children
/// concatenated by AO). At most one non-gto "extra" block is present in the
/// composites jqmc builds (pgto+pw OR pgto+bspline), so `pw` and `bspline` are
/// mutually exclusive; `gto_first` orders the gto block against that extra block.
/// Any other composite is refused with [`unsupported_composite_err`]. `skipped_rows`
/// counts the AO rows of complex children skipped on purpose (see below).
#[allow(clippy::type_complexity)]
fn extract_aos(
    aos_obj: &Bound<'_, PyAny>,
) -> PyResult<(
    AosGtoData,
    Option<crate::atomic_orbital::AosPwData>,
    Option<crate::bspline::AosBsplineData>,
    bool,
    usize,
)> {
    if aos_obj.hasattr("children")? {
        let children = aos_obj.getattr("children")?;
        let n: usize = children.len()?;
        let mut gto: Option<AosGtoData> = None;
        let mut pw: Option<crate::atomic_orbital::AosPwData> = None;
        let mut bspline: Option<crate::bspline::AosBsplineData> = None;
        let (mut gto_idx, mut extra_idx) = (0usize, 0usize);
        let mut skipped_rows = 0usize;
        for i in 0..n {
            let child = children.get_item(i)?;
            if child.hasattr("phase_is_sin")? {
                if pw.is_some() || bspline.is_some() {
                    return Err(unsupported_composite_err(
                        "real",
                        "more than one plane-wave/B-spline child",
                    ));
                }
                pw = Some(crate::atomic_orbital::AosPwData::from_py(&child)?);
                extra_idx = i;
            } else if child.hasattr("spline_coefficients")? {
                extra_idx = i;
                if child.get_type().name()?.to_string().contains("complex") {
                    // A COMPLEX twist B-spline child (`AOs_bspline_complex_data`) has no
                    // real kernel. Like the complex plane-wave child below it is owned by
                    // the complex observable path (`ComplexOrb`), so skip it -> the real
                    // OrbData carries the gto child only (bspline = None) (issue #88).
                    skipped_rows += child.getattr("num_ao")?.extract::<usize>()?;
                    continue;
                }
                if pw.is_some() || bspline.is_some() {
                    return Err(unsupported_composite_err(
                        "real",
                        "more than one plane-wave/B-spline child",
                    ));
                }
                // Real-Gamma B-spline child of a pgto+bspline composite. The spline
                // block concatenates after (or before) the gto block exactly like the
                // pw child.
                bspline = Some(crate::bspline::AosBsplineData::from_py(&child)?);
            } else if child.get_type().name()?.to_string().contains("plane_wave") {
                // A COMPLEX plane-wave child (`AOs_plane_wave_complex_data`, no real
                // sin/cos `phase_is_sin`) has no real PW kernel. It reaches the real
                // `OrbData` only as a J3 presence gate for the complex-twist path (whose
                // kernels consume the parallel `ComplexOrb`, not this real leaf), so skip
                // it -> the real OrbData carries the gto child only (pw = None).
                extra_idx = i;
                skipped_rows += child.getattr("num_ao")?.extract::<usize>()?;
                continue;
            } else {
                if gto.is_some() {
                    return Err(unsupported_composite_err(
                        "real",
                        "more than one GTO/pGTO child",
                    ));
                }
                gto = Some(AosGtoData::from_py(&child)?);
                gto_idx = i;
            }
        }
        let gto = gto.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "composite AO without a gto child is unsupported",
            )
        })?;
        let gto_first = (pw.is_none() && bspline.is_none()) || gto_idx < extra_idx;
        Ok((gto, pw, bspline, gto_first, skipped_rows))
    } else if aos_obj.hasattr("phase_is_sin")? {
        // Pure plane-wave leaf (`AOs_plane_wave_data`): no GTO primitives. Use the
        // zero-AO GTO placeholder + `Some(pw)`, so `eval_aos` / `orb_value_grad_lap`
        // yield the plane-wave stack directly (concatenate with a 0-row GTO block).
        Ok((
            AosGtoData::empty(),
            Some(crate::atomic_orbital::AosPwData::from_py(aos_obj)?),
            None,
            true,
            0,
        ))
    } else {
        Ok((AosGtoData::from_py(aos_obj)?, None, None, true, 0))
    }
}

impl OrbData {
    /// Parse a real orbital (`MOs_data` or AO leaf / composite) and check its AO rank
    /// against Python (issue #165, see [`check_native_ao_rank`]).
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (orb, skipped_rows) = Self::from_py_unchecked(obj)?;
        let aos_obj = if obj.hasattr("mo_coefficients")? {
            obj.getattr("aos_data")?
        } else {
            obj.clone()
        };
        check_native_ao_rank(
            "real",
            &aos_obj,
            orb.ao_count(),
            skipped_rows,
            orb.mo_coefficients.as_ref().map(|c| c.ncols()),
        )?;
        Ok(orb)
    }

    /// `(orbital, skipped_rows)` without the AO-rank check of [`OrbData::from_py`].
    fn from_py_unchecked(obj: &Bound<'_, PyAny>) -> PyResult<(Self, usize)> {
        // MOs_data carries `mo_coefficients` + `aos_data`; AO leaves do not.
        if obj.hasattr("mo_coefficients")? {
            let aos_obj = obj.getattr("aos_data")?;
            let identity = obj
                .getattr("identity")
                .and_then(|value| value.extract::<bool>())
                .unwrap_or(false);
            if identity {
                if aos_obj.hasattr("spline_coefficients")? {
                    return Ok((
                        OrbData {
                            precision_policy: crate::construction_precision_policy(),
                            aos: AosGtoData::empty(),
                            pw: None,
                            gto_first: true,
                            mo_coefficients: None,
                            mo_coefficients_t: None,
                            bspline: Some(crate::bspline::AosBsplineData::from_py(&aos_obj)?),
                        },
                        0,
                    ));
                }
                let (aos, pw, bspline, gto_first, skipped_rows) = extract_aos(&aos_obj)?;
                return Ok((
                    OrbData {
                        precision_policy: crate::construction_precision_policy(),
                        aos,
                        pw,
                        gto_first,
                        mo_coefficients: None,
                        mo_coefficients_t: None,
                        bspline,
                    },
                    skipped_rows,
                ));
            }
            // Python may store coefficients in an fp32 forward zone in mixed
            // mode.  The owned Rust representation is f64 and applies the
            // selected arithmetic width inside its kernels, so widen once when
            // constructing the context.
            let np = obj.py().import("numpy")?;
            let mut c_src = np.call_method1("asarray", (obj.getattr("mo_coefficients")?,))?;
            // A complex128 MO matrix with an exactly zero imaginary part (the dtype that
            // `MOs_data.sanity_check` requires over a phase-free Gamma-complex pGTO child) is
            // real-representable: take its real part explicitly, which is exact, instead of the
            // `asarray(float64)` complex->real cast below that emits `ComplexWarning` (PR #93).
            // Any other input keeps the unchanged cast.
            if np
                .call_method1("iscomplexobj", (&c_src,))?
                .extract::<bool>()?
                && !np
                    .call_method1("any", (c_src.getattr("imag")?,))?
                    .extract::<bool>()?
            {
                c_src = c_src.getattr("real")?;
            }
            let c_obj = np.call_method1("asarray", (c_src, np.getattr("float64")?))?;
            let c_arr: numpy::PyReadonlyArray2<f64> = c_obj.extract()?;
            let c = c_arr.as_array().to_owned();
            let c_t = c.t().as_standard_layout().to_owned(); // (num_ao, num_mo), C-contiguous
            if aos_obj.hasattr("spline_coefficients")? {
                return Ok((
                    OrbData {
                        precision_policy: crate::construction_precision_policy(),
                        aos: AosGtoData::empty(),
                        pw: None,
                        gto_first: true,
                        mo_coefficients: Some(c),
                        mo_coefficients_t: Some(c_t),
                        bspline: Some(crate::bspline::AosBsplineData::from_py(&aos_obj)?),
                    },
                    0,
                ));
            }
            let (aos, pw, bspline, gto_first, skipped_rows) = extract_aos(&aos_obj)?;
            Ok((
                OrbData {
                    precision_policy: crate::construction_precision_policy(),
                    aos,
                    pw,
                    gto_first,
                    mo_coefficients: Some(c),
                    mo_coefficients_t: Some(c_t),
                    bspline,
                },
                skipped_rows,
            ))
        } else if obj.hasattr("spline_coefficients")? {
            Ok((
                OrbData {
                    precision_policy: crate::construction_precision_policy(),
                    aos: AosGtoData::empty(),
                    pw: None,
                    gto_first: true,
                    mo_coefficients: None,
                    mo_coefficients_t: None,
                    bspline: Some(crate::bspline::AosBsplineData::from_py(obj)?),
                },
                0,
            ))
        } else {
            let (aos, pw, bspline, gto_first, skipped_rows) = extract_aos(obj)?;
            Ok((
                OrbData {
                    precision_policy: crate::construction_precision_policy(),
                    aos,
                    pw,
                    gto_first,
                    mo_coefficients: None,
                    mo_coefficients_t: None,
                    bspline,
                },
                skipped_rows,
            ))
        }
    }

    /// Home-cell-wrap electron rows for the geminal-VALUE observable paths (jqmc's
    /// `_wrap_into_cell_if_pbc`, applied inside `compute_AOs`). `Some(wrapped)` for a periodic
    /// pGTO leaf (whose static Gaussian image tiling underflows to 0 for a walker roamed a few
    /// lattice vectors out -> a fresh geminal inverse would NaN in the force drift/ratio);
    /// `None` (identity) for molecular / plane-wave / B-spline leaves (bounded / grid-periodic,
    /// no underflow). NOT applied in the sweep VALUE path (`eval_aos`/`compute`), which stays
    /// bit-identical to JAX's accept/reject; only the observable callers wrap. For a composite
    /// the PW/B-spline child is wrap-invariant (real-Gamma phase / periodic grid), so wrapping
    /// the whole coordinate is safe.
    pub(crate) fn wrap_electrons(&self, r: &ArrayView2<f64>) -> Option<Array2<f64>> {
        if self.aos.wrap_cell.is_some() {
            Some(crate::atomic_orbital::wrap_r(&self.aos, r))
        } else {
            None
        }
    }

    /// The orbital->PW-AO coefficient block `(num_orb, pw.num_ao)` used to map an
    /// orbital-rank determinant cotangent onto the plane-wave child rows. For an
    /// `MOs_data` orbital this is the PW column slice of `mo_coefficients`; for a
    /// RAW-AO orbital (dense AO-AGP composite) the orbital basis IS the concatenated AO
    /// basis, so `C = I(num_ao_total)` and the block is the identity selector of the PW
    /// rows -- exactly `bar[pw0..pw0+pw.num_ao, :]` after the `c_pw.t().dot(bar)`.
    pub(crate) fn pw_coeff_block(&self) -> Array2<f64> {
        let pw = self
            .pw
            .as_ref()
            .expect("pw_coeff_block requires a PW child");
        let pw0 = if self.gto_first { self.aos.num_ao } else { 0 };
        match &self.mo_coefficients {
            Some(c) => c
                .slice(numpy::ndarray::s![.., pw0..pw0 + pw.num_ao])
                .to_owned(),
            None => {
                // Raw-AO: num_orb == num_ao_total == aos.num_ao + pw.num_ao.
                let n_tot = self.aos.num_ao + pw.num_ao;
                let mut block = Array2::<f64>::zeros((n_tot, pw.num_ao));
                for j in 0..pw.num_ao {
                    block[[pw0 + j, j]] = 1.0;
                }
                block
            }
        }
    }

    /// AO values `(num_ao_total, N_e)`, concatenating the gto and (optional) pw
    /// children in the composite's child order.
    pub(crate) fn eval_aos(&self, r: &ArrayView2<f64>) -> Array2<f64> {
        let ao_f32 = self.precision_policy.is_f32("ao_eval");
        // The "extra" (non-gto) block is the pw child or the B-spline child --
        // mutually exclusive in jqmc composites. A pure B-spline / pure PW leaf
        // has a 0-row gto placeholder, so the extra block is returned verbatim.
        let extra: Option<Array2<f64>> = match (&self.pw, &self.bspline) {
            (Some(pw), _) => Some(crate::timed!(
                "ao.pw",
                crate::atomic_orbital::compute_pw_with_policy(pw, r, ao_f32)
            )),
            (None, Some(bs)) => Some(bs.value_with_policy(r, ao_f32)),
            (None, None) => None,
        };
        match extra {
            None => crate::atomic_orbital::compute_aos_with_policy(&self.aos, r, ao_f32),
            Some(ex) => {
                if self.aos.num_ao == 0 {
                    return ex;
                }
                let gto = crate::timed!(
                    "ao.gto",
                    crate::atomic_orbital::compute_aos_with_policy(&self.aos, r, ao_f32)
                );
                let (a, b) = if self.gto_first { (gto, ex) } else { (ex, gto) };
                numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[a.view(), b.view()]).unwrap()
            }
        }
    }

    /// EXACT (un-screened) sibling of [`eval_aos`]: the GTO block is evaluated with
    /// [`crate::atomic_orbital::compute_aos_exact`], i.e. every kernel AO of the
    /// static periodic image tiling at every point, with no runtime LEPSBAS distance
    /// screen -- the XLA convention. Observable / parity kernels whose OUTPUT is
    /// indexed BY AO (dV/dC, dV/dlambda) must use this: a screened AO row does not
    /// merely shift a contracted scalar at the fp floor, it lands in the output as an
    /// exact 0.0 where XLA has a small non-zero.
    /// [`eval_aos`] keeps the screen for the MCMC per-move sweep.
    pub(crate) fn eval_aos_exact(&self, r: &ArrayView2<f64>) -> Array2<f64> {
        let ao_f32 = self.precision_policy.is_f32("ao_eval");
        let extra: Option<Array2<f64>> = match (&self.pw, &self.bspline) {
            (Some(pw), _) => Some(crate::atomic_orbital::compute_pw_with_policy(pw, r, ao_f32)),
            (None, Some(bs)) => Some(bs.value_with_policy(r, ao_f32)),
            (None, None) => None,
        };
        let gto_exact = |r: &ArrayView2<f64>| {
            crate::atomic_orbital::compute_aos_with_policy(&self.aos, r, ao_f32)
        };
        match extra {
            None => gto_exact(r),
            Some(ex) => {
                if self.aos.num_ao == 0 {
                    return ex;
                }
                let gto = gto_exact(r);
                let (a, b) = if self.gto_first { (gto, ex) } else { (ex, gto) };
                numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[a.view(), b.view()]).unwrap()
            }
        }
    }

    /// EXACT (un-screened) sibling of [`compute`] -- [`eval_aos_exact`] + MO projection.
    pub fn compute_exact(&self, r_carts: &ArrayView2<f64>) -> Array2<f64> {
        let aos = self.eval_aos_exact(r_carts);
        match &self.mo_coefficients {
            None => aos,
            Some(c) => matmul(&c.view(), &aos.view()),
        }
    }

    /// Evaluate the orbital column at a single position, shape `(orb_num,)`.
    /// Fast path: raw-slice AO eval + tight contiguous MO projection (the
    /// constant `C` matrix stays L2-resident across moves -- the optimization
    /// XLA can't replicate, since its op boundaries re-read `C` from DRAM).
    /// Total AO-buffer length for this orbital (bspline-aware: the empty GTO
    /// placeholder reports 0, so the spline's `num_ao` must be used instead).
    #[inline]
    pub fn ao_count(&self) -> usize {
        let extra = self
            .pw
            .as_ref()
            .map(|p| p.num_ao)
            .or_else(|| self.bspline.as_ref().map(|b| b.num_ao))
            .unwrap_or(0);
        self.aos.num_ao + extra
    }

    /// AO-axis offset of the GTO/pGTO child inside the composite AO buffer:
    /// `0` if the gto block leads (or there is no extra child), else the extra
    /// (pw / bspline) block's `num_ao`. The composite concatenates children in
    /// child order (`compute_single` / `bspline_mo_block` use the same rule), so a
    /// cotangent pulled to the full composite AO axis has this many non-GTO rows
    /// ahead of the GTO home AOs. The basis-radial param-grad (which only the GTO
    /// child carries) slices the GTO rows out with this offset.
    #[inline]
    pub(crate) fn gto_ao_offset(&self) -> usize {
        if self.gto_first {
            0
        } else {
            self.pw
                .as_ref()
                .map(|p| p.num_ao)
                .or_else(|| self.bspline.as_ref().map(|b| b.num_ao))
                .unwrap_or(0)
        }
    }

    /// MO-coefficient sub-block for the B-spline child (composite-aware), shape
    /// `(num_orb, bspline.num_ao)`. A raw composite uses the identity selector of
    /// its B-spline rows. `None` for a pure raw B-spline or non-B-spline orbital. For a
    /// PURE B-spline the whole map (offset 0, since the gto placeholder is 0-AO); for a
    /// pgto(+)bspline composite the B-spline AO columns `[off .. off+bspline.num_ao]`
    /// (`off = gto_first ? aos.num_ao : 0`, matching `eval_aos`' child concatenation).
    /// The B-spline force kernels (`bspline_det_egrad_side`) must project through THIS
    /// sub-block, not the full composite map (whose extra pgto columns overrun the
    /// bspline AO fields -> index-out-of-bounds).
    pub(crate) fn bspline_mo_block(&self) -> Option<Array2<f64>> {
        let bs = self.bspline.as_ref()?;
        let off = if self.gto_first { self.aos.num_ao } else { 0 };
        let c = match &self.mo_coefficients {
            Some(c) => c,
            None if self.aos.num_ao == 0 => return None,
            None => {
                // Raw composite cotangents have the complete AO rank. Select
                // only the B-spline rows, preserving the AO child order.
                let mut block = Array2::<f64>::zeros((self.ao_count(), bs.num_ao));
                for j in 0..bs.num_ao {
                    block[[off + j, j]] = 1.0;
                }
                return Some(block);
            }
        };
        Some(
            c.slice(numpy::ndarray::s![.., off..off + bs.num_ao])
                .to_owned(),
        )
    }

    pub fn compute_single(&self, pos: [f64; 3]) -> Vec<f64> {
        let ao_f32 = self.precision_policy.is_f32("ao_eval");
        if ao_f32 {
            if self.pw.is_none() && self.bspline.is_none() {
                let mut ao = vec![0.0_f64; self.aos.num_ao];
                return match &self.mo_coefficients {
                    None => {
                        crate::atomic_orbital::compute_aos_single_f32_into(
                            &self.aos, pos, false, &mut ao,
                        );
                        ao
                    }
                    Some(c) => {
                        let mut out = vec![0.0_f64; c.nrows()];
                        self.compute_single_gto_f32_into(pos, &mut ao, &mut out);
                        out
                    }
                };
            }
            // Keep every leaf (GTO/PW/B-spline) in ao_eval fp32 until the
            // explicit fp32 -> fp64 MO projection boundary.
            let r = Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap();
            return self.compute(&r.view()).column(0).to_vec();
        }
        self.compute_single_f64_body(pos, false)
    }

    /// Orbital column in XLA's FUSED value convention.
    ///
    /// jqmc builds the geminal whose inverse meets the fp64 spatial-derivative
    /// tables from `compute_AOs_value_grad_lap` (`determinant.py::
    /// _orbital_value_grad_lap_tables`): the AO body is evaluated in the fp64
    /// `ao_grad_lap` zone and only the completed AO value is cast to the
    /// `ao_eval` dtype, before the (fp64 `mo_eval`) MO projection.  This is NOT
    /// the zone-local fp32 arithmetic of [`compute_single`] (the `compute_AOs`
    /// convention of the MCMC sweep and of the ratio-only kernels).  The GFMC
    /// fast state seeds, hops and candidate rows of ONE carried geminal use
    /// this convention so that the carried `G` never mixes the two families.
    /// Full precision: identical to the fp64 [`compute_single`] (no rounding).
    pub(crate) fn compute_single_fused(&self, pos: [f64; 3]) -> Vec<f64> {
        self.compute_single_f64_body(pos, self.precision_policy.is_f32("ao_eval"))
    }

    /// fp64 single-position AO stack (+ MO projection).  `round_ao_eval`
    /// rounds every completed AO value to fp32 before the projection.
    fn compute_single_f64_body(&self, pos: [f64; 3], round_ao_eval: bool) -> Vec<f64> {
        if round_ao_eval
            && self.pw.is_none()
            && self.bspline.is_none()
            && self.aos.n_image == 1
            && !self.aos.shells.is_empty()
        {
            // Mixed-only fast path (full keeps the generic kernel below, bitwise):
            // the shell-reuse fp64 single-position kernel of the full-precision
            // MCMC sweep. Its fp64 values differ from the generic kernel only in
            // the last bits, which the fp32 rounding removes.
            let mut ao = vec![0.0_f64; self.aos.num_ao_kernel];
            crate::atomic_orbital::compute_aos_single_into(&self.aos, pos, &mut ao);
            round_to_ao_eval(&mut ao);
            return match &self.mo_coefficients {
                None => ao,
                Some(c) => {
                    let mut out = vec![0.0_f64; c.nrows()];
                    project_8acc(
                        c.as_slice().expect("C contiguous (C-order)"),
                        &ao,
                        c.ncols(),
                        c.nrows(),
                        &mut out,
                    );
                    out
                }
            };
        }
        // "extra" (non-gto) AO block: the pw child or the B-spline child (mutually
        // exclusive). A pure PW / B-spline leaf carries a 0-AO gto placeholder.
        let extra: Option<Vec<f64>> = match (&self.pw, &self.bspline) {
            (Some(pw), _) => Some(crate::timed!(
                "sw.pw",
                crate::atomic_orbital::compute_pw_single(pw, pos)
            )),
            (None, Some(bs)) => {
                let mut v = vec![0.0_f64; bs.num_ao];
                bs.value_single(pos, &mut v);
                Some(v)
            }
            (None, None) => None,
        };
        let mut ao = match extra {
            None => crate::atomic_orbital::compute_aos_single_with_policy(&self.aos, pos, false),
            Some(ex) => {
                if self.aos.num_ao == 0 {
                    ex
                } else {
                    let gto = crate::timed!(
                        "sw.gto",
                        crate::atomic_orbital::compute_aos_single_with_policy(
                            &self.aos, pos, false
                        )
                    );
                    if self.gto_first {
                        let mut v = gto;
                        v.extend(ex);
                        v
                    } else {
                        let mut v = ex;
                        v.extend(gto);
                        v
                    }
                }
            }
        };
        if round_ao_eval {
            round_to_ao_eval(&mut ao);
        }
        match &self.mo_coefficients {
            None => ao,
            Some(c) => {
                let mut out = vec![0.0_f64; c.nrows()];
                crate::timed!(
                    "sw.moproj",
                    project_8acc(
                        c.as_slice().expect("C contiguous (C-order)"),
                        &ao,
                        c.ncols(),
                        c.nrows(),
                        &mut out,
                    )
                );
                out
            }
        }
    }

    /// No-alloc single-position orbital column into caller buffers (molecular
    /// only -- `aos.n_image == 1`). `ao_buf` length `num_ao`, `out` length
    /// `num_mo` (MO) or `num_ao` (raw AO). The hot per-move path: scratch buffers
    /// are reused across moves so the constant `C` stays L2-resident (no
    /// per-move allocator churn evicting it to DRAM).
    #[inline]
    pub fn compute_single_into(&self, pos: [f64; 3], ao_buf: &mut [f64], out: &mut [f64]) {
        if self.precision_policy.is_f32("ao_eval") && self.pw.is_none() && self.bspline.is_none() {
            self.compute_single_gto_f32_into(pos, ao_buf, out);
            return;
        }
        if self.precision_policy.is_f32("ao_eval") {
            // The f32 implementation keeps the AO materialized until the
            // explicit f32->f64 mo_eval boundary.
            let r = Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap();
            let values = self.compute(&r.view());
            for (dst, src) in out.iter_mut().zip(values.column(0)) {
                *dst = *src;
            }
            return;
        }
        if let Some(bs) = &self.bspline {
            bs.value_single(pos, &mut ao_buf[..bs.num_ao]);
            match &self.mo_coefficients {
                None => out.copy_from_slice(&ao_buf[..bs.num_ao]),
                Some(c) => project_faer(
                    c.as_slice().expect("C contiguous"),
                    &ao_buf[..bs.num_ao],
                    c.ncols(),
                    c.nrows(),
                    out,
                ),
            }
            return;
        }
        match &self.pw {
            None => crate::atomic_orbital::compute_aos_single_into(&self.aos, pos, ao_buf),
            Some(pw) => {
                let n_gto = self.aos.num_ao;
                let n_pw = pw.num_ao;
                if self.gto_first {
                    crate::timed!(
                        "sw.gto",
                        crate::atomic_orbital::compute_aos_single_into(
                            &self.aos,
                            pos,
                            &mut ao_buf[..n_gto]
                        )
                    );
                    crate::timed!(
                        "sw.pw",
                        crate::atomic_orbital::compute_pw_single_into(
                            pw,
                            pos,
                            &mut ao_buf[n_gto..n_gto + n_pw]
                        )
                    );
                } else {
                    crate::timed!(
                        "sw.pw",
                        crate::atomic_orbital::compute_pw_single_into(pw, pos, &mut ao_buf[..n_pw])
                    );
                    crate::timed!(
                        "sw.gto",
                        crate::atomic_orbital::compute_aos_single_into(
                            &self.aos,
                            pos,
                            &mut ao_buf[n_pw..n_pw + n_gto]
                        )
                    );
                }
            }
        }
        match &self.mo_coefficients {
            None => out.copy_from_slice(ao_buf),
            Some(c) => project_faer(
                c.as_slice().expect("C contiguous"),
                ao_buf,
                c.ncols(),
                c.nrows(),
                out,
            ),
        }
    }

    /// Pure-GTO leaf under an fp32 `ao_eval` zone: the single-position fp32 AO
    /// kernel (bit-identical to the one-point batch value of [`Self::eval_aos`])
    /// into `ao_buf[..num_ao]`, then the fp64 (`mo_eval`) projection
    /// [`project_8acc`] of the full-precision single-position path
    /// ([`Self::compute_single_f64_body`]). `out` takes one entry per orbital.
    fn compute_single_gto_f32_into(&self, pos: [f64; 3], ao_buf: &mut [f64], out: &mut [f64]) {
        let na = self.aos.num_ao;
        crate::atomic_orbital::compute_aos_single_f32_into(
            &self.aos,
            pos,
            false,
            &mut ao_buf[..na],
        );
        match &self.mo_coefficients {
            None => out[..na].copy_from_slice(&ao_buf[..na]),
            Some(c) => project_8acc(
                c.as_slice().expect("C contiguous (C-order)"),
                &ao_buf[..na],
                c.ncols(),
                c.nrows(),
                &mut out[..c.nrows()],
            ),
        }
    }

    /// A raw-AO copy of this orbital (drops the MO layer). Cheap clone of the AO
    /// leaves, `mo_coefficients = None`. Used by [`j3_lower_to_ao`].
    #[allow(dead_code)]
    pub(crate) fn as_raw_ao(&self) -> OrbData {
        OrbData {
            precision_policy: self.precision_policy,
            aos: self.aos.clone(),
            pw: self.pw.clone(),
            gto_first: self.gto_first,
            mo_coefficients: None,
            mo_coefficients_t: None,
            bspline: self.bspline.clone(),
        }
    }

    /// Evaluate the orbital matrix at `r_carts`, shape `(orb_num, N_e)`.
    pub fn compute(&self, r_carts: &ArrayView2<f64>) -> Array2<f64> {
        let aos = crate::timed!("orb.eval_aos", self.eval_aos(r_carts)); // (num_ao_total, N_e)
        match &self.mo_coefficients {
            None => aos,
            Some(c) => crate::timed!("orb.mo_matmul", matmul(&c.view(), &aos.view())), // (num_mo, num_ao)*(num_ao, N_e)
        }
    }

    /// Like [`compute`] but with a per-shell molecular distance cutoff (skips
    /// shells whose every AO value is below `FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO
    /// units at the point, `Shell::pz_val`, issue #100). Used by the
    /// ECP-nonlocal mesh eval (points cluster near one nucleus => most of a large
    /// basis is negligible). Falls back to the exact [`compute`] for any leaf the
    /// cutoff doesn't cover (periodic images, PW/bspline child, bare-AO geminal).
    /// Approximation (dropped AO values < 1e-12) -- the caller opts in (production e_L, not the
    /// bit-exact test path).
    pub fn compute_mo_cutoff(&self, r_carts: &ArrayView2<f64>) -> Array2<f64> {
        use crate::atomic_orbital::AoKind;
        // Periodic sphe GTO: the batch `compute` loops all `num_ao_kernel =
        // n_image*num_ao` tiled AOs per point. The sweep's
        // `compute_aos_single_fold_into` is SHELL-based over the contiguous
        // `sh_*` arrays (one radial per shell image) and folds straight into the
        // home rank; it applies NO runtime distance screen (static image tiling,
        // the XLA convention). This is the periodic ECP-mesh hot path (silicon:
        // 96% of orb.compute).
        let pbc_sphe = self.aos.n_image > 1
            && self.pw.is_none()
            && self.bspline.is_none()
            && self.aos.kind == AoKind::Sphe
            && !self.aos.shells.is_empty();
        if pbc_sphe {
            if let Some(c) = &self.mo_coefficients {
                let (num_ao, num_mo) = (self.aos.num_ao, c.nrows());
                let cs = c.as_slice().expect("C contiguous");
                let n = r_carts.nrows();
                let mut out = Array2::<f64>::zeros((num_mo, n));
                let mut ao = vec![0.0_f64; num_ao];
                let mut mo = vec![0.0_f64; num_mo];
                for e in 0..n {
                    let pos = [r_carts[[e, 0]], r_carts[[e, 1]], r_carts[[e, 2]]];
                    crate::atomic_orbital::compute_aos_single_fold_into(&self.aos, pos, &mut ao);
                    project_8acc(cs, &ao, num_ao, num_mo, &mut mo);
                    for m in 0..num_mo {
                        out[[m, e]] = mo[m];
                    }
                }
                return out;
            }
        }
        let molecular_gto = self.aos.n_image == 1
            && self.pw.is_none()
            && self.bspline.is_none()
            && !self.aos.shells.is_empty();
        let c_t = match (&self.mo_coefficients_t, molecular_gto) {
            (Some(ct), true) => ct,
            _ => return self.compute(r_carts), // exact fallback
        };
        let ct = c_t.as_slice().expect("C_T contiguous");
        let num_mo = self
            .mo_coefficients
            .as_ref()
            .map(|c| c.nrows())
            .unwrap_or(0);
        let n = r_carts.nrows();
        let mut out = Array2::<f64>::zeros((num_mo, n));
        let mut mo = vec![0.0_f64; num_mo];
        let is_cart = self.aos.kind == AoKind::Cart;
        for e in 0..n {
            let pos = [r_carts[[e, 0]], r_carts[[e, 1]], r_carts[[e, 2]]];
            if is_cart {
                crate::atomic_orbital::compute_mo_single_cutoff_cart_into(
                    &self.aos, ct, num_mo, pos, &mut mo,
                );
            } else {
                crate::atomic_orbital::compute_mo_single_cutoff_into(
                    &self.aos, ct, num_mo, pos, &mut mo,
                );
            }
            for m in 0..num_mo {
                out[[m, e]] = mo[m];
            }
        }
        out
    }
}

/// Geminal data shared by dense-lambda AGP/AGPn and occupied-orbital SD.
///
/// Canonical ``SD_data`` owns no lambda.  For that representation `lambda` is
/// `None` and the determinant is contracted directly from the occupied MO rows:
/// ``[O_up[:N_dn]^T O_dn | O_up[N_dn:]^T]``.  Only AGP/AGPn retain a dense
/// lambda payload.
pub struct GeminalData {
    pub orb_up: OrbData,
    pub orb_dn: OrbData,
    pub lambda: Option<Array2<f64>>,
    pub orb_num_up: usize,
    pub orb_num_dn: usize,
}

impl OrbData {
    /// Build an MO-basis orb from a cart AO leaf + transform `mo` (`orb = mo @ AO`).
    /// Used for sphe->cart (`_aos_sphe_to_cart` gives the c2s transform as `mo`).
    pub fn new_mo(aos: AosGtoData, mo: Array2<f64>) -> Self {
        let mo_t = mo.t().to_owned();
        OrbData {
            precision_policy: crate::construction_precision_policy(),
            aos,
            pw: None,
            gto_first: true,
            mo_coefficients: Some(mo),
            mo_coefficients_t: Some(mo_t),
            bspline: None,
        }
    }
}

impl GeminalData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let orb_up = OrbData::from_py(&obj.getattr("orb_data_up_spin")?)?;
        let orb_dn = OrbData::from_py(&obj.getattr("orb_data_dn_spin")?)?;
        let orb_num_up: usize = obj.getattr("orb_num_up")?.extract()?;
        let orb_num_dn: usize = obj.getattr("orb_num_dn")?.extract()?;
        let lambda = if obj.hasattr("lambda_matrix")? {
            let lam: numpy::PyReadonlyArray2<f64> = obj.getattr("lambda_matrix")?.extract()?;
            Some(lam.as_array().to_owned())
        } else {
            None
        };
        Ok(GeminalData {
            orb_up,
            orb_dn,
            lambda,
            orb_num_up,
            orb_num_dn,
        })
    }

    #[inline]
    pub fn is_sd(&self) -> bool {
        self.lambda.is_none()
    }

    #[inline]
    pub fn orb_num_up(&self) -> usize {
        self.orb_num_up
    }

    #[inline]
    pub fn n_unpaired(&self) -> usize {
        match &self.lambda {
            Some(lam) => lam.ncols() - self.orb_num_dn,
            None => self.orb_num_up() - self.orb_num_dn,
        }
    }

    #[inline]
    pub fn lambda_value(&self, row: usize, col: usize) -> f64 {
        match &self.lambda {
            Some(lam) => lam[[row, col]],
            None => (row == col) as u8 as f64,
        }
    }

    /// ``lambda_paired * O_dn`` without materialising the SD selector.
    pub fn paired_times(&self, od: &ArrayView2<f64>) -> Array2<f64> {
        if let Some(lam) = &self.lambda {
            return matmul(&lam.slice(numpy::ndarray::s![.., 0..self.orb_num_dn]), od);
        }
        let mut out = Array2::<f64>::zeros((self.orb_num_up(), od.ncols()));
        out.slice_mut(numpy::ndarray::s![0..self.orb_num_dn, ..])
            .assign(od);
        out
    }

    /// ``lambda_paired^T * X`` without materialising the SD selector.
    pub fn paired_t_times(&self, x: &ArrayView2<f64>) -> Array2<f64> {
        if let Some(lam) = &self.lambda {
            return mat_t_mul(&lam.slice(numpy::ndarray::s![.., 0..self.orb_num_dn]), x);
        }
        x.slice(numpy::ndarray::s![0..self.orb_num_dn, ..])
            .to_owned()
    }

    /// ``O_up^T * lambda_paired`` without materialising the SD selector.
    pub fn up_t_times_paired(&self, ou: &ArrayView2<f64>) -> Array2<f64> {
        if let Some(lam) = &self.lambda {
            return mat_t_mul(ou, &lam.slice(numpy::ndarray::s![.., 0..self.orb_num_dn]));
        }
        ou.slice(numpy::ndarray::s![0..self.orb_num_dn, ..])
            .t()
            .as_standard_layout()
            .to_owned()
    }

    /// ``W = [lambda_paired O_dn | lambda_unpaired]``.  The SD branch writes
    /// occupied rows directly: the paired down table in the first rows and an
    /// identity selector for excess-up occupied rows.
    pub fn w_from_dn(&self, od: &ArrayView2<f64>, n_up: usize) -> Array2<f64> {
        let mut w = Array2::<f64>::zeros((self.orb_num_up(), n_up));
        let paired = self.paired_times(od);
        w.slice_mut(numpy::ndarray::s![.., 0..od.ncols()])
            .assign(&paired);
        let n_unpaired = self.n_unpaired();
        for a in 0..self.orb_num_up() {
            for j in 0..n_unpaired {
                w[[a, od.ncols() + j]] = self.lambda_value(a, self.orb_num_dn + j);
            }
        }
        w
    }
}

/// C[m,n] = A[m,k] * B[k,n] (hand-rolled, row-major friendly: i,k,j order).
pub(crate) fn matmul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    debug_assert_eq!(k, b.nrows());
    // Large GEMMs (the det/J3 MO projections in the e_L, ~10^7 flops) route
    // through faer (cache-blocked, BLAS-class): the naive triple loop below is
    // memory-bound at that size and scales poorly when many MPI ranks share the
    // memory bandwidth of a node. Small
    // GEMMs keep the naive zero-skip path (avoids the to_owned + faer setup).
    if (m as u64) * (n as u64) * (k as u64) >= 100_000 && m != 0 && n != 0 && k != 0 {
        let ac = a.as_standard_layout(); // force C-contiguous (a may be a .t() view)
        let bc = b.as_standard_layout();
        let mut out = Array2::<f64>::zeros((m, n));
        // row-major (p,q) viewed column-major (q,p) == transpose; faer computes
        // outT = bT*aT = (a*b)T, so `out` holds a*b on return.
        let a_faer = faer::MatRef::from_column_major_slice(ac.as_slice().unwrap(), k, m);
        let b_faer = faer::MatRef::from_column_major_slice(bc.as_slice().unwrap(), n, k);
        let mut res = faer::MatMut::from_column_major_slice_mut(out.as_slice_mut().unwrap(), n, m);
        faer::linalg::matmul::matmul(
            res.as_mut(),
            faer::Accum::Replace,
            b_faer,
            a_faer,
            1.0,
            faer::Par::Seq,
        );
        return out;
    }
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for p in 0..k {
            let aip = a[[i, p]];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += aip * b[[p, j]];
            }
        }
    }
    c
}

/// out[i,j] = Sum_k aT[i,k]*b[k,j] = Sum_k a[k,i]*b[k,j]  (a is (K, M), result (M, N)).
pub(crate) fn mat_t_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let (kk, m) = (a.nrows(), a.ncols());
    let n = b.ncols();
    debug_assert_eq!(kk, b.nrows());
    let mut c = Array2::<f64>::zeros((m, n));
    for p in 0..kk {
        for i in 0..m {
            let api = a[[p, i]];
            if api == 0.0 {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += api * b[[p, j]];
            }
        }
    }
    c
}

/// Build the full geminal matrix G, shape `(N_up, N_up)`.
pub fn compute_geminal_all_elements(
    data: &GeminalData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> Array2<f64> {
    // Home-cell-wrap for the geminal-VALUE observable path (jqmc `compute_AOs`): a far-roamed
    // periodic walker would otherwise underflow the static pGTO image tiling to a 0 geminal.
    let ruw = data
        .orb_up
        .wrap_electrons(r_up_carts)
        .unwrap_or_else(|| r_up_carts.to_owned());
    let rdw = data
        .orb_dn
        .wrap_electrons(r_dn_carts)
        .unwrap_or_else(|| r_dn_carts.to_owned());
    let ou = data.orb_up.compute_exact(&ruw.view()); // (orb_num_up, N_up)
    let od = data.orb_dn.compute_exact(&rdw.view()); // (orb_num_dn, N_dn)

    let n_up = ou.ncols();
    let w = data.w_from_dn(&od.view(), n_up);
    mat_t_mul(&ou.view(), &w.view())
}

/// ln|det(G)| via LU with partial pivoting (consumes a copy).
pub fn ln_abs_det(g: &ArrayView2<f64>) -> f64 {
    let n = g.nrows();
    let mut a = g.to_owned();
    let mut ln = 0.0_f64;
    for k in 0..n {
        // partial pivot
        let mut p = k;
        let mut maxv = a[[k, k]].abs();
        for i in (k + 1)..n {
            let v = a[[i, k]].abs();
            if v > maxv {
                maxv = v;
                p = i;
            }
        }
        if p != k {
            for j in 0..n {
                let tmp = a[[k, j]];
                a[[k, j]] = a[[p, j]];
                a[[p, j]] = tmp;
            }
        }
        let piv = a[[k, k]];
        ln += piv.abs().ln();
        if piv == 0.0 {
            continue;
        }
        for i in (k + 1)..n {
            let f = a[[i, k]] / piv;
            if f == 0.0 {
                continue;
            }
            for j in (k + 1)..n {
                let akj = a[[k, j]];
                a[[i, j]] -= f * akj;
            }
        }
    }
    ln
}

/// `(sign, ln|det(G)|)` via LU with partial pivoting. `det = sign*exp(ln_abs)`.
/// Lets ratios `det_num/det_den = sign*exp(ln_num - ln_den)` avoid overflow.
pub fn det_sign_lnabs(g: &ArrayView2<f64>) -> (f64, f64) {
    let n = g.nrows();
    let mut a = g.to_owned();
    let mut ln = 0.0_f64;
    let mut sign = 1.0_f64;
    for k in 0..n {
        let mut p = k;
        let mut maxv = a[[k, k]].abs();
        for i in (k + 1)..n {
            let v = a[[i, k]].abs();
            if v > maxv {
                maxv = v;
                p = i;
            }
        }
        if p != k {
            for j in 0..n {
                let tmp = a[[k, j]];
                a[[k, j]] = a[[p, j]];
                a[[p, j]] = tmp;
            }
            sign = -sign;
        }
        let piv = a[[k, k]];
        if piv == 0.0 {
            return (0.0, f64::NEG_INFINITY);
        }
        if piv < 0.0 {
            sign = -sign;
        }
        ln += piv.abs().ln();
        for i in (k + 1)..n {
            let f = a[[i, k]] / piv;
            if f == 0.0 {
                continue;
            }
            for j in (k + 1)..n {
                let akj = a[[k, j]];
                a[[i, j]] -= f * akj;
            }
        }
    }
    (sign, ln)
}

/// ln|det of the geminal| for the given configuration.
pub fn compute_ln_det_geminal_all_elements(
    data: &GeminalData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    let g = compute_geminal_all_elements(data, r_up_carts, r_dn_carts);
    ln_abs_det(&g.view())
}

// ============================================================================
// Sherman-Morrison ratio streaming state.
// Mirrors Det_ratio_streaming_state + _init/_advance/_compute_v/_compute_u
// (determinant.py) and the rank-1 SM update (jqmc_mcmc.py).
// ============================================================================

/// Value-only orbital + paired tables for the ratio kernels.
pub struct DetRatioState {
    pub ao_up: Array2<f64>,            // (orb_num_up, N_up)  = compute_orb(orb_up)
    pub ao_dn: Array2<f64>,            // (orb_num_dn, N_dn)
    pub paired_dn: Array2<f64>,        // AGP: (orb_num_up,N_dn); SD: compact (orb_num_dn,N_dn)
    pub paired_up_lambda: Array2<f64>, // (N_up, orb_num_dn) = ao_upT * lambda_paired
}

/// `_init_det_ratio_streaming_state`.
pub fn init_det_ratio_state(
    data: &GeminalData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> DetRatioState {
    let ao_up = data.orb_up.compute(r_up_carts); // (orb_num_up, N_up)
    let ao_dn = data.orb_dn.compute(r_dn_carts); // (orb_num_dn, N_dn)
    init_det_ratio_state_from_orb_mats(data, ao_up, ao_dn)
}

/// Orbital matrix `(orb_num, N)` built column-by-column via the per-move
/// [`OrbData::compute_single`] path, rather than the batch [`OrbData::compute`].
/// These differ on a PERIODIC CART pGTO leaf: the batch `compute_aos` applies the
/// `LEPSBAS` far-image truncation while `compute_single` (cart) sums all images
/// (as JAX does) -- a ~1e-9 gap. The MCMC sweep seeds its streaming det-ratio
/// state with THIS single-path matrix so the seed is bit-consistent with (a) the
/// per-move `compute_single` ratio stream and (b) the JAX-built `geminal_inv0` it
/// is handed; the batch-seeded state instead drifted the SM-streamed inverse over
/// the sweep and failed the composite pGTO+B-spline force check.
/// Round completed AO values to the `ao_eval` fp32 dtype in place (the cast
/// jqmc's fused `compute_AOs_value_grad_lap` applies to its value output).
#[inline]
pub(crate) fn round_to_ao_eval(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = f64::from(*v as f32);
    }
}

/// [`orb_matrix_via_single`] in the fused value convention
/// ([`OrbData::compute_single_fused`]): the GFMC fast-state seed.  Identical
/// to [`orb_matrix_via_single`] in full precision.
pub(crate) fn orb_matrix_via_single_fused(orb: &OrbData, r: &ArrayView2<f64>) -> Array2<f64> {
    if !orb.precision_policy.is_f32("ao_eval") {
        return orb_matrix_via_single(orb, r);
    }
    let n = r.nrows();
    let m = orb
        .mo_coefficients
        .as_ref()
        .map(|c| c.nrows())
        .unwrap_or_else(|| orb.ao_count());
    let mut out = Array2::<f64>::zeros((m, n));
    for e in 0..n {
        let col = orb.compute_single_fused([r[[e, 0]], r[[e, 1]], r[[e, 2]]]);
        for a in 0..m {
            out[[a, e]] = col[a];
        }
    }
    out
}

pub fn orb_matrix_via_single(orb: &OrbData, r: &ArrayView2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let m = orb
        .mo_coefficients
        .as_ref()
        .map(|c| c.nrows())
        .unwrap_or_else(|| orb.ao_count());
    let mut out = Array2::<f64>::zeros((m, n));
    for e in 0..n {
        let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
        let col = orb.compute_single(pos);
        for a in 0..m {
            out[[a, e]] = col[a];
        }
    }
    out
}

/// Build the streaming det-ratio state from already-evaluated orbital matrices
/// `ao_up` (orb_num_up, N_up) and `ao_dn` (orb_num_dn, N_dn).
pub fn init_det_ratio_state_from_orb_mats(
    data: &GeminalData,
    ao_up: Array2<f64>,
    ao_dn: Array2<f64>,
) -> DetRatioState {
    let paired_dn = if data.is_sd() {
        ao_dn.clone()
    } else {
        data.paired_times(&ao_dn.view())
    };
    let paired_up_lambda = data.up_t_times_paired(&ao_up.view()); // (N_up, orb_num_dn)
    DetRatioState {
        ao_up,
        ao_dn,
        paired_dn,
        paired_up_lambda,
    }
}

/// Rebuild the full geminal `G (N_up, N_up)` from a [`DetRatioState`]'s cached
/// orbital matrices (`ao_up`, `paired_dn`) plus the unpaired lambda columns --
/// the SAME `G = ou^T (lambda_p od) | ou^T lambda_unp` as [`compute_geminal_all_elements`],
/// but from the single-path AO matrices the streaming ratio stream is seeded with.
/// Used by the GFMC fast discretized-kinetic path so the base `G^-1` and every
/// Sherman-Morrison candidate column come from ONE consistent AO evaluation (the
/// mesh det ratios are then exact `det_new/det_base` for that basis).
pub fn geminal_from_det_ratio_state(data: &GeminalData, state: &DetRatioState) -> Array2<f64> {
    let n_up = state.ao_up.ncols();
    let n_dn = state.ao_dn.ncols();
    let n_unpaired = data.n_unpaired();
    // g_paired = ao_up^T * paired_dn  (N_up, N_dn)
    let up_paired = state
        .ao_up
        .slice(numpy::ndarray::s![0..state.paired_dn.nrows(), ..]);
    let g_paired = mat_t_mul(&up_paired, &state.paired_dn.view());
    let mut g = Array2::<f64>::zeros((n_up, n_dn + n_unpaired));
    for i in 0..n_up {
        for j in 0..n_dn {
            g[[i, j]] = g_paired[[i, j]];
        }
    }
    if n_unpaired > 0 {
        for i in 0..n_up {
            for j in 0..n_unpaired {
                let mut value = 0.0;
                for a in 0..state.ao_up.nrows() {
                    value += state.ao_up[[a, i]] * data.lambda_value(a, data.orb_num_dn + j);
                }
                g[[i, n_dn + j]] = value;
            }
        }
    }
    g
}

/// Single orbital column at one new position (orb_num,). Uses the fast
/// shell-reuse single-position path, not the naive batch `compute` -- the mesh
/// det ratios in the ECP nonlocal call this ~Ne*Nmesh times per e_L.
fn orb_col_at(orb: &OrbData, r_new: [f64; 3]) -> Vec<f64> {
    orb.compute_single(r_new)
}

/// `_compute_v_up_move_from_det_ratio_state`: row-delta v (N_up,).
pub fn compute_v_up_move(
    data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    r_up_proposed: [f64; 3],
) -> Vec<f64> {
    let ao_new = orb_col_at(&data.orb_up, r_up_proposed);
    compute_v_up_move_from_orb(data, state, moved_index, &ao_new)
}

/// As [`compute_v_up_move`] but with the orbital column at the new position
/// supplied (computed once per move and reused by the advance).
pub fn compute_v_up_move_from_orb(
    data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    ao_new: &[f64],
) -> Vec<f64> {
    let n_orb_up = ao_new.len();
    let delta: Vec<f64> = (0..n_orb_up)
        .map(|a| ao_new[a] - state.ao_up[[a, moved_index]])
        .collect();
    // v_paired = delta * paired_dn  -> (N_dn,)
    let n_dn = state.paired_dn.ncols();
    let mut v = vec![0.0_f64; n_dn];
    for (a, delta_a) in delta[..state.paired_dn.nrows()].iter().enumerate() {
        let d = *delta_a;
        if d == 0.0 {
            continue;
        }
        for (j, v_j) in v[..n_dn].iter_mut().enumerate() {
            *v_j += d * state.paired_dn[[a, j]];
        }
    }
    // v_unpaired = delta * lambda_unpaired -> (num_unpaired,)
    let n_unpaired = data.n_unpaired();
    for j in 0..n_unpaired {
        let mut s = 0.0;
        for (a, delta_a) in delta[..n_orb_up].iter().enumerate() {
            s += *delta_a * data.lambda_value(a, data.orb_num_dn + j);
        }
        v.push(s);
    }
    v // (N_dn + num_unpaired) = (N_up,)
}

/// `_compute_u_dn_move_from_det_ratio_state`: col-delta u (N_up,).
pub fn compute_u_dn_move(
    data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    r_dn_proposed: [f64; 3],
) -> Vec<f64> {
    let ao_new = orb_col_at(&data.orb_dn, r_dn_proposed);
    compute_u_dn_move_from_orb(data, state, moved_index, &ao_new)
}

/// As [`compute_u_dn_move`] but with the orbital column supplied.
pub fn compute_u_dn_move_from_orb(
    _data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    ao_new: &[f64],
) -> Vec<f64> {
    let n_orb_dn = ao_new.len();
    let delta: Vec<f64> = (0..n_orb_dn)
        .map(|a| ao_new[a] - state.ao_dn[[a, moved_index]])
        .collect();
    // u = paired_up_lambda * delta -> (N_up,)
    let n_up = state.paired_up_lambda.nrows();
    let mut u = vec![0.0_f64; n_up];
    for (i, u_i) in u[..n_up].iter_mut().enumerate() {
        let mut s = 0.0;
        for (a, delta_a) in delta[..n_orb_dn].iter().enumerate() {
            s += state.paired_up_lambda[[i, a]] * *delta_a;
        }
        *u_i = s;
    }
    u
}

/// Batched [`compute_v_up_move_from_orb`] over `npts` proposed orbital columns
/// `orb_mesh` (orb_num_up, npts) -- returns V (N_up, npts). The ECP-nonlocal mesh
/// det ratios use this so the paired contraction is ONE GEMM instead of `npts`
/// matvecs (and the orbital columns come from one batched `orb.compute`).
pub fn compute_v_up_move_batch(
    data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    orb_mesh: &ArrayView2<f64>,
) -> Array2<f64> {
    let orb_num_up = orb_mesh.nrows();
    let npts = orb_mesh.ncols();
    let n_dn = state.paired_dn.ncols();
    let n_unpaired = data.n_unpaired();
    let mut delta = Array2::<f64>::zeros((orb_num_up, npts));
    for a in 0..orb_num_up {
        let old = state.ao_up[[a, moved_index]];
        for v in 0..npts {
            delta[[a, v]] = orb_mesh[[a, v]] - old;
        }
    }
    let mut out = Array2::<f64>::zeros((n_dn + n_unpaired, npts));
    // paired: paired_dnT * delta -> (N_dn, npts)
    let delta_paired = delta.slice(numpy::ndarray::s![0..state.paired_dn.nrows(), ..]);
    let vp = mat_t_mul(&state.paired_dn.view(), &delta_paired);
    for j in 0..n_dn {
        for v in 0..npts {
            out[[j, v]] = vp[[j, v]];
        }
    }
    for j in 0..n_unpaired {
        for v in 0..npts {
            let mut s = 0.0;
            for a in 0..orb_num_up {
                s += delta[[a, v]] * data.lambda_value(a, data.orb_num_dn + j);
            }
            out[[n_dn + j, v]] = s;
        }
    }
    out
}

/// Batched [`compute_u_dn_move_from_orb`]: `orb_mesh` (orb_num_dn, npts) -> U
/// (N_up, npts) via one GEMM `paired_up_lambda * delta`.
pub fn compute_u_dn_move_batch(
    data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    orb_mesh: &ArrayView2<f64>,
) -> Array2<f64> {
    let _ = data;
    let orb_num_dn = orb_mesh.nrows();
    let npts = orb_mesh.ncols();
    let mut delta = Array2::<f64>::zeros((orb_num_dn, npts));
    for a in 0..orb_num_dn {
        let old = state.ao_dn[[a, moved_index]];
        for v in 0..npts {
            delta[[a, v]] = orb_mesh[[a, v]] - old;
        }
    }
    matmul(&state.paired_up_lambda.view(), &delta.view())
}

/// No-alloc [`compute_v_up_move_from_orb`]: fills `v_out[0..N_up]`.
pub fn compute_v_up_move_from_orb_into(
    data: &GeminalData,
    state: &DetRatioState,
    moved_index: usize,
    ao_new: &[f64],
    v_out: &mut [f64],
) {
    let n_orb_up = ao_new.len();
    let n_dn = state.paired_dn.ncols();
    let pd = state.paired_dn.as_slice().expect("paired_dn contiguous");
    for x in v_out.iter_mut() {
        *x = 0.0;
    }
    // v_paired = delta * paired_dn -> (N_dn,)
    for a in 0..state.paired_dn.nrows() {
        let d = ao_new[a] - state.ao_up[[a, moved_index]];
        if d == 0.0 {
            continue;
        }
        let row = &pd[a * n_dn..a * n_dn + n_dn];
        for j in 0..n_dn {
            v_out[j] += d * row[j];
        }
    }
    // v_unpaired = delta * lambda_unpaired -> (num_unpaired,)
    let n_unpaired = data.n_unpaired();
    for j in 0..n_unpaired {
        let mut s = 0.0;
        for (a, ao_new_a) in ao_new[..n_orb_up].iter().enumerate() {
            let d = *ao_new_a - state.ao_up[[a, moved_index]];
            s += d * data.lambda_value(a, data.orb_num_dn + j);
        }
        v_out[n_dn + j] = s;
    }
}

/// No-alloc [`compute_u_dn_move_from_orb`]: fills `u_out[0..N_up]`.
pub fn compute_u_dn_move_from_orb_into(
    state: &DetRatioState,
    moved_index: usize,
    ao_new: &[f64],
    u_out: &mut [f64],
) {
    let n_orb_dn = ao_new.len();
    let n_up = state.paired_up_lambda.nrows();
    let pul = state
        .paired_up_lambda
        .as_slice()
        .expect("paired_up_lambda contiguous");
    for i in 0..n_up {
        let row = &pul[i * n_orb_dn..i * n_orb_dn + n_orb_dn];
        let mut s = 0.0;
        for a in 0..n_orb_dn {
            s += row[a] * (ao_new[a] - state.ao_dn[[a, moved_index]]);
        }
        u_out[i] = s;
    }
}

/// Issue #134: proposed geminal row `G_new[k, :]` of an up move, built directly
/// from the new orbital column (`compute_v_up_move_from_orb_into` with `ao_new`
/// in place of `ao_new - ao_old`). `G[k, :] + v` cancels to round-off (or exactly
/// zero) once `|G_new[k, :]| < eps |G[k, :]|`; the AS factor of the proposal needs
/// the direct value.
pub fn compute_up_row_from_orb_into(
    data: &GeminalData,
    state: &DetRatioState,
    ao_new: &[f64],
    row_out: &mut [f64],
) {
    let n_orb_up = ao_new.len();
    let n_dn = state.paired_dn.ncols();
    let pd = state.paired_dn.as_slice().expect("paired_dn contiguous");
    for x in row_out.iter_mut() {
        *x = 0.0;
    }
    for a in 0..state.paired_dn.nrows() {
        let d = ao_new[a];
        if d == 0.0 {
            continue;
        }
        let row = &pd[a * n_dn..a * n_dn + n_dn];
        for j in 0..n_dn {
            row_out[j] += d * row[j];
        }
    }
    let n_unpaired = data.n_unpaired();
    for j in 0..n_unpaired {
        let mut s = 0.0;
        for (a, ao_new_a) in ao_new[..n_orb_up].iter().enumerate() {
            s += *ao_new_a * data.lambda_value(a, data.orb_num_dn + j);
        }
        row_out[n_dn + j] = s;
    }
}

/// Issue #134: proposed geminal column `G_new[:, k]` of a dn move, built directly
/// from the new orbital column (see [`compute_up_row_from_orb_into`]).
pub fn compute_dn_col_from_orb_into(state: &DetRatioState, ao_new: &[f64], col_out: &mut [f64]) {
    let n_orb_dn = ao_new.len();
    let n_up = state.paired_up_lambda.nrows();
    let pul = state
        .paired_up_lambda
        .as_slice()
        .expect("paired_up_lambda contiguous");
    for i in 0..n_up {
        let row = &pul[i * n_orb_dn..i * n_orb_dn + n_orb_dn];
        let mut s = 0.0;
        for a in 0..n_orb_dn {
            s += row[a] * ao_new[a];
        }
        col_out[i] = s;
    }
}

/// No-alloc Sherman-Morrison that computes only the det ratio (`1 + vTG^-^1u`)
/// plus the rank-1 factors `ainv_u = G^-^1u` and `vt_ainv = vTG^-^1` needed to
/// update the inverse/geminal in place on accept. No 80x80 copies per move.
#[inline]
pub fn sm_ratio_into(
    geminal_inv: &ArrayView2<f64>,
    u: &[f64],
    v: &[f64],
    ainv_u: &mut [f64],
    vt_ainv: &mut [f64],
) -> f64 {
    let n = u.len();
    let gi = geminal_inv.as_slice().expect("G_inv contiguous");
    for x in vt_ainv.iter_mut() {
        *x = 0.0;
    }
    for i in 0..n {
        let row = &gi[i * n..i * n + n];
        let mut s = 0.0;
        for k in 0..n {
            s += row[k] * u[k];
        }
        ainv_u[i] = s;
        let vi = v[i];
        if vi != 0.0 {
            for j in 0..n {
                vt_ainv[j] += vi * row[j];
            }
        }
    }
    let mut ratio = 1.0_f64;
    for k in 0..n {
        ratio += v[k] * ainv_u[k];
    }
    ratio
}

/// Issue #127: determinant-ratio magnitude above which the accepted rank-1
/// inverse update writes the moved electron's inverse column (up move) or row
/// (dn move) exactly. ``1 / sqrt(f64::EPSILON) = 2**26``, the float64 value of
/// the Python drivers' ``jqmc._setting.sm_large_ratio_threshold``. Not a user knob.
pub use crate::setting_consts::SM_LARGE_RATIO_THRESHOLD_FP64 as SM_LARGE_RATIO_THRESHOLD;

/// Issue #127: after the rank-1 update ``X - (X u)(vT X) / R`` of a row-major
/// inverse, overwrite column ``k`` (up move, ``u = e_k``) by ``X e_k / R`` or
/// row ``k`` (dn move, ``v = e_k``) by ``e_kT X / R`` when ``|R|`` exceeds
/// [`SM_LARGE_RATIO_THRESHOLD`]. The rank-1 form evaluates that column / row as
/// ``X e_k (1 - (R - 1) / R)`` and loses ``log10(eps |R|)`` digits (all of them
/// once ``|R| > 1/eps``), after which the next move's ratio and Metropolis
/// decision are wrong. Same arithmetic as the XLA
/// ``determinant._sherman_morrison_ratio_and_inverse``; a no-op below the
/// threshold, so ordinary moves stay bit-identical.
#[inline]
pub fn sm_large_ratio_exact_slice(
    ins: &mut [f64],
    n: usize,
    ainv_u: &[f64],
    vt_ainv: &[f64],
    ratio: f64,
    moved_spin_is_up: bool,
    moved_index: usize,
) {
    if ratio.abs().partial_cmp(&SM_LARGE_RATIO_THRESHOLD) != Some(std::cmp::Ordering::Greater) {
        return;
    }
    if moved_spin_is_up {
        for i in 0..n {
            ins[i * n + moved_index] = ainv_u[i] / ratio;
        }
    } else {
        for j in 0..n {
            ins[moved_index * n + j] = vt_ainv[j] / ratio;
        }
    }
}

/// Apply the accepted rank-1 update IN PLACE: `G^-^1 -= (ainv_u (x) vt_ainv)/ratio`
/// and `G += e_i vT` (up) or `G += u e_jT` (dn). Only called on accept (~1/3 of
/// moves), so the O(N^2) write never happens on a rejected move.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "rank-1 Sherman-Morrison update operands"
)]
pub fn sm_apply_inplace(
    geminal_inv: &mut Array2<f64>,
    geminal: &mut Array2<f64>,
    u: &[f64],
    v: &[f64],
    ainv_u: &[f64],
    vt_ainv: &[f64],
    ratio: f64,
    moved_spin_is_up: bool,
    moved_index: usize,
) {
    let n = u.len();
    let ins = geminal_inv.as_slice_mut().expect("inv contiguous");
    for i in 0..n {
        let ai = ainv_u[i] / ratio;
        if ai == 0.0 {
            continue;
        }
        let row = &mut ins[i * n..i * n + n];
        for j in 0..n {
            row[j] -= ai * vt_ainv[j];
        }
    }
    sm_large_ratio_exact_slice(
        ins,
        n,
        ainv_u,
        vt_ainv,
        ratio,
        moved_spin_is_up,
        moved_index,
    );
    if moved_spin_is_up {
        let gs = geminal.as_slice_mut().expect("G contiguous");
        let row = &mut gs[moved_index * n..moved_index * n + n];
        for j in 0..n {
            row[j] += v[j];
        }
    } else {
        for i in 0..n {
            geminal[[i, moved_index]] += u[i];
        }
    }
}

/// `_advance_det_ratio_streaming_state` after an accepted single-electron move.
pub fn advance_det_ratio_state(
    data: &GeminalData,
    state: &mut DetRatioState,
    moved_spin_is_up: bool,
    moved_index: usize,
    r_new: [f64; 3],
) {
    let orb = if moved_spin_is_up {
        &data.orb_up
    } else {
        &data.orb_dn
    };
    let ao_col = orb_col_at(orb, r_new);
    advance_det_ratio_state_from_orb(data, state, moved_spin_is_up, moved_index, &ao_col);
}

/// As [`advance_det_ratio_state`] but with the orbital column supplied.
pub fn advance_det_ratio_state_from_orb(
    data: &GeminalData,
    state: &mut DetRatioState,
    moved_spin_is_up: bool,
    moved_index: usize,
    ao_col: &[f64],
) {
    if moved_spin_is_up {
        for (a, ao_col_a) in ao_col.iter().enumerate() {
            state.ao_up[[a, moved_index]] = *ao_col_a;
        }
        // paired_up_lambda[k, :] = ao_col * lambda_paired  (orb_num_dn,)
        for j in 0..data.orb_num_dn {
            let mut s = 0.0;
            for (a, ao_col_a) in ao_col.iter().enumerate() {
                s += *ao_col_a * data.lambda_value(a, j);
            }
            state.paired_up_lambda[[moved_index, j]] = s;
        }
    } else {
        for (a, ao_col_a) in ao_col.iter().enumerate() {
            state.ao_dn[[a, moved_index]] = *ao_col_a;
        }
        // paired_dn[:, k] = lambda_paired * ao_col  (orb_num_up,)
        let orb_num_up = state.paired_dn.nrows();
        for i in 0..orb_num_up {
            let mut s = 0.0;
            for (a, ao_col_a) in ao_col.iter().enumerate() {
                s += data.lambda_value(i, a) * *ao_col_a;
            }
            state.paired_dn[[i, moved_index]] = s;
        }
    }
}

/// Rank-1 Sherman-Morrison det ratio + inverse + matrix update.
/// up-move: `u = e_index`, `v = row-delta`; dn-move: `u = col-delta`, `v = e_index`.
/// Returns `(Det_T_ratio, geminal_inv_new, geminal_new)`.
pub fn sm_update(
    geminal_inv: &ArrayView2<f64>,
    geminal: &ArrayView2<f64>,
    u: &[f64],
    v: &[f64],
    moved_spin_is_up: bool,
    moved_index: usize,
) -> (f64, Array2<f64>, Array2<f64>) {
    let n = u.len();
    let gi = geminal_inv.as_slice().expect("G_inv contiguous");
    // Ainv_u = G^{-1} * u  (row-sequential); vT_Ainv = vT * G^{-1} (accumulate over rows).
    let mut ainv_u = vec![0.0_f64; n];
    let mut vt_ainv = vec![0.0_f64; n];
    for i in 0..n {
        let row = &gi[i * n..i * n + n];
        let mut s = 0.0;
        for k in 0..n {
            s += row[k] * u[k];
        }
        ainv_u[i] = s;
        let vi = v[i];
        if vi != 0.0 {
            for j in 0..n {
                vt_ainv[j] += vi * row[j];
            }
        }
    }
    // Det_T_ratio = 1 + vT Ainv_u
    let mut ratio = 1.0_f64;
    for k in 0..n {
        ratio += v[k] * ainv_u[k];
    }
    // G^{-1}_new = G^{-1} - (Ainv_u (x) vT_Ainv) / ratio (raw-slice rank-1 update).
    let mut inv_new = geminal_inv.to_owned();
    {
        let ins = inv_new.as_slice_mut().expect("inv contiguous");
        for i in 0..n {
            let ai = ainv_u[i] / ratio;
            if ai == 0.0 {
                continue;
            }
            let row = &mut ins[i * n..i * n + n];
            for j in 0..n {
                row[j] -= ai * vt_ainv[j];
            }
        }
        sm_large_ratio_exact_slice(
            ins,
            n,
            &ainv_u,
            &vt_ainv,
            ratio,
            moved_spin_is_up,
            moved_index,
        );
    }
    // G_new = G + e_i vT (up) or G + u e_jT (dn)
    let mut gem_new = geminal.to_owned();
    if moved_spin_is_up {
        let gs = gem_new.as_slice_mut().expect("G contiguous");
        let row = &mut gs[moved_index * n..moved_index * n + n];
        for j in 0..n {
            row[j] += v[j];
        }
    } else {
        for i in 0..n {
            gem_new[[i, moved_index]] += u[i];
        }
    }
    (ratio, inv_new, gem_new)
}

/// Orbital value + gradient (x,y,z) + Laplacian at `r_carts`, each shape
/// `(orb_num, N_e)`. AO arrays are projected by the MO coefficients (linear,
/// so grad/lap project the same way) when this block is an MO basis.
pub(crate) fn orb_value_grad_lap(
    orb: &OrbData,
    r: &ArrayView2<f64>,
) -> crate::atomic_orbital::ValueGradLap {
    orb_value_grad_lap_cut(orb, r, false)
}

/// [`orb_value_grad_lap`] whose molecular GTO grad/Laplacian (no PW / B-spline
/// child) is the FAST derivative-screened `compute_aos_grad_lap_fast_cut` when
/// `fast_cut` -- the tables the FAST e_L kinetic contracts
/// ([`init_det_grad_lap_cache`]). Values are unchanged.
pub(crate) fn orb_value_grad_lap_cut(
    orb: &OrbData,
    r: &ArrayView2<f64>,
    fast_cut: bool,
) -> crate::atomic_orbital::ValueGradLap {
    if let Some(bs) = &orb.bspline {
        let (bv, bgx, bgy, bgz, blap) = bs.value_grad_lap(r);
        // Pure B-spline leaf (0-AO gto placeholder): the spline block IS the whole
        // AO stack. Composite pgto+bspline (`orb.aos.num_ao > 0`): concatenate the
        // gto grad/laplacian in the child (`gto_first`) order -- the VALUE path
        // `eval_aos` / `compute_single` concatenate the same way. The GTO block
        // must be kept: the MO-projection matmul indexes the full
        // `(num_ao_total, N)` stack of a periodic pGTO+B-spline orbital.
        let (av, gx, gy, gz, lap) = if orb.aos.num_ao == 0 {
            (bv, bgx, bgy, bgz, blap)
        } else {
            // EXACT (un-screened) value, matching the un-screened grad/lap below --
            // value and grad/lap must share one AO convention or dT/dC picks up a
            // term on only one of them.
            let gv = crate::atomic_orbital::compute_aos_fused_value(&orb.aos, r);
            let (ggx, ggy, ggz, glap) = crate::atomic_orbital::compute_aos_grad_lap(&orb.aos, r);
            let cat = |a: &Array2<f64>, b: &Array2<f64>| -> Array2<f64> {
                let (x, y) = if orb.gto_first { (a, b) } else { (b, a) };
                numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[x.view(), y.view()]).unwrap()
            };
            (
                cat(&gv, &bv),
                cat(&ggx, &bgx),
                cat(&ggy, &bgy),
                cat(&ggz, &bgz),
                cat(&glap, &blap),
            )
        };
        return match &orb.mo_coefficients {
            None => (av, gx, gy, gz, lap),
            Some(c) => (
                matmul(&c.view(), &av.view()),
                matmul(&c.view(), &gx.view()),
                matmul(&c.view(), &gy.view()),
                matmul(&c.view(), &gz.view()),
                matmul(&c.view(), &lap.view()),
            ),
        };
    }
    let av = crate::timed!(
        "kdet.ao_val",
        crate::atomic_orbital::compute_aos_fused_value(&orb.aos, r)
    );
    // Fused grad+lap in one AO sweep (not two -- compute_aos_grad + _laplacian, each
    // re-doing the radial exp). Value keeps the SIMD `compute_aos` path (bit-identical).
    let (gx, gy, gz, lap) = crate::timed!(
        "kdet.ao_gradlap",
        if fast_cut && orb.pw.is_none() && orb.aos.n_image == 1 {
            crate::atomic_orbital::compute_aos_grad_lap_fast_cut(&orb.aos, r)
        } else {
            crate::atomic_orbital::compute_aos_grad_lap(&orb.aos, r)
        }
    );
    // Concatenate the plane-wave child (value/grad/lap) for a pgto(+)pw composite.
    let (av, gx, gy, gz, lap) = match &orb.pw {
        None => (av, gx, gy, gz, lap),
        Some(pw) => {
            let (pv, pgx, pgy, pgz, plap) = crate::atomic_orbital::compute_pw_value_grad_lap(pw, r);
            let cat = |a: &Array2<f64>, b: &Array2<f64>| -> Array2<f64> {
                let (x, y) = if orb.gto_first { (a, b) } else { (b, a) };
                numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[x.view(), y.view()]).unwrap()
            };
            (
                cat(&av, &pv),
                cat(&gx, &pgx),
                cat(&gy, &pgy),
                cat(&gz, &pgz),
                cat(&lap, &plap),
            )
        }
    };
    match &orb.mo_coefficients {
        None => (av, gx, gy, gz, lap),
        Some(c) => {
            // Five projections share the SAME coefficient matrix C (orb_num x
            // num_ao, ~1.5MB, doesn't fit L2). Five separate GEMMs re-stream C
            // from DRAM 5x, which bounds the e_L by memory bandwidth when many
            // MPI ranks share a node. Stack the AO
            // components into one (num_ao, 5*Ne) matrix so a single wide GEMM
            // reads C once (faer's column panel >= 5*Ne here). Stack/slice copies
            // are sequential and cheap next to the 160x1140x400 GEMM.
            let na = av.nrows();
            let ne = av.ncols();
            let mut stacked = Array2::<f64>::zeros((na, 5 * ne));
            for a in 0..na {
                for e in 0..ne {
                    stacked[[a, e]] = av[[a, e]];
                    stacked[[a, ne + e]] = gx[[a, e]];
                    stacked[[a, 2 * ne + e]] = gy[[a, e]];
                    stacked[[a, 3 * ne + e]] = gz[[a, e]];
                    stacked[[a, 4 * ne + e]] = lap[[a, e]];
                }
            }
            let y = crate::timed!("kdet.mo_gemm", matmul(&c.view(), &stacked.view())); // (orb_num, 5*ne)
            let sl = |k: usize| {
                y.slice(numpy::ndarray::s![.., k * ne..(k + 1) * ne])
                    .to_owned()
            };
            (sl(0), sl(1), sl(2), sl(3), sl(4))
        }
    }
}

/// Orbital value + grad + laplacian, `(orb_num, N)` each, built with the per-move
/// single full-image AO convention on the periodic GTO block. This is the
/// grad/laplacian sibling of [`orb_matrix_via_single`] and is used by the
/// OBSERVABLE reused-inverse kinetic (`compute_grads_and_laplacian_ln_det_with_inv`
/// -> `compute_kinetic_energy_with_inv` -> `compute_local_energy_fast`), so its
/// value matrix AND grad^2 G are convention-consistent with the sweep's reused SM
/// inverse (`inv(G_single)`), which the batch [`orb_value_grad_lap`] is NOT (the
/// ~1e-9 single-vs-batch periodic-image gap, amplified by the Gaussian curvature
/// into ~0.06 Ha). Molecular GTO / pure PW / B-spline are already
/// convention-consistent, so they DELEGATE to the batch path -- keeping the
/// molecular kinetic BIT-IDENTICAL.
pub(crate) fn orb_value_grad_lap_via_single(
    orb: &OrbData,
    r: &ArrayView2<f64>,
) -> crate::atomic_orbital::ValueGradLap {
    // Only a PERIODIC GTO block (n_image > 1) differs between the batch and single
    // AO conventions. B-spline (exact eval), pure PW (cos/sin, wrap-invariant) and
    // the molecular GTO leaf are already consistent -> reuse the batch path (this
    // also guarantees molecular bit-identity).
    if orb.bspline.is_some() || orb.aos.n_image <= 1 {
        return orb_value_grad_lap(orb, r);
    }
    let n = r.nrows();
    let num_gto = orb.aos.num_ao;
    let mut av = Array2::<f64>::zeros((num_gto, n));
    let mut gx = Array2::<f64>::zeros((num_gto, n));
    let mut gy = Array2::<f64>::zeros((num_gto, n));
    let mut gz = Array2::<f64>::zeros((num_gto, n));
    let mut lap = Array2::<f64>::zeros((num_gto, n));
    crate::timed!(
        "kd.gto_gl",
        for e in 0..n {
            let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            // Value AND grad/laplacian from the SAME single-position `ao_value_grad_lap`
            // sweep (num_ao_kernel iteration), so they are mutually self-consistent and
            // in the single (full-image) AO convention the reused SM inverse uses.
            // Value in the fused `ao_eval` convention (per-image round + fp32
            // fold, as `compute_AOs_value_grad_lap`), derivatives fp64.
            let (vcol, gxc, gyc, gzc, lc) =
                crate::atomic_orbital::compute_aos_single_value_grad_lap_with_policy(
                    &orb.aos,
                    pos,
                    orb.aos.precision_policy.is_f32("ao_eval"),
                );
            for a in 0..num_gto {
                av[[a, e]] = vcol[a];
                gx[[a, e]] = gxc[a];
                gy[[a, e]] = gyc[a];
                gz[[a, e]] = gzc[a];
                lap[[a, e]] = lc[a];
            }
        }
    );
    // Plane-wave child (batch value+grad+lap is convention-consistent).
    let (av, gx, gy, gz, lap) = match &orb.pw {
        None => (av, gx, gy, gz, lap),
        Some(pw) => {
            let (pv, pgx, pgy, pgz, plap) = crate::timed!(
                "kd.pw_gl",
                crate::atomic_orbital::compute_pw_value_grad_lap(pw, r)
            );
            let cat = |a: &Array2<f64>, b: &Array2<f64>| -> Array2<f64> {
                let (x, y) = if orb.gto_first { (a, b) } else { (b, a) };
                numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[x.view(), y.view()]).unwrap()
            };
            (
                cat(&av, &pv),
                cat(&gx, &pgx),
                cat(&gy, &pgy),
                cat(&gz, &pgz),
                cat(&lap, &plap),
            )
        }
    };
    // MO projection (same fused wide-GEMM as `orb_value_grad_lap`).
    match &orb.mo_coefficients {
        None => (av, gx, gy, gz, lap),
        Some(c) => {
            let na = av.nrows();
            let ne = av.ncols();
            let mut stacked = Array2::<f64>::zeros((na, 5 * ne));
            for a in 0..na {
                for e in 0..ne {
                    stacked[[a, e]] = av[[a, e]];
                    stacked[[a, ne + e]] = gx[[a, e]];
                    stacked[[a, 2 * ne + e]] = gy[[a, e]];
                    stacked[[a, 3 * ne + e]] = gz[[a, e]];
                    stacked[[a, 4 * ne + e]] = lap[[a, e]];
                }
            }
            let y = matmul(&c.view(), &stacked.view());
            let sl = |k: usize| {
                y.slice(numpy::ndarray::s![.., k * ne..(k + 1) * ne])
                    .to_owned()
            };
            (sl(0), sl(1), sl(2), sl(3), sl(4))
        }
    }
}

/// AO-layer value + grad(x,y,z) + laplacian, each `(num_ao_total, N)`, BEFORE any
/// MO projection -- the raw gto(+pw) or B-spline AO stack in `mo_coefficients`
/// column order. `grad_t_mo`'s `dT/dC` contracts this AO layer directly, so the
/// composite plane-wave child and pure-PW leaves are included, not only the GTO
/// block `compute_aos(orb.aos)`.
pub(crate) fn aos_stack_value_grad_lap(
    orb: &OrbData,
    r: &ArrayView2<f64>,
) -> crate::atomic_orbital::ValueGradLap {
    if let Some(bs) = &orb.bspline {
        return bs.value_grad_lap(r);
    }
    let av = crate::atomic_orbital::compute_aos_exact(&orb.aos, r);
    let (gx, gy, gz, lap) = crate::atomic_orbital::compute_aos_grad_lap(&orb.aos, r);
    match &orb.pw {
        None => (av, gx, gy, gz, lap),
        Some(pw) => {
            let (pv, pgx, pgy, pgz, plap) = crate::atomic_orbital::compute_pw_value_grad_lap(pw, r);
            let cat = |a: &Array2<f64>, b: &Array2<f64>| -> Array2<f64> {
                let (x, y) = if orb.gto_first { (a, b) } else { (b, a) };
                numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[x.view(), y.view()]).unwrap()
            };
            (
                cat(&av, &pv),
                cat(&gx, &pgx),
                cat(&gy, &pgy),
                cat(&gz, &pgz),
                cat(&lap, &plap),
            )
        }
    }
}

/// `compute_grads_and_laplacian_ln_Det` (determinant.py): per-electron
/// `grad_i ln|det G|` (shape `(N,3)`) and `grad^2_i ln|det G|` (shape `(N,)`).
/// Uses `grad_i ln|det G| = tr(G^-^1 grad_i G)` and `grad^2_i ln|det G| = tr(G^-^1 grad^2_i G) -
/// |grad_i ln|det G||^2`. Contractions reduced to two GEMMs (MG = M*G^-^1,
/// GU = O_up*G^-T) plus column-wise dots (1:1 with the jqmc einsums).
pub fn compute_grads_and_laplacian_ln_det(
    data: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    // Periodicity guarantor (real-Gamma pGTO), applied ONCE at this OBSERVABLE
    // entry so BOTH the geminal G (built from AO VALUES) and the AO grad/lap are
    // evaluated at the SAME home-cell-wrapped coordinate. jqmc wraps every electron
    // coordinate into the home cell (`_wrap_into_cell_if_pbc`) before an AO eval:
    // the truncated image sum is enumerated relative to the home cell, so a raw
    // coordinate several lattice vectors out falls beyond the static tiling and
    // grad/lap, and a near-singular g_inv if G is built raw while grad/lap is
    // wrapped, blow the kinetic up. Wrapping here rather than in the
    // shared low-level AO evaluations keeps the per-move MCMC sweep value path
    // (`compute_aos` via `OrbData::eval_aos`) identical to JAX's accept/reject
    // calculation; grad/lap is never on the sweep's Metropolis path. No-op for molecular
    // (wrap_cell == None) and for PW (e^{iG.(r+T)} = e^{iG.r}, translation-invariant).
    let rwu = crate::atomic_orbital::wrap_r(&data.orb_up.aos, r_up);
    let rwd = crate::atomic_orbital::wrap_r(&data.orb_dn.aos, r_dn);
    let (vu, vd) = (rwu.view(), rwd.view());
    let (r_up, r_dn) = (&vu, &vd);
    // Fresh-inverse entry (1:1 jqmc `compute_grads_and_laplacian_ln_Det`): build G
    // and invert it here, then delegate to the shared body.
    //
    // G comes from the FUSED-VALUE orbital tables -- the same value convention
    // `orb_value_grad_lap` feeds the contractions below -- and NOT from
    // `compute_geminal_all_elements`, whose standalone `eval_aos` kernel performs
    // its zone-local arithmetic in fp32 under a mixed `ao_eval` policy. jqmc's
    // kinetic (determinant.py `compute_grads_and_laplacian_ln_Det`) builds G from
    // its `_orbital_value_grad_lap_tables`, whose shared analytic body runs in the
    // fp64 `ao_grad_lap` zone with only the completed value rounded; feeding a
    // full-f32-arithmetic G into `inverse` here makes the mixed-precision Rust kinetic
    // deviate from the fp64 answer 136x more than XLA on the water-ccECP/ccpVQZ
    // fixture (+3.0e-5 vs -2.2e-7 on e_L; the deviation bisects to the ao_eval zone).
    // In full precision both kernels reduce to the identical
    // `fold_image_axis(compute_aos_kernel_screened(.., false))` body, so this
    // choice is bit-for-bit neutral there. The tables are recomputed by the
    // shared body below; this entry is not on the per-move MCMC sweep (which
    // reuses the SM inverse via `..._with_inv`), so the extra pass stays off the
    // hot path.
    let g = {
        let (ou, _gx, _gy, _gz, _lap) = orb_value_grad_lap(&data.orb_up, r_up);
        let (od, _dx, _dy, _dz, _dl) = orb_value_grad_lap(&data.orb_dn, r_dn);
        let n_up = ou.ncols();
        let w = data.w_from_dn(&od.view(), n_up);
        mat_t_mul(&ou.view(), &w.view())
    }; // (n_up, n_up)
    let g_inv = inverse(&g.view());
    // BATCH AO convention: `g_inv` here is inv(G) with G from the BATCH fused
    // tables above, so the grad/lap must use the batch path too
    // (self-consistent). `via_single = false`.
    compute_grads_and_laplacian_ln_det_with_inv(data, r_up, r_dn, &g_inv.view(), false)
}

/// Determinant grad/laplacian reusing a caller-supplied geminal inverse
/// `g_inv == G(r_up, r_dn)^-1`. Identical body to
/// `compute_grads_and_laplacian_ln_det` but skips the fresh LU so the MCMC
/// observable path reuses the Sherman-Morrison running inverse the sweep
/// emits -- matching jqmc's `compute_kinetic_energy_all_elements_fast_update`.
/// The fresh LU can NaN at a near-singular geminal whereas the AS-regularized
/// SM inverse is stable (jqmc `compute_local_energy_fast` docstring).
///
/// `via_single` selects the AO convention for the value + grad/laplacian, which
/// must match the convention used to build `g_inv`:
/// `false` = BATCH `orb_value_grad_lap` (for the fresh-LU entry, whose `g_inv`
/// comes from the batch `compute_geminal_all_elements`); `true` = SINGLE
/// `orb_value_grad_lap_via_single` (for the OBSERVABLE path, whose reused SM
/// `g_inv` is seeded/streamed via the per-move single AO path). Molecular / pure
/// PW / B-spline are convention-consistent, so `true` delegates to the batch path
/// there -> molecular kinetic stays BIT-IDENTICAL.
pub fn compute_grads_and_laplacian_ln_det_with_inv(
    data: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: &ArrayView2<f64>,
    via_single: bool,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();

    // The observable `via_single` path home-cell-wraps the determinant AO
    // coordinate, replicating jqmc's `compute_AOs`, which wraps
    // internally (`_wrap_into_cell_if_pbc`) while the MCMC walker stays UNWRAPPED.
    // So the determinant grad/laplacian is the full-image value (consistent with
    // the reused SM inverse = inv(G(full-image))) even for an electron sitting
    // several cells out, WHILE the Jastrow (evaluated by the caller at the RAW
    // coordinate, matching jqmc's non-invariant reference kinetic) is untouched.
    // No-op for molecular / PW. The fresh-LU (`via_single == false`) entry already
    // wrapped at its call site, so it stays raw here.
    let ru_owned = if via_single {
        crate::atomic_orbital::wrap_r(&data.orb_up.aos, r_up)
    } else {
        r_up.to_owned()
    };
    let rd_owned = if via_single {
        crate::atomic_orbital::wrap_r(&data.orb_dn.aos, r_dn)
    } else {
        r_dn.to_owned()
    };
    let ru = ru_owned.view();
    let rd = rd_owned.view();
    let ao_vgl = |orb: &OrbData, r: &ArrayView2<f64>| {
        if via_single {
            orb_value_grad_lap_via_single(orb, r)
        } else {
            orb_value_grad_lap(orb, r)
        }
    };
    let (ou, oux, ouy, ouz, oul) = ao_vgl(&data.orb_up, &ru); // (orb_num_up, n_up)
    let (_od, odx, ody, odz, odl) = ao_vgl(&data.orb_dn, &rd); // (orb_num_dn, n_dn)
    let od = _od;

    let orb_num_up = ou.nrows();
    let m = data.w_from_dn(&od.view(), n_up);
    let mg = crate::timed!("kdet.lam_gemm", matmul(&m.view(), &g_inv.view())); // (orb_num_up, n_up)
    let gu = crate::timed!("kdet.lam_gemm", matmul(&ou.view(), &g_inv.t())); // O_up * G^-T  (orb_num_up, n_up)

    // lambda_paired * O_dn grad/lap  (orb_num_up, n_dn)
    let pdn_gx = crate::timed!("kdet.lam_gemm", data.paired_times(&odx.view()));
    let pdn_gy = crate::timed!("kdet.lam_gemm", data.paired_times(&ody.view()));
    let pdn_gz = crate::timed!("kdet.lam_gemm", data.paired_times(&odz.view()));
    let pdn_lap = crate::timed!("kdet.lam_gemm", data.paired_times(&odl.view()));

    // up electrons
    let mut grad_up = Array2::<f64>::zeros((n_up, 3));
    let mut lap_up = Array1::<f64>::zeros(n_up);
    for i in 0..n_up {
        let (mut gxs, mut gys, mut gzs, mut tr) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..orb_num_up {
            let mgi = mg[[a, i]];
            gxs += oux[[a, i]] * mgi;
            gys += ouy[[a, i]] * mgi;
            gzs += ouz[[a, i]] * mgi;
            tr += oul[[a, i]] * mgi;
        }
        grad_up[[i, 0]] = gxs;
        grad_up[[i, 1]] = gys;
        grad_up[[i, 2]] = gzs;
        lap_up[i] = -(gxs * gxs + gys * gys + gzs * gzs) + tr;
    }

    // dn electrons (paired only)
    let mut grad_dn = Array2::<f64>::zeros((n_dn, 3));
    let mut lap_dn = Array1::<f64>::zeros(n_dn);
    for i in 0..n_dn {
        let (mut gxs, mut gys, mut gzs, mut tr) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..orb_num_up {
            let gui = gu[[a, i]];
            gxs += pdn_gx[[a, i]] * gui;
            gys += pdn_gy[[a, i]] * gui;
            gzs += pdn_gz[[a, i]] * gui;
            tr += pdn_lap[[a, i]] * gui;
        }
        grad_dn[[i, 0]] = gxs;
        grad_dn[[i, 1]] = gys;
        grad_dn[[i, 2]] = gzs;
        lap_dn[i] = -(gxs * gxs + gys * gys + gzs * gzs) + tr;
    }

    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Fast molecular determinant grad/laplacian reusing the sweep's streaming
/// state: the orbital value matrices and the
/// `lambda_paired*O_dn` product are taken from [`DetRatioState`] (already owned
/// by the chunk loop) instead of re-evaluated, and the AO grad/lap sweep applies
/// the molecular FAST derivative shell cutoff ([`compute_aos_grad_lap_fast_cut`]:
/// every dropped AO gradient component / Laplacian is `< FAST_SCREEN_ABS_BOUND = 1e-12`
/// in absolute AO units, see `Shell::pz_deriv`).
/// MOLECULAR MO-projected GTO only (caller-gated); the exact
/// [`compute_grads_and_laplacian_ln_det_with_inv`] stays the bit-parity path.
pub struct DetGradLapCache {
    up_x: Array2<f64>,
    up_y: Array2<f64>,
    up_z: Array2<f64>,
    up_lap: Array2<f64>,
    dn_x: Array2<f64>,
    dn_y: Array2<f64>,
    dn_z: Array2<f64>,
    dn_lap: Array2<f64>,
}

impl DetGradLapCache {
    /// The carried MO derivative tables `([x, y, z, lap] up, [x, y, z, lap] dn)`,
    /// each `(M, N)` (the screened FAST kinetic tables at the cache's positions).
    pub(crate) fn mo_tables(&self) -> ([&Array2<f64>; 4], [&Array2<f64>; 4]) {
        (
            [&self.up_x, &self.up_y, &self.up_z, &self.up_lap],
            [&self.dn_x, &self.dn_y, &self.dn_z, &self.dn_lap],
        )
    }
}

/// Build the TurboRVB-style derivative-orbital table used by the GFMC interval.
/// Both spin blocks contain raw MO derivatives.  The down-spin determinant
/// weights already include the paired lambda block, so keeping the derivative
/// table in MO space lets the GFMC interval share the exact contraction rows
/// used by its candidate-ratio mesh.
pub fn init_det_grad_lap_cache(
    data: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DetGradLapCache {
    // MO-projected AO grad/lap, 4-wide stacked GEMM (value comes from `st`).
    let mo_grad_lap = |orb: &OrbData, r: &ArrayView2<f64>| {
        let (gx, gy, gz, lap) = crate::timed!(
            "kdet.ao_gradlap",
            crate::atomic_orbital::compute_aos_grad_lap_fast_cut(&orb.aos, r)
        );
        let c = orb
            .mo_coefficients
            .as_ref()
            .expect("fast kinetic requires MO");
        let na = gx.nrows();
        let ne = gx.ncols();
        let mut stacked = Array2::<f64>::zeros((na, 4 * ne));
        for a in 0..na {
            for e in 0..ne {
                stacked[[a, e]] = gx[[a, e]];
                stacked[[a, ne + e]] = gy[[a, e]];
                stacked[[a, 2 * ne + e]] = gz[[a, e]];
                stacked[[a, 3 * ne + e]] = lap[[a, e]];
            }
        }
        let y = crate::timed!("kdet.mo_gemm", matmul(&c.view(), &stacked.view()));
        let sl = |k: usize| {
            y.slice(numpy::ndarray::s![.., k * ne..(k + 1) * ne])
                .to_owned()
        };
        (sl(0), sl(1), sl(2), sl(3))
    };
    let (oux, ouy, ouz, oul) = mo_grad_lap(&data.orb_up, r_up);
    let (odx, ody, odz, odl) = mo_grad_lap(&data.orb_dn, r_dn);
    DetGradLapCache {
        up_x: oux,
        up_y: ouy,
        up_z: ouz,
        up_lap: oul,
        dn_x: odx,
        dn_y: ody,
        dn_z: odz,
        dn_lap: odl,
    }
}

/// Replace only the moved electron's derivative column. This is the direct
/// analogue of TurboRVB updating one `winvup/winvdo` row after a hop.
pub fn update_det_grad_lap_cache_and_value(
    data: &GeminalData,
    cache: &mut DetGradLapCache,
    is_up: bool,
    electron: usize,
    pos: [f64; 3],
) -> Vec<f64> {
    let orb = if is_up { &data.orb_up } else { &data.orb_dn };
    let na = orb.aos.num_ao_kernel;
    let mut value = vec![0.0; na];
    let mut gx = vec![0.0; na];
    let mut gy = vec![0.0; na];
    let mut gz = vec![0.0; na];
    let mut lap = vec![0.0; na];
    crate::timed!(
        "gfmc.adv_det_vgl_ao",
        crate::atomic_orbital::compute_aos_single_value_grad_lap_fast_cut_into(
            &orb.aos, pos, &mut value, &mut gx, &mut gy, &mut gz, &mut lap,
        )
    );
    // The returned VALUE column joins the carried geminal: give it the fused
    // `ao_eval` convention of the seed. Derivatives stay fp64.
    if orb.precision_policy.is_f32("ao_eval") {
        round_to_ao_eval(&mut value);
    }
    let c = orb
        .mo_coefficients
        .as_ref()
        .expect("fast kinetic derivative cache requires MO coefficients");
    let nmo = c.nrows();
    let cs = c.as_slice().expect("C contiguous");
    let mut px = vec![0.0; nmo];
    let mut py = vec![0.0; nmo];
    let mut pz = vec![0.0; nmo];
    let mut pl = vec![0.0; nmo];
    let mut pv = vec![0.0; nmo];
    crate::timed!(
        "gfmc.adv_det_vgl_mo",
        project5(
            cs, &gx, &gy, &gz, &lap, &value, na, nmo, &mut px, &mut py, &mut pz, &mut pl, &mut pv,
        )
    );
    let mut projected = Array2::<f64>::zeros((nmo, 4));
    for m in 0..nmo {
        projected[[m, 0]] = px[m];
        projected[[m, 1]] = py[m];
        projected[[m, 2]] = pz[m];
        projected[[m, 3]] = pl[m];
    }
    let assign = |dst: &mut Array2<f64>, component: usize| {
        for a in 0..dst.nrows() {
            dst[[a, electron]] = projected[[a, component]];
        }
    };
    if is_up {
        assign(&mut cache.up_x, 0);
        assign(&mut cache.up_y, 1);
        assign(&mut cache.up_z, 2);
        assign(&mut cache.up_lap, 3);
    } else {
        assign(&mut cache.dn_x, 0);
        assign(&mut cache.dn_y, 1);
        assign(&mut cache.dn_z, 2);
        assign(&mut cache.dn_lap, 3);
    }
    pv
}

/// Basis-generic sibling of [`init_det_grad_lap_cache`] for the
/// non-molecular GFMC fast routes (periodic pGTO / B-spline / pGTO+PW):
/// per-electron single-point value+grad+lap via
/// [`composite_single_value_grad_lap_into`] (image-FOLDED, buffers sized at
/// the HOME `orb.ao_count()` rank -- the latent-B2 fix: the molecular twin
/// sizes on `num_ao_kernel` and its single-point kernel asserts
/// `n_image == 1`), then optional MO projection through the SAME `project5`
/// kernel the per-hop update uses, so seed and refresh columns share one fp
/// chain. Raw-AO orbitals keep AO-rank derivative tables (the contraction
/// side is orbital-rank generic).
///
/// Coordinate contract: `r_up` and `r_dn` must be pre-wrapped into
/// the home cell by the caller under PBC (`MeshFastState` wraps with the same
/// `wrap_rows` its value chain was seeded with), mirroring how the reference
/// `compute_grads_and_laplacian_ln_det` wraps before its AO sweep. The
/// molecular chain keeps [`init_det_grad_lap_cache`] untouched
/// (bit-identical).
///
/// AO convention: the SCREENED fast-cut family (LEPSBAS image screen on the
/// periodic value, `derivative_alive` zeroing on the derivatives), pairing
/// with the screened candidate VALUE batches -- fast-mode ~1e-9 class under
/// the fast-chain tolerance contract (see
/// [`composite_single_value_grad_lap_into`] for the one-convention rationale).
pub fn init_det_grad_lap_cache_folded(
    data: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DetGradLapCache {
    let build = |orb: &OrbData, r: &ArrayView2<f64>| {
        let n_tot = orb.ao_count();
        let ne = r.nrows();
        let n_out = orb.mo_coefficients.as_ref().map_or(n_tot, |c| c.nrows());
        let mut ox = Array2::<f64>::zeros((n_out, ne));
        let mut oy = Array2::<f64>::zeros((n_out, ne));
        let mut oz = Array2::<f64>::zeros((n_out, ne));
        let mut ol = Array2::<f64>::zeros((n_out, ne));
        let mut scratch: Vec<f64> = Vec::new();
        let mut value = vec![0.0_f64; n_tot];
        let mut gx = vec![0.0_f64; n_tot];
        let mut gy = vec![0.0_f64; n_tot];
        let mut gz = vec![0.0_f64; n_tot];
        let mut lap = vec![0.0_f64; n_tot];
        let mut px = vec![0.0_f64; n_out];
        let mut py = vec![0.0_f64; n_out];
        let mut pz = vec![0.0_f64; n_out];
        let mut pl = vec![0.0_f64; n_out];
        let mut pv = vec![0.0_f64; n_out];
        for e in 0..ne {
            let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
            composite_single_value_grad_lap_into(
                orb,
                pos,
                &mut scratch,
                &mut value,
                &mut gx,
                &mut gy,
                &mut gz,
                &mut lap,
            );
            match &orb.mo_coefficients {
                Some(c) => {
                    // Release-mode shape contract: MO columns must span the
                    // concatenated AO stack.
                    assert_eq!(
                        c.ncols(),
                        n_tot,
                        "MO coefficient columns must match the composite AO stack"
                    );
                    let cs = c.as_slice().expect("C contiguous");
                    project5(
                        cs, &gx, &gy, &gz, &lap, &value, n_tot, n_out, &mut px, &mut py, &mut pz,
                        &mut pl, &mut pv,
                    );
                    for a in 0..n_out {
                        ox[[a, e]] = px[a];
                        oy[[a, e]] = py[a];
                        oz[[a, e]] = pz[a];
                        ol[[a, e]] = pl[a];
                    }
                }
                None => {
                    for a in 0..n_tot {
                        ox[[a, e]] = gx[a];
                        oy[[a, e]] = gy[a];
                        oz[[a, e]] = gz[a];
                        ol[[a, e]] = lap[a];
                    }
                }
            }
        }
        (ox, oy, oz, ol)
    };
    let (oux, ouy, ouz, oul) = build(&data.orb_up, r_up);
    let (odx, ody, odz, odl) = build(&data.orb_dn, r_dn);
    DetGradLapCache {
        up_x: oux,
        up_y: ouy,
        up_z: ouz,
        up_lap: oul,
        dn_x: odx,
        dn_y: ody,
        dn_z: odz,
        dn_lap: odl,
    }
}

/// Basis-generic sibling of [`update_det_grad_lap_cache_and_value`]:
/// replace the moved electron's derivative column and return its orbital VALUE
/// column (MO-projected when the orbital is MO-wrapped, raw-AO otherwise) for
/// the Sherman-Morrison advance. Buffers are sized at the home
/// `orb.ao_count()` rank; the single-point kernel and the MO
/// projection (`project5`) are EXACTLY the ones
/// [`init_det_grad_lap_cache_folded`] used per column, so seed and refresh
/// stay on one floating-point chain. `pos` must be pre-wrapped into the home
/// cell by the caller under PBC; the GFMC hop advance already wraps it.
pub fn update_det_grad_lap_cache_and_value_folded(
    data: &GeminalData,
    cache: &mut DetGradLapCache,
    is_up: bool,
    electron: usize,
    pos: [f64; 3],
) -> Vec<f64> {
    let orb = if is_up { &data.orb_up } else { &data.orb_dn };
    let n_tot = orb.ao_count();
    let mut scratch: Vec<f64> = Vec::new();
    let mut value = vec![0.0_f64; n_tot];
    let mut gx = vec![0.0_f64; n_tot];
    let mut gy = vec![0.0_f64; n_tot];
    let mut gz = vec![0.0_f64; n_tot];
    let mut lap = vec![0.0_f64; n_tot];
    composite_single_value_grad_lap_into(
        orb,
        pos,
        &mut scratch,
        &mut value,
        &mut gx,
        &mut gy,
        &mut gz,
        &mut lap,
    );
    // Fused `ao_eval` convention for the carried value column.
    if orb.precision_policy.is_f32("ao_eval") {
        round_to_ao_eval(&mut value);
    }
    let (col_x, col_y, col_z, col_l, col_v) = match &orb.mo_coefficients {
        Some(c) => {
            assert_eq!(
                c.ncols(),
                n_tot,
                "MO coefficient columns must match the composite AO stack"
            );
            let n_out = c.nrows();
            let cs = c.as_slice().expect("C contiguous");
            let mut px = vec![0.0_f64; n_out];
            let mut py = vec![0.0_f64; n_out];
            let mut pz = vec![0.0_f64; n_out];
            let mut pl = vec![0.0_f64; n_out];
            let mut pv = vec![0.0_f64; n_out];
            project5(
                cs, &gx, &gy, &gz, &lap, &value, n_tot, n_out, &mut px, &mut py, &mut pz, &mut pl,
                &mut pv,
            );
            (px, py, pz, pl, pv)
        }
        None => (gx, gy, gz, lap, value),
    };
    let assign = |dst: &mut Array2<f64>, src: &[f64]| {
        debug_assert_eq!(dst.nrows(), src.len());
        for a in 0..dst.nrows() {
            dst[[a, electron]] = src[a];
        }
    };
    if is_up {
        assign(&mut cache.up_x, &col_x);
        assign(&mut cache.up_y, &col_y);
        assign(&mut cache.up_z, &col_z);
        assign(&mut cache.up_lap, &col_l);
    } else {
        assign(&mut cache.dn_x, &col_x);
        assign(&mut cache.dn_y, &col_y);
        assign(&mut cache.dn_z, &col_z);
        assign(&mut cache.dn_lap, &col_l);
    }
    col_v
}

/// Candidate AO VALUE rows for ANY real orbital basis, written row-major into
/// `out` (`(npoint, orb.ao_count())`) -- the composite/non-molecular
/// counterpart of GFMC's `molecular_aos_candidate_batch`. Children are concatenated along the
/// AO axis in the composite's `gto_first` child order, exactly like
/// [`OrbData::eval_aos`] / `compute_single` (`gto_ao_offset` gives the GTO
/// column offset). Blocks:
/// - GTO/pGTO: the image-folded FAST candidate kernel
///   [`crate::atomic_orbital::compute_aos_single_value_fast_cut_folded_into`]
///   (LEPSBAS image screen when `n_image > 1`; identity fold and NO screen for
///   a molecular child, bit-matching the molecular candidate convention).
/// - PW: [`crate::atomic_orbital::compute_pw_single_into`] (per-move libm-cos
///   convention; no internal wrap).
/// - B-spline: [`crate::bspline::AosBsplineData::value_single`] (sweep
///   bit-chain convention; wraps in fractional space internally).
///   Contract: under PBC (periodic pGTO child, `n_image > 1`) the caller passes
///   candidate positions ALREADY wrapped into the home cell -- the pGTO image
///   tiling is enumerated relative to the home cell and this builder does not
///   wrap (one wrap site per hop, owned by the caller).
// Used by the GFMC fixed-mesh candidate and derivative routes.
pub(crate) fn composite_candidate_batch_into(
    orb: &OrbData,
    positions: &[[f64; 3]],
    out: &mut [f64],
) {
    let n_tot = orb.ao_count();
    debug_assert_eq!(out.len(), positions.len() * n_tot);
    let num_gto = orb.aos.num_ao;
    let gto_off = orb.gto_ao_offset();
    let extra_off = if orb.gto_first { num_gto } else { 0 };
    let mut kernel_scratch: Vec<f64> = Vec::new();
    for (v, &pos) in positions.iter().enumerate() {
        let row = &mut out[v * n_tot..(v + 1) * n_tot];
        if num_gto > 0 {
            crate::atomic_orbital::compute_aos_single_value_fast_cut_folded_into(
                &orb.aos,
                pos,
                &mut kernel_scratch,
                &mut row[gto_off..gto_off + num_gto],
            );
        }
        match (&orb.pw, &orb.bspline) {
            (Some(pw), _) => crate::atomic_orbital::compute_pw_single_into(
                pw,
                pos,
                &mut row[extra_off..extra_off + pw.num_ao],
            ),
            (None, Some(bs)) => bs.value_single(pos, &mut row[extra_off..extra_off + bs.num_ao]),
            (None, None) => {}
        }
    }
}

/// Composite AO-layer value + grad(x,y,z) + laplacian at ONE position into
/// caller buffers (each `orb.ao_count()`), concatenated in the same
/// `gto_first` child order as [`composite_candidate_batch_into`] /
/// `orb_value_grad_lap` -- the AO half of a non-molecular
/// `update_det_grad_lap_cache_and_value` (MO projection stays a separate step,
/// e.g. [`project_candidates_to_mos`] / `project5`).
///
/// AO-convention choice (the converse of the [`orb_value_grad_lap`] choice):
/// the GTO block uses the SCREENED fast-cut kernels
/// ([`crate::atomic_orbital::compute_aos_single_value_grad_lap_fast_cut_folded_into`]:
/// LEPSBAS image screen on the value when periodic, `derivative_alive` zeroing
/// on the derivatives always), because this kernel pairs with the SCREENED
/// candidate value batch above -- value and grad/lap feeding one fast-chain
/// table must share one AO convention. [`orb_value_grad_lap`] deliberately
/// pairs UN-screened grad/lap with the un-screened exact value on the
/// observable dT/dC path for the same one-convention reason; mixing this
/// kernel with an exact-convention value table would re-introduce exactly the
/// one-sided-term defect documented there.
///
/// Contract: under PBC pass a pre-wrapped position (see
/// [`composite_candidate_batch_into`]). `scratch` is reused by the GTO phase
/// (`5 * num_ao_kernel`) and then by the B-spline phase (`5 * num_ao`);
/// allocation-free in steady state.
// Used by the GFMC fixed-mesh candidate and derivative routes.
#[allow(
    clippy::too_many_arguments,
    reason = "caller-owned value/grad/lap output buffers"
)]
pub(crate) fn composite_single_value_grad_lap_into(
    orb: &OrbData,
    pos: [f64; 3],
    scratch: &mut Vec<f64>,
    value: &mut [f64],
    gx: &mut [f64],
    gy: &mut [f64],
    gz: &mut [f64],
    lap: &mut [f64],
) {
    let n_tot = orb.ao_count();
    debug_assert_eq!(value.len(), n_tot);
    debug_assert_eq!(gx.len(), n_tot);
    debug_assert_eq!(gy.len(), n_tot);
    debug_assert_eq!(gz.len(), n_tot);
    debug_assert_eq!(lap.len(), n_tot);
    let num_gto = orb.aos.num_ao;
    let gto_off = orb.gto_ao_offset();
    let extra_off = if orb.gto_first { num_gto } else { 0 };
    if num_gto > 0 {
        let (a, b) = (gto_off, gto_off + num_gto);
        crate::atomic_orbital::compute_aos_single_value_grad_lap_fast_cut_folded_into(
            &orb.aos,
            pos,
            scratch,
            &mut value[a..b],
            &mut gx[a..b],
            &mut gy[a..b],
            &mut gz[a..b],
            &mut lap[a..b],
        );
    }
    match (&orb.pw, &orb.bspline) {
        (Some(pw), _) => {
            let (a, b) = (extra_off, extra_off + pw.num_ao);
            crate::atomic_orbital::compute_pw_single_value_grad_lap_into(
                pw,
                pos,
                &mut value[a..b],
                &mut gx[a..b],
                &mut gy[a..b],
                &mut gz[a..b],
                &mut lap[a..b],
            );
        }
        (None, Some(bs)) => {
            let (a, b) = (extra_off, extra_off + bs.num_ao);
            bs.value_grad_lap_single_into(
                pos,
                scratch,
                &mut value[a..b],
                &mut gx[a..b],
                &mut gy[a..b],
                &mut gz[a..b],
                &mut lap[a..b],
            );
        }
        (None, None) => {}
    }
}

/// Project candidate AO rows into MO space: `(npoint, num_ao_total)` rows
/// contracted with `mo_coefficients` `(num_mo, num_ao_total)` into
/// `(npoint, num_mo)` -- the generic projection half of GFMC's
/// `molecular_mos_candidate_batch`. Plain sequential per-(point, MO) dot
/// (deterministic order, same as gfmc's scalar `dot_product_fast` fallback);
/// gfmc's AVX2 six-row `candidate_wdot` kernel is where the vector variant
/// slots in.
// Used by the GFMC fixed-mesh candidate and derivative routes.
pub(crate) fn project_candidates_to_mos(
    aos: &ArrayView2<f64>,
    coeff: &ArrayView2<f64>,
) -> Array2<f64> {
    debug_assert_eq!(aos.ncols(), coeff.ncols());
    let mut out = Array2::<f64>::zeros((aos.nrows(), coeff.nrows()));
    for s in 0..aos.nrows() {
        let ao = aos.row(s);
        let ao = ao.as_slice().expect("candidate AO row contiguous");
        for m in 0..coeff.nrows() {
            let c = coeff.row(m);
            let c = c.as_slice().expect("MO coefficient row contiguous");
            let mut acc = 0.0_f64;
            for j in 0..ao.len() {
                acc += ao[j] * c[j];
            }
            out[[s, m]] = acc;
        }
    }
    out
}

pub fn contract_det_grad_lap_cache(
    data: &GeminalData,
    g_inv: &ArrayView2<f64>,
    st: &DetRatioState,
    cache: &DetGradLapCache,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_up = st.ao_up.ncols();
    let n_dn = st.ao_dn.ncols();
    let orb_num_up = st.ao_up.nrows();
    let n_unpaired = data.n_unpaired();

    // M = [ lambda_paired*O_dn | lambda_unpaired ] with the paired product from `st`.
    let paired_dn = &st.paired_dn;
    let mut m = Array2::<f64>::zeros((orb_num_up, n_up));
    for a in 0..orb_num_up {
        for j in 0..n_dn {
            m[[a, j]] = if a < paired_dn.nrows() {
                paired_dn[[a, j]]
            } else {
                0.0
            };
        }
        for j in 0..n_unpaired {
            m[[a, n_dn + j]] = data.lambda_value(a, data.orb_num_dn + j);
        }
    }
    let mg = crate::timed!("kdet.lam_gemm", matmul(&m.view(), &g_inv.view()));
    let td = crate::timed!(
        "kdet.lam_gemm",
        matmul(&g_inv.view(), &st.paired_up_lambda.view())
    );
    let mut grad_up = Array2::<f64>::zeros((n_up, 3));
    let mut lap_up = Array1::<f64>::zeros(n_up);
    for i in 0..n_up {
        let (mut gxs, mut gys, mut gzs, mut tr) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..orb_num_up {
            let mgi = mg[[a, i]];
            gxs += cache.up_x[[a, i]] * mgi;
            gys += cache.up_y[[a, i]] * mgi;
            gzs += cache.up_z[[a, i]] * mgi;
            tr += cache.up_lap[[a, i]] * mgi;
        }
        grad_up[[i, 0]] = gxs;
        grad_up[[i, 1]] = gys;
        grad_up[[i, 2]] = gzs;
        lap_up[i] = -(gxs * gxs + gys * gys + gzs * gzs) + tr;
    }
    let mut grad_dn = Array2::<f64>::zeros((n_dn, 3));
    let mut lap_dn = Array1::<f64>::zeros(n_dn);
    for i in 0..n_dn {
        let (mut gxs, mut gys, mut gzs, mut tr) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..cache.dn_x.nrows() {
            let tdi = td[[i, a]];
            gxs += cache.dn_x[[a, i]] * tdi;
            gys += cache.dn_y[[a, i]] * tdi;
            gzs += cache.dn_z[[a, i]] * tdi;
            tr += cache.dn_lap[[a, i]] * tdi;
        }
        grad_dn[[i, 0]] = gxs;
        grad_dn[[i, 1]] = gys;
        grad_dn[[i, 2]] = gzs;
        lap_dn[i] = -(gxs * gxs + gys * gys + gzs * gzs) + tr;
    }
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Contract the derivative table with determinant-ratio rows already built by
/// the fixed-mesh candidate kernel.  `up_rows[i]` is column `i` of
/// `[lambda O_dn | lambda_unpaired] G^-1`; `dn_rows[i]` is row `i` of
/// `G^-1 O_up^T lambda`.  These are exactly the two weights needed here, so the
/// local-energy path need not repeat either GEMM on every LRDMC hop.
pub fn contract_det_grad_lap_cache_with_wdot(
    cache: &DetGradLapCache,
    up_rows: &ArrayView2<f64>,
    dn_rows: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_up = cache.up_x.ncols();
    let n_dn = cache.dn_x.ncols();
    debug_assert_eq!(up_rows.nrows(), n_up);
    debug_assert_eq!(up_rows.ncols(), cache.up_x.nrows());
    debug_assert!(dn_rows.nrows() >= n_dn);
    debug_assert_eq!(dn_rows.ncols(), cache.dn_x.nrows());

    let mut grad_up = Array2::<f64>::zeros((n_up, 3));
    let mut lap_up = Array1::<f64>::zeros(n_up);
    for i in 0..n_up {
        let (mut gx, mut gy, mut gz, mut tr) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..cache.up_x.nrows() {
            let w = up_rows[[i, a]];
            gx += cache.up_x[[a, i]] * w;
            gy += cache.up_y[[a, i]] * w;
            gz += cache.up_z[[a, i]] * w;
            tr += cache.up_lap[[a, i]] * w;
        }
        grad_up[[i, 0]] = gx;
        grad_up[[i, 1]] = gy;
        grad_up[[i, 2]] = gz;
        lap_up[i] = -(gx * gx + gy * gy + gz * gz) + tr;
    }

    let mut grad_dn = Array2::<f64>::zeros((n_dn, 3));
    let mut lap_dn = Array1::<f64>::zeros(n_dn);
    for i in 0..n_dn {
        let (mut gx, mut gy, mut gz, mut tr) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..cache.dn_x.nrows() {
            let w = dn_rows[[i, a]];
            gx += cache.dn_x[[a, i]] * w;
            gy += cache.dn_y[[a, i]] * w;
            gz += cache.dn_z[[a, i]] * w;
            tr += cache.dn_lap[[a, i]] * w;
        }
        grad_dn[[i, 0]] = gx;
        grad_dn[[i, 1]] = gy;
        grad_dn[[i, 2]] = gz;
        lap_dn[i] = -(gx * gx + gy * gy + gz * gz) + tr;
    }
    (grad_up, grad_dn, lap_up, lap_dn)
}

pub fn compute_grads_and_laplacian_ln_det_fast(
    data: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: &ArrayView2<f64>,
    st: &DetRatioState,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    // Periodic leaves: the sweep evaluated the carried value columns `st` at the
    // home-cell-wrapped coordinate (the image fold is home-cell centered), so the
    // derivative tables must use the same wrapped coordinate. Molecular: no copy,
    // raw coordinates exactly as before.
    if data.orb_up.aos.n_image > 1 || data.orb_dn.aos.n_image > 1 {
        let ru = crate::atomic_orbital::wrap_r(&data.orb_up.aos, r_up);
        let rd = crate::atomic_orbital::wrap_r(&data.orb_dn.aos, r_dn);
        let cache = init_det_grad_lap_cache(data, &ru.view(), &rd.view());
        return contract_det_grad_lap_cache(data, g_inv, st, &cache);
    }
    let cache = init_det_grad_lap_cache(data, r_up, r_dn);
    contract_det_grad_lap_cache(data, g_inv, st, &cache)
}

/// Attaccalite-Sorella exponent: `R_AS = (S*F)^(-AS_THETA)`.
const AS_THETA: f64 = 3.0 / 8.0;

/// Issue #131: smallest `S` for which `R_AS` is evaluated as the linear
/// `(S*F)^(-3/8)` (float64 value of `jqmc._setting.as_linear_min_sq_norm`,
/// `tiny / eps`). Below it (or with a non-finite `F` / `S*F`) the factor comes
/// from max-abs-scaled `ln S + ln F` ([`as_factor_scaled`]). Not a user knob.
pub use crate::setting_consts::AS_LINEAR_MIN_SQ_NORM_FP64 as AS_LINEAR_MIN_SQ_NORM;

/// Issue #134: an incrementally updated squared norm of the proposed state below
/// this fraction of the cached one has lost at least half of its digits and is
/// recomputed directly (float64 value `2**-26` of
/// `jqmc._setting.as_incremental_norm_cancellation_threshold`). Not a user knob.
pub use crate::setting_consts::AS_NORM_CANCELLATION_THRESHOLD_FP64 as AS_NORM_CANCELLATION_THRESHOLD;

#[inline]
fn as_linear_ok(s_fac: f64, f: f64, sf: f64) -> bool {
    s_fac >= AS_LINEAR_MIN_SQ_NORM && f.is_finite() && sf.is_finite() && sf > 0.0
}

/// `ln sum_t |x_t|^2` over `len` magnitudes with max-abs scaling (no underflow /
/// overflow). An all-zero slice gives `-inf`; `inf` or `NaN` gives `NaN`.
pub fn ln_sum_sq_scaled(len: usize, abs_at: impl Fn(usize) -> f64) -> f64 {
    let mut m = 0.0_f64;
    for t in 0..len {
        let a = abs_at(t);
        if a.is_nan() {
            return f64::NAN;
        }
        if a > m {
            m = a;
        }
    }
    if m == 0.0 {
        return f64::NEG_INFINITY;
    }
    if !m.is_finite() {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for t in 0..len {
        let r = abs_at(t) / m;
        s += r * r;
    }
    2.0 * m.ln() + s.ln()
}

/// Issue #131: `R_AS = exp(-3/8 (ln S + ln F))` from max-abs-scaled logarithms,
/// for any entry accessors: `g_abs(i, j)` the magnitudes of the `nr x nc`
/// geminal, `gi_abs(t)` those of the `ni` inverse entries. Invariant under the
/// scale of a row (a far electron), so a tiny row neither underflows `S` nor
/// overflows `F`. Returns 0 only for a degenerate state (an exactly zero row or
/// column, or a non-finite entry).
pub fn as_factor_scaled(
    nr: usize,
    nc: usize,
    g_abs: impl Fn(usize, usize) -> f64,
    ni: usize,
    gi_abs: impl Fn(usize) -> f64,
) -> f64 {
    let mut ln_s = f64::INFINITY;
    let mut bad = false;
    for i in 0..nr {
        let l = ln_sum_sq_scaled(nc, |j| g_abs(i, j));
        bad |= l.is_nan();
        if l < ln_s {
            ln_s = l;
        }
    }
    for j in 0..nc {
        let l = ln_sum_sq_scaled(nr, |i| g_abs(i, j));
        bad |= l.is_nan();
        if l < ln_s {
            ln_s = l;
        }
    }
    let ln_f = ln_sum_sq_scaled(ni, gi_abs);
    if !bad && ln_s.is_finite() && ln_f.is_finite() {
        (-AS_THETA * (ln_s + ln_f)).exp()
    } else {
        0.0
    }
}

/// Issues #131/#134: AS guiding factor `(max(R_p, eps)/R_p) / (max(R_o, eps)/R_o)`
/// of one proposal, mirror of `jqmc_mcmc._as_guiding_ratio`. Exactly 1 for
/// `epsilon_as == 0`; a degenerate proposal (`R_p == 0`, guiding amplitude 0)
/// gives 0 (rejected) instead of `eps / 0 = inf`; a degenerate current state is
/// left by any proposal (`inf`). Bit-identical to the plain expression above for
/// positive factors.
#[inline]
pub fn as_guiding_ratio(r_as_p: f64, r_as_o: f64, epsilon_as: f64) -> f64 {
    if epsilon_as <= 0.0 || epsilon_as.is_nan() {
        return 1.0;
    }
    if r_as_p <= 0.0 || r_as_p.is_nan() {
        return 0.0;
    }
    if r_as_o <= 0.0 || r_as_o.is_nan() {
        return f64::INFINITY;
    }
    (r_as_p.max(epsilon_as) / r_as_p) / (r_as_o.max(epsilon_as) / r_as_o)
}

/// Issue #131: walker weight `(R_AS / max(R_AS, eps))^2`, mirror of
/// `jqmc_mcmc._as_walker_weight`: exactly 1 for `epsilon_as == 0` (the plain
/// expression would be `0 / 0 = NaN` at `R_AS == 0`).
#[inline]
pub fn as_walker_weight(r_as: f64, epsilon_as: f64) -> f64 {
    if epsilon_as <= 0.0 || epsilon_as.is_nan() {
        return 1.0;
    }
    let ratio = r_as / r_as.max(epsilon_as);
    ratio * ratio
}

/// Attaccalite-Sorella regularization factor `R_AS = (S*F)^(-3/8)`
/// (real path), 1:1 with `compute_AS_regularization_factor_fast_update`.
/// `F = ||G^-^1||^2_F`; `S = min(min_i Sum_j G_ij^2, min_j Sum_i G_ij^2)`.
/// Issue #131: outside the linear range ([`AS_LINEAR_MIN_SQ_NORM`]) the factor
/// comes from [`as_factor_scaled`].
pub fn compute_as_regularization_factor(
    geminal: &ArrayView2<f64>,
    geminal_inv: &ArrayView2<f64>,
) -> f64 {
    let theta = AS_THETA;
    let n = geminal.nrows();
    let m = geminal.ncols();
    // Raw-slice access (G, G^-^1 are contiguous NxN) -- avoids ndarray 2D-index
    // bounds checks in the per-move AS factor (~50 mus with them).
    let g = geminal.as_slice().expect("G contiguous");
    let gi = geminal_inv.as_slice().expect("G_inv contiguous");

    // F = ||G^-^1||^2_F (one sequential pass).
    let mut f = 0.0_f64;
    for &x in gi {
        f += x * x;
    }

    // row norms^2 and col norms^2 of G (single pass each, sequential).
    let mut min_row = f64::INFINITY;
    for i in 0..n {
        let row = &g[i * m..i * m + m];
        let mut s = 0.0;
        for &x in row {
            s += x * x;
        }
        if s < min_row {
            min_row = s;
        }
    }
    let mut col = vec![0.0_f64; m];
    for i in 0..n {
        let row = &g[i * m..i * m + m];
        for j in 0..m {
            col[j] += row[j] * row[j];
        }
    }
    let min_col = col.iter().cloned().fold(f64::INFINITY, f64::min);
    let s_fac = min_row.min(min_col);

    let sf = s_fac * f;
    if as_linear_ok(s_fac, f, sf) {
        sf.powf(-theta)
    } else {
        as_factor_scaled(n, m, |i, j| g[i * m + j].abs(), gi.len(), |t| gi[t].abs())
    }
}

/// Full AS pass that also fills the row/col squared-norm caches and returns
/// `R_AS`. Used at init and after an accepted move (~=1/3 of moves), so the
/// O(N^2) recompute is amortized. Issue #131: robust to tiny rows (see
/// [`compute_as_regularization_factor`]).
pub fn as_recompute(
    geminal: &ArrayView2<f64>,
    geminal_inv: &ArrayView2<f64>,
    row_n2: &mut [f64],
    col_n2: &mut [f64],
) -> f64 {
    let n = geminal.nrows();
    let m = geminal.ncols();
    let g = geminal.as_slice().expect("G contiguous");
    let gi = geminal_inv.as_slice().expect("G_inv contiguous");
    let mut f = 0.0_f64;
    for &x in gi {
        f += x * x;
    }
    for c in col_n2.iter_mut() {
        *c = 0.0;
    }
    let mut min_row = f64::INFINITY;
    for i in 0..n {
        let row = &g[i * m..i * m + m];
        let mut s = 0.0;
        for j in 0..m {
            let x = row[j];
            s += x * x;
            col_n2[j] += x * x;
        }
        row_n2[i] = s;
        if s < min_row {
            min_row = s;
        }
    }
    let min_col = col_n2.iter().cloned().fold(f64::INFINITY, f64::min);
    let s_fac = min_row.min(min_col);
    let sf = s_fac * f;
    if as_linear_ok(s_fac, f, sf) {
        sf.powf(-AS_THETA)
    } else {
        as_factor_scaled(n, m, |i, j| g[i * m + j].abs(), gi.len(), |t| gi[t].abs())
    }
}

/// AS factor of the PROPOSED state from the cached old state + rank-1 factors,
/// with no materialization of `G_new`/`G^-^1_new`.
///
/// Issues #131/#134:
/// * The moved row (up) / column (dn) is `G_k + v` unless that lost digits
///   ([`AS_NORM_CANCELLATION_THRESHOLD`]); then it is rebuilt directly by
///   `fill_direct_line` (the new orbital values contracted with the cached
///   state), and so is every cross column (up) / row (dn) norm whose
///   incremental update `n2 - G_kj^2 + G'_kj^2` cancelled (O(N) each).
/// * `F_new = ||G^-1_new||_F^2` is summed directly over the rank-1 entries,
///   with the moved inverse column (up) / row (dn) in its exact form
///   `X e_k / R` (`e_kT X / R`) -- the expansion
///   `F_old - 2<G^-1,B> + ||B||^2` cancels for `F_new << F_old` and is
///   `inf - inf` for a tiny old row.
/// * Outside the linear range the factor comes from [`as_factor_scaled`] over
///   the same proposed entries.
#[allow(clippy::too_many_arguments)]
pub fn as_proposed(
    geminal: &ArrayView2<f64>,
    geminal_inv: &ArrayView2<f64>,
    row_n2: &[f64],
    col_n2: &[f64],
    u: &[f64],
    v: &[f64],
    ainv_u: &[f64],
    vt_ainv: &[f64],
    ratio: f64,
    moved_spin_is_up: bool,
    idx: usize,
    line: &mut [f64],
    fill_direct_line: &mut dyn FnMut(&mut [f64]),
) -> f64 {
    let n = u.len();
    let gi = geminal_inv.as_slice().expect("G_inv contiguous");
    let g = geminal.as_slice().expect("G contiguous");
    let thr = AS_NORM_CANCELLATION_THRESHOLD;
    let up = moved_spin_is_up;
    // Old moved line G[idx, :] (up) or G[:, idx] (dn); its delta v or u.
    let old_at = |t: usize| if up { g[idx * n + t] } else { g[t * n + idx] };
    let delta = if up { v } else { u };
    for t in 0..n {
        line[t] = old_at(t) + delta[t];
    }
    let sum_sq = |x: &[f64]| x.iter().map(|a| a * a).sum::<f64>();
    let mut direct = false;
    let mut n_line = sum_sq(line);
    let old_line_n2 = if up { row_n2[idx] } else { col_n2[idx] };
    if n_line < thr * old_line_n2 {
        fill_direct_line(line);
        direct = true;
        n_line = sum_sq(line);
    }
    // Unchanged lines of the moved kind.
    let same = if up { row_n2 } else { col_n2 };
    let mut min_same = n_line;
    for (t, same_t) in same[..n].iter().enumerate() {
        if t != idx && *same_t < min_same {
            min_same = *same_t;
        }
    }
    // Cross lines (columns for an up move, rows for a dn move), each changed in
    // one entry: n2 - G_kt^2 + G'_kt^2 (not n2 + 2 G_kt d_t + d_t^2).
    let cross = if up { col_n2 } else { row_n2 };
    let mut min_cross = f64::INFINITY;
    for t in 0..n {
        let gk = old_at(t);
        let mut c = cross[t] + 2.0 * gk * delta[t] + delta[t] * delta[t];
        if c < thr * cross[t] {
            if !direct {
                fill_direct_line(line);
                direct = true;
                n_line = sum_sq(line);
                if n_line < min_same {
                    min_same = n_line;
                }
            }
            let mut acc = 0.0_f64;
            for o in 0..n {
                if o != idx {
                    let x = if up { g[o * n + t] } else { g[t * n + o] };
                    acc += x * x;
                }
            }
            c = acc + line[t] * line[t];
        }
        if c < min_cross {
            min_cross = c;
        }
    }
    let s_new = min_same.min(min_cross);
    // Proposed inverse entries: X - (X u)(vT X)/R, moved column / row exact.
    let inv_at = |i: usize, j: usize| -> f64 {
        if up && j == idx {
            ainv_u[i] / ratio
        } else if !up && i == idx {
            vt_ainv[j] / ratio
        } else {
            gi[i * n + j] - (ainv_u[i] / ratio) * vt_ainv[j]
        }
    };
    let mut f_new = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let x = inv_at(i, j);
            f_new += x * x;
        }
    }
    let sf = s_new * f_new;
    if as_linear_ok(s_new, f_new, sf) {
        return sf.powf(-AS_THETA);
    }
    if !direct {
        fill_direct_line(line);
    }
    let line_ref: &[f64] = line;
    as_factor_scaled(
        n,
        n,
        |i, j| {
            let moved = if up { i == idx } else { j == idx };
            if moved {
                (if up { line_ref[j] } else { line_ref[i] }).abs()
            } else {
                g[i * n + j].abs()
            }
        },
        n * n,
        |t| inv_at(t / n, t % n).abs(),
    )
}

/// O(N^2) probe residual ``max|G (X P) - P|`` of a carried inverse.
///
/// ``P`` holds the same two deterministic probe columns as the JAX driver
/// (``jqmc_mcmc._running_geminal_inv_probe_residual``): all ones and
/// ``cos(1 + 0.7 j)``. A non-finite inverse gives NaN, which never triggers.
pub fn running_inverse_probe_residual(g: &ArrayView2<f64>, gi: &ArrayView2<f64>) -> f64 {
    let n = gi.nrows();
    let mut worst = 0.0_f64;
    for probe in 0..2 {
        let p: Vec<f64> = (0..n)
            .map(|j| {
                if probe == 0 {
                    1.0
                } else {
                    (1.0 + 0.7 * j as f64).cos()
                }
            })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| gi[[i, j]] * p[j]).sum::<f64>())
            .collect();
        for i in 0..n {
            let r = (0..n).map(|j| g[[i, j]] * y[j]).sum::<f64>() - p[i];
            let a = r.abs();
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

/// Real geminal pseudo-inverse, the native mirror of
/// ``jqmc_mcmc._geminal_pinv_from_matrix`` (see
/// [`crate::complex_det::geminal_pinv_from_matrix_complex`]; with a real input
/// every Jacobi rotation and the result stay real).
///
/// Real arithmetic, column-contiguous storage. With a real input every
/// imaginary part of the complex routine stays a signed zero, and each real
/// part it forms is the real operation done here (`hypot(x, 0) == |x|`,
/// `gamma / |gamma| == +-1`, a real phase leaves `x c - y s` unchanged), in the
/// same order, so the result equals the real part of the complex routine
/// (`pinv_real_matches_complex_routine`); only the sign of an exact zero may
/// differ. The column layout makes the Jacobi rotations unit-stride.
pub fn geminal_pinv_from_matrix(g: &ArrayView2<f64>) -> Array2<f64> {
    use crate::complex_det::{unscale_pow2_f64, RCOND_SVD};
    let n = g.nrows();
    if g.iter().any(|x| !x.is_finite()) {
        return Array2::from_elem((n, n), f64::NAN);
    }
    // wt[p * n + i] = W[i, p] (column p contiguous).
    let (row_scale, col_scale, mut wt) =
        equilibrate_geminal_col_major(g, crate::complex_det::pinv_equilibration_scale);
    let mut vt = vec![0.0_f64; n * n];
    for i in 0..n {
        vt[i * n + i] = 1.0;
    }
    #[inline(always)]
    fn rotate(m: &mut [f64], n: usize, p: usize, q: usize, ph: f64, c: f64, s: f64) {
        let (lo, hi) = m.split_at_mut(q * n);
        let cp = &mut lo[p * n..(p + 1) * n];
        let cq = &mut hi[..n];
        for i in 0..n {
            let x = cp[i];
            let y = cq[i] * ph;
            cp[i] = x * c - y * s;
            cq[i] = x * s + y * c;
        }
    }
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0_f64, 0.0_f64, 0.0_f64);
                {
                    let cp = &wt[p * n..(p + 1) * n];
                    let cq = &wt[q * n..(q + 1) * n];
                    for i in 0..n {
                        alpha += cp[i] * cp[i];
                        beta += cq[i] * cq[i];
                        gamma += cp[i] * cq[i];
                    }
                }
                let gabs = gamma.abs();
                let orthogonal = gabs.partial_cmp(&(f64::EPSILON * alpha.sqrt() * beta.sqrt()))
                    != Some(std::cmp::Ordering::Greater);
                if alpha == 0.0 || beta == 0.0 || orthogonal {
                    continue;
                }
                rotated = true;
                let ph = gamma / gabs;
                let zeta = (beta - alpha) / (2.0 * gabs);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                rotate(&mut wt, n, p, q, ph, c, s);
                rotate(&mut vt, n, p, q, ph, c, s);
            }
        }
        if !rotated {
            break;
        }
    }
    let sv: Vec<f64> = (0..n)
        .map(|j| {
            wt[j * n..(j + 1) * n]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let s_max = sv.iter().cloned().fold(0.0, f64::max);
    let mut out = vec![0.0_f64; n * n];
    for j in 0..n {
        if sv[j].partial_cmp(&(RCOND_SVD * s_max)) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let inv_s2 = 1.0 / (sv[j] * sv[j]);
        let wj = &wt[j * n..(j + 1) * n];
        for i in 0..n {
            let vij = vt[j * n + i] * inv_s2;
            let row = &mut out[i * n..(i + 1) * n];
            for k in 0..n {
                row[k] += vij * wj[k];
            }
        }
    }
    Array2::from_shape_fn((n, n), |(i, k)| {
        unscale_pow2_f64(out[i * n + k], col_scale[i], row_scale[k])
    })
}

/// Row and column equilibration of a real geminal (see
/// ``complex_det::equilibrate_geminal_complex``): ``(r, c, A^T)`` with
/// ``A = diag(r) G diag(c)`` stored column-major (``at[j * n + i] = A[i, j]``).
fn equilibrate_geminal_col_major(
    g: &ArrayView2<f64>,
    scale: fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = g.nrows();
    let row_scale: Vec<f64> = (0..n)
        .map(|i| scale((0..n).map(|j| g[[i, j]].abs()).fold(0.0, f64::max)))
        .collect();
    let mut at = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            at[j * n + i] = g[[i, j]] * row_scale[i];
        }
    }
    let col_scale: Vec<f64> = (0..n)
        .map(|j| {
            scale(
                at[j * n..(j + 1) * n]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0, f64::max),
            )
        })
        .collect();
    for j in 0..n {
        for x in &mut at[j * n..(j + 1) * n] {
            *x *= col_scale[j];
        }
    }
    (row_scale, col_scale, at)
}

/// Issue #605: the seed / carried-inverse refresh inverse of one real geminal, the real
/// twin of [`crate::complex_det::geminal_inverse_guarded_complex`]:
/// ``C inv(A) R`` from Gauss-Jordan elimination of the equilibrated
/// ``A = R G C`` (full exponent range), kept when it is finite and ``max|A (inv(A) P) - P|`` is at
/// most ``GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD``; otherwise the equilibrated
/// pseudo-inverse [`geminal_pinv_from_matrix`].
pub fn geminal_inverse_guarded(g: &ArrayView2<f64>) -> Array2<f64> {
    use crate::complex_det::{unscale_pow2_f64, GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD};
    let n = g.nrows();
    if g.iter().all(|x| x.is_finite()) {
        let (row_scale, col_scale, at) =
            equilibrate_geminal_col_major(g, crate::complex_det::pow2_scale);
        let a = Array2::from_shape_fn((n, n), |(i, j)| at[j * n + i]);
        let y = inverse(&a.view());
        let residual = running_inverse_probe_residual(&a.view(), &y.view());
        if residual <= GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD {
            let x = Array2::from_shape_fn((n, n), |(i, k)| {
                unscale_pow2_f64(y[[i, k]], col_scale[i], row_scale[k])
            });
            if x.iter().all(|v| v.is_finite()) {
                return x;
            }
        }
    }
    geminal_pinv_from_matrix(g)
}

/// Replace a drifted carried inverse by the inverse of the SAME carried
/// geminal when its probe residual exceeds ``threshold`` (``inf`` disables).
/// Returns whether a refresh happened. O(N^2) unless triggered.
pub fn refresh_running_inverse_if_needed(
    g: &ArrayView2<f64>,
    gi: &mut Array2<f64>,
    threshold: f64,
) -> bool {
    if !threshold.is_finite() {
        return false;
    }
    if running_inverse_probe_residual(g, &gi.view()) > threshold {
        // Issues #127 / #605: the guarded equilibrated LU inverse, falling back
        // to the equilibrated SVD pseudo-inverse (``_geminal_pinv_from_matrix``)
        // for a singular, rank-deficient or out-of-range carried G, the same
        // rule as the Python carried-inverse refresh (finite instead of a Gauss-Jordan inf).
        *gi = geminal_inverse_guarded(g);
        true
    } else {
        false
    }
}

/// Probe-residual threshold of the carried Sherman-Morrison
/// inverse. The SAME internal constant as the Python drivers'
/// ``jqmc._setting.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`` (pinned equal by
/// ``tests/test_rs_kernel/test_lrdmc_mesh_chain_rust.py`` through
/// ``jqmc_rs.sm_inverse_refresh_residual_threshold``). Not a user knob.
pub use crate::setting_consts::SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD;

/// The probe residual ``max|G (X P) - P|`` of a carried inverse
/// ``X`` against the geminal ``G`` the streaming caches imply
/// ([`geminal_from_det_ratio_state`]), WITHOUT forming ``G``: ``G y`` is
/// applied in its factored form ``ao_up^T (paired_dn y_p + lambda_u y_u)``,
/// so the cost is O(N^2 + N M) per call (two probe columns, the same
/// deterministic ``P`` as [`running_inverse_probe_residual`]). A non-finite
/// inverse gives NaN, which never triggers a refresh.
pub fn det_state_inverse_probe_residual(
    data: &GeminalData,
    state: &DetRatioState,
    gi: &ArrayView2<f64>,
) -> f64 {
    let n = gi.nrows();
    let n_dn = state.paired_dn.ncols();
    let m_paired = state.paired_dn.nrows();
    let m_up = state.ao_up.nrows();
    let n_unpaired = data.n_unpaired();
    // Both probe columns in one pass over every operand, row-major contiguous
    // traversal: y = X P, z = paired_dn y_p (+ lambda_u y_u), G y = ao_up^T z.
    let p1: Vec<f64> = (0..n).map(|j| (1.0 + 0.7 * j as f64).cos()).collect();
    let gi = gi.as_standard_layout();
    let gis = gi.as_slice().expect("standard layout");
    let mut y0 = vec![0.0_f64; n];
    let mut y1 = vec![0.0_f64; n];
    for i in 0..n {
        let row = &gis[i * n..(i + 1) * n];
        let (mut s0, mut s1) = (0.0, 0.0);
        for (x, q) in row.iter().zip(p1.iter()) {
            s0 += x;
            s1 += x * q;
        }
        y0[i] = s0;
        y1[i] = s1;
    }
    let pdn = state.paired_dn.as_standard_layout();
    let pdns = pdn.as_slice().expect("standard layout");
    let mut z0 = vec![0.0_f64; m_up];
    let mut z1 = vec![0.0_f64; m_up];
    for a in 0..m_paired {
        let row = &pdns[a * n_dn..(a + 1) * n_dn];
        let (mut s0, mut s1) = (0.0, 0.0);
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
                let l = data.lambda_value(a, data.orb_num_dn + k);
                z0[a] += l * y0[n_dn + k];
                z1[a] += l * y1[n_dn + k];
            }
        }
    }
    let aup = state.ao_up.as_standard_layout();
    let aups = aup.as_slice().expect("standard layout");
    let mut g0 = vec![0.0_f64; n];
    let mut g1 = vec![0.0_f64; n];
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
        for r in [(g0[i] - 1.0).abs(), (g1[i] - p1[i]).abs()] {
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

/// The parameter-free in-interval screen. The carried inverse is kept
/// iff its probe residual does not exceed
/// [`SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`]; otherwise the caller rebuilds it
/// (and every table derived from it) from scratch. NaN keeps the carry, the
/// existing carried-inverse refresh convention.
#[inline]
pub fn interval_ginv_carry_ok(residual: f64) -> bool {
    residual.partial_cmp(&SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD)
        != Some(std::cmp::Ordering::Greater)
}

/// Hand-rolled matrix inverse via Gauss-Jordan with partial pivoting.
pub fn inverse(a: &ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut m = a.to_owned();
    let mut inv = Array2::<f64>::eye(n);
    for col in 0..n {
        // pivot
        let mut p = col;
        let mut maxv = m[[col, col]].abs();
        for i in (col + 1)..n {
            let v = m[[i, col]].abs();
            if v > maxv {
                maxv = v;
                p = i;
            }
        }
        if p != col {
            for j in 0..n {
                m.swap([col, j], [p, j]);
                inv.swap([col, j], [p, j]);
            }
        }
        let piv = m[[col, col]];
        for j in 0..n {
            m[[col, j]] /= piv;
            inv[[col, j]] /= piv;
        }
        for i in 0..n {
            if i == col {
                continue;
            }
            let f = m[[i, col]];
            if f == 0.0 {
                continue;
            }
            for j in 0..n {
                let mcj = m[[col, j]];
                let icj = inv[[col, j]];
                m[[i, j]] -= f * mcj;
                inv[[i, j]] -= f * icj;
            }
        }
    }
    inv
}

#[cfg(test)]
mod tests {

    #[test]
    fn pinv_real_matches_complex_routine() {
        // The real Jacobi pseudo-inverse equals the real part of the complex
        // routine on real input (values; only the sign of an exact zero may differ).
        use numpy::ndarray::Array2;
        let mut seed = 0x9e3779b97f4a7c15_u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for (case, n) in [(0, 1), (1, 7), (2, 24), (3, 24), (4, 31), (5, 16)] {
            let mut g = Array2::<f64>::from_shape_fn((n, n), |_| rnd());
            match case {
                3 => {
                    // graded rows / columns over many decades
                    for i in 0..n {
                        for j in 0..n {
                            g[[i, j]] *=
                                10f64.powi(i as i32 * 9 - 100) * 10f64.powi(-(j as i32) * 5);
                        }
                    }
                }
                4 => {
                    // rank deficient: last row = first row, one zero column
                    for j in 0..n {
                        g[[n - 1, j]] = g[[0, j]];
                        g[[j, 3]] = 0.0;
                    }
                }
                5 => g[[2, 2]] = 0.0,
                _ => {}
            }
            let got = geminal_pinv_from_matrix(&g.view());
            let gc = g.mapv(|x| numpy::Complex64::new(x, 0.0));
            let want = crate::complex_det::geminal_pinv_from_matrix_complex(&gc.view());
            for i in 0..n {
                for k in 0..n {
                    let (a, b) = (got[[i, k]], want[[i, k]]);
                    assert!(
                        a == b.re && b.im == 0.0,
                        "case {case} ({i},{k}): {a:e} vs {b}"
                    );
                }
            }
        }
        let mut bad = Array2::<f64>::zeros((3, 3));
        bad[[1, 1]] = f64::NAN;
        assert!(geminal_pinv_from_matrix(&bad.view())
            .iter()
            .all(|x| x.is_nan()));
    }
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn ln_det_2x2() {
        // [[2,0],[0,3]] det=6
        let g = array![[2.0, 0.0], [0.0, 3.0]];
        assert!(
            (ln_abs_det(&g.view()) - 6.0_f64.ln()).abs()
                < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE
        );
    }

    #[test]
    fn ln_det_pivot() {
        // [[0,1],[1,0]] det=-1 -> ln|det|=0
        let g = array![[0.0, 1.0], [1.0, 0.0]];
        assert!(ln_abs_det(&g.view()).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
    }

    #[test]
    fn matmul_basic() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a.view(), &b.view());
        assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn inverse_roundtrip() {
        let a = array![[4.0, 3.0, 0.0], [6.0, 3.0, 1.0], [0.0, 2.0, 5.0]];
        let ai = inverse(&a.view());
        let prod = matmul(&a.view(), &ai.view());
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[[i, j]] - expect).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE
                );
            }
        }
    }

    #[test]
    fn sm_row_update_matches_full_recompute() {
        // G, invert, do an up-move (row 1 += v), check ratio = det(G_new)/det(G)
        // and inv_new = inverse(G_new).
        let g = array![[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]];
        let g_inv = inverse(&g.view());
        let v = vec![0.5, -1.0, 2.0]; // row delta
        let u = vec![0.0, 1.0, 0.0]; // e_1
        let (ratio, inv_new, gem_new) = sm_update(&g_inv.view(), &g.view(), &u, &v, true, 1);
        // direct ratio via ln|det|
        let direct = (ln_abs_det(&gem_new.view()) - ln_abs_det(&g.view())).exp();
        assert!(
            (ratio.abs() - direct).abs()
                < crate::setting_consts::ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
            "{} vs {}",
            ratio.abs(),
            direct
        );
        // inverse check
        let inv_direct = inverse(&gem_new.view());
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (inv_new[[i, j]] - inv_direct[[i, j]]).abs()
                        < crate::setting_consts::ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT
                );
            }
        }
    }
}
