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

//! DTN (Drummond-Towler-Needs) cutoff-polynomial Jastrow -- `J = J_u + J_chi + J_f`.
//! Each channel is `cutoff(r)=max(L-r,0)^C` times a polynomial (1-D for u/chi,
//! 3-D in (r_iI,r_jI,r_ij) for f). 1:1 with `compute_Jastrow_DTN` (jastrow_factor.py).
//! Inflated coefficients (alpha/beta/gamma, cusp constraints baked in) are pulled from jqmc's
//! own `_dtn_inflate_*` via PyO3 so the constraint matrices need not be re-derived.

use crate::jastrow_factor::{min_image_dist, CellInfo};
use numpy::ndarray::{Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

pub struct JastrowDtnData {
    pub use_u: bool,
    pub use_chi: bool,
    pub use_f: bool,
    pub c: i32,
    pub n_en: usize,
    pub n_ee: usize,
    pub l_u: f64,
    pub l_chi: Vec<f64>,           // per species
    pub l_f: Vec<f64>,             // per species
    pub alpha: Array2<f64>,        // (2, N_u+1): row 0 = parallel, row 1 = antiparallel
    pub beta: Array2<f64>,         // (n_species, N_chi+1)
    pub gamma: Vec<f64>,           // flat (n_species, N_eN+1, N_eN+1, N_ee+1)
    pub alpha_policy: Array2<f64>, // d alpha / d ln(ell_V)
    pub beta_policy: Array2<f64>,  // d beta / d ln(ell_V)
    pub gamma_policy: Vec<f64>,    // d gamma / d ln(ell_V), same layout as gamma
    pub species_index: Vec<usize>,
    pub positions: Array2<f64>,
    pub cell: Option<Array2<f64>>,
    pub(crate) precision_policy: crate::PrecisionPolicy,
    /// The same DTN with its coefficient tensors inflated in the other precision, present
    /// only when the `jastrow_eval` or the `jastrow_ratio` dtype differs from the
    /// `jastrow_grad_lap` dtype of the tensors above. The tensors above are read by the
    /// gradient / Laplacian, drift, carried grad/lap, Pulay and strain kernels, so only
    /// `jastrow_grad_lap` selects their precision (issue #774). The value kernel reads the
    /// `jastrow_eval`-zone tensors via [`Self::eval_zone`] and the single-move ratio kernels
    /// (streaming, grouped, candidate-table, mesh and direct `compute_dtn_ratio`) read the
    /// `jastrow_ratio`-zone tensors via [`Self::ratio_zone`] (issue #766), like the XLA
    /// value and ratio kernels, which inflate in their own zone dtype. Two dtypes exist, so
    /// one copy serves both zones whenever both differ from `jastrow_grad_lap`.
    pub zone_coefficients: Option<Box<JastrowDtnData>>,
    /// Whether the coefficient tensors of this struct are float32-inflated.
    pub coefficients_f32: bool,
}

#[inline]
fn cutoff(r: f64, l: f64, c: i32) -> f64 {
    let t = l - r;
    if t > 0.0 {
        t.powi(c)
    } else {
        0.0
    }
}

#[inline]
fn poly(r: f64, coeffs: &[f64]) -> f64 {
    // Horner from the top degree.
    let mut p = 0.0_f64;
    for &a in coeffs.iter().rev() {
        p = p * r + a;
    }
    p
}

impl JastrowDtnData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let precision_policy = crate::construction_precision_policy();
        let use_u: bool = obj.getattr("use_u")?.extract()?;
        let use_chi: bool = obj.getattr("use_chi")?.extract()?;
        let use_f: bool = obj.getattr("use_f")?.extract()?;
        let c: i32 = obj.getattr("cutoff_smoothness")?.extract()?;
        let n_en: usize = obj.getattr("n_order_eN")?.extract()?;
        let n_ee: usize = obj.getattr("n_order_ee")?.extract()?;

        let np = py.import("numpy")?;
        let jnp = py.import("jax.numpy")?;
        let jf = py.import("jqmc.jastrow_factor")?;
        let np_f64ty = np.getattr("float64")?;
        // Coefficient inflation is arithmetic owned by the DTN Jastrow zones.
        // XLA performs M @ independent-parameter in the selected component
        // precision; inflating in fp64 and only casting the finished tensor is
        // observably different near constrained cancellations.
        // The shared tensors follow only the zone of their consumers (`jastrow_grad_lap`);
        // `jastrow_eval`, `jastrow_ratio` and the carried grad/lap
        // (`jastrow_grad_lap_carry`) read `zone_coefficients` when they differ.
        let coefficients_f32 = precision_policy.is_f32(DTN_SHARED_COEFFICIENT_ZONE);
        let other_zone_needed = ["jastrow_eval", "jastrow_ratio", DTN_GRAD_LAP_CARRY_ZONE]
            .iter()
            .any(|zone| precision_policy.is_f32(zone) != coefficients_f32);
        let dtype_of = |f32_zone: bool| jnp.getattr(if f32_zone { "float32" } else { "float64" });
        let coefficient_dtype = dtype_of(coefficients_f32)?;
        let as2 = |o: Bound<'_, PyAny>| -> PyResult<Array2<f64>> {
            Ok(np
                .call_method1("asarray", (o, &np_f64ty))?
                .extract::<PyReadonlyArray2<f64>>()?
                .as_array()
                .to_owned())
        };
        let item = |o: Bound<'_, PyAny>| -> PyResult<f64> {
            np.call_method1("asarray", (o,))?
                .call_method0("item")?
                .extract()
        };
        let vecf = |o: Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
            np.call_method1("asarray", (o, &np_f64ty))?
                .call_method0("ravel")?
                .call_method0("tolist")?
                .extract()
        };

        // Inflated coefficients via jqmc's own constraint maps.
        let inflate = |dtype: &Bound<'_, PyAny>| -> PyResult<DtnCoefficientSet> {
            let policy = jf.call_method1("_dtn_cutoff_policy_full_coefficients", (obj, dtype))?;
            Ok(DtnCoefficientSet {
                alpha: as2(jf.call_method1("_dtn_inflate_alpha", (obj, dtype))?)?,
                beta: as2(jf.call_method1("_dtn_inflate_beta", (obj, dtype))?)?,
                gamma: vecf(jf.call_method1("_dtn_inflate_gamma", (obj, dtype))?)?, // (I,l,m,n) C-flat
                alpha_policy: as2(policy.get_item(0)?)?,
                beta_policy: as2(policy.get_item(1)?)?,
                gamma_policy: vecf(policy.get_item(2)?)?,
            })
        };
        let coefficients = inflate(&coefficient_dtype)?;
        let other_set = if other_zone_needed {
            Some(inflate(&dtype_of(!coefficients_f32)?)?)
        } else {
            None
        };

        // Comoving cutoff scale (pytree leaf, 1.0 unless the Hamiltonian was strained through
        // ``_apply_strain``); the kernels use ``L * cutoff_scale`` exactly like ``_dtn_effective_cutoffs``.
        let cutoff_scale: f64 = match obj.getattr("cutoff_scale") {
            Ok(v) => item(v)?,
            Err(_) => 1.0,
        };
        let l_u: f64 = item(obj.getattr("L_u")?)? * cutoff_scale;
        let l_chi: Vec<f64> = vecf(obj.getattr("L_chi")?)?
            .into_iter()
            .map(|x| x * cutoff_scale)
            .collect();
        let l_f: Vec<f64> = vecf(obj.getattr("L_f")?)?
            .into_iter()
            .map(|x| x * cutoff_scale)
            .collect();
        let species_index: Vec<usize> = obj.getattr("species_index")?.extract()?;

        let sd = obj.getattr("structure_data")?;
        let positions = as2(sd.getattr("positions")?)?;
        let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
        // The DTN minimum image wraps in the basis `mic_basis_change @ cell` of
        // the same lattice (jqmc `_dtn_minimum_image`; identity unless the given
        // basis is far from reduced, issue #68). Only the image selection reads
        // `cell`, so the kernels receive that basis; `CellInfo` then sizes the
        // image shell for it.
        let cell = if pbc {
            let given = as2(sd.getattr("cell")?)?;
            let change: Vec<Vec<i64>> = match obj.getattr("mic_basis_change") {
                Ok(v) => v.extract()?,
                Err(_) => vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
            };
            let identity = (0..3).all(|i| (0..3).all(|j| change[i][j] == i64::from(i == j)));
            if identity {
                Some(given)
            } else {
                let mut wrap_basis = Array2::<f64>::zeros((3, 3));
                for i in 0..3 {
                    for c in 0..3 {
                        wrap_basis[[i, c]] =
                            (0..3).map(|k| change[i][k] as f64 * given[[k, c]]).sum();
                    }
                }
                Some(wrap_basis)
            }
        } else {
            None
        };

        let build =
            |set: DtnCoefficientSet,
             f32_set: bool,
             zone_coefficients: Option<Box<JastrowDtnData>>| JastrowDtnData {
                use_u,
                use_chi,
                use_f,
                c,
                n_en,
                n_ee,
                l_u,
                l_chi: l_chi.clone(),
                l_f: l_f.clone(),
                alpha: set.alpha,
                beta: set.beta,
                gamma: set.gamma,
                alpha_policy: set.alpha_policy,
                beta_policy: set.beta_policy,
                gamma_policy: set.gamma_policy,
                species_index: species_index.clone(),
                positions: positions.clone(),
                cell: cell.clone(),
                precision_policy,
                zone_coefficients,
                coefficients_f32: f32_set,
            };
        let zone_coefficients = other_set.map(|set| Box::new(build(set, !coefficients_f32, None)));
        Ok(build(coefficients, coefficients_f32, zone_coefficients))
    }

    /// This DTN as read by `zone`: `self` when its tensors have that zone's precision,
    /// else the copy inflated in the other precision (see `zone_coefficients`), which
    /// `from_py` builds whenever a zone that reads it differs. Idempotent.
    #[inline]
    fn zone(&self, zone: &str) -> &JastrowDtnData {
        if self.precision_policy.is_f32(zone) == self.coefficients_f32 {
            return self;
        }
        debug_assert!(
            self.zone_coefficients.is_some(),
            "DTN coefficient tensors of the {zone} precision were not inflated"
        );
        self.zone_coefficients.as_deref().unwrap_or(self)
    }

    /// This DTN as read by the `jastrow_ratio` zone (issue #766). Idempotent.
    #[inline]
    pub fn ratio_zone(&self) -> &JastrowDtnData {
        self.zone("jastrow_ratio")
    }

    /// This DTN as read by the `jastrow_eval` zone (issue #774). Idempotent.
    #[inline]
    pub fn eval_zone(&self) -> &JastrowDtnData {
        self.zone("jastrow_eval")
    }
}

/// The zone whose precision selects the shared coefficient tensors of [`JastrowDtnData`]:
/// the one their consumers (gradient / Laplacian, drift, carried grad/lap, Pulay, strain)
/// belong to. Python mirrors it in `jastrow_factor._DTN_RUST_COEFFICIENT_ZONES`.
pub(crate) const DTN_SHARED_COEFFICIENT_ZONE: &str = "jastrow_grad_lap";

/// One full DTN coefficient set (`_dtn_inflate_alpha|beta|gamma` and their cutoff-policy
/// derivatives) inflated in one dtype.
struct DtnCoefficientSet {
    alpha: Array2<f64>,
    beta: Array2<f64>,
    gamma: Vec<f64>,
    alpha_policy: Array2<f64>,
    beta_policy: Array2<f64>,
    gamma_policy: Vec<f64>,
}

/// DTN distance geometry in the mixed jastrow_grad_lap zone. Periodic
/// subtraction and minimum-image reconstruction use fp64; the displacement
/// enters fp32 for the radial derivatives. The molecular path retains fp32
/// coordinate subtraction, matching XLA.
#[inline]
fn dtn_dist_grad_lap_f32(
    a: [f64; 3],
    b: [f64; 3],
    ci: &Option<CellInfo>,
) -> (f64, [f64; 3], f64, f64) {
    let u = if let Some(ci) = ci {
        crate::jastrow_factor::min_image_disp([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci)
    } else {
        [
            ((a[0] as f32) - (b[0] as f32)) as f64,
            ((a[1] as f32) - (b[1] as f32)) as f64,
            ((a[2] as f32) - (b[2] as f32)) as f64,
        ]
    };
    let (x, y, z) = (u[0] as f32, u[1] as f32, u[2] as f32);
    let r = (x * x + y * y + z * z)
        .sqrt()
        .max(crate::jastrow_factor::EPS_SAFE_DISTANCE as f32);
    let gx = x / r;
    let gy = y / r;
    let gz = z / r;
    (
        r as f64,
        [gx as f64, gy as f64, gz as f64],
        1.0,
        (2.0_f32 / r) as f64,
    )
}

#[inline]
fn dtn_dist(a: [f64; 3], b: [f64; 3], ci: &Option<CellInfo>) -> f64 {
    match ci {
        Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci).0,
        None => {
            let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
    }
}

/// DTN ratio-zone distance.  XLA casts both point arrays to the selected dtype
/// before forming their difference; for PBC the image selection is then done
/// in fp64 and the selected displacement returns to the ratio dtype.
#[inline]
fn dtn_dist_ratio(a: [f64; 3], b: [f64; 3], ci: &Option<CellInfo>, mixed: bool) -> f64 {
    if !mixed {
        return dtn_dist(a, b, ci);
    }
    let diff = [
        ((a[0] as f32) - (b[0] as f32)) as f64,
        ((a[1] as f32) - (b[1] as f32)) as f64,
        ((a[2] as f32) - (b[2] as f32)) as f64,
    ];
    let disp = match ci {
        Some(ci) => crate::jastrow_factor::min_image_disp(diff, ci),
        None => diff,
    };
    let (x, y, z) = (disp[0] as f32, disp[1] as f32, disp[2] as f32);
    (x * x + y * y + z * z).sqrt() as f64
}

#[inline]
fn cutoff_ratio(r: f64, l: f64, c: i32, mixed: bool) -> f64 {
    if mixed {
        let t = (l as f32) - (r as f32);
        if t > 0.0 {
            t.powi(c) as f64
        } else {
            0.0
        }
    } else {
        cutoff(r, l, c)
    }
}

#[inline]
fn ratio_add(acc: &mut f64, value: f64, mixed: bool) {
    if mixed {
        *acc = ((*acc as f32) + (value as f32)) as f64;
    } else {
        *acc += value;
    }
}

#[inline]
fn ratio_sub(a: f64, b: f64, mixed: bool) -> f64 {
    if mixed {
        ((a as f32) - (b as f32)) as f64
    } else {
        a - b
    }
}

/// Geometry cache for the DTN single-electron move ratio.  This mirrors
/// `Jastrow_DTN_ratio_state` in `jastrow_factor.py`: electron--atom and
/// electron--electron distances plus every polynomial power needed by the
/// u/chi/f channels.  A proposal computes only the moved electron's new rows;
/// an accepted move replaces those rows and the symmetric e--e columns.
#[derive(Clone, Debug)]
pub struct DtnRatioState {
    pub r_ei: Vec<f64>,   // (n_e, n_atom) smooth e--atom distance
    pub cut_ei: Vec<f64>, // (n_e, n_atom) cutoff(r_ei, L_f[sp])
    pub pow_ei: Vec<f64>, // (n_e, n_atom, max_ei_order + 1)
    pub r_ee: Vec<f64>,   // (n_e, n_e) smooth e--e distance
    pub pow_ee: Vec<f64>, // (n_e, n_e, max_ee_order + 1)
    pub n_e: usize,
    pub n_atom: usize,
    pub max_ei_order: usize,
    pub max_ee_order: usize,
}

#[inline]
fn fill_powers(dst: &mut [f64], r: f64) {
    let mut power = 1.0_f64;
    for value in dst {
        *value = power;
        power *= r;
    }
}

#[inline]
fn fill_powers_ratio(dst: &mut [f64], r: f64, mixed: bool) {
    if !mixed {
        fill_powers(dst, r);
        return;
    }
    let rr = r as f32;
    // XLA constructs the exponent vector in the zone dtype and evaluates
    // `r ** arange(...)`, i.e. general f32 power rather than a recurrence.
    for (k, value) in dst.iter_mut().enumerate() {
        *value = rr.powf(k as f32) as f64;
    }
}

fn atom_pos(d: &JastrowDtnData, ia: usize) -> [f64; 3] {
    [
        d.positions[[ia, 0]],
        d.positions[[ia, 1]],
        d.positions[[ia, 2]],
    ]
}

pub fn init_dtn_ratio_state(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DtnRatioState {
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = d.positions.nrows();
    let max_ei_order = d.n_en.max(d.beta.ncols().saturating_sub(1));
    let max_ee_order = d.n_ee.max(d.alpha.ncols().saturating_sub(1));
    let n_pow_ei = max_ei_order + 1;
    let n_pow_ee = max_ee_order + 1;
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let mut r_ei = vec![0.0_f64; n_e * n_atom];
    let mut cut_ei = vec![0.0_f64; n_e * n_atom];
    let mut pow_ei = vec![0.0_f64; n_e * n_atom * n_pow_ei];
    for e in 0..n_e {
        let re = rget(e);
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let r = dtn_dist_ratio(re, atom_pos(d, ia), &ci, mixed);
            let leg = e * n_atom + ia;
            r_ei[leg] = r;
            cut_ei[leg] = cutoff_ratio(r, d.l_f[sp], d.c, mixed);
            let power_offset = leg * n_pow_ei;
            fill_powers_ratio(&mut pow_ei[power_offset..power_offset + n_pow_ei], r, mixed);
        }
    }
    let mut r_ee = vec![0.0_f64; n_e * n_e];
    let mut pow_ee = vec![0.0_f64; n_e * n_e * n_pow_ee];
    for e in 0..n_e {
        let re = rget(e);
        for k in 0..n_e {
            let r = dtn_dist_ratio(re, rget(k), &ci, mixed);
            let pair = e * n_e + k;
            r_ee[pair] = r;
            let power_offset = pair * n_pow_ee;
            fill_powers_ratio(&mut pow_ee[power_offset..power_offset + n_pow_ee], r, mixed);
        }
    }
    DtnRatioState {
        r_ei,
        cut_ei,
        pow_ei,
        r_ee,
        pow_ee,
        n_e,
        n_atom,
        max_ei_order,
        max_ee_order,
    }
}

/// Commit an accepted move to all five cached geometry tables.
pub fn advance_dtn_ratio_state(
    d: &JastrowDtnData,
    st: &mut DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
) {
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let n_up = r_up.nrows();
    let g = if is_up { idx } else { n_up + idx };
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let new_pos = rget(g);
    let n_pow_ei = st.max_ei_order + 1;
    let n_pow_ee = st.max_ee_order + 1;
    for ia in 0..st.n_atom {
        let sp = d.species_index[ia];
        let r = dtn_dist_ratio(new_pos, atom_pos(d, ia), &ci, mixed);
        let leg = g * st.n_atom + ia;
        st.r_ei[leg] = r;
        st.cut_ei[leg] = cutoff_ratio(r, d.l_f[sp], d.c, mixed);
        let power_offset = leg * n_pow_ei;
        fill_powers_ratio(
            &mut st.pow_ei[power_offset..power_offset + n_pow_ei],
            r,
            mixed,
        );
    }
    for k in 0..st.n_e {
        let r = dtn_dist_ratio(new_pos, rget(k), &ci, mixed);
        let row_pair = g * st.n_e + k;
        let col_pair = k * st.n_e + g;
        st.r_ee[row_pair] = r;
        st.r_ee[col_pair] = r;

        let row_offset = row_pair * n_pow_ee;
        fill_powers_ratio(&mut st.pow_ee[row_offset..row_offset + n_pow_ee], r, mixed);
        if col_pair != row_pair {
            let col_offset = col_pair * n_pow_ee;
            fill_powers_ratio(&mut st.pow_ee[col_offset..col_offset + n_pow_ee], r, mixed);
        }
    }
}

/// f-kernel value from cached polynomial powers and precomputed cutoffs.
fn f_value_from_powers(
    d: &JastrowDtnData,
    sp: usize,
    pow_a: &[f64],
    cut_a: f64,
    pow_b: &[f64],
    cut_b: f64,
    pow_ab: &[f64],
) -> f64 {
    if cut_a == 0.0 || cut_b == 0.0 {
        return 0.0;
    }
    let stride_n = d.n_ee + 1;
    let stride_m = (d.n_en + 1) * stride_n;
    let stride_l = (d.n_en + 1) * stride_m;
    let base = sp * stride_l;
    let mut inner = 0.0_f64;
    for (l, pow_a_l) in pow_a[..d.n_en + 1].iter().enumerate() {
        for (m, pow_b_m) in pow_b[..d.n_en + 1].iter().enumerate() {
            let off = base + l * stride_m + m * stride_n;
            let coef = *pow_a_l * *pow_b_m;
            for (n, &pair_power) in pow_ab.iter().take(d.n_ee + 1).enumerate() {
                inner += d.gamma[off + n] * coef * pair_power;
            }
        }
    }
    cut_a * cut_b * inner
}

#[allow(
    clippy::too_many_arguments,
    reason = "per-leg power tables and cutoffs of the DTN f term"
)]
fn f_value_from_powers_ratio(
    d: &JastrowDtnData,
    sp: usize,
    pow_a: &[f64],
    cut_a: f64,
    pow_b: &[f64],
    cut_b: f64,
    pow_ab: &[f64],
    mixed: bool,
) -> f64 {
    if !mixed {
        return f_value_from_powers(d, sp, pow_a, cut_a, pow_b, cut_b, pow_ab);
    }
    if cut_a == 0.0 || cut_b == 0.0 {
        return 0.0;
    }
    let stride_n = d.n_ee + 1;
    let stride_m = (d.n_en + 1) * stride_n;
    let stride_l = (d.n_en + 1) * stride_m;
    let base = sp * stride_l;
    let mut inner = 0.0_f32;
    for (l, pow_a_l) in pow_a[..d.n_en + 1].iter().enumerate() {
        for (m, pow_b_m) in pow_b[..d.n_en + 1].iter().enumerate() {
            let off = base + l * stride_m + m * stride_n;
            let coef = (*pow_a_l as f32) * (*pow_b_m as f32);
            for (n, &pair_power) in pow_ab.iter().take(d.n_ee + 1).enumerate() {
                inner += (d.gamma[off + n] as f32) * coef * (pair_power as f32);
            }
        }
    }
    ((cut_a as f32) * (cut_b as f32) * inner) as f64
}

#[inline]
fn poly_from_powers(coeffs: &[f64], powers: &[f64]) -> f64 {
    coeffs
        .iter()
        .zip(powers.iter())
        .map(|(&coefficient, &power)| coefficient * power)
        .sum()
}

#[inline]
fn poly_from_powers_ratio(coeffs: &[f64], powers: &[f64], mixed: bool) -> f64 {
    if !mixed {
        return poly_from_powers(coeffs, powers);
    }
    let mut sum = 0.0_f32;
    for (&coefficient, &power) in coeffs.iter().zip(powers.iter()) {
        sum += (coefficient as f32) * (power as f32);
    }
    sum as f64
}

/// Streaming DTN move log-ratio `DeltaJ` using the precomputed leg state.
///
/// This is also the exact mover-row rebuild used by the projection-scoped
/// candidate table.  Keeping the logarithm avoids an `exp`/`ln` round trip
/// when that table is advanced after an accepted hop.
pub(crate) fn compute_dtn_log_ratio_streaming(
    d: &JastrowDtnData,
    st: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    new_pos: [f64; 3],
) -> f64 {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = st.n_atom;
    debug_assert_eq!(st.n_e, n_e);
    debug_assert_eq!(st.n_atom, d.positions.nrows());
    let n_pow_ei = st.max_ei_order + 1;
    let n_pow_ee = st.max_ee_order + 1;
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let g_idx = if is_up { idx } else { n_up + idx };
    let mut dj = 0.0_f64;

    // Proposed moved-electron rows.  Every old row is read from the state.
    let mut r_new = vec![0.0_f64; n_atom];
    let mut cf_new = vec![0.0_f64; n_atom];
    let mut pow_ei_new = vec![0.0_f64; n_atom * n_pow_ei];
    for ia in 0..n_atom {
        let sp = d.species_index[ia];
        let r = dtn_dist_ratio(new_pos, atom_pos(d, ia), &ci, mixed);
        r_new[ia] = r;
        cf_new[ia] = cutoff_ratio(r, d.l_f[sp], d.c, mixed);
        let power_offset = ia * n_pow_ei;
        fill_powers_ratio(
            &mut pow_ei_new[power_offset..power_offset + n_pow_ei],
            r,
            mixed,
        );
    }
    let mut r_ee_new = vec![0.0_f64; n_e];
    let mut pow_ee_new = vec![0.0_f64; n_e * n_pow_ee];
    for (k, r_ee_new_k) in r_ee_new[..n_e].iter_mut().enumerate() {
        let r = dtn_dist_ratio(new_pos, rget(k), &ci, mixed);
        *r_ee_new_k = r;
        let power_offset = k * n_pow_ee;
        fill_powers_ratio(
            &mut pow_ee_new[power_offset..power_offset + n_pow_ee],
            r,
            mixed,
        );
    }

    // u: proposed powers versus the cached old e--e row.
    if d.use_u {
        let mut channel_delta = 0.0_f64;
        for (k, r_ee_new_k) in r_ee_new[..n_e].iter().enumerate() {
            if k == g_idx {
                continue;
            }
            let same = (k < n_up) == is_up;
            let channel = if same { 0 } else { 1 };
            let coefficients = d.alpha.row(channel);
            let coefficients = coefficients.as_slice().unwrap();
            let new_power_offset = k * n_pow_ee;
            let old_pair = g_idx * n_e + k;
            let old_power_offset = old_pair * n_pow_ee;
            let u_new = cutoff_ratio(*r_ee_new_k, d.l_u, d.c, mixed)
                * poly_from_powers_ratio(
                    coefficients,
                    &pow_ee_new[new_power_offset..new_power_offset + n_pow_ee],
                    mixed,
                );
            let u_old = cutoff_ratio(st.r_ee[old_pair], d.l_u, d.c, mixed)
                * poly_from_powers_ratio(
                    coefficients,
                    &st.pow_ee[old_power_offset..old_power_offset + n_pow_ee],
                    mixed,
                );
            ratio_add(&mut channel_delta, ratio_sub(u_new, u_old, mixed), mixed);
        }
        ratio_add(&mut dj, channel_delta, mixed);
    }

    // chi: proposed powers versus the cached old e--I row.
    if d.use_chi {
        let mut channel_delta = 0.0_f64;
        for (ia, r_new_ia) in r_new[..n_atom].iter().enumerate() {
            let sp = d.species_index[ia];
            let beta = d.beta.row(sp);
            let beta = beta.as_slice().unwrap();
            let new_power_offset = ia * n_pow_ei;
            let old_leg = g_idx * n_atom + ia;
            let old_power_offset = old_leg * n_pow_ei;
            let chi_n = cutoff_ratio(*r_new_ia, d.l_chi[sp], d.c, mixed)
                * poly_from_powers_ratio(
                    beta,
                    &pow_ei_new[new_power_offset..new_power_offset + n_pow_ei],
                    mixed,
                );
            let chi_o = cutoff_ratio(st.r_ei[old_leg], d.l_chi[sp], d.c, mixed)
                * poly_from_powers_ratio(
                    beta,
                    &st.pow_ei[old_power_offset..old_power_offset + n_pow_ei],
                    mixed,
                );
            ratio_add(&mut channel_delta, ratio_sub(chi_n, chi_o, mixed), mixed);
        }
        ratio_add(&mut dj, channel_delta, mixed);
    }

    // f: only atoms where idx is near (new or old) contribute; for those, only
    // electrons k with a non-zero leg cutoff to that atom.
    if d.use_f {
        let mut channel_delta = 0.0_f64;
        for (ia, cf_new_ia) in cf_new[..n_atom].iter().enumerate() {
            let cf_o = st.cut_ei[g_idx * n_atom + ia];
            if *cf_new_ia == 0.0 && cf_o == 0.0 {
                continue;
            }
            let sp = d.species_index[ia];
            let new_i_power_offset = ia * n_pow_ei;
            let old_i_leg = g_idx * n_atom + ia;
            let old_i_power_offset = old_i_leg * n_pow_ei;
            let row = ia; // atom column index into the per-electron flat table
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let ck = st.cut_ei[k * n_atom + row];
                if ck == 0.0 {
                    continue;
                }
                let k_leg = k * n_atom + row;
                let k_power_offset = k_leg * n_pow_ei;
                let new_pair_power_offset = k * n_pow_ee;
                let old_pair = g_idx * n_e + k;
                let old_pair_power_offset = old_pair * n_pow_ee;
                let pow_i_new = &pow_ei_new[new_i_power_offset..new_i_power_offset + n_pow_ei];
                let pow_i_old = &st.pow_ei[old_i_power_offset..old_i_power_offset + n_pow_ei];
                let pow_k = &st.pow_ei[k_power_offset..k_power_offset + n_pow_ei];
                let pow_pair_new =
                    &pow_ee_new[new_pair_power_offset..new_pair_power_offset + n_pow_ee];
                let pow_pair_old =
                    &st.pow_ee[old_pair_power_offset..old_pair_power_offset + n_pow_ee];
                let (fnew, fold) = if g_idx < k {
                    (
                        f_value_from_powers_ratio(
                            d,
                            sp,
                            pow_i_new,
                            *cf_new_ia,
                            pow_k,
                            ck,
                            pow_pair_new,
                            mixed,
                        ),
                        f_value_from_powers_ratio(
                            d,
                            sp,
                            pow_i_old,
                            cf_o,
                            pow_k,
                            ck,
                            pow_pair_old,
                            mixed,
                        ),
                    )
                } else {
                    (
                        f_value_from_powers_ratio(
                            d,
                            sp,
                            pow_k,
                            ck,
                            pow_i_new,
                            *cf_new_ia,
                            pow_pair_new,
                            mixed,
                        ),
                        f_value_from_powers_ratio(
                            d,
                            sp,
                            pow_k,
                            ck,
                            pow_i_old,
                            cf_o,
                            pow_pair_old,
                            mixed,
                        ),
                    )
                };
                ratio_add(&mut channel_delta, ratio_sub(fnew, fold, mixed), mixed);
            }
        }
        ratio_add(&mut dj, channel_delta, mixed);
    }
    dj
}

/// Streaming DTN move ratio `exp(DeltaJ)` using the precomputed leg state. u/chi are
/// O(N+atoms); the f-term iterates only idx's near atoms x their near electrons.
pub fn compute_dtn_ratio_streaming(
    d: &JastrowDtnData,
    st: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    new_pos: [f64; 3],
) -> f64 {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
    let dj = compute_dtn_log_ratio_streaming(d, st, r_up, r_dn, is_up, idx, new_pos);
    if d.precision_policy.is_f32("jastrow_ratio") {
        (dj as f32).exp() as f64
    } else {
        dj.exp()
    }
}

/// Group twin of [`compute_dtn_log_ratio_streaming`] for many candidate positions
/// of ONE moved electron: `out_log[c]` is bitwise the single-candidate log ratio
/// at `candidates[c]`, and `out_r_ei[c * n_atom + ia]` the candidate--atom
/// distance `dtn_dist_ratio(candidates[c], atom ia)` the candidate tables store.
/// Every candidate-independent term of the single-candidate kernel (the old u
/// pair rows, the old chi legs and the old f triples of the moved electron) is
/// evaluated once per group with the identical call and reused, the cell metric
/// is built once, and the per-candidate rows live in reused scratch; each
/// candidate still accumulates the same terms in the same order (`ratio_sub(new,
/// old)` per pair, channel sums as in the single kernel), so the result is the
/// same bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_dtn_log_ratio_streaming_group(
    d: &JastrowDtnData,
    st: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    candidates: &[[f64; 3]],
    out_log: &mut [f64],
    out_r_ei: &mut [f64],
) {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = st.n_atom;
    debug_assert_eq!(st.n_e, n_e);
    debug_assert_eq!(st.n_atom, d.positions.nrows());
    debug_assert_eq!(out_log.len(), candidates.len());
    debug_assert_eq!(out_r_ei.len(), candidates.len() * n_atom);
    let n_pow_ei = st.max_ei_order + 1;
    let n_pow_ee = st.max_ee_order + 1;
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let g_idx = if is_up { idx } else { n_up + idx };
    let rk: Vec<[f64; 3]> = (0..n_e).map(rget).collect();

    // Candidate-independent old-side terms, each the exact expression of the
    // single-candidate kernel.
    let u_old: Vec<f64> = if d.use_u {
        (0..n_e)
            .map(|k| {
                if k == g_idx {
                    return 0.0;
                }
                let same = (k < n_up) == is_up;
                let coefficients = d.alpha.row(if same { 0 } else { 1 });
                let coefficients = coefficients.as_slice().unwrap();
                let old_pair = g_idx * n_e + k;
                let old_power_offset = old_pair * n_pow_ee;
                cutoff_ratio(st.r_ee[old_pair], d.l_u, d.c, mixed)
                    * poly_from_powers_ratio(
                        coefficients,
                        &st.pow_ee[old_power_offset..old_power_offset + n_pow_ee],
                        mixed,
                    )
            })
            .collect()
    } else {
        Vec::new()
    };
    let chi_old: Vec<f64> = if d.use_chi {
        (0..n_atom)
            .map(|ia| {
                let sp = d.species_index[ia];
                let beta = d.beta.row(sp);
                let beta = beta.as_slice().unwrap();
                let old_leg = g_idx * n_atom + ia;
                let old_power_offset = old_leg * n_pow_ei;
                cutoff_ratio(st.r_ei[old_leg], d.l_chi[sp], d.c, mixed)
                    * poly_from_powers_ratio(
                        beta,
                        &st.pow_ei[old_power_offset..old_power_offset + n_pow_ei],
                        mixed,
                    )
            })
            .collect()
    } else {
        Vec::new()
    };
    // f_old[ia * n_e + k]: the old (g_idx, k, ia) triple, needed by every
    // candidate whenever the old leg or the candidate leg survives the cutoff.
    let mut f_old = Vec::new();
    if d.use_f {
        f_old = vec![0.0_f64; n_atom * n_e];
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let cf_o = st.cut_ei[g_idx * n_atom + ia];
            let old_i_leg = g_idx * n_atom + ia;
            let pow_i_old = &st.pow_ei[old_i_leg * n_pow_ei..(old_i_leg + 1) * n_pow_ei];
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let ck = st.cut_ei[k * n_atom + ia];
                if ck == 0.0 {
                    continue;
                }
                let k_leg = k * n_atom + ia;
                let pow_k = &st.pow_ei[k_leg * n_pow_ei..(k_leg + 1) * n_pow_ei];
                let old_pair = g_idx * n_e + k;
                let pow_pair_old = &st.pow_ee[old_pair * n_pow_ee..(old_pair + 1) * n_pow_ee];
                f_old[ia * n_e + k] = if g_idx < k {
                    f_value_from_powers_ratio(
                        d,
                        sp,
                        pow_i_old,
                        cf_o,
                        pow_k,
                        ck,
                        pow_pair_old,
                        mixed,
                    )
                } else {
                    f_value_from_powers_ratio(
                        d,
                        sp,
                        pow_k,
                        ck,
                        pow_i_old,
                        cf_o,
                        pow_pair_old,
                        mixed,
                    )
                };
            }
        }
    }

    let atoms: Vec<[f64; 3]> = (0..n_atom).map(|ia| atom_pos(d, ia)).collect();
    let mut cf_new = vec![0.0_f64; n_atom];
    let mut pow_ei_new = vec![0.0_f64; n_atom * n_pow_ei];
    let mut r_ee_new = vec![0.0_f64; n_e];
    let mut pow_ee_new = vec![0.0_f64; n_e * n_pow_ee];
    for (c, &new_pos) in candidates.iter().enumerate() {
        let r_new = &mut out_r_ei[c * n_atom..(c + 1) * n_atom];
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let r = dtn_dist_ratio(new_pos, atoms[ia], &ci, mixed);
            r_new[ia] = r;
            cf_new[ia] = cutoff_ratio(r, d.l_f[sp], d.c, mixed);
            let power_offset = ia * n_pow_ei;
            fill_powers_ratio(
                &mut pow_ei_new[power_offset..power_offset + n_pow_ei],
                r,
                mixed,
            );
        }
        for k in 0..n_e {
            let r = dtn_dist_ratio(new_pos, rk[k], &ci, mixed);
            r_ee_new[k] = r;
            let power_offset = k * n_pow_ee;
            fill_powers_ratio(
                &mut pow_ee_new[power_offset..power_offset + n_pow_ee],
                r,
                mixed,
            );
        }
        let mut dj = 0.0_f64;
        if d.use_u {
            let mut channel_delta = 0.0_f64;
            for k in 0..n_e {
                if k == g_idx {
                    continue;
                }
                let same = (k < n_up) == is_up;
                let coefficients = d.alpha.row(if same { 0 } else { 1 });
                let coefficients = coefficients.as_slice().unwrap();
                let new_power_offset = k * n_pow_ee;
                let u_new = cutoff_ratio(r_ee_new[k], d.l_u, d.c, mixed)
                    * poly_from_powers_ratio(
                        coefficients,
                        &pow_ee_new[new_power_offset..new_power_offset + n_pow_ee],
                        mixed,
                    );
                ratio_add(&mut channel_delta, ratio_sub(u_new, u_old[k], mixed), mixed);
            }
            ratio_add(&mut dj, channel_delta, mixed);
        }
        if d.use_chi {
            let mut channel_delta = 0.0_f64;
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let beta = d.beta.row(sp);
                let beta = beta.as_slice().unwrap();
                let new_power_offset = ia * n_pow_ei;
                let chi_n = cutoff_ratio(r_new[ia], d.l_chi[sp], d.c, mixed)
                    * poly_from_powers_ratio(
                        beta,
                        &pow_ei_new[new_power_offset..new_power_offset + n_pow_ei],
                        mixed,
                    );
                ratio_add(
                    &mut channel_delta,
                    ratio_sub(chi_n, chi_old[ia], mixed),
                    mixed,
                );
            }
            ratio_add(&mut dj, channel_delta, mixed);
        }
        if d.use_f {
            let mut channel_delta = 0.0_f64;
            for ia in 0..n_atom {
                let cf_o = st.cut_ei[g_idx * n_atom + ia];
                if cf_new[ia] == 0.0 && cf_o == 0.0 {
                    continue;
                }
                let sp = d.species_index[ia];
                let new_i_power_offset = ia * n_pow_ei;
                let pow_i_new = &pow_ei_new[new_i_power_offset..new_i_power_offset + n_pow_ei];
                for k in 0..n_e {
                    if k == g_idx {
                        continue;
                    }
                    let ck = st.cut_ei[k * n_atom + ia];
                    if ck == 0.0 {
                        continue;
                    }
                    let k_leg = k * n_atom + ia;
                    let pow_k = &st.pow_ei[k_leg * n_pow_ei..(k_leg + 1) * n_pow_ei];
                    let pow_pair_new = &pow_ee_new[k * n_pow_ee..(k + 1) * n_pow_ee];
                    let fnew = if g_idx < k {
                        f_value_from_powers_ratio(
                            d,
                            sp,
                            pow_i_new,
                            cf_new[ia],
                            pow_k,
                            ck,
                            pow_pair_new,
                            mixed,
                        )
                    } else {
                        f_value_from_powers_ratio(
                            d,
                            sp,
                            pow_k,
                            ck,
                            pow_i_new,
                            cf_new[ia],
                            pow_pair_new,
                            mixed,
                        )
                    };
                    ratio_add(
                        &mut channel_delta,
                        ratio_sub(fnew, f_old[ia * n_e + k], mixed),
                        mixed,
                    );
                }
            }
            ratio_add(&mut dj, channel_delta, mixed);
        }
        out_log[c] = dj;
    }
}

/// Rust analogue of TurboRVB's DTN part of `tabpip`: one log-ratio for every
/// electron and every point of the fixed kinetic lattice.  The table is built
/// once per projection interval and then advanced after accepted hops.
#[derive(Clone, Debug)]
pub(crate) struct DtnCandidateTable {
    log_delta: Vec<f64>, // (n_e, n_k), electron-major
    candidate_pos: Vec<[f64; 3]>,
    cand_r_ei: Vec<f64>, // (n_e, n_k, n_atom)
    shifts: Vec<[f64; 3]>,
    n_up: usize,
    n_dn: usize,
    n_k: usize,
}

impl DtnCandidateTable {
    pub(crate) fn matches_layout(
        &self,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        shifts: &[[f64; 3]],
    ) -> bool {
        self.n_up == r_up.nrows()
            && self.n_dn == r_dn.nrows()
            && self.n_k == shifts.len()
            && self
                .shifts
                .iter()
                .zip(shifts)
                .all(|(a, b)| a.iter().zip(b).all(|(&x, &y)| x.to_bits() == y.to_bits()))
    }

    pub(crate) fn ratios_for_group(
        &self,
        d: &JastrowDtnData,
        is_up: bool,
        electron: usize,
        positions: &[[f64; 3]],
    ) -> Option<Vec<f64>> {
        let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
        let global = if is_up {
            electron
        } else {
            self.n_up + electron
        };
        if global >= self.n_up + self.n_dn || positions.is_empty() || positions.len() > self.n_k {
            return None;
        }
        let row_offset = global * self.n_k;
        let expected = &self.candidate_pos[row_offset..row_offset + self.n_k];
        // Kinetic requests the complete six-point row. ECP requests one
        // nucleus' Nv-point group from the electron's concatenated NN*Nv row;
        // find that exact contiguous block without introducing a caller-side
        // group-index convention.
        let start = expected.windows(positions.len()).position(|window| {
            window
                .iter()
                .zip(positions)
                .all(|(a, b)| a.iter().zip(b).all(|(&x, &y)| x.to_bits() == y.to_bits()))
        })?;
        let offset = row_offset + start;
        let mixed = d.precision_policy.is_f32("jastrow_ratio");
        Some(
            self.log_delta[offset..offset + positions.len()]
                .iter()
                .map(|&x| {
                    if mixed {
                        (x as f32).exp() as f64
                    } else {
                        x.exp()
                    }
                })
                .collect(),
        )
    }

    /// Shape-only gate for a candidate mesh whose points are not generated by
    /// one common translation table (the ECP quadrature mesh).  Exact point
    /// identity is still checked by `ratios_for_group`; after a hop the moved
    /// row is refreshed from the newly generated positions before any read.
    pub(crate) fn matches_position_layout(
        &self,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        positions_up: &[[f64; 3]],
        positions_dn: &[[f64; 3]],
        n_k: usize,
    ) -> bool {
        self.n_up == r_up.nrows()
            && self.n_dn == r_dn.nrows()
            && self.n_k == n_k
            && positions_up.len() == self.n_up * n_k
            && positions_dn.len() == self.n_dn * n_k
    }

    /// Test-probe dimensions `(n_electrons, n_candidates_per_electron, n_atom)`
    /// (long-chain test probe; not used by any driver).
    pub(crate) fn probe_dims(&self) -> (usize, usize, usize) {
        let n_e = self.n_up + self.n_dn;
        let n_atom = self
            .cand_r_ei
            .len()
            .checked_div(n_e * self.n_k)
            .unwrap_or(0);
        (n_e, self.n_k, n_atom)
    }

    /// Test-probe copies of the carried `(log_delta, candidate positions
    /// (flattened xyz), candidate e-atom distances)` tables, electron-major.
    pub(crate) fn probe_arrays(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            self.log_delta.clone(),
            self.candidate_pos.iter().flatten().copied().collect(),
            self.cand_r_ei.clone(),
        )
    }
}

pub(crate) fn init_dtn_candidate_table(
    d: &JastrowDtnData,
    st: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    shifts: &[[f64; 3]],
) -> DtnCandidateTable {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_k = shifts.len();
    let n_atom = d.positions.nrows();
    let mut log_delta = vec![0.0_f64; n_e * n_k];
    let mut candidate_pos = Vec::with_capacity(n_e * n_k);
    let mut cand_r_ei = vec![0.0_f64; n_e * n_k * n_atom];
    for global in 0..n_e {
        let (is_up, electron, base) = if global < n_up {
            (
                true,
                global,
                [r_up[[global, 0]], r_up[[global, 1]], r_up[[global, 2]]],
            )
        } else {
            let electron = global - n_up;
            (
                false,
                electron,
                [
                    r_dn[[electron, 0]],
                    r_dn[[electron, 1]],
                    r_dn[[electron, 2]],
                ],
            )
        };
        let row = candidate_pos.len();
        for shift in shifts {
            candidate_pos.push([base[0] + shift[0], base[1] + shift[1], base[2] + shift[2]]);
        }
        // One group call per electron row: bitwise the per-candidate
        // `compute_dtn_log_ratio_streaming` / `dtn_dist_ratio` values.
        compute_dtn_log_ratio_streaming_group(
            d,
            st,
            r_up,
            r_dn,
            is_up,
            electron,
            &candidate_pos[row..row + n_k],
            &mut log_delta[row..row + n_k],
            &mut cand_r_ei[row * n_atom..(row + n_k) * n_atom],
        );
    }
    DtnCandidateTable {
        log_delta,
        candidate_pos,
        cand_r_ei,
        shifts: shifts.to_vec(),
        n_up,
        n_dn,
        n_k,
    }
}

/// ECP twin of [`init_dtn_candidate_table`].  ECP quadrature points are not a
/// common set of translations: each electron/nucleus pair has its own radial
/// shell, so seed directly from the exact electron-major candidate arrays.
pub(crate) fn init_dtn_candidate_table_from_positions(
    d: &JastrowDtnData,
    st: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions_up: &[[f64; 3]],
    positions_dn: &[[f64; 3]],
    n_k: usize,
) -> DtnCandidateTable {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    assert_eq!(positions_up.len(), n_up * n_k);
    assert_eq!(positions_dn.len(), n_dn * n_k);
    let n_e = n_up + n_dn;
    let n_atom = d.positions.nrows();
    let mut log_delta = vec![0.0_f64; n_e * n_k];
    let mut candidate_pos = Vec::with_capacity(n_e * n_k);
    let mut cand_r_ei = vec![0.0_f64; n_e * n_k * n_atom];
    for global in 0..n_e {
        let (is_up, electron, positions) = if global < n_up {
            (true, global, positions_up)
        } else {
            (false, global - n_up, positions_dn)
        };
        let offset = electron * n_k;
        let cands = &positions[offset..offset + n_k];
        candidate_pos.extend_from_slice(cands);
        let row = global * n_k;
        // One group call per electron row: bitwise the per-candidate
        // `compute_dtn_log_ratio_streaming` / `dtn_dist_ratio` values.
        compute_dtn_log_ratio_streaming_group(
            d,
            st,
            r_up,
            r_dn,
            is_up,
            electron,
            cands,
            &mut log_delta[row..row + n_k],
            &mut cand_r_ei[row * n_atom..(row + n_k) * n_atom],
        );
    }
    DtnCandidateTable {
        log_delta,
        candidate_pos,
        cand_r_ei,
        // Empty by construction: ECP moved rows must be supplied explicitly
        // to the position-based advance below.
        shifts: Vec::new(),
        n_up,
        n_dn,
        n_k,
    }
}

struct DtnCorrectionScratch {
    cand_pow_ei: Vec<f64>,
    pair_powers: [Vec<f64>; 4], // cand-new, cand-old, base-new, base-old
}

impl DtnCorrectionScratch {
    fn new(d: &JastrowDtnData) -> Self {
        let pair_len = d.n_ee.max(d.alpha.ncols().saturating_sub(1)) + 1;
        Self {
            cand_pow_ei: vec![0.0; d.n_en + 1],
            pair_powers: std::array::from_fn(|_| vec![0.0; pair_len]),
        }
    }
}

#[inline]
fn dtn_u_from_pair_powers(d: &JastrowDtnData, r: f64, powers: &[f64], same_spin: bool) -> f64 {
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let channel = if same_spin { 0 } else { 1 };
    cutoff_ratio(r, d.l_u, d.c, mixed)
        * poly_from_powers_ratio(d.alpha.row(channel).as_slice().unwrap(), powers, mixed)
}

/// Contract the mover's e--atom polynomial leg into gamma once per hop.
/// The remaining axes are the other electron's eN order and the pair order.
fn fill_dtn_mover_gamma(
    d: &JastrowDtnData,
    sp: usize,
    mover_pow: &[f64],
    mover_cut: f64,
    mover_is_first: bool,
    out: &mut [f64],
) {
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let n_en = d.n_en + 1;
    let n_ee = d.n_ee + 1;
    let stride_m = n_en * n_ee;
    let stride_l = n_en * stride_m;
    let base = sp * stride_l;
    for row_order in 0..n_en {
        for n in 0..n_ee {
            let value = if mixed {
                let mut sum = 0.0_f32;
                for (mover_order, mover_pow_mover_order) in mover_pow[..n_en].iter().enumerate() {
                    let gamma_idx = if mover_is_first {
                        base + mover_order * stride_m + row_order * n_ee + n
                    } else {
                        base + row_order * stride_m + mover_order * n_ee + n
                    };
                    sum += (d.gamma[gamma_idx] as f32) * (*mover_pow_mover_order as f32);
                }
                ((mover_cut as f32) * sum) as f64
            } else {
                let mut sum = 0.0_f64;
                for (mover_order, mover_pow_mover_order) in mover_pow[..n_en].iter().enumerate() {
                    let gamma_idx = if mover_is_first {
                        base + mover_order * stride_m + row_order * n_ee + n
                    } else {
                        base + row_order * stride_m + mover_order * n_ee + n
                    };
                    sum += d.gamma[gamma_idx] * *mover_pow_mover_order;
                }
                mover_cut * sum
            };
            out[row_order * n_ee + n] = value;
        }
    }
}

/// Safety margins (bohr) of the far-row skip bound in
/// [`advance_dtn_candidate_table_impl`]: they absorb the rounding of the
/// distances the bound compares (fp64 ~1e-15 relative; the mixed route forms
/// distances from f32 coordinates, ~1e-6 relative on |r| < 1e2 bohr).
const DTN_FAR_ROW_MARGIN_F64: f64 = 1.0e-9;
const DTN_FAR_ROW_MARGIN_F32: f64 = 1.0e-3;

fn dtn_f_from_mover_gamma(
    row_pow: &[f64],
    row_cut: f64,
    pair_pow: &[f64],
    mover_gamma: &[f64],
    n_en: usize,
    n_ee: usize,
    mixed: bool,
) -> f64 {
    if row_cut == 0.0 {
        return 0.0;
    }
    if mixed {
        let mut sum = 0.0_f32;
        for l in 0..n_en {
            let leg = (row_cut as f32) * (row_pow[l] as f32);
            for n in 0..n_ee {
                sum += leg * (mover_gamma[l * n_ee + n] as f32) * (pair_pow[n] as f32);
            }
        }
        sum as f64
    } else {
        let mut sum = 0.0_f64;
        for l in 0..n_en {
            let leg = row_cut * row_pow[l];
            for n in 0..n_ee {
                sum += leg * mover_gamma[l * n_ee + n] * pair_pow[n];
            }
        }
        sum
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_dtn_candidate_table(
    d: &JastrowDtnData,
    table: &mut DtnCandidateTable,
    new_state: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    electron: usize,
    old_pos: [f64; 3],
) {
    advance_dtn_candidate_table_impl(
        d, table, new_state, r_up, r_dn, is_up, electron, old_pos, None,
    );
}

/// Advance an arbitrary-position (ECP) candidate table.  Unmoved rows use the
/// same O(N*k) pair correction as kinetic tabpip; only the moved electron's
/// row is rebuilt from the freshly generated quadrature points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_dtn_candidate_table_from_positions(
    d: &JastrowDtnData,
    table: &mut DtnCandidateTable,
    new_state: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    electron: usize,
    old_pos: [f64; 3],
    moved_candidates: &[[f64; 3]],
) {
    assert_eq!(moved_candidates.len(), table.n_k);
    advance_dtn_candidate_table_impl(
        d,
        table,
        new_state,
        r_up,
        r_dn,
        is_up,
        electron,
        old_pos,
        Some(moved_candidates),
    );
}

#[allow(clippy::too_many_arguments)]
fn advance_dtn_candidate_table_impl(
    d: &JastrowDtnData,
    table: &mut DtnCandidateTable,
    new_state: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    electron: usize,
    old_pos: [f64; 3],
    moved_candidates: Option<&[[f64; 3]]>,
) {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
    debug_assert_eq!(table.n_up, r_up.nrows());
    debug_assert_eq!(table.n_dn, r_dn.nrows());
    let mixed = d.precision_policy.is_f32("jastrow_ratio");
    let n_e = table.n_up + table.n_dn;
    let mover = if is_up {
        electron
    } else {
        table.n_up + electron
    };
    let position = |global: usize| -> [f64; 3] {
        if global < table.n_up {
            [r_up[[global, 0]], r_up[[global, 1]], r_up[[global, 2]]]
        } else {
            let e = global - table.n_up;
            [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]]
        }
    };
    let new_pos = position(mover);
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let n_atom = d.positions.nrows();
    let n_pow_ei = new_state.max_ei_order + 1;
    let mut mover_old_cut = vec![0.0_f64; n_atom];
    let mut mover_old_pow = vec![0.0_f64; n_atom * n_pow_ei];
    for ia in 0..n_atom {
        let sp = d.species_index[ia];
        let r = dtn_dist_ratio(old_pos, atom_pos(d, ia), &ci, mixed);
        mover_old_cut[ia] = cutoff_ratio(r, d.l_f[sp], d.c, mixed);
        fill_powers_ratio(
            &mut mover_old_pow[ia * n_pow_ei..(ia + 1) * n_pow_ei],
            r,
            mixed,
        );
    }
    let n_en = d.n_en + 1;
    let n_ee = d.n_ee + 1;
    let gamma_leg = n_en * n_ee;
    // Atoms whose mover leg is live at the old or the new position. For every
    // other atom both contracted mover legs are `0 * sum` (exactly zero), so each
    // of its four f contributions is a signed zero and adding it leaves the
    // accumulator unchanged (the accumulators start at +0 and never become -0).
    // Skipping them -- and their gamma contraction -- is bit-identical.
    let active_atoms: Vec<usize> = if d.use_f {
        (0..n_atom)
            .filter(|&ia| new_state.cut_ei[mover * n_atom + ia] != 0.0 || mover_old_cut[ia] != 0.0)
            .collect()
    } else {
        Vec::new()
    };
    let mut g_new_first = vec![0.0_f64; n_atom * gamma_leg];
    let mut g_old_first = vec![0.0_f64; n_atom * gamma_leg];
    let mut g_new_second = vec![0.0_f64; n_atom * gamma_leg];
    let mut g_old_second = vec![0.0_f64; n_atom * gamma_leg];
    for &ia in &active_atoms {
        let sp = d.species_index[ia];
        let mover_leg = mover * n_atom + ia;
        let mover_new_pow = &new_state.pow_ei[mover_leg * n_pow_ei..(mover_leg + 1) * n_pow_ei];
        let mover_old_pow = &mover_old_pow[ia * n_pow_ei..(ia + 1) * n_pow_ei];
        let out = ia * gamma_leg..(ia + 1) * gamma_leg;
        fill_dtn_mover_gamma(
            d,
            sp,
            mover_new_pow,
            new_state.cut_ei[mover_leg],
            true,
            &mut g_new_first[out.clone()],
        );
        fill_dtn_mover_gamma(
            d,
            sp,
            mover_old_pow,
            mover_old_cut[ia],
            true,
            &mut g_old_first[out.clone()],
        );
        fill_dtn_mover_gamma(
            d,
            sp,
            mover_new_pow,
            new_state.cut_ei[mover_leg],
            false,
            &mut g_new_second[out.clone()],
        );
        fill_dtn_mover_gamma(
            d,
            sp,
            mover_old_pow,
            mover_old_cut[ia],
            false,
            &mut g_old_second[out],
        );
    }
    let mut scratch = DtnCorrectionScratch::new(d);

    // IMPL-RS-DTN-TABPIP-FAR-ROW-SKIP-001: a row whose every candidate is
    // provably outside L_u of both mover positions and outside L_f of every
    // active atom gets a correction built only from exact-zero cutoff terms
    // (`cutoff_ratio` returns 0 for r >= L), so adding it cannot change any
    // stored value except -0 -> +0. The bound is the triangle inequality
    // `d(c,x) >= d(b,x) - |c-b|` on the true minimum-image metric with
    // `delta = max_k |c_k - b|` (raw Euclidean, an upper bound on the minimum
    // image distance). The implemented minimum image equals the true one for
    // molecules, orthorhombic cells, and (by the shell rule documented on
    // `CellInfo::mic_half_widths`) for any displacement shorter than
    // `rho = min_i |b_i| / 2`; it is never shorter than the true one. So
    // `min(d_impl(b,x), rho_eff)` lower-bounds the true `d(b,x)`.
    let far_skip = std::env::var_os("JQMC_RS_DTN_TABPIP_NO_FAR_SKIP").is_none();
    let rho_eff = match ci.as_ref() {
        None => f64::INFINITY,
        Some(c) => {
            let dot = |a: usize, b: usize| (0..3).map(|k| c.cell[a][k] * c.cell[b][k]).sum::<f64>();
            if dot(0, 1) == 0.0 && dot(0, 2) == 0.0 && dot(1, 2) == 0.0 {
                f64::INFINITY
            } else {
                (0..3)
                    .map(|a| dot(a, a).sqrt())
                    .fold(f64::INFINITY, f64::min)
                    * 0.5
            }
        }
    };
    let far_margin = if mixed {
        DTN_FAR_ROW_MARGIN_F32
    } else {
        DTN_FAR_ROW_MARGIN_F64
    };
    let mut far_rows_skipped = 0u64;

    // Every unmoved candidate row changes only through its pair with the
    // accepted mover: P(c,n)-P(c,o)-P(r,n)+P(r,o).  The f pair is evaluated
    // in canonical global-electron order, as in the full ratio kernel.
    //
    // The base terms P(r,n), P(r,o) depend on the row only, so they are formed
    // once per row and reused for its n_k candidates: the same operations on the
    // same operands, hence the same values the per-candidate evaluation gave. In
    // the f sum a skipped (atom, candidate) pair contributes an exact +0 to the
    // base accumulators (`dtn_f_from_mover_gamma` returns +0 for a zero row
    // cutoff), so the base sum over the live atoms is the per-candidate value.
    for row in 0..n_e {
        if row == mover {
            continue;
        }
        let base = position(row);
        let same_spin = (row < table.n_up) == is_up;
        let d_bn = dtn_dist_ratio(base, new_pos, &ci, mixed);
        let d_bo = dtn_dist_ratio(base, old_pos, &ci, mixed);
        if far_skip {
            let mut delta2 = 0.0_f64;
            for k in 0..table.n_k {
                let c = table.candidate_pos[row * table.n_k + k];
                let (x, y, z) = (c[0] - base[0], c[1] - base[1], c[2] - base[2]);
                delta2 = delta2.max(x * x + y * y + z * z);
            }
            let reach = delta2.sqrt() + far_margin;
            let u_far = !d.use_u || d_bn.min(d_bo).min(rho_eff) >= d.l_u + reach;
            let f_far = u_far
                && active_atoms.iter().all(|&ia| {
                    let r = new_state.r_ei[row * n_atom + ia];
                    r.min(rho_eff) >= d.l_f[d.species_index[ia]] + reach
                });
            if u_far && f_far {
                far_rows_skipped += 1;
                continue;
            }
        }
        {
            let [_, _, pb_n, pb_o] = &mut scratch.pair_powers;
            fill_powers_ratio(pb_n, d_bn, mixed);
            fill_powers_ratio(pb_o, d_bo, mixed);
        }
        let base_delta_u = if d.use_u {
            let u_bn = dtn_u_from_pair_powers(d, d_bn, &scratch.pair_powers[2], same_spin);
            let u_bo = dtn_u_from_pair_powers(d, d_bo, &scratch.pair_powers[3], same_spin);
            ratio_sub(u_bn, u_bo, mixed)
        } else {
            0.0
        };
        let base_delta_f = if d.use_f {
            let mut f_bn = 0.0_f64;
            let mut f_bo = 0.0_f64;
            for &ia in &active_atoms {
                let base_leg = row * n_atom + ia;
                let base_cut = new_state.cut_ei[base_leg];
                if base_cut == 0.0 {
                    continue;
                }
                let base_pow = &new_state.pow_ei[base_leg * n_pow_ei..(base_leg + 1) * n_pow_ei];
                let g_range = ia * gamma_leg..(ia + 1) * gamma_leg;
                let (g_new, g_old) = if row < mover {
                    (&g_new_second[g_range.clone()], &g_old_second[g_range])
                } else {
                    (&g_new_first[g_range.clone()], &g_old_first[g_range])
                };
                ratio_add(
                    &mut f_bn,
                    dtn_f_from_mover_gamma(
                        base_pow,
                        base_cut,
                        &scratch.pair_powers[2],
                        g_new,
                        n_en,
                        n_ee,
                        mixed,
                    ),
                    mixed,
                );
                ratio_add(
                    &mut f_bo,
                    dtn_f_from_mover_gamma(
                        base_pow,
                        base_cut,
                        &scratch.pair_powers[3],
                        g_old,
                        n_en,
                        n_ee,
                        mixed,
                    ),
                    mixed,
                );
            }
            ratio_sub(f_bn, f_bo, mixed)
        } else {
            0.0
        };
        for k in 0..table.n_k {
            let offset = row * table.n_k + k;
            let candidate = table.candidate_pos[offset];
            let d_cn = dtn_dist_ratio(candidate, new_pos, &ci, mixed);
            let d_co = dtn_dist_ratio(candidate, old_pos, &ci, mixed);
            {
                let [pc_n, pc_o, _, _] = &mut scratch.pair_powers;
                fill_powers_ratio(pc_n, d_cn, mixed);
                fill_powers_ratio(pc_o, d_co, mixed);
            }
            let mut correction = 0.0_f64;
            if d.use_u {
                let u_cn = dtn_u_from_pair_powers(d, d_cn, &scratch.pair_powers[0], same_spin);
                let u_co = dtn_u_from_pair_powers(d, d_co, &scratch.pair_powers[1], same_spin);
                let candidate_delta = ratio_sub(u_cn, u_co, mixed);
                ratio_add(
                    &mut correction,
                    ratio_sub(candidate_delta, base_delta_u, mixed),
                    mixed,
                );
            }
            if d.use_f {
                let mut f_cn = 0.0_f64;
                let mut f_co = 0.0_f64;
                for &ia in &active_atoms {
                    let sp = d.species_index[ia];
                    let cand_leg = (offset * n_atom) + ia;
                    let cand_r = table.cand_r_ei[cand_leg];
                    let cand_cut = cutoff_ratio(cand_r, d.l_f[sp], d.c, mixed);
                    if cand_cut == 0.0 {
                        continue;
                    }
                    fill_powers_ratio(&mut scratch.cand_pow_ei, cand_r, mixed);
                    let g_range = ia * gamma_leg..(ia + 1) * gamma_leg;
                    let (g_new, g_old) = if row < mover {
                        (&g_new_second[g_range.clone()], &g_old_second[g_range])
                    } else {
                        (&g_new_first[g_range.clone()], &g_old_first[g_range])
                    };
                    ratio_add(
                        &mut f_cn,
                        dtn_f_from_mover_gamma(
                            &scratch.cand_pow_ei,
                            cand_cut,
                            &scratch.pair_powers[0],
                            g_new,
                            n_en,
                            n_ee,
                            mixed,
                        ),
                        mixed,
                    );
                    ratio_add(
                        &mut f_co,
                        dtn_f_from_mover_gamma(
                            &scratch.cand_pow_ei,
                            cand_cut,
                            &scratch.pair_powers[1],
                            g_old,
                            n_en,
                            n_ee,
                            mixed,
                        ),
                        mixed,
                    );
                }
                let candidate_delta = ratio_sub(f_cn, f_co, mixed);
                ratio_add(
                    &mut correction,
                    ratio_sub(candidate_delta, base_delta_f, mixed),
                    mixed,
                );
            }
            ratio_add(&mut table.log_delta[offset], correction, mixed);
        }
    }

    crate::gfmc::bump_route_counter_by(&crate::gfmc::GFMC_DTN_FAR_ROWS, far_rows_skipped);

    // Rebuild only the mover row exactly so chi and all pair channels enter
    // once. Kinetic points share translations; ECP supplies its newly
    // generated nucleus-dependent quadrature positions explicitly.
    let row = mover * table.n_k;
    for k in 0..table.n_k {
        table.candidate_pos[row + k] = match moved_candidates {
            Some(points) => points[k],
            None => [
                new_pos[0] + table.shifts[k][0],
                new_pos[1] + table.shifts[k][1],
                new_pos[2] + table.shifts[k][2],
            ],
        };
    }
    compute_dtn_log_ratio_streaming_group(
        d,
        new_state,
        r_up,
        r_dn,
        is_up,
        electron,
        &table.candidate_pos[row..row + table.n_k],
        &mut table.log_delta[row..row + table.n_k],
        &mut table.cand_r_ei[row * n_atom..(row + table.n_k) * n_atom],
    );
}

/// Scalar f-kernel value `cutoff(r_a)*cutoff(r_b)*Sum_lmn gamma r_a^l r_b^m r_ab^n`.
fn f_value(d: &JastrowDtnData, sp: usize, r_a: f64, r_b: f64, r_ab: f64) -> f64 {
    let cut_a = cutoff(r_a, d.l_f[sp], d.c);
    if cut_a == 0.0 {
        return 0.0;
    }
    let cut_b = cutoff(r_b, d.l_f[sp], d.c);
    if cut_b == 0.0 {
        return 0.0;
    }
    let stride_n = d.n_ee + 1;
    let stride_m = (d.n_en + 1) * stride_n;
    let stride_l = (d.n_en + 1) * stride_m;
    let base = sp * stride_l;
    let mut pa = vec![0.0; d.n_en + 1];
    let mut t = 1.0;
    for v in pa.iter_mut() {
        *v = t;
        t *= r_a;
    }
    let mut pb = vec![0.0; d.n_en + 1];
    t = 1.0;
    for v in pb.iter_mut() {
        *v = t;
        t *= r_b;
    }
    let mut pab = vec![0.0; d.n_ee + 1];
    t = 1.0;
    for v in pab.iter_mut() {
        *v = t;
        t *= r_ab;
    }
    let mut inner = 0.0_f64;
    for (l, pa_l) in pa[..d.n_en + 1].iter().enumerate() {
        for (m, pb_m) in pb[..d.n_en + 1].iter().enumerate() {
            let off = base + l * stride_m + m * stride_n;
            let coef = *pa_l * *pb_m;
            for (n, pabn) in pab.iter().enumerate() {
                inner += d.gamma[off + n] * coef * pabn;
            }
        }
    }
    cut_a * cut_b * inner
}

/// DTN single-electron-move ratio `exp(J_new - J_old)` for moving electron
/// `idx` (spin `is_up`) to `new_pos`. Incremental over the terms touching it.
pub fn compute_dtn_ratio(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    new_pos: [f64; 3],
) -> f64 {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
                            // XLA evaluates the mixed DTN move through the cached-leg formulation.  Use
                            // the same formulation here so both the leaf arithmetic and its reduction
                            // order are governed by the `jastrow_ratio` f32 zone.  Keep the established
                            // direct f64 implementation below byte-for-byte for full precision.
    if d.precision_policy.is_f32("jastrow_ratio") {
        let state = init_dtn_ratio_state(d, r_up, r_dn);
        return compute_dtn_ratio_streaming(d, &state, r_up, r_dn, is_up, idx, new_pos);
    }
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
        match &ci {
            Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci).0,
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
        }
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
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
    let g_idx = if is_up { idx } else { n_up + idx };
    let r_old = rget(g_idx);
    let mut dj = 0.0_f64;

    // u: pairs (idx, k) for all k != idx; alpha[0] same-spin / alpha[1] anti-spin.
    if d.use_u {
        let a_par: Vec<f64> = d.alpha.row(0).to_vec();
        let a_anti: Vec<f64> = d.alpha.row(1).to_vec();
        for k in 0..(n_up + n_dn) {
            if k == g_idx {
                continue;
            }
            let same = (k < n_up) == is_up;
            let a = if same { &a_par } else { &a_anti };
            let rk = rget(k);
            let uk = |r: f64| cutoff(r, d.l_u, d.c) * poly(r, a);
            dj += uk(dist(new_pos, rk)) - uk(dist(r_old, rk));
        }
    }

    // chi: idx x all atoms.
    if d.use_chi {
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let p = pos(ia);
            let chik =
                |r: f64| cutoff(r, d.l_chi[sp], d.c) * poly(r, d.beta.row(sp).as_slice().unwrap());
            dj += chik(dist(new_pos, p)) - chik(dist(r_old, p));
        }
    }

    // f: triples (idx, k, I); r_kI fixed. Respect the i<j ordering of the pair.
    if d.use_f {
        for k in 0..(n_up + n_dn) {
            if k == g_idx {
                continue;
            }
            let rk = rget(k);
            let r_ik_new = dist(new_pos, rk);
            let r_ik_old = dist(r_old, rk);
            let idx_first = g_idx < k;
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let p = pos(ia);
                let r_ki = dist(rk, p);
                if d.l_f[sp] - r_ki <= 0.0 {
                    continue; // other leg cut -> F=0 for both old and new
                }
                let r_ii_new = dist(new_pos, p);
                let r_ii_old = dist(r_old, p);
                let (fnew, fold) = if idx_first {
                    (
                        f_value(d, sp, r_ii_new, r_ki, r_ik_new),
                        f_value(d, sp, r_ii_old, r_ki, r_ik_old),
                    )
                } else {
                    (
                        f_value(d, sp, r_ki, r_ii_new, r_ik_new),
                        f_value(d, sp, r_ki, r_ii_old, r_ik_old),
                    )
                };
                dj += fnew - fold;
            }
        }
    }
    dj.exp()
}

/// Batched DTN single-move ratio for the ECP-nonlocal mesh: `exp(J(mesh_v) - J(r_old))`
/// for ALL `nv` mesh points of electron `idx` (spin `is_up`) at once, into `out`.
/// Equivalent to calling [`compute_dtn_ratio`] per mesh point, but the mesh points all
/// share electron `idx`'s OLD position and the moved electron's partners/atoms are fixed,
/// so the entire OLD-side `J(r_old)` and the constant electron-ion `r_kI` / leg-cutoff
/// table are computed ONCE and reused across the mesh (the per-mesh work drops to the
/// NEW-side `J(mesh_v)` only). Bit-behaviour: sums the per-position `J` then subtracts
/// (fp-reordered vs the per-term `dj += new - old` of `compute_dtn_ratio`), matching to
/// the fp floor -- the ECP-nl energy it feeds is floor-compared (fast chunk).
pub fn compute_dtn_ratio_mesh(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    mesh: &ArrayView2<f64>,
    out: &mut [f64],
) {
    compute_dtn_ratio_mesh_impl(d, None, r_up, r_dn, is_up, idx, mesh, out);
}

/// State-borrowing twin of [`compute_dtn_ratio_mesh`] for callers that already
/// hold a `DtnRatioState` for this walker (the GFMC candidate loops and the
/// force mesh). Identical results; it only avoids rebuilding the geometry cache
/// that the mixed-precision rank-1 reduction needs.
#[allow(
    clippy::too_many_arguments,
    reason = "streaming ratio state, move and mesh inputs"
)]
pub(crate) fn compute_dtn_ratio_mesh_cached(
    d: &JastrowDtnData,
    state: &DtnRatioState,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    mesh: &ArrayView2<f64>,
    out: &mut [f64],
) {
    compute_dtn_ratio_mesh_impl(d, Some(state), r_up, r_dn, is_up, idx, mesh, out);
}

#[allow(clippy::too_many_arguments)]
fn compute_dtn_ratio_mesh_impl(
    d: &JastrowDtnData,
    cached_state: Option<&DtnRatioState>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    idx: usize,
    mesh: &ArrayView2<f64>,
    out: &mut [f64],
) {
    let d = d.ratio_zone(); // `jastrow_ratio`-zone coefficients (issue #766)
                            // Preserve the rank-1 mixed reduction order for every ECP mesh point.  The
                            // geometry cache is built only once; the full-precision optimized mesh
                            // implementation remains below.
    if d.precision_policy.is_f32("jastrow_ratio") {
        let owned = cached_state
            .is_none()
            .then(|| init_dtn_ratio_state(d, r_up, r_dn));
        let state = cached_state.or(owned.as_ref()).expect("mixed DTN geometry");
        for v in 0..mesh.nrows() {
            out[v] = compute_dtn_ratio_streaming(
                d,
                state,
                r_up,
                r_dn,
                is_up,
                idx,
                [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]],
            );
        }
        return;
    }
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
        match &ci {
            Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci).0,
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
        }
    };
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = d.positions.nrows();
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let g_idx = if is_up { idx } else { n_up + idx };
    let r_old = rget(g_idx);
    let a_par: Vec<f64> = d.alpha.row(0).to_vec();
    let a_anti: Vec<f64> = d.alpha.row(1).to_vec();
    // Constant across mesh points: partner positions, atom positions, and the
    // f-channel passing triples (k, atom) whose fixed leg r_kI clears the cutoff.
    let rk_pos: Vec<[f64; 3]> = (0..n_e).map(rget).collect();
    let atom_pos: Vec<[f64; 3]> = (0..n_atom)
        .map(|ia| {
            [
                d.positions[[ia, 0]],
                d.positions[[ia, 1]],
                d.positions[[ia, 2]],
            ]
        })
        .collect();
    let mut fpass: Vec<(usize, usize, usize, f64, bool)> = Vec::new();
    if d.use_f {
        for (k, rk_pos_k) in rk_pos[..n_e].iter().enumerate() {
            if k == g_idx {
                continue;
            }
            let idx_first = g_idx < k;
            for (ia, atom_pos_ia) in atom_pos[..n_atom].iter().enumerate() {
                let sp = d.species_index[ia];
                let r_ki = dist(*rk_pos_k, *atom_pos_ia);
                if d.l_f[sp] - r_ki <= 0.0 {
                    continue; // other leg cut -> F=0 regardless of the moved electron
                }
                fpass.push((k, ia, sp, r_ki, idx_first));
            }
        }
    }
    let mut d_ik = vec![0.0_f64; n_e];
    let mut d_ii = vec![0.0_f64; n_atom];
    // Total DTN self-part J at position `p` for the moved electron (u + chi + f).
    let j_at = |p: [f64; 3], d_ik: &mut [f64], d_ii: &mut [f64]| -> f64 {
        for k in 0..n_e {
            if k != g_idx {
                d_ik[k] = dist(p, rk_pos[k]);
            }
        }
        for ia in 0..n_atom {
            d_ii[ia] = dist(p, atom_pos[ia]);
        }
        let mut j = 0.0_f64;
        if d.use_u {
            for (k, d_ik_k) in d_ik[..n_e].iter().enumerate() {
                if k == g_idx {
                    continue;
                }
                let same = (k < n_up) == is_up;
                let a = if same { &a_par } else { &a_anti };
                j += cutoff(*d_ik_k, d.l_u, d.c) * poly(*d_ik_k, a);
            }
        }
        if d.use_chi {
            for (ia, d_ii_ia) in d_ii[..n_atom].iter().enumerate() {
                let sp = d.species_index[ia];
                j += cutoff(*d_ii_ia, d.l_chi[sp], d.c)
                    * poly(*d_ii_ia, d.beta.row(sp).as_slice().unwrap());
            }
        }
        if d.use_f {
            for &(k, ia, sp, r_ki, idx_first) in &fpass {
                j += if idx_first {
                    f_value(d, sp, d_ii[ia], r_ki, d_ik[k])
                } else {
                    f_value(d, sp, r_ki, d_ii[ia], d_ik[k])
                };
            }
        }
        j
    };
    let j_old = j_at(r_old, &mut d_ik, &mut d_ii);
    for v in 0..mesh.nrows() {
        let j_new = j_at(
            [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]],
            &mut d_ik,
            &mut d_ii,
        );
        out[v] = (j_new - j_old).exp();
    }
}

/// cutoff(r)=max(L-r,0)^C and its 1st/2nd derivs (`_dtn_cutoff_with_derivs`).
#[inline]
pub(crate) fn cutoff_derivs(r: f64, l: f64, c: i32) -> (f64, f64, f64) {
    let t = l - r;
    if t > 0.0 {
        let cf = t.powi(c);
        let cp = -(c as f64) * t.powi(c - 1);
        let cpp = (c as f64) * ((c - 1) as f64) * t.powi(c - 2);
        (cf, cp, cpp)
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Cutoff `(l-r)^c` value + 1st/2nd/3rd derivatives w.r.t. `r` (for the f-channel
/// kinetic STRESS, which strain-differentiates the 2nd-order kinetic terms).
pub(crate) fn cutoff_derivs3(r: f64, l: f64, c: i32) -> (f64, f64, f64, f64) {
    let t = l - r;
    if t > 0.0 {
        let cc = c as f64;
        let cf = t.powi(c);
        let cp = -cc * t.powi(c - 1);
        let cpp = cc * (cc - 1.0) * t.powi(c - 2);
        let cppp = -cc * (cc - 1.0) * (cc - 2.0) * t.powi(c - 3);
        (cf, cp, cpp, cppp)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

/// All 1st/2nd/3rd-order partials of the f-channel kernel `f = A(x)*B(y)*G(x,y,z)`
/// (x=r_iI, y=r_jI, z=r_ij; A,B = cutoff; G = Sumgamma x^l y^m z^n) needed by the
/// f-channel kinetic STRESS. Ordering of the returned tuple is documented at the
/// call site (`FDerivs3` fields).
#[allow(clippy::too_many_arguments)]
pub(crate) fn f_atom_derivs3(
    d: &JastrowDtnData,
    sp: usize,
    x: f64,
    y: f64,
    z: f64,
    l: f64,
) -> FDerivs3 {
    let (a, ap, app, appp) = cutoff_derivs3(x, l, d.c);
    let (b, bp, bpp, bppp) = cutoff_derivs3(y, l, d.c);
    // power tables, per power k: [r^k, k r^{k-1}, k(k-1) r^{k-2}, k(k-1)(k-2) r^{k-3}]
    let ptab = |r: f64, kmax: usize| -> Vec<[f64; 4]> {
        let mut p = vec![[0.0; 4]; kmax + 1];
        let (mut rk, mut rkm1, mut rkm2, mut rkm3) = (1.0_f64, 0.0, 0.0, 0.0);
        for (k, pk) in p.iter_mut().enumerate() {
            let kf = k as f64;
            *pk = [
                rk,
                kf * rkm1,
                kf * (kf - 1.0) * rkm2,
                kf * (kf - 1.0) * (kf - 2.0) * rkm3,
            ];
            rkm3 = rkm2;
            rkm2 = rkm1;
            rkm1 = rk;
            rk *= r;
        }
        p
    };
    let px = ptab(x, d.n_en);
    let py = ptab(y, d.n_en);
    let pz = ptab(z, d.n_ee);
    let stride_n = d.n_ee + 1;
    let stride_m = (d.n_en + 1) * stride_n;
    let stride_l = (d.n_en + 1) * stride_m;
    let base = sp * stride_l;
    // G partials: gp[i][j][k] = d^{i+j+k} G / dx^i dy^j dz^k, i,jin0..3 (x/y), kin0..3.
    let mut g = [[[0.0_f64; 4]; 4]; 4];
    for (li, px_l) in px.iter().enumerate() {
        for (mi, py_m) in py.iter().enumerate() {
            let off = base + li * stride_m + mi * stride_n;
            for (ni, pz_n) in pz.iter().enumerate() {
                let gam = d.gamma[off + ni];
                if gam == 0.0 {
                    continue;
                }
                for i in 0..4 {
                    let pxi = px_l[i];
                    if pxi == 0.0 {
                        continue;
                    }
                    for j in 0..4 {
                        let pyj = py_m[j];
                        if pyj == 0.0 {
                            continue;
                        }
                        let c = gam * pxi * pyj;
                        for k in 0..4 {
                            g[i][j][k] += c * pz_n[k];
                        }
                    }
                }
            }
        }
    }
    let gp = |i: usize, j: usize, k: usize| g[i][j][k];
    // f = A*B*G; partials via product rule.
    FDerivs3 {
        fx: ap * b * gp(0, 0, 0) + a * b * gp(1, 0, 0),
        fy: a * bp * gp(0, 0, 0) + a * b * gp(0, 1, 0),
        fz: a * b * gp(0, 0, 1),
        fxx: app * b * gp(0, 0, 0) + 2.0 * ap * b * gp(1, 0, 0) + a * b * gp(2, 0, 0),
        fyy: a * bpp * gp(0, 0, 0) + 2.0 * a * bp * gp(0, 1, 0) + a * b * gp(0, 2, 0),
        fzz: a * b * gp(0, 0, 2),
        fxy: ap * bp * gp(0, 0, 0)
            + ap * b * gp(0, 1, 0)
            + a * bp * gp(1, 0, 0)
            + a * b * gp(1, 1, 0),
        fxz: ap * b * gp(0, 0, 1) + a * b * gp(1, 0, 1),
        fyz: a * bp * gp(0, 0, 1) + a * b * gp(0, 1, 1),
        fxxx: appp * b * gp(0, 0, 0)
            + 3.0 * app * b * gp(1, 0, 0)
            + 3.0 * ap * b * gp(2, 0, 0)
            + a * b * gp(3, 0, 0),
        fxxy: app * bp * gp(0, 0, 0)
            + app * b * gp(0, 1, 0)
            + 2.0 * ap * bp * gp(1, 0, 0)
            + 2.0 * ap * b * gp(1, 1, 0)
            + a * bp * gp(2, 0, 0)
            + a * b * gp(2, 1, 0),
        fxxz: app * b * gp(0, 0, 1) + 2.0 * ap * b * gp(1, 0, 1) + a * b * gp(2, 0, 1),
        fxyy: ap * bpp * gp(0, 0, 0)
            + 2.0 * ap * bp * gp(0, 1, 0)
            + ap * b * gp(0, 2, 0)
            + a * bpp * gp(1, 0, 0)
            + 2.0 * a * bp * gp(1, 1, 0)
            + a * b * gp(1, 2, 0),
        fxyz: ap * bp * gp(0, 0, 1)
            + ap * b * gp(0, 1, 1)
            + a * bp * gp(1, 0, 1)
            + a * b * gp(1, 1, 1),
        fxzz: ap * b * gp(0, 0, 2) + a * b * gp(1, 0, 2),
        fyyy: a * bppp * gp(0, 0, 0)
            + 3.0 * a * bpp * gp(0, 1, 0)
            + 3.0 * a * bp * gp(0, 2, 0)
            + a * b * gp(0, 3, 0),
        fyyz: a * bpp * gp(0, 0, 1) + 2.0 * a * bp * gp(0, 1, 1) + a * b * gp(0, 2, 1),
        fyzz: a * bp * gp(0, 0, 2) + a * b * gp(0, 1, 2),
        fzzz: a * b * gp(0, 0, 3),
    }
}

/// 1st/2nd/3rd-order partials of the f kernel w.r.t. `(x=r_iI, y=r_jI, z=r_ij)`.
pub(crate) struct FDerivs3 {
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
    pub fxx: f64,
    pub fyy: f64,
    pub fzz: f64,
    pub fxy: f64,
    pub fxz: f64,
    pub fyz: f64,
    pub fxxx: f64,
    pub fxxy: f64,
    pub fxxz: f64,
    pub fxyy: f64,
    pub fxyz: f64,
    pub fxzz: f64,
    pub fyyy: f64,
    pub fyyz: f64,
    pub fyzz: f64,
    pub fzzz: f64,
}

/// P(r)=Sum coeffs[k] r^k and its 1st/2nd derivs.
#[inline]
fn poly_derivs(r: f64, coeffs: &[f64]) -> (f64, f64, f64) {
    let (mut p, mut pp, mut ppp) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut rk, mut rkm1, mut rkm2) = (1.0_f64, 0.0_f64, 0.0_f64); // r^k, r^{k-1}, r^{k-2}
    for (k, &a) in coeffs.iter().enumerate() {
        let kf = k as f64;
        p += a * rk;
        pp += a * kf * rkm1;
        ppp += a * kf * (kf - 1.0) * rkm2;
        // advance powers
        rkm2 = rkm1;
        rkm1 = rk;
        rk *= r;
    }
    (p, pp, ppp)
}

/// Radial kernel f=cutoff*P and (f, f', f'') by product rule (u/chi channel).
#[inline]
pub(crate) fn radial_kernel_derivs(r: f64, coeffs: &[f64], l: f64, c: i32) -> (f64, f64, f64) {
    let (cf, cp, cpp) = cutoff_derivs(r, l, c);
    let (p, pp, ppp) = poly_derivs(r, coeffs);
    let f = cf * p;
    let fp = cp * p + cf * pp;
    let fpp = cpp * p + 2.0 * cp * pp + cf * ppp;
    (f, fp, fpp)
}

/// f-channel atom kernel value+grad+hess for arbitrary coefficients and
/// independent cutoff powers on the two electron--atom legs.
#[allow(clippy::too_many_arguments)]
fn f_atom_value_grad_hess_raw(
    d: &JastrowDtnData,
    gamma: &[f64],
    sp: usize,
    r_ii: f64,
    r_ji: f64,
    r_ij: f64,
    l_i: f64,
    c_x: i32,
    c_y: i32,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let (gx, gpx, gppx) = cutoff_derivs(r_ii, l_i, c_x);
    let (gy, gpy, gppy) = cutoff_derivs(r_ji, l_i, c_y);
    // power tables [r^k, k r^{k-1}, k(k-1) r^{k-2}]
    let ptab = |r: f64, kmax: usize| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut p0, mut p1, mut p2) = (
            vec![0.0; kmax + 1],
            vec![0.0; kmax + 1],
            vec![0.0; kmax + 1],
        );
        let (mut rk, mut rkm1, mut rkm2) = (1.0_f64, 0.0_f64, 0.0_f64);
        for k in 0..=kmax {
            let kf = k as f64;
            p0[k] = rk;
            p1[k] = kf * rkm1;
            p2[k] = kf * (kf - 1.0) * rkm2;
            rkm2 = rkm1;
            rkm1 = rk;
            rk *= r;
        }
        (p0, p1, p2)
    };
    let (x0, x1, x2) = ptab(r_ii, d.n_en);
    let (y0, y1, y2) = ptab(r_ji, d.n_en);
    let (z0, z1, z2) = ptab(r_ij, d.n_ee);
    let stride_n = d.n_ee + 1;
    let stride_m = (d.n_en + 1) * stride_n;
    let stride_l = (d.n_en + 1) * stride_m;
    let base = sp * stride_l;
    // T[a,b,c] for the 9 needed (a,b,c). G partials.
    let (mut t000, mut t100, mut t200, mut t010, mut t020, mut t001, mut t002, mut t101, mut t011) =
        (0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for l in 0..=d.n_en {
        for m in 0..=d.n_en {
            let off = base + l * stride_m + m * stride_n;
            for n in 0..=d.n_ee {
                let g = gamma[off + n];
                if g == 0.0 {
                    continue;
                }
                t000 += g * x0[l] * y0[m] * z0[n];
                t100 += g * x1[l] * y0[m] * z0[n];
                t200 += g * x2[l] * y0[m] * z0[n];
                t010 += g * x0[l] * y1[m] * z0[n];
                t020 += g * x0[l] * y2[m] * z0[n];
                t001 += g * x0[l] * y0[m] * z1[n];
                t002 += g * x0[l] * y0[m] * z2[n];
                t101 += g * x1[l] * y0[m] * z1[n];
                t011 += g * x0[l] * y1[m] * z1[n];
            }
        }
    }
    let (gg, g_x, g_y, g_z) = (t000, t100, t010, t001);
    let (g_xx, g_yy, g_zz, g_xz, g_yz) = (t200, t020, t002, t101, t011);
    let gh = gx * gy;
    let f_x = gpx * gy * gg + gh * g_x;
    let f_y = gx * gpy * gg + gh * g_y;
    let f_z = gh * g_z;
    let f_xx = gppx * gy * gg + 2.0 * gpx * gy * g_x + gh * g_xx;
    let f_yy = gx * gppy * gg + 2.0 * gx * gpy * g_y + gh * g_yy;
    let f_zz = gh * g_zz;
    let f_xz = gpx * gy * g_z + gh * g_xz;
    let f_yz = gx * gpy * g_z + gh * g_yz;
    let f = gh * gg;
    (f, f_x, f_y, f_z, f_xx, f_yy, f_zz, f_xz, f_yz)
}

/// f-channel atom kernel grad+hess w.r.t. (x=r_iI, y=r_jI, z=r_ij).
/// Returns (F_x, F_y, F_z, H_xx, H_yy, H_zz, H_xz, H_yz) -- the entries the
/// chain rule consumes (`_dtn_f_atom_value_grad_hess`, F_xy dropped).
#[allow(clippy::too_many_arguments)]
pub(crate) fn f_atom_grad_hess(
    d: &JastrowDtnData,
    sp: usize,
    r_ii: f64,
    r_ji: f64,
    r_ij: f64,
    l_i: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let (_, fx, fy, fz, fxx, fyy, fzz, fxz, fyz) =
        f_atom_value_grad_hess_raw(d, &d.gamma, sp, r_ii, r_ji, r_ij, l_i, d.c, d.c);
    (fx, fy, fz, fxx, fyy, fzz, fxz, fyz)
}

/// Cutoff-policy f field: d/d ln(ell_V) of the atom contribution, including
/// the transported constrained coefficients and both comoving cutoff legs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn f_atom_policy_value_grad_hess(
    d: &JastrowDtnData,
    sp: usize,
    r_ii: f64,
    r_ji: f64,
    r_ij: f64,
    l_i: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let p0 = f_atom_value_grad_hess_raw(d, &d.gamma_policy, sp, r_ii, r_ji, r_ij, l_i, d.c, d.c);
    let px = f_atom_value_grad_hess_raw(d, &d.gamma, sp, r_ii, r_ji, r_ij, l_i, d.c - 1, d.c);
    let py = f_atom_value_grad_hess_raw(d, &d.gamma, sp, r_ii, r_ji, r_ij, l_i, d.c, d.c - 1);
    let scale = (d.c as f64) * l_i;
    (
        p0.0 + scale * (px.0 + py.0),
        p0.1 + scale * (px.1 + py.1),
        p0.2 + scale * (px.2 + py.2),
        p0.3 + scale * (px.3 + py.3),
        p0.4 + scale * (px.4 + py.4),
        p0.5 + scale * (px.5 + py.5),
        p0.6 + scale * (px.6 + py.6),
        p0.7 + scale * (px.7 + py.7),
        p0.8 + scale * (px.8 + py.8),
    )
}

#[inline]
fn cutoff_derivs_f32(r: f32, l: f32, c: i32) -> (f32, f32, f32) {
    let t = l - r;
    if t > 0.0 {
        let cc = c as f32;
        (
            t.powi(c),
            -cc * t.powi(c - 1),
            cc * (cc - 1.0) * t.powi(c - 2),
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

#[inline]
fn radial_kernel_derivs_f32(r: f32, coeffs: &[f64], l: f32, c: i32) -> (f32, f32, f32) {
    let (cf, cp, cpp) = cutoff_derivs_f32(r, l, c);
    let (mut p, mut pp, mut ppp) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut rk, mut rkm1, mut rkm2) = (1.0_f32, 0.0_f32, 0.0_f32);
    for (k, &coefficient) in coeffs.iter().enumerate() {
        let a = coefficient as f32;
        let kf = k as f32;
        p += a * rk;
        pp += a * kf * rkm1;
        ppp += a * kf * (kf - 1.0) * rkm2;
        rkm2 = rkm1;
        rkm1 = rk;
        rk *= r;
    }
    (cf * p, cp * p + cf * pp, cpp * p + 2.0 * cp * pp + cf * ppp)
}

#[allow(clippy::too_many_arguments)]
fn f_atom_grad_hess_f32(
    d: &JastrowDtnData,
    sp: usize,
    r_ii: f32,
    r_ji: f32,
    r_ij: f32,
    l_i: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let (gx, gpx, gppx) = cutoff_derivs_f32(r_ii, l_i, d.c);
    let (gy, gpy, gppy) = cutoff_derivs_f32(r_ji, l_i, d.c);
    let ptab = |r: f32, kmax: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut p0, mut p1, mut p2) = (
            vec![0.0_f32; kmax + 1],
            vec![0.0_f32; kmax + 1],
            vec![0.0_f32; kmax + 1],
        );
        let (mut rk, mut rkm1, mut rkm2) = (1.0_f32, 0.0_f32, 0.0_f32);
        for k in 0..=kmax {
            let kf = k as f32;
            p0[k] = rk;
            p1[k] = kf * rkm1;
            p2[k] = kf * (kf - 1.0) * rkm2;
            rkm2 = rkm1;
            rkm1 = rk;
            rk *= r;
        }
        (p0, p1, p2)
    };
    let (x0, x1, x2) = ptab(r_ii, d.n_en);
    let (y0, y1, y2) = ptab(r_ji, d.n_en);
    let (z0, z1, z2) = ptab(r_ij, d.n_ee);
    let stride_n = d.n_ee + 1;
    let stride_m = (d.n_en + 1) * stride_n;
    let stride_l = (d.n_en + 1) * stride_m;
    let base = sp * stride_l;
    let (mut t000, mut t100, mut t200, mut t010, mut t020, mut t001, mut t002, mut t101, mut t011) = (
        0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32,
    );
    for l in 0..=d.n_en {
        for m in 0..=d.n_en {
            let off = base + l * stride_m + m * stride_n;
            for n in 0..=d.n_ee {
                let gamma = d.gamma[off + n] as f32;
                if gamma == 0.0 {
                    continue;
                }
                t000 += gamma * x0[l] * y0[m] * z0[n];
                t100 += gamma * x1[l] * y0[m] * z0[n];
                t200 += gamma * x2[l] * y0[m] * z0[n];
                t010 += gamma * x0[l] * y1[m] * z0[n];
                t020 += gamma * x0[l] * y2[m] * z0[n];
                t001 += gamma * x0[l] * y0[m] * z1[n];
                t002 += gamma * x0[l] * y0[m] * z2[n];
                t101 += gamma * x1[l] * y0[m] * z1[n];
                t011 += gamma * x0[l] * y1[m] * z1[n];
            }
        }
    }
    let gh = gx * gy;
    (
        gpx * gy * t000 + gh * t100,
        gx * gpy * t000 + gh * t010,
        gh * t001,
        gppx * gy * t000 + 2.0 * gpx * gy * t100 + gh * t200,
        gx * gppy * t000 + 2.0 * gx * gpy * t010 + gh * t020,
        gh * t002,
        gpx * gy * t001 + gh * t101,
        gx * gpy * t001 + gh * t011,
    )
}

/// DTN grad/lap of `J` w.r.t. electron positions (analytic, 1:1 with
/// `compute_grads_and_laplacian_Jastrow_DTN`), in the `jastrow_grad_lap` zone.
/// Returns (grad_up (N_up,3), grad_dn (N_dn,3), lap_up (N_up,), lap_dn (N_dn,)).
pub fn compute_grads_and_laplacian_dtn(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Vec<f64>, Vec<f64>) {
    if d.precision_policy.is_f32("jastrow_grad_lap") {
        return compute_grads_and_laplacian_dtn_f32(d, r_up, r_dn);
    }
    compute_grads_and_laplacian_dtn_f64(d, r_up, r_dn)
}

/// fp64 body of [`compute_grads_and_laplacian_dtn`]: the full-precision
/// `jastrow_grad_lap` kernel and the seed of the `jastrow_grad_lap_carry` zone
/// ([`init_dtn_grad_lap_cache`]).
fn compute_grads_and_laplacian_dtn_f64(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Vec<f64>, Vec<f64>) {
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    // distance + (grad vec, |grad|^2, lap) w.r.t. the FIRST argument.
    let dgl = |a: [f64; 3], b: [f64; 3]| -> (f64, [f64; 3], f64, f64) {
        match &ci {
            Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci),
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                let r = (dx * dx + dy * dy + dz * dz)
                    .sqrt()
                    .max(crate::setting_consts::EPS_safe_distance);
                (r, [dx / r, dy / r, dz / r], 1.0, 2.0 / r)
            }
        }
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_tot = n_up + n_dn;
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
    let mut grad = vec![[0.0_f64; 3]; n_tot];
    let mut lap = vec![0.0_f64; n_tot];

    // u-term: each electron i sees every other; alpha[0] same-spin, alpha[1] anti-spin.
    if d.use_u {
        let a_par: Vec<f64> = d.alpha.row(0).to_vec();
        let a_anti: Vec<f64> = d.alpha.row(1).to_vec();
        for i in 0..n_tot {
            let ri = rget(i);
            let i_up = i < n_up;
            for k in 0..n_tot {
                if k == i {
                    continue;
                }
                let same = (k < n_up) == i_up;
                let a = if same { &a_par } else { &a_anti };
                let (r, gv, gn2, ld) = dgl(ri, rget(k));
                let (_f, fp, fpp) = radial_kernel_derivs(r, a, d.l_u, d.c);
                for c in 0..3 {
                    grad[i][c] += fp * gv[c];
                }
                lap[i] += fpp * gn2 + fp * ld;
            }
        }
    }

    // chi-term: each electron x all atoms.
    if d.use_chi {
        for i in 0..n_tot {
            let ri = rget(i);
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let (r, gv, gn2, ld) = dgl(ri, pos(ia));
                let (cf, _, _) = cutoff_derivs(r, d.l_chi[sp], d.c);
                if cf == 0.0 {
                    continue;
                }
                let (_f, fp, fpp) =
                    radial_kernel_derivs(r, d.beta.row(sp).as_slice().unwrap(), d.l_chi[sp], d.c);
                for c in 0..3 {
                    grad[i][c] += fp * gv[c];
                }
                lap[i] += fpp * gn2 + fp * ld;
            }
        }
    }

    // f-term: each electron pair (i<j) x all atoms; accumulate into BOTH i and j.
    if d.use_f {
        for i in 0..n_tot {
            let ri = rget(i);
            for j in (i + 1)..n_tot {
                let rj = rget(j);
                let (r_ij, g_ij_i, gn2_ij, lap_ij) = dgl(ri, rj);
                let g_ij_j = [-g_ij_i[0], -g_ij_i[1], -g_ij_i[2]];
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let l_i = d.l_f[sp];
                    let p = pos(ia);
                    let (r_ii, g_ii, gn2_ii, lap_ii) = dgl(ri, p);
                    if l_i - r_ii <= 0.0 {
                        continue;
                    }
                    let (r_ji, g_ji, gn2_ji, lap_ji) = dgl(rj, p);
                    if l_i - r_ji <= 0.0 {
                        continue;
                    }
                    let (f_x, f_y, f_z, h_xx, h_yy, h_zz, h_xz, h_yz) =
                        f_atom_grad_hess(d, sp, r_ii, r_ji, r_ij, l_i);
                    // electron i
                    let dot_ii_ij = g_ii[0] * g_ij_i[0] + g_ii[1] * g_ij_i[1] + g_ii[2] * g_ij_i[2];
                    for c in 0..3 {
                        grad[i][c] += f_x * g_ii[c] + f_z * g_ij_i[c];
                    }
                    lap[i] += h_xx * gn2_ii
                        + 2.0 * h_xz * dot_ii_ij
                        + h_zz * gn2_ij
                        + f_x * lap_ii
                        + f_z * lap_ij;
                    // electron j
                    let dot_ji_ijj =
                        g_ji[0] * g_ij_j[0] + g_ji[1] * g_ij_j[1] + g_ji[2] * g_ij_j[2];
                    for c in 0..3 {
                        grad[j][c] += f_y * g_ji[c] + f_z * g_ij_j[c];
                    }
                    lap[j] += h_yy * gn2_ji
                        + 2.0 * h_yz * dot_ji_ijj
                        + h_zz * gn2_ij
                        + f_y * lap_ji
                        + f_z * lap_ij;
                }
            }
        }
    }

    let mut grad_up = Array2::<f64>::zeros((n_up, 3));
    let mut grad_dn = Array2::<f64>::zeros((n_dn, 3));
    let mut lap_up = vec![0.0_f64; n_up];
    let mut lap_dn = vec![0.0_f64; n_dn];
    for i in 0..n_up {
        for c in 0..3 {
            grad_up[[i, c]] = grad[i][c];
        }
        lap_up[i] = lap[i];
    }
    for j in 0..n_dn {
        for c in 0..3 {
            grad_dn[[j, c]] = grad[n_up + j][c];
        }
        lap_dn[j] = lap[n_up + j];
    }
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Analytic DTN electron gradient/Laplacian evaluated entirely in the
/// `jastrow_grad_lap` f32 zone.  No finite differences are used.
fn compute_grads_and_laplacian_dtn_f32(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Vec<f64>, Vec<f64>) {
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dgl = |a: [f64; 3], b: [f64; 3]| -> (f32, [f32; 3], f32, f32) {
        let (r, g, gn2, lap) = dtn_dist_grad_lap_f32(a, b, &ci);
        (
            r as f32,
            [g[0] as f32, g[1] as f32, g[2] as f32],
            gn2 as f32,
            lap as f32,
        )
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_tot = n_up + n_dn;
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let n_atom = d.positions.nrows();
    let mut grad = vec![[0.0_f32; 3]; n_tot];
    let mut lap = vec![0.0_f32; n_tot];

    if d.use_u {
        for i in 0..n_tot {
            let i_up = i < n_up;
            for k in 0..n_tot {
                if k == i {
                    continue;
                }
                let coefficients = d.alpha.row(if (k < n_up) == i_up { 0 } else { 1 });
                let coefficients = coefficients.as_slice().unwrap();
                let (r, gv, gn2, ld) = dgl(rget(i), rget(k));
                let (_, fp, fpp) = radial_kernel_derivs_f32(r, coefficients, d.l_u as f32, d.c);
                for c in 0..3 {
                    grad[i][c] += fp * gv[c];
                }
                lap[i] += fpp * gn2 + fp * ld;
            }
        }
    }

    if d.use_chi {
        for i in 0..n_tot {
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let (r, gv, gn2, ld) = dgl(rget(i), atom_pos(d, ia));
                let (cf, _, _) = cutoff_derivs_f32(r, d.l_chi[sp] as f32, d.c);
                if cf == 0.0 {
                    continue;
                }
                let (_, fp, fpp) = radial_kernel_derivs_f32(
                    r,
                    d.beta.row(sp).as_slice().unwrap(),
                    d.l_chi[sp] as f32,
                    d.c,
                );
                for c in 0..3 {
                    grad[i][c] += fp * gv[c];
                }
                lap[i] += fpp * gn2 + fp * ld;
            }
        }
    }

    if d.use_f {
        for i in 0..n_tot {
            for j in (i + 1)..n_tot {
                let (r_ij, g_ij_i, gn2_ij, lap_ij) = dgl(rget(i), rget(j));
                let g_ij_j = [-g_ij_i[0], -g_ij_i[1], -g_ij_i[2]];
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let l_i = d.l_f[sp] as f32;
                    let (r_ii, g_ii, gn2_ii, lap_ii) = dgl(rget(i), atom_pos(d, ia));
                    if l_i - r_ii <= 0.0 {
                        continue;
                    }
                    let (r_ji, g_ji, gn2_ji, lap_ji) = dgl(rget(j), atom_pos(d, ia));
                    if l_i - r_ji <= 0.0 {
                        continue;
                    }
                    let (f_x, f_y, f_z, h_xx, h_yy, h_zz, h_xz, h_yz) =
                        f_atom_grad_hess_f32(d, sp, r_ii, r_ji, r_ij, l_i);
                    let dot_ii_ij = g_ii[0] * g_ij_i[0] + g_ii[1] * g_ij_i[1] + g_ii[2] * g_ij_i[2];
                    for c in 0..3 {
                        grad[i][c] += f_x * g_ii[c] + f_z * g_ij_i[c];
                    }
                    lap[i] += h_xx * gn2_ii
                        + 2.0 * h_xz * dot_ii_ij
                        + h_zz * gn2_ij
                        + f_x * lap_ii
                        + f_z * lap_ij;

                    let dot_ji_ijj =
                        g_ji[0] * g_ij_j[0] + g_ji[1] * g_ij_j[1] + g_ji[2] * g_ij_j[2];
                    for c in 0..3 {
                        grad[j][c] += f_y * g_ji[c] + f_z * g_ij_j[c];
                    }
                    lap[j] += h_yy * gn2_ji
                        + 2.0 * h_yz * dot_ji_ijj
                        + h_zz * gn2_ij
                        + f_y * lap_ji
                        + f_z * lap_ij;
                }
            }
        }
    }

    let mut grad_up = Array2::<f64>::zeros((n_up, 3));
    let mut grad_dn = Array2::<f64>::zeros((n_dn, 3));
    let mut lap_up = vec![0.0_f64; n_up];
    let mut lap_dn = vec![0.0_f64; n_dn];
    for i in 0..n_up {
        for c in 0..3 {
            grad_up[[i, c]] = grad[i][c] as f64;
        }
        lap_up[i] = lap[i] as f64;
    }
    for j in 0..n_dn {
        for c in 0..3 {
            grad_dn[[j, c]] = grad[n_up + j][c] as f64;
        }
        lap_dn[j] = lap[n_up + j] as f64;
    }
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Every gradient term of [`compute_grads_and_laplacian_dtn_f32`] at one base configuration,
/// cached per row in the full kernel's accumulation order, so that a one-electron move can be
/// re-summed bit-identically without re-evaluating the terms it does not touch.
///
/// Row `i` (all-electron index, up electrons first) of the full kernel accumulates, in this
/// order: the u terms of its partners `k != i` ascending, the chi terms of the atoms ascending
/// (only where the chi cutoff is non-zero), then the f terms of its partners `p != i` ascending,
/// atoms ascending within each partner (only where both e-I distances are inside the f cutoff).
/// The partner order of the f terms is that of the full kernel's `(i < j)` pair loop: pairs
/// `(p, i)` with `p < i` come before pairs `(i, p)` with `p > i`. Each term is evaluated exactly
/// as the full kernel evaluates it (same operand order inside each pair), and a row is the
/// sequential fp32 sum of its present terms, so every row equals the full kernel's row bitwise.
///
/// A move of electron `g` changes only the terms with partner `g` in every other row, and every
/// term of row `g`. [`Self::move_delta`] re-evaluates those `O(Ne * Natom)` terms and re-sums
/// each row in order, giving `grad(moved) - grad(base)` bit-identical to the full-kernel
/// difference of [`dtn_drift_move_delta`] under an fp32 `jastrow_grad_lap` zone, at
/// `O(Ne * Natom)` term evaluations instead of two `O(Ne^2 * Natom)` kernels per move.
pub(crate) struct DtnGradTermsF32 {
    n_up: usize,
    n_tot: usize,
    n_atom: usize,
    /// Base electron positions (all-electron index).
    pos: Vec<[f64; 3]>,
    /// `n_tot` rows of `n_tot` u slots, `n_atom` chi slots and `n_tot * n_atom` f slots.
    terms: Vec<Option<[f32; 3]>>,
    /// Base gradient rows (the ordered sums of `terms`).
    base: Vec<[f32; 3]>,
}

impl DtnGradTermsF32 {
    fn slots(&self) -> usize {
        self.n_tot + self.n_atom + self.n_tot * self.n_atom
    }

    fn u_slot(&self, i: usize, k: usize) -> usize {
        i * self.slots() + k
    }

    fn chi_slot(&self, i: usize, ia: usize) -> usize {
        i * self.slots() + self.n_tot + ia
    }

    fn f_slot(&self, i: usize, p: usize, ia: usize) -> usize {
        i * self.slots() + self.n_tot + self.n_atom + p * self.n_atom + ia
    }

    /// u term of row `i` with partner `k` (the full kernel's `dgl(r_i, r_k)` operand order).
    fn u_term(
        d: &JastrowDtnData,
        ci: &Option<CellInfo>,
        ri: [f64; 3],
        rk: [f64; 3],
        same_spin: bool,
    ) -> [f32; 3] {
        let coefficients = d.alpha.row(if same_spin { 0 } else { 1 });
        let (r, gv, _, _) = dtn_dist_grad_lap_f32(ri, rk, ci);
        let (r, gv) = (r as f32, [gv[0] as f32, gv[1] as f32, gv[2] as f32]);
        let (_, fp, _) =
            radial_kernel_derivs_f32(r, coefficients.as_slice().unwrap(), d.l_u as f32, d.c);
        [fp * gv[0], fp * gv[1], fp * gv[2]]
    }

    /// chi term of row `i` with atom `ia`; `None` where the chi cutoff vanishes.
    fn chi_term(
        d: &JastrowDtnData,
        ci: &Option<CellInfo>,
        ri: [f64; 3],
        ia: usize,
    ) -> Option<[f32; 3]> {
        let sp = d.species_index[ia];
        let (r, gv, _, _) = dtn_dist_grad_lap_f32(ri, atom_pos(d, ia), ci);
        let (r, gv) = (r as f32, [gv[0] as f32, gv[1] as f32, gv[2] as f32]);
        let (cf, _, _) = cutoff_derivs_f32(r, d.l_chi[sp] as f32, d.c);
        if cf == 0.0 {
            return None;
        }
        let (_, fp, _) = radial_kernel_derivs_f32(
            r,
            d.beta.row(sp).as_slice().unwrap(),
            d.l_chi[sp] as f32,
            d.c,
        );
        Some([fp * gv[0], fp * gv[1], fp * gv[2]])
    }

    /// f terms of the pair `(a, b)`, `a < b` in the full kernel's loop, with atom `ia`:
    /// `(term of row a, term of row b)`, `None` where either e-I distance leaves the f cutoff.
    fn f_terms(
        d: &JastrowDtnData,
        ci: &Option<CellInfo>,
        ra: [f64; 3],
        rb: [f64; 3],
        r_ab: f32,
        g_ab_a: [f32; 3],
        ia: usize,
    ) -> Option<([f32; 3], [f32; 3])> {
        let dgl = |x: [f64; 3], y: [f64; 3]| -> (f32, [f32; 3]) {
            let (r, g, _, _) = dtn_dist_grad_lap_f32(x, y, ci);
            (r as f32, [g[0] as f32, g[1] as f32, g[2] as f32])
        };
        let g_ab_b = [-g_ab_a[0], -g_ab_a[1], -g_ab_a[2]];
        let sp = d.species_index[ia];
        let l_i = d.l_f[sp] as f32;
        let (r_ai, g_ai) = dgl(ra, atom_pos(d, ia));
        if l_i - r_ai <= 0.0 {
            return None;
        }
        let (r_bi, g_bi) = dgl(rb, atom_pos(d, ia));
        if l_i - r_bi <= 0.0 {
            return None;
        }
        let (f_x, f_y, f_z, _, _, _, _, _) = f_atom_grad_hess_f32(d, sp, r_ai, r_bi, r_ab, l_i);
        let ta = [
            f_x * g_ai[0] + f_z * g_ab_a[0],
            f_x * g_ai[1] + f_z * g_ab_a[1],
            f_x * g_ai[2] + f_z * g_ab_a[2],
        ];
        let tb = [
            f_y * g_bi[0] + f_z * g_ab_b[0],
            f_y * g_bi[1] + f_z * g_ab_b[1],
            f_y * g_bi[2] + f_z * g_ab_b[2],
        ];
        Some((ta, tb))
    }

    /// The pair `(a, b)` operands `dgl(r_a, r_b)` of the full kernel's f loop.
    fn pair_geom(ci: &Option<CellInfo>, ra: [f64; 3], rb: [f64; 3]) -> (f32, [f32; 3]) {
        let (r, g, _, _) = dtn_dist_grad_lap_f32(ra, rb, ci);
        (r as f32, [g[0] as f32, g[1] as f32, g[2] as f32])
    }

    pub(crate) fn new(d: &JastrowDtnData, r_up: &ArrayView2<f64>, r_dn: &ArrayView2<f64>) -> Self {
        let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
        let n_up = r_up.nrows();
        let n_tot = n_up + r_dn.nrows();
        let n_atom = d.positions.nrows();
        let pos: Vec<[f64; 3]> = (0..n_tot)
            .map(|i| {
                if i < n_up {
                    [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
                } else {
                    let j = i - n_up;
                    [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
                }
            })
            .collect();
        let mut t = DtnGradTermsF32 {
            n_up,
            n_tot,
            n_atom,
            pos,
            terms: Vec::new(),
            base: Vec::new(),
        };
        t.terms = vec![None; n_tot * t.slots()];
        for i in 0..n_tot {
            if d.use_u {
                for k in 0..n_tot {
                    if k != i {
                        let s = t.u_slot(i, k);
                        t.terms[s] = Some(Self::u_term(
                            d,
                            &ci,
                            t.pos[i],
                            t.pos[k],
                            (k < n_up) == (i < n_up),
                        ));
                    }
                }
            }
            if d.use_chi {
                for ia in 0..n_atom {
                    let s = t.chi_slot(i, ia);
                    t.terms[s] = Self::chi_term(d, &ci, t.pos[i], ia);
                }
            }
        }
        if d.use_f {
            for a in 0..n_tot {
                for b in (a + 1)..n_tot {
                    let (r_ab, g_ab_a) = Self::pair_geom(&ci, t.pos[a], t.pos[b]);
                    for ia in 0..n_atom {
                        if let Some((ta, tb)) =
                            Self::f_terms(d, &ci, t.pos[a], t.pos[b], r_ab, g_ab_a, ia)
                        {
                            let (sa, sb) = (t.f_slot(a, b, ia), t.f_slot(b, a, ia));
                            t.terms[sa] = Some(ta);
                            t.terms[sb] = Some(tb);
                        }
                    }
                }
            }
        }
        t.base = (0..n_tot)
            .map(|i| {
                let row = &t.terms[i * t.slots()..(i + 1) * t.slots()];
                let mut acc = [0.0_f32; 3];
                for term in row.iter().flatten() {
                    for c in 0..3 {
                        acc[c] += term[c];
                    }
                }
                acc
            })
            .collect();
        t
    }

    /// `grad(moved) - grad(base)` of the full fp32 kernel for the move of electron `li`
    /// (spin `is_up`) to `new_pos`, as `(up rows, dn rows)`; bit-identical to the fp32 branch
    /// of [`dtn_drift_move_delta`]. `r_up` / `r_dn` must be the cache's base configuration.
    pub(crate) fn move_delta(
        &self,
        d: &JastrowDtnData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        li: usize,
        new_pos: [f64; 3],
    ) -> (Array2<f64>, Array2<f64>) {
        let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
        let (n_tot, n_atom, n_up) = (self.n_tot, self.n_atom, self.n_up);
        debug_assert!(
            r_up.nrows() == n_up
                && r_dn.nrows() == n_tot - n_up
                && (0..n_tot).all(|i| {
                    let r = if i < n_up {
                        r_up.row(i)
                    } else {
                        r_dn.row(i - n_up)
                    };
                    [r[0], r[1], r[2]] == self.pos[i]
                }),
            "DtnGradTermsF32: move off a configuration other than the cached base"
        );
        let g = if is_up { li } else { n_up + li };
        let rpos = |i: usize| if i == g { new_pos } else { self.pos[i] };
        // Re-evaluated terms with partner g: per other row i its u slot and n_atom f slots;
        // row g in full (its own u, chi and f slots).
        let mut u_new = vec![[0.0_f32; 3]; n_tot]; // u term of row i (!= g) with partner g
        let mut f_new = vec![None; n_tot * n_atom]; // f term of row i (!= g) with partner g
        let mut row_g = vec![None; self.slots()];
        for i in 0..n_tot {
            if i == g {
                continue;
            }
            if d.use_u {
                let same = (i < n_up) == (g < n_up);
                u_new[i] = Self::u_term(d, &ci, rpos(i), new_pos, same);
                row_g[i] = Some(Self::u_term(d, &ci, new_pos, rpos(i), same));
            }
            if d.use_f {
                let (a, b) = if i < g { (i, g) } else { (g, i) };
                let (r_ab, g_ab_a) = Self::pair_geom(&ci, rpos(a), rpos(b));
                for ia in 0..n_atom {
                    if let Some((ta, tb)) =
                        Self::f_terms(d, &ci, rpos(a), rpos(b), r_ab, g_ab_a, ia)
                    {
                        let (t_i, t_g) = if i < g { (ta, tb) } else { (tb, ta) };
                        f_new[i * n_atom + ia] = Some(t_i);
                        row_g[n_tot + n_atom + i * n_atom + ia] = Some(t_g);
                    }
                }
            }
        }
        if d.use_chi {
            for ia in 0..n_atom {
                row_g[n_tot + ia] = Self::chi_term(d, &ci, new_pos, ia);
            }
        }
        let mut du = Array2::<f64>::zeros((n_up, 3));
        let mut ddn = Array2::<f64>::zeros((n_tot - n_up, 3));
        for i in 0..n_tot {
            let mut acc = [0.0_f32; 3];
            if i == g {
                for term in row_g.iter().flatten() {
                    for c in 0..3 {
                        acc[c] += term[c];
                    }
                }
            } else {
                let row = &self.terms[i * self.slots()..(i + 1) * self.slots()];
                for (s, slot) in row.iter().enumerate() {
                    let term = if d.use_u && s == g {
                        Some(u_new[i])
                    } else if s >= n_tot + n_atom && (s - n_tot - n_atom) / n_atom == g {
                        f_new[i * n_atom + (s - n_tot - n_atom) % n_atom]
                    } else {
                        *slot
                    };
                    if let Some(term) = term {
                        for c in 0..3 {
                            acc[c] += term[c];
                        }
                    }
                }
            }
            let (dst, r) = if i < n_up {
                (&mut du, i)
            } else {
                (&mut ddn, i - n_up)
            };
            for c in 0..3 {
                dst[[r, c]] = acc[c] as f64 - self.base[i][c] as f64;
            }
        }
        (du, ddn)
    }
}

/// Row `g` (all-electron index, up electrons first) of the gradient of
/// [`compute_grads_and_laplacian_dtn_f32`], with electron `g` at `pos_g` and every other
/// electron at its `r_up` / `r_dn` row. It evaluates exactly the terms that the full fp32
/// kernel accumulates into `grad[g]`, with the same operands and in the same order (u partners
/// `k` ascending, chi atoms ascending, then the f pairs `(i, g)` for `i < g` followed by `(g, j)`
/// for `j > g`, atoms ascending within each pair), so the value is bit-identical to that row of
/// the full kernel at `O(Ne * Natom)` instead of `O(Ne^2 * Natom)`. No Laplacian.
pub(crate) fn dtn_grad_row_f32(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g: usize,
    pos_g: [f64; 3],
) -> [f64; 3] {
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dgl = |a: [f64; 3], b: [f64; 3]| -> (f32, [f32; 3]) {
        let (r, gv, _, _) = dtn_dist_grad_lap_f32(a, b, &ci);
        (r as f32, [gv[0] as f32, gv[1] as f32, gv[2] as f32])
    };
    let n_up = r_up.nrows();
    let n_tot = n_up + r_dn.nrows();
    let rget = |i: usize| -> [f64; 3] {
        if i == g {
            pos_g
        } else if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };
    let n_atom = d.positions.nrows();
    let mut acc = [0.0_f32; 3];

    if d.use_u {
        let g_up = g < n_up;
        for k in 0..n_tot {
            if k == g {
                continue;
            }
            let coefficients = d.alpha.row(if (k < n_up) == g_up { 0 } else { 1 });
            let coefficients = coefficients.as_slice().unwrap();
            let (r, gv) = dgl(rget(g), rget(k));
            let (_, fp, _) = radial_kernel_derivs_f32(r, coefficients, d.l_u as f32, d.c);
            for c in 0..3 {
                acc[c] += fp * gv[c];
            }
        }
    }

    if d.use_chi {
        for ia in 0..n_atom {
            let sp = d.species_index[ia];
            let (r, gv) = dgl(rget(g), atom_pos(d, ia));
            let (cf, _, _) = cutoff_derivs_f32(r, d.l_chi[sp] as f32, d.c);
            if cf == 0.0 {
                continue;
            }
            let (_, fp, _) = radial_kernel_derivs_f32(
                r,
                d.beta.row(sp).as_slice().unwrap(),
                d.l_chi[sp] as f32,
                d.c,
            );
            for c in 0..3 {
                acc[c] += fp * gv[c];
            }
        }
    }

    if d.use_f {
        // One f pair (i < j) touching g, as the full kernel's (i, j) iteration evaluates it.
        let pair = |i: usize, j: usize, acc: &mut [f32; 3]| {
            let (r_ij, g_ij_i) = dgl(rget(i), rget(j));
            let g_ij_j = [-g_ij_i[0], -g_ij_i[1], -g_ij_i[2]];
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let l_i = d.l_f[sp] as f32;
                let (r_ii, g_ii) = dgl(rget(i), atom_pos(d, ia));
                if l_i - r_ii <= 0.0 {
                    continue;
                }
                let (r_ji, g_ji) = dgl(rget(j), atom_pos(d, ia));
                if l_i - r_ji <= 0.0 {
                    continue;
                }
                let (f_x, f_y, f_z, _, _, _, _, _) =
                    f_atom_grad_hess_f32(d, sp, r_ii, r_ji, r_ij, l_i);
                if i == g {
                    for c in 0..3 {
                        acc[c] += f_x * g_ii[c] + f_z * g_ij_i[c];
                    }
                } else {
                    for c in 0..3 {
                        acc[c] += f_y * g_ji[c] + f_z * g_ij_j[c];
                    }
                }
            }
        };
        for i in 0..g {
            pair(i, g, &mut acc);
        }
        for j in (g + 1)..n_tot {
            pair(g, j, &mut acc);
        }
    }
    [acc[0] as f64, acc[1] as f64, acc[2] as f64]
}

/// DTN drift (`grad_r J`) DELTA for a single-electron hop of electron `li` (spin
/// `is_up`) to `new_pos`: `grad(moved) - grad(base)`, restricted to the terms that
/// touch the moved electron `g` (every other pair/leg is invariant and cancels). The
/// per-term formulas are byte-identical to [`compute_grads_and_laplacian_dtn`] (u/chi/f
/// grad parts), so `base + delta` reproduces the full moved-config drift to fp noise.
/// O(Ne+Natom) per hop (moved electron's partners/legs), NOT the full O(Ne^2+Ne*Natom).
pub fn dtn_drift_move_delta(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_pos: [f64; 3],
) -> (Array2<f64>, Array2<f64>) {
    if d.precision_policy.is_f32("jastrow_grad_lap") {
        // The analytic full-gradient leaf is the mixed-zone authority. Taking
        // the difference of two analytic gradients keeps all DTN drift
        // arithmetic in fp32 without duplicating the sizeable delta algebra.
        // This is not a finite-difference derivative.
        let (base_up, base_dn, _, _) = compute_grads_and_laplacian_dtn_f32(d, r_up, r_dn);
        let mut moved_up = r_up.to_owned();
        let mut moved_dn = r_dn.to_owned();
        if is_up {
            for c in 0..3 {
                moved_up[[li, c]] = new_pos[c];
            }
        } else {
            for c in 0..3 {
                moved_dn[[li, c]] = new_pos[c];
            }
        }
        let (new_up, new_dn, _, _) =
            compute_grads_and_laplacian_dtn_f32(d, &moved_up.view(), &moved_dn.view());
        return (new_up - base_up, new_dn - base_dn);
    }
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dgl = |a: [f64; 3], b: [f64; 3]| -> (f64, [f64; 3]) {
        match &ci {
            Some(ci) => {
                let (r, gv, _, _) = min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci);
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
    let n_dn = r_dn.nrows();
    let n_tot = n_up + n_dn;
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
    let g_idx = if is_up { li } else { n_up + li };
    let r_old = rget(g_idx);
    let a_par: Vec<f64> = d.alpha.row(0).to_vec();
    let a_anti: Vec<f64> = d.alpha.row(1).to_vec();
    let mut delta = vec![[0.0_f64; 3]; n_tot];
    // Accumulate the touching-`g` grad terms with `g` at `g_pos`, scaled by `sign`.
    let accumulate = |g_pos: [f64; 3], sign: f64, delta: &mut [[f64; 3]]| {
        // u-term: antisymmetric pair (g,k) -> grad[g] += fp*gv, grad[k] -= fp*gv.
        if d.use_u {
            for k in 0..n_tot {
                if k == g_idx {
                    continue;
                }
                let same = (k < n_up) == is_up;
                let a = if same { &a_par } else { &a_anti };
                let (r, gv) = dgl(g_pos, rget(k));
                let (_f, fp, _fpp) = radial_kernel_derivs(r, a, d.l_u, d.c);
                for c in 0..3 {
                    delta[g_idx][c] += sign * fp * gv[c];
                    delta[k][c] -= sign * fp * gv[c];
                }
            }
        }
        // chi-term: only the moved electron's e-N legs.
        if d.use_chi {
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let (r, gv) = dgl(g_pos, pos(ia));
                let (cf, _, _) = cutoff_derivs(r, d.l_chi[sp], d.c);
                if cf == 0.0 {
                    continue;
                }
                let (_f, fp, _fpp) =
                    radial_kernel_derivs(r, d.beta.row(sp).as_slice().unwrap(), d.l_chi[sp], d.c);
                for c in 0..3 {
                    delta[g_idx][c] += sign * fp * gv[c];
                }
            }
        }
        // f-term: pairs (g,k) x atoms, respecting the global i<j ordering (x=r_iI leg = a).
        if d.use_f {
            for k in 0..n_tot {
                if k == g_idx {
                    continue;
                }
                let (a_glob, b_glob, a_pos, b_pos) = if g_idx < k {
                    (g_idx, k, g_pos, rget(k))
                } else {
                    (k, g_idx, rget(k), g_pos)
                };
                let (r_ij, g_ij_a) = dgl(a_pos, b_pos);
                let g_ij_b = [-g_ij_a[0], -g_ij_a[1], -g_ij_a[2]];
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let l_i = d.l_f[sp];
                    let p = pos(ia);
                    let (r_ii, g_ii) = dgl(a_pos, p);
                    if l_i - r_ii <= 0.0 {
                        continue;
                    }
                    let (r_ji, g_ji) = dgl(b_pos, p);
                    if l_i - r_ji <= 0.0 {
                        continue;
                    }
                    let (f_x, f_y, f_z, ..) = f_atom_grad_hess(d, sp, r_ii, r_ji, r_ij, l_i);
                    for c in 0..3 {
                        delta[a_glob][c] += sign * (f_x * g_ii[c] + f_z * g_ij_a[c]);
                        delta[b_glob][c] += sign * (f_y * g_ji[c] + f_z * g_ij_b[c]);
                    }
                }
            }
        }
    };
    accumulate(new_pos, 1.0, &mut delta);
    accumulate(r_old, -1.0, &mut delta);

    let mut du = Array2::<f64>::zeros((n_up, 3));
    let mut dd = Array2::<f64>::zeros((n_dn, 3));
    for i in 0..n_up {
        for c in 0..3 {
            du[[i, c]] = delta[i][c];
        }
    }
    for j in 0..n_dn {
        for c in 0..3 {
            dd[[j, c]] = delta[n_up + j][c];
        }
    }
    (du, dd)
}

/// Precision zone of [`DtnGradLapCache`]: its seed, per-hop delta and
/// accumulator. fp64 in both modes (Python `_precision.py`); its native
/// capability is fp64 only (`native_precision_support_policy` does not list it).
const DTN_GRAD_LAP_CARRY_ZONE: &str = "jastrow_grad_lap_carry";

/// Projection-scoped DTN electron grad/Laplacian cache (real-valued -- DTN is a
/// pure-distance Jastrow, identical on the real and complex-twist chains). The
/// full O(Ne^2 x Natom) rebuild runs once per interval; each GFMC hop then
/// applies the O(Ne x Natom) moved-electron delta of
/// [`update_dtn_grad_lap_cache`]. Seed, delta and accumulator are in the
/// `jastrow_grad_lap_carry` zone, fp64 in both precision modes, so a fresh
/// state built at the same configuration agrees with the carried one to the
/// fp64 delta-chain floor. An f32 seed would carry its rounding unchanged over
/// the whole interval (#679). The coefficients are inflated in the carry zone too
/// (`JastrowDtnData::zone`), not in the `jastrow_grad_lap` dtype of the shared set,
/// which the carry does not consume (issue #781); the XLA carry
/// (`jastrow_factor._dtn_carry_coefficient_dtype`) inflates in the same precision.
pub struct DtnGradLapCache {
    pub(crate) gu: Array2<f64>,
    pub(crate) gd: Array2<f64>,
    pub(crate) lu: Vec<f64>,
    pub(crate) ld: Vec<f64>,
}

pub fn init_dtn_grad_lap_cache(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> DtnGradLapCache {
    // Seed in the carry zone. It has fp64-only native capability, so an f32
    // carry policy would be a desynchronised Python/Rust schema; checked once
    // per interval.
    assert!(
        !d.precision_policy.is_f32(DTN_GRAD_LAP_CARRY_ZONE),
        "the DTN grad/lap carry zone is fp64 only"
    );
    let d = d.zone(DTN_GRAD_LAP_CARRY_ZONE); // carry-zone coefficients (issue #781)
    let (gu, gd, lu, ld) = compute_grads_and_laplacian_dtn_f64(d, r_up, r_dn);
    DtnGradLapCache { gu, gd, lu, ld }
}

/// Snapshot the cached tables in the `compute_grads_and_laplacian_dtn` return
/// shape (grad_up, grad_dn, lap_up, lap_dn).
pub fn clone_dtn_grad_lap_cache(
    cache: &DtnGradLapCache,
) -> (Array2<f64>, Array2<f64>, Vec<f64>, Vec<f64>) {
    (
        cache.gu.clone(),
        cache.gd.clone(),
        cache.lu.clone(),
        cache.ld.clone(),
    )
}

/// Advance the DTN grad/lap cache after an accepted single-electron hop:
/// grad AND Laplacian deltas restricted to the terms touching the moved
/// electron (the full-derivative sibling of [`dtn_drift_move_delta`], which
/// carries only the drift). `r_up`/`r_dn` are the POST-move coordinates and
/// `old_pos` the moved electron's previous position (the GFMC advance
/// convention, mirroring `update_j12_grad_lap_cache_generic`). Per-term
/// formulas are byte-identical to [`compute_grads_and_laplacian_dtn`]'s
/// u/chi/f blocks; partner-side u/f Laplacian legs reuse the moved-side
/// metric derivatives (`gn2`/`lap` of the smooth/min-image distance are even
/// in the separation, exactly the symmetry the full kernel's double loop
/// evaluates from each side).
#[allow(clippy::too_many_arguments)]
pub fn update_dtn_grad_lap_cache(
    d: &JastrowDtnData,
    cache: &mut DtnGradLapCache,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    old_pos: [f64; 3],
) {
    // The deltas and the accumulation are in the fp64 carry zone in both
    // precision modes (see [`DtnGradLapCache`]), with carry-zone coefficients.
    debug_assert!(!d.precision_policy.is_f32(DTN_GRAD_LAP_CARRY_ZONE));
    let d = d.zone(DTN_GRAD_LAP_CARRY_ZONE);
    debug_assert!(
        cache.gu.nrows() == r_up.nrows()
            && cache.lu.len() == r_up.nrows()
            && cache.gd.nrows() == r_dn.nrows()
            && cache.ld.len() == r_dn.nrows(),
        "DTN grad/lap cache shape does not match the configuration"
    );
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    // distance + (grad vec, |grad|^2, lap) w.r.t. the FIRST argument -- the
    // exact dgl of `compute_grads_and_laplacian_dtn`.
    let dgl = |a: [f64; 3], b: [f64; 3]| -> (f64, [f64; 3], f64, f64) {
        match &ci {
            Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci),
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                let r = (dx * dx + dy * dy + dz * dz)
                    .sqrt()
                    .max(crate::setting_consts::EPS_safe_distance);
                (r, [dx / r, dy / r, dz / r], 1.0, 2.0 / r)
            }
        }
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_tot = n_up + n_dn;
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
    let g_idx = if is_up { li } else { n_up + li };
    let new_pos = rget(g_idx);
    let a_par: Vec<f64> = d.alpha.row(0).to_vec();
    let a_anti: Vec<f64> = d.alpha.row(1).to_vec();
    let mut dgrad = vec![[0.0_f64; 3]; n_tot];
    let mut dlap = vec![0.0_f64; n_tot];
    // Accumulate the touching-`g` grad+lap terms with `g` at `g_pos`, scaled by `sign`.
    let accumulate = |g_pos: [f64; 3], sign: f64, dgrad: &mut [[f64; 3]], dlap: &mut [f64]| {
        // u-term: pair (g,k) -> grad antisymmetric, lap symmetric (even metric).
        if d.use_u {
            for k in 0..n_tot {
                if k == g_idx {
                    continue;
                }
                let same = (k < n_up) == is_up;
                let a = if same { &a_par } else { &a_anti };
                let (r, gv, gn2, ld) = dgl(g_pos, rget(k));
                let (_f, fp, fpp) = radial_kernel_derivs(r, a, d.l_u, d.c);
                for c in 0..3 {
                    dgrad[g_idx][c] += sign * fp * gv[c];
                    dgrad[k][c] -= sign * fp * gv[c];
                }
                let lterm = fpp * gn2 + fp * ld;
                dlap[g_idx] += sign * lterm;
                dlap[k] += sign * lterm;
            }
        }
        // chi-term: only the moved electron's e-N legs.
        if d.use_chi {
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let (r, gv, gn2, ld) = dgl(g_pos, pos(ia));
                let (cf, _, _) = cutoff_derivs(r, d.l_chi[sp], d.c);
                if cf == 0.0 {
                    continue;
                }
                let (_f, fp, fpp) =
                    radial_kernel_derivs(r, d.beta.row(sp).as_slice().unwrap(), d.l_chi[sp], d.c);
                for c in 0..3 {
                    dgrad[g_idx][c] += sign * fp * gv[c];
                }
                dlap[g_idx] += sign * (fpp * gn2 + fp * ld);
            }
        }
        // f-term: pairs (g,k) x atoms, respecting the global i<j ordering
        // (x = r_iI leg belongs to the smaller global index `a`).
        if d.use_f {
            // The moved electron's e-N legs do not depend on the partner `k`: form
            // them once per position and keep only the atoms inside its f cutoff.
            // An atom outside it fails the `l_i - r <= 0` leg test for EVERY pair,
            // so it contributed nothing; the surviving (k, atom) terms are the same
            // operations on the same values in the same order (bit-identical).
            let mover_legs: Vec<_> = (0..n_atom)
                .filter_map(|ia| {
                    let leg = dgl(g_pos, pos(ia));
                    (d.l_f[d.species_index[ia]] - leg.0 > 0.0).then_some((ia, leg))
                })
                .collect();
            for k in 0..n_tot {
                if k == g_idx || mover_legs.is_empty() {
                    continue;
                }
                let (a_glob, b_glob, a_pos, b_pos) = if g_idx < k {
                    (g_idx, k, g_pos, rget(k))
                } else {
                    (k, g_idx, rget(k), g_pos)
                };
                let mut pair: Option<(f64, [f64; 3], f64, f64)> = None;
                for &(ia, mover_leg) in &mover_legs {
                    let sp = d.species_index[ia];
                    let l_i = d.l_f[sp];
                    let partner_leg = dgl(rget(k), pos(ia));
                    if l_i - partner_leg.0 <= 0.0 {
                        continue;
                    }
                    let ((r_ii, g_ii, gn2_ii, lap_ii), (r_ji, g_ji, gn2_ji, lap_ji)) = if g_idx < k
                    {
                        (mover_leg, partner_leg)
                    } else {
                        (partner_leg, mover_leg)
                    };
                    let (r_ij, g_ij_a, gn2_ij, lap_ij) =
                        *pair.get_or_insert_with(|| dgl(a_pos, b_pos));
                    let g_ij_b = [-g_ij_a[0], -g_ij_a[1], -g_ij_a[2]];
                    let (f_x, f_y, f_z, h_xx, h_yy, h_zz, h_xz, h_yz) =
                        f_atom_grad_hess(d, sp, r_ii, r_ji, r_ij, l_i);
                    // electron a (the i-leg holder)
                    let dot_ii_ij = g_ii[0] * g_ij_a[0] + g_ii[1] * g_ij_a[1] + g_ii[2] * g_ij_a[2];
                    for c in 0..3 {
                        dgrad[a_glob][c] += sign * (f_x * g_ii[c] + f_z * g_ij_a[c]);
                    }
                    dlap[a_glob] += sign
                        * (h_xx * gn2_ii
                            + 2.0 * h_xz * dot_ii_ij
                            + h_zz * gn2_ij
                            + f_x * lap_ii
                            + f_z * lap_ij);
                    // electron b (the j-leg holder)
                    let dot_ji_ijj =
                        g_ji[0] * g_ij_b[0] + g_ji[1] * g_ij_b[1] + g_ji[2] * g_ij_b[2];
                    for c in 0..3 {
                        dgrad[b_glob][c] += sign * (f_y * g_ji[c] + f_z * g_ij_b[c]);
                    }
                    dlap[b_glob] += sign
                        * (h_yy * gn2_ji
                            + 2.0 * h_yz * dot_ji_ijj
                            + h_zz * gn2_ij
                            + f_y * lap_ji
                            + f_z * lap_ij);
                }
            }
        }
    };
    accumulate(new_pos, 1.0, &mut dgrad, &mut dlap);
    accumulate(old_pos, -1.0, &mut dgrad, &mut dlap);
    for i in 0..n_up {
        for (c, &dg) in dgrad[i].iter().enumerate() {
            cache.gu[[i, c]] += dg;
        }
        cache.lu[i] += dlap[i];
    }
    for j in 0..n_dn {
        for (c, &dg) in dgrad[n_up + j].iter().enumerate() {
            cache.gd[[j, c]] += dg;
        }
        cache.ld[j] += dlap[n_up + j];
    }
}

/// DTN log-Jastrow value `J = J_u + J_chi + J_f` (before exp), in the `jastrow_eval` zone.
pub fn compute_dtn_value(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    compute_dtn_value_in_zone(d, "jastrow_eval", r_up, r_dn)
}

/// DTN log-Jastrow value evaluated in `zone`: that zone's coefficient tensors and
/// arithmetic precision. The reference / fallback ratio routes that difference two
/// whole values pass `"jastrow_ratio"`, so `J(R') - J(R)` never consumes the
/// `jastrow_eval` rounding (issue #781).
pub(crate) fn compute_dtn_value_in_zone(
    d: &JastrowDtnData,
    zone: &str,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let d = d.zone(zone); // the zone's coefficients (issues #774, #781)
    if d.precision_policy.is_f32(zone) {
        return compute_dtn_value_f32(d, r_up, r_dn);
    }
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
        match &ci {
            Some(ci) => min_image_dist([a[0] - b[0], a[1] - b[1], a[2] - b[2]], ci).0,
            None => {
                let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
        }
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
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
    let mut j = 0.0_f64;

    // u-term (e-e): up-up / dn-dn use alpha[0] (parallel), up-dn alpha[1] (anti).
    if d.use_u {
        let a_par = d.alpha.row(0);
        let a_par = a_par.as_slice().unwrap();
        let a_anti = d.alpha.row(1);
        let a_anti = a_anti.as_slice().unwrap();
        let uk = |r: f64, a: &[f64]| cutoff(r, d.l_u, d.c) * poly(r, a);
        for i in 0..n_up {
            let ri = rget(i);
            for k in (i + 1)..n_up {
                j += uk(dist(ri, rget(k)), a_par);
            }
        }
        for i in 0..n_dn {
            let ri = [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]];
            for k in (i + 1)..n_dn {
                j += uk(dist(ri, [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]), a_par);
            }
        }
        for i in 0..n_up {
            let ri = [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]];
            for k in 0..n_dn {
                j += uk(dist(ri, [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]), a_anti);
            }
        }
    }

    // chi-term (e-n): all electrons x all atoms.
    if d.use_chi {
        for e in 0..(n_up + n_dn) {
            let re = rget(e);
            for ia in 0..n_atom {
                let sp = d.species_index[ia];
                let r_ei = dist(re, pos(ia));
                let cut = cutoff(r_ei, d.l_chi[sp], d.c);
                if cut != 0.0 {
                    j += cut * poly(r_ei, d.beta.row(sp).as_slice().unwrap());
                }
            }
        }
    }

    // f-term (e-e-n): all electron pairs (i<j) x all atoms.
    if d.use_f {
        let n_tot = n_up + n_dn;
        let stride_n = d.n_ee + 1;
        let stride_m = (d.n_en + 1) * stride_n;
        let stride_l = (d.n_en + 1) * stride_m;
        for i in 0..n_tot {
            let ri = rget(i);
            for k in (i + 1)..n_tot {
                let rk = rget(k);
                let r_ij = dist(ri, rk);
                let mut pij = vec![0.0_f64; d.n_ee + 1];
                let mut t = 1.0;
                for p in pij.iter_mut() {
                    *p = t;
                    t *= r_ij;
                }
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let p = pos(ia);
                    let r_ii = dist(ri, p);
                    let cut_i = cutoff(r_ii, d.l_f[sp], d.c);
                    if cut_i == 0.0 {
                        continue;
                    }
                    let r_ji = dist(rk, p);
                    let cut_j = cutoff(r_ji, d.l_f[sp], d.c);
                    if cut_j == 0.0 {
                        continue;
                    }
                    // inner = Sum_{l,m,n} gamma[sp,l,m,n] r_ii^l r_ji^m r_ij^n
                    let base = sp * stride_l;
                    let mut inner = 0.0_f64;
                    let mut pl = 1.0;
                    for l in 0..=d.n_en {
                        let mut pm = 1.0;
                        for m in 0..=d.n_en {
                            let off = base + l * stride_m + m * stride_n;
                            let coef = pl * pm;
                            for (n, pijn) in pij.iter().enumerate() {
                                inner += d.gamma[off + n] * coef * pijn;
                            }
                            pm *= r_ji;
                        }
                        pl *= r_ii;
                    }
                    j += cut_i * cut_j * inner;
                }
            }
        }
    }
    j
}

/// Mixed-precision DTN value.  Storage remains f64 at the Python/Rust ABI, but
/// every coordinate, coefficient, polynomial operation and channel reduction
/// is rounded in f32, matching `compute_Jastrow_DTN`'s `jastrow_eval` zone.
fn compute_dtn_value_f32(
    d: &JastrowDtnData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_tot = n_up + n_dn;
    let n_atom = d.positions.nrows();
    let rget = |i: usize| -> [f64; 3] {
        if i < n_up {
            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
        } else {
            let j = i - n_up;
            [r_dn[[j, 0]], r_dn[[j, 1]], r_dn[[j, 2]]]
        }
    };

    let mut total = 0.0_f32;

    if d.use_u {
        let mut u_total = 0.0_f32;
        let pair_value = |i: usize, k: usize, coefficient_row: usize| -> f32 {
            let coefficients = d.alpha.row(coefficient_row);
            let coefficients = coefficients.as_slice().unwrap();
            let r = dtn_dist_ratio(rget(i), rget(k), &ci, true);
            let mut powers = vec![0.0_f64; coefficients.len()];
            fill_powers_ratio(&mut powers, r, true);
            (cutoff_ratio(r, d.l_u, d.c, true) as f32)
                * (poly_from_powers_ratio(coefficients, &powers, true) as f32)
        };
        // Keep the same block boundaries as XLA: uu, dd and ud are three
        // separate reductions, each rounded before it is added to J.
        if n_up >= 2 {
            let mut block = 0.0_f32;
            for i in 0..n_up {
                for k in (i + 1)..n_up {
                    block += pair_value(i, k, 0);
                }
            }
            u_total += block;
        }
        if n_dn >= 2 {
            let mut block = 0.0_f32;
            for i in 0..n_dn {
                for k in (i + 1)..n_dn {
                    block += pair_value(n_up + i, n_up + k, 0);
                }
            }
            u_total += block;
        }
        if n_up >= 1 && n_dn >= 1 {
            let mut block = 0.0_f32;
            for i in 0..n_up {
                for k in 0..n_dn {
                    block += pair_value(i, n_up + k, 1);
                }
            }
            u_total += block;
        }
        total += u_total;
    }

    if d.use_chi {
        let mut chi_total = 0.0_f32;
        let spin_block = |start: usize, end: usize| -> f32 {
            let mut block = 0.0_f32;
            for e in start..end {
                let mut electron = 0.0_f32;
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let coefficients = d.beta.row(sp);
                    let coefficients = coefficients.as_slice().unwrap();
                    let r = dtn_dist_ratio(rget(e), atom_pos(d, ia), &ci, true);
                    let mut powers = vec![0.0_f64; coefficients.len()];
                    fill_powers_ratio(&mut powers, r, true);
                    electron += (cutoff_ratio(r, d.l_chi[sp], d.c, true) as f32)
                        * (poly_from_powers_ratio(coefficients, &powers, true) as f32);
                }
                block += electron;
            }
            block
        };
        if n_up > 0 {
            chi_total += spin_block(0, n_up);
        }
        if n_dn > 0 {
            chi_total += spin_block(n_up, n_tot);
        }
        total += chi_total;
    }

    if d.use_f {
        let n_pow_ei = d.n_en + 1;
        let n_pow_ee = d.n_ee + 1;
        let mut pair_block = 0.0_f32;
        for i in 0..n_tot {
            for k in (i + 1)..n_tot {
                let r_ij = dtn_dist_ratio(rget(i), rget(k), &ci, true);
                let mut pow_ij = vec![0.0_f64; n_pow_ee];
                fill_powers_ratio(&mut pow_ij, r_ij, true);
                let mut atom_block = 0.0_f32;
                for ia in 0..n_atom {
                    let sp = d.species_index[ia];
                    let r_ii = dtn_dist_ratio(rget(i), atom_pos(d, ia), &ci, true);
                    let r_ji = dtn_dist_ratio(rget(k), atom_pos(d, ia), &ci, true);
                    let cut_i = cutoff_ratio(r_ii, d.l_f[sp], d.c, true);
                    let cut_j = cutoff_ratio(r_ji, d.l_f[sp], d.c, true);
                    if cut_i == 0.0 || cut_j == 0.0 {
                        continue;
                    }
                    let mut pow_i = vec![0.0_f64; n_pow_ei];
                    let mut pow_j = vec![0.0_f64; n_pow_ei];
                    fill_powers_ratio(&mut pow_i, r_ii, true);
                    fill_powers_ratio(&mut pow_j, r_ji, true);
                    atom_block += f_value_from_powers_ratio(
                        d, sp, &pow_i, cut_i, &pow_j, cut_j, &pow_ij, true,
                    ) as f32;
                }
                pair_block += atom_block;
            }
        }
        total += pair_block;
    }

    total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    fn fixture() -> JastrowDtnData {
        let n_species = 2;
        let n_en = 2;
        let n_ee = 2;
        let gamma_len = n_species * (n_en + 1) * (n_en + 1) * (n_ee + 1);
        let gamma = (0..gamma_len)
            .map(|i| ((i as f64) + 1.0) * 0.0007)
            .collect();
        JastrowDtnData {
            use_u: true,
            use_chi: true,
            use_f: true,
            c: 3,
            n_en,
            n_ee,
            l_u: 4.0,
            l_chi: vec![3.4, 3.8],
            l_f: vec![3.1, 3.6],
            // u has order 3, deliberately larger than the f e--e order.
            alpha: array![[0.11, -0.03, 0.007, -0.001], [0.09, 0.02, -0.004, 0.0005]],
            // chi has order 3, deliberately larger than the f e--I order.
            beta: array![[0.13, -0.02, 0.006, -0.0008], [0.07, 0.03, -0.005, 0.0006]],
            gamma,
            alpha_policy: Array2::zeros((2, 4)),
            beta_policy: Array2::zeros((2, 4)),
            gamma_policy: vec![0.0; gamma_len],
            species_index: vec![0, 1],
            positions: array![[0.1, -0.2, 0.3], [1.4, 0.7, -0.5]],
            cell: None,
            precision_policy: crate::PrecisionPolicy::full(),
            zone_coefficients: None,
            coefficients_f32: false,
        }
    }

    /// Round every coefficient tensor of `d` to float32, as `from_py` inflates a float32 set.
    fn round_coefficients_to_f32(d: &mut JastrowDtnData) {
        let r = |x: f64| x as f32 as f64;
        d.alpha.mapv_inplace(r);
        d.beta.mapv_inplace(r);
        d.alpha_policy.mapv_inplace(r);
        d.beta_policy.mapv_inplace(r);
        d.gamma.iter_mut().for_each(|x| *x = r(*x));
        d.gamma_policy.iter_mut().for_each(|x| *x = r(*x));
    }

    /// `fixture()` under `policy` with coherent coefficient metadata (issue #779),
    /// as `from_py` builds it: the shared set has the precision of
    /// `DTN_SHARED_COEFFICIENT_ZONE`, and a second set in the other precision is attached
    /// whenever `jastrow_eval`, `jastrow_ratio` or `jastrow_grad_lap_carry` differs from it.
    fn fixture_with_policy(
        policy: crate::PrecisionPolicy,
        cell: Option<Array2<f64>>,
    ) -> JastrowDtnData {
        let shared_f32 = policy.is_f32(DTN_SHARED_COEFFICIENT_ZONE);
        // The same zones as `from_py`, the carry included (issue #781).
        let needs_other = ["jastrow_eval", "jastrow_ratio", DTN_GRAD_LAP_CARRY_ZONE]
            .iter()
            .any(|z| policy.is_f32(z) != shared_f32);
        let make = |f32_set: bool| {
            let mut d = fixture();
            d.precision_policy = policy;
            d.cell = cell.clone();
            d.coefficients_f32 = f32_set;
            if f32_set {
                round_coefficients_to_f32(&mut d);
            }
            d
        };
        let mut d = make(shared_f32);
        if needs_other {
            d.zone_coefficients = Some(Box::new(make(!shared_f32)));
        }
        d
    }

    fn assert_close_slice(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (i, (&a, &b)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (a - b).abs() <= crate::setting_consts::ATOL_RS_UNIT_NEAR_MACHINE,
                "cache mismatch at {i}: {a:.17e} != {b:.17e}"
            );
        }
    }

    #[test]
    fn grad_terms_f32_move_delta_matches_full_kernel_bitwise() {
        // The fp32 DTN term cache must reproduce the full fp32 kernel bit for bit: its base rows
        // equal the kernel's gradient, and every one-electron move delta equals the fp32 branch
        // of `dtn_drift_move_delta` (two full-kernel evaluations), for every electron of both
        // spins, molecular and periodic, with all channels and with the chi channel off.
        for (pbc, use_chi) in [(false, true), (true, true), (true, false)] {
            let mut d = fixture();
            d.precision_policy = crate::native_precision_support_policy();
            d.use_chi = use_chi;
            if pbc {
                d.cell = Some(array![[4.2, 0.0, 0.0], [0.3, 3.9, 0.0], [0.0, 0.2, 4.4]]);
            }
            assert!(
                d.precision_policy.is_f32("jastrow_grad_lap"),
                "mixed policy must take the fp32 lane"
            );
            let r_up = array![[0.3, 0.4, -0.1], [1.1, -0.6, 0.8], [2.9, 1.7, 0.4]];
            let r_dn = array![[-0.7, 0.2, 1.0], [1.3, 0.5, -0.2]];
            let n_up = r_up.nrows();
            let t = DtnGradTermsF32::new(&d, &r_up.view(), &r_dn.view());
            let (gu, gd, _, _) =
                compute_grads_and_laplacian_dtn_f32(&d, &r_up.view(), &r_dn.view());
            for i in 0..(n_up + r_dn.nrows()) {
                let want = if i < n_up {
                    gu.row(i).to_vec()
                } else {
                    gd.row(i - n_up).to_vec()
                };
                for (c, want_c) in want[..3].iter().enumerate() {
                    assert_eq!(
                        (t.base[i][c] as f64).to_bits(),
                        want_c.to_bits(),
                        "pbc={pbc} chi={use_chi} base row {i}"
                    );
                }
            }
            let mut n_nonzero = 0usize;
            for (is_up, li) in [(true, 0), (true, 1), (true, 2), (false, 0), (false, 1)] {
                let r = if is_up { &r_up } else { &r_dn };
                let base = [r[[li, 0]], r[[li, 1]], r[[li, 2]]];
                for shift in [[0.37, -0.21, 0.15], [-0.9, 0.6, 1.3], [2.5, -1.8, 0.4]] {
                    let pos = [base[0] + shift[0], base[1] + shift[1], base[2] + shift[2]];
                    let (gu_t, gd_t) = t.move_delta(&d, &r_up.view(), &r_dn.view(), is_up, li, pos);
                    let (gu_w, gd_w) =
                        dtn_drift_move_delta(&d, &r_up.view(), &r_dn.view(), is_up, li, pos);
                    for (got, want) in [(&gu_t, &gu_w), (&gd_t, &gd_w)] {
                        assert_eq!(got.dim(), want.dim());
                        for (a, b) in got.iter().zip(want.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "pbc={pbc} chi={use_chi} up={is_up} li={li} shift={shift:?}"
                            );
                            n_nonzero += usize::from(*b != 0.0);
                        }
                    }
                }
            }
            assert!(n_nonzero > 0, "pbc={pbc} chi={use_chi}: vacuous comparison");
        }
    }

    #[test]
    fn grad_row_f32_matches_full_kernel_row_bitwise() {
        // The single-row fp32 DTN gradient must reproduce the corresponding row of the full
        // fp32 kernel bit for bit, at the base configuration and with the electron moved,
        // molecular and periodic, for every electron of both spins.
        for pbc in [false, true] {
            let cell = pbc.then(|| array![[4.2, 0.0, 0.0], [0.3, 3.9, 0.0], [0.0, 0.2, 4.4]]);
            let d = fixture_with_policy(crate::native_precision_support_policy(), cell);
            let r_up = array![[0.3, 0.4, -0.1], [1.1, -0.6, 0.8], [2.9, 1.7, 0.4]];
            let r_dn = array![[-0.7, 0.2, 1.0], [1.3, 0.5, -0.2]];
            let n_up = r_up.nrows();
            let mut n_nonzero = 0usize;
            for g in 0..(n_up + r_dn.nrows()) {
                let base = if g < n_up {
                    r_up.row(g).to_vec()
                } else {
                    r_dn.row(g - n_up).to_vec()
                };
                for shift in [[0.0, 0.0, 0.0], [0.37, -0.21, 0.15], [-0.9, 0.6, 1.3]] {
                    let pos = [base[0] + shift[0], base[1] + shift[1], base[2] + shift[2]];
                    let (mut mu, mut md) = (r_up.clone(), r_dn.clone());
                    let row = if g < n_up {
                        mu.row_mut(g)
                    } else {
                        md.row_mut(g - n_up)
                    };
                    for (dst, &v) in row.into_iter().zip(pos.iter()) {
                        *dst = v;
                    }
                    let (gu, gd, _, _) =
                        compute_grads_and_laplacian_dtn_f32(&d, &mu.view(), &md.view());
                    let want = if g < n_up {
                        gu.row(g).to_vec()
                    } else {
                        gd.row(g - n_up).to_vec()
                    };
                    let got = dtn_grad_row_f32(&d, &r_up.view(), &r_dn.view(), g, pos);
                    for c in 0..3 {
                        assert_eq!(
                            got[c].to_bits(),
                            want[c].to_bits(),
                            "pbc={pbc} g={g} shift={shift:?} c={c}"
                        );
                        n_nonzero += usize::from(want[c] != 0.0);
                    }
                }
            }
            assert!(n_nonzero > 0, "pbc={pbc}: vacuous comparison");
        }
    }

    #[test]
    fn group_log_ratio_matches_single_candidate_bitwise() {
        // Route twin: the per-electron group kernel that seeds the kinetic / ECP
        // candidate tables must reproduce the single-candidate streaming kernel
        // (log ratio) and the stored candidate--atom distances bit for bit, in
        // both precision lanes, molecular and periodic.
        for (mixed, pbc) in [(false, false), (true, false), (false, true), (true, true)] {
            let cell = pbc.then(|| array![[4.2, 0.0, 0.0], [0.3, 3.9, 0.0], [0.0, 0.2, 4.4]]);
            let policy = if mixed {
                crate::native_precision_support_policy()
            } else {
                crate::PrecisionPolicy::full()
            };
            let d = fixture_with_policy(policy, cell);
            let r_up = array![[0.3, 0.4, -0.1], [1.1, -0.6, 0.8], [2.9, 1.7, 0.4]];
            let r_dn = array![[-0.7, 0.2, 1.0], [1.3, 0.5, -0.2]];
            let st = init_dtn_ratio_state(&d, &r_up.view(), &r_dn.view());
            let ci = d.cell.as_ref().map(|c| CellInfo::new(&c.view()));
            let n_atom = d.positions.nrows();
            let n_k = 13;
            let (mut n_f_live, mut n_nonzero) = (0usize, 0usize);
            for (is_up, idx) in [(true, 0), (true, 1), (true, 2), (false, 0), (false, 1)] {
                let r = if is_up { &r_up } else { &r_dn };
                let base = [r[[idx, 0]], r[[idx, 1]], r[[idx, 2]]];
                let cands: Vec<[f64; 3]> = (0..n_k)
                    .map(|c| {
                        let t = c as f64 * 0.61;
                        [
                            base[0] + 0.9 * t.cos(),
                            base[1] + 0.7 * t.sin(),
                            base[2] + 0.05 * c as f64 - 0.3,
                        ]
                    })
                    .collect();
                let mut log = vec![0.0_f64; n_k];
                let mut rei = vec![0.0_f64; n_k * n_atom];
                compute_dtn_log_ratio_streaming_group(
                    &d,
                    &st,
                    &r_up.view(),
                    &r_dn.view(),
                    is_up,
                    idx,
                    &cands,
                    &mut log,
                    &mut rei,
                );
                for c in 0..n_k {
                    let want = compute_dtn_log_ratio_streaming(
                        &d,
                        &st,
                        &r_up.view(),
                        &r_dn.view(),
                        is_up,
                        idx,
                        cands[c],
                    );
                    assert_eq!(
                        want.to_bits(),
                        log[c].to_bits(),
                        "mixed={mixed} pbc={pbc} up={is_up} e={idx} c={c}"
                    );
                    n_nonzero += usize::from(want != 0.0);
                    for ia in 0..n_atom {
                        let r_ia = dtn_dist_ratio(cands[c], atom_pos(&d, ia), &ci, mixed);
                        assert_eq!(r_ia.to_bits(), rei[c * n_atom + ia].to_bits());
                        n_f_live += usize::from(
                            cutoff_ratio(r_ia, d.l_f[d.species_index[ia]], d.c, mixed) != 0.0,
                        );
                    }
                }
            }
            // Non-vacuity: non-trivial ratios with live f-channel legs.
            assert!(
                n_nonzero == 5 * n_k,
                "every candidate must move the DTN value"
            );
            assert!(n_f_live > 0, "candidates must engage the f channel");
        }
    }

    #[test]
    fn ratio_state_caches_all_five_geometry_tables() {
        let d = fixture();
        let r_up = array![[0.3, 0.4, -0.1], [1.1, -0.6, 0.8]];
        let r_dn = array![[-0.7, 0.2, 1.0]];
        let state = init_dtn_ratio_state(&d, &r_up.view(), &r_dn.view());

        assert_eq!(state.n_e, 3);
        assert_eq!(state.n_atom, 2);
        assert_eq!(state.max_ei_order, 3);
        assert_eq!(state.max_ee_order, 3);
        assert_eq!(state.r_ei.len(), 3 * 2);
        assert_eq!(state.cut_ei.len(), 3 * 2);
        assert_eq!(state.pow_ei.len(), 3 * 2 * 4);
        assert_eq!(state.r_ee.len(), 3 * 3);
        assert_eq!(state.pow_ee.len(), 3 * 3 * 4);

        for leg in 0..state.r_ei.len() {
            let powers = &state.pow_ei[leg * 4..leg * 4 + 4];
            for (order, &value) in powers.iter().enumerate() {
                assert!(
                    (value - state.r_ei[leg].powi(order as i32)).abs()
                        <= crate::setting_consts::ATOL_RS_UNIT_NEAR_MACHINE
                );
            }
        }
        for pair in 0..state.r_ee.len() {
            let powers = &state.pow_ee[pair * 4..pair * 4 + 4];
            for (order, &value) in powers.iter().enumerate() {
                assert!(
                    (value - state.r_ee[pair].powi(order as i32)).abs()
                        <= crate::setting_consts::ATOL_RS_UNIT_NEAR_MACHINE
                );
            }
        }
    }

    #[test]
    fn streaming_ratio_and_accepted_update_match_independent_recomputation() {
        let d = fixture();
        let mut r_up = array![[0.3, 0.4, -0.1], [1.1, -0.6, 0.8]];
        let r_dn = array![[-0.7, 0.2, 1.0]];
        let mut state = init_dtn_ratio_state(&d, &r_up.view(), &r_dn.view());
        let new_pos = [0.55, -0.15, 0.65];

        let cached =
            compute_dtn_ratio_streaming(&d, &state, &r_up.view(), &r_dn.view(), true, 1, new_pos);
        let reference = compute_dtn_ratio(&d, &r_up.view(), &r_dn.view(), true, 1, new_pos);
        assert!((cached - reference).abs() <= crate::setting_consts::ATOL_RS_UNIT_REDUCTION);

        for axis in 0..3 {
            r_up[[1, axis]] = new_pos[axis];
        }
        advance_dtn_ratio_state(&d, &mut state, &r_up.view(), &r_dn.view(), true, 1);
        let fresh = init_dtn_ratio_state(&d, &r_up.view(), &r_dn.view());
        assert_close_slice(&state.r_ei, &fresh.r_ei);
        assert_close_slice(&state.cut_ei, &fresh.cut_ei);
        assert_close_slice(&state.pow_ei, &fresh.pow_ei);
        assert_close_slice(&state.r_ee, &fresh.r_ee);
        assert_close_slice(&state.pow_ee, &fresh.pow_ee);
    }

    #[test]
    fn value_strain_move_delta_batch_matches_per_hop() {
        // The grouped one-electron DTN value-strain delta (light f kernels, shared e-N tables,
        // base terms once per group) against the per-hop `dtn_value_strain_move_delta`, for
        // every channel and a nonzero cutoff policy, molecular and periodic.
        for pbc in [false, true] {
            let mut d = fixture();
            d.alpha_policy = array![[0.02, 0.01, -0.003, 0.0004], [-0.01, 0.015, 0.002, -0.0003]];
            d.beta_policy = array![[0.03, -0.01, 0.004, -0.0002], [-0.02, 0.01, -0.001, 0.0003]];
            d.gamma_policy = (0..d.gamma.len())
                .map(|i| ((i % 7) as f64 - 3.0) * 0.0004)
                .collect();
            if pbc {
                d.cell = Some(array![[4.2, 0.0, 0.0], [0.3, 3.9, 0.0], [0.0, 0.2, 4.4]]);
            }
            let r_up = array![[0.3, 0.4, -0.1], [1.1, -0.6, 0.8], [2.9, 1.7, 0.4]];
            let r_dn = array![[-0.7, 0.2, 1.0], [1.3, 0.5, -0.2]];
            for (is_up, li) in [(true, 0), (true, 2), (false, 1)] {
                let r = if is_up { &r_up } else { &r_dn };
                let hops: Vec<[f64; 3]> = (0..5)
                    .map(|k| {
                        let t = 0.37 * (k as f64 + 1.0);
                        [
                            r[[li, 0]] + t.cos(),
                            r[[li, 1]] + 0.8 * t.sin(),
                            r[[li, 2]] - 0.5 * t,
                        ]
                    })
                    .collect();
                let batch = crate::stress_pbc::dtn_value_strain_move_delta_batch(
                    &d,
                    &r_up.view(),
                    &r_dn.view(),
                    is_up,
                    li,
                    &hops,
                );
                for (h, &p) in hops.iter().enumerate() {
                    let one = crate::stress_pbc::dtn_value_strain_move_delta(
                        &d,
                        &r_up.view(),
                        &r_dn.view(),
                        is_up,
                        li,
                        p,
                    );
                    assert!(one.iter().any(|v| v.abs() > 0.0));
                    for (a, b) in batch[h].iter().zip(one.iter()) {
                        assert!(
                            (a - b).abs() <= crate::setting_consts::ATOL_RS_UNIT_REDUCTION,
                            "pbc={pbc} hop {h}: {a:.17e} != {b:.17e}"
                        );
                    }
                }
            }
        }
    }
}
