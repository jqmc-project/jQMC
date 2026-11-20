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

//! Data structs mirror the jqmc pytree dataclasses and expose `from_py`
//! constructors.
//!
//! Each struct reads the Python dataclass fields by name. Large arrays are
//! taken as owned `ndarray` copies. Scalars and integer lists are copied too,
//! so the native structs do not borrow Python storage after construction.
//!
//! Periodic fields (`cell`) are read when `pbc_flag` is set; the periodic J1/J2
//! kernels then use the Williamson smooth periodic distance.

use numpy::ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// 1:1 with `jqmc.structure.Structure_data` (the fields the value functions use).
pub struct StructureData {
    /// Cartesian atomic positions, shape `(N, 3)`, Bohr.
    pub positions: Array2<f64>,
    /// Atomic numbers `Z` per site (kept as f64 for arithmetic).
    pub atomic_numbers: Vec<f64>,
    pub pbc_flag: bool,
    /// Lattice matrix `(3, 3)` (rows are lattice vectors), only when periodic.
    pub cell: Option<Array2<f64>>,
}

/// Read a `(3, 3)` cell from a Python object via `np.asarray` (handles jax arrays).
fn read_cell(obj: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("asarray", (obj.getattr("cell")?,))?;
    let c: PyReadonlyArray2<f64> = arr.extract()?;
    Ok(c.as_array().to_owned())
}

impl StructureData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let pbc_flag: bool = obj.getattr("pbc_flag")?.extract()?;
        // `positions` is an (N, 3) float64 ndarray.
        let positions_arr: PyReadonlyArray2<f64> = obj.getattr("positions")?.extract()?;
        let positions = positions_arr.as_array().to_owned();
        // `atomic_numbers` is a tuple/list of ints; widen to f64.
        let atomic_numbers: Vec<f64> = obj.getattr("atomic_numbers")?.extract()?;
        let cell = if pbc_flag {
            Some(read_cell(obj)?)
        } else {
            None
        };
        Ok(StructureData {
            positions,
            atomic_numbers,
            pbc_flag,
            cell,
        })
    }

    #[inline]
    pub fn n_atoms(&self) -> usize {
        self.positions.nrows()
    }
}

/// 1:1 with `jqmc.jastrow_factor.Jastrow_one_body_data`.
pub struct JastrowOneBodyData {
    pub(crate) precision_policy: crate::PrecisionPolicy,
    pub jastrow_1b_param: f64,
    pub jastrow_1b_type: String,
    pub structure_data: StructureData,
    pub core_electrons: Vec<f64>,
}

impl JastrowOneBodyData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // jastrow_1b_param is a 0-d numpy array; __float__ extraction is exact.
        let jastrow_1b_param: f64 = obj.getattr("jastrow_1b_param")?.extract()?;
        let jastrow_1b_type: String = obj.getattr("jastrow_1b_type")?.extract()?;
        if !matches!(jastrow_1b_type.as_str(), "exp" | "pade") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown jastrow_1b_type: {jastrow_1b_type}"
            )));
        }
        let structure_data = StructureData::from_py(&obj.getattr("structure_data")?)?;
        let core_electrons: Vec<f64> = obj.getattr("core_electrons")?.extract()?;
        Ok(JastrowOneBodyData {
            precision_policy: crate::construction_precision_policy(),
            jastrow_1b_param,
            jastrow_1b_type,
            structure_data,
            core_electrons,
        })
    }
}

/// Mirrors `jqmc.jastrow_factor.Jastrow_three_body_data`: the
/// shared orbital basis + the `(orb_num, orb_num+1)` J3/J1 matrix. Owned so it
/// can live as a `JastrowData` field, so the value/force/stress kernels take it
/// with the rest of the Jastrow data.
pub struct J3Data {
    /// Orbital basis (AOs/MOs) shared across spins, `orb_data`.
    pub orb: crate::determinant::OrbData,
    /// J3/J1 matrix, shape `(orb_num, orb_num+1)`.
    pub j_matrix: Array2<f64>,
}

impl J3Data {
    /// Borrow the `(orb, j_matrix view)` pair the value/kinetic kernels consume.
    #[inline]
    pub fn as_ref(
        &self,
    ) -> (
        &crate::determinant::OrbData,
        numpy::ndarray::ArrayView2<'_, f64>,
    ) {
        (&self.orb, self.j_matrix.view())
    }
}

/// Combined `Jastrow_data` ({one,two,three}_body + DTN), mirroring
/// `jqmc.jastrow_factor.Jastrow_data`.
pub struct JastrowData {
    pub(crate) precision_policy: crate::PrecisionPolicy,
    pub one_body: Option<JastrowOneBodyData>,
    pub two_body: Option<JastrowTwoBodyData>,
    pub three_body: Option<J3Data>,
    /// `true` when an ACTIVE J3 was left out of `three_body` because only the complex
    /// observable path can evaluate its orbital; see [`JastrowData::require_real_three_body`].
    pub three_body_complex_only: bool,
    pub dtn: Option<crate::dtn::JastrowDtnData>,
}

impl JastrowData {
    /// Parse for a REAL native consumer (the default).
    ///
    /// Raises `NotImplementedError` (see [`Self::require_real_three_body`]) when the active J3
    /// orbital is one only the complex observable path can evaluate, so no real consumer ever
    /// sees `three_body == None` for an active J3 it cannot represent and silently evaluates the
    /// wavefunction without it (PR #93: the exported `compute_local_energy` /
    /// `compute_kinetic_energy` free kernels did exactly that). Consumers that ingest the J3
    /// through the complex path themselves use [`Self::from_py_complex_aware`].
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let jdata = Self::from_py_impl(obj, true)?;
        jdata.require_real_three_body("real native kernel")?;
        Ok(jdata)
    }

    /// Parse WITHOUT rejecting a complex-only J3: `three_body` is `None` for it and
    /// `three_body_complex_only` records why.
    ///
    /// Only for consumers that evaluate the J3 through the complex path themselves
    /// (`parse_j3_complex`, `parse_jastrow_complex`, `J3ComplexOwned`, `ComplexOrb`), or that
    /// branch on real versus complex and call [`Self::require_real_three_body`] on the real
    /// branch, or that never read the J3 (`jastrow_1b2b_ratio`,
    /// `compute_grads_and_laplacian_jastrow_part`).
    pub fn from_py_complex_aware(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_py_impl(obj, true)
    }

    /// J1 / J2 / DTN only; `three_body` is left `None`.
    ///
    /// The complex-twist parameter-derivative kernels ingest the J3 leaf SEPARATELY as a
    /// [`crate::complex_det::ComplexOrb`] (`parse_j3_complex`) and thread it explicitly --
    /// they never read `JastrowData::three_body`. Building a REAL [`crate::determinant::OrbData`]
    /// for it there is not just dead work, it HARD-FAILS on a genuinely complex J3 leaf:
    /// `AOs_bspline_complex_data` exposes no `_complex` flag and
    /// `AOs_plane_wave_complex_data` no `_num_ao_kernel` (measured -- this is what kept the
    /// MO-wrapped complex J3 c-axis unserved). Callers that DO consume `three_body`
    /// (the value / kinetic / native-sweep paths) must keep using [`Self::from_py`].
    pub fn from_py_without_three_body(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_py_impl(obj, false)
    }

    fn from_py_impl(obj: &Bound<'_, PyAny>, want_three_body: bool) -> PyResult<Self> {
        let ob = obj.getattr("jastrow_one_body_data")?;
        let one_body = if ob.is_none() {
            None
        } else {
            Some(JastrowOneBodyData::from_py(&ob)?)
        };
        let tb = obj.getattr("jastrow_two_body_data")?;
        let two_body = if tb.is_none() {
            None
        } else {
            Some(JastrowTwoBodyData::from_py(&tb)?)
        };
        // three-body: skip an INACTIVE (all-zero coupling) J3 entirely, exactly
        // as the native sweep / Turbo do -- an all-zero j_matrix contributes
        // nothing, so `three_body = None` keeps the J3 streaming/eval off on
        // every value/kinetic path.
        let mut three_body_complex_only = false;
        let three_body = if !want_three_body {
            None
        } else {
            let j3d = obj.getattr("jastrow_three_body_data")?;
            if j3d.is_none() {
                None
            } else {
                // `j_matrix` may be a jnp ArrayImpl (j123 complex/PBC fixtures) --
                // coerce via np.asarray before the PyReadonlyArray extract.
                let np = obj.py().import("numpy")?;
                let jm: PyReadonlyArray2<f64> = np
                    .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
                    .call_method1("astype", (np.getattr("float64")?,))?
                    .extract()?;
                let j_matrix = jm.as_array().to_owned();
                if j_matrix.iter().all(|&x| x == 0.0) {
                    None
                } else {
                    let orb_py = j3d.getattr("orb_data")?;
                    // Ownership is decided by what the REAL `OrbData` parser can represent
                    // exactly, not by child type names (the determinant, not the J3, picks real
                    // versus complex dispatch). A complex PW / B-spline leaf or child, or a
                    // genuinely complex MO, is owned by the complex observable path
                    // (`parse_j3_complex` / `J3ComplexOwned`): skip it here (the real parser
                    // cannot read it, and a complex MO would be cast to its real part, issue #88)
                    // and record that, so a REAL consumer rejects it loudly instead of silently
                    // evaluating the wavefunction without its J3. A phase-free Gamma-complex
                    // pGTO leaf or child has unit image phases: the real gto parser reads the
                    // same primitives, so it is consumed (PR #93).
                    if j3_orbital_real_representable(&orb_py)? {
                        let orb = crate::determinant::OrbData::from_py(&orb_py)?;
                        Some(J3Data { orb, j_matrix })
                    } else {
                        three_body_complex_only = true;
                        None
                    }
                }
            }
        };
        let dtn = if obj.hasattr("jastrow_dtn_data")? {
            let dd = obj.getattr("jastrow_dtn_data")?;
            if dd.is_none() {
                None
            } else {
                Some(crate::dtn::JastrowDtnData::from_py(&dd)?)
            }
        } else {
            None
        };
        Ok(JastrowData {
            precision_policy: crate::construction_precision_policy(),
            one_body,
            two_body,
            three_body,
            three_body_complex_only,
            dtn,
        })
    }

    /// Reject an active J3 that only the complex observable path can evaluate.
    ///
    /// [`Self::from_py`] calls this for every real consumer; the real/complex branching
    /// consumers (`ClefArgs`, the observable contexts) call it on their real branch after
    /// [`Self::from_py_complex_aware`]. So for a real consumer `three_body == None` means "no
    /// active J3", never "a J3 this real kernel cannot represent" (which would silently omit
    /// the J3 from the energy).
    pub fn require_real_three_body(&self, consumer: &str) -> PyResult<()> {
        if self.three_body_complex_only {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "{consumer}: the active J3 orbital (complex plane-wave / B-spline basis or a \
                 complex MO) is supported only by the complex native path, but the determinant \
                 selects the real path; refusing to evaluate the wavefunction without its J3"
            )));
        }
        Ok(())
    }
}

/// Whether the REAL J3 `OrbData` parser represents this J3 orbital (`MOs_data` or AO leaf) exactly.
///
/// Not representable: a complex plane-wave or B-spline leaf / composite child (no real kernel;
/// `extract_aos` skips such children), a complex pGTO with a nonzero Bloch `k_point`, or MO
/// coefficients with a nonzero imaginary part. A phase-free Gamma-complex pGTO
/// (`build_AOs_cart_pgto_complex_data(k_point=None or 0, ...)`, unit image phases) has the same
/// primitives and values as the real pGTO, so it is representable.
pub(crate) fn j3_orbital_real_representable(orb_py: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = orb_py.py().import("numpy")?;
    let leaf = if orb_py.hasattr("mo_coefficients")? {
        let c = np.call_method1("asarray", (orb_py.getattr("mo_coefficients")?,))?;
        if np.call_method1("iscomplexobj", (&c,))?.extract::<bool>()?
            && np
                .call_method1("any", (c.getattr("imag")?,))?
                .extract::<bool>()?
        {
            return Ok(false);
        }
        orb_py.getattr("aos_data")?
    } else {
        orb_py.clone()
    };
    let nodes: Vec<Bound<'_, PyAny>> = if leaf.hasattr("children")? {
        let children = leaf.getattr("children")?;
        let mut v = Vec::new();
        for i in 0..children.len()? {
            v.push(children.get_item(i)?);
        }
        v
    } else {
        vec![leaf]
    };
    for node in &nodes {
        let name = node.get_type().name()?.to_string();
        if !name.contains("complex") {
            continue;
        }
        if !name.contains("pgto") {
            return Ok(false);
        }
        let k = node.getattr("k_point")?;
        if !k.is_none()
            && np
                .call_method1("any", (np.call_method1("asarray", (&k,))?,))?
                .extract::<bool>()?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// 1:1 with `jqmc.jastrow_factor.Jastrow_two_body_data`.
pub struct JastrowTwoBodyData {
    pub(crate) precision_policy: crate::PrecisionPolicy,
    pub jastrow_2b_param: f64,
    pub jastrow_2b_type: String,
    pub pbc_flag: bool,
    /// Lattice matrix `(3, 3)` (rows are lattice vectors), only when periodic.
    pub cell: Option<Array2<f64>>,
}

impl JastrowTwoBodyData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let jastrow_2b_param: f64 = obj.getattr("jastrow_2b_param")?.extract()?;
        let jastrow_2b_type: String = obj.getattr("jastrow_2b_type")?.extract()?;
        if !matches!(jastrow_2b_type.as_str(), "exp" | "pade") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown jastrow_2b_type: {jastrow_2b_type}"
            )));
        }
        let sd = obj.getattr("structure_data")?;
        let pbc_flag: bool = sd.getattr("pbc_flag")?.extract()?;
        let cell = if pbc_flag {
            Some(read_cell(&sd)?)
        } else {
            None
        };
        Ok(JastrowTwoBodyData {
            precision_policy: crate::construction_precision_policy(),
            jastrow_2b_param,
            jastrow_2b_type,
            pbc_flag,
            cell,
        })
    }
}
