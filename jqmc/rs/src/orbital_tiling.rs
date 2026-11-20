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

//! Representation-independent primitive k blocks. No global AO/MO matrix is built.
//!
//! This is the native counterpart of orbital_tiling.py's fixed-geometry provider.
//! Determinant/driver integration is intentionally a separate consumer of this model.

use std::collections::HashMap;
use std::sync::Arc;

use numpy::ndarray::{s, Array2, ArrayView2};
use numpy::{Complex64, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::complex_det::ComplexOrb;

struct OrbitalKBlock {
    backend_key: usize,
    coefficient_key: usize,
    backend: Arc<ComplexOrb>,
    coefficients: Option<Arc<Array2<Complex64>>>,
    occupied: usize,
    conjugate: bool,
    images: Vec<[i64; 3]>,
    kernel_atoms: Vec<usize>,
}

/// Ordered, ragged primitive blocks with a single supercell normalization.
pub struct TiledOrbitalSet {
    blocks: Vec<OrbitalKBlock>,
    occupied: usize,
    normalization: f64,
    primitive_atoms: usize,
    physical_atoms: usize,
    quotient: Vec<[i128; 3]>,
    adjugate: [[i128; 3]; 3],
    determinant: i128,
}

#[derive(Clone, Copy)]
pub enum TiledCoordinate {
    Atoms,
    Strain,
    Electrons,
}

impl OrbitalKBlock {
    fn contract(&self, basis: Array2<Complex64>, normalization: f64) -> Array2<Complex64> {
        let mut final_values = match &self.coefficients {
            Some(c) => c.dot(&basis),
            None => basis,
        };
        final_values.mapv_inplace(|v| {
            if self.conjugate {
                v.conj() * normalization
            } else {
                v * normalization
            }
        });
        final_values
    }
}

impl TiledOrbitalSet {
    pub fn num_orb(&self) -> usize {
        self.occupied
    }

    /// Parse the same versioned block schema as Python; preserve within-set aliases.
    pub fn from_py(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_py_with_source(data, None)
    }

    pub fn from_py_with_source(data: &Bound<'_, PyAny>, source: Option<&Self>) -> PyResult<Self> {
        // Validates schema/normalization, k cosets, backend types, local matrix
        // dimensions and physical geometry before native kernels see any arrays.
        data.call_method0("sanity_check")?;
        let blocks_py = data.getattr("blocks")?;
        let n_blocks = blocks_py.len()?;
        let np = data.py().import("numpy")?;
        let geometry = data.py().import("jqmc.structure")?;
        let matrix = data.getattr("tiling_matrix")?;
        let determinant_adjugate =
            geometry.call_method1("_tiling_determinant_adjugate", (&matrix,))?;
        let determinant: i128 = determinant_adjugate.get_item(0)?.extract()?;
        let adjugate: [[i128; 3]; 3] = determinant_adjugate.get_item(1)?.extract()?;
        let quotient: Vec<[i128; 3]> = geometry
            .call_method1("_tiling_quotient_residues", (&matrix,))?
            .extract()?;
        let primitive_atoms = data
            .getattr("primitive_structure")?
            .getattr("natom")?
            .extract()?;
        let physical_atoms = data
            .getattr("structure_data")?
            .getattr("natom")?
            .extract()?;
        let mut backends: HashMap<usize, Arc<ComplexOrb>> = HashMap::new();
        let mut operators: HashMap<usize, Arc<Array2<Complex64>>> = HashMap::new();
        if let Some(source) = source {
            for block in &source.blocks {
                backends.insert(block.backend_key, Arc::clone(&block.backend));
                if let Some(c) = &block.coefficients {
                    operators.insert(block.coefficient_key, Arc::clone(c));
                }
            }
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut occupied = 0;
        for block in blocks_py.try_iter()? {
            let block = block?;
            let backend_py = block.getattr("backend")?;
            let key = backend_py.as_ptr() as usize;
            let backend = match backends.get(&key) {
                Some(stored) => Arc::clone(stored),
                None => {
                    let parsed = Arc::new(ComplexOrb::from_py(&backend_py)?);
                    backends.insert(key, Arc::clone(&parsed));
                    parsed
                }
            };
            let coefficients_py = block.getattr("coefficients")?;
            let coefficients = if coefficients_py.is_none() {
                None
            } else {
                let key = coefficients_py.as_ptr() as usize;
                Some(match operators.get(&key) {
                    Some(stored) => Arc::clone(stored),
                    None => {
                        let array: PyReadonlyArray2<Complex64> = np
                            .call_method1("asarray", (&coefficients_py, np.getattr("complex128")?))?
                            .extract()?;
                        let parsed = Arc::new(array.as_array().to_owned());
                        operators.insert(key, Arc::clone(&parsed));
                        parsed
                    }
                })
            };
            let count: usize = block.getattr("num_orbitals")?.extract()?;
            occupied += count;
            // A composite owns one pGTO child and one PW child. Keep the
            // pGTO image map beside the already shared native composite;
            // do not extract a second copy of either child's orbital data.
            let mut image_leaf = backend_py.clone();
            if backend_py.hasattr("children")? {
                for child in backend_py.getattr("children")?.try_iter()? {
                    let child = child?;
                    if child.hasattr("_image_indices")? {
                        image_leaf = child;
                        break;
                    }
                }
            }
            blocks.push(OrbitalKBlock {
                backend_key: backend_py.as_ptr() as usize,
                coefficient_key: coefficients_py.as_ptr() as usize,
                backend,
                coefficients,
                occupied: count,
                conjugate: block.getattr("conjugate")?.extract()?,
                images: if image_leaf.hasattr("_image_indices")? {
                    image_leaf.getattr("_image_indices")?.extract()?
                } else {
                    Vec::new()
                },
                kernel_atoms: if image_leaf.hasattr("_ragged_kernel_atom")? {
                    image_leaf.getattr("_ragged_kernel_atom")?.extract()?
                } else {
                    Vec::new()
                },
            });
        }
        Ok(Self {
            blocks,
            occupied,
            normalization: 1.0 / (n_blocks as f64).sqrt(),
            primitive_atoms,
            physical_atoms,
            quotient,
            adjugate,
            determinant,
        })
    }

    /// Holomorphic VGL cotangent response at each physical atom, raw strain,
    /// or electron. Only local block coefficients and kernel image arrays occur.
    pub fn field_vjp(
        &self,
        r: &ArrayView2<f64>,
        bar_value: &Array2<Complex64>,
        bar_grad: Option<&[Array2<Complex64>; 3]>,
        bar_lap: Option<&Array2<Complex64>>,
        coordinate: TiledCoordinate,
    ) -> Array2<Complex64> {
        use crate::atomic_orbital::{
            compute_aos_gradlap_kernel_sphe, compute_aos_hessian_kernel_sphe,
            compute_aos_vgl_kernel,
        };
        use crate::complex_ao::{compute_pw_hessian_complex, compute_pw_vgl_complex};
        let rows = match coordinate {
            TiledCoordinate::Atoms => self.physical_atoms,
            TiledCoordinate::Strain => 3,
            TiledCoordinate::Electrons => r.nrows(),
        };
        let mut result = Array2::<Complex64>::zeros((rows, 3));
        let mut offset = 0;
        for block in &self.blocks {
            let start = offset;
            offset += block.occupied;
            if block.occupied == 0 {
                continue;
            }
            let orb = &block.backend;
            let is_gto = !block.images.is_empty();
            if matches!(coordinate, TiledCoordinate::Atoms) && !is_gto {
                continue;
            }
            let project = |bar: Option<&Array2<Complex64>>| {
                let mut values = match bar {
                    Some(bar) => bar.slice(s![start..offset, ..]).to_owned(),
                    None => Array2::zeros((block.occupied, r.nrows())),
                };
                if block.conjugate {
                    values.mapv_inplace(|v| v.conj());
                }
                let mut out = match &block.coefficients {
                    Some(c) => c.t().dot(&values),
                    None => values,
                };
                out.mapv_inplace(|v| v * self.normalization);
                out
            };
            let cv = project(Some(bar_value));
            let cl = project(bar_lap);
            let cg: [Array2<Complex64>; 3] =
                std::array::from_fn(|d| project(bar_grad.map(|g| &g[d])));
            let mut contribution = Array2::<Complex64>::zeros((rows, 3));
            if is_gto {
                let (rw, restore) = crate::complex_stress::wrap_restore(orb, r);
                let (_, gx, gy, gz, _) = compute_aos_vgl_kernel(&orb.aos, &rw.view(), false);
                let grad = [gx, gy, gz];
                let gl =
                    bar_lap.map(|_| compute_aos_gradlap_kernel_sphe(&orb.aos, &rw.view(), false));
                let hess =
                    bar_grad.map(|_| compute_aos_hessian_kernel_sphe(&orb.aos, &rw.view(), false));
                let wrap = orb
                    .wrap
                    .as_ref()
                    .expect("tiled pGTO requires primitive Bloch wrapping");
                let gto_offset = if orb.gto_first {
                    0
                } else {
                    orb.pw.as_ref().map_or(0, |pw| pw.num_ao)
                };
                for e in 0..r.nrows() {
                    let n0: [i128; 3] = std::array::from_fn(|d| {
                        (r[[e, 0]] * wrap.inv_cell[0][d]
                            + r[[e, 1]] * wrap.inv_cell[1][d]
                            + r[[e, 2]] * wrap.inv_cell[2][d])
                            .floor() as i128
                    });
                    for kernel in 0..orb.aos.num_ao_kernel {
                        let home = gto_offset + orb.aos.kernel_to_home[kernel];
                        let phase = orb.image_phase[kernel] * restore[e];
                        let image = block.images[kernel];
                        let label: [i128; 3] = std::array::from_fn(|d| {
                            ((0..3)
                                .map(|a| (image[a] as i128 + n0[a]) * self.adjugate[a][d])
                                .sum::<i128>()
                                * self.determinant.signum())
                            .rem_euclid(self.determinant.abs())
                        });
                        let coset = self
                            .quotient
                            .binary_search(&label)
                            .expect("complete real-space quotient");
                        let atom = coset * self.primitive_atoms + block.kernel_atoms[kernel];
                        for c in 0..3 {
                            let mut field = cv[[home, e]] * grad[c][[kernel, e]];
                            if let Some(gl) = &gl {
                                field += cl[[home, e]] * gl[c][[kernel, e]];
                            }
                            if let Some(h) = &hess {
                                for d in 0..3 {
                                    field += cg[d][[home, e]] * h[c][d][[kernel, e]];
                                }
                            }
                            field *= phase;
                            match coordinate {
                                TiledCoordinate::Atoms => contribution[[atom, c]] -= field,
                                TiledCoordinate::Electrons => contribution[[e, c]] += field,
                                TiledCoordinate::Strain => {
                                    for d in 0..3 {
                                        contribution[[c, d]] +=
                                            field * (rw[[e, d]] - orb.aos.centers[[kernel, d]]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !matches!(coordinate, TiledCoordinate::Atoms)
                && (orb.pw.is_some() || orb.bspline.is_some())
                && (bar_grad.is_some()
                    || bar_lap.is_some()
                    || matches!(coordinate, TiledCoordinate::Electrons))
            {
                let (grad, gl, h) = if let Some(pw) = &orb.pw {
                    let (_, grad, _) = compute_pw_vgl_complex(pw, r);
                    let gl = std::array::from_fn(|d| {
                        Array2::from_shape_fn(grad[d].raw_dim(), |(a, e)| {
                            -pw.gk[a].iter().map(|v| v * v).sum::<f64>() * grad[d][[a, e]]
                        })
                    });
                    (grad, gl, compute_pw_hessian_complex(pw, r))
                } else {
                    orb.bspline
                        .as_ref()
                        .expect("supported primitive leaf")
                        .det_egrad_fields(r)
                };
                let extra_offset = if orb.gto_first { orb.aos.num_ao } else { 0 };
                for a in 0..grad[0].nrows() {
                    for e in 0..r.nrows() {
                        for c in 0..3 {
                            if matches!(coordinate, TiledCoordinate::Strain) {
                                for d in 0..3 {
                                    contribution[[c, d]] -=
                                        2.0 * cl[[a + extra_offset, e]] * h[c][d][[a, e]]
                                            + cg[d][[a + extra_offset, e]] * grad[c][[a, e]];
                                }
                            } else {
                                let mut field = cv[[a + extra_offset, e]] * grad[c][[a, e]]
                                    + cl[[a + extra_offset, e]] * gl[c][[a, e]];
                                for d in 0..3 {
                                    field += cg[d][[a + extra_offset, e]] * h[c][d][[a, e]];
                                }
                                contribution[[e, c]] += field;
                            }
                        }
                    }
                }
            }
            if block.conjugate {
                contribution.mapv_inplace(|v| v.conj());
            }
            result += &contribution;
        }
        result
    }

    pub fn value(&self, positions: &ArrayView2<f64>) -> Array2<Complex64> {
        let mut result = Array2::zeros((self.occupied, positions.nrows()));
        let mut offset = 0;
        for block in &self.blocks {
            if block.occupied == 0 {
                continue;
            }
            result
                .slice_mut(s![offset..offset + block.occupied, ..])
                .assign(&block.contract(block.backend.value(positions), self.normalization));
            offset += block.occupied;
        }
        result
    }

    /// Final (value, gx, gy, gz, lap) rows. Only block-local bases are evaluated.
    pub fn value_grad_lap(&self, positions: &ArrayView2<f64>) -> [Array2<Complex64>; 5] {
        let mut result = std::array::from_fn(|_| Array2::zeros((self.occupied, positions.nrows())));
        let mut offset = 0;
        for block in &self.blocks {
            // Retain empty k blocks in metadata, but do not allocate their AO
            // evaluation when they contribute no occupied rows.
            if block.occupied == 0 {
                continue;
            }
            let (v, [gx, gy, gz], lap) = block.backend.value_grad_lap(positions);
            for (output, local) in result.iter_mut().zip([v, gx, gy, gz, lap]) {
                output
                    .slice_mut(s![offset..offset + block.occupied, ..])
                    .assign(&block.contract(local, self.normalization));
            }
            offset += block.occupied;
        }
        result
    }
}

/// `(atoms, strain, electrons)` cotangent responses of [`py_compute_tiled_orbitals_vjp`].
type VjpArrays<'py> = (
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
);

type VglArrays<'py> = (
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
);

/// Evaluate the common tiled provider without an identity-MO compatibility wrapper.
#[pyfunction(name = "compute_tiled_orbitals_vgl")]
pub fn py_compute_tiled_orbitals_vgl<'py>(
    py: Python<'py>,
    orbitals: &Bound<'py, PyAny>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<VglArrays<'py>> {
    let model = TiledOrbitalSet::from_py(orbitals)?;
    if positions.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "electron positions must have shape (N_e, 3)",
        ));
    }
    let [v, gx, gy, gz, lap] = model.value_grad_lap(&positions.as_array());
    Ok((
        v.into_pyarray(py),
        gx.into_pyarray(py),
        gy.into_pyarray(py),
        gz.into_pyarray(py),
        lap.into_pyarray(py),
    ))
}
/// Native analytic VGL cotangent responses in physical coordinates.
#[pyfunction(name = "compute_tiled_orbitals_vjp")]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
pub fn py_compute_tiled_orbitals_vjp<'py>(
    py: Python<'py>,
    orbitals: &Bound<'_, PyAny>,
    positions: PyReadonlyArray2<'py, f64>,
    bar_value: PyReadonlyArray2<'py, Complex64>,
    bar_gx: PyReadonlyArray2<'py, Complex64>,
    bar_gy: PyReadonlyArray2<'py, Complex64>,
    bar_gz: PyReadonlyArray2<'py, Complex64>,
    bar_lap: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<VjpArrays<'py>> {
    let model = TiledOrbitalSet::from_py(orbitals)?;
    if positions.shape()[1] != 3
        || [&bar_value, &bar_gx, &bar_gy, &bar_gz, &bar_lap]
            .iter()
            .any(|v| v.shape() != [model.occupied, positions.shape()[0]])
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tiled VJP requires positions (N,3) and each cotangent (occupied,N)",
        ));
    }
    let value = bar_value.as_array().to_owned();
    let gradient = [
        bar_gx.as_array().to_owned(),
        bar_gy.as_array().to_owned(),
        bar_gz.as_array().to_owned(),
    ];
    let laplacian = bar_lap.as_array().to_owned();
    let response = |coordinate| {
        model.field_vjp(
            &positions.as_array(),
            &value,
            Some(&gradient),
            Some(&laplacian),
            coordinate,
        )
    };
    Ok((
        response(TiledCoordinate::Atoms).into_pyarray(py),
        response(TiledCoordinate::Strain).into_pyarray(py),
        response(TiledCoordinate::Electrons).into_pyarray(py),
    ))
}
