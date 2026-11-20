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

//! Native CPU kernels for jQMC.
//!
//! This crate ports autograd-free and RNG-free value functions from the Python
//! package. Module layout mirrors `jqmc/*.py` so ownership remains explicit.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// Thread-local-heap allocator: under `mpirun -np 30` the e_L's transient GEMM /
// Vec allocations otherwise serialize on glibc's arena lock across ranks.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[allow(non_upper_case_globals)]
pub mod setting_consts {
    include!(concat!(env!("OUT_DIR"), "/setting_consts.rs"));
}

#[cfg(test)]
fn central_setting_f64(name: &str) -> f64 {
    setting_consts::MANIFEST
        .iter()
        .find(|(key, _)| *key == name)
        .unwrap_or_else(|| panic!("unknown central setting: {name}"))
        .1
}

pub mod ao_cart_deriv;
pub mod atomic_orbital;
pub mod bspline;
pub mod orbital_tiling;
use orbital_tiling::{py_compute_tiled_orbitals_vgl, py_compute_tiled_orbitals_vjp};
pub mod complex_ao;
pub mod complex_det;
pub mod complex_div;
pub mod complex_ecp;
pub mod complex_stress;
pub mod coulomb_potential;
pub mod data;
pub mod determinant;
pub mod dtn;
pub mod ewald;
pub mod fastmath;
pub mod forces;
pub mod forces_dtn;
pub mod forces_ecp;
pub mod forces_ewald;
pub mod forces_pbc;
pub mod gfmc;
pub mod gfmc_forces;
pub mod hamiltonians;
pub mod jastrow_factor;
pub mod mcmc;
pub mod mesh_vgs;
pub mod mpc;
pub mod native_driver;
pub mod optimizer_linalg;
pub mod optimizer_transform;
pub mod param_grad;
pub mod param_grad_basis;
pub mod param_grad_dtn;
pub mod param_grad_ecp;
pub mod param_grad_el;
pub mod param_grad_twist;
pub mod prof;
pub mod stress;
pub mod stress_pbc;
pub mod threefry;
pub mod walker_batch;
pub mod wavefunction;

use numpy::IntoPyArray;

/// Four / five Python objects returned together by a PyO3 entry point.
type PyObj4 = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);
type PyObj5 = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

// --- PyO3 wrappers: thin zero-copy boundary over the pure-Rust core. ---------
// Names match `jqmc/coulomb_potential.py`.

#[pyfunction]
#[pyo3(name = "compute_bare_coulomb_potential_el_el")]
fn py_compute_bare_coulomb_potential_el_el(
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> f64 {
    coulomb_potential::compute_bare_coulomb_potential_el_el(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    )
}

#[pyfunction]
#[pyo3(name = "compute_bare_coulomb_potential_ion_ion")]
fn py_compute_bare_coulomb_potential_ion_ion(
    r_carts: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
) -> f64 {
    coulomb_potential::compute_bare_coulomb_potential_ion_ion(
        &r_carts.as_array(),
        charges.as_slice().expect("charges must be contiguous"),
    )
}

#[pyfunction]
#[pyo3(name = "compute_bare_coulomb_potential_el_ion_element_wise")]
fn py_compute_bare_coulomb_potential_el_ion_element_wise(
    r_carts: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    coulomb_potential::compute_bare_coulomb_potential_el_ion_element_wise(
        &r_carts.as_array(),
        charges.as_slice().expect("charges must be contiguous"),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    )
}

#[pyfunction]
#[pyo3(name = "compute_bare_coulomb_potential")]
fn py_compute_bare_coulomb_potential(
    r_carts: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> f64 {
    coulomb_potential::compute_bare_coulomb_potential(
        &r_carts.as_array(),
        charges.as_slice().expect("charges must be contiguous"),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    )
}

// --- Jastrow value functions. Take the Python dataclass object directly and
// build the Rust struct through `from_py`. -------------------------------------------

/// DEBUG: J1/J2 cell-straining kinetic stress with an
/// ARBITRARY fixed cotangent, to isolate the J1/J2 kernel from the wf drift. (3x3.)
#[pyfunction]
#[pyo3(name = "_jastrow_kstress_debug", signature = (jastrow_data, cot, r_up_carts, r_dn_carts, with_j1, with_j2))]
fn py_jastrow_kstress_debug(
    py: Python<'_>,
    jastrow_data: &Bound<'_, PyAny>,
    cot: PyReadonlyArray2<f64>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    with_j1: bool,
    with_j2: bool,
) -> PyResult<Py<PyAny>> {
    let j1o = jastrow_data.getattr("jastrow_one_body_data")?;
    let j2o = jastrow_data.getattr("jastrow_two_body_data")?;
    let j1 = if with_j1 && !j1o.is_none() {
        Some(data::JastrowOneBodyData::from_py(&j1o)?)
    } else {
        None
    };
    let j2 = if with_j2 && !j2o.is_none() {
        Some(data::JastrowTwoBodyData::from_py(&j2o)?)
    } else {
        None
    };
    let mut out = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    let cot = cot.as_array().to_owned();
    stress_pbc::jastrow_kstress(
        j1.as_ref(),
        j2.as_ref(),
        &cot,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &mut out,
    );
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// DEBUG: Williamson smooth distance (d, g, q, L) for x, cell. For FD-checking the Hessian.
#[pyfunction]
#[pyo3(name = "_smooth_dist_debug")]
fn py_smooth_dist_debug(x: [f64; 3], cell: PyReadonlyArray2<f64>) -> (f64, [f64; 3], f64, f64) {
    let ci = jastrow_factor::CellInfo::new(&cell.as_array());
    jastrow_factor::smooth_dist(x, &ci)
}

/// DEBUG: Williamson smooth dist + Hessian + grad-Laplacian (d, g, q, L, H 3x3 flat, gL).
#[pyfunction]
#[pyo3(name = "_smooth_dist_hess_debug")]
fn py_smooth_dist_hess_debug(
    x: [f64; 3],
    cell: PyReadonlyArray2<f64>,
) -> (f64, [f64; 3], f64, f64, Vec<f64>, [f64; 3]) {
    let ci = jastrow_factor::CellInfo::new(&cell.as_array());
    let (d, g, q, l, h, gl) = jastrow_factor::smooth_dist_hess(x, &ci);
    let hflat = vec![
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    ];
    (d, g, q, l, hflat, gl)
}

#[pyfunction]
#[pyo3(name = "compute_Jastrow_one_body")]
fn py_compute_jastrow_one_body(
    jastrow_one_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let data = data::JastrowOneBodyData::from_py(jastrow_one_body_data)?;
    Ok(jastrow_factor::compute_jastrow_one_body(
        &data,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_Jastrow_two_body")]
fn py_compute_jastrow_two_body(
    jastrow_two_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let data = data::JastrowTwoBodyData::from_py(jastrow_two_body_data)?;
    Ok(jastrow_factor::compute_jastrow_two_body(
        &data,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_Jastrow_three_body_value")]
fn py_compute_jastrow_three_body_value(
    j_matrix: PyReadonlyArray2<f64>,
    aos_up: PyReadonlyArray2<f64>,
    aos_dn: PyReadonlyArray2<f64>,
) -> f64 {
    jastrow_factor::compute_jastrow_three_body_value(
        &j_matrix.as_array(),
        &aos_up.as_array(),
        &aos_dn.as_array(),
    )
}

#[pyfunction]
#[pyo3(name = "jastrow_1b2b_ratio")]
fn py_jastrow_1b2b_ratio(
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    is_up: bool,
    moved_index: usize,
    new_pos: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let p = new_pos.as_slice()?;
    Ok(jastrow_factor::jastrow_1b2b_ratio(
        &data,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        is_up,
        moved_index,
        [p[0], p[1], p[2]],
    ))
}

// --- J3 ratio streaming state. Orbital matrices are supplied by the caller. ---

// J3 symmetry collapses the state from 8 tables to 6 (w_up/w_dn each serve
// as both the j3*aos and j3^T*aos operand; each cross vec is that table's rowsum).
// The exported tuple mirrors the struct: (aos_up, aos_dn, w_up, w_dn, cross_up,
// cross_dn). The coupling block must satisfy the production symmetry contract.
fn _j3state_tuple(py: Python<'_>, s: &jastrow_factor::J3RatioState) -> Vec<Py<PyAny>> {
    vec![
        s.aos_up.clone().into_pyarray(py).into_any().unbind(),
        s.aos_dn.clone().into_pyarray(py).into_any().unbind(),
        s.w_up.clone().into_pyarray(py).into_any().unbind(),
        s.w_dn.clone().into_pyarray(py).into_any().unbind(),
        s.cross_up.clone().into_pyarray(py).into_any().unbind(),
        s.cross_dn.clone().into_pyarray(py).into_any().unbind(),
    ]
}

/// Fetch the fp64 storage-symmetry floor from
/// `jqmc._setting.{atol,rtol}_consistency` (never hardcode a tolerance) and assert
/// the J3 coupling block is symmetric. Called ONCE per run at native-ctx build.
fn check_j3_symmetric(py: Python<'_>, jm: &numpy::ndarray::ArrayView2<f64>) -> PyResult<()> {
    let setting = py.import("jqmc._setting")?;
    let atol: f64 = setting.getattr("atol_consistency")?.extract()?;
    let rtol: f64 = setting.getattr("rtol_consistency")?.extract()?;
    jastrow_factor::assert_j3_block_symmetric(jm, atol, rtol)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn _j3_init_state(
    py: Python<'_>,
    j_matrix: PyReadonlyArray2<f64>,
    aos_up: PyReadonlyArray2<f64>,
    aos_dn: PyReadonlyArray2<f64>,
) -> Vec<Py<PyAny>> {
    let s = jastrow_factor::init_j3_ratio_state(
        &j_matrix.as_array(),
        &aos_up.as_array(),
        &aos_dn.as_array(),
    );
    _j3state_tuple(py, &s)
}

#[pyfunction]
fn _j3_ratio(
    j_matrix: PyReadonlyArray2<f64>,
    aos_up: PyReadonlyArray2<f64>,
    aos_dn: PyReadonlyArray2<f64>,
    is_up: bool,
    moved_index: usize,
    aos_new_col: PyReadonlyArray1<f64>,
) -> f64 {
    let s = jastrow_factor::init_j3_ratio_state(
        &j_matrix.as_array(),
        &aos_up.as_array(),
        &aos_dn.as_array(),
    );
    jastrow_factor::j3_ratio(
        &j_matrix.as_array(),
        &s,
        is_up,
        moved_index,
        aos_new_col.as_slice().unwrap(),
    )
}

#[pyfunction]
fn _j3_advance_state(
    py: Python<'_>,
    j_matrix: PyReadonlyArray2<f64>,
    aos_up: PyReadonlyArray2<f64>,
    aos_dn: PyReadonlyArray2<f64>,
    is_up: bool,
    moved_index: usize,
    aos_new_col: PyReadonlyArray1<f64>,
) -> Vec<Py<PyAny>> {
    let mut s = jastrow_factor::init_j3_ratio_state(
        &j_matrix.as_array(),
        &aos_up.as_array(),
        &aos_dn.as_array(),
    );
    jastrow_factor::advance_j3_ratio_state(
        &j_matrix.as_array(),
        &mut s,
        is_up,
        moved_index,
        aos_new_col.as_slice().unwrap(),
    );
    _j3state_tuple(py, &s)
}

#[pyfunction]
#[pyo3(name = "pgto_cart_deriv")]
fn py_pgto_cart_deriv(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
    ax: u32,
    ay: u32,
    az: u32,
) -> PyResult<Py<PyAny>> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let out = ao_cart_deriv::pgto_deriv(&data, &r_carts.as_array(), &cell.as_array(), (ax, ay, az));
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "ao_cart_deriv")]
fn py_ao_cart_deriv(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
    ax: u32,
    ay: u32,
    az: u32,
) -> PyResult<Py<PyAny>> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let out = ao_cart_deriv::ao_deriv(&data, &r_carts.as_array(), (ax, ay, az));
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "compute_AOs")]
fn py_compute_aos(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    // Default backend-parity convention: evaluate every image retained by the
    // build-time image set, exactly as XLA does.
    let out = atomic_orbital::compute_aos(&data, &r_carts.as_array());
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// TEST-ONLY binding (parity net): the EXACT (un-screened) batch AO value --
/// the OBSERVABLE convention (`compute_aos_exact`, what `orb_value_grad_lap`
/// and every e_L/param-grad/force aggregate evaluates internally). Not on any
/// production dispatch path itself; exists so the Rust==XLA AO-leaf parity can
/// pin the observable convention at strict tolerance on anisotropic/slab cells.
#[pyfunction]
#[pyo3(name = "compute_AOs_exact")]
fn py_compute_aos_exact(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let out = atomic_orbital::compute_aos_exact(&data, &r_carts.as_array());
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// TEST-ONLY binding (parity net): the EXACT (un-screened) batch AO gradient --
/// the observable convention (`compute_aos_grad_exact`, used by the density-grid
/// consumers). See `compute_AOs_exact` above for why this exists.
#[pyfunction]
#[pyo3(name = "compute_AOs_grad_exact")]
fn py_compute_aos_grad_exact(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let (gx, gy, gz) = atomic_orbital::compute_aos_grad_exact(&data, &r_carts.as_array());
    Ok((
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
    ))
}

/// TEST-ONLY binding (issue #87): the FAST-mode screened AO gradient and
/// Laplacian, `(gx, gy, gz, lap)` each `(num_ao, n_points)`, through one of the
/// three production derivative-screen kernels:
/// `route="batch"` -> `compute_aos_grad_lap_fast_cut` (FAST chunk kinetic, J3),
/// `route="single"` -> `compute_aos_single_value_grad_lap_fast_cut_into`
/// (GFMC derivative-cache column update; molecular leaves only),
/// `route="folded"` -> `compute_aos_single_value_grad_lap_fast_cut_folded_into`
/// (image-folded column update; under PBC the points must already be wrapped
/// into the home cell). Not on any production dispatch path itself; exists so
/// the dropped-derivative bound of the screen can be pinned against the exact
/// XLA AO derivatives.
#[pyfunction]
#[pyo3(name = "compute_AOs_grad_lap_fast_cut")]
fn py_compute_aos_grad_lap_fast_cut(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
    route: &str,
) -> PyResult<PyObj4> {
    use numpy::ndarray::Array2;
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let r = r_carts.as_array();
    let (gx, gy, gz, lap) = match route {
        "batch" => atomic_orbital::compute_aos_grad_lap_fast_cut(&data, &r),
        "single" | "folded" => {
            if data.shells.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "compute_AOs_grad_lap_fast_cut: single-point routes need a GTO leaf with shells",
                ));
            }
            let single = route == "single";
            if single && data.n_image != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "compute_AOs_grad_lap_fast_cut: route='single' is molecular-only (use 'folded')",
                ));
            }
            let rows = if single {
                data.num_ao_kernel
            } else {
                data.num_ao
            };
            let n = r.nrows();
            let mut out = [
                Array2::<f64>::zeros((rows, n)),
                Array2::<f64>::zeros((rows, n)),
                Array2::<f64>::zeros((rows, n)),
                Array2::<f64>::zeros((rows, n)),
            ];
            let mut v = vec![0.0; rows];
            let mut bx = vec![0.0; rows];
            let mut by = vec![0.0; rows];
            let mut bz = vec![0.0; rows];
            let mut bl = vec![0.0; rows];
            let mut scratch: Vec<f64> = Vec::new();
            for e in 0..n {
                let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
                if single {
                    atomic_orbital::compute_aos_single_value_grad_lap_fast_cut_into(
                        &data, pos, &mut v, &mut bx, &mut by, &mut bz, &mut bl,
                    );
                } else {
                    atomic_orbital::compute_aos_single_value_grad_lap_fast_cut_folded_into(
                        &data,
                        pos,
                        &mut scratch,
                        &mut v,
                        &mut bx,
                        &mut by,
                        &mut bz,
                        &mut bl,
                    );
                }
                for a in 0..rows {
                    out[0][[a, e]] = bx[a];
                    out[1][[a, e]] = by[a];
                    out[2][[a, e]] = bz[a];
                    out[3][[a, e]] = bl[a];
                }
            }
            let [ox, oy, oz, ol] = out;
            (ox, oy, oz, ol)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compute_AOs_grad_lap_fast_cut: unknown route {route:?} (batch | single | folded)"
            )))
        }
    };
    Ok((
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
        lap.into_pyarray(py).into_any().unbind(),
    ))
}

/// TEST-ONLY binding (issue #100): AO VALUES `(num_ao, n_points)` as seen
/// through one of the production FAST-mode value-screen kernels:
/// `route="mo_single"` -> `compute_mo_single_cutoff_into` (sphe) /
/// `compute_mo_single_cutoff_cart_into` (cart), the per-move determinant
/// orbital evaluation, projected with an identity coefficient matrix so the
/// output IS the screened AO vector (molecular leaves only);
/// `route="mesh_dot"` -> `dot_aos_multi_cutoff_sphe` / `dot_aos_multi_cutoff_cart`
/// (ECP non-local mesh AO contraction), contracted against each unit weight
/// vector in turn (home AO rank; all points form one mesh);
/// `route="dot_single"` -> `dot_aos_single_cutoff_sphe` (sphe only), unit weights;
/// `route="candidate"` -> the GFMC molecular candidate AO batch
/// (`gfmc::molecular_aos_candidate_batch`; molecular leaves only).
/// Not on any production dispatch path itself; exists so the dropped-value bound
/// of the screens can be pinned against the exact XLA AO values.
#[pyfunction]
#[pyo3(name = "compute_AOs_value_fast_cut")]
fn py_compute_aos_value_fast_cut(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
    route: &str,
) -> PyResult<Py<PyAny>> {
    use numpy::ndarray::Array2;
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    if data.shells.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "compute_AOs_value_fast_cut: needs a GTO leaf with shells",
        ));
    }
    let r = r_carts.as_array();
    let n = r.nrows();
    let num_ao = data.num_ao;
    let molecular_only = |name: &str| -> PyResult<()> {
        if data.n_image != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compute_AOs_value_fast_cut: route={name:?} is molecular-only"
            )));
        }
        Ok(())
    };
    let mut out = Array2::<f64>::zeros((num_ao, n));
    match route {
        "mo_single" => {
            molecular_only(route)?;
            let mut c_t = vec![0.0_f64; num_ao * num_ao];
            for a in 0..num_ao {
                c_t[a * num_ao + a] = 1.0;
            }
            let mut mo = vec![0.0_f64; num_ao];
            for e in 0..n {
                let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
                match data.kind {
                    atomic_orbital::AoKind::Sphe => atomic_orbital::compute_mo_single_cutoff_into(
                        &data, &c_t, num_ao, pos, &mut mo,
                    ),
                    atomic_orbital::AoKind::Cart => {
                        atomic_orbital::compute_mo_single_cutoff_cart_into(
                            &data, &c_t, num_ao, pos, &mut mo,
                        )
                    }
                }
                for a in 0..num_ao {
                    out[[a, e]] = mo[a];
                }
            }
        }
        "single_f32" => {
            // The per-move FAST orbital route of an fp32 `ao_eval` zone
            // (`eval_orb_into`): the screened single-position fp32 kernel
            // (molecular or periodic, image-folded).
            let mut ao = vec![0.0_f64; num_ao];
            for e in 0..n {
                let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
                atomic_orbital::compute_aos_single_f32_into(&data, pos, true, &mut ao);
                for a in 0..num_ao {
                    out[[a, e]] = ao[a];
                }
            }
        }
        "mesh_dot" => {
            let mut w = vec![0.0_f64; num_ao];
            let mut dots = vec![0.0_f64; n];
            for a in 0..num_ao {
                w.fill(0.0);
                w[a] = 1.0;
                match data.kind {
                    atomic_orbital::AoKind::Sphe => {
                        atomic_orbital::dot_aos_multi_cutoff_sphe(&data, &r, &w, &mut dots)
                    }
                    atomic_orbital::AoKind::Cart => {
                        atomic_orbital::dot_aos_multi_cutoff_cart(&data, &r, &w, &mut dots)
                    }
                }
                for e in 0..n {
                    out[[a, e]] = dots[e];
                }
            }
        }
        "dot_single" => {
            if data.kind != atomic_orbital::AoKind::Sphe {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "compute_AOs_value_fast_cut: route='dot_single' is sphe-only",
                ));
            }
            let mut w = vec![0.0_f64; num_ao];
            for a in 0..num_ao {
                w.fill(0.0);
                w[a] = 1.0;
                for e in 0..n {
                    let pos = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
                    out[[a, e]] = atomic_orbital::dot_aos_single_cutoff_sphe(&data, pos, &w);
                }
            }
        }
        "candidate" => {
            molecular_only(route)?;
            let positions: Vec<[f64; 3]> =
                (0..n).map(|e| [r[[e, 0]], r[[e, 1]], r[[e, 2]]]).collect();
            let batch = gfmc::molecular_aos_candidate_batch(&data, &positions);
            for e in 0..n {
                for a in 0..num_ao {
                    out[[a, e]] = batch[[e, a]];
                }
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compute_AOs_value_fast_cut: unknown route {route:?} \
                 (mo_single | single_f32 | mesh_dot | dot_single | candidate)"
            )))
        }
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "compute_AOs_grad")]
fn py_compute_aos_grad(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let (gx, gy, gz) = atomic_orbital::compute_aos_grad(&data, &r_carts.as_array());
    Ok((
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_AOs_laplacian")]
fn py_compute_aos_laplacian(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let data = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let lap = atomic_orbital::compute_aos_laplacian(&data, &r_carts.as_array());
    Ok(lap.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "compute_ln_det_geminal_all_elements")]
fn py_compute_ln_det_geminal_all_elements(
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    Ok(determinant::compute_ln_det_geminal_all_elements(
        &data,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_geminal_all_elements")]
fn py_compute_geminal_all_elements(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    let g = determinant::compute_geminal_all_elements(
        &data,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "compute_grads_and_laplacian_ln_det")]
fn py_compute_grads_and_laplacian_ln_det(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj4> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    let (gu, gd, lu, ld) = determinant::compute_grads_and_laplacian_ln_det(
        &data,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
        lu.into_pyarray(py).into_any().unbind(),
        ld.into_pyarray(py).into_any().unbind(),
    ))
}

/// Local-channel ECP inputs read from a `Coulomb_potential_data`: nucleus
/// positions, the flattened local terms, and the periodic cell with its
/// Wigner-Seitz `n_max` (`None` for molecules), so the standalone local-ECP
/// bindings take the same minimum image as the production callers.
type EcpLocalInputs = (
    numpy::ndarray::Array2<f64>,
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Option<(numpy::ndarray::Array2<f64>, i32)>,
);

fn ecp_local_inputs(cp: &Bound<'_, PyAny>) -> PyResult<EcpLocalInputs> {
    let sd = cp.getattr("structure_data")?;
    let pos_attr = sd.getattr("positions")?;
    let positions = pos_attr
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nuc: Vec<i64> = cp.getattr("_nucleus_index_local_part")?.extract()?;
    let exps: Vec<f64> = cp
        .getattr("_exponents_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let coefs: Vec<f64> = cp
        .getattr("_coefficients_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let powers: Vec<f64> = cp
        .getattr("_powers_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    let cell = if pbc {
        let np = cp.py().import("numpy")?;
        let c = np
            .call_method1("asarray", (sd.getattr("cell")?,))?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let n_max: i32 = sd.getattr("_n_max")?.extract()?;
        Some((c, n_max))
    } else {
        None
    };
    Ok((positions, nuc, exps, coefs, powers, cell))
}

#[pyfunction]
#[pyo3(name = "compute_ecp_local_parts_all_pairs")]
fn py_compute_ecp_local_parts_all_pairs(
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let (positions, nuc, exps, coefs, powers, cell) = ecp_local_inputs(coulomb_potential_data)?;
    let cell_view = cell.as_ref().map(|(c, n)| (c.view(), *n));
    Ok(coulomb_potential::compute_ecp_local_parts_all_pairs(
        &positions.view(),
        &nuc,
        &exps,
        &coefs,
        &powers,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        cell_view.as_ref().map(|(c, n)| (c, *n)),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_ewald_potential_ee")]
#[allow(clippy::too_many_arguments)]
fn py_compute_ewald_potential_ee(
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
) -> PyResult<f64> {
    Ok(ewald::compute_ewald_potential_ee(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        &cell.as_array(),
        n_max,
        None, // bit-exact test path: compute electron SF inline
    ))
}

#[pyfunction]
#[pyo3(name = "compute_ewald_potential_ei")]
#[allow(clippy::too_many_arguments)]
fn py_compute_ewald_potential_ei(
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
    z_ion: PyReadonlyArray1<f64>,
    r_ion: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    Ok(ewald::compute_ewald_potential_ei(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        &cell.as_array(),
        n_max,
        z_ion.as_slice()?,
        &r_ion.as_array(),
        None, // bit-exact test path: compute SI inline
        None, // and electron SF inline
    ))
}

/// Ion-ion Ewald energy through Rust for caller-supplied tables. Production
/// evaluates the same [`ewald::compute_ewald_potential_ii`] inside
/// [`ewald::EwaldData::from_py`] from tables rebuilt for the current structure;
/// this binding exposes it for independent XLA-vs-Rust parity checks of the
/// ion-ion channel.
#[pyfunction]
#[pyo3(name = "compute_ewald_potential_ii")]
#[allow(clippy::too_many_arguments)]
fn py_compute_ewald_potential_ii(
    charges: PyReadonlyArray1<f64>,
    positions: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    cell: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    Ok(ewald::compute_ewald_potential_ii(
        charges.as_slice()?,
        &positions.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        &cell.as_array(),
    ))
}

/// Per-electron Ewald ei (up then dn) for the LRDMC ZV; `alat=None` bare, `Some` discretized.
#[pyfunction]
#[pyo3(name = "_ewald_ei_per_electron", signature = (r_up_carts, r_dn_carts, eta, real_image_list, g_vectors, g_factors, cell, n_max, z_ion, r_ion, alat=None))]
#[allow(clippy::too_many_arguments)]
fn py_ewald_ei_per_electron(
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
    z_ion: PyReadonlyArray1<f64>,
    r_ion: PyReadonlyArray2<f64>,
    alat: Option<f64>,
) -> PyResult<Vec<f64>> {
    Ok(ewald::compute_ewald_potential_ei_per_electron(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        &cell.as_array(),
        n_max,
        z_ion.as_slice()?,
        &r_ion.as_array(),
        alat,
    ))
}

#[pyfunction]
fn _bench_faer_mo(geminal_data: &Bound<'_, PyAny>, n: usize) -> PyResult<(f64, f64)> {
    use std::time::Instant;
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let c = gem.orb_up.mo_coefficients.as_ref().expect("MO");
    let nmo = c.nrows();
    let na = c.ncols();
    let cs = c.as_slice().expect("C contiguous");
    let ao: Vec<f64> = (0..na)
        .map(|i| ((i * 7 % 13) as f64) * 0.01 + 0.001)
        .collect();
    let mut acc = 0.0_f64;
    // hand AVX-512
    let mut out = vec![0.0_f64; nmo];
    let t = Instant::now();
    for _ in 0..n {
        determinant::project_8acc(cs, &ao, na, nmo, &mut out);
        acc += out[0];
    }
    let avx_us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    // faer GEMV: C(nmo,na row-major) = column-major (na,nmo) = CT; out = CTT*ao
    let ct = faer::MatRef::from_column_major_slice(cs, na, nmo);
    let aov = faer::MatRef::from_column_major_slice(&ao, na, 1);
    let mut res = faer::Mat::<f64>::zeros(nmo, 1);
    let t = Instant::now();
    for _ in 0..n {
        faer::linalg::matmul::matmul(
            res.as_mut(),
            faer::Accum::Replace,
            ct.transpose(),
            aov,
            1.0_f64,
            faer::Par::Seq,
        );
        acc += res[(0, 0)];
    }
    let faer_us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    let _ = acc;
    Ok((avx_us, faer_us))
}

#[pyfunction]
#[pyo3(name = "compute_local_energy")]
#[allow(clippy::too_many_arguments)]
fn py_compute_local_energy(
    hamiltonian_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let h = hamiltonian_data;
    let wf = h.getattr("wavefunction_data")?;
    let gem = determinant::GeminalData::from_py(&wf.getattr("geminal_data")?)?;
    let jdata = data::JastrowData::from_py_complex_aware(&wf.getattr("jastrow_data")?)?;
    jdata.require_real_three_body("compute_local_energy (real)")?;
    let cp = h.getattr("coulomb_potential_data")?;
    let r_up = r_up_carts.as_array();
    let r_dn = r_dn_carts.as_array();

    // T (kinetic, incl. J3 via `jdata.three_body` if present)
    let t = wavefunction::compute_kinetic_energy(&gem, &jdata, &r_up, &r_dn);

    let sd = cp.getattr("structure_data")?;
    let pos_attr = sd.getattr("positions")?;
    let positions = pos_attr
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;

    // PBC: cell + WS n_max for the periodic min-image (None => molecular).
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    let py = h.py();
    let np = py.import("numpy")?;
    let as_arr2 = |o: Bound<'_, PyAny>| -> PyResult<numpy::ndarray::Array2<f64>> {
        Ok(np
            .call_method1("asarray", (o,))?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned())
    };
    let n_max: i32 = if pbc {
        sd.getattr("_n_max")?.extract()?
    } else {
        0
    };
    let cell_owned = if pbc {
        as_arr2(sd.getattr("cell")?)?
    } else {
        numpy::ndarray::Array2::<f64>::zeros((3, 3))
    };
    let cell_view = cell_owned.view();
    let cell_nmax = if pbc { Some((&cell_view, n_max)) } else { None };

    // V = bare Coulomb (molecular) OR Ewald ee+ei+ii (periodic).
    let mut v = if pbc {
        // Tables + ii derived from the CURRENT ewald_data.structure_data (issue #373),
        // the same single parse every native context uses.
        let ew = ewald::EwaldData::from_py(&cp)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
        })?;
        let ee = ewald::compute_ewald_potential_ee(
            &r_up,
            &r_dn,
            ew.eta,
            &ew.real_images.view(),
            &ew.g_vectors.view(),
            &ew.g_factors,
            &ew.cell.view(),
            ew.n_max,
            None,
        );
        let ei = ewald::compute_ewald_potential_ei(
            &r_up,
            &r_dn,
            ew.eta,
            &ew.real_images.view(),
            &ew.g_vectors.view(),
            &ew.g_factors,
            &ew.cell.view(),
            ew.n_max,
            &eff_charges,
            &positions.view(),
            None,
            None,
        );
        ee + ei + ew.ii
    } else {
        coulomb_potential::compute_bare_coulomb_potential(
            &positions.view(),
            &eff_charges,
            &r_up,
            &r_dn,
        )
    };

    // ECP local + non-local (periodic uses the min-image distance via cell_nmax).
    let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
    if ecp_flag {
        let nuc_l: Vec<i64> = cp.getattr("_nucleus_index_local_part")?.extract()?;
        let e_l: Vec<f64> = cp
            .getattr("_exponents_local_part")?
            .call_method0("tolist")?
            .extract()?;
        let c_l: Vec<f64> = cp
            .getattr("_coefficients_local_part")?
            .call_method0("tolist")?
            .extract()?;
        let p_l: Vec<f64> = cp
            .getattr("_powers_local_part")?
            .call_method0("tolist")?
            .extract()?;
        v += coulomb_potential::compute_ecp_local_parts_all_pairs(
            &positions.view(),
            &nuc_l,
            &e_l,
            &c_l,
            &p_l,
            &r_up,
            &r_dn,
            cell_nmax,
        );
        let nuc_n: Vec<i64> = cp.getattr("_nucleus_index_non_local_part")?.extract()?;
        let am: Vec<i64> = cp.getattr("_ang_mom_non_local_part")?.extract()?;
        let e_n: Vec<f64> = cp
            .getattr("_exponents_non_local_part")?
            .call_method0("tolist")?
            .extract()?;
        let c_n: Vec<f64> = cp
            .getattr("_coefficients_non_local_part")?
            .call_method0("tolist")?
            .extract()?;
        let p_n: Vec<f64> = cp
            .getattr("_powers_non_local_part")?
            .call_method0("tolist")?
            .extract()?;
        // Base geminal inverse once -> Sherman-Morrison mesh det ratios (O(N) per
        // mesh point instead of an O(N^3) full det), matching jqmc's fast e_L.
        let g_base = determinant::compute_geminal_all_elements(&gem, &r_up, &r_dn);
        let g_inv = determinant::inverse(&g_base.view());
        // J3 for the ECP-nonlocal ratio, from the single `three_body` field.
        let j3_data = jdata.three_body.as_ref().map(|j3| j3.as_ref());
        let j3_ref = j3_data.as_ref().map(|(o, m)| (*o, m));
        v += coulomb_potential::compute_ecp_nonlocal_parts_all_pairs(
            &gem,
            &jdata,
            &r_up,
            &r_dn,
            &positions.view(),
            &nuc_n,
            &am,
            &e_n,
            &c_n,
            &p_n,
            weights.as_slice()?,
            &grid_points.as_array(),
            false,
            1, // NN_default = 1
            j3_ref,
            cell_nmax,
            Some(&g_inv.view()),
            false, // bit-exact e_L (no mesh cutoff)
            None,
            None, // per-step/bit-exact: no sweep J3 state reuse (rebuild)
        );
    }

    Ok(t + v)
}

/// Local energy reusing a caller-supplied geminal inverse. Identical to
/// `compute_local_energy` except both the
/// kinetic and the ECP non-local Sherman-Morrison mesh ratios reuse
/// `geminal_inv == G(r_up, r_dn)^-1` (the MCMC sweep's running SM inverse)
/// instead of a fresh LU. This is the 1:1 of jqmc `compute_local_energy_fast`:
/// the fresh LU NaNs at a near-singular geminal, the AS-regularized SM inverse
/// is stable. The pure-Rust implementation performs no XLA/LAPACK lowering, so
/// it cannot enter `lapack_dgesdd_ffi` from this route.
///
/// When `mpc_data` is `Some` for a periodic MPC run,
/// the Ewald e-e channel is swapped for the finite-size-corrected MPC e-e term
/// (1:1 jqmc `compute_local_energy_fast_mpc`): the SAME kinetic (SM inverse) + ECP
/// pieces are kept, only the Ewald ee is skipped and `E_mpc_ee` is added. The
/// built `MPC_data` (from `build_mpc_data`, marshalled per-run like the ham) is
/// read inline via `mpc_ee_from_py`. `None` -> the plain Ewald/bare path.
#[pyfunction]
#[pyo3(name = "compute_local_energy_fast")]
#[pyo3(signature = (hamiltonian_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
#[allow(clippy::too_many_arguments)]
fn py_compute_local_energy_fast(
    hamiltonian_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let h = hamiltonian_data;
    let wf = h.getattr("wavefunction_data")?;
    let a = ClefArgs::from_py(
        &wf.getattr("geminal_data")?,
        &wf.getattr("jastrow_data")?,
        &h.getattr("coulomb_potential_data")?,
    )?;
    a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        &geminal_inv.as_array(),
        false, // per-step dispatch: exact mesh (bit-exact path)
        None,
        None, // per-step dispatch: no sweep J3 state reuse (rebuild)
        None,
        None,
    )
}

/// TEST-ONLY: the screened FAST chunk e_L of a periodic real-Gamma MO-GTO Hamiltonian at
/// `(r_up, r_dn)`, built from scratch as the chunk builds it: orbital values from the FAST
/// screened single-electron evaluator (`mcmc::fast_orb_matrix`), the determinant state and
/// inverse from those values, the FAST folded kinetic tables
/// (`init_det_grad_lap_cache_folded`) and the ECP mesh AO cutoff. Returns
/// `(e_L, geminal_inv)`. A definite function of the configuration, so finite differences of it
/// are the reference for the periodic force / stress kernels, which differentiate this
/// screened energy. Not on any production dispatch path.
#[pyfunction]
#[pyo3(name = "_e_L_fast_screened")]
fn py_e_l_fast_screened(
    py: Python<'_>,
    hamiltonian_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(f64, Py<PyAny>)> {
    let h = hamiltonian_data;
    let wf = h.getattr("wavefunction_data")?;
    let a = ClefArgs::from_py(
        &wf.getattr("geminal_data")?,
        &wf.getattr("jastrow_data")?,
        &h.getattr("coulomb_potential_data")?,
    )?;
    let gem = &a.gem;
    let (ru, rd) = (r_up_carts.as_array(), r_dn_carts.as_array());
    let ruw = atomic_orbital::wrap_r(&gem.orb_up.aos, &ru);
    let rdw = atomic_orbital::wrap_r(&gem.orb_dn.aos, &rd);
    let st = determinant::init_det_ratio_state_from_orb_mats(
        gem,
        mcmc::fast_orb_matrix(&gem.orb_up, &ruw.view()),
        mcmc::fast_orb_matrix(&gem.orb_dn, &rdw.view()),
    );
    let g = determinant::geminal_from_det_ratio_state(gem, &st);
    let gi = determinant::inverse(&g.view());
    let kin = determinant::init_det_grad_lap_cache_folded(gem, &ruw.view(), &rdw.view());
    let e = a.compute(
        &ru,
        &rd,
        weights.as_slice()?,
        &grid_points.as_array(),
        &gi.view(),
        true,
        Some(&st),
        None,
        Some(&kin),
        None,
    )?;
    Ok((e, gi.into_pyarray(py).into_any().unbind()))
}

/// Complex-twist (k != Gamma) local energy reusing a caller-supplied complex
/// geminal inverse -- the complex sibling of
/// `compute_local_energy_fast`, returning `(re, im)`. The det ratio is complex at
/// a twist, so the kinetic (`compute_kinetic_withj_complex_with_inv`) and the ECP
/// non-local (`compute_ecp_nonlocal_withj_complex`, a complex det ratio) are
/// complex; the Ewald e-e/e-i/i-i and ECP-local are REAL distance functions
/// (identical to the real path, no complex analog). Reuses the running complex SM
/// inverse the MCMC path carries (the fresh `inverse_complex` NaNs at a
/// near-singular geminal). Unlike the real periodic path this needs NO
/// `via_single`: the complex batch AO home-cell-wraps internally (its `WrapCtx`),
/// so the AO value (which `g_inv` = inv(JAX `compute_geminal_all_elements`) was
/// built from) and the AO grad/laplacian share the same full-image convention --
/// verified far-from-home vs the JAX complex kinetic. Watches the complex-twist
/// Re-projection trap: T and V stay complex through the sum, never Re-projected
/// factor-by-factor (Re(A)Re(B) != Re(A*B)). RAW coordinates throughout (jqmc's
/// non-invariant reference); the AO evals wrap internally, the Jastrow is min-image.
///
/// When `mpc_data` is `Some` for periodic MPC at a twist, the real Ewald e-e
/// channel is swapped for the finite-size-corrected
/// (REAL) `E_mpc_ee`, added into `v_re`; the complex kinetic + complex ECP
/// non-local are unchanged, so `Im(e_L) == Im(T + V_nl)`. The MPC e-e term is
/// a real distance function (no complex analog), consistent with jqmc's
/// dtype-promoted `compute_local_energy_fast_mpc` on a complex Bloch twist.
#[pyfunction]
#[pyo3(name = "compute_local_energy_fast_complex")]
#[pyo3(signature = (hamiltonian_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
#[allow(clippy::too_many_arguments)]
fn py_compute_local_energy_fast_complex(
    hamiltonian_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<numpy::Complex64>,
) -> PyResult<(f64, f64)> {
    let h = hamiltonian_data;
    let wf = h.getattr("wavefunction_data")?;
    let a = ClefComplexArgs::from_py(
        &wf.getattr("geminal_data")?,
        &wf.getattr("jastrow_data")?,
        &h.getattr("coulomb_potential_data")?,
    )?;
    a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        &geminal_inv.as_array(),
        false,
        None,
    )
}

#[pyfunction]
#[pyo3(name = "compute_dtn_value")]
fn py_compute_dtn_value(
    dtn_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let d = dtn::JastrowDtnData::from_py(dtn_data)?;
    Ok(dtn::compute_dtn_value(
        &d,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_dtn_ratio")]
fn py_compute_dtn_ratio(
    dtn_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    is_up: bool,
    idx: usize,
    new_pos: [f64; 3],
) -> PyResult<f64> {
    let d = dtn::JastrowDtnData::from_py(dtn_data)?;
    Ok(dtn::compute_dtn_ratio(
        &d,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        is_up,
        idx,
        new_pos,
    ))
}

#[pyfunction]
#[pyo3(name = "compute_grads_and_laplacian_dtn")]
fn py_compute_grads_and_laplacian_dtn(
    py: Python<'_>,
    dtn_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj4> {
    let d = dtn::JastrowDtnData::from_py(dtn_data)?;
    let (gu, gd, lu, ld) =
        dtn::compute_grads_and_laplacian_dtn(&d, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
        numpy::ndarray::Array1::from(lu)
            .into_pyarray(py)
            .into_any()
            .unbind(),
        numpy::ndarray::Array1::from(ld)
            .into_pyarray(py)
            .into_any()
            .unbind(),
    ))
}

/// Test probe of the projection-scoped DTN grad/Laplacian carry (`DtnGradLapCache`):
/// the seed at `(r_up, r_dn)` and the carried tables after the single-electron hop of
/// electron `index` (spin `is_up`) to `new_pos`, as two `(grad_up, grad_dn, lap_up,
/// lap_dn)` tuples. The carry is in the fp64 `jastrow_grad_lap_carry` zone with
/// carry-zone coefficients (issue #781), so its zone-isolation detector can read it
/// without the GFMC projection that owns it.
#[pyfunction]
#[pyo3(name = "_dtn_grad_lap_carry_probe")]
#[allow(clippy::type_complexity)]
fn py_dtn_grad_lap_carry_probe(
    py: Python<'_>,
    dtn_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    is_up: bool,
    index: usize,
    new_pos: [f64; 3],
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let d = dtn::JastrowDtnData::from_py(dtn_data)?;
    let mut r_up = r_up_carts.as_array().to_owned();
    let mut r_dn = r_dn_carts.as_array().to_owned();
    let n = if is_up { r_up.nrows() } else { r_dn.nrows() };
    if index >= n {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "moved electron index out of range",
        ));
    }
    let mut cache = dtn::init_dtn_grad_lap_cache(&d, &r_up.view(), &r_dn.view());
    let pack = |c: &dtn::DtnGradLapCache| -> PyResult<Py<PyAny>> {
        let (gu, gd, lu, ld) = dtn::clone_dtn_grad_lap_cache(c);
        let t = (
            gu.into_pyarray(py),
            gd.into_pyarray(py),
            numpy::ndarray::Array1::from(lu).into_pyarray(py),
            numpy::ndarray::Array1::from(ld).into_pyarray(py),
        );
        Ok(t.into_pyobject(py)?.into_any().unbind())
    };
    let seed = pack(&cache)?;
    let moved = if is_up { &mut r_up } else { &mut r_dn };
    let old_pos = [moved[[index, 0]], moved[[index, 1]], moved[[index, 2]]];
    for c in 0..3 {
        moved[[index, c]] = new_pos[c];
    }
    dtn::update_dtn_grad_lap_cache(
        &d,
        &mut cache,
        &r_up.view(),
        &r_dn.view(),
        is_up,
        index,
        old_pos,
    );
    Ok((seed, pack(&cache)?))
}

/// Parse the J3 `(orb_data, j_matrix)` pair that the free real wrappers thread explicitly
/// next to `JastrowData`. An INACTIVE J3 (all-zero `j_matrix`) is identically zero and is
/// skipped exactly as `data::JastrowData::from_py` skips it, so its orbital is never parsed
/// or evaluated: re-reading it unconditionally made the real `OrbData` parser drop a complex
/// plane-wave child of a composite J3 orbital and `matmul` overrun (issues #140, #170). An
/// active J3 is parsed as before; one whose orbital the real parser cannot represent raises
/// `NotImplementedError` (as `JastrowData::require_real_three_body`, #93) instead of being
/// truncated. For value, energy and non-J3-derivative consumers only; a J3 `j_matrix`
/// derivative uses [`parse_explicit_real_j3_for_jmatrix_deriv`].
fn parse_explicit_real_j3(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    split_real_j3(parse_real_j3_parts(
        &jastrow_data.getattr("jastrow_three_body_data")?,
        false,
    )?)
}

/// [`parse_explicit_real_j3`] for a J3 `j_matrix`-derivative consumer: the derivative of an
/// inactive J3 is NOT zero (it is a product of J3 orbital values), so an inactive J3 is kept
/// and parsed like an active one, and an orbital the real parser cannot represent raises
/// `NotImplementedError` whatever the `j_matrix` (issue #170), pointing to the JAX route.
fn parse_explicit_real_j3_for_jmatrix_deriv(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    split_real_j3(parse_real_j3_parts(
        &jastrow_data.getattr("jastrow_three_body_data")?,
        true,
    )?)
}

#[allow(clippy::type_complexity)]
fn split_real_j3(
    parts: Option<(determinant::OrbData, numpy::ndarray::Array2<f64>)>,
) -> PyResult<(
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    Ok(match parts {
        Some((orb, jm)) => (Some(orb), Some(jm)),
        None => (None, None),
    })
}

/// The one real-path J3 parse shared by the free wrappers (issue #170): `j3d` is a
/// `Jastrow_three_body_data` or `None`. `j_matrix` is coerced to float64 exactly as
/// `JastrowData::from_py` coerces it. Returns `None` for no J3 and, unless
/// `for_jmatrix_deriv`, for an inactive (all-zero `j_matrix`) J3.
fn parse_real_j3_parts(
    j3d: &Bound<'_, PyAny>,
    for_jmatrix_deriv: bool,
) -> PyResult<Option<(determinant::OrbData, numpy::ndarray::Array2<f64>)>> {
    if j3d.is_none() {
        return Ok(None);
    }
    let np = j3d.py().import("numpy")?;
    let jm: numpy::PyReadonlyArray2<f64> = np
        .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
        .call_method1("astype", (np.getattr("float64")?,))?
        .extract()?;
    let jm = jm.as_array().to_owned();
    let inactive = jm.iter().all(|&x| x == 0.0);
    if inactive && !for_jmatrix_deriv {
        return Ok(None);
    }
    let orb_py = j3d.getattr("orb_data")?;
    if !data::j3_orbital_real_representable(&orb_py)? {
        let what = if inactive {
            "its j_matrix derivative is nonzero even for an all-zero j_matrix"
        } else {
            "refusing to evaluate the wavefunction without its J3"
        };
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "real native kernel: the J3 orbital (complex plane-wave / B-spline basis or a complex \
             MO) is supported only by the complex native path, but the determinant selects the \
             real path; {what}. Disable flag_rust_on_CPU to use the JAX path for this family."
        )));
    }
    let orb = determinant::OrbData::from_py(&orb_py)?;
    Ok(Some((orb, jm)))
}

/// `parse_jastrow_full` for the free real value / energy / non-J3-derivative wrappers: the
/// `JastrowData` plus the explicit J3 pair of [`parse_explicit_real_j3`] (an inactive J3 is
/// skipped, an unrepresentable active one refused). A J3 `j_matrix`-derivative consumer must
/// not use it (issue #170).
#[allow(clippy::type_complexity)]
fn parse_jastrow_real_value(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<data::JastrowData>,
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    if jastrow_data.is_none() {
        return Ok((None, None, None));
    }
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    Ok((Some(jas), j3_orb, j3_jm))
}

#[pyfunction]
#[pyo3(name = "compute_ecp_nonlocal_parts_all_pairs", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, det_only, nn=0))]
#[allow(clippy::too_many_arguments)]
fn py_compute_ecp_nonlocal_parts_all_pairs(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    det_only: bool,
    nn: usize,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos_attr = sd.getattr("positions")?;
    let positions = pos_attr
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nuc: Vec<i64> = cp.getattr("_nucleus_index_non_local_part")?.extract()?;
    let angmom: Vec<i64> = cp.getattr("_ang_mom_non_local_part")?.extract()?;
    let exps: Vec<f64> = cp
        .getattr("_exponents_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let coefs: Vec<f64> = cp
        .getattr("_coefficients_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let powers: Vec<f64> = cp
        .getattr("_powers_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    // Periodic min-image when the cell is present (None => molecular).
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    let cell_owned = if pbc {
        let np = cp.py().import("numpy")?;
        Some(
            np.call_method1("asarray", (sd.getattr("cell")?,))?
                .extract::<numpy::PyReadonlyArray2<f64>>()?
                .as_array()
                .to_owned(),
        )
    } else {
        None
    };
    let n_max: i32 = if pbc {
        sd.getattr("_n_max")?.extract()?
    } else {
        0
    };
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    Ok(coulomb_potential::compute_ecp_nonlocal_parts_all_pairs(
        &gem,
        &jdata,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &positions.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        det_only,
        nn, // 0 => all-pairs (layer-1 test); >0 => NN nearest-nucleus filter
        j3_ref,
        cell_nmax,
        None,  // standalone wrapper keeps the exact full-recompute reference
        false, // bit-exact: no mesh AO cutoff
        None,
        None, // standalone wrapper: no sweep J3 state reuse (rebuild)
    ))
}

/// The FAST ECP non-local radial cut `coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT`
/// (Ha), exported only so a test can pin it equal to `jqmc._setting`.
#[pyfunction]
#[pyo3(name = "fast_ecp_nonlocal_radial_cut")]
fn py_fast_ecp_nonlocal_radial_cut() -> f64 {
    coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT
}

/// Test-only probe of the FAST ECP non-local route with an explicit radial cut:
/// the production FAST evaluation (Sherman-Morrison inverse of the base geminal,
/// mesh AO cutoff, NN nearest ECP nuclei) of `compute_ecp_nonlocal_parts_all_pairs`
/// with `radial_cut` (Ha) in place of `FAST_ECP_NONLOCAL_RADIAL_CUT`; `0.0` is the
/// uncut FAST value. Returns `(value, kept_pairs, dropped_pairs)`. Molecular only.
#[pyfunction]
#[pyo3(name = "_ecp_nonlocal_fast_radial_cut_probe", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, radial_cut, nn=1))]
#[allow(clippy::too_many_arguments)]
fn py_ecp_nonlocal_fast_radial_cut_probe(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    radial_cut: f64,
    nn: usize,
) -> PyResult<(f64, u64, u64)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    if pbc {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "_ecp_nonlocal_fast_radial_cut_probe: molecular input only",
        ));
    }
    if radial_cut < 0.0 || radial_cut.is_nan() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "_ecp_nonlocal_fast_radial_cut_probe: radial_cut must be >= 0",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nuc: Vec<i64> = cp.getattr("_nucleus_index_non_local_part")?.extract()?;
    let angmom: Vec<i64> = cp.getattr("_ang_mom_non_local_part")?.extract()?;
    let exps: Vec<f64> = cp
        .getattr("_exponents_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let coefs: Vec<f64> = cp
        .getattr("_coefficients_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let powers: Vec<f64> = cp
        .getattr("_powers_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let r_up = r_up_carts.as_array();
    let r_dn = r_dn_carts.as_array();
    let g_base = determinant::compute_geminal_all_elements(&gem, &r_up, &r_dn);
    let g_inv = determinant::inverse(&g_base.view());
    Ok(
        coulomb_potential::compute_ecp_nonlocal_parts_all_pairs_radial_cut(
            &gem,
            &jdata,
            &r_up,
            &r_dn,
            &positions.view(),
            &nuc,
            &angmom,
            &exps,
            &coefs,
            &powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            false,
            nn,
            j3_ref,
            None,
            Some(&g_inv.view()),
            true, // FAST route: mesh AO cutoff + w-dot tables
            None,
            None,
            radial_cut,
        ),
    )
}

#[pyfunction]
#[pyo3(name = "compute_kinetic_energy")]
fn py_compute_kinetic_energy(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    jdata.require_real_three_body("compute_kinetic_energy (real)")?;
    Ok(wavefunction::compute_kinetic_energy(
        &gem,
        &jdata,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

// ---- Parameter gradients O_k = dlnPsi/dc ----

#[pyfunction]
#[pyo3(name = "grad_ln_Psi_jastrow_1b_param")]
fn py_grad_j1_param(
    jastrow_one_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let d = data::JastrowOneBodyData::from_py(jastrow_one_body_data)?;
    Ok(param_grad::grad_j1_param(
        &d,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_ln_Psi_jastrow_2b_param")]
fn py_grad_j2_param(
    jastrow_two_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let d = data::JastrowTwoBodyData::from_py(jastrow_two_body_data)?;
    Ok(param_grad::grad_j2_param(
        &d,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_ln_Psi_jastrow_3b_jmatrix")]
fn py_grad_j3_jmatrix(
    py: Python<'_>,
    jastrow_three_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    // The `j_matrix` derivative is nonzero even for an all-zero `j_matrix`, so the J3 is kept
    // whatever its activity; an orbital the real parser cannot represent is refused (#170).
    let (orb, _) = parse_real_j3_parts(jastrow_three_body_data, true)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "grad_ln_Psi_jastrow_3b_jmatrix: jastrow_three_body_data is None",
        )
    })?;
    let g = param_grad::grad_j3_jmatrix(&orb, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_ln_det_wrt_lambda")]
fn py_grad_ln_det_lambda(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let g = param_grad::grad_ln_det_lambda(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_ln_det_wrt_mo_coeff")]
fn py_grad_ln_det_mo(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (gu, gd) = param_grad::grad_ln_det_mo(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// AGPn `O_k` on the `(lambda_diag, lambda_unpaired)` leaves (real Gamma path):
/// diagonal + unpaired slice of the dense-`lambda_matrix` gradient.
#[pyfunction]
#[pyo3(name = "grad_ln_det_wrt_lambda_diag")]
fn py_grad_ln_det_lambda_diag(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (d, u) =
        param_grad::grad_ln_det_lambda_diag(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

/// Complex Bloch AO values for a twisted (k!=Gamma) pgto leaf. Verifies the complex
/// core's AO layer against jqmc `_compute_AOs`.
#[pyfunction]
#[pyo3(name = "compute_aos_complex", signature = (aos_data, r_carts))]
fn py_compute_aos_complex(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let aos = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let np = py.import("numpy")?;
    let ipf: numpy::PyReadonlyArray1<numpy::Complex64> = np
        .call_method1("asarray", (aos_data.getattr("_image_phase_factors")?,))?
        .extract()?;
    let out = complex_ao::compute_aos_complex(&aos, ipf.as_slice()?, &r_carts.as_array());
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// B-spline AO value + Cartesian gradient + Laplacian, each `(num_ao, N_e)`. Verifies
/// the B-spline basis (`AOs_bspline_data`) against jqmc `_compute_AOs_value_grad_lap`.
#[pyfunction]
#[pyo3(name = "compute_aos_bspline_vgl", signature = (aos_data, r_carts))]
fn py_compute_aos_bspline_vgl(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj5> {
    let bs = bspline::AosBsplineData::from_py(aos_data)?;
    let (v, gx, gy, gz, l) = bs.value_grad_lap(&r_carts.as_array());
    Ok((
        v.into_pyarray(py).into_any().unbind(),
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
        l.into_pyarray(py).into_any().unbind(),
    ))
}

/// COMPLEX (twist) B-spline AO value + Cartesian gradient + Laplacian, each `(num_ao, N_e)`
/// complex128. Verifies `AOs_bspline_complex_data` against jqmc
/// `_compute_AOs_value_grad_lap` (Bloch-phase product rule on the real spline u).
#[pyfunction]
#[pyo3(name = "compute_aos_bspline_complex_vgl", signature = (aos_data, r_carts))]
fn py_compute_aos_bspline_complex_vgl(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj5> {
    let bs = bspline::AosBsplineComplexData::from_py(aos_data)?;
    let (v, g, l) = bs.value_grad_lap(&r_carts.as_array());
    let [gx, gy, gz] = g;
    Ok((
        v.into_pyarray(py).into_any().unbind(),
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
        l.into_pyarray(py).into_any().unbind(),
    ))
}

/// Verify the complex per-move Sherman-Morrison det ratio (1a foundation): build the
/// streaming state at (r_up,r_dn), G^-^1 = inverse_complex(G), then the rank-1 ratio for
/// moving electron `idx` (spin `is_up`) to `new_pos`. Returns `(re, im)` of `det_t` -- the
/// Python side compares to det_complex(G_moved)/det_complex(G_base).
#[pyfunction]
#[pyo3(name = "_verify_sm_ratio_complex", signature = (geminal_data, r_up_carts, r_dn_carts, is_up, idx, new_pos))]
fn py_verify_sm_ratio_complex(
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    is_up: bool,
    idx: usize,
    new_pos: [f64; 3],
) -> PyResult<(f64, f64)> {
    use numpy::Complex64;
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let r_up = r_up_carts.as_array();
    let r_dn = r_dn_carts.as_array();
    let state = complex_det::init_det_ratio_state_complex(&gem, &r_up, &r_dn);
    let g = complex_det::compute_geminal_complex(&gem, &r_up, &r_dn);
    let g_inv = complex_det::inverse_complex(&g);
    let n = g.nrows();
    let (u, v): (Vec<Complex64>, Vec<Complex64>) = if is_up {
        let ao_new = complex_det::orb_single_complex(&gem.orb_up, new_pos);
        let v = complex_det::compute_v_up_move_complex(&gem, &state, idx, &ao_new);
        let mut u = vec![Complex64::new(0.0, 0.0); n];
        u[idx] = Complex64::new(1.0, 0.0);
        (u, v)
    } else {
        let ao_new = complex_det::orb_single_complex(&gem.orb_dn, new_pos);
        let u = complex_det::compute_u_dn_move_complex(&state, idx, &ao_new);
        let mut v = vec![Complex64::new(0.0, 0.0); n];
        v[idx] = Complex64::new(1.0, 0.0);
        (u, v)
    };
    let mut ainv_u = vec![Complex64::new(0.0, 0.0); n];
    let mut vt_ainv = vec![Complex64::new(0.0, 0.0); n];
    let r = complex_det::sm_ratio_complex(&g_inv.view(), &u, &v, &mut ainv_u, &mut vt_ainv);
    Ok((r.re, r.im))
}

/// B-spline determinant KINETIC stress dT_det/deps (full 3x3, affine strain). The det side of
/// the full bspline kinetic stress (Jastrow/DTN side added separately). Verifies vs FD of
/// compute_kinetic_energy under the full cell+electron strain.
#[pyfunction]
#[pyo3(name = "grad_T_bspline_kinetic_strain", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_bspline_kinetic_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    if gem.orb_up.bspline.is_none() || gem.orb_dn.bspline.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "determinant orbital is not a B-spline leaf",
        ));
    }
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    let j3jm_v = j3_jm.as_ref().map(|m| m.view());
    let s = stress_pbc::bspline_kinetic_stress_det(
        &gem,
        Some(&jdata),
        j3_orb.as_ref(),
        j3jm_v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// B-spline determinant KINETIC stress directional derivative dT_det/deps along `dcell`.
/// Verifies vs jax.jvp(compute_kinetic_energy, dcell). jnone + withJ (J1/J2/J3/DTN).
#[pyfunction]
#[pyo3(name = "grad_T_bspline_dcell", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, dcell))]
fn py_grad_t_bspline_dcell(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    dcell: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    if gem.orb_up.bspline.is_none() || gem.orb_dn.bspline.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "determinant orbital is not a B-spline leaf",
        ));
    }
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    let j3jm_v = j3_jm.as_ref().map(|m| m.view());
    let dc = dcell.as_array();
    let dcm = [
        [dc[[0, 0]], dc[[0, 1]], dc[[0, 2]]],
        [dc[[1, 0]], dc[[1, 1]], dc[[1, 2]]],
        [dc[[2, 0]], dc[[2, 1]], dc[[2, 2]]],
    ];
    Ok(stress_pbc::grad_t_bspline_dcell(
        &gem,
        Some(&jdata),
        j3_orb.as_ref(),
        j3jm_v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &dcm,
    ))
}

/// B-spline AO directional cell-derivative d(val,grad,lap)/deps along `dcell` (the STRESS
/// building block), each `(num_ao, N_e)`. Verifies vs jqmc `bspline_value_grad_lap_dcell`.
#[pyfunction]
#[pyo3(name = "compute_aos_bspline_vgl_dcell", signature = (aos_data, r_carts, dcell))]
fn py_compute_aos_bspline_vgl_dcell(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
    dcell: PyReadonlyArray2<f64>,
) -> PyResult<PyObj5> {
    let bs = bspline::AosBsplineData::from_py(aos_data)?;
    let dc = dcell.as_array();
    let dcm = [
        [dc[[0, 0]], dc[[0, 1]], dc[[0, 2]]],
        [dc[[1, 0]], dc[[1, 1]], dc[[1, 2]]],
        [dc[[2, 0]], dc[[2, 1]], dc[[2, 2]]],
    ];
    let (v, gx, gy, gz, l) = bs.value_grad_lap_dcell(&r_carts.as_array(), &dcm);
    Ok((
        v.into_pyarray(py).into_any().unbind(),
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
        l.into_pyarray(py).into_any().unbind(),
    ))
}

/// Complex Bloch geminal matrix `G` (N_up, N_up) for a twisted geminal.
#[pyfunction]
#[pyo3(name = "compute_geminal_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_compute_geminal_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let g =
        complex_det::compute_geminal_complex(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex Pulay force `dln|det|/dR` (real, n_atomx3) for a twisted geminal.
#[pyfunction]
#[pyo3(name = "grad_lndet_wrt_R_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_lndet_wrt_r_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // home-AO atom index (length num_ao) for the (possibly image-tiled) pgto leaf
    // (GTO child of a composite pgto(+)pw; PW has no atom force).
    let aos_up = gto_leaf_of(
        &geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("aos_data")?,
    )?;
    let aos_dn = gto_leaf_of(
        &geminal_data
            .getattr("orb_data_dn_spin")?
            .getattr("aos_data")?,
    )?;
    let nuc_up = pgto_home_atom(&aos_up)?;
    let nuc_dn = pgto_home_atom(&aos_dn)?;
    let n_atom = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let g = complex_det::grad_lndet_wrt_r_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        n_atom,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex PBC stress helper: extract home/kernel atom maps plus cell/positions.
/// The GTO leaf of a (possibly pgto(+)pw composite) AO container -- for the pgto
/// home/kernel/structure. Non-composite -> the leaf itself.
fn gto_leaf_of<'p>(aos: &Bound<'p, PyAny>) -> PyResult<Bound<'p, PyAny>> {
    if aos.hasattr("children")? {
        let children = aos.getattr("children")?;
        let c0 = children.get_item(0)?;
        let c1 = children.get_item(1)?;
        let c1_is_pw = c1.get_type().name()?.to_string().contains("plane_wave");
        Ok(if c1_is_pw { c0 } else { c1 })
    } else {
        Ok(aos.clone())
    }
}

/// The gto-block home atoms (per gto AO) of a complex geminal orb leaf's `aos_data`:
/// `pgto_home_atom` of the gto child (composite pgto(+)pw / pgto(+)bspline) or the bare
/// complex-pgto leaf, and EMPTY for a pure plane-wave / B-spline leaf (no atom-centered
/// AO -> zero nuclear force, 0 gto rows). Robust to child order and to a pgto(+)bspline
/// composite (unlike `gto_leaf_of`, which only special-cases the plane-wave child).
fn complex_gto_home_atoms(aos: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if aos.get_type().name()? == "TiledOrbitalSet_data" {
        // The provider owns its electron-dependent image -> physical atom map.
        return Ok(Vec::new());
    }
    if aos.hasattr("children")? {
        let children = aos.getattr("children")?;
        let n: usize = children.len()?;
        for i in 0..n {
            let c = children.get_item(i)?;
            let is_pw = c.get_type().name()?.to_string().contains("plane_wave");
            let is_bs = c.hasattr("spline_coefficients")?;
            if !is_pw && !is_bs {
                return pgto_home_atom(&c);
            }
        }
        Ok(Vec::new())
    } else if aos.get_type().name()?.to_string().contains("plane_wave")
        || aos.hasattr("spline_coefficients")?
    {
        Ok(Vec::new())
    } else {
        pgto_home_atom(aos)
    }
}

/// The atom-centered pGTO leaf of a J3 orbital: unwrap the identity MO wrapper
/// (`MOs_data` -> `aos_data`) then pick the GTO child of a pgto(+)pw composite. Used to
/// build the J3 home/kernel atom maps (the PW child is R-independent -> no atom force).
fn j3_gto_leaf<'p>(orb_data: &Bound<'p, PyAny>) -> PyResult<Bound<'p, PyAny>> {
    let aos = if orb_data.get_type().name()?.to_string().contains("MOs_data") {
        orb_data.getattr("aos_data")?
    } else {
        orb_data.clone()
    };
    gto_leaf_of(&aos)
}

/// The gto-block home + kernel atom maps of a complex geminal orb leaf's `aos_data`, for the
/// complex STRESS field-strain (analogous to `complex_gto_home_atoms` for the force). EMPTY for a
/// pure plane-wave / B-spline leaf (no atom-centered AO -> 0 gto rows -> the gto field stress is
/// a no-op over 0 AOs); the gto child's maps for a pgto(+)pw / pgto(+)bspline composite; the bare
/// pgto leaf's maps otherwise. Robust to child order.
fn complex_gto_stress_maps(aos: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<usize>)> {
    if aos.get_type().name()? == "TiledOrbitalSet_data" {
        return Ok((Vec::new(), Vec::new()));
    }
    if aos.hasattr("children")? {
        let children = aos.getattr("children")?;
        let n: usize = children.len()?;
        for i in 0..n {
            let c = children.get_item(i)?;
            let is_pw = c.get_type().name()?.to_string().contains("plane_wave");
            let is_bs = c.hasattr("spline_coefficients")?;
            if !is_pw && !is_bs {
                return Ok((pgto_home_atom(&c)?, pgto_kernel_atom(&c)?));
            }
        }
        Ok((Vec::new(), Vec::new()))
    } else if aos.get_type().name()?.to_string().contains("plane_wave")
        || aos.hasattr("spline_coefficients")?
    {
        Ok((Vec::new(), Vec::new()))
    } else {
        Ok((pgto_home_atom(aos)?, pgto_kernel_atom(aos)?))
    }
}

/// `(home_up, kernel_up, home_dn, kernel_dn, cell, positions)` of [`complex_stress_args`].
type ComplexStressArgs = (
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    numpy::ndarray::Array2<f64>,
    numpy::ndarray::Array2<f64>,
);

fn complex_stress_args(geminal_data: &Bound<'_, PyAny>) -> PyResult<ComplexStressArgs> {
    let aos_up = orb_aos_leaf(&geminal_data.getattr("orb_data_up_spin")?)?;
    let aos_dn = orb_aos_leaf(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let (home_up, kernel_up) = complex_gto_stress_maps(&aos_up)?;
    let (home_dn, kernel_dn) = complex_gto_stress_maps(&aos_dn)?;
    // structure_data is carried by the pgto gto-leaf AND by a pure plane-wave / B-spline leaf,
    // so `gto_leaf_of` (which falls through to the leaf itself when there is no gto child) always
    // reaches a leaf that exposes cell/positions.
    let sd = gto_leaf_of(&aos_up)?.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    Ok((home_up, kernel_up, home_dn, kernel_dn, cell, positions))
}

/// Complex-Bloch J3 leaf for the twist stress/force cores: `(ComplexOrb, j_matrix,
/// per-sphe-orb home atom, per-kernel-AO atom)`. `(None, ..)` when the Jastrow carries no
/// three-body term. Shared by the `_complex` entries that accept a `jastrow_data`.
#[allow(clippy::type_complexity)]
fn complex_j3_leaf(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<complex_det::ComplexOrb>,
    Option<numpy::ndarray::Array2<f64>>,
    Option<Vec<usize>>,
    Option<Vec<usize>>,
)> {
    if jastrow_data.is_none() {
        return Ok((None, None, None, None));
    }
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Ok((None, None, None, None));
    }
    let od = j3d.getattr("orb_data")?;
    let np = jastrow_data.py().import("numpy")?;
    let jm: numpy::PyReadonlyArray2<f64> = np
        .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
        .call_method1("astype", (np.getattr("float64")?,))?
        .extract()?;
    Ok((
        Some(complex_det::ComplexOrb::from_py(&od)?),
        Some(jm.as_array().to_owned()),
        Some(pgto_home_atom(&j3_gto_leaf(&od)?)?),
        Some(pgto_kernel_atom(&j3_gto_leaf(&od)?)?),
    ))
}

/// Complex Pulay stress `dln|Psi|/deps` (real 3x3) for a twisted geminal, with or without a Jastrow.
#[pyfunction]
#[pyo3(name = "grad_lnPsi_wrt_strain_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_lnpsi_wrt_strain_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (hu, ku, hd, kd, cell, pos) = complex_stress_args(geminal_data)?;
    let (jas, ..) = parse_jastrow_full_complex_aware(jastrow_data)?;
    let (j3c, j3m, j3_home, j3_kernel) = complex_j3_leaf(jastrow_data)?;
    let g = complex_stress::grad_lnpsi_wrt_strain_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        &pos.view(),
        &hu,
        &ku,
        &hd,
        &kd,
        jas.as_ref(),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        j3_home.as_deref(),
        j3_kernel.as_deref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex kinetic stress `dT/deps` (real 3x3) for a twisted geminal without Jastrow.
#[pyfunction]
#[pyo3(name = "grad_T_wrt_strain_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_t_wrt_strain_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (hu, ku, hd, kd, _, pos) = complex_stress_args(geminal_data)?;
    let g = complex_stress::grad_t_wrt_strain_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &hu,
        &ku,
        &hd,
        &kd,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex withJ kinetic stress `dT_withJ/deps` (real 3x3): det gou (sphe AO Hessian
/// field strain) + J1/J2 cell-straining kinetic stress. J3 uses the
/// complex Bloch path when present.
#[pyfunction]
#[pyo3(name = "grad_T_wrt_strain_withJ_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_wrt_strain_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (hu, ku, hd, kd, _, pos) = complex_stress_args(geminal_data)?;
    let (jas, j3o, j3m) = parse_jastrow_full_complex_aware(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j1 = if jas.is_some() && !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    let j2 = if jas.is_some() && !jastrow_data.getattr("jastrow_two_body_data")?.is_none() {
        Some(data::JastrowTwoBodyData::from_py(
            &jastrow_data.getattr("jastrow_two_body_data")?,
        )?)
    } else {
        None
    };
    // Complex Bloch-sphe J3 leaf (its own ComplexOrb, no MO) + home/kernel maps.
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3c, j3_home, j3_kernel) = if j3d.is_none() {
        (None, None, None)
    } else {
        let od = j3d.getattr("orb_data")?;
        let orb = complex_det::ComplexOrb::from_py(&od)?;
        let gto_leaf = j3_gto_leaf(&od)?;
        (
            Some(orb),
            Some(pgto_home_atom(&gto_leaf)?),
            Some(pgto_kernel_atom(&gto_leaf)?),
        )
    };
    let g = complex_stress::grad_t_wrt_strain_complex_withj(
        &gem,
        j1.as_ref(),
        j2.as_ref(),
        jas.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &hu,
        &ku,
        &hd,
        &kd,
        j3c.as_ref(),
        j3_home.as_deref(),
        j3_kernel.as_deref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex ECP non-local param gradient `dV_ecp_nl/dMO` (up, dn complex) for a twisted ECP geminal.
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_mo_complex", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
fn py_grad_v_ecp_nonlocal_mo_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let (gu, gd) = complex_stress::grad_v_ecp_nonlocal_mo_complex(
        &gem,
        None,
        None,
        None,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// Parse (j1, j2, j3 ComplexOrb, j_matrix) from a jastrow_data for the complex ECP-withJ gradients.
#[allow(clippy::type_complexity)]
fn parse_jastrow_complex(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<data::JastrowOneBodyData>,
    Option<data::JastrowTwoBodyData>,
    Option<complex_det::ComplexOrb>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    let j1 = {
        let o = jastrow_data.getattr("jastrow_one_body_data")?;
        if o.is_none() {
            None
        } else {
            Some(data::JastrowOneBodyData::from_py(&o)?)
        }
    };
    let j2 = {
        let o = jastrow_data.getattr("jastrow_two_body_data")?;
        if o.is_none() {
            None
        } else {
            Some(data::JastrowTwoBodyData::from_py(&o)?)
        }
    };
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3c, j3m) = if j3d.is_none() {
        (None, None)
    } else {
        let np = jastrow_data.py().import("numpy")?;
        let jm: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
            .call_method1("astype", (np.getattr("float64")?,))?
            .extract()?;
        let orb = complex_det::ComplexOrb::from_py(&j3d.getattr("orb_data")?)?;
        orb.check_j3_matrix_shape(jm.as_array().nrows(), jm.as_array().ncols())?;
        (Some(orb), Some(jm.as_array().to_owned()))
    };
    Ok((j1, j2, j3c, j3m))
}

/// Parse the complex-twist J3 context `(ComplexOrb, j_matrix, home atoms, kernel atoms)`
/// from a jastrow_data, for the complex GFMC force/stress. `(None,..)` when J3 is absent.
#[allow(clippy::type_complexity)]
fn parse_j3_complex_ctx(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<complex_det::ComplexOrb>,
    Option<numpy::ndarray::Array2<f64>>,
    Option<Vec<usize>>,
    Option<Vec<usize>>,
)> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Ok((None, None, None, None));
    }
    let od = j3d.getattr("orb_data")?;
    let np = jastrow_data.py().import("numpy")?;
    let jm: numpy::PyReadonlyArray2<f64> = np
        .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
        .call_method1("astype", (np.getattr("float64")?,))?
        .extract()?;
    let orb = complex_det::ComplexOrb::from_py(&od)?;
    orb.check_j3_matrix_shape(jm.as_array().nrows(), jm.as_array().ncols())?;
    // A PURE complex plane-wave / B-spline J3 leaf has no cart-lowerable AO and no atom
    // center (`_nucleus_index_jnp` / `pgto_home_atom` would raise): its nuclear-force (R)
    // Pulay and Pulay value-strain (eps) are exactly 0 (verified: the J3 R- and
    // eps-contributions to dln|Psi| are 0 for pure PW / B-spline, real AND complex), and
    // its electron drift (r) routes through the leaf-owning complex batch evaluator
    // `complex_det::j3_grads_lap_complex` (holomorphic, `Re` LAST) instead. Signal it by
    // `home == None` so `J3ComplexOwned::as_ref()` yields None (the cart complex J3
    // force/stress kernels skip it, adding the correct zero) while the orb+jm survive for
    // the batch drift the ln|Psi| caller adds.
    if orb.aos.num_ao == 0 && (orb.bspline.is_some() || orb.pw.is_some()) {
        return Ok((Some(orb), Some(jm.as_array().to_owned()), None, None));
    }
    // The atom-centered `home` / `kernel` maps describe ONLY the pGTO AO rows (the cart complex
    // J3 nuclear-force / value-strain kernels scatter the pGTO block; the PW child, if any, is
    // NOT atom-centered -> zero R-Pulay / value-strain). Navigate to the pGTO leaf: unwrap an
    // (identity-)MO `MOs_data` to its `aos_data`, and for a composite pgto(+){pw,bspline} pick
    // the gto child (the one that is neither a plane-wave nor a B-spline). A bare complex pgto
    // leaf is its own gto leaf. Mirrors `ComplexOrb::from_py`'s child selection.
    let leaf = if od.hasattr("aos_data")? {
        od.getattr("aos_data")?
    } else {
        od.clone()
    };
    let gto_leaf = if leaf.hasattr("children")? {
        let children = leaf.getattr("children")?;
        let c0 = children.get_item(0)?;
        let c1 = children.get_item(1)?;
        let c0_extra = c0.hasattr("spline_coefficients")?
            || c0.get_type().name()?.to_string().contains("plane_wave");
        if c0_extra {
            c1
        } else {
            c0
        }
    } else {
        leaf
    };
    Ok((
        Some(orb),
        Some(jm.as_array().to_owned()),
        Some(pgto_home_atom(&gto_leaf)?),
        Some(pgto_kernel_atom(&gto_leaf)?),
    ))
}

/// Complex ECP non-local d/dMO WITH Jastrow (det MO grad; the Jastrow multiplies the ratio).
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_mo_withJ_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_grad_v_ecp_nonlocal_mo_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    // J1/J2/DTN as one `JastrowData` (its policy places the mesh ratio in
    // `jastrow_ratio`); the complex J3 leaf is parsed separately (issue #717).
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (gu, gd) = complex_stress::grad_v_ecp_nonlocal_mo_complex(
        &gem,
        Some(&jas),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// Dedicated holomorphic complex ECP weight kernel
/// for the c-axis `dV_ecp_nl/dc` on the MO block (per-leaf up/dn). The holomorphic
/// per-grid weight remains an unprojected complex ratio. Element-wise real
/// projection occurs at the source; this weight multiplies the complex O_k
/// delta and its real part is taken last, so the
/// `Im(wf_ratio).Im(O_k)` cross term survives at the twist. The driver / test folds
/// the closed-shell singlet tie `C_dn = conj(C_up)` (`up + conj(dn)`) on top, exactly
/// as `_reduce_c_param_grads`.
#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_mo_coeff_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_grad_el_ecp_wrt_mo_coeff_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    // J1/J2/DTN as one `JastrowData` (its policy places the mesh ratio in
    // `jastrow_ratio`); the complex J3 leaf is parsed separately (issue #717).
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (gu, gd) = complex_stress::grad_v_ecp_nonlocal_mo_complex(
        &gem,
        Some(&jas),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// Complex ECP non-local force `dV_ecp_nl/dR` (real n_atomx3) for a twisted ECP geminal, jnone.
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_R_complex", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
fn py_grad_v_ecp_nonlocal_force_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // Pure plane-wave / B-spline determinant leaves have no atom-centred AO
    // rows: their nuclear Pulay force is exactly zero, while their electron
    // derivatives are evaluated by the leaf-specific complex kernels.  The
    // leaf-aware helper returns an empty map for those cases and the pGTO
    // block's home atoms for composites.
    let nuc_up =
        complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_up_spin")?)?)?;
    let nuc_dn =
        complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_dn_spin")?)?)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let n_atom = pos.nrows();
    let g = complex_stress::grad_v_ecp_nonlocal_force_complex(
        &gem,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
        &nuc_up,
        &nuc_dn,
        n_atom,
        None,
        None,
        None,
        None,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex ECP non-local force `dV_ecp_nl/dR` (real n_atomx3) WITH Jastrow (J1 Williamson
/// value-force + complex J3 value-force; J2 e-e has no nuclear force), twisted ECP geminal.
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_R_withJ_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(clippy::too_many_arguments)]
fn py_grad_v_ecp_nonlocal_force_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let nuc_up = pgto_home_atom(&gto_leaf_of(
        &geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("aos_data")?,
    )?)?;
    let nuc_dn = pgto_home_atom(&gto_leaf_of(
        &geminal_data
            .getattr("orb_data_dn_spin")?
            .getattr("aos_data")?,
    )?)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let n_atom = pos.nrows();
    let jas = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3c, j3m, j3_nuc) = if j3d.is_none() {
        (None, None, None)
    } else {
        let od = j3d.getattr("orb_data")?;
        let np = jastrow_data.py().import("numpy")?;
        let jm: numpy::PyReadonlyArray2<f64> = np
            .call_method1(
                "asarray",
                (j3d.getattr("j_matrix")?, np.getattr("float64")?),
            )?
            .extract()?;
        (
            Some(complex_det::ComplexOrb::from_py(&od)?),
            Some(jm.as_array().to_owned()),
            Some(pgto_home_atom(&j3_gto_leaf(&od)?)?),
        )
    };
    let g = complex_stress::grad_v_ecp_nonlocal_force_complex(
        &gem,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
        &nuc_up,
        &nuc_dn,
        n_atom,
        Some(&jas),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        j3_nuc.as_deref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex ECP non-local stress `dV_ecp_nl/deps` (real 3x3) for a twisted ECP geminal.
/// `jastrow_data=None` -> jnone; with a Jastrow the mesh wf-ratio carries the J1/J2
/// Williamson + DTN + complex-Bloch J3 (`A = Re(complex sphe AO)`) contributions, exactly
/// as the real twin's optional `jastrow_data` does.
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_strain_complex", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, jastrow_data=None))]
#[allow(clippy::too_many_arguments)]
fn py_grad_v_ecp_nonlocal_strain_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    jastrow_data: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (hu, ku, hd, kd, _cell0, pos) = complex_stress_args(geminal_data)?;
    let cp = coulomb_potential_data;
    let (_pos2, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let jas = match jastrow_data {
        Some(j) if !j.is_none() => Some(data::JastrowData::from_py_complex_aware(j)?),
        _ => None,
    };
    let (j3c, j3m, j3_home, j3_kernel) = match jastrow_data {
        Some(j) => complex_j3_leaf(j)?,
        None => (None, None, None, None),
    };
    let g = complex_stress::grad_v_ecp_nonlocal_strain_complex(
        &gem,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
        &hu,
        &ku,
        &hd,
        &kd,
        jas.as_ref(),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        j3_home.as_deref(),
        j3_kernel.as_deref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex ECP non-local stress `dV_ecp_nl/deps` (real 3x3) WITH Jastrow (J1/J2 Williamson
/// + J3 = Re(complex sphe AO)), twisted ECP geminal.
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_strain_withJ_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(clippy::too_many_arguments)]
fn py_grad_v_ecp_nonlocal_strain_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (hu, ku, hd, kd, _cell0, pos) = complex_stress_args(geminal_data)?;
    let cp = coulomb_potential_data;
    let (_pos2, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let jas = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3c, j3m, j3_home, j3_kernel) = if j3d.is_none() {
        (None, None, None, None)
    } else {
        let od = j3d.getattr("orb_data")?;
        let np = jastrow_data.py().import("numpy")?;
        let jm: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
            .extract()?;
        (
            Some(complex_det::ComplexOrb::from_py(&od)?),
            Some(jm.as_array().to_owned()),
            Some(pgto_home_atom(&j3_gto_leaf(&od)?)?),
            Some(pgto_kernel_atom(&j3_gto_leaf(&od)?)?),
        )
    };
    let g = complex_stress::grad_v_ecp_nonlocal_strain_complex(
        &gem,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
        &hu,
        &ku,
        &hd,
        &kd,
        Some(&jas),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        j3_home.as_deref(),
        j3_kernel.as_deref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex kinetic force `dT/dR` (real, n_atomx3) for a twisted geminal.
/// An EMPTY `jastrow_data` (no J1/J2/J3/DTN) takes the jnone core; otherwise the withJ core
/// (det `gou` cross term via the sphe AO Hessian + the Jastrow nuclear side).
#[pyfunction]
#[pyo3(name = "grad_T_wrt_R_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_wrt_r_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // Composite pgto(+)pw: the GTO home map comes from the pgto child (the PW is
    // R-independent -> no atom force).
    let nuc_up = pgto_home_atom(&gto_leaf_of(
        &geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("aos_data")?,
    )?)?;
    let nuc_dn = pgto_home_atom(&gto_leaf_of(
        &geminal_data
            .getattr("orb_data_dn_spin")?
            .getattr("aos_data")?,
    )?)?;
    let n_atom = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let (jas, j3o, j3m) = parse_jastrow_full_complex_aware(jastrow_data)?;
    let has_j = jas
        .as_ref()
        .is_some_and(|j| j.one_body.is_some() || j.two_body.is_some() || j.dtn.is_some())
        || j3o.is_some();
    if !has_j {
        let g = complex_det::grad_t_wrt_r_complex(
            &gem,
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            &nuc_up,
            &nuc_dn,
            n_atom,
        );
        return Ok(g.into_pyarray(py).into_any().unbind());
    }
    let j3v = j3m.as_ref().map(|m| m.view());
    let j1 = if !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    let (j3c, nuc_j3) = if j3o.is_some() {
        let od = jastrow_data
            .getattr("jastrow_three_body_data")?
            .getattr("orb_data")?;
        (
            Some(complex_det::ComplexOrb::from_py(&od)?),
            Some(pgto_home_atom(&j3_gto_leaf(&od)?)?),
        )
    } else {
        (None, None)
    };
    let g = complex_det::grad_t_wrt_r_complex_withj(
        &gem,
        jas.as_ref(),
        j1.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
        n_atom,
        j3c.as_ref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex kinetic force dT_withJ/dR for a twisted geminal WITH Jastrow (J1/J2/J3).
/// Det gou cross term via the sphe AO Hessian; Jastrow side reuses the real
/// j1/j3_side with Re(bar_gj). Verified vs FD of complex compute_kinetic_energy.
#[pyfunction]
#[pyo3(name = "grad_T_wrt_R_withJ_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_wrt_r_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // Composite pgto(+)pw: the GTO home map comes from the pgto child (the PW is
    // R-independent -> no atom force).
    let nuc_up = pgto_home_atom(&gto_leaf_of(
        &geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("aos_data")?,
    )?)?;
    let nuc_dn = pgto_home_atom(&gto_leaf_of(
        &geminal_data
            .getattr("orb_data_dn_spin")?
            .getattr("aos_data")?,
    )?)?;
    let n_atom = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let (jas, j3o, j3m) = parse_jastrow_full_complex_aware(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j1 = if jas.is_some() && !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    // Complex Bloch J3: its ComplexOrb + GTO-child->atom home map (MO-wrapped composite
    // unwrapped; the PW child is R-independent so only the pGTO home map is needed).
    let (j3c, nuc_j3) = if j3o.is_some() {
        let od = jastrow_data
            .getattr("jastrow_three_body_data")?
            .getattr("orb_data")?;
        (
            Some(complex_det::ComplexOrb::from_py(&od)?),
            Some(pgto_home_atom(&j3_gto_leaf(&od)?)?),
        )
    } else {
        (None, None)
    };
    let g = complex_det::grad_t_wrt_r_complex_withj(
        &gem,
        jas.as_ref(),
        j1.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
        n_atom,
        j3c.as_ref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex `dT/dMO` withJ (up, dn): exercises the gj!=0 path of geminal_reverse.
#[pyfunction]
#[pyo3(name = "grad_T_mo_withJ_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv=None))]
fn py_grad_t_mo_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // The Jastrow drift `gj` MUST use the COMPLEX J3 leaf: at a twist the J3 orbital is a
    // complex Bloch AO (composite pgto(+)pw / pgto(+)bspline, or a pure plane_wave /
    // bspline complex leaf), so building a REAL `OrbData` for it (as `parse_jastrow_full`
    // does) indexes off the end of a complex-only leaf and panics. `twist_gj` takes the
    // `Re`-projected complex J3 drift (J3 is real, built on `Re(orbital)`), exactly as the
    // J1/J2/DTN twist kinetic kernels do. `from_py_without_three_body` avoids the eager
    // real-J3 OrbData that `JastrowData::from_py` would build.
    let (jas, j3c, j3m) = if jastrow_data.is_none() {
        (None, None, None)
    } else {
        let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
        (
            Some(data::JastrowData::from_py_without_three_body(jastrow_data)?),
            j3c,
            j3m,
        )
    };
    let (gj_up, gj_dn) = param_grad_twist::twist_gj(
        jas.as_ref(),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    let gi = geminal_inv.as_ref().map(|value| value.as_array());
    let (gu, gd) = complex_det::grad_t_mo_complex_gj_gi(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &gj_up,
        &gj_dn,
        gi.as_ref(),
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// Complex `dT/dMO` (up, dn) for a twisted geminal without Jastrow.
#[pyfunction]
#[pyo3(name = "grad_T_mo_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_t_mo_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (gu, gd) =
        complex_det::grad_t_mo_complex(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// Complex `dT/dlambda` for a twisted geminal without Jastrow.
#[pyfunction]
#[pyo3(name = "grad_T_lambda_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_t_lambda_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let g =
        complex_det::grad_t_lambda_complex(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex `dT/dlambda` for a twisted geminal WITH a Jastrow (the with-J twin of
/// `grad_T_lambda_complex`). `J(r)` is real and lambda-INDEPENDENT, so the only Jastrow
/// entry into the kinetic lambda c-axis is the cross-term seed `gj` (the full Jastrow drift
/// incl. the `Re`-projected complex J3 leaf), exactly as `grad_T_mo_withJ_complex` threads
/// it. The dense-complex-`lambda` AGP twist Det e_L rides this; the reverse carries the
/// FULL complex `lambda_c`, so `Im(lambda)` reaches dE_L/dlambda. Returns the HOLOMORPHIC
/// (un-`Re`'d) complex `dT/dlambda` -- the SR/LM complex-DOF convention takes `Re` last, in
/// the driver, never here.
#[pyfunction]
#[pyo3(name = "grad_T_lambda_withJ_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_lambda_withj_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // Same complex-J3-safe Jastrow drift assembly as `grad_T_mo_withJ_complex`.
    let (jas, j3c, j3m) = if jastrow_data.is_none() {
        (None, None, None)
    } else {
        let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
        (
            Some(data::JastrowData::from_py_without_three_body(jastrow_data)?),
            j3c,
            j3m,
        )
    };
    let (gj_up, gj_dn) = param_grad_twist::twist_gj(
        jas.as_ref(),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    let g = complex_det::grad_t_lambda_complex_gj(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &gj_up,
        &gj_dn,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Complex `O_k = dln det/dMO` (up, dn) for a twisted geminal.
#[pyfunction]
#[pyo3(name = "grad_ln_det_mo_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_ln_det_mo_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (gu, gd) =
        complex_det::grad_ln_det_mo_complex(&gem, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

/// `O_k = dln det/dlambda` HOLOMORPHIC (un-conjugated complex) for a twisted geminal
/// This is the c-axis O_k the driver stores for a complex `AGP_data` lambda.
#[pyfunction]
#[pyo3(name = "grad_ln_det_lambda_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_ln_det_lambda_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    // Read the FULL complex lambda (ComplexGeminal stores Re-only; a genuine complex
    // AGP twist has nonzero Im that the un-conjugated holomorphic O_k must carry).
    let np = py.import("numpy")?;
    let lam_c: numpy::PyReadonlyArray2<numpy::Complex64> = np
        .call_method1("asarray", (geminal_data.getattr("lambda_matrix")?,))?
        .call_method1("astype", (np.getattr("complex128")?,))?
        .extract()?;
    let g = complex_det::grad_ln_det_lambda_complex_holo_clam(
        &gem,
        &lam_c.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// AGPn twist `O_k` on the `(lambda_diag, lambda_unpaired)` leaves (Re diagonal +
/// unpaired slice of the complex-twist dense-`lambda` gradient).
#[pyfunction]
#[pyo3(name = "grad_ln_det_lambda_diag_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_ln_det_lambda_diag_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (d, u) = complex_det::grad_ln_det_lambda_diag_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

/// AGPn twist dE_L on the `(lambda_diag, lambda_unpaired)` leaves (Jastrow-free twist
/// kinetic; Re diagonal + unpaired slice of `grad_t_lambda_complex`).
#[pyfunction]
#[pyo3(name = "grad_T_lambda_diag_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_t_lambda_diag_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (d, u) = complex_det::grad_t_lambda_diag_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

/// `ln|det G|` (real |Psi|^2 amplitude) for a twisted geminal.
#[pyfunction]
#[pyo3(name = "compute_ln_det_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_compute_ln_det_complex(
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    Ok(complex_det::compute_ln_det_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

/// Complex ECP non-local energy (jnone, all-pairs) for a twisted geminal -> (re, im).
#[pyfunction]
#[pyo3(name = "compute_ecp_nonlocal_complex", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
fn py_compute_ecp_nonlocal_complex(
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(f64, f64)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    let cell_owned = if pbc {
        Some(
            sd.getattr("cell")?
                .extract::<numpy::PyReadonlyArray2<f64>>()?
                .as_array()
                .to_owned(),
        )
    } else {
        None
    };
    let nmax: i32 = if pbc {
        sd.getattr("_n_max")?.extract()?
    } else {
        0
    };
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let v = complex_ecp::compute_ecp_nonlocal_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        nmax,
    );
    Ok((v.re, v.im))
}

/// Complex ECP non-local energy WITH Jastrow (J1/J2 Williamson + J3 = Re(complex sphe
/// AO)) for a twisted ECP geminal -> (re, im). Jastrow ratio exp(J(r')-J(r)) x det ratio.
#[pyfunction]
#[pyo3(name = "compute_ecp_nonlocal_withJ_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
fn py_compute_ecp_nonlocal_withj_complex(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(f64, f64)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    let cell_owned = if pbc {
        Some(
            sd.getattr("cell")?
                .extract::<numpy::PyReadonlyArray2<f64>>()?
                .as_array()
                .to_owned(),
        )
    } else {
        None
    };
    let nmax: i32 = if pbc {
        sd.getattr("_n_max")?.extract()?
    } else {
        0
    };
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    // J1/J2/DTN as one `JastrowData` (its policy places the mesh ratio in `jastrow_ratio`);
    // the complex J3 leaf is parsed separately (issue #757).
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let v = complex_ecp::compute_ecp_nonlocal_withj_complex(
        &gem,
        Some(&jas),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        nmax,
        0,
    );
    Ok((v.re, v.im))
}

/// Complex kinetic energy WITH Jastrow for a twisted geminal -> (re, im).
#[pyfunction]
#[pyo3(name = "compute_kinetic_withj_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_compute_kinetic_withj_complex(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(f64, f64)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (jas, _j3o, j3m) = parse_jastrow_full_complex_aware(jastrow_data)?;
    let jas = jas.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("jastrow_data is None"))?;
    let j3v = j3m.as_ref().map(|m| m.view());
    // complex Bloch-sphe J3 leaf (ComplexOrb), not the real cart OrbData
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let j3c = if j3d.is_none() {
        None
    } else {
        Some(complex_det::ComplexOrb::from_py(&j3d.getattr("orb_data")?)?)
    };
    let j3 = match (&j3c, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let t = complex_det::compute_kinetic_withj_complex(
        &gem,
        &jas,
        j3,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((t.re, t.im))
}

/// Complex determinant kinetic energy (jnone) for a twisted geminal -> (re, im).
#[pyfunction]
#[pyo3(name = "compute_kinetic_det_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_compute_kinetic_det_complex(
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(f64, f64)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let t = complex_det::compute_kinetic_det_complex(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((t.re, t.im))
}

/// Complex Bloch AO Hessian for a twisted sphe pgto leaf (returns 3x3 list of
/// (num_ao,N) complex arrays). Verifies the sphe AO Hessian closed form.
#[pyfunction]
#[pyo3(name = "compute_aos_hessian_complex", signature = (aos_data, r_carts))]
fn py_compute_aos_hessian_complex(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    let aos = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let np = py.import("numpy")?;
    let ipf: numpy::PyReadonlyArray1<numpy::Complex64> = np
        .call_method1("asarray", (aos_data.getattr("_image_phase_factors")?,))?
        .extract()?;
    let h = complex_ao::compute_aos_hessian_complex(&aos, ipf.as_slice()?, &r_carts.as_array());
    Ok(h.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|m| m.into_pyarray(py).into_any().unbind())
                .collect()
        })
        .collect())
}

/// Complex Bloch AO value+grad+laplacian for a twisted pgto leaf (returns a
/// 5-tuple: val, gx, gy, gz, lap). Verifies the complex AO derivative layer.
#[pyfunction]
#[pyo3(name = "compute_aos_vgl_complex", signature = (aos_data, r_carts))]
fn py_compute_aos_vgl_complex(
    py: Python<'_>,
    aos_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj5> {
    let aos = atomic_orbital::AosGtoData::from_py(aos_data)?;
    let np = py.import("numpy")?;
    let ipf: numpy::PyReadonlyArray1<numpy::Complex64> = np
        .call_method1("asarray", (aos_data.getattr("_image_phase_factors")?,))?
        .extract()?;
    let (v, gx, gy, gz, l) =
        complex_ao::compute_aos_vgl_complex(&aos, ipf.as_slice()?, &r_carts.as_array());
    Ok((
        v.into_pyarray(py).into_any().unbind(),
        gx.into_pyarray(py).into_any().unbind(),
        gy.into_pyarray(py).into_any().unbind(),
        gz.into_pyarray(py).into_any().unbind(),
        l.into_pyarray(py).into_any().unbind(),
    ))
}

/// The GTO/pGTO AO leaf owned by an `orb_data`, matching Python's
/// `determinant._gto_leaves_of`: unwrap `MOs_data.aos_data`, and for an
/// `AOs_composite_data` return its GTO child (the one carrying `radial`; jqmc's
/// composites hold exactly one GTO child, pgto(+)pw / pgto(+)bspline). A bare GTO
/// leaf returns itself; a pure PW / B-spline leaf (no `radial`, no GTO child)
/// returns `None`.
fn gto_leaf_obj<'py>(orb_data: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let aos = if orb_data.hasattr("aos_data")? {
        orb_data.getattr("aos_data")?
    } else {
        orb_data.clone()
    };
    if aos.hasattr("children")? {
        let children = aos.getattr("children")?;
        let n: usize = children.len()?;
        for i in 0..n {
            let child = children.get_item(i)?;
            if child.hasattr("radial")? {
                return Ok(Some(child));
            }
        }
        return Ok(None);
    }
    if aos.hasattr("radial")? {
        return Ok(Some(aos));
    }
    Ok(None)
}

/// `(ao_prim_to_shell_prim, n_shell_prim)` for the GTO AO leaf of an orb_data
/// (the GTO child for a composite -- see `gto_leaf_obj`).
fn radial_map(orb_data: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, usize)> {
    let aos = gto_leaf_obj(orb_data)?.ok_or_else(|| {
        pyo3::exceptions::PyNotImplementedError::new_err(
            "radial_map: orbital leaf carries no GTO/pGTO `radial` child",
        )
    })?;
    let py = aos.py();
    let np = py.import("numpy")?;
    let a2s: Vec<i64> = np
        .call_method1("asarray", (aos.getattr("ao_prim_to_shell_prim")?,))?
        .call_method0("tolist")?
        .extract()?;
    let exps: numpy::PyReadonlyArray1<f64> = np
        .call_method1("asarray", (aos.getattr("radial")?.getattr("exponents")?,))?
        .extract()?;
    let n_shell = exps.as_array().len();
    Ok((a2s.into_iter().map(|x| x as usize).collect(), n_shell))
}

/// Home-cell lattice `(3,3)` if the orb leaf is periodic (pbc_flag), else None.
/// Used to wrap electrons into the home cell so the per-image radial attribution
/// in the pGTO basis param-grad matches jqmc's wrapping AO eval.
fn pbc_cell(orb_data: &Bound<'_, PyAny>) -> PyResult<Option<numpy::ndarray::Array2<f64>>> {
    // Resolve to the GTO leaf (its `structure_data` field is the shared home cell;
    // for a composite the container's `structure_data` property delegates to the
    // same children[0] structure anyway).
    let aos = match gto_leaf_obj(orb_data)? {
        Some(a) => a,
        None => return Ok(None),
    };
    if !aos.hasattr("structure_data")? {
        return Ok(None);
    }
    let sd = aos.getattr("structure_data")?;
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    if !pbc {
        return Ok(None);
    }
    let np = orb_data.py().import("numpy")?;
    let cell: numpy::PyReadonlyArray2<f64> = np
        .call_method1("asarray", (sd.getattr("cell")?,))?
        .extract()?;
    Ok(Some(cell.as_array().to_owned()))
}

/// Raise `NotImplementedError` for orbital leaves whose radial param-grad Rust
/// cannot serve (a PURE plane-wave / B-spline leaf carries no GTO radial): a clean
/// PyErr rather than a Rust panic on the strict dispatch. A composite is served
/// child-wise via its GTO child (`gto_leaf_obj`), matching Python's
/// `_gto_leaves_of`; only a leaf with NO GTO child is rejected here.
fn ensure_gto_radial(orb_data: &Bound<'_, PyAny>) -> PyResult<()> {
    if gto_leaf_obj(orb_data)?.is_some() {
        return Ok(());
    }
    let aos = if orb_data.hasattr("aos_data")? {
        orb_data.getattr("aos_data")?
    } else {
        orb_data.clone()
    };
    let tn = aos.get_type().name()?.to_string();
    Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "grad_*_wrt_radial: no GTO/pGTO radial param-grad for leaf {tn} (pure plane-wave / B-spline carries no GTO radial)"
    )))
}

#[pyfunction]
#[pyo3(name = "grad_ln_det_wrt_radial", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_radial_basis(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let up_py = geminal_data.getattr("orb_data_up_spin")?;
    let dn_py = geminal_data.getattr("orb_data_dn_spin")?;
    ensure_gto_radial(&up_py)?;
    ensure_gto_radial(&dn_py)?;
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (a2s_up, nsh_up) = radial_map(&up_py)?;
    let (a2s_dn, nsh_dn) = radial_map(&dn_py)?;
    let cell = pbc_cell(&up_py)?;
    let cell_v = cell.as_ref().map(|c| c.view());
    let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_basis::grad_radial(
        &gem,
        &a2s_up,
        nsh_up,
        &a2s_dn,
        nsh_dn,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        cell_v.as_ref(),
    );
    let mk = |v: Vec<f64>| {
        numpy::ndarray::Array1::from_vec(v)
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    let dict = pyo3::types::PyDict::new(py);
    let up = pyo3::types::PyTuple::new(py, [mk(gz_up), mk(gc_up)])?;
    let dn = pyo3::types::PyTuple::new(py, [mk(gz_dn), mk(gc_dn)])?;
    dict.set_item("orb_data_up_spin", up)?;
    dict.set_item("orb_data_dn_spin", dn)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_radial", signature = (geminal_data, r_up_carts, r_dn_carts, jastrow_data=None))]
fn py_grad_radial_kinetic(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    jastrow_data: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let up_py = geminal_data.getattr("orb_data_up_spin")?;
    let dn_py = geminal_data.getattr("orb_data_dn_spin")?;
    ensure_gto_radial(&up_py)?;
    ensure_gto_radial(&dn_py)?;
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (a2s_up, nsh_up) = radial_map(&up_py)?;
    let (a2s_dn, nsh_dn) = radial_map(&dn_py)?;
    let cell = pbc_cell(&up_py)?;
    let cell_v = cell.as_ref().map(|c| c.view());
    // `jastrow_data` is OPTIONAL and trailing: omitting it (or passing None) keeps the
    // pinned jnone det-kinetic dT/dc that the existing free-kernel callers rely on;
    // passing a `Jastrow_data` adds the Jastrow-determinant kinetic cross term
    // (`-Sum gJ.gD` carried through `geminal_reverse` into the orbital-GRADIENT
    // cotangent), i.e. the SAME with-Jastrow reduction the production ctx
    // `pd_el_basis` performs. J3 is threaded through `parse_jastrow_real_value` exactly as
    // for `grad_eL_wrt_{mo_coeff,lambda}` (an inactive J3 is identically 0 and skipped, #170).
    let (jas, j3_orb, j3_jm) = match jastrow_data {
        None => (None, None, None),
        Some(j) => parse_jastrow_real_value(j)?,
    };
    let j3v = j3_jm.as_ref().map(|m| m.view());
    let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_basis::grad_radial_kinetic(
        &gem,
        jas.as_ref(),
        j3_orb.as_ref(),
        j3v.as_ref(),
        &a2s_up,
        nsh_up,
        &a2s_dn,
        nsh_dn,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        cell_v.as_ref(),
    );
    let mk = |v: Vec<f64>| {
        numpy::ndarray::Array1::from_vec(v)
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    let dict = pyo3::types::PyDict::new(py);
    let up = pyo3::types::PyTuple::new(py, [mk(gz_up), mk(gc_up)])?;
    let dn = pyo3::types::PyTuple::new(py, [mk(gz_dn), mk(gc_dn)])?;
    dict.set_item("orb_data_up_spin", up)?;
    dict.set_item("orb_data_dn_spin", dn)?;
    Ok(dict.into_any().unbind())
}

/// FULL complex `lambda_matrix` of a (possibly complex-twist) geminal.
/// `ComplexGeminal::from_py` stores lambda Re-only -- right for the SD / real-pairing
/// twists, but a genuine complex `AGP_data` twist carries a nonzero imaginary part
/// (measured: max|Im| = 6.4 on the committed composite twist AGP fixture) that every
/// HOLOMORPHIC c-axis must carry. `astype(complex128)` makes the real case a no-op.
fn complex_lambda(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
) -> PyResult<numpy::ndarray::Array2<numpy::Complex64>> {
    let np = py.import("numpy")?;
    let lam: numpy::PyReadonlyArray2<numpy::Complex64> = np
        .call_method1("asarray", (geminal_data.getattr("lambda_matrix")?,))?
        .call_method1("astype", (np.getattr("complex128")?,))?
        .extract()?;
    Ok(lam.as_array().to_owned())
}

/// Pack `((gZ_up,gc_up),(gZ_dn,gc_dn))` into the per-spin dict the radial c-axis
/// kernels return (shared by the real and complex-twist entry points).
fn radial_dict(
    py: Python<'_>,
    up: (Vec<f64>, Vec<f64>),
    dn: (Vec<f64>, Vec<f64>),
) -> PyResult<Py<PyAny>> {
    let mk = |v: Vec<f64>| {
        numpy::ndarray::Array1::from_vec(v)
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "orb_data_up_spin",
        pyo3::types::PyTuple::new(py, [mk(up.0), mk(up.1)])?,
    )?;
    dict.set_item(
        "orb_data_dn_spin",
        pyo3::types::PyTuple::new(py, [mk(dn.0), mk(dn.1)])?,
    )?;
    Ok(dict.into_any().unbind())
}

/// Complex-valued twin used by exact per-walker local-energy derivative sources.
fn radial_dict_complex(
    py: Python<'_>,
    up: (Vec<numpy::Complex64>, Vec<numpy::Complex64>),
    dn: (Vec<numpy::Complex64>, Vec<numpy::Complex64>),
) -> PyResult<Py<PyAny>> {
    let mk = |v: Vec<numpy::Complex64>| {
        numpy::ndarray::Array1::from_vec(v)
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "orb_data_up_spin",
        pyo3::types::PyTuple::new(py, [mk(up.0), mk(up.1)])?,
    )?;
    dict.set_item(
        "orb_data_dn_spin",
        pyo3::types::PyTuple::new(py, [mk(dn.0), mk(dn.1)])?,
    )?;
    Ok(dict.into_any().unbind())
}

fn radial_complex_re(
    values: (Vec<numpy::Complex64>, Vec<numpy::Complex64>),
) -> (Vec<f64>, Vec<f64>) {
    (
        values.0.into_iter().map(|z| z.re).collect(),
        values.1.into_iter().map(|z| z.re).collect(),
    )
}

/// `O_k = dln|det|/dradial` for a COMPLEX-TWIST (Bloch) geminal -- the complex twin of
/// `grad_ln_det_wrt_radial`. Holomorphic chain, `Re` taken LAST (see
/// `param_grad_basis::grad_radial_complex`). Serves the complex sphe/cart-pGTO leaves
/// and the pgto(+)pw composite at a twist (the pw child carries no radial DOF).
#[pyfunction]
#[pyo3(name = "grad_ln_det_wrt_radial_complex", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_radial_basis_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let up_py = geminal_data.getattr("orb_data_up_spin")?;
    let dn_py = geminal_data.getattr("orb_data_dn_spin")?;
    ensure_gto_radial(&up_py)?;
    ensure_gto_radial(&dn_py)?;
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (a2s_up, nsh_up) = radial_map(&up_py)?;
    let (a2s_dn, nsh_dn) = radial_map(&dn_py)?;
    let lam = if gem.is_sd() {
        None
    } else {
        Some(complex_lambda(py, geminal_data)?)
    };
    let (up, dn) = param_grad_basis::grad_radial_complex_re(
        &gem,
        lam.as_ref().map(|l| l.view()).as_ref(),
        &a2s_up,
        nsh_up,
        &a2s_dn,
        nsh_dn,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    radial_dict(py, up, dn)
}

/// `de_L/dradial = Re(dT_det/dradial)` for a COMPLEX-TWIST geminal -- the complex twin
/// of `grad_eL_wrt_radial`. `jastrow_data` is OPTIONAL and trailing (same contract as
/// the real kernel): omitted/None keeps the jnone det kinetic, a `Jastrow_data` adds
/// the `-Sum gJ.gD` cross term through the complex reverse's orbital-GRADIENT
/// cotangent. J3 rides the complex Bloch AO (`parse_j3_complex`).
#[pyfunction]
#[pyo3(name = "grad_eL_wrt_radial_complex", signature = (geminal_data, r_up_carts, r_dn_carts, jastrow_data=None, source_mode=false))]
fn py_grad_radial_kinetic_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    jastrow_data: Option<&Bound<'_, PyAny>>,
    source_mode: bool,
) -> PyResult<Py<PyAny>> {
    let up_py = geminal_data.getattr("orb_data_up_spin")?;
    let dn_py = geminal_data.getattr("orb_data_dn_spin")?;
    ensure_gto_radial(&up_py)?;
    ensure_gto_radial(&dn_py)?;
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (a2s_up, nsh_up) = radial_map(&up_py)?;
    let (a2s_dn, nsh_dn) = radial_map(&dn_py)?;
    let (jas, j3c, j3m) = match jastrow_data {
        Some(j) if !j.is_none() => {
            let (c, m) = parse_j3_complex(j)?;
            (Some(data::JastrowData::from_py_complex_aware(j)?), c, m)
        }
        _ => (None, None, None),
    };
    let lam = if gem.is_sd() {
        None
    } else {
        Some(complex_lambda(py, geminal_data)?)
    };
    let (up, dn) = param_grad_basis::grad_radial_kinetic_complex(
        &gem,
        lam.as_ref().map(|l| l.view()).as_ref(),
        jas.as_ref(),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &a2s_up,
        nsh_up,
        &a2s_dn,
        nsh_dn,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    if source_mode {
        radial_dict_complex(py, up, dn)
    } else {
        radial_dict(py, radial_complex_re(up), radial_complex_re(dn))
    }
}

/// `dV_ecp(non-local)/dradial` at a TWIST -- the complex twin of
/// `grad_eL_ecp_wrt_radial`. Complex per-grid weight x HOLOMORPHIC radial O_k,
/// `Re` taken on the product (see `param_grad_twist::grad_ecp_radial_twist`).
#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_radial_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, source_mode=false))]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_radial_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<Py<PyAny>> {
    let up_py = geminal_data.getattr("orb_data_up_spin")?;
    let dn_py = geminal_data.getattr("orb_data_dn_spin")?;
    ensure_gto_radial(&up_py)?;
    ensure_gto_radial(&dn_py)?;
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (a2s_up, nsh_up) = radial_map(&up_py)?;
    let (a2s_dn, nsh_dn) = radial_map(&dn_py)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let lam = if gem.is_sd() {
        None
    } else {
        Some(complex_lambda(py, geminal_data)?)
    };
    let (up, dn) = param_grad_twist::grad_ecp_radial_twist(
        &gem,
        lam.as_ref().map(|l| l.view()).as_ref(),
        &jas,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &a2s_up,
        nsh_up,
        &a2s_dn,
        nsh_dn,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    if source_mode {
        radial_dict_complex(py, up, dn)
    } else {
        radial_dict(py, radial_complex_re(up), radial_complex_re(dn))
    }
}

/// J3 orbital radial O_k for a complex-Bloch leaf.  Returns the one shared
/// J3 `(exp, coeff)` block (the Python production assembler assigns simple or
/// child-qualified names from the leaf protocol).
#[pyfunction]
#[pyo3(name = "grad_ln_Psi_jastrow_3b_radial_complex")]
fn py_grad_j3_radial_complex(
    py: Python<'_>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_ln_Psi_jastrow_3b_radial_complex: missing J3 data",
        ));
    }
    let orb_py = j3d.getattr("orb_data")?;
    ensure_gto_radial(&orb_py)?;
    let (a2s, nsh) = radial_map(&orb_py)?;
    let (orb, jm) = parse_j3_complex(jastrow_data)?;
    let orb = orb.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inactive complex J3"))?;
    let jm = jm.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inactive complex J3"))?;
    let (gz, gc) = param_grad_basis::grad_j3_radial_complex(
        &orb,
        &jm.view(),
        &a2s,
        nsh,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((vec_pyarray(py, gz), vec_pyarray(py, gc)))
}

/// Complex-Bloch J3 radial kinetic derivative `dT/dc`.
#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_3b_radial_complex")]
fn py_grad_j3_radial_kinetic_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_eL_wrt_jastrow_3b_radial_complex: missing J3 data",
        ));
    }
    let orb_py = j3d.getattr("orb_data")?;
    ensure_gto_radial(&orb_py)?;
    let (a2s, nsh) = radial_map(&orb_py)?;
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (orb, jm) = parse_j3_complex(jastrow_data)?;
    let orb = orb.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inactive complex J3"))?;
    let jm = jm.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inactive complex J3"))?;
    let (gz, gc) = param_grad_basis::grad_j3_radial_kinetic_complex(
        &gem,
        &jas,
        &orb,
        &jm.view(),
        &a2s,
        nsh,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((vec_pyarray(py, gz), vec_pyarray(py, gc)))
}

/// Complex-Bloch ECP non-local add for the J3 radial derivative.
#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_3b_radial_complex")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_j3_radial_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_eL_ecp_wrt_jastrow_3b_radial_complex: missing J3 data",
        ));
    }
    let orb_py = j3d.getattr("orb_data")?;
    ensure_gto_radial(&orb_py)?;
    let (a2s, nsh) = radial_map(&orb_py)?;
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (orb, jm) = parse_j3_complex(jastrow_data)?;
    let orb = orb.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inactive complex J3"))?;
    let jm = jm.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("inactive complex J3"))?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell, nmax) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell.as_ref().map(|c| c.view());
    let (gz, gc) = param_grad_twist::grad_ecp_j3_radial_twist(
        &gem,
        &jas,
        &orb,
        &jm.view(),
        &a2s,
        nsh,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        nmax,
    );
    Ok((vec_pyarray(py, gz), vec_pyarray(py, gc)))
}

// ---- Local-energy parameter derivatives de_L/dc = dT/dc ----

/// `(JastrowData, J3 orbital, J3 j_matrix)` of [`parse_jastrow_full`].
type JastrowJ3Parts = (
    Option<data::JastrowData>,
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
);

/// Parse (JastrowData, Option<OrbData j3>, Option<Array2 j_matrix>) from a
/// Python `jastrow_data` (or None for jnone).
fn parse_jastrow_full(jastrow_data: &Bound<'_, PyAny>) -> PyResult<JastrowJ3Parts> {
    parse_jastrow_full_impl(jastrow_data, false)
}

/// [`parse_jastrow_full`] for the complex-twist consumers, which evaluate the J3 through
/// their own complex leaf: a complex-only J3 is not rejected (see
/// [`data::JastrowData::from_py_complex_aware`]).
#[allow(clippy::type_complexity)]
fn parse_jastrow_full_complex_aware(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<data::JastrowData>,
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    parse_jastrow_full_impl(jastrow_data, true)
}

#[allow(clippy::type_complexity)]
fn parse_jastrow_full_impl(
    jastrow_data: &Bound<'_, PyAny>,
    complex_aware: bool,
) -> PyResult<(
    Option<data::JastrowData>,
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    if jastrow_data.is_none() {
        return Ok((None, None, None));
    }
    let jas = if complex_aware {
        data::JastrowData::from_py_complex_aware(jastrow_data)?
    } else {
        data::JastrowData::from_py(jastrow_data)?
    };
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3_orb, j3_jm) = if j3d.is_none() {
        (None, None)
    } else {
        let orb = determinant::OrbData::from_py(&j3d.getattr("orb_data")?)?;
        let np = jastrow_data.py().import("numpy")?;
        let jm: numpy::PyReadonlyArray2<f64> = np
            .call_method1(
                "asarray",
                (j3d.getattr("j_matrix")?, np.getattr("float64")?),
            )?
            .extract()?;
        (Some(orb), Some(jm.as_array().to_owned()))
    };
    Ok((Some(jas), j3_orb, j3_jm))
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_lambda", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_el_lambda(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    if gem.lambda.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_t_lambda is undefined for occupied-only SD",
        ));
    }
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let g = param_grad_el::grad_t_lambda(
        &gem,
        jas.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// AGPn dE_L on the `(lambda_diag, lambda_unpaired)` leaves (real path): diagonal +
/// unpaired slice of the dense-`lambda` kinetic gradient.
#[pyfunction]
#[pyo3(name = "grad_eL_wrt_lambda_diag", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_el_lambda_diag(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let (d, u) = param_grad_el::grad_t_lambda_diag(
        &gem,
        jas.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_mo_coeff", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv=None))]
fn py_grad_el_mo(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: Option<PyReadonlyArray2<f64>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let gi = geminal_inv.as_ref().map(|value| value.as_array());
    let (gu, gd) = param_grad_el::grad_t_mo_gi(
        &gem,
        jas.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        gi.as_ref(),
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_1b_param")]
fn py_grad_el_j1(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let jas = jas.expect("grad_eL_wrt_jastrow_1b_param: jastrow_data is None");
    let j1 = data::JastrowOneBodyData::from_py(&jastrow_data.getattr("jastrow_one_body_data")?)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    Ok(param_grad_el::grad_t_j1(
        &gem,
        &jas,
        &j1,
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_2b_param")]
fn py_grad_el_j2(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let jas = jas.expect("grad_eL_wrt_jastrow_2b_param: jastrow_data is None");
    let j2 = data::JastrowTwoBodyData::from_py(&jastrow_data.getattr("jastrow_two_body_data")?)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    Ok(param_grad_el::grad_t_j2(
        &gem,
        &jas,
        &j2,
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_3b_jmatrix")]
fn py_grad_el_j3(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3o, j3m) = parse_explicit_real_j3_for_jmatrix_deriv(jastrow_data)?;
    let (j3o, j3m) = match (j3o, j3m) {
        (Some(orb), Some(matrix)) => (orb, matrix),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "grad_eL_wrt_jastrow_3b_jmatrix: no jastrow_three_body_data",
            ));
        }
    };
    let g = param_grad_el::grad_t_j3(
        &gem,
        &jas,
        &j3o,
        &j3m.view(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

// ---- PBC kinetic force dT_det/dR (forces_pbc.rs, real Gamma pgto) ----

/// Per-molecular-AO home atom of a (cart) pgto leaf (`_pgto_home_atom`).
fn pgto_home_atom(aos_py: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let py = aos_py.py();
    let np = py.import("numpy")?;
    let num_ao: usize = aos_py.getattr("num_ao")?.extract()?;
    let has_ragged = aos_py.hasattr("_ragged_kernel_to_home")?
        && !aos_py.getattr("_ragged_kernel_to_home")?.is_none();
    if has_ragged {
        let home: Vec<i64> = np
            .call_method1("asarray", (aos_py.getattr("_ragged_kernel_to_home")?,))?
            .call_method0("tolist")?
            .extract()?;
        let katom: Vec<i64> = np
            .call_method1("asarray", (aos_py.getattr("_ragged_kernel_atom")?,))?
            .call_method0("tolist")?
            .extract()?;
        let mut out = vec![0usize; num_ao];
        for k in 0..home.len() {
            out[home[k] as usize] = katom[k] as usize;
        }
        Ok(out)
    } else {
        let nuc: Vec<i64> = np
            .call_method1("asarray", (aos_py.getattr("_nucleus_index_jnp")?,))?
            .call_method0("tolist")?
            .extract()?;
        Ok(nuc.into_iter().map(|x| x as usize).collect())
    }
}

type PgtoLeaf = (
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
);

/// One cart-pgto AO leaf (`aos_py`) + its MO-coeff column block `c_block`
/// (num_mo, n_block) -> `(cart_aos, C_cart, home_atom)`. Sphe leaves convert via
/// `_aos_sphe_to_cart` (folding the molecular c2s into the block).
fn pgto_leaf_pieces(
    aos_py: &Bound<'_, PyAny>,
    c_block: &numpy::ndarray::ArrayView2<f64>,
) -> PyResult<PgtoLeaf> {
    let py = aos_py.py();
    if aos_py.get_type().name()?.to_string().contains("sphe") {
        let np = py.import("numpy")?;
        let ao_mod = py.import("jqmc.atomic_orbital")?;
        let res = ao_mod.call_method1("_aos_sphe_to_cart", (aos_py,))?;
        let cart_aos_py = res.get_item(0)?;
        let t_mol: numpy::PyReadonlyArray2<f64> =
            np.call_method1("asarray", (res.get_item(1)?,))?.extract()?;
        let c_cart = c_block.dot(&t_mol.as_array());
        let cart_aos = atomic_orbital::AosGtoData::from_py(&cart_aos_py)?;
        let home = pgto_home_atom(&cart_aos_py)?;
        Ok((cart_aos, c_cart, home))
    } else {
        let cart_aos = atomic_orbital::AosGtoData::from_py(aos_py)?;
        let home = pgto_home_atom(aos_py)?;
        Ok((cart_aos, c_block.to_owned(), home))
    }
}

/// `(AO leaf, orbital->AO coefficient matrix C (num_orb, num_ao_total))` of an orb_data.
/// An `MOs_data` reports its own `aos_data` + `mo_coefficients`; a RAW-AO orbital (the
/// dense AO-AGP representation, whose `lambda_matrix` pairs AOs directly) IS its own AO
/// leaf and its orbital basis IS the AO basis, i.e. `C = I(num_ao)`. Threading the
/// identity here is exact (not an approximation): every downstream leaf kernel contracts
/// `C` with an orbital-space cotangent, and for a raw-AO orbital those cotangents are
/// already AO-indexed.
fn orb_leaf_and_coeff<'p>(
    orb_data: &Bound<'p, PyAny>,
) -> PyResult<(Bound<'p, PyAny>, numpy::ndarray::Array2<f64>)> {
    if orb_data.hasattr("mo_coefficients")? {
        let np = orb_data.py().import("numpy")?;
        let c_mo: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (orb_data.getattr("mo_coefficients")?,))?
            .extract()?;
        Ok((orb_data.getattr("aos_data")?, c_mo.as_array().to_owned()))
    } else {
        let n: usize = orb_data.getattr("num_ao")?.extract()?;
        Ok((orb_data.clone(), numpy::ndarray::Array2::<f64>::eye(n)))
    }
}

/// The AO leaf of an orb_data: `aos_data` for an `MOs_data` wrapper, the object itself
/// for a RAW-AO orbital (dense AO-AGP). Sibling of [`orb_leaf_and_coeff`] for the callers
/// that need only the leaf (structure/cell/home-atom reads).
fn orb_aos_leaf<'p>(orb_data: &Bound<'p, PyAny>) -> PyResult<Bound<'p, PyAny>> {
    if orb_data.hasattr("aos_data")? {
        orb_data.getattr("aos_data")
    } else {
        Ok(orb_data.clone())
    }
}

/// All atom-centered (pgto) AO leaves of an MO orb_data + their MO-coefficient
/// blocks. Composite leaves recurse -- PW children are skipped (columns dropped).
fn pgto_force_leaves(orb_data: &Bound<'_, PyAny>) -> PyResult<Vec<PgtoLeaf>> {
    let (aos, c_owned) = orb_leaf_and_coeff(orb_data)?;
    let c = c_owned.view(); // (num_orb, num_ao_total)
    let mut leaves = Vec::new();
    // A B-spline geminal has NO atom-centered pgto leaf: the control points sit on a fixed
    // cell-fractional grid, so dAO/dR = 0 and the determinant nuclear (Pulay) force is exactly
    // zero. Return no leaves -> the det force side contributes nothing (the electron-axis
    // kinetic force is served separately via `bspline_det_egrad_side`).
    if aos.get_type().name()?.to_string().contains("bspline") {
        return Ok(leaves);
    }
    // A pure plane-wave geminal likewise has NO atom-centered pgto leaf (the PW basis is
    // R-independent -> dAO/dR = 0, det nuclear/Pulay force exactly zero). The electron-axis
    // kinetic force is served separately via `pw_det_egrad_side`. (Composite pgto(+)pw is
    // handled below -- its PW child is skipped by the `atom_centered` filter.)
    if aos.get_type().name()?.to_string().contains("plane_wave") {
        return Ok(leaves);
    }
    if aos.get_type().name()?.to_string().contains("composite") {
        let mut off = 0usize;
        for child in aos.getattr("children")?.try_iter()? {
            let child = child?;
            let nao: usize = child.getattr("num_ao")?.extract()?;
            let atom_centered =
                child.hasattr("_nucleus_index_np")? || child.hasattr("_ragged_kernel_to_home")?;
            if atom_centered {
                let block = c.slice(numpy::ndarray::s![.., off..off + nao]);
                leaves.push(pgto_leaf_pieces(&child, &block)?);
            }
            off += nao;
        }
    } else {
        leaves.push(pgto_leaf_pieces(&aos, &c)?);
    }
    Ok(leaves)
}

/// Per-kernel-AO atom index of a (cart) pgto leaf (for the stress image-T term).
fn pgto_kernel_atom(aos_py: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let py = aos_py.py();
    let np = py.import("numpy")?;
    let attr = if aos_py.hasattr("_ragged_kernel_atom")?
        && !aos_py.getattr("_ragged_kernel_atom")?.is_none()
    {
        "_ragged_kernel_atom"
    } else {
        "_nucleus_index_jnp"
    };
    let v: Vec<i64> = np
        .call_method1("asarray", (aos_py.getattr(attr)?,))?
        .call_method0("tolist")?
        .extract()?;
    Ok(v.into_iter().map(|x| x as usize).collect())
}

type PgtoStressLeaf = (
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
    Vec<usize>,
);

/// Like `pgto_force_leaves` but also returns each leaf's per-kernel-AO atom map.
fn pgto_stress_leaves(orb_data: &Bound<'_, PyAny>) -> PyResult<Vec<PgtoStressLeaf>> {
    let py = orb_data.py();
    let np = py.import("numpy")?;
    let (aos, c_owned) = orb_leaf_and_coeff(orb_data)?;
    let c = c_owned.view();
    // A pure B-spline / plane-wave determinant has NO atom-centered pgto leaf (control points on a
    // fixed cell-fractional grid / R-independent PW basis): the det VALUE Pulay stress is exactly 0
    // (frac-invariant) and the det KINETIC stress is served separately (B-spline via
    // `bspline_kstress_side`, PW via `pw_lap_stress`/`pw_gou_stress`). Return no leaves, mirroring
    // `pgto_force_leaves`. (Composite pgto(+)pw / pgto(+)bspline is the `composite` branch below --
    // its non-pgto child is skipped by the atom-centered filter.)
    if aos.get_type().name()?.to_string().contains("bspline")
        || aos.get_type().name()?.to_string().contains("plane_wave")
    {
        return Ok(Vec::new());
    }
    let leaf = |child: &Bound<'_, PyAny>,
                block: numpy::ndarray::ArrayView2<f64>|
     -> PyResult<PgtoStressLeaf> {
        // cart leaf (sphe -> cart) + its kernel-atom map (from the cart leaf)
        if child.get_type().name()?.to_string().contains("sphe") {
            let ao_mod = py.import("jqmc.atomic_orbital")?;
            let res = ao_mod.call_method1("_aos_sphe_to_cart", (child,))?;
            let cart_py = res.get_item(0)?;
            let t: numpy::PyReadonlyArray2<f64> =
                np.call_method1("asarray", (res.get_item(1)?,))?.extract()?;
            let cart = atomic_orbital::AosGtoData::from_py(&cart_py)?;
            Ok((
                cart,
                block.dot(&t.as_array()),
                pgto_home_atom(&cart_py)?,
                pgto_kernel_atom(&cart_py)?,
            ))
        } else {
            Ok((
                atomic_orbital::AosGtoData::from_py(child)?,
                block.to_owned(),
                pgto_home_atom(child)?,
                pgto_kernel_atom(child)?,
            ))
        }
    };
    let mut out = Vec::new();
    if aos.get_type().name()?.to_string().contains("composite") {
        let mut off = 0usize;
        for child in aos.getattr("children")?.try_iter()? {
            let child = child?;
            let nao: usize = child.getattr("num_ao")?.extract()?;
            if child.hasattr("_nucleus_index_np")? || child.hasattr("_ragged_kernel_to_home")? {
                out.push(leaf(
                    &child,
                    c.slice(numpy::ndarray::s![.., off..off + nao]),
                )?);
            }
            off += nao;
        }
    } else {
        out.push(leaf(&aos, c.view())?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "grad_lnPsi_pbc_wrt_strain", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_lnpsi_pbc_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let f = param_grad_el::gem_forward(&gem, &ru, &rd);
    let bar_orb_up = f.wg.clone(); // dln|det|/dorb_up
    let bar_orb_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    for (side, bar_orb, r) in [
        ("orb_data_up_spin", &bar_orb_up, &ru),
        ("orb_data_dn_spin", &bar_orb_dn, &rd),
    ] {
        for (cart_aos, c_cart, home, kernel_atom) in
            pgto_stress_leaves(&geminal_data.getattr(side)?)?
        {
            stress_pbc::pgto_det_stress_side(
                &cart_aos,
                &c_cart,
                bar_orb,
                r,
                &cell.view(),
                &home,
                &kernel_atom,
                &positions.view(),
                &mut out,
            );
        }
    }
    // Jastrow J1/J2 strain
    if !jastrow_data.is_none() {
        let j1o = jastrow_data.getattr("jastrow_one_body_data")?;
        let j2o = jastrow_data.getattr("jastrow_two_body_data")?;
        let j1 = if j1o.is_none() {
            None
        } else {
            Some(data::JastrowOneBodyData::from_py(&j1o)?)
        };
        let j2 = if j2o.is_none() {
            None
        } else {
            Some(data::JastrowTwoBodyData::from_py(&j2o)?)
        };
        // J1 and J2 Williamson metrics strain with the physical lattice.
        stress_pbc::jastrow_strain(j1.as_ref(), j2.as_ref(), &ru, &rd, &mut out);
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_T_det_pbc_wrt_strain", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_t_det_pbc_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let zu = numpy::ndarray::Array2::<f64>::zeros((ru.nrows(), 3));
    let zd = numpy::ndarray::Array2::<f64>::zeros((rd.nrows(), 3));
    let b = param_grad_el::geminal_reverse(&gem, &ru, &rd, &zu, &zd);
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    for (side, bar_orb, bar_lap, r) in [
        ("orb_data_up_spin", &b.bar_orb_up, &b.bar_lap_up, &ru),
        ("orb_data_dn_spin", &b.bar_orb_dn, &b.bar_lap_dn, &rd),
    ] {
        let od_py = geminal_data.getattr(side)?;
        for (cart_aos, c_cart, home, kernel_atom) in pgto_stress_leaves(&od_py)? {
            stress_pbc::pgto_kstress_side(
                &cart_aos,
                &c_cart,
                bar_orb,
                bar_lap,
                None,
                r,
                &cell.view(),
                &home,
                &kernel_atom,
                &positions.view(),
                &mut out,
            );
        }
        let orb = determinant::OrbData::from_py(&od_py)?;
        if let Some(pw) = &orb.pw {
            let np = py.import("numpy")?;
            let c_mo: numpy::PyReadonlyArray2<f64> = np
                .call_method1("asarray", (od_py.getattr("mo_coefficients")?,))?
                .extract()?;
            let c = c_mo.as_array();
            let pw0 = if orb.gto_first { orb.aos.num_ao } else { 0 };
            let c_pw = c.slice(numpy::ndarray::s![.., pw0..pw0 + pw.num_ao]);
            let cm_lap_pw = c_pw.t().dot(bar_lap);
            stress_pbc::pw_lap_stress(pw, &cm_lap_pw, r, &mut out);
        }
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// `(cart_j3_aos, T_mol (n_sphe,n_cart), home_atom)` for a bare pgto J3 AO leaf.
fn pgto_j3_pieces(
    j3_leaf: &Bound<'_, PyAny>,
) -> PyResult<(
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
)> {
    let py = j3_leaf.py();
    let np = py.import("numpy")?;
    let home = pgto_home_atom(j3_leaf)?;
    if j3_leaf.get_type().name()?.to_string().contains("sphe") {
        let ao_mod = py.import("jqmc.atomic_orbital")?;
        let res = ao_mod.call_method1("_aos_sphe_to_cart", (j3_leaf,))?;
        let cart_aos = atomic_orbital::AosGtoData::from_py(&res.get_item(0)?)?;
        let t_mol: numpy::PyReadonlyArray2<f64> =
            np.call_method1("asarray", (res.get_item(1)?,))?.extract()?;
        Ok((cart_aos, t_mol.as_array().to_owned(), home))
    } else {
        let n: usize = j3_leaf.getattr("num_ao")?.extract()?;
        let cart_aos = atomic_orbital::AosGtoData::from_py(j3_leaf)?;
        Ok((cart_aos, numpy::ndarray::Array2::<f64>::eye(n), home))
    }
}

/// `(cart_j3_aos, T_mol, home_atom, kernel_atom)` of [`pgto_j3_stress_pieces`].
type PgtoJ3StressPieces = (
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
    Vec<usize>,
);

/// `(cart_j3_aos, T_mol, home_atom, kernel_atom)` for a bare pgto J3 AO leaf
/// (stress variant of [`pgto_j3_pieces`], adds the per-kernel-AO atom map).
fn pgto_j3_stress_pieces(j3_leaf: &Bound<'_, PyAny>) -> PyResult<PgtoJ3StressPieces> {
    let py = j3_leaf.py();
    let np = py.import("numpy")?;
    // home/kernel are per CART-AO (field_stress indexes by cart-AO row), unlike the
    // force J3 path which scatters by the sphe-basis index.
    if j3_leaf.get_type().name()?.to_string().contains("sphe") {
        let ao_mod = py.import("jqmc.atomic_orbital")?;
        let res = ao_mod.call_method1("_aos_sphe_to_cart", (j3_leaf,))?;
        let cart_py = res.get_item(0)?;
        let cart_aos = atomic_orbital::AosGtoData::from_py(&cart_py)?;
        let t_mol: numpy::PyReadonlyArray2<f64> =
            np.call_method1("asarray", (res.get_item(1)?,))?.extract()?;
        Ok((
            cart_aos,
            t_mol.as_array().to_owned(),
            pgto_home_atom(&cart_py)?,
            pgto_kernel_atom(&cart_py)?,
        ))
    } else {
        let n: usize = j3_leaf.getattr("num_ao")?.extract()?;
        let cart_aos = atomic_orbital::AosGtoData::from_py(j3_leaf)?;
        Ok((
            cart_aos,
            numpy::ndarray::Array2::<f64>::eye(n),
            pgto_home_atom(j3_leaf)?,
            pgto_kernel_atom(j3_leaf)?,
        ))
    }
}

/// J3 leaf pieces for the AO LEAF of a real periodic J3 orbital: a bare pGTO leaf or an
/// `AOs_composite_data` pGTO(+)PW / pGTO(+)B-spline. Returns the pGTO child's cart AO +
/// T_mol + both spherical-row (force) and Cartesian-row (stress) home-atom maps + the
/// Cartesian kernel-atom map, plus the optional non-atom-centred child and `gto_first`.
/// An MO-wrapped J3 must first be lowered to AO rank (`J3_ao = C^T J3_mo C`, see
/// [`lower_real_j3`]); the pieces then describe exactly the AO basis the lowered
/// `j_matrix` indexes, so they never depend on `C` (issue #104: passing the `MOs_data`
/// wrapper here rejected the QE converter's dense `-j3 mo` orbital, and the B-spline
/// child was misread as the pGTO child). `extra = None` reproduces the single-leaf path.
#[allow(clippy::type_complexity)]
fn pgto_j3_pieces_composite(
    leaf: &Bound<'_, PyAny>,
) -> PyResult<(
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Option<forces_pbc::J3ExtraOwned>,
    bool,
)> {
    if leaf.hasattr("mo_coefficients")? {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pgto_j3_pieces_composite takes the AO leaf of a J3 lowered to AO rank, not an MOs_data",
        ));
    }
    if leaf.hasattr("children")? {
        let children = leaf.getattr("children")?;
        let n: usize = children.len()?;
        let mut gto_child: Option<Bound<'_, PyAny>> = None;
        let mut extra: Option<forces_pbc::J3ExtraOwned> = None;
        let (mut gto_idx, mut extra_idx) = (0usize, 0usize);
        for i in 0..n {
            let child = children.get_item(i)?;
            let name = child.get_type().name()?.to_string();
            let is_extra =
                child.hasattr("phase_is_sin")? || child.hasattr("spline_coefficients")?;
            // Only one block per family is held (issues #104 and #176): a second
            // plane-wave / B-spline child or a second GTO/pGTO child would leave the
            // native AO rank below the j_matrix rank. Refuse it like the OrbData parser
            // (issue #165); the free `_grad_lnJ3_pbc_strain` reaches this function
            // without parsing an OrbData first.
            if is_extra && extra.is_some() {
                return Err(determinant::unsupported_composite_err(
                    "real J3",
                    &format!(
                        "more than one plane-wave or B-spline child; got a second {name} child"
                    ),
                ));
            }
            if !is_extra && child.hasattr("_nucleus_index_jnp")? && gto_child.is_some() {
                return Err(determinant::unsupported_composite_err(
                    "real J3",
                    &format!("more than one GTO/pGTO child; got a second {name} child"),
                ));
            }
            if child.hasattr("phase_is_sin")? {
                extra = Some(forces_pbc::J3ExtraOwned::Pw(
                    atomic_orbital::AosPwData::from_py(&child)?,
                ));
                extra_idx = i;
            } else if child.hasattr("spline_coefficients")? {
                // `from_py` rejects a complex (twist) B-spline child.
                extra = Some(forces_pbc::J3ExtraOwned::Bspline(
                    bspline::AosBsplineData::from_py(&child)?,
                ));
                extra_idx = i;
            } else if child.hasattr("_nucleus_index_jnp")? {
                gto_idx = i;
                gto_child = Some(child);
            } else {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "real periodic J3 force/stress has no kernel for a composite {name} child"
                )));
            }
        }
        let gto_child = gto_child.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "composite J3 orbital without a gto child is unsupported",
            )
        })?;
        let gto_first = extra.is_none() || gto_idx < extra_idx;
        let force_home = pgto_home_atom(&gto_child)?;
        let (cart_j3, t_mol, stress_home, kernel) = pgto_j3_stress_pieces(&gto_child)?;
        Ok((
            cart_j3,
            t_mol,
            force_home,
            stress_home,
            kernel,
            extra,
            gto_first,
        ))
    } else {
        let force_home = pgto_home_atom(leaf)?;
        let (cart_j3, t_mol, stress_home, kernel) = pgto_j3_stress_pieces(leaf)?;
        Ok((cart_j3, t_mol, force_home, stress_home, kernel, None, true))
    }
}

/// Lower a REAL periodic J3 (`orb_data`, `j_matrix`) to AO rank for the cart J3 kernels:
/// `(J3 AO leaf, j_matrix_ao)`. An MO-wrapped J3 becomes `M_ao = C^T M_mo C`,
/// `j1_ao = C^T j1_mo` over the MO's `aos_data` (`gfmc_forces::lower_j3_to_ao`; exact
/// because the Gamma-point C is real); a raw-AO J3 passes through unchanged. Used by the
/// free J3 force/stress bindings, which feed the same kernels as the driver contexts
/// ([`j3_pbc_ctx_owned`], [`j3_stress_ctx_owned`]).
#[allow(clippy::type_complexity)]
fn lower_real_j3<'py>(
    j3d: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, numpy::ndarray::Array2<f64>)> {
    let np = j3d.py().import("numpy")?;
    let jm_obj = np.call_method1(
        "asarray",
        (j3d.getattr("j_matrix")?, np.getattr("float64")?),
    )?;
    let jm: numpy::PyReadonlyArray2<f64> = jm_obj.extract()?;
    let jm_arr = jm.as_array().to_owned();
    let orb_py = j3d.getattr("orb_data")?;
    if !orb_py.hasattr("mo_coefficients")? {
        return Ok((orb_py, jm_arr));
    }
    let orb = determinant::OrbData::from_py(&orb_py)?;
    let (_orb_low, jm_low) = gfmc_forces::lower_j3_to_ao(Some(&orb), Some(&jm_arr.view()))
        .expect("lower_j3_to_ao returns Some when orb+jm present");
    Ok((orb_py.getattr("aos_data")?, jm_low))
}

/// PBC DTN value strain `dJ_DTN/deps` (3x3, u+chi+f) -- test hook for `dtn_value_strain`.
#[pyfunction]
#[pyo3(name = "_dtn_value_strain", signature = (jastrow_data, r_up_carts, r_dn_carts))]
fn py_dtn_value_strain(
    py: Python<'_>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let dtn = dtn::JastrowDtnData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let s = stress_pbc::dtn_value_strain(&dtn, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// PBC J3 value strain `d(lnJ3)/deps` (3x3) -- test hook for `j3_pbc_value_strain`.
#[pyfunction]
#[pyo3(name = "_grad_lnJ3_pbc_strain", signature = (jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_lnj3_pbc_strain(
    py: Python<'_>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (od, jm) = lower_real_j3(&j3d)?;
    let (cart_j3, t_mol, _force_home, home, kernel, pw, gto_first) = pgto_j3_pieces_composite(&od)?;
    let sd = od.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    stress_pbc::j3_pbc_value_strain(
        &cart_j3,
        &t_mol,
        &jm.view(),
        &ru,
        &rd,
        &cell.view(),
        &home,
        &kernel,
        &positions.view(),
        pw.as_ref().map(|e| e.view()),
        gto_first,
        &mut out,
    );
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_T_pbc_wrt_R", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_pbc(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let with_j = jas.is_some();
    // gJ (J1/J2/DTN + J3) + gD; det-VJP cotangents (incl. gou cross term if Jastrow)
    let (gj_up, gj_dn) =
        param_grad_el::jastrow_grads(jas.as_ref(), j3o.as_ref(), j3v.as_ref(), &ru, &rd);
    let b = param_grad_el::geminal_reverse(&gem, &ru, &rd, &gj_up, &gj_dn);
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let n_atom = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((n_atom, 3));
    // orbital side (bar_gou Hessian term only when Jastrow present)
    for (side, bar_orb, bar_lap, bar_gou, r) in [
        (
            "orb_data_up_spin",
            &b.bar_orb_up,
            &b.bar_lap_up,
            &b.bar_gou,
            &ru,
        ),
        (
            "orb_data_dn_spin",
            &b.bar_orb_dn,
            &b.bar_lap_dn,
            &b.bar_godn,
            &rd,
        ),
    ] {
        let bg = if with_j { Some(bar_gou) } else { None };
        for (cart_aos, c_cart, home_atom) in pgto_force_leaves(&geminal_data.getattr(side)?)? {
            forces_pbc::pgto_det_force_side(
                &cart_aos,
                &c_cart,
                bar_orb,
                bar_lap,
                bg,
                r,
                &cell.view(),
                &home_atom,
                n_atom,
                &mut out,
            );
        }
    }
    // Jastrow side: J1 (PBC smooth-dist) and J3 (below); J2 has no ion dependence.
    if with_j {
        let (gd_up, gd_dn) = param_grad_el::det_grad_field(&gem, &ru, &rd);
        let n_up = ru.nrows();
        let n_e = n_up + rd.nrows();
        let mut bar_gj = numpy::ndarray::Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                let (gj, gd) = if e < n_up {
                    (gj_up[[e, c]], gd_up[[e, c]])
                } else {
                    let l = e - n_up;
                    (gj_dn[[l, c]], gd_dn[[l, c]])
                };
                bar_gj[[e, c]] = -(gj + gd);
            }
        }
        let j1_obj = jastrow_data.getattr("jastrow_one_body_data")?;
        if !j1_obj.is_none() {
            let j1 = data::JastrowOneBodyData::from_py(&j1_obj)?;
            forces_pbc::j1_pbc_side(&j1, &ru, &rd, &bar_gj, &mut out);
        }
        // J3 PBC side (pgto leaf or pgto(+)PW / pgto(+)B-spline composite; an MO-wrapped J3
        // is lowered to AO rank first)
        // `j3m` is `None` for an absent or inactive J3 (issue #170).
        if j3m.is_some() {
            let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
            let (j3_leaf, jm) = lower_real_j3(&j3d)?;
            let (cart_j3, t_mol, home_j3, _stress_home, _kernel, pw, gto_first) =
                pgto_j3_pieces_composite(&j3_leaf)?;
            let n_up = ru.nrows();
            let bg_up = bar_gj.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
            let bg_dn = bar_gj.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
            forces_pbc::j3_pbc_side_w(
                &cart_j3,
                &t_mol,
                &jm.view(),
                &ru,
                &rd,
                &bg_up,
                &bg_dn,
                &cell.view(),
                &home_j3,
                None,
                pw.as_ref().map(|e| e.view()),
                gto_first,
                &mut out,
            );
        }
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// Periodic electron-axis det-kinetic gradient `dT_det/dr` `(N_e,3)` (jnone;
/// det identity => no gou term). Verifies vs jax.grad of compute_kinetic_energy
/// (jnone). The building block for the PBC GFMC HF force's continuum-kinetic
/// electron axis.
#[pyfunction]
#[pyo3(name = "_grad_T_pbc_det_wrt_r", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_t_pbc_det_wrt_r(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let n_up = ru.nrows();
    let n_e = n_up + rd.nrows();
    let zu = numpy::ndarray::Array2::<f64>::zeros((n_up, 3));
    let zd = numpy::ndarray::Array2::<f64>::zeros((rd.nrows(), 3));
    let b = param_grad_el::geminal_reverse(&gem, &ru, &rd, &zu, &zd);
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((n_e, 3));
    for (side, bar_orb, bar_lap, r, off) in [
        (
            "orb_data_up_spin",
            &b.bar_orb_up,
            &b.bar_lap_up,
            &ru,
            0usize,
        ),
        ("orb_data_dn_spin", &b.bar_orb_dn, &b.bar_lap_dn, &rd, n_up),
    ] {
        let mut spin_out = numpy::ndarray::Array2::<f64>::zeros((r.nrows(), 3));
        for (cart_aos, c_cart, _home) in pgto_force_leaves(&geminal_data.getattr(side)?)? {
            forces_pbc::pgto_det_egrad_side(
                &cart_aos,
                &c_cart,
                bar_orb,
                bar_lap,
                None,
                r,
                &cell.view(),
                &mut spin_out,
            );
        }
        // B-spline geminal electron-axis kinetic (fixed grid -> no nuclear force, but the
        // electron move drives the AO grad/lap).
        let orb = if off == 0 { &gem.orb_up } else { &gem.orb_dn };
        if let Some(bs) = &orb.bspline {
            forces_pbc::bspline_det_egrad_side(
                bs,
                orb.mo_coefficients.as_ref(),
                bar_orb,
                bar_lap,
                None,
                r,
                &mut spin_out,
            );
        }
        for k in 0..r.nrows() {
            for d in 0..3 {
                out[[off + k, d]] = spin_out[[k, d]];
            }
        }
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// Periodic withJ (det+J1+J2+J3) electron-axis kinetic `dT/dr (N_e,3)`. Verifies
/// vs jax.grad(compute_kinetic_energy) for a j12/j123 periodic system.
#[pyfunction]
#[pyo3(name = "_grad_T_pbc_withj_wrt_r", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_pbc_withj_wrt_r(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let up_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
    let dn_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let n_atom = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    // Optional J3 context.
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let j3_owned = j3_pbc_ctx_owned(&j3d)?;
    let j3 = j3_owned.as_ref().map(
        |(orb, cart, t, jm, home, pw, gto_first)| gfmc_forces::J3Pbc {
            orb,
            cart_j3: cart,
            t_mol: t.view(),
            jm: jm.view(),
            home,
            pw: pw.as_ref().map(|e| e.view()),
            gto_first: *gto_first,
        },
    );
    let (_d_r, d_re) = gfmc_forces::pbc_kinetic_withj_grads(
        &gem,
        &jas,
        j3.as_ref(),
        &up_leaves,
        &dn_leaves,
        &cell.view(),
        n_atom,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        None,
        None,
        None,
        None,
    );
    Ok(d_re.into_pyarray(py).into_any().unbind())
}

/// Extract the owned J3 periodic context `(OrbData, cart_j3, T_mol, j_matrix, home)`
/// from a `jastrow_three_body_data` (None / all-zero j_matrix => no J3).
#[allow(clippy::type_complexity)]
fn j3_pbc_ctx_owned(
    j3d: &Bound<'_, PyAny>,
) -> PyResult<
    Option<(
        determinant::OrbData,
        atomic_orbital::AosGtoData,
        numpy::ndarray::Array2<f64>,
        numpy::ndarray::Array2<f64>,
        Vec<usize>,
        Option<forces_pbc::J3ExtraOwned>,
        bool,
    )>,
> {
    if j3d.is_none() {
        return Ok(None);
    }
    // `j_matrix` belongs to the jastrow evaluation zone and is therefore an
    // fp32 Python array in mixed mode.  Rust owns f64 coefficients internally
    // and selects f32 arithmetic from PrecisionPolicy, so widen once at this
    // construction boundary (never in the hot loop).
    let np = j3d.py().import("numpy")?;
    let jm_obj = np.call_method1(
        "asarray",
        (j3d.getattr("j_matrix")?, np.getattr("float64")?),
    )?;
    let jm: numpy::PyReadonlyArray2<f64> = jm_obj.extract()?;
    let jm_arr = jm.as_array().to_owned();
    if jm_arr.iter().all(|&x| x == 0.0) {
        return Ok(None);
    }
    let orb_py = j3d.getattr("orb_data")?;
    let orb = determinant::OrbData::from_py(&orb_py)?;
    // A PURE plane-wave / B-spline J3 leaf (no cart-lowerable pGTO AO) has NO
    // `cart_j3`/`t_mol`/`home` for the cart `pgto_deriv` force/stress path: its nuclear
    // Pulay is exactly 0 (verified: the J3 dR-contribution == 0 for both PW and B-spline,
    // the orbital carries no atom center) and its electron drift routes through the
    // leaf-owning batch evaluator instead (`j3_batch_ctx_owned` + the caller). Return None
    // here so the cart J3 kernels skip it (adding the correct zero R-Pulay).
    if j3_leaf_is_batch(&orb) {
        return Ok(None);
    }
    // Lower an MO-wrapped J3 (num_mo != num_ao) to its exact AO-equivalent: a raw-AO
    // orbital + AO-space j_matrix `M_ao = C^T M_mo C`, `j1_ao = C^T j1_mo` (see
    // `gfmc_forces::lower_j3_to_ao`). The periodic J3 FORCE kernels (`j3_pbc_side_w`,
    // `grad_j3_wrt_r_pbc`, `j3_pbc_egrad_side`) read the RAW `cart_j3`+`t_mol` basis and
    // slice `jm[.., 0..num_ao]` -- a genuine MO J3 (production lithium is 1<-26) panicked
    // that slice, so the periodic force stayed JAX-gated. Lowering is value/grad/lap
    // invariant (R enters only via the R-independent-C AO centers), matching the molecular
    // Pulay/HF path. VALID here because the periodic (real-Gamma) J3 `mo_coefficients` are
    // real; a complex-C twist J3 (Re-per-MO, not bilinearly lowerable) routes to the complex
    // ctx instead. `lower_j3_to_ao` is a bit-identical no-op for a raw-AO (non-MO) J3.
    let (orb_low, jm_low) = gfmc_forces::lower_j3_to_ao(Some(&orb), Some(&jm_arr.view()))
        .expect("lower_j3_to_ao returns Some when orb+jm present");
    // The raw-AO leaf for `cart_j3`/`t_mol`/`home`: unwrap a `MOs_data` wrapper to its AO
    // leaf so the pgto pieces describe the num_ao basis the lowered `jm_low` indexes.
    let leaf_py = if orb.mo_coefficients.is_some() {
        orb_py.getattr("aos_data")?
    } else {
        orb_py.clone()
    };
    let (cart_j3, t_mol, home, pw, gto_first) = if leaf_py.hasattr("children")? {
        let (cart, t, home, _stress_home, _kernel, pw, first) = pgto_j3_pieces_composite(&leaf_py)?;
        (cart, t, home, pw, first)
    } else {
        let (cart, t, home) = pgto_j3_pieces(&leaf_py)?;
        (cart, t, home, None, true)
    };
    Ok(Some((orb_low, cart_j3, t_mol, jm_low, home, pw, gto_first)))
}

/// True when the J3 leaf is a PURE plane-wave or B-spline orbital (no cart-lowerable
/// pGTO AO). Such a leaf has no atom-centered AO, so its nuclear-force (R) Pulay is
/// exactly 0 and its electron-drift (r) routes through the leaf-owning batch evaluator
/// [`jastrow_factor::compute_grads_and_laplacian_jastrow_three_body`] (which evaluates
/// the orbital grad via `orb_value_grad_lap`, serving PW / B-spline identically to the
/// determinant kinetic path), NOT the cart `pgto_deriv` kernels. A composite
/// (pGTO(+)PW) leaf keeps `aos.num_ao > 0` and stays on the cart / composite path.
fn j3_leaf_is_batch(orb: &determinant::OrbData) -> bool {
    orb.aos.num_ao == 0 && (orb.bspline.is_some() || orb.pw.is_some())
}

/// Real-path GFMC J3 parse shared by every projection / e_L entry point that feeds
/// `gfmc::mesh_fast_state`: `(None, None)` for an absent or all-zero `j_matrix`
/// (a vacuous J3 contributes nothing), else the owned `(orbital, j_matrix)`. A
/// nonzero MOLECULAR MO-wrapped J3 (the production `-j3 mo` form) is lowered to its
/// exact raw-AO equivalent (`gfmc::lower_molecular_mo_j3`) so the mesh
/// streams it instead of taking the full-value J3 slow fallback.
#[allow(clippy::type_complexity)]
fn gfmc_real_j3_parse(
    j3d: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    if j3d.is_none() {
        return Ok((None, None));
    }
    let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
    let jm_arr = jm.as_array().to_owned();
    if jm_arr.iter().all(|&x| x == 0.0) {
        return Ok((None, None));
    }
    let orb = determinant::OrbData::from_py(&j3d.getattr("orb_data")?)?;
    let (orb, jm_arr) = gfmc::lower_molecular_mo_j3(orb, jm_arr);
    Ok((Some(orb), Some(jm_arr)))
}

/// Owned `(OrbData, j_matrix)` for a PURE PW / B-spline J3 leaf; `None` for absent /
/// all-zero / cart-lowerable J3. Companion to [`j3_pbc_ctx_owned`] (which returns None
/// for the same batch leaves): the caller adds the batch electron drift on top of the
/// J3-less base drift.
#[allow(clippy::type_complexity)]
fn j3_batch_ctx_owned(
    j3d: &Bound<'_, PyAny>,
) -> PyResult<Option<(determinant::OrbData, numpy::ndarray::Array2<f64>)>> {
    if j3d.is_none() {
        return Ok(None);
    }
    let np = j3d.py().import("numpy")?;
    let jm_obj = np.call_method1(
        "asarray",
        (j3d.getattr("j_matrix")?, np.getattr("float64")?),
    )?;
    let jm: numpy::PyReadonlyArray2<f64> = jm_obj.extract()?;
    let jm_arr = jm.as_array().to_owned();
    if jm_arr.iter().all(|&x| x == 0.0) {
        return Ok(None);
    }
    let orb = determinant::OrbData::from_py(&j3d.getattr("orb_data")?)?;
    if j3_leaf_is_batch(&orb) {
        Ok(Some((orb, jm_arr)))
    } else {
        Ok(None)
    }
}

/// Periodic Pulay `dln|detG|/dR` `(n_atom,3)` (the lndet part of the GFMC Pulay
/// force / mesh terms). Uses the lndet cotangents (wg up, G_i_n_v*Q dn) scattered by
/// pgto_det_force_side (bar_lap=0). Verifies vs jax.grad of compute_ln_det.
#[pyfunction]
#[pyo3(name = "_grad_lnDet_pbc_wrt_R", signature = (geminal_data, r_up_carts, r_dn_carts))]
fn py_grad_lndet_pbc_wrt_r(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let f = param_grad_el::gem_forward(&gem, &ru, &rd);
    // lndet cotangents: orb_up value coeff = WG; orb_dn value coeff = (G_i_n_v*Q)[:N_dn]T.
    let bar_up = f.wg.clone();
    let bar_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let n_atom = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((n_atom, 3));
    for (side, bar, r) in [
        ("orb_data_up_spin", &bar_up, &ru),
        ("orb_data_dn_spin", &bar_dn, &rd),
    ] {
        let zlap = numpy::ndarray::Array2::<f64>::zeros(bar.raw_dim());
        for (cart_aos, c_cart, home_atom) in pgto_force_leaves(&geminal_data.getattr(side)?)? {
            forces_pbc::pgto_det_force_side(
                &cart_aos,
                &c_cart,
                bar,
                &zlap,
                None,
                r,
                &cell.view(),
                &home_atom,
                n_atom,
                &mut out,
            );
        }
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_T_pbc_wrt_strain", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_pbc_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let with_j = jas.is_some();
    // jastrow_grads already includes the DTN drift (jastrow_part dispatches on jdata.dtn),
    // so gj is the full jastrow drift used by the det gou cross + J1/J2 kinetic cotangent.
    let (gj_up, gj_dn) =
        param_grad_el::jastrow_grads(jas.as_ref(), j3o.as_ref(), j3v.as_ref(), &ru, &rd);
    let b = param_grad_el::geminal_reverse(&gem, &ru, &rd, &gj_up, &gj_dn);
    let up_leaf = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("aos_data")?;
    let sd = up_leaf.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let mut out = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    // orbital side: value (bar_orb) + lap (bar_lap) + gou (bar_gou Hessian, withJ only)
    for (side, bar_orb, bar_lap, bar_gou, r) in [
        (
            "orb_data_up_spin",
            &b.bar_orb_up,
            &b.bar_lap_up,
            &b.bar_gou,
            &ru,
        ),
        (
            "orb_data_dn_spin",
            &b.bar_orb_dn,
            &b.bar_lap_dn,
            &b.bar_godn,
            &rd,
        ),
    ] {
        let bg = if with_j { Some(bar_gou) } else { None };
        let od_py = geminal_data.getattr(side)?;
        for (cart_aos, c_cart, home, kernel_atom) in pgto_stress_leaves(&od_py)? {
            stress_pbc::pgto_kstress_side(
                &cart_aos,
                &c_cart,
                bar_orb,
                bar_lap,
                bg,
                r,
                &cell.view(),
                &home,
                &kernel_atom,
                &positions.view(),
                &mut out,
            );
        }
        // plane-wave child: lap (|kappa|^2) + gou (kappa<->kappa) strain
        let orb = determinant::OrbData::from_py(&od_py)?;
        if let Some(pw) = &orb.pw {
            let np = py.import("numpy")?;
            let c_mo: numpy::PyReadonlyArray2<f64> = np
                .call_method1("asarray", (od_py.getattr("mo_coefficients")?,))?
                .extract()?;
            let c = c_mo.as_array();
            let pw0 = if orb.gto_first { orb.aos.num_ao } else { 0 };
            let c_pw = c.slice(numpy::ndarray::s![.., pw0..pw0 + pw.num_ao]);
            let cm_lap_pw = c_pw.t().dot(bar_lap);
            stress_pbc::pw_lap_stress(pw, &cm_lap_pw, r, &mut out);
            if with_j {
                let cm_gou_pw = [
                    c_pw.t().dot(&bar_gou[0]),
                    c_pw.t().dot(&bar_gou[1]),
                    c_pw.t().dot(&bar_gou[2]),
                ];
                stress_pbc::pw_gou_stress(pw, &cm_gou_pw, r, &mut out);
            }
        }
    }
    // Jastrow J1/J2 cell-strain (cot = gJ + gD).
    if with_j {
        let (gd_up, gd_dn) = param_grad_el::det_grad_field(&gem, &ru, &rd);
        let n_up = ru.nrows();
        let n_e = n_up + rd.nrows();
        let mut cot = numpy::ndarray::Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                let (gj, gd) = if e < n_up {
                    (gj_up[[e, c]], gd_up[[e, c]])
                } else {
                    let l = e - n_up;
                    (gj_dn[[l, c]], gd_dn[[l, c]])
                };
                cot[[e, c]] = gj + gd;
            }
        }
        let j1o = jastrow_data.getattr("jastrow_one_body_data")?;
        let j2o = jastrow_data.getattr("jastrow_two_body_data")?;
        let j1 = if j1o.is_none() {
            None
        } else {
            Some(data::JastrowOneBodyData::from_py(&j1o)?)
        };
        let j2 = if j2o.is_none() {
            None
        } else {
            Some(data::JastrowTwoBodyData::from_py(&j2o)?)
        };
        // J1 and J2 Williamson metrics strain with the physical lattice.
        stress_pbc::jastrow_kstress(j1.as_ref(), j2.as_ref(), &cot, &ru, &rd, &mut out);
        // J3 PBC kinetic stress (pgto leaf or pgto(+)PW / pgto(+)B-spline composite; an
        // MO-wrapped J3 is lowered to AO rank first)
        // `j3m` is `None` for an absent or inactive J3 (issue #170).
        if j3m.is_some() {
            let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
            let (j3_leaf, jm) = lower_real_j3(&j3d)?;
            let (cart_j3, t_mol, _force_home, home_j3, kernel_j3, pw, gto_first) =
                pgto_j3_pieces_composite(&j3_leaf)?;
            // bar on gJ3 = -(gJ+gD) = -cot
            let bg_up = -&cot.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
            let bg_dn = -&cot.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
            stress_pbc::j3_pbc_kstress_w(
                &cart_j3,
                &t_mol,
                &jm.view(),
                &ru,
                &rd,
                &bg_up,
                &bg_dn,
                None,
                &cell.view(),
                &home_j3,
                &kernel_j3,
                &positions.view(),
                pw.as_ref().map(|e| e.view()),
                gto_first,
                &mut out,
            );
        }
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}

// ---- Nuclear forces de_L/dR (Cartesian leaf via spherical-to-Cartesian lowering) ----

/// Convert any orbital leaf to a CART-basis `OrbData` (`orb = C @ AO_cart`):
/// MOs -> `to_cartesian()`; bare sphe AO -> `_aos_sphe_to_cart` (c2s as `C`);
/// cart AO -> as-is. Lets the cart AO-derivative kernel serve sphe systems too.
fn orbdata_to_cart(orb_data: &Bound<'_, PyAny>) -> PyResult<determinant::OrbData> {
    if orb_data.hasattr("to_cartesian")? {
        let cart = orb_data.call_method0("to_cartesian")?;
        return determinant::OrbData::from_py(&cart);
    }
    if orb_data.get_type().name()?.to_string().contains("sphe") {
        let py = orb_data.py();
        let ao_mod = py.import("jqmc.atomic_orbital")?;
        let res = ao_mod.call_method1("_aos_sphe_to_cart", (orb_data,))?;
        let cart_aos = res.get_item(0)?;
        let np = py.import("numpy")?;
        let t: numpy::PyReadonlyArray2<f64> =
            np.call_method1("asarray", (res.get_item(1)?,))?.extract()?;
        let aos = atomic_orbital::AosGtoData::from_py(&cart_aos)?;
        return Ok(determinant::OrbData::new_mo(aos, t.as_array().to_owned()));
    }
    determinant::OrbData::from_py(orb_data)
}

/// Cart-leaf `GeminalData` (orb_up/dn converted via [`orbdata_to_cart`]).
fn geminal_cart_from_py(geminal_data: &Bound<'_, PyAny>) -> PyResult<determinant::GeminalData> {
    let orb_up = orbdata_to_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let orb_dn = orbdata_to_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let lambda = if geminal_data.hasattr("lambda_matrix")? {
        let lam: numpy::PyReadonlyArray2<f64> = geminal_data.getattr("lambda_matrix")?.extract()?;
        Some(lam.as_array().to_owned())
    } else {
        None
    };
    let orb_num_up: usize = geminal_data.getattr("orb_num_up")?.extract()?;
    let orb_num_dn: usize = geminal_data.getattr("orb_num_dn")?.extract()?;
    Ok(determinant::GeminalData {
        orb_up,
        orb_dn,
        lambda,
        orb_num_up,
        orb_num_dn,
    })
}

/// `nucleus_index` from the CART leaf an orb converts to.
fn extract_nuc_cart(orb_data: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let conv = if orb_data.hasattr("to_cartesian")? {
        orb_data.call_method0("to_cartesian")?
    } else if orb_data.get_type().name()?.to_string().contains("sphe")
        && !orb_data.hasattr("aos_data")?
    {
        let py = orb_data.py();
        let ao_mod = py.import("jqmc.atomic_orbital")?;
        ao_mod
            .call_method1("_aos_sphe_to_cart", (orb_data,))?
            .get_item(0)?
    } else {
        orb_data.clone()
    };
    extract_nuc(&conv)
}

/// `(cart J3 orbital, cart j_matrix, per-cart-AO nucleus index)` of [`parse_j3_cart_force`].
type J3OrbParts = (
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
    Option<Vec<usize>>,
);

/// Cart-lower a molecular J3 `(orb_data, j_matrix)` for the raw-AO force/stress/drift
/// kernels (`forces::grad_j3_wrt_r`, `forces::j3_side`, `stress::j3_grad_wrt_r`), which
/// evaluate cartesian AO derivatives via `ao_cart_deriv` (empty `poly_x/y/z` on a sphe
/// leaf -> panic). Reuses the established convention: `orbdata_to_cart` (raw sphe->cart
/// via `_aos_sphe_to_cart`, or MO `to_cartesian`) yields an MO-wrapped cart orbital whose
/// `C` is the sphe->cart transform `T`, then `gfmc_forces::lower_j3_to_ao` lowers it to a
/// RAW cart-AO orbital with `j_matrix_cart = T^T M T` (block) and `T^T j1`. The J3 factor
/// `A^T M A` is basis-invariant, so every physical force/stress/drift observable is
/// unchanged; the returned nucleus index is per-CART-AO (`extract_nuc_cart`) so the Pulay
/// scatter lands on the right atom. Identity for a cart raw-AO leaf. `(None,None,None)` if
/// there is no J3. NOTE: do NOT use for the j_matrix PARAMETER gradient
/// (`grad_j3_jmatrix`/`grad_t_j3`/`grad_ecp_j3`), whose result is basis-DEPENDENT and must
/// stay in the stored (sphe) basis; those use the sphe-safe `orb_value_grad_lap` and never
/// reach `ao_cart_deriv`.
fn parse_j3_cart_force(jastrow_data: &Bound<'_, PyAny>) -> PyResult<J3OrbParts> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Ok((None, None, None));
    }
    lower_j3_leaf_cart(&j3d.getattr("orb_data")?, &j3d.getattr("j_matrix")?)
        .map(|(o, m, n)| (Some(o), Some(m), Some(n)))
}

/// Cart-lowered `(raw-cart-AO orbital, cart j_matrix, per-cart-AO nucleus index)` for a J3
/// AO leaf + its (spherical-basis) `j_matrix`. See [`parse_j3_cart_force`].
fn lower_j3_leaf_cart(
    orb_leaf: &Bound<'_, PyAny>,
    j_matrix: &Bound<'_, PyAny>,
) -> PyResult<(
    determinant::OrbData,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
)> {
    // See `j3_pbc_ctx_owned`: mixed precision supplies an fp32 j_matrix, while
    // the owned Rust lowering representation is f64.
    let np = j_matrix.py().import("numpy")?;
    let jm_obj = np.call_method1("asarray", (j_matrix, np.getattr("float64")?))?;
    let jm: numpy::PyReadonlyArray2<f64> = jm_obj.extract()?;
    let mo = orbdata_to_cart(orb_leaf)?;
    let (raw, jm_cart) = gfmc_forces::lower_j3_to_ao(Some(&mo), Some(&jm.as_array()))
        .expect("lower_j3_to_ao returns Some when orb+jm present");
    let nuc = extract_nuc_cart(orb_leaf)?;
    Ok((raw, jm_cart, nuc))
}

/// Per-AO atom index `nucleus_index` from an orb_data (MOs or AOs leaf).
fn extract_nuc(orb_data: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let aos = if orb_data.hasattr("aos_data")? {
        orb_data.getattr("aos_data")?
    } else {
        orb_data.clone()
    };
    let py = aos.py();
    let np = py.import("numpy")?;
    let v: Vec<i64> = np
        .call_method1("asarray", (aos.getattr("_nucleus_index_jnp")?,))?
        .call_method0("tolist")?
        .extract()?;
    Ok(v.into_iter().map(|x| x as usize).collect())
}

#[pyfunction]
#[pyo3(name = "grad_lndet_wrt_R")]
fn py_grad_lndet_r(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let n_atom = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let g = forces::grad_lndet_wrt_r(
        &gem,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        n_atom,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_J1_wrt_R")]
fn py_grad_j1_r(
    py: Python<'_>,
    jastrow_one_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let j1 = data::JastrowOneBodyData::from_py(jastrow_one_body_data)?;
    let g = forces::grad_j1_wrt_r(&j1, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_J3_wrt_R")]
fn py_grad_j3_r(
    py: Python<'_>,
    jastrow_three_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let (orb, jm, nuc) = lower_j3_leaf_cart(
        &jastrow_three_body_data.getattr("orb_data")?,
        &jastrow_three_body_data.getattr("j_matrix")?,
    )?;
    let n_atom = jastrow_three_body_data
        .getattr("orb_data")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let g = forces::grad_j3_wrt_r(
        &orb,
        &jm.view(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc,
        n_atom,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_J_dtn_wrt_R")]
fn py_grad_j_dtn_r(
    py: Python<'_>,
    jastrow_dtn_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let d = dtn::JastrowDtnData::from_py(jastrow_dtn_data)?;
    let g = forces_dtn::grad_j_dtn_wrt_r_atom(&d, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_V_wrt_R")]
fn py_grad_v_r(
    py: Python<'_>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let znum: Vec<f64> = {
        let np = cp.py().import("numpy")?;
        np.call_method1("asarray", (sd.getattr("atomic_numbers")?,))?
            .call_method0("tolist")?
            .extract()?
    };
    let zc: Vec<f64> = {
        let o = cp.getattr("z_cores")?;
        if o.is_none() {
            vec![0.0; znum.len()]
        } else {
            let np = cp.py().import("numpy")?;
            let v: Vec<f64> = np
                .call_method1("asarray", (o,))?
                .call_method0("tolist")?
                .extract()?;
            if v.is_empty() {
                vec![0.0; znum.len()]
            } else {
                v
            }
        }
    };
    let z_eff: Vec<f64> = znum.iter().zip(zc.iter()).map(|(z, c)| z - c).collect();
    let g = forces::grad_v_wrt_r(
        &pos.view(),
        &z_eff,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_T_wrt_R", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_r(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let n_atom = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let (jas, ..) = parse_jastrow_full(jastrow_data)?;
    // J3: cart-lower for the raw-AO force kernel (`j3_side`); nuc is per-cart-AO.
    let (j3o, j3m, nuc_j3) = parse_j3_cart_force(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    // J1 nucleus index (only when Jastrow present)
    let j1 = if jas.is_some() && !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    let g = forces::grad_t_wrt_r(
        &gem,
        jas.as_ref(),
        j1.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
        n_atom,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_lnPsi_wrt_strain", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_lnpsi_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let positions = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let (jas, ..) = parse_jastrow_full(jastrow_data)?;
    // J3: cart-lower for the raw-AO Pulay kernel (`grad_j3_wrt_r`); nuc is per-cart-AO.
    let (j3o, j3m, nuc_j3) = parse_j3_cart_force(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j1 = if jas.is_some() && !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    let mut s = stress::grad_lnpsi_wrt_strain(
        &gem,
        jas.as_ref(),
        j1.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &positions.view(),
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
    );
    // Molecular ``eps`` is the coordinate-affine (virial) capability, not a
    // physical cell stress. ``stress::grad_lnpsi_wrt_strain`` already receives
    // the DTN electron drift through ``JastrowData``; its nuclear-force assembly,
    // however, owns only J1/J3. Add exactly the missing DTN nuclear virial
    // ``dJ_DTN/dR . R``. Adding ``dtn_value_strain`` here would double-count the
    // electron part.
    if !jastrow_data.is_none()
        && jastrow_data.hasattr("jastrow_dtn_data")?
        && !jastrow_data.getattr("jastrow_dtn_data")?.is_none()
    {
        let dtn = dtn::JastrowDtnData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
        let d_r_dtn =
            forces_dtn::grad_j_dtn_wrt_r_atom(&dtn, &r_up_carts.as_array(), &r_dn_carts.as_array());
        for c in 0..3 {
            for axis in 0..3 {
                for ia in 0..positions.nrows() {
                    s[[c, axis]] += d_r_dtn[[ia, c]] * positions[[ia, axis]];
                }
            }
        }
    }
    Ok(s.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_strain", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts))]
fn py_grad_el_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let np = cp.py().import("numpy")?;
    let znum: Vec<f64> = np
        .call_method1("asarray", (sd.getattr("atomic_numbers")?,))?
        .call_method0("tolist")?
        .extract()?;
    let zc: Vec<f64> = {
        let o = cp.getattr("z_cores")?;
        if o.is_none() {
            vec![0.0; znum.len()]
        } else {
            let v: Vec<f64> = np
                .call_method1("asarray", (o,))?
                .call_method0("tolist")?
                .extract()?;
            if v.is_empty() {
                vec![0.0; znum.len()]
            } else {
                v
            }
        }
    };
    let z_eff: Vec<f64> = znum.iter().zip(zc.iter()).map(|(z, c)| z - c).collect();
    let s = stress::grad_el_wrt_strain_jnone(
        &gem,
        &pos.view(),
        &z_eff,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_strain_withJ", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts))]
fn py_grad_el_strain_withj(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let np = cp.py().import("numpy")?;
    let znum: Vec<f64> = np
        .call_method1("asarray", (sd.getattr("atomic_numbers")?,))?
        .call_method0("tolist")?
        .extract()?;
    let zc: Vec<f64> = {
        let o = cp.getattr("z_cores")?;
        if o.is_none() {
            vec![0.0; znum.len()]
        } else {
            let v: Vec<f64> = np
                .call_method1("asarray", (o,))?
                .call_method0("tolist")?
                .extract()?;
            if v.is_empty() {
                vec![0.0; znum.len()]
            } else {
                v
            }
        }
    };
    let z_eff: Vec<f64> = znum.iter().zip(zc.iter()).map(|(z, c)| z - c).collect();
    let (jas, ..) = parse_jastrow_full(jastrow_data)?;
    // J3: cart-lower for the raw-AO stress kernels (`grad_j3_wrt_r`/`j3_grad_wrt_r`).
    let (j3o, j3m, nuc_j3) = parse_j3_cart_force(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j1 = if jas.is_some() && !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    let j2 = if jas.is_some() && !jastrow_data.getattr("jastrow_two_body_data")?.is_none() {
        Some(data::JastrowTwoBodyData::from_py(
            &jastrow_data.getattr("jastrow_two_body_data")?,
        )?)
    } else {
        None
    };
    let s = stress::grad_el_wrt_strain_withj(
        &gem,
        jas.as_ref(),
        j1.as_ref(),
        j2.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &pos.view(),
        &z_eff,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_strain_ecp", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(clippy::too_many_arguments)]
fn py_grad_el_strain_ecp(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let znum: Vec<f64> = np
        .call_method1("asarray", (sd.getattr("atomic_numbers")?,))?
        .call_method0("tolist")?
        .extract()?;
    let zc: Vec<f64> = {
        let o = cp.getattr("z_cores")?;
        if o.is_none() {
            vec![0.0; znum.len()]
        } else {
            let v: Vec<f64> = np
                .call_method1("asarray", (o,))?
                .call_method0("tolist")?
                .extract()?;
            if v.is_empty() {
                vec![0.0; znum.len()]
            } else {
                v
            }
        }
    };
    let z_eff: Vec<f64> = znum.iter().zip(zc.iter()).map(|(z, c)| z - c).collect();
    // FULL ECP channel arrays (all l), used by the electron-axis pieces & local force.
    let nuc_all = geti("nucleus_index")?;
    let ang_all = geti("ang_moms")?;
    let exps_all = getf("exponents")?;
    let coefs_all = getf("coefficients")?;
    let powers_all = getf("powers")?;
    let max_ang = geti("max_ang_mom_plus_1")?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let n_e = ru.nrows() + rd.nrows();
    // dV_ecp/dr = local + non-local (Ne,3)
    let mut der_r_ecp = forces_ecp::grad_vlocal_wrt_r(
        &pos.view(),
        &nuc_all,
        &ang_all,
        &exps_all,
        &coefs_all,
        &powers_all,
        &max_ang,
        &ru,
        &rd,
    );
    der_r_ecp += &forces_ecp::grad_v_ecp_wrt_r(
        &gem,
        &ru,
        &rd,
        &pos.view(),
        &nuc_all,
        &ang_all,
        &exps_all,
        &coefs_all,
        &powers_all,
        &max_ang,
        weights.as_slice()?,
        &grid_points.as_array(),
    );
    debug_assert_eq!(der_r_ecp.nrows(), n_e);
    // dV_ecp/dR = local (translation id) + non-local (wf-ratio + explicit)
    let mut d_r_ecp = forces_ecp::grad_vlocal_force(
        &pos.view(),
        &nuc_all,
        &ang_all,
        &exps_all,
        &coefs_all,
        &powers_all,
        &max_ang,
        &ru,
        &rd,
    );
    {
        let jas = data::JastrowData::from_py(jastrow_data)?;
        let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
        let j3v = j3m.as_ref().map(|m| m.view());
        let j3 = match (&j3o, &j3v) {
            (Some(o), Some(v)) => Some((o, v)),
            _ => None,
        };
        let (pos2, nuc_nl, ang_nl, exps_nl, coefs_nl, powers_nl) = parse_ecp_arrays(cp)?;
        d_r_ecp += &forces_ecp::grad_ecp_force(
            &gem,
            &jas,
            j3,
            &ru,
            &rd,
            &nuc_up,
            &nuc_dn,
            &pos2.view(),
            &nuc_nl,
            &ang_nl,
            &exps_nl,
            &coefs_nl,
            &powers_nl,
            weights.as_slice()?,
            &grid_points.as_array(),
        );
    }
    let s = stress::grad_el_wrt_strain_ecp(
        &gem,
        &pos.view(),
        &z_eff,
        &der_r_ecp,
        &d_r_ecp,
        &ru,
        &rd,
        &nuc_up,
        &nuc_dn,
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_strain_dtn", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts))]
fn py_grad_el_strain_dtn(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let np = cp.py().import("numpy")?;
    let znum: Vec<f64> = np
        .call_method1("asarray", (sd.getattr("atomic_numbers")?,))?
        .call_method0("tolist")?
        .extract()?;
    let zc: Vec<f64> = {
        let o = cp.getattr("z_cores")?;
        if o.is_none() {
            vec![0.0; znum.len()]
        } else {
            let v: Vec<f64> = np
                .call_method1("asarray", (o,))?
                .call_method0("tolist")?
                .extract()?;
            if v.is_empty() {
                vec![0.0; znum.len()]
            } else {
                v
            }
        }
    };
    let z_eff: Vec<f64> = znum.iter().zip(zc.iter()).map(|(z, c)| z - c).collect();
    let (jas, ..) = parse_jastrow_full(jastrow_data)?;
    // J3: cart-lower for the raw-AO kinetic-stress kernels (`j3_grad_wrt_r`/`grad_j3_wrt_r`).
    let (j3o, j3m, nuc_j3) = parse_j3_cart_force(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j1 = if jas.is_some() && !jastrow_data.getattr("jastrow_one_body_data")?.is_none() {
        Some(data::JastrowOneBodyData::from_py(
            &jastrow_data.getattr("jastrow_one_body_data")?,
        )?)
    } else {
        None
    };
    let j2 = if jas.is_some() && !jastrow_data.getattr("jastrow_two_body_data")?.is_none() {
        Some(data::JastrowTwoBodyData::from_py(
            &jastrow_data.getattr("jastrow_two_body_data")?,
        )?)
    } else {
        None
    };
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    // base stress: orbital (incl DTN cotangents) + J1/J2/J3 kinetic + bare virial
    let base = stress::grad_el_wrt_strain_withj(
        &gem,
        jas.as_ref(),
        j1.as_ref(),
        j2.as_ref(),
        j3o.as_ref(),
        j3v.as_ref(),
        &pos.view(),
        &z_eff,
        &ru,
        &rd,
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
    );
    // DTN kinetic virial: cot = gJ(incl DTN) + gD
    let dtn = dtn::JastrowDtnData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let (gj_up, gj_dn) =
        param_grad_el::jastrow_grads(jas.as_ref(), j3o.as_ref(), j3v.as_ref(), &ru, &rd);
    let (gd_up, gd_dn) = param_grad_el::det_grad_field(&gem, &ru, &rd);
    let n_up = ru.nrows();
    let n_e = n_up + rd.nrows();
    let mut cot = numpy::ndarray::Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            cot[[e, c]] = if e < n_up {
                gj_up[[e, c]] + gd_up[[e, c]]
            } else {
                let l = e - n_up;
                gj_dn[[l, c]] + gd_dn[[l, c]]
            };
        }
    }
    let der_r_dtn = forces_dtn::grad_t_dtn_wrt_r(&dtn, &cot, &ru, &rd);
    let d_r_dtn = forces_dtn::grad_t_dtn_wrt_r_atom(&dtn, &cot, &ru, &rd);
    let extra = stress::virial(&der_r_dtn, &d_r_dtn, &ru, &rd, &pos.view());
    let s = base + &extra;
    Ok(s.into_pyarray(py).into_any().unbind())
}

// ---- Non-local ECP parameter derivative dV_ecp/dc ----

/// Extract molecular ECP non-local arrays from a Python `coulomb_potential_data`.
type EcpArrays = (
    numpy::ndarray::Array2<f64>,
    Vec<i64>,
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
);
fn parse_ecp_arrays(cp: &Bound<'_, PyAny>) -> PyResult<EcpArrays> {
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nuc: Vec<i64> = cp.getattr("_nucleus_index_non_local_part")?.extract()?;
    let angmom: Vec<i64> = cp.getattr("_ang_mom_non_local_part")?.extract()?;
    let exps: Vec<f64> = cp
        .getattr("_exponents_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let coefs: Vec<f64> = cp
        .getattr("_coefficients_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let powers: Vec<f64> = cp
        .getattr("_powers_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    Ok((positions, nuc, angmom, exps, coefs, powers))
}

/// Owned `(cell, n_max)` for the periodic min-image of the ECP-nonlocal mesh
/// (`None` cell => molecular). Mirrors the value path `py_compute_ecp_nonlocal_...`.
fn parse_ecp_cell(cp: &Bound<'_, PyAny>) -> PyResult<(Option<numpy::ndarray::Array2<f64>>, i32)> {
    let sd = cp.getattr("structure_data")?;
    let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
    if !pbc {
        return Ok((None, 0));
    }
    let np = cp.py().import("numpy")?;
    let cell = np
        .call_method1("asarray", (sd.getattr("cell")?,))?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let n_max: i32 = sd.getattr("_n_max")?.extract()?;
    Ok((Some(cell), n_max))
}

#[pyfunction]
#[pyo3(name = "grad_Eee_wrt_r")]
#[allow(clippy::too_many_arguments)]
fn py_grad_eee_r(
    py: Python<'_>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
) -> PyResult<Py<PyAny>> {
    let g = forces_ewald::grad_eee_wrt_r(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        &cell.as_array(),
        n_max,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_Eei")]
#[allow(clippy::too_many_arguments)]
fn py_grad_eei(
    py: Python<'_>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    z_ion: PyReadonlyArray1<f64>,
    r_ion: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (dr, drr) = forces_ewald::grad_eei(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        z_ion.as_slice()?,
        &r_ion.as_array(),
        &cell.as_array(),
        n_max,
    );
    Ok((
        dr.into_pyarray(py).into_any().unbind(),
        drr.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_Eii_wrt_R")]
#[allow(clippy::too_many_arguments)]
fn py_grad_eii_r(
    py: Python<'_>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    z_ion: PyReadonlyArray1<f64>,
    r_ion: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let g = forces_ewald::grad_eii_wrt_r(
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        z_ion.as_slice()?,
        &r_ion.as_array(),
        &cell.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_Eee_wrt_strain")]
#[allow(clippy::too_many_arguments)]
fn py_grad_eee_strain(
    py: Python<'_>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
) -> PyResult<Py<PyAny>> {
    let g = forces_ewald::grad_eee_strain(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        &cell.as_array(),
        n_max,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_Eei_wrt_strain")]
#[allow(clippy::too_many_arguments)]
fn py_grad_eei_strain(
    py: Python<'_>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    z_ion: PyReadonlyArray1<f64>,
    r_ion: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
) -> PyResult<Py<PyAny>> {
    let g = forces_ewald::grad_eei_strain(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        z_ion.as_slice()?,
        &r_ion.as_array(),
        &cell.as_array(),
        n_max,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_Eii_wrt_strain")]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_grad_eii_strain(
    py: Python<'_>,
    eta: f64,
    real_image_list: PyReadonlyArray2<f64>,
    g_vectors: PyReadonlyArray2<f64>,
    g_factors: PyReadonlyArray1<f64>,
    z_ion: PyReadonlyArray1<f64>,
    r_ion: PyReadonlyArray2<f64>,
    cell: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let g = forces_ewald::grad_eii_strain(
        eta,
        &real_image_list.as_array(),
        &g_vectors.as_array(),
        g_factors.as_slice()?,
        z_ion.as_slice()?,
        &r_ion.as_array(),
        &cell.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_T_dtn_pbc_wrt_strain", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_t_dtn_pbc_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    let (jas, j3o, j3m) = parse_jastrow_real_value(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    // jastrow_grads already includes the DTN drift (compute_grads_and_laplacian_jastrow_part
    // dispatches on jdata.dtn), so cot = gJ_total + gD is the full drift -- do NOT re-add DTN.
    let (gj_up, gj_dn) =
        param_grad_el::jastrow_grads(jas.as_ref(), j3o.as_ref(), j3v.as_ref(), &ru, &rd);
    let (gd_up, gd_dn) = param_grad_el::det_grad_field(&gem, &ru, &rd);
    let dtn = dtn::JastrowDtnData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let n_up = ru.nrows();
    let n_e = n_up + rd.nrows();
    let mut cot = numpy::ndarray::Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            cot[[e, c]] = if e < n_up {
                gj_up[[e, c]] + gd_up[[e, c]]
            } else {
                let l = e - n_up;
                gj_dn[[l, c]] + gd_dn[[l, c]]
            };
        }
    }
    let mut out = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    stress_pbc::dtn_kstress(&dtn, &cot, &ru, &rd, &mut out);
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_strain", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, jastrow_data=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_grad_v_ecp_nonlocal_strain(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    jastrow_data: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let leaves_up = pgto_stress_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
    let leaves_dn = pgto_stress_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    // optional DTN + J1 + J2 in the wf ratio (u+chi + e-N + e-e)
    let (dtn, j1, j2) = parse_ecp_ratio_jastrow(jastrow_data)?;
    let g = stress_pbc::grad_v_ecp_nonlocal_strain(
        &gem,
        &leaves_up,
        &leaves_dn,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
        dtn.as_ref(),
        j1.as_ref(),
        j2.as_ref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// Parse the DTN + J1 + J2 pieces of the wf ratio used by the PBC ECP non-local stress. A J3 in
/// `jastrow_data` is not read: this ratio carries DTN, J1 and J2 only.
#[allow(clippy::type_complexity)]
fn parse_ecp_ratio_jastrow(
    jastrow_data: Option<&Bound<'_, PyAny>>,
) -> PyResult<(
    Option<dtn::JastrowDtnData>,
    Option<data::JastrowOneBodyData>,
    Option<data::JastrowTwoBodyData>,
)> {
    match jastrow_data {
        Some(jd) if !jd.is_none() => {
            let dobj = jd.getattr("jastrow_dtn_data")?;
            let dtn = if dobj.is_none() {
                None
            } else {
                Some(dtn::JastrowDtnData::from_py(&dobj)?)
            };
            let j1o = jd.getattr("jastrow_one_body_data")?;
            let j1 = if j1o.is_none() {
                None
            } else {
                Some(data::JastrowOneBodyData::from_py(&j1o)?)
            };
            let j2o = jd.getattr("jastrow_two_body_data")?;
            let j2 = if j2o.is_none() {
                None
            } else {
                Some(data::JastrowTwoBodyData::from_py(&j2o)?)
            };
            Ok((dtn, j1, j2))
        }
        _ => Ok((None, None, None)),
    }
}

/// ECP non-local stress dV_nl/deps for a B-spline geminal via pulay=0 (empty GTO leaves).
/// JUSTIFIED BY COMPOSITION (a direct end-to-end FD is blocked, see below):
///  1. The bspline AO VALUE is affine-strain-invariant (frac=r*inv_cell invariant under the
///     simultaneous cell+electron strain) => dln|det|/deps == 0 for EVERY electron config (base AND
///     ECP-mesh), FD-verified 7e-7. So the det Pulay stress pbc_pulay_det -> 0 (empty leaves) is
///     EXACT, not an approximation.
///  2. Everything else in `grad_v_ecp_nonlocal_strain` (the det ratio, the drift via
///     det_grad_field, the explicit dA/deps, the mesh-motion correction, the DTN u+chi terms) is
///     BASIS-AGNOSTIC and verified for GTO (lithium_k025 full ECP e_L stress = jax.grad, 1e-5,
///     using this same function); the bspline geminal eval (compute_geminal_all_elements,
///     det_grad_field) is verified via ln|det|/kinetic.
/// A direct FD on the only bspline fixture (a 96-electron periodic cell) is intractable: jqmc's
/// `compute_ecp_non_local_parts` OOMs at NN=natom and has a >400 s XLA compile at small NN, and
/// jqmc's `_apply_strain` freezes the Ewald tables (so a full-e_L FD is invalid for the Coulomb
/// part) -- reference-tooling limits, NOT a Rust issue.
#[pyfunction]
#[pyo3(name = "grad_V_ecp_nonlocal_wrt_strain_bspline", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, jastrow_data=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_grad_v_ecp_nonlocal_strain_bspline(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    jastrow_data: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let cp = coulomb_potential_data;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
    let sd = cp.getattr("structure_data")?;
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let (dtn, j1, j2) = parse_ecp_ratio_jastrow(jastrow_data)?;
    let empty: Vec<stress_pbc::StressLeaf> = Vec::new();
    let g = stress_pbc::grad_v_ecp_nonlocal_strain(
        &gem,
        &empty,
        &empty,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
        dtn.as_ref(),
        j1.as_ref(),
        j2.as_ref(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_V_ecp_local_wrt_strain", signature = (coulomb_potential_data, r_up_carts, r_dn_carts))]
fn py_grad_vlocal_strain(
    py: Python<'_>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let cell = sd
        .getattr("cell")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let nmax: i32 = sd.getattr("_n_max")?.extract()?;
    let g = forces_ecp::grad_vlocal_strain(
        &pos.view(),
        &geti("_nucleus_index_local_part")?,
        &getf("_exponents_local_part")?,
        &getf("_coefficients_local_part")?,
        &getf("_powers_local_part")?,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &cell.view(),
        nmax,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_local_force_wrt_R")]
fn py_grad_ecp_local_force(
    py: Python<'_>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc = geti("nucleus_index")?;
    let ang = geti("ang_moms")?;
    let exps = getf("exponents")?;
    let coefs = getf("coefficients")?;
    let powers = getf("powers")?;
    let max_ang = geti("max_ang_mom_plus_1")?;
    let g = forces_ecp::grad_vlocal_force(
        &pos.view(),
        &nuc,
        &ang,
        &exps,
        &coefs,
        &powers,
        &max_ang,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_force_wrt_R")]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_grad_ecp_force(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let g = forces_ecp::grad_ecp_force(
        &gem,
        &jas,
        j3,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_lambda")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_lambda(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    if gem.lambda.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_ln_det_lambda is undefined for occupied-only SD",
        ));
    }
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    let g = param_grad_ecp::grad_ecp_lambda(
        &gem,
        &jas,
        j3,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// AGPn `dV_ecp(non-local)/d(lambda_diag, lambda_unpaired)` (real path).
#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_lambda_diag", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_lambda_diag(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    if gem.lambda.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_ln_det_lambda is undefined for occupied-only SD",
        ));
    }
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    let (d, u) = param_grad_ecp::grad_ecp_lambda_diag(
        &gem,
        &jas,
        j3,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    );
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_mo_coeff")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_mo(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    if gem.orb_up.mo_coefficients.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_ln_det_mo: up spin has no mo_coefficients",
        ));
    }
    if gem.orb_dn.mo_coefficients.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_ln_det_mo: dn spin has no mo_coefficients",
        ));
    }
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    let (gu, gd) = param_grad_ecp::grad_ecp_mo(
        &gem,
        &jas,
        j3,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_dtn_kinetic_params")]
fn py_grad_el_dtn_kinetic(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let d = param_grad_dtn::DtnOkData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let (a, b, g) = param_grad_dtn::compute_dtn_kinetic_grads(
        &d,
        &gem,
        &jas,
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    let opt = |x: Option<numpy::ndarray::Array2<f64>>| -> Py<PyAny> {
        match x {
            Some(arr) => arr.into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    };
    Ok((opt(a), opt(b), opt(g)))
}

#[pyfunction]
#[pyo3(name = "grad_ln_Psi_dtn_params")]
fn py_grad_dtn_params(
    py: Python<'_>,
    jastrow_dtn_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let d = param_grad_dtn::DtnOkData::from_py(jastrow_dtn_data)?;
    let (a, b, g) =
        param_grad_dtn::compute_dtn_param_grads(&d, &r_up_carts.as_array(), &r_dn_carts.as_array());
    let opt = |x: Option<numpy::ndarray::Array2<f64>>| -> Py<PyAny> {
        match x {
            Some(arr) => arr.into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    };
    Ok((opt(a), opt(b), opt(g)))
}

// ---- Complex-twist (k != Gamma) Jastrow-param de_L/dc = Re(dT/dc). At a twist
//      the Jastrow is real but the determinant drift is complex;
//      the reduction re-projects to Re. For an ECP twist add the complex-mesh
//      dV_ecp_nl/dc (the *_complex ECP wrappers below). ----

/// Parse the COMPLEX-Bloch J3 orbital + j_matrix from a jastrow_data (None if no J3).
fn parse_j3_complex(
    jastrow_data: &Bound<'_, PyAny>,
) -> PyResult<(
    Option<complex_det::ComplexOrb>,
    Option<numpy::ndarray::Array2<f64>>,
)> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Ok((None, None));
    }
    let od = j3d.getattr("orb_data")?;
    let np = jastrow_data.py().import("numpy")?;
    let jm: numpy::PyReadonlyArray2<f64> = np
        .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
        .call_method1("astype", (np.getattr("float64")?,))?
        .extract()?;
    let orb = complex_det::ComplexOrb::from_py(&od)?;
    orb.check_j3_matrix_shape(jm.as_array().nrows(), jm.as_array().ncols())?;
    Ok((Some(orb), Some(jm.as_array().to_owned())))
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_1b_param_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, source_mode=false))]
fn py_grad_el_j1_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let j1 = data::JastrowOneBodyData::from_py(&jastrow_data.getattr("jastrow_one_body_data")?)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let value = param_grad_twist::grad_t_j1_twist(
        &gem,
        &jas,
        &j1,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    if source_mode {
        Ok(pyo3::types::PyComplex::from_doubles(py, value.re, value.im)
            .into_any()
            .unbind())
    } else {
        Ok(pyo3::types::PyFloat::new(py, value.re).into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_2b_param_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, source_mode=false))]
fn py_grad_el_j2_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let j2 = data::JastrowTwoBodyData::from_py(&jastrow_data.getattr("jastrow_two_body_data")?)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let value = param_grad_twist::grad_t_j2_twist(
        &gem,
        &jas,
        &j2,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    if source_mode {
        Ok(pyo3::types::PyComplex::from_doubles(py, value.re, value.im)
            .into_any()
            .unbind())
    } else {
        Ok(pyo3::types::PyFloat::new(py, value.re).into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(name = "grad_eL_wrt_jastrow_3b_jmatrix_complex")]
fn py_grad_el_j3_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let j3c = j3c.expect("grad_eL_wrt_jastrow_3b_jmatrix_complex: no jastrow_three_body_data");
    let j3m = j3m.unwrap();
    let g = param_grad_twist::grad_t_j3_twist(
        &gem,
        &jas,
        &j3c,
        &j3m.view(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

/// `O_k = dJ3/dj_matrix` on a complex-Bloch (twist) J3 orbital leaf. The complex twin of
/// `grad_ln_Psi_jastrow_3b_jmatrix`: identical arguments, real `(orb, orb+1)` output
/// (J3 is a real correlation factor built on `Re(orbital)`).
#[pyfunction]
#[pyo3(name = "grad_ln_Psi_jastrow_3b_jmatrix_complex")]
fn py_grad_j3_jmatrix_twist(
    py: Python<'_>,
    jastrow_three_body_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let orb = complex_det::ComplexOrb::from_py(&jastrow_three_body_data.getattr("orb_data")?)?;
    let g = param_grad_twist::grad_j3_jmatrix_complex(
        &orb,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_dtn_kinetic_params_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, source_mode=false))]
fn py_grad_el_dtn_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let d = param_grad_dtn::DtnOkData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (a, b, g) = param_grad_twist::grad_t_dtn_twist(
        &d,
        &gem,
        &jas,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    let opt = |x: Option<numpy::ndarray::Array2<numpy::Complex64>>| -> Py<PyAny> {
        match x {
            Some(arr) if source_mode => arr.into_pyarray(py).into_any().unbind(),
            Some(arr) => arr.mapv(|z| z.re).into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    };
    Ok((opt(a), opt(b), opt(g)))
}

/// AGPn twist `dE_L/d(lambda_diag, lambda_unpaired)` WITH Jastrow -- the complex twin
/// of `grad_eL_wrt_lambda_diag`. `jastrow_data = None` (jnone) reduces to the
/// Jastrow-free `grad_T_lambda_diag_complex`.
#[pyfunction]
#[pyo3(name = "grad_eL_wrt_lambda_diag_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_grad_el_lambda_diag_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let (jas, j3c, j3m) = if jastrow_data.is_none() {
        (None, None, None)
    } else {
        let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
        (
            Some(data::JastrowData::from_py_complex_aware(jastrow_data)?),
            j3c,
            j3m,
        )
    };
    let (d, u) = param_grad_twist::grad_t_lambda_diag_twist(
        &gem,
        jas.as_ref(),
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

/// Dense AGP twist `dV_ecp(non-local)/d(lambda_matrix)`, retaining the holomorphic
/// complex mesh weight and lambda derivative until the optimizer projection.
#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_lambda_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_lambda_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let np = py.import("numpy")?;
    let lam_c: numpy::PyReadonlyArray2<numpy::Complex64> = np
        .call_method1("asarray", (geminal_data.getattr("lambda_matrix")?,))?
        .call_method1("astype", (np.getattr("complex128")?,))?
        .extract()?;
    let out = param_grad_twist::grad_ecp_lambda_twist(
        &gem,
        &lam_c.as_array(),
        &jas,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// AGPn twist `dV_ecp(non-local)/d(lambda_diag, lambda_unpaired)` -- the complex twin
/// of `grad_eL_ecp_wrt_lambda_diag`.
#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_lambda_diag_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points))]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_lambda_diag_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let (d, u) = param_grad_twist::grad_ecp_lambda_diag_twist(
        &gem,
        &jas,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    Ok((
        d.into_pyarray(py).into_any().unbind(),
        u.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_1b_param_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, source_mode=false))]
#[allow(clippy::too_many_arguments)]
fn py_grad_el_ecp_j1_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let j1 = data::JastrowOneBodyData::from_py(&jastrow_data.getattr("jastrow_one_body_data")?)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let value = param_grad_twist::grad_ecp_j1_twist(
        &gem,
        &jas,
        &j1,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    if source_mode {
        Ok(pyo3::types::PyComplex::from_doubles(py, value.re, value.im)
            .into_any()
            .unbind())
    } else {
        Ok(pyo3::types::PyFloat::new(py, value.re).into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_2b_param_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, source_mode=false))]
#[allow(clippy::too_many_arguments)]
fn py_grad_el_ecp_j2_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let j2 = data::JastrowTwoBodyData::from_py(&jastrow_data.getattr("jastrow_two_body_data")?)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let value = param_grad_twist::grad_ecp_j2_twist(
        &gem,
        &jas,
        &j2,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    if source_mode {
        Ok(pyo3::types::PyComplex::from_doubles(py, value.re, value.im)
            .into_any()
            .unbind())
    } else {
        Ok(pyo3::types::PyFloat::new(py, value.re).into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex")]
#[allow(clippy::too_many_arguments)]
fn py_grad_el_ecp_j3_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let j3c = j3c.expect("grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex: no jastrow_three_body_data");
    let j3m = j3m.unwrap();
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let g = param_grad_twist::grad_ecp_j3_twist(
        &gem,
        &jas,
        &j3c,
        &j3m.view(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_dtn_params_complex")]
#[pyo3(signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, source_mode=false))]
#[allow(clippy::too_many_arguments)]
fn py_grad_el_ecp_dtn_twist(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    source_mode: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_without_three_body(jastrow_data)?;
    let d = param_grad_dtn::DtnOkData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let (j3c, j3m) = parse_j3_complex(jastrow_data)?;
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let (a, b, g) = param_grad_twist::grad_ecp_dtn_twist(
        &d,
        &gem,
        &jas,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_view.as_ref(),
        n_max,
    );
    let opt = |x: Option<numpy::ndarray::Array2<numpy::Complex64>>| -> Py<PyAny> {
        match x {
            Some(arr) if source_mode => arr.into_pyarray(py).into_any().unbind(),
            Some(arr) => arr.mapv(|z| z.re).into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    };
    Ok((opt(a), opt(b), opt(g)))
}

// ---- ECP non-local dV_ecp/dc for the Jastrow / DTN / basis param blocks ----
// (completes de_L/dc = dT/dc + dV_ecp_nl/dc for ECP systems; the Det piece is
// grad_eL_ecp_wrt_{lambda,mo_coeff} above.)

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_1b_param")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_j1(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let j1 = data::JastrowOneBodyData::from_py(&jastrow_data.getattr("jastrow_one_body_data")?)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    Ok(param_grad_ecp::grad_ecp_j1(
        &gem,
        &jas,
        j3,
        &j1,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_2b_param")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_j2(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let j2 = data::JastrowTwoBodyData::from_py(&jastrow_data.getattr("jastrow_two_body_data")?)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    Ok(param_grad_ecp::grad_ecp_j2(
        &gem,
        &jas,
        j3,
        &j2,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    ))
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_jastrow_3b_jmatrix")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_j3(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (j3o, j3m) = parse_explicit_real_j3_for_jmatrix_deriv(jastrow_data)?;
    let (j3o, j3m) = match (j3o, j3m) {
        (Some(orb), Some(matrix)) => (orb, matrix),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "grad_eL_ecp_wrt_jastrow_3b_jmatrix: no jastrow_three_body_data",
            ));
        }
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    let g = param_grad_ecp::grad_ecp_j3(
        &gem,
        &jas,
        &j3o,
        &j3m.view(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    );
    Ok(g.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_dtn_params")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_dtn(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let d = param_grad_dtn::DtnOkData::from_py(&jastrow_data.getattr("jastrow_dtn_data")?)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    let (a, b, g) = param_grad_ecp::grad_ecp_dtn(
        &d,
        &gem,
        &jas,
        j3,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    );
    let opt = |x: Option<numpy::ndarray::Array2<f64>>| -> Py<PyAny> {
        match x {
            Some(arr) => arr.into_pyarray(py).into_any().unbind(),
            None => py.None(),
        }
    };
    Ok((opt(a), opt(b), opt(g)))
}

#[pyfunction]
#[pyo3(name = "grad_eL_ecp_wrt_radial")]
#[allow(clippy::too_many_arguments)]
fn py_grad_ecp_radial(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    let (a2s_up, nsh_up) = radial_map(&geminal_data.getattr("orb_data_up_spin")?)?;
    let (a2s_dn, nsh_dn) = radial_map(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let (j3o, j3m) = parse_explicit_real_j3(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let j3 = match (&j3o, &j3v) {
        (Some(o), Some(v)) => Some((o, v)),
        _ => None,
    };
    let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(coulomb_potential_data)?;
    let (cell_owned, n_max) = parse_ecp_cell(coulomb_potential_data)?;
    let cell_view = cell_owned.as_ref().map(|c| c.view());
    let cell_nmax = cell_view.as_ref().map(|v| (v, n_max));
    let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_ecp::grad_ecp_radial(
        &gem,
        &jas,
        j3,
        &a2s_up,
        nsh_up,
        &a2s_dn,
        nsh_dn,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &pos.view(),
        &nuc,
        &angmom,
        &exps,
        &coefs,
        &powers,
        weights.as_slice()?,
        &grid_points.as_array(),
        cell_nmax,
    );
    let mk = |v: Vec<f64>| {
        numpy::ndarray::Array1::from_vec(v)
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    let dict = pyo3::types::PyDict::new(py);
    let up = pyo3::types::PyTuple::new(py, [mk(gz_up), mk(gc_up)])?;
    let dn = pyo3::types::PyTuple::new(py, [mk(gz_dn), mk(gc_dn)])?;
    dict.set_item("orb_data_up_spin", up)?;
    dict.set_item("orb_data_dn_spin", dn)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "compute_grads_and_laplacian_jastrow_three_body")]
fn py_compute_grads_and_laplacian_jastrow_three_body(
    py: Python<'_>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj4> {
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    if j3d.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "compute_grads_and_laplacian_jastrow_three_body: jastrow_three_body_data is None",
        ));
    }
    let ru = r_up_carts.as_array();
    let rd = r_dn_carts.as_array();
    // An INACTIVE J3 (all-zero `j_matrix`) is identically 0: its gradients and Laplacians are
    // exactly 0 and its orbital is never parsed; an active J3 whose orbital the real parser
    // cannot represent is refused (issue #170).
    let (gu, gd, lu, ld) = match parse_real_j3_parts(&j3d, false)? {
        Some((orb, jm)) => jastrow_factor::compute_grads_and_laplacian_jastrow_three_body(
            &orb,
            &jm.view(),
            &ru,
            &rd,
        ),
        None => (
            numpy::ndarray::Array2::<f64>::zeros((ru.nrows(), 3)),
            numpy::ndarray::Array2::<f64>::zeros((rd.nrows(), 3)),
            numpy::ndarray::Array1::<f64>::zeros(ru.nrows()),
            numpy::ndarray::Array1::<f64>::zeros(rd.nrows()),
        ),
    };
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
        lu.into_pyarray(py).into_any().unbind(),
        ld.into_pyarray(py).into_any().unbind(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_grads_and_laplacian_jastrow_part")]
fn py_compute_grads_and_laplacian_jastrow_part(
    py: Python<'_>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj4> {
    // J1 + J2 + DTN only (the J3 grad / Laplacian is computed separately): a complex-only J3
    // is not read here, so it is not rejected either.
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (gu, gd, lu, ld) = jastrow_factor::compute_grads_and_laplacian_jastrow_part(
        &jdata,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
    );
    Ok((
        gu.into_pyarray(py).into_any().unbind(),
        gd.into_pyarray(py).into_any().unbind(),
        lu.into_pyarray(py).into_any().unbind(),
        ld.into_pyarray(py).into_any().unbind(),
    ))
}

// --- Det ratio streaming state + Sherman-Morrison validation wrappers. --

fn _state_tuple(
    py: Python<'_>,
    s: &determinant::DetRatioState,
) -> (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>) {
    (
        s.ao_up.clone().into_pyarray(py).into_any().unbind(),
        s.ao_dn.clone().into_pyarray(py).into_any().unbind(),
        s.paired_dn.clone().into_pyarray(py).into_any().unbind(),
        s.paired_up_lambda
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind(),
    )
}

#[pyfunction]
fn _det_init_state(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<PyObj4> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    let s =
        determinant::init_det_ratio_state(&data, &r_up_carts.as_array(), &r_dn_carts.as_array());
    Ok(_state_tuple(py, &s))
}

#[pyfunction]
fn _det_advance_state(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    moved_spin_is_up: bool,
    moved_index: usize,
    new_pos: PyReadonlyArray1<f64>,
) -> PyResult<PyObj4> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    let mut s =
        determinant::init_det_ratio_state(&data, &r_up_carts.as_array(), &r_dn_carts.as_array());
    let p = new_pos.as_slice()?;
    determinant::advance_det_ratio_state(
        &data,
        &mut s,
        moved_spin_is_up,
        moved_index,
        [p[0], p[1], p[2]],
    );
    Ok(_state_tuple(py, &s))
}

#[pyfunction]
fn _det_v_up_move(
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    moved_index: usize,
    new_pos: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    let s =
        determinant::init_det_ratio_state(&data, &r_up_carts.as_array(), &r_dn_carts.as_array());
    let p = new_pos.as_slice()?;
    Ok(determinant::compute_v_up_move(
        &data,
        &s,
        moved_index,
        [p[0], p[1], p[2]],
    ))
}

#[pyfunction]
fn _det_u_dn_move(
    geminal_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    moved_index: usize,
    new_pos: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let data = determinant::GeminalData::from_py(geminal_data)?;
    let s =
        determinant::init_det_ratio_state(&data, &r_up_carts.as_array(), &r_dn_carts.as_array());
    let p = new_pos.as_slice()?;
    Ok(determinant::compute_u_dn_move(
        &data,
        &s,
        moved_index,
        [p[0], p[1], p[2]],
    ))
}

#[pyfunction]
fn _det_sm_update(
    py: Python<'_>,
    geminal: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    moved_spin_is_up: bool,
    moved_index: usize,
) -> (f64, Py<PyAny>, Py<PyAny>) {
    let (ratio, inv_new, gem_new) = determinant::sm_update(
        &geminal_inv.as_array(),
        &geminal.as_array(),
        u.as_slice().unwrap(),
        v.as_slice().unwrap(),
        moved_spin_is_up,
        moved_index,
    );
    (
        ratio,
        inv_new.into_pyarray(py).into_any().unbind(),
        gem_new.into_pyarray(py).into_any().unbind(),
    )
}

// --- MCMC move driver ---------------------------------------------------

/// Draw one move's RNG choices (electron selection, axis, Metropolis b) in
/// jqmc's exact threefry split order. Returns (is_up, selected_index, axis, b,
/// final_key0, final_key1).
#[pyfunction]
fn _move_draws(k0: u32, k1: u32, n_up: usize, n_dn: usize) -> (bool, usize, usize, f64, u32, u32) {
    let d = mcmc::move_draws([k0, k1], n_up, n_dn);
    (
        d.is_up,
        d.selected_index,
        d.axis,
        d.metropolis_b,
        d.final_key[0],
        d.final_key[1],
    )
}

/// One native single-electron move (molecular or PBC via optional `cell`/`n_max`).
/// Returns a dict-like tuple.
#[pyfunction]
#[pyo3(signature = (geminal_data, jastrow_data, positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, k0, k1, injected_normal, cell=None, n_max=0))]
#[allow(clippy::too_many_arguments)]
fn _single_move(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    positions: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
    r_up: PyReadonlyArray2<f64>,
    r_dn: PyReadonlyArray2<f64>,
    geminal: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
    dt: f64,
    epsilon_as: f64,
    k0: u32,
    k1: u32,
    injected_normal: f64,
    cell: Option<PyReadonlyArray2<f64>>,
    n_max: i32,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;
    // Optional J3: parse orb_data + j_matrix from jastrow_three_body_data, mirroring
    // ``_mcmc_sweep``'s parse so ``single_move`` sees the SAME J3 channel the sweep does
    // so every active Jastrow channel participates in the acceptance ratio.
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    let j3_jm_view = j3_jm.as_ref().map(|m| m.view());
    let cell_arr = cell.as_ref().map(|c| c.as_array());
    let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
    let res = mcmc::single_move(
        &gem,
        &jas,
        j3_orb.as_ref(),
        j3_jm_view.as_ref(),
        &positions.as_array(),
        charges.as_slice()?,
        &r_up.as_array(),
        &r_dn.as_array(),
        &geminal.as_array(),
        &geminal_inv.as_array(),
        dt,
        epsilon_as,
        [k0, k1],
        injected_normal,
        cell_nmax,
    );
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("accepted", res.accepted)?;
    dict.set_item("is_up", res.is_up)?;
    dict.set_item("selected_index", res.selected_index)?;
    dict.set_item("new_pos", res.new_pos.to_vec())?;
    dict.set_item("r_up", res.r_up.into_pyarray(py))?;
    dict.set_item("r_dn", res.r_dn.into_pyarray(py))?;
    dict.set_item("geminal", res.geminal.into_pyarray(py))?;
    dict.set_item("geminal_inv", res.geminal_inv.into_pyarray(py))?;
    dict.set_item("det_t_ratio", res.det_t_ratio)?;
    dict.set_item("r_ratio", res.r_ratio)?;
    dict.set_item("t_ratio", res.t_ratio)?;
    dict.set_item("acceptance_ratio", res.acceptance_ratio)?;
    dict.set_item("metropolis_b", res.metropolis_b)?;
    dict.set_item("final_key", res.final_key.to_vec())?;
    Ok(dict.into_any().unbind())
}

/// LRDMC discretized-kinetic hopping amplitudes (`compute_discretized_kinetic_energy`).
/// Verifies vs jqmc `compute_discretized_kinetic_energy`. `rt` = the (rotated) 3x3 axis matrix.
#[pyfunction]
#[pyo3(name = "_gfmc_discretized_kinetic_elements", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, alat, rt))]
fn py_gfmc_discretized_kinetic_elements(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
) -> PyResult<Vec<f64>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3_orb, j3_jm) = if j3d.is_none() {
        (None, None)
    } else {
        let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
        let jm_arr = jm.as_array().to_owned();
        if jm_arr.iter().all(|&x| x == 0.0) {
            (None, None)
        } else {
            (
                Some(determinant::OrbData::from_py(&j3d.getattr("orb_data")?)?),
                Some(jm_arr),
            )
        }
    };
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    Ok(gfmc::compute_discretized_kinetic_elements(
        &gem,
        &jdata,
        j3_ref,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
    ))
}

/// LRDMC all-electron diagonal local energy e_L + off-diagonal rate b_L. Verifies vs jqmc.
#[pyfunction]
#[pyo3(name = "_gfmc_lrdmc_ae", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, fast=false))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_gfmc_lrdmc_ae(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    fast: bool,
) -> PyResult<(f64, f64)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = gfmc_real_j3_parse(&jastrow_data.getattr("jastrow_three_body_data")?)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let pbc_owned = crate::ewald::EwaldData::from_py(cp)?;
    let pbc = pbc_owned.as_ref().map(|a| a.as_ref());
    let (e_l, _e_l_meas, b_l, _) = gfmc::gfmc_lrdmc_ae(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        pbc.as_ref(),
        fast,
    );
    Ok((e_l, b_l))
}

/// GFMC HF-stress `de_L^t/deps (3x3)` for the molecular AE LRDMC diagonal e_L
/// (jnone / j12 / j123 / DTN). stress_PP = the existing VMC Pulay
/// stress `grad_lnPsi_wrt_strain`, reused as-is.
#[pyfunction]
#[pyo3(name = "_gfmc_stress_ae", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_stress_ae(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    // J3: cart-lower for the raw-AO stress kernels; nuc is per-cart-AO. Skip a
    // vacuous (all-zero j_matrix) J3. J3 factor is basis-invariant.
    let (j3_orb, j3_jm, nuc_j3) = {
        let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
        if j3d.is_none() {
            (None, None, None)
        } else {
            let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
            if jm.as_array().iter().all(|&x| x == 0.0) {
                (None, None, None)
            } else {
                let (o, m, n) =
                    lower_j3_leaf_cart(&j3d.getattr("orb_data")?, &j3d.getattr("j_matrix")?)?;
                (Some(o), Some(m), Some(n))
            }
        }
    };
    let j3v = j3_jm.as_ref().map(|m| m.view());
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let s = gfmc_forces::gfmc_stress_ae(
        &gem,
        &jdata,
        j3_orb.as_ref(),
        j3v.as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// GFMC HF-stress `de_L^t/deps (3x3)` for the molecular ECP jnone LRDMC diagonal e_L (tmove/dltmove).
#[pyfunction]
#[pyo3(name = "_gfmc_stress_ecp_jnone", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_stress_ecp_jnone(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let s = gfmc_forces::gfmc_stress_ecp_jnone(
        &gem,
        &pos.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &nuc_up,
        &nuc_dn,
        &nuc_l,
        &e_l_p,
        &c_l_p,
        &p_l_p,
        &nuc_n,
        &am_n,
        &e_n,
        &c_n,
        &p_n,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// GFMC HF-stress `de_L^t/deps` for molecular ECP with J1/J2/J3/DTN.
#[pyfunction]
#[pyo3(name = "_gfmc_stress_ecp_withj", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_stress_ecp_withj(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
) -> PyResult<Py<PyAny>> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let (j3_orb, j3_jm, nuc_j3) = if j3d.is_none() {
        (None, None, None)
    } else {
        let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
        if jm.as_array().iter().all(|&x| x == 0.0) {
            (None, None, None)
        } else {
            let (o, m, n) =
                lower_j3_leaf_cart(&j3d.getattr("orb_data")?, &j3d.getattr("j_matrix")?)?;
            (Some(o), Some(m), Some(n))
        }
    };
    let j3v = j3_jm.as_ref().map(|m| m.view());
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    if sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_stress_ecp_withj: molecular only",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let np = cp.py().import("numpy")?;
    let geti = |name: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(name)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |name: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(name)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let sigma = gfmc_forces::gfmc_stress_ecp_withj(
        &gem,
        &jdata,
        j3_orb.as_ref(),
        j3v.as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
        &nuc_l,
        &e_l_p,
        &c_l_p,
        &p_l_p,
        &nuc_n,
        &am_n,
        &e_n,
        &c_n,
        &p_n,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
    );
    Ok(sigma.into_pyarray(py).into_any().unbind())
}

/// GFMC HF-stress `de_L^t/deps (3x3)` for the PBC (Ewald, Gamma) ECP jnone LRDMC diagonal e_L.
#[pyfunction]
#[pyo3(name = "_gfmc_stress_ecp_pbc_jnone", signature = (geminal_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, debug=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_stress_ecp_pbc_jnone(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    debug: bool,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?; // native (composite-aware)
    let up_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
    let dn_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    if !sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_stress_ecp_pbc_jnone: PBC only",
        ));
    }
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let ew_arr = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
    })?;
    let ew = ew_arr.as_ref();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let pos_v = pos.view();
    let r_up_v = r_up_carts.as_array();
    let r_dn_v = r_dn_carts.as_array();
    let grid_v = grid_points.as_array();
    let weights_v = weights.as_slice()?;
    let s = if debug {
        gfmc_forces::gfmc_stress_ecp_pbc_jnone_debug(
            &gem,
            &up_leaves,
            &dn_leaves,
            &pos_v,
            &eff_charges,
            &r_up_v,
            &r_dn_v,
            alat,
            &rtm,
            &ew,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights_v,
            &grid_v,
            nn,
        )
    } else {
        gfmc_forces::gfmc_stress_ecp_pbc_jnone(
            &gem,
            &up_leaves,
            &dn_leaves,
            &pos_v,
            &eff_charges,
            &r_up_v,
            &r_dn_v,
            alat,
            &rtm,
            &ew,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights_v,
            &grid_v,
            nn,
        )
    };
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// GFMC HF-stress `de_L^t/deps (3x3)` for the PBC (Ewald, Gamma) ECP **withJ (J1/J2)**
/// LRDMC diagonal e_L. J3/DTN guarded (no periodic J3 value-strain).
#[pyfunction]
#[pyo3(name = "_gfmc_stress_ecp_pbc_withj", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false, debug=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_stress_ecp_pbc_withj(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    debug: bool,
) -> PyResult<Py<PyAny>> {
    // Native geminal (composite pgto(+)pw aware; the cart-convert path raises on
    // composite). The stress leaves + PW blocks handle both representations.
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let j3s_owned = j3_stress_ctx_owned(&j3d)?;
    let j3_batch = if j3s_owned.is_none() {
        j3_batch_ctx_owned(&j3d)?
    } else {
        None
    };
    let jm_v = j3_batch.as_ref().map(|(_, m)| m.view());
    let batch = j3_batch.as_ref().map(|(o, _)| (o, jm_v.as_ref().unwrap()));
    let j3s = j3s_owned
        .as_ref()
        .map(
            |(orb, cart, t, jm, home, ker, pw, gto_first)| gfmc_forces::J3StressPbc {
                orb,
                cart_j3: cart,
                t_mol: t.view(),
                jm: jm.view(),
                home,
                kernel: ker,
                pw: pw.as_ref().map(|e| e.view()),
                gto_first: *gto_first,
            },
        );
    let j1o = jastrow_data.getattr("jastrow_one_body_data")?;
    let j2o = jastrow_data.getattr("jastrow_two_body_data")?;
    let j1 = if j1o.is_none() {
        None
    } else {
        Some(data::JastrowOneBodyData::from_py(&j1o)?)
    };
    let j2 = if j2o.is_none() {
        None
    } else {
        Some(data::JastrowTwoBodyData::from_py(&j2o)?)
    };
    let up_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
    let dn_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    if !sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_stress_ecp_pbc_withj: PBC only",
        ));
    }
    let pos = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let ew_arr = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
    })?;
    let ew = ew_arr.as_ref();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let pos_v = pos.view();
    let r_up_v = r_up_carts.as_array();
    let r_dn_v = r_dn_carts.as_array();
    let grid_v = grid_points.as_array();
    let weights_v = weights.as_slice()?;
    let s = if debug {
        gfmc_forces::gfmc_stress_ecp_pbc_withj_debug(
            &gem,
            &jdata,
            &up_leaves,
            &dn_leaves,
            &pos_v,
            &eff_charges,
            &r_up_v,
            &r_dn_v,
            alat,
            &rtm,
            &ew,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights_v,
            &grid_v,
            nn,
            dltmove,
            j1.as_ref(),
            j2.as_ref(),
            jdata.dtn.as_ref(),
            j3s.as_ref(),
            batch,
        )
    } else {
        gfmc_forces::gfmc_stress_ecp_pbc_withj(
            &gem,
            &jdata,
            &up_leaves,
            &dn_leaves,
            &pos_v,
            &eff_charges,
            &r_up_v,
            &r_dn_v,
            alat,
            &rtm,
            &ew,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights_v,
            &grid_v,
            nn,
            dltmove,
            j1.as_ref(),
            j2.as_ref(),
            jdata.dtn.as_ref(),
            j3s.as_ref(),
            batch,
        )
    };
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// J3 stress context (owned): `(drift orb, cart_j3, t_mol, jm, home, kernel)`.
/// Returns None when J3 is absent or the j_matrix is all-zero.
#[allow(clippy::type_complexity)]
fn j3_stress_ctx_owned(
    j3d: &Bound<'_, PyAny>,
) -> PyResult<
    Option<(
        determinant::OrbData,
        atomic_orbital::AosGtoData,
        numpy::ndarray::Array2<f64>,
        numpy::ndarray::Array2<f64>,
        Vec<usize>,
        Vec<usize>,
        Option<forces_pbc::J3ExtraOwned>,
        bool,
    )>,
> {
    if j3d.is_none() {
        return Ok(None);
    }
    let np = j3d.py().import("numpy")?;
    let jm_obj = np.call_method1(
        "asarray",
        (j3d.getattr("j_matrix")?, np.getattr("float64")?),
    )?;
    let jm: numpy::PyReadonlyArray2<f64> = jm_obj.extract()?;
    let jm_arr = jm.as_array().to_owned();
    if jm_arr.iter().all(|&x| x == 0.0) {
        return Ok(None);
    }
    let orb_py = j3d.getattr("orb_data")?;
    let orb = determinant::OrbData::from_py(&orb_py)?;
    // A PURE plane-wave / B-spline J3 leaf has no cart-lowerable AO and no atom center: its
    // Pulay VALUE-strain d(lnJ3)/deps is exactly 0 (affine-invariant; verified: the J3
    // eps-contribution to dln|Psi|/deps == 0 for both PW and B-spline), so skip it here.
    // (The e_L kinetic strain, which IS nonzero for these leaves, is a separate unported
    // kernel guarded at its own caller.)
    if j3_leaf_is_batch(&orb) {
        return Ok(None);
    }
    // Lower an MO-wrapped J3 (num_mo != num_ao, e.g. production lithium 1<-26) to its exact
    // AO-equivalent (raw-AO orbital + `M_ao = C^T M_mo C`), exactly as the FORCE ctx
    // (`j3_pbc_ctx_owned`) does -- the stress field kernels (`j3_pbc_value_strain` /
    // `j3_pbc_kstress_w`) index the raw cart/sphe AO basis, so an un-lowered MO J3 read
    // `MOs_data.num_ao` (AttributeError) on the production lithium MCMC stress. Bilinear
    // lowering is valid for a REAL C (real-Gamma); the complex-C twist J3 routes to the
    // complex ctx. No-op for a raw-AO J3.
    let (orb_low, jm_low) = gfmc_forces::lower_j3_to_ao(Some(&orb), Some(&jm_arr.view()))
        .expect("lower_j3_to_ao returns Some when orb+jm present");
    let leaf_py = if orb.mo_coefficients.is_some() {
        orb_py.getattr("aos_data")?
    } else {
        orb_py.clone()
    };
    let (cart_j3, t_mol, home, kernel, pw, gto_first) = if leaf_py.hasattr("children")? {
        let (cart, t, _force_home, stress_home, kernel, pw, first) =
            pgto_j3_pieces_composite(&leaf_py)?;
        (cart, t, stress_home, kernel, pw, first)
    } else {
        let (cart, t, home, kernel) = pgto_j3_stress_pieces(&leaf_py)?;
        (cart, t, home, kernel, None, true)
    };
    Ok(Some((
        orb_low, cart_j3, t_mol, jm_low, home, kernel, pw, gto_first,
    )))
}

/// Canonical MPC (Model Periodic Coulomb) e-e potential (Ha) for one configuration,
/// 1:1 `jqmc.mpc.compute_mpc_potential_ee`: min-image pair sum + periodic tricubic
/// B-spline of `v_lr_coef` + `v_const`. Consumes a built `MPC_data`; `build_mpc_data`
/// (including the B-spline coefficient deconvolution) stays in Python.
#[pyfunction]
#[pyo3(name = "_compute_mpc_potential_ee", signature = (mpc_data, r_up_carts, r_dn_carts))]
fn py_compute_mpc_potential_ee(
    mpc_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let m = mpc::MpcData::from_py(mpc_data)?;
    let r_up = r_up_carts.as_array();
    let r_dn = r_dn_carts.as_array();
    if r_up.ncols() != 3 || r_dn.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "electron coordinates must have shape (N, 3), got up={:?}, dn={:?}",
            r_up.shape(),
            r_dn.shape()
        )));
    }
    Ok(mpc::compute_mpc_potential_ee(
        &m.cell.view(),
        &m.origin,
        m.n_max,
        &m.v_lr_coef.view(),
        m.v_const,
        &r_up,
        &r_dn,
    ))
}

/// Canonical MPC finite-size correction `dV = V_ee^MPC - V_ee^Ewald` (Ha) for one
/// configuration, 1:1 `jqmc.mpc.compute_mpc_correction_ee`. This is the passive,
/// energy-only observable (`physical = no`): walkers are Ewald-propagated and this
/// difference is averaged over the SAME configurations. REUSES the existing Rust Ewald
/// e-e value kernel (`ewald::compute_ewald_potential_ee`) -- NO Ewald reimplementation and
/// NO MPC derivative. `eta` / `real_image_list` / `g_vectors` / `g_factors` and the lattice
/// come from `ewald_data` (+ its `structure_data`); the MPC half from `mpc_data`.
#[pyfunction]
#[pyo3(name = "_compute_mpc_correction_ee", signature = (mpc_data, ewald_data, r_up_carts, r_dn_carts))]
fn py_compute_mpc_correction_ee(
    mpc_data: &Bound<'_, PyAny>,
    ewald_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let m = mpc::MpcData::from_py(mpc_data)?;
    let r_up = r_up_carts.as_array();
    let r_dn = r_dn_carts.as_array();
    if r_up.ncols() != 3 || r_dn.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "electron coordinates must have shape (N, 3), got up={:?}, dn={:?}",
            r_up.shape(),
            r_dn.shape()
        )));
    }
    // Ewald e-e: REBUILD the Cartesian tables from the integer index sets + the
    // (possibly strained) cell, exactly like the JAX `_ewald_tables_from_cell` that
    // `compute_ewald_potential_ee` uses in-trace -- the cached `real_image_list` /
    // `g_vectors` / `g_factors` on `ewald_data` are stale under a strain (they were
    // built for the unstrained cell). Then feed the SAME Rust Ewald value kernel via
    // the shared `mpc::mpc_correction_value`; no Ewald reimplementation, no MPC deriv.
    let tables = mpc::mpc_ewald_tables_from_py(ewald_data)?;
    Ok(mpc::mpc_correction_value(&m, &tables, &r_up, &r_dn))
}

/// Batched twin of [`py_compute_mpc_correction_ee`]: the
/// passive energy-only MPC correction `dV = V_ee^MPC - V_ee^Ewald` for a whole walker
/// batch. `mpc_data` / `ewald_data` (the big statics) are parsed EXACTLY ONCE before the
/// parallel region; each walker's `dV` is computed by the SAME shared
/// `mpc::mpc_correction_value` the scalar entry calls, so `out[w]` is bit-identical to
/// `_compute_mpc_correction_ee(mpc_data, ewald_data, r_up[w], r_dn[w])` regardless of the
/// thread count. Walker parallelism goes through the [`walker_batch::for_each_walker`]
/// harness only. Returns `(nw,)` f64. Switches the 4 driver MPC list loops
/// (jqmc_mcmc.py chunk / per-step, jqmc_gfmc.py GFMC-t / GFMC-n) to one call per batch.
#[pyfunction]
#[pyo3(name = "_compute_mpc_correction_ee_batch", signature = (mpc_data, ewald_data, r_up_carts, r_dn_carts))]
fn py_compute_mpc_correction_ee_batch(
    py: Python<'_>,
    mpc_data: &Bound<'_, PyAny>,
    ewald_data: &Bound<'_, PyAny>,
    r_up_carts: numpy::PyReadonlyArray3<f64>,
    r_dn_carts: numpy::PyReadonlyArray3<f64>,
) -> PyResult<Py<PyAny>> {
    use numpy::ndarray::Axis;
    // Parse the big statics ONCE (shared by reference across every walker).
    let m = mpc::MpcData::from_py(mpc_data)?;
    let tables = mpc::mpc_ewald_tables_from_py(ewald_data)?;
    // Extract the per-walker coords to owned ndarrays (Send) BEFORE releasing the GIL;
    // `PyReadonlyArray3` is not Send.
    let r_up = r_up_carts.as_array();
    let r_dn = r_dn_carts.as_array();
    if r_up.ndim() != 3 || r_dn.ndim() != 3 || r_up.shape()[2] != 3 || r_dn.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "batched electron coordinates must have shape (nw, N, 3), got up={:?}, dn={:?}",
            r_up.shape(),
            r_dn.shape()
        )));
    }
    if r_up.shape()[0] != r_dn.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "walker-axis mismatch: r_up nw={} vs r_dn nw={}",
            r_up.shape()[0],
            r_dn.shape()[0]
        )));
    }
    let ru = r_up.to_owned();
    let rd = r_dn.to_owned();
    let nw = ru.shape()[0];
    let out = walker_batch::for_each_walker(py, nw, |w| {
        let rus = ru.index_axis(Axis(0), w);
        let rds = rd.index_axis(Axis(0), w);
        mpc::mpc_correction_value(&m, &tables, &rus, &rds)
    });
    Ok(numpy::ndarray::Array1::from_vec(out)
        .into_pyarray(py)
        .into_any()
        .unbind())
}

/// GFMC HF-force building block: `de_L^t/d(R, r_up, r_dn)` of the LRDMC diagonal
/// local energy (AE molecular; 1:1 jax.grad of `_compute_local_energy_t`).
/// Returns `(dR (n_atom,3), dr_up (N_up,3), dr_dn (N_dn,3))`. DTN unsupported.
#[pyfunction]
#[pyo3(name = "_gfmc_grad_e_L_ae", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_grad_e_l_ae(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    // J3 (skip when the bench j_matrix is all-zero, mirroring _gfmc_lrdmc_ae)
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    // J3: cart-lower for the raw-AO force kernels; nuc is per-cart-AO. Skip a vacuous
    // (all-zero j_matrix) J3, mirroring _gfmc_lrdmc_ae. J3 factor is basis-invariant.
    let (j3_orb, j3_jm, nuc_j3) = if j3d.is_none() {
        (None, None, None)
    } else {
        let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
        if jm.as_array().iter().all(|&x| x == 0.0) {
            (None, None, None)
        } else {
            let (o, m, n) =
                lower_j3_leaf_cart(&j3d.getattr("orb_data")?, &j3d.getattr("j_matrix")?)?;
            (Some(o), Some(m), Some(n))
        }
    };
    let j3v = j3_jm.as_ref().map(|m| m.view());
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    if sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_grad_e_L_ae: molecular AE only (PBC is a follow-up)",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let g = gfmc_forces::gfmc_grad_e_l_ae(
        &gem,
        &jdata,
        j3_orb.as_ref(),
        j3v.as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
    );
    Ok((
        g.d_r.into_pyarray(py).into_any().unbind(),
        g.d_rup.into_pyarray(py).into_any().unbind(),
        g.d_rdn.into_pyarray(py).into_any().unbind(),
    ))
}

/// Parse only host geometry: native SWCT must never invoke a JAX callback.
fn swct_geometry_from_py(structure_data: &Bound<'_, PyAny>) -> PyResult<gfmc_forces::SwctGeometry> {
    // `Structure_data.replace` can retain concrete JAX array leaves. Transfer
    // their stored values to host exactly as other native geometry boundaries
    // do; this is not a JAX evaluation or a numerical fallback.
    let np = structure_data.py().import("numpy")?;
    let positions = np
        .call_method1("asarray", (structure_data.getattr("positions")?,))?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    if positions.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SWCT positions must have shape (n_atom, 3)",
        ));
    }
    let pbc: bool = structure_data.getattr("pbc_flag")?.extract()?;
    let cell_nmax = if pbc {
        let cell = np
            .call_method1("asarray", (structure_data.getattr("cell")?,))?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let nmax: i32 = structure_data.getattr("_n_max")?.extract()?;
        if cell.dim() != (3, 3) || nmax < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "SWCT periodic cell must be (3, 3) with nonnegative n_max",
            ));
        }
        let determinant = ewald::det3(&cell.view());
        if !determinant.is_finite() || determinant == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "SWCT periodic cell must be finite and nonsingular",
            ));
        }
        Some((cell, nmax))
    } else {
        None
    };
    Ok(gfmc_forces::SwctGeometry::new(positions, cell_nmax))
}

/// SWCT weights `omega (n_atom, N_e)` (1:1 `swct.evaluate_swct_omega`).
#[pyfunction]
#[pyo3(name = "_swct_omega", signature = (structure_data, r_carts))]
fn py_swct_omega(
    py: Python<'_>,
    structure_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let geometry = swct_geometry_from_py(structure_data)?;
    let r_carts = r_carts.as_array();
    if r_carts.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SWCT coordinates must have shape (n_electron, 3)",
        ));
    }
    let o = gfmc_forces::swct_omega_geometry(&geometry, &r_carts);
    Ok(o.into_pyarray(py).into_any().unbind())
}

/// SWCT `Sum_i gradomega (n_atom, 3)` (1:1 `swct.evaluate_swct_domega`).
#[pyfunction]
#[pyo3(name = "_swct_domega", signature = (structure_data, r_carts))]
fn py_swct_domega(
    py: Python<'_>,
    structure_data: &Bound<'_, PyAny>,
    r_carts: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let geometry = swct_geometry_from_py(structure_data)?;
    let r_carts = r_carts.as_array();
    if r_carts.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SWCT coordinates must have shape (n_electron, 3)",
        ));
    }
    let o = gfmc_forces::swct_domega_geometry(&geometry, &r_carts);
    Ok(o.into_pyarray(py).into_any().unbind())
}

fn swct_corrections_batch<T>(
    py: Python<'_>,
    geometry: &gfmc_forces::SwctGeometry,
    r_carts: &numpy::ndarray::ArrayView3<f64>,
    fields: &numpy::ndarray::ArrayView3<T>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)>
where
    T: numpy::Element
        + Copy
        + Default
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::Mul<f64, Output = T>,
{
    use numpy::ndarray::{s, Array3};
    let (batch, n_e, xyz) = r_carts.dim();
    let (field_batch, field_ne, n_field) = fields.dim();
    if xyz != 3 || batch != field_batch || n_e != field_ne {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SWCT requires coordinates (batch, n_electron, 3) and fields (batch, n_electron, n_field)",
        ));
    }
    let results = walker_batch::for_each_walker(py, batch, |b| {
        gfmc_forces::swct_corrections(
            geometry,
            &r_carts.slice(s![b, .., ..]),
            &fields.slice(s![b, .., ..]),
        )
    });
    let n_atom = geometry.n_atom();
    let mut dot = Array3::from_elem((batch, n_atom, n_field), T::default());
    let mut domega = Array3::<f64>::zeros((batch, n_atom, 3));
    for (b, (one_dot, one_domega)) in results.into_iter().enumerate() {
        dot.slice_mut(s![b, .., ..]).assign(&one_dot);
        domega.slice_mut(s![b, .., ..]).assign(&one_domega);
    }
    Ok((
        dot.into_pyarray(py).into_any().unbind(),
        domega.into_pyarray(py).into_any().unbind(),
    ))
}

/// Native fused SWCT field contraction and weight-gradient sum. The SWCT
/// precision zone is float64 in both modes; complex128 fields retain phase.
/// Geometry scratch is O(n_atom) per worker, including empty spin channels.
#[pyfunction]
#[pyo3(name = "_swct_corrections_batched", signature = (structure_data, r_carts_batch, field_batch))]
fn py_swct_corrections_batched(
    py: Python<'_>,
    structure_data: &Bound<'_, PyAny>,
    r_carts_batch: numpy::PyReadonlyArray3<f64>,
    field_batch: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let geometry = swct_geometry_from_py(structure_data)?;
    let r_carts = r_carts_batch.as_array();
    if let Ok(fields) = field_batch.extract::<numpy::PyReadonlyArray3<f64>>() {
        swct_corrections_batch(py, &geometry, &r_carts, &fields.as_array())
    } else {
        let fields = field_batch.extract::<numpy::PyReadonlyArray3<numpy::Complex64>>()?;
        swct_corrections_batch(py, &geometry, &r_carts, &fields.as_array())
    }
}

/// GFMC HF-force for the PBC (Ewald, Gamma) ECP case -- jnone or withJ (J1/J2).
/// `de_L^t/d(R,r)`. DTN and J3 are follow-ups (guarded).
#[pyfunction]
#[pyo3(name = "_gfmc_grad_e_L_ecp_pbc", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false, debug=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_grad_e_l_ecp_pbc(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    debug: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?; // native (composite-aware)
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    let j3_owned = j3_pbc_ctx_owned(&j3d)?;
    let j3_batch = if j3_owned.is_none() {
        j3_batch_ctx_owned(&j3d)?
    } else {
        None
    };
    let jm_v = j3_batch.as_ref().map(|(_, m)| m.view());
    let batch = j3_batch.as_ref().map(|(o, _)| (o, jm_v.as_ref().unwrap()));
    let j3 = j3_owned.as_ref().map(
        |(orb, cart, t, jm, home, pw, gto_first)| gfmc_forces::J3Pbc {
            orb,
            cart_j3: cart,
            t_mol: t.view(),
            jm: jm.view(),
            home,
            pw: pw.as_ref().map(|e| e.view()),
            gto_first: *gto_first,
        },
    );
    let up_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
    let dn_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    if !sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_grad_e_L_ecp_pbc: PBC only",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let ew_arr = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
    })?;
    let ew = ew_arr.as_ref();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let positions_v = positions.view();
    let r_up_v = r_up_carts.as_array();
    let r_dn_v = r_dn_carts.as_array();
    let grid_v = grid_points.as_array();
    let weights_v = weights.as_slice()?;
    let g = if debug {
        gfmc_forces::gfmc_grad_e_l_ecp_pbc_debug(
            &gem,
            &jdata,
            j3.as_ref(),
            &up_leaves,
            &dn_leaves,
            &positions_v,
            &eff_charges,
            &r_up_v,
            &r_dn_v,
            alat,
            &rtm,
            &ew,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights_v,
            &grid_v,
            nn,
            dltmove,
            batch,
        )
    } else {
        gfmc_forces::gfmc_grad_e_l_ecp_pbc(
            &gem,
            &jdata,
            j3.as_ref(),
            &up_leaves,
            &dn_leaves,
            &positions_v,
            &eff_charges,
            &r_up_v,
            &r_dn_v,
            alat,
            &rtm,
            &ew,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights_v,
            &grid_v,
            nn,
            dltmove,
            batch,
        )
    };
    Ok((
        g.d_r.into_pyarray(py).into_any().unbind(),
        g.d_rup.into_pyarray(py).into_any().unbind(),
        g.d_rdn.into_pyarray(py).into_any().unbind(),
    ))
}

/// COMPLEX-TWIST (k != 0) GFMC/LRDMC HF force `de_L^t/d(R, r_up, r_dn)` (real, Re-projected)
/// for the periodic ECP twisted geminal. jnone + j12 (J1/J2) + DTN. GFMC_t and GFMC_n share
/// this reduction; verify vs jqmc `gfmc.grad_e_L_{t,n}_wrt_positions`.
#[pyfunction]
#[pyo3(name = "_gfmc_grad_e_L_ecp_pbc_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false, debug=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_grad_e_l_ecp_pbc_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    debug: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (j3_orb, j3_jm, j3_home, j3_kernel) = parse_j3_complex_ctx(jastrow_data)?;
    let j3ctx = match (&j3_orb, &j3_jm, &j3_home, &j3_kernel) {
        (Some(o), Some(m), Some(h), Some(k)) => Some(gfmc_forces::J3PbcComplex {
            orb: o,
            jm: m.view(),
            home: h,
            kernel: k,
        }),
        _ => None,
    };
    // Pure plane-wave / B-spline leaves have no atom-centred determinant
    // rows; composites expose only their pGTO block here.  Their electron
    // derivatives remain present in ``ComplexGeminal`` and are evaluated by
    // the corresponding leaf-specific kinetic kernels.
    let nuc_up =
        complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_up_spin")?)?)?;
    let nuc_dn =
        complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_dn_spin")?)?)?;
    let j3_jm_v = j3_jm.as_ref().map(|m| m.view());
    let batch_j3 = match (&j3_orb, &j3_jm_v, &j3_home) {
        (Some(o), Some(m), None) => Some((o, m)),
        _ => None,
    };
    let cp = coulomb_potential_data;
    let ecp_flag = cp.getattr("ecp_flag")?.extract::<bool>()?;
    let sd = cp.getattr("structure_data")?;
    if !sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_grad_e_L_ecp_pbc_complex: PBC only",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let ew_arr = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
    })?;
    let ew = ew_arr.as_ref();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let positions_v = positions.view();
    let r_up_v = r_up_carts.as_array();
    let r_dn_v = r_dn_carts.as_array();
    let grid_v = grid_points.as_array();
    let weights_v = weights.as_slice()?;
    let args = (
        &gem,
        &jdata,
        j3ctx.as_ref(),
        &nuc_up[..],
        &nuc_dn[..],
        &positions_v,
        &eff_charges[..],
        &r_up_v,
        &r_dn_v,
        alat,
        &rtm,
        &ew,
        ecp_flag,
        &nuc_l[..],
        &e_l_p[..],
        &c_l_p[..],
        &p_l_p[..],
        &nuc_n[..],
        &am_n[..],
        &e_n[..],
        &c_n[..],
        &p_n[..],
        weights_v,
        &grid_v,
        nn,
        dltmove,
        batch_j3,
    );
    let g = if debug {
        gfmc_forces::gfmc_grad_e_l_ecp_pbc_complex_debug(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11, args.12, args.13, args.14, args.15, args.16, args.17, args.18,
            args.19, args.20, args.21, args.22, args.23, args.24, args.25, args.26,
        )
    } else {
        gfmc_forces::gfmc_grad_e_l_ecp_pbc_complex(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11, args.12, args.13, args.14, args.15, args.16, args.17, args.18,
            args.19, args.20, args.21, args.22, args.23, args.24, args.25, args.26,
        )
    };
    Ok((
        g.d_r.into_pyarray(py).into_any().unbind(),
        g.d_rup.into_pyarray(py).into_any().unbind(),
        g.d_rdn.into_pyarray(py).into_any().unbind(),
    ))
}

/// COMPLEX-TWIST (k != 0) GFMC/LRDMC HF stress `de_L^t/deps` (real 3x3) for the periodic ECP
/// twisted geminal. jnone + j12 (J1/J2) + DTN + j123ao (complex-sphe J3). Verify vs jqmc `gfmc.grad_e_L_{t,n}_wrt_strain`.
#[pyfunction]
#[pyo3(name = "_gfmc_stress_ecp_pbc_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false, debug=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_stress_ecp_pbc_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    debug: bool,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (home_up, kernel_up, home_dn, kernel_dn, _cell0, _pos0) =
        complex_stress_args(geminal_data)?;
    let (j3_orb, j3_jm, j3_home, j3_kernel) = parse_j3_complex_ctx(jastrow_data)?;
    let j3ctx = match (&j3_orb, &j3_jm, &j3_home, &j3_kernel) {
        (Some(o), Some(m), Some(h), Some(k)) => Some(gfmc_forces::J3PbcComplex {
            orb: o,
            jm: m.view(),
            home: h,
            kernel: k,
        }),
        _ => None,
    };
    let j3_jm_v = j3_jm.as_ref().map(|m| m.view());
    let batch_j3 = match (&j3_orb, &j3_jm_v, &j3_home) {
        (Some(o), Some(m), None) => Some((o, m)),
        _ => None,
    };
    let cp = coulomb_potential_data;
    let ecp_flag = cp.getattr("ecp_flag")?.extract::<bool>()?;
    let sd = cp.getattr("structure_data")?;
    if !sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_stress_ecp_pbc_complex: PBC only",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let ew_arr = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
    })?;
    let ew = ew_arr.as_ref();
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let s = if debug {
        gfmc_forces::gfmc_stress_ecp_pbc_complex_debug(
            &gem,
            &jdata,
            j3ctx.as_ref(),
            &home_up,
            &kernel_up,
            &home_dn,
            &kernel_dn,
            &positions.view(),
            &eff_charges,
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            alat,
            &rtm,
            &ew,
            ecp_flag,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights.as_slice()?,
            &grid_points.as_array(),
            nn,
            dltmove,
            batch_j3,
        )
    } else {
        gfmc_forces::gfmc_stress_ecp_pbc_complex(
            &gem,
            &jdata,
            j3ctx.as_ref(),
            &home_up,
            &kernel_up,
            &home_dn,
            &kernel_dn,
            &positions.view(),
            &eff_charges,
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            alat,
            &rtm,
            &ew,
            ecp_flag,
            &nuc_l,
            &e_l_p,
            &c_l_p,
            &p_l_p,
            &nuc_n,
            &am_n,
            &e_n,
            &c_n,
            &p_n,
            weights.as_slice()?,
            &grid_points.as_array(),
            nn,
            dltmove,
            batch_j3,
        )
    };
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// GFMC HF-force building block for the ECP (tmove/dltmove) molecular case:
/// `de_L^t/d(R, r_up, r_dn)`. DTN unsupported; molecular only.
#[pyfunction]
#[pyo3(name = "_gfmc_grad_e_L_ecp", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_grad_e_l_ecp(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
    // J3: cart-lower for the raw-AO force kernels; nuc is per-cart-AO. Skip a vacuous
    // (all-zero j_matrix) J3, mirroring _gfmc_lrdmc_ae. J3 factor is basis-invariant.
    let (j3_orb, j3_jm, nuc_j3) = if j3d.is_none() {
        (None, None, None)
    } else {
        let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
        if jm.as_array().iter().all(|&x| x == 0.0) {
            (None, None, None)
        } else {
            let (o, m, n) =
                lower_j3_leaf_cart(&j3d.getattr("orb_data")?, &j3d.getattr("j_matrix")?)?;
            (Some(o), Some(m), Some(n))
        }
    };
    let j3v = j3_jm.as_ref().map(|m| m.view());
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    if sd.getattr("pbc_flag")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "_gfmc_grad_e_L_ecp: molecular only (PBC is a follow-up)",
        ));
    }
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let np = cp.py().import("numpy")?;
    let geti = |a: &str| -> PyResult<Vec<i64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let getf = |a: &str| -> PyResult<Vec<f64>> {
        np.call_method1("asarray", (cp.getattr(a)?,))?
            .call_method0("tolist")?
            .extract()
    };
    let nuc_l = geti("_nucleus_index_local_part")?;
    let e_l_p = getf("_exponents_local_part")?;
    let c_l_p = getf("_coefficients_local_part")?;
    let p_l_p = getf("_powers_local_part")?;
    let nuc_n = geti("_nucleus_index_non_local_part")?;
    let am_n = geti("_ang_mom_non_local_part")?;
    let e_n = getf("_exponents_non_local_part")?;
    let c_n = getf("_coefficients_non_local_part")?;
    let p_n = getf("_powers_non_local_part")?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let g = gfmc_forces::gfmc_grad_e_l_ecp(
        &gem,
        &jdata,
        j3_orb.as_ref(),
        j3v.as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
        &nuc_l,
        &e_l_p,
        &c_l_p,
        &p_l_p,
        &nuc_n,
        &am_n,
        &e_n,
        &c_n,
        &p_n,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
    );
    Ok((
        g.d_r.into_pyarray(py).into_any().unbind(),
        g.d_rup.into_pyarray(py).into_any().unbind(),
        g.d_rdn.into_pyarray(py).into_any().unbind(),
    ))
}

/// VMC (MCMC) molecular AE HF-force core `(de_L/dR, de_L/dr_up, de_L/dr_dn)`
/// reusing the sweep SM `geminal_inv`. This is the VMC
/// continuous local-energy gradient (`compute_grad_e_L` wrt R/r). The novar=0
/// LRDMC measurement derivative uses the same continuous composition. Assembled
/// and Omega-contracted into `force_HF` by the driver.
#[pyfunction]
#[pyo3(name = "vmc_grad_e_L_ae", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, geminal_inv))]
#[allow(clippy::too_many_arguments)]
fn py_vmc_grad_e_l_ae(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = MolForceArgs::from_py(
        geminal_data,
        jastrow_data,
        coulomb_potential_data,
        "vmc_grad_e_L_ae",
    )?;
    let (d_r, du, dd) = a.compute_ae(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &geminal_inv.as_array(),
    );
    Ok(triple(py, d_r, du, dd))
}

/// VMC (MCMC) molecular ECP HF-force core `(de_L/dR, de_L/dr_up, de_L/dr_dn)` =
/// `vmc_grad_e_L_ae` diagonal + ECP local/non-local force (shared VMC/LRDMC: the
/// ECP potential is not alat-discretized), reusing the sweep SM `geminal_inv` for
/// the base-config kinetic + ECP `O` fields (mesh re-evals stay fresh). Mirrors
/// jqmc `compute_grad_e_L` wrt R + `_deL_dr` ECP assembly.
#[pyfunction]
///
/// `fast_mesh_screen = True` differentiates the FAST-chunk e_L's screened ECP mesh
/// term instead of the exact one ([`gfmc_forces::FastScreen`]; the FAST MCMC
/// chunk sets it from its own e_L route).
#[pyo3(name = "vmc_grad_e_L_ecp", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv, fast_mesh_screen=false))]
#[allow(clippy::too_many_arguments)]
fn py_vmc_grad_e_l_ecp(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
    fast_mesh_screen: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = MolForceArgs::from_py(
        geminal_data,
        jastrow_data,
        coulomb_potential_data,
        "vmc_grad_e_L_ecp",
    )?;
    let (d_r, du, dd) = a.compute_ecp(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        &geminal_inv.as_array(),
        gfmc_forces::FastScreen {
            det: fast_mesh_screen,
            j3: fast_mesh_screen,
            ecp_radial: fast_mesh_screen,
            mesh_shell_only: false,
        },
    );
    Ok(triple(py, d_r, du, dd))
}

/// GFMC Pulay-force building block: `(dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn)`
/// (1:1 `_grad_ln_psi_rR`; ln|Psi| is LRDMC-independent). J1/J2/J3(MO or AO)/DTN all
/// included (the Jastrow pieces are the AO-basis `grad_j*`/`jastrow_grads` kernels).
#[pyfunction]
#[pyo3(name = "_gfmc_grad_ln_Psi", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts))]
fn py_gfmc_grad_ln_psi(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    // Bounded guard mirroring the sibling `_gfmc_grad_e_L_ae` (molecular AE only):
    // a periodic geminal leaf makes `geminal_cart_from_py` -> `orbdata_to_cart`
    // index out of bounds (raw `PanicException`). Raise a typed PyNotImplementedError
    // naming the unsupported leaf instead. PBC is a follow-up.
    let up_orb = geminal_data.getattr("orb_data_up_spin")?;
    if up_orb
        .getattr("structure_data")?
        .getattr("pbc_flag")?
        .extract::<bool>()?
    {
        let leaf = if up_orb.hasattr("aos_data")? {
            up_orb.getattr("aos_data")?
        } else {
            up_orb.clone()
        };
        let leaf_name = leaf.get_type().name()?.to_string();
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "_gfmc_grad_ln_Psi: molecular only (periodic geminal leaf '{leaf_name}' unsupported; PBC is a follow-up)"
        )));
    }
    let gem = geminal_cart_from_py(geminal_data)?;
    let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
    let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
    let n_atom = geminal_data
        .getattr("orb_data_up_spin")?
        .getattr("structure_data")?
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .nrows();
    let (jas, j3o, j3m) = parse_jastrow_full(jastrow_data)?;
    let j3v = j3m.as_ref().map(|m| m.view());
    let nuc_j3 = if j3o.is_some() {
        Some(extract_nuc(
            &jastrow_data
                .getattr("jastrow_three_body_data")?
                .getattr("orb_data")?,
        )?)
    } else {
        None
    };
    let j1_ref = jas.as_ref().and_then(|j| j.one_body.as_ref());
    let (d_r, du, dd) = gfmc_forces::gfmc_grad_ln_psi(
        &gem,
        jas.as_ref(),
        j1_ref,
        j3o.as_ref(),
        j3v.as_ref(),
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &nuc_up,
        &nuc_dn,
        nuc_j3.as_deref(),
        n_atom,
    );
    Ok((
        d_r.into_pyarray(py).into_any().unbind(),
        du.into_pyarray(py).into_any().unbind(),
        dd.into_pyarray(py).into_any().unbind(),
    ))
}

/// VMC (MCMC) molecular Pulay WF-gradient core `(dln|Psi|/dR, dln|Psi|/dr_up,
/// dln|Psi|/dr_dn)` (1:1 the JAX `_grad_ln_psi_rR` = `compute_grad_ln_Psi` wrt R/r),
/// reusing the sweep SM `geminal_inv` for the determinant Pulay + drift (the Jastrow
/// J1/J2/J3/DTN pieces are inverse-free). Identical to `_gfmc_grad_ln_Psi` but threads
/// the running inverse instead of a fresh LU, remaining stable near singular
/// configurations and avoiding a second solve. Molecular only.
#[pyfunction]
#[pyo3(name = "vmc_grad_ln_psi", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv))]
fn py_vmc_grad_ln_psi(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = MolLnPsiArgs::from_py(geminal_data, jastrow_data)?;
    let (d_r, du, dd) = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &geminal_inv.as_array(),
    );
    Ok(triple(py, d_r, du, dd))
}

/// VMC (MCMC) periodic real-Gamma HF-force core `(de_L/dR, de_L/dr_up, de_L/dr_dn)`,
/// reusing the sweep SM `geminal_inv`. This is the
/// continuous local-energy gradient (`compute_grad_e_L` wrt R/r); novar=0
/// LRDMC measurement uses the same composition. AE (bare Ewald)
/// when `ecp_flag=False`; ECP local (min-image) + non-local (NN=1 continuum) added
/// otherwise. Grid points arrive PRE-ROTATED (rt=identity inside). MPC swaps the
/// Ewald e-e for the finite-size e-e force when `mpc_data` is Some.
#[pyfunction]
#[pyo3(name = "vmc_grad_e_L_pbc", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
#[allow(clippy::too_many_arguments)]
fn py_vmc_grad_e_l_pbc(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = PbcForceArgs::from_py(geminal_data, jastrow_data, coulomb_potential_data)?;
    let (d_r, du, dd) = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        &geminal_inv.as_array(),
    )?;
    Ok(triple(py, d_r, du, dd))
}

/// VMC (MCMC) PERIODIC real-Gamma Pulay-force core `(dln|Psi|/dR, dln|Psi|/dr_up,
/// dln|Psi|/dr_dn)` (1:1 JAX `compute_grad_ln_Psi` wrt R/r), reusing the sweep SM
/// `geminal_inv`. Pure WF derivative (LRDMC-independent, no S', no ECP/mesh) -- AE
/// and ECP share this one call.
#[pyfunction]
#[pyo3(name = "vmc_grad_ln_psi_pbc", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv))]
fn py_vmc_grad_ln_psi_pbc(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = PbcLnPsiArgs::from_py(geminal_data, jastrow_data)?;
    let (d_r, du, dd) = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &geminal_inv.as_array(),
    );
    Ok(triple(py, d_r, du, dd))
}

/// VMC (MCMC) COMPLEX-twist (k != Gamma) HF-force core `(de_L/dR, de_L/dr_up,
/// de_L/dr_dn)` (real, Re-projected). Smooth base of `_gfmc_grad_e_L_ecp_pbc_complex`
/// (complex continuum kinetic + Ewald + ECP local/non-local) WITHOUT the S' block.
/// Reuses the sweep's running COMPLEX Sherman-Morrison `geminal_inv` for every base-config
/// determinant field (kinetic reverse + Pulay + drift), exactly as the real twin does: a
/// fresh per-config `inverse_complex` NaNs at a near-singular geminal while the
/// AS-regularized swept inverse stays finite. Reflected ECP-mesh candidates keep a fresh
/// inverse (a different, non-near-singular config). MPC-twist force unsupported.
#[pyfunction]
#[pyo3(name = "vmc_grad_e_L_pbc_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv=None))]
#[allow(clippy::too_many_arguments)]
fn py_vmc_grad_e_l_pbc_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = PbcForceComplexArgs::from_py(geminal_data, jastrow_data, coulomb_potential_data)?;
    let gi = geminal_inv.as_ref().map(|g| g.as_array());
    let (d_r, du, dd) = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        gi.as_ref(),
    )?;
    Ok(triple(py, d_r, du, dd))
}

/// VMC (MCMC) COMPLEX-twist Pulay-force core `(dln|Psi|/dR, dln|Psi|/dr_up,
/// dln|Psi|/dr_dn)` (real, Re-projected): Re(holomorphic det Pulay/drift) + real
/// Jastrow (1:1 JAX `compute_grad_ln_Psi` `take_real`). No S', no ECP/mesh.
#[pyfunction]
#[pyo3(name = "vmc_grad_ln_psi_pbc_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv=None))]
fn py_vmc_grad_ln_psi_pbc_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let a = PbcLnPsiComplexArgs::from_py(geminal_data, jastrow_data)?;
    let gi = geminal_inv.as_ref().map(|g| g.as_array());
    let (d_r, du, dd) = a.compute(&r_up_carts.as_array(), &r_dn_carts.as_array(), gi.as_ref());
    Ok(triple(py, d_r, du, dd))
}

// ===================== VMC (MCMC) periodic stress aggregates =====================
// de_L/deps (HF stress) + dln|Psi|/deps (Pulay stress), periodic only, replacing
// the JAX `_grad_e_L_eps` / `_grad_ln_psi_eps`. Both real-Gamma and
// complex-twist reuse the sweep SM `geminal_inv` (complex128 at a twist); omitting it falls
// back to a fresh per-config inverse (the GFMC / oracle path).

/// VMC PERIODIC real-Gamma Pulay stress `dln|Psi|/deps` (3x3), reusing the sweep SM
/// `geminal_inv`. AE and ECP share this call (pure WF derivative).
#[pyfunction]
#[pyo3(name = "vmc_grad_ln_psi_eps_pbc", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv))]
fn py_vmc_grad_ln_psi_eps_pbc(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let a = PbcStressPpArgs::from_py(geminal_data, jastrow_data)?;
    let s = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        &geminal_inv.as_array(),
    );
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// VMC PERIODIC real-Gamma HF stress `de_L/deps` (3x3), reusing the sweep SM `geminal_inv`.
/// `ecp_flag` drives the ECP local/non-local strain. MPC swaps the Ewald ee strain.
#[pyfunction]
#[pyo3(name = "vmc_grad_e_L_eps_pbc", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv, fast_mesh=true))]
#[allow(clippy::too_many_arguments)]
fn py_vmc_grad_e_l_eps_pbc(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
    fast_mesh: bool,
) -> PyResult<Py<PyAny>> {
    let a = PbcStressHfArgs::from_py(geminal_data, jastrow_data, coulomb_potential_data)?;
    let s = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        &geminal_inv.as_array(),
        fast_mesh,
    )?;
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// VMC PERIODIC complex-twist Pulay stress `dln|Psi|/deps` (real 3x3, Re-projected).
#[pyfunction]
#[pyo3(name = "vmc_grad_ln_psi_eps_pbc_complex", signature = (geminal_data, jastrow_data, r_up_carts, r_dn_carts, geminal_inv=None))]
fn py_vmc_grad_ln_psi_eps_pbc_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
) -> PyResult<Py<PyAny>> {
    let a = PbcStressPpComplexArgs::from_py(geminal_data, jastrow_data)?;
    let gi = geminal_inv.as_ref().map(|g| g.as_array());
    let s = a.compute(&r_up_carts.as_array(), &r_dn_carts.as_array(), gi.as_ref());
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// VMC PERIODIC complex-twist HF stress `de_L/deps` (real 3x3, Re-projected). `grid_points`
/// are pre-rotated by the driver's per-measurement RT (rt=identity inside). MPC-twist unsupported.
/// Supports the pgto / pure plane-wave / pure B-spline / pgto(+)pw complex determinant leaf: the
/// complex kinetic-stress core folds the gto field strain (0 AOs for a pure pw/bspline leaf), the
/// PW metric response (`pw_lap/gou_stress_complex`) and the B-spline metric response
/// (`bspline_kstress_side_complex`). A pgto(+)bspline composite stays on JAX (driver gate).
#[pyfunction]
#[pyo3(name = "vmc_grad_e_L_eps_pbc_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, weights, grid_points, geminal_inv=None, fast_mesh=true))]
#[allow(clippy::too_many_arguments)]
fn py_vmc_grad_e_l_eps_pbc_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
    fast_mesh: bool,
) -> PyResult<Py<PyAny>> {
    let a = PbcStressHfComplexArgs::from_py(geminal_data, jastrow_data, coulomb_potential_data)?;
    let gi = geminal_inv.as_ref().map(|g| g.as_array());
    let s = a.compute(
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        weights.as_slice()?,
        &grid_points.as_array(),
        gi.as_ref(),
        fast_mesh,
    )?;
    Ok(s.into_pyarray(py).into_any().unbind())
}

/// Complex Bloch-twist (k!=Gamma) all-electron LRDMC diagonal e_L, jnone. Returns
/// (Re e_L, Im e_L, Re b_L, Im b_L). Requires a periodic (twist) coulomb_potential_data.
#[pyfunction]
#[pyo3(name = "_gfmc_lrdmc_ae_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, fast=false))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_gfmc_lrdmc_ae_complex(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    fast: bool,
) -> PyResult<(f64, f64, f64, f64)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (_, _, j3c, j3m) = parse_jastrow_complex(jastrow_data)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let pbc_owned = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "_gfmc_lrdmc_ae_complex requires a periodic (twist) system",
        )
    })?;
    let pbc = pbc_owned.as_ref();
    let (e_l, _e_l_meas, b_l, _) = gfmc::gfmc_lrdmc_ae_complex(
        &gem,
        &jdata,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &pbc,
        fast,
    );
    Ok((e_l.re, e_l.im, b_l.re, b_l.im))
}

/// Complex Bloch-twist ECP (tmove) LRDMC diagonal e_L. Returns (Re,Im e_L, Re,Im b_L, n_cands).
#[pyfunction]
#[pyo3(name = "_gfmc_lrdmc_ecp_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false, fast=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_lrdmc_ecp_complex(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    fast: bool,
) -> PyResult<(f64, f64, f64, f64, usize)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (_, _, j3c, j3m) = parse_jastrow_complex(jastrow_data)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let nuc_l: Vec<i64> = cp.getattr("_nucleus_index_local_part")?.extract()?;
    let e_l_p: Vec<f64> = cp
        .getattr("_exponents_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let c_l_p: Vec<f64> = cp
        .getattr("_coefficients_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let p_l_p: Vec<f64> = cp
        .getattr("_powers_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let nuc_n: Vec<i64> = cp.getattr("_nucleus_index_non_local_part")?.extract()?;
    let am: Vec<i64> = cp.getattr("_ang_mom_non_local_part")?.extract()?;
    let e_n: Vec<f64> = cp
        .getattr("_exponents_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let c_n: Vec<f64> = cp
        .getattr("_coefficients_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let p_n: Vec<f64> = cp
        .getattr("_powers_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let pbc_owned = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "_gfmc_lrdmc_ecp_complex requires a periodic (twist) system",
        )
    })?;
    let pbc = pbc_owned.as_ref();
    let (e_l, _e_l_meas, b_l, _, cands) = gfmc::gfmc_lrdmc_ecp_complex(
        &gem,
        &jdata,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &pbc,
        &nuc_l,
        &e_l_p,
        &c_l_p,
        &p_l_p,
        &nuc_n,
        &am,
        &e_n,
        &c_n,
        &p_n,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
        fast,
    );
    Ok((e_l.re, e_l.im, b_l.re, b_l.im, cands.len()))
}

/// Single-walker complex Bloch-twist AE GFMC_t chain (jnone). Returns (energy, e_L_hist, w_L_hist).
#[pyfunction]
#[pyo3(name = "_gfmc_run_ae_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, tau, alat, num_collect, n_warmup, n_measure, seed, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ae_complex(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (_, _, j3c, j3m) = parse_jastrow_complex(jastrow_data)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let pbc_owned = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "_gfmc_run_ae_complex requires a periodic (twist) system",
        )
    })?;
    let pbc = pbc_owned.as_ref();
    let (e, eh, wh) = gfmc::gfmc_run_ae_complex(
        &gem,
        &jdata,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        tau,
        alat,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        random_mesh,
        &pbc,
    );
    Ok((e, eh, wh))
}

/// Single-walker all-electron LRDMC/GFMC chain. Returns (energy, e_L_hist, w_L_hist).
#[pyfunction]
#[pyo3(name = "_gfmc_run_ae", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, tau, alat, num_collect, n_warmup, n_measure, seed, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ae(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = gfmc_real_j3_parse(&jastrow_data.getattr("jastrow_three_body_data")?)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let pbc_owned = crate::ewald::EwaldData::from_py(cp)?;
    let pbc = pbc_owned.as_ref().map(|a| a.as_ref());
    let (e, eh, wh) = gfmc::gfmc_run_ae(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        tau,
        alat,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        random_mesh,
        pbc.as_ref(),
    );
    Ok((e, eh, wh))
}

/// Multi-walker (population) all-electron GFMC_t with stochastic reconfiguration.
/// Returns (energy, e_L_hist, w_L_hist) -- walker-averaged per branching step.
#[pyfunction]
#[pyo3(name = "_gfmc_run_ae_pop", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, tau, alat, num_collect, n_warmup, n_measure, seed, num_walkers, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ae_pop(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    num_walkers: usize,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = gfmc_real_j3_parse(&jastrow_data.getattr("jastrow_three_body_data")?)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let (e, eh, wh) = gfmc::gfmc_run_ae_pop(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        tau,
        alat,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        num_walkers,
        random_mesh,
    );
    Ok((e, eh, wh))
}

/// Single-walker all-electron GFMC_n (lrdmc-bra, fixed-step branching projector).
/// Returns (energy, e_L_hist, w_L_hist). `e_scf0` seeds the self-tuning E_scf.
#[pyfunction]
#[pyo3(name = "_gfmc_run_ae_n", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, num_per_meas, num_collect, n_warmup, n_measure, seed, e_scf0=0.0, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ae_n(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    num_per_meas: usize,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    e_scf0: f64,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = gfmc_real_j3_parse(&jastrow_data.getattr("jastrow_three_body_data")?)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let (e, eh, wh) = gfmc::gfmc_run_ae_n(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        num_per_meas,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        e_scf0,
        random_mesh,
    );
    Ok((e, eh, wh))
}

/// Shared ECP-GFMC parser: geminal + jastrow(+J3) + structure positions/eff-charges +
/// ECP local/non-local parameter arrays (nuc_l,e_l,c_l,p_l, nuc_n,am,e_n,c_n,p_n).
#[allow(clippy::type_complexity)]
fn gfmc_ecp_parse(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
) -> PyResult<(
    determinant::GeminalData,
    data::JastrowData,
    Option<determinant::OrbData>,
    Option<numpy::ndarray::Array2<f64>>,
    numpy::ndarray::Array2<f64>,
    Vec<f64>,
    (
        Vec<i64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<i64>,
        Vec<i64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ),
)> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py(jastrow_data)?;
    let (j3_orb, j3_jm) = gfmc_real_j3_parse(&jastrow_data.getattr("jastrow_three_body_data")?)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let e = (
        cp.getattr("_nucleus_index_local_part")?.extract()?,
        cp.getattr("_exponents_local_part")?
            .call_method0("tolist")?
            .extract()?,
        cp.getattr("_coefficients_local_part")?
            .call_method0("tolist")?
            .extract()?,
        cp.getattr("_powers_local_part")?
            .call_method0("tolist")?
            .extract()?,
        cp.getattr("_nucleus_index_non_local_part")?.extract()?,
        cp.getattr("_ang_mom_non_local_part")?.extract()?,
        cp.getattr("_exponents_non_local_part")?
            .call_method0("tolist")?
            .extract()?,
        cp.getattr("_coefficients_non_local_part")?
            .call_method0("tolist")?
            .extract()?,
        cp.getattr("_powers_non_local_part")?
            .call_method0("tolist")?
            .extract()?,
    );
    Ok((gem, jdata, j3_orb, j3_jm, positions, eff_charges, e))
}

/// ECP (Casula T-move) LRDMC/GFMC local energy at one config. Returns
/// (e_L, b_L, n_ecp_candidates). `weights`/`grid_points` are the Nv quadrature sphere.
#[pyfunction]
#[pyo3(name = "_gfmc_lrdmc_ecp", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, rt, weights, grid_points, nn, dltmove=false, fast=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_lrdmc_ecp(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    rt: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    fast: bool,
) -> PyResult<(f64, f64, usize)> {
    let (gem, jdata, j3_orb, j3_jm, positions, eff_charges, e) =
        gfmc_ecp_parse(geminal_data, jastrow_data, coulomb_potential_data)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let r = rt.as_array();
    let rtm = [
        [r[[0, 0]], r[[0, 1]], r[[0, 2]]],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]]],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]]],
    ];
    let pbc_owned = crate::ewald::EwaldData::from_py(coulomb_potential_data)?;
    let pbc = pbc_owned.as_ref().map(|a| a.as_ref());
    let (e_l, _e_l_meas, b_l, _, cands) = gfmc::gfmc_lrdmc_ecp(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        &rtm,
        &e.0,
        &e.1,
        &e.2,
        &e.3,
        &e.4,
        &e.5,
        &e.6,
        &e.7,
        &e.8,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
        pbc.as_ref(),
        fast,
    );
    Ok((e_l, b_l, cands.len()))
}

/// Single-walker ECP (T-move) LRDMC/GFMC chain. Returns (energy, e_L_hist, w_L_hist).
#[pyfunction]
#[pyo3(name = "_gfmc_run_ecp", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, tau, alat, num_collect, n_warmup, n_measure, seed, weights, grid_points, nn, dltmove=false, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ecp(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let (gem, jdata, j3_orb, j3_jm, positions, eff_charges, e) =
        gfmc_ecp_parse(geminal_data, jastrow_data, coulomb_potential_data)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let pbc_owned = crate::ewald::EwaldData::from_py(coulomb_potential_data)?;
    let pbc = pbc_owned.as_ref().map(|a| a.as_ref());
    let (en, eh, wh) = gfmc::gfmc_run_ecp(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        tau,
        alat,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        &e.0,
        &e.1,
        &e.2,
        &e.3,
        &e.4,
        &e.5,
        &e.6,
        &e.7,
        &e.8,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
        random_mesh,
        pbc.as_ref(),
    );
    Ok((en, eh, wh))
}

/// Single-walker ECP GFMC_n (lrdmc-bra, fixed-step b_x projector). Returns (energy, e_L_hist, w_L_hist).
#[pyfunction]
#[pyo3(name = "_gfmc_run_ecp_n", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, alat, num_per_meas, num_collect, n_warmup, n_measure, seed, weights, grid_points, nn, e_scf0=0.0, dltmove=false, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ecp_n(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    alat: f64,
    num_per_meas: usize,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    e_scf0: f64,
    dltmove: bool,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let (gem, jdata, j3_orb, j3_jm, positions, eff_charges, e) =
        gfmc_ecp_parse(geminal_data, jastrow_data, coulomb_potential_data)?;
    let j3 = match (&j3_orb, &j3_jm) {
        (Some(o), Some(m)) => Some((o, m.view())),
        _ => None,
    };
    let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
    let pbc_owned = crate::ewald::EwaldData::from_py(coulomb_potential_data)?;
    let pbc = pbc_owned.as_ref().map(|a| a.as_ref());
    let (en, eh, wh) = gfmc::gfmc_run_ecp_n(
        &gem,
        &jdata,
        j3_ref,
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        alat,
        num_per_meas,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        e_scf0,
        &e.0,
        &e.1,
        &e.2,
        &e.3,
        &e.4,
        &e.5,
        &e.6,
        &e.7,
        &e.8,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
        random_mesh,
        pbc.as_ref(),
    );
    Ok((en, eh, wh))
}

/// Single-walker complex Bloch-twist ECP GFMC_t chain (tmove). Returns (energy, e_L_hist, w_L_hist).
#[pyfunction]
#[pyo3(name = "_gfmc_run_ecp_complex", signature = (geminal_data, jastrow_data, coulomb_potential_data, r_up_carts, r_dn_carts, tau, alat, num_collect, n_warmup, n_measure, seed, weights, grid_points, nn, dltmove=false, random_mesh=false))]
#[allow(clippy::too_many_arguments)]
fn py_gfmc_run_ecp_complex(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    r_up_carts: PyReadonlyArray2<f64>,
    r_dn_carts: PyReadonlyArray2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    weights: PyReadonlyArray1<f64>,
    grid_points: PyReadonlyArray2<f64>,
    nn: usize,
    dltmove: bool,
    random_mesh: bool,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (_, _, j3c, j3m) = parse_jastrow_complex(jastrow_data)?;
    let cp = coulomb_potential_data;
    let sd = cp.getattr("structure_data")?;
    let positions = sd
        .getattr("positions")?
        .extract::<numpy::PyReadonlyArray2<f64>>()?
        .as_array()
        .to_owned();
    let eff_charges: Vec<f64> = cp
        .getattr("_effective_charges")?
        .call_method0("tolist")?
        .extract()?;
    let nuc_l: Vec<i64> = cp.getattr("_nucleus_index_local_part")?.extract()?;
    let e_l_p: Vec<f64> = cp
        .getattr("_exponents_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let c_l_p: Vec<f64> = cp
        .getattr("_coefficients_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let p_l_p: Vec<f64> = cp
        .getattr("_powers_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let nuc_n: Vec<i64> = cp.getattr("_nucleus_index_non_local_part")?.extract()?;
    let am: Vec<i64> = cp.getattr("_ang_mom_non_local_part")?.extract()?;
    let e_n: Vec<f64> = cp
        .getattr("_exponents_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let c_n: Vec<f64> = cp
        .getattr("_coefficients_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let p_n: Vec<f64> = cp
        .getattr("_powers_non_local_part")?
        .call_method0("tolist")?
        .extract()?;
    let pbc_owned = crate::ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "_gfmc_run_ecp_complex requires a periodic (twist) system",
        )
    })?;
    let pbc = pbc_owned.as_ref();
    let (en, eh, wh) = gfmc::gfmc_run_ecp_complex(
        &gem,
        &jdata,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &positions.view(),
        &eff_charges,
        &r_up_carts.as_array(),
        &r_dn_carts.as_array(),
        tau,
        alat,
        num_collect,
        n_warmup,
        n_measure,
        seed,
        random_mesh,
        &pbc,
        &nuc_l,
        &e_l_p,
        &c_l_p,
        &p_l_p,
        &nuc_n,
        &am,
        &e_n,
        &c_n,
        &p_n,
        weights.as_slice()?,
        &grid_points.as_array(),
        nn,
        dltmove,
    );
    Ok((en, eh, wh))
}

/// GFMC accumulated weights `G_L` (`jqmc_gfmc._compute_G_L`): sliding-window
/// product over `k = num_gfmc_collect_steps` consecutive branching weights.
/// `w_l` is `(A, x)`; returns `(A-k, x)`.
#[pyfunction]
#[pyo3(name = "compute_G_L")]
fn py_compute_g_l(
    py: Python<'_>,
    w_l: PyReadonlyArray2<f64>,
    num_gfmc_collect_steps: usize,
) -> PyResult<Py<PyAny>> {
    let out = gfmc::compute_g_l(&w_l.as_array(), num_gfmc_collect_steps);
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// GFMC systematic reconfiguration (branching): resampled source-walker indices
/// for `weights` and one uniform draw `zeta` in [0,1) (single-process).
#[pyfunction]
#[pyo3(name = "gfmc_reconfigure_indices")]
fn py_gfmc_reconfigure_indices(weights: PyReadonlyArray1<f64>, zeta: f64) -> PyResult<Vec<usize>> {
    Ok(gfmc::reconfigure_indices(weights.as_slice()?, zeta))
}

/// Debug: PBC Wigner-Seitz nearest-atom index + min-image distance for one point.
/// Mirrors jqmc `_find_nearest_index_jnp` + `|_get_min_dist_rel_R_cart_jnp|`.
#[pyfunction]
fn _nearest_atom_periodic(
    positions: PyReadonlyArray2<f64>,
    r: [f64; 3],
    cell: PyReadonlyArray2<f64>,
    n_max: i32,
) -> PyResult<(usize, f64)> {
    let cell_arr = cell.as_array();
    let inv = ewald::inv3(&cell_arr);
    Ok(mcmc::nearest_atom_periodic(
        &positions.as_array(),
        r,
        &cell_arr,
        &inv,
        n_max,
    ))
}

/// Native MCMC sweep over `num_moves` single-electron moves (molecular or PBC
/// via optional `cell`/`n_max`).
#[pyfunction]
#[pyo3(signature = (geminal_data, jastrow_data, positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, k0, k1, injected_normals, cell=None, n_max=0))]
#[allow(clippy::too_many_arguments)]
fn _mcmc_sweep(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    positions: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
    r_up: PyReadonlyArray2<f64>,
    r_dn: PyReadonlyArray2<f64>,
    geminal: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
    dt: f64,
    epsilon_as: f64,
    k0: u32,
    k1: u32,
    injected_normals: PyReadonlyArray1<f64>,
    cell: Option<PyReadonlyArray2<f64>>,
    n_max: i32,
) -> PyResult<Py<PyAny>> {
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py(jastrow_data)?;

    // Optional J3: parse orb_data + j_matrix from jastrow_three_body_data.
    let (j3_orb, j3_jm) = parse_explicit_real_j3(jastrow_data)?;
    let j3_jm_view = j3_jm.as_ref().map(|m| m.view());

    let normals = injected_normals.as_slice()?;
    let cell_arr = cell.as_ref().map(|c| c.as_array());
    let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
    let res = mcmc::mcmc_sweep(
        &gem,
        &jas,
        j3_orb.as_ref(),
        j3_jm_view.as_ref(),
        &positions.as_array(),
        charges.as_slice()?,
        &r_up.as_array(),
        &r_dn.as_array(),
        &geminal.as_array(),
        &geminal_inv.as_array(),
        dt,
        epsilon_as,
        [k0, k1],
        normals,
        normals.len(),
        cell_nmax,
        None,  // per-step dispatch: fresh determinant state (bit-exact path)
        None,  // per-step dispatch: fresh J3 state per sweep (bit-exact path)
        false, // per-step dispatch is a parity path -> full eval (no cutoff)
    );
    dict_from_sweep(py, res)
}

/// Native MCMC sweep for a COMPLEX Bloch twist geminal. Returns the post-sweep
/// walkers + complex geminal_inv + the complex streaming det-ratio state.
/// J1/J2/J3/DTN Jastrow (real, geometry-only factors).
#[pyfunction]
#[pyo3(signature = (geminal_data, jastrow_data, positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, k0, k1, injected_normals, cell=None, n_max=0))]
#[allow(clippy::too_many_arguments)]
fn _mcmc_sweep_complex(
    py: Python<'_>,
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    positions: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
    r_up: PyReadonlyArray2<f64>,
    r_dn: PyReadonlyArray2<f64>,
    geminal: PyReadonlyArray2<numpy::Complex64>,
    geminal_inv: PyReadonlyArray2<numpy::Complex64>,
    dt: f64,
    epsilon_as: f64,
    k0: u32,
    k1: u32,
    injected_normals: PyReadonlyArray1<f64>,
    cell: Option<PyReadonlyArray2<f64>>,
    n_max: i32,
) -> PyResult<Py<PyAny>> {
    let gem = complex_det::ComplexGeminal::from_py(geminal_data)?;
    let jas = data::JastrowData::from_py_complex_aware(jastrow_data)?;
    let (_j1, _j2, j3c, j3m) = parse_jastrow_complex(jastrow_data)?;
    let normals = injected_normals.as_slice()?;
    let cell_arr = cell.as_ref().map(|c| c.as_array());
    let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
    let res = mcmc::mcmc_sweep_complex(
        &gem,
        &jas,
        j3c.as_ref(),
        j3m.as_ref().map(|m| m.view()).as_ref(),
        &positions.as_array(),
        charges.as_slice()?,
        &r_up.as_array(),
        &r_dn.as_array(),
        &geminal.as_array(),
        &geminal_inv.as_array(),
        dt,
        epsilon_as,
        [k0, k1],
        normals,
        normals.len(),
        cell_nmax,
    );
    dict_from_sweep_complex(py, res)
}

/// Microbench: time n calls of orb MO eval (compute_single) and n calls of the
/// AO-only eval, on the geminal up-spin orbital, to split orb_eval into AO vs MO.
#[pyfunction]
fn _bench_orb_split(
    geminal_data: &Bound<'_, PyAny>,
    x: f64,
    y: f64,
    z: f64,
    n: usize,
) -> PyResult<(f64, f64)> {
    use std::time::Instant;
    let gem = determinant::GeminalData::from_py(geminal_data)?;
    let pos = [x, y, z];
    // MO (full compute_single)
    let mut acc = 0.0;
    let t = Instant::now();
    for _ in 0..n {
        let v = gem.orb_up.compute_single(pos);
        acc += v[0];
    }
    let mo_us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    // AO only
    let t = Instant::now();
    for _ in 0..n {
        let v = atomic_orbital::compute_aos_single(&gem.orb_up.aos, pos);
        acc += v[0];
    }
    let ao_us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    let _ = acc;
    Ok((ao_us, mo_us))
}

#[pyfunction]
fn _kato_f_l(z: f64, norm_r_r: f64) -> f64 {
    mcmc::kato_f_l(z, norm_r_r)
}

#[pyfunction]
fn _t_ratio(f_l: f64, f_prime_l: f64, dist2: f64, dt: f64) -> f64 {
    mcmc::t_ratio(f_l, f_prime_l, dist2, dt)
}

#[pyfunction]
fn _metropolis_accept(r_ratio: f64, t_ratio_val: f64, b: f64) -> bool {
    mcmc::metropolis_accept(r_ratio, t_ratio_val, b)
}

#[pyfunction]
#[pyo3(name = "compute_AS_regularization_factor_fast_update")]
fn py_compute_as_regularization_factor(
    geminal: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
) -> f64 {
    determinant::compute_as_regularization_factor(&geminal.as_array(), &geminal_inv.as_array())
}

/// Hand-rolled matrix inverse (for tests comparing against numpy/jqmc).
#[pyfunction]
fn _matrix_inverse(py: Python<'_>, a: PyReadonlyArray2<f64>) -> Py<PyAny> {
    determinant::inverse(&a.as_array())
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// Refresh of a carried real inverse (test hook, issue #127): runs the
/// production ``determinant::refresh_running_inverse_if_needed`` on a copy of
/// ``geminal_inv`` and returns ``(refreshed, inverse)``.
#[pyfunction]
fn _refresh_running_inverse_if_needed(
    py: Python<'_>,
    geminal: PyReadonlyArray2<f64>,
    geminal_inv: PyReadonlyArray2<f64>,
    threshold: f64,
) -> (bool, Py<PyAny>) {
    let mut gi = geminal_inv.as_array().to_owned();
    let refreshed =
        determinant::refresh_running_inverse_if_needed(&geminal.as_array(), &mut gi, threshold);
    (refreshed, gi.into_pyarray(py).into_any().unbind())
}

/// Complex twin of [`_refresh_running_inverse_if_needed`] (test hook, issue
/// #127): the production ``complex_det::refresh_running_inverse_if_needed_complex``.
#[pyfunction]
fn _refresh_running_inverse_if_needed_complex(
    py: Python<'_>,
    geminal: PyReadonlyArray2<numpy::Complex64>,
    geminal_inv: PyReadonlyArray2<numpy::Complex64>,
    threshold: f64,
) -> (bool, Py<PyAny>) {
    let g = geminal.as_array().to_owned();
    let mut gi = geminal_inv.as_array().to_owned();
    let refreshed =
        crate::complex_det::refresh_running_inverse_if_needed_complex(&g, &mut gi, threshold);
    (refreshed, gi.into_pyarray(py).into_any().unbind())
}

// --- Threefry RNG validation wrappers. Keys/counts as (u32, u32). ------------

#[pyfunction]
fn _tf_threefry(k0: u32, k1: u32, c0: u32, c1: u32) -> (u32, u32) {
    let o = threefry::threefry_2x32([k0, k1], [c0, c1]);
    (o[0], o[1])
}

#[pyfunction]
fn _tf_split(k0: u32, k1: u32, n: usize) -> Vec<(u32, u32)> {
    threefry::split([k0, k1], n)
        .into_iter()
        .map(|p| (p[0], p[1]))
        .collect()
}

#[pyfunction]
fn _tf_randint_i32(k0: u32, k1: u32, n: usize, minval: i32, maxval: i32) -> Vec<i32> {
    threefry::randint_i32([k0, k1], n, minval, maxval)
}

#[pyfunction]
fn _tf_randint_i64(k0: u32, k1: u32, n: usize, minval: i64, maxval: i64) -> Vec<i64> {
    threefry::randint_i64([k0, k1], n, minval, maxval)
}

#[pyfunction]
fn _tf_uniform_f64(k0: u32, k1: u32, n: usize, minval: f64, maxval: f64) -> Vec<f64> {
    threefry::uniform_f64([k0, k1], n, minval, maxval)
}

/// Debug helper: round-trip `Structure_data` through `from_py` and return the
/// parsed fields so the marshalling layer can be tested directly.
#[pyfunction]
fn _structure_data_from_py_roundtrip(
    py: Python<'_>,
    structure_data: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Vec<f64>, bool)> {
    let sd = data::StructureData::from_py(structure_data)?;
    let positions = sd.positions.into_pyarray(py).into_any().unbind();
    Ok((positions, sd.atomic_numbers, sd.pbc_flag))
}

// ============================================================================
// Native observable context.
//
// Every per-walker native observable call (`compute_local_energy_fast[_complex]`,
// the `vmc_grad_e_L_*` force/stress cores) re-parsed the WHOLE Hamiltonian from
// Python -- GeminalData / JastrowData / EwaldData (asarray on real_image_list /
// g_vectors + tolist on g_factors), `pgto_force_leaves`, all ECP channel arrays.
// That per-step `from_py` was 80% of the lithium HF force call (paid AGAIN in the
// Pulay + stress calls). The Hamiltonian is FIXED during a VMC run (only
// r_up/r_dn/geminal_inv/RT change per step) -- so, mirroring `native_driver::LeTerms`
// (the e_L benchmark chain's "extract once"), each observable entry point is split
// into a `*_Args` struct (owned extraction, `from_py`) + a `compute` (the kernel
// call). The `#[pyclass] NativeObsCtx` owns every `*_Args` the driver's fixed path
// needs (built ONCE by `build_obs_ctx`); the existing `py_*` wrappers stay thin
// (`Args::from_py(..)?.compute(..)`) so every current caller/test is bit-identical.
// ============================================================================

/// (de_L/dR, de_L/dr_up, de_L/dr_dn) -- the owned force/Pulay triple returned by the
/// `compute` methods; the py entry points wrap these into numpy.
type Grad3 = (
    numpy::ndarray::Array2<f64>,
    numpy::ndarray::Array2<f64>,
    numpy::ndarray::Array2<f64>,
);

/// ECP local + non-local channel arrays as `compute_local_energy_fast[_complex]`
/// read them (direct `extract()` for the int indices / `tolist()` for the float
/// exponents-coefficients-powers). Present only when `ecp_flag` is set.
struct ClefEcp {
    nuc_l: Vec<i64>,
    e_l: Vec<f64>,
    c_l: Vec<f64>,
    p_l: Vec<f64>,
    nuc_n: Vec<i64>,
    am: Vec<i64>,
    e_n: Vec<f64>,
    c_n: Vec<f64>,
    p_n: Vec<f64>,
}
impl ClefEcp {
    fn from_py(cp: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ClefEcp {
            nuc_l: cp.getattr("_nucleus_index_local_part")?.extract()?,
            e_l: cp
                .getattr("_exponents_local_part")?
                .call_method0("tolist")?
                .extract()?,
            c_l: cp
                .getattr("_coefficients_local_part")?
                .call_method0("tolist")?
                .extract()?,
            p_l: cp
                .getattr("_powers_local_part")?
                .call_method0("tolist")?
                .extract()?,
            nuc_n: cp.getattr("_nucleus_index_non_local_part")?.extract()?,
            am: cp.getattr("_ang_mom_non_local_part")?.extract()?,
            e_n: cp
                .getattr("_exponents_non_local_part")?
                .call_method0("tolist")?
                .extract()?,
            c_n: cp
                .getattr("_coefficients_non_local_part")?
                .call_method0("tolist")?
                .extract()?,
            p_n: cp
                .getattr("_powers_non_local_part")?
                .call_method0("tolist")?
                .extract()?,
        })
    }
}

/// ECP local + non-local channel arrays as the periodic `vmc_grad_e_L_pbc*` /
/// `vmc_grad_e_L_eps_pbc*` force/stress cores read them (`np.asarray(..).tolist()`
/// coercion for every field). Numerically identical to [`ClefEcp`]; a separate
/// reader, so the extraction of each of these wrappers stays byte-for-byte.
struct EcpChannels {
    nuc_l: Vec<i64>,
    e_l: Vec<f64>,
    c_l: Vec<f64>,
    p_l: Vec<f64>,
    nuc_n: Vec<i64>,
    am: Vec<i64>,
    e_n: Vec<f64>,
    c_n: Vec<f64>,
    p_n: Vec<f64>,
}
impl EcpChannels {
    /// `ecp_flag` gate matches the wrappers: AE hams carry empty channel vecs.
    fn from_py(cp: &Bound<'_, PyAny>, ecp_flag: bool) -> PyResult<Self> {
        if !ecp_flag {
            return Ok(EcpChannels {
                nuc_l: vec![],
                e_l: vec![],
                c_l: vec![],
                p_l: vec![],
                nuc_n: vec![],
                am: vec![],
                e_n: vec![],
                c_n: vec![],
                p_n: vec![],
            });
        }
        let np = cp.py().import("numpy")?;
        let geti = |a: &str| -> PyResult<Vec<i64>> {
            np.call_method1("asarray", (cp.getattr(a)?,))?
                .call_method0("tolist")?
                .extract()
        };
        let getf = |a: &str| -> PyResult<Vec<f64>> {
            np.call_method1("asarray", (cp.getattr(a)?,))?
                .call_method0("tolist")?
                .extract()
        };
        Ok(EcpChannels {
            nuc_l: geti("_nucleus_index_local_part")?,
            e_l: getf("_exponents_local_part")?,
            c_l: getf("_coefficients_local_part")?,
            p_l: getf("_powers_local_part")?,
            nuc_n: geti("_nucleus_index_non_local_part")?,
            am: geti("_ang_mom_non_local_part")?,
            e_n: getf("_exponents_non_local_part")?,
            c_n: getf("_coefficients_non_local_part")?,
            p_n: getf("_powers_non_local_part")?,
        })
    }
}

/// Owned complex-twist J3 context `(ComplexOrb, j_matrix, home atoms, kernel atoms)`
/// (from [`parse_j3_complex_ctx`]); rebuilds the borrowed [`gfmc_forces::J3PbcComplex`]
/// per `compute` call (cheap -- the owned pieces stay put).
struct J3ComplexOwned {
    orb: Option<complex_det::ComplexOrb>,
    jm: Option<numpy::ndarray::Array2<f64>>,
    home: Option<Vec<usize>>,
    kernel: Option<Vec<usize>>,
}
impl J3ComplexOwned {
    fn from_py(jastrow_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (orb, jm, home, kernel) = parse_j3_complex_ctx(jastrow_data)?;
        Ok(J3ComplexOwned {
            orb,
            jm,
            home,
            kernel,
        })
    }
    fn as_ref(&self) -> Option<gfmc_forces::J3PbcComplex<'_>> {
        match (&self.orb, &self.jm, &self.home, &self.kernel) {
            (Some(o), Some(m), Some(h), Some(k)) => Some(gfmc_forces::J3PbcComplex {
                orb: o,
                jm: m.view(),
                home: h,
                kernel: k,
            }),
            _ => None,
        }
    }

    /// PURE PW / B-spline complex J3 leaf `(orb, jm)` (the leaves whose cart `home` is None so
    /// [`Self::as_ref`] discards them): its kinetic SELF force/strain is served by the
    /// leaf-owning complex kernels, threaded as `batch_j3` into the continuum kinetic force/stress
    /// -- the complex twin of the real `j3_batch` channel. `None` for absent / cart-lowerable J3.
    fn batch(
        &self,
    ) -> Option<(
        &complex_det::ComplexOrb,
        numpy::ndarray::ArrayView2<'_, f64>,
    )> {
        match (&self.orb, &self.jm, &self.home) {
            (Some(o), Some(m), None) => Some((o, m.view())),
            _ => None,
        }
    }

    /// Electron drift `(dln J3/dr_up, dln J3/dr_dn)` (REAL, holomorphic `Re` LAST) for a
    /// PURE PW / B-spline complex J3 leaf -- the leaves whose cart `home` is None so
    /// [`Self::as_ref`] discards the cart J3 (its R-Pulay / eps-strain contribution is
    /// exactly 0). `None` for absent / cart-lowerable J3 (those ride the cart complex path).
    fn batch_drift(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
    ) -> Option<(numpy::ndarray::Array2<f64>, numpy::ndarray::Array2<f64>)> {
        match (&self.orb, &self.jm, &self.home) {
            (Some(o), Some(m), None) => {
                let (gu, gd, _, _) = complex_det::j3_grads_lap_complex(o, &m.view(), r_up, r_dn);
                Some((gu, gd))
            }
            _ => None,
        }
    }
}

const RT_ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ---------------------------------------------------------------------------
// Sweep-side extraction. The per-step `_mcmc_sweep[_complex]` re-parses the WHOLE
// sweep Hamiltonian (`GeminalData` / `JastrowData` + the J3 orb_data / j_matrix)
// from Python on EVERY call -- ~140-150 ms/step for a 20-water cluster with
// J1+J2+J3, dominated by the J3 j_matrix `asarray`. The Hamiltonian is FIXED during a run,
// so `SweepArgs` owns that extraction ONCE (built inside `build_obs_ctx` from the
// same fp64-coerced geminal/jastrow the observable Args read), and the ctx-taking
// sweep entries (`NativeObsCtx::mcmc_sweep[_complex]`) reuse it -- bit-identical to
// the per-call parse, only paid once.
// ---------------------------------------------------------------------------
struct SweepArgs {
    jas: data::JastrowData,
    /// Real J3 j_matrix (shared by real + complex sweeps; `None` == no active J3).
    j3_jm: Option<numpy::ndarray::Array2<f64>>,
    // real-Gamma / molecular sweep
    gem: Option<determinant::GeminalData>,
    j3_orb: Option<determinant::OrbData>,
    // complex-twist sweep
    cgem: Option<Arc<complex_det::ComplexGeminal>>,
    j3c: Option<complex_det::ComplexOrb>,
}
impl SweepArgs {
    fn from_py_with_cgem(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        complex: bool,
        cgem: Option<Arc<complex_det::ComplexGeminal>>,
    ) -> PyResult<Self> {
        let jas = data::JastrowData::from_py_complex_aware(jastrow_data)?;
        if complex {
            // Mirror `_mcmc_sweep_complex`'s parse exactly.
            let cgem = cgem.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "complex SweepArgs requires a shared ComplexGeminal",
                )
            })?;
            let (_j1, _j2, j3c, j3m) = parse_jastrow_complex(jastrow_data)?;
            if let Some(m) = j3m.as_ref() {
                check_j3_symmetric(jastrow_data.py(), &m.view())?;
            }
            Ok(SweepArgs {
                jas,
                j3_jm: j3m,
                gem: None,
                j3_orb: None,
                cgem: Some(cgem),
                j3c,
            })
        } else {
            // Mirror `_mcmc_sweep`'s parse exactly (which rejects a complex-only J3).
            jas.require_real_three_body("mcmc sweep (real)")?;
            let gem = determinant::GeminalData::from_py(geminal_data)?;
            let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
            // `jas.three_body` is `None` here only for an absent or INACTIVE (all-zero
            // `j_matrix`) J3 (a complex-only active J3 was rejected just above). An inactive
            // J3 is identically 0, so the sweep runs without it, as `JastrowData::from_py`
            // does; re-reading its orbital would hand the real `OrbData` parser a complex
            // child it drops (fewer AO rows than MO columns -> out-of-bounds panic, #157).
            let (j3_orb, j3_jm) = if j3d.is_none() || jas.three_body.is_none() {
                (None, None)
            } else {
                let orb = determinant::OrbData::from_py(&j3d.getattr("orb_data")?)?;
                let jm: numpy::PyReadonlyArray2<f64> = j3d.getattr("j_matrix")?.extract()?;
                let jm = jm.as_array().to_owned();
                check_j3_symmetric(jastrow_data.py(), &jm.view())?;
                (Some(orb), Some(jm))
            };
            Ok(SweepArgs {
                jas,
                j3_jm,
                gem: Some(gem),
                j3_orb,
                cgem: None,
                j3c: None,
            })
        }
    }
}

/// Build the `_mcmc_sweep` result dict (real path) from a [`mcmc::SweepResult`].
/// Shared by the free `_mcmc_sweep` wrapper and the ctx-taking `mcmc_sweep`.
fn dict_from_sweep(py: Python<'_>, res: mcmc::SweepResult) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("accepted", res.accepted)?;
    dict.set_item("r_up", res.r_up.into_pyarray(py))?;
    dict.set_item("r_dn", res.r_dn.into_pyarray(py))?;
    dict.set_item("geminal", res.geminal.into_pyarray(py))?;
    dict.set_item("geminal_inv", res.geminal_inv.into_pyarray(py))?;
    dict.set_item("ao_up", res.det_state.ao_up.into_pyarray(py))?;
    dict.set_item("ao_dn", res.det_state.ao_dn.into_pyarray(py))?;
    dict.set_item("paired_dn", res.det_state.paired_dn.into_pyarray(py))?;
    dict.set_item(
        "paired_up_lambda",
        res.det_state.paired_up_lambda.into_pyarray(py),
    )?;
    dict.set_item("final_key", res.final_key.to_vec())?;
    Ok(dict.into_any().unbind())
}

/// Build the `_mcmc_sweep_complex` result dict from a [`mcmc::SweepResultC`].
fn dict_from_sweep_complex(py: Python<'_>, res: mcmc::SweepResultC) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("accepted", res.accepted)?;
    dict.set_item("r_up", res.r_up.into_pyarray(py))?;
    dict.set_item("r_dn", res.r_dn.into_pyarray(py))?;
    // Emit the SWEPT complex geminal alongside its inverse so the driver can carry the
    // Sherman-Morrison streaming state forward instead of rebuilding it every measurement.
    dict.set_item("geminal", res.geminal.into_pyarray(py))?;
    dict.set_item("geminal_inv", res.geminal_inv.into_pyarray(py))?;
    dict.set_item("ao_up", res.det_state.ao_up.into_pyarray(py))?;
    dict.set_item("ao_dn", res.det_state.ao_dn.into_pyarray(py))?;
    dict.set_item("paired_dn", res.det_state.paired_dn.into_pyarray(py))?;
    dict.set_item(
        "paired_up_lambda",
        res.det_state.paired_up_lambda.into_pyarray(py),
    )?;
    dict.set_item("final_key", res.final_key.to_vec())?;
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Rayon batch helpers stack per-walker owned matrices into a
// leading-walker-axis array. `mats[w]` becomes `out[w, .., ..]`, preserving walker
// order (the `for_each_walker` harness already returns walker-ordered results, so
// the stack is bit-identical to a per-walker `np.stack`).
// ---------------------------------------------------------------------------
fn stack_axis0_f64(mats: &[numpy::ndarray::Array2<f64>]) -> numpy::ndarray::Array3<f64> {
    use numpy::ndarray::Axis;
    let nw = mats.len();
    let (r, c) = if nw == 0 {
        (0, 0)
    } else {
        (mats[0].nrows(), mats[0].ncols())
    };
    let mut out = numpy::ndarray::Array3::<f64>::zeros((nw, r, c));
    for (w, m) in mats.iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(m);
    }
    out
}

fn stack_axis0_c64(
    mats: &[numpy::ndarray::Array2<numpy::Complex64>],
) -> numpy::ndarray::Array3<numpy::Complex64> {
    use numpy::ndarray::Axis;
    let nw = mats.len();
    let (r, c) = if nw == 0 {
        (0, 0)
    } else {
        (mats[0].nrows(), mats[0].ncols())
    };
    let mut out = numpy::ndarray::Array3::<numpy::Complex64>::zeros((nw, r, c));
    for (w, m) in mats.iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(m);
    }
    out
}

/// Split a per-walker `Vec<Grad3>` into three batched `(nw, r, c)` pyarrays
/// (`dR`, `dr_up`, `dr_dn`) -- the batched twin of the per-walker `triple`.
fn grad3_batch_to_py(py: Python<'_>, results: Vec<Grad3>) -> (Py<PyAny>, Py<PyAny>, Py<PyAny>) {
    let n = results.len();
    let mut dr = Vec::with_capacity(n);
    let mut du = Vec::with_capacity(n);
    let mut dd = Vec::with_capacity(n);
    for (a, b, c) in results {
        dr.push(a);
        du.push(b);
        dd.push(c);
    }
    (
        stack_axis0_f64(&dr).into_pyarray(py).into_any().unbind(),
        stack_axis0_f64(&du).into_pyarray(py).into_any().unbind(),
        stack_axis0_f64(&dd).into_pyarray(py).into_any().unbind(),
    )
}

/// Batched `_mcmc_sweep` result dict (leading walker axis on every array).
/// Bit-identical to a per-walker loop that calls [`dict_from_sweep`] and
/// `np.stack`s the outputs in Python.
fn dict_from_sweep_batch(py: Python<'_>, results: Vec<mcmc::SweepResult>) -> PyResult<Py<PyAny>> {
    let nw = results.len();
    let mut accepted = Vec::<i64>::with_capacity(nw);
    let mut final_keys = numpy::ndarray::Array2::<u32>::zeros((nw, 2));
    let (mut ru, mut rd, mut g, mut gi) = (
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
    );
    let (mut aou, mut aod, mut pdn, mut pul) = (
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
    );
    for (w, res) in results.into_iter().enumerate() {
        accepted.push(res.accepted as i64);
        final_keys[[w, 0]] = res.final_key[0];
        final_keys[[w, 1]] = res.final_key[1];
        ru.push(res.r_up);
        rd.push(res.r_dn);
        g.push(res.geminal);
        gi.push(res.geminal_inv);
        aou.push(res.det_state.ao_up);
        aod.push(res.det_state.ao_dn);
        pdn.push(res.det_state.paired_dn);
        pul.push(res.det_state.paired_up_lambda);
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "accepted",
        numpy::ndarray::Array1::from(accepted).into_pyarray(py),
    )?;
    dict.set_item("r_up", stack_axis0_f64(&ru).into_pyarray(py))?;
    dict.set_item("r_dn", stack_axis0_f64(&rd).into_pyarray(py))?;
    dict.set_item("geminal", stack_axis0_f64(&g).into_pyarray(py))?;
    dict.set_item("geminal_inv", stack_axis0_f64(&gi).into_pyarray(py))?;
    dict.set_item("ao_up", stack_axis0_f64(&aou).into_pyarray(py))?;
    dict.set_item("ao_dn", stack_axis0_f64(&aod).into_pyarray(py))?;
    dict.set_item("paired_dn", stack_axis0_f64(&pdn).into_pyarray(py))?;
    dict.set_item("paired_up_lambda", stack_axis0_f64(&pul).into_pyarray(py))?;
    dict.set_item("final_key", final_keys.into_pyarray(py))?;
    Ok(dict.into_any().unbind())
}

/// Batched `_mcmc_sweep_complex` result dict (leading walker axis; complex geminal state).
fn dict_from_sweep_complex_batch(
    py: Python<'_>,
    results: Vec<mcmc::SweepResultC>,
) -> PyResult<Py<PyAny>> {
    let nw = results.len();
    let mut accepted = Vec::<i64>::with_capacity(nw);
    let mut final_keys = numpy::ndarray::Array2::<u32>::zeros((nw, 2));
    let (mut ru, mut rd) = (Vec::with_capacity(nw), Vec::with_capacity(nw));
    let (mut g, mut gi) = (Vec::with_capacity(nw), Vec::with_capacity(nw));
    let (mut aou, mut aod, mut pdn, mut pul) = (
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
        Vec::with_capacity(nw),
    );
    for (w, res) in results.into_iter().enumerate() {
        accepted.push(res.accepted as i64);
        final_keys[[w, 0]] = res.final_key[0];
        final_keys[[w, 1]] = res.final_key[1];
        ru.push(res.r_up);
        rd.push(res.r_dn);
        g.push(res.geminal);
        gi.push(res.geminal_inv);
        aou.push(res.det_state.ao_up);
        aod.push(res.det_state.ao_dn);
        pdn.push(res.det_state.paired_dn);
        pul.push(res.det_state.paired_up_lambda);
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "accepted",
        numpy::ndarray::Array1::from(accepted).into_pyarray(py),
    )?;
    dict.set_item("r_up", stack_axis0_f64(&ru).into_pyarray(py))?;
    dict.set_item("r_dn", stack_axis0_f64(&rd).into_pyarray(py))?;
    dict.set_item("geminal", stack_axis0_c64(&g).into_pyarray(py))?;
    dict.set_item("geminal_inv", stack_axis0_c64(&gi).into_pyarray(py))?;
    dict.set_item("ao_up", stack_axis0_c64(&aou).into_pyarray(py))?;
    dict.set_item("ao_dn", stack_axis0_c64(&aod).into_pyarray(py))?;
    dict.set_item("paired_dn", stack_axis0_c64(&pdn).into_pyarray(py))?;
    dict.set_item("paired_up_lambda", stack_axis0_c64(&pul).into_pyarray(py))?;
    dict.set_item("final_key", final_keys.into_pyarray(py))?;
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// (1) real local energy: compute_local_energy_fast  (molecular + periodic real-Gamma)
// ---------------------------------------------------------------------------
struct ClefArgs {
    gem: determinant::GeminalData,
    jdata: data::JastrowData,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    /// `Some` -> periodic (Ewald tables + cell/n_max); `None` -> molecular (bare Coulomb).
    ewald: Option<ewald::EwaldData>,
    ecp: Option<ClefEcp>,
    /// Ion Ewald structure factor (re, im), precomputed ONCE per ctx (ions fixed).
    /// Reused by every e_L's ei reciprocal sum instead of the per-step
    /// `O(n_g*n_atom)` sin/cos recompute (mirrors the native driver's `LeTerms`).
    /// Bit-identical: it is the SAME `compute_ion_structure_factor` the inline
    /// (`si=None`) path computes, only memoized. Periodic-only (`ewald.is_some()`).
    ewald_si: Option<(Vec<f64>, Vec<f64>)>,
}
impl ClefArgs {
    fn from_py(
        gem_py: &Bound<'_, PyAny>,
        jas_py: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let gem = determinant::GeminalData::from_py(gem_py)?;
        let jdata = data::JastrowData::from_py_complex_aware(jas_py)?;
        jdata.require_real_three_body("compute_local_energy_fast (real)")?;
        let sd = cp.getattr("structure_data")?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ewald = ewald::EwaldData::from_py(cp)?;
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let ecp = if ecp_flag {
            Some(ClefEcp::from_py(cp)?)
        } else {
            None
        };
        let ewald_si = ewald.as_ref().map(|ew| {
            ewald::compute_ion_structure_factor(
                &ew.g_vectors.view(),
                &eff_charges,
                &positions.view(),
            )
        });
        Ok(ClefArgs {
            gem,
            jdata,
            positions,
            eff_charges,
            ewald,
            ecp,
            ewald_si,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
        // ECP-nl mesh AO distance cutoff (`compute_mo_cutoff` / `dot_aos_multi_cutoff_*`):
        // an adaptive per-shell truncation that drops only AO values below
        // FAST_SCREEN_ABS_BOUND = 1e-12 in absolute AO units (`Shell::pz_val`, issue #100;
        // mesh points cluster on a sphere around ONE nucleus, so most of a large
        // basis is negligible). OFF for the bit-exact / per-step
        // dispatch path (equivalence tests); ON only for the FAST chunk chain
        // (`native_rng`), which is already statistically-guarded not bit-exact --
        // measured bit-identical e_L to 5 decimals on a 20-water cluster with J1+J2, halves
        // ecp.orb_compute.
        mesh_cutoff: bool,
        // A sweep-owned streaming determinant-ratio state skips the ECP-nl
        // per-e_L `init_det_ratio_state` re-evaluation. Fast chunk
        // only, molecular only (see `compute_ecp_nonlocal_parts_all_pairs`).
        det_state: Option<&crate::determinant::DetRatioState>,
        // A sweep-owned streaming J3-ratio state skips the ECP-nl
        // per-e_L `init_j3_ratio_state` rebuild (2 dense faer GEMMs). Fast chunk +
        // molecular J3 leaf only; `None` keeps the per-e_L rebuild (bit-exact path).
        j3_state: Option<&crate::jastrow_factor::J3RatioState>,
        // Chunk-carried molecular MO derivative tables at (r_up, r_dn); used by
        // the FAST kinetic only (see `run_energy_chunk_real_body`).
        det_gl_cache: Option<&crate::determinant::DetGradLapCache>,
        // Chunk-carried raw-AO J3 derivative table at (r_up, r_dn), contracted
        // against `j3_state` (both FAST chunk, molecular).
        j3_gl_cache: Option<&crate::jastrow_factor::J3GradLapCache>,
    ) -> PyResult<f64> {
        // Kinetic energy reusing the SM inverse on raw coordinates.
        // FAST chunk (det_state supplied => molecular MO-GTO, driver-gated): reuse
        // the sweep state for the det values/paired product + LEPSBAS grad/lap
        // cutoff (floor-compared). Exact path otherwise (bit-parity).
        // Both precision modes: under an fp32 `ao_eval` zone the sweep state carries the
        // zone-local fp32 values (`compute_aos_single_f32_into`, the `compute_AOs`
        // convention), as on the molecular route; the derivative tables stay fp64
        // (`ao_grad_lap`, `pz_deriv` screen).
        let fast_kin = det_state.is_some()
            && mesh_cutoff
            && self.gem.orb_up.aos.n_image == self.gem.orb_dn.aos.n_image
            && (self.gem.orb_up.aos.n_image == 1
                || (!self.gem.orb_up.aos.shells.is_empty()
                    && !self.gem.orb_dn.aos.shells.is_empty()))
            && self.gem.orb_up.pw.is_none()
            && self.gem.orb_up.bspline.is_none()
            && self.gem.orb_dn.pw.is_none()
            && self.gem.orb_dn.bspline.is_none()
            && self.gem.orb_up.mo_coefficients.is_some()
            && self.gem.orb_dn.mo_coefficients.is_some();
        let t = if fast_kin {
            crate::timed!(
                "eL.kinetic",
                wavefunction::compute_kinetic_energy_fast(
                    &self.gem,
                    &self.jdata,
                    r_up,
                    r_dn,
                    g_inv,
                    det_state.unwrap(),
                    det_gl_cache,
                    j3_gl_cache.zip(j3_state),
                )
            )
        } else {
            crate::timed!(
                "eL.kinetic",
                wavefunction::compute_kinetic_energy_with_inv(
                    &self.gem,
                    &self.jdata,
                    r_up,
                    r_dn,
                    g_inv
                )
            )
        };

        let (cell_nmax, mut v) = if let Some(ew) = &self.ewald {
            let cell_view = ew.cell.view();
            // Electron structure factor: computed ONCE per e_L, shared by the ee and
            // ei reciprocal sums (turns ei recip into O(n_g)). Same value the
            // internal `esf=None` path computes -> bit-identical (mirrors LeTerms).
            let esf_owned = crate::timed!(
                "eL.esf",
                ewald::compute_electron_structure_factor(
                    r_up,
                    r_dn,
                    &ew.g_vectors.view(),
                    &cell_view
                )
            );
            let esf = Some((esf_owned.0.as_slice(), esf_owned.1.as_slice()));
            let ee = crate::timed!(
                "eL.ewald_ee",
                ewald::compute_ewald_potential_ee(
                    r_up,
                    r_dn,
                    ew.eta,
                    &ew.real_images.view(),
                    &ew.g_vectors.view(),
                    &ew.g_factors,
                    &cell_view,
                    ew.n_max,
                    esf,
                )
            );
            let si = self
                .ewald_si
                .as_ref()
                .map(|(sr, sii)| (sr.as_slice(), sii.as_slice()));
            let ei = crate::timed!(
                "eL.ewald_ei",
                ewald::compute_ewald_potential_ei(
                    r_up,
                    r_dn,
                    ew.eta,
                    &ew.real_images.view(),
                    &ew.g_vectors.view(),
                    &ew.g_factors,
                    &cell_view,
                    ew.n_max,
                    &self.eff_charges,
                    &self.positions.view(),
                    si,
                    esf,
                )
            );
            (Some((cell_view, ew.n_max)), ee + ei + ew.ii)
        } else {
            (
                None,
                coulomb_potential::compute_bare_coulomb_potential(
                    &self.positions.view(),
                    &self.eff_charges,
                    r_up,
                    r_dn,
                ),
            )
        };
        let cell_nmax = cell_nmax.as_ref().map(|(c, n)| (c, *n));

        if let Some(e) = &self.ecp {
            v += crate::timed!(
                "eL.ecp_local",
                coulomb_potential::compute_ecp_local_parts_all_pairs(
                    &self.positions.view(),
                    &e.nuc_l,
                    &e.e_l,
                    &e.c_l,
                    &e.p_l,
                    r_up,
                    r_dn,
                    cell_nmax,
                )
            );
            let j3_data = self.jdata.three_body.as_ref().map(|j3| j3.as_ref());
            let j3_ref = j3_data.as_ref().map(|(o, m)| (*o, m));
            v += crate::timed!(
                "eL.ecp_nonlocal",
                coulomb_potential::compute_ecp_nonlocal_parts_all_pairs(
                    &self.gem,
                    &self.jdata,
                    r_up,
                    r_dn,
                    &self.positions.view(),
                    &e.nuc_n,
                    &e.am,
                    &e.e_n,
                    &e.c_n,
                    &e.p_n,
                    weights,
                    grid_points,
                    false,
                    1,
                    j3_ref,
                    cell_nmax,
                    Some(g_inv),
                    mesh_cutoff,
                    det_state,
                    j3_state,
                )
            );
        }
        Ok(t + v)
    }
}

// ---------------------------------------------------------------------------
// (2) complex-twist local energy: compute_local_energy_fast_complex
// ---------------------------------------------------------------------------
struct ClefComplexArgs {
    cgem: Arc<complex_det::ComplexGeminal>,
    jas: data::JastrowData,
    j3c: Option<complex_det::ComplexOrb>,
    j3m: Option<numpy::ndarray::Array2<f64>>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    ewald: Option<ewald::EwaldData>,
    ecp: Option<ClefEcp>,
    /// Ion structure factor, fixed for the run (as `ClefArgs::ewald_si`).
    ewald_si: Option<(Vec<f64>, Vec<f64>)>,
}
impl ClefComplexArgs {
    fn from_py(
        gem_py: &Bound<'_, PyAny>,
        jas_py: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let cgem = Arc::new(complex_det::ComplexGeminal::from_py(gem_py)?);
        Self::from_py_with_gem(gem_py, jas_py, cp, cgem)
    }

    fn from_py_with_gem(
        _gem_py: &Bound<'_, PyAny>,
        jas_py: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
        cgem: Arc<complex_det::ComplexGeminal>,
    ) -> PyResult<Self> {
        if jas_py.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "compute_local_energy_fast_complex: jastrow_data is None",
            ));
        }
        // The complex J3 is parsed by `parse_jastrow_complex` below. Do not also parse it
        // as a REAL `OrbData`: that would cast a complex MO to its real part
        // (ComplexWarning) and cannot read a complex B-spline child (issue #88).
        // J1/J2/DTN are read from `jas` (parsed above from the same attributes).
        let jas = data::JastrowData::from_py_complex_aware(jas_py)?;
        let (_, _, j3c, j3m) = parse_jastrow_complex(jas_py)?;
        let sd = cp.getattr("structure_data")?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ewald = ewald::EwaldData::from_py(cp)?;
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let ecp = if ecp_flag {
            Some(ClefEcp::from_py(cp)?)
        } else {
            None
        };
        let ewald_si = ewald.as_ref().map(|ew| {
            ewald::compute_ion_structure_factor(
                &ew.g_vectors.view(),
                &eff_charges,
                &positions.view(),
            )
        });
        Ok(ClefComplexArgs {
            cgem,
            jas,
            j3c,
            j3m,
            positions,
            eff_charges,
            ewald,
            ecp,
            ewald_si,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "complex e_L inputs: configuration, ECP mesh, inverse and seed"
    )]
    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<numpy::Complex64>,
        mesh_wdot: bool,
        det_seed: Option<&complex_det::DetRatioStateC>,
    ) -> PyResult<(f64, f64)> {
        self.compute_with_kin(
            r_up,
            r_dn,
            weights,
            grid_points,
            g_inv,
            mesh_wdot,
            det_seed,
            None,
        )
    }

    /// The local energy with the kinetic term optionally contracted from the FAST
    /// chunk's carried derivative tables (`kin_carry`, at exactly `(r_up, r_dn)`);
    /// `None` is the fresh kinetic of [`Self::compute`].
    #[allow(clippy::too_many_arguments)]
    fn compute_with_kin(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<numpy::Complex64>,
        // FAST chunk chain only (`native_rng`): contract the ECP-nl mesh det ratio
        // in AO space (`ComplexOrb::mesh_wdot_into`; FAST value screens, dropped AO
        // values < FAST_SCREEN_ABS_BOUND = 1e-12 absolute) instead of forming the
        // orbital mesh. OFF for the per-step dispatch (`e_L_complex[_batch]`,
        // `compute_local_energy_fast_complex`), as the real path's `mesh_cutoff`.
        mesh_wdot: bool,
        // FAST chunk only: the sweep's carried determinant tables for these
        // positions; with `g_inv` they seed the ECP-nl mesh state instead of a
        // fresh orbital evaluation + `G` + inverse (`mesh_fast_state_complex_from`).
        det_seed: Option<&complex_det::DetRatioStateC>,
        kin_carry: Option<&gfmc::ComplexChunkKinCarry>,
    ) -> PyResult<(f64, f64)> {
        use numpy::Complex64;
        let j3v = self.j3m.as_ref().map(|m| m.view());
        let j3 = match (&self.j3c, &j3v) {
            (Some(o), Some(v)) => Some((o, v)),
            _ => None,
        };
        let t = match kin_carry {
            Some(kc) => {
                debug_assert!(kc.positions().0.view() == *r_up && kc.positions().1.view() == *r_dn);
                timed!("eL_c.kinetic", kc.kinetic(&self.cgem, &self.jas, j3))
            }
            None => timed!(
                "eL_c.kinetic",
                complex_det::compute_kinetic_withj_complex_with_inv(
                    &self.cgem, &self.jas, j3, r_up, r_dn, g_inv,
                )
            ),
        };

        let (cell_opt, mut v_re) = if let Some(ew) = &self.ewald {
            let cell_view = ew.cell.view();
            // Electron structure factor once per e_L, shared by ee and ei (ei reciprocal
            // becomes O(n_g)); ion structure factor cached at construction. Same forms as
            // the real `ClefArgs` path.
            let esf_owned = timed!(
                "eL_c.esf",
                ewald::compute_electron_structure_factor(
                    r_up,
                    r_dn,
                    &ew.g_vectors.view(),
                    &cell_view
                )
            );
            let esf = Some((esf_owned.0.as_slice(), esf_owned.1.as_slice()));
            let ee = ewald::compute_ewald_potential_ee(
                r_up,
                r_dn,
                ew.eta,
                &ew.real_images.view(),
                &ew.g_vectors.view(),
                &ew.g_factors,
                &cell_view,
                ew.n_max,
                esf,
            );
            let si = self
                .ewald_si
                .as_ref()
                .map(|(sr, sii)| (sr.as_slice(), sii.as_slice()));
            let ei = ewald::compute_ewald_potential_ei(
                r_up,
                r_dn,
                ew.eta,
                &ew.real_images.view(),
                &ew.g_vectors.view(),
                &ew.g_factors,
                &cell_view,
                ew.n_max,
                &self.eff_charges,
                &self.positions.view(),
                si,
                esf,
            );
            (Some((cell_view, ew.n_max)), ee + ei + ew.ii)
        } else {
            (
                None,
                coulomb_potential::compute_bare_coulomb_potential(
                    &self.positions.view(),
                    &self.eff_charges,
                    r_up,
                    r_dn,
                ),
            )
        };
        if let Some(e) = &self.ecp {
            let cell_nmax = cell_opt.as_ref().map(|(c, n)| (c, *n));
            v_re += coulomb_potential::compute_ecp_local_parts_all_pairs(
                &self.positions.view(),
                &e.nuc_l,
                &e.e_l,
                &e.c_l,
                &e.p_l,
                r_up,
                r_dn,
                cell_nmax,
            );
            let cell_ref = cell_opt.as_ref().map(|(c, _)| *c);
            let nmax = self.ewald.as_ref().map(|ew| ew.n_max).unwrap_or(0);
            // Rank-1 (Sherman-Morrison + incremental Jastrow) route; the full-recompute
            // reference stays as the fallback and behind JQMC_RS_ECP_COMPLEX_REFERENCE.
            let fast = if std::env::var_os("JQMC_RS_ECP_COMPLEX_REFERENCE").is_some() {
                None
            } else {
                timed!(
                    "eL_c.ecp_nl_fast",
                    complex_ecp::compute_ecp_nonlocal_withj_complex_fast(
                        &self.cgem,
                        &self.jas,
                        self.j3c.as_ref(),
                        j3v.as_ref(),
                        r_up,
                        r_dn,
                        &self.positions.view(),
                        &e.nuc_n,
                        &e.am,
                        &e.e_n,
                        &e.c_n,
                        &e.p_n,
                        weights,
                        grid_points,
                        cell_ref.as_ref(),
                        nmax,
                        1,
                        mesh_wdot,
                        det_seed.map(|d| (d, g_inv)),
                    )
                )
            };
            let vc = match fast {
                Some(vc) => vc,
                None => complex_ecp::compute_ecp_nonlocal_withj_complex(
                    &self.cgem,
                    Some(&self.jas),
                    self.j3c.as_ref(),
                    j3v.as_ref(),
                    r_up,
                    r_dn,
                    &self.positions.view(),
                    &e.nuc_n,
                    &e.am,
                    &e.e_n,
                    &e.c_n,
                    &e.p_n,
                    weights,
                    grid_points,
                    cell_ref.as_ref(),
                    nmax,
                    1,
                ),
            };
            // Preserve VMC H Psi/Psi; fixed-phase LRDMC projection is separate.
            return Ok((t.re + v_re + vc.re, t.im + vc.im));
        }
        let e = Complex64::new(t.re + v_re, t.im);
        Ok((e.re, e.im))
    }
}

// ---------------------------------------------------------------------------
// (3) molecular HF force: vmc_grad_e_L_ae / vmc_grad_e_L_ecp
// ---------------------------------------------------------------------------
/// Molecular ECP channel arrays for the electron-axis/local force (all-`l`) + the
/// non-local wf-ratio arrays. `None` for an AE ham.
struct MolEcp {
    nuc_all: Vec<i64>,
    ang_all: Vec<i64>,
    exps_all: Vec<f64>,
    coefs_all: Vec<f64>,
    powers_all: Vec<f64>,
    max_ang: Vec<i64>,
    nuc_nl: Vec<i64>,
    ang_nl: Vec<i64>,
    exps_nl: Vec<f64>,
    coefs_nl: Vec<f64>,
    powers_nl: Vec<f64>,
}
struct MolForceArgs {
    gem: determinant::GeminalData,
    /// Stored/native orbital representation used for determinant forward fields.
    /// AO-centre derivatives continue to use `gem`, which is Cartesian-lowered.
    gem_forward: Option<determinant::GeminalData>,
    nuc_up: Vec<usize>,
    nuc_dn: Vec<usize>,
    jdata: data::JastrowData,
    j3_orb: Option<determinant::OrbData>,
    j3_jm: Option<numpy::ndarray::Array2<f64>>,
    nuc_j3: Option<Vec<usize>>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    ecp: Option<MolEcp>,
}
impl MolForceArgs {
    fn from_py(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
        fn_name: &str,
    ) -> PyResult<Self> {
        let gem = geminal_cart_from_py(geminal_data)?;
        let native_gem = determinant::GeminalData::from_py(geminal_data)?;
        // The split representation requires the same orbital rank on both sides.
        // This is guaranteed for MO-backed spherical orbitals.  A raw spherical
        // AO AGP changes rank when lowered, so it keeps the established all-cart
        // path until a rank-transform-aware cotangent bridge is added.
        let compatible_mo_rank = [
            (&native_gem.orb_up, &gem.orb_up),
            (&native_gem.orb_dn, &gem.orb_dn),
        ]
        .into_iter()
        .all(
            |(native, cart)| match (&native.mo_coefficients, &cart.mo_coefficients) {
                (Some(n), Some(c)) => n.nrows() == c.nrows(),
                _ => false,
            },
        );
        let needs_native_forward = native_gem.orb_up.aos.kind == atomic_orbital::AoKind::Sphe
            || native_gem.orb_dn.aos.kind == atomic_orbital::AoKind::Sphe;
        let gem_forward = (compatible_mo_rank && needs_native_forward).then_some(native_gem);
        let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
        let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
        let jdata = data::JastrowData::from_py(jastrow_data)?;
        // J3: cart-lower for the raw-AO force kernels (`j3_side`/`grad_j3_wrt_r`); the
        // returned nucleus index is per-cart-AO. J3 factor is basis-invariant.
        let (j3_orb, j3_jm, nuc_j3) = parse_j3_cart_force(jastrow_data)?;
        let sd = cp.getattr("structure_data")?;
        if sd.getattr("pbc_flag")?.extract::<bool>()? {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "{fn_name}: molecular only (periodic HF force is a follow-up phase)"
            )));
        }
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let ecp = if ecp_flag {
            let np = cp.py().import("numpy")?;
            let geti = |a: &str| -> PyResult<Vec<i64>> {
                np.call_method1("asarray", (cp.getattr(a)?,))?
                    .call_method0("tolist")?
                    .extract()
            };
            let getf = |a: &str| -> PyResult<Vec<f64>> {
                np.call_method1("asarray", (cp.getattr(a)?,))?
                    .call_method0("tolist")?
                    .extract()
            };
            let (_pos_nl, nuc_nl, ang_nl, exps_nl, coefs_nl, powers_nl) = parse_ecp_arrays(cp)?;
            Some(MolEcp {
                nuc_all: geti("nucleus_index")?,
                ang_all: geti("ang_moms")?,
                exps_all: getf("exponents")?,
                coefs_all: getf("coefficients")?,
                powers_all: getf("powers")?,
                max_ang: geti("max_ang_mom_plus_1")?,
                nuc_nl,
                ang_nl,
                exps_nl,
                coefs_nl,
                powers_nl,
            })
        } else {
            None
        };
        Ok(MolForceArgs {
            gem,
            gem_forward,
            nuc_up,
            nuc_dn,
            jdata,
            j3_orb,
            j3_jm,
            nuc_j3,
            positions,
            eff_charges,
            ecp,
        })
    }

    fn forward_gem(&self) -> &determinant::GeminalData {
        self.gem_forward.as_ref().unwrap_or(&self.gem)
    }

    fn compute_ae(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
    ) -> Grad3 {
        self.compute_ae_pp(
            r_up,
            r_dn,
            Some(g_inv),
            gfmc_forces::FastScreen::default(),
            false,
            None,
        )
        .0
    }

    /// [`Self::compute_ae`] plus, when `want_pulay` and the force's determinant
    /// forward is the Cartesian one, the Pulay `gfmc_grad_ln_psi_gi` triple from
    /// the same forward (`MolLnPsiArgs::compute` at the same inputs).
    /// `g_inv = None` takes a fresh LU (no Pulay); `screen` / `kin_cache` as in
    /// [`Self::compute_ecp_pp`].
    fn compute_ae_pp(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: Option<&numpy::ndarray::ArrayView2<f64>>,
        screen: gfmc_forces::FastScreen,
        want_pulay: bool,
        kin_cache: Option<&determinant::DetGradLapCache>,
    ) -> (Grad3, Option<Grad3>) {
        let j3v = self.j3_jm.as_ref().map(|m| m.view());
        let out = gfmc_forces::vmc_diag_grads_with_forward(
            &self.gem,
            self.forward_gem(),
            &self.jdata,
            self.j3_orb.as_ref(),
            j3v.as_ref(),
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            &self.nuc_up,
            &self.nuc_dn,
            self.nuc_j3.as_deref(),
            g_inv,
            screen,
            gfmc_forces::VmcShare {
                pulay: want_pulay,
                mesh: false,
            },
            self.kin_tables(screen, kin_cache),
        );
        let n_up = r_up.nrows();
        let d_rup = out.d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
        let d_rdn = out.d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
        ((out.d_r, d_rup, d_rdn), out.pulay)
    }

    fn compute_ecp(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
        screen: gfmc_forces::FastScreen,
    ) -> Grad3 {
        self.compute_ecp_pp(
            r_up,
            r_dn,
            weights,
            grid_points,
            &RT_ID,
            false,
            Some(g_inv),
            screen,
            false,
            None,
        )
        .0
    }

    /// The carried kinetic tables the forward may take: the caller's screened MO
    /// grad/Laplacian tables at the same positions (Cartesian forward only).
    fn kin_tables<'c>(
        &self,
        screen: gfmc_forces::FastScreen,
        kin_cache: Option<&'c determinant::DetGradLapCache>,
    ) -> Option<param_grad_el::MoDerivTables<'c>> {
        kin_cache
            .filter(|_| screen.det && self.gem_forward.is_none())
            .map(|c| c.mo_tables())
    }

    /// [`Self::compute_ecp`] plus the Pulay triple from the same forward (see
    /// [`Self::compute_ae_pp`]); the ECP mesh base reuses that forward too.
    ///
    /// `rt` rotates the ECP quadrature (`grid_points @ rt`) and `dltmove` selects
    /// the LRDMC DLT-move non-local split (both identity/false for VMC). `screen`
    /// names the caller e_L's AO screens; `kin_cache` its carried screened kinetic
    /// tables at `(r_up, r_dn)`. `g_inv = None` takes a fresh LU (no Pulay share).
    #[allow(clippy::too_many_arguments)]
    fn compute_ecp_pp(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        rt: &[[f64; 3]; 3],
        dltmove: bool,
        g_inv: Option<&numpy::ndarray::ArrayView2<f64>>,
        screen: gfmc_forces::FastScreen,
        want_pulay: bool,
        kin_cache: Option<&determinant::DetGradLapCache>,
    ) -> (Grad3, Option<Grad3>) {
        let e = self
            .ecp
            .as_ref()
            .expect("compute_ecp on an AE MolForceArgs");
        let j3v = self.j3_jm.as_ref().map(|m| m.view());
        let n_up = r_up.nrows();
        let out = gfmc_forces::vmc_diag_grads_with_forward(
            &self.gem,
            self.forward_gem(),
            &self.jdata,
            self.j3_orb.as_ref(),
            j3v.as_ref(),
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            &self.nuc_up,
            &self.nuc_dn,
            self.nuc_j3.as_deref(),
            g_inv,
            screen,
            gfmc_forces::VmcShare {
                pulay: want_pulay,
                mesh: true,
            },
            self.kin_tables(screen, kin_cache),
        );
        let (mut d_r, mut d_re) = (out.d_r, out.d_re);
        let pulay = out.pulay;
        let (nl_re, nl_r, _grid_stress) = gfmc_forces::ecp_nonlocal_force_mol(
            &self.gem,
            self.forward_gem(),
            &self.jdata,
            self.j3_orb.as_ref(),
            j3v.as_ref(),
            &self.positions.view(),
            r_up,
            r_dn,
            rt,
            &self.nuc_up,
            &self.nuc_dn,
            self.nuc_j3.as_deref(),
            &e.nuc_nl,
            &e.ang_nl,
            &e.exps_nl,
            &e.coefs_nl,
            &e.powers_nl,
            weights,
            grid_points,
            param_grad_ecp::NN_DEFAULT,
            dltmove,
            g_inv,
            screen,
            out.mesh,
        );
        d_re =
            d_re + &forces_ecp::grad_vlocal_wrt_r(
                &self.positions.view(),
                &e.nuc_all,
                &e.ang_all,
                &e.exps_all,
                &e.coefs_all,
                &e.powers_all,
                &e.max_ang,
                r_up,
                r_dn,
            ) + &nl_re;
        d_r =
            d_r + &forces_ecp::grad_vlocal_force(
                &self.positions.view(),
                &e.nuc_all,
                &e.ang_all,
                &e.exps_all,
                &e.coefs_all,
                &e.powers_all,
                &e.max_ang,
                r_up,
                r_dn,
            ) + &nl_r;
        let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
        let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
        ((d_r, d_rup, d_rdn), pulay)
    }
}

// ---------------------------------------------------------------------------
// (4) molecular Pulay: vmc_grad_ln_psi
// ---------------------------------------------------------------------------
struct MolLnPsiArgs {
    gem: determinant::GeminalData,
    nuc_up: Vec<usize>,
    nuc_dn: Vec<usize>,
    n_atom: usize,
    jas: Option<data::JastrowData>,
    j3o: Option<determinant::OrbData>,
    j3m: Option<numpy::ndarray::Array2<f64>>,
    nuc_j3: Option<Vec<usize>>,
}
impl MolLnPsiArgs {
    fn from_py(geminal_data: &Bound<'_, PyAny>, jastrow_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let gem = geminal_cart_from_py(geminal_data)?;
        let nuc_up = extract_nuc_cart(&geminal_data.getattr("orb_data_up_spin")?)?;
        let nuc_dn = extract_nuc_cart(&geminal_data.getattr("orb_data_dn_spin")?)?;
        let n_atom = geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("structure_data")?
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .nrows();
        let (jas, ..) = parse_jastrow_full(jastrow_data)?;
        // J3: cart-lower for the raw-AO Pulay kernel (`grad_j3_wrt_r`); nuc is per-cart-AO.
        let (j3o, j3m, nuc_j3) = parse_j3_cart_force(jastrow_data)?;
        Ok(MolLnPsiArgs {
            gem,
            nuc_up,
            nuc_dn,
            n_atom,
            jas,
            j3o,
            j3m,
            nuc_j3,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
    ) -> Grad3 {
        let j3v = self.j3m.as_ref().map(|m| m.view());
        let j1_ref = self.jas.as_ref().and_then(|j| j.one_body.as_ref());
        gfmc_forces::gfmc_grad_ln_psi_gi(
            &self.gem,
            self.jas.as_ref(),
            j1_ref,
            self.j3o.as_ref(),
            j3v.as_ref(),
            r_up,
            r_dn,
            &self.nuc_up,
            &self.nuc_dn,
            self.nuc_j3.as_deref(),
            self.n_atom,
            Some(g_inv),
        )
    }

    /// GFMC Pulay `(dln|Psi|/dR, dln|Psi|/dr)` fallback with a FRESH LU
    /// (`g_inv=None`), byte-identical to the free `_gfmc_grad_ln_Psi` wrapper.
    /// Production projection callers use `compute` with the carried inverse;
    /// this route remains the stateless/debug oracle and safety fallback.
    fn compute_fresh(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
    ) -> Grad3 {
        let j3v = self.j3m.as_ref().map(|m| m.view());
        let j1_ref = self.jas.as_ref().and_then(|j| j.one_body.as_ref());
        gfmc_forces::gfmc_grad_ln_psi_gi(
            &self.gem,
            self.jas.as_ref(),
            j1_ref,
            self.j3o.as_ref(),
            j3v.as_ref(),
            r_up,
            r_dn,
            &self.nuc_up,
            &self.nuc_dn,
            self.nuc_j3.as_deref(),
            self.n_atom,
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// (5) periodic real-Gamma HF force: vmc_grad_e_L_pbc
// ---------------------------------------------------------------------------
type J3PbcOwned = (
    determinant::OrbData,
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
    Option<forces_pbc::J3ExtraOwned>,
    bool,
);
struct PbcForceArgs {
    gem: determinant::GeminalData,
    jdata: data::JastrowData,
    j3_owned: Option<J3PbcOwned>,
    j3_batch: Option<(determinant::OrbData, numpy::ndarray::Array2<f64>)>,
    up_leaves: Vec<PgtoLeaf>,
    dn_leaves: Vec<PgtoLeaf>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    ew: ewald::EwaldData,
    ecp_flag: bool,
    ecp: EcpChannels,
}
impl PbcForceArgs {
    fn from_py(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let gem = determinant::GeminalData::from_py(geminal_data)?;
        let jdata = data::JastrowData::from_py(jastrow_data)?;
        let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
        let j3_owned = j3_pbc_ctx_owned(&j3d)?;
        let j3_batch = j3_batch_ctx_owned(&j3d)?;
        let up_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
        let dn_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
        let sd = cp.getattr("structure_data")?;
        if !sd.getattr("pbc_flag")?.extract::<bool>()? {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "vmc_grad_e_L_pbc: PBC only",
            ));
        }
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ew = ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
        })?;
        let ecp = EcpChannels::from_py(cp, ecp_flag)?;
        Ok(PbcForceArgs {
            gem,
            jdata,
            j3_owned,
            j3_batch,
            up_leaves,
            dn_leaves,
            positions,
            eff_charges,
            ew,
            ecp_flag,
            ecp,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
    ) -> PyResult<Grad3> {
        let j3 = self
            .j3_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, pw, gto_first)| gfmc_forces::J3Pbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        let jm_v = self.j3_batch.as_ref().map(|(_, m)| m.view());
        let batch = self
            .j3_batch
            .as_ref()
            .map(|(o, _)| (o, jm_v.as_ref().unwrap()));
        let e = &self.ecp;
        let g = gfmc_forces::vmc_grad_e_l_pbc(
            &self.gem,
            &self.jdata,
            j3.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            &RT_ID,
            param_grad_ecp::NN_DEFAULT,
            Some(g_inv),
            batch,
        );
        Ok((g.d_r, g.d_rup, g.d_rdn))
    }

    /// GFMC (LRDMC) periodic real-Gamma ECP HF-force at the projection final config,
    /// with the interval's FINAL mesh rotation `rt`. Drives `gfmc_grad_e_l_ecp_pbc`
    /// (the alat-discretized periodic kernel; parity-pinned by
    /// `test_gfmc_grad_e_l_ecp_pbc*`). Fresh LU internally (GFMC has no sweep inverse).
    /// ECP-only (the periodic GFMC production path is pseudopotential solids); the
    /// driver gate keeps a bare-Ewald periodic-AE force run on the per-step JAX path.
    #[allow(clippy::too_many_arguments)]
    fn compute_gfmc(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        alat: f64,
        rt: &[[f64; 3]; 3],
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        dltmove: bool,
    ) -> PyResult<Grad3> {
        let j3 = self
            .j3_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, pw, gto_first)| gfmc_forces::J3Pbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        let jm_v = self.j3_batch.as_ref().map(|(_, m)| m.view());
        let batch = self
            .j3_batch
            .as_ref()
            .map(|(o, _)| (o, jm_v.as_ref().unwrap()));
        let e = &self.ecp;
        let g = gfmc_forces::gfmc_grad_e_l_ecp_pbc(
            &self.gem,
            &self.jdata,
            j3.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            alat,
            rt,
            &self.ew.as_ref(),
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            param_grad_ecp::NN_DEFAULT,
            dltmove,
            batch,
        );
        Ok((g.d_r, g.d_rup, g.d_rdn))
    }
}

// ---------------------------------------------------------------------------
// (6) periodic real-Gamma Pulay: vmc_grad_ln_psi_pbc
// ---------------------------------------------------------------------------
struct PbcLnPsiArgs {
    gem: determinant::GeminalData,
    jdata: data::JastrowData,
    j3_owned: Option<J3PbcOwned>,
    j3_batch: Option<(determinant::OrbData, numpy::ndarray::Array2<f64>)>,
    up_leaves: Vec<PgtoLeaf>,
    dn_leaves: Vec<PgtoLeaf>,
    cell: numpy::ndarray::Array2<f64>,
    n_atom: usize,
}
impl PbcLnPsiArgs {
    fn from_py(geminal_data: &Bound<'_, PyAny>, jastrow_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let gem = determinant::GeminalData::from_py(geminal_data)?;
        let jdata = data::JastrowData::from_py(jastrow_data)?;
        let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
        let j3_owned = j3_pbc_ctx_owned(&j3d)?;
        let j3_batch = j3_batch_ctx_owned(&j3d)?;
        let up_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
        let dn_leaves = pgto_force_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
        let sd = geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("structure_data")?;
        let cell = sd
            .getattr("cell")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let n_atom = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .nrows();
        Ok(PbcLnPsiArgs {
            gem,
            jdata,
            j3_owned,
            j3_batch,
            up_leaves,
            dn_leaves,
            cell,
            n_atom,
        })
    }

    /// Add the PURE PW / B-spline J3 electron drift (leaf-owning batch evaluator) onto the
    /// J3-less base drift. The R-Pulay is untouched (the J3 dR-contribution is exactly 0
    /// for these atom-center-free leaves), and the determinant PW / B-spline drift already
    /// rides `pbc_lndet_drift`.
    fn add_batch_j3_drift(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        du: numpy::ndarray::Array2<f64>,
        dd: numpy::ndarray::Array2<f64>,
    ) -> (numpy::ndarray::Array2<f64>, numpy::ndarray::Array2<f64>) {
        match &self.j3_batch {
            None => (du, dd),
            Some((orb, jm)) => {
                let (j3u, j3d, _, _) =
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_three_body(
                        orb,
                        &jm.view(),
                        r_up,
                        r_dn,
                    );
                (du + &j3u, dd + &j3d)
            }
        }
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
    ) -> Grad3 {
        let j3 = self
            .j3_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, pw, gto_first)| gfmc_forces::J3Pbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        let g = gfmc_forces::vmc_grad_ln_psi_pbc(
            &self.gem,
            &self.jdata,
            j3.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.cell.view(),
            self.n_atom,
            r_up,
            r_dn,
            Some(g_inv),
        );
        let (du, dd) = self.add_batch_j3_drift(r_up, r_dn, g.d_rup, g.d_rdn);
        (g.d_r, du, dd)
    }

    /// GFMC periodic Pulay with a FRESH LU (`g_inv=None`); ln|Psi| is LRDMC-independent
    /// so this is the same `vmc_grad_ln_psi_pbc` kernel the VMC Pulay drives, matching
    /// the free `_gfmc_grad_ln_Psi` fresh-LU semantics.
    fn compute_fresh(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
    ) -> Grad3 {
        let j3 = self
            .j3_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, pw, gto_first)| gfmc_forces::J3Pbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        let g = gfmc_forces::vmc_grad_ln_psi_pbc(
            &self.gem,
            &self.jdata,
            j3.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.cell.view(),
            self.n_atom,
            r_up,
            r_dn,
            None,
        );
        let (du, dd) = self.add_batch_j3_drift(r_up, r_dn, g.d_rup, g.d_rdn);
        (g.d_r, du, dd)
    }
}

// ---------------------------------------------------------------------------
// (7) periodic real-Gamma HF stress: vmc_grad_e_L_eps_pbc
// ---------------------------------------------------------------------------
type J3StressOwned = (
    determinant::OrbData,
    atomic_orbital::AosGtoData,
    numpy::ndarray::Array2<f64>,
    numpy::ndarray::Array2<f64>,
    Vec<usize>,
    Vec<usize>,
    Option<forces_pbc::J3ExtraOwned>,
    bool,
);
struct PbcStressHfArgs {
    gem: determinant::GeminalData,
    jdata: data::JastrowData,
    j3s_owned: Option<J3StressOwned>,
    j3_batch: Option<(determinant::OrbData, numpy::ndarray::Array2<f64>)>,
    up_leaves: Vec<PgtoStressLeaf>,
    dn_leaves: Vec<PgtoStressLeaf>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    ew: ewald::EwaldData,
    ecp_flag: bool,
    ecp: EcpChannels,
}
impl PbcStressHfArgs {
    fn from_py(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let gem = determinant::GeminalData::from_py(geminal_data)?;
        let jdata = data::JastrowData::from_py(jastrow_data)?;
        let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
        let j3s_owned = j3_stress_ctx_owned(&j3d)?;
        let j3_batch = j3_batch_ctx_owned(&j3d)?;
        let up_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
        let dn_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
        let sd = cp.getattr("structure_data")?;
        if !sd.getattr("pbc_flag")?.extract::<bool>()? {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "vmc_grad_e_L_eps_pbc: PBC only",
            ));
        }
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ew = ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
        })?;
        let ecp = EcpChannels::from_py(cp, ecp_flag)?;
        Ok(PbcStressHfArgs {
            gem,
            jdata,
            j3s_owned,
            j3_batch,
            up_leaves,
            dn_leaves,
            positions,
            eff_charges,
            ew,
            ecp_flag,
            ecp,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
        fast_mesh: bool,
    ) -> PyResult<numpy::ndarray::Array2<f64>> {
        let j3s = self
            .j3s_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, ker, pw, gto_first)| gfmc_forces::J3StressPbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    kernel: ker,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        let jm_v = self.j3_batch.as_ref().map(|(_, m)| m.view());
        let batch = self
            .j3_batch
            .as_ref()
            .map(|(o, _)| (o, jm_v.as_ref().unwrap()));
        let e = &self.ecp;
        let s = gfmc_forces::vmc_grad_e_l_eps_pbc(
            &self.gem,
            &self.jdata,
            j3s.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            param_grad_ecp::NN_DEFAULT,
            Some(g_inv),
            fast_mesh,
            batch,
        );
        Ok(s)
    }

    /// GFMC (LRDMC) periodic HF stress `de_L^t/deps (3x3)` at the projection final
    /// config with the interval's FINAL mesh rotation `rt`. Drives
    /// `gfmc_stress_ecp_pbc_withj` for every Jastrow content: with an empty Jastrow its
    /// Jastrow terms vanish (ratio 1, zero strain / drift), and its ECP mesh is the shared
    /// rank-1 `ecp_nonlocal_stress_mesh_pbc` (the `gfmc_stress_ecp_pbc_jnone` route recomputes
    /// every mesh candidate and stays only as a parity-pinned kernel). Fresh LU
    /// internally. ECP-only (periodic GFMC = pseudopotential solids).
    #[allow(clippy::too_many_arguments)]
    fn compute_gfmc(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        alat: f64,
        rt: &[[f64; 3]; 3],
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        dltmove: bool,
    ) -> PyResult<numpy::ndarray::Array2<f64>> {
        let e = &self.ecp;
        let jm_v = self.j3_batch.as_ref().map(|(_, m)| m.view());
        let batch = self
            .j3_batch
            .as_ref()
            .map(|(o, _)| (o, jm_v.as_ref().unwrap()));
        let s = {
            let j3s =
                self.j3s_owned
                    .as_ref()
                    .map(
                        |(orb, cart, t, jm, home, ker, pw, gto_first)| gfmc_forces::J3StressPbc {
                            orb,
                            cart_j3: cart,
                            t_mol: t.view(),
                            jm: jm.view(),
                            home,
                            kernel: ker,
                            pw: pw.as_ref().map(|e| e.view()),
                            gto_first: *gto_first,
                        },
                    );
            gfmc_forces::gfmc_stress_ecp_pbc_withj(
                &self.gem,
                &self.jdata,
                &self.up_leaves,
                &self.dn_leaves,
                &self.positions.view(),
                &self.eff_charges,
                r_up,
                r_dn,
                alat,
                rt,
                &self.ew.as_ref(),
                &e.nuc_l,
                &e.e_l,
                &e.c_l,
                &e.p_l,
                &e.nuc_n,
                &e.am,
                &e.e_n,
                &e.c_n,
                &e.p_n,
                weights,
                grid_points,
                param_grad_ecp::NN_DEFAULT,
                dltmove,
                self.jdata.one_body.as_ref(),
                self.jdata.two_body.as_ref(),
                self.jdata.dtn.as_ref(),
                j3s.as_ref(),
                batch,
            )
        };
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// (8) periodic real-Gamma Pulay stress: vmc_grad_ln_psi_eps_pbc
// ---------------------------------------------------------------------------
struct PbcStressPpArgs {
    gem: determinant::GeminalData,
    jdata: data::JastrowData,
    j3s_owned: Option<J3StressOwned>,
    up_leaves: Vec<PgtoStressLeaf>,
    dn_leaves: Vec<PgtoStressLeaf>,
    positions: numpy::ndarray::Array2<f64>,
    cell: numpy::ndarray::Array2<f64>,
}
impl PbcStressPpArgs {
    fn from_py(geminal_data: &Bound<'_, PyAny>, jastrow_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let gem = determinant::GeminalData::from_py(geminal_data)?;
        let jdata = data::JastrowData::from_py(jastrow_data)?;
        let j3d = jastrow_data.getattr("jastrow_three_body_data")?;
        let j3s_owned = j3_stress_ctx_owned(&j3d)?;
        let up_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_up_spin")?)?;
        let dn_leaves = pgto_stress_leaves(&geminal_data.getattr("orb_data_dn_spin")?)?;
        let sd =
            orb_aos_leaf(&geminal_data.getattr("orb_data_up_spin")?)?.getattr("structure_data")?;
        let cell = sd
            .getattr("cell")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        Ok(PbcStressPpArgs {
            gem,
            jdata,
            j3s_owned,
            up_leaves,
            dn_leaves,
            positions,
            cell,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: &numpy::ndarray::ArrayView2<f64>,
    ) -> numpy::ndarray::Array2<f64> {
        let j3s = self
            .j3s_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, ker, pw, gto_first)| gfmc_forces::J3StressPbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    kernel: ker,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        gfmc_forces::vmc_grad_ln_psi_eps_pbc(
            &self.gem,
            &self.jdata,
            j3s.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.positions.view(),
            &self.cell.view(),
            r_up,
            r_dn,
            Some(g_inv),
        )
    }

    /// GFMC periodic Pulay stress with a FRESH LU (`g_inv=None`); Pulay stress is
    /// LRDMC-independent so this is the same `vmc_grad_ln_psi_eps_pbc` the VMC path drives.
    fn compute_fresh(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
    ) -> numpy::ndarray::Array2<f64> {
        let j3s = self
            .j3s_owned
            .as_ref()
            .map(
                |(orb, cart, t, jm, home, ker, pw, gto_first)| gfmc_forces::J3StressPbc {
                    orb,
                    cart_j3: cart,
                    t_mol: t.view(),
                    jm: jm.view(),
                    home,
                    kernel: ker,
                    pw: pw.as_ref().map(|e| e.view()),
                    gto_first: *gto_first,
                },
            );
        gfmc_forces::vmc_grad_ln_psi_eps_pbc(
            &self.gem,
            &self.jdata,
            j3s.as_ref(),
            &self.up_leaves,
            &self.dn_leaves,
            &self.positions.view(),
            &self.cell.view(),
            r_up,
            r_dn,
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// (9) complex-twist HF force: vmc_grad_e_L_pbc_complex
// ---------------------------------------------------------------------------
struct PbcForceComplexArgs {
    gem: Arc<complex_det::ComplexGeminal>,
    jdata: data::JastrowData,
    j3: J3ComplexOwned,
    nuc_up: Vec<usize>,
    nuc_dn: Vec<usize>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    ew: ewald::EwaldData,
    ecp_flag: bool,
    ecp: EcpChannels,
}
impl PbcForceComplexArgs {
    fn from_py(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let gem = Arc::new(complex_det::ComplexGeminal::from_py(geminal_data)?);
        Self::from_py_with_gem(geminal_data, jastrow_data, cp, gem)
    }

    fn from_py_with_gem(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
        gem: Arc<complex_det::ComplexGeminal>,
    ) -> PyResult<Self> {
        let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
        let j3 = J3ComplexOwned::from_py(jastrow_data)?;
        let nuc_up =
            complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_up_spin")?)?)?;
        let nuc_dn =
            complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_dn_spin")?)?)?;
        let sd = cp.getattr("structure_data")?;
        if !sd.getattr("pbc_flag")?.extract::<bool>()? {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "vmc_grad_e_L_pbc_complex: PBC only",
            ));
        }
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ew = ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
        })?;
        let ecp = EcpChannels::from_py(cp, ecp_flag)?;
        Ok(PbcForceComplexArgs {
            gem,
            jdata,
            j3,
            nuc_up,
            nuc_dn,
            positions,
            eff_charges,
            ew,
            ecp_flag,
            ecp,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: Option<&numpy::ndarray::ArrayView2<numpy::Complex64>>,
    ) -> PyResult<Grad3> {
        let j3ctx = self.j3.as_ref();
        let j3_batch = self.j3.batch();
        let batch = j3_batch.as_ref().map(|(o, m)| (*o, m));
        let e = &self.ecp;
        let g = gfmc_forces::vmc_grad_e_l_pbc_complex(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.nuc_up,
            &self.nuc_dn,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            &RT_ID,
            param_grad_ecp::NN_DEFAULT,
            g_inv,
            batch,
        );
        Ok((g.d_r, g.d_rup, g.d_rdn))
    }

    /// GFMC/LRDMC complex-twist HF force on the projection's
    /// final config with the interval's final quadrature rotation: the SAME
    /// `gfmc_grad_e_l_ecp_pbc_complex` core the free `_gfmc_grad_e_L_ecp_pbc_complex`
    /// wrapper drives (zero kernel duplication). The HF derivative is the
    /// novar=0 continuous measurement derivative; `rt` still legitimately
    /// controls ECP non-local quadrature. Holomorphic throughout; `Re` is taken
    /// once, last, inside the core at the observable boundary.
    #[allow(clippy::too_many_arguments)]
    fn compute_gfmc(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        alat: f64,
        rt: &[[f64; 3]; 3],
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        nn: usize,
        dltmove: bool,
    ) -> PyResult<Grad3> {
        let j3ctx = self.j3.as_ref();
        let j3_batch = self.j3.batch();
        let batch = j3_batch.as_ref().map(|(o, m)| (*o, m));
        let e = &self.ecp;
        let g = gfmc_forces::gfmc_grad_e_l_ecp_pbc_complex(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.nuc_up,
            &self.nuc_dn,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            alat,
            rt,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            nn,
            dltmove,
            batch,
        );
        Ok((g.d_r, g.d_rup, g.d_rdn))
    }
}

// ---------------------------------------------------------------------------
// (10) complex-twist Pulay: vmc_grad_ln_psi_pbc_complex
// ---------------------------------------------------------------------------
struct PbcLnPsiComplexArgs {
    gem: Arc<complex_det::ComplexGeminal>,
    jdata: data::JastrowData,
    j3: J3ComplexOwned,
    nuc_up: Vec<usize>,
    nuc_dn: Vec<usize>,
    n_atom: usize,
}
impl PbcLnPsiComplexArgs {
    fn from_py(geminal_data: &Bound<'_, PyAny>, jastrow_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let gem = Arc::new(complex_det::ComplexGeminal::from_py(geminal_data)?);
        Self::from_py_with_gem(geminal_data, jastrow_data, gem)
    }

    fn from_py_with_gem(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        gem: Arc<complex_det::ComplexGeminal>,
    ) -> PyResult<Self> {
        let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
        let j3 = J3ComplexOwned::from_py(jastrow_data)?;
        let nuc_up =
            complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_up_spin")?)?)?;
        let nuc_dn =
            complex_gto_home_atoms(&orb_aos_leaf(&geminal_data.getattr("orb_data_dn_spin")?)?)?;
        let n_atom = geminal_data
            .getattr("orb_data_up_spin")?
            .getattr("structure_data")?
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .nrows();
        Ok(PbcLnPsiComplexArgs {
            gem,
            jdata,
            j3,
            nuc_up,
            nuc_dn,
            n_atom,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: Option<&numpy::ndarray::ArrayView2<numpy::Complex64>>,
    ) -> Grad3 {
        let j3ctx = self.j3.as_ref();
        let g = gfmc_forces::vmc_grad_ln_psi_pbc_complex(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.nuc_up,
            &self.nuc_dn,
            self.n_atom,
            r_up,
            r_dn,
            g_inv,
        );
        match self.j3.batch_drift(r_up, r_dn) {
            None => (g.d_r, g.d_rup, g.d_rdn),
            Some((j3u, j3d)) => (g.d_r, g.d_rup + &j3u, g.d_rdn + &j3d),
        }
    }
}

// ---------------------------------------------------------------------------
// (11) complex-twist HF stress: vmc_grad_e_L_eps_pbc_complex
// ---------------------------------------------------------------------------
struct PbcStressHfComplexArgs {
    gem: Arc<complex_det::ComplexGeminal>,
    jdata: data::JastrowData,
    j3: J3ComplexOwned,
    home_up: Vec<usize>,
    kernel_up: Vec<usize>,
    home_dn: Vec<usize>,
    kernel_dn: Vec<usize>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    ew: ewald::EwaldData,
    ecp_flag: bool,
    ecp: EcpChannels,
}
impl PbcStressHfComplexArgs {
    fn from_py(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let gem = Arc::new(complex_det::ComplexGeminal::from_py(geminal_data)?);
        Self::from_py_with_gem(geminal_data, jastrow_data, cp, gem)
    }

    fn from_py_with_gem(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
        gem: Arc<complex_det::ComplexGeminal>,
    ) -> PyResult<Self> {
        let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
        let (home_up, kernel_up, home_dn, kernel_dn, _cell0, _pos0) =
            complex_stress_args(geminal_data)?;
        let j3 = J3ComplexOwned::from_py(jastrow_data)?;
        let sd = cp.getattr("structure_data")?;
        if !sd.getattr("pbc_flag")?.extract::<bool>()? {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "vmc_grad_e_L_eps_pbc_complex: PBC only",
            ));
        }
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ew = ewald::EwaldData::from_py(cp)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("expected periodic coulomb_potential_data")
        })?;
        let ecp = EcpChannels::from_py(cp, ecp_flag)?;
        Ok(PbcStressHfComplexArgs {
            gem,
            jdata,
            j3,
            home_up,
            kernel_up,
            home_dn,
            kernel_dn,
            positions,
            eff_charges,
            ew,
            ecp_flag,
            ecp,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        g_inv: Option<&numpy::ndarray::ArrayView2<numpy::Complex64>>,
        fast_mesh: bool,
    ) -> PyResult<numpy::ndarray::Array2<f64>> {
        let j3ctx = self.j3.as_ref();
        let j3_batch = self.j3.batch();
        let batch = j3_batch.as_ref().map(|(o, m)| (*o, m));
        let e = &self.ecp;
        Ok(gfmc_forces::vmc_grad_e_l_eps_pbc_complex(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.home_up,
            &self.kernel_up,
            &self.home_dn,
            &self.kernel_dn,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            &RT_ID,
            param_grad_ecp::NN_DEFAULT,
            g_inv,
            fast_mesh,
            batch,
        ))
    }

    /// GFMC/LRDMC complex-twist HF stress (3x3, real) on the
    /// projection's final config with the interval's final mesh rotation: the SAME
    /// `gfmc_stress_ecp_pbc_complex` core the free `_gfmc_stress_ecp_pbc_complex` wrapper
    /// drives. `dltmove` selects the DLA split inside the complex ECP non-local strain
    /// chain (det-only amplitude clamp + det-only strain-Pulay/drift), exactly as the real
    /// `PbcStressHfArgs::compute_gfmc` threads it into `gfmc_stress_ecp_pbc_withj`: a
    /// `dltmove` run must NOT be served the t-move stress.
    #[allow(clippy::too_many_arguments)]
    fn compute_gfmc(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        alat: f64,
        rt: &[[f64; 3]; 3],
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        nn: usize,
        dltmove: bool,
    ) -> PyResult<numpy::ndarray::Array2<f64>> {
        let j3ctx = self.j3.as_ref();
        let j3_batch = self.j3.batch();
        let batch = j3_batch.as_ref().map(|(o, m)| (*o, m));
        let e = &self.ecp;
        Ok(gfmc_forces::gfmc_stress_ecp_pbc_complex(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.home_up,
            &self.kernel_up,
            &self.home_dn,
            &self.kernel_dn,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            alat,
            rt,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            nn,
            dltmove,
            batch,
        ))
    }

    /// [`Self::compute_gfmc`] together with the Pulay stress at the inverse `pp_gi` (`None` =
    /// the HF kernel's fresh base inverse), as `PbcStressPpComplexArgs::compute` gives it,
    /// sharing the base forward, base Jastrow strain and final base-table pass
    /// (`gfmc_stress_ecp_pbc_complex_with_pp`).
    #[allow(clippy::too_many_arguments)]
    fn compute_gfmc_with_pp(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        alat: f64,
        rt: &[[f64; 3]; 3],
        weights: &[f64],
        grid_points: &numpy::ndarray::ArrayView2<f64>,
        nn: usize,
        dltmove: bool,
        pp_gi: Option<&numpy::ndarray::ArrayView2<numpy::Complex64>>,
    ) -> PyResult<(numpy::ndarray::Array2<f64>, numpy::ndarray::Array2<f64>)> {
        let j3ctx = self.j3.as_ref();
        let j3_batch = self.j3.batch();
        let batch = j3_batch.as_ref().map(|(o, m)| (*o, m));
        let e = &self.ecp;
        Ok(gfmc_forces::gfmc_stress_ecp_pbc_complex_with_pp(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.home_up,
            &self.kernel_up,
            &self.home_dn,
            &self.kernel_dn,
            &self.positions.view(),
            &self.eff_charges,
            r_up,
            r_dn,
            alat,
            rt,
            &self.ew.as_ref(),
            self.ecp_flag,
            &e.nuc_l,
            &e.e_l,
            &e.c_l,
            &e.p_l,
            &e.nuc_n,
            &e.am,
            &e.e_n,
            &e.c_n,
            &e.p_n,
            weights,
            grid_points,
            nn,
            dltmove,
            batch,
            pp_gi,
        ))
    }
}

// ---------------------------------------------------------------------------
// (12) complex-twist Pulay stress: vmc_grad_ln_psi_eps_pbc_complex
// ---------------------------------------------------------------------------
struct PbcStressPpComplexArgs {
    gem: Arc<complex_det::ComplexGeminal>,
    jdata: data::JastrowData,
    j3: J3ComplexOwned,
    home_up: Vec<usize>,
    kernel_up: Vec<usize>,
    home_dn: Vec<usize>,
    kernel_dn: Vec<usize>,
    cell0: numpy::ndarray::Array2<f64>,
    pos0: numpy::ndarray::Array2<f64>,
}
impl PbcStressPpComplexArgs {
    fn from_py(geminal_data: &Bound<'_, PyAny>, jastrow_data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let gem = Arc::new(complex_det::ComplexGeminal::from_py(geminal_data)?);
        Self::from_py_with_gem(geminal_data, jastrow_data, gem)
    }

    fn from_py_with_gem(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        gem: Arc<complex_det::ComplexGeminal>,
    ) -> PyResult<Self> {
        let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
        let (home_up, kernel_up, home_dn, kernel_dn, cell0, pos0) =
            complex_stress_args(geminal_data)?;
        let j3 = J3ComplexOwned::from_py(jastrow_data)?;
        Ok(PbcStressPpComplexArgs {
            gem,
            jdata,
            j3,
            home_up,
            kernel_up,
            home_dn,
            kernel_dn,
            cell0,
            pos0,
        })
    }

    fn compute(
        &self,
        r_up: &numpy::ndarray::ArrayView2<f64>,
        r_dn: &numpy::ndarray::ArrayView2<f64>,
        g_inv: Option<&numpy::ndarray::ArrayView2<numpy::Complex64>>,
    ) -> numpy::ndarray::Array2<f64> {
        let j3ctx = self.j3.as_ref();
        gfmc_forces::vmc_grad_ln_psi_eps_pbc_complex(
            &self.gem,
            &self.jdata,
            j3ctx.as_ref(),
            &self.home_up,
            &self.kernel_up,
            &self.home_dn,
            &self.kernel_dn,
            &self.pos0.view(),
            &self.cell0.view(),
            r_up,
            r_dn,
            g_inv,
        )
    }
}

// ---------------------------------------------------------------------------
// GFMC projection-interval chunk arguments: the once-parsed Hamiltonian
// view the `run_gfmc_projection_{t,n}` ctx methods drive the verified
// `gfmc::gfmc_lrdmc_{ae,ecp}` kernels with (zero kernel duplication -- the SAME
// kernels the `_gfmc_run_*` reference chains and the `_gfmc_lrdmc_*` parity
// wrappers call). Mirrors `gfmc_ecp_parse` + `EwaldData::from_py`, extracted
// once per run instead of per interval. Real and complex-twist paths store their
// respective geminal representation in the same argument owner.
// ---------------------------------------------------------------------------
/// ECP channels (local + non-local) tuple: (nuc_l, e_l, c_l, p_l, nuc_n, am, e_n, c_n, p_n)
/// -- the `gfmc_ecp_parse` layout, shared by the value e_L kernels and the
/// molecular-ECP HF-force fold-in.
type GfmcEcpTuple = (
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<i64>,
    Vec<i64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
);
struct GfmcArgs {
    /// Real geminal (real-Gamma / molecular); `None` on the complex-twist path.
    gem: Option<determinant::GeminalData>,
    /// Complex-twist geminal (k != Gamma); `None` on the real path.
    cgem: Option<Arc<complex_det::ComplexGeminal>>,
    jdata: data::JastrowData,
    j3_orb: Option<determinant::OrbData>,
    j3_jm: Option<numpy::ndarray::Array2<f64>>,
    /// Complex-twist J3 (sphe complex AO orbital + real j_matrix); `None` on real path.
    j3c: Option<complex_det::ComplexOrb>,
    j3cm: Option<numpy::ndarray::Array2<f64>>,
    positions: numpy::ndarray::Array2<f64>,
    eff_charges: Vec<f64>,
    /// ECP channels (local + non-local), `None` for all-electron.
    ecp: Option<GfmcEcpTuple>,
    /// `Some` -> periodic real-Gamma (Ewald tables); `None` -> molecular. The complex
    /// twist always carries Ewald (periodic-only), verified at build.
    ewald: Option<ewald::EwaldData>,
}
impl GfmcArgs {
    fn from_py_with_cgem(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
        complex: bool,
        shared_cgem: Option<Arc<complex_det::ComplexGeminal>>,
    ) -> PyResult<Self> {
        let jdata = data::JastrowData::from_py_complex_aware(jastrow_data)?;
        // Real geminal + real J3 (real path only). Complex-twist parses the complex
        // geminal / complex J3 instead (the complex e_L kernels are complex-native).
        let (gem, j3_orb, j3_jm, cgem, j3c, j3cm) = if complex {
            let cgem = shared_cgem.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "complex GfmcArgs requires a shared ComplexGeminal",
                )
            })?;
            let (_, _, j3c, j3cm) = parse_jastrow_complex(jastrow_data)?;
            (None, None, None, Some(cgem), j3c, j3cm)
        } else {
            jdata.require_real_three_body("gfmc (real)")?;
            let gem = determinant::GeminalData::from_py(geminal_data)?;
            // J3 parse mirrors `gfmc_ecp_parse` exactly (incl. the all-zero j_matrix skip).
            let (j3_orb, j3_jm) =
                gfmc_real_j3_parse(&jastrow_data.getattr("jastrow_three_body_data")?)?;
            (Some(gem), j3_orb, j3_jm, None, None, None)
        };
        let sd = cp.getattr("structure_data")?;
        let positions = sd
            .getattr("positions")?
            .extract::<numpy::PyReadonlyArray2<f64>>()?
            .as_array()
            .to_owned();
        let eff_charges: Vec<f64> = cp
            .getattr("_effective_charges")?
            .call_method0("tolist")?
            .extract()?;
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let ecp = if ecp_flag {
            Some((
                cp.getattr("_nucleus_index_local_part")?.extract()?,
                cp.getattr("_exponents_local_part")?
                    .call_method0("tolist")?
                    .extract()?,
                cp.getattr("_coefficients_local_part")?
                    .call_method0("tolist")?
                    .extract()?,
                cp.getattr("_powers_local_part")?
                    .call_method0("tolist")?
                    .extract()?,
                cp.getattr("_nucleus_index_non_local_part")?.extract()?,
                cp.getattr("_ang_mom_non_local_part")?.extract()?,
                cp.getattr("_exponents_non_local_part")?
                    .call_method0("tolist")?
                    .extract()?,
                cp.getattr("_coefficients_non_local_part")?
                    .call_method0("tolist")?
                    .extract()?,
                cp.getattr("_powers_non_local_part")?
                    .call_method0("tolist")?
                    .extract()?,
            ))
        } else {
            None
        };
        let ewald = ewald::EwaldData::from_py(cp)?;
        if complex && ewald.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "build_obs_ctx(want_gfmc, complex): complex-twist GFMC requires a periodic \
                 (Ewald) system",
            ));
        }
        Ok(GfmcArgs {
            gem,
            cgem,
            jdata,
            j3_orb,
            j3_jm,
            j3c,
            j3cm,
            positions,
            eff_charges,
            ecp,
            ewald,
        })
    }
}

// ---------------------------------------------------------------------------
// `ParamDerivArgs` for `want_param_deriv`:
// the once-parsed Geminal / full Jastrow (+ J1/J2/J3/DTN leaves) / ECP channels the
// c-axis optimization-observable kernels (O_k = dlnPsi/dc, dE_L/dc) re-parsed PER
// walker per family on the free-kernel `_rs_native_c_param_grads` path.
// Extracted ONCE at ctx build; the per-step ctx methods below take only
// (r_up, r_dn[, weights, grid_points]) and call the SAME inner kernels with the SAME
// owned data -> byte-identical to the free path (pinned by the A/B dispatch rows).
// Parse mirrors the free wrappers exactly: `parse_jastrow_full` (jas + j3 orb/jm,
// shared by every e_L kinetic kernel), the J1/J2 leaf structs (O_k Jastrow blocks),
// the DTN `DtnOkData` (O_k + kinetic + ECP DTN), and `parse_ecp_arrays`/`parse_ecp_cell`
// (the dV_ecp_nl/dc mesh add). Real path only (the gate excludes complex/MPC/basis).
// ---------------------------------------------------------------------------
struct EcpPdChannels {
    pos: numpy::ndarray::Array2<f64>,
    nuc: Vec<i64>,
    angmom: Vec<i64>,
    exps: Vec<f64>,
    coefs: Vec<f64>,
    powers: Vec<f64>,
    cell: Option<numpy::ndarray::Array2<f64>>,
    n_max: i32,
}
/// Geminal AO radial (GTO/pGTO) basis-opt param-grad args: the per-shell prim->shell
/// map + shell count for each spin, plus the home cell for periodic wrapping. Built
/// only when both spin leaves carry a GTO `radial` (PW / B-spline / composite carry
/// none), so a non-GTO geminal simply has `basis: None` and the driver's basis branch
/// never fires (matching the reducer's `_gto_leaves_of` gate).
struct BasisPdArgs {
    a2s_up: Vec<usize>,
    nsh_up: usize,
    a2s_dn: Vec<usize>,
    nsh_dn: usize,
    cell: Option<numpy::ndarray::Array2<f64>>,
}

/// Three-body-Jastrow ORBITAL AO radial (GTO/pGTO) basis-opt param-grad args
/// (`opt_J3_basis_exp` / `opt_J3_basis_coeff`): the prim->shell map + shell count of
/// the J3 orbital's GTO leaf. ONE block (not per-spin): the J3 orbital is shared
/// across both spins. `None` when the J3 leaf carries no GTO radial (pure PW /
/// B-spline) or when there is no three-body Jastrow at all -- mirroring the Python
/// publisher `Jastrow_data.collect_param_grads`, which emits `j3_basis_*` only for
/// an `AOs_GTO_base_data` leaf.
struct J3BasisPdArgs {
    a2s: Vec<usize>,
    nsh: usize,
}

struct ParamDerivArgs {
    gem: determinant::GeminalData,
    jas: Option<data::JastrowData>,
    j1: Option<data::JastrowOneBodyData>,
    j2: Option<data::JastrowTwoBodyData>,
    j3_orb: Option<determinant::OrbData>,
    j3_jm: Option<numpy::ndarray::Array2<f64>>,
    /// Inactive J3 whose orbital the real parser cannot represent (#157); see `j3_pd_orb`.
    j3_real_unrepresentable: bool,
    dtn: Option<param_grad_dtn::DtnOkData>,
    ecp: Option<EcpPdChannels>,
    basis: Option<BasisPdArgs>,
    j3_basis: Option<J3BasisPdArgs>,
}

/// Complex-twist DTN optimizer arguments.  Unlike [`ParamDerivArgs`], this keeps the
/// complex geminal behind the SAME [`Arc`] used by the sweep/local-energy/force
/// contexts. A complex B-spline geminal owns grid-sized real/imag coefficient storage;
/// parsing it again for every DTN walker would create a grid-sized transient allocation
/// whose high-water mark mimalloc retains.
struct ComplexDtnPdArgs {
    gem: Arc<complex_det::ComplexGeminal>,
    jas: data::JastrowData,
    dtn: Option<param_grad_dtn::DtnOkData>,
    j3c: Option<complex_det::ComplexOrb>,
    j3m: Option<numpy::ndarray::Array2<f64>>,
    ecp: Option<EcpPdChannels>,
}

impl ComplexDtnPdArgs {
    fn from_py_with_gem(
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
        gem: Arc<complex_det::ComplexGeminal>,
    ) -> PyResult<Self> {
        let jas = data::JastrowData::from_py_without_three_body(jastrow_data).map_err(|err| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "complex DTN context: failed to parse JastrowData: {err}"
            ))
        })?;
        let dtna = jastrow_data.getattr("jastrow_dtn_data")?;
        let dtn = if dtna.is_none() {
            None
        } else {
            Some(param_grad_dtn::DtnOkData::from_py(&dtna).map_err(|err| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "complex DTN context: failed to parse DTN data: {err}"
                ))
            })?)
        };
        let (j3c, j3m) = parse_j3_complex(jastrow_data).map_err(|err| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "complex DTN context: failed to parse J3 data: {err}"
            ))
        })?;
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let ecp = if ecp_flag {
            let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp).map_err(|err| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "complex DTN context: failed to parse ECP arrays: {err}"
                ))
            })?;
            let (cell, n_max) = parse_ecp_cell(cp).map_err(|err| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "complex DTN context: failed to parse ECP cell: {err}"
                ))
            })?;
            Some(EcpPdChannels {
                pos,
                nuc,
                angmom,
                exps,
                coefs,
                powers,
                cell,
                n_max,
            })
        } else {
            None
        };
        Ok(Self {
            gem,
            jas,
            dtn,
            j3c,
            j3m,
            ecp,
        })
    }
}
impl ParamDerivArgs {
    fn from_py(
        geminal_data: &Bound<'_, PyAny>,
        jastrow_data: &Bound<'_, PyAny>,
        cp: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let gem = determinant::GeminalData::from_py(geminal_data)?;
        // jas (full) + j3 orb / j_matrix -- identical parse the e_L kinetic wrappers use.
        // Exception (#157): a J3 whose orbital the real `OrbData` parser cannot represent
        // (complex plane-wave / B-spline child or complex MO). `JastrowData::from_py`
        // rejects it when ACTIVE, so past that call it is INACTIVE (all-zero `j_matrix`):
        // identically 0, so every value / kinetic / MO / lambda consumer runs without it,
        // while its own `j_matrix` derivative (nonzero) is refused by `j3_pd_orb`.
        let j3_real_unrepresentable = !jastrow_data.is_none() && {
            let j3a = jastrow_data.getattr("jastrow_three_body_data")?;
            !j3a.is_none() && !data::j3_orbital_real_representable(&j3a.getattr("orb_data")?)?
        };
        let (jas, j3_orb, j3_jm) = if j3_real_unrepresentable {
            (Some(data::JastrowData::from_py(jastrow_data)?), None, None)
        } else {
            parse_jastrow_full(jastrow_data)?
        };
        let (j1, j2, dtn) = if jastrow_data.is_none() {
            (None, None, None)
        } else {
            let j1a = jastrow_data.getattr("jastrow_one_body_data")?;
            let j1 = if j1a.is_none() {
                None
            } else {
                Some(data::JastrowOneBodyData::from_py(&j1a)?)
            };
            let j2a = jastrow_data.getattr("jastrow_two_body_data")?;
            let j2 = if j2a.is_none() {
                None
            } else {
                Some(data::JastrowTwoBodyData::from_py(&j2a)?)
            };
            let dtna = jastrow_data.getattr("jastrow_dtn_data")?;
            let dtn = if dtna.is_none() {
                None
            } else {
                Some(param_grad_dtn::DtnOkData::from_py(&dtna)?)
            };
            (j1, j2, dtn)
        };
        let ecp_flag: bool = cp.getattr("ecp_flag")?.extract()?;
        let ecp = if ecp_flag {
            let (pos, nuc, angmom, exps, coefs, powers) = parse_ecp_arrays(cp)?;
            let (cell, n_max) = parse_ecp_cell(cp)?;
            Some(EcpPdChannels {
                pos,
                nuc,
                angmom,
                exps,
                coefs,
                powers,
                cell,
                n_max,
            })
        } else {
            None
        };
        // Geminal AO radial (GTO) basis-opt args -- only if BOTH spin leaves carry a
        // GTO `radial` (mirrors the reducer's `_gto_leaves_of` guard). PW / B-spline /
        // composite leaves -> `basis: None`, so the driver's basis branch is inert.
        let up_py = geminal_data.getattr("orb_data_up_spin")?;
        let dn_py = geminal_data.getattr("orb_data_dn_spin")?;
        let basis = if ensure_gto_radial(&up_py).is_ok() && ensure_gto_radial(&dn_py).is_ok() {
            let (a2s_up, nsh_up) = radial_map(&up_py)?;
            let (a2s_dn, nsh_dn) = radial_map(&dn_py)?;
            let cell = pbc_cell(&up_py)?;
            Some(BasisPdArgs {
                a2s_up,
                nsh_up,
                a2s_dn,
                nsh_dn,
                cell,
            })
        } else {
            None
        };
        // J3 orbital AO radial basis-opt args -- built iff a three-body Jastrow is
        // present AND its orbital resolves to a GTO/pGTO leaf (`radial_map` walks the
        // MO wrapper / composite children exactly as the geminal path does). A pure
        // PW / B-spline J3 orbital leaves this `None`, so the driver's j3-basis branch
        // is inert -- matching `collect_param_grads`, which publishes no `j3_basis_*`
        // block for such a leaf.
        let j3_basis = if jastrow_data.is_none() {
            None
        } else {
            let j3a = jastrow_data.getattr("jastrow_three_body_data")?;
            if j3a.is_none() {
                None
            } else {
                let orb_py = j3a.getattr("orb_data")?;
                if ensure_gto_radial(&orb_py).is_ok() {
                    let (a2s, nsh) = radial_map(&orb_py)?;
                    Some(J3BasisPdArgs { a2s, nsh })
                } else {
                    None
                }
            }
        };
        Ok(ParamDerivArgs {
            gem,
            jas,
            j1,
            j2,
            j3_orb,
            j3_jm,
            j3_real_unrepresentable,
            dtn,
            ecp,
            basis,
            j3_basis,
        })
    }

    /// The J3 orbital for a J3-parameter-derivative method (`pd_*_j3*`).
    ///
    /// Raises `NotImplementedError` for an inactive J3 whose orbital only the complex path
    /// can evaluate (#157): its `j_matrix` / basis derivatives are nonzero, so neither a
    /// truncated real orbital nor an omitted block is correct. Use the JAX route
    /// (`JQMC_FLAG_RUST_ON_CPU=0`), as for the active case (#93).
    fn j3_pd_orb(&self) -> PyResult<&determinant::OrbData> {
        if self.j3_real_unrepresentable {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "native real parameter derivatives: the J3 orbital (complex plane-wave / \
                 B-spline basis or a complex MO) is supported only by the complex native path, \
                 but the determinant selects the real path; its j_matrix derivative is nonzero \
                 even for an all-zero j_matrix. Disable flag_rust_on_CPU to use the JAX \
                 param-deriv path for this family.",
            ));
        }
        self.j3_orb
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))
    }
}

// ---------------------------------------------------------------------------
// NativeObsCtx: owns every *_Args the driver's (fixed-per-run) path needs. Built
// ONCE per run() by `build_obs_ctx`; the driver threads it through the per-step
// e_L / force / Pulay / stress calls, so the whole Hamiltonian `from_py` is paid
// once instead of per walker per step.
// ---------------------------------------------------------------------------

/// Precision-zone wire schema shared with `jqmc._precision`. Python remains the
/// source of truth for the selected dtype; Rust validates that every known zone
/// is present before storing the immutable per-context snapshot.
const PRECISION_ZONES: [&str; 23] = [
    "ao_eval",
    "ao_bloch_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "tiling_eval",
    "tiling_grad",
    "tiling_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "jastrow_grad_lap_carry",
    "jastrow_ratio",
    "det_eval",
    "det_grad_lap",
    "det_ratio",
    "coulomb",
    "ewald",
    "wf_eval",
    "wf_kinetic",
    "wf_ratio",
    "local_energy",
    "swct",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RealDType {
    F32,
    F64,
}

impl RealDType {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "float32" => Ok(Self::F32),
            "float64" => Ok(Self::F64),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid native precision dtype {other:?}; expected 'float32' or 'float64'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrecisionPolicy {
    dtypes: [RealDType; PRECISION_ZONES.len()],
}

impl PrecisionPolicy {
    fn full() -> Self {
        Self {
            dtypes: [RealDType::F64; PRECISION_ZONES.len()],
        }
    }

    /// Parse a complete precision-zone policy dict. There is deliberately no
    /// `None` -> full default here: a caller that omits the policy must fail
    /// loudly instead of silently running fp64 kernels in a mixed run.
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mapping = value.downcast::<pyo3::types::PyDict>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "precision_policy must be a dict containing all precision zones",
            )
        })?;
        if mapping.len() != PRECISION_ZONES.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "precision_policy must contain exactly {} zones, got {}",
                PRECISION_ZONES.len(),
                mapping.len()
            )));
        }
        let mut out = [RealDType::F64; PRECISION_ZONES.len()];
        for (key, value) in mapping.iter() {
            let key: String = key.extract()?;
            let dtype: String = value.extract()?;
            let index = PRECISION_ZONES
                .iter()
                .position(|known| *known == key)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown native precision zone {key:?}"
                    ))
                })?;
            out[index] = RealDType::parse(&dtype)?;
        }
        // The exact-length + no-unknown checks imply completeness for a Python
        // dict (which cannot carry duplicate keys).
        Ok(Self { dtypes: out })
    }

    fn to_pydict<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let out = pyo3::types::PyDict::new(py);
        for (zone, dtype) in PRECISION_ZONES.iter().zip(self.dtypes) {
            out.set_item(*zone, dtype.as_str())?;
        }
        Ok(out)
    }

    #[inline]
    pub(crate) fn real(&self, zone: &str) -> RealDType {
        let index = PRECISION_ZONES
            .iter()
            .position(|known| *known == zone)
            .unwrap_or_else(|| panic!("unknown native precision zone {zone:?}"));
        self.dtypes[index]
    }

    #[inline]
    pub(crate) fn is_f32(&self, zone: &str) -> bool {
        self.real(zone) == RealDType::F32
    }

    fn bitmask(&self) -> u32 {
        self.dtypes
            .iter()
            .enumerate()
            .fold(0_u32, |mask, (i, dtype)| {
                mask | (u32::from(*dtype == RealDType::F32) << i)
            })
    }

    fn from_bitmask(mask: u32) -> Self {
        let mut dtypes = [RealDType::F64; PRECISION_ZONES.len()];
        for (i, dtype) in dtypes.iter_mut().enumerate() {
            if mask & (1_u32 << i) != 0 {
                *dtype = RealDType::F32;
            }
        }
        Self { dtypes }
    }
}

// Standalone PyO3 kernels have no NativeObsCtx owner.  Their explicit session
// policy is process-local and configured once by the parity harness / CLI run
// head.  Production contexts do not read this atomic after construction.
static STANDALONE_PRECISION_MASK: AtomicU32 = AtomicU32::new(0);

thread_local! {
    // NativeObsCtx construction is single-threaded. Nested from_py calls copy
    // this policy into the numerical data they create; the guard below restores
    // the previous value, so contexts with different policies coexist safely.
    static CONSTRUCTION_PRECISION: Cell<PrecisionPolicy> = Cell::new(PrecisionPolicy::full());
}

pub(crate) fn standalone_precision_policy() -> PrecisionPolicy {
    PrecisionPolicy::from_bitmask(STANDALONE_PRECISION_MASK.load(Ordering::Relaxed))
}

pub(crate) fn construction_precision_policy() -> PrecisionPolicy {
    CONSTRUCTION_PRECISION.with(Cell::get)
}

struct ConstructionPrecisionGuard {
    previous: PrecisionPolicy,
}

impl ConstructionPrecisionGuard {
    fn enter(policy: PrecisionPolicy) -> Self {
        let previous = CONSTRUCTION_PRECISION.with(|slot| slot.replace(policy));
        Self { previous }
    }
}

impl Drop for ConstructionPrecisionGuard {
    fn drop(&mut self) {
        CONSTRUCTION_PRECISION.with(|slot| slot.set(self.previous));
    }
}

/// Scoped construction policy for the context-less `jqmc_rs` free kernels.
///
/// A free kernel parses its Python data on the calling thread, and the parsed
/// leaves capture the construction policy (`AosGtoData::precision_policy` and
/// friends). Without a scope that is the thread's standalone setting, which a
/// production run never configures (it would silently run fp64 in a mixed
/// run). Production adapters that call free kernels (the complex-twist
/// parameter-derivative adapter) therefore enter this scope with the policy of
/// the run's `NativeObsCtx`:
///
/// ```python
/// with jqmc_rs.NativeConstructionPrecision(ctx.precision_policy()):
///     jqmc_rs.grad_ln_det_mo_complex(...)
/// ```
///
/// The previous thread-local policy is restored on exit; the process-wide
/// standalone setting is neither read nor written.
#[pyclass(name = "NativeConstructionPrecision", unsendable)]
struct NativeConstructionPrecision {
    policy: PrecisionPolicy,
    previous: Option<PrecisionPolicy>,
}

#[pymethods]
impl NativeConstructionPrecision {
    #[new]
    fn new(policy: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            policy: PrecisionPolicy::from_py(policy)?,
            previous: None,
        })
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let policy = slf.policy;
        slf.previous = Some(CONSTRUCTION_PRECISION.with(|slot| slot.replace(policy)));
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        if let Some(previous) = self.previous.take() {
            CONSTRUCTION_PRECISION.with(|slot| slot.set(previous));
        }
        false
    }

    /// The policy this scope installs (exact 18-zone dict).
    fn precision_policy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.policy.to_pydict(py)
    }
}

#[pyfunction]
#[pyo3(name = "configure_native_precision")]
fn py_configure_native_precision(policy: &Bound<'_, PyAny>) -> PyResult<()> {
    let policy = PrecisionPolicy::from_py(policy)?;
    STANDALONE_PRECISION_MASK.store(policy.bitmask(), Ordering::Relaxed);
    CONSTRUCTION_PRECISION.with(|slot| slot.set(policy));
    Ok(())
}

/// Current implementation capability.  This is kept separate from the
/// requested policy so an unsupported f32 request can still be reported as the
/// f64 fallback selected by the native dispatcher.
fn native_precision_support_policy() -> PrecisionPolicy {
    let mut policy = PrecisionPolicy::full();
    for zone in [
        "ao_eval",
        "jastrow_eval",
        "jastrow_grad_lap",
        "jastrow_ratio",
    ] {
        let index = PRECISION_ZONES
            .iter()
            .position(|known| *known == zone)
            .unwrap();
        policy.dtypes[index] = RealDType::F32;
    }
    policy
}

fn native_precision_dispatch_policy(requested: PrecisionPolicy) -> PrecisionPolicy {
    let capability = native_precision_support_policy();
    let mut selected = PrecisionPolicy::full();
    for i in 0..PRECISION_ZONES.len() {
        if requested.dtypes[i] == RealDType::F32 && capability.dtypes[i] == RealDType::F32 {
            selected.dtypes[i] = RealDType::F32;
        }
    }
    selected
}

#[pyfunction]
#[pyo3(name = "native_precision_zones")]
fn py_native_precision_zones() -> Vec<&'static str> {
    PRECISION_ZONES.to_vec()
}

#[pyfunction]
#[pyo3(name = "native_precision_support")]
fn py_native_precision_support(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(
        native_precision_dispatch_policy(standalone_precision_policy())
            .to_pydict(py)?
            .into_any()
            .unbind(),
    )
}

#[pyclass(unsendable)]
pub struct NativeObsCtx {
    /// Immutable Python-resolved precision snapshot. Kernels do not consume it
    /// yet; the separate native-support contract therefore honestly reports f64
    /// and makes mixed-zone tests RED instead of pretending support exists.
    precision_policy: PrecisionPolicy,
    /// Sweep-side extraction: the once-parsed Geminal/Jastrow/J3 that context-taking
    /// `mcmc_sweep[_complex]` reuse. Always built (the sweep is native on every run).
    sweep: Option<SweepArgs>,
    clef: Option<ClefArgs>,
    clef_c: Option<ClefComplexArgs>,
    mol_force: Option<MolForceArgs>,
    mol_lnpsi: Option<MolLnPsiArgs>,
    pbc_force: Option<PbcForceArgs>,
    pbc_lnpsi: Option<PbcLnPsiArgs>,
    pbc_shf: Option<PbcStressHfArgs>,
    pbc_spp: Option<PbcStressPpArgs>,
    pbc_force_c: Option<PbcForceComplexArgs>,
    pbc_lnpsi_c: Option<PbcLnPsiComplexArgs>,
    pbc_shf_c: Option<PbcStressHfComplexArgs>,
    pbc_spp_c: Option<PbcStressPpComplexArgs>,
    /// GFMC projection-interval arguments, built only for GFMC runs
    /// (`want_gfmc=True`).
    gfmc: Option<GfmcArgs>,
    /// C-axis optimization-observable arguments;
    /// built only when `want_param_deriv=True` on the real path.
    param_deriv: Option<ParamDerivArgs>,
    /// Complex Jastrow c-axis methods (DTN and the J1 / J2 / J3 j_matrix
    /// `pd_*_complex` methods), sharing the context's single complex geminal and the
    /// once-parsed Jastrow / J3 orbital / ECP tables. The geminal and basis families
    /// still use their existing free-kernel adapter.
    param_deriv_c_dtn: Option<ComplexDtnPdArgs>,
}

/// Compile-time guarantee that the Rayon batch harness may call
/// `&NativeObsCtx` methods from several worker threads at once. Concurrent `&self`
/// access requires `&NativeObsCtx: Send`, i.e. `NativeObsCtx: Sync`. This function is
/// never called; it FAILS TO COMPILE the moment any field -- or a struct it
/// transitively owns -- gains interior mutability that is not itself `Sync` (a bare
/// `Cell`/`RefCell`, an `Rc`, a non-atomic lazy cache, or a raw pointer without a
/// sound `unsafe impl Sync`). The property is therefore enforced by the compiler,
/// not by review.
///
/// Every `*_Args` family held by the context (`SweepArgs`,
/// `Clef[Complex]Args`, `{Mol,Pbc}Force/LnPsi/Stress*Args`, `GfmcArgs`,
/// `ParamDerivArgs`, `ComplexDtnPdArgs`): each is plain owned data -- `ndarray`
/// arrays, `Vec`, `f64`/`Complex<f64>`, `usize`/`bool` metadata, and
/// `Arc<ComplexGeminal>` (immutable, shared read-only). No `RefCell`/`Cell`/
/// `UnsafeCell`/`Rc`/`&mut`-scratch anywhere in the reachable graph. The only
/// interior mutability in the whole crate is `prof.rs` (a `thread_local!` `RefCell`
/// plus an `AtomicBool` -- both `Sync`, and neither is reachable through a shared
/// `&self` scratch). `NativeObsCtx` carries no per-walker scratch, so nothing had to
/// be moved out; the harness closures own all their mutable state.
#[allow(dead_code)]
fn _assert_obs_ctx_sync() {
    fn is_sync<T: Sync>() {}
    is_sync::<NativeObsCtx>();
}

/// Build a [`NativeObsCtx`] ONCE from the (fp64-coerced) geminal / jastrow /
/// coulomb sub-objects. `complex` selects the complex-twist family; `pbc` the
/// periodic family; `want_stress` gates the (pGTO/composite-leaf-only) stress
/// aggregates -- the driver passes its `_rs_stress_leaf_ok` so a pure-PW / B-spline
/// determinant (no stress kernel) never triggers `pgto_stress_leaves`.
///
/// Requested-observable laziness: the `e_L` arguments (`clef[_c]`) and the
/// sweep args are ALWAYS built (every run sweeps + measures energy); the position-
/// derivative args (force / Pulay) are built only when `want_force`, and the stress
/// args only when `want_stress` (which additionally needs the pGTO/composite leaf
/// gate). An energy-only run therefore never pays the `MolForceArgs`/`PbcForceArgs`
/// `pgto_force_leaves` extraction (the measured ~4.8 s on a 20-water cluster).
#[pyfunction]
#[pyo3(name = "build_obs_ctx", signature = (geminal_data, jastrow_data, coulomb_potential_data, complex, pbc, want_stress, want_force=true, want_gfmc=false, want_param_deriv=false, precision_policy=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 entry point: the parameters are the Python signature"
)]
fn py_build_obs_ctx(
    geminal_data: &Bound<'_, PyAny>,
    jastrow_data: &Bound<'_, PyAny>,
    coulomb_potential_data: &Bound<'_, PyAny>,
    complex: bool,
    pbc: bool,
    want_stress: bool,
    want_force: bool,
    want_gfmc: bool,
    want_param_deriv: bool,
    precision_policy: Option<&Bound<'_, PyAny>>,
) -> PyResult<NativeObsCtx> {
    let g = geminal_data;
    let j = jastrow_data;
    let cp = coulomb_potential_data;
    // An omitted (or explicit `None`) policy is an
    // error, never a silent full-precision default. Production passes
    // `jqmc._precision.precision_policy_snapshot()`; a harness that forgets it
    // would otherwise build fp64 kernels inside a mixed-precision run.
    let Some(precision_policy) = precision_policy.filter(|value| !value.is_none()) else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "build_obs_ctx requires an explicit precision_policy (pass \
             jqmc._precision.precision_policy_snapshot()); there is no default, \
             because omitting it would silently build full-precision (float64) \
             kernels regardless of the active precision mode",
        ));
    };
    let precision_policy = PrecisionPolicy::from_py(precision_policy)?;
    let _precision_guard = ConstructionPrecisionGuard::enter(precision_policy);
    // A complex B-spline parse materializes grid-sized real/imag coefficient storage.
    // Parse it exactly once and share it across every fixed-per-run argument family.
    let shared_cgem = if complex {
        Some(Arc::new(complex_det::ComplexGeminal::from_py(g)?))
    } else {
        None
    };
    let mut ctx = NativeObsCtx {
        precision_policy,
        sweep: Some(SweepArgs::from_py_with_cgem(
            g,
            j,
            complex,
            shared_cgem.clone(),
        )?),
        clef: None,
        clef_c: None,
        mol_force: None,
        mol_lnpsi: None,
        pbc_force: None,
        pbc_lnpsi: None,
        pbc_shf: None,
        pbc_spp: None,
        pbc_force_c: None,
        pbc_lnpsi_c: None,
        pbc_shf_c: None,
        pbc_spp_c: None,
        gfmc: None,
        param_deriv: None,
        param_deriv_c_dtn: None,
    };
    if want_param_deriv {
        if complex {
            // Parsed ONCE per run for every complex Jastrow family (J1 / J2 / J3
            // j_matrix and DTN): the per-walker free kernels re-parsed the geminal,
            // Jastrow, J3 orbital and ECP tables on every call.
            if !j.is_none() {
                ctx.param_deriv_c_dtn = Some(ComplexDtnPdArgs::from_py_with_gem(
                    j,
                    cp,
                    shared_cgem
                        .clone()
                        .ok_or_else(|| ctx_missing("shared complex geminal"))?,
                )?);
            }
        } else {
            ctx.param_deriv = Some(ParamDerivArgs::from_py(g, j, cp)?);
        }
    }
    if want_gfmc {
        // The complex-twist GFMC projection chunk drives
        // `gfmc_interval_{n,t}_complex` over the verified complex e_L kernels.
        // A complex `comput_position_deriv` run is not
        // energy-only -- `want_force` / `want_stress` build the complex force/stress
        // `*_ComplexArgs` in the `if complex` block below, and
        // `gfmc_force_block_complex` folds them out of the projection call.
        ctx.gfmc = Some(GfmcArgs::from_py_with_cgem(
            g,
            j,
            cp,
            complex,
            shared_cgem.clone(),
        )?);
    }
    if complex {
        let cgem = shared_cgem
            .clone()
            .ok_or_else(|| ctx_missing("shared complex geminal"))?;
        ctx.clef_c = Some(ClefComplexArgs::from_py_with_gem(g, j, cp, cgem.clone())?);
        if want_force {
            ctx.pbc_force_c = Some(PbcForceComplexArgs::from_py_with_gem(
                g,
                j,
                cp,
                cgem.clone(),
            )?);
            ctx.pbc_lnpsi_c = Some(PbcLnPsiComplexArgs::from_py_with_gem(g, j, cgem.clone())?);
        }
        if want_stress {
            ctx.pbc_shf_c = Some(PbcStressHfComplexArgs::from_py_with_gem(
                g,
                j,
                cp,
                cgem.clone(),
            )?);
            ctx.pbc_spp_c = Some(PbcStressPpComplexArgs::from_py_with_gem(g, j, cgem)?);
        }
    } else if pbc {
        ctx.clef = Some(ClefArgs::from_py(g, j, cp)?);
        if want_force {
            ctx.pbc_force = Some(PbcForceArgs::from_py(g, j, cp)?);
            ctx.pbc_lnpsi = Some(PbcLnPsiArgs::from_py(g, j)?);
        }
        if want_stress {
            ctx.pbc_shf = Some(PbcStressHfArgs::from_py(g, j, cp)?);
            ctx.pbc_spp = Some(PbcStressPpArgs::from_py(g, j)?);
        }
    } else {
        ctx.clef = Some(ClefArgs::from_py(g, j, cp)?);
        if want_force {
            ctx.mol_force = Some(MolForceArgs::from_py(g, j, cp, "vmc_grad_e_L")?);
            ctx.mol_lnpsi = Some(MolLnPsiArgs::from_py(g, j)?);
        }
    }
    Ok(ctx)
}

fn ctx_missing(name: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "NativeObsCtx: {name} was not built for this Hamiltonian (build_obs_ctx flag mismatch)"
    ))
}

// ===================================================================================
// Whole-run energy chunk over the walker batch.
//
// The per-walker chunk body is factored out of the `run_energy_chunk` PyO3 entry into
// two shared internal methods -- `run_energy_chunk_real_body` (real / real-Gamma) and
// `run_energy_chunk_complex_body` (complex twist). BOTH the per-walker entry
// (`run_energy_chunk`) and the batch entry (`run_energy_chunk_batch`) call these
// exact bodies, so there is ONE copy of the kernel logic (no duplication). The bodies
// take Send-able ndarray views / slices (never `PyReadonlyArray`, which is not `Send`)
// and return an owned Rust output struct (no `py`, no `into_pyarray`), so a batch entry
// can run them across walkers inside `walker_batch::for_each_walker`'s GIL-released
// Rayon region and build the batched dictionary afterwards.
// ===================================================================================

/// Owned per-walker outputs of the REAL energy chunk (no `py` state). Which optional
/// fields are `Some` encodes the chunk mode / requested observables, mirroring exactly
/// the keys the per-walker dict sets:
///   * `w_l` + `g_final`/`gi_final` present  <-> FAST (native_rng) chunk
///   * `g_steps`/`gi_steps` present          <-> BITEXACT chunk (per-step geminal stacks)
///   * `hf_*`/`pp_*` present                 <-> want_force
///   * `ru_steps`/`rd_steps` present         <-> want_force || want_steps
///   * `shf`/`spp` present                   <-> want_stress
struct RealChunkOut {
    e_l: Vec<f64>,
    w_l: Option<Vec<f64>>,
    g_final: Option<numpy::ndarray::Array2<f64>>,
    gi_final: Option<numpy::ndarray::Array2<f64>>,
    g_steps: Option<numpy::ndarray::Array3<f64>>,
    gi_steps: Option<numpy::ndarray::Array3<f64>>,
    r_up: numpy::ndarray::Array2<f64>,
    r_dn: numpy::ndarray::Array2<f64>,
    final_key: [u32; 2],
    accepted: usize,
    ao_up: numpy::ndarray::Array2<f64>,
    ao_dn: numpy::ndarray::Array2<f64>,
    paired_dn: numpy::ndarray::Array2<f64>,
    paired_up_lambda: numpy::ndarray::Array2<f64>,
    hf_dr: Option<numpy::ndarray::Array3<f64>>,
    hf_dru: Option<numpy::ndarray::Array3<f64>>,
    hf_drd: Option<numpy::ndarray::Array3<f64>>,
    pp_dr: Option<numpy::ndarray::Array3<f64>>,
    pp_dru: Option<numpy::ndarray::Array3<f64>>,
    pp_drd: Option<numpy::ndarray::Array3<f64>>,
    ru_steps: Option<numpy::ndarray::Array3<f64>>,
    rd_steps: Option<numpy::ndarray::Array3<f64>>,
    shf: Option<numpy::ndarray::Array3<f64>>,
    spp: Option<numpy::ndarray::Array3<f64>>,
    /// Steps whose carried inverse was refreshed from the same geminal.
    inverse_refreshes: usize,
}

/// Owned per-walker outputs of the COMPLEX-twist energy chunk (fast-mode, energy-only).
struct ComplexChunkOut {
    e_l: Vec<numpy::Complex64>,
    w_l: Vec<f64>,
    geminal: numpy::ndarray::Array2<numpy::Complex64>,
    geminal_inv: numpy::ndarray::Array2<numpy::Complex64>,
    r_up: numpy::ndarray::Array2<f64>,
    r_dn: numpy::ndarray::Array2<f64>,
    final_key: [u32; 2],
    accepted: usize,
    ao_up: numpy::ndarray::Array2<numpy::Complex64>,
    ao_dn: numpy::ndarray::Array2<numpy::Complex64>,
    paired_dn: numpy::ndarray::Array2<numpy::Complex64>,
    paired_up_lambda: numpy::ndarray::Array2<numpy::Complex64>,
    ru_steps: Option<numpy::ndarray::Array3<f64>>,
    rd_steps: Option<numpy::ndarray::Array3<f64>>,
    /// Steps whose carried inverse was refreshed from the same geminal.
    inverse_refreshes: usize,
}

impl NativeObsCtx {
    /// The once-parsed complex-twist Jastrow parameter-derivative tables
    /// (`build_obs_ctx(complex=True, want_param_deriv=True)` with a Jastrow).
    fn complex_pd_args(&self) -> PyResult<&ComplexDtnPdArgs> {
        self.param_deriv_c_dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv_complex"))
    }

    /// Shared REAL per-walker chunk body (see the module comment above): the
    /// `run_energy_chunk` real-path computation, with the
    /// PyO3 marshalling left to the caller (it takes ndarray views / slices) and
    /// a `RealChunkOut` in place of the terminal dict. Called once per walker by the
    /// per-walker entry and by every walker of `run_energy_chunk_batch`.
    #[allow(clippy::too_many_arguments)]
    fn run_energy_chunk_real_body(
        &self,
        positions_arr: numpy::ndarray::ArrayView2<f64>,
        charges_slice: &[f64],
        r_up: numpy::ndarray::ArrayView2<f64>,
        r_dn: numpy::ndarray::ArrayView2<f64>,
        geminal: numpy::ndarray::ArrayView2<f64>,
        geminal_inv: numpy::ndarray::ArrayView2<f64>,
        dt: f64,
        epsilon_as: f64,
        key0: [u32; 2],
        normals_arr: numpy::ndarray::ArrayView2<f64>, // (n_steps, num_moves) -- bitexact only
        grids_arr: numpy::ndarray::ArrayView3<f64>,   // (n_steps, n_grid, 3) -- bitexact only
        weights_slice: &[f64],
        num_moves: usize,
        cell_nmax: Option<(&numpy::ndarray::ArrayView2<f64>, i32)>,
        native_rng: bool,
        injected_inputs: bool,
        random_mesh: bool,
        base_grid: Option<numpy::ndarray::ArrayView2<f64>>, // (n_grid, 3) -- fast only
        n_steps_fast: usize,
        want_force: bool,
        want_stress: bool,
        want_steps: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<RealChunkOut> {
        use numpy::ndarray::{Array2, Array3, Axis};
        crate::prof::set_from_env();
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let gem = s.gem.as_ref().ok_or_else(|| ctx_missing("sweep_real"))?;
        let clef = self.clef.as_ref().ok_or_else(|| ctx_missing("clef"))?;
        let j3_jm_view = s.j3_jm.as_ref().map(|m| m.view());
        // The FAST chunk e_L (`mesh_cutoff = native_rng`) contracts the ECP mesh AOs
        // with the FAST value screen whenever these leaves qualify (the same
        // `ecp_wdot_ok` / `ecp_j3_wdot_ok` predicates). For molecular Cartesian
        // leaves the force then differentiates that same screened mesh term.
        let mol_cart = |o: &determinant::OrbData| {
            o.aos.kind == atomic_orbital::AoKind::Cart && o.aos.n_image == 1
        };
        let ecp_screen = gfmc_forces::FastScreen {
            det: native_rng
                && [&clef.gem.orb_up, &clef.gem.orb_dn]
                    .iter()
                    .all(|o| mol_cart(o) && coulomb_potential::ecp_wdot_ok(o, true)),
            j3: native_rng
                && clef.jdata.three_body.as_ref().is_some_and(|j3| {
                    mol_cart(&j3.orb) && coulomb_potential::ecp_j3_wdot_ok(&j3.orb, true)
                }),
            // FAST e_L ECP radial cut: `cutoff && geminal_inv.is_some()` (always here).
            ecp_radial: native_rng,
            mesh_shell_only: false,
        };

        let n_steps = if injected_inputs {
            normals_arr.nrows()
        } else {
            n_steps_fast
        };
        if !injected_inputs && base_grid.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_energy_chunk: fast mode (native_rng) requires base_grid_points",
            ));
        }

        // Threaded walker state (owned; advanced per step by the SM sweep).
        let mut r_up_cur = r_up.to_owned();
        let mut r_dn_cur = r_dn.to_owned();
        let mut g_cur = geminal.to_owned();
        let mut gi_cur = geminal_inv.to_owned();
        let mut key = key0;

        let (n, m) = (g_cur.nrows(), g_cur.ncols());
        let mut e_l_out: Vec<f64> = Vec::with_capacity(n_steps);
        let mut w_l_out: Vec<f64> = Vec::new();
        let (mut g_out, mut gi_out) = if native_rng {
            (None, None)
        } else {
            (
                Some(Array3::<f64>::zeros((n_steps, n, m))),
                Some(Array3::<f64>::zeros((n_steps, n, m))),
            )
        };
        let mut accepted_total = 0usize;
        let mut inverse_refreshes = 0usize;
        let mut last_det: Option<crate::determinant::DetRatioState> = None;

        let (n_atom, n_up, n_dn) = (positions_arr.nrows(), r_up_cur.nrows(), r_dn_cur.nrows());
        let mut hf_dr_out = if want_force {
            Some(Array3::<f64>::zeros((n_steps, n_atom, 3)))
        } else {
            None
        };
        let mut hf_dru_out = if want_force {
            Some(Array3::<f64>::zeros((n_steps, n_up, 3)))
        } else {
            None
        };
        let mut hf_drd_out = if want_force {
            Some(Array3::<f64>::zeros((n_steps, n_dn, 3)))
        } else {
            None
        };
        let mut pp_dr_out = if want_force {
            Some(Array3::<f64>::zeros((n_steps, n_atom, 3)))
        } else {
            None
        };
        let mut pp_dru_out = if want_force {
            Some(Array3::<f64>::zeros((n_steps, n_up, 3)))
        } else {
            None
        };
        let mut pp_drd_out = if want_force {
            Some(Array3::<f64>::zeros((n_steps, n_dn, 3)))
        } else {
            None
        };
        let mut ru_steps_out = if want_force || want_steps {
            Some(Array3::<f64>::zeros((n_steps, n_up, 3)))
        } else {
            None
        };
        let mut rd_steps_out = if want_force || want_steps {
            Some(Array3::<f64>::zeros((n_steps, n_dn, 3)))
        } else {
            None
        };
        let mut shf_out = if want_stress {
            Some(Array3::<f64>::zeros((n_steps, 3, 3)))
        } else {
            None
        };
        let mut spp_out = if want_stress {
            Some(Array3::<f64>::zeros((n_steps, 3, 3)))
        } else {
            None
        };

        let j3_carry_ok = native_rng && s.j3_orb.as_ref().is_some_and(|o| o.aos.n_image == 1);
        // Carry the sweep's streaming det state across the chunk's steps (and hand
        // it to the e_L). Molecular leaves, plus periodic MO-GTO leaves whose FAST
        // sweep evaluates every column with a screened periodic kernel
        // (`compute_aos_single_fold_cutoff_into` for fp64 `ao_eval`,
        // `compute_aos_single_f32_into` with its `pz_val` shell screen for fp32
        // `ao_eval`), so the carried columns, the carried G and the e_L's det tables
        // share one AO convention.
        let periodic_fast_gto = |o: &crate::determinant::OrbData| {
            o.aos.n_image > 1
                && o.pw.is_none()
                && o.bspline.is_none()
                && o.mo_coefficients.is_some()
                && !o.aos.shells.is_empty()
        };
        let det_carry_ok = native_rng
            && ((gem.orb_up.aos.n_image == 1 && gem.orb_dn.aos.n_image == 1)
                || (periodic_fast_gto(&gem.orb_up) && periodic_fast_gto(&gem.orb_dn)));
        let mut j3_carry: Option<crate::jastrow_factor::J3RatioState> = None;
        // FAST chunk, molecular MO-GTO determinant: carry the kinetic MO
        // derivative tables across the chunk's steps and refresh only the
        // columns of electrons whose position changed in the sweep (a sweep
        // moves `num_moves` electrons, typically far fewer than N). Same
        // derivative screen as the per-e_L build (`pz_deriv`, dropped values
        // < FAST_SCREEN_ABS_BOUND); the refreshed columns come from the
        // single-position kernel + `project5`, the per-hop GFMC chain (floor-level
        // vs the batch build, FAST chunk only). Never crosses a chunk boundary.
        let mol_mo_gto = |o: &crate::determinant::OrbData| {
            o.aos.n_image == 1
                && o.pw.is_none()
                && o.bspline.is_none()
                && o.mo_coefficients.is_some()
                && !o.aos.shells.is_empty()
        };
        // Periodic MO-GTO twin (the det state is carried, `det_carry_ok`): the
        // image-folded GFMC tables (`init/update_det_grad_lap_cache_folded`, the
        // screened periodic value + `pz_deriv` derivative family) on home-cell
        // wrapped positions, as the per-e_L FAST kinetic wraps them.
        // Force / stress runs take this FAST kinetic too: their periodic derivative kernels
        // differentiate the SAME screened energy (`gfmc_forces::fast_screens`), so a
        // force/stress run reports the e_L of an energy-only run.
        let kin_carry_periodic = native_rng
            && det_carry_ok
            && periodic_fast_gto(&gem.orb_up)
            && periodic_fast_gto(&gem.orb_dn);
        // Molecular: carried in force/stress runs too, so their e_L is the same FAST
        // kinetic route as the energy-only run (the force differentiates that same
        // screened kinetic, see `gfmc_forces::FastScreen`).
        let kin_carry = kin_carry_periodic
            || (native_rng && det_carry_ok && mol_mo_gto(&gem.orb_up) && mol_mo_gto(&gem.orb_dn));
        // J3 twin: raw-AO molecular J3 leaf whose streaming ratio state is carried
        // (`j3_carry_ok`), contracted by `contract_j3_grad_lap_cache` (the GFMC
        // J3 derivative table). Both precision modes: as in the GFMC kinetic
        // (`MeshFastState::j3_grad_lap`), the carried table and the fp64
        // `jastrow_ratio` state are contracted in fp64; under an fp32
        // `jastrow_grad_lap` zone `compute_kinetic_energy_fast` rounds the summed
        // Jastrow gradient / Laplacian to that zone, as for the fresh rebuild.
        let j3_kin_carry = kin_carry
            && j3_carry_ok
            && s.j3_orb.as_ref().is_some_and(|o| {
                o.aos.n_image == 1
                    && o.pw.is_none()
                    && o.bspline.is_none()
                    && o.mo_coefficients.is_none()
                    && !o.aos.shells.is_empty()
            });
        let mut kc_pos: Option<(Array2<f64>, Array2<f64>)> = None;
        let mut kin_cache: Option<crate::determinant::DetGradLapCache> = None;
        let mut j3_kin_cache: Option<crate::jastrow_factor::J3GradLapCache> = None;

        for step in 0..n_steps {
            // Route-predicate counters (observability only, off in production).
            crate::gfmc::record_mcmc_det_carry(det_carry_ok);
            if s.j3_orb.is_some() {
                crate::gfmc::record_mcmc_j3_carry(j3_carry_ok);
            }
            let normals_owned: Vec<f64> = if injected_inputs {
                normals_arr.index_axis(Axis(0), step).to_vec()
            } else {
                crate::threefry::native_step_normals(key, num_moves).0
            };
            let res = crate::timed!(
                "chunk.sweep",
                mcmc::mcmc_sweep(
                    gem,
                    &s.jas,
                    s.j3_orb.as_ref(),
                    j3_jm_view.as_ref(),
                    &positions_arr,
                    charges_slice,
                    &r_up_cur.view(),
                    &r_dn_cur.view(),
                    &g_cur.view(),
                    &gi_cur.view(),
                    dt,
                    epsilon_as,
                    key,
                    &normals_owned,
                    num_moves,
                    cell_nmax,
                    if det_carry_ok { last_det.take() } else { None },
                    if j3_carry_ok { j3_carry.take() } else { None },
                    native_rng,
                )
            );
            r_up_cur = res.r_up;
            r_dn_cur = res.r_dn;
            g_cur = res.geminal;
            gi_cur = res.geminal_inv;
            // Refresh a drifted carried inverse from the same carried
            // geminal before any consumer (e_L, AS weight, force, stress) and
            // the next sweep see it. O(N^2) probe; O(N^3) only when triggered.
            if crate::timed!(
                "chunk.inv_refresh",
                crate::determinant::refresh_running_inverse_if_needed(
                    &g_cur.view(),
                    &mut gi_cur,
                    inverse_refresh_threshold,
                )
            ) {
                inverse_refreshes += 1;
            }
            key = res.final_key;
            accepted_total += res.accepted;
            last_det = Some(res.det_state);
            let last_j3 = res.j3_state;
            let det_state_for_el = if det_carry_ok {
                last_det.as_ref()
            } else {
                None
            };
            let j3_state_for_el =
                if native_rng && s.j3_orb.as_ref().is_some_and(|o| o.aos.n_image == 1) {
                    last_j3.as_ref()
                } else {
                    None
                };

            let grid_fast_owned: Option<Array2<f64>> = if !injected_inputs {
                let base = base_grid.as_ref().unwrap();
                if random_mesh {
                    let rt = crate::threefry::native_rotation_matrix(key);
                    let ng = base.nrows();
                    let mut rot = Array2::<f64>::zeros((ng, 3));
                    for gp in 0..ng {
                        for c in 0..3 {
                            rot[[gp, c]] = base[[gp, 0]] * rt[0][c]
                                + base[[gp, 1]] * rt[1][c]
                                + base[[gp, 2]] * rt[2][c];
                        }
                    }
                    Some(rot)
                } else {
                    Some(base.to_owned())
                }
            } else {
                None
            };
            let grid_view = match &grid_fast_owned {
                Some(g) => g.view(),
                None => grids_arr.index_axis(Axis(0), step),
            };
            if kin_carry {
                crate::timed!("chunk.kin_cache", {
                    match kc_pos.as_mut() {
                        None => {
                            kin_cache = Some(if kin_carry_periodic {
                                crate::determinant::init_det_grad_lap_cache_folded(
                                    gem,
                                    &crate::atomic_orbital::wrap_r(
                                        &gem.orb_up.aos,
                                        &r_up_cur.view(),
                                    )
                                    .view(),
                                    &crate::atomic_orbital::wrap_r(
                                        &gem.orb_dn.aos,
                                        &r_dn_cur.view(),
                                    )
                                    .view(),
                                )
                            } else {
                                crate::determinant::init_det_grad_lap_cache(
                                    gem,
                                    &r_up_cur.view(),
                                    &r_dn_cur.view(),
                                )
                            });
                            if j3_kin_carry {
                                j3_kin_cache = Some(crate::jastrow_factor::init_j3_grad_lap_cache(
                                    s.j3_orb.as_ref().unwrap(),
                                    &r_up_cur.view(),
                                    &r_dn_cur.view(),
                                ));
                            }
                            kc_pos = Some((r_up_cur.clone(), r_dn_cur.clone()));
                        }
                        Some((pu, pd)) => {
                            for (is_up, rc, rp) in [(true, &r_up_cur, pu), (false, &r_dn_cur, pd)] {
                                for i in 0..rc.nrows() {
                                    let pos = [rc[[i, 0]], rc[[i, 1]], rc[[i, 2]]];
                                    if pos != [rp[[i, 0]], rp[[i, 1]], rp[[i, 2]]] {
                                        if let Some(c) = kin_cache.as_mut() {
                                            if kin_carry_periodic {
                                                let orb =
                                                    if is_up { &gem.orb_up } else { &gem.orb_dn };
                                                crate::determinant::update_det_grad_lap_cache_and_value_folded(
                                                    gem,
                                                    c,
                                                    is_up,
                                                    i,
                                                    crate::atomic_orbital::wrap_pos(&orb.aos, pos),
                                                );
                                            } else {
                                                crate::determinant::update_det_grad_lap_cache_and_value(
                                                    gem, c, is_up, i, pos,
                                                );
                                            }
                                        }
                                        if let Some(c) = j3_kin_cache.as_mut() {
                                            crate::jastrow_factor::update_j3_grad_lap_cache(
                                                s.j3_orb.as_ref().unwrap(),
                                                c,
                                                is_up,
                                                i,
                                                pos,
                                            );
                                        }
                                        rp[[i, 0]] = pos[0];
                                        rp[[i, 1]] = pos[1];
                                        rp[[i, 2]] = pos[2];
                                    }
                                }
                            }
                        }
                    }
                });
            }
            let e = crate::timed!(
                "chunk.eL",
                clef.compute(
                    &r_up_cur.view(),
                    &r_dn_cur.view(),
                    weights_slice,
                    &grid_view,
                    &gi_cur.view(),
                    native_rng,
                    det_state_for_el,
                    j3_state_for_el,
                    kin_cache.as_ref(),
                    j3_kin_cache.as_ref(),
                )?
            );
            e_l_out.push(e);
            if j3_carry_ok {
                j3_carry = last_j3;
            }

            if native_rng {
                // Issue #131: robust R_AS; w_L = 1 at epsilon_as == 0.
                let r_as = crate::timed!(
                    "chunk.as_weight",
                    crate::determinant::compute_as_regularization_factor(
                        &g_cur.view(),
                        &gi_cur.view()
                    )
                );
                w_l_out.push(crate::determinant::as_walker_weight(r_as, epsilon_as));
            } else {
                g_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&g_cur);
                gi_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&gi_cur);
            }

            if want_force {
                let ru_v = r_up_cur.view();
                let rd_v = r_dn_cur.view();
                let gi_v = gi_cur.view();
                // FAST molecular Pulay can reuse the HF forward. The BITEXACT chunk
                // uses the standalone Pulay kernel to retain its evaluation order.
                let mut pp_shared: Option<Grad3> = None;
                let (hdr, hdu, hdd) = crate::timed!(
                    "chunk.force_hf",
                    if let Some(pf) = self.pbc_force.as_ref() {
                        pf.compute(&ru_v, &rd_v, weights_slice, &grid_view, &gi_v)?
                    } else {
                        let mf = self
                            .mol_force
                            .as_ref()
                            .ok_or_else(|| ctx_missing("mol_force"))?;
                        let want_pp =
                            native_rng && self.pbc_lnpsi.is_none() && self.mol_lnpsi.is_some();
                        let (hf, pp) = if mf.ecp.is_some() {
                            mf.compute_ecp_pp(
                                &ru_v,
                                &rd_v,
                                weights_slice,
                                &grid_view,
                                &RT_ID,
                                false,
                                Some(&gi_v),
                                ecp_screen,
                                want_pp,
                                kin_cache.as_ref(),
                            )
                        } else {
                            mf.compute_ae_pp(
                                &ru_v,
                                &rd_v,
                                Some(&gi_v),
                                gfmc_forces::FastScreen::default(),
                                want_pp,
                                None,
                            )
                        };
                        pp_shared = pp;
                        hf
                    }
                );
                hf_dr_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&hdr);
                hf_dru_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&hdu);
                hf_drd_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&hdd);
                let (pdr, pdu, pdd) = crate::timed!(
                    "chunk.force_pp",
                    if let Some(pp) = pp_shared {
                        pp
                    } else if let Some(pl) = self.pbc_lnpsi.as_ref() {
                        pl.compute(&ru_v, &rd_v, &gi_v)
                    } else {
                        let ml = self
                            .mol_lnpsi
                            .as_ref()
                            .ok_or_else(|| ctx_missing("mol_lnpsi"))?;
                        ml.compute(&ru_v, &rd_v, &gi_v)
                    }
                );
                pp_dr_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&pdr);
                pp_dru_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&pdu);
                pp_drd_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&pdd);
            }
            if want_stress {
                let ru_v = r_up_cur.view();
                let rd_v = r_dn_cur.view();
                let gi_v = gi_cur.view();
                let shf = crate::timed!(
                    "chunk.stress_hf",
                    self.pbc_shf
                        .as_ref()
                        .ok_or_else(|| ctx_missing("pbc_stress_hf"))?
                        .compute(&ru_v, &rd_v, weights_slice, &grid_view, &gi_v, true)?
                );
                let spp = crate::timed!(
                    "chunk.stress_pp",
                    self.pbc_spp
                        .as_ref()
                        .ok_or_else(|| ctx_missing("pbc_stress_pp"))?
                        .compute(&ru_v, &rd_v, &gi_v)
                );
                shf_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&shf);
                spp_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&spp);
            }
            if want_force || want_steps {
                ru_steps_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&r_up_cur);
                rd_steps_out
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), step)
                    .assign(&r_dn_cur);
            }
        }

        if crate::prof::on() {
            crate::prof::dump("run_energy_chunk");
        }
        let det = last_det.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("run_energy_chunk: n_steps must be >= 1")
        })?;
        Ok(RealChunkOut {
            e_l: e_l_out,
            w_l: if native_rng { Some(w_l_out) } else { None },
            g_final: if native_rng { Some(g_cur) } else { None },
            gi_final: if native_rng { Some(gi_cur) } else { None },
            g_steps: g_out,
            gi_steps: gi_out,
            r_up: r_up_cur,
            r_dn: r_dn_cur,
            final_key: key,
            accepted: accepted_total,
            ao_up: det.ao_up,
            ao_dn: det.ao_dn,
            paired_dn: det.paired_dn,
            paired_up_lambda: det.paired_up_lambda,
            hf_dr: hf_dr_out,
            hf_dru: hf_dru_out,
            hf_drd: hf_drd_out,
            pp_dr: pp_dr_out,
            pp_dru: pp_dru_out,
            pp_drd: pp_drd_out,
            ru_steps: ru_steps_out,
            rd_steps: rd_steps_out,
            shf: shf_out,
            spp: spp_out,
            inverse_refreshes,
        })
    }

    /// Shared COMPLEX-twist per-walker chunk body (fast-mode, energy-only): the
    /// `run_energy_chunk` `if complex { .. }` path, with marshalling left to the caller
    /// and a `ComplexChunkOut` in place of the terminal dict. The caller validates fast-mode /
    /// energy-only BEFORE calling this.
    #[allow(clippy::too_many_arguments)]
    fn run_energy_chunk_complex_body(
        &self,
        positions_arr: numpy::ndarray::ArrayView2<f64>,
        charges_slice: &[f64],
        r_up: numpy::ndarray::ArrayView2<f64>,
        r_dn: numpy::ndarray::ArrayView2<f64>,
        dt: f64,
        epsilon_as: f64,
        key0: [u32; 2],
        weights_slice: &[f64],
        num_moves: usize,
        cell_nmax: Option<(&numpy::ndarray::ArrayView2<f64>, i32)>,
        random_mesh: bool,
        base: numpy::ndarray::ArrayView2<f64>, // (n_grid, 3) -- required
        n_steps: usize,
        want_steps: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<ComplexChunkOut> {
        use numpy::ndarray::{Array2, Array3, Axis};
        use numpy::Complex64;
        crate::prof::set_from_env();
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let cgem = s
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("sweep_complex"))?;
        let clef_c = self
            .clef_c
            .as_ref()
            .ok_or_else(|| ctx_missing("clef_complex"))?;
        let j3m_view = s.j3_jm.as_ref().map(|m| m.view());

        let mut r_up_cur = r_up.to_owned();
        let mut r_dn_cur = r_dn.to_owned();
        // Determinant base carry: build G, its inverse and the sweep's
        // `DetRatioStateC` ONCE per chunk and thread them across the steps (the
        // sweep's SM-advanced G / inverse after the carried-inverse refresh check, and its
        // advanced streaming state), as the real chunk's `det_carry_ok`. The carried
        // columns are the sweep's own per-move `orb_single_complex` evaluations, so
        // the carried base differs from a per-step rebuild only at the SM-carry fp
        // floor (FAST chain). Test-only A/B hook: JQMC_RS_DET_COMPLEX_FRESH_BASE
        // rebuilds the base every step (the pre-carry route).
        let det_base_carry_ok = std::env::var_os("JQMC_RS_DET_COMPLEX_FRESH_BASE").is_none();
        let (mut g_cur_c, mut gi_cur_c) = if det_base_carry_ok {
            crate::timed!("chunk.base_c", {
                let g = crate::complex_det::compute_geminal_complex(
                    cgem,
                    &r_up_cur.view(),
                    &r_dn_cur.view(),
                );
                let gi = crate::complex_det::inverse_complex(&g);
                (g, gi)
            })
        } else {
            (
                Array2::<Complex64>::zeros((0, 0)),
                Array2::<Complex64>::zeros((0, 0)),
            )
        };
        let mut key = key0;
        let mut e_l_out: Vec<Complex64> = Vec::with_capacity(n_steps);
        let mut w_l_out: Vec<f64> = Vec::with_capacity(n_steps);
        let mut accepted_total = 0usize;
        let mut inverse_refreshes = 0usize;
        let mut g_last: Option<Array2<Complex64>> = None;
        let mut gi_last: Option<Array2<Complex64>> = None;
        let mut det_last: Option<crate::complex_det::DetRatioStateC> = None;
        let (n_up_c, n_dn_c) = (r_up_cur.nrows(), r_dn_cur.nrows());
        // The chunk e_L seeds its ECP-nl mesh state on the sweep's carried
        // determinant base (orbital tables + running SM inverse, the inverse the
        // kinetic term already consumes). Test-only A/B hook:
        // JQMC_RS_ECP_COMPLEX_FRESH_SEED re-seeds it fresh every e_L (the pre-carry
        // route), so a test can bound the carried-vs-fresh difference.
        let carry_ecp_seed = std::env::var_os("JQMC_RS_ECP_COMPLEX_FRESH_SEED").is_none();
        // Kinetic derivative-table carry (`gfmc::ComplexChunkKinCarry`, the complex
        // GFMC e_L's det / J1-J2 / DTN / J3 derivative caches): seeded once at the
        // chunk's start positions, advanced by replaying each sweep's accepted moves
        // in order, contracted per e_L. Supported determinant classes only (the
        // complex GFMC gate); a decline keeps the fresh per-e_L kinetic. Test-only
        // A/B hook: JQMC_RS_KIN_COMPLEX_FRESH keeps the fresh kinetic.
        let kin_carry_ok = std::env::var_os("JQMC_RS_KIN_COMPLEX_FRESH").is_none()
            && crate::gfmc::complex_chunk_kin_carry_covers(cgem);
        let clef_j3v = clef_c.j3m.as_ref().map(|m| m.view());
        let clef_j3 = match (&clef_c.j3c, &clef_j3v) {
            (Some(o), Some(v)) => Some((o, v)),
            _ => None,
        };
        let mut kin_carry: Option<crate::gfmc::ComplexChunkKinCarry> = if kin_carry_ok {
            Some(crate::timed!(
                "chunk.kin_cache_c",
                crate::gfmc::ComplexChunkKinCarry::seed(
                    cgem,
                    &clef_c.jas,
                    clef_j3,
                    &r_up_cur.view(),
                    &r_dn_cur.view(),
                )
            ))
        } else {
            None
        };
        let mut ru_steps_c = if want_steps {
            Some(Array3::<f64>::zeros((n_steps, n_up_c, 3)))
        } else {
            None
        };
        let mut rd_steps_c = if want_steps {
            Some(Array3::<f64>::zeros((n_steps, n_dn_c, 3)))
        } else {
            None
        };

        for _step in 0..n_steps {
            crate::gfmc::record_cplx_det_base_carry(det_base_carry_ok);
            crate::gfmc::record_cplx_kin_carry(kin_carry_ok);
            let normals_owned = crate::threefry::native_step_normals(key, num_moves).0;
            if !det_base_carry_ok {
                crate::timed!("chunk.base_c", {
                    g_cur_c = crate::complex_det::compute_geminal_complex(
                        cgem,
                        &r_up_cur.view(),
                        &r_dn_cur.view(),
                    );
                    gi_cur_c = crate::complex_det::inverse_complex(&g_cur_c);
                });
            }
            let res = crate::timed!(
                "chunk.sweep_c",
                mcmc::mcmc_sweep_complex_carry(
                    cgem,
                    &s.jas,
                    s.j3c.as_ref(),
                    j3m_view.as_ref(),
                    &positions_arr,
                    charges_slice,
                    &r_up_cur.view(),
                    &r_dn_cur.view(),
                    &g_cur_c.view(),
                    &gi_cur_c.view(),
                    dt,
                    epsilon_as,
                    key,
                    &normals_owned,
                    num_moves,
                    cell_nmax,
                    if det_base_carry_ok {
                        det_last.take()
                    } else {
                        None
                    },
                    kin_carry_ok,
                )
            );
            r_up_cur = res.r_up;
            r_dn_cur = res.r_dn;
            g_cur_c = res.geminal;
            gi_cur_c = res.geminal_inv;
            // Refresh a drifted running inverse (carried across the chunk's
            // steps under `det_base_carry_ok`, otherwise seeded fresh each step but
            // still carrying a near-singular start's error) from the same G before
            // the e_L, the AS weight and the next sweep see it.
            if crate::timed!(
                "chunk.inv_refresh_c",
                crate::complex_det::refresh_running_inverse_if_needed_complex(
                    &g_cur_c,
                    &mut gi_cur_c,
                    inverse_refresh_threshold,
                )
            ) {
                inverse_refreshes += 1;
            }
            key = res.final_key;
            accepted_total += res.accepted;
            if let Some(kc) = kin_carry.as_mut() {
                crate::timed!("chunk.kin_cache_c", {
                    for mv in &res.accepted_moves {
                        kc.advance(cgem, &clef_c.jas, clef_j3, mv);
                    }
                    kc.screen(cgem, &clef_c.jas, clef_j3);
                });
                debug_assert!(kc.positions().0 == r_up_cur && kc.positions().1 == r_dn_cur);
            }
            // Issue #131: robust R_AS; w_L = 1 at epsilon_as == 0.
            let r_as = crate::complex_det::compute_as_regularization_factor_complex(
                &g_cur_c.view(),
                &gi_cur_c.view(),
            );
            w_l_out.push(crate::determinant::as_walker_weight(r_as, epsilon_as));
            let grid_owned: Array2<f64> = if random_mesh {
                let rt = crate::threefry::native_rotation_matrix(key);
                let ng = base.nrows();
                let mut rot = Array2::<f64>::zeros((ng, 3));
                for gp in 0..ng {
                    for c in 0..3 {
                        rot[[gp, c]] = base[[gp, 0]] * rt[0][c]
                            + base[[gp, 1]] * rt[1][c]
                            + base[[gp, 2]] * rt[2][c];
                    }
                }
                rot
            } else {
                base.to_owned()
            };
            let (re, im) = crate::timed!(
                "chunk.eL_c",
                clef_c.compute_with_kin(
                    &r_up_cur.view(),
                    &r_dn_cur.view(),
                    weights_slice,
                    &grid_owned.view(),
                    &gi_cur_c.view(),
                    true,
                    if carry_ecp_seed {
                        Some(&res.det_state)
                    } else {
                        None
                    },
                    kin_carry.as_ref(),
                )?
            );
            e_l_out.push(Complex64::new(re, im));
            if want_steps {
                ru_steps_c
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), _step)
                    .assign(&r_up_cur);
                rd_steps_c
                    .as_mut()
                    .unwrap()
                    .index_axis_mut(Axis(0), _step)
                    .assign(&r_dn_cur);
            }
            g_last = Some(g_cur_c.clone());
            gi_last = Some(gi_cur_c.clone());
            det_last = Some(res.det_state);
        }

        if crate::prof::on() {
            crate::prof::dump("run_energy_chunk");
        }
        let det = det_last.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("run_energy_chunk: n_steps must be >= 1")
        })?;
        Ok(ComplexChunkOut {
            e_l: e_l_out,
            w_l: w_l_out,
            geminal: g_last.unwrap(),
            geminal_inv: gi_last.unwrap(),
            r_up: r_up_cur,
            r_dn: r_dn_cur,
            final_key: key,
            accepted: accepted_total,
            ao_up: det.ao_up,
            ao_dn: det.ao_dn,
            paired_dn: det.paired_dn,
            paired_up_lambda: det.paired_up_lambda,
            ru_steps: ru_steps_c,
            rd_steps: rd_steps_c,
            inverse_refreshes,
        })
    }
}

// ---- dict builders (GIL held; called AFTER the parallel region) -------------------

/// Build the per-walker dict for the REAL chunk (the keys/shapes of the
/// `run_energy_chunk` real path).
fn real_out_to_pydict(py: Python<'_>, out: RealChunkOut) -> PyResult<Py<PyAny>> {
    use numpy::IntoPyArray;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("e_L", out.e_l)?;
    if let Some(w_l) = out.w_l {
        dict.set_item("w_L", w_l)?;
        dict.set_item("geminal", out.g_final.unwrap().into_pyarray(py))?;
        dict.set_item("geminal_inv", out.gi_final.unwrap().into_pyarray(py))?;
    } else {
        dict.set_item("geminal", out.g_steps.unwrap().into_pyarray(py))?;
        dict.set_item("geminal_inv", out.gi_steps.unwrap().into_pyarray(py))?;
    }
    dict.set_item("r_up", out.r_up.into_pyarray(py))?;
    dict.set_item("r_dn", out.r_dn.into_pyarray(py))?;
    dict.set_item("final_key", out.final_key.to_vec())?;
    dict.set_item("accepted", out.accepted)?;
    dict.set_item("ao_up", out.ao_up.into_pyarray(py))?;
    dict.set_item("ao_dn", out.ao_dn.into_pyarray(py))?;
    dict.set_item("paired_dn", out.paired_dn.into_pyarray(py))?;
    dict.set_item("paired_up_lambda", out.paired_up_lambda.into_pyarray(py))?;
    if let Some(hf_dr) = out.hf_dr {
        dict.set_item("hf_dR", hf_dr.into_pyarray(py))?;
        dict.set_item("hf_dr_up", out.hf_dru.unwrap().into_pyarray(py))?;
        dict.set_item("hf_dr_dn", out.hf_drd.unwrap().into_pyarray(py))?;
        dict.set_item("pp_dR", out.pp_dr.unwrap().into_pyarray(py))?;
        dict.set_item("pp_dr_up", out.pp_dru.unwrap().into_pyarray(py))?;
        dict.set_item("pp_dr_dn", out.pp_drd.unwrap().into_pyarray(py))?;
    }
    if let Some(ru_steps) = out.ru_steps {
        dict.set_item("r_up_steps", ru_steps.into_pyarray(py))?;
        dict.set_item("r_dn_steps", out.rd_steps.unwrap().into_pyarray(py))?;
    }
    if let Some(shf) = out.shf {
        dict.set_item("stress_HF", shf.into_pyarray(py))?;
        dict.set_item("stress_PP", out.spp.unwrap().into_pyarray(py))?;
    }
    dict.set_item("inverse_refreshes", out.inverse_refreshes)?;
    Ok(dict.into_any().unbind())
}

/// Build the per-walker dict for the COMPLEX chunk (the keys/shapes of the
/// `run_energy_chunk` complex path).
fn complex_out_to_pydict(py: Python<'_>, out: ComplexChunkOut) -> PyResult<Py<PyAny>> {
    use numpy::IntoPyArray;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("e_L", out.e_l.into_pyarray(py))?;
    dict.set_item("w_L", out.w_l)?;
    dict.set_item("geminal", out.geminal.into_pyarray(py))?;
    dict.set_item("geminal_inv", out.geminal_inv.into_pyarray(py))?;
    dict.set_item("r_up", out.r_up.into_pyarray(py))?;
    dict.set_item("r_dn", out.r_dn.into_pyarray(py))?;
    dict.set_item("final_key", out.final_key.to_vec())?;
    dict.set_item("accepted", out.accepted)?;
    dict.set_item("ao_up", out.ao_up.into_pyarray(py))?;
    dict.set_item("ao_dn", out.ao_dn.into_pyarray(py))?;
    dict.set_item("paired_dn", out.paired_dn.into_pyarray(py))?;
    dict.set_item("paired_up_lambda", out.paired_up_lambda.into_pyarray(py))?;
    if let Some(ru_steps) = out.ru_steps {
        dict.set_item("r_up_steps", ru_steps.into_pyarray(py))?;
        dict.set_item("r_dn_steps", out.rd_steps.unwrap().into_pyarray(py))?;
    }
    dict.set_item("inverse_refreshes", out.inverse_refreshes)?;
    Ok(dict.into_any().unbind())
}

// ---- Walker-axis stacking helpers with a leading nw axis. --------------------------

fn stack_scalar_vec_f64(v: &[Vec<f64>]) -> numpy::ndarray::Array2<f64> {
    let nw = v.len();
    let ns = if nw > 0 { v[0].len() } else { 0 };
    let mut out = numpy::ndarray::Array2::<f64>::zeros((nw, ns));
    for (w, e) in v.iter().enumerate() {
        for (i, &x) in e.iter().enumerate() {
            out[[w, i]] = x;
        }
    }
    out
}

fn stack_scalar_vec_c(v: &[Vec<numpy::Complex64>]) -> numpy::ndarray::Array2<numpy::Complex64> {
    let nw = v.len();
    let ns = if nw > 0 { v[0].len() } else { 0 };
    let mut out = numpy::ndarray::Array2::<numpy::Complex64>::zeros((nw, ns));
    for (w, e) in v.iter().enumerate() {
        for (i, &x) in e.iter().enumerate() {
            out[[w, i]] = x;
        }
    }
    out
}

fn stack_a2_f64(v: &[numpy::ndarray::Array2<f64>]) -> numpy::ndarray::Array3<f64> {
    use numpy::ndarray::Axis;
    let nw = v.len();
    let (r, c) = v[0].dim();
    let mut out = numpy::ndarray::Array3::<f64>::zeros((nw, r, c));
    for (w, a) in v.iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(a);
    }
    out
}

fn stack_a2_c(
    v: &[numpy::ndarray::Array2<numpy::Complex64>],
) -> numpy::ndarray::Array3<numpy::Complex64> {
    use numpy::ndarray::Axis;
    let nw = v.len();
    let (r, c) = v[0].dim();
    let mut out = numpy::ndarray::Array3::<numpy::Complex64>::zeros((nw, r, c));
    for (w, a) in v.iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(a);
    }
    out
}

fn stack_a3_f64(v: &[numpy::ndarray::Array3<f64>]) -> numpy::ndarray::Array4<f64> {
    use numpy::ndarray::Axis;
    let nw = v.len();
    let (a, b, c) = v[0].dim();
    let mut out = numpy::ndarray::Array4::<f64>::zeros((nw, a, b, c));
    for (w, x) in v.iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(x);
    }
    out
}

/// Batched real dictionary: same keys as `real_out_to_pydict`, every value gaining a
/// leading walker axis. `outs[w]` is walker `w`, in walker order.
fn real_batch_to_pydict(
    py: Python<'_>,
    outs: Vec<RealChunkOut>,
    native_rng: bool,
    want_force: bool,
    want_stress: bool,
    want_steps: bool,
) -> PyResult<Py<PyAny>> {
    use numpy::ndarray::{Array1, Array2};
    use numpy::IntoPyArray;
    let nw = outs.len();
    let dict = pyo3::types::PyDict::new(py);

    // e_L (nw, n_steps)
    let e_l: Vec<Vec<f64>> = outs.iter().map(|o| o.e_l.clone()).collect();
    dict.set_item("e_L", stack_scalar_vec_f64(&e_l).into_pyarray(py))?;

    if native_rng {
        let w_l: Vec<Vec<f64>> = outs.iter().map(|o| o.w_l.clone().unwrap()).collect();
        dict.set_item("w_L", stack_scalar_vec_f64(&w_l).into_pyarray(py))?;
        let g: Vec<Array2<f64>> = outs.iter().map(|o| o.g_final.clone().unwrap()).collect();
        let gi: Vec<Array2<f64>> = outs.iter().map(|o| o.gi_final.clone().unwrap()).collect();
        dict.set_item("geminal", stack_a2_f64(&g).into_pyarray(py))?;
        dict.set_item("geminal_inv", stack_a2_f64(&gi).into_pyarray(py))?;
    } else {
        // bitexact: per-step geminal stacks -> (nw, n_steps, n, m)
        let g: Vec<_> = outs.iter().map(|o| o.g_steps.clone().unwrap()).collect();
        let gi: Vec<_> = outs.iter().map(|o| o.gi_steps.clone().unwrap()).collect();
        dict.set_item("geminal", stack_a3_f64(&g).into_pyarray(py))?;
        dict.set_item("geminal_inv", stack_a3_f64(&gi).into_pyarray(py))?;
    }

    let r_up: Vec<_> = outs.iter().map(|o| o.r_up.clone()).collect();
    let r_dn: Vec<_> = outs.iter().map(|o| o.r_dn.clone()).collect();
    dict.set_item("r_up", stack_a2_f64(&r_up).into_pyarray(py))?;
    dict.set_item("r_dn", stack_a2_f64(&r_dn).into_pyarray(py))?;

    let mut fk = Array2::<u32>::zeros((nw, 2));
    for (w, o) in outs.iter().enumerate() {
        fk[[w, 0]] = o.final_key[0];
        fk[[w, 1]] = o.final_key[1];
    }
    dict.set_item("final_key", fk.into_pyarray(py))?;

    let mut acc = Array1::<i64>::zeros(nw);
    for (w, o) in outs.iter().enumerate() {
        acc[w] = o.accepted as i64;
    }
    dict.set_item("accepted", acc.into_pyarray(py))?;
    let refreshes: Vec<i64> = outs.iter().map(|o| o.inverse_refreshes as i64).collect();
    dict.set_item(
        "inverse_refreshes",
        Array1::from_vec(refreshes).into_pyarray(py),
    )?;

    let ao_up: Vec<_> = outs.iter().map(|o| o.ao_up.clone()).collect();
    let ao_dn: Vec<_> = outs.iter().map(|o| o.ao_dn.clone()).collect();
    let pdn: Vec<_> = outs.iter().map(|o| o.paired_dn.clone()).collect();
    let pul: Vec<_> = outs.iter().map(|o| o.paired_up_lambda.clone()).collect();
    dict.set_item("ao_up", stack_a2_f64(&ao_up).into_pyarray(py))?;
    dict.set_item("ao_dn", stack_a2_f64(&ao_dn).into_pyarray(py))?;
    dict.set_item("paired_dn", stack_a2_f64(&pdn).into_pyarray(py))?;
    dict.set_item("paired_up_lambda", stack_a2_f64(&pul).into_pyarray(py))?;

    if want_force {
        let hf_dr: Vec<_> = outs.iter().map(|o| o.hf_dr.clone().unwrap()).collect();
        let hf_dru: Vec<_> = outs.iter().map(|o| o.hf_dru.clone().unwrap()).collect();
        let hf_drd: Vec<_> = outs.iter().map(|o| o.hf_drd.clone().unwrap()).collect();
        let pp_dr: Vec<_> = outs.iter().map(|o| o.pp_dr.clone().unwrap()).collect();
        let pp_dru: Vec<_> = outs.iter().map(|o| o.pp_dru.clone().unwrap()).collect();
        let pp_drd: Vec<_> = outs.iter().map(|o| o.pp_drd.clone().unwrap()).collect();
        dict.set_item("hf_dR", stack_a3_f64(&hf_dr).into_pyarray(py))?;
        dict.set_item("hf_dr_up", stack_a3_f64(&hf_dru).into_pyarray(py))?;
        dict.set_item("hf_dr_dn", stack_a3_f64(&hf_drd).into_pyarray(py))?;
        dict.set_item("pp_dR", stack_a3_f64(&pp_dr).into_pyarray(py))?;
        dict.set_item("pp_dr_up", stack_a3_f64(&pp_dru).into_pyarray(py))?;
        dict.set_item("pp_dr_dn", stack_a3_f64(&pp_drd).into_pyarray(py))?;
    }
    if want_force || want_steps {
        let ru_s: Vec<_> = outs.iter().map(|o| o.ru_steps.clone().unwrap()).collect();
        let rd_s: Vec<_> = outs.iter().map(|o| o.rd_steps.clone().unwrap()).collect();
        dict.set_item("r_up_steps", stack_a3_f64(&ru_s).into_pyarray(py))?;
        dict.set_item("r_dn_steps", stack_a3_f64(&rd_s).into_pyarray(py))?;
    }
    if want_stress {
        let shf: Vec<_> = outs.iter().map(|o| o.shf.clone().unwrap()).collect();
        let spp: Vec<_> = outs.iter().map(|o| o.spp.clone().unwrap()).collect();
        dict.set_item("stress_HF", stack_a3_f64(&shf).into_pyarray(py))?;
        dict.set_item("stress_PP", stack_a3_f64(&spp).into_pyarray(py))?;
    }
    Ok(dict.into_any().unbind())
}

/// Batched COMPLEX dict: same keys as `complex_out_to_pydict`, leading walker axis.
fn complex_batch_to_pydict(
    py: Python<'_>,
    outs: Vec<ComplexChunkOut>,
    want_steps: bool,
) -> PyResult<Py<PyAny>> {
    use numpy::ndarray::{Array1, Array2};
    use numpy::IntoPyArray;
    let nw = outs.len();
    let dict = pyo3::types::PyDict::new(py);

    let e_l: Vec<_> = outs.iter().map(|o| o.e_l.clone()).collect();
    dict.set_item("e_L", stack_scalar_vec_c(&e_l).into_pyarray(py))?;
    let w_l: Vec<_> = outs.iter().map(|o| o.w_l.clone()).collect();
    dict.set_item("w_L", stack_scalar_vec_f64(&w_l).into_pyarray(py))?;

    let g: Vec<_> = outs.iter().map(|o| o.geminal.clone()).collect();
    let gi: Vec<_> = outs.iter().map(|o| o.geminal_inv.clone()).collect();
    dict.set_item("geminal", stack_a2_c(&g).into_pyarray(py))?;
    dict.set_item("geminal_inv", stack_a2_c(&gi).into_pyarray(py))?;

    let r_up: Vec<_> = outs.iter().map(|o| o.r_up.clone()).collect();
    let r_dn: Vec<_> = outs.iter().map(|o| o.r_dn.clone()).collect();
    dict.set_item("r_up", stack_a2_f64(&r_up).into_pyarray(py))?;
    dict.set_item("r_dn", stack_a2_f64(&r_dn).into_pyarray(py))?;

    let mut fk = Array2::<u32>::zeros((nw, 2));
    for (w, o) in outs.iter().enumerate() {
        fk[[w, 0]] = o.final_key[0];
        fk[[w, 1]] = o.final_key[1];
    }
    dict.set_item("final_key", fk.into_pyarray(py))?;

    let mut acc = Array1::<i64>::zeros(nw);
    for (w, o) in outs.iter().enumerate() {
        acc[w] = o.accepted as i64;
    }
    dict.set_item("accepted", acc.into_pyarray(py))?;
    let refreshes: Vec<i64> = outs.iter().map(|o| o.inverse_refreshes as i64).collect();
    dict.set_item(
        "inverse_refreshes",
        Array1::from_vec(refreshes).into_pyarray(py),
    )?;

    let ao_up: Vec<_> = outs.iter().map(|o| o.ao_up.clone()).collect();
    let ao_dn: Vec<_> = outs.iter().map(|o| o.ao_dn.clone()).collect();
    let pdn: Vec<_> = outs.iter().map(|o| o.paired_dn.clone()).collect();
    let pul: Vec<_> = outs.iter().map(|o| o.paired_up_lambda.clone()).collect();
    dict.set_item("ao_up", stack_a2_c(&ao_up).into_pyarray(py))?;
    dict.set_item("ao_dn", stack_a2_c(&ao_dn).into_pyarray(py))?;
    dict.set_item("paired_dn", stack_a2_c(&pdn).into_pyarray(py))?;
    dict.set_item("paired_up_lambda", stack_a2_c(&pul).into_pyarray(py))?;

    if want_steps {
        let ru_s: Vec<_> = outs.iter().map(|o| o.ru_steps.clone().unwrap()).collect();
        let rd_s: Vec<_> = outs.iter().map(|o| o.rd_steps.clone().unwrap()).collect();
        dict.set_item("r_up_steps", stack_a3_f64(&ru_s).into_pyarray(py))?;
        dict.set_item("r_dn_steps", stack_a3_f64(&rd_s).into_pyarray(py))?;
    }
    Ok(dict.into_any().unbind())
}

#[pymethods]
impl NativeObsCtx {
    /// Exact immutable policy snapshot supplied when this context was built.
    fn precision_policy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.precision_policy.to_pydict(py)?.into_any().unbind())
    }

    /// Actual arithmetic dispatch selected for this immutable context.
    fn precision_dispatch_support(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(native_precision_dispatch_policy(self.precision_policy)
            .to_pydict(py)?
            .into_any()
            .unbind())
    }

    /// Internal regression probe: all fixed-per-run complex observable families must
    /// point at one parsed geminal.  This intentionally reports UNIQUE backing objects,
    /// not Arc's reference count (temporary clones must not make the contract flaky).
    fn _debug_complex_geminal_storage_count(&self) -> usize {
        let mut ptrs = std::collections::HashSet::<usize>::new();
        let mut add = |gem: &Arc<complex_det::ComplexGeminal>| {
            ptrs.insert(Arc::as_ptr(gem) as usize);
        };
        if let Some(gem) = self.sweep.as_ref().and_then(|a| a.cgem.as_ref()) {
            add(gem);
        }
        if let Some(a) = self.clef_c.as_ref() {
            add(&a.cgem);
        }
        if let Some(a) = self.pbc_force_c.as_ref() {
            add(&a.gem);
        }
        if let Some(a) = self.pbc_lnpsi_c.as_ref() {
            add(&a.gem);
        }
        if let Some(a) = self.pbc_shf_c.as_ref() {
            add(&a.gem);
        }
        if let Some(a) = self.pbc_spp_c.as_ref() {
            add(&a.gem);
        }
        if let Some(gem) = self.gfmc.as_ref().and_then(|a| a.cgem.as_ref()) {
            add(gem);
        }
        if let Some(a) = self.param_deriv_c_dtn.as_ref() {
            add(&a.gem);
        }
        ptrs.len()
    }

    /// Internal memory-layout regression probe. A time-reversal complex B-spline
    /// geminal must own only two unique grid-sized allocations (real and imaginary),
    /// shared by up/down; independent spins own four.
    fn _debug_complex_bspline_coefficient_storage_count(&self) -> usize {
        self.sweep
            .as_ref()
            .and_then(|args| args.cgem.as_ref())
            .map_or(0, |gem| gem.bspline_coefficient_storage_count())
    }

    /// Context-taking real sweep: reuses the once-extracted `SweepArgs` instead of
    /// re-parsing the Geminal/Jastrow/J3 from Python each step. Bit-identical to the
    /// free `_mcmc_sweep` (same kernel, same owned data). Per-step args only.
    #[pyo3(signature = (positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, k0, k1, injected_normals, cell=None, n_max=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc_sweep(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<f64>,
        charges: PyReadonlyArray1<f64>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        geminal: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
        dt: f64,
        epsilon_as: f64,
        k0: u32,
        k1: u32,
        injected_normals: PyReadonlyArray1<f64>,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
    ) -> PyResult<Py<PyAny>> {
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let gem = s.gem.as_ref().ok_or_else(|| ctx_missing("sweep_real"))?;
        let j3_jm_view = s.j3_jm.as_ref().map(|m| m.view());
        let normals = injected_normals.as_slice()?;
        let cell_arr = cell.as_ref().map(|c| c.as_array());
        let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
        let res = mcmc::mcmc_sweep(
            gem,
            &s.jas,
            s.j3_orb.as_ref(),
            j3_jm_view.as_ref(),
            &positions.as_array(),
            charges.as_slice()?,
            &r_up.as_array(),
            &r_dn.as_array(),
            &geminal.as_array(),
            &geminal_inv.as_array(),
            dt,
            epsilon_as,
            [k0, k1],
            normals,
            normals.len(),
            cell_nmax,
            None,  // per-step ctx dispatch: fresh determinant state (bit-exact path)
            None,  // per-step ctx dispatch: fresh J3 state per sweep (bit-exact path)
            false, // per-step ctx dispatch is a parity path -> full eval (no cutoff)
        );
        dict_from_sweep(py, res)
    }

    /// Context-taking complex-twist sweep, matching `_mcmc_sweep_complex`.
    #[pyo3(signature = (positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, k0, k1, injected_normals, cell=None, n_max=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc_sweep_complex(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<f64>,
        charges: PyReadonlyArray1<f64>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        geminal: PyReadonlyArray2<numpy::Complex64>,
        geminal_inv: PyReadonlyArray2<numpy::Complex64>,
        dt: f64,
        epsilon_as: f64,
        k0: u32,
        k1: u32,
        injected_normals: PyReadonlyArray1<f64>,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
    ) -> PyResult<Py<PyAny>> {
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let cgem = s
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("sweep_complex"))?;
        let j3m_view = s.j3_jm.as_ref().map(|m| m.view());
        let normals = injected_normals.as_slice()?;
        let cell_arr = cell.as_ref().map(|c| c.as_array());
        let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
        let res = mcmc::mcmc_sweep_complex(
            cgem,
            &s.jas,
            s.j3c.as_ref(),
            j3m_view.as_ref(),
            &positions.as_array(),
            charges.as_slice()?,
            &r_up.as_array(),
            &r_dn.as_array(),
            &geminal.as_array(),
            &geminal_inv.as_array(),
            dt,
            epsilon_as,
            [k0, k1],
            normals,
            normals.len(),
            cell_nmax,
        );
        dict_from_sweep_complex(py, res)
    }

    /// e_L (real) reusing the ctx-owned Hamiltonian. `mpc_data` still crosses per-call
    /// (built once by the driver); `None` off the MPC path.
    #[pyo3(name = "e_L", signature = (r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
    fn e_l(
        &self,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<f64> {
        let a = self.clef.as_ref().ok_or_else(|| ctx_missing("clef"))?;
        a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            &geminal_inv.as_array(),
            false, // per-step ctx dispatch: exact mesh (bit-exact path)
            None,
            None,
            None,
            None,
        )
    }

    #[pyo3(name = "e_L_complex", signature = (r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
    fn e_l_complex(
        &self,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<numpy::Complex64>,
    ) -> PyResult<(f64, f64)> {
        let a = self
            .clef_c
            .as_ref()
            .ok_or_else(|| ctx_missing("clef_complex"))?;
        a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            &geminal_inv.as_array(),
            false,
            None,
        )
    }

    #[pyo3(name = "grad_e_L_ae")]
    fn grad_e_l_ae(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .mol_force
            .as_ref()
            .ok_or_else(|| ctx_missing("mol_force"))?;
        let (d_r, du, dd) = a.compute_ae(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            &geminal_inv.as_array(),
        );
        Ok(triple(py, d_r, du, dd))
    }

    #[pyo3(name = "grad_e_L_ecp")]
    fn grad_e_l_ecp(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .mol_force
            .as_ref()
            .ok_or_else(|| ctx_missing("mol_force"))?;
        let (d_r, du, dd) = a.compute_ecp(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            &geminal_inv.as_array(),
            gfmc_forces::FastScreen::default(),
        );
        Ok(triple(py, d_r, du, dd))
    }

    fn grad_ln_psi(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .mol_lnpsi
            .as_ref()
            .ok_or_else(|| ctx_missing("mol_lnpsi"))?;
        let (d_r, du, dd) = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            &geminal_inv.as_array(),
        );
        Ok(triple(py, d_r, du, dd))
    }

    #[pyo3(name = "grad_e_L_pbc", signature = (r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
    fn grad_e_l_pbc(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .pbc_force
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_force"))?;
        let (d_r, du, dd) = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            &geminal_inv.as_array(),
        )?;
        Ok(triple(py, d_r, du, dd))
    }

    fn grad_ln_psi_pbc(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .pbc_lnpsi
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_lnpsi"))?;
        let (d_r, du, dd) = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            &geminal_inv.as_array(),
        );
        Ok(triple(py, d_r, du, dd))
    }

    #[pyo3(name = "grad_e_L_pbc_complex", signature = (r_up_carts, r_dn_carts, weights, grid_points, geminal_inv=None))]
    fn grad_e_l_pbc_complex(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .pbc_force_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_force_complex"))?;
        let gi = geminal_inv.as_ref().map(|g| g.as_array());
        let (d_r, du, dd) = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            gi.as_ref(),
        )?;
        Ok(triple(py, d_r, du, dd))
    }

    #[pyo3(signature = (r_up_carts, r_dn_carts, geminal_inv=None))]
    fn grad_ln_psi_pbc_complex(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .pbc_lnpsi_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_lnpsi_complex"))?;
        let gi = geminal_inv.as_ref().map(|g| g.as_array());
        let (d_r, du, dd) = a.compute(&r_up_carts.as_array(), &r_dn_carts.as_array(), gi.as_ref());
        Ok(triple(py, d_r, du, dd))
    }

    #[pyo3(name = "grad_e_L_eps_pbc", signature = (r_up_carts, r_dn_carts, weights, grid_points, geminal_inv))]
    fn grad_e_l_eps_pbc(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .pbc_shf
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_hf"))?;
        let s = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            &geminal_inv.as_array(),
            true,
        )?;
        Ok(s.into_pyarray(py).into_any().unbind())
    }

    fn grad_ln_psi_eps_pbc(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .pbc_spp
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_pp"))?;
        let s = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            &geminal_inv.as_array(),
        );
        Ok(s.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(name = "grad_e_L_eps_pbc_complex", signature = (r_up_carts, r_dn_carts, weights, grid_points, geminal_inv=None))]
    fn grad_e_l_eps_pbc_complex(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
        geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .pbc_shf_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_hf_complex"))?;
        let gi = geminal_inv.as_ref().map(|g| g.as_array());
        let s = a.compute(
            &r_up_carts.as_array(),
            &r_dn_carts.as_array(),
            weights.as_slice()?,
            &grid_points.as_array(),
            gi.as_ref(),
            true,
        )?;
        Ok(s.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (r_up_carts, r_dn_carts, geminal_inv=None))]
    fn grad_ln_psi_eps_pbc_complex(
        &self,
        py: Python<'_>,
        r_up_carts: PyReadonlyArray2<f64>,
        r_dn_carts: PyReadonlyArray2<f64>,
        geminal_inv: Option<PyReadonlyArray2<numpy::Complex64>>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .pbc_spp_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_pp_complex"))?;
        let gi = geminal_inv.as_ref().map(|g| g.as_array());
        let s = a.compute(&r_up_carts.as_array(), &r_dn_carts.as_array(), gi.as_ref());
        Ok(s.into_pyarray(py).into_any().unbind())
    }

    // ======================================================================
    // Rayon batch twins. Each fans the
    // per-walker method's SHARED INNER FN (`SweepArgs`/`*Args::compute*`) over
    // the walker axis through the ONE `for_each_walker` harness -- no per-walker
    // kernel is re-implemented, so `batch[w] == per-walker loop[w]` bit-for-
    // bit and (per-walker threefry) independent of the thread count. Every
    // `PyReadonlyArray` is extracted to an owned array / Vec BEFORE the harness
    // releases the GIL (`PyReadonlyArray` is not `Send`); the ctx statics are
    // borrowed `&self` across the workers; `NativeObsCtx` is `Sync`.
    // ======================================================================

    /// Batched real sweep seed: the geminal `G (nw,n,m)` implied by the streaming
    /// caches `mcmc_sweep` builds at `r_up (nw,n_up,3)` / `r_dn (nw,n_dn,3)`
    /// ([`mcmc::sweep_seed_geminal`]). The driver seeds its carried `G` with this
    /// so the rank-1 delta updates never inherit a JAX-vs-native seed offset.
    #[pyo3(signature = (r_up, r_dn, cell=None, n_max=0))]
    fn sweep_seed_geminal_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let gem = s.gem.as_ref().ok_or_else(|| ctx_missing("sweep_real"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let cell_owned = cell.map(|c| c.as_array().to_owned());
        let nw = r_up.shape()[0];
        let seeds = walker_batch::for_each_walker(py, nw, |w| {
            let cell_v = cell_owned.as_ref().map(|c| c.view());
            let cell_nmax = cell_v.as_ref().map(|c| (c, n_max));
            mcmc::sweep_seed_geminal(
                gem,
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                cell_nmax,
            )
        });
        Ok(stack_axis0_f64(&seeds).into_pyarray(py).into_any().unbind())
    }

    /// Batched complex-twist sweep seed. The parsed complex geminal is owned by
    /// this context, so restart validation does not reparse a potentially large
    /// pGTO/PW/B-spline orbital once per walker.
    fn sweep_seed_geminal_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let gem = s
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("sweep_complex"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let nw = r_up.shape()[0];
        let seeds = walker_batch::for_each_walker(py, nw, |w| {
            crate::complex_det::compute_geminal_complex(
                gem,
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
            )
        });
        Ok(stack_axis0_c64(&seeds).into_pyarray(py).into_any().unbind())
    }

    /// Batched real sweep. `r_up (nw,n_up,3)`, `r_dn (nw,n_dn,3)`,
    /// `geminal / geminal_inv (nw,n,m)`, `keys (nw,2) u32`,
    /// `injected_normals (nw,num_moves)`; shared `positions / charges / cell`.
    #[pyo3(signature = (positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, keys, injected_normals, cell=None, n_max=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc_sweep_batch(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<f64>,
        charges: PyReadonlyArray1<f64>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
        dt: f64,
        epsilon_as: f64,
        keys: numpy::PyReadonlyArray2<u32>,
        injected_normals: numpy::PyReadonlyArray2<f64>,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let gem = s.gem.as_ref().ok_or_else(|| ctx_missing("sweep_real"))?;
        let positions = positions.as_array().to_owned();
        let charges = charges.as_array().to_vec();
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal = geminal.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let keys = keys.as_array().to_owned();
        let normals = injected_normals.as_array().to_owned();
        let cell_owned = cell.map(|c| c.as_array().to_owned());
        let nw = r_up.shape()[0];
        let num_moves = normals.shape()[1];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            let j3_jm_view = s.j3_jm.as_ref().map(|m| m.view());
            let cell_v = cell_owned.as_ref().map(|c| c.view());
            let cell_nmax = cell_v.as_ref().map(|c| (c, n_max));
            let ru = r_up.index_axis(Axis(0), w);
            let rd = r_dn.index_axis(Axis(0), w);
            let g = geminal.index_axis(Axis(0), w);
            let gi = geminal_inv.index_axis(Axis(0), w);
            let nrm = normals.index_axis(Axis(0), w);
            mcmc::mcmc_sweep(
                gem,
                &s.jas,
                s.j3_orb.as_ref(),
                j3_jm_view.as_ref(),
                &positions.view(),
                &charges,
                &ru,
                &rd,
                &g,
                &gi,
                dt,
                epsilon_as,
                [keys[[w, 0]], keys[[w, 1]]],
                nrm.as_slice().expect("contiguous normals row"),
                num_moves,
                cell_nmax,
                None,
                None,
                false,
            )
        });
        dict_from_sweep_batch(py, results)
    }

    /// Batched complex-twist sweep.
    #[pyo3(signature = (positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, keys, injected_normals, cell=None, n_max=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc_sweep_complex_batch(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<f64>,
        charges: PyReadonlyArray1<f64>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal: numpy::PyReadonlyArray3<numpy::Complex64>,
        geminal_inv: numpy::PyReadonlyArray3<numpy::Complex64>,
        dt: f64,
        epsilon_as: f64,
        keys: numpy::PyReadonlyArray2<u32>,
        injected_normals: numpy::PyReadonlyArray2<f64>,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let s = self.sweep.as_ref().ok_or_else(|| ctx_missing("sweep"))?;
        let cgem = s
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("sweep_complex"))?;
        let positions = positions.as_array().to_owned();
        let charges = charges.as_array().to_vec();
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal = geminal.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let keys = keys.as_array().to_owned();
        let normals = injected_normals.as_array().to_owned();
        let cell_owned = cell.map(|c| c.as_array().to_owned());
        let nw = r_up.shape()[0];
        let num_moves = normals.shape()[1];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            let j3m_view = s.j3_jm.as_ref().map(|m| m.view());
            let cell_v = cell_owned.as_ref().map(|c| c.view());
            let cell_nmax = cell_v.as_ref().map(|c| (c, n_max));
            let ru = r_up.index_axis(Axis(0), w);
            let rd = r_dn.index_axis(Axis(0), w);
            let g = geminal.index_axis(Axis(0), w);
            let gi = geminal_inv.index_axis(Axis(0), w);
            let nrm = normals.index_axis(Axis(0), w);
            mcmc::mcmc_sweep_complex(
                cgem,
                &s.jas,
                s.j3c.as_ref(),
                j3m_view.as_ref(),
                &positions.view(),
                &charges,
                &ru,
                &rd,
                &g,
                &gi,
                dt,
                epsilon_as,
                [keys[[w, 0]], keys[[w, 1]]],
                nrm.as_slice().expect("contiguous normals row"),
                num_moves,
                cell_nmax,
            )
        });
        dict_from_sweep_complex_batch(py, results)
    }

    /// Batched real e_L. `weights` shared; `grid_points (nw,ng,3)`,
    /// `geminal_inv (nw,n,m)` per walker. Returns `(nw,)` f64.
    #[pyo3(name = "e_L_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn e_l_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let a = self.clef.as_ref().ok_or_else(|| ctx_missing("clef"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
                false,
                None,
                None,
                None,
                None,
            )
        });
        let vals: Vec<f64> = results.into_iter().collect::<PyResult<Vec<f64>>>()?;
        Ok(numpy::ndarray::Array1::from(vals)
            .into_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Batched complex-twist e_L. Returns `(nw,)` complex128.
    #[pyo3(name = "e_L_complex_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn e_l_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<numpy::Complex64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let a = self
            .clef_c
            .as_ref()
            .ok_or_else(|| ctx_missing("clef_complex"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
                false,
                None,
            )
        });
        let pairs: Vec<(f64, f64)> = results.into_iter().collect::<PyResult<Vec<_>>>()?;
        let out: Vec<numpy::Complex64> = pairs
            .into_iter()
            .map(|(re, im)| numpy::Complex64::new(re, im))
            .collect();
        Ok(numpy::ndarray::Array1::from(out)
            .into_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Batched molecular AE HF force. No ECP mesh.
    #[pyo3(name = "grad_e_L_ae_batch")]
    fn grad_e_l_ae_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .mol_force
            .as_ref()
            .ok_or_else(|| ctx_missing("mol_force"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute_ae(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
            )
        });
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched molecular ECP HF force. `weights` shared; `grid_points (nw,ng,3)`.
    #[pyo3(name = "grad_e_L_ecp_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn grad_e_l_ecp_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .mol_force
            .as_ref()
            .ok_or_else(|| ctx_missing("mol_force"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute_ecp(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
                gfmc_forces::FastScreen::default(),
            )
        });
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched periodic real-Gamma HF force.
    #[pyo3(name = "grad_e_L_pbc_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn grad_e_l_pbc_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_force
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_force"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
            )
        });
        let results: Vec<Grad3> = results.into_iter().collect::<PyResult<Vec<_>>>()?;
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched complex-twist periodic HF force.
    #[pyo3(name = "grad_e_L_pbc_complex_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn grad_e_l_pbc_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<numpy::Complex64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_force_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_force_complex"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            let gi = geminal_inv.index_axis(Axis(0), w);
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                Some(&gi),
            )
        });
        let results: Vec<Grad3> = results.into_iter().collect::<PyResult<Vec<_>>>()?;
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched molecular Pulay WF gradient. No ECP mesh (pure WF deriv).
    fn grad_ln_psi_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .mol_lnpsi
            .as_ref()
            .ok_or_else(|| ctx_missing("mol_lnpsi"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
            )
        });
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched periodic real-Gamma Pulay.
    fn grad_ln_psi_pbc_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_lnpsi
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_lnpsi"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
            )
        });
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched complex-twist Pulay.
    fn grad_ln_psi_pbc_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<numpy::Complex64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_lnpsi_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_lnpsi_complex"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            let gi = geminal_inv.index_axis(Axis(0), w);
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                Some(&gi),
            )
        });
        Ok(grad3_batch_to_py(py, results))
    }

    /// Batched periodic real-Gamma HF stress. Returns `(nw,3,3)`.
    #[pyo3(name = "grad_e_L_eps_pbc_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn grad_e_l_eps_pbc_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_shf
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_hf"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
                true,
            )
        });
        let results: Vec<numpy::ndarray::Array2<f64>> =
            results.into_iter().collect::<PyResult<Vec<_>>>()?;
        Ok(stack_axis0_f64(&results)
            .into_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Batched periodic real-Gamma Pulay stress. Returns `(nw,3,3)`.
    fn grad_ln_psi_eps_pbc_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_spp
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_pp"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &geminal_inv.index_axis(Axis(0), w),
            )
        });
        Ok(stack_axis0_f64(&results)
            .into_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Batched complex-twist HF stress. Returns `(nw,3,3)`.
    #[pyo3(name = "grad_e_L_eps_pbc_complex_batch", signature = (r_up, r_dn, weights, grid_points, geminal_inv))]
    fn grad_e_l_eps_pbc_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<numpy::Complex64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_shf_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_hf_complex"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let weights = weights.as_array().to_vec();
        let grid_points = grid_points.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            let gi = geminal_inv.index_axis(Axis(0), w);
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                &weights,
                &grid_points.index_axis(Axis(0), w),
                Some(&gi),
                true,
            )
        });
        let results: Vec<numpy::ndarray::Array2<f64>> =
            results.into_iter().collect::<PyResult<Vec<_>>>()?;
        Ok(stack_axis0_f64(&results)
            .into_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Batched complex-twist Pulay stress. Returns `(nw,3,3)`.
    fn grad_ln_psi_eps_pbc_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        geminal_inv: numpy::PyReadonlyArray3<numpy::Complex64>,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Axis;
        let a = self
            .pbc_spp_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_pp_complex"))?;
        let r_up = r_up.as_array().to_owned();
        let r_dn = r_dn.as_array().to_owned();
        let geminal_inv = geminal_inv.as_array().to_owned();
        let nw = r_up.shape()[0];
        let results = walker_batch::for_each_walker(py, nw, |w| {
            let gi = geminal_inv.index_axis(Axis(0), w);
            a.compute(
                &r_up.index_axis(Axis(0), w),
                &r_dn.index_axis(Axis(0), w),
                Some(&gi),
            )
        });
        Ok(stack_axis0_f64(&results)
            .into_pyarray(py)
            .into_any()
            .unbind())
    }

    // ----- O_k = d ln|Psi|/dc context methods for `want_param_deriv`. ----------------
    // Each reuses the once-parsed `ParamDerivArgs` and calls the same inner kernel
    // the free wrapper calls, so the result is byte-identical to the free path.

    fn pd_ok_j1(&self, r_up: PyReadonlyArray2<f64>, r_dn: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j1 = a.j1.as_ref().ok_or_else(|| ctx_missing("param_deriv j1"))?;
        Ok(param_grad::grad_j1_param(
            j1,
            &r_up.as_array(),
            &r_dn.as_array(),
        ))
    }

    fn pd_ok_j2(&self, r_up: PyReadonlyArray2<f64>, r_dn: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j2 = a.j2.as_ref().ok_or_else(|| ctx_missing("param_deriv j2"))?;
        Ok(param_grad::grad_j2_param(
            j2,
            &r_up.as_array(),
            &r_dn.as_array(),
        ))
    }

    fn pd_ok_j3(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let orb = a.j3_pd_orb()?;
        let g = param_grad::grad_j3_jmatrix(orb, &r_up.as_array(), &r_dn.as_array());
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    fn pd_ok_dtn(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let d = if let Some(a) = self.param_deriv.as_ref() {
            a.dtn.as_ref()
        } else {
            self.param_deriv_c_dtn.as_ref().and_then(|a| a.dtn.as_ref())
        }
        .ok_or_else(|| ctx_missing("param_deriv dtn"))?;
        let (x, y, z) =
            param_grad_dtn::compute_dtn_param_grads(d, &r_up.as_array(), &r_dn.as_array());
        Ok((opt_arr(py, x), opt_arr(py, y), opt_arr(py, z)))
    }

    fn pd_ok_mo(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let (gu, gd) = param_grad::grad_ln_det_mo(&a.gem, &r_up.as_array(), &r_dn.as_array());
        Ok((
            gu.into_pyarray(py).into_any().unbind(),
            gd.into_pyarray(py).into_any().unbind(),
        ))
    }

    fn pd_ok_lambda_diag(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let (d, u) =
            param_grad::grad_ln_det_lambda_diag(&a.gem, &r_up.as_array(), &r_dn.as_array());
        Ok((
            d.into_pyarray(py).into_any().unbind(),
            u.into_pyarray(py).into_any().unbind(),
        ))
    }

    fn pd_ok_lambda(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let g = param_grad::grad_ln_det_lambda(&a.gem, &r_up.as_array(), &r_dn.as_array());
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    // ----- dE_L/dc kinetic (dT/dc) context methods. ----------------------------------

    fn pd_el_j1(&self, r_up: PyReadonlyArray2<f64>, r_dn: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j1 = a.j1.as_ref().ok_or_else(|| ctx_missing("param_deriv j1"))?;
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        Ok(param_grad_el::grad_t_j1(
            &a.gem,
            jas,
            j1,
            a.j3_orb.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
        ))
    }

    fn pd_el_j2(&self, r_up: PyReadonlyArray2<f64>, r_dn: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j2 = a.j2.as_ref().ok_or_else(|| ctx_missing("param_deriv j2"))?;
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        Ok(param_grad_el::grad_t_j2(
            &a.gem,
            jas,
            j2,
            a.j3_orb.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
        ))
    }

    fn pd_el_j3(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let g = param_grad_el::grad_t_j3(
            &a.gem,
            jas,
            orb,
            &jm.view(),
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    fn pd_el_dtn(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if let Some(a) = self.param_deriv.as_ref() {
            let jas = a
                .jas
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv jas"))?;
            let d = a
                .dtn
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv dtn"))?;
            let j3v = a.j3_jm.as_ref().map(|m| m.view());
            let (x, y, z) = param_grad_dtn::compute_dtn_kinetic_grads(
                d,
                &a.gem,
                jas,
                a.j3_orb.as_ref(),
                j3v.as_ref(),
                &r_up.as_array(),
                &r_dn.as_array(),
            );
            Ok((opt_arr(py, x), opt_arr(py, y), opt_arr(py, z)))
        } else {
            let a = self
                .param_deriv_c_dtn
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv_complex_dtn"))?;
            let d = a
                .dtn
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv complex dtn"))?;
            let j3v = a.j3m.as_ref().map(|m| m.view());
            let (x, y, z) = param_grad_twist::grad_t_dtn_twist(
                d,
                &a.gem,
                &a.jas,
                a.j3c.as_ref(),
                j3v.as_ref(),
                &r_up.as_array(),
                &r_dn.as_array(),
            );
            Ok((opt_arr_c(py, x), opt_arr_c(py, y), opt_arr_c(py, z)))
        }
    }

    #[pyo3(signature = (r_up, r_dn, geminal_inv=None))]
    fn pd_el_mo(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        geminal_inv: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let gi = geminal_inv.as_ref().map(|value| value.as_array());
        let (gu, gd) = param_grad_el::grad_t_mo_gi(
            &a.gem,
            a.jas.as_ref(),
            a.j3_orb.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
            gi.as_ref(),
        );
        Ok((
            gu.into_pyarray(py).into_any().unbind(),
            gd.into_pyarray(py).into_any().unbind(),
        ))
    }

    fn pd_el_lambda_diag(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let (d, u) = param_grad_el::grad_t_lambda_diag(
            &a.gem,
            a.jas.as_ref(),
            a.j3_orb.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok((
            d.into_pyarray(py).into_any().unbind(),
            u.into_pyarray(py).into_any().unbind(),
        ))
    }

    fn pd_el_lambda(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let g = param_grad_el::grad_t_lambda(
            &a.gem,
            a.jas.as_ref(),
            a.j3_orb.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    // ----- dV_ecp_nl/dc mesh add for each walker's rotated grid. ---------------------

    fn pd_el_ecp_j1(
        &self,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<f64> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j1 = a.j1.as_ref().ok_or_else(|| ctx_missing("param_deriv j1"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        Ok(param_grad_ecp::grad_ecp_j1(
            &a.gem,
            jas,
            j3,
            j1,
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        ))
    }

    fn pd_el_ecp_j2(
        &self,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<f64> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j2 = a.j2.as_ref().ok_or_else(|| ctx_missing("param_deriv j2"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        Ok(param_grad_ecp::grad_ecp_j2(
            &a.gem,
            jas,
            j3,
            j2,
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        ))
    }

    fn pd_el_ecp_j3(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let g = param_grad_ecp::grad_ecp_j3(
            &a.gem,
            jas,
            orb,
            &jm.view(),
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        );
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    fn pd_el_ecp_dtn(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if let Some(a) = self.param_deriv.as_ref() {
            let jas = a
                .jas
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv jas"))?;
            let d = a
                .dtn
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv dtn"))?;
            let e = a
                .ecp
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
            let cell_v = e.cell.as_ref().map(|c| c.view());
            let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
            let j3o = match (&a.j3_orb, &a.j3_jm) {
                (Some(o), Some(m)) => Some((o, m.view())),
                _ => None,
            };
            let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
            let (x, y, z) = param_grad_ecp::grad_ecp_dtn(
                d,
                &a.gem,
                jas,
                j3,
                &r_up.as_array(),
                &r_dn.as_array(),
                &e.pos.view(),
                &e.nuc,
                &e.angmom,
                &e.exps,
                &e.coefs,
                &e.powers,
                weights.as_slice()?,
                &grid_points.as_array(),
                cell_nmax,
            );
            Ok((opt_arr(py, x), opt_arr(py, y), opt_arr(py, z)))
        } else {
            let a = self
                .param_deriv_c_dtn
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv_complex_dtn"))?;
            let d = a
                .dtn
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv complex dtn"))?;
            let e = a
                .ecp
                .as_ref()
                .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
            let cell_v = e.cell.as_ref().map(|c| c.view());
            let j3v = a.j3m.as_ref().map(|m| m.view());
            let (x, y, z) = param_grad_twist::grad_ecp_dtn_twist(
                d,
                &a.gem,
                &a.jas,
                a.j3c.as_ref(),
                j3v.as_ref(),
                &r_up.as_array(),
                &r_dn.as_array(),
                &e.pos.view(),
                &e.nuc,
                &e.angmom,
                &e.exps,
                &e.coefs,
                &e.powers,
                weights.as_slice()?,
                &grid_points.as_array(),
                cell_v.as_ref(),
                e.n_max,
            );
            Ok((opt_arr_c(py, x), opt_arr_c(py, y), opt_arr_c(py, z)))
        }
    }

    /// True when this context carries the once-parsed complex-twist Jastrow
    /// parameter-derivative tables served by the `pd_*_complex` methods
    /// (`complex=True, want_param_deriv=True` with a Jastrow).
    fn has_complex_param_deriv(&self) -> bool {
        self.param_deriv_c_dtn.is_some()
    }

    // ----- Complex-twist Jastrow c-axis methods ------------------------------
    // Same kernels as the free `grad_eL_*_complex` / `grad_eL_ecp_*_complex` /
    // `grad_ln_Psi_jastrow_3b_jmatrix_complex` wrappers (J1 / J2 in source mode:
    // complex scalars), served from the once-built `ComplexDtnPdArgs` instead of a
    // per-call `from_py` of the geminal, Jastrow, J3 orbital and ECP tables.

    fn pd_ok_j3_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j3c = a
            .j3c
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let g = param_grad_twist::grad_j3_jmatrix_complex(j3c, &r_up.as_array(), &r_dn.as_array());
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    fn pd_el_j1_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j1 = a
            .jas
            .one_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j1"))?;
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let v = param_grad_twist::grad_t_j1_twist(
            &a.gem,
            &a.jas,
            j1,
            a.j3c.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok(pyo3::types::PyComplex::from_doubles(py, v.re, v.im)
            .into_any()
            .unbind())
    }

    fn pd_el_j2_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j2 = a
            .jas
            .two_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j2"))?;
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let v = param_grad_twist::grad_t_j2_twist(
            &a.gem,
            &a.jas,
            j2,
            a.j3c.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok(pyo3::types::PyComplex::from_doubles(py, v.re, v.im)
            .into_any()
            .unbind())
    }

    fn pd_el_j3_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j3c = a
            .j3c
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let jm = a
            .j3m
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let g = param_grad_twist::grad_t_j3_twist(
            &a.gem,
            &a.jas,
            j3c,
            &jm.view(),
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    fn pd_el_ecp_j1_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j1 = a
            .jas
            .one_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j1"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let v = param_grad_twist::grad_ecp_j1_twist(
            &a.gem,
            &a.jas,
            j1,
            a.j3c.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_v.as_ref(),
            e.n_max,
        );
        Ok(pyo3::types::PyComplex::from_doubles(py, v.re, v.im)
            .into_any()
            .unbind())
    }

    fn pd_el_ecp_j2_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j2 = a
            .jas
            .two_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j2"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let v = param_grad_twist::grad_ecp_j2_twist(
            &a.gem,
            &a.jas,
            j2,
            a.j3c.as_ref(),
            j3v.as_ref(),
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_v.as_ref(),
            e.n_max,
        );
        Ok(pyo3::types::PyComplex::from_doubles(py, v.re, v.im)
            .into_any()
            .unbind())
    }

    fn pd_el_ecp_j3_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j3c = a
            .j3c
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let jm = a
            .j3m
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let g = param_grad_twist::grad_ecp_j3_twist(
            &a.gem,
            &a.jas,
            j3c,
            &jm.view(),
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_v.as_ref(),
            e.n_max,
        );
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    fn pd_el_ecp_mo(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let (gu, gd) = param_grad_ecp::grad_ecp_mo(
            &a.gem,
            jas,
            j3,
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        );
        Ok((
            gu.into_pyarray(py).into_any().unbind(),
            gd.into_pyarray(py).into_any().unbind(),
        ))
    }

    fn pd_el_ecp_lambda(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let g = param_grad_ecp::grad_ecp_lambda(
            &a.gem,
            jas,
            j3,
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        );
        Ok(g.into_pyarray(py).into_any().unbind())
    }

    // ----- Geminal AO radial basis-optimization parameter gradients.
    // Returns (exp_grad, coeff_grad), each up+dn concatenated to match the reducer's
    // `lambda_basis_{exp,coeff}` = concat([..._up, ..._dn], axis=-1). Threads the same
    // radial kernels the free `grad_{ln_det,eL,eL_ecp}_wrt_radial`
    // wrappers call, reusing the once-parsed geminal / radial map / cell / ecp. --------

    /// O_k radial block (ln_Psi): d ln|det| / d{exp,coeff}.
    fn pd_ok_basis(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a
            .basis
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv basis (non-GTO geminal leaf)"))?;
        let cell_v = b.cell.as_ref().map(|c| c.view());
        let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_basis::grad_radial(
            &a.gem,
            &b.a2s_up,
            b.nsh_up,
            &b.a2s_dn,
            b.nsh_dn,
            &r_up.as_array(),
            &r_dn.as_array(),
            cell_v.as_ref(),
        );
        Ok((cat_pyarray(py, gz_up, gz_dn), cat_pyarray(py, gc_up, gc_dn)))
    }

    /// dE_L/dc radial block (kinetic dT/dc): d T / d{exp,coeff}.
    fn pd_el_basis(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a
            .basis
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv basis (non-GTO geminal leaf)"))?;
        let cell_v = b.cell.as_ref().map(|c| c.view());
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_basis::grad_radial_kinetic(
            &a.gem,
            a.jas.as_ref(),
            a.j3_orb.as_ref(),
            j3v.as_ref(),
            &b.a2s_up,
            b.nsh_up,
            &b.a2s_dn,
            b.nsh_dn,
            &r_up.as_array(),
            &r_dn.as_array(),
            cell_v.as_ref(),
        );
        Ok((cat_pyarray(py, gz_up, gz_dn), cat_pyarray(py, gc_up, gc_dn)))
    }

    /// dE_L/dc radial ECP-nonlocal mesh add (per-walker RT-rotated grid).
    fn pd_el_ecp_basis(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let b = a
            .basis
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv basis (non-GTO geminal leaf)"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_ecp::grad_ecp_radial(
            &a.gem,
            jas,
            j3,
            &b.a2s_up,
            b.nsh_up,
            &b.a2s_dn,
            b.nsh_dn,
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        );
        Ok((cat_pyarray(py, gz_up, gz_dn), cat_pyarray(py, gc_up, gc_dn)))
    }

    // ----- J3 three-body-Jastrow orbital AO radial basis-optimization gradients
    // for `opt_J3_basis_exp` and `opt_J3_basis_coeff`.
    // Returns `(exp_grad, coeff_grad)` as a SINGLE block each (not up+dn concatenated
    // like the geminal `pd_*_basis`): the three-body Jastrow has ONE orbital shared by
    // both spins, so the up and dn electron contributions are SUMMED inside the kernel
    // and the published block carries that one leaf's shell rank -- exactly the shape
    // `Jastrow_data.collect_param_grads` reads off
    // `grad.jastrow_three_body_data.ao_{exponents,coefficients}`. -----------------

    /// O_k block: `d ln|Psi| / d(J3 radial)` = `dJ3/d(J3 radial)`
    /// (`basis.grad_J3_wrt_basis`).
    fn pd_ok_j3_basis(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a.j3_basis.as_ref().ok_or_else(|| {
            ctx_missing("param_deriv j3_basis (non-GTO three-body-Jastrow orbital leaf)")
        })?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let (gz, gc) = param_grad_basis::grad_j3_radial(
            orb,
            &jm.view(),
            &b.a2s,
            b.nsh,
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok((vec_pyarray(py, gz), vec_pyarray(py, gc)))
    }

    /// dE_L/dc block: the kinetic `dT/d(J3 radial)` (`basis.grad_T_wrt_J3_basis`).
    fn pd_el_j3_basis(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a.j3_basis.as_ref().ok_or_else(|| {
            ctx_missing("param_deriv j3_basis (non-GTO three-body-Jastrow orbital leaf)")
        })?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let (gz, gc) = param_grad_basis::grad_j3_radial_kinetic(
            &a.gem,
            jas,
            orb,
            &jm.view(),
            &b.a2s,
            b.nsh,
            &r_up.as_array(),
            &r_dn.as_array(),
        );
        Ok((vec_pyarray(py, gz), vec_pyarray(py, gc)))
    }

    /// dE_L/dc `dV_ecp_nl/dc` mesh add for the J3 radial block (per-walker RT-rotated grid).
    fn pd_el_ecp_j3_basis(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a.j3_basis.as_ref().ok_or_else(|| {
            ctx_missing("param_deriv j3_basis (non-GTO three-body-Jastrow orbital leaf)")
        })?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let (gz, gc) = param_grad_ecp::grad_ecp_j3_radial(
            &a.gem,
            jas,
            orb,
            &jm.view(),
            &b.a2s,
            b.nsh,
            &r_up.as_array(),
            &r_dn.as_array(),
            &e.pos.view(),
            &e.nuc,
            &e.angmom,
            &e.exps,
            &e.coefs,
            &e.powers,
            weights.as_slice()?,
            &grid_points.as_array(),
            cell_nmax,
        );
        Ok((vec_pyarray(py, gz), vec_pyarray(py, gc)))
    }

    // ================= Batched pd_* twins. ======================================
    // Each reuses the SAME inner kernel its scalar sibling calls, fanned over the
    // walker axis by `batch_map` / `batch_map_ecp` (the `for_each_walker` harness).
    // The `&ParamDerivArgs` extraction (fallible) happens ONCE before the parallel
    // region so the closure is infallible. Only the real path reaches these (ctx =
    // NativeObsCtx); the complex adapter has no `_batch` methods.

    // ---- J1 ----
    fn pd_ok_j1_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j1 = a.j1.as_ref().ok_or_else(|| ctx_missing("param_deriv j1"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad::grad_j1_param(j1, ru, rd)
        });
        Ok(stack_f64(py, out))
    }

    fn pd_el_j1_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j1 = a.j1.as_ref().ok_or_else(|| ctx_missing("param_deriv j1"))?;
        let gem = &a.gem;
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_el::grad_t_j1(gem, jas, j1, j3_orb, j3v.as_ref(), ru, rd)
        });
        Ok(stack_f64(py, out))
    }

    fn pd_el_ecp_j1_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j1 = a.j1.as_ref().ok_or_else(|| ctx_missing("param_deriv j1"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_ecp::grad_ecp_j1(
                    gem,
                    jas,
                    j3,
                    j1,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                )
            },
        )?;
        Ok(stack_f64(py, out))
    }

    // ---- J2 ----
    fn pd_ok_j2_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let j2 = a.j2.as_ref().ok_or_else(|| ctx_missing("param_deriv j2"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad::grad_j2_param(j2, ru, rd)
        });
        Ok(stack_f64(py, out))
    }

    fn pd_el_j2_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j2 = a.j2.as_ref().ok_or_else(|| ctx_missing("param_deriv j2"))?;
        let gem = &a.gem;
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_el::grad_t_j2(gem, jas, j2, j3_orb, j3v.as_ref(), ru, rd)
        });
        Ok(stack_f64(py, out))
    }

    fn pd_el_ecp_j2_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let j2 = a.j2.as_ref().ok_or_else(|| ctx_missing("param_deriv j2"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_ecp::grad_ecp_j2(
                    gem,
                    jas,
                    j3,
                    j2,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                )
            },
        )?;
        Ok(stack_f64(py, out))
    }

    // ---- J3 j_matrix ----
    fn pd_ok_j3_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let orb = a.j3_pd_orb()?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad::grad_j3_jmatrix(orb, ru, rd)
        });
        Ok(stack_arr2(py, out))
    }

    fn pd_el_j3_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let gem = &a.gem;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_el::grad_t_j3(gem, jas, orb, &jm.view(), ru, rd)
        });
        Ok(stack_arr2(py, out))
    }

    fn pd_el_ecp_j3_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_ecp::grad_ecp_j3(
                    gem,
                    jas,
                    orb,
                    &jm.view(),
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                )
            },
        )?;
        Ok(stack_arr2(py, out))
    }

    // ---- DTN (u/chi/f) -- real path only (complex delegates through the adapter) ----
    fn pd_ok_dtn_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let d = a
            .dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv dtn"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_dtn::compute_dtn_param_grads(d, ru, rd)
        });
        let (av, bv, gv) = unzip3(out);
        Ok((
            stack_opt_arr2(py, av),
            stack_opt_arr2(py, bv),
            stack_opt_arr2(py, gv),
        ))
    }

    fn pd_el_dtn_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let d = a
            .dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv dtn"))?;
        let gem = &a.gem;
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_dtn::compute_dtn_kinetic_grads(d, gem, jas, j3_orb, j3v.as_ref(), ru, rd)
        });
        let (av, bv, gv) = unzip3(out);
        Ok((
            stack_opt_arr2(py, av),
            stack_opt_arr2(py, bv),
            stack_opt_arr2(py, gv),
        ))
    }

    fn pd_el_ecp_dtn_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let d = a
            .dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv dtn"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_ecp::grad_ecp_dtn(
                    d,
                    gem,
                    jas,
                    j3,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                )
            },
        )?;
        let (av, bv, gv) = unzip3(out);
        Ok((
            stack_opt_arr2(py, av),
            stack_opt_arr2(py, bv),
            stack_opt_arr2(py, gv),
        ))
    }

    // ---- Complex-twist Jastrow families: walker-batch twins of the scalar
    // `pd_*_complex` / complex `pd_*_dtn` methods (same kernels and tables, walkers on
    // rayon). Walker w of each batch equals the scalar call on walker w. ----
    fn pd_ok_j3_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j3c = a
            .j3c
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_twist::grad_j3_jmatrix_complex(j3c, ru, rd)
        });
        Ok(stack_arr2(py, out))
    }

    fn pd_el_j1_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j1 = a
            .jas
            .one_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j1"))?;
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_twist::grad_t_j1_twist(
                &a.gem,
                &a.jas,
                j1,
                a.j3c.as_ref(),
                j3v.as_ref(),
                ru,
                rd,
            )
        });
        Ok(stack_c64(py, out))
    }

    fn pd_el_j2_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j2 = a
            .jas
            .two_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j2"))?;
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_twist::grad_t_j2_twist(
                &a.gem,
                &a.jas,
                j2,
                a.j3c.as_ref(),
                j3v.as_ref(),
                ru,
                rd,
            )
        });
        Ok(stack_c64(py, out))
    }

    fn pd_el_j3_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j3c = a
            .j3c
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let jm = a
            .j3m
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_twist::grad_t_j3_twist(&a.gem, &a.jas, j3c, &jm.view(), ru, rd)
        });
        Ok(stack_arr2(py, out))
    }

    fn pd_el_ecp_j1_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j1 = a
            .jas
            .one_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j1"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_twist::grad_ecp_j1_twist(
                    &a.gem,
                    &a.jas,
                    j1,
                    a.j3c.as_ref(),
                    j3v.as_ref(),
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_v.as_ref(),
                    e.n_max,
                )
            },
        )?;
        Ok(stack_c64(py, out))
    }

    fn pd_el_ecp_j2_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j2 = a
            .jas
            .two_body
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j2"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_twist::grad_ecp_j2_twist(
                    &a.gem,
                    &a.jas,
                    j2,
                    a.j3c.as_ref(),
                    j3v.as_ref(),
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_v.as_ref(),
                    e.n_max,
                )
            },
        )?;
        Ok(stack_c64(py, out))
    }

    fn pd_el_ecp_j3_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self.complex_pd_args()?;
        let j3c = a
            .j3c
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let jm = a
            .j3m
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex j3"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_twist::grad_ecp_j3_twist(
                    &a.gem,
                    &a.jas,
                    j3c,
                    &jm.view(),
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_v.as_ref(),
                    e.n_max,
                )
            },
        )?;
        Ok(stack_arr2(py, out))
    }

    fn pd_ok_dtn_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let d = self
            .complex_pd_args()?
            .dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex dtn"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_dtn::compute_dtn_param_grads(d, ru, rd)
        });
        let (av, bv, gv) = unzip3(out);
        Ok((
            stack_opt_arr2(py, av),
            stack_opt_arr2(py, bv),
            stack_opt_arr2(py, gv),
        ))
    }

    fn pd_el_dtn_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self.complex_pd_args()?;
        let d = a
            .dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex dtn"))?;
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_twist::grad_t_dtn_twist(
                d,
                &a.gem,
                &a.jas,
                a.j3c.as_ref(),
                j3v.as_ref(),
                ru,
                rd,
            )
        });
        let (av, bv, gv) = unzip3(out);
        Ok((
            stack_opt_arr2_c(py, av),
            stack_opt_arr2_c(py, bv),
            stack_opt_arr2_c(py, gv),
        ))
    }

    fn pd_el_ecp_dtn_complex_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let a = self.complex_pd_args()?;
        let d = a
            .dtn
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex dtn"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv complex ecp"))?;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let j3v = a.j3m.as_ref().map(|m| m.view());
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_twist::grad_ecp_dtn_twist(
                    d,
                    &a.gem,
                    &a.jas,
                    a.j3c.as_ref(),
                    j3v.as_ref(),
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_v.as_ref(),
                    e.n_max,
                )
            },
        )?;
        let (av, bv, gv) = unzip3(out);
        Ok((
            stack_opt_arr2_c(py, av),
            stack_opt_arr2_c(py, bv),
            stack_opt_arr2_c(py, gv),
        ))
    }

    // ---- MO (direct mo_coefficients, SD/AGPn) ----
    fn pd_ok_mo_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let gem = &a.gem;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad::grad_ln_det_mo(gem, ru, rd)
        });
        let (uv, dv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr2(py, uv), stack_arr2(py, dv)))
    }

    #[pyo3(signature = (r_up_b, r_dn_b, geminal_inv=None))]
    fn pd_el_mo_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        geminal_inv: Option<numpy::PyReadonlyArray3<f64>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let gem = &a.gem;
        let jas = a.jas.as_ref();
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let ru = r_up_b.as_array().to_owned();
        let rd = r_dn_b.as_array().to_owned();
        let gi = geminal_inv.map(|value| value.as_array().to_owned());
        let nw = ru.shape()[0];
        if rd.shape()[0] != nw || gi.as_ref().is_some_and(|value| value.shape()[0] != nw) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pd_el_mo_batch: r_up, r_dn, and geminal_inv walker axes must match",
            ));
        }
        let out = walker_batch::for_each_walker(py, nw, |w| {
            let ru_w = ru.index_axis(numpy::ndarray::Axis(0), w);
            let rd_w = rd.index_axis(numpy::ndarray::Axis(0), w);
            let gi_w = gi
                .as_ref()
                .map(|value| value.index_axis(numpy::ndarray::Axis(0), w));
            param_grad_el::grad_t_mo_gi(gem, jas, j3_orb, j3v.as_ref(), &ru_w, &rd_w, gi_w.as_ref())
        });
        let (uv, dv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr2(py, uv), stack_arr2(py, dv)))
    }

    fn pd_el_ecp_mo_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_ecp::grad_ecp_mo(
                    gem,
                    jas,
                    j3,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                )
            },
        )?;
        let (uv, dv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr2(py, uv), stack_arr2(py, dv)))
    }

    // ---- lambda_diag (AGPn: diag + unpaired) -- NO ecp add (matches scalar path) ----
    fn pd_ok_lambda_diag_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let gem = &a.gem;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad::grad_ln_det_lambda_diag(gem, ru, rd)
        });
        let (dv, uv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, dv), stack_arr2(py, uv)))
    }

    fn pd_el_lambda_diag_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let gem = &a.gem;
        let jas = a.jas.as_ref();
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_el::grad_t_lambda_diag(gem, jas, j3_orb, j3v.as_ref(), ru, rd)
        });
        let (dv, uv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, dv), stack_arr2(py, uv)))
    }

    // ---- lambda_matrix (AGP) ----
    fn pd_ok_lambda_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let gem = &a.gem;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad::grad_ln_det_lambda(gem, ru, rd)
        });
        Ok(stack_arr2(py, out))
    }

    fn pd_el_lambda_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let gem = &a.gem;
        let jas = a.jas.as_ref();
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            param_grad_el::grad_t_lambda(gem, jas, j3_orb, j3v.as_ref(), ru, rd)
        });
        Ok(stack_arr2(py, out))
    }

    fn pd_el_ecp_lambda_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<Py<PyAny>> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                param_grad_ecp::grad_ecp_lambda(
                    gem,
                    jas,
                    j3,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                )
            },
        )?;
        Ok(stack_arr2(py, out))
    }

    // ---- geminal AO radial basis-opt (up+dn concatenated) ----
    fn pd_ok_basis_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a
            .basis
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv basis (non-GTO geminal leaf)"))?;
        let gem = &a.gem;
        let cell_v = b.cell.as_ref().map(|c| c.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_basis::grad_radial(
                gem,
                &b.a2s_up,
                b.nsh_up,
                &b.a2s_dn,
                b.nsh_dn,
                ru,
                rd,
                cell_v.as_ref(),
            );
            (cat1(gz_up, gz_dn), cat1(gc_up, gc_dn))
        });
        let (ev, cv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, ev), stack_arr1(py, cv)))
    }

    fn pd_el_basis_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a
            .basis
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv basis (non-GTO geminal leaf)"))?;
        let gem = &a.gem;
        let jas = a.jas.as_ref();
        let j3_orb = a.j3_orb.as_ref();
        let j3v = a.j3_jm.as_ref().map(|m| m.view());
        let cell_v = b.cell.as_ref().map(|c| c.view());
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_basis::grad_radial_kinetic(
                gem,
                jas,
                j3_orb,
                j3v.as_ref(),
                &b.a2s_up,
                b.nsh_up,
                &b.a2s_dn,
                b.nsh_dn,
                ru,
                rd,
                cell_v.as_ref(),
            );
            (cat1(gz_up, gz_dn), cat1(gc_up, gc_dn))
        });
        let (ev, cv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, ev), stack_arr1(py, cv)))
    }

    fn pd_el_ecp_basis_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let b = a
            .basis
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv basis (non-GTO geminal leaf)"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let j3o = match (&a.j3_orb, &a.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3 = j3o.as_ref().map(|(o, m)| (*o, m));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                let ((gz_up, gc_up), (gz_dn, gc_dn)) = param_grad_ecp::grad_ecp_radial(
                    gem,
                    jas,
                    j3,
                    &b.a2s_up,
                    b.nsh_up,
                    &b.a2s_dn,
                    b.nsh_dn,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                );
                (cat1(gz_up, gz_dn), cat1(gc_up, gc_dn))
            },
        )?;
        let (ev, cv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, ev), stack_arr1(py, cv)))
    }

    // ---- J3 orbital AO radial basis-opt (single shared block) ----
    fn pd_ok_j3_basis_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a.j3_basis.as_ref().ok_or_else(|| {
            ctx_missing("param_deriv j3_basis (non-GTO three-body-Jastrow orbital leaf)")
        })?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            let (gz, gc) = param_grad_basis::grad_j3_radial(orb, &jm.view(), &b.a2s, b.nsh, ru, rd);
            (
                numpy::ndarray::Array1::from_vec(gz),
                numpy::ndarray::Array1::from_vec(gc),
            )
        });
        let (ev, cv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, ev), stack_arr1(py, cv)))
    }

    fn pd_el_j3_basis_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a.j3_basis.as_ref().ok_or_else(|| {
            ctx_missing("param_deriv j3_basis (non-GTO three-body-Jastrow orbital leaf)")
        })?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let gem = &a.gem;
        let out = batch_map(py, &r_up_b, &r_dn_b, |ru, rd| {
            let (gz, gc) = param_grad_basis::grad_j3_radial_kinetic(
                gem,
                jas,
                orb,
                &jm.view(),
                &b.a2s,
                b.nsh,
                ru,
                rd,
            );
            (
                numpy::ndarray::Array1::from_vec(gz),
                numpy::ndarray::Array1::from_vec(gc),
            )
        });
        let (ev, cv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, ev), stack_arr1(py, cv)))
    }

    fn pd_el_ecp_j3_basis_batch(
        &self,
        py: Python<'_>,
        r_up_b: numpy::PyReadonlyArray3<f64>,
        r_dn_b: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        grid_points_b: numpy::PyReadonlyArray3<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let a = self
            .param_deriv
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv"))?;
        let b = a.j3_basis.as_ref().ok_or_else(|| {
            ctx_missing("param_deriv j3_basis (non-GTO three-body-Jastrow orbital leaf)")
        })?;
        let jas = a
            .jas
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv jas"))?;
        let orb = a.j3_pd_orb()?;
        let jm = a
            .j3_jm
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv j3"))?;
        let e = a
            .ecp
            .as_ref()
            .ok_or_else(|| ctx_missing("param_deriv ecp"))?;
        let gem = &a.gem;
        let cell_v = e.cell.as_ref().map(|c| c.view());
        let cell_nmax = cell_v.as_ref().map(|v| (v, e.n_max));
        let out = batch_map_ecp(
            py,
            &r_up_b,
            &r_dn_b,
            &weights,
            &grid_points_b,
            |ru, rd, w, gp| {
                let (gz, gc) = param_grad_ecp::grad_ecp_j3_radial(
                    gem,
                    jas,
                    orb,
                    &jm.view(),
                    &b.a2s,
                    b.nsh,
                    ru,
                    rd,
                    &e.pos.view(),
                    &e.nuc,
                    &e.angmom,
                    &e.exps,
                    &e.coefs,
                    &e.powers,
                    w,
                    gp,
                    cell_nmax,
                );
                (
                    numpy::ndarray::Array1::from_vec(gz),
                    numpy::ndarray::Array1::from_vec(gc),
                )
            },
        )?;
        let (ev, cv): (Vec<_>, Vec<_>) = out.into_iter().unzip();
        Ok((stack_arr1(py, ev), stack_arr1(py, cv)))
    }

    /// Per-walker whole-run energy chunk. This thin wrapper marshals
    /// the PyO3 arrays to views and delegates to the shared `run_energy_chunk_*_body`
    /// (the ONE copy of the chunk kernel logic, also driven per-walker by
    /// `run_energy_chunk_batch`). Behavior and dictionary keys match the per-step route.
    ///
    /// Two chain modes (see the body docs): `native_rng` (FAST, production default) vs
    /// the caller-injected JAX normals/mesh (BITEXACT). `complex=true` routes to the
    /// complex-twist body (fast-mode + energy-only).
    #[pyo3(signature = (positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, k0, k1, injected_normals, grid_points, weights, num_moves, cell=None, n_max=0, native_rng=false, random_mesh=false, base_grid_points=None, n_steps_fast=0, want_force=false, want_stress=false, want_steps=false, complex=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_energy_chunk(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<f64>,
        charges: PyReadonlyArray1<f64>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        geminal: PyReadonlyArray2<f64>,
        geminal_inv: PyReadonlyArray2<f64>,
        dt: f64,
        epsilon_as: f64,
        k0: u32,
        k1: u32,
        injected_normals: numpy::PyReadonlyArray2<f64>, // (n_steps, num_moves) -- bitexact only
        grid_points: numpy::PyReadonlyArray3<f64>,      // (n_steps, n_grid, 3) -- bitexact only
        weights: PyReadonlyArray1<f64>,
        num_moves: usize,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
        native_rng: bool,
        random_mesh: bool,
        base_grid_points: Option<PyReadonlyArray2<f64>>, // (n_grid, 3) -- fast only
        n_steps_fast: usize,
        want_force: bool,
        want_stress: bool,
        want_steps: bool,
        complex: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let positions_arr = positions.as_array();
        let charges_slice = charges.as_slice()?;
        let weights_slice = weights.as_slice()?;
        let cell_arr = cell.as_ref().map(|c| c.as_array());
        let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
        let base_arr = base_grid_points.as_ref().map(|b| b.as_array());

        if complex {
            if !native_rng {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_energy_chunk: complex-twist chunk supports fast mode only \
                     (native_rng=true); bitexact complex stays on the per-step dispatch",
                ));
            }
            if want_force || want_stress {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_energy_chunk: complex-twist chunk is energy-only \
                     (complex force/stress stays on the per-step dispatch)",
                ));
            }
            let base = base_arr.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "run_energy_chunk: fast mode (native_rng) requires base_grid_points",
                )
            })?;
            let out = self.run_energy_chunk_complex_body(
                positions_arr,
                charges_slice,
                r_up.as_array(),
                r_dn.as_array(),
                dt,
                epsilon_as,
                [k0, k1],
                weights_slice,
                num_moves,
                cell_nmax,
                random_mesh,
                base,
                n_steps_fast,
                want_steps,
                inverse_refresh_threshold,
            )?;
            return complex_out_to_pydict(py, out);
        }

        let out = self.run_energy_chunk_real_body(
            positions_arr,
            charges_slice,
            r_up.as_array(),
            r_dn.as_array(),
            geminal.as_array(),
            geminal_inv.as_array(),
            dt,
            epsilon_as,
            [k0, k1],
            injected_normals.as_array(),
            grid_points.as_array(),
            weights_slice,
            num_moves,
            cell_nmax,
            native_rng,
            !native_rng,
            random_mesh,
            base_arr,
            n_steps_fast,
            want_force,
            want_stress,
            want_steps,
            inverse_refresh_threshold,
        )?;
        real_out_to_pydict(py, out)
    }

    /// Walker-batched whole-run energy chunk. Leading
    /// walker axis on every per-walker array (`r_up (nw,n_up,3)`, `r_dn (nw,n_dn,3)`,
    /// `geminal`/`geminal_inv (nw,n,m)`, `keys (nw,2) u32`, and -- bitexact only --
    /// `injected_normals (nw,n_steps,num_moves)` / `grid_points (nw,n_steps,n_grid,3)`);
    /// per-run statics (`positions`, `charges`, `weights`, `cell`, `base_grid_points`)
    /// stay un-batched. Fans the SAME `run_energy_chunk_*_body` the per-walker entry uses
    /// across walkers through `walker_batch::for_each_walker` (the ONE rayon site), then
    /// stacks the owned per-walker outputs into a batched dict whose keys match the
    /// per-walker entry with a leading walker axis. Per-walker output
    /// is bit-identical to the per-walker entry independent of thread count (per-walker
    /// threefry key -> walker-local RNG stream; `NativeObsCtx: Sync` carries no scratch).
    #[pyo3(signature = (positions, charges, r_up, r_dn, geminal, geminal_inv, dt, epsilon_as, keys, injected_normals, grid_points, weights, num_moves, cell=None, n_max=0, native_rng=false, random_mesh=false, base_grid_points=None, n_steps_fast=0, want_force=false, want_stress=false, want_steps=false, complex=false, test_injected_inputs=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_energy_chunk_batch(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<f64>,
        charges: PyReadonlyArray1<f64>,
        r_up: numpy::PyReadonlyArray3<f64>,    // (nw, n_up, 3)
        r_dn: numpy::PyReadonlyArray3<f64>,    // (nw, n_dn, 3)
        geminal: numpy::PyReadonlyArray3<f64>, // (nw, n, m)
        geminal_inv: numpy::PyReadonlyArray3<f64>,
        dt: f64,
        epsilon_as: f64,
        keys: numpy::PyReadonlyArray2<u32>,             // (nw, 2)
        injected_normals: numpy::PyReadonlyArray3<f64>, // (nw, n_steps, num_moves) -- bitexact
        grid_points: numpy::PyReadonlyArray4<f64>,      // (nw, n_steps, n_grid, 3) -- bitexact
        weights: PyReadonlyArray1<f64>,
        num_moves: usize,
        cell: Option<PyReadonlyArray2<f64>>,
        n_max: i32,
        native_rng: bool,
        random_mesh: bool,
        base_grid_points: Option<PyReadonlyArray2<f64>>, // (n_grid, 3) -- fast only
        n_steps_fast: usize,
        want_force: bool,
        want_stress: bool,
        want_steps: bool,
        complex: bool,
        test_injected_inputs: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::{Array2, Array3, Axis};
        // Extract every `PyReadonlyArray` to a Send ndarray view before the
        // GIL-released Rayon region because `PyReadonlyArray` is not Send.
        let positions_arr = positions.as_array();
        let charges_slice = charges.as_slice()?;
        let weights_slice = weights.as_slice()?;
        let cell_arr = cell.as_ref().map(|c| c.as_array());
        let cell_nmax = cell_arr.as_ref().map(|c| (c, n_max));
        let base_arr = base_grid_points.as_ref().map(|b| b.as_array());
        let base_view = base_arr.as_ref().map(|b| b.view());
        let r_up_all = r_up.as_array(); // (nw, n_up, 3)
        let r_dn_all = r_dn.as_array();
        let gem_all = geminal.as_array();
        let gi_all = geminal_inv.as_array();
        let keys_all = keys.as_array(); // (nw, 2)
        let normals_all = injected_normals.as_array(); // (nw, n_steps, num_moves)
        let grids_all = grid_points.as_array(); // (nw, n_steps, n_grid, 3)
        let nw = r_up_all.shape()[0];
        // Fast mode never indexes injected_normals / grid_points; hand every walker an
        // empty view (the body ignores it) rather than slicing the (nw,0,0)/(nw,0,0,0)
        // placeholders the driver passes.
        let empty2 = Array2::<f64>::zeros((0, 0));
        let empty3 = Array3::<f64>::zeros((0, 0, 0));

        if complex {
            if !native_rng || test_injected_inputs {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_energy_chunk_batch: complex-twist chunk supports fast mode only \
                     (native_rng=true); bitexact complex stays on the per-step dispatch",
                ));
            }
            if want_force || want_stress {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_energy_chunk_batch: complex-twist chunk is energy-only \
                     (complex force/stress stays on the per-step dispatch)",
                ));
            }
            let base = base_view.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "run_energy_chunk_batch: fast mode (native_rng) requires base_grid_points",
                )
            })?;
            let outs = walker_batch::for_each_walker(py, nw, |w| {
                let key0 = [keys_all[[w, 0]], keys_all[[w, 1]]];
                self.run_energy_chunk_complex_body(
                    positions_arr,
                    charges_slice,
                    r_up_all.index_axis(Axis(0), w),
                    r_dn_all.index_axis(Axis(0), w),
                    dt,
                    epsilon_as,
                    key0,
                    weights_slice,
                    num_moves,
                    cell_nmax,
                    random_mesh,
                    base,
                    n_steps_fast,
                    want_steps,
                    inverse_refresh_threshold,
                )
            });
            let outs = outs.into_iter().collect::<PyResult<Vec<_>>>()?;
            return complex_batch_to_pydict(py, outs, want_steps);
        }

        let injected_inputs = !native_rng || test_injected_inputs;
        let outs = walker_batch::for_each_walker(py, nw, |w| {
            let key0 = [keys_all[[w, 0]], keys_all[[w, 1]]];
            let normals_w = if injected_inputs {
                normals_all.index_axis(Axis(0), w)
            } else {
                empty2.view()
            };
            let grids_w = if injected_inputs {
                grids_all.index_axis(Axis(0), w)
            } else {
                empty3.view()
            };
            crate::timed!(
                "chunk.walker",
                self.run_energy_chunk_real_body(
                    positions_arr,
                    charges_slice,
                    r_up_all.index_axis(Axis(0), w),
                    r_dn_all.index_axis(Axis(0), w),
                    gem_all.index_axis(Axis(0), w),
                    gi_all.index_axis(Axis(0), w),
                    dt,
                    epsilon_as,
                    key0,
                    normals_w,
                    grids_w,
                    weights_slice,
                    num_moves,
                    cell_nmax,
                    native_rng,
                    injected_inputs,
                    random_mesh,
                    base_view,
                    n_steps_fast,
                    want_force,
                    want_stress,
                    want_steps,
                    inverse_refresh_threshold,
                )
            )
        });
        let outs = outs.into_iter().collect::<PyResult<Vec<_>>>()?;
        real_batch_to_pydict(py, outs, native_rng, want_force, want_stress, want_steps)
    }

    /// Test probe: drive the production GFMC interval chain state
    /// through the prescribed hops (`hop_is_up[k]`, `hop_electron[k]`,
    /// `hop_positions[k]`) and return, for every hop count in `checkpoints`
    /// (0 = seed), a dict with the carried determinant grad/lap (`carried`), a
    /// from-scratch evaluation at the same configuration (`fresh`) and the
    /// matrix-product contraction of the carried cache (`twin`), each a
    /// `(grad_up, grad_dn, lap_up, lap_dn)` tuple. Dispatches on the ctx's
    /// real or complex geminal; not used by any driver.
    #[pyo3(signature = (r_up, r_dn, hop_is_up, hop_electron, hop_positions, checkpoints))]
    #[allow(clippy::too_many_arguments)]
    fn debug_det_grad_lap_chain(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        hop_is_up: Vec<bool>,
        hop_electron: Vec<usize>,
        hop_positions: PyReadonlyArray2<f64>,
        checkpoints: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let pos = hop_positions.as_array();
        if hop_is_up.len() != hop_electron.len()
            || pos.nrows() != hop_is_up.len()
            || pos.ncols() != 3
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "debug_det_grad_lap_chain: hop arrays must have matching lengths and (n_hops, 3) positions",
            ));
        }
        let hops: Vec<(bool, usize, [f64; 3])> = (0..hop_is_up.len())
            .map(|k| {
                (
                    hop_is_up[k],
                    hop_electron[k],
                    [pos[[k, 0]], pos[[k, 1]], pos[[k, 2]]],
                )
            })
            .collect();
        let to_err = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let list = pyo3::types::PyList::empty(py);
        if let Some(cgem) = g.cgem.as_ref() {
            let j3cm_view = g.j3cm.as_ref().map(|m| m.view());
            let rows = gfmc::debug_det_grad_lap_chain_complex(
                cgem,
                &g.jdata,
                g.j3c.as_ref(),
                j3cm_view.as_ref(),
                &r_up.as_array(),
                &r_dn.as_array(),
                &hops,
                &checkpoints,
            )
            .map_err(to_err)?;
            for (hop, carried, fresh, twin) in rows {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("hop", hop)?;
                for (name, v) in [("carried", carried), ("fresh", fresh), ("twin", twin)] {
                    d.set_item(
                        name,
                        (
                            v.0.into_pyarray(py),
                            v.1.into_pyarray(py),
                            v.2.into_pyarray(py),
                            v.3.into_pyarray(py),
                        ),
                    )?;
                }
                list.append(d)?;
            }
        } else {
            let gem = g
                .gem
                .as_ref()
                .ok_or_else(|| ctx_missing("GFMC geminal (build_obs_ctx want_gfmc=True)"))?;
            let j3 = match (&g.j3_orb, &g.j3_jm) {
                (Some(o), Some(m)) => Some((o, m.view())),
                _ => None,
            };
            let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
            let pbc = g.ewald.as_ref().map(|a| a.as_ref());
            let rows = gfmc::debug_det_grad_lap_chain(
                gem,
                &g.jdata,
                j3_ref,
                &r_up.as_array(),
                &r_dn.as_array(),
                pbc.as_ref(),
                &hops,
                &checkpoints,
            )
            .map_err(to_err)?;
            for (hop, carried, fresh, twin) in rows {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("hop", hop)?;
                for (name, v) in [("carried", carried), ("fresh", fresh), ("twin", twin)] {
                    d.set_item(
                        name,
                        (
                            v.0.into_pyarray(py),
                            v.1.into_pyarray(py),
                            v.2.into_pyarray(py),
                            v.3.into_pyarray(py),
                        ),
                    )?;
                }
                list.append(d)?;
            }
        }
        Ok(list.into_any().unbind())
    }

    /// Test probe: drive the production projection-scoped GFMC
    /// interval state (real `MeshFastState` or complex `MeshFastStateC`, per the
    /// ctx geminal) through `n_hops` sub-steps with fixed kinetic / ECP
    /// rotations: every sub-step runs the production interval evaluation
    /// (kinetic mesh, ECP mesh, e_L), picks the hop (`policy` = "categorical":
    /// the production FN hop with `uniforms[k]`; "adversarial": the kinetic
    /// candidate of smallest |element| above `floor` of an electron other than
    /// the previous hop's) and applies the production `advance_after_hop`.
    /// Returns, for every hop count in `checkpoints`, a dict with the
    /// configuration and three snapshots (`carried`, `fresh`, `oracle`), each a
    /// dict of named arrays (G^-1, winv tables, DTN candidate tables, DTN
    /// grad/lap cache, mesh elements, e_L / b_L). Not used by any driver.
    #[pyo3(signature = (r_up, r_dn, alat, kinetic_rt, ecp_rt, weights, base_grid_points, nn, dltmove, policy, floor, uniforms, n_hops, checkpoints))]
    #[allow(clippy::too_many_arguments)]
    fn debug_mesh_chain(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        alat: f64,
        kinetic_rt: PyReadonlyArray2<f64>,
        ecp_rt: PyReadonlyArray2<f64>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        nn: usize,
        dltmove: bool,
        policy: &str,
        floor: f64,
        uniforms: Vec<f64>,
        n_hops: usize,
        checkpoints: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let to_err = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let rt3 = |a: &PyReadonlyArray2<f64>| -> PyResult<[[f64; 3]; 3]> {
            let v = a.as_array();
            if v.shape() != [3, 3] {
                return Err(to_err("debug_mesh_chain: rotations must be (3, 3)".into()));
            }
            let mut m = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] = v[[i, j]];
                }
            }
            Ok(m)
        };
        let kin_rt = rt3(&kinetic_rt)?;
        let e_rt = rt3(&ecp_rt)?;
        let pol = match policy {
            "categorical" => gfmc::ProbeHopPolicy::Categorical,
            "adversarial" => gfmc::ProbeHopPolicy::Adversarial { floor },
            other => {
                return Err(to_err(format!(
                    "debug_mesh_chain: unknown policy {other:?}"
                )))
            }
        };
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let pbc = g.ewald.as_ref().map(|a| a.as_ref());
        let rows = if let Some(cgem) = g.cgem.as_ref() {
            let pbc = pbc.as_ref().ok_or_else(|| {
                to_err("debug_mesh_chain: complex geminal needs a periodic ctx".into())
            })?;
            let j3cm_view = g.j3cm.as_ref().map(|m| m.view());
            gfmc::debug_mesh_chain_complex(
                cgem,
                &g.jdata,
                g.j3c.as_ref(),
                j3cm_view.as_ref(),
                &g.positions.view(),
                &g.eff_charges,
                &r_up.as_array(),
                &r_dn.as_array(),
                alat,
                &kin_rt,
                &e_rt,
                ecp_params.as_ref(),
                pbc,
                pol,
                &uniforms,
                n_hops,
                &checkpoints,
            )
            .map_err(to_err)?
        } else {
            let gem = g
                .gem
                .as_ref()
                .ok_or_else(|| ctx_missing("GFMC geminal (build_obs_ctx want_gfmc=True)"))?;
            let j3 = match (&g.j3_orb, &g.j3_jm) {
                (Some(o), Some(m)) => Some((o, m.view())),
                _ => None,
            };
            let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
            gfmc::debug_mesh_chain(
                gem,
                &g.jdata,
                j3_ref,
                &g.positions.view(),
                &g.eff_charges,
                &r_up.as_array(),
                &r_dn.as_array(),
                alat,
                &kin_rt,
                &e_rt,
                ecp_params.as_ref(),
                pbc.as_ref(),
                pol,
                &uniforms,
                n_hops,
                &checkpoints,
            )
            .map_err(to_err)?
        };
        let snap =
            |s: Vec<(&'static str, gfmc::ProbeArray)>| -> PyResult<Bound<'_, pyo3::types::PyDict>> {
                let d = pyo3::types::PyDict::new(py);
                for (name, arr) in s {
                    match arr {
                        gfmc::ProbeArray::Real(shape, data) => {
                            let a = numpy::ndarray::ArrayD::from_shape_vec(
                                numpy::ndarray::IxDyn(&shape),
                                data,
                            )
                            .map_err(|e| to_err(e.to_string()))?;
                            d.set_item(name, a.into_pyarray(py))?;
                        }
                        gfmc::ProbeArray::Complex(shape, data) => {
                            let a = numpy::ndarray::ArrayD::from_shape_vec(
                                numpy::ndarray::IxDyn(&shape),
                                data,
                            )
                            .map_err(|e| to_err(e.to_string()))?;
                            d.set_item(name, a.into_pyarray(py))?;
                        }
                    }
                }
                Ok(d)
            };
        let list = pyo3::types::PyList::empty(py);
        for row in rows {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("hop", row.hop)?;
            d.set_item("r_up", row.r_up.into_pyarray(py))?;
            d.set_item("r_dn", row.r_dn.into_pyarray(py))?;
            d.set_item("carried", snap(row.carried)?)?;
            d.set_item("fresh", snap(row.fresh)?)?;
            d.set_item("oracle", snap(row.oracle)?)?;
            list.append(d)?;
        }
        Ok(list.into_any().unbind())
    }

    /// GFMC_n whole-interval native projection for one walker: the fixed
    /// `num_per_meas` sub-steps of the discrete projector (`w_L *= -b_L/(V_diag - E_scf)`)
    /// + FN categorical hop, then the final-config e_L / V_diag measurement at the last
    /// sub-step's mesh rotation -- one native call per branching step per walker (the GFMC
    /// analogue of `run_energy_chunk`). Python keeps the MPI branching / reconfiguration.
    ///
    /// 2-mode chain (mirroring the MCMC chunk's divergence-point discipline):
    ///  * BITEXACT: `injected_rts` = the JAX-drawn `(num_per_meas, 3, 3)` `R^T` stack (the
    ///    ONLY XLA-non-reproducible stream: the rotation trig). The key chain and the hop
    ///    uniforms replay jqmc's `_split_step_keys` threefry stream bit-exactly, so the
    ///    walk reproduces the JAX per-step chain (kernel-parity fp floor only) and
    ///    `final_key` is byte-equal.
    ///  * FAST: `injected_rts` empty (`(0, 3, 3)`) -> the rotation is built natively from
    ///    the SAME bit-exact threefry angles + native trig (statistically guarded).
    #[pyo3(signature = (r_up, r_dn, alat, e_scf, num_per_meas, k0, k1, injected_rts, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_n(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        alat: f64,
        e_scf: f64,
        num_per_meas: usize,
        k0: u32,
        k1: u32,
        injected_rts: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        // Evaluate the per-interval HF/Pulay force and optional periodic
        // stress) on the final config with the interval's last mesh rotation, folded into
        // stress in this one native call, analogous to `run_energy_chunk`.
        want_force: bool,
        want_stress: bool,
        // The debug/oracle twin disables the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let j3 = match (&g.j3_orb, &g.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
        let pbc = g.ewald.as_ref().map(|a| a.as_ref());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let rts_arr = injected_rts.as_array();
        let inj = if rts_arr.shape()[0] == 0 {
            None
        } else {
            Some(&rts_arr)
        };
        if inj.is_some() && rts_arr.shape()[0] != num_per_meas {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gfmc_projection_n: injected_rts must be (num_per_meas, 3, 3)",
            ));
        }
        let gem = g.gem.as_ref().ok_or_else(|| {
            ctx_missing(
                "GFMC real geminal (complex-twist run must use run_gfmc_projection_n_complex)",
            )
        })?;
        let out = gfmc::gfmc_interval_n(
            gem,
            &g.jdata,
            j3_ref,
            &g.positions.view(),
            &g.eff_charges,
            &r_up.as_array(),
            &r_dn.as_array(),
            alat,
            e_scf,
            num_per_meas,
            [k0, k1],
            inj,
            random_mesh,
            ecp_params.as_ref(),
            pbc.as_ref(),
            debug_full_recompute,
        );
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("w_L", out.w_l)?;
        dict.set_item("e_L", out.e_l)?;
        dict.set_item("V_diag", out.v_diag)?;
        gfmc_force_block(
            self,
            py,
            &dict,
            &out.r_up.view(),
            &out.r_dn.view(),
            &out.last_rt,
            screen_projection_ginv(self, out.chain_state.as_ref(), inverse_refresh_threshold),
            out.chain_state.as_ref(),
            alat,
            want_force,
            want_stress,
            w_slice,
            &gp_view,
            dltmove,
        )?;
        dict.set_item("r_up", out.r_up.into_pyarray(py))?;
        dict.set_item("r_dn", out.r_dn.into_pyarray(py))?;
        dict.set_item("final_key", out.final_key.to_vec())?;
        dict.set_item("n_substeps", out.n_substeps)?;
        dict.set_item("n_pot_updates", out.n_pot_updates)?;
        Ok(dict.into_any().unbind())
    }

    /// GFMC_t whole-interval native tau-projection for one walker, fast chain
    /// mode ONLY: the escape-time loop (`tau_update = min(tau_left, ln(1-xi)/b_L)`,
    /// `w_L *= exp(-tau_update * e_L)`) + FN hop until tau exhausts. BITEXACT mode for
    /// GFMC_t is BY DESIGN the pure-JAX per-step path (driver gate): the sub-step COUNT
    /// itself is an fp function of `b_L` through `log`, so a byte-for-byte chain contract
    /// vs XLA cannot exist. Returns the last sub-step's e_L
    /// (jqmc `_projection_t` semantics) and the per-walker sub-step count
    /// (`n_substeps` -> `projection_counter`).
    #[pyo3(signature = (r_up, r_dn, alat, tau, k0, k1, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_t(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        alat: f64,
        tau: f64,
        k0: u32,
        k1: u32,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let j3 = match (&g.j3_orb, &g.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
        let pbc = g.ewald.as_ref().map(|a| a.as_ref());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let gem = g.gem.as_ref().ok_or_else(|| {
            ctx_missing(
                "GFMC real geminal (complex-twist run must use run_gfmc_projection_t_complex)",
            )
        })?;
        let out = crate::timed!(
            "gfmc.interval_total",
            gfmc::gfmc_interval_t(
                gem,
                &g.jdata,
                j3_ref,
                &g.positions.view(),
                &g.eff_charges,
                &r_up.as_array(),
                &r_dn.as_array(),
                alat,
                tau,
                [k0, k1],
                random_mesh,
                ecp_params.as_ref(),
                pbc.as_ref(),
                debug_full_recompute,
            )
        );
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("w_L", out.w_l)?;
        dict.set_item("e_L", out.e_l)?;
        gfmc_force_block(
            self,
            py,
            &dict,
            &out.r_up.view(),
            &out.r_dn.view(),
            &out.last_rt,
            screen_projection_ginv(self, out.chain_state.as_ref(), inverse_refresh_threshold),
            out.chain_state.as_ref(),
            alat,
            want_force,
            want_stress,
            w_slice,
            &gp_view,
            dltmove,
        )?;
        dict.set_item("r_up", out.r_up.into_pyarray(py))?;
        dict.set_item("r_dn", out.r_dn.into_pyarray(py))?;
        dict.set_item("final_key", out.final_key.to_vec())?;
        dict.set_item("n_substeps", out.n_substeps)?;
        dict.set_item("n_pot_updates", out.n_pot_updates)?;
        Ok(dict.into_any().unbind())
    }

    /// Complex-twist GFMC_n whole-interval native projection for one walker.
    /// Drives `gfmc_interval_n_complex` over the verified complex e_L kernels
    /// (`gfmc_lrdmc_{ae,ecp}_complex`). Returns the complex measurement e_L, the
    /// Re-projected `w_L` / `V_diag`, the final config, key, sub-step count, and -- when
    /// `want_force` / `want_stress` -- the per-interval complex-twist HF/Pulay force and
    /// periodic stress folded out on the final config with the interval's last mesh
    /// rotation (`gfmc_force_block_complex`).
    ///
    /// Argument set and order mirror the real `run_gfmc_projection_n` exactly, including
    /// `injected_rts`: the complex GFMC_n interval has the SAME fixed sub-step count as
    /// the real one, so the injected (JAX-drawn) `R^T` stack selects the BITEXACT chain
    /// (byte-exact reference mesh kernels) and an empty `(0,3,3)` selects FAST (native
    /// trig off the same bit-exact threefry angles + the complex Sherman-Morrison mesh
    /// state). The driver's chain-mode gate decides which one production uses; the
    /// argument is not synthesised here.
    #[pyo3(signature = (r_up, r_dn, alat, e_scf, num_per_meas, k0, k1, injected_rts, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_n_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        alat: f64,
        e_scf: f64,
        num_per_meas: usize,
        k0: u32,
        k1: u32,
        injected_rts: numpy::PyReadonlyArray3<f64>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let cgem = g
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC complex geminal (build_obs_ctx complex=True)"))?;
        let pbc_ref = g.ewald.as_ref().map(|a| a.as_ref()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("complex-twist GFMC requires a periodic system")
        })?;
        let j3cm_view = g.j3cm.as_ref().map(|m| m.view());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let rts_arr = injected_rts.as_array();
        let inj = if rts_arr.shape()[0] == 0 {
            None
        } else {
            Some(&rts_arr)
        };
        if inj.is_some() && rts_arr.shape()[0] != num_per_meas {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gfmc_projection_n_complex: injected_rts must be (num_per_meas, 3, 3)",
            ));
        }
        let out = gfmc::gfmc_interval_n_complex(
            cgem,
            &g.jdata,
            g.j3c.as_ref(),
            j3cm_view.as_ref(),
            &g.positions.view(),
            &g.eff_charges,
            &r_up.as_array(),
            &r_dn.as_array(),
            alat,
            e_scf,
            num_per_meas,
            [k0, k1],
            inj,
            random_mesh,
            ecp_params.as_ref(),
            &pbc_ref,
            debug_full_recompute,
        );
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("w_L", out.w_l)?;
        dict.set_item(
            "e_L",
            pyo3::types::PyComplex::from_doubles(py, out.e_l.re, out.e_l.im),
        )?;
        dict.set_item("V_diag", out.v_diag)?;
        gfmc_force_block_complex(
            self,
            py,
            &dict,
            &out.r_up.view(),
            &out.r_dn.view(),
            &out.last_rt,
            screen_projection_ginv_complex(
                self,
                out.chain_state.as_ref(),
                inverse_refresh_threshold,
            ),
            alat,
            want_force,
            want_stress,
            w_slice,
            &gp_view,
            nn,
            dltmove,
        )?;
        dict.set_item("r_up", out.r_up.into_pyarray(py))?;
        dict.set_item("r_dn", out.r_dn.into_pyarray(py))?;
        dict.set_item("final_key", out.final_key.to_vec())?;
        dict.set_item("n_substeps", out.n_substeps)?;
        dict.set_item("n_pot_updates", out.n_pot_updates)?;
        Ok(dict.into_any().unbind())
    }

    /// Complex-twist GFMC_t whole-interval native tau-projection for one walker,
    /// FAST chain mode ONLY (the escape-time sub-step count is an fp function of
    /// `Re(b_L)`, exactly the real GFMC_t rationale -- BITEXACT for GFMC_t IS the pure-JAX
    /// path). Drives `gfmc_interval_t_complex`, and folds out the complex-twist HF/Pulay
    /// force + periodic stress on the measured final config when requested. Argument set
    /// and ORDER mirror the real `run_gfmc_projection_t` exactly.
    #[pyo3(signature = (r_up, r_dn, alat, tau, k0, k1, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_t_complex(
        &self,
        py: Python<'_>,
        r_up: PyReadonlyArray2<f64>,
        r_dn: PyReadonlyArray2<f64>,
        alat: f64,
        tau: f64,
        k0: u32,
        k1: u32,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let cgem = g
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC complex geminal (build_obs_ctx complex=True)"))?;
        let pbc_ref = g.ewald.as_ref().map(|a| a.as_ref()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("complex-twist GFMC requires a periodic system")
        })?;
        let j3cm_view = g.j3cm.as_ref().map(|m| m.view());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let out = crate::timed!(
            "gfmc.interval_total",
            gfmc::gfmc_interval_t_complex(
                cgem,
                &g.jdata,
                g.j3c.as_ref(),
                j3cm_view.as_ref(),
                &g.positions.view(),
                &g.eff_charges,
                &r_up.as_array(),
                &r_dn.as_array(),
                alat,
                tau,
                [k0, k1],
                random_mesh,
                ecp_params.as_ref(),
                &pbc_ref,
                debug_full_recompute,
            )
        );
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("w_L", out.w_l)?;
        dict.set_item(
            "e_L",
            pyo3::types::PyComplex::from_doubles(py, out.e_l.re, out.e_l.im),
        )?;
        gfmc_force_block_complex(
            self,
            py,
            &dict,
            &out.r_up.view(),
            &out.r_dn.view(),
            &out.last_rt,
            screen_projection_ginv_complex(
                self,
                out.chain_state.as_ref(),
                inverse_refresh_threshold,
            ),
            alat,
            want_force,
            want_stress,
            w_slice,
            &gp_view,
            nn,
            dltmove,
        )?;
        dict.set_item("r_up", out.r_up.into_pyarray(py))?;
        dict.set_item("r_dn", out.r_dn.into_pyarray(py))?;
        dict.set_item("final_key", out.final_key.to_vec())?;
        dict.set_item("n_substeps", out.n_substeps)?;
        dict.set_item("n_pot_updates", out.n_pot_updates)?;
        Ok(dict.into_any().unbind())
    }

    /// Batched real GFMC_t whole-interval projection over all
    /// walkers on the rayon global pool. Per-walker inputs gain a leading walker axis
    /// (`r_up (nw,n_up,3)`, `r_dn (nw,n_dn,3)`, `keys (nw,2)`); every shared static
    /// (`alat`, `tau`, the ECP grid, the once-parsed `GfmcArgs`) is read-only and passed
    /// by reference into the per-walker closure. The closure is the SAME projection body
    /// the per-walker `run_gfmc_projection_t` runs (`gfmc::gfmc_interval_t` +
    /// `gfmc_force_compute`), invoked through the ONE `for_each_walker` harness so walker
    /// order and per-walker Threefry determinism hold at any thread count. The
    /// returned dictionary has the same keys as the per-walker entry,
    /// each value gaining a leading walker axis. Completed walkers' final PRNG keys
    /// advance to the batch's maximum sub-step count, matching the XLA masked loop.
    #[pyo3(signature = (r_up, r_dn, alat, tau, keys, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_t_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        alat: f64,
        tau: f64,
        keys: numpy::PyReadonlyArray2<u32>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let j3 = match (&g.j3_orb, &g.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
        let pbc = g.ewald.as_ref().map(|a| a.as_ref());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let gem = g.gem.as_ref().ok_or_else(|| {
            ctx_missing(
                "GFMC real geminal (complex-twist run must use run_gfmc_projection_t_complex_batch)",
            )
        })?;
        let r_up_all = r_up.as_array();
        let r_dn_all = r_dn.as_array();
        let keys_all = keys.as_array();
        let nw = r_up_all.shape()[0];
        if r_dn_all.shape()[0] != nw || keys_all.shape()[0] != nw {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gfmc_projection_t_batch: r_up/r_dn/keys walker axis (dim 0) mismatch",
            ));
        }
        let ctx = self;
        crate::prof::set_from_env();
        let raw = walker_batch::for_each_walker(
            py,
            nw,
            |w| -> PyResult<(gfmc::GfmcIntervalOut, Option<GfmcForceOut>)> {
                let ru = r_up_all.index_axis(numpy::ndarray::Axis(0), w);
                let rd = r_dn_all.index_axis(numpy::ndarray::Axis(0), w);
                let out = crate::timed!(
                    "gfmc.interval_total",
                    gfmc::gfmc_interval_t(
                        gem,
                        &g.jdata,
                        j3_ref,
                        &g.positions.view(),
                        &g.eff_charges,
                        &ru,
                        &rd,
                        alat,
                        tau,
                        [keys_all[[w, 0]], keys_all[[w, 1]]],
                        random_mesh,
                        ecp_params.as_ref(),
                        pbc.as_ref(),
                        debug_full_recompute,
                    )
                );
                let forces = gfmc_force_compute(
                    ctx,
                    &out.r_up.view(),
                    &out.r_dn.view(),
                    &out.last_rt,
                    screen_projection_ginv(
                        ctx,
                        out.chain_state.as_ref(),
                        inverse_refresh_threshold,
                    ),
                    out.chain_state.as_ref(),
                    alat,
                    want_force,
                    want_stress,
                    w_slice,
                    &gp_view,
                    dltmove,
                )?;
                if w == 0 && crate::prof::on() {
                    crate::prof::dump("gfmc_t_walker0");
                }
                Ok((out, forces))
            },
        );
        let results = raw.into_iter().collect::<PyResult<Vec<_>>>()?;
        let max_substeps = results
            .iter()
            .map(|(out, _)| out.n_substeps)
            .max()
            .unwrap_or(0);
        let dict = pyo3::types::PyDict::new(py);
        let mut w_l = Vec::with_capacity(nw);
        let mut e_l = Vec::with_capacity(nw);
        let mut nsub = Vec::with_capacity(nw);
        let mut npot = Vec::with_capacity(nw);
        let mut ru_v = Vec::with_capacity(nw);
        let mut rd_v = Vec::with_capacity(nw);
        let mut fk = numpy::ndarray::Array2::<u32>::zeros((nw, 2));
        let mut forces = Vec::with_capacity(nw);
        for (w, (out, f)) in results.into_iter().enumerate() {
            w_l.push(out.w_l);
            e_l.push(out.e_l);
            nsub.push(out.n_substeps as u64);
            npot.push(out.n_pot_updates);
            // XLA keeps splitting keys for masked, completed walkers. Only the
            // retained child is needed: split(key, 2)[0] == Threefry(key, [0, 0]).
            // Do not replay rotations, uniforms, hops, or force evaluations.
            let mut final_key = out.final_key;
            for _ in out.n_substeps..max_substeps {
                for _ in 0..3 {
                    final_key = crate::threefry::threefry_2x32(final_key, [0, 0]);
                }
            }
            fk[[w, 0]] = final_key[0];
            fk[[w, 1]] = final_key[1];
            ru_v.push(out.r_up);
            rd_v.push(out.r_dn);
            forces.push(f);
        }
        dict.set_item(
            "w_L",
            numpy::ndarray::Array1::from_vec(w_l).into_pyarray(py),
        )?;
        dict.set_item(
            "e_L",
            numpy::ndarray::Array1::from_vec(e_l).into_pyarray(py),
        )?;
        dict.set_item(
            "n_substeps",
            numpy::ndarray::Array1::from_vec(nsub).into_pyarray(py),
        )?;
        dict.set_item(
            "n_pot_updates",
            numpy::ndarray::Array1::from_vec(npot).into_pyarray(py),
        )?;
        dict.set_item("r_up", stack_walker_2d(ru_v).into_pyarray(py))?;
        dict.set_item("r_dn", stack_walker_2d(rd_v).into_pyarray(py))?;
        dict.set_item("final_key", fk.into_pyarray(py))?;
        marshal_gfmc_forces_batched(py, &dict, forces, want_force, want_stress)?;
        Ok(dict.into_any().unbind())
    }

    /// Batched real GFMC_n whole-interval projection over all
    /// walkers. Same ABI as `run_gfmc_projection_t_batch` plus `e_scf`, `num_per_meas`,
    /// and the per-walker `injected_rts`: pass `(nw, num_per_meas, 3, 3)` for the BITEXACT
    /// chain (JAX-drawn mesh rotations, one stack per walker) or an empty `(0, 3, 3)` /
    /// `(0, ...)` array for FAST (native mesh). Returns the per-walker dict with a walker
    /// axis first (adds `V_diag`).
    #[pyo3(signature = (r_up, r_dn, alat, e_scf, num_per_meas, keys, injected_rts, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_n_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        alat: f64,
        e_scf: f64,
        num_per_meas: usize,
        keys: numpy::PyReadonlyArray2<u32>,
        injected_rts: numpy::PyReadonlyArray4<f64>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let j3 = match (&g.j3_orb, &g.j3_jm) {
            (Some(o), Some(m)) => Some((o, m.view())),
            _ => None,
        };
        let j3_ref = j3.as_ref().map(|(o, m)| (*o, m));
        let pbc = g.ewald.as_ref().map(|a| a.as_ref());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let gem = g.gem.as_ref().ok_or_else(|| {
            ctx_missing(
                "GFMC real geminal (complex-twist run must use run_gfmc_projection_n_complex_batch)",
            )
        })?;
        let r_up_all = r_up.as_array();
        let r_dn_all = r_dn.as_array();
        let keys_all = keys.as_array();
        let rts_all = injected_rts.as_array();
        let nw = r_up_all.shape()[0];
        if r_dn_all.shape()[0] != nw || keys_all.shape()[0] != nw {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gfmc_projection_n_batch: r_up/r_dn/keys walker axis (dim 0) mismatch",
            ));
        }
        // BITEXACT (per-walker injected R^T stack) vs FAST (empty). The walker axis on
        // injected_rts must match nw; an empty dim-0 selects FAST for every walker.
        let inj_on = rts_all.shape()[0] != 0;
        if inj_on {
            if rts_all.shape()[0] != nw {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_gfmc_projection_n_batch: injected_rts must be (nw, num_per_meas, 3, 3) or empty",
                ));
            }
            if rts_all.shape()[1] != num_per_meas {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_gfmc_projection_n_batch: injected_rts dim 1 must be num_per_meas",
                ));
            }
        }
        let ctx = self;
        crate::prof::set_from_env();
        let raw = walker_batch::for_each_walker(
            py,
            nw,
            |w| -> PyResult<(gfmc::GfmcIntervalOut, Option<GfmcForceOut>)> {
                let ru = r_up_all.index_axis(numpy::ndarray::Axis(0), w);
                let rd = r_dn_all.index_axis(numpy::ndarray::Axis(0), w);
                let inj_w = if inj_on {
                    Some(rts_all.index_axis(numpy::ndarray::Axis(0), w))
                } else {
                    None
                };
                let out = gfmc::gfmc_interval_n(
                    gem,
                    &g.jdata,
                    j3_ref,
                    &g.positions.view(),
                    &g.eff_charges,
                    &ru,
                    &rd,
                    alat,
                    e_scf,
                    num_per_meas,
                    [keys_all[[w, 0]], keys_all[[w, 1]]],
                    inj_w.as_ref(),
                    random_mesh,
                    ecp_params.as_ref(),
                    pbc.as_ref(),
                    debug_full_recompute,
                );
                let forces = gfmc_force_compute(
                    ctx,
                    &out.r_up.view(),
                    &out.r_dn.view(),
                    &out.last_rt,
                    screen_projection_ginv(
                        ctx,
                        out.chain_state.as_ref(),
                        inverse_refresh_threshold,
                    ),
                    out.chain_state.as_ref(),
                    alat,
                    want_force,
                    want_stress,
                    w_slice,
                    &gp_view,
                    dltmove,
                )?;
                if w == 0 && crate::prof::on() {
                    crate::prof::dump("gfmc_n_walker0");
                }
                Ok((out, forces))
            },
        );
        let results = raw.into_iter().collect::<PyResult<Vec<_>>>()?;
        let dict = pyo3::types::PyDict::new(py);
        let mut w_l = Vec::with_capacity(nw);
        let mut e_l = Vec::with_capacity(nw);
        let mut v_diag = Vec::with_capacity(nw);
        let mut nsub = Vec::with_capacity(nw);
        let mut npot = Vec::with_capacity(nw);
        let mut ru_v = Vec::with_capacity(nw);
        let mut rd_v = Vec::with_capacity(nw);
        let mut fk = numpy::ndarray::Array2::<u32>::zeros((nw, 2));
        let mut forces = Vec::with_capacity(nw);
        for (w, (out, f)) in results.into_iter().enumerate() {
            w_l.push(out.w_l);
            e_l.push(out.e_l);
            v_diag.push(out.v_diag);
            nsub.push(out.n_substeps as u64);
            npot.push(out.n_pot_updates);
            fk[[w, 0]] = out.final_key[0];
            fk[[w, 1]] = out.final_key[1];
            ru_v.push(out.r_up);
            rd_v.push(out.r_dn);
            forces.push(f);
        }
        dict.set_item(
            "w_L",
            numpy::ndarray::Array1::from_vec(w_l).into_pyarray(py),
        )?;
        dict.set_item(
            "e_L",
            numpy::ndarray::Array1::from_vec(e_l).into_pyarray(py),
        )?;
        dict.set_item(
            "V_diag",
            numpy::ndarray::Array1::from_vec(v_diag).into_pyarray(py),
        )?;
        dict.set_item(
            "n_substeps",
            numpy::ndarray::Array1::from_vec(nsub).into_pyarray(py),
        )?;
        dict.set_item(
            "n_pot_updates",
            numpy::ndarray::Array1::from_vec(npot).into_pyarray(py),
        )?;
        dict.set_item("r_up", stack_walker_2d(ru_v).into_pyarray(py))?;
        dict.set_item("r_dn", stack_walker_2d(rd_v).into_pyarray(py))?;
        dict.set_item("final_key", fk.into_pyarray(py))?;
        marshal_gfmc_forces_batched(py, &dict, forces, want_force, want_stress)?;
        Ok(dict.into_any().unbind())
    }

    /// Batched complex-twist GFMC_t projection over all walkers.
    /// Complex-twist is periodic + FAST-only (the escape-time sub-step count is an fp
    /// function of `Re(b_L)`; see `run_gfmc_projection_t_complex`). `e_L` comes back as a
    /// `(nw,)` complex128 array. As in the real batch, final PRNG keys advance through
    /// masked steps up to the batch maximum; physical interval outputs are unchanged.
    #[pyo3(signature = (r_up, r_dn, alat, tau, keys, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_t_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        alat: f64,
        tau: f64,
        keys: numpy::PyReadonlyArray2<u32>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let cgem = g
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC complex geminal (build_obs_ctx complex=True)"))?;
        let pbc_ref = g.ewald.as_ref().map(|a| a.as_ref()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("complex-twist GFMC requires a periodic system")
        })?;
        let j3cm_view = g.j3cm.as_ref().map(|m| m.view());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let r_up_all = r_up.as_array();
        let r_dn_all = r_dn.as_array();
        let keys_all = keys.as_array();
        let nw = r_up_all.shape()[0];
        if r_dn_all.shape()[0] != nw || keys_all.shape()[0] != nw {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gfmc_projection_t_complex_batch: r_up/r_dn/keys walker axis (dim 0) mismatch",
            ));
        }
        let ctx = self;
        crate::prof::set_from_env();
        let raw = walker_batch::for_each_walker(
            py,
            nw,
            |w| -> PyResult<(gfmc::GfmcIntervalOutComplex, Option<GfmcForceOut>)> {
                let ru = r_up_all.index_axis(numpy::ndarray::Axis(0), w);
                let rd = r_dn_all.index_axis(numpy::ndarray::Axis(0), w);
                let out = crate::timed!(
                    "gfmc.interval_total",
                    gfmc::gfmc_interval_t_complex(
                        cgem,
                        &g.jdata,
                        g.j3c.as_ref(),
                        j3cm_view.as_ref(),
                        &g.positions.view(),
                        &g.eff_charges,
                        &ru,
                        &rd,
                        alat,
                        tau,
                        [keys_all[[w, 0]], keys_all[[w, 1]]],
                        random_mesh,
                        ecp_params.as_ref(),
                        &pbc_ref,
                        debug_full_recompute,
                    )
                );
                let forces = gfmc_force_compute_complex(
                    ctx,
                    &out.r_up.view(),
                    &out.r_dn.view(),
                    &out.last_rt,
                    screen_projection_ginv_complex(
                        ctx,
                        out.chain_state.as_ref(),
                        inverse_refresh_threshold,
                    ),
                    alat,
                    want_force,
                    want_stress,
                    w_slice,
                    &gp_view,
                    nn,
                    dltmove,
                )?;
                if w == 0 && crate::prof::on() {
                    crate::prof::dump("gfmc_t_complex_walker0");
                }
                Ok((out, forces))
            },
        );
        let results = raw.into_iter().collect::<PyResult<Vec<_>>>()?;
        let max_substeps = results
            .iter()
            .map(|(out, _)| out.n_substeps)
            .max()
            .unwrap_or(0);
        let dict = pyo3::types::PyDict::new(py);
        let mut w_l = Vec::with_capacity(nw);
        let mut e_l = Vec::with_capacity(nw);
        let mut nsub = Vec::with_capacity(nw);
        let mut npot = Vec::with_capacity(nw);
        let mut ru_v = Vec::with_capacity(nw);
        let mut rd_v = Vec::with_capacity(nw);
        let mut fk = numpy::ndarray::Array2::<u32>::zeros((nw, 2));
        let mut forces = Vec::with_capacity(nw);
        for (w, (out, f)) in results.into_iter().enumerate() {
            w_l.push(out.w_l);
            e_l.push(out.e_l);
            nsub.push(out.n_substeps as u64);
            npot.push(out.n_pot_updates);
            // Retain the XLA batch-wide rotation / escape / move split schedule
            // after this walker finishes, without repeating any physical work.
            let mut final_key = out.final_key;
            for _ in out.n_substeps..max_substeps {
                for _ in 0..3 {
                    final_key = crate::threefry::threefry_2x32(final_key, [0, 0]);
                }
            }
            fk[[w, 0]] = final_key[0];
            fk[[w, 1]] = final_key[1];
            ru_v.push(out.r_up);
            rd_v.push(out.r_dn);
            forces.push(f);
        }
        dict.set_item(
            "w_L",
            numpy::ndarray::Array1::from_vec(w_l).into_pyarray(py),
        )?;
        dict.set_item(
            "e_L",
            numpy::ndarray::Array1::from_vec(e_l).into_pyarray(py),
        )?;
        dict.set_item(
            "n_substeps",
            numpy::ndarray::Array1::from_vec(nsub).into_pyarray(py),
        )?;
        dict.set_item(
            "n_pot_updates",
            numpy::ndarray::Array1::from_vec(npot).into_pyarray(py),
        )?;
        dict.set_item("r_up", stack_walker_2d(ru_v).into_pyarray(py))?;
        dict.set_item("r_dn", stack_walker_2d(rd_v).into_pyarray(py))?;
        dict.set_item("final_key", fk.into_pyarray(py))?;
        marshal_gfmc_forces_batched(py, &dict, forces, want_force, want_stress)?;
        Ok(dict.into_any().unbind())
    }

    /// Batched complex-twist GFMC_n projection over all walkers.
    /// Periodic by construction. `injected_rts` selects BITEXACT vs FAST exactly as the
    /// real `run_gfmc_projection_n_batch` (the complex GFMC_n interval has the same fixed
    /// sub-step count), though the driver currently drives complex FAST-only. `e_L` is a
    /// `(nw,)` complex128 array.
    #[pyo3(signature = (r_up, r_dn, alat, e_scf, num_per_meas, keys, injected_rts, weights, base_grid_points, random_mesh=false, dltmove=false, nn=1, want_force=false, want_stress=false, debug_full_recompute=false, inverse_refresh_threshold=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)]
    fn run_gfmc_projection_n_complex_batch(
        &self,
        py: Python<'_>,
        r_up: numpy::PyReadonlyArray3<f64>,
        r_dn: numpy::PyReadonlyArray3<f64>,
        alat: f64,
        e_scf: f64,
        num_per_meas: usize,
        keys: numpy::PyReadonlyArray2<u32>,
        injected_rts: numpy::PyReadonlyArray4<f64>,
        weights: PyReadonlyArray1<f64>,
        base_grid_points: PyReadonlyArray2<f64>,
        random_mesh: bool,
        dltmove: bool,
        nn: usize,
        want_force: bool,
        want_stress: bool,
        // Debug/oracle twin route (IMPL-GFMC-INCREMENTAL-TWIN-001): disable the
        // carried interval potential state (full per-substep ee/ei/local-ECP
        // recompute). The chain (hops, keys, sub-step count) is identical.
        debug_full_recompute: bool,
        inverse_refresh_threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let g = self
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let cgem = g
            .cgem
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC complex geminal (build_obs_ctx complex=True)"))?;
        let pbc_ref = g.ewald.as_ref().map(|a| a.as_ref()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("complex-twist GFMC requires a periodic system")
        })?;
        let j3cm_view = g.j3cm.as_ref().map(|m| m.view());
        let w_slice = weights.as_slice()?;
        let gp_view = base_grid_points.as_array();
        let ecp_params = g.ecp.as_ref().map(|e| gfmc::GfmcEcpParams {
            nuc_l: &e.0,
            e_l: &e.1,
            c_l: &e.2,
            p_l: &e.3,
            nuc_n: &e.4,
            am: &e.5,
            e_n: &e.6,
            c_n: &e.7,
            p_n: &e.8,
            weights: w_slice,
            grid_points: gp_view,
            nn,
            dltmove,
        });
        let r_up_all = r_up.as_array();
        let r_dn_all = r_dn.as_array();
        let keys_all = keys.as_array();
        let rts_all = injected_rts.as_array();
        let nw = r_up_all.shape()[0];
        if r_dn_all.shape()[0] != nw || keys_all.shape()[0] != nw {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "run_gfmc_projection_n_complex_batch: r_up/r_dn/keys walker axis (dim 0) mismatch",
            ));
        }
        let inj_on = rts_all.shape()[0] != 0;
        if inj_on {
            if rts_all.shape()[0] != nw {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_gfmc_projection_n_complex_batch: injected_rts must be (nw, num_per_meas, 3, 3) or empty",
                ));
            }
            if rts_all.shape()[1] != num_per_meas {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "run_gfmc_projection_n_complex_batch: injected_rts dim 1 must be num_per_meas",
                ));
            }
        }
        let ctx = self;
        crate::prof::set_from_env();
        let raw = walker_batch::for_each_walker(
            py,
            nw,
            |w| -> PyResult<(gfmc::GfmcIntervalOutComplex, Option<GfmcForceOut>)> {
                let ru = r_up_all.index_axis(numpy::ndarray::Axis(0), w);
                let rd = r_dn_all.index_axis(numpy::ndarray::Axis(0), w);
                let inj_w = if inj_on {
                    Some(rts_all.index_axis(numpy::ndarray::Axis(0), w))
                } else {
                    None
                };
                let out = gfmc::gfmc_interval_n_complex(
                    cgem,
                    &g.jdata,
                    g.j3c.as_ref(),
                    j3cm_view.as_ref(),
                    &g.positions.view(),
                    &g.eff_charges,
                    &ru,
                    &rd,
                    alat,
                    e_scf,
                    num_per_meas,
                    [keys_all[[w, 0]], keys_all[[w, 1]]],
                    inj_w.as_ref(),
                    random_mesh,
                    ecp_params.as_ref(),
                    &pbc_ref,
                    debug_full_recompute,
                );
                let forces = gfmc_force_compute_complex(
                    ctx,
                    &out.r_up.view(),
                    &out.r_dn.view(),
                    &out.last_rt,
                    screen_projection_ginv_complex(
                        ctx,
                        out.chain_state.as_ref(),
                        inverse_refresh_threshold,
                    ),
                    alat,
                    want_force,
                    want_stress,
                    w_slice,
                    &gp_view,
                    nn,
                    dltmove,
                )?;
                if w == 0 && crate::prof::on() {
                    crate::prof::dump("gfmc_n_complex_walker0");
                }
                Ok((out, forces))
            },
        );
        let results = raw.into_iter().collect::<PyResult<Vec<_>>>()?;
        let dict = pyo3::types::PyDict::new(py);
        let mut w_l = Vec::with_capacity(nw);
        let mut e_l = Vec::with_capacity(nw);
        let mut v_diag = Vec::with_capacity(nw);
        let mut nsub = Vec::with_capacity(nw);
        let mut npot = Vec::with_capacity(nw);
        let mut ru_v = Vec::with_capacity(nw);
        let mut rd_v = Vec::with_capacity(nw);
        let mut fk = numpy::ndarray::Array2::<u32>::zeros((nw, 2));
        let mut forces = Vec::with_capacity(nw);
        for (w, (out, f)) in results.into_iter().enumerate() {
            w_l.push(out.w_l);
            e_l.push(out.e_l);
            v_diag.push(out.v_diag);
            nsub.push(out.n_substeps as u64);
            npot.push(out.n_pot_updates);
            fk[[w, 0]] = out.final_key[0];
            fk[[w, 1]] = out.final_key[1];
            ru_v.push(out.r_up);
            rd_v.push(out.r_dn);
            forces.push(f);
        }
        dict.set_item(
            "w_L",
            numpy::ndarray::Array1::from_vec(w_l).into_pyarray(py),
        )?;
        dict.set_item(
            "e_L",
            numpy::ndarray::Array1::from_vec(e_l).into_pyarray(py),
        )?;
        dict.set_item(
            "V_diag",
            numpy::ndarray::Array1::from_vec(v_diag).into_pyarray(py),
        )?;
        dict.set_item(
            "n_substeps",
            numpy::ndarray::Array1::from_vec(nsub).into_pyarray(py),
        )?;
        dict.set_item(
            "n_pot_updates",
            numpy::ndarray::Array1::from_vec(npot).into_pyarray(py),
        )?;
        dict.set_item("r_up", stack_walker_2d(ru_v).into_pyarray(py))?;
        dict.set_item("r_dn", stack_walker_2d(rd_v).into_pyarray(py))?;
        dict.set_item("final_key", fk.into_pyarray(py))?;
        marshal_gfmc_forces_batched(py, &dict, forces, want_force, want_stress)?;
        Ok(dict.into_any().unbind())
    }
}

/// Owned GFMC force/stress fold-in buffers for ONE walker (the raw kernel outputs
/// before any Python marshalling). `gfmc_force_compute[_complex]` produce these off the
/// GIL inside the Rayon closure; `PyReadonlyArray` and `into_pyarray`
/// must not appear under `py.allow_threads`); the batch entry stacks them into the
/// walker-axis dict afterwards, and the per-walker entry marshals one directly via
/// [`set_gfmc_force_block_single`].
struct GfmcForceOut {
    /// HF force (dE_L/dR, dE_L/dr_up, dE_L/dr_dn).
    hf: Option<Grad3>,
    /// Pulay force (dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn).
    pp: Option<Grad3>,
    /// Periodic (HF stress 3x3, Pulay stress 3x3) when `want_stress`, else `None`.
    stress: Option<(numpy::ndarray::Array2<f64>, numpy::ndarray::Array2<f64>)>,
}

/// Screen the projection-carried inverse handed to the Pulay force/stress
/// consumers. The O(N^2) probe residual is evaluated against the geminal implied by
/// the interval's streaming determinant caches (the matrix the running inverse
/// inverts); a drifted inverse (residual above the driver's
/// ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``) is dropped so the consumers take
/// their existing fresh-factorization route. ``inf`` keeps the handoff as-is.
fn screen_projection_ginv<'a>(
    ctx: &NativeObsCtx,
    chain_state: Option<&'a gfmc::MeshFastState>,
    threshold: f64,
) -> Option<&'a numpy::ndarray::Array2<f64>> {
    let state = chain_state?;
    let gi = state.ginv();
    if !threshold.is_finite() {
        return Some(gi);
    }
    let gem = match ctx.gfmc.as_ref().and_then(|g| g.gem.as_ref()) {
        Some(gem) => gem,
        None => return Some(gi),
    };
    let g = determinant::geminal_from_det_ratio_state(gem, state.det_state());
    if determinant::running_inverse_probe_residual(&g.view(), &gi.view()) > threshold {
        None
    } else {
        Some(gi)
    }
}

/// Complex analogue of [`screen_projection_ginv`].
fn screen_projection_ginv_complex<'a>(
    ctx: &NativeObsCtx,
    chain_state: Option<&'a gfmc::MeshFastStateC>,
    threshold: f64,
) -> Option<&'a numpy::ndarray::Array2<numpy::Complex64>> {
    let state = chain_state?;
    let gi = state.ginv();
    if !threshold.is_finite() {
        return Some(gi);
    }
    let cgem = match ctx.gfmc.as_ref().and_then(|g| g.cgem.as_ref()) {
        Some(cgem) => cgem,
        None => return Some(gi),
    };
    let g = complex_det::geminal_from_det_ratio_state_complex(cgem, state.det_state());
    if complex_det::running_inverse_probe_residual_complex(&g.view(), &gi.view()) > threshold {
        None
    } else {
        Some(gi)
    }
}

/// Pure-Rust (GIL-free) real GFMC force/stress fold-in: the compute half of
/// [`gfmc_force_block`], returning owned buffers instead of writing a `PyDict`. Runs
/// inside the rayon per-walker closure, so it touches no Python. `None` when neither
/// observable is requested. Family selection (periodic / molecular-ecp / molecular-ae)
/// and the projection-inverse handoff discipline match the scalar block.
///
/// Molecular: the VMC force functions (`MolForceArgs::compute_{ecp,ae}_pp`) at the
/// final configuration, with the ECP rotation `last_rt` and `dltmove`, the carried
/// inverse, and from `chain_state` the measurement e_L's AO screens
/// (`MeshFastState::force_screen`) and its carried screened kinetic tables; the
/// Pulay force comes from the same forward. `chain_state = None` (BITEXACT) keeps
/// the exact unscreened kernels.
#[allow(clippy::too_many_arguments)]
fn gfmc_force_compute(
    ctx: &NativeObsCtx,
    r_up: &numpy::ndarray::ArrayView2<f64>,
    r_dn: &numpy::ndarray::ArrayView2<f64>,
    last_rt: &[[f64; 3]; 3],
    projection_ginv: Option<&numpy::ndarray::Array2<f64>>,
    chain_state: Option<&gfmc::MeshFastState>,
    alat: f64,
    want_force: bool,
    want_stress: bool,
    weights: &[f64],
    grid_points: &numpy::ndarray::ArrayView2<f64>,
    dltmove: bool,
) -> PyResult<Option<GfmcForceOut>> {
    if !want_force && !want_stress {
        return Ok(None);
    }
    gfmc::record_force_state_handoff(projection_ginv.is_some());
    let projection_ginv_view = projection_ginv.map(|x| x.view());
    let (hf, pp) = if want_force {
        let g = ctx
            .gfmc
            .as_ref()
            .ok_or_else(|| ctx_missing("GFMC projection args (build_obs_ctx want_gfmc=True)"))?;
        let is_pbc = g.ewald.is_some();
        // Molecular: the VMC force path (one shared forward, fused kinetic sweep,
        // ECP mesh on the same base) on the projection's carried inverse, with the
        // measurement e_L's own AO screens and carried kinetic tables; the Pulay
        // comes from the same forward.
        let mut pp_shared: Option<Grad3> = None;
        let hf = crate::timed!(
            "gfmc.force_hf",
            if is_pbc {
                ctx.pbc_force
                    .as_ref()
                    .ok_or_else(|| ctx_missing("pbc_force (GFMC want_force)"))?
                    .compute_gfmc(r_up, r_dn, alat, last_rt, weights, grid_points, dltmove)?
            } else {
                let mf = ctx
                    .mol_force
                    .as_ref()
                    .ok_or_else(|| ctx_missing("mol_force (GFMC want_force)"))?;
                let screen = match (chain_state, g.gem.as_ref()) {
                    (Some(st), Some(gem)) => st.force_screen(gem, g.j3_orb.as_ref()),
                    _ => gfmc_forces::FastScreen::default(),
                };
                let det_cache = chain_state.map(|st| st.det_grad_cache());
                let kin_cache = det_cache.as_ref().and_then(|c| c.as_ref());
                let want_pp = ctx.mol_lnpsi.is_some();
                let gi = projection_ginv_view.as_ref();
                let (hf, pp) = if mf.ecp.is_some() {
                    mf.compute_ecp_pp(
                        r_up,
                        r_dn,
                        weights,
                        grid_points,
                        last_rt,
                        dltmove,
                        gi,
                        screen,
                        want_pp,
                        kin_cache,
                    )
                } else {
                    mf.compute_ae_pp(r_up, r_dn, gi, screen, want_pp, kin_cache)
                };
                pp_shared = pp;
                hf
            }
        );
        let pp = crate::timed!(
            "gfmc.force_pp",
            if let Some(pp) = pp_shared {
                pp
            } else if is_pbc {
                let args = ctx
                    .pbc_lnpsi
                    .as_ref()
                    .ok_or_else(|| ctx_missing("pbc_lnpsi (GFMC want_force)"))?;
                match projection_ginv_view.as_ref() {
                    Some(gi) => args.compute(r_up, r_dn, gi),
                    None => args.compute_fresh(r_up, r_dn),
                }
            } else {
                let args = ctx
                    .mol_lnpsi
                    .as_ref()
                    .ok_or_else(|| ctx_missing("mol_lnpsi (GFMC want_force)"))?;
                match projection_ginv_view.as_ref() {
                    Some(gi) => args.compute(r_up, r_dn, gi),
                    None => args.compute_fresh(r_up, r_dn),
                }
            }
        );
        (Some(hf), Some(pp))
    } else {
        (None, None)
    };
    let stress = if want_stress {
        let shf = crate::timed!(
            "gfmc.stress_hf",
            ctx.pbc_shf
                .as_ref()
                .ok_or_else(|| ctx_missing("pbc_stress_hf (GFMC want_stress)"))?
                .compute_gfmc(r_up, r_dn, alat, last_rt, weights, grid_points, dltmove,)?
        );
        let spp_args = ctx
            .pbc_spp
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_pp (GFMC want_stress)"))?;
        let spp = crate::timed!(
            "gfmc.stress_pp",
            match projection_ginv_view.as_ref() {
                Some(gi) => spp_args.compute(r_up, r_dn, gi),
                None => spp_args.compute_fresh(r_up, r_dn),
            }
        );
        Some((shf, spp))
    } else {
        None
    };
    Ok(Some(GfmcForceOut { hf, pp, stress }))
}

/// Pure-Rust (GIL-free) complex-twist GFMC force/stress fold-in: the compute half of
/// [`gfmc_force_block_complex`]. Periodic-only (a twist needs a cell), holomorphic
/// derivative chains, and `Re` taken once at the observable boundary inside the
/// kernels. `None` when neither observable is requested.
#[allow(clippy::too_many_arguments)]
fn gfmc_force_compute_complex(
    ctx: &NativeObsCtx,
    r_up: &numpy::ndarray::ArrayView2<f64>,
    r_dn: &numpy::ndarray::ArrayView2<f64>,
    last_rt: &[[f64; 3]; 3],
    projection_ginv: Option<&numpy::ndarray::Array2<numpy::Complex64>>,
    alat: f64,
    want_force: bool,
    want_stress: bool,
    weights: &[f64],
    grid_points: &numpy::ndarray::ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
) -> PyResult<Option<GfmcForceOut>> {
    if !want_force && !want_stress {
        return Ok(None);
    }
    gfmc::record_force_state_handoff(projection_ginv.is_some());
    let projection_ginv_view = projection_ginv.map(|x| x.view());
    let (hf, pp) = if want_force {
        let hf = crate::timed!(
            "gfmc.force_hf_c",
            ctx.pbc_force_c
                .as_ref()
                .ok_or_else(|| ctx_missing("pbc_force_c (complex GFMC want_force)"))?
                .compute_gfmc(r_up, r_dn, alat, last_rt, weights, grid_points, nn, dltmove,)?
        );
        let pp = crate::timed!(
            "gfmc.force_pp_c",
            ctx.pbc_lnpsi_c
                .as_ref()
                .ok_or_else(|| ctx_missing("pbc_lnpsi_c (complex GFMC want_force)"))?
                .compute(r_up, r_dn, projection_ginv_view.as_ref())
        );
        (Some(hf), Some(pp))
    } else {
        (None, None)
    };
    let stress = if want_stress {
        let hf_args = ctx
            .pbc_shf_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_hf_c (complex GFMC want_stress)"))?;
        let pp_args = ctx
            .pbc_spp_c
            .as_ref()
            .ok_or_else(|| ctx_missing("pbc_stress_pp_c (complex GFMC want_stress)"))?;
        if hf_args.positions == pp_args.pos0 {
            // One base forward / Jastrow strain / base-table pass for the HF and Pulay stress.
            let (shf, spp) = crate::timed!(
                "gfmc.stress_hf_c",
                hf_args.compute_gfmc_with_pp(
                    r_up,
                    r_dn,
                    alat,
                    last_rt,
                    weights,
                    grid_points,
                    nn,
                    dltmove,
                    projection_ginv_view.as_ref(),
                )?
            );
            Some((shf, spp))
        } else {
            let shf = crate::timed!(
                "gfmc.stress_hf_c",
                hf_args.compute_gfmc(
                    r_up,
                    r_dn,
                    alat,
                    last_rt,
                    weights,
                    grid_points,
                    nn,
                    dltmove,
                )?
            );
            let spp = crate::timed!(
                "gfmc.stress_pp_c",
                pp_args.compute(r_up, r_dn, projection_ginv_view.as_ref())
            );
            Some((shf, spp))
        }
    } else {
        None
    };
    Ok(Some(GfmcForceOut { hf, pp, stress }))
}

/// Marshal ONE walker's [`GfmcForceOut`] into the projection dict under the raw force
/// keys the driver's SWCT assembly consumes (shared by the real and complex per-walker
/// entries; keys are identical by construction).
fn set_gfmc_force_block_single(
    py: Python<'_>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    forces: Option<GfmcForceOut>,
) -> PyResult<()> {
    let Some(f) = forces else {
        return Ok(());
    };
    if let (Some(hf), Some(pp)) = (f.hf, f.pp) {
        dict.set_item("hf_dR", hf.0.into_pyarray(py))?;
        dict.set_item("hf_dr_up", hf.1.into_pyarray(py))?;
        dict.set_item("hf_dr_dn", hf.2.into_pyarray(py))?;
        dict.set_item("pp_dR", pp.0.into_pyarray(py))?;
        dict.set_item("pp_dr_up", pp.1.into_pyarray(py))?;
        dict.set_item("pp_dr_dn", pp.2.into_pyarray(py))?;
    }
    if let Some((shf, spp)) = f.stress {
        dict.set_item("stress_HF", shf.into_pyarray(py))?;
        dict.set_item("stress_PP", spp.into_pyarray(py))?;
    }
    Ok(())
}

/// Stack a per-walker `Vec` of `(a,b)` arrays into a walker-axis `(nw,a,b)` array.
fn stack_walker_2d(items: Vec<numpy::ndarray::Array2<f64>>) -> numpy::ndarray::Array3<f64> {
    let nw = items.len();
    let d0 = items[0].shape()[0];
    let d1 = items[0].shape()[1];
    let mut out = numpy::ndarray::Array3::<f64>::zeros((nw, d0, d1));
    for (w, it) in items.into_iter().enumerate() {
        out.index_axis_mut(numpy::ndarray::Axis(0), w).assign(&it);
    }
    out
}

/// Marshal all walkers' [`GfmcForceOut`] into the batch dictionary, gaining a
/// leading walker axis on every force/stress block. Called once under the GIL,
/// after `for_each_walker` returns. `forces[w]` is walker `w` (walker order preserved by
/// the harness), and every entry is `Some` when either derivative block was requested.
fn marshal_gfmc_forces_batched(
    py: Python<'_>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    forces: Vec<Option<GfmcForceOut>>,
    want_force: bool,
    want_stress: bool,
) -> PyResult<()> {
    if !want_force && !want_stress {
        return Ok(());
    }
    let n = forces.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "GFMC batch: a requested derivative block has no walkers",
        ));
    }
    let mut hfr = Vec::with_capacity(n);
    let mut hfu = Vec::with_capacity(n);
    let mut hfd = Vec::with_capacity(n);
    let mut ppr = Vec::with_capacity(n);
    let mut ppu = Vec::with_capacity(n);
    let mut ppd = Vec::with_capacity(n);
    let mut shf = Vec::with_capacity(n);
    let mut spp = Vec::with_capacity(n);
    for f in forces {
        let f = f.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "GFMC batch: a requested derivative block is missing for a walker",
            )
        })?;
        if want_force {
            let hf = f.hf.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "GFMC batch: want_force set but a walker returned no HF force block",
                )
            })?;
            let pp = f.pp.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "GFMC batch: want_force set but a walker returned no Pulay force block",
                )
            })?;
            hfr.push(hf.0);
            hfu.push(hf.1);
            hfd.push(hf.2);
            ppr.push(pp.0);
            ppu.push(pp.1);
            ppd.push(pp.2);
        }
        if want_stress {
            let (h, p) = f.stress.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "GFMC batch: want_stress set but a walker returned no stress block",
                )
            })?;
            shf.push(h);
            spp.push(p);
        }
    }
    if want_force {
        dict.set_item("hf_dR", stack_walker_2d(hfr).into_pyarray(py))?;
        dict.set_item("hf_dr_up", stack_walker_2d(hfu).into_pyarray(py))?;
        dict.set_item("hf_dr_dn", stack_walker_2d(hfd).into_pyarray(py))?;
        dict.set_item("pp_dR", stack_walker_2d(ppr).into_pyarray(py))?;
        dict.set_item("pp_dr_up", stack_walker_2d(ppu).into_pyarray(py))?;
        dict.set_item("pp_dr_dn", stack_walker_2d(ppd).into_pyarray(py))?;
    }
    if want_stress {
        dict.set_item("stress_HF", stack_walker_2d(shf).into_pyarray(py))?;
        dict.set_item("stress_PP", stack_walker_2d(spp).into_pyarray(py))?;
    }
    Ok(())
}

/// GFMC force/stress fold-in: on the projection's final configuration (the measured
/// e_L config) with the interval's FINAL mesh rotation `last_rt`, evaluate the
/// per-interval HF force + Pulay force (+ periodic HF/Pulay stress when `want_stress`)
/// (see [`gfmc_force_compute`] for the kernels). The raw (dR, dr_up, dr_dn) grads + 3x3 stress buffers are written into the
/// projection dict; the driver keeps the SWCT Omega/dOmega drift assembly + the walker
/// weighting (`w_L` for GFMC_t; `w_L / (V_diag - E_scf)` for GFMC_n) in Python, exactly
/// like the MCMC chunk hands its per-step force buffers back for the host SWCT einsum.
/// The HF kernels differentiate the continuous measurement local energy, while
/// Pulay force/stress consumes the projection's carried determinant inverse.
/// Stateless/debug entry points retain independent parity surfaces. Family (molecular ae/ecp vs
/// periodic ecp) is picked from the once-parsed `GfmcArgs`; the excluded cases
/// (periodic-AE force, MPC HF, complex-C MO-J3, non-pgto stress leaves) never build the
/// force args and stay on the per-step JAX path (driver gate).
#[allow(clippy::too_many_arguments)]
fn gfmc_force_block(
    ctx: &NativeObsCtx,
    py: Python<'_>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    r_up: &numpy::ndarray::ArrayView2<f64>,
    r_dn: &numpy::ndarray::ArrayView2<f64>,
    last_rt: &[[f64; 3]; 3],
    projection_ginv: Option<&numpy::ndarray::Array2<f64>>,
    chain_state: Option<&gfmc::MeshFastState>,
    alat: f64,
    want_force: bool,
    want_stress: bool,
    weights: &[f64],
    grid_points: &numpy::ndarray::ArrayView2<f64>,
    dltmove: bool,
) -> PyResult<()> {
    // Compute the GIL-free half and marshal it so the per-walker and batch entries
    // share one dispatch and kernel source.
    let forces = gfmc_force_compute(
        ctx,
        r_up,
        r_dn,
        last_rt,
        projection_ginv,
        chain_state,
        alat,
        want_force,
        want_stress,
        weights,
        grid_points,
        dltmove,
    )?;
    set_gfmc_force_block_single(py, dict, forces)
}

/// Complex-twist GFMC force/stress fold-in: the exact
/// analogue of [`gfmc_force_block`] for `k != Gamma`. On the projection's FINAL config
/// (the measured `e_L` config) with the interval's FINAL mesh rotation `last_rt`, it
/// evaluates the per-interval complex HF force + Pulay force (+ periodic HF/Pulay stress
/// when `want_stress`) through the SAME parity-pinned complex kernels the free
/// `_gfmc_grad_e_L_ecp_pbc_complex` / `_gfmc_stress_ecp_pbc_complex` wrappers drive, and
/// writes the raw buffers into the projection dict under the SAME keys the real block
/// uses -- so the Python driver's SWCT Omega/dOmega assembly and walker weighting are
/// literally the same code on both families.
///
/// Fixed-phase convention: the derivative chains inside those
/// kernels stay HOLOMORPHIC -- complex geminal reverse and complex J3/ECP
/// cotangents -- and `Re` is taken ONCE, LAST, at the observable boundary, because
/// `Re(A)*Re(B) != Re(A*B)`. The buffers handed back here are therefore already real and
/// equal `d(Re e_L)/dX`, which is exactly what the XLA path differentiates
/// (`jnp.real(...)` wrapper on `_e_L_{t,n}_with_position_mask`). Nothing downstream
/// re-projects, and nothing upstream may Re a factor early.
///
/// Complex-twist GFMC is periodic by construction (a twist needs a cell), so there is no
/// molecular branch: the periodic complex force/stress `*_ComplexArgs` are the only
/// family, and a missing one is a loud ctx error rather than a silent JAX fallback.
#[allow(clippy::too_many_arguments)]
fn gfmc_force_block_complex(
    ctx: &NativeObsCtx,
    py: Python<'_>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    r_up: &numpy::ndarray::ArrayView2<f64>,
    r_dn: &numpy::ndarray::ArrayView2<f64>,
    last_rt: &[[f64; 3]; 3],
    projection_ginv: Option<&numpy::ndarray::Array2<numpy::Complex64>>,
    alat: f64,
    want_force: bool,
    want_stress: bool,
    weights: &[f64],
    grid_points: &numpy::ndarray::ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
) -> PyResult<()> {
    // Compute (GIL-free half) + marshal (see [`gfmc_force_block`]).
    let forces = gfmc_force_compute_complex(
        ctx,
        r_up,
        r_dn,
        last_rt,
        projection_ginv,
        alat,
        want_force,
        want_stress,
        weights,
        grid_points,
        nn,
        dltmove,
    )?;
    set_gfmc_force_block_single(py, dict, forces)
}

/// An optional owned `Array2` -> numpy array or `None` (the DTN param-grad blocks
/// return `None` for an unused u/chi/f channel; mirrors the free wrappers' `opt`).
fn opt_arr(py: Python<'_>, x: Option<numpy::ndarray::Array2<f64>>) -> Py<PyAny> {
    match x {
        Some(arr) => arr.into_pyarray(py).into_any().unbind(),
        None => py.None(),
    }
}

/// Complex-twist twin of [`opt_arr`].  Complex optimizer sources must retain their
/// imaginary component until the Python reducer performs the final real projection.
fn opt_arr_c(py: Python<'_>, x: Option<numpy::ndarray::Array2<numpy::Complex64>>) -> Py<PyAny> {
    match x {
        Some(arr) => arr.into_pyarray(py).into_any().unbind(),
        None => py.None(),
    }
}

/// Concatenate two owned `Vec<f64>` (up then dn) into one 1-D numpy array. Used by
/// the basis-optimization context methods to build the `lambda_basis_{exp,coeff}` block in
/// the reducer's `concat([..._up, ..._dn], axis=-1)` convention.
fn cat_pyarray(py: Python<'_>, mut up: Vec<f64>, dn: Vec<f64>) -> Py<PyAny> {
    up.extend(dn);
    numpy::ndarray::Array1::from_vec(up)
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// One owned `Vec<f64>` as a 1-D numpy array. Used by the J3-orbital radial
/// basis-opt ctx methods, whose block is a SINGLE shell-rank vector (the shared J3
/// leaf's up + dn contributions are summed in the kernel, not concatenated).
fn vec_pyarray(py: Python<'_>, v: Vec<f64>) -> Py<PyAny> {
    numpy::ndarray::Array1::from_vec(v)
        .into_pyarray(py)
        .into_any()
        .unbind()
}

// ---------------------------------------------------------------------------
// Batched `pd_*` helpers fan one parameter-derivative kernel
// over the walker axis through the `for_each_walker` harness, then stack the
// per-walker owned results into a leading-walker-axis numpy array. The scalar
// `pd_*` methods stay intact (other callers/tests use them); each `pd_*_batch`
// twin reuses the SAME inner kernel, so `out[w]` is bit-identical to the scalar
// on walker `w`. All batch methods are reached ONLY on the real path (the ctx is
// the `NativeObsCtx` directly); the complex-twist adapter `_RsComplexPdCtx` has no
// `_batch` methods and the Python assembler falls back to a per-walker loop for it,
// so these Rust helpers never need the complex `param_deriv_c_dtn` branch.
// ---------------------------------------------------------------------------

/// Fan `f(r_up[w], r_dn[w])` over the walker axis. `r_up_b`/`r_dn_b` are
/// `(nw, N, 3)`; the owned copies are captured `Sync` by the closure (they are read
/// only). Callers extract `&ParamDerivArgs` (fallible) BEFORE calling this so the
/// closure itself is infallible (no `PyErr` under `allow_threads`).
fn batch_map<T: Send>(
    py: Python<'_>,
    r_up_b: &numpy::PyReadonlyArray3<f64>,
    r_dn_b: &numpy::PyReadonlyArray3<f64>,
    f: impl Fn(&numpy::ndarray::ArrayView2<f64>, &numpy::ndarray::ArrayView2<f64>) -> T + Sync + Send,
) -> Vec<T> {
    use numpy::ndarray::Axis;
    let ru = r_up_b.as_array().to_owned();
    let rd = r_dn_b.as_array().to_owned();
    let nw = ru.shape()[0];
    walker_batch::for_each_walker(py, nw, |w| {
        f(&ru.index_axis(Axis(0), w), &rd.index_axis(Axis(0), w))
    })
}

/// ECP twin of [`batch_map`]: also threads the shared `weights` (RT-invariant, same
/// for every walker) and the per-walker rotated `grid_points` batch `(nw, n_grid, 3)`
/// = `ecp_gp @ RT[w]`. The scalar `pd_el_ecp_*(r_up, r_dn, weights, grid_points[w])`.
fn batch_map_ecp<T: Send>(
    py: Python<'_>,
    r_up_b: &numpy::PyReadonlyArray3<f64>,
    r_dn_b: &numpy::PyReadonlyArray3<f64>,
    weights: &numpy::PyReadonlyArray1<f64>,
    grid_points_b: &numpy::PyReadonlyArray3<f64>,
    f: impl Fn(
            &numpy::ndarray::ArrayView2<f64>,
            &numpy::ndarray::ArrayView2<f64>,
            &[f64],
            &numpy::ndarray::ArrayView2<f64>,
        ) -> T
        + Sync
        + Send,
) -> PyResult<Vec<T>> {
    use numpy::ndarray::Axis;
    let ru = r_up_b.as_array().to_owned();
    let rd = r_dn_b.as_array().to_owned();
    let gp = grid_points_b.as_array().to_owned();
    let w_vec: Vec<f64> = weights.as_slice()?.to_vec();
    let nw = ru.shape()[0];
    Ok(walker_batch::for_each_walker(py, nw, |w| {
        f(
            &ru.index_axis(Axis(0), w),
            &rd.index_axis(Axis(0), w),
            &w_vec,
            &gp.index_axis(Axis(0), w),
        )
    }))
}

/// Stack per-walker `f64` scalars into a `(nw,)` numpy array.
fn stack_f64(py: Python<'_>, v: Vec<f64>) -> Py<PyAny> {
    numpy::ndarray::Array1::from_vec(v)
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// Stack per-walker `Complex64` scalars into a `(nw,)` complex numpy array.
fn stack_c64(py: Python<'_>, v: Vec<numpy::Complex64>) -> Py<PyAny> {
    numpy::ndarray::Array1::from_vec(v)
        .into_pyarray(py)
        .into_any()
        .unbind()
}

/// Complex twin of [`stack_opt_arr2`] (the complex-twist DTN dE_L/dc channels).
fn stack_opt_arr2_c(
    py: Python<'_>,
    v: Vec<Option<numpy::ndarray::Array2<numpy::Complex64>>>,
) -> Py<PyAny> {
    if v.iter().all(|x| x.is_none()) {
        return py.None();
    }
    let mats: Vec<_> = v.into_iter().map(|x| x.unwrap()).collect();
    stack_axis0_c64(&mats).into_pyarray(py).into_any().unbind()
}

/// Stack per-walker `Array1` blocks into a `(nw, L)` numpy array.
fn stack_arr1(py: Python<'_>, v: Vec<numpy::ndarray::Array1<f64>>) -> Py<PyAny> {
    use numpy::ndarray::{Array2, Axis};
    let nw = v.len();
    let l = if nw == 0 { 0 } else { v[0].len() };
    let mut out = Array2::<f64>::zeros((nw, l));
    for (w, a) in v.into_iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(&a);
    }
    out.into_pyarray(py).into_any().unbind()
}

/// Stack per-walker `Array2` blocks into a `(nw, R, C)` numpy array.
fn stack_arr2(py: Python<'_>, v: Vec<numpy::ndarray::Array2<f64>>) -> Py<PyAny> {
    use numpy::ndarray::{Array3, Axis};
    let nw = v.len();
    let (r, c) = if nw == 0 {
        (0, 0)
    } else {
        (v[0].nrows(), v[0].ncols())
    };
    let mut out = Array3::<f64>::zeros((nw, r, c));
    for (w, a) in v.into_iter().enumerate() {
        out.index_axis_mut(Axis(0), w).assign(&a);
    }
    out.into_pyarray(py).into_any().unbind()
}

/// Stack per-walker optional `Array2` blocks (the DTN u/chi/f channels). If the
/// channel is unused (`None` for every walker -- the flag is per-run constant, so
/// the walkers always agree) the block is Python `None`, matching the scalar path.
fn stack_opt_arr2(py: Python<'_>, v: Vec<Option<numpy::ndarray::Array2<f64>>>) -> Py<PyAny> {
    if v.iter().all(|x| x.is_none()) {
        return py.None();
    }
    stack_arr2(py, v.into_iter().map(|x| x.unwrap()).collect())
}

/// up+dn concatenated `Array1` (the geminal `lambda_basis_{exp,coeff}` convention,
/// `cat_pyarray` as an owned `Array1` for stacking).
fn cat1(mut up: Vec<f64>, dn: Vec<f64>) -> numpy::ndarray::Array1<f64> {
    up.extend(dn);
    numpy::ndarray::Array1::from_vec(up)
}

/// Split a `Vec<(A, B, C)>` into three column `Vec`s (the DTN u/chi/f channels).
#[allow(clippy::type_complexity)]
fn unzip3<A, B, C>(v: Vec<(A, B, C)>) -> (Vec<A>, Vec<B>, Vec<C>) {
    let n = v.len();
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    for (x, y, z) in v {
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

/// Convert a `(d_r, d_rup, d_rdn)` owned triple into the numpy tuple the force
/// entry points return.
fn triple(
    py: Python<'_>,
    d_r: numpy::ndarray::Array2<f64>,
    du: numpy::ndarray::Array2<f64>,
    dd: numpy::ndarray::Array2<f64>,
) -> (Py<PyAny>, Py<PyAny>, Py<PyAny>) {
    (
        d_r.into_pyarray(py).into_any().unbind(),
        du.into_pyarray(py).into_any().unbind(),
        dd.into_pyarray(py).into_any().unbind(),
    )
}

/// `native_normals(k0, k1, n)` -> `n` native `N(0,1)` draws, one per threefry
/// subkey of `split([k0,k1], n)`. Exposes the `fast`-mode Gaussian transform
/// (bit-exact `uniform` + native `erfinv`) for pointwise independent inverse-
/// function and XLA comparisons using the central consistency tolerance pair.
#[pyfunction]
#[pyo3(name = "native_normals")]
fn py_native_normals(k0: u32, k1: u32, n: usize) -> Vec<f64> {
    crate::threefry::split([k0, k1], n)
        .into_iter()
        .map(crate::threefry::native_normal)
        .collect()
}

/// Profiler control for kernel-level microbenches: `_prof_enable()` arms the
/// `timed!` scopes (same as `JQMC_RS_PROFILE=1`), `_prof_dump(tag)` prints the
/// thread-local accumulators to stderr and clears them. Instrumentation only.
#[pyfunction]
#[pyo3(name = "_prof_enable")]
fn py_prof_enable() {
    crate::prof::PROFILE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[pyfunction]
#[pyo3(name = "_prof_dump")]
fn py_prof_dump(tag: &str) {
    crate::prof::dump(tag);
}

/// GFMC route-engagement counters: proof of which projection-scoped cache tier
/// actually engaged (per-fixture assertion surface -- never assume engagement
/// from timings). Observability only. Returned as a list (append-only index
/// order):
///  0 dtn_cache_hits, 1 det_fast_intervals (molecular),
///  2 reduced_state_intervals (non-molecular),
///  3/4 candidate build/refresh molecular GTO,
///  5/6 build/refresh periodic pGTO, 7/8 build/refresh B-spline,
///  9/10 build/refresh pGTO+PW, 11 det candidate slow-path mesh evals,
///  12 J3 slow fallbacks, 13 generic det derivative-cache builds,
///  14-23 complex candidate-table counters, 24-40 complex derivative-cache
///  counters,
///  41 det grad/lap slow, 42 J1/J2 grad/lap slow, 43 J3 grad/lap slow,
///  44 DTN grad/lap slow -- the real-path derivative-cache miss counters.
///  Supported fixtures require zero in both full and mixed precision. Index 44
///  also reports DTN work repeated by an unexpected J1/J2 full rebuild
///  45-47 kinetic DTN tabpip, 48-55 J1/J2 kinetic/ECP tabpip,
///  56-60 determinant winv, 61-63 ECP DTN tabpip,
///  and 64/65 force inverse handoff/fallback
///  (full append-only index list at `gfmc::gfmc_route_counters`).
#[pyfunction]
#[pyo3(name = "gfmc_route_counters")]
fn py_gfmc_route_counters() -> Vec<u64> {
    crate::gfmc::gfmc_route_counters().to_vec()
}

/// Route-predicate counters, a separate append-only array
/// sharing the enable flag and reset window of `gfmc_route_counters`:
///  0/1 ECP determinant w-dot table built / `ecp_wdot_ok` declined,
///  2/3 ECP J3 w-dot table built / `ecp_j3_wdot_ok` declined,
///  4/5 MCMC chunk steps with `det_carry_ok` engaged / declined,
///  6/7 MCMC chunk steps (J3 present) with `j3_carry_ok` engaged / declined,
///  8/9 ECP parameter-gradient mesh candidates with `sm_ratio_ok` engaged / declined,
///  10/11 LRDMC interval hops whose carried `G^-1` passed / failed the
///  per-hop residual screen (11 = in-interval inverse refreshes),
///  12/13 twist ECP parameter-gradient meshes on the fast mesh / reference-mesh candidates,
///  14/15 J1/J2/J3 ECP O_k mesh reduces done incrementally / full O_k evaluations,
///  16/17 complex-twist FAST chunk ECP spins with the AO-space mesh w-dot
///  (`ComplexOrb::mesh_wdot_ok`) engaged / declined,
///  18/19 FAST ECP non-local (electron, term) pairs kept / dropped by the radial cut
///  `|v_t| < FAST_ECP_NONLOCAL_RADIAL_CUT`
///  (full list at `gfmc::route_predicate_counters`). Zeroed and enabled by
/// `gfmc_route_counters_reset`.
#[pyfunction]
#[pyo3(name = "route_predicate_counters")]
fn py_route_predicate_counters() -> Vec<u64> {
    crate::gfmc::route_predicate_counters().to_vec()
}

/// The internal probe-residual threshold of the Rust carried
/// Sherman-Morrison inverse (`determinant::SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`),
/// exported only so a test can pin it equal to the Python drivers' constant.
#[pyfunction]
#[pyo3(name = "sm_inverse_refresh_residual_threshold")]
fn py_sm_inverse_refresh_residual_threshold() -> f64 {
    crate::determinant::SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
}

/// Issue #605: the LU guard residual of the seed / carried-inverse refresh inverse
/// (`complex_det::GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD`), exported only so a test
/// can pin it equal to the Python drivers' constant.
#[pyfunction]
#[pyo3(name = "geminal_lu_guard_residual_threshold")]
fn py_geminal_lu_guard_residual_threshold() -> f64 {
    crate::complex_det::GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD
}

/// Issue #605: the `|e|` clip of the geminal equilibration exponents
/// (`complex_det::GEMINAL_EQUILIBRATION_MAX_EXPONENT`), exported for the same pin.
#[pyfunction]
#[pyo3(name = "geminal_equilibration_max_exponent")]
fn py_geminal_equilibration_max_exponent() -> i32 {
    crate::complex_det::GEMINAL_EQUILIBRATION_MAX_EXPONENT
}

/// Zero the GFMC route-engagement counters and the route-predicate counters
/// (start of a measurement window).
#[pyfunction]
#[pyo3(name = "gfmc_route_counters_reset")]
fn py_gfmc_route_counters_reset() {
    crate::gfmc::gfmc_route_counters_reset()
}

/// Whether any GFMC projection mesh took the full-value J3 slow fallback
/// (no streaming J3 ratio state) since the last call; clears the witness. Unlike
/// `gfmc_route_counters` this witness is always on, so production can log a
/// once-per-run warning about the (O(10-100x) slower) route.
#[pyfunction]
#[pyo3(name = "gfmc_take_j3_slow_fallback_seen")]
fn py_gfmc_take_j3_slow_fallback_seen() -> bool {
    crate::gfmc::gfmc_take_j3_slow_fallback_seen()
}

// ---------------------------------------------------------------------------
// Rayon thread control.
// ---------------------------------------------------------------------------

/// Initialize the rayon global thread pool.
///
/// Precedence for the thread count: explicit `n` > env `JQMC_RS_NUM_THREADS` >
/// rayon's affinity-aware default. The default is `available_parallelism()`, which
/// on Linux honors the cgroup and CPU-affinity mask, so `mpirun --bind-to socket`
/// derives the per-rank thread count automatically.
///
/// `build_global` may only succeed ONCE per process, so this MUST be called BEFORE
/// the first parallel `for_each_walker` call. Calling it again is tolerated only as a
/// no-op when the request already matches the live pool width (so a test harness that
/// calls `configure_threads(4)` twice does not fail); a genuine attempt to RESIZE an
/// already-initialized pool raises `RuntimeError`, because rayon cannot honor it.
#[pyfunction]
#[pyo3(name = "configure_threads", signature = (n=None))]
fn py_configure_threads(n: Option<usize>) -> PyResult<()> {
    // explicit arg > env > default (None => rayon's affinity-aware default).
    let requested = n.or_else(|| {
        std::env::var("JQMC_RS_NUM_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    });
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(k) = requested {
        builder = builder.num_threads(k);
    }
    match builder.build_global() {
        Ok(()) => Ok(()),
        Err(e) => {
            // The pool is already initialized. Tolerate iff we are not asking to
            // change its width (idempotent re-call, or a default-valued call).
            let live = rayon::current_num_threads();
            match requested {
                None => Ok(()),
                Some(k) if k == live => Ok(()),
                Some(k) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "jqmc_rs.configure_threads({k}): the rayon global pool is already \
                     initialized with {live} threads and cannot be resized. \
                     build_global runs once per process -- call configure_threads BEFORE \
                     the first parallel kernel. rayon: {e}"
                ))),
            }
        }
    }
}

/// The live rayon global-pool thread count. Reading this initializes the pool with
/// the default width if `configure_threads` was never called, so callers see the
/// count that the batch harness will actually use.
#[pyfunction]
#[pyo3(name = "current_num_threads")]
fn py_current_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Smoke probe over the [`walker_batch::for_each_walker`] harness: returns the
/// list of walker indices actually processed, in walker order. It exercises the exact
/// GIL-release + `into_par_iter().map().collect()` path the batch entries use, so the
/// smoke proof can show (a) the output is identical at threads=1 and threads=N and
/// (b) `current_num_threads` reports as configured. Harmless; kept for future debugging.
#[pyfunction]
#[pyo3(name = "_walker_batch_probe")]
fn py_walker_batch_probe(py: Python<'_>, nw: usize) -> Vec<usize> {
    walker_batch::for_each_walker(py, nw, |w| w)
}

#[pymodule]
fn jqmc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // FNV-1a digest of the Rust source tree at build time, so testing a stale
    // extension fails loudly.
    m.add("__source_fnv64__", env!("JQMC_RS_SOURCE_FNV64"))?;
    // Read the bindings actually used by production, not only the generated
    // producer table. A reintroduced copied alias must fail the Python setting
    // check when its central value changes, even if usual numerics miss it.
    let mut compiled_settings = setting_consts::MANIFEST.to_vec();
    for (name, actual) in [
        (
            "FAST_AO_SCREEN_ABS_BOUND",
            atomic_orbital::FAST_SCREEN_ABS_BOUND,
        ),
        ("NATIVE_AO_SCREEN_EXPONENT", atomic_orbital::LEPSBAS),
        ("EPS_safe_distance", jastrow_factor::EPS_SAFE_DISTANCE),
        ("EPS_rcond_SVD", complex_det::RCOND_SVD),
        (
            "SM_LARGE_RATIO_THRESHOLD_FP64",
            determinant::SM_LARGE_RATIO_THRESHOLD,
        ),
        (
            "AS_LINEAR_MIN_SQ_NORM_FP64",
            determinant::AS_LINEAR_MIN_SQ_NORM,
        ),
        (
            "AS_NORM_CANCELLATION_THRESHOLD_FP64",
            determinant::AS_NORM_CANCELLATION_THRESHOLD,
        ),
        (
            "SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD",
            determinant::SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD,
        ),
    ] {
        let entry = compiled_settings
            .iter_mut()
            .find(|(key, _)| *key == name)
            .expect("every production policy binding must have a generated entry");
        entry.1 = actual;
    }
    m.add("__setting_consts__", compiled_settings)?;
    m.add_class::<NativeObsCtx>()?;
    m.add_class::<NativeConstructionPrecision>()?;
    // Rayon thread control and walker-batch harness probe.
    m.add_function(wrap_pyfunction!(py_configure_threads, m)?)?;
    m.add_function(wrap_pyfunction!(py_current_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_wide_sr_real_cg,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_wide_sr_real_direct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_tall_sr_real_cg,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_tall_sr_real_direct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_lm_pencil_core,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_sr_operator_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_linalg::py_optimizer_sr_precondition_sample_major,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_walker_batch_probe, m)?)?;
    m.add_function(wrap_pyfunction!(py_native_precision_zones, m)?)?;
    m.add_function(wrap_pyfunction!(py_native_precision_support, m)?)?;
    m.add_function(wrap_pyfunction!(py_configure_native_precision, m)?)?;
    m.add_function(wrap_pyfunction!(py_prof_enable, m)?)?;
    m.add_function(wrap_pyfunction!(py_prof_dump, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_route_counters, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_route_counters_reset, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_take_j3_slow_fallback_seen, m)?)?;
    m.add_function(wrap_pyfunction!(py_route_predicate_counters, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_sm_inverse_refresh_residual_threshold,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_geminal_lu_guard_residual_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(py_geminal_equilibration_max_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_obs_ctx, m)?)?;
    m.add_function(wrap_pyfunction!(py_native_normals, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_compute_bare_coulomb_potential_el_el,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_compute_bare_coulomb_potential_ion_ion,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_compute_bare_coulomb_potential_el_ion_element_wise,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_compute_bare_coulomb_potential, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_jastrow_one_body, m)?)?;
    m.add_function(wrap_pyfunction!(py_jastrow_kstress_debug, m)?)?;
    m.add_function(wrap_pyfunction!(py_smooth_dist_debug, m)?)?;
    m.add_function(wrap_pyfunction!(py_smooth_dist_hess_debug, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_jastrow_two_body, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_jastrow_three_body_value, m)?)?;
    m.add_function(wrap_pyfunction!(py_jastrow_1b2b_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_j3_init_state, m)?)?;
    m.add_function(wrap_pyfunction!(_j3_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_j3_advance_state, m)?)?;
    m.add_function(wrap_pyfunction!(_structure_data_from_py_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(py_ao_cart_deriv, m)?)?;
    m.add_function(wrap_pyfunction!(py_pgto_cart_deriv, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_bspline_vgl, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_bspline_complex_vgl, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_tiled_orbitals_vgl, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_tiled_orbitals_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_bspline_vgl_dcell, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_bspline_dcell, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_bspline_kinetic_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_exact, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_grad_exact, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_grad_lap_fast_cut, m)?)?;
    m.add_function(wrap_pyfunction!(py_e_l_fast_screened, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_value_fast_cut, m)?)?;
    m.add_function(wrap_pyfunction!(_tf_threefry, m)?)?;
    m.add_function(wrap_pyfunction!(_tf_split, m)?)?;
    m.add_function(wrap_pyfunction!(_tf_randint_i32, m)?)?;
    m.add_function(wrap_pyfunction!(_tf_randint_i64, m)?)?;
    m.add_function(wrap_pyfunction!(_tf_uniform_f64, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ln_det_geminal_all_elements, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_geminal_all_elements, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_grads_and_laplacian_ln_det, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_compute_grads_and_laplacian_jastrow_part,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_grad_j1_param, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j2_param, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j3_jmatrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ln_det_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ln_det_lambda_diag, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ln_det_mo, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_vgl_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aos_hessian_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_geminal_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_verify_sm_ratio_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ln_det_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ln_det_mo_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ln_det_lambda_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ln_det_lambda_diag_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_lambda_diag_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_mo_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_mo_withj_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_lambda_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_lambda_withj_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lndet_wrt_r_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_wrt_r_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_wrt_r_withj_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lnpsi_wrt_strain_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_wrt_strain_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_wrt_strain_withj_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_v_ecp_nonlocal_strain_complex, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_grad_v_ecp_nonlocal_strain_withj_complex,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(_mcmc_sweep_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_v_ecp_nonlocal_force_complex, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_grad_v_ecp_nonlocal_force_withj_complex,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_grad_v_ecp_nonlocal_mo_complex, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_grad_v_ecp_nonlocal_mo_withj_complex,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_ecp_wrt_mo_coeff_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_kinetic_det_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_kinetic_withj_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ecp_nonlocal_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ecp_nonlocal_withj_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_radial_basis, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_radial_kinetic, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_radial_basis_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_radial_kinetic_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_radial_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j3_radial_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j3_radial_kinetic_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_j3_radial_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j_dtn_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_dtn_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_dtn_kinetic, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_j1_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_j2_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_j3_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j3_jmatrix_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_ecp_j3_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_dtn_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_ecp_j1_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_ecp_j2_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_ecp_dtn_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_lambda_diag, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_lambda_diag_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_lambda_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_lambda_diag_twist, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_mo, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_j1, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_j2, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_j3, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_force, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_local_force, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_vlocal_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_v_ecp_nonlocal_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_v_ecp_nonlocal_strain_bspline, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_dtn_pbc_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_eee_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_eei, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_eii_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_eee_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_eei_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_eii_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_lambda_diag, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_mo, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_j1, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_j2, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_j3, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_dtn, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_ecp_radial, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lndet_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j1_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_j3_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_v_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_pbc_det_wrt_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_pbc_withj_wrt_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lndet_pbc_wrt_r, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lnpsi_pbc_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_det_pbc_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_t_pbc_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lnpsi_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_strain_withj, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_strain_ecp, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_el_strain_dtn, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_compute_grads_and_laplacian_jastrow_three_body,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_compute_kinetic_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ecp_local_parts_all_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_compute_ecp_nonlocal_parts_all_pairs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_fast_ecp_nonlocal_radial_cut, m)?)?;
    m.add_function(wrap_pyfunction!(py_ecp_nonlocal_fast_radial_cut_probe, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_local_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_local_energy_fast, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_local_energy_fast_complex, m)?)?;
    m.add_function(wrap_pyfunction!(native_driver::py_native_vmc, m)?)?;
    m.add_function(wrap_pyfunction!(native_driver::py_native_vmc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_transform::py_optimizer_transform_derivatives,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        optimizer_transform::py_optimizer_coordinate_force_jackknife,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_compute_dtn_value, m)?)?;
    m.add_function(wrap_pyfunction!(py_dtn_grad_lap_carry_probe, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_grads_and_laplacian_dtn, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_dtn_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_bench_faer_mo, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ewald_potential_ee, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ewald_potential_ei, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ewald_potential_ii, m)?)?;
    m.add_function(wrap_pyfunction!(py_ewald_ei_per_electron, m)?)?;
    m.add_function(wrap_pyfunction!(_det_init_state, m)?)?;
    m.add_function(wrap_pyfunction!(_det_advance_state, m)?)?;
    m.add_function(wrap_pyfunction!(_det_v_up_move, m)?)?;
    m.add_function(wrap_pyfunction!(_det_u_dn_move, m)?)?;
    m.add_function(wrap_pyfunction!(_det_sm_update, m)?)?;
    m.add_function(wrap_pyfunction!(_matrix_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(_refresh_running_inverse_if_needed, m)?)?;
    m.add_function(wrap_pyfunction!(
        _refresh_running_inverse_if_needed_complex,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_compute_as_regularization_factor, m)?)?;
    m.add_function(wrap_pyfunction!(_move_draws, m)?)?;
    m.add_function(wrap_pyfunction!(_single_move, m)?)?;
    m.add_function(wrap_pyfunction!(_mcmc_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(_nearest_atom_periodic, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_g_l, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_discretized_kinetic_elements, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_lrdmc_ae, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_grad_e_l_ae, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_e_l_ae, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_e_l_ecp, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_stress_ae, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_stress_ecp_jnone, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_stress_ecp_withj, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_stress_ecp_pbc_jnone, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_stress_ecp_pbc_withj, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_mpc_potential_ee, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_mpc_correction_ee, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_mpc_correction_ee_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad_lnj3_pbc_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_dtn_value_strain, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_grad_e_l_ecp, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_grad_e_l_ecp_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_grad_e_l_ecp_pbc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_stress_ecp_pbc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_grad_ln_psi, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_ln_psi, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_e_l_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_ln_psi_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_e_l_pbc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_ln_psi_pbc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_ln_psi_eps_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_e_l_eps_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_ln_psi_eps_pbc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_vmc_grad_e_l_eps_pbc_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_swct_omega, m)?)?;
    m.add_function(wrap_pyfunction!(py_swct_domega, m)?)?;
    m.add_function(wrap_pyfunction!(py_swct_corrections_batched, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_lrdmc_ae_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_lrdmc_ecp_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ae_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ae, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ae_pop, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ae_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_lrdmc_ecp, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ecp, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ecp_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_run_ecp_complex, m)?)?;
    m.add_function(wrap_pyfunction!(py_gfmc_reconfigure_indices, m)?)?;
    m.add_function(wrap_pyfunction!(_bench_orb_split, m)?)?;
    m.add_function(wrap_pyfunction!(_kato_f_l, m)?)?;
    m.add_function(wrap_pyfunction!(_t_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_metropolis_accept, m)?)?;
    Ok(())
}
