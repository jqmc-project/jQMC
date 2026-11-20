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

//! End-to-end native VMC driver. A Python-side block loop over these kernels
//! would round-trip r_up/r_dn/G/G_inv through numpy, and every
//! `compute_local_energy(ham, ...)` would re-extract the whole Hamiltonian from
//! the Python object (getattr/tolist).
//! Here the Hamiltonian is extracted ONCE into owned Rust structs (`LeTerms`),
//! and the block loop (sweep + e_L + accumulation) runs entirely in Rust -- the
//! shape a production MPI rank would take. This sheds the per-call extraction and
//! the per-block numpy copies, the Python overhead that remains in a
//! many-rank MPI run.

use crate::data::JastrowData;
use crate::determinant::{self, GeminalData};
use crate::timed;
use crate::{coulomb_potential, ewald, mcmc, wavefunction};
use numpy::ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// splitmix64 -- proposals + initial positions only (NOT the accept/reject stream,
/// which the sweep threads through its threefry `key`). Reproducible per seed.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self
            .uniform()
            .max(crate::setting_consts::EPS_machine_precision);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

/// All local-energy inputs, owned -- extracted ONCE from the Python Hamiltonian.
pub struct LeTerms {
    gem: GeminalData,
    jdata: JastrowData,
    coulomb: coulomb_potential::CoulombPotentialData,
    /// Ion Ewald structure factor (re, im), precomputed once (ions fixed). Reused
    /// by every e_L instead of the per-call `O(n_g*n_atom)` sin/cos.
    ewald_si: Option<(Vec<f64>, Vec<f64>)>,
    weights: Vec<f64>,
    grid_points: Array2<f64>,
    n_up: usize,
    n_dn: usize,
}

impl LeTerms {
    fn from_py(
        ham: &Bound<'_, PyAny>,
        weights: Vec<f64>,
        grid_points: Array2<f64>,
    ) -> PyResult<Self> {
        let gd = ham.getattr("wavefunction_data")?.getattr("geminal_data")?;
        let n_up: usize = gd.getattr("num_electron_up")?.extract()?;
        let n_dn: usize = gd.getattr("num_electron_dn")?.extract()?;

        // Parse the whole Hamiltonian hierarchy once (Structure / Coulomb->Ewald /
        // Wavefunction->{Geminal, Jastrow}); J3 lives on `jastrow.three_body`.
        let hd = crate::hamiltonians::HamiltonianData::from_py(ham)?;

        // Cache the ion Ewald structure factor once (constant across the run).
        let ewald_si = hd.coulomb.ewald.as_ref().map(|ew| {
            ewald::compute_ion_structure_factor(
                &ew.g_vectors.view(),
                &hd.coulomb.effective_charges,
                &hd.coulomb.positions.view(),
            )
        });

        Ok(LeTerms {
            gem: hd.wavefunction.geminal,
            jdata: hd.wavefunction.jastrow,
            coulomb: hd.coulomb,
            ewald_si,
            weights,
            grid_points,
            n_up,
            n_dn,
        })
    }

    /// Pure-Rust local energy (mirrors `py_compute_local_energy`, no Python).
    fn local_energy(&self, r_up: &ArrayView2<f64>, r_dn: &ArrayView2<f64>) -> f64 {
        let t = timed!(
            "eL.kinetic",
            wavefunction::compute_kinetic_energy(&self.gem, &self.jdata, r_up, r_dn)
        );
        // J3 for the ECP-nonlocal ratio (still an explicit kernel arg; sourced
        // from the single `three_body` field via as_ref()).
        let j3_data = self.jdata.three_body.as_ref().map(|j3| j3.as_ref());
        let j3_ref = j3_data.as_ref().map(|(o, m)| (*o, m));

        let cell_view = self.coulomb.cell.view();
        let cell_nmax = if self.coulomb.pbc {
            Some((&cell_view, self.coulomb.n_max))
        } else {
            None
        };

        let mut v = if let Some(ew) = &self.coulomb.ewald {
            // Electron structure factor: computed ONCE per e_L, shared by the ee
            // and ei reciprocal sums (turns ei recip into O(n_g)).
            let esf_owned = ewald::compute_electron_structure_factor(
                r_up,
                r_dn,
                &ew.g_vectors.view(),
                &cell_view,
            );
            let esf = Some((esf_owned.0.as_slice(), esf_owned.1.as_slice()));
            let ee = timed!(
                "eL.ewald_ee",
                ewald::compute_ewald_potential_ee(
                    r_up,
                    r_dn,
                    ew.eta,
                    &ew.real_images.view(),
                    &ew.g_vectors.view(),
                    &ew.g_factors,
                    &cell_view,
                    self.coulomb.n_max,
                    esf,
                )
            );
            let si = self
                .ewald_si
                .as_ref()
                .map(|(sr, sii)| (sr.as_slice(), sii.as_slice()));
            let ei = timed!(
                "eL.ewald_ei",
                ewald::compute_ewald_potential_ei(
                    r_up,
                    r_dn,
                    ew.eta,
                    &ew.real_images.view(),
                    &ew.g_vectors.view(),
                    &ew.g_factors,
                    &cell_view,
                    self.coulomb.n_max,
                    &self.coulomb.effective_charges,
                    &self.coulomb.positions.view(),
                    si,
                    esf,
                )
            );
            ee + ei + ew.ii
        } else {
            coulomb_potential::compute_bare_coulomb_potential(
                &self.coulomb.positions.view(),
                &self.coulomb.effective_charges,
                r_up,
                r_dn,
            )
        };

        if let Some(e) = &self.coulomb.ecp {
            v += timed!(
                "eL.ecp_local",
                coulomb_potential::compute_ecp_local_parts_all_pairs(
                    &self.coulomb.positions.view(),
                    &e.nuc_l,
                    &e.e_l,
                    &e.c_l,
                    &e.p_l,
                    r_up,
                    r_dn,
                    cell_nmax,
                )
            );
            let g_base = determinant::compute_geminal_all_elements(&self.gem, r_up, r_dn);
            let g_inv = timed!("eL.geminv", determinant::inverse(&g_base.view()));
            v += timed!(
                "eL.ecp_nonlocal",
                coulomb_potential::compute_ecp_nonlocal_parts_all_pairs(
                    &self.gem,
                    &self.jdata,
                    r_up,
                    r_dn,
                    &self.coulomb.positions.view(),
                    &e.nuc_n,
                    &e.am,
                    &e.e_n,
                    &e.c_n,
                    &e.p_n,
                    &self.weights,
                    &self.grid_points.view(),
                    false,
                    1,
                    j3_ref,
                    cell_nmax,
                    Some(&g_inv.view()),
                    true, // native_vmc: opt-in mesh AO distance cutoff (production e_L)
                    None,
                    None, // native_vmc standalone e_L: no sweep J3 state reuse (rebuild)
                )
            );
        }

        t + v
    }

    /// Random initial positions (molecular: atom + N(0,0.5); pbc: uniform cell).
    fn init_positions(&self, rng: &mut Rng) -> (Array2<f64>, Array2<f64>) {
        let mut place = |n: usize| -> Array2<f64> {
            let mut r = Array2::<f64>::zeros((n, 3));
            for e in 0..n {
                if self.coulomb.pbc {
                    let u = [rng.uniform(), rng.uniform(), rng.uniform()];
                    for c in 0..3 {
                        r[[e, c]] = u[0] * self.coulomb.cell[[0, c]]
                            + u[1] * self.coulomb.cell[[1, c]]
                            + u[2] * self.coulomb.cell[[2, c]];
                    }
                } else {
                    let a = rng.below(self.coulomb.positions.nrows());
                    for c in 0..3 {
                        r[[e, c]] = self.coulomb.positions[[a, c]] + 0.5 * rng.normal();
                    }
                }
            }
            r
        };
        (place(self.n_up), place(self.n_dn))
    }
}

/// Run the full native VMC chain. Returns (energies, accepted, total_moves).
fn run(
    terms: &LeTerms,
    seed: u64,
    dt: f64,
    eps: f64,
    moves_per_block: usize,
    n_warmup: usize,
    n_blocks: usize,
) -> (Vec<f64>, usize, usize) {
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let (mut r_up, mut r_dn) = terms.init_positions(&mut rng);
    let mut g = determinant::compute_geminal_all_elements(&terms.gem, &r_up.view(), &r_dn.view());
    let mut g_inv = determinant::inverse(&g.view());
    let mut key = [seed as u32, (seed >> 32) as u32];

    let cell_view = terms.coulomb.cell.view();
    let cell_nmax = if terms.coulomb.pbc {
        Some((&cell_view, terms.coulomb.n_max))
    } else {
        None
    };
    // J3 for the sweep ratio, sourced from the single `three_body` field.
    let j3_data = terms.jdata.three_body.as_ref().map(|j3| j3.as_ref());
    let j3_orb = j3_data.as_ref().map(|(o, _)| *o);
    let j3_jm_ref = j3_data.as_ref().map(|(_, m)| m);

    let mut energies = Vec::with_capacity(n_blocks);
    let mut accepted = 0usize;
    let mut total_moves = 0usize;

    for blk in 0..(n_warmup + n_blocks) {
        let normals: Vec<f64> = (0..moves_per_block).map(|_| rng.normal()).collect();
        let res = timed!(
            "sweep",
            mcmc::mcmc_sweep(
                &terms.gem,
                &terms.jdata,
                j3_orb,
                j3_jm_ref,
                &terms.coulomb.positions.view(),
                &terms.coulomb.effective_charges,
                &r_up.view(),
                &r_dn.view(),
                &g.view(),
                &g_inv.view(),
                dt,
                eps,
                key,
                &normals,
                moves_per_block,
                cell_nmax,
                None, // native_vmc standalone: fresh determinant state per sweep
                None, // native_vmc standalone: fresh J3 state per sweep (no chunk carry)
                true, // whole-run native is statistical (own RNG, no JAX byte-contract) -> cutoff on
            )
        );
        r_up = res.r_up;
        r_dn = res.r_dn;
        g = res.geminal;
        g_inv = res.geminal_inv;
        key = res.final_key;
        accepted += res.accepted;
        total_moves += moves_per_block;
        if blk >= n_warmup {
            energies.push(timed!(
                "e_L",
                terms.local_energy(&r_up.view(), &r_dn.view())
            ));
        }
    }
    if crate::prof::on() {
        crate::prof::dump("native_vmc");
    }
    (energies, accepted, total_moves)
}

/// `native_vmc(hamiltonian, weights, grid_points, seed, dt, epsilon, moves_per_block,
/// n_warmup, n_blocks)` -> (energies[n_blocks], accept_rate). The whole chain runs
/// in Rust; only the final per-block energies cross back to Python.
#[pyfunction]
#[pyo3(name = "native_vmc", signature = (hamiltonian_data, weights, grid_points, seed, dt, epsilon, moves_per_block, n_warmup, n_blocks))]
#[allow(clippy::too_many_arguments)]
pub fn py_native_vmc(
    hamiltonian_data: &Bound<'_, PyAny>,
    weights: numpy::PyReadonlyArray1<f64>,
    grid_points: numpy::PyReadonlyArray2<f64>,
    seed: u64,
    dt: f64,
    epsilon: f64,
    moves_per_block: usize,
    n_warmup: usize,
    n_blocks: usize,
) -> PyResult<(Vec<f64>, f64)> {
    crate::prof::set_from_env();
    let terms = LeTerms::from_py(
        hamiltonian_data,
        weights.as_slice()?.to_vec(),
        grid_points.as_array().to_owned(),
    )?;
    let (energies, accepted, total_moves) = hamiltonian_data.py().allow_threads(|| {
        run(
            &terms,
            seed,
            dt,
            epsilon,
            moves_per_block,
            n_warmup,
            n_blocks,
        )
    });
    let acc = accepted as f64 / total_moves.max(1) as f64;
    Ok((energies, acc))
}

// ============================================================================
// Complex-twist native VMC chain. Mirror of LeTerms/run but with the
// complex sweep (mcmc_sweep_complex) + Re(complex e_L). e_L = T(complex) + V(real);
// the VMC energy is <Re(e_L)>_{|Psi|^2}. J3 = Re(complex sphe AO). jnone/j12/j123ao/jdtn.
// ============================================================================

struct LeTermsC {
    cgem: crate::complex_det::ComplexGeminal,
    jdata: JastrowData,
    j3c: Option<crate::complex_det::ComplexOrb>,
    j3m: Option<Array2<f64>>,
    coulomb: coulomb_potential::CoulombPotentialData,
    ewald_si: Option<(Vec<f64>, Vec<f64>)>,
    weights: Vec<f64>,
    grid_points: Array2<f64>,
    n_up: usize,
    n_dn: usize,
}

impl LeTermsC {
    fn from_py(
        ham: &Bound<'_, PyAny>,
        weights: Vec<f64>,
        grid_points: Array2<f64>,
    ) -> PyResult<Self> {
        let wf = ham.getattr("wavefunction_data")?;
        let gd_py = wf.getattr("geminal_data")?;
        let cgem = crate::complex_det::ComplexGeminal::from_py(&gd_py)?;
        let jd_obj = wf.getattr("jastrow_data")?;
        let jdata = JastrowData::from_py_complex_aware(&jd_obj)?;
        let np = ham.py().import("numpy")?;
        let j3d = jd_obj.getattr("jastrow_three_body_data")?;
        let (j3c, j3m) = if j3d.is_none() {
            (None, None)
        } else {
            let jm: numpy::PyReadonlyArray2<f64> = np
                .call_method1("asarray", (j3d.getattr("j_matrix")?,))?
                .call_method1("astype", (np.getattr("float64")?,))?
                .extract()?;
            let jm_arr = jm.as_array().to_owned();
            if jm_arr.iter().all(|&x| x == 0.0) {
                (None, None)
            } else {
                (
                    Some(crate::complex_det::ComplexOrb::from_py(
                        &j3d.getattr("orb_data")?,
                    )?),
                    Some(jm_arr),
                )
            }
        };
        let n_up: usize = gd_py.getattr("num_electron_up")?.extract()?;
        let n_dn: usize = gd_py.getattr("num_electron_dn")?.extract()?;
        let coulomb = coulomb_potential::CoulombPotentialData::from_py(
            &ham.getattr("coulomb_potential_data")?,
        )?;
        let ewald_si = coulomb.ewald.as_ref().map(|ew| {
            ewald::compute_ion_structure_factor(
                &ew.g_vectors.view(),
                &coulomb.effective_charges,
                &coulomb.positions.view(),
            )
        });
        Ok(LeTermsC {
            cgem,
            jdata,
            j3c,
            j3m,
            coulomb,
            ewald_si,
            weights,
            grid_points,
            n_up,
            n_dn,
        })
    }

    /// Re(complex e_L) = Re(T_complex) + Ewald/bare + ECP-local + Re(ECP-nl-complex).
    fn local_energy(&self, r_up: &ArrayView2<f64>, r_dn: &ArrayView2<f64>) -> f64 {
        let j3m_view = self.j3m.as_ref().map(|m| m.view());
        let j3_kin: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)> =
            match (&self.j3c, &j3m_view) {
                (Some(o), Some(m)) => Some((o, m)),
                _ => None,
            };
        let t = crate::complex_det::compute_kinetic_withj_complex(
            &self.cgem,
            &self.jdata,
            j3_kin,
            r_up,
            r_dn,
        )
        .re;
        let cell_view = self.coulomb.cell.view();
        let cell_nmax = if self.coulomb.pbc {
            Some((&cell_view, self.coulomb.n_max))
        } else {
            None
        };
        let mut v = if let Some(ew) = &self.coulomb.ewald {
            let esf_owned = ewald::compute_electron_structure_factor(
                r_up,
                r_dn,
                &ew.g_vectors.view(),
                &cell_view,
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
                self.coulomb.n_max,
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
                self.coulomb.n_max,
                &self.coulomb.effective_charges,
                &self.coulomb.positions.view(),
                si,
                esf,
            );
            ee + ei + ew.ii
        } else {
            coulomb_potential::compute_bare_coulomb_potential(
                &self.coulomb.positions.view(),
                &self.coulomb.effective_charges,
                r_up,
                r_dn,
            )
        };
        if let Some(e) = &self.coulomb.ecp {
            v += coulomb_potential::compute_ecp_local_parts_all_pairs(
                &self.coulomb.positions.view(),
                &e.nuc_l,
                &e.e_l,
                &e.c_l,
                &e.p_l,
                r_up,
                r_dn,
                cell_nmax,
            );
            let cell_opt = if self.coulomb.pbc {
                Some(&cell_view)
            } else {
                None
            };
            // Rank-1 (Sherman-Morrison + incremental Jastrow) route; the full-recompute
            // reference stays as the fallback and behind JQMC_RS_ECP_COMPLEX_REFERENCE.
            let fast = if std::env::var_os("JQMC_RS_ECP_COMPLEX_REFERENCE").is_some() {
                None
            } else {
                crate::complex_ecp::compute_ecp_nonlocal_withj_complex_fast(
                    &self.cgem,
                    &self.jdata,
                    self.j3c.as_ref(),
                    j3m_view.as_ref(),
                    r_up,
                    r_dn,
                    &self.coulomb.positions.view(),
                    &e.nuc_n,
                    &e.am,
                    &e.e_n,
                    &e.c_n,
                    &e.p_n,
                    &self.weights,
                    &self.grid_points.view(),
                    cell_opt,
                    self.coulomb.n_max,
                    1,
                    false,
                    None,
                )
            };
            let vnl = match fast {
                Some(vnl) => vnl,
                None => crate::complex_ecp::compute_ecp_nonlocal_withj_complex(
                    &self.cgem,
                    Some(&self.jdata),
                    self.j3c.as_ref(),
                    j3m_view.as_ref(),
                    r_up,
                    r_dn,
                    &self.coulomb.positions.view(),
                    &e.nuc_n,
                    &e.am,
                    &e.e_n,
                    &e.c_n,
                    &e.p_n,
                    &self.weights,
                    &self.grid_points.view(),
                    cell_opt,
                    self.coulomb.n_max,
                    1, // NN_default = 1 (nearest-neighbour, matching jqmc's forward ECP-nl)
                ),
            };
            v += vnl.re;
        }
        t + v
    }

    fn init_positions(&self, rng: &mut Rng) -> (Array2<f64>, Array2<f64>) {
        let mut place = |n: usize| -> Array2<f64> {
            let mut r = Array2::<f64>::zeros((n, 3));
            for e in 0..n {
                if self.coulomb.pbc {
                    let u = [rng.uniform(), rng.uniform(), rng.uniform()];
                    for c in 0..3 {
                        r[[e, c]] = u[0] * self.coulomb.cell[[0, c]]
                            + u[1] * self.coulomb.cell[[1, c]]
                            + u[2] * self.coulomb.cell[[2, c]];
                    }
                } else {
                    let a = rng.below(self.coulomb.positions.nrows());
                    for c in 0..3 {
                        r[[e, c]] = self.coulomb.positions[[a, c]] + 0.5 * rng.normal();
                    }
                }
            }
            r
        };
        (place(self.n_up), place(self.n_dn))
    }
}

fn run_complex(
    terms: &LeTermsC,
    seed: u64,
    dt: f64,
    eps: f64,
    moves_per_block: usize,
    n_warmup: usize,
    n_blocks: usize,
) -> (Vec<f64>, usize, usize) {
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let (mut r_up, mut r_dn) = terms.init_positions(&mut rng);
    let mut key = [seed as u32, (seed >> 32) as u32];
    let cell_view = terms.coulomb.cell.view();
    let cell_nmax = if terms.coulomb.pbc {
        Some((&cell_view, terms.coulomb.n_max))
    } else {
        None
    };
    let j3m_v = terms.j3m.as_ref().map(|m| m.view());
    let mut energies = Vec::with_capacity(n_blocks);
    let (mut accepted, mut total_moves) = (0usize, 0usize);
    // Streaming complex geminal + inverse: built once, then carried through the sweep's
    // Sherman-Morrison chain (the sweep does not rebuild it per block).
    let mut geminal =
        crate::complex_det::compute_geminal_complex(&terms.cgem, &r_up.view(), &r_dn.view());
    let mut geminal_inv = crate::complex_det::inverse_complex(&geminal);
    for blk in 0..(n_warmup + n_blocks) {
        let normals: Vec<f64> = (0..moves_per_block).map(|_| rng.normal()).collect();
        let res = mcmc::mcmc_sweep_complex(
            &terms.cgem,
            &terms.jdata,
            terms.j3c.as_ref(),
            j3m_v.as_ref(),
            &terms.coulomb.positions.view(),
            &terms.coulomb.effective_charges,
            &r_up.view(),
            &r_dn.view(),
            &geminal.view(),
            &geminal_inv.view(),
            dt,
            eps,
            key,
            &normals,
            moves_per_block,
            cell_nmax,
        );
        r_up = res.r_up;
        r_dn = res.r_dn;
        geminal = res.geminal;
        geminal_inv = res.geminal_inv;
        key = res.final_key;
        accepted += res.accepted;
        total_moves += moves_per_block;
        if blk >= n_warmup {
            energies.push(terms.local_energy(&r_up.view(), &r_dn.view()));
        }
    }
    (energies, accepted, total_moves)
}

/// `native_vmc_complex(...)` -- the full native VMC chain for a COMPLEX Bloch twist geminal.
/// Returns (Re(e_L)[n_blocks], accept_rate).
#[pyfunction]
#[pyo3(name = "native_vmc_complex")]
#[allow(clippy::too_many_arguments)]
pub fn py_native_vmc_complex(
    hamiltonian_data: &Bound<'_, PyAny>,
    weights: numpy::PyReadonlyArray1<f64>,
    grid_points: numpy::PyReadonlyArray2<f64>,
    seed: u64,
    dt: f64,
    epsilon: f64,
    moves_per_block: usize,
    n_warmup: usize,
    n_blocks: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let terms = LeTermsC::from_py(
        hamiltonian_data,
        weights.as_slice()?.to_vec(),
        grid_points.as_array().to_owned(),
    )?;
    let (energies, accepted, total_moves) = hamiltonian_data.py().allow_threads(|| {
        run_complex(
            &terms,
            seed,
            dt,
            epsilon,
            moves_per_block,
            n_warmup,
            n_blocks,
        )
    });
    Ok((energies, accepted as f64 / total_moves.max(1) as f64))
}
