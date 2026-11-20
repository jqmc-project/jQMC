"""Native complex-twist paths on the QE converter's dense-MO composite J3 (issues #88 and #89).

``jqmc-tool qe convert-to -j3 mo`` (``run_qe2jqmc(..., {"j3_basis_type": "mo"})``) on a single
k != 0 save builds the J3 orbital with ``jastrow_factor._phase_free_j3_orbitals``: a DENSE complex
MO matrix (n_occ, n_ao) over the same AO basis as the determinant. For ``kind="pgto+pw"`` and
``kind="pgto+bspline"`` that basis is a composite ``AOs_composite_data`` (complex pGTO child plus a
complex plane-wave or B-spline child). Every committed composite-J3 fixture wraps the composite in a
SQUARE identity MO instead, so this shape was never exercised:

* issue #88: the native context builder sent the complex B-spline child of the J3 composite to the
  real ``AosBsplineData::from_py`` (``determinant.rs`` ``extract_aos``), which raised
  ``AttributeError: 'AOs_bspline_complex_data' object has no attribute '_complex'`` from every
  native CPU driver, energy-only MCMC included;
* issue #89: the native complex J3 force / stress kernels (``complex_det.rs`` ``j3_side_complex``,
  ``j3_egrad_side_complex``; ``complex_stress.rs`` ``j3_pbc_kstress_complex``,
  ``j3_value_strain_complex``, ``j3_value_force_complex``) took their MO-aware branch only for a
  non-composite orbital. For a dense MO over a composite they built the AO-rank basis, ignored the
  MO, and sliced the (n_occ, n_occ + 1) ``j_matrix`` with AO rows, which aborted the process with
  an ndarray panic.

Input: the committed QE save ``tests/qe_example_files/Li_k025`` (Li, k = (1/4, 1/4, 1/4), ccECP
auto-discovered), converted by the public ``run_qe2jqmc`` with ``j1 = j2 = 1`` and
``j3_basis_type="mo"``. The force / stress test replaces the converter's zero ``j_matrix`` by a
seeded symmetric one, so the J3 terms are live.

References are independent of the native code:

* #88: the XLA ``compute_local_energy`` recomputed from scratch at the walkers the native
  production MCMC returns (the driver oracle of ``test_richer_production_drivers``);
* #89: reverse-mode AD of the XLA ``Re ln Psi`` and ``Re compute_local_energy`` (identity
  quadrature rotation) with respect to every ``Structure_data.positions`` leaf, the electron
  coordinates, or a homogeneous strain of cell, nuclei and electrons (``Hamiltonian_data._apply_strain``
  and ``r @ F.T``), in the active precision policy. The XLA *analytic* J3 kinetic force is not used:
  it is itself wrong for a complex MO J3 (issue #90). In full precision the AD force agrees with a
  fourth-order central finite difference of the same energy to 1.9e-7 (h = 1e-3, recorded in #89).

Precision scope (#163): full retains the original independent energy / ln-wavefunction
AD formulas at strict. Mixed compares the same native analytical blocks with independently
rebuilt full native blocks from identical serialized physical input and deterministic
coordinates, under the central parameter-derivative fidelity budget. The original medium
nonzero thresholds are captured on mixed entry and checked in both analytical callbacks.
All native arguments, parameter cases and ordinary public-driver backend comparisons remain.

"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import functools
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._test_data_cache import qe_example_files_root
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from tests.test_rs_kernel._native_precision_fidelity import (
    full_source_hamiltonian_factory,
)
from jqmc._precision import get_tolerance_min, precision_policy_snapshot, is_mixed_precision_enabled
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, MCMC, Nv_default, _geminal_inv_batched
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import evaluate_ln_wavefunction

jax.config.update("jax_enable_x64", True)

# The tests read the committed QE save through an absolute path and write the converted
# Hamiltonian to a private temporary directory.
pytestmark = pytest.mark.with_chdir

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAVE = os.path.join(qe_example_files_root(), "Li_k025", "out", "li_k025.save")
_EXTRA_CHILD = {"pgto+pw": AOs_plane_wave_complex_data, "pgto+bspline": AOs_bspline_complex_data}
_J_SEED = 7
_J_SCALE = 0.3
_SEED = 1
_SCALE = 0.6
#: GFMC lattice spacing; the T-move measurement force / stress does not depend on it.
_ALAT = 0.3
_NN = 1
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by the complex J123 ln Psi / local energy and their derivatives.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "wf_kinetic",
    "ewald",
    "coulomb",
)
#: Production MCMC segments of the #88 driver test (three ``run`` calls).
_MCMC_SEGMENTS = (6, 6, 6)
_MCMC_SEED = 34356


@functools.lru_cache(maxsize=None)
def _converted(kind):
    """Public QE conversion of the committed Li k=1/4 save with a J3 MO basis (cached per kind)."""
    from jqmc.jqmc_qe import run_qe2jqmc

    out = os.path.join(tempfile.mkdtemp(prefix="jqmc_j3mo_"), f"ham_{kind.replace('+', '_')}_j3mo.h5")
    run_qe2jqmc(
        {"qe_save_dir": _SAVE, "kind": kind, "upf_paths": [], "ecutwfc": None, "alpha_min": None},
        {"j1_parameter": 1.0, "j2_parameter": 1.0, "j3_basis_type": "mo"},
        out,
    )
    return out


def _converter_hamiltonian(kind, source_path=None):
    ham = Hamiltonian_data.load_from_hdf5(_converted(kind) if source_path is None else source_path)
    j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
    orb = j3.orb_data
    # The converter's J3: a DENSE complex MO over a (pGTO, extra) composite.
    assert isinstance(orb, MOs_data), type(orb)
    assert isinstance(orb.aos_data, AOs_composite_data), type(orb.aos_data)
    gto, extra = orb.aos_data.children
    assert isinstance(extra, _EXTRA_CHILD[kind]), orb.aos_data.children
    coefficients = np.asarray(orb.mo_coefficients)
    assert np.iscomplexobj(coefficients)
    assert coefficients.shape[0] < coefficients.shape[1] == int(gto.num_ao) + int(extra.num_ao), coefficients.shape
    return ham


def _with_seeded_j3(ham, order):
    """Seeded symmetric j_matrix; ``extra_first`` swaps the J3 children and permutes the MO columns."""
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    shape = np.asarray(j3.j_matrix).shape
    n = shape[0]
    j_matrix = _J_SCALE * np.random.default_rng(_J_SEED).standard_normal(shape)
    j_matrix[:, :n] = 0.5 * (j_matrix[:, :n] + j_matrix[:, :n].T)
    orb = j3.orb_data
    if order == "extra_first":
        gto, extra = orb.aos_data.children
        num_gto = int(gto.num_ao)
        c = jnp.asarray(orb.mo_coefficients)
        # The orbital (and so every observable) is unchanged: only the AO order differs.
        orb = orb.replace(
            aos_data=AOs_composite_data.build((extra, gto)),
            mo_coefficients=jnp.concatenate([c[:, num_gto:], c[:, :num_gto]], axis=1),
        )
    j3 = j3.replace(orb_data=orb, j_matrix=jnp.asarray(j_matrix))
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jastrow.replace(jastrow_three_body_data=j3)))


def _full_hamiltonian_source(kind, order):
    """Convert fresh full physical input without consuming the per-mode cached H5."""
    assert not is_mixed_precision_enabled()
    return _with_seeded_j3(_converter_hamiltonian(kind, source_path=_converted.__wrapped__(kind)), order)


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _configuration(ham):
    geminal = ham.wavefunction_data.geminal_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(_SEED)
    r_up = positions[np.arange(num_up) % len(positions)] + rng.normal(scale=_SCALE, size=(num_up, 3))
    r_dn = positions[np.arange(num_dn) % len(positions)] + rng.normal(scale=_SCALE, size=(num_dn, 3))
    return np.ascontiguousarray(r_up, dtype=np.float64), np.ascontiguousarray(r_dn, dtype=np.float64)


def _is_structure(x):
    return isinstance(x, Structure_data)


def _displaced(ham, d_positions):
    """Hamiltonian with every Structure_data.positions leaf displaced by ``d_positions``."""
    return jax.tree_util.tree_map(
        lambda x: x.replace(positions=jnp.asarray(x.positions) + d_positions) if _is_structure(x) else x,
        ham,
        is_leaf=_is_structure,
    )


def _ad_references(ham, r_up, r_dn):
    """Reverse-mode AD of Re ln Psi and Re e_L with respect to nuclei, electrons and strain."""
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))

    def ln_psi(h, a, b):
        return jnp.real(evaluate_ln_wavefunction(h.wavefunction_data, a, b))

    def e_l(h, a, b):
        return jnp.real(compute_local_energy(h, a, b, jnp.eye(3)))

    def strain(scalar):
        def at_strain(eps):
            deformation = jnp.eye(3) + eps
            return scalar(ham._apply_strain(deformation), up @ deformation.T, dn @ deformation.T)

        return np.asarray(jax.jacrev(at_strain)(jnp.zeros((3, 3))), dtype=np.float64)

    as_np = functools.partial(np.asarray, dtype=np.float64)
    pulay_nuclei = as_np(jax.grad(lambda d: ln_psi(_displaced(ham, d), up, dn))(zero))
    drift_up, drift_dn = (as_np(x) for x in jax.grad(lambda a, b: ln_psi(ham, a, b), argnums=(0, 1))(up, dn))
    hf_nuclei = as_np(jax.grad(lambda d: e_l(_displaced(ham, d), up, dn))(zero))
    hf_up, hf_dn = (as_np(x) for x in jax.grad(lambda a, b: e_l(ham, a, b), argnums=(0, 1))(up, dn))
    return {
        "pulay_nuclei": pulay_nuclei,
        "drift_up": drift_up,
        "drift_dn": drift_dn,
        "hf_nuclei": hf_nuclei,
        "hf_up": hf_up,
        "hf_dn": hf_dn,
        "pulay_strain": strain(ln_psi),
        "hf_strain": strain(e_l),
    }


@pytest.mark.heavy
def test_converter_complex_bspline_mo_j3_native_mcmc_energies_match_xla(monkeypatch):
    """Native complex MCMC runs on the converter's pGTO+B-spline MO J3 and stores the XLA e_L.

    The native context builder accepts this Hamiltonian (the complex B-spline child of the J3
    composite) before sampling. Two walkers, seed 34356, fixed quadrature mesh, energy only, three
    production ``run`` calls of six steps on the native complex chunk. After each call the last
    stored complex e_L of every walker must equal ``compute_local_energy`` of the unchanged
    Hamiltonian at the returned walker positions, within the MCMC Rust/XLA driver policy of
    ``test_chain_modes``, and the walkers must move.
    """
    from tests.test_rs_kernel.test_chain_modes import _assert_value_parity
    from tests.test_rs_kernel.test_richer_production_drivers import _complex_chunk_spy

    kind = "pgto+bspline"
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    ham = _converter_hamiltonian(kind)
    calls = _complex_chunk_spy(monkeypatch)
    mcmc = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=_MCMC_SEED,
        epsilon_AS=1.0e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        random_discretized_mesh=False,
    )
    compared = 0
    moved = False
    previous = None
    for steps in _MCMC_SEGMENTS:
        mcmc.run(num_mcmc_steps=steps)
        r_up = np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64)
        r_dn = np.asarray(mcmc._MCMC__latest_r_dn_carts, dtype=np.float64)
        stored = np.asarray(mcmc.e_L)[-1]
        assert np.iscomplexobj(stored) and np.all(np.isfinite(stored)), (kind, stored)
        for w in range(r_up.shape[0]):
            ref = complex(np.asarray(compute_local_energy(ham, jnp.asarray(r_up[w]), jnp.asarray(r_dn[w]), jnp.eye(3))))
            assert np.isfinite(ref), (kind, w, ref)
            _assert_value_parity(complex(stored[w]), ref, f"{kind} MO-J3 MCMC walker {w} stored e_L vs XLA")
            compared += 1
        if previous is not None and not np.array_equal(previous, r_up):
            moved = True
        previous = r_up
    assert compared == 2 * len(_MCMC_SEGMENTS)
    assert moved, f"{kind}: walkers did not move between production calls"
    # The native complex energy chunk (complex=True, native random stream) was the route taken.
    assert calls and all(c == (True, True) for c in calls), (kind, calls)


def _native_derivative_outputs(ham, r_up, r_dn):
    """Original native analytical blocks, with unchanged kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    free_pulay = jqmc_rs.vmc_grad_ln_psi_pbc_complex(geminal, jastrow, r_up, r_dn)
    free_hf = jqmc_rs.vmc_grad_e_L_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, None)
    geminal_inv = np.asarray(
        _geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.complex128
    )
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, True, True, True, precision_policy=precision_policy_snapshot())
    ctx_pulay = ctx.grad_ln_psi_pbc_complex_batch(r_up[None], r_dn[None], geminal_inv)
    ctx_hf = ctx.grad_e_L_pbc_complex_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    gfmc_hf = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(
        geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False
    )
    vmc_pulay_stress = jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(geminal, jastrow, r_up, r_dn)
    vmc_hf_stress = jqmc_rs.vmc_grad_e_L_eps_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID)
    gfmc_stress = jqmc_rs._gfmc_stress_ecp_pbc_complex(
        geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False
    )

    return {
        "Pulay dR (VMC free kernel)": np.asarray(free_pulay[0], dtype=np.float64),
        "Pulay dR (VMC driver ctx)": np.asarray(np.asarray(ctx_pulay[0])[0], dtype=np.float64),
        "drift dr_up (VMC free kernel)": np.asarray(free_pulay[1], dtype=np.float64),
        "drift dr_dn (VMC free kernel)": np.asarray(free_pulay[2], dtype=np.float64),
        "HF dR (VMC free kernel)": np.asarray(free_hf[0], dtype=np.float64),
        "HF dR (VMC driver ctx)": np.asarray(np.asarray(ctx_hf[0])[0], dtype=np.float64),
        "HF dR (GFMC T-move)": np.asarray(gfmc_hf[0], dtype=np.float64),
        "HF dr_up (VMC free kernel)": np.asarray(free_hf[1], dtype=np.float64),
        "HF dr_up (VMC driver ctx)": np.asarray(np.asarray(ctx_hf[1])[0], dtype=np.float64),
        "HF dr_up (GFMC T-move)": np.asarray(gfmc_hf[1], dtype=np.float64),
        "HF dr_dn (VMC free kernel)": np.asarray(free_hf[2], dtype=np.float64),
        "HF dr_dn (VMC driver ctx)": np.asarray(np.asarray(ctx_hf[2])[0], dtype=np.float64),
        "HF dr_dn (GFMC T-move)": np.asarray(gfmc_hf[2], dtype=np.float64),
        "Pulay stress (VMC free kernel)": np.asarray(vmc_pulay_stress, dtype=np.float64),
        "HF stress (VMC free kernel)": np.asarray(vmc_hf_stress, dtype=np.float64),
        "HF stress (GFMC T-move)": np.asarray(gfmc_stress, dtype=np.float64),
    }


@pytest.mark.parametrize("order", ["gto_first", "extra_first"])
@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("pgto+pw", marks=pytest.mark.heavy),
        pytest.param("pgto+bspline", marks=pytest.mark.heavy),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_converter_dense_mo_composite_j3_complex_forces_and_stress_match_ad(kind, order):
    """Native complex VMC / GFMC forces and stress equal AD for the converter's dense-MO composite J3.

    These kernels handle a dense MO over a pGTO(+)PW or pGTO(+)B-spline composite without an
    ``end <= axis_len`` panic. At one seeded configuration, with a seeded symmetric j_matrix and
    the J3 composite in its converted (pGTO, extra) order or in the (extra, pGTO) order with the
    MO columns permuted to match, every native quantity below must equal reverse-mode AD of the
    XLA energy in full precision: the Pulay force and drift (free kernel and MCMC driver
    context), the HF force on nuclei and electrons (free kernel, driver context and the GFMC/LRDMC
    T-move measurement force), and the VMC Pulay, VMC HF and GFMC T-move HF stress.
    """
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_derivative_outputs(fresh, r_up, r_dn)
            expected_fields = {
                "HF dr_dn (VMC free kernel)",
                "HF dR (GFMC T-move)",
                "HF dR (VMC free kernel)",
                "HF stress (GFMC T-move)",
                "drift dr_up (VMC free kernel)",
                "HF dR (VMC driver ctx)",
                "HF stress (VMC free kernel)",
                "Pulay dR (VMC free kernel)",
                "HF dr_dn (VMC driver ctx)",
                "HF dr_up (GFMC T-move)",
                "HF dr_dn (GFMC T-move)",
                "drift dr_dn (VMC free kernel)",
                "Pulay dR (VMC driver ctx)",
                "HF dr_up (VMC driver ctx)",
                "HF dr_up (VMC free kernel)",
                "Pulay stress (VMC free kernel)",
            }
            assert set(values) == expected_fields, values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                expected_shape = (
                    (3, 3)
                    if "stress" in name
                    else r_up.shape
                    if "dr_up" in name
                    else r_dn.shape
                    if "dr_dn" in name
                    else np.asarray(fresh.structure_data.positions).shape
                )
                assert value.shape == expected_shape, (name, value.shape, expected_shape)
                assert np.all(np.isfinite(value)), (name, value)
                assert np.max(np.abs(value)) > live_atol + live_rtol * np.max(np.abs(value)), (name, value)
            return values

        factory = full_source_hamiltonian_factory(lambda: _full_hamiltonian_source(kind, order))
        mixed, full = mixed_vs_full(
            factory,
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ham = _with_seeded_j3(_converter_hamiltonian(kind), order)
    r_up, r_dn = _configuration(ham)
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ref = _ad_references(ham, r_up, r_dn)

    values = _native_derivative_outputs(ham, r_up, r_dn)

    checks = [
        ("Pulay dR (VMC free kernel)", values["Pulay dR (VMC free kernel)"], ref["pulay_nuclei"]),
        ("Pulay dR (VMC driver ctx)", values["Pulay dR (VMC driver ctx)"], ref["pulay_nuclei"]),
        ("drift dr_up (VMC free kernel)", values["drift dr_up (VMC free kernel)"], ref["drift_up"]),
        ("drift dr_dn (VMC free kernel)", values["drift dr_dn (VMC free kernel)"], ref["drift_dn"]),
        ("HF dR (VMC free kernel)", values["HF dR (VMC free kernel)"], ref["hf_nuclei"]),
        ("HF dR (VMC driver ctx)", values["HF dR (VMC driver ctx)"], ref["hf_nuclei"]),
        ("HF dR (GFMC T-move)", values["HF dR (GFMC T-move)"], ref["hf_nuclei"]),
        ("HF dr_up (VMC free kernel)", values["HF dr_up (VMC free kernel)"], ref["hf_up"]),
        ("HF dr_up (VMC driver ctx)", values["HF dr_up (VMC driver ctx)"], ref["hf_up"]),
        ("HF dr_up (GFMC T-move)", values["HF dr_up (GFMC T-move)"], ref["hf_up"]),
        ("HF dr_dn (VMC free kernel)", values["HF dr_dn (VMC free kernel)"], ref["hf_dn"]),
        ("HF dr_dn (VMC driver ctx)", values["HF dr_dn (VMC driver ctx)"], ref["hf_dn"]),
        ("HF dr_dn (GFMC T-move)", values["HF dr_dn (GFMC T-move)"], ref["hf_dn"]),
        ("Pulay stress (VMC free kernel)", values["Pulay stress (VMC free kernel)"], ref["pulay_strain"]),
        ("HF stress (VMC free kernel)", values["HF stress (VMC free kernel)"], ref["hf_strain"]),
        ("HF stress (GFMC T-move)", values["HF stress (GFMC T-move)"], ref["hf_strain"]),
    ]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    for name, value, reference in checks:
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(reference)), (name, reference)
        assert np.all(np.isfinite(value)), (name, value)
        # Non-vacuous: the reference is distinguishable from zero at the comparison tolerance.
        assert np.max(np.abs(reference)) > live_atol + live_rtol * np.max(np.abs(reference)), (name, reference)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"Rust complex dense-MO composite J3 != AD reference: {name} ({kind}, J3 {order})",
        )
