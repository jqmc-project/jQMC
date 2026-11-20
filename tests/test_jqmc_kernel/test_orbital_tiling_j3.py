"""Independent physical-supercell J3 with a frozen primitive-block determinant."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_three_body_data, compute_Jastrow_three_body
from jqmc.jqmc_tool import build_supercell_jastrow_three_body
from jqmc.molecular_orbital import MOs_data
from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction
from jqmc.upf_wrapper import PrimitiveShell
from jqmc.wavefunction import compute_kinetic_energy, evaluate_wavefunction
from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
from tests.test_jqmc_kernel.test_orbital_tiling_supercell import _expanded_mos, _source
from tests.test_jqmc_kernel.test_qe_orbital_tiling import _explicit_supercell_pw_sd


def _supercell_j3(structure, mo=False):
    shells = [PrimitiveShell(0, (0.8, 1.3), (1.0, 1.0), 0.0, 1.0)]
    j3 = build_supercell_jastrow_three_body(structure, [shells] * structure.natom, "ao-full")
    if mo:
        # Independent supercell MOs, deliberately unrelated to Slater bands.
        coefficients = np.random.default_rng(89).normal(size=(3, j3.orb_num))
        j3 = Jastrow_three_body_data.build(orb_data=MOs_data.build(aos_data=j3.orb_data, mo_coefficients=coefficients))
    matrix = np.random.default_rng(92).uniform(-0.03, 0.03, size=(j3.orb_num, j3.orb_num + 1))
    matrix[:, :-1] = (matrix[:, :-1] + matrix[:, :-1].T) / 2
    return j3.replace(j_matrix=matrix)


def _positions(structure):
    centers = np.asarray(structure.positions)
    return (
        centers + [[0.35, 0.41, 0.27], [0.61, -0.32, 0.28]],
        centers + [[-0.29, 0.37, 0.49], [0.44, 0.25, -0.36]],
    )


@pytest.mark.heavy
@pytest.mark.without_chdir
@pytest.mark.parametrize("backend", ["bspline3", "bspline5", "pgto_cart", "pgto_sphe"])
@pytest.mark.parametrize("sector", ["gamma", "general"])
@pytest.mark.parametrize("restricted", [True, False])
def test_supercell_j3_matches_expanded_determinant(backend, sector, restricted):
    """A nonzero cross-cell J3 preserves wavefunction and kinetic/local energy equality."""
    source = _source(backend, sector)
    down = None if restricted else _source(backend, sector, independent_down=True)
    sd = SD_data.build_tiled(up_source=source, down_source=down)
    j3 = _supercell_j3(source.structure_data)
    assert j3.orb_num == 2 * source.structure_data.natom
    assert np.any(j3.j_matrix[:2, 2:4] != 0)
    ham = build_hamiltonian_from_wavefunction(source.structure_data, source, sd, j2_param=None, jastrow_three_body_data=j3)
    explicit_sd = SD_data.build(
        num_electron_up=2,
        num_electron_dn=2,
        mos_up=_expanded_mos(sd.orb_data_up_spin, backend),
        mos_dn=_expanded_mos(sd.orb_data_dn_spin, backend),
        time_reversal=False,
    )
    explicit = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=explicit_sd))
    up, dn = _positions(ham.structure_data)
    atol, rtol = get_tolerance_min(("tiling_eval", "jastrow_eval", "det_grad_lap", "local_energy"), "medium")
    for evaluate in (
        lambda h: evaluate_wavefunction(h.wavefunction_data, up, dn),
        lambda h: compute_kinetic_energy(h.wavefunction_data, up, dn),
        lambda h: compute_local_energy(h, up, dn, np.eye(3)),
    ):
        actual, expected = evaluate(ham), evaluate(explicit)
        assert np.isfinite(actual) and np.isfinite(expected)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    # Translating one electron by a physical lattice vector changes no J3.
    moved = up.copy()
    moved[0] += ham.structure_data.cell[0]
    np.testing.assert_allclose(
        compute_Jastrow_three_body(j3, moved, dn), compute_Jastrow_three_body(j3, up, dn), atol=atol, rtol=rtol
    )


@pytest.mark.heavy
@pytest.mark.without_chdir
@pytest.mark.parametrize("mo", [False, True])
def test_supercell_j3_matrix_gradient_and_hdf5_with_tiled_pw(mo, tmp_path):
    """J3 M remains optimizable; its independent basis and full matrix survive persistence."""
    ham = _hamiltonian("pw", True)
    j3 = _supercell_j3(ham.structure_data, mo)
    wf = ham.wavefunction_data.replace(jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3))
    ham = ham.replace(wavefunction_data=wf)
    ham.sanity_check()
    up, dn = _positions(ham.structure_data)
    explicit = ham.replace(wavefunction_data=wf.replace(geminal_data=_explicit_supercell_pw_sd(wf.geminal_data)))
    flags = dict(opt_geminal=False, opt_lambda_basis_exp=False, opt_lambda_basis_coeff=False)
    blocks = wf.get_variational_blocks(state_token=("j3-tiling", 0), **flags)
    assert [b.name for b in blocks] == ["j3_matrix"]

    def log_amplitude(matrix, enabled):
        current = wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=matrix)))
        current = current.with_param_grad_mask(opt_J3_param=enabled, **flags)
        return jnp.log(jnp.abs(evaluate_wavefunction(current, up, dn)))

    # Independent pair sum gives d log|Psi| / dM, including the linear column.
    phi = np.asarray(j3.compute_orb_api(j3.orb_data, np.concatenate([up, dn]))).real
    expected = sum(np.outer(phi[:, i], phi[:, k]) for i in range(4) for k in range(i + 1, 4))
    expected = np.column_stack([(expected + expected.T) / 2, phi.sum(axis=1)])
    gradient = jax.grad(log_amplitude)(jnp.asarray(j3.j_matrix), True)
    # M is constrained symmetric; compare its symmetric tangent derivative.
    gradient = np.array(gradient)
    gradient[:, :-1] = (gradient[:, :-1] + gradient[:, :-1].T) / 2
    atol, rtol = get_tolerance_min(("jastrow_eval", "det_grad_lap", "local_energy"), "medium")
    np.testing.assert_allclose(gradient, expected, atol=atol, rtol=rtol)
    np.testing.assert_array_equal(jax.grad(log_amplitude)(jnp.asarray(j3.j_matrix), False), np.zeros_like(j3.j_matrix))
    with pytest.raises(NotImplementedError, match="frozen"):
        wf.get_variational_blocks(state_token=("j3-tiling", 0), opt_geminal=True)
    path = tmp_path / "tiled-j3.h5"
    ham.save_to_hdf5(path)
    restored = Hamiltonian_data.load_from_hdf5(path)
    saved_j3 = restored.wavefunction_data.jastrow_data.jastrow_three_body_data
    np.testing.assert_array_equal(saved_j3.j_matrix, j3.j_matrix)
    np.testing.assert_array_equal(saved_j3.orb_data.structure_data.cell, ham.structure_data.cell)
    for reference in (explicit, restored):
        np.testing.assert_allclose(
            evaluate_wavefunction(reference.wavefunction_data, up, dn), evaluate_wavefunction(wf, up, dn), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            compute_local_energy(reference, up, dn, np.eye(3)),
            compute_local_energy(ham, up, dn, np.eye(3)),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.light
@pytest.mark.without_chdir
@pytest.mark.parametrize("flag", ["opt_J3_basis_exp", "opt_J3_basis_coeff"])
@pytest.mark.parametrize("mo", [False, True])
def test_tiled_j3_basis_flags_are_admitted(flag, mo):
    """Determinant tiling does not disable the independent J3 basis flags.

    This is an admission contract, not a claim that the shared periodic-basis
    optimizer/persistence defect has been repaired.
    """
    from jqmc.jqmc_mcmc import MCMC
    from jqmc.jqmc_optimizer import ParameterSelection

    ham = _hamiltonian("pw", True)
    j3 = _supercell_j3(ham.structure_data, mo)
    wf = ham.wavefunction_data.replace(jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3))
    ham = ham.replace(wavefunction_data=wf)
    flags = dict(opt_geminal=False, **{flag: True})
    expected = "j3_basis_exp" if flag == "opt_J3_basis_exp" else "j3_basis_coeff"
    blocks = wf.get_variational_blocks(state_token=("j3-basis", 0), **flags)
    assert [b.name for b in blocks] == ["j3_matrix", expected]
    assert all(np.isfinite(b.values).all() for b in blocks)
    wf.with_param_grad_mask(**flags)

    driver = MCMC(hamiltonian_data=ham, num_walkers=2, mcmc_seed=718, comput_log_WF_param_deriv=True)
    parameters = ParameterSelection(**flags)
    driver.select_parameters(parameters.gradient_flag_kwargs())
    assert [b.name for b in driver.variational_blocks(parameters)] == ["j3_matrix", expected]


@pytest.mark.light
@pytest.mark.without_chdir
def test_j3_rejects_determinant_blocks_and_primitive_cell():
    """J3 cannot silently reuse tiled Bloch orbitals or a primitive-cell periodic basis."""
    ham = _hamiltonian("pw", True)
    orbitals = ham.wavefunction_data.geminal_data.orb_data_up_spin
    with pytest.raises(ValueError, match=r"independent.*supercell"):
        Jastrow_three_body_data.init_jastrow_three_body_data(orbitals)
    primitive_j3 = _supercell_j3(orbitals.primitive_structure)
    invalid = ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(
            jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=primitive_j3)
        )
    )
    with pytest.raises(ValueError, match="lattice"):
        invalid.sanity_check()


@pytest.mark.heavy
@pytest.mark.with_chdir
def test_j3_tiled_mcmc_checkpoint_continuation(tmp_path, monkeypatch):
    """Nonzero supercell J3 participates in XLA sampling and resumed RNG streams."""
    from jqmc._checkpoint import merge_rank_checkpoints
    from jqmc._setting import get_tol_hdf5_roundtrip
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.chdir(tmp_path)
    ham = _hamiltonian("pw", True)
    ham = ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(
            jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=_supercell_j3(ham.structure_data))
        )
    )
    driver = MCMC(
        hamiltonian_data=ham,
        num_walkers=2,
        mcmc_seed=812,
        num_mcmc_per_measurement=1,
        comput_log_WF_param_deriv=True,
    )
    initial_matrix = np.array(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix)
    driver.run_optimize(
        num_mcmc_steps=4,
        num_opt_steps=1,
        num_mcmc_bin_blocks=2,
        dump_wavefunction=False,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=True,
        opt_geminal=False,
        optimizer_kwargs={"method": "sr", "delta": 1.0e-3, "epsilon": 1.0e-3},
    )
    ham = driver.hamiltonian_data
    updated_matrix = np.asarray(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix)
    assert np.isfinite(updated_matrix).all() and np.any(updated_matrix != initial_matrix)
    driver.run(num_mcmc_steps=2)
    pattern = str(tmp_path / "rank-{rank}.h5")
    driver.save_to_hdf5(pattern.format(rank=0))
    merged = str(tmp_path / "restart.h5")
    merge_rank_checkpoints(merged, mpi_size=1, driver_type="MCMC", hamiltonian_data=ham, tmp_pattern=pattern)
    restored = MCMC.load_from_hdf5(merged)
    driver.run(num_mcmc_steps=2)
    restored.run(num_mcmc_steps=2)
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    for name in ("latest_r_up_carts", "latest_r_dn_carts", "stored_e_L", "stored_w_L"):
        actual = np.asarray(getattr(restored, "_MCMC__" + name))
        assert actual.size and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, getattr(driver, "_MCMC__" + name), atol=atol, rtol=rtol)
    np.testing.assert_array_equal(restored._MCMC__jax_PRNG_key_list, driver._MCMC__jax_PRNG_key_list)


@pytest.mark.light
@pytest.mark.without_chdir
@pytest.mark.parametrize("choice", ["none", "ao-small", "ao-full"])
def test_qe_tiling_converter_uses_independent_supercell_j3(choice, tmp_path, monkeypatch):
    """The converter assembles common k blocks and a separate UPF-seeded supercell basis."""
    from dataclasses import replace
    from jqmc.jqmc_qe import run_qe2jqmc
    from jqmc.orbital_tiling import TiledOrbitalSet_data
    from tests.test_jqmc_kernel.test_qe_orbital_tiling import _synthetic_source

    references, projections = _synthetic_source("bspline")
    references = [replace(r, wfc=replace(r.wfc, k_point=np.array([i / 2, 0, 0]) @ r.cell.b)) for i, r in enumerate(references)]

    class Seed:
        def __init__(self, *args, **kwargs):
            pass

        def shells_for_element(self, label):
            return [PrimitiveShell(0, (0.8, 1.3), (1.0, 1.0), 0.0, 1.0)]

        def is_coulomb_ae_for_element(self, label):
            return True

        def z_valence_for_element(self, label):
            return 2

    monkeypatch.setattr("jqmc.upf_wrapper.UPFExponentInit", Seed)
    monkeypatch.setattr("jqmc.qe_wrapper.qe_save_is_lsda", lambda path: False)
    monkeypatch.setattr("jqmc.qe_wrapper._find_save_dir", lambda path: str(tmp_path))
    monkeypatch.setattr("jqmc.qe_wrapper.load_qe_kpoint_list", lambda path: [(r.wfc.k_point, 0.5) for r in references])
    monkeypatch.setattr("jqmc.qe_wrapper.load_qe_reference", lambda path, ik=1: references[ik - 1])
    monkeypatch.setattr("jqmc.pw_pgto_projection.taper_qe_reference", lambda ref, **kw: (ref, None))
    monkeypatch.setattr(
        "jqmc.pw_pgto_projection.project_pw_to_spline", lambda ref, **kw: projections[int(np.any(ref.wfc.k_point))]
    )
    path = tmp_path / "converted.h5"
    run_qe2jqmc(
        {"qe_save_dir": str(tmp_path), "kind": "bspline", "tiling": [2, 1, 1], "upf_paths": ["He=unused.upf"]},
        {"j3_basis_type": choice},
        str(path),
    )
    restored = Hamiltonian_data.load_from_hdf5(path)
    orbitals = restored.wavefunction_data.geminal_data.orb_data_up_spin
    assert isinstance(orbitals, TiledOrbitalSet_data)
    assert orbitals.primitive_structure.natom == 1 and restored.structure_data.natom == 2
    j3 = restored.wavefunction_data.jastrow_data.jastrow_three_body_data
    if choice == "none":
        assert j3 is None
    else:
        assert not isinstance(j3.orb_data, TiledOrbitalSet_data)
        np.testing.assert_array_equal(j3.orb_data.structure_data.cell, restored.structure_data.cell)
        from jqmc.hamiltonians import _detile_aos_data_for_save

        home = _detile_aos_data_for_save(j3.orb_data)
        assert set(home.nucleus_index) == {0, 1}
        assert set(np.asarray(home.exponents)) <= {0.8, 1.3}
        assert j3.orb_data.k_point is None or not np.any(j3.orb_data.k_point)
        assert j3.j_matrix.shape == (j3.orb_num, j3.orb_num + 1)
        np.testing.assert_array_equal(j3.j_matrix, np.zeros_like(j3.j_matrix))


@pytest.mark.light
@pytest.mark.without_chdir
@pytest.mark.parametrize("matrix", ["3 3 3", "2 1 1 1 1 1 3 2 1"])
def test_cli_tiling_matrix_is_one_argument(matrix, monkeypatch):
    """CLI and workflow preserve the whole integer matrix as one option value."""
    from click.testing import CliRunner
    from jqmc.jqmc_tool import typer_click_qe
    from jqmc_workflow.qe_workflow import QE_Workflow

    captured = {}

    def convert(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("jqmc.jqmc_qe.run_qe2jqmc", convert)
    values = [int(v) for v in matrix.split()]
    command = QE_Workflow(qe_save_dir="source.save", kind="bspline", tiling=values, j3_basis_type="ao-small")._build_command()
    assert command.count("--tiling") == 1
    assert command[command.index("--tiling") + 1] == matrix
    result = CliRunner().invoke(typer_click_qe, command[2:])
    assert result.exit_code == 0, result.output
    assert captured["qe_params"]["tiling"] == values
    assert captured["jastrow_params"]["j3_basis_type"] == "ao-small"


@pytest.mark.light
@pytest.mark.without_chdir
@pytest.mark.parametrize("matrix", ["2 2", "2.5 1 1", "0 1 1"])
def test_cli_tiling_rejects_malformed_matrix(matrix, monkeypatch):
    """Malformed matrices fail before converter work starts."""
    from click import unstyle
    from click.testing import CliRunner
    from jqmc.jqmc_tool import typer_click_qe

    def unexpected(**kwargs):
        pytest.fail("invalid matrix reached converter")

    monkeypatch.setattr("jqmc.jqmc_qe.run_qe2jqmc", unexpected)
    result = CliRunner().invoke(typer_click_qe, ["source.save", "--tiling", matrix])
    assert result.exit_code == 2
    assert "--tiling" in unstyle(result.output)
