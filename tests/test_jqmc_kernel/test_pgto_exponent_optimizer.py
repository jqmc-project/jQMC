"""Tests for exponent-only variational optimisation of uncontracted pGTOs."""

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

import itertools

import numpy as np
import pytest

from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_PGTO_CONDITION_AUDIT,
    RTOL_PGTO_EXPONENT_OPT_RECOVERY,
    RTOL_PYTHON_FP64_LOOSE,
)

pytestmark = [pytest.mark.with_chdir]


def _one_gaussian_reference(target_exponent: float):
    from jqmc.pw_pgto_projection import CartPrimitive, build_gto_pw_matrix
    from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction

    cell = QECell(a=np.eye(3, dtype=np.float64) * 8.0)
    millers = np.asarray(list(itertools.product(range(-4, 5), repeat=3)), dtype=np.int64)
    bands = QEBands(
        eigenvalues=np.zeros(1, dtype=np.float64),
        occupations=np.ones(1, dtype=np.float64),
        n_electrons=1.0,
        fermi_energy=0.0,
    )
    empty_wfc = QEWavefunction(
        millers=millers,
        coeffs=np.zeros((1, millers.shape[0]), dtype=np.complex128),
        gamma_only=False,
        k_point=np.zeros(3, dtype=np.float64),
    )
    empty_ref = QEReference(
        cell=cell,
        atom_labels=("X",),
        atom_positions=np.zeros((1, 3), dtype=np.float64),
        ecutwfc=20.0,
        fft_grid=(16, 16, 16),
        bands=bands,
        wfc=empty_wfc,
        atom_elements=("X",),
    )
    target = build_gto_pw_matrix(
        empty_ref,
        [CartPrimitive(center=(0.0, 0.0, 0.0), ang=(0, 0, 0), exponent=target_exponent)],
    )[:, 0]
    wfc = QEWavefunction(
        millers=millers,
        coeffs=target[None, :],
        gamma_only=False,
        k_point=np.zeros(3, dtype=np.float64),
    )
    return QEReference(
        cell=cell,
        atom_labels=("X",),
        atom_positions=np.zeros((1, 3), dtype=np.float64),
        ecutwfc=20.0,
        fft_grid=(16, 16, 16),
        bands=bands,
        wfc=wfc,
        atom_elements=("X",),
    )


@pytest.mark.light
def test_exponent_only_optimizer_recovers_gaussian_and_keeps_fit_coefficient():
    """Exponent only optimizer recovers gaussian and keeps fit coefficient."""
    from jqmc.pgto_exponent_optimizer import optimize_pgto_exponents
    from jqmc.upf_wrapper import PrimitiveShell

    target_exponent = 1.7
    reference = _one_gaussian_reference(target_exponent)
    fit_coefficient = -2.5  # deliberately non-unit: it must remain inert metadata
    shells = [
        [
            PrimitiveShell(
                angular_l=0,
                exponents=(0.25,),
                coefficients=(fit_coefficient,),
                rel_error=0.1,
                overlap_cond=1.0,
            )
        ]
    ]

    result = optimize_pgto_exponents(reference, shells, maxiter=100)

    assert result.accepted
    assert result.deployed_optimized_spilling < result.deployed_initial_spilling * 1.0e-4
    assert result.shells_per_atom[0][0].exponents[0] == pytest.approx(
        target_exponent,
        rel=RTOL_PGTO_EXPONENT_OPT_RECOVERY,
    )
    # The PSWFC-fit coefficient is neither an optimisation variable nor a
    # contraction coefficient in the deployed all-primitive projection.
    assert result.shells_per_atom[0][0].coefficients == (fit_coefficient,)


@pytest.mark.light
def test_exponent_only_optimizer_handles_mixed_pgto_pw_objective():
    """Exponent only optimizer handles mixed pgto pw objective."""
    from jqmc.pgto_exponent_optimizer import optimize_pgto_exponents
    from jqmc.upf_wrapper import PrimitiveShell

    target_exponent = 1.7
    reference = _one_gaussian_reference(target_exponent)
    shells = [
        [
            PrimitiveShell(
                angular_l=0,
                exponents=(0.25,),
                coefficients=(9.0,),
                rel_error=0.1,
                overlap_cond=1.0,
            )
        ]
    ]

    result = optimize_pgto_exponents(
        reference,
        shells,
        g_cut=0.8,
        exponent_floor=0.1,
        maxiter=100,
    )

    assert result.accepted
    assert result.deployed_optimized_spilling < result.deployed_initial_spilling * 1.0e-4
    assert result.shells_per_atom[0][0].exponents[0] == pytest.approx(
        target_exponent,
        rel=RTOL_PGTO_EXPONENT_OPT_RECOVERY,
    )


@pytest.mark.light
def test_exponent_only_optimizer_handles_fixed_cutoff_fft_overlap():
    """Numeric ecut uses the FFT-complete fixed mixed-basis projector."""
    from jqmc.pgto_exponent_optimizer import optimize_pgto_exponents
    from jqmc.upf_wrapper import PrimitiveShell

    reference = _one_gaussian_reference(target_exponent=1.7)
    shells = [
        [
            PrimitiveShell(
                angular_l=0,
                exponents=(0.25,),
                coefficients=(3.0,),
                rel_error=0.1,
                overlap_cond=1.0,
            )
        ]
    ]

    result = optimize_pgto_exponents(
        reference,
        shells,
        g_cut=0.8,
        exponent_floor=0.1,
        fixed_mixed_overlap=True,
        maxiter=100,
    )

    assert result.accepted
    assert result.deployed_optimized_spilling < result.deployed_initial_spilling
    assert result.optimized_condition == pytest.approx(
        result.deployed_optimized_condition,
        rel=RTOL_PGTO_CONDITION_AUDIT,
    )


@pytest.mark.light
@pytest.mark.parametrize("k_fraction", [0.0, 0.24])
def test_mixed_constraint_is_generated_ao_full_overlap_condition(k_fraction):
    """The safety constraint must condition the raw emitted pGTO+PW overlap."""
    from dataclasses import replace

    from jqmc.pgto_exponent_optimizer import _build_layout, _reference_block, _spilling_and_logcond
    from jqmc.upf_wrapper import PrimitiveShell

    reference = _one_gaussian_reference(target_exponent=1.7)
    reference = replace(
        reference,
        wfc=replace(reference.wfc, k_point=k_fraction * reference.cell.b[0]),
    )
    shells = [
        [
            PrimitiveShell(
                angular_l=0,
                exponents=(0.25, 0.7),
                coefficients=(1.0, 1.0),
                rel_error=0.1,
                overlap_cond=1.0,
            )
        ]
    ]
    layout = _build_layout(reference, shells)
    block = _reference_block(
        reference,
        occ_only=True,
        occ_tol=1.0e-6,
        g_cut=0.8,
        fixed_mixed_overlap=False,
        chunk_rows=128,
        condition_exponent_floor=0.1,
        condition_max_angular=0,
    )
    _, logcond = _spilling_and_logcond(
        np.log(layout.initial_exponents),
        layout,
        (block,),
        1.0e-8,
    )
    from jqmc.pw_pgto_projection import (
        _generated_pgto_overlap_condition,
        build_pgto_primitives,
        pw_gvectors_within,
    )

    prims = build_pgto_primitives(shells, reference.atom_positions)
    pw_millers, _ = pw_gvectors_within(reference, 0.8)
    direct_condition = _generated_pgto_overlap_condition(reference, prims, pw_millers)

    assert float(np.exp(logcond)) == pytest.approx(
        direct_condition,
        rel=RTOL_PGTO_CONDITION_AUDIT,
    )


@pytest.mark.light
def test_periodic_pgto_overlap_matches_emitted_ao_leaf():
    """The converter-side analytic overlap must match the emitted pGTO AO."""
    from dataclasses import replace

    from jqmc.atomic_orbital import compute_overlap_matrix
    from jqmc.pw_pgto_projection import build_pgto_primitives, periodic_pgto_overlap_matrix, project_pw_to_pgto
    from jqmc.qe_to_jqmc import _build_pgto_leaf_and_mo, _build_structure_data
    from jqmc.upf_wrapper import PrimitiveShell

    reference = _one_gaussian_reference(target_exponent=1.7)
    reference = replace(
        reference,
        atom_labels=("H",),
        atom_elements=("H",),
        wfc=replace(reference.wfc, k_point=0.24 * reference.cell.b[0]),
    )
    shells = [
        [
            PrimitiveShell(0, (0.25, 0.7), (1.0, 1.0), 0.1, 1.0),
            PrimitiveShell(1, (0.4,), (1.0,), 0.1, 1.0),
        ]
    ]
    prims = build_pgto_primitives(shells, reference.atom_positions)
    expected = periodic_pgto_overlap_matrix(reference, prims)

    projection = project_pw_to_pgto(reference, shells)
    structure, k_frac = _build_structure_data(reference)
    pgto_leaf, _ = _build_pgto_leaf_and_mo(
        projection,
        structure,
        reference.atom_positions,
        k_frac,
    )
    actual = np.asarray(compute_overlap_matrix(pgto_leaf))

    assert actual == pytest.approx(expected, rel=RTOL_PYTHON_FP64_LOOSE, abs=ATOL_PYTHON_FP64_LOOSE)


@pytest.mark.light
def test_unsuccessful_slsqp_point_is_never_adopted(monkeypatch):
    """SLSQP without success (here: the iteration limit) raises; its returned point is not used."""
    from scipy.optimize import OptimizeResult

    from jqmc import pgto_exponent_optimizer
    from jqmc.upf_wrapper import PrimitiveShell

    shells = [[PrimitiveShell(angular_l=0, exponents=(0.25,), coefficients=(1.0,), rel_error=0.1, overlap_cond=1.0)]]

    def _unsuccessful(fun, z0, **kwargs):
        return OptimizeResult(x=np.full_like(z0, 3.4e3), success=False, status=9, message="Iteration limit reached", nit=2)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("the unsuccessful SLSQP point was evaluated")

    monkeypatch.setattr(pgto_exponent_optimizer, "minimize", _unsuccessful)
    monkeypatch.setattr(pgto_exponent_optimizer, "_deployed_diagnostics", _must_not_run)
    with pytest.raises(pgto_exponent_optimizer.PGTOExponentOptimizationError, match="status=9, nit=2: Iteration limit"):
        pgto_exponent_optimizer.optimize_pgto_exponents(_one_gaussian_reference(target_exponent=1.7), shells, maxiter=20)


@pytest.mark.light
def test_failed_production_condition_audit_restores_seed_basis(monkeypatch):
    """A rejected SLSQP point must not be modified or partially adopted."""
    from jqmc import pgto_exponent_optimizer
    from jqmc.upf_wrapper import PrimitiveShell

    reference = _one_gaussian_reference(target_exponent=1.7)
    shells = [
        [
            PrimitiveShell(
                angular_l=0,
                exponents=(0.25,),
                coefficients=(1.0,),
                rel_error=0.1,
                overlap_cond=1.0,
            )
        ]
    ]
    deployed = iter(((1.0e-2, 1.0), (1.0e-3, float("inf"))))
    monkeypatch.setattr(
        pgto_exponent_optimizer,
        "_deployed_diagnostics",
        lambda *args, **kwargs: next(deployed),
    )

    result = pgto_exponent_optimizer.optimize_pgto_exponents(reference, shells, maxiter=20)

    assert not result.accepted
    assert result.shells_per_atom == shells
    assert result.deployed_optimized_condition == float("inf")


def _gamma_overlap_probe(n_pairs=8, n_gto=3, cross_rank="full"):
    """A nonsingular raw overlap with the actual Gamma cos/sin PW map."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    from jqmc.pw_pgto_projection import _gamma_real_pw_basis_map

    millers = np.zeros((2 * n_pairs + 1, 3), dtype=np.int64)
    millers[:, 0] = np.arange(-n_pairs, n_pairs + 1)
    indices, weights, scales = _gamma_real_pw_basis_map(millers)
    block = SimpleNamespace(
        cell_volume=8.0,
        condition_pw_source_indices=jnp.asarray(indices),
        condition_pw_source_weights=jnp.asarray(weights),
        condition_pw_diagonal_scale=jnp.asarray(scales),
        condition_pw_bulk_indices=jnp.asarray(np.flatnonzero(scales == 0.5)),
        condition_pw_fixed_indices=jnp.asarray(np.flatnonzero(scales != 0.5)),
    )
    # Complex Fourier coefficients with real Gamma cos/sin coefficients.
    row = millers[:, 0, None]
    col = np.arange(1, n_gto + 1)[None, :]
    cross = np.exp(1j * row * col) / (4 * np.sqrt(len(millers)))
    if cross_rank == "dependent":
        cross = np.repeat(cross[:, :1], n_gto, axis=1)
    elif cross_rank == "zero":
        cross = np.zeros_like(cross)
    return block, jnp.asarray(cross), jnp.diag(jnp.arange(1, n_gto + 1, dtype=jnp.float64) + 1)


def _dense_gamma_logcond(s_gg, g_low, normalisations, block):
    """Independent unreduced matrix oracle, including every repeated PW mode."""
    import jax.numpy as jnp

    cross = (g_low * normalisations[None, :]).conj().T
    cross = jnp.sum(cross[:, block.condition_pw_source_indices] * block.condition_pw_source_weights[None, :, :], axis=2)
    full = jnp.block([[s_gg, cross], [cross.conj().T, block.cell_volume * jnp.diag(block.condition_pw_diagonal_scale)]])
    eig = jnp.linalg.eigvalsh((full + full.conj().T) / 2)
    return jnp.log(eig[-1]) - jnp.log(eig[0])


@pytest.mark.light
@pytest.mark.parametrize("cross_rank", ["full", "dependent", "zero"])
@pytest.mark.parametrize("n_pairs", [0, 1, 2, 8])
def test_gamma_condition_compression_preserves_dense_value_and_gradient(n_pairs, cross_rank):
    """Thin Gamma reduction preserves raw eigenvalue extrema and first derivatives."""
    import jax
    import jax.numpy as jnp

    from jqmc.pgto_exponent_optimizer import _full_overlap_logcond

    block, cross, s_gg = _gamma_overlap_probe(n_pairs=n_pairs, n_gto=4, cross_rank=cross_rank)

    def evaluate(z, implementation):
        return implementation(
            s_gg + jnp.diag(z),
            cross * jnp.exp(z)[None, :] + z[None, :] / (1 + cross.shape[0]),
            jnp.ones_like(z),
            block,
        )

    z = jnp.zeros(s_gg.shape[0])
    actual, actual_grad = jax.jit(jax.value_and_grad(lambda z: evaluate(z, _full_overlap_logcond)))(z)
    expected, expected_grad = jax.value_and_grad(lambda z: evaluate(z, _dense_gamma_logcond))(z)
    np.testing.assert_allclose(actual, expected, rtol=RTOL_PGTO_CONDITION_AUDIT, atol=ATOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=RTOL_PGTO_CONDITION_AUDIT, atol=ATOL_PYTHON_FP64_LOOSE)
    assert np.all(np.isfinite(actual_grad))


@pytest.mark.light
def test_gamma_exponent_condition_gradient_matches_dense_generated_overlap():
    """Real pGTO exponents retain the unreduced full-overlap analytic gradient."""
    import jax
    import jax.numpy as jnp

    from jqmc.pgto_exponent_optimizer import (
        _build_g,
        _build_layout,
        _cart_primitive_normalisations,
        _full_overlap_logcond,
        _periodic_pgto_overlap,
        _reference_block,
    )
    from jqmc.upf_wrapper import PrimitiveShell

    reference = _one_gaussian_reference(1.7)
    layout = _build_layout(reference, [[PrimitiveShell(0, (0.4, 1.3), (1.0, 1.0), 0.1, 1.0)]])
    block = _reference_block(
        reference,
        occ_only=True,
        occ_tol=1.0e-6,
        g_cut=1.2,
        fixed_mixed_overlap=False,
        chunk_rows=32,
        condition_exponent_floor=0.3,
        condition_max_angular=0,
    )
    assert block.condition_pw_bulk_indices.shape[0] > len(layout.initial_exponents)
    q_low = block.subtract_q_chunks.reshape((-1, 3))[: block.n_pw]

    def evaluate(z, implementation):
        return implementation(
            _periodic_pgto_overlap(z, layout, block),
            _build_g(z, layout, q_low),
            _cart_primitive_normalisations(z, layout),
            block,
        )

    z = jnp.log(layout.initial_exponents)
    actual, actual_grad = jax.jit(jax.value_and_grad(lambda z: evaluate(z, _full_overlap_logcond)))(z)
    expected, expected_grad = jax.value_and_grad(lambda z: evaluate(z, _dense_gamma_logcond))(z)
    np.testing.assert_allclose(actual, expected, rtol=RTOL_PGTO_CONDITION_AUDIT, atol=ATOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=RTOL_PGTO_CONDITION_AUDIT, atol=ATOL_PYTHON_FP64_LOOSE)


@pytest.mark.light
def test_gamma_condition_value_and_gradient_do_not_build_pw_squared_arrays():
    """Both objective and derivative traces keep the PW block implicit."""
    import jax
    import jax.numpy as jnp

    from jqmc.pgto_exponent_optimizer import _full_overlap_logcond

    block, cross, s_gg = _gamma_overlap_probe(n_pairs=64)
    n_pw, n_gto = cross.shape
    trace = jax.make_jaxpr(jax.value_and_grad(lambda z: _full_overlap_logcond(s_gg + jnp.diag(z), cross, jnp.exp(z), block)))(
        jnp.zeros(n_gto)
    )

    def check_graph(graph):
        if hasattr(graph, "jaxpr"):
            graph = graph.jaxpr
        for eqn in graph.eqns:
            for variable in eqn.outvars:
                shape = getattr(variable.aval, "shape", ())
                assert len(shape) < 2 or not (shape[-2] >= n_pw and shape[-1] >= n_pw), (
                    f"PW-squared intermediate: {eqn.primitive} {shape}"
                )
            for parameter in eqn.params.values():
                if hasattr(parameter, "jaxpr") or hasattr(parameter, "eqns"):
                    check_graph(parameter)
                elif isinstance(parameter, (tuple, list)):
                    for item in parameter:
                        if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                            check_graph(item)

    check_graph(trace)
