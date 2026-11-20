"""Jastrow contracts: full AD correctness and mixed analytical/full fidelity.

Unrelated value, parity, finite-difference and invariant checks keep both modes.
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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance
from jqmc.swct import (
    _evaluate_swct_domega_debug,
    _evaluate_swct_omega_debug,
    evaluate_swct_domega,
    evaluate_swct_omega,
)
from jqmc.trexio_wrapper import read_trexio_file

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block


def _assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Check a homogeneous electron-coordinate derivative block at its central budget."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


class TestSWCT:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_debug_and_jax_SWCT_omega(self, trexio_file: str):
        """Test SWCT omega, compare debug and jax."""
        atol, rtol = get_tolerance("swct", "strict")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        num_ele_up = geminal_mo_data.num_electron_up
        num_ele_dn = geminal_mo_data.num_electron_dn
        r_cart_min, r_cart_max = -5.0, +5.0
        r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
        r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

        omega_up_debug = _evaluate_swct_omega_debug(structure_data=structure_data, r_carts=r_up_carts)
        omega_dn_debug = _evaluate_swct_omega_debug(structure_data=structure_data, r_carts=r_dn_carts)
        omega_up_jax = evaluate_swct_omega(structure_data=structure_data, r_carts=r_up_carts)
        omega_dn_jax = evaluate_swct_omega(structure_data=structure_data, r_carts=r_dn_carts)

        assert not np.any(np.isnan(np.asarray(omega_up_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(omega_up_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(omega_up_debug, omega_up_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(omega_dn_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(omega_dn_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(omega_dn_debug, omega_dn_jax, atol=atol, rtol=rtol)

        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    def test_swct_domega_production_vs_debug(self, trexio_file: str):
        """Guardrail for the ``evaluate_swct_domega`` pair.

        Asserts the production jacrev path matches the ``_debug`` finite-
        difference oracle at strict swct tolerance, as a guardrail of its own
        separate from ``test_debug_and_jax_SWCT_omega``.
        """
        atol, rtol = get_tolerance("swct", "strict")
        (
            structure_data,
            _,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        num_ele_up = geminal_mo_data.num_electron_up
        num_ele_dn = geminal_mo_data.num_electron_dn
        r_cart_min, r_cart_max = -5.0, +5.0
        r_up_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_up, 3) + r_cart_min
        r_dn_carts = (r_cart_max - r_cart_min) * np.random.rand(num_ele_dn, 3) + r_cart_min

        domega_up_debug = _evaluate_swct_domega_debug(structure_data=structure_data, r_carts=r_up_carts)
        domega_dn_debug = _evaluate_swct_domega_debug(structure_data=structure_data, r_carts=r_dn_carts)
        domega_up_jax = evaluate_swct_domega(structure_data=structure_data, r_carts=r_up_carts)
        domega_dn_jax = evaluate_swct_domega(structure_data=structure_data, r_carts=r_dn_carts)

        assert not np.any(np.isnan(np.asarray(domega_up_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(domega_up_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(domega_up_debug, domega_up_jax, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(domega_dn_debug))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(domega_dn_jax))), "NaN detected in second argument"
        np.testing.assert_allclose(domega_dn_debug, domega_dn_jax, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestSWCTDomegaClosedForm:
    """``evaluate_swct_domega``'s closed form vs a locally-built ``jacrev`` oracle.

    Production contains no ``jacrev`` (the ``(N_a, N_e, N_e, 3)`` jacobian it
    would materialise is 1/N_e sparse and costs ``0.343 * nw * N_a^2 * N_e^2 *
    24 B`` of temp, about 18 GB for a large periodic cell even with a single
    walker); the guard ``test_no_autodiff_in_production.py`` forbids it. This
    test keeps the mathematical identity pinned by rebuilding the jacobian
    oracle HERE, where autodiff is allowed. Machine-precision agreement expected
    (the omega jacobian is exactly diagonal in the electron indices); asserted
    at the zone-strict tolerance. Complements the finite-difference oracle in
    ``test_swct_domega_production_vs_debug``.
    """

    @pytest.mark.light
    def test_domega_matches_jacrev_oracle(self):
        """Domega matches jacrev oracle."""

        atol, rtol = get_tolerance("swct", "strict")

        def case_factory():
            from jqmc.structure import Structure_data

            box = 9.3
            rng = np.random.default_rng(3)
            n_atoms = 5
            structure_data = Structure_data.build(
                positions=rng.random((n_atoms, 3)) * box,
                pbc_flag=True,
                vec_a=(box, 0.0, 0.0),
                vec_b=(0.7, box, 0.0),
                vec_c=(0.2, 0.4, box),
                atomic_numbers=tuple([6] * n_atoms),
                element_symbols=tuple(["C"] * n_atoms),
                atomic_labels=tuple(["C"] * n_atoms),
            )
            r_carts = jax.numpy.asarray(np.random.default_rng(7).random((11, 3)) * box)

            return (structure_data, r_carts)

        def analytical(case):
            structure_data, r_carts = case
            return evaluate_swct_domega(structure_data, r_carts)

        if is_mixed_precision_enabled():
            mixed, full = mixed_vs_full(case_factory, analytical)
            blocks = mixed_fidelity_blocks(mixed, full)
            mixed_atol, mixed_rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, mixed_atol, mixed_rtol)
            mutated = out_of_budget_mixed_block(*blocks[0], atol=mixed_atol, rtol=mixed_rtol)
            with pytest.raises(AssertionError):
                _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], mixed_atol, mixed_rtol)
            jax.clear_caches()
            return

        structure_data, r_carts = case_factory()
        got = evaluate_swct_domega(structure_data, r_carts)
        oracle = jax.numpy.sum(jax.jacrev(evaluate_swct_omega, argnums=1)(structure_data, r_carts), axis=(1, 2))

        assert not np.any(np.isnan(np.asarray(got))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(oracle))), "NaN detected in second argument"
        np.testing.assert_allclose(np.asarray(got), np.asarray(oracle), atol=atol, rtol=rtol)
        jax.clear_caches()


class TestSWCTOmegaDot:
    """``evaluate_swct_omega_dot`` vs materialising ``omega`` and contracting.

    The full ``(N_a, N_e)`` SWCT weight matrix per walker feeds only one
    ``einsum("jk,kl->jl")``. The weights are a contraction operand, never an
    output (materialised, they take ~9.6 GiB per chunked force block at
    A = N_e / 3 = 288), so the fused form lets XLA drop the intermediate
    entirely. Agreement is to fp round-off, not
    bit-exact: the electron sum runs in a different order (fused device
    reduction vs BLAS). ``evaluate_swct_omega`` stays the reference.
    """

    @pytest.mark.light
    @pytest.mark.parametrize("pbc", [False, True], ids=["molecular", "periodic"])
    def test_omega_dot_matches_materialised_contraction(self, pbc):
        """Omega dot matches materialised contraction."""
        from jqmc.structure import Structure_data
        from jqmc.swct import evaluate_swct_omega_dot

        atol, rtol = get_tolerance("swct", "strict")
        box = 9.3
        rng = np.random.default_rng(13)
        n_atoms, n_e = 6, 14
        kwargs = (
            dict(pbc_flag=True, vec_a=(box, 0.0, 0.0), vec_b=(0.7, box, 0.0), vec_c=(0.2, 0.4, box))
            if pbc
            else dict(pbc_flag=False)
        )
        structure_data = Structure_data.build(
            positions=rng.random((n_atoms, 3)) * box,
            atomic_numbers=tuple([6] * n_atoms),
            element_symbols=tuple(["C"] * n_atoms),
            atomic_labels=tuple(["C"] * n_atoms),
            **kwargs,
        )
        r_carts = jax.numpy.asarray(rng.random((n_e, 3)) * box)
        field = jax.numpy.asarray(rng.standard_normal((n_e, 3)))

        got = np.asarray(evaluate_swct_omega_dot(structure_data, r_carts, field))
        omega = np.asarray(evaluate_swct_omega(structure_data, r_carts))  # (N_a, N_e)
        reference = np.einsum("ai,ic->ac", omega, np.asarray(field))

        assert got.shape == (n_atoms, 3)
        assert not np.any(np.isnan(got)), "NaN detected in first argument"
        assert not np.any(np.isnan(reference)), "NaN detected in second argument"
        np.testing.assert_allclose(got, reference, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.heavy
    def test_omega_dot_gemm_miscompile_shape(self):
        """Pin the (N_a=16, N_e=432, B=1024) shape that jax<=0.7.x XLA:GPU miscompiles.

        With the contraction written as einsum/dot_general, the fp64 GEMM
        consumer of the fused kappa/normaliser producer returns nondeterministic
        NaN and rel ~ 0.28 against the CPU backend at exactly this shape on the
        jax 0.7.x GPU lowering (N_a = 8 and 32 are clean, an isolated
        dot_general at the same shape is clean, and an optimization_barrier
        does not protect it). ``evaluate_swct_omega_dot`` therefore uses a
        broadcast-multiply-reduce contraction; this test fails if anyone
        rewrites it back to a GEMM form and runs an affected jax on GPU.
        """
        from jqmc.structure import Structure_data
        from jqmc.swct import evaluate_swct_omega_dot

        atol, rtol = get_tolerance("swct", "strict")
        box = 8.3
        rng = np.random.default_rng(5)
        n_atoms, n_e, n_b = 16, 432, 1024
        structure_data = Structure_data.build(
            positions=rng.random((n_atoms, 3)) * box,
            pbc_flag=True,
            vec_a=(box, 0.0, 0.0),
            vec_b=(0.0, box, 0.0),
            vec_c=(0.0, 0.0, box),
            atomic_numbers=tuple([18] * n_atoms),
            element_symbols=tuple(["Ar"] * n_atoms),
            atomic_labels=tuple(["Ar"] * n_atoms),
        )
        R = np.asarray(structure_data.positions)
        r_b = jax.numpy.asarray(R[rng.integers(0, n_atoms, (n_b, n_e))] + rng.normal(scale=0.6, size=(n_b, n_e, 3)))
        f_b = jax.numpy.asarray(rng.standard_normal((n_b, n_e, 3)))

        got = np.asarray(jax.vmap(evaluate_swct_omega_dot, in_axes=(None, 0, 0))(structure_data, r_b, f_b))
        omega = np.asarray(jax.vmap(evaluate_swct_omega, in_axes=(None, 0))(structure_data, r_b))  # (B, N_a, N_e)
        reference = np.einsum("wai,wic->wac", omega, np.asarray(f_b))

        assert not np.any(np.isnan(got)), "GEMM-lowering miscompilation reproduced (nondeterministic NaN)"
        np.testing.assert_allclose(got, reference, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    def test_batched_streaming_matches_plain_vmap(self, monkeypatch):
        """The streamed batch axis reproduces the unstreamed batch result.

        ``batch >= B`` is the plain ``vmap``; a budget of one byte forces the
        minimum batch. Both must agree with the per-element reference, which is
        what makes the budget a pure memory knob.
        """
        import jqmc.swct as swct_mod
        from jqmc.structure import Structure_data
        from jqmc.swct import evaluate_swct_omega_dot_batched

        atol, rtol = get_tolerance("swct", "strict")
        box = 9.3
        rng = np.random.default_rng(29)
        n_atoms, n_e, n_b = 6, 14, 37  # B not a multiple of any batch -> exercises the pad/trim
        structure_data = Structure_data.build(
            positions=rng.random((n_atoms, 3)) * box,
            pbc_flag=True,
            vec_a=(box, 0.0, 0.0),
            vec_b=(0.7, box, 0.0),
            vec_c=(0.2, 0.4, box),
            atomic_numbers=tuple([6] * n_atoms),
            element_symbols=tuple(["C"] * n_atoms),
            atomic_labels=tuple(["C"] * n_atoms),
        )
        r_b = jax.numpy.asarray(rng.random((n_b, n_e, 3)) * box)
        f_b = jax.numpy.asarray(rng.standard_normal((n_b, n_e, 3)))

        monkeypatch.setattr(swct_mod, "SWCT_OMEGA_BATCH_BUDGET_BYTES", 1 << 60)  # one batch
        unstreamed = np.asarray(evaluate_swct_omega_dot_batched(structure_data, r_b, f_b))
        monkeypatch.setattr(swct_mod, "SWCT_OMEGA_BATCH_BUDGET_BYTES", 1)  # minimum batch
        streamed = np.asarray(evaluate_swct_omega_dot_batched(structure_data, r_b, f_b))

        assert streamed.shape == (n_b, n_atoms, 3)
        assert not np.any(np.isnan(streamed)), "NaN detected in first argument"
        assert not np.any(np.isnan(unstreamed)), "NaN detected in second argument"
        np.testing.assert_allclose(streamed, unstreamed, atol=atol, rtol=rtol)
        jax.clear_caches()


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
