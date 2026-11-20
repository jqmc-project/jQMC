"""CPU coordinate casts must not displace the exact nodes or mirror pairs of an odd spline."""

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

from jqmc._precision import get_complex_dtype_np, get_dtype_np, get_tolerance
from jqmc.atomic_orbital import compute_AOs
from tests._bspline_nodal_fixture import mirror_complex_spline_case, nodal_spline_case

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("cell_case", ["home", "translated"])
def test_real_spline_half_cell_nodal_planes(cell_case):
    """Odd mode-5 controls vanish on each half-cell plane in a triclinic cell.

    The symmetric cardinal kernel preserves oddness, so zero is an exact
    reference independent of either evaluator. The off-plane samples ensure
    that the values are nontrivial. Both batches retain caller fp64 coordinates.
    """
    assert jax.default_backend() == "cpu", "This geometry regression requires JAX_PLATFORMS=cpu"
    aos, nodes, off_nodes = nodal_spline_case(cell_case)
    atol, rtol = get_tolerance("ao_eval", "strict")
    values = np.asarray(compute_AOs(aos, jnp.asarray(nodes.reshape(-1, 3))))
    off_values = np.asarray(compute_AOs(aos, jnp.asarray(off_nodes.reshape(-1, 3))))
    assert values.shape == off_values.shape == (3, 96)
    assert values.dtype == off_values.dtype == np.dtype(get_dtype_np("ao_eval"))
    assert np.isfinite(values).all() and np.isfinite(off_values).all()
    node_values = np.concatenate([values[axis, axis * 32 : (axis + 1) * 32] for axis in range(3)])
    off_values = np.concatenate([off_values[axis, axis * 32 : (axis + 1) * 32] for axis in range(3)])
    assert np.min(np.abs(off_values)) > 100 * atol
    np.testing.assert_allclose(node_values, np.zeros_like(node_values), atol=atol, rtol=rtol, equal_nan=False)
    # The independent NumPy forward oracle must also preserve these known
    # zeros; comparing only production and debug could hide a shared fault.
    debug = np.asarray(aos._compute_AOs_debug(nodes.reshape(-1, 3)))
    debug_off = np.asarray(aos._compute_AOs_debug(off_nodes.reshape(-1, 3)))
    assert debug.shape == debug_off.shape == (3, 96)
    assert debug.dtype == debug_off.dtype == np.dtype(get_dtype_np("ao_eval"))
    assert np.isfinite(debug).all() and np.isfinite(debug_off).all()
    debug_nodes = np.concatenate([debug[axis, axis * 32 : (axis + 1) * 32] for axis in range(3)])
    debug_off = np.concatenate([debug_off[axis, axis * 32 : (axis + 1) * 32] for axis in range(3)])
    assert np.min(np.abs(debug_off)) > 100 * atol
    np.testing.assert_allclose(debug_nodes, np.zeros_like(debug_nodes), atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("cell_case", ["home", "translated"])
@pytest.mark.parametrize("leaf_kind", ["complex", "tiled"])
def test_complex_spline_mirror_pairs_cancel(leaf_kind, cell_case):
    """Odd complex controls give ``u(c + d) + u(c - d) = 0`` at mirror points.

    The periodic part is recovered by removing the Bloch phase in float64
    (``u = psi exp(-i k.r)``). The zero sum follows from the symmetric cardinal
    kernel, independent of either evaluator. The mirror centre lies at
    ``t = 28.5`` on a 30-point grid, so the sum is sensitive to the local offset
    ``t - floor(t)``. A local offset formed after rounding the fractional
    coordinate to fp32 carries an absolute error of about 2e-6 there, above the
    strict ``ao_eval`` absolute tolerance. The XLA evaluator and the
    independent NumPy oracle are both checked, for the single-k leaf and the
    tiled primitive-cell leaf, in the home cell and a lattice translation.
    """
    assert jax.default_backend() == "cpu", "This geometry regression requires JAX_PLATFORMS=cpu"
    aos, plus, minus, k_cart, axes = mirror_complex_spline_case(leaf_kind, cell_case)
    atol, rtol = get_tolerance("ao_eval", "strict")
    flat_plus, flat_minus = plus.reshape(-1, 3), minus.reshape(-1, 3)
    for label, evaluate in (
        ("xla", lambda r: np.asarray(compute_AOs(aos, jnp.asarray(r)))),
        ("numpy", lambda r: np.asarray(aos._compute_AOs_debug(r))),
    ):
        values_plus, values_minus = evaluate(flat_plus), evaluate(flat_minus)
        assert values_plus.dtype == values_minus.dtype == np.dtype(get_complex_dtype_np("ao_eval")), label
        assert np.isfinite(values_plus).all() and np.isfinite(values_minus).all(), label
        sums, magnitudes = [], []
        for band, axis in enumerate(axes):
            columns = slice(axis * 64, (axis + 1) * 64)
            u_plus = values_plus[band, columns] * np.exp(-1j * (flat_plus[columns] @ k_cart[band]))
            u_minus = values_minus[band, columns] * np.exp(-1j * (flat_minus[columns] @ k_cart[band]))
            sums.append(u_plus + u_minus)
            magnitudes.append(np.abs(u_plus))
        sums, magnitudes = np.concatenate(sums), np.concatenate(magnitudes)
        assert np.min(magnitudes) > 100 * atol, label
        np.testing.assert_allclose(sums, np.zeros_like(sums), atol=atol, rtol=rtol, equal_nan=False, err_msg=label)
