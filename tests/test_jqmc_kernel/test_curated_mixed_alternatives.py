"""Same-input, same-route AD alternatives for the eight unmatched curated recipes.

The seven recipes already present in the generated matrix stay there.  These
eight keep the curated recipes' seeds, two walkers, masks, and actual bridge calls.
The MPC label is only a recipe identifier: that recipe
does not enable MPC in the builder, and no MPC accuracy is claimed here.
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

import pytest
import jax.numpy as jnp
import numpy as np

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import bridge

from tests.test_jqmc_kernel._grad_fidelity_harness import (
    _NW,
    _QE_STEM,
    _assert_axis_allclose,
    _batched_geminal_inv,
    _build_ham,
    _clear_bridge_caches,  # noqa: F401
    _draw_batch,
    _flags_for_axis,
    _gap_inventory_report,  # noqa: F401
    requires_qe,
)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_IDS = [
    "N_cart_J",
    "H2_ecp_cart_J",
    "H2_ecp_cart_noJ",
    "Li_pbc_twist_J",
    "Li_pbc_twist_J3cplx",
    "H2_SD_J3MO",
    "Li_AGP_J3MO",
    "Li_AGP_MPC_J3MO",
]
# Explicit finite recipes make the ordinary receipt selector statically
# resolvable.  These are copied verbatim from the eight original rows.
_RECIPES = [
    ("N_cart_J", "N_ae_ccpvdz_cart.h5", True, None),
    ("H2_ecp_cart_J", "H2_ecp_ccpvdz_cart.h5", True, None),
    ("H2_ecp_cart_noJ", "H2_ecp_ccpvdz_cart.h5", False, None),
    ("Li_pbc_twist_J", "lithium_k025.h5", True, None),
    ("Li_pbc_twist_J3cplx", "lithium_k025.h5", True, {"j3": "complex"}),
    ("H2_SD_J3MO", "H2_ae_ccpvdz_cart.h5", True, {"j3": "mo"}),
    pytest.param("Li_AGP_J3MO", _QE_STEM, True, {"dense_agp": True, "j3": "mo"}, marks=requires_qe),
    pytest.param("Li_AGP_MPC_J3MO", _QE_STEM, True, {"dense_agp": True, "mpc": True, "j3": "mo"}, marks=requires_qe),
]


def _compare_curated_routes(fixture_id, stem, with_jastrow, spec, quantity, axis):
    """Original batched bridge calls, plus one unconditional receipt comparison site.

    The shared comparator supplies exactly its compared operands and retains
    its parameter-block selection, finite, and nonvacuity checks.  Rechecking
    those operands at the identical central budget makes one required site
    usable for both branches without optional comparison obligations.
    """
    ham = _build_ham(stem, with_jastrow, spec)
    up, down = _draw_batch(ham)
    inverse = _batched_geminal_inv(ham, up, down)
    mask = _flags_for_axis("c") if axis == "c" else None
    if quantity == "ln_Psi":

        def evaluate(analytic):
            return bridge.batched_grad_ln_Psi(wrt=axis, mask=mask, with_geminal_inv=True, analytic=analytic)(
                ham, up, down, inverse
            )
    else:
        rotation = jnp.broadcast_to(jnp.eye(3), (_NW, 3, 3))

        def evaluate(analytic):
            return bridge.batched_grad_e_L(wrt=axis, mask=mask, with_geminal_inv=True, analytic=analytic)(
                ham, up, down, rotation, inverse
            )

    pairs = _assert_axis_allclose(ham, axis, evaluate(True), evaluate(False), cell_id=fixture_id, quantity=quantity)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), level)
    for actual, reference in pairs:
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("fixture_id,stem,with_jastrow,spec", _RECIPES, ids=_IDS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_curated_unmatched_parameters_match_same_mode_autodiff(fixture_id, stem, with_jastrow, spec, quantity):
    """The 16 c-axis cases retain the original parameter blocks and nonvacuity checks."""
    _compare_curated_routes(fixture_id, stem, with_jastrow, spec, quantity, "c")


@pytest.mark.parametrize("fixture_id,stem,with_jastrow,spec", _RECIPES, ids=_IDS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("axis", ["r", "R", "eps"])
def test_curated_unmatched_physical_axes_match_same_mode_autodiff(fixture_id, stem, with_jastrow, spec, quantity, axis):
    """The other 48 cases: disjoint from c, with unchanged recipe/quantity/axis coverage."""
    _compare_curated_routes(fixture_id, stem, with_jastrow, spec, quantity, axis)
