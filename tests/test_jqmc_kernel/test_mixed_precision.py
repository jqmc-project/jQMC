"""Mixed precision dtype propagation tests.

These tests verify that when ``--precision-mode=mixed`` is active, each
Precision Zone produces outputs in the expected dtype. They catch JAX
dtype-promotion bugs where fp64 data (io/optimization zone) leaks into
fp32 compute kernels and silently promotes the entire computation to fp64.

Every test:
  1. Configures ``mode="mixed"`` explicitly (independent of the CLI flag).
  2. Calls the target function with realistic inputs.
  3. Asserts the output dtype matches the zone's configured dtype.

In ``mode="full"`` (the default), all zones are fp64 and these tests are
trivially satisfied, so they are skipped to save time.

Run with::

    pytest tests/test_jqmc_kernel/test_mixed_precision.py -v --precision-mode=mixed
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
import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine.evidence import ObservationBinding, observe_non_numerical
from tests.test_jqmc_kernel._gdd_adapters import case_receipt, committed_input, seeded_single_electron_receipt

pytestmark = [
    pytest.mark.with_chdir,
    pytest.mark.skip_full_precision(reason="mixed-precision policy tests run only with --precision-mode=mixed"),
]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import (
    cast_to_zone,
    configure,
    get_complex_dtype_jnp,
    get_complex_dtype_np,
    get_dtype_jnp,
    get_tolerance,
)
from jqmc._setting import (
    ATOL_BIT_EXACT,
    RTOL_PYTHON_FP64_TIGHT,
)
from jqmc.atomic_orbital import (
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.coulomb_potential import (
    compute_bare_coulomb_potential,
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_part_all_pairs_jax_weights_grid_points,
)
from jqmc.derivatives import bridge, gfmc
from jqmc.determinant import (
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
    compute_grads_and_laplacian_ln_Det,
    compute_ln_det_geminal_all_elements,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_Jastrow_one_body,
    compute_Jastrow_part,
    compute_Jastrow_three_body,
    compute_Jastrow_two_body,
)
from jqmc.molecular_orbital import (
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    compute_kinetic_energy,
    evaluate_ln_wavefunction,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


@pytest.fixture(autouse=True)
def _configure_mixed():
    """Ensure mixed mode is active and JIT caches are cleared."""
    configure("mixed")
    jax.clear_caches()
    yield
    # Restore full mode after each test to avoid polluting other tests
    configure("full")
    jax.clear_caches()


def _load_trexio(filename: str) -> dict:
    """Helper that loads a TREXIO file and synthesizes random electron coordinates."""
    trexio_file = os.path.join(TREXIO_DIR, filename)
    structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_data = read_trexio_file(
        trexio_file=trexio_file, store_tuple=True
    )
    rng = np.random.default_rng(42)
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn
    r_up = jnp.array(rng.standard_normal((n_up, 3)) * 0.5, dtype=jnp.float64)
    r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 0.5, dtype=jnp.float64)
    return {
        "structure_data": structure_data,
        "aos_data": aos_data,
        "mos_data_up": mos_data_up,
        "mos_data_dn": mos_data_dn,
        "geminal_data": geminal_data,
        "coulomb_data": coulomb_data,
        "r_up": r_up,
        "r_dn": r_dn,
    }


@pytest.fixture
def h2_data():
    """Load H2 all-electron Cartesian basis test data."""
    return _load_trexio("H2_ae_ccpvdz_cart.h5")


@pytest.fixture
def h2_sphe_data():
    """Load H2 all-electron spherical basis test data (covers AO_sphe path)."""
    return _load_trexio("H2_ae_ccpvdz_sphe.h5")


@pytest.fixture
def h2_ecp_data():
    """Load H2 ECP test data (covers ECP local/non-local paths)."""
    return _load_trexio("H2_ecp_ccpvtz.h5")


def _assert_dtype(arr, expected, label):
    """Helper: assert array (or scalar) dtype matches expected."""
    actual = jnp.asarray(arr).dtype
    assert actual == expected, (
        f"{label} dtype is {actual}, expected {expected}. "
        "Check kernel-entry casts of (1) input r_carts, (2) all pytree float fields, "
        "(3) jnp.array/zeros/ones literals."
    )


def _assert_eval_shape_dtype(fn, expected, label, *args, **kwargs):
    """Helper: use jax.eval_shape (no actual execution) to assert output dtype.

    Useful for heavy kernels (ECP non-local) where executing in mixed mode
    is slow. Returns a ShapeDtypeStruct (or pytree thereof) and we walk the
    leaves to assert every float leaf has the expected dtype.
    """
    out = jax.eval_shape(fn, *args, **kwargs)
    leaves = jax.tree_util.tree_leaves(out)
    bad = [(i, leaf.dtype) for i, leaf in enumerate(leaves) if leaf.dtype.kind == "f" and leaf.dtype != expected]
    assert not bad, (
        f"{label} has float leaves with unexpected dtype: {bad}. Expected {expected}. "
        "Heavy kernels checked via jax.eval_shape (no execution)."
    )


def _hamiltonian(data) -> Hamiltonian_data:
    return Hamiltonian_data(
        structure_data=data["structure_data"],
        wavefunction_data=Wavefunction_data(geminal_data=data["geminal_data"]),
        coulomb_potential_data=data["coulomb_data"],
    )


def _assert_finite_pytree(value, label: str) -> None:
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(value) if np.asarray(leaf).dtype.kind in "fc"]
    assert leaves, f"{label}: analytical route returned no floating-point leaves"
    for index, leaf in enumerate(leaves):
        assert np.all(np.isfinite(leaf)), f"{label}: non-finite leaf {index}"


@pytest.mark.light
def test_cast_to_zone_preserves_number_kind_and_uses_zone_component_width():
    """The input selects real/complex; the zone alone selects component width."""
    assert get_complex_dtype_jnp("ao_eval") is jnp.complex64
    assert get_complex_dtype_np("ao_eval") is np.complex64
    assert get_complex_dtype_jnp("det_eval") is jnp.complex128
    assert get_complex_dtype_np("det_eval") is np.complex128

    assert cast_to_zone(jnp.asarray(1.0, dtype=jnp.float64), "ao_eval").dtype == jnp.float32
    assert cast_to_zone(jnp.asarray(1.0 + 2.0j, dtype=jnp.complex128), "ao_eval").dtype == jnp.complex64
    assert cast_to_zone(jnp.asarray(1.0, dtype=jnp.float64), "ao_eval", jnp.complex128).dtype == jnp.complex64
    assert cast_to_zone(jnp.asarray(1.0 + 2.0j, dtype=jnp.complex64), "det_eval").dtype == jnp.complex128


@pytest.mark.light
def test_mixed_jastrow_ratio_stays_float64():
    """LRDMC-amplified Jastrow ratio arithmetic is a mixed-mode fp64 zone."""
    assert get_dtype_jnp("jastrow_ratio") is jnp.float64


@pytest.mark.light
def test_configure_invalidates_jit_cache_when_precision_policy_changes():
    """A cached executable must not retain the previous precision mode."""

    @jax.jit
    def cast_in_ao_zone(x):
        return x.astype(get_dtype_jnp("ao_eval"))

    x = jnp.ones(1, dtype=jnp.float64)
    configure("full")
    assert cast_in_ao_zone(x).dtype == jnp.float64

    configure("mixed")
    assert cast_in_ao_zone(x).dtype == jnp.float32

    configure("full")
    assert cast_in_ao_zone(x).dtype == jnp.float64


@pytest.mark.light
def test_medium_tolerance_uses_strict_fp32_to_fp64_scale():
    """Medium must not introduce a broader fp32/fp64 tolerance ratio."""
    configure("full")
    strict_fp64 = get_tolerance("ao_eval", "strict")
    medium_fp64 = get_tolerance("ao_eval", "medium")

    configure("mixed")
    strict_fp32 = get_tolerance("ao_eval", "strict")
    medium_fp32 = get_tolerance("ao_eval", "medium")

    strict_scale = np.asarray(strict_fp32) / np.asarray(strict_fp64)
    medium_scale = np.asarray(medium_fp32) / np.asarray(medium_fp64)
    np.testing.assert_allclose(
        medium_scale,
        strict_scale,
        atol=ATOL_BIT_EXACT,
        rtol=RTOL_PYTHON_FP64_TIGHT,
    )
    assert medium_fp32 == (1.0e-5, 2.0e-3)


@pytest.mark.light
def test_configure_clears_cache_only_for_an_effective_policy_change(monkeypatch):
    """Idempotent and rejected configuration must leave caches and policy intact."""
    configure("mixed")
    clear_calls = []
    monkeypatch.setattr(jax, "clear_caches", lambda: clear_calls.append(None))

    configure("mixed")
    assert not clear_calls

    configure("full")
    assert len(clear_calls) == 1
    configure("full")
    assert len(clear_calls) == 1

    with pytest.raises(ValueError, match="Unknown precision mode"):
        configure("invalid")
    assert len(clear_calls) == 1
    assert get_dtype_jnp("ao_eval") is jnp.float64

    configure("mixed")
    assert len(clear_calls) == 2


@pytest.mark.light
def test_configure_requests_highest_precision_for_fp32_matmuls():
    """fp32 matrix products must not run at reduced (TF32-class) precision.

    With the JAX matmul precision unset, XLA lowers fp32 dots at DEFAULT
    precision, which Ampere GPUs execute at TF32-class
    accuracy, far outside the fp32 tolerances of the mixed zones.  The
    lowered StableHLO precision_config is backend-independent, so this
    checks on any device what the GPU will execute.  An explicit user choice
    of matmul precision is left alone.
    """
    previous = jax.config.jax_default_matmul_precision
    try:
        jax.config.update("jax_default_matmul_precision", None)
        configure("mixed")
        x = jnp.ones((4, 4), dtype=get_dtype_jnp("ao_eval"))
        assert x.dtype == jnp.float32
        assert "precision = [HIGHEST, HIGHEST]" in jax.jit(jnp.matmul).lower(x, x).as_text()

        jax.config.update("jax_default_matmul_precision", "default")
        configure("full")
        configure("mixed")
        assert jax.config.jax_default_matmul_precision == "default"
    finally:
        jax.config.update("jax_default_matmul_precision", previous)
        configure("mixed")


# This is the independently reviewed mixed-precision contract.  It deliberately
# contains no autograd oracle: mixed validates the production analytical graph,
# while analytical-vs-autograd correctness is an fp64-only contract.
MIXED_ANALYTICAL_SMOKE_CONTRACT = {
    "continuous": {
        "quantities": ("ln_Psi", "e_L"),
        "routes": ("c", "r+R", "eps"),
    },
    "gfmc": {
        "families": ("t", "n"),
        "routes": ("positions", "strain"),
    },
}

_MIXED_EXPECTED_LEAF_SHAPES = {
    ("continuous", "ln_Psi", "c"): ((1, 2, 3), (1, 0), (1, 0), (1, 0), (1, 10), (1, 10), (1, 1, 10)) * 2,
    ("continuous", "ln_Psi", "r+R"): ((1, 2, 3), (1, 1, 3), (1, 1, 3)),
    ("continuous", "ln_Psi", "eps"): ((1, 3, 3),),
    ("continuous", "e_L", "c"): ((1, 2, 3), (1, 0), (1, 0), (1, 0), (1, 10), (1, 10), (1, 1, 10)) * 2,
    ("continuous", "e_L", "r+R"): ((1, 1, 3), (1, 1, 3), (1, 2, 3)),
    ("continuous", "e_L", "eps"): ((1, 3, 3),),
    ("gfmc", "t", "positions"): ((2, 3), (1, 3), (1, 3)),
    ("gfmc", "t", "strain"): ((3, 3),),
    ("gfmc", "n", "positions"): ((2, 3), (1, 3), (1, 3)),
    ("gfmc", "n", "strain"): ((3, 3),),
}


def mixed_route_obligation_id(route_class: str, family_or_quantity: str, route: str) -> str:
    """The MIXED production-route obligation id; shared with the derivative registry."""
    return f"MIXED::{route_class}::{family_or_quantity}::{route}"


def _record_mixed_route_receipt(h2_data, output, *, route_class, family_or_quantity, route, pytest_node_id):
    obligation_id = mixed_route_obligation_id(route_class, family_or_quantity, route)
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(output) if np.asarray(leaf).dtype.kind in "fc"]
    expected_shapes = _MIXED_EXPECTED_LEAF_SHAPES[(route_class, family_or_quantity, route)]
    actual_shapes = tuple(leaf.shape for leaf in leaves)
    actual_dtypes = tuple(leaf.dtype for leaf in leaves)
    check = observe_non_numerical(
        ObservationBinding(
            binding_id=f"MIXED-WITNESS::{obligation_id}",
            policy_id="OBS-MIXED-ROUTE-SMOKE-001",
            precision_mode="mixed",
            subject_role="production-analytical-derivative-route",
            required_observations=("finite", "expected-shape", "expected-dtype", "route-executed"),
        ),
        {
            "finite": bool(leaves and all(np.all(np.isfinite(leaf)) for leaf in leaves)),
            "expected-shape": actual_shapes == expected_shapes,
            "expected-dtype": bool(
                actual_dtypes
                and all(dtype == np.dtype(np.float64) for dtype in actual_dtypes)
                and get_dtype_jnp("ao_eval") is jnp.float32
            ),
            "route-executed": True,
        },
    )
    electrons = seeded_single_electron_receipt(
        obligation_id,
        h2_data["r_up"],
        h2_data["r_dn"],
        generator_operation="tests.test_jqmc_kernel.test_mixed_precision._load_trexio",
        seed=42,
        generator_arguments=(("distribution", "standard-normal"), ("scale", 0.5)),
        generation_pattern="direct",
    )
    case_receipt(
        guarantee_id="NUM-DERIV-MIXED-001",
        obligation_id=obligation_id,
        pytest_node_id=pytest_node_id,
        input_binding=committed_input(
            obligation_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL",),
            builder_operation="tests.test_jqmc_kernel.test_mixed_precision._hamiltonian",
        ),
        electron_configuration=electrons,
        primary_check=check,
    )


@pytest.mark.light
@pytest.mark.parametrize("quantity", MIXED_ANALYTICAL_SMOKE_CONTRACT["continuous"]["quantities"])
@pytest.mark.parametrize("route", MIXED_ANALYTICAL_SMOKE_CONTRACT["continuous"]["routes"])
def test_mixed_production_continuous_analytical_smoke(h2_data, quantity, route):
    """Execute the same jitted analytical closures used by production, without an AD oracle."""
    ham = _hamiltonian(h2_data)
    r_up = h2_data["r_up"][None, ...]
    r_dn = h2_data["r_dn"][None, ...]
    wrt = ("r", "R") if route == "r+R" else route
    mask = {"opt_geminal": True} if route == "c" else None
    if quantity == "ln_Psi":
        fn = bridge.batched_grad_ln_Psi(wrt=wrt, mask=mask, analytic=True)
        output = fn(ham, r_up, r_dn)
    else:
        fn = bridge.batched_grad_e_L(wrt=wrt, mask=mask, analytic=True)
        output = fn(ham, r_up, r_dn, jnp.eye(3, dtype=jnp.float64)[None, ...])
    _assert_finite_pytree(output, f"{quantity}/{route}")
    _record_mixed_route_receipt(
        h2_data,
        output,
        route_class="continuous",
        family_or_quantity=quantity,
        route=route,
        pytest_node_id=("tests/test_jqmc_kernel/test_mixed_precision.py::test_mixed_production_continuous_analytical_smoke"),
    )


@pytest.mark.heavy
@pytest.mark.parametrize("family", MIXED_ANALYTICAL_SMOKE_CONTRACT["gfmc"]["families"])
@pytest.mark.parametrize("route", MIXED_ANALYTICAL_SMOKE_CONTRACT["gfmc"]["routes"])
def test_mixed_production_gfmc_analytical_smoke(h2_data, family, route):
    """Execute every public GFMC/LRDMC analytical wrapper in mixed precision."""
    ham = _hamiltonian(h2_data)
    suffix = "t" if family == "t" else "n"
    fn = getattr(gfmc, f"grad_e_L_{suffix}_wrt_{route}")
    output = fn(
        ham,
        h2_data["r_up"],
        h2_data["r_dn"],
        jnp.eye(3, dtype=jnp.float64),
        0.8,
        non_local_move=None,
    )
    _assert_finite_pytree(output, f"G{2 if family == 't' else 3}/{route}")
    _record_mixed_route_receipt(
        h2_data,
        output,
        route_class="gfmc",
        family_or_quantity=family,
        route=route,
        pytest_node_id=("tests/test_jqmc_kernel/test_mixed_precision.py::test_mixed_production_gfmc_analytical_smoke"),
    )


# ---------------------------------------------------------------------------
# A. AO zone (orb_eval -> float32 in mixed)
# ---------------------------------------------------------------------------


class TestAODtype:
    """Verify AO evaluation outputs are float32 in mixed mode."""

    @pytest.mark.light
    def test_compute_AOs_output_dtype(self, h2_data):
        """compute_AOs must return float32 (ao_eval zone)."""
        AOs = compute_AOs(h2_data["aos_data"], h2_data["r_up"])
        expected = get_dtype_jnp("ao_eval")
        assert AOs.dtype == expected, (
            f"compute_AOs output dtype is {AOs.dtype}, expected {expected}. "
            "Likely cause: fp64 data (R_carts, exponents, coefficients) not cast to ao_eval dtype inside kernel."
        )

    @pytest.mark.light
    def test_compute_AOs_grad_output_dtype(self, h2_data):
        """compute_AOs_grad must return ao_grad_lap zone dtype."""
        grad_x, grad_y, grad_z = compute_AOs_grad(h2_data["aos_data"], h2_data["r_up"])
        expected = get_dtype_jnp("ao_grad_lap")
        for name, arr in [("grad_x", grad_x), ("grad_y", grad_y), ("grad_z", grad_z)]:
            assert arr.dtype == expected, f"compute_AOs_grad {name} dtype is {arr.dtype}, expected {expected}."

    @pytest.mark.light
    def test_compute_AOs_laplacian_output_dtype(self, h2_data):
        """compute_AOs_laplacian must return ao_grad_lap zone dtype."""
        lap = compute_AOs_laplacian(h2_data["aos_data"], h2_data["r_up"])
        expected = get_dtype_jnp("ao_grad_lap")
        assert lap.dtype == expected, f"compute_AOs_laplacian dtype is {lap.dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# B. MO zone (orb_eval -> float32 in mixed)
# ---------------------------------------------------------------------------


class TestMODtype:
    """Verify MO evaluation outputs use the determinant-zone dtype.

    Note: AO evaluation itself runs in ``orb_eval`` precision (e.g. fp32 in mixed
    mode), but the (small) MO matmul is upcast to the ``determinant`` zone dtype
    (fp64 by default).  This avoids amplifying fp32 round-off through the
    32x32 determinant / kinetic / energy paths while keeping the heavy AO
    kernels in fp32 (see bug/fp32 diagnostics).
    """

    @pytest.mark.light
    def test_compute_MOs_output_dtype(self, h2_data):
        """compute_MOs must return mo_eval-zone dtype (fp64 in mixed)."""
        MOs = compute_MOs(h2_data["mos_data_up"], h2_data["r_up"])
        expected = get_dtype_jnp("mo_eval")
        assert MOs.dtype == expected, (
            f"compute_MOs output dtype is {MOs.dtype}, expected {expected}. "
            "compute_MOs should upcast its small matmul to the mo_eval zone "
            "to avoid fp32 amplification downstream."
        )


# ---------------------------------------------------------------------------
# C. Jastrow zone (jastrow -> float32 in mixed)
# ---------------------------------------------------------------------------


class TestJastrowDtype:
    """Verify Jastrow outputs are float32 in mixed mode."""

    @pytest.mark.light
    def test_jastrow_two_body_output_dtype(self, h2_data):
        """compute_Jastrow_two_body must return float32 (jastrow zone).

        Catches: jastrow_2b_param (fp64) not cast to jastrow dtype.
        """
        j2_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="pade")
        J2 = compute_Jastrow_two_body(j2_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("jastrow_eval")
        assert jnp.asarray(J2).dtype == expected, (
            f"compute_Jastrow_two_body dtype is {jnp.asarray(J2).dtype}, expected {expected}. "
            "Likely cause: jastrow_2b_param not cast to jastrow dtype."
        )

    @pytest.mark.light
    def test_jastrow_three_body_output_dtype(self, h2_data):
        """compute_Jastrow_three_body must return float32 (jastrow zone).

        Catches: j_matrix (fp64) not cast to jastrow dtype.
        """
        j3_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=h2_data["aos_data"], random_init=True, random_scale=1e-3
        )
        J3 = compute_Jastrow_three_body(j3_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("jastrow_eval")
        assert jnp.asarray(J3).dtype == expected, (
            f"compute_Jastrow_three_body dtype is {jnp.asarray(J3).dtype}, expected {expected}. "
            "Likely cause: j_matrix not cast to jastrow dtype."
        )

    @pytest.mark.light
    def test_jastrow_part_output_dtype(self, h2_data):
        """compute_Jastrow_part (J1+J2+J3) must return float32 (jastrow zone)."""
        j2_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        j3_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=h2_data["aos_data"], random_init=True, random_scale=1e-3
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=j2_data,
            jastrow_three_body_data=j3_data,
        )
        J = compute_Jastrow_part(jastrow_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("jastrow_eval")
        assert jnp.asarray(J).dtype == expected, f"compute_Jastrow_part dtype is {jnp.asarray(J).dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# D. Geminal zone (geminal -> float32 in mixed)
# ---------------------------------------------------------------------------


class TestGeminalDtype:
    """Verify geminal matrix is float32 in mixed mode."""

    @pytest.mark.light
    def test_geminal_matrix_output_dtype(self, h2_data):
        """compute_geminal_all_elements must return float32 (geminal zone).

        Catches: lambda_matrix or AO data not cast to geminal dtype.
        """
        G = compute_geminal_all_elements(h2_data["geminal_data"], h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("det_eval")
        assert G.dtype == expected, f"compute_geminal_all_elements dtype is {G.dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# E. Determinant zone (determinant -> float64 in mixed)
# ---------------------------------------------------------------------------


class TestDeterminantDtype:
    """Verify determinant outputs stay float64 in mixed mode."""

    @pytest.mark.light
    def test_ln_det_output_dtype(self, h2_data):
        """compute_ln_det_geminal_all_elements must return float64 (determinant zone).

        Even though geminal matrix is float32, the log-det must be computed in float64.
        """
        ln_det = compute_ln_det_geminal_all_elements(h2_data["geminal_data"], h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("det_eval")
        assert jnp.asarray(ln_det).dtype == expected, (
            f"compute_ln_det dtype is {jnp.asarray(ln_det).dtype}, expected {expected}."
        )

    @pytest.mark.light
    def test_compute_grads_and_laplacian_ln_Det_output_dtype(self, h2_data):
        """compute_grads_and_laplacian_ln_Det outputs must use det_grad_lap zone dtype."""
        grad_up, grad_dn, lap_up, lap_dn = compute_grads_and_laplacian_ln_Det(
            h2_data["geminal_data"], h2_data["r_up"], h2_data["r_dn"]
        )
        expected = get_dtype_jnp("det_grad_lap")
        _assert_dtype(grad_up, expected, "compute_grads_and_laplacian_ln_Det grad_up")
        _assert_dtype(grad_dn, expected, "compute_grads_and_laplacian_ln_Det grad_dn")
        _assert_dtype(lap_up, expected, "compute_grads_and_laplacian_ln_Det lap_up")
        _assert_dtype(lap_dn, expected, "compute_grads_and_laplacian_ln_Det lap_dn")


# ---------------------------------------------------------------------------
# F. Coulomb zone (coulomb -> float32 in mixed)
# ---------------------------------------------------------------------------


class TestCoulombDtype:
    """Verify Coulomb outputs are float32 in mixed mode."""

    @pytest.mark.light
    def test_bare_coulomb_el_el_output_dtype(self, h2_data):
        """Electron-electron Coulomb must return float32 (coulomb zone)."""
        V = compute_bare_coulomb_potential_el_el(h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("coulomb")
        assert jnp.asarray(V).dtype == expected, (
            f"compute_bare_coulomb_potential_el_el dtype is {jnp.asarray(V).dtype}, expected {expected}."
        )

    @pytest.mark.light
    def test_bare_coulomb_el_ion_output_dtype(self, h2_data):
        """Electron-ion Coulomb must return float32 (coulomb zone).

        Catches: R_charges (fp64) or structure positions (fp64) not cast.
        """
        V_up, V_dn = compute_bare_coulomb_potential_el_ion_element_wise(
            h2_data["coulomb_data"], h2_data["r_up"], h2_data["r_dn"]
        )
        expected = get_dtype_jnp("coulomb")
        assert V_up.dtype == expected, (
            f"el_ion V_up dtype is {V_up.dtype}, expected {expected}. "
            "Likely cause: R_charges or R_carts not cast to coulomb dtype."
        )

    @pytest.mark.light
    def test_bare_coulomb_total_output_dtype(self, h2_data):
        """Total bare Coulomb must return float32 (coulomb zone)."""
        V = compute_bare_coulomb_potential(h2_data["coulomb_data"], h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("coulomb")
        assert jnp.asarray(V).dtype == expected, (
            f"compute_bare_coulomb_potential dtype is {jnp.asarray(V).dtype}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# G. Kinetic zone (kinetic -> float64 in mixed)
# ---------------------------------------------------------------------------


class TestKineticDtype:
    """Verify kinetic energy stays float64 in mixed mode."""

    @pytest.mark.light
    def test_kinetic_energy_output_dtype(self, h2_data):
        """compute_kinetic_energy must return float64 (wf_kinetic zone)."""
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        T = compute_kinetic_energy(wf_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("wf_kinetic")
        assert jnp.asarray(T).dtype == expected, f"compute_kinetic_energy dtype is {jnp.asarray(T).dtype}, expected {expected}."


# ---------------------------------------------------------------------------
# H. Wavefunction zone boundary
# ---------------------------------------------------------------------------


class TestWavefunctionDtype:
    """Verify wavefunction evaluation zone boundaries."""

    @pytest.mark.light
    def test_ln_wavefunction_output_dtype(self, h2_data):
        """evaluate_ln_wavefunction must return float64 (determinant zone).

        The output combines Jastrow (float32) and log-det (float64),
        cast to determinant zone dtype.
        """
        j2_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=j2_data,
            jastrow_three_body_data=None,
        )
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"], jastrow_data=jastrow_data)
        ln_psi = evaluate_ln_wavefunction(wf_data, h2_data["r_up"], h2_data["r_dn"])
        expected = get_dtype_jnp("wf_eval")
        assert jnp.asarray(ln_psi).dtype == expected, (
            f"evaluate_ln_wavefunction dtype is {jnp.asarray(ln_psi).dtype}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# Extended coverage (a): additional boundary kernels
# ---------------------------------------------------------------------------


class TestAOSpheDtype:
    """Verify AO spherical basis path also returns float32 (orb_eval)."""

    @pytest.mark.light
    def test_compute_AOs_sphe_output_dtype(self, h2_sphe_data):
        """Compute AOs sphe output dtype."""
        AOs = compute_AOs(h2_sphe_data["aos_data"], h2_sphe_data["r_up"])
        _assert_dtype(AOs, get_dtype_jnp("ao_eval"), "compute_AOs (sphe)")

    @pytest.mark.light
    def test_compute_AOs_sphe_grad_output_dtype(self, h2_sphe_data):
        """Compute AOs sphe grad output dtype."""
        gx, gy, gz = compute_AOs_grad(h2_sphe_data["aos_data"], h2_sphe_data["r_up"])
        for name, arr in [("grad_x", gx), ("grad_y", gy), ("grad_z", gz)]:
            _assert_dtype(arr, get_dtype_jnp("ao_grad_lap"), f"compute_AOs_grad sphe {name}")

    @pytest.mark.light
    def test_compute_AOs_sphe_laplacian_output_dtype(self, h2_sphe_data):
        """Compute AOs sphe laplacian output dtype."""
        lap = compute_AOs_laplacian(h2_sphe_data["aos_data"], h2_sphe_data["r_up"])
        _assert_dtype(lap, get_dtype_jnp("ao_grad_lap"), "compute_AOs_laplacian (sphe)")


class TestMOExtendedDtype:
    """MO derivative kernels (kinetic zone)."""

    @pytest.mark.light
    def test_compute_MOs_grad_output_dtype(self, h2_data):
        """Compute MOs grad output dtype."""
        gx, gy, gz = compute_MOs_grad(h2_data["mos_data_up"], h2_data["r_up"])
        expected = get_dtype_jnp("mo_grad")
        for name, arr in [("grad_x", gx), ("grad_y", gy), ("grad_z", gz)]:
            _assert_dtype(arr, expected, f"compute_MOs_grad {name}")

    @pytest.mark.light
    def test_compute_MOs_laplacian_output_dtype(self, h2_data):
        """Compute MOs laplacian output dtype."""
        lap = compute_MOs_laplacian(h2_data["mos_data_up"], h2_data["r_up"])
        _assert_dtype(lap, get_dtype_jnp("mo_lap"), "compute_MOs_laplacian")


class TestJastrowOneBodyDtype:
    """Verify J1 forward standalone output (jastrow zone)."""

    @pytest.mark.light
    def test_compute_Jastrow_one_body_output_dtype(self, h2_data):
        """Compute Jastrow one body output dtype."""
        core_electrons = tuple([0.0] * len(h2_data["structure_data"].positions))
        j1_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=h2_data["structure_data"],
            core_electrons=core_electrons,
            jastrow_1b_type="pade",
        )
        J1 = compute_Jastrow_one_body(j1_data, h2_data["r_up"], h2_data["r_dn"])
        _assert_dtype(J1, get_dtype_jnp("jastrow_eval"), "compute_Jastrow_one_body")


class TestGeminalFastUpdateDtype:
    """Verify Geminal row/column kernels used in MCMC fast updates (geminal zone)."""

    @pytest.fixture
    def water_data(self):
        """Water ECP data with multiple electrons (needed for row/column tests)."""
        return _load_trexio("water_ccecp_ccpvqz.h5")

    @pytest.mark.light
    def test_geminal_up_one_row_output_dtype(self, water_data):
        # Use [0:1] to get shape (1, 3) -- compute_orb_api requires (N, 3), not (3,)
        """Geminal up one row output dtype."""
        row = compute_geminal_up_one_row_elements(water_data["geminal_data"], water_data["r_up"][0:1], water_data["r_dn"])
        _assert_dtype(row, get_dtype_jnp("det_ratio"), "compute_geminal_up_one_row_elements")

    @pytest.mark.light
    def test_geminal_dn_one_column_output_dtype(self, water_data):
        # Use [0:1] to get shape (1, 3) -- compute_orb_api requires (N, 3), not (3,)
        """Geminal dn one column output dtype."""
        col = compute_geminal_dn_one_column_elements(water_data["geminal_data"], water_data["r_up"], water_data["r_dn"][0:1])
        _assert_dtype(col, get_dtype_jnp("det_ratio"), "compute_geminal_dn_one_column_elements")


class TestECPDtype:
    """Verify ECP local + non-local paths (coulomb zone).

    Local: executed directly. Non-local: heavy, checked via jax.eval_shape (no run).
    """

    @pytest.mark.light
    def test_compute_ecp_local_parts_output_dtype(self, h2_ecp_data):
        """Compute ECP local parts output dtype."""
        V_loc = compute_ecp_local_parts_all_pairs(h2_ecp_data["coulomb_data"], h2_ecp_data["r_up"], h2_ecp_data["r_dn"])
        _assert_dtype(V_loc, get_dtype_jnp("coulomb"), "compute_ecp_local_parts_all_pairs")

    @pytest.mark.light
    def test_compute_ecp_non_local_eval_shape_dtype(self, h2_ecp_data):
        """Heavy ECP non-local kernel: verify dtype via jax.eval_shape (no execution)."""
        wf_data = Wavefunction_data(geminal_data=h2_ecp_data["geminal_data"])
        # Minimal Lebedev-like quadrature placeholder (6-point). Values are not
        # used by eval_shape; only shapes/dtypes matter for static dtype tracing.
        weights = jnp.ones((6,), dtype=jnp.float64) / 6.0
        grid_points = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=jnp.float64,
        )
        _assert_eval_shape_dtype(
            compute_ecp_non_local_part_all_pairs_jax_weights_grid_points,
            get_dtype_jnp("coulomb"),
            "compute_ecp_non_local_part_all_pairs_jax_weights_grid_points",
            h2_ecp_data["coulomb_data"],
            wf_data,
            h2_ecp_data["r_up"],
            h2_ecp_data["r_dn"],
            weights,
            grid_points,
        )


class TestLocalEnergyDtype:
    """Final boundary: compute_local_energy aggregates kinetic + coulomb (kinetic zone)."""

    @pytest.mark.light
    def test_compute_local_energy_output_dtype(self, h2_data):
        """Compute local energy output dtype."""
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        ham = Hamiltonian_data(
            structure_data=h2_data["structure_data"],
            wavefunction_data=wf_data,
            coulomb_potential_data=h2_data["coulomb_data"],
        )
        RT = jnp.eye(3, dtype=jnp.float64)
        e_L = compute_local_energy(ham, h2_data["r_up"], h2_data["r_dn"], RT)
        _assert_dtype(e_L, get_dtype_jnp("local_energy"), "compute_local_energy")


class TestKineticEvalShape:
    """jax.eval_shape coverage for kinetic energy fast/discretized variants.

    These exercise different code paths (Sherman-Morrison fast update,
    discretized DMC) without actually executing the heavy autodiff laplacian.
    """

    @pytest.mark.light
    def test_kinetic_energy_eval_shape_dtype(self, h2_data):
        """Kinetic energy eval shape dtype."""
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        _assert_eval_shape_dtype(
            compute_kinetic_energy,
            get_dtype_jnp("wf_kinetic"),
            "compute_kinetic_energy (eval_shape)",
            wf_data,
            h2_data["r_up"],
            h2_data["r_dn"],
        )

    @pytest.mark.light
    def test_evaluate_ln_wavefunction_eval_shape_dtype(self, h2_data):
        """Evaluate ln wavefunction eval shape dtype."""
        wf_data = Wavefunction_data(geminal_data=h2_data["geminal_data"])
        _assert_eval_shape_dtype(
            evaluate_ln_wavefunction,
            get_dtype_jnp("wf_eval"),
            "evaluate_ln_wavefunction (eval_shape)",
            wf_data,
            h2_data["r_up"],
            h2_data["r_dn"],
        )
