"""The molecular GTO ``build()`` factories store the supplied fp64 basis exactly (issue #156).

AO radial exponents and contraction coefficients are fp64 model storage in
both precision modes (basis-data storage exemption in ``jqmc/_precision.py``,
IMPL-HAM-HDF5-BASIS-FP64-STORAGE-001). ``AOs_cart_gto_data.build`` and
``AOs_sphe_gto_data.build`` documented this ("coerces exponents /
coefficients to float64 ndarrays") but rounded the inputs to the ``ao_eval``
dtype first, so in mixed precision they stored float32-rounded values, an
``inf`` for a 1e40 coefficient and a float32 subnormal for 1e-40, unlike the
TREXIO route (``_from_full_basis``) for the same inputs.
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

import jax
import numpy as np
import pytest

import jqmc_rs
from jqmc.atomic_orbital import (
    AOs_cart_gto_data,
    AOs_sphe_gto_data,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.structure import Structure_data
from tests._precision_fidelity import mixed_vs_full


jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# One value per distinct primitive (see ``_basis_kwargs``). None of the
# exponents or ordinary coefficients is representable in float32; the two
# extreme coefficients lie outside the float32 normal range (1e40 overflows to
# inf, 1e-40 becomes a float32 subnormal).
_EXPONENTS = (0.1234567891234567, 1.7654321987654321, 0.4321098765432109, 13.37133713371337, 0.987654321987654)
_COEFFICIENTS = (0.7654321098765432, 0.3141592653589793, 1.0e40, 0.2718281828459045, 1.0e-40)


def _structure():
    return Structure_data.build(
        positions=np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 1.1]]),
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H", "H"),
        pbc_flag=False,
    )


def _basis_kwargs(ao_kind, exponents, coefficients):
    """s shell on H0 (3 primitives), s shell on H1 (1 primitive), p shell on H1 (1 primitive).

    ``exponents`` / ``coefficients`` hold one value per distinct primitive; they
    are expanded to the per-AO-primitive layout the factories take (the p
    shell's primitive is repeated for its three components).
    """
    expand = (0, 1, 2, 3, 4, 4, 4)
    kwargs = dict(
        structure_data=_structure(),
        nucleus_index=(0, 1, 1, 1, 1),
        num_ao=5,
        num_ao_prim=len(expand),
        orbital_indices=(0, 0, 0, 1, 2, 3, 4),
        exponents=np.array([exponents[i] for i in expand], dtype=np.float64),
        coefficients=np.array([coefficients[i] for i in expand], dtype=np.float64),
        angular_momentums=(0, 0, 1, 1, 1),
    )
    if ao_kind == "cart":
        return dict(
            kwargs,
            polynominal_order_x=(0, 0, 1, 0, 0),
            polynominal_order_y=(0, 0, 0, 1, 0),
            polynominal_order_z=(0, 0, 0, 0, 1),
        )
    return dict(kwargs, magnetic_quantum_numbers=(0, 0, -1, 0, 1))


_CLASSES = {"cart": AOs_cart_gto_data, "sphe": AOs_sphe_gto_data}


def _build(ao_kind, exponents=_EXPONENTS, coefficients=_COEFFICIENTS):
    return _CLASSES[ao_kind].build(**_basis_kwargs(ao_kind, exponents, coefficients))


def _trexio_route(ao_kind, exponents=_EXPONENTS, coefficients=_COEFFICIENTS):
    """The construction used by ``trexio_wrapper.read_trexio_file`` (no ``ao_eval`` cast)."""
    return _CLASSES[ao_kind]._from_full_basis(**_basis_kwargs(ao_kind, exponents, coefficients))


def _electron_positions():
    """Five seeded points around the two nuclei (no screening or resampling)."""
    return np.random.default_rng(156).normal(scale=1.0, size=(5, 3))


@pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Checks stored basis parameters of the built leaf; no electron positions are evaluated."
)
def test_gto_build_stores_raw_fp64_basis_like_trexio_route(ao_kind):
    """``build()`` stores the supplied exponents/coefficients bit-exactly, as the TREXIO route does.

    Independent reference: the raw fp64 inputs themselves (five exponents and
    five coefficients, none representable in float32, including the
    coefficients 1e40 and 1e-40 outside the float32 normal range). The raw
    radial leaf arrays must be float64 numpy arrays, the per-primitive
    ``exponents`` / ``coefficients`` must equal the supplied arrays bit for
    bit, and every radial array must equal that of the leaf built from the
    same inputs by ``_from_full_basis`` (the TREXIO route). Checked in the
    active precision mode; in mixed, a factory that rounds to float32 would
    store ``inf`` for 1e40.
    """
    kwargs = _basis_kwargs(ao_kind, _EXPONENTS, _COEFFICIENTS)
    raw_exponents = np.array(kwargs["exponents"], copy=True)
    raw_coefficients = np.array(kwargs["coefficients"], copy=True)
    # Precondition: no input survives a float32 round trip (1e40 overflows to inf by design).
    with np.errstate(over="ignore"):
        assert np.all(raw_exponents.astype(np.float32).astype(np.float64) != raw_exponents)
        assert np.all(raw_coefficients.astype(np.float32).astype(np.float64) != raw_coefficients)

    built = _build(ao_kind)
    trexio = _trexio_route(ao_kind)

    for field in ("exponents", "coefficients"):
        array = getattr(built.radial, field)
        assert isinstance(array, np.ndarray), field
        assert array.dtype == np.dtype(np.float64), field
        np.testing.assert_array_equal(array, getattr(trexio.radial, field), err_msg=field, strict=True)
    np.testing.assert_array_equal(
        np.asarray(built.ao_prim_to_shell_prim), np.asarray(trexio.ao_prim_to_shell_prim), strict=True
    )
    np.testing.assert_array_equal(built.exponents, raw_exponents, err_msg="exponents", strict=True)
    np.testing.assert_array_equal(built.coefficients, raw_coefficients, err_msg="coefficients", strict=True)


# Ordinary (float32-range) coefficients for the evaluation test, so every
# compared quantity is finite; still none is representable in float32.
_ORDINARY_COEFFICIENTS = (0.7654321098765432, 0.3141592653589793, 0.5772156649015329, 0.2718281828459045, 1.4142135623730951)
_FP64_ZONE = ("grad_x", "grad_y", "grad_z", "laplacian")


def _evaluate(leaves, backend):
    """AO value, gradients and Laplacian of each leaf at five seeded points."""
    r_carts = _electron_positions()
    if backend == "xla":
        value, grad, lap = compute_AOs, compute_AOs_grad, compute_AOs_laplacian
    else:
        value, grad, lap = jqmc_rs.compute_AOs, jqmc_rs.compute_AOs_grad, jqmc_rs.compute_AOs_laplacian
    results = {}
    for route, aos in leaves.items():
        gx, gy, gz = grad(aos, r_carts)
        results[route] = {
            "value": value(aos, r_carts),
            "grad_x": gx,
            "grad_y": gy,
            "grad_z": gz,
            "laplacian": lap(aos, r_carts),
        }
    return results


@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 156, five normal(scale=1.0) points around the two H centres; no cond(G) screen or resampling.",
)
def test_gto_build_aos_equal_trexio_route_in_full_and_fp64_zone_in_mixed(ao_kind, backend):
    """A ``build()`` leaf evaluates like the TREXIO-route leaf, and its fp64 zone is mode-independent.

    The shared ``mixed_vs_full`` helper rebuilds the ``build()`` leaf and the
    ``_from_full_basis`` (TREXIO-route) leaf for the same inputs after each
    precision-policy switch and evaluates AO value, gradient and Laplacian
    (XLA or Rust) at five seeded points.

    * Full precision (correctness): the ``build()`` leaf's value, gradient
      and Laplacian are bit-identical to the TREXIO-route leaf's.
    * Mixed precision (mixed-versus-full fidelity of the ``build()`` route):
      the gradient and Laplacian are the float64 ``ao_grad_lap`` zone in both
      modes and the basis is fp64 storage, so the mixed ``build()`` leaf must
      reproduce the full-precision ``build()`` gradient and Laplacian bit for
      bit. The AO value (float32 ``ao_eval`` zone in mixed) is not compared in
      mixed, and no same-precision cross-route comparison is made in mixed.

    No tolerance is involved. A mixed factory that rounds the basis to float32
    shifts the mixed gradient/Laplacian by ~3e-8.
    """
    native = (
        dict(configure_native=jqmc_rs.configure_native_precision, native_policy_snapshot=jqmc_rs.native_precision_support)
        if backend == "rust"
        else {}
    )
    mixed, full = mixed_vs_full(
        lambda: {
            "build": _build(ao_kind, coefficients=_ORDINARY_COEFFICIENTS),
            "trexio": _trexio_route(ao_kind, coefficients=_ORDINARY_COEFFICIENTS),
        },
        lambda leaves: _evaluate(leaves, backend),
        **native,
    )
    for name, reference in full["trexio"].items():
        reference = np.asarray(reference)
        assert np.all(np.isfinite(reference)) and np.any(reference != 0.0), name
        np.testing.assert_array_equal(
            np.asarray(full["build"][name]), reference, err_msg=f"full {ao_kind} {backend} {name}", strict=True
        )
    for name in _FP64_ZONE:
        assert np.asarray(full["build"][name]).dtype == np.dtype(np.float64), name
        np.testing.assert_array_equal(
            np.asarray(mixed["build"][name]),
            np.asarray(full["build"][name]),
            err_msg=f"mixed vs full {ao_kind} {backend} {name}",
            strict=True,
        )
