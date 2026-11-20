"""pGTO/PW packaging against elementary image and Fourier sums."""

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

from jqmc import qe_to_jqmc as conversion
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.molecular_orbital import compute_MOs
from jqmc.pw_pgto_projection import CartPrimitive, ProjectionResult
from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _projection(skew, twisted):
    cell = QECell(
        np.array([[5.0, 0.0, 0.0], [0.8 if skew else 0.0, 6.0, 0.0], [-0.4 if skew else 0.0, 0.6 if skew else 0.0, 7.0]])
    )
    center = np.array([0.3, 0.7, 0.9])
    k = np.array([0.25, -0.125, 0.375]) if twisted else np.zeros(3)
    millers = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 1], [0, -1, -1]])
    pw = np.array(
        [
            [0.3, 0.12 + 0.07j, 0.12 - 0.07j, -0.04 + 0.03j, -0.04 - 0.03j],
            [-0.2, 0.09 - 0.02j, 0.09 + 0.02j, 0.05 + 0.08j, 0.05 - 0.08j],
        ]
    )
    coeff = np.array([[0.9, -0.4], [0.25, 0.8]], complex)
    if twisted:
        pw *= np.array([[1 + 0.2j], [0.8 - 0.3j]])
        coeff += np.array([[0.2j, 0.1j], [-0.15j, 0.3j]])
    prims = (
        CartPrimitive(center=tuple(center), exponent=0.7, ang=(0, 0, 0)),
        CartPrimitive(center=tuple(center), exponent=1.1, ang=(1, 0, 0)),
    )
    ref = QEReference(
        cell=cell,
        atom_labels=("Be",),
        atom_positions=center[None, :],
        ecutwfc=5.0,
        fft_grid=(8, 8, 8),
        bands=QEBands(np.array([-0.4, -0.2]), np.ones(2), 4.0, None),
        wfc=QEWavefunction(millers, pw, not twisted, k @ cell.b),
        atom_elements=("Be",),
    )
    projection = ProjectionResult(np.zeros(2), np.array([0, 1]), np.ones(2), coeff, prims, 2, 5, 0, 1.0, 1.0, 1.0, millers, pw)
    return ref, projection, k


def _elementary_orbitals(ref, projection, k, points):
    """Raw Gaussian polynomials and complex exponentials, without AO/projection helpers."""
    real_dtype = get_dtype_np("ao_eval")
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    images = np.array(list(itertools.product(range(-4, 5), repeat=3)))
    phase = np.exp((2j * np.pi * (images @ k)).astype(complex_dtype))
    basis = []
    for primitive in projection.primitives:
        displacement = (points[:, None, :] - np.array(primitive.center) - images @ ref.cell.a).astype(real_dtype)
        values = np.prod(displacement ** np.array(primitive.ang), axis=-1) * np.exp(
            -real_dtype(primitive.exponent) * np.sum(displacement**2, axis=-1)
        )
        basis.append(values @ phase)
    pgto = np.array(basis).T @ projection.coefficients.T
    reciprocal = 2 * np.pi * np.linalg.inv(ref.cell.a).T
    pw_phase = (points @ ((projection.pw_millers + k) @ reciprocal).T).astype(real_dtype)
    pw = np.exp((1j * pw_phase).astype(complex_dtype)) @ projection.pw_coefficients.T
    return (pgto + pw).T


def _compare_conversion(skew, twisted):
    ref, projection, k = _projection(skew, twisted)
    _, _, geminal = conversion.build_wavefunction_from_projection(ref, projection, kind="pgto+pw")
    fractions = np.array([[0.12, 0.21, 0.32], [0.4, 0.3, 0.2], [0.73, 0.26, 0.48], [-0.1, 0.04, 0.14], [1.12, 0.21, 0.32]])
    r_cart = fractions @ ref.cell.a
    reference = _elementary_orbitals(ref, projection, k, r_cart)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval"), "strict")
    for spin, orbitals in (("up", geminal.orb_data_up_spin), ("down", geminal.orb_data_dn_spin)):
        actual = np.asarray(compute_MOs(orbitals, r_cart))
        expected = reference if spin == "up" else reference.conj()
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=spin)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("twisted", [False, True])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_converted_pgto_pw_orbitals_match_elementary_sums(skew, twisted):
    """Public pGTO+PW conversion preserves both projected blocks and both spin orbitals.

    Synthetic Be projection data contain two occupied SD/MO rows (2 up/2 down),
    one unnormalized s Gaussian of exponent 0.7, one x Gaussian of exponent
    1.1, and five nonzero PW columns. No TREXIO or external QE file is read.
    Test the orthorhombic and triclinic cells, Gamma and fractional twist
    (1/4,-1/8,3/8), on five fixed fractional points including an image pair
    and a negative coordinate. No random positions or cond(G) filtering.
    Independent image enumeration [-4,4]^3 and elementary Fourier sums use
    active AO precision, then the input MO coefficients. Compare every band
    and point in both spin channels using same-mode central strict tolerances.
    This verifies faithful conversion of the supplied projection, not that
    a pGTO tail is representable by the original finite QE plane-wave set.
    """
    _compare_conversion(skew, twisted)


@pytest.mark.parametrize("fault", ["pw-scale", "pgto-normalization", "pw-column-order", "bloch-sign"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_pgto_pw_reference_rejects_actual_conversion_faults(monkeypatch, fault):
    """Corrupting either actual conversion block, its order or twist must fail the independent sum."""
    _compare_conversion(True, True)
    name = "_build_pgto_leaf_and_mo" if fault == "pgto-normalization" else "_build_pw_leaf_and_mo"
    original = getattr(conversion, name)

    def corrupted(*args, **kwargs):
        leaf, coefficients = original(*args, **kwargs)
        if fault in ("pw-scale", "pgto-normalization"):
            coefficients = 2 * coefficients
        elif fault == "pw-column-order":
            coefficients = coefficients[:, ::-1]
        else:
            leaf = leaf.with_negated_k()
        return leaf, coefficients

    monkeypatch.setattr(conversion, name, corrupted)
    if fault == "bloch-sign":
        # A plane-wave child at -k beside a +k Gaussian child does not build:
        # one spin channel must carry one twist.
        with pytest.raises(ValueError, match="carry different twists"):
            _compare_conversion(True, True)
        return
    with pytest.raises(AssertionError):
        _compare_conversion(True, True)
