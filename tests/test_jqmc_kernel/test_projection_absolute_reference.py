"""Absolute projection residuals from independent Fourier integrals and SVD."""

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

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_hermite

from jqmc._setting import (
    ATOL_PW_PROJECTION_FP64_LOOSE,
    PW_PGTO_SVD_RCOND,
    RTOL_PW_PROJECTION_FP64_LOOSE,
)
from jqmc.pw_pgto_projection import project_pw_to_pgto
from jqmc.qe_wrapper import load_qe_reference
from jqmc.upf_wrapper import UPFExponentInit
from tests._test_data_cache import qe_example_files_root

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _fourier_columns(reference, shells, alpha_min):
    # Hermite generating law for the unnormalized Cartesian Gaussian integral.
    # No production primitive enumeration, Fourier helper, overlap, or projection.
    lattice = np.asarray(reference.cell.a)
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    q = np.asarray(reference.wfc.millers) @ reciprocal + np.asarray(reference.wfc.k_point)
    volume = abs(np.linalg.det(lattice))
    columns = []
    for center, atom_shells in zip(reference.atom_positions, shells, strict=True):
        for shell in atom_shells:
            powers = sorted(
                (v for v in product(range(shell.angular_l + 1), repeat=3) if sum(v) == shell.angular_l), reverse=True
            )
            for powers_xyz in powers:
                for alpha in shell.exponents:
                    if alpha_min is not None and alpha < alpha_min:
                        continue
                    integral = np.exp(-1j * (q @ center)) / np.sqrt(volume)
                    for axis, order in enumerate(powers_xyz):
                        scaled_q = q[:, axis] / (2 * np.sqrt(alpha))
                        integral *= (
                            (-1j / (2 * np.sqrt(alpha))) ** order
                            * eval_hermite(order, scaled_q)
                            * np.sqrt(np.pi / alpha)
                            * np.exp(-(scaled_q**2))
                        )
                    columns.append(integral)
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("alpha_min", [None, 1.0])
def test_committed_li_pgto_residual_independent_svd(alpha_min):
    """Li QE pGTO per-band, weighted-mean and maximum residuals equal independent SVD values.

    Both modes exercise the converter's declared float64 IO computation, not
    mixed-versus-full values. The UPF exponent initializer is shared input;
    Cartesian enumeration, Gaussian Fourier integrals and the rectangular
    orthogonal projector are independently expressed. The existing float64
    projection comparison pair is used. No production values are
    saved as reference constants and no random sampling error enters the verdict.
    """
    source = qe_example_files_root() / "Li_gamma/out/li.save"
    reference = load_qe_reference(str(source))
    strategy = UPFExponentInit({"li": str(source / "Li.ccECP.upf")}, g_target=float(np.sqrt(2 * reference.ecutwfc)))
    shells = [strategy.shells_for_element(label) for label in reference.atom_labels]
    actual = project_pw_to_pgto(reference, shells, alpha_min=alpha_min)
    assert np.asarray(actual.residuals).dtype == np.dtype("float64")
    matrix = _fourier_columns(reference, shells, alpha_min)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    retained = singular**2 > PW_PGTO_SVD_RCOND * singular[0] ** 2
    occupations = np.asarray(reference.bands.occupations)
    occupied = np.flatnonzero(occupations > 1e-6)
    np.testing.assert_array_equal(actual.band_indices, occupied)
    bands = np.asarray(reference.wfc.coeffs)[occupied].T
    projected = left[:, retained] @ (left[:, retained].conj().T @ bands)
    expected = np.sum(np.abs(bands - projected) ** 2, axis=0) / np.sum(np.abs(bands) ** 2, axis=0)
    assert np.all(expected > ATOL_PW_PROJECTION_FP64_LOOSE)
    np.testing.assert_allclose(
        actual.residuals, expected, atol=ATOL_PW_PROJECTION_FP64_LOOSE, rtol=RTOL_PW_PROJECTION_FP64_LOOSE
    )
    weights = occupations[occupied]
    mean_expected = np.sum(weights * expected) / np.sum(weights)
    np.testing.assert_allclose(
        actual.mean_residual, mean_expected, atol=ATOL_PW_PROJECTION_FP64_LOOSE, rtol=RTOL_PW_PROJECTION_FP64_LOOSE
    )
    np.testing.assert_allclose(
        actual.max_residual, np.max(expected), atol=ATOL_PW_PROJECTION_FP64_LOOSE, rtol=RTOL_PW_PROJECTION_FP64_LOOSE
    )
