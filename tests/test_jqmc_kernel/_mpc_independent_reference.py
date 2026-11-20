"""Independent references for MPC geometry, singular kernel and cubic interpolation.

No production MPC helper supplies a reference value. The smooth box-face
quadrature is for an orthorhombic physical lattice, even when its basis is
represented by an integer unimodular shear. The spline reference instead uses
SciPy's periodic cubic linear-system construction, not FFT deconvolution.
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

import numpy as np
from scipy.interpolate import CubicSpline


def periodic_cubic_reference(values, fractions):
    """Evaluate the tensor product of three independently solved periodic cubics."""
    values = np.asarray(values, dtype=np.float64)
    answers = []
    for point in np.asarray(fractions, dtype=np.float64) % 1.0:
        work = values
        for coordinate in point:
            count = work.shape[0]
            closed = np.concatenate((work, work[:1]), axis=0)
            work = CubicSpline(np.arange(count + 1) / count, closed, axis=0, bc_type="periodic")(coordinate)
        answers.append(float(work))
    return np.asarray(answers)


def manufactured_field(fractions):
    """A fixed real Fourier polynomial with mixed axes and nonzero phases."""
    fractions = np.asarray(fractions, dtype=np.float64)
    waves = np.array([[1, 0, 0], [0, 1, 1], [1, -1, 1]])
    phases = 2.0 * np.pi * (fractions @ waves.T)
    return 0.4 + np.cos(phases) @ np.array([0.3, -0.2, 0.15]) + np.sin(phases) @ np.array([0.1, 0.25, -0.12])


def box_kernel_coefficients(lengths, cartesian_vectors, order):
    r"""Compute volume-normalized integral of cos(G.r)/|r| over the centered box.

    Split the box into six face pyramids, r=t*p. The Jacobian divided by
    |r| is t*d/|p|, hence smooth even at t=0. Opposite faces combine into
    cos(G.r). Integrating t analytically gives sin(a)/a+(cos(a)-1)/a**2,
    evaluated by a convergent power series to avoid cancellation. Only the
    smooth two-dimensional face integrals use Gauss--Legendre quadrature.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    vectors = np.asarray(cartesian_vectors, dtype=np.float64)
    x, w = np.polynomial.legendre.leggauss(order)
    coefficients = np.zeros(len(vectors))
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        points = np.zeros((order, order, 3))
        points[..., axis] = lengths[axis] / 2.0
        points[..., other[0]] = lengths[other[0]] * x[:, None] / 2.0
        points[..., other[1]] = lengths[other[1]] * x[None, :] / 2.0
        weights = np.outer(w, w) * lengths[other[0]] * lengths[other[1]] / 4.0
        alpha = points @ vectors.T
        # Integral series: sum_n (-1)^n a^(2n)/((2n)! (2n+2)).
        # The finite fixture frequencies have |a| <= 2*pi; 32 terms exceed
        # the binary64 precision requirement without a numerical cutoff.
        term = np.ones_like(alpha)
        radial = term / 2.0
        for n in range(1, 32):
            term *= -(alpha * alpha) / ((2 * n - 1) * (2 * n))
            radial += term / (2 * n + 2)
        coefficients += np.einsum("ij,ijk->k", weights * lengths[axis] / np.linalg.norm(points, axis=-1), radial)
    return coefficients / np.prod(lengths)


#: Fixed table-calibration fixture: physical box, Cartesian
#: plane-wave modes, MO coefficients, twist fractions, cutoff (Ha) and grid.
TABLE_FIXTURE_LENGTHS = np.array([5.0, 6.0, 7.0])
TABLE_FIXTURE_PW_MODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
TABLE_FIXTURE_MO_COEFFICIENTS = np.array([1.0, 0.5 + 0.25j, -0.25 + 0.5j])
TABLE_FIXTURE_TWIST = np.array([0.25, -0.125, 0.375])
TABLE_FIXTURE_CUTOFF_HA = 1.3
TABLE_FIXTURE_GRID = 16
#: Integer unimodular basis transforms; both describe the same physical box.
TABLE_FIXTURE_TRANSFORMS = {
    "orthorhombic": np.eye(3, dtype=int),
    "sheared-basis": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
}


def plane_wave_mpc_reference(
    transform, ao_real_dtype, ao_complex_dtype, fractions, order=48, *, node_density=None, n_electron=2
):
    """Independent V_LR on the table nodes and at ``fractions`` for the fixed fixture.

    The kernel K_G = 4 pi/(volume G^2) - f_G (K_0 = -f_0) uses
    :func:`box_kernel_coefficients` for f_G over the physical box. The density
    is the doubly occupied plane-wave MO evaluated directly by NumPy on the
    table nodes, with the plane-wave phase in the caller's AO precision (the
    production AO precision of the active mode), normalised to two electrons;
    its Fourier coefficients and every later sum are fp64. Returns
    ``(node_values, off_grid_values)``; node values are ordered like a
    C-order flattening of the (N, N, N) table.

    ``node_density`` optionally replaces that fixture density by a caller-supplied,
    directly evaluated (unnormalised) density on the same C-order table nodes of
    the basis ``transform @ diag(lengths)``; it is normalised to ``n_electron``
    electrons. Box, kernel, cutoff and grid stay those of the fixed fixture.
    """
    lengths = TABLE_FIXTURE_LENGTHS
    size = TABLE_FIXTURE_GRID
    cell = np.asarray(transform) @ np.diag(lengths)
    volume = float(np.prod(lengths))
    modes = np.array(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij")).reshape(3, -1).T
    vectors = 2.0 * np.pi * modes / lengths
    keep = np.sum(vectors**2, axis=1) / 2.0 <= TABLE_FIXTURE_CUTOFF_HA
    modes, vectors = modes[keep], vectors[keep]
    g2 = np.sum(vectors**2, axis=1)
    kernel = -box_kernel_coefficients(lengths, vectors, order)
    kernel[g2 > 0] += 4.0 * np.pi / (volume * g2[g2 > 0])
    basis_modes = modes @ np.asarray(transform).T
    pw = 2.0 * np.pi * TABLE_FIXTURE_PW_MODES / lengths
    k_cart = TABLE_FIXTURE_TWIST @ (2.0 * np.pi * np.linalg.inv(cell).T)
    nodes = np.array(np.meshgrid(*[np.arange(size) / size] * 3, indexing="ij")).reshape(3, -1).T
    if node_density is None:
        phases = (nodes @ cell).astype(ao_real_dtype) @ (pw + k_cart).astype(ao_real_dtype).T
        orbitals = np.exp(ao_complex_dtype(1j) * phases)
        rho = 2.0 * np.abs(orbitals @ TABLE_FIXTURE_MO_COEFFICIENTS) ** 2
    else:
        rho = np.asarray(node_density, dtype=np.float64).reshape(-1)
    rho = rho * (n_electron / (np.mean(rho) * volume))
    exponentials = np.exp(2j * np.pi * (nodes @ basis_modes.T))
    rho_modes = exponentials.conj().T @ rho / len(rho)
    node_values = volume * (exponentials @ (kernel * rho_modes)).real
    off_grid = volume * (np.exp(2j * np.pi * (np.asarray(fractions) @ basis_modes.T)) @ (kernel * rho_modes)).real
    return node_values, off_grid


def plane_wave_mpc_energy_reference(transform, ao_real_dtype, ao_complex_dtype, r_up, r_dn, order=48, *, node_density=None):
    """Independent canonical-MPC e-e energy of the fixed fixture for one configuration.

    ``V = sum_{i<j} 1/|r_ij^MI| + sum_i V_LR(r_i) + V_const`` with every term built
    without production helpers: the minimum image is the exact orthorhombic wrap of
    the physical box (``d - L round(d / L)``, independent of the cell basis),
    ``V_LR`` comes from :func:`plane_wave_mpc_reference` at the electron fractions of
    the basis ``transform @ diag(lengths)``, and ``V_const = -(volume^2 / 2)
    sum_G K_G |rho_G|^2`` uses the same box-kernel ``K_G`` and node density as that
    reference. Returns a Python float (Ha).

    ``node_density`` is passed to :func:`plane_wave_mpc_reference` (and used for
    ``V_const``) normalised to ``len(r_up) + len(r_dn)`` electrons; when omitted the
    fixed doubly occupied fixture density (two electrons) is used.
    """
    lengths = TABLE_FIXTURE_LENGTHS
    size = TABLE_FIXTURE_GRID
    cell = np.asarray(transform) @ np.diag(lengths)
    volume = float(np.prod(lengths))
    r_all = np.concatenate([np.asarray(r_up, dtype=np.float64), np.asarray(r_dn, dtype=np.float64)], axis=0)

    pair = 0.0
    for i in range(len(r_all)):
        for j in range(i + 1, len(r_all)):
            d = r_all[i] - r_all[j]
            d = d - lengths * np.round(d / lengths)
            pair += 1.0 / float(np.linalg.norm(d))

    fractions = r_all @ np.linalg.inv(cell)
    n_electron = len(r_all) if node_density is not None else 2
    _, v_lr = plane_wave_mpc_reference(
        transform, ao_real_dtype, ao_complex_dtype, fractions, order=order, node_density=node_density, n_electron=n_electron
    )

    modes = np.array(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij")).reshape(3, -1).T
    vectors = 2.0 * np.pi * modes / lengths
    keep = np.sum(vectors**2, axis=1) / 2.0 <= TABLE_FIXTURE_CUTOFF_HA
    modes, vectors = modes[keep], vectors[keep]
    g2 = np.sum(vectors**2, axis=1)
    kernel = -box_kernel_coefficients(lengths, vectors, order)
    kernel[g2 > 0] += 4.0 * np.pi / (volume * g2[g2 > 0])
    basis_modes = modes @ np.asarray(transform).T
    pw = 2.0 * np.pi * TABLE_FIXTURE_PW_MODES / lengths
    k_cart = TABLE_FIXTURE_TWIST @ (2.0 * np.pi * np.linalg.inv(cell).T)
    nodes = np.array(np.meshgrid(*[np.arange(size) / size] * 3, indexing="ij")).reshape(3, -1).T
    if node_density is None:
        phases = (nodes @ cell).astype(ao_real_dtype) @ (pw + k_cart).astype(ao_real_dtype).T
        orbitals = np.exp(ao_complex_dtype(1j) * phases)
        rho = 2.0 * np.abs(orbitals @ TABLE_FIXTURE_MO_COEFFICIENTS) ** 2
    else:
        rho = np.asarray(node_density, dtype=np.float64).reshape(-1)
    rho = rho * (n_electron / (np.mean(rho) * volume))
    rho_modes = np.exp(2j * np.pi * (nodes @ basis_modes.T)).conj().T @ rho / len(rho)
    v_const = -0.5 * volume**2 * float(np.sum(kernel * np.abs(rho_modes) ** 2))
    return float(pair + np.sum(v_lr) + v_const)
