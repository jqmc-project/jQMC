"""Forward non-local ECP quadrature parity under NON-identity mesh rotations,
Rust vs XLA.

The jQMC physics battery (``tests/test_jqmc_kernel/test_coulomb_potential.py``,
around lines 795 / 1048 / 1390, driven by ``angle_params`` at 141-157) validates
the XLA non-local ECP quadrature under 9 non-identity Euler-mesh rotations x
Nv in {4, 6, 12, 18}. But EVERY forward Rust==XLA parity call fixes the mesh at
the identity: ``test_ecp_local.py`` lines 88/109/127 (``RT = np.eye(3)``) and
``test_mcmc_observables.py`` 460/478. Only the GFMC path
(``test_gfmc_projection.py`` 516-529) ever compared a rotated mesh. A bug in how
the FORWARD Rust ECP quadrature consumes rotated grid points -- ignoring the
rotation, re-deriving it, or transposing the ``RT`` vs ``R`` convention -- would
pass every existing forward parity test.

Convention (mirrors the identity-mesh siblings exactly): the XLA oracle takes the
rotation as ``RT=rt``; the Rust kernel takes the SAME points already rotated,
``gp = mesh.grid_points @ rt``. The rotation matrices are INPUTS (a deterministic
sweep of the ``random_discretized_mesh`` degree of freedom), never reference data.

NOTE: the Rust forward ECP kernels expose only the SCALAR summed
non-local energy (``compute_ecp_nonlocal_parts_all_pairs`` returns a ``float``;
``compute_ecp_nonlocal_withJ_complex`` returns a ``(re, im)`` float pair). They do
NOT return the per-mesh-point ``V`` array that the XLA
``compute_ecp_non_local_parts_*`` functions return as their [0]/[1]/[2] outputs, so
element-wise per-mesh-point parity is NOT reachable from the current PyO3 surface.
Closing that would require a new test-only Rust binding (out of scope here -- no
.rs file is edited); this module pins the summed quadrature, which is the value
the production local energy consumes.
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
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import (
    compute_ecp_non_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_holomorphic,
    icosahedron_sym_mesh_Nv12,
    octahedron_sym_mesh_Nv6,
    octahedron_sym_mesh_Nv18,
    tetrahedron_sym_mesh_Nv4,
)

from tests.test_rs_kernel._rs_fixtures import rs_ham

# Reuse the battery's deterministic non-identity R^T generator (a stack of
# Rz(c) @ Ry(b) @ Rx(a) rotations, transposed) rather than re-inventing it.
from tests.test_rs_kernel.test_gfmc_projection import _rt_stack

# Same precision zones the identity-mesh sibling uses (test_ecp_local.py:42).
_ZONES_ECP_NONLOCAL = ("coulomb", "det_ratio", "jastrow_ratio")

_MESHES = {
    4: tetrahedron_sym_mesh_Nv4,
    6: octahedron_sym_mesh_Nv6,
    12: icosahedron_sym_mesh_Nv12,
    18: octahedron_sym_mesh_Nv18,
}

#: A deterministic sweep of non-identity mesh rotations (>= 3), shared by every
#: case. Each is a proper rotation (det = +1); none is the identity.
_RTS = _rt_stack(3, seed=5)


def _electrons(ham, seed, scale=0.5):
    """Deterministic electron positions scattered off the nuclei (copy of the
    identity-mesh sibling's helper, test_ecp_local.py:58)."""
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    r_dn = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return r_up.astype(np.float64), r_dn.astype(np.float64)


@pytest.mark.light
@pytest.mark.parametrize("nv", [4, 6, 12, 18])
@pytest.mark.parametrize("pattern", ["mol_ecp_cart_j12", "mol_ecp_sphe_j12"])
def test_ecp_nonlocal_rotated_mesh_molecular(pattern, nv):
    """Molecular forward non-local ECP quadrature (all-pairs) under a sweep of
    non-identity mesh rotations, Rust vs XLA, for every quadrature sphere."""
    mesh = _MESHES[nv]
    ham = rs_ham(pattern)
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    r_up, r_dn = _electrons(ham, 4)
    w = np.asarray(mesh.weights, dtype=float)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    for rotation_index, rt in enumerate(_RTS):
        rotation = f"rotation-{rotation_index}"
        *_, sumV = compute_ecp_non_local_parts_all_pairs(cp, wf, r_up, r_dn, rt, Nv=nv, flag_determinant_only=False)
        gp = np.asarray(mesh.grid_points, dtype=float) @ rt
        vr = jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, False)
        assert np.isclose(vr, float(sumV), rtol=rtol, atol=atol), (rotation, pattern, nv, vr, float(sumV))


@pytest.mark.heavy
@pytest.mark.parametrize("nv", [6, 18])
def test_ecp_nonlocal_rotated_mesh_periodic(nv):
    """Periodic (Gamma) forward non-local ECP quadrature under non-identity mesh
    rotations, Rust vs XLA. XLA's all-pairs variant is image-blind, so the
    nearest-neighbour enumeration with ``NN = n_atom`` is the matching reference
    for the Rust ``nn`` argument (as in test_mcmc_observables.py:478)."""
    mesh = _MESHES[nv]
    ham = rs_ham("pbc_gamma_sphepgto_j12")
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    natom = len(np.asarray(ham.structure_data.positions))
    r_up, r_dn = _electrons(ham, 4)
    w = np.asarray(mesh.weights, dtype=float)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    for rotation_index, rt in enumerate(_RTS):
        rotation = f"rotation-{rotation_index}"
        xp = compute_ecp_non_local_parts_nearest_neighbors(cp, wf, r_up, r_dn, rt, NN=natom, Nv=nv, flag_determinant_only=False)
        ref = float(np.real(xp[3]))
        gp = np.asarray(mesh.grid_points, dtype=float) @ rt
        vr = jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, False, natom)
        assert np.isclose(vr, ref, rtol=rtol, atol=atol), (rotation, nv, vr, ref)


@pytest.mark.heavy
def test_ecp_nonlocal_rotated_mesh_twist_complex(nv=6):
    """Complex-twist (k != Gamma) forward non-local ECP quadrature under
    non-identity mesh rotations, Rust vs XLA. The Rust complex kernel returns a
    ``(re, im)`` pair keeping Im(Vnl), the continuum VMC matrix element, so the
    oracle is the un-projected XLA holomorphic nearest-neighbour sum (the fixed-phase
    ``compute_ecp_non_local_parts_nearest_neighbors`` projects Re per candidate for
    LRDMC). Both real and imaginary parts are pinned, and Im must be active."""
    mesh = _MESHES[nv]
    ham = rs_ham("twist_sphepgto_j12")
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    natom = len(np.asarray(ham.structure_data.positions))
    r_up, r_dn = _electrons(ham, 4)
    w = np.asarray(mesh.weights, dtype=float)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")
    for rotation_index, rt in enumerate(_RTS):
        rotation = f"rotation-{rotation_index}"
        xc = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(
            cp, wf, r_up, r_dn, rt, NN=natom, Nv=nv, flag_determinant_only=False
        )
        ref = complex(xc[3])
        assert abs(ref.imag) > atol + rtol * abs(ref.real), (rotation, "inactive im", ref)
        gp = np.asarray(mesh.grid_points, dtype=float) @ rt
        re, im = jqmc_rs.compute_ecp_nonlocal_withJ_complex(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp)
        assert np.isclose(re, ref.real, rtol=rtol, atol=atol), (rotation, "re", re, ref.real)
        assert np.isclose(im, ref.imag, rtol=rtol, atol=atol), (rotation, "im", im, ref.imag)


@pytest.mark.light
@pytest.mark.parametrize("nv,seed", [(6, 4)])
def test_ecp_rotated_vs_identity_mesh_differs(nv, seed):
    """Anti-vacuity: prove the rotation is LIVE. The rotated-mesh summed non-local
    energy must DIFFER from the identity-mesh value beyond the comparison
    tolerance for at least one rotation, on BOTH backends. If it did not, every
    parity case above would silently be a re-test of the identity mesh (e.g. a
    Rust kernel that discards the incoming rotation would still match an XLA
    oracle whose rotation had also been cancelled)."""
    mesh = _MESHES[nv]
    ham = rs_ham("mol_ecp_sphe_j12")
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    r_up, r_dn = _electrons(ham, seed)
    w = np.asarray(mesh.weights, dtype=float)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_ECP_NONLOCAL, level="strict")

    gp0 = np.asarray(mesh.grid_points, dtype=float)  # identity mesh
    v0_rust = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp0, False))
    *_, s0 = compute_ecp_non_local_parts_all_pairs(cp, wf, r_up, r_dn, np.eye(3), Nv=nv, flag_determinant_only=False)
    v0_xla = float(s0)

    rust_values = []
    xla_values = []
    for rt in _RTS:
        gp = np.asarray(mesh.grid_points, dtype=float) @ rt
        v_rust = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gem, wf.jastrow_data, cp, r_up, r_dn, w, gp, False))
        *_, sv = compute_ecp_non_local_parts_all_pairs(cp, wf, r_up, r_dn, rt, Nv=nv, flag_determinant_only=False)
        v_xla = float(sv)
        rust_values.append(v_rust)
        xla_values.append(v_xla)
    # At least one rotation must differ on each backend, using the unchanged
    # element-wise tolerance. Expose the compared values at the assertion site.
    assert not np.allclose(rust_values, v0_rust, rtol=rtol, atol=atol), (
        "Rust non-local ECP energy is identity-mesh-invariant -- the rotation is dead in Rust",
        v0_rust,
    )
    assert not np.allclose(xla_values, v0_xla, rtol=rtol, atol=atol), (
        "XLA non-local ECP energy is identity-mesh-invariant -- the rotation is dead in XLA",
        v0_xla,
    )
    np.testing.assert_allclose(v0_rust, v0_xla, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(rust_values, xla_values, atol=atol, rtol=rtol, equal_nan=False)
