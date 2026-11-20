"""Analytic VGL cotangents for ragged tiled providers, with no global AO matrix.

Precision Zones: derivative accumulators use float64/complex128; existing
analytic AO kernels retain their own precision contracts.
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
import jax.numpy as jnp
import numpy as np

from jqmc.atomic_orbital import AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data, AOs_plane_wave_complex_data
from jqmc.orbital_tiling_geometry import (
    physical_block_backend,
    physical_center_displacements,
    physical_pgto_view,
    image_physical_atoms,
)
from jqmc.structure import _wrap_into_cell_if_pbc
from . import ao as A
from . import ao_deriv as G


def _cartesian_kernel_terms(leaf):
    """Expand each real harmonic into monomials without requiring complete shells.

    This is a transient raw-kernel view, never a stored AO provider or a folded
    Cartesian basis. Radial data remains shared; each term remembers the original
    home spherical AO and physical image. The transform is local to one harmonic.
    """
    from jqmc.atomic_orbital import _spherical_to_cart_matrix

    if isinstance(leaf, AOs_cart_pgto_complex_data):
        return leaf, jnp.ones(leaf._num_ao_kernel)
    source_kernel, weights, powers, primitives, indices = [], [], [], [], []
    prims_by_kernel = [[] for _ in range(leaf._num_ao_kernel)]
    for p, k in enumerate(leaf.orbital_indices):
        prims_by_kernel[k].append(p)
    for kernel, (l, m) in enumerate(zip(leaf.angular_momentums, leaf.magnetic_quantum_numbers, strict=True)):
        orders = [(x, y, l - x - y) for x in range(l, -1, -1) for y in range(l - x, -1, -1)]
        for order, weight in zip(orders, _spherical_to_cart_matrix(l)[m + l], strict=True):
            if weight == 0:
                continue
            term = len(source_kernel)
            source_kernel.append(kernel)
            weights.append(weight)
            powers.append(order)
            primitives.extend(prims_by_kernel[kernel])
            indices.extend([term] * len(prims_by_kernel[kernel]))
    take = tuple(source_kernel)
    view = AOs_cart_pgto_complex_data(
        structure_data=leaf.structure_data,
        radial=leaf.radial,
        ao_prim_to_shell_prim=tuple(leaf.ao_prim_to_shell_prim[p] for p in primitives),
        num_ao=leaf.num_ao,
        num_ao_prim=len(primitives),
        nucleus_index=tuple(range(len(take))),
        orbital_indices=tuple(indices),
        angular_momentums=tuple(leaf.angular_momentums[k] for k in take),
        polynominal_order_x=tuple(v[0] for v in powers),
        polynominal_order_y=tuple(v[1] for v in powers),
        polynominal_order_z=tuple(v[2] for v in powers),
        k_point=leaf.k_point,
        _n_image=leaf._n_image,
        _image_indices=tuple(leaf._image_indices[k] for k in take),
        _ragged_kernel_atom=tuple(leaf._ragged_kernel_atom[k] for k in take),
        _ragged_kernel_to_home=tuple(leaf._ragged_kernel_to_home[k] for k in take),
        _image_phase_factors=jnp.asarray(leaf._image_phase_factors)[jnp.asarray(take)],
    )
    return view, jnp.asarray(weights)


def tiled_field_vjp(orbitals, r, bar_value, bar_grad=None, bar_lap=None, *, wrt):
    """Contract final-orbital VGL bars with physical R, affine eps or electron r.

    The caller supplies holomorphic bars. Re is taken by the real observable,
    not here, so this also supports complex nonlocal wavefunction ratios.
    Logical conjugation is handled by conjugating bars and the final result;
    local C and primitive backend storage stay shared and unchanged.
    """
    if wrt not in ("R", "eps", "r"):
        raise ValueError(f"Unknown tiled derivative coordinate: {wrt}")
    shape = (3, 3) if wrt == "eps" else ((len(r), 3) if wrt == "r" else (orbitals.structure_data.natom, 3))
    out = jnp.zeros(shape, jnp.complex128)
    offsets = orbitals.orbital_offsets
    displacement = physical_center_displacements(orbitals)
    for index, block in enumerate(orbitals.blocks):
        C = None if block.coefficients is None else jax.lax.stop_gradient(jnp.asarray(block.coefficients))

        def project(bar):
            if bar is None:
                return None
            value = jnp.asarray(bar)[..., offsets[index] : offsets[index + 1], :]
            if block.conjugate:
                value = jnp.conj(value)
            if C is not None:
                value = jnp.einsum("ab,...ai->...bi", C, value)
            return value / np.sqrt(len(orbitals.blocks))

        values, gradients, laps = project(bar_value), project(bar_grad), project(bar_lap)
        backend = physical_block_backend(orbitals, block.backend)
        offset = 0
        for leaf in getattr(backend, "children", (backend,)):
            stop = offset + leaf.num_ao
            cv = values[offset:stop]
            cg = None if gradients is None else gradients[:, offset:stop]
            cl = None if laps is None else laps[offset:stop]
            offset = stop
            pgto = isinstance(leaf, (AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data))
            if not pgto and wrt == "R":
                continue
            if pgto:
                leaf, term_weights = _cartesian_kernel_terms(leaf)
                home = jnp.asarray(leaf._ragged_kernel_to_home, jnp.int32)

                def one(electron, position):
                    view = physical_pgto_view(orbitals, leaf, position, displacement)
                    wrapped = _wrap_into_cell_if_pbc(view.structure_data, position[None, :])
                    phase = jnp.asarray(view._image_phase_factors) * view._restore_phase(position[None, :], "ao_grad_lap")[0]
                    field = G.ao_grad(view, wrapped)[:, :, 0] * cv[home, electron][None, :]
                    if cl is not None:
                        field = field + G.ao_gradlap(view, wrapped)[:, :, 0] * cl[home, electron][None, :]
                    if cg is not None:
                        field = field + jnp.einsum("cdk,dk->ck", G.ao_hessian(view, wrapped)[:, :, :, 0], cg[:, home, electron])
                    field = field * (phase * term_weights)[None, :]
                    if wrt == "r":
                        return field.sum(axis=1)
                    if wrt == "eps":
                        diff = wrapped[0] - view._expanded_positions_cart_jnp
                        return jnp.einsum("ck,kd->cd", field, diff)
                    atoms = image_physical_atoms(orbitals, leaf, position)
                    return jax.ops.segment_sum(-field.T, atoms, num_segments=orbitals.structure_data.natom)

                response = jax.vmap(one)(jnp.arange(len(r)), jnp.asarray(r))
                contribution = response if wrt == "r" else response.sum(axis=0)
            elif wrt == "eps":
                if cg is None and cl is None:
                    continue  # PW/spline VALUES are affine invariant.
                cg = jnp.zeros((3, leaf.num_ao, len(r)), jnp.complex128) if cg is None else cg
                cl = jnp.zeros((leaf.num_ao, len(r)), jnp.complex128) if cl is None else cl
                if isinstance(leaf, AOs_plane_wave_complex_data):
                    from .pbc_stress import _pw_kinetic_stress_contrib

                    contribution = _pw_kinetic_stress_contrib(leaf, cv, cl, cg, r)
                else:
                    contribution = A.bspline_complex_kinetic_stress_contrib(leaf, cl, cg, r)
            else:
                if isinstance(leaf, AOs_plane_wave_complex_data):
                    gradient, gradlap, hessian = A.pw_grad(leaf, r), A.pw_gradlap(leaf, r), A.pw_hessian(leaf, r)
                else:
                    gradient, gradlap, hessian = A._bspline_complex_kinetic_fields(leaf, r)
                contribution = jnp.einsum("bi,cbi->ic", cv, gradient)
                if cl is not None:
                    contribution += jnp.einsum("bi,cbi->ic", cl, gradlap)
                if cg is not None:
                    contribution += jnp.einsum("dbi,cdbi->ic", cg, hessian)
            out = out + (jnp.conj(contribution) if block.conjugate else contribution)
    return out
