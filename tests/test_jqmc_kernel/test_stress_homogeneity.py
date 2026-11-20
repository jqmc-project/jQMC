"""Joint-scaling homogeneity (virial) oracle for the periodic stress channels.

For an all-electron Coulomb system whose orbitals are functions of the
fractional coordinates only (a Gamma real plane-wave determinant, no Jastrow),
scaling the cell, every nucleus and every electron by ``s`` gives exactly

* ``T(s) = T / s^2`` (kinetic energy; ``Psi_s(s r) = Psi(r)``), and
* ``V_ch(s) = V_ch / s`` for each Ewald channel ``ch`` in ``ii``, ``ei``,
  ``ee`` (Coulomb homogeneity; each channel carries its own neutralising
  background and is independent of the split parameter).

Hence ``dT/ds|_1 = -2 T`` and ``dV_ch/ds|_1 = -V_ch``.  A hydrostatic strain
``F = (1 + s') I`` is this scaling, so the trace of every channel's strain
derivative through the production strain map ``Hamiltonian_data._apply_strain``
must equal the same closed forms, and the trace of the production HF stress
kernel must equal ``-2 T - V``.

The oracle has three legs, none of which differentiates through
``_apply_strain``:

1. value homogeneity: each channel evaluated on a Hamiltonian rebuilt from raw
   inputs by the public builders at the scaled cell (fresh Ewald split, image
   and reciprocal sets, fresh plane-wave indices) equals ``s^-k`` times the
   unscaled value;
2. AD homogeneity: ``d/ds`` by ``jax.grad`` of each channel on a Hamiltonian
   whose every ``Structure_data`` leaf is constructed in this file from the
   scaled raw cell and nuclei (constructor calls, not the strain helpers),
   equals ``-k`` times the value; the same construction under a general
   ``F = I + eps`` gives the full 3x3 tensor reference for leg 3;
3. the production route: ``jax.jacrev`` over ``eps`` of each channel through
   ``_apply_strain`` must match the leg-2 tensor entry-wise and its trace must
   equal the closed form; the production analytic HF stress kernel's trace must
   equal the virial ``-2 T - V``.
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
import pytest

from jqmc import derivatives
from jqmc._precision import get_tolerance_min
from jqmc.atomic_orbital import AOs_plane_wave_data
from jqmc.coulomb_potential import (
    Coulomb_potential_data,
    Ewald_data,
    build_ewald_data,
    compute_ewald_potential_ee,
    compute_ewald_potential_ei,
    compute_ewald_potential_ii,
)
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data, compute_kinetic_energy

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

#: Triclinic cell (rows are lattice vectors, bohr).
_CELL = 5.5 * (np.eye(3) + np.array([[0.05, 0.12, -0.04], [0.03, -0.06, 0.09], [-0.08, 0.04, 0.10]]))
_NUCLEI_FRAC = np.array([[0.11, 0.23, 0.17], [0.58, 0.61, 0.52]])
_ELECTRON_UP_FRAC = np.array([[0.31, 0.72, 0.44]])
_ELECTRON_DN_FRAC = np.array([[0.83, 0.15, 0.66]])
#: Miller indices of the Gamma real plane waves (cos for all, sin for G != 0).
_MILLERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, -1], [1, 0, 1]])
_C_UP = np.array([[2.1, 0.3, -0.4, 0.25, 0.1, -0.2, 0.15, 0.35, -0.1, 0.05, 0.2, -0.3, 0.12]])
_C_DN = np.array([[1.8, -0.2, 0.3, 0.1, -0.35, 0.25, 0.05, -0.15, 0.3, 0.2, -0.1, 0.1, 0.22]])
_SCALES = (0.9, 1.1)
#: Homogeneity degree k of each channel: Q(s) = Q(1) / s^k.
_DEGREE = {"T": 2, "ii": 1, "ei": 1, "ee": 1}
#: Precision zones on each channel's evaluation path (tolerance = loosest zone, strict tier).
_KINETIC_ZONES = ("ao_eval", "ao_grad_lap", "mo_eval", "mo_grad", "mo_lap", "det_eval", "det_grad_lap", "wf_kinetic")
_ZONES = {"T": _KINETIC_ZONES, "ii": ("ewald",), "ei": ("ewald",), "ee": ("ewald",)}
_VIRIAL_ZONES = _KINETIC_ZONES + ("ewald", "coulomb", "local_energy")


def _basis():
    millers, is_sin = [], []
    for row in _MILLERS:
        millers.append(row)
        is_sin.append(0.0)
        if np.any(row != 0):
            millers.append(row)
            is_sin.append(1.0)
    return np.asarray(millers, dtype=np.float64), np.asarray(is_sin)


_BASIS_MILLERS, _BASIS_IS_SIN = _basis()


def _structure_kwargs():
    return dict(pbc_flag=True, atomic_numbers=(1, 1), element_symbols=("H", "H"), atomic_labels=("H0", "H1"))


def build_hamiltonian(scale=1.0):
    """Fixture rebuilt from raw inputs by the public builders at cell ``scale * _CELL``."""
    cell = scale * _CELL
    structure = Structure_data.build(
        positions=_NUCLEI_FRAC @ cell, vec_a=cell[0], vec_b=cell[1], vec_c=cell[2], **_structure_kwargs()
    )
    leaf = AOs_plane_wave_data.build(structure, _BASIS_MILLERS @ (2.0 * np.pi * np.linalg.inv(cell).T), _BASIS_IS_SIN)
    determinant = SD_data.build(
        num_electron_up=1,
        num_electron_dn=1,
        mos_up=MOs_data.build(aos_data=leaf, mo_coefficients=_C_UP),
        mos_dn=MOs_data.build(aos_data=leaf, mo_coefficients=_C_DN),
        time_reversal=False,
    )
    wavefunction = Wavefunction_data(
        geminal_data=determinant,
        jastrow_data=Jastrow_data(
            jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
        ),
    )
    potential = Coulomb_potential_data(structure_data=structure)
    potential = potential.replace(ewald_data=build_ewald_data(structure, potential, n_electron=2))
    ham = Hamiltonian_data.build(structure_data=structure, coulomb_potential_data=potential, wavefunction_data=wavefunction)
    return ham, _ELECTRON_UP_FRAC @ cell, _ELECTRON_DN_FRAC @ cell


def _constructed_hamiltonian(base, F):
    """The fixture at cell ``_CELL @ F.T`` with every structure leaf constructed here.

    ``F`` may be a tracer.  Each ``Structure_data`` is a fresh constructor call
    on the mapped raw cell and nuclei; every consumer (plane-wave leaf, both
    MO sets, determinant, Coulomb container, Ewald tables, Hamiltonian) is a
    fresh constructor call around it.  Only integer/static data (Miller
    indices, Ewald split and index sets, the minimum-image box) come from the
    unstrained build ``base``; they are piecewise constant in the strain.
    """
    cell = jnp.asarray(_CELL) @ F.T
    structure = Structure_data(
        positions=jnp.asarray(_NUCLEI_FRAC) @ cell,
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
        _n_max=base.structure_data._n_max,
        **_structure_kwargs(),
    )
    base_leaf = base.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    leaf = AOs_plane_wave_data(
        structure_data=structure, num_ao=base_leaf.num_ao, _g_indices=base_leaf._g_indices, phase_is_sin=_BASIS_IS_SIN
    )
    base_gem = base.wavefunction_data.geminal_data
    determinant = SD_data(
        num_electron_up=1,
        num_electron_dn=1,
        orb_data_up_spin=MOs_data(num_mo=1, mo_coefficients=_C_UP, aos_data=leaf),
        orb_data_dn_spin=MOs_data(num_mo=1, mo_coefficients=_C_DN, aos_data=leaf),
        time_reversal=base_gem.time_reversal,
    )
    wavefunction = Wavefunction_data(geminal_data=determinant, jastrow_data=base.wavefunction_data.jastrow_data)
    base_ewald = base.coulomb_potential_data.ewald_data
    ewald = Ewald_data(
        structure_data=structure,
        eta=base_ewald.eta,
        r_real_cut=base_ewald.r_real_cut,
        g_max=base_ewald.g_max,
        real_image_indices=base_ewald.real_image_indices,
        g_indices=base_ewald.g_indices,
    )
    potential = Coulomb_potential_data(structure_data=structure, ewald_data=ewald)
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=potential, wavefunction_data=wavefunction)


def channels(ham, r_up, r_dn):
    """Kinetic energy and the three Ewald channels of the production evaluators."""
    potential = ham.coulomb_potential_data
    ei_up, ei_dn = compute_ewald_potential_ei(potential, r_up, r_dn, potential.ewald_data)
    return {
        "T": jnp.real(compute_kinetic_energy(ham.wavefunction_data, r_up, r_dn)),
        "ii": compute_ewald_potential_ii(potential, potential.ewald_data),
        "ei": jnp.sum(ei_up) + jnp.sum(ei_dn),
        "ee": compute_ewald_potential_ee(r_up, r_dn, potential.ewald_data),
    }


def _electrons(F):
    return jnp.asarray(_ELECTRON_UP_FRAC @ _CELL) @ F.T, jnp.asarray(_ELECTRON_DN_FRAC @ _CELL) @ F.T


def _constructed_strain_tensor(base):
    """``d Q / d eps`` of every channel on the constructed (non-``_apply_strain``) route."""

    def route(eps):
        F = jnp.eye(3) + eps
        return channels(_constructed_hamiltonian(base, F), *_electrons(F))

    return jax.jacrev(route)(jnp.zeros((3, 3)))


def _production_strain_tensor(ham):
    """``d Q / d eps`` of every channel through the production ``_apply_strain``."""

    def route(eps):
        F = jnp.eye(3) + eps
        return channels(ham._apply_strain(F), *_electrons(F))

    return jax.jacrev(route)(jnp.zeros((3, 3)))


def _assert_close(actual, expected, label, zones):
    actual, expected = np.asarray(actual), np.asarray(expected)
    atol, rtol = get_tolerance_min(zones, "strict")
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), label
    assert np.any(np.abs(expected) > atol + rtol * np.abs(expected)), f"{label}: vacuous reference"
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def test_rebuilt_channels_are_homogeneous():
    """Leg 1 and leg 2: rebuilt values scale as s^-k and their AD scaling derivative is -k Q.

    Fixture: triclinic cell, two all-electron H nuclei, one up and one down
    electron in a 13-function Gamma real plane-wave SD, no Jastrow, fixed
    electron fractional coordinates.  For s in {0.9, 1.1} every channel of a
    Hamiltonian rebuilt from raw inputs by the public builders equals
    ``Q(1) s^-k`` (k = 2 for T, 1 for ii/ei/ee).  ``jax.grad`` in s of each
    channel on the constructed-leaf route (module docstring) equals ``-k Q(1)``.
    Deterministic, strict zone tolerance, full and mixed each in their own mode.
    """
    ham, r_up, r_dn = build_hamiltonian()
    values = {key: float(value) for key, value in channels(ham, r_up, r_dn).items()}
    for scale in _SCALES:
        rebuilt, up, dn = build_hamiltonian(scale)
        for key, value in channels(rebuilt, up, dn).items():
            _assert_close(value, values[key] / scale ** _DEGREE[key], f"value/{key}/s={scale}", _ZONES[key])

    for key, degree in _DEGREE.items():

        def along_scale(s, key=key):
            F = s * jnp.eye(3)
            return channels(_constructed_hamiltonian(ham, F), *_electrons(F))[key]

        _assert_close(jax.grad(along_scale)(1.0), -degree * values[key], f"ad/{key}", _ZONES[key])


def test_production_strain_channels_obey_homogeneity():
    """Leg 3: production ``_apply_strain`` strain derivatives against the homogeneity oracle.

    Same fixture.  For each channel the 3x3 ``jax.jacrev`` over eps through
    ``Hamiltonian_data._apply_strain`` equals entry-wise the constructed-leaf
    tensor (full tensor, including shear entries) and its trace equals
    ``-k Q``.  The production analytic HF stress kernel
    (``derivatives.compute_grad_e_L(..., {"opt_eps": True}, analytic=True)``)
    has trace ``-2 T - V_ii - V_ei - V_ee``.  Deterministic, strict zone
    tolerance, full and mixed each in their own mode.
    """
    ham, r_up, r_dn = build_hamiltonian()
    values = {key: float(value) for key, value in channels(ham, r_up, r_dn).items()}
    constructed = _constructed_strain_tensor(ham)
    production = _production_strain_tensor(ham)
    for key, degree in _DEGREE.items():
        _assert_close(production[key], constructed[key], f"tensor/{key}", _ZONES[key])
        _assert_close(np.trace(np.asarray(production[key])), -degree * values[key], f"trace/{key}", _ZONES[key])
    stress = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), {"opt_eps": True}, analytic=True)["eps"]
    virial = -sum(degree * values[key] for key, degree in _DEGREE.items())
    _assert_close(np.trace(np.real(np.asarray(stress))), virial, "trace/analytic-stress", _VIRIAL_ZONES)


def test_homogeneity_oracle_rejects_unstrained_ewald_leaf(monkeypatch):
    """The oracle is red when ``_apply_strain`` stops rebuilding the Ewald tables' cell.

    Same fixture.  ``Hamiltonian_data._apply_strain`` is replaced by a copy
    that strains every leaf except ``coulomb_potential_data.ewald_data``, so the
    in-trace Ewald tables (real-space images, reciprocal vectors, g-factors,
    volume) keep the unstrained cell.  The ii, ei and ee trace checks of
    ``test_production_strain_channels_obey_homogeneity`` must then fail, while
    the kinetic channel is unaffected.
    """
    from jqmc.structure import _strain_structure_data

    def skip_ewald(self, F):
        potential = self.coulomb_potential_data
        return self.replace(
            structure_data=_strain_structure_data(self.structure_data, F),
            coulomb_potential_data=potential.replace(structure_data=_strain_structure_data(potential.structure_data, F)),
            wavefunction_data=self.wavefunction_data._apply_strain(F),
        )

    ham, r_up, r_dn = build_hamiltonian()
    values = {key: float(value) for key, value in channels(ham, r_up, r_dn).items()}
    monkeypatch.setattr(Hamiltonian_data, "_apply_strain", skip_ewald)
    production = _production_strain_tensor(ham)
    _assert_close(np.trace(np.asarray(production["T"])), -2.0 * values["T"], "mutated/trace/T", _ZONES["T"])
    for key in ("ii", "ei", "ee"):
        with pytest.raises(AssertionError, match=rf"(?s)Not equal to tolerance.*mutated/trace/{key}"):
            _assert_close(np.trace(np.asarray(production[key])), -values[key], f"mutated/trace/{key}", _ZONES[key])
