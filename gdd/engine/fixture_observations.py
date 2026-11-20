"""Structural facts from a Hamiltonian consumed at a test's call boundary.

These facts prove presence, not nonzero physical activation. In particular an
ECP flag is not a nonlocal contribution and non-Gamma is not twist wiring.
Production imports are lazy; receipt validation needs only Python and NumPy.
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

import hashlib
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

from gdd.engine.wavefunction_fixtures import classify_cell_metric


AO_SIGNATURES = {
    "cart_gto": "gto-cartesian-real",
    "sphe_gto": "gto-spherical-real",
    "cart_pgto": "pgto-cartesian-real",
    "sphe_pgto": "pgto-spherical-real",
    "cart_pgto_complex": "pgto-cartesian-complex",
    "sphe_pgto_complex": "pgto-spherical-complex",
    "plane_wave": "plane-wave-real",
    "plane_wave_complex": "plane-wave-complex",
    "bspline": "bspline-real",
    "bspline_complex": "bspline-complex",
    "bspline_tiled_complex": "bspline-tiled-complex",
}
COMPOSITE_SIGNATURES = {
    ("pgto-cartesian-real", "plane-wave-real"): "pgto-plane-wave-real",
    ("pgto-cartesian-complex", "plane-wave-complex"): "pgto-plane-wave-complex",
    ("pgto-cartesian-real", "bspline-real"): "pgto-bspline-real",
    ("pgto-cartesian-complex", "bspline-complex"): "pgto-bspline-complex",
}


def input_object_snapshot(value) -> str:
    """Fingerprint concrete dataclass fields, not names, labels or cached properties."""
    digest = hashlib.sha256()

    def add(item):
        if is_dataclass(item) and not isinstance(item, type):
            digest.update(f"{type(item).__module__}.{type(item).__qualname__}".encode())
            for field in fields(item):
                digest.update(field.name.encode())
                add(getattr(item, field.name))
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}:".encode())
            for child in item:
                add(child)
        elif item is None or isinstance(item, str):
            digest.update(repr(item).encode())
        else:
            array = np.asarray(item)
            if array.dtype.kind not in "biufc":
                raise AssertionError(f"unsupported fixture field type: {type(item).__name__}")
            if not np.all(np.isfinite(array)):
                raise AssertionError("fixture field contains a nonfinite value")
            digest.update(f"{array.dtype.str}:{array.shape}:".encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b";")

    add(value)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class ComponentBoundaryReceipt:
    """Concrete cell on a consumed component; basis centres are not cell identity."""

    role: str
    periodic: bool
    cell: tuple[tuple[float, ...], ...]

    def validate(self) -> None:
        if not self.role or type(self.periodic) is not bool:
            raise AssertionError("invalid component boundary identity")
        if self.periodic:
            try:
                classify_cell_metric(self.cell)
            except (TypeError, ValueError) as exc:
                raise AssertionError("invalid component boundary cell") from exc
        elif self.cell:
            raise AssertionError("open component boundary carries a periodic cell")


def orbital_boundary_roles(role: str, signature: str) -> tuple[str, ...]:
    """Name every consumed leaf, including the second child of a composite."""
    if signature in AO_SIGNATURES.values():
        return (role,)
    for children, name in COMPOSITE_SIGNATURES.items():
        if signature == name:
            return tuple(f"{role}.{index}" for index in range(len(children)))
    raise AssertionError("unknown orbital signature for component boundary")


@dataclass(frozen=True)
class NuclearPotentialReceipt:
    """Per-nucleus ECP identity, not evidence of a nonzero ECP contribution."""

    atomic_number: int
    core_charge: float
    ecp_channels: int
    ecp_terms: int

    @property
    def pseudized(self) -> bool:
        return self.ecp_channels > 0 or self.ecp_terms > 0

    @property
    def effective_charge(self) -> float:
        return self.atomic_number - self.core_charge

    def validate(self) -> None:
        if type(self.atomic_number) is not int or self.atomic_number <= 0:
            raise AssertionError("invalid observed nuclear charge")
        if any(type(value) is not int or value < 0 for value in (self.ecp_channels, self.ecp_terms)):
            raise AssertionError("invalid observed ECP inventory")
        if not np.isfinite(self.core_charge) or not 0 <= self.core_charge <= self.atomic_number:
            raise AssertionError("invalid observed nuclear core charge")
        if self.core_charge and not self.pseudized:
            raise AssertionError("removed core charge without explicit ECP identity")


@dataclass(frozen=True)
class DTNCuspReceipt:
    """Species data and required physical targets, not measured polynomial slopes."""

    constraint_enabled: bool
    species_index: tuple[int, ...]
    effective_charges: tuple[float, ...]

    def validate(self, nuclei: tuple[NuclearPotentialReceipt, ...]) -> None:
        if self.constraint_enabled is not True:
            raise AssertionError("DTN cusp constraint must remain enabled")
        if (
            len(self.species_index) != len(nuclei)
            or any(type(value) is not int or value < 0 for value in self.species_index)
            or set(self.species_index) != set(range(len(self.effective_charges)))
            or not self.effective_charges
            or not np.all(np.isfinite(self.effective_charges))
        ):
            raise AssertionError("invalid DTN species inventory")
        for species, charge in enumerate(self.effective_charges):
            members = tuple(atom for atom, index in zip(nuclei, self.species_index, strict=True) if index == species)
            for atom in members:
                atom.validate()
            identities = {(atom.atomic_number, atom.effective_charge, atom.pseudized) for atom in members}
            if len(identities) != 1:
                raise AssertionError("DTN species mixes distinct nuclear potential identities")
            if charge != members[0].effective_charge:
                raise AssertionError("DTN effective charge differs from nuclear potential")

    def physical_targets(self, nuclei: tuple[NuclearPotentialReceipt, ...]) -> tuple[str, ...]:
        """One target per species; a mixed system must not become one global label."""
        self.validate(nuclei)
        return tuple(
            "ecp-zero" if nuclei[self.species_index.index(species)].pseudized else "ae-nonzero"
            for species in range(len(self.effective_charges))
        )


@dataclass(frozen=True)
class HamiltonianFixtureReceipt:
    """Observed structure of one concrete input, independent of its matrix label."""

    content_snapshot: str
    determinant_kind: str
    representation: str
    orbital_signatures: tuple[str, str]
    electron_counts: tuple[int, int]
    jastrow_components: tuple[str, ...]
    j1_form: str
    j2_form: str
    j3_representation: str
    j3_orbital_signature: str
    dtn_channels: tuple[str, ...]
    periodic: bool
    cell_metric: str
    k_points: tuple[tuple[float, float, float], ...]
    ecp: bool
    component_boundaries: tuple[ComponentBoundaryReceipt, ...]
    nuclei: tuple[NuclearPotentialReceipt, ...]
    dtn_cusp: DTNCuspReceipt | None

    def validate(self) -> None:
        if not self.content_snapshot.startswith("sha256:") or len(self.content_snapshot) != 71:
            raise AssertionError("fixture observation has no concrete content snapshot")
        if self.determinant_kind not in {"SD", "AGPn", "AGP"} or self.representation not in {"AO", "MO"}:
            raise AssertionError("unknown determinant structure")
        if self.determinant_kind != "AGP" and self.representation != "MO":
            raise AssertionError("SD/AGPn require MO representation")
        signatures = set(AO_SIGNATURES.values()) | set(COMPOSITE_SIGNATURES.values())
        if len(self.orbital_signatures) != 2 or any(item not in signatures for item in self.orbital_signatures):
            raise AssertionError("unknown orbital signature")
        if (
            len(self.electron_counts) != 2
            or any(type(item) is not int for item in self.electron_counts)
            or not 0 <= self.electron_counts[1] <= self.electron_counts[0]
            or self.electron_counts[0] == 0
        ):
            raise AssertionError("invalid observed electron counts")
        if self.jastrow_components != tuple(name for name in ("J1", "J2", "J3", "DTN") if name in self.jastrow_components):
            raise AssertionError("invalid observed Jastrow components")
        for channel, form in (("J1", self.j1_form), ("J2", self.j2_form)):
            if form not in {"off", "exp", "pade"} or (form != "off") != (channel in self.jastrow_components):
                raise AssertionError("Jastrow form disagrees with observed components")
        if "J3" not in self.jastrow_components:
            if (self.j3_representation, self.j3_orbital_signature) != ("off", "not-applicable"):
                raise AssertionError("J3 orbital without a J3 component")
        elif self.j3_representation not in {"AO", "MO"} or self.j3_orbital_signature not in signatures:
            raise AssertionError("invalid observed J3 orbital")
        elif self.j3_representation == "AO" and not self.j3_orbital_signature.startswith(
            ("gto-", "pgto-cartesian-", "pgto-spherical-")
        ):
            raise AssertionError("J3 AO representation requires a GTO or pGTO leaf")
        if self.dtn_channels != tuple(name for name in ("u", "chi", "f") if name in self.dtn_channels):
            raise AssertionError("invalid DTN channels")
        if self.dtn_channels and "DTN" not in self.jastrow_components:
            raise AssertionError("DTN channels without a DTN object")
        metrics = {"cubic", "tetragonal", "orthorhombic", "monoclinic", "hexagonal", "rhombohedral", "triclinic"}
        if type(self.periodic) is not bool or type(self.ecp) is not bool:
            raise AssertionError("fixture flags must be boolean")
        if self.cell_metric not in (metrics if self.periodic else {"not-applicable"}):
            raise AssertionError("cell metric disagrees with boundary")
        if not self.k_points or any(len(k) != 3 or not np.all(np.isfinite(k)) for k in self.k_points):
            raise AssertionError("invalid observed k points")
        if not self.periodic and any(value != 0 for k in self.k_points for value in k):
            raise AssertionError("nonperiodic input has a nonzero k point")
        if not self.nuclei:
            raise AssertionError("fixture has no observed nuclei")
        for atom in self.nuclei:
            atom.validate()
            if atom.pseudized and not self.ecp:
                raise AssertionError("observed ECP nucleus without an ECP Hamiltonian")
        if (self.dtn_cusp is not None) != ("DTN" in self.jastrow_components):
            raise AssertionError("DTN cusp facts differ from observed components")
        if self.dtn_cusp is not None:
            self.dtn_cusp.validate(self.nuclei)
        self._validate_component_boundaries()

    def _validate_component_boundaries(self) -> None:
        expected = {"hamiltonian", "coulomb"}
        for role, signature in zip(("determinant.up", "determinant.down"), self.orbital_signatures, strict=True):
            expected.update(orbital_boundary_roles(role, signature))
            if signature.startswith("gto-") == self.periodic:
                raise AssertionError("determinant orbital family disagrees with boundary")
        expected.update(f"jastrow.{channel}" for channel in ("J1", "J2", "DTN") if channel in self.jastrow_components)
        if "J3" in self.jastrow_components:
            expected.update(orbital_boundary_roles("jastrow.J3", self.j3_orbital_signature))
            if self.j3_orbital_signature.startswith("gto-") == self.periodic:
                raise AssertionError("J3 orbital family disagrees with boundary")
        roles = tuple(item.role for item in self.component_boundaries)
        if len(set(roles)) != len(roles) or set(roles) != expected:
            raise AssertionError("component boundary inventory differs from observed fixture")
        for item in self.component_boundaries:
            item.validate()
        root = next(item for item in self.component_boundaries if item.role == "hamiltonian")
        if root.periodic != self.periodic:
            raise AssertionError("Hamiltonian boundary flag differs from fixture")
        if self.periodic and classify_cell_metric(root.cell) != self.cell_metric:
            raise AssertionError("Hamiltonian cell metric differs from fixture")
        for item in self.component_boundaries:
            if (item.periodic, item.cell) != (root.periodic, root.cell):
                raise AssertionError(f"component boundary differs from Hamiltonian: {item.role}")


def _observed_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise AssertionError(f"{name} must be an integer")
    return int(value)


def _require_same_nuclei(reference, observed, role: str) -> None:
    """Nuclear identity is ordered input equality, not a numerical tolerance test."""
    if tuple(reference.atomic_numbers) != tuple(observed.atomic_numbers) or not np.array_equal(
        np.asarray(reference.positions), np.asarray(observed.positions)
    ):
        raise AssertionError(f"component nuclei differ from Hamiltonian: {role}")


def _observe_nuclear_potentials(ham) -> tuple[NuclearPotentialReceipt, ...]:
    from jqmc.coulomb_potential import Coulomb_potential_data

    coulomb = ham.coulomb_potential_data
    if type(coulomb) is not Coulomb_potential_data or type(coulomb.ecp_flag) not in (bool, np.bool_):
        raise AssertionError("fixture requires concrete Coulomb potential data")
    _require_same_nuclei(ham.structure_data, coulomb.structure_data, "coulomb")
    numbers = tuple(_observed_int(value, "nuclear charge") for value in ham.structure_data.atomic_numbers)
    if np.asarray(ham.structure_data.positions).shape != (len(numbers), 3):
        raise AssertionError("invalid observed nuclear positions")
    if not coulomb.ecp_flag:
        # Unused ECP fields cannot turn an all-electron evaluation into ECP evidence.
        return tuple(NuclearPotentialReceipt(number, 0.0, 0, 0) for number in numbers)
    if len(coulomb.z_cores) != len(numbers) or len(coulomb.max_ang_mom_plus_1) != len(numbers):
        raise AssertionError("ECP nuclear inventory has the wrong length")
    if _observed_int(coulomb.num_ecps, "ECP term count") != len(coulomb.nucleus_index):
        raise AssertionError("ECP term inventory has the wrong length")
    counts = [0] * len(numbers)
    for value in coulomb.nucleus_index:
        index = _observed_int(value, "ECP nucleus index")
        if not 0 <= index < len(numbers):
            raise AssertionError("ECP nucleus index is outside the nuclear inventory")
        counts[index] += 1
    return tuple(
        NuclearPotentialReceipt(number, float(core), _observed_int(channels, "ECP channel count"), count)
        for number, core, channels, count in zip(numbers, coulomb.z_cores, coulomb.max_ang_mom_plus_1, counts, strict=True)
    )


def observe_hamiltonian(ham) -> HamiltonianFixtureReceipt:
    """Read only real production object types; never infer structure from test IDs."""
    from jqmc import atomic_orbital
    from jqmc.determinant import AGP_data, AGPn_data, SD_data
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_DTN_data
    from jqmc.molecular_orbital import MOs_data
    from jqmc.structure import Structure_data

    if type(ham) is not Hamiltonian_data:
        raise AssertionError("fixture observer requires a Hamiltonian_data")
    gem = ham.wavefunction_data.geminal_data
    kinds = {SD_data: "SD", AGPn_data: "AGPn", AGP_data: "AGP"}
    if type(gem) not in kinds:
        raise AssertionError("unknown determinant object")
    k_points = []
    boundaries = []
    leaf_types = {getattr(atomic_orbital, f"AOs_{token}_data"): name for token, name in AO_SIGNATURES.items()}

    def boundary(role, structure):
        if type(structure) is not Structure_data or type(structure.pbc_flag) not in (bool, np.bool_):
            raise AssertionError("fixture component requires a concrete Structure_data boundary")
        vectors = tuple(tuple(float(value) for value in row) for row in (structure.vec_a, structure.vec_b, structure.vec_c))
        # Molecular J2 legitimately has a default, empty Structure_data. Its
        # nuclear positions are unused; its false boundary flag is still real.
        if not structure.pbc_flag and any(vectors):
            raise AssertionError("open component boundary carries cell vectors")
        observed = ComponentBoundaryReceipt(role, bool(structure.pbc_flag), vectors if structure.pbc_flag else ())
        observed.validate()
        boundaries.append(observed)

    boundary("hamiltonian", ham.structure_data)
    boundary("coulomb", ham.coulomb_potential_data.structure_data)

    def orbital(ao, role, *, determinant):
        if type(ao) is atomic_orbital.AOs_composite_data:
            children = tuple(
                orbital(child, f"{role}.{index}", determinant=determinant) for index, child in enumerate(ao.children)
            )
            if children not in COMPOSITE_SIGNATURES:
                raise AssertionError(f"unregistered ordered composite signature: {children}")
            return COMPOSITE_SIGNATURES[children]
        if type(ao) not in leaf_types:
            raise AssertionError(f"unknown AO object: {type(ao).__name__}")
        boundary(role, ao.structure_data)
        k = getattr(ao, "k_point", None)
        vector = np.asarray((0.0, 0.0, 0.0) if k is None else k)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise AssertionError("invalid AO k point")
        if determinant:
            k_points.append(tuple(float(value) for value in vector))
        else:
            # Observe the J3 basis itself, independently of constructor
            # validation. Complex storage does not imply a Bloch twist.
            if np.any(vector != 0):
                raise AssertionError("J3 fixture basis must be phase-free and shared across twists")
            image_phases = getattr(ao, "_image_phase_factors", None)
            if image_phases is not None and np.any(np.asarray(image_phases) != 1):
                raise AssertionError("J3 fixture carries nontrivial Bloch image phases")
        return leaf_types[type(ao)]

    spins = (gem.orb_data_up_spin, gem.orb_data_dn_spin)
    representations = tuple("MO" if type(spin) is MOs_data else "AO" for spin in spins)
    if representations[0] != representations[1]:
        raise AssertionError("mixed AO/MO spin representations are not registered")
    signatures = tuple(
        orbital(spin.aos_data if type(spin) is MOs_data else spin, f"determinant.{role}", determinant=True)
        for role, spin in zip(("up", "down"), spins, strict=True)
    )
    jas = ham.wavefunction_data.jastrow_data
    components = tuple(
        label
        for label, field in (("J1", "one_body"), ("J2", "two_body"), ("J3", "three_body"), ("DTN", "dtn"))
        if getattr(jas, f"jastrow_{field}_data") is not None
    )
    dtn = jas.jastrow_dtn_data
    for channel, field in (("J1", "one_body"), ("J2", "two_body"), ("DTN", "dtn")):
        if channel in components:
            boundary(f"jastrow.{channel}", getattr(jas, f"jastrow_{field}_data").structure_data)
    j3_representation, j3_signature = "off", "not-applicable"
    if "J3" in components:
        j3_orbital = jas.jastrow_three_body_data.orb_data
        j3_representation = "MO" if type(j3_orbital) is MOs_data else "AO"
        j3_signature = orbital(
            j3_orbital.aos_data if type(j3_orbital) is MOs_data else j3_orbital, "jastrow.J3", determinant=False
        )
    structure = ham.structure_data
    dtn_cusp = None
    if dtn is not None:
        if type(dtn) is not Jastrow_DTN_data or type(dtn.apply_en_cusp) not in (bool, np.bool_):
            raise AssertionError("fixture requires concrete DTN cusp data")
        _require_same_nuclei(structure, dtn.structure_data, "DTN")
        dtn_cusp = DTNCuspReceipt(
            constraint_enabled=bool(dtn.apply_en_cusp),
            species_index=tuple(_observed_int(value, "DTN species index") for value in dtn.species_index),
            effective_charges=tuple(float(value) for value in dtn.z_eff_per_species),
        )
    receipt = HamiltonianFixtureReceipt(
        content_snapshot=input_object_snapshot(ham),
        determinant_kind=kinds[type(gem)],
        representation=representations[0],
        orbital_signatures=signatures,
        electron_counts=(int(gem.num_electron_up), int(gem.num_electron_dn)),
        jastrow_components=components,
        j1_form=jas.jastrow_one_body_data.jastrow_1b_type if "J1" in components else "off",
        j2_form=jas.jastrow_two_body_data.jastrow_2b_type if "J2" in components else "off",
        j3_representation=j3_representation,
        j3_orbital_signature=j3_signature,
        dtn_channels=tuple(name for name in ("u", "chi", "f") if dtn is not None and getattr(dtn, f"use_{name}")),
        periodic=bool(structure.pbc_flag),
        cell_metric=classify_cell_metric((structure.vec_a, structure.vec_b, structure.vec_c))
        if structure.pbc_flag
        else "not-applicable",
        k_points=tuple(k_points),
        ecp=bool(ham.coulomb_potential_data.ecp_flag),
        component_boundaries=tuple(boundaries),
        nuclei=_observe_nuclear_potentials(ham),
        dtn_cusp=dtn_cusp,
    )
    receipt.validate()
    return receipt


def validate_matrix_fixture(cell, fixture: HamiltonianFixtureReceipt) -> None:
    """Enforce the matrix's declared axes against measured, jointly realized facts.

    Historical J1/J2 rows both request the coupled J1+J2 recipe. This does not
    certify isolated-channel canonical patterns or count them as two samples.
    """
    fixture.validate()
    ao, jas, ecp, boundary, gem = cell
    expected_jastrow = {
        "none": (),
        "J1": ("J1", "J2"),
        "J2": ("J1", "J2"),
        "J3poly": ("J1", "J2", "J3"),
        "DTN": ("J1", "J2", "DTN"),
    }
    if jas not in expected_jastrow or boundary not in {"mol", "pbc_gamma", "twist"} or ecp not in {"AE", "ECP"}:
        raise AssertionError("unknown matrix fixture axis")
    if gem not in {"SD", "AGPn", "AGP", "AGP_MO"}:
        raise AssertionError("unknown matrix determinant axis")
    expected_kind = "AGP" if gem == "AGP_MO" else gem
    expected_representation = "AO" if gem == "AGP" else "MO"
    if (fixture.determinant_kind, fixture.representation) != (expected_kind, expected_representation):
        raise AssertionError("matrix determinant differs from consumed fixture")
    if ao == "composite":
        # This matrix's composite recipe is specifically ordered pGTO+PW.
        signature = "pgto-plane-wave-complex" if boundary == "twist" else "pgto-plane-wave-real"
    else:
        if ao not in AO_SIGNATURES:
            raise AssertionError("unknown matrix AO axis")
        signature = AO_SIGNATURES[ao]
    if fixture.orbital_signatures != (signature, signature):
        raise AssertionError("matrix orbitals differ from consumed fixture")
    if fixture.jastrow_components != expected_jastrow[jas]:
        raise AssertionError("matrix Jastrow differs from consumed fixture")
    if fixture.ecp != (ecp == "ECP") or fixture.periodic != (boundary != "mol"):
        raise AssertionError("matrix potential/boundary differs from consumed fixture")
    if any(atom.pseudized for atom in fixture.nuclei) != (ecp == "ECP"):
        raise AssertionError("matrix ECP nuclei differ from consumed fixture")
    nonzero_k = tuple(any(value != 0 for value in k) for k in fixture.k_points)
    if any(value != (boundary == "twist") for value in nonzero_k):
        raise AssertionError("matrix Bloch condition differs from consumed fixture")


def validate_matrix_fixture_route(receipt) -> None:
    """Require every XLA matrix call to carry jointly observed input and positions."""
    parts = receipt.obligation_id.split("|")
    if len(parts) != 4 or parts[1] != "matrix" or len(parts[2].split("__")) != 5:
        raise AssertionError("invalid matrix obligation for fixture verification")
    route = receipt.route_capture
    if route is None or not route.fixture_calls:
        raise AssertionError("matrix receipt has no consumed Hamiltonian observations")
    route.validate()
    cell = tuple(parts[2].split("__"))
    counts = dict(
        zip(
            receipt.electron_configuration.ordered_coordinate_snapshots,
            receipt.electron_configuration.electron_counts_per_configuration,
            strict=True,
        )
    )
    for call in route.fixture_calls:
        validate_matrix_fixture(cell, call.fixture)
        if any(counts[s] != call.fixture.electron_counts for s in call.coordinate_snapshots if s in counts):
            raise AssertionError("matrix fixture electron counts differ from position evidence")
    if set(receipt.electron_configuration.ordered_coordinate_snapshots) - set(route.coordinate_snapshots):
        raise AssertionError("matrix fixture did not consume the declared coordinates")
