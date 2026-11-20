"""Small parameter-set Hamiltonians realizing the canonical fixture requirements.

The SSoT supplies requests, never observed facts. Callers must observe the built
objects and bind their own electron coordinates and numerical checks. These are
synthetic neutral atomic cells, not converged electronic-structure solutions.
No coordinates, conditioning screen, skip or fallback lives in this provider.
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

from pathlib import Path

import numpy as np
import yaml


CANONICAL_PATTERN_IDS = (
    "WF-PATTERN-001",
    "WF-PATTERN-002",
    "WF-PATTERN-003",
    "WF-PATTERN-004",
    "WF-PATTERN-005",
    "WF-PATTERN-006",
    "WF-PATTERN-007",
    "WF-PATTERN-008",
    "WF-PATTERN-009",
    "WF-PATTERN-010",
    "WF-PATTERN-011",
    "WF-PATTERN-012",
    "WF-PATTERN-013",
    "WF-PATTERN-014",
    "WF-PATTERN-015",
    "WF-PATTERN-016",
    "WF-PATTERN-017",
)


def canonical_signature(pattern_id: str) -> dict:
    """Read an independently maintained requirement, failing on inventory drift."""
    path = Path(__file__).resolve().parents[1] / "gdd/ssot/fixtures/wavefunctions.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = document["canonical_suite"]["cases"]
    if tuple(row["id"] for row in rows) != CANONICAL_PATTERN_IDS:
        raise AssertionError("canonical provider inventory differs from the SSoT")
    for row in rows:
        if row["id"] == pattern_id:
            return row["signature"]
    raise ValueError(f"unknown canonical pattern: {pattern_id}")


def _cell(metric: str) -> np.ndarray:
    cells = {
        "cubic": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
        "tetragonal": [[6, 0, 0], [0, 6, 0], [0, 0, 8]],
        "orthorhombic": [[6, 0, 0], [0, 7, 0], [0, 0, 8]],
        "monoclinic": [[6, 0, 0], [0, 7, 0], [1.2, 0, 8]],
        "hexagonal": [[6, 0, 0], [-3, 3 * np.sqrt(3), 0], [0, 0, 8]],
        "rhombohedral": [[6, 1, 1], [1, 6, 1], [1, 1, 6]],
        "triclinic": [[6, 0, 0], [1, 7, 0], [0.5, 1.5, 8]],
    }
    return np.asarray(cells[metric], dtype=np.float64)


def _gto(structure, spherical: bool):
    from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data

    common = dict(
        structure_data=structure,
        nucleus_index=(0, 0, 0, 0),
        num_ao=4,
        num_ao_prim=4,
        orbital_indices=(0, 1, 2, 3),
        exponents=np.asarray([0.8, 0.6, 0.6, 0.6]),
        coefficients=np.ones(4),
        angular_momentums=(0, 1, 1, 1),
    )
    if spherical:
        return AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=(0, -1, 0, 1))
    return AOs_cart_gto_data._from_full_basis(
        **common, polynominal_order_x=(0, 1, 0, 0), polynominal_order_y=(0, 0, 1, 0), polynominal_order_z=(0, 0, 0, 1)
    )


def _orbital(signature: str, structure, *, variant: int = 0):
    from jqmc.atomic_orbital import (
        AOs_bspline_complex_data,
        AOs_bspline_data,
        AOs_bspline_tiled_complex_data,
        AOs_composite_data,
        AOs_plane_wave_complex_data,
        AOs_plane_wave_data,
        _apply_image_expansion,
        _apply_image_expansion_complex,
    )
    from jqmc.pw_pgto_projection import bspline_prefilter_periodic

    if signature == "bspline-tiled-complex":
        shape = (8, 8, 8)
        x, y, z = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
        phase = 2 * np.pi
        bands = np.stack(
            (
                1 + 0.2 * np.cos(phase * x),
                np.sin(phase * x) + 0.3 * np.cos(phase * y),
                np.sin(phase * y) + 0.4 * np.cos(phase * z),
                np.sin(phase * z) + 0.25 * np.cos(phase * (x + y)),
            )
        ).astype(np.complex128)
        bands += 0.3j * np.roll(bands, 1, axis=0)
        controls = bspline_prefilter_periodic(bands, 5).reshape(4, -1)
        return AOs_bspline_tiled_complex_data.build(
            structure_data=structure,
            spline_coefficients_by_k=(controls[:2], controls[2:]),
            spline_shape=shape,
            spline_order=5,
            tiling_matrix=(2, 1, 1),
            k_points=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        )

    complex_leaf = signature.endswith("-complex")
    k = (0.25, 0.125, 0.0)
    suffix = "complex" if complex_leaf else "real"
    if signature in (f"pgto-plane-wave-{suffix}", f"pgto-bspline-{suffix}"):
        right = "plane-wave" if "plane-wave" in signature else "bspline"
        return AOs_composite_data.build(
            (
                _orbital(f"pgto-cartesian-{suffix}", structure, variant=variant),
                _orbital(f"{right}-{suffix}", structure, variant=variant),
            )
        )
    if signature in (
        "gto-cartesian-real",
        "gto-spherical-real",
        "pgto-cartesian-real",
        "pgto-spherical-real",
        "pgto-cartesian-complex",
        "pgto-spherical-complex",
    ):
        base = _gto(structure, "spherical" in signature)
        if variant:
            base = base.replace(
                radial=base.radial.replace(coefficients=np.asarray(base.radial.coefficients) * (1 + 0.2 * variant))
            )
        if signature.startswith("gto-"):
            return base
        if complex_leaf:
            return _apply_image_expansion_complex(base, structure, k)
        return _apply_image_expansion(base, structure)
    if signature in ("plane-wave-real", "plane-wave-complex"):
        millers = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        if variant:
            millers[-1] = [1, 1, 1]
        if complex_leaf:
            return AOs_plane_wave_complex_data.build(structure, millers @ structure.recip_cell, k_point=k)
        indices = np.asarray([0, 1, 1, 2, 2, 3, 3])
        return AOs_plane_wave_data.build(
            structure, millers[indices] @ structure.recip_cell, np.asarray([0, 0, 1, 0, 1, 0, 1], dtype=float)
        )
    if signature in ("bspline-real", "bspline-complex"):
        shape = (8, 8, 8)
        x, y, z = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
        phase = 2 * np.pi
        bands = np.stack(
            (
                1 + 0.2 * np.cos(phase * x),
                np.sin(phase * x) + 0.3 * np.cos(phase * y),
                np.sin(phase * y) + 0.4 * np.cos(phase * z),
                np.sin(phase * z) + 0.25 * np.cos(phase * (x + y)),
            )
        )
        if variant:
            bands[1] += 0.2 * variant * np.cos(phase * (y + z))
        if complex_leaf:
            bands = bands + 0.3j * np.roll(bands, 1, axis=0)
        controls = bspline_prefilter_periodic(bands.astype(np.complex128), 5).reshape(4, -1)
        if complex_leaf:
            return AOs_bspline_complex_data.build(structure, controls, shape, 5, k_point=k)
        return AOs_bspline_data.build(structure, controls.real, shape, 5)
    raise ValueError(f"unsupported canonical orbital signature: {signature}")


def _mos(orbital, seed: int):
    from jqmc.atomic_orbital import is_complex_aos
    from jqmc.molecular_orbital import MOs_data

    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(4, orbital.num_ao))
    if is_complex_aos(orbital):
        coefficients = coefficients + 0.3j * rng.normal(size=coefficients.shape)
    # Independent, normalized rows without changing the AO type or its cell.
    coefficients = np.linalg.qr(coefficients.T)[0].T
    return MOs_data.build(aos_data=orbital, mo_coefficients=coefficients)


def build_canonical_hamiltonian(pattern_id: str):
    """Construct one requested pattern; positions and certification belong to callers."""
    from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
    from jqmc.determinant import AGP_data, AGPn_data, SD_data
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import (
        Jastrow_data,
        Jastrow_DTN_data,
        Jastrow_one_body_data,
        Jastrow_three_body_data,
        Jastrow_two_body_data,
    )
    from jqmc.structure import Structure_data
    from jqmc.wavefunction import Wavefunction_data

    signature = canonical_signature(pattern_id)
    det, jas, electrons = (signature[key] for key in ("determinant", "jastrow", "electrons"))
    restricted = electrons["spin_restriction"] == "spin-restricted"
    n_up, n_dn = (2, 2) if electrons["shell_structure"] == "closed-shell" else (3, 1)
    if electrons["spin_restriction"] == "not-applicable":
        n_dn = 0
    ecp = jas["dtn_en_cusp_target"] == "ecp-zero"
    core = 2.0 if ecp else 0.0
    tiled = det["orbitals"]["up"] == "bspline-tiled-complex"
    atomic_number = (n_up + n_dn + int(core)) // 2 if tiled else n_up + n_dn + int(core)
    symbol = {2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C"}[atomic_number]
    periodic = signature["periodicity"] == "periodic"
    cell = _cell(signature["cell_metric_family"]) if periodic else None
    structure = Structure_data.build(
        pbc_flag=periodic,
        positions=np.zeros((1, 3)),
        atomic_numbers=(atomic_number,),
        element_symbols=(symbol,),
        atomic_labels=(symbol + "1",),
        **({"vec_a": cell[0], "vec_b": cell[1], "vec_c": cell[2]} if periodic else {}),
    )
    if tiled:
        from jqmc.structure import tile_structure_data

        structure = tile_structure_data(structure.replace(vec_a=structure.vec_a / 2.0), (2, 1, 1))
    if ecp:
        # Synthetic finite local potential: +Z_eff exp(-a r^2)/r cancels
        # the ionic -Z_eff/r singularity. A nonzero s projector is included.
        coulomb = Coulomb_potential_data(
            structure_data=structure,
            ecp_flag=True,
            z_cores=(core,),
            max_ang_mom_plus_1=(1,),
            num_ecps=3,
            nucleus_index=(0, 0, 0),
            ang_moms=(1, 1, 0),
            exponents=(1.1, 0.6, 0.9),
            coefficients=(float(n_up + n_dn), -0.35, 0.2),
            powers=(1, 2, 2),
        )
    else:
        coulomb = Coulomb_potential_data(structure_data=structure)
    if periodic:
        coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, n_electron=n_up + n_dn))

    up = _orbital(det["orbitals"]["up"], structure)
    down_signature = det["orbitals"]["down"]
    if restricted or down_signature == "not-applicable":
        down = up.with_negated_k() if hasattr(up, "with_negated_k") else up
    else:
        # Independent down basis, at the opposite twist like every down channel.
        down = _orbital(down_signature, structure, variant=1)
        down = down.with_negated_k() if hasattr(down, "with_negated_k") else down
    if det["representation"] == "MO":
        up = _mos(up, 321)
        down = up.replace(aos_data=down, mo_coefficients=np.conj(up.mo_coefficients)) if restricted else _mos(down, 654)
    if det["kind"] == "SD":
        geminal = SD_data.build(num_electron_up=n_up, num_electron_dn=n_dn, mos_up=up, mos_dn=down, time_reversal=restricted)
    elif det["kind"] == "AGPn":
        geminal = AGPn_data.build(
            num_electron_up=n_up,
            num_electron_dn=n_dn,
            mos_up=up,
            mos_dn=down,
            lambda_diag=np.asarray([1.0, 0.8, 0.3, 0.15]),
            time_reversal=restricted,
        )
    elif det["kind"] == "AGP":
        rank = up.num_mo if det["representation"] == "MO" else up.num_ao
        rng = np.random.default_rng(987)
        matrix = np.eye(rank) + 0.03 * rng.normal(size=(rank, rank))
        matrix = 0.5 * (matrix + matrix.T)
        if "complex" in det["orbitals"]["up"]:
            skew = rng.normal(size=(rank, rank))
            matrix = matrix + 0.02j * (skew - skew.T)
        unpaired = 0.3 * rng.normal(size=(rank, n_up - n_dn))
        geminal = AGP_data.build(
            num_electron_up=n_up,
            num_electron_dn=n_dn,
            orb_data_up_spin=up,
            orb_data_dn_spin=down,
            lambda_matrix=np.column_stack((matrix, unpaired)),
            time_reversal=restricted,
        )
    else:
        raise ValueError(f"unsupported canonical determinant: {det['kind']}")

    j1 = (
        None
        if jas["j1_form"] == "off"
        else Jastrow_one_body_data.build(
            jastrow_1b_param=0.7,
            jastrow_1b_type=jas["j1_form"],
            structure_data=structure,
            core_electrons=(core,),
        )
    )
    j2 = (
        None
        if jas["j2_form"] == "off"
        else Jastrow_two_body_data.build(
            jastrow_2b_param=0.8,
            jastrow_2b_type=jas["j2_form"],
            structure_data=structure,
        )
    )
    j3 = None
    if jas["j3_representation"] != "off":
        from jqmc.jastrow_factor import _phase_free_j3_orbitals

        j3_orbital = _orbital(jas["j3_orbital_signature"], structure)
        if jas["j3_representation"] == "MO":
            j3_orbital = _mos(j3_orbital, 963)
        # Complex storage is a kernel-coverage axis, not permission to
        # attach the determinant's Bloch twist to the common Jastrow basis.
        j3_orbital = _phase_free_j3_orbitals(j3_orbital)
        j3 = Jastrow_three_body_data.build(orb_data=j3_orbital, random_init=True, random_scale=0.03, seed=741)
    dtn = None
    if jas["dtn_channel_mask"] != "off":
        channels = jas["dtn_channel_mask"].split("_")
        dtn = Jastrow_DTN_data.build(
            structure_data=structure,
            z_eff_per_species=(float(n_up + n_dn),),
            pseudized_per_species=(ecp,),
            use_u="u" in channels,
            use_chi="chi" in channels,
            use_f="f" in channels,
            n_order_u=2,
            n_order_chi=2,
            n_order_eN=2,
            n_order_ee=2,
            random_init=True,
            seed=852,
        )
    wavefunction = Wavefunction_data(
        geminal_data=geminal,
        jastrow_data=Jastrow_data(
            jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=dtn
        ),
    )
    return Hamiltonian_data.build(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)
