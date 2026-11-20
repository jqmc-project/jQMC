"""jQMC tools.

Precision Zones:
    This module owns no Precision Zone.

See :mod:`jqmc._precision` for details.
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

import contextlib
import hashlib
import itertools
import math
import os
import re
import sys
from enum import Enum
from logging import Formatter, StreamHandler, getLogger
from typing import Any

import click
import h5py
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tomlkit
import typer
from uncertainties import ufloat

from ._build_info import build_info
from ._checkpoint import (
    compute_retained_accumulated_weights,
    load_driver_config_from_checkpoint,
    load_hamiltonian_from_checkpoint,
    load_observables_from_checkpoint,
    open_observable_datasets,
)
from ._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    PGTO_EXP_TAIL_EPS_USER_DEFAULT,
    Bohr_to_Angstrom,
)
from ._twist_checkpoint import load_twist_checkpoint_manifest, load_twist_checkpoint_summary
from .atomic_orbital import (
    AOs_base_data,
    AOs_cart_gto_data,
    AOs_composite_data,
    AOs_GTO_base_data,
    AOs_pGTO_base_data,
    AOs_sphe_gto_data,
    _apply_image_expansion_complex,
)
from .coulomb_potential import Coulomb_potential_data
from .determinant import Geminal_data, to_AGPn
from .hamiltonians import Hamiltonian_data, _detile_aos_data_for_save
from .jastrow_factor import (
    DTN_DEFAULT_CUTOFF_RATIO,
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _phase_free_j3_orbitals,
)
from .jqmc_miscs import GENERATED_INPUT_OMITTED_KEYS, cli_parameters
from .molecular_orbital import MOs_data
from .output_parser import parse_vmc_steps
from .structure import Structure_data
from .trexio_wrapper import convert_trexio_to_twist_average_input, read_trexio_file
from .wavefunction import Wavefunction_data

log = getLogger("jqmc")
# jqmc-tool is a one-shot diagnostic CLI: default to INFO so that
# per-command progress and projection diagnostics (mean / max residual,
# cond(S), dropped count, ...) are visible without an extra flag.
# Override per command with --verbosity.
log.setLevel("INFO")
stream_handler = StreamHandler(sys.stdout)
stream_handler.setLevel("INFO")
handler_format = Formatter("%(message)s")
stream_handler.setFormatter(handler_format)
log.addHandler(stream_handler)


def _show_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print the version lines and exit (``jqmc-tool --version``)."""
    if not value or ctx.resilient_parsing:
        return
    click.echo("\n".join(build_info().banner_lines()))
    ctx.exit()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_show_version,
    help="Show the jQMC version and git commit, then exit.",
)
def _cli() -> None:
    """The jQMC tools."""
    # Every subcommand run reports where its code comes from. stderr keeps the
    # commands' stdout (parsed by jqmc_workflow) unchanged.
    click.echo("\n".join(build_info().banner_lines()), err=True)


# trexio_app
trexio_app = typer.Typer(help="Read and Convert TREXIO files.")


@trexio_app.command("show-info")
def trexio_show_info(
    filename: str = typer.Argument(..., help="TREXIO file name."),
) -> None:
    """Show information stored in the TREXIO file."""
    (structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data) = read_trexio_file(filename)

    for line in structure_data._get_info():
        typer.echo(line)
    for line in aos_data._get_info():
        typer.echo(line)
    for line in mos_data_up._get_info():
        typer.echo(line)
    for line in mos_data_dn._get_info():
        typer.echo(line)
    for line in geminal_data._get_info():
        typer.echo(line)
    for line in coulomb_potential_data._get_info():
        typer.echo(line)


@trexio_app.command("show-detail")
def trexio_show_detail(
    filename: str = typer.Argument(..., help="TREXIO file name."),
) -> None:
    """Show information stored in the TREXIO file."""
    (structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data) = read_trexio_file(filename)

    typer.echo(structure_data)
    typer.echo(aos_data)
    typer.echo(mos_data_up)
    typer.echo(mos_data_dn)
    typer.echo(geminal_data)
    typer.echo(coulomb_potential_data)


class orbital_type(str, Enum):
    """Orbital type."""

    ao = "ao"
    mo = "mo"
    ao_full = "ao-full"
    ao_small = "ao-small"
    ao_medium = "ao-medium"
    ao_large = "ao-large"
    none = "none"


_J3_AO_SELECTION_TABLE = {
    "period1": {
        "ao-small": "3s",
        "ao-medium": "3s1p",
        "ao-large": "4s2p1d",
    },
    "period2": {
        "ao-small": "3s1p",
        "ao-medium": "4s2p1d",
        "ao-large": "5s3p2d1f",
    },
    "period3": {
        "ao-small": "4s2p",
        "ao-medium": "5s3p1d",
        "ao-large": "6s4p3d1f",
    },
    "period4plus_main": {
        "ao-small": "4s2p1d",
        "ao-medium": "5s3p2d1f",
        "ao-large": "6s4p3d2f",
    },
    "period4plus_transition": {
        "ao-small": "4s2p2d1f",
        "ao-medium": "5s3p3d2f1g",
        "ao-large": "6s4p4d3f2g1h",
    },
}

_J3_TRANSITION_Z_RANGES = (
    (21, 30),  # 3d transition
    (39, 48),  # 4d transition
    (57, 71),  # lanthanides
    (72, 80),  # 5d transition (excluding La)
    (89, 103),  # actinides
    (104, 112),  # 6d transition
)

_J3_PERIOD_RANGES = (
    (1, 2),
    (3, 10),
    (11, 18),
    (19, 36),
    (37, 54),
    (55, 86),
    (87, 118),
)

_J3_SHELL_L_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


# ---------------------------------------------------------------------------
# TREXIO conversion uses both builders below. QE conversion reuses the J3
# builder, but owns its DTN builder because the library entry point
# (jqmc.jqmc_qe.run_qe2jqmc) must not raise click-specific exceptions.
# ---------------------------------------------------------------------------


_DTN_PRESET_MAP: dict[str, tuple[bool, bool, bool]] = {
    "u": (True, False, False),
    "u_chi": (True, True, False),
    "u_f": (True, False, True),
    "u_chi_f": (True, True, True),
}


def _build_jastrow_dtn_from_cli(
    structure_data: Structure_data,
    *,
    jastrow_dtn: str | None,
    coulomb_potential_data: Coulomb_potential_data,
    dtn_c: int = 4,
    dtn_n_u: int = 8,
    dtn_n_chi: int = 8,
    dtn_n_eN: int = 4,
    dtn_n_ee: int = 4,
    dtn_L_u: float | None = None,
    dtn_L_chi: float | None = None,
    dtn_L_f: float | None = None,
    dtn_cutoff_abs_cap: float = 8.0,
    dtn_cutoff_ratio_u: float | None = None,
    dtn_cutoff_ratio_chi: float | None = None,
    dtn_cutoff_ratio_f: float | None = None,
) -> Jastrow_DTN_data | None:
    """Build a :class:`Jastrow_DTN_data` from the CLI flags, or return ``None``.

    Effective charge and ECP identity both come from
    ``coulomb_potential_data``.  This distinction is required for zero-core
    pseudopotentials, whose effective charge equals the atomic number.
    """
    if jastrow_dtn is None:
        return None
    choice = str(jastrow_dtn).lower()
    if choice == "none":
        return None
    if choice not in _DTN_PRESET_MAP:
        raise click.ClickException(f"--jastrow-dtn={jastrow_dtn!r} is not recognised; allowed: none, u, u_chi, u_f, u_chi_f.")
    use_u, use_chi, use_f = _DTN_PRESET_MAP[choice]
    if not structure_data.pbc_flag:
        # Molecular cutoffs are lengths; L <= 0 would silently drop the channel and its cusp (issue #589).
        for option, value in (
            ("--dtn-Lu", dtn_L_u),
            ("--dtn-Lchi", dtn_L_chi),
            ("--dtn-Lf", dtn_L_f),
            ("--dtn-cutoff-abs-cap", dtn_cutoff_abs_cap),
        ):
            if value is not None and not (np.isfinite(float(value)) and float(value) > 0.0):
                raise click.ClickException(f"{option} = {value} must be a finite cutoff length > 0 (Bohr).")

    atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.int64)
    unique_atomic_numbers = np.unique(atomic_numbers)
    coulomb_atomic_numbers = np.asarray(coulomb_potential_data.structure_data.atomic_numbers, dtype=np.int64)
    if not np.array_equal(coulomb_atomic_numbers, atomic_numbers):
        raise click.ClickException("DTN and Coulomb data must describe the same structure.")
    effective_charge_per_atom = np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)
    pseudized_per_atom = np.asarray(coulomb_potential_data.pseudized_per_atom, dtype=np.bool_)
    if effective_charge_per_atom.shape != atomic_numbers.shape or pseudized_per_atom.shape != atomic_numbers.shape:
        raise click.ClickException("DTN and Coulomb data must describe the same number of atoms.")
    effective_charge_per_species: list[float] = []
    pseudized_per_species: list[bool] = []
    for atomic_number in unique_atomic_numbers:
        atoms_of_species = atomic_numbers == atomic_number
        species_effective_charges = np.unique(effective_charge_per_atom[atoms_of_species])
        species_pseudized_flags = np.unique(pseudized_per_atom[atoms_of_species])
        if species_effective_charges.size != 1 or species_pseudized_flags.size != 1:
            raise click.ClickException(
                f"Atoms of element Z={int(atomic_number)} do not share one ECP/effective-charge policy; "
                "assign distinct DTN species before building the Hamiltonian."
            )
        effective_charge_per_species.append(float(species_effective_charges[0]))
        pseudized_per_species.append(bool(species_pseudized_flags[0]))

    return Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=tuple(effective_charge_per_species),
        pseudized_per_species=tuple(pseudized_per_species),
        use_u=use_u,
        use_chi=use_chi,
        use_f=use_f,
        cutoff_smoothness=int(dtn_c),
        n_order_u=int(dtn_n_u),
        n_order_chi=int(dtn_n_chi),
        n_order_eN=int(dtn_n_eN),
        n_order_ee=int(dtn_n_ee),
        cutoff_abs_cap=float(dtn_cutoff_abs_cap),
        cutoff_ratio_u=dtn_cutoff_ratio_u,
        cutoff_ratio_chi=dtn_cutoff_ratio_chi,
        cutoff_ratio_f=dtn_cutoff_ratio_f,
        L_u=None if dtn_L_u is None else float(dtn_L_u),
        L_chi=None if dtn_L_chi is None else float(dtn_L_chi),
        L_f=None if dtn_L_f is None else float(dtn_L_f),
    )


def _resolve_j3_gto_leaf(aos_data: AOs_base_data) -> AOs_GTO_base_data:
    """Resolve the GTO/pGTO leaf used to build an ``ao*`` J3 basis.

    The ``pgto+pw`` / ``pgto+bspline`` projections produce an
    :class:`AOs_composite_data`; the polynomial J3 expansion needs a single
    GTO/pGTO leaf to size and shell-select from. Extract it, with explicit
    ``NotImplementedError`` on the cases that are not meaningfully supported:

    * composite with no GTO/pGTO child (plane-wave / B-spline only) -> error;
    * composite with more than one GTO/pGTO child                   -> error;
    * a bare GTO/pGTO leaf                                          -> returned as-is;
    * a bare plane-wave / B-spline leaf                            -> error (no shells).

    For the supported single-leaf cases the caller applies the usual
    ao-small / ao-medium / ao-large / ao-full shell-selection criterion.
    """
    if isinstance(aos_data, AOs_composite_data):
        gto_children = [c for c in aos_data.children if isinstance(c, AOs_GTO_base_data)]
        if len(gto_children) == 0:
            raise NotImplementedError(
                "J3 ao-basis on a composite AO container with no GTO/pGTO child "
                "(plane-wave / B-spline only) is not supported; use j3_basis_type='mo'."
            )
        if len(gto_children) > 1:
            raise NotImplementedError(
                "J3 ao-basis on a composite AO container with more than one GTO/pGTO child is not supported."
            )
        return gto_children[0]
    if isinstance(aos_data, AOs_GTO_base_data):
        return aos_data
    raise NotImplementedError(
        f"J3 ao-basis is not supported for AO container {type(aos_data).__name__} "
        "(plane-wave / B-spline have no GTO shells); use j3_basis_type='mo'."
    )


def resolve_twist_average_j3_choice(j3_choice):
    """Return the AO J3 choice shared by every untiled twist, or ``None``.

    A combined twist input stores one phase-free J3 in ``/shared``. An AO basis
    is k-independent, so one ordinary periodic copy serves every twist. MO J3
    orbitals are the Bloch MOs of one twist and cannot be one shared object, so
    ``mo`` is rejected before any source I/O.
    """
    value = "none" if j3_choice is None else str(getattr(j3_choice, "value", j3_choice)).lower()
    if value == "none":
        return None
    if value == "mo":
        raise ValueError(
            "Untiled twist averaging does not support an MO-based J3 (-j3 mo / j3_basis_type='mo'): its orbitals "
            "are the Bloch MOs of one twist, so they are twist-dependent, while every twist must share one "
            "phase-free J3 basis. Use ao, ao-small, ao-medium, ao-large or ao-full."
        )
    if value not in {"ao", "ao-full", "ao-small", "ao-medium", "ao-large"}:
        raise ValueError(f"Invalid J3 basis type {value!r}; use none, ao, ao-small, ao-medium, ao-large or ao-full.")
    return value


def build_supercell_jastrow_three_body(structure_data, shells_per_atom, j3_choice="ao-small"):
    """Build an independent phase-free periodic AO J3 on every physical atom.

    Shells describe the physical cell (the supercell when tiled), not the
    determinant's primitive blocks or Bloch orbitals. Select the uncontracted
    home basis before expanding periodic images. The full symmetric J matrix is unconstrained across atoms/cells and starts
    at zero, as for the ordinary converter. No determinant coefficients enter.
    """
    from .pw_pgto_projection import build_pgto_primitives
    from ._setting import AO_IMAGE_TRUNCATION_ATOL

    if not structure_data.pbc_flag:
        raise ValueError("supercell J3 requires a periodic physical structure")
    if j3_choice not in {"ao", "ao-full", "ao-small", "ao-medium", "ao-large"}:
        raise ValueError("tiled conversion requires an independent J3 AO basis (ao-small/medium/large/full), not Slater MOs")
    prims = []
    nuclei = []
    if len(shells_per_atom) != structure_data.natom:
        raise ValueError("J3 requires shells for every physical-supercell atom")
    for atom, (shells, position) in enumerate(zip(shells_per_atom, structure_data.positions, strict=True)):
        atom_prims = build_pgto_primitives([shells], np.asarray([position]))
        prims.extend(atom_prims)
        nuclei.extend([atom] * len(atom_prims))
    if not prims:
        raise ValueError("J3 requires a nonempty independent AO basis")
    home = AOs_cart_gto_data._from_full_basis(
        structure_data=structure_data,
        nucleus_index=tuple(nuclei),
        num_ao=len(prims),
        num_ao_prim=len(prims),
        orbital_indices=tuple(range(len(prims))),
        exponents=np.asarray([p.exponent for p in prims]),
        coefficients=np.ones(len(prims)),
        angular_momentums=tuple(sum(p.ang) for p in prims),
        polynominal_order_x=tuple(p.ang[0] for p in prims),
        polynominal_order_y=tuple(p.ang[1] for p in prims),
        polynominal_order_z=tuple(p.ang[2] for p in prims),
    )
    j3 = build_jastrow_three_body_for_basis(home, None, j3_choice)
    periodic = _apply_image_expansion_complex(j3.orb_data, structure_data, k_point=None, tol=AO_IMAGE_TRUNCATION_ATOL)
    result = j3.replace(orb_data=periodic)
    result.sanity_check()
    return result


def build_jastrow_three_body_for_basis(
    aos_data: AOs_base_data,
    mos_data: MOs_data,
    j3_choice: str | None,
) -> Jastrow_three_body_data | None:
    """Build a ``Jastrow_three_body_data`` from an AO/MO container + choice.

    Shared by the TREXIO (``trexio convert-to``) and QE (``run_qe2jqmc``) paths
    so both expose an identical J3 surface. ``j3_choice`` is the already-resolved
    lower-case string (``None`` / ``"none"`` -> returns ``None``). For the
    ``ao*`` choices, a composite container must contain exactly one GTO/pGTO
    leaf. That leaf is uncontracted and filtered with the ao-small /
    ao-medium / ao-large / ao-full shell-selection criterion.
    Every J3 basis is phase-free. For MO-based J3 the fixed reference MO
    coefficients are retained, not recomputed or projected to real numbers;
    combined twist inputs must share this one resulting J3 object.
    """

    if j3_choice is not None:
        if j3_choice in {"ao", "ao-full", "ao-small", "ao-medium", "ao-large"}:
            aos_data = _resolve_j3_gto_leaf(aos_data)
            aos_data = aos_data._build_uncontracted_aos()
            # PBC pgto leaves carry tiled per-primitive arrays; the J3 trim
            # logic below was written for the molecular convention where
            # ``num_ao`` matches the per-array length. De-tile to a home-only
            # gto sibling for the duration of the selection, then re-tile the
            # filtered result back to a pgto so the saved Hamiltonian h5 has
            # ``_n_image`` / ``_image_truncation_tol`` consistent with the
            # home-cell basis. J3 deliberately drops the determinant twist.
            _was_pgto = isinstance(aos_data, AOs_pGTO_base_data)
            if _was_pgto:
                _pgto_tol = float(aos_data._image_truncation_tol)
                aos_data = _detile_aos_data_for_save(aos_data)
            else:
                _pgto_tol = None
            exponent_by_ao_index = [None] * aos_data.num_ao
            for primitive_index, ao_index in enumerate(aos_data.orbital_indices):
                if exponent_by_ao_index[ao_index] is None:
                    exponent_by_ao_index[ao_index] = aos_data.exponents[primitive_index]

            selected_ao_indices_across_nuclei = []

            # Resolve every AO nucleus index against the home-cell atomic
            # numbers. Molecular and de-tiled periodic data are already
            # home-cell indexed, so the modulo is a no-op for the supported
            # inputs while keeping the lookup valid for image-expanded data.
            num_atoms_home = len(aos_data.structure_data.atomic_numbers)

            for nucleus_index in set(aos_data.nucleus_index):
                nucleus_ao_indices = [
                    ao_index
                    for ao_index, ao_nucleus_index in enumerate(aos_data.nucleus_index)
                    if ao_nucleus_index == nucleus_index
                ]
                if not nucleus_ao_indices:
                    continue

                home_nucleus_index = nucleus_index % num_atoms_home
                atomic_number = aos_data.structure_data.atomic_numbers[home_nucleus_index]
                if j3_choice in ("ao", "ao-full"):
                    shell_counts = None
                else:
                    period = 7
                    for period_index, (z_min, z_max) in enumerate(_J3_PERIOD_RANGES, start=1):
                        if z_min <= atomic_number <= z_max:
                            period = period_index
                            break
                    if period == 1:
                        selection_key = "period1"
                    elif period == 2:
                        selection_key = "period2"
                    elif period == 3:
                        selection_key = "period3"
                    else:
                        is_transition_or_lanthanide = any(
                            z_min <= atomic_number <= z_max for z_min, z_max in _J3_TRANSITION_Z_RANGES
                        )
                        selection_key = "period4plus_transition" if is_transition_or_lanthanide else "period4plus_main"
                    shell_spec = _J3_AO_SELECTION_TABLE[selection_key][j3_choice]
                    shell_counts = {}
                    for shell_count, shell_name in re.findall(r"(\d+)([spdfgh])", shell_spec):
                        angular_momentum = _J3_SHELL_L_MAP[shell_name]
                        shell_counts[angular_momentum] = int(shell_count)
                if shell_counts is None:
                    selected_ao_indices_across_nuclei.extend(nucleus_ao_indices)
                    continue

                # Select the requested radial shells independently for each
                # angular momentum on this nucleus.
                for angular_momentum in set(aos_data.angular_momentums):
                    angular_momentum_ao_indices = [
                        ao_index for ao_index in nucleus_ao_indices if aos_data.angular_momentums[ao_index] == angular_momentum
                    ]
                    if not angular_momentum_ao_indices:
                        continue

                    number_of_shells = shell_counts.get(angular_momentum, 0)
                    if number_of_shells <= 0:
                        continue

                    unique_basis_exponents = sorted(
                        {float(exponent_by_ao_index[ao_index]) for ao_index in angular_momentum_ao_indices}
                    )
                    if number_of_shells >= len(unique_basis_exponents):
                        selected_basis_exponents = unique_basis_exponents
                    else:
                        # Center small selections near the median of the
                        # log-exponent distribution. Larger selections widen
                        # the window across the exponent range; the 2.5 divisor
                        # maps a 2-of-10 selection to about 32%-68% of the CDF.
                        log_exponents = np.log(unique_basis_exponents)
                        selected_fraction = number_of_shells / len(unique_basis_exponents)
                        margin = (1.0 - selected_fraction) / 2.5
                        cumulative_coordinates = np.linspace(0, 1, len(log_exponents))
                        lower_log_exponent = np.interp(margin, cumulative_coordinates, log_exponents)
                        upper_log_exponent = np.interp(1.0 - margin, cumulative_coordinates, log_exponents)
                        target_log_exponents = np.linspace(lower_log_exponent, upper_log_exponent, number_of_shells)
                        selected_basis_exponents = []
                        remaining_exponents = list(unique_basis_exponents)
                        remaining_log_exponents = list(log_exponents)
                        for target_log_exponent in target_log_exponents:
                            nearest_index = int(
                                np.argmin([abs(log_exponent - target_log_exponent) for log_exponent in remaining_log_exponents])
                            )
                            selected_basis_exponents.append(remaining_exponents.pop(nearest_index))
                            remaining_log_exponents.pop(nearest_index)
                        selected_basis_exponents.sort()

                    selected_ao_indices_across_nuclei.extend(
                        ao_index
                        for ao_index in angular_momentum_ao_indices
                        if float(exponent_by_ao_index[ao_index]) in selected_basis_exponents
                    )

            selected_ao_indices = sorted(set(selected_ao_indices_across_nuclei))

            new_idx_map = {old: new for new, old in enumerate(selected_ao_indices)}

            # Keep the primitives belonging to selected AOs, then remap their
            # orbital indices to the compact output basis.
            new_prims = [p for p, orb in enumerate(aos_data.orbital_indices) if orb in new_idx_map]

            new_orbital_indices = [new_idx_map[aos_data.orbital_indices[p]] for p in new_prims]
            dtype_io_jnp = jnp.float64
            new_exponents = jnp.array([float(aos_data.exponents[p]) for p in new_prims], dtype=dtype_io_jnp)
            new_coefficients = jnp.array([float(aos_data.coefficients[p]) for p in new_prims], dtype=dtype_io_jnp)
            new_nucleus_index = [aos_data.nucleus_index[i] for i in selected_ao_indices]
            new_angular_momentums = [aos_data.angular_momentums[i] for i in selected_ao_indices]

            if isinstance(aos_data, AOs_cart_gto_data):
                new_polynominal_order_x = [aos_data.polynominal_order_x[i] for i in selected_ao_indices]
                new_polynominal_order_y = [aos_data.polynominal_order_y[i] for i in selected_ao_indices]
                new_polynominal_order_z = [aos_data.polynominal_order_z[i] for i in selected_ao_indices]
            elif isinstance(aos_data, AOs_sphe_gto_data):
                new_magnetic_quantum_numbers = [aos_data.magnetic_quantum_numbers[i] for i in selected_ao_indices]
            else:
                raise ImportError(f"Invalid AOs data type: {type(aos_data)}")

            common_kwargs = {
                "structure_data": aos_data.structure_data,
                "nucleus_index": new_nucleus_index,
                "num_ao": len(selected_ao_indices),
                "num_ao_prim": len(new_orbital_indices),
                "orbital_indices": new_orbital_indices,
                "exponents": new_exponents,
                "coefficients": new_coefficients,
                "angular_momentums": new_angular_momentums,
            }

            if isinstance(aos_data, AOs_cart_gto_data):
                common_kwargs.update(
                    {
                        "polynominal_order_x": new_polynominal_order_x,
                        "polynominal_order_y": new_polynominal_order_y,
                        "polynominal_order_z": new_polynominal_order_z,
                    }
                )
            else:
                common_kwargs["magnetic_quantum_numbers"] = new_magnetic_quantum_numbers

            # Exponents and coefficients belong to the per-shell radial leaf,
            # so rebuild through the full-basis factory instead of calling the
            # dataclass constructor directly.
            aos_data = type(aos_data)._from_full_basis(**common_kwargs)

            # Re-tile the trimmed home gto back to a pgto leaf when the
            # input was periodic, so the J3 ``orb_data`` saved into the
            # Hamiltonian h5 carries ``_n_image`` / ``_image_truncation_tol``
            # consistent with the phase-free home-cell basis. Without
            # this step the load-time ``_apply_image_expansion_post_load``
            # would feed ``tol=0`` into ``_primitive_image_radius`` and
            # crash with ``OverflowError``.
            if _was_pgto:
                aos_data = _apply_image_expansion_complex(aos_data, aos_data.structure_data, k_point=None, tol=_pgto_tol)

            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)
        elif j3_choice == "mo":
            jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
                orb_data=_phase_free_j3_orbitals(mos_data)
            )
        else:
            raise ImportError(f"Invalid j3_basis_type = {j3_choice}.")
    else:
        jastrow_threebody_data = None
    return jastrow_threebody_data


def _parse_tiling_selection(tiling, supertwist, twist_average):
    """Validate the common quoted tiling/sector CLI options before source I/O."""
    from .structure import normalize_tiling_matrix

    def value(option):
        return option.default if isinstance(option, typer.models.OptionInfo) else option

    try:
        matrix = [int(token) for entry in (value(tiling) or []) for token in str(entry).split()]
        if matrix:
            normalize_tiling_matrix(matrix)
        elif value(tiling):
            raise ValueError("empty tiling matrix")
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--tiling") from exc
    twist = value(supertwist)
    if twist is not None:
        try:
            twist = [float(token) for token in twist.split()]
            if len(twist) != 3 or not np.isfinite(twist).all():
                raise ValueError("supertwist must contain three finite fractional coordinates")
            if not matrix or value(twist_average):
                raise ValueError("supertwist requires tiling and cannot be combined with twist-average")
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--supertwist") from exc
    return matrix, twist


@trexio_app.command("convert-to")
def trexio_convert_to(
    trexio_file: str = typer.Argument(..., help="TREXIO filename."),
    hamiltonian_file: str = typer.Option("hamiltonian_data.h5", "-o", "--output", help="Output file name."),
    j1_parmeter: float = typer.Option(None, "-j1", "--jastrow-1b-parameter", help="Jastrow one-body parameter."),
    j1_type: str = typer.Option("exp", "--jastrow-1b-type", help="Jastrow one-body functional form: 'exp' or 'pade'."),
    j2_parmeter: float = typer.Option(None, "-j2", "--jastrow-2b-parameter", help="Jastrow two-body parameter."),
    j2_type: str = typer.Option("pade", "--jastrow-2b-type", help="Jastrow two-body functional form: 'pade' or 'exp'."),
    j3_basis_type: orbital_type = typer.Option(
        orbital_type.none,
        "-j3",
        "--jastrow-3b-basis-set-type",
        help="Jastrow three-body basis-set type (use 'none' to disable atomic/molecular-orbital-based J3 term).",
    ),
    jastrow_dtn: str = typer.Option(
        "none",
        "--jastrow-dtn",
        help=(
            "DTN (Drummond-Towler-Needs) polynomial Jastrow preset; selects "
            "which of u (e-e), chi (e-n), f (e-e-n) channels are active. "
            "Allowed values: 'none' (default), 'u', 'u_chi', 'u_f', 'u_chi_f'. "
            "When != none, the DTN family is mutually exclusive with the Pade "
            "j1 / j2 / AO j3 family."
        ),
    ),
    dtn_c: int = typer.Option(4, "--dtn-C", help="DTN cutoff polynomial smoothness order C (>= 3)."),
    dtn_n_u: int = typer.Option(8, "--dtn-Nu", help="DTN u-term polynomial order N_u."),
    dtn_n_chi: int = typer.Option(8, "--dtn-Nchi", help="DTN chi-term polynomial order N_chi."),
    dtn_n_eN: int = typer.Option(4, "--dtn-NeN", help="DTN f-term electron-ion polynomial order N_eN."),
    dtn_n_ee: int = typer.Option(4, "--dtn-Nee", help="DTN f-term electron-electron polynomial order N_ee."),
    dtn_L_u: float = typer.Option(
        None,
        "--dtn-Lu",
        help=(
            "DTN u-term cutoff (Bohr). Omit for the geometry-derived default: a fraction of the "
            "Wigner-Seitz inscribed radius under PBC, or min(R_max + 10 Bohr, cap) for a molecule."
        ),
    ),
    dtn_L_chi: float = typer.Option(
        None,
        "--dtn-Lchi",
        help="DTN chi-term cutoff (Bohr, broadcast across species). Omit for auto.",
    ),
    dtn_L_f: float = typer.Option(
        None,
        "--dtn-Lf",
        help="DTN f-term cutoff (Bohr, broadcast across species). Omit for auto.",
    ),
    dtn_cutoff_ratio_u: float = typer.Option(
        None, "--dtn-cutoff-ratio-u", help=f"PBC u cutoff ratio relative to R_ins (default {DTN_DEFAULT_CUTOFF_RATIO})."
    ),
    dtn_cutoff_ratio_chi: float = typer.Option(
        None, "--dtn-cutoff-ratio-chi", help=f"PBC chi cutoff ratio relative to R_ins (default {DTN_DEFAULT_CUTOFF_RATIO})."
    ),
    dtn_cutoff_ratio_f: float = typer.Option(
        None, "--dtn-cutoff-ratio-f", help=f"PBC f cutoff ratio relative to R_ins/2 (default {DTN_DEFAULT_CUTOFF_RATIO})."
    ),
    dtn_cutoff_abs_cap: float = typer.Option(
        8.0,
        "--dtn-cutoff-abs-cap",
        help="Auto-cutoff absolute cap (Bohr).",
    ),
    ao_conv_to: str = typer.Option(
        None,
        "--ao-conv-to",
        help=(
            "Convert AOs after building the Hamiltonian: 'cart' (Cartesian) or 'sphe' (spherical). Default: keep original. "
            "'sphe' projects Cartesian MOs onto the spherical subspace, discards the rest (e.g. the r^2 trace "
            "of d shells) and warns with the discarded relative S-norm when it is not negligible."
        ),
    ),
    twist_average: bool = typer.Option(
        False,
        "--twist-average",
        help="Convert every k-point of a multi-k TREXIO file into one combined twist-average HDF5 file.",
    ),
    tiling: list[str] = typer.Option(
        None, "--tiling", help='Primitive k blocks in a supercell: one quoted matrix, e.g. "3 3 3" or nine row-major integers.'
    ),
    supertwist: str = typer.Option(
        None, "--supertwist", help='Select a fractional supercell twist, e.g. "0.2 0.1 0"; otherwise infer the single sector.'
    ),
) -> None:
    """Convert a TREXIO file to hamiltonian_data."""

    # Allow direct string inputs when trexio_convert_to is called programmatically (e.g., in tests)
    if isinstance(j3_basis_type, str):
        try:
            j3_basis_type = orbital_type(j3_basis_type)
        except ValueError:
            # Leave as-is; downstream validation will raise a clearer error message.
            pass

    if isinstance(j1_type, typer.models.OptionInfo):
        j1_type = j1_type.default
    if isinstance(j2_type, typer.models.OptionInfo):
        j2_type = j2_type.default
    if isinstance(ao_conv_to, typer.models.OptionInfo):
        ao_conv_to = ao_conv_to.default
    if isinstance(twist_average, typer.models.OptionInfo):
        twist_average = twist_average.default

    tiling_values, selected_twist = _parse_tiling_selection(tiling, supertwist, twist_average)
    if (twist_average or tiling_values) and ao_conv_to is not None:
        raise click.ClickException("--twist-average/--tiling preserve the source angular form; omit --ao-conv-to.")
    if twist_average and not tiling_values:
        try:
            resolve_twist_average_j3_choice(j3_basis_type)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    if tiling_values:
        from .orbital_tiling_sources import read_tiled_trexio_sector, select_tiling_sector, trexio_tiling_sectors

        if getattr(j3_basis_type, "value", j3_basis_type) == "mo":
            raise click.ClickException("Tiling requires an independent supercell J3 basis; choose ao-small/medium/large/full.")
        sectors = trexio_tiling_sectors(trexio_file, tiling_values)
        sector = sectors[0] if twist_average else select_tiling_sector(sectors, selected_twist)
        loaded = read_tiled_trexio_sector(trexio_file, tiling_values, sector)
    else:
        loaded = read_trexio_file(trexio_file, store_tuple=True, twist_ik=0 if twist_average else None)
    (structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = loaded

    if j1_parmeter is not None:
        if coulomb_potential_data.ecp_flag:
            core_electrons = coulomb_potential_data.z_cores
        else:
            core_electrons = [0] * structure_data.natom
        # ``build`` runs the same sanity_check that load_from_hdf5 applies, so a
        # value the loader would refuse is refused here, before anything is written.
        try:
            jastrow_onebody_data = Jastrow_one_body_data.build(
                jastrow_1b_param=j1_parmeter,
                structure_data=structure_data,
                core_electrons=core_electrons,
                jastrow_1b_type=j1_type,
            )
        except ValueError as exc:
            raise click.ClickException(f"Invalid one-body Jastrow option (-j1 / --jastrow-1b-type): {exc}") from exc
    else:
        jastrow_onebody_data = None
    if j2_parmeter is not None:
        try:
            jastrow_twobody_data = Jastrow_two_body_data.build(
                jastrow_2b_param=j2_parmeter,
                jastrow_2b_type=j2_type,
                structure_data=structure_data,
            )
        except ValueError as exc:
            raise click.ClickException(f"Invalid two-body Jastrow option (-j2 / --jastrow-2b-type): {exc}") from exc
    else:
        jastrow_twobody_data = None
    if j3_basis_type is None:
        j3_choice = None
    else:
        j3_choice = getattr(j3_basis_type, "value", j3_basis_type)
    if j3_choice == "none":
        j3_choice = None
    # Untiled twist averaging: the TREXIO AO basis is shared by every k point,
    # so the ik=0 basis yields the one phase-free J3 that /shared stores; the
    # builder drops the Bloch phase and keeps the source angular form.
    jastrow_threebody_data = build_jastrow_three_body_for_basis(aos_data, mos_data, j3_choice)

    # DTN polynomial Jastrow (CASINO-style u / chi / f cutoff polynomials).
    # Mutually exclusive with the Pade J1 + J2 + AO J3 family. The CLI
    # exposes the five physically meaningful presets; the dataclass itself
    # accepts any (use_u, use_chi, use_f) triple.
    if isinstance(jastrow_dtn, typer.models.OptionInfo):
        jastrow_dtn = jastrow_dtn.default
    if isinstance(dtn_cutoff_ratio_u, typer.models.OptionInfo):
        dtn_cutoff_ratio_u = dtn_cutoff_ratio_u.default
    if isinstance(dtn_cutoff_ratio_chi, typer.models.OptionInfo):
        dtn_cutoff_ratio_chi = dtn_cutoff_ratio_chi.default
    if isinstance(dtn_cutoff_ratio_f, typer.models.OptionInfo):
        dtn_cutoff_ratio_f = dtn_cutoff_ratio_f.default
    jastrow_dtn_data = _build_jastrow_dtn_from_cli(
        structure_data,
        jastrow_dtn=jastrow_dtn,
        coulomb_potential_data=coulomb_potential_data,
        dtn_c=dtn_c,
        dtn_n_u=dtn_n_u,
        dtn_n_chi=dtn_n_chi,
        dtn_n_eN=dtn_n_eN,
        dtn_n_ee=dtn_n_ee,
        dtn_L_u=dtn_L_u,
        dtn_L_chi=dtn_L_chi,
        dtn_L_f=dtn_L_f,
        dtn_cutoff_abs_cap=dtn_cutoff_abs_cap,
        dtn_cutoff_ratio_u=dtn_cutoff_ratio_u,
        dtn_cutoff_ratio_chi=dtn_cutoff_ratio_chi,
        dtn_cutoff_ratio_f=dtn_cutoff_ratio_f,
    )
    if jastrow_dtn_data is not None and (
        jastrow_onebody_data is not None or jastrow_twobody_data is not None or jastrow_threebody_data is not None
    ):
        raise click.ClickException(
            "--jastrow-dtn != none is mutually exclusive with the Pade "
            "J1 / J2 + AO J3 family. Remove -j1/-j2/-j3 or set "
            "--jastrow-dtn=none."
        )

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
        jastrow_dtn_data=jastrow_dtn_data,
    )

    if twist_average and not tiling_values:
        convert_trexio_to_twist_average_input(trexio_file, hamiltonian_file, jastrow_data=jastrow_data)
        typer.echo(f"Twist-average Hamiltonian data is saved in {hamiltonian_file}.")
        return

    # Optional AO conversion (cart / sphe)
    if ao_conv_to is not None:
        ao_conv_to = ao_conv_to.lower()
        if ao_conv_to == "cart":
            jastrow_data = jastrow_data.to_cartesian()
            geminal_data = geminal_data.to_cartesian()
        elif ao_conv_to == "sphe":
            # ``to_spherical`` projects a Cartesian MO onto the pure-spherical
            # subspace and discards the rest. The projection is accepted, but a
            # non-negligible discarded part is reported (#591).
            # A non-finite residual means broken source MOs or overlap, not a
            # projection loss, and is refused.
            from ._setting import AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD

            for spin in ("up", "dn"):
                orb_data = getattr(geminal_data, f"orb_data_{spin}_spin")
                if not isinstance(orb_data, MOs_data):
                    continue
                residual = orb_data._spherical_projection_residual()
                if residual.size == 0:
                    continue
                if not np.all(np.isfinite(residual)):
                    raise click.ClickException(
                        f"--ao-conv-to sphe: the spherical-projection residual of the spin-{spin} occupied MOs is "
                        "not finite; the Cartesian source MOs or their overlap matrix are invalid."
                    )
                worst = int(np.argmax(residual))
                if residual[worst] > AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD:
                    log.warning(
                        "WARNING: --ao-conv-to sphe discards the part of the Cartesian wavefunction outside the "
                        "spherical-harmonic subspace: spin-%s occupied MOs, maximum relative S-norm residual %.3e "
                        "(MO %d; threshold %.1e). The written wavefunction is the S-metric projection of the "
                        "source, not the source itself; omit --ao-conv-to to keep the source unchanged.",
                        spin,
                        residual[worst],
                        worst,
                        AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD,
                    )
            jastrow_data = jastrow_data.to_spherical()
            geminal_data = geminal_data.to_spherical()
        else:
            raise ValueError(f"Invalid ao_conv_to = {ao_conv_to!r}. Must be 'cart', 'sphe', or None.")

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    if twist_average:
        from ._twist_average import TwistAverageWriter

        with TwistAverageWriter(hamiltonian_file, hamiltonian_data, len(sectors)) as writer:
            for index, sector in enumerate(sectors):
                sd = (
                    geminal_data
                    if index == 0
                    else read_tiled_trexio_sector(trexio_file, tiling_values, sector, expand_j3_basis=False)[4]
                )
                writer.write_tiled_twist(index, sector[2], sd)
        typer.echo(f"Tiled twist-average Hamiltonian data is saved in {hamiltonian_file}.")
        return

    hamiltonian_data.save_to_hdf5(hamiltonian_file)

    typer.echo(f"Hamiltonian data is saved in {hamiltonian_file}.")


typer_click_trexio = typer.main.get_command(trexio_app)

_cli.add_command(typer_click_trexio, "trexio")


# qe_app -- Quantum ESPRESSO PW projection. Thin CLI wrapper around
# :func:`jqmc.jqmc_qe.run_qe2jqmc`; the body just builds the
# ``qe_params`` / ``jastrow_params`` dicts the converter expects.
qe_app = typer.Typer(help="Read and convert Quantum ESPRESSO PW projections.")


@qe_app.command("convert-to")
def qe_convert_to(
    qe_save_dir: str = typer.Argument(
        ...,
        help=(
            "QE 'prefix.save' directory containing data-file-schema.xml and wfc<ik>.hdf5, "
            "or its parent directory holding exactly one '*.save'."
        ),
    ),
    kind: str = typer.Option(
        "pgto+pw",
        "-k",
        "--kind",
        help="Projection target: 'pw', 'bspline', 'pgto', 'pgto+pw', or 'pgto+bspline'.",
    ),
    upf_paths: list[str] = typer.Option(
        None,
        "-u",
        "--upf",
        help=(
            "UPF mapping 'element=path/to/element.upf' (repeat once per element). "
            "Default: auto-discovered from <prefix>.save/data-file-schema.xml. "
            "Every kind takes the analytic ECP from the UPF, so a pseudopotential "
            "save whose UPFs cannot be found is rejected; the pGTO-bearing kinds "
            "also seed their basis from it."
        ),
    ),
    twist_average: bool = typer.Option(
        False,
        "--twist-average",
        help="Convert every QE k-point into one combined twist-average HDF5 file.",
    ),
    tiling: list[str] = typer.Option(
        None,
        "--tiling",
        help=(
            "Unfold complete QE k cosets into a supercell while retaining the primitive "
            'k blocks. Pass one quoted argument: --tiling "3 3 3" for a diagonal matrix, '
            'or --tiling "2 1 1 1 1 1 3 2 1" for a row-major full matrix. '
            "Supports --kind pw/bspline/pgto/pgto+pw and --twist-average."
        ),
    ),
    supertwist: str = typer.Option(
        None, "--supertwist", help='Select a fractional supercell twist, e.g. "0.2 0.1 0"; otherwise infer the single sector.'
    ),
    mesh_factor: float = typer.Option(
        0.75,
        "--mesh-factor",
        help=(
            "B-spline grid size relative to the QE FFT grid (grid = round(mesh_factor * fft_grid) "
            "per axis). The control-point table scales as mesh_factor^3. The default 0.75 keeps the "
            "spline Nyquist at 1.5x the PW cutoff; 0.5 puts it exactly AT the cutoff, which leaves "
            "the L2 residual small but biases the determinant KINETIC energy (Laplacian error is "
            "O(h^4) vs O(h^6) for values; a bias of -9.5 mHa per twist was measured at a 300 Ry "
            "cutoff). Below 0.625 a Nyquist-headroom warning fires: converge the VMC energy AND "
            "variance against mesh_factor before trusting such grids. The build-time L2 residual "
            "alone cannot certify the grid."
        ),
    ),
    spline_order: int = typer.Option(
        5, "--spline-order", help="B-spline polynomial degree: 5 = quintic (C4, recommended for forces), 3 = cubic (C2)."
    ),
    orthonormalize: bool = typer.Option(
        True,
        "--orthonormalize/--no-orthonormalize",
        help=(
            "kind='pgto' only: orthonormalize the projected MOs of every (k, spin) block in the periodic "
            "AO metric by band-order Cholesky (keeps the span of every lowest-N band subset). Ignored "
            "for the other kinds."
        ),
    ),
    ecutwfc: str = typer.Option(
        None,
        "--ecutwfc",
        help=("PW handoff cutoff. Numeric value in Rydberg or the sentinel 'auto' (pgto+pw only). Omit for the kind default."),
    ),
    alpha_min: float = typer.Option(None, "--alpha-min", help="Diffuse-pGTO exponent floor. Omit for the kind default."),
    exp_tail_eps: float = typer.Option(
        PGTO_EXP_TAIL_EPS_USER_DEFAULT,
        "--exp-tail-eps",
        help=(
            "Exponent-tail bound of the pgto+pw exponent optimiser: no primitive may place more than this "
            "fraction of its momentum-space norm outside the QE WFC sphere (alpha <= alpha_max(l, eps)). "
            "0 disables the bound."
        ),
    ),
    max_cond_number: float = typer.Option(None, "--max-cond-number", help="Upper bound on cond(S) for the pGTO fit."),
    hamiltonian_file: str = typer.Option("hamiltonian_data.h5", "-o", "--output", help="Output HDF5 file."),
    j1_parameter: float = typer.Option(None, "-j1", "--jastrow-1b-parameter", help="Jastrow one-body parameter."),
    j1_type: str = typer.Option("exp", "--jastrow-1b-type", help="Jastrow one-body form: 'exp' or 'pade'."),
    j2_parameter: float = typer.Option(
        None,
        "-j2",
        "--jastrow-2b-parameter",
        help="Jastrow two-body parameter. Omit to disable; must be omitted when --jastrow-dtn != none.",
    ),
    j2_type: str = typer.Option("pade", "--jastrow-2b-type", help="Jastrow two-body form: 'pade' or 'exp'."),
    j3_basis_type: str = typer.Option(
        "none",
        "-j3",
        "--jastrow-3b-basis-set-type",
        help=(
            "Jastrow three-body basis: none, ao-small/medium/large/full, ao, or mo. "
            "With tiling, ao* builds an independent supercell pGTO basis from UPF; mo is unavailable."
        ),
    ),
    jastrow_dtn: str = typer.Option(
        "none",
        "--jastrow-dtn",
        help="DTN preset: 'none', 'u', 'u_chi', 'u_f', 'u_chi_f'. Mutually exclusive with j1/j2/j3.",
    ),
    dtn_c: int = typer.Option(4, "--dtn-C", help="DTN cutoff smoothness order C."),
    dtn_n_u: int = typer.Option(8, "--dtn-Nu", help="DTN u-term polynomial order N_u."),
    dtn_n_chi: int = typer.Option(8, "--dtn-Nchi", help="DTN chi-term polynomial order N_chi."),
    dtn_n_eN: int = typer.Option(4, "--dtn-NeN", help="DTN f-term electron-ion polynomial order N_eN."),
    dtn_n_ee: int = typer.Option(4, "--dtn-Nee", help="DTN f-term electron-electron polynomial order N_ee."),
    dtn_cutoff_ratio_u: float = typer.Option(
        DTN_DEFAULT_CUTOFF_RATIO, "--dtn-cutoff-ratio-u", help="PBC u cutoff ratio relative to R_ins."
    ),
    dtn_cutoff_ratio_chi: float = typer.Option(
        DTN_DEFAULT_CUTOFF_RATIO, "--dtn-cutoff-ratio-chi", help="PBC chi cutoff ratio relative to R_ins."
    ),
    dtn_cutoff_ratio_f: float = typer.Option(
        DTN_DEFAULT_CUTOFF_RATIO, "--dtn-cutoff-ratio-f", help="PBC f cutoff ratio relative to R_ins/2."
    ),
    ewald_tol: float = typer.Option(1.0e-10, "--ewald-tol", help="Per-channel truncation tolerance of the Ewald summation."),
    ecp_z_cores: list[float] = typer.Option(None, "--ecp-z-core", help="Per-atom core-electron count."),
    verbosity: str = typer.Option(None, "--verbosity", help="Logger verbosity ('low', 'high', ...)."),
) -> None:
    """Convert a QE 'prefix.save' directory to a jQMC Hamiltonian HDF5 file."""
    from .jqmc_qe import run_qe2jqmc

    def _v(v: Any) -> Any:
        # typer OptionInfo default leak when called programmatically.
        return v.default if isinstance(v, typer.models.OptionInfo) else v

    verbosity_value = _v(verbosity)
    if verbosity_value is not None:
        verbosity_levels = {
            "low": "INFO",
            "high": "DEBUG",
            "devel": "DEBUG",
            "mpi-low": "INFO",
            "mpi-high": "DEBUG",
            "mpi-devel": "DEBUG",
            "warning": "WARNING",
            "error": "ERROR",
        }
        log_level = verbosity_levels.get(verbosity_value.lower(), "INFO")
        log.setLevel(log_level)
        stream_handler.setLevel(log_level)

    tiling_values, selected_twist = _parse_tiling_selection(tiling, supertwist, twist_average)

    qe_params = {
        "qe_save_dir": qe_save_dir,
        "kind": _v(kind),
        "upf_paths": list(_v(upf_paths) or []),
        "twist_average": bool(_v(twist_average)),
        "tiling": tiling_values,
        "supertwist": selected_twist,
        "ecutwfc": _v(ecutwfc),
        "alpha_min": _v(alpha_min),
        "exp_tail_eps": _v(exp_tail_eps),
        "max_cond_number": _v(max_cond_number),
        "mesh_factor": _v(mesh_factor),
        "spline_order": _v(spline_order),
        "orthonormalize": bool(_v(orthonormalize)),
        "ewald_tol": _v(ewald_tol),
        "ecp_z_cores": list(_v(ecp_z_cores) or []),
    }
    jastrow_params = {
        "j1_parameter": _v(j1_parameter),
        "j1_type": _v(j1_type),
        "j2_parameter": _v(j2_parameter),
        "j2_type": _v(j2_type),
        "j3_basis_type": _v(j3_basis_type),
        "jastrow_dtn": _v(jastrow_dtn),
        "dtn_c": _v(dtn_c),
        "dtn_n_u": _v(dtn_n_u),
        "dtn_n_chi": _v(dtn_n_chi),
        "dtn_n_eN": _v(dtn_n_eN),
        "dtn_n_ee": _v(dtn_n_ee),
        "dtn_cutoff_ratio_u": _v(dtn_cutoff_ratio_u),
        "dtn_cutoff_ratio_chi": _v(dtn_cutoff_ratio_chi),
        "dtn_cutoff_ratio_f": _v(dtn_cutoff_ratio_f),
    }
    # CLI-layer guard: surface the DTN-vs-Pade mutual exclusion as a clean
    # ``click.ClickException`` (no traceback), matching the TREXIO ``wf``
    # converter. ``run_qe2jqmc`` re-checks and raises a raw ``ValueError`` as
    # a library-level backstop; this guard fires first for CLI users.
    if str(jastrow_params["jastrow_dtn"]).lower() != "none" and (
        jastrow_params["j1_parameter"] is not None
        or jastrow_params["j2_parameter"] is not None
        or str(jastrow_params["j3_basis_type"]).lower() != "none"
    ):
        raise click.ClickException(
            "--jastrow-dtn != none is mutually exclusive with the Pade "
            "J1 / J2 + AO J3 family. Remove -j1/-j2/-j3 or set "
            "--jastrow-dtn=none."
        )
    run_qe2jqmc(
        qe_params=qe_params,
        jastrow_params=jastrow_params,
        hamiltonian_h5=_v(hamiltonian_file),
    )


typer_click_qe = typer.main.get_command(qe_app)

_cli.add_command(typer_click_qe, "qe")


# hamiltonian_app
hamiltonian_app = typer.Typer(help="Read and convert Hamiltonian data.")


@hamiltonian_app.command("show-info")
def hamiltonian_show_info(
    hamiltonian_data: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.h5"),
    twist_index: int = typer.Option(0, "--twist", help="Twist index to show for a twist-averaged input."),
) -> None:
    """Show information stored in the Hamiltonian data.

    A twist-averaged input (hamiltonian_twists.h5, combined inputs) prints the
    twist table followed by the full information of the twist selected with
    --twist (the structure and Jastrow are shared by every twist).
    """
    hamiltonian = _load_hamiltonian_for_tool(hamiltonian_data, twist_index, echo_twist_table=True)
    hamiltonian.sanity_check()
    for line in hamiltonian._get_info():
        typer.echo(line)


def _load_hamiltonian_for_tool(path: str, twist_index: int, *, echo_twist_table: bool = False) -> Hamiltonian_data:
    """Load a single-twist file, or one twist of a twist-averaged (combined) file."""
    from ._twist_average import is_twist_average_input, load_twist, load_twist_average_input

    is_twist = is_twist_average_input(path)
    if not is_twist:
        return Hamiltonian_data.load_from_hdf5(path)
    meta = load_twist_average_input(path)
    if not 0 <= twist_index < meta.n_twist:
        raise typer.BadParameter(f"--twist {twist_index} is out of range; {path} holds {meta.n_twist} twists.")
    if echo_twist_table:
        typer.echo(f"Twist-averaged input: {meta.n_twist} twists (sum of BZ weights = {meta.bz_weight_sum:.6f}).")
        typer.echo("  index  k-point (reduced or Cartesian, as written by the converter)   weight")
        for idx, (k, w) in enumerate(zip(meta.k_points, meta.weights, strict=True)):
            k_str = "Gamma" if k is None else "(" + ", ".join(f"{x: .6f}" for x in k) + ")"
            typer.echo(f"  {idx:5d}  {k_str:<58s}  {w:.6f}")
        typer.echo(f"Showing twist {twist_index}:")
    return load_twist(path, twist_index)


@hamiltonian_app.command("to-xyz")
def hamiltonian_to_xyz(
    hamiltonian_data: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.h5"),
    xyz_file: str = typer.Option("struct.xyz", "-o", "--output", help="Output file name."),
) -> None:
    """Write the structure stored in the Hamiltonian data to an xyz file."""
    # Every twist of a twist-averaged input shares the structure; take twist 0.
    hamiltonian = _load_hamiltonian_for_tool(hamiltonian_data, 0)
    structure_data = hamiltonian.structure_data

    with open(xyz_file, "w") as f:
        f.write(f"{structure_data.natom}\n")
        f.write("\n")
        for atom, coord in zip(structure_data.atomic_labels, structure_data.positions, strict=True):
            f.write(f"{atom} {coord[0] * Bohr_to_Angstrom} {coord[1] * Bohr_to_Angstrom} {coord[2] * Bohr_to_Angstrom}\n")


class ansatz_type(str, Enum):
    """Wavefunction ansatz that ``conv-wf`` converts to."""

    jagp = "jagp"
    jagpn = "jagpn"


@hamiltonian_app.command("conv-wf")
def hamiltonian_convert_wavefunction(
    hamiltonian_data_org_file: str = typer.Argument(..., help="hamiltonian_data file, e.g. hamiltonian_data.h5"),
    convert_to: ansatz_type = typer.Option(..., "-c", "--convert-to", help="Ansatz to convert to: jagp or jagpn."),
    hamiltonian_data_conv_file: str = typer.Option(
        "hamiltonian_data_conv.h5", "-o", "--output", help="Output hamiltonian_data file."
    ),
) -> None:
    """Convert wavefunction data in the Hamiltonian data."""
    hamiltonian_org = Hamiltonian_data.load_from_hdf5(hamiltonian_data_org_file)

    wavefunction_data = hamiltonian_org.wavefunction_data
    structure_data = hamiltonian_org.structure_data
    coulomb_potential_data = hamiltonian_org.coulomb_potential_data

    geminal_data = wavefunction_data.geminal_data
    Jastrow_data = wavefunction_data.jastrow_data

    if convert_to == "jagp":
        # conversion of SD to AGP
        typer.echo("Convert SD to AGP.")
        geminal_data = Geminal_data.convert_from_MOs_to_AOs(geminal_data)
    else:
        # conversion of SD to AGPn (natural-orbital diagonal form)
        typer.echo("Convert SD to AGPn.")
        geminal_data = to_AGPn(geminal_data)

    wavefunction_data = Wavefunction_data(jastrow_data=Jastrow_data, geminal_data=geminal_data)

    # Validate before writing: a conversion that Hamiltonian_data.load_from_hdf5 would
    # reject (e.g. an AO-basis AGP with unequal up/dn AO counts, issue #593) must fail
    # here without creating or replacing the output file.
    hamiltonian_conv_data = Hamiltonian_data.build(
        structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wavefunction_data
    )

    hamiltonian_conv_data.save_to_hdf5(hamiltonian_data_conv_file)

    typer.echo(f"Hamiltonian data is saved in {hamiltonian_data_conv_file}.")


typer_click_hamiltonian = typer.main.get_command(hamiltonian_app)

_cli.add_command(typer_click_hamiltonian, "hamiltonian")

# VMC app
vmc_app = typer.Typer(help="Pre- and Post-Processing for VMC calculations.")


@vmc_app.command("generate-input")
def vmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for VMCopt calculations."),
    filename: str = typer.Option("vmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
) -> None:
    """Generate an input file for VMCopt calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                control_table[key] = str(value)
            else:
                control_table[key] = value
            # tomlkit returns a boolean value as a plain bool, which cannot carry a comment.
            if not exclude_comment and not isinstance(value, bool):
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "vmc"
        doc.add("control", control_table)

        vmc_table = tomlkit.table()
        for key, value in cli_parameters["vmc"].items():
            if value is None:
                vmc_table[key] = str(value)
            else:
                vmc_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                vmc_table[key].comment(cli_parameters["vmc_comments"][key])
        doc.add("vmc", vmc_table)

        coulomb_table = tomlkit.table()
        for key, value in cli_parameters["coulomb"].items():
            if value is None:
                coulomb_table[key] = str(value)
            else:
                coulomb_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                coulomb_table[key].comment(cli_parameters["coulomb_comments"][key])
        doc.add("coulomb", coulomb_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


@vmc_app.command("analyze-output")
def vmc_analyze_output(
    filenames: list[str] = typer.Argument(..., help="Output files of vmc optimizations."),
    plot_graph: bool = typer.Option(
        False,
        "-p",
        "--plot_graph",
        help="Open an interactive matplotlib window summarizing the result (skipped, with a message, when the matplotlib backend is non-interactive, e.g. MPLBACKEND=Agg on a headless machine; use -s to save the figure instead).",
    ),
    save_graph: str = typer.Option(None, "-s", "--save-graph", help="Specify a graph filename."),
) -> None:
    """Analyze the output files of vmc optimizations.

    Each file is read with jqmc.output_parser.parse_vmc_steps, so every value
    stays with the "Optimization step" header it follows. A step whose energy,
    force or signal-to-noise line is absent (a step that did not finish) is
    shown with "-" and reported on stderr. A non-finite value is a diverged
    optimization: its row is shown, the steps are named on stderr and the command
    exits with status 1 without writing a graph. Input without any optimization
    step is refused the same way.
    """
    steps = []
    for filename in filenames:
        with open(filename) as f:
            file_steps = parse_vmc_steps(f.read())
        if not file_steps:
            typer.echo(f"Warning: {filename} contains no 'Optimization step' header; no rows were read from it.", err=True)
        steps.extend((filename, step) for step in file_steps)

    if not steps:
        typer.echo(f"Error: no VMC optimization step was found in {', '.join(filenames)}.", err=True)
        raise typer.Exit(code=1)

    keys = ("energy", "energy_error", "max_force", "max_force_error", "signal_to_noise_ratio")
    nonfinite = [
        (filename, step)
        for filename, step in steps
        if any(step[key] is not None and not math.isfinite(step[key]) for key in keys)
    ]
    incomplete = [
        (filename, step)
        for filename, step in steps
        if any(step[key] is None for key in keys) and not any(step is bad for _, bad in nonfinite)
    ]

    iter_list = []
    E_list = []
    max_f_list = []
    signal_to_noise_list = []

    sep = 54
    typer.echo("-" * sep)
    typer.echo(f"{'Iter':<8} {'E (Ha)':<10} {'Max generalized force (Ha/a.u.)':<32} {'Max force S/N':<16}")
    typer.echo("-" * sep)
    for _, step in steps:
        has_E = step["energy"] is not None and step["energy_error"] is not None
        has_f = step["max_force"] is not None and step["max_force_error"] is not None
        has_sn = step["signal_to_noise_ratio"] is not None
        E_text = f"{ufloat(step['energy'], step['energy_error']):8.2uS}" if has_E else f"{'-':>8}"
        f_text = f"{ufloat(step['max_force'], step['max_force_error']):+10.2uS}" if has_f else f"{'-':>10}"
        sn_text = f"{step['signal_to_noise_ratio']:8.3f}" if has_sn else f"{'-':>8}"
        typer.echo(f"{step['step']:4}  {E_text}  {f_text}  {sn_text}")
        if has_E and has_f and has_sn:
            iter_list.append(step["step"])
            E_list.append(ufloat(step["energy"], step["energy_error"]))
            max_f_list.append(ufloat(step["max_force"], step["max_force_error"]))
            signal_to_noise_list.append(step["signal_to_noise_ratio"])
    typer.echo("-" * sep)

    def _step_names(selection):
        return ", ".join(f"{step['step']} ({filename})" for filename, step in selection)

    if incomplete:
        typer.echo(
            "Warning: optimization step(s) without an energy, max generalized force or signal-to-noise line "
            f"(the step did not finish): {_step_names(incomplete)}.",
            err=True,
        )
    if nonfinite:
        typer.echo(
            f"Error: non-finite energy or generalized force in optimization step(s) {_step_names(nonfinite)}; "
            "the optimization diverged and these rows are not a result.",
            err=True,
        )
        raise typer.Exit(code=1)

    # plot graphs
    if plot_graph or save_graph is not None:
        iters = []
        E_means = []
        E_errs = []
        max_f_means = []
        max_f_errs = []

        for iter, E, max_f in zip(iter_list, E_list, max_f_list, strict=True):
            iters.append(iter)
            E_means.append(E.n)
            E_errs.append(E.s)
            max_f_means.append(max_f.n)
            max_f_errs.append(max_f.s)

        plt.rcParams["font.size"] = 8
        plt.rcParams["font.family"] = "sans-serif"

        fig = plt.figure(figsize=(8, 4), facecolor="white", dpi=300, tight_layout=True)

        ax11 = fig.add_subplot(1, 2, 1)
        ax12 = ax11.twinx()

        ax11.tick_params(axis="both", which="both", direction="in")
        ax11.errorbar(iters, E_means, yerr=E_errs, fmt="o-", markersize=3, capsize=3, color="blue", label="Energy")
        ax11.set_xlabel("Iteration")
        ax11.set_ylabel("Energy (Ha)")

        ax12.errorbar(
            iters, max_f_means, yerr=max_f_errs, fmt="s-", markersize=3, capsize=3, color="red", label="Max generalized force"
        )
        ax12.set_ylabel("Max generalized force (Ha/a.u.)")

        lines11, labels11 = ax11.get_legend_handles_labels()
        lines12, labels12 = ax12.get_legend_handles_labels()
        ax11.legend(lines11 + lines12, labels11 + labels12, loc="best")

        ax21 = fig.add_subplot(1, 2, 2)
        ax22 = ax21.twinx()

        ax21.tick_params(axis="both", which="both", direction="in")
        ax21.errorbar(iters, E_means, yerr=E_errs, fmt="o-", markersize=3, capsize=3, color="blue", label="Energy")
        ax21.set_xlabel("Iteration")
        ax21.set_ylabel("Energy (Ha)")

        ax22.plot(
            iters, signal_to_noise_list, marker="s", linestyle="-", markersize=3, color="red", label="Max generalized-force S/N"
        )
        ax22.set_ylabel("Max generalized-force signal-to-noise")

        # Combine legend handles and labels for the second subplot
        lines21, labels21 = ax21.get_legend_handles_labels()
        lines22, labels22 = ax22.get_legend_handles_labels()
        ax21.legend(lines21 + lines22, labels21 + labels22, loc="best")

        if save_graph is not None:
            plt.savefig(save_graph)
            typer.echo(f"Graph is saved in {save_graph}.")

        if plot_graph:
            _show_figure_if_interactive()


#: matplotlib backends that cannot open a window. ``plt.show()`` is a no-op on
#: most of them but blocks forever on some interactive backends without a
#: display, so the plot flags consult this set before calling it.
_NON_INTERACTIVE_MPL_BACKENDS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})


def _show_figure_if_interactive() -> None:
    """Show the current matplotlib figure only when a window can be opened.

    ``plt.show()`` blocks until the user closes the window. That is the
    intended behaviour on a desktop, but on a headless machine (batch job,
    SSH session without X forwarding, CI) it either does nothing or hangs the
    command. When the active backend is one of the known non-interactive
    backends the figure is therefore not shown and a hint to use ``-s`` is
    printed instead.
    """
    backend = matplotlib.get_backend().lower()
    if backend in _NON_INTERACTIVE_MPL_BACKENDS:
        typer.echo(
            f"matplotlib backend {backend!r} cannot open a window; the figure is not shown. "
            "Use -s FILE to save it, or set MPLBACKEND to an interactive backend."
        )
        return
    plt.show()


# ---------------------------------------------------------------------------
# vmc average-wf: uniform average of the wavefunction parameters of several
# ``hamiltonian_data_opt_step_*.h5`` files of one optimization (issue #749).
#
# The average works on the HDF5 tree, not on a reloaded ``Hamiltonian_data``:
# the post-load hooks cast leaves to the active precision zone and re-expand
# periodic AO images, so a load -> save cycle would not keep the invariant
# leaves bitwise. The output is a byte copy of the first input in which only
# the variational-parameter leaves that differ between the inputs are
# overwritten by their arithmetic mean.
# ---------------------------------------------------------------------------

#: Variational parameters as (owning dataclass ``_class_name``, field name):
#: the leaves that ``Wavefunction_data.get_variational_blocks`` publishes and
#: the ``apply_block_update`` owners write back (J1/J2 scalars, the J3 matrix,
#: the DTN independent coefficients, the AGP/AGPn lambda blocks and the AO
#: radial exponents/contraction coefficients of the J3 and geminal bases).
_AVERAGE_WF_PARAMETER_FIELDS = frozenset(
    {
        ("Jastrow_one_body_data", "jastrow_1b_param"),
        ("Jastrow_two_body_data", "jastrow_2b_param"),
        ("Jastrow_three_body_data", "j_matrix"),
        ("Jastrow_DTN_data", "alpha_indep"),
        ("Jastrow_DTN_data", "beta_indep"),
        ("Jastrow_DTN_data", "gamma_indep"),
        ("AGP_data", "lambda_matrix"),
        ("AGPn_data", "lambda_diag"),
        ("AGPn_data", "lambda_unpaired"),
        ("AO_radial_shell_data", "exponents"),
        ("AO_radial_shell_data", "coefficients"),
    }
)
#: Determinant MO coefficients are variational only below a geminal (J3 MOs are fixed).
_AVERAGE_WF_GEMINAL_CLASSES = frozenset({"SD_data", "AGP_data", "AGPn_data"})
#: Per-twist determinant payload of a combined file (``/twists/twist_NNN``).
_AVERAGE_WF_TWIST_GROUP = re.compile(r"^/twists/twist_\d+$")
_AVERAGE_WF_TWIST_FIELDS = frozenset({"mo_coefficients_up", "mo_coefficients_dn", "lambda_matrix"})
#: Root attributes naming the code that wrote a file; a difference is reported, not refused.
_AVERAGE_WF_WRITER_ATTRS = frozenset({"jqmc_version", "jqmc_git_commit"})
#: Root provenance attributes written by this command; ignored when comparing inputs.
_AVERAGE_WF_PROVENANCE_PREFIX = "opt_step_average_"
#: Read granularity (bytes per input) of the streaming comparison and mean.
_AVERAGE_WF_BLOCK_BYTES = 1 << 24
_AVERAGE_WF_STEP_PATTERN = re.compile(r"_opt_step_(\d+)\.h5$")


class _AverageWfRefusal(ValueError):
    """The inputs of ``vmc average-wf`` cannot be averaged (message names the cause)."""


def _average_wf_attr_str(value: Any) -> str:
    """Decode an HDF5 string attribute (``bytes`` or ``str``); other values become ``""``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else ""


def _average_wf_leaf_name(group_path: str, name: str, *, attribute: bool) -> str:
    """``/a/b/name`` for a dataset, ``/a/b@name`` for an attribute."""
    if attribute:
        return f"{group_path}@{name}"
    return f"{group_path.rstrip('/')}/{name}"


def _average_wf_is_parameter(group: h5py.Group, name: str) -> bool:
    """Whether ``group[name]`` (dataset or attribute) is a variational parameter."""
    class_name = _average_wf_attr_str(group.attrs.get("_class_name"))
    if (class_name, name) in _AVERAGE_WF_PARAMETER_FIELDS:
        return True
    if class_name == "MOs_data" and name == "mo_coefficients":
        return _average_wf_attr_str(group.parent.attrs.get("_class_name")) in _AVERAGE_WF_GEMINAL_CLASSES
    return bool(_AVERAGE_WF_TWIST_GROUP.match(group.name)) and name in _AVERAGE_WF_TWIST_FIELDS


def _average_wf_is_mo_rows(group: h5py.Group, name: str) -> bool:
    """Whether the leaf holds determinant MO rows, whose sign/phase is a gauge."""
    if bool(_AVERAGE_WF_TWIST_GROUP.match(group.name)) and name in ("mo_coefficients_up", "mo_coefficients_dn"):
        return True
    return _average_wf_attr_str(group.attrs.get("_class_name")) == "MOs_data" and name == "mo_coefficients"


def _average_wf_topic(leaf: str) -> str:
    """What a non-variational difference at ``leaf`` means, for the refusal message."""
    if "structure_data" in leaf:
        return "the atomic structure differs"
    if (
        leaf.startswith("/_meta")
        or leaf == "/twists"
        or re.search(r"^/twists/twist_\d+(/k_point|/weight|@is_gamma)$", leaf) is not None
    ):
        return "the twist set (count, k-points or weights) differs"
    if "coulomb_potential_data" in leaf:
        return "the Coulomb / ECP data differ"
    if any(token in leaf for token in ("aos_data", "orb_data", "/radial", "composite_child")):
        return "the basis set differs"
    return "the wavefunction structure (ansatz, Jastrow channels or stored optimization state) differs"


def _average_wf_same_value(a: Any, b: Any) -> bool:
    """Bitwise equality of two attribute values, with value equality for strings/objects."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.dtype != b_arr.dtype or a_arr.shape != b_arr.shape:
        return False
    if a_arr.dtype.kind in "biufc":
        return a_arr.tobytes() == b_arr.tobytes()
    return bool(np.array_equal(a_arr, b_arr))


def _average_wf_blocks(shape: tuple[int, ...], itemsize: int):
    """Slices along the first axis holding about ``_AVERAGE_WF_BLOCK_BYTES`` each (``()`` for a scalar)."""
    if len(shape) == 0:
        yield ()
        return
    row_bytes = max(1, itemsize * int(np.prod(shape[1:], dtype=np.int64)))
    step = max(1, _AVERAGE_WF_BLOCK_BYTES // row_bytes)
    for start in range(0, int(shape[0]), step):
        yield np.s_[start : min(start + step, int(shape[0]))]


def _average_wf_bitwise_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise bit equality of two same-dtype arrays (``-0.0 != 0.0``; equal NaN bits compare equal)."""
    shape = np.shape(a)
    itemsize = np.asarray(a).dtype.itemsize
    a_bytes = np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(-1, itemsize)
    b_bytes = np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(-1, itemsize)
    return np.all(a_bytes == b_bytes, axis=1).reshape(shape)


def _average_wf_dataset_differs(datasets: list[h5py.Dataset], names: list[str], leaf: str) -> bool:
    """Check one dataset across the inputs; ``True`` when its values differ."""
    reference = datasets[0]
    for name, dataset in zip(names[1:], datasets[1:], strict=True):
        if dataset.shape != reference.shape or dataset.dtype != reference.dtype:
            raise _AverageWfRefusal(
                f"{leaf} has shape {dataset.shape} / dtype {dataset.dtype} in {name} but "
                f"{reference.shape} / {reference.dtype} in {names[0]}: {_average_wf_topic(leaf)} "
                "(a different precision mode stores different dtypes)."
            )
    if reference.size == 0:
        return False
    if reference.dtype.kind not in "biufc":
        values = np.asarray(reference[()])
        return any(not np.array_equal(values, np.asarray(dataset[()])) for dataset in datasets[1:])
    for selection in _average_wf_blocks(reference.shape, reference.dtype.itemsize):
        reference_bytes = np.asarray(reference[selection]).tobytes()
        if any(np.asarray(dataset[selection]).tobytes() != reference_bytes for dataset in datasets[1:]):
            return True
    return False


def _average_wf_require_finite(datasets: list[h5py.Dataset], names: list[str], leaf: str) -> None:
    """Refuse a variational-parameter dataset holding a non-finite value in any input.

    Runs for every parameter leaf, changed or not and with a single input, so
    an unchanged NaN cannot be copied into the output. Reads block-wise.
    """
    reference = datasets[0]
    if reference.size == 0 or reference.dtype.kind not in "fc":
        return
    for name, dataset in zip(names, datasets, strict=True):
        for selection in _average_wf_blocks(dataset.shape, dataset.dtype.itemsize):
            if not np.all(np.isfinite(np.asarray(dataset[selection]))):
                raise _AverageWfRefusal(f"{leaf} holds a non-finite value in {name}; a diverged step cannot be averaged.")


def _average_wf_scan(files: list[h5py.File], names: list[str]) -> tuple[list[tuple[str, str, bool, bool]], list[str]]:
    """Compare the inputs and list the leaves to average.

    Returns ``(leaves, warnings)`` where every leaf is ``(group_path, name,
    is_attribute, is_mo_rows)``. Raises :class:`_AverageWfRefusal` for any
    structural mismatch or for a difference outside the variational parameters.
    """
    leaves: list[tuple[str, str, bool, bool]] = []
    warnings: list[str] = []

    def attribute_names(group: h5py.Group, path: str) -> set[str]:
        keys = set(group.attrs.keys())
        if path == "/":
            keys = {key for key in keys if not key.startswith(_AVERAGE_WF_PROVENANCE_PREFIX)}
        return keys

    def scan_group(path: str) -> None:
        groups = [file[path] for file in files]
        reference = groups[0]
        reference_attrs = attribute_names(reference, path)
        for name, group in zip(names[1:], groups[1:], strict=True):
            other = attribute_names(group, path)
            if other != reference_attrs:
                leaf = _average_wf_leaf_name(path, sorted(other ^ reference_attrs)[0], attribute=True)
                raise _AverageWfRefusal(
                    f"{name} and {names[0]} store different attributes at {path} (first: {leaf}): {_average_wf_topic(leaf)}."
                )
        for attr in sorted(reference_attrs):
            leaf = _average_wf_leaf_name(path, attr, attribute=True)
            values = [group.attrs[attr] for group in groups]
            if _average_wf_is_parameter(reference, attr):
                for name, value in zip(names, values, strict=True):
                    array = np.asarray(value)
                    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                        raise _AverageWfRefusal(
                            f"{leaf} holds a non-finite value in {name}; a diverged step cannot be averaged."
                        )
            differing = [names[i] for i in range(1, len(values)) if not _average_wf_same_value(values[0], values[i])]
            if not differing:
                continue
            if path == "/" and attr in _AVERAGE_WF_WRITER_ATTRS:
                warnings.append(
                    f"{leaf} differs between {names[0]} and {differing[0]}: the inputs were written by different "
                    f"jQMC builds. The output keeps the value of {names[0]}."
                )
                continue
            numeric = all(np.asarray(value).dtype.kind in "fc" for value in values)
            if not (_average_wf_is_parameter(reference, attr) and numeric):
                raise _AverageWfRefusal(
                    f"{leaf} differs between {names[0]} and {differing[0]} and is not a variational parameter: "
                    f"{_average_wf_topic(leaf)}. The inputs are not steps of one optimization."
                )
            leaves.append((path, attr, True, False))

        reference_keys = sorted(reference.keys())
        for name, group in zip(names[1:], groups[1:], strict=True):
            keys = sorted(group.keys())
            if keys != reference_keys:
                leaf = _average_wf_leaf_name(path, sorted(set(keys) ^ set(reference_keys))[0], attribute=False)
                raise _AverageWfRefusal(
                    f"{name} and {names[0]} hold different members at {path} (first: {leaf}): {_average_wf_topic(leaf)}."
                )
        for key in reference_keys:
            child = _average_wf_leaf_name(path, key, attribute=False)
            links = [group.get(key, getlink=True) for group in groups]
            link_types = {type(link) for link in links}
            if len(link_types) != 1:
                raise _AverageWfRefusal(f"{child} is linked differently in the inputs: {_average_wf_topic(child)}.")
            if isinstance(links[0], (h5py.SoftLink, h5py.ExternalLink)):
                targets = {(getattr(link, "filename", ""), link.path) for link in links}
                if len(targets) != 1:
                    raise _AverageWfRefusal(f"{child} links to different targets in the inputs.")
                continue
            objects = [group[key] for group in groups]
            if isinstance(objects[0], h5py.Group):
                if not all(isinstance(obj, h5py.Group) for obj in objects):
                    raise _AverageWfRefusal(f"{child} is a group in {names[0]} but not in every input.")
                scan_group(child)
                continue
            if not all(isinstance(obj, h5py.Dataset) for obj in objects):
                raise _AverageWfRefusal(f"{child} is a dataset in {names[0]} but not in every input.")
            is_parameter = _average_wf_is_parameter(reference, key)
            if is_parameter:
                # Before the unchanged-leaf shortcut (and for a single input): a copied
                # parameter must be finite too.
                _average_wf_require_finite(objects, names, child)
            if not _average_wf_dataset_differs(objects, names, child):
                continue
            if not is_parameter:
                differing = next(
                    names[i]
                    for i in range(1, len(objects))
                    if _average_wf_dataset_differs([objects[0], objects[i]], [names[0], names[i]], child)
                )
                raise _AverageWfRefusal(
                    f"{child} differs between {names[0]} and {differing} and is not a variational parameter: "
                    f"{_average_wf_topic(child)}. The inputs are not steps of one optimization."
                )
            if objects[0].dtype.kind not in "fc":
                raise _AverageWfRefusal(f"{child} differs between the inputs but has non-floating dtype {objects[0].dtype}.")
            leaves.append((path, key, False, _average_wf_is_mo_rows(reference, key)))

    scan_group("/")
    return leaves, warnings


def _average_wf_dataset(
    datasets: list[h5py.Dataset], names: list[str], leaf: str, output: h5py.Dataset, *, mo_rows: bool
) -> tuple[list[bytes], float | None]:
    """Stream the uniform mean of one dataset into ``output``.

    The mean is accumulated in float64 / complex128 one block at a time, so at
    most a few blocks are held at once. Elements that are bitwise identical in
    every input are copied from the first input instead of being re-rounded.
    Returns the per-input SHA-256 digests of the leaf and, for MO rows, the
    smallest row coherence ``|sum_i c_ik| / sum_i |c_ik|``.
    """
    dtype = datasets[0].dtype
    work_dtype = np.complex128 if dtype.kind == "c" else np.float64
    digests = [hashlib.sha256() for _ in datasets]
    min_coherence: float | None = None
    for selection in _average_wf_blocks(datasets[0].shape, dtype.itemsize):
        first = None
        same = None
        total = None
        row_norm_sum = None
        for name, dataset, digest in zip(names, datasets, digests, strict=True):
            values = np.asarray(dataset[selection])
            if not np.all(np.isfinite(values)):
                raise _AverageWfRefusal(f"{leaf} holds a non-finite value in {name}; a diverged step cannot be averaged.")
            digest.update(values.tobytes())
            work = values.astype(work_dtype)
            if first is None:
                first, same, total = values, np.ones(values.shape, dtype=bool), work.copy()
                row_norm_sum = np.linalg.norm(work, axis=-1) if mo_rows else None
            else:
                same &= _average_wf_bitwise_equal(values, first)
                total += work
                if mo_rows:
                    row_norm_sum += np.linalg.norm(work, axis=-1)
        mean = np.where(same, first, (total / len(datasets)).astype(dtype))
        output[selection] = mean
        if mo_rows:
            populated = row_norm_sum > 0.0
            if np.any(populated):
                coherence = np.linalg.norm(total, axis=-1)[populated] / row_norm_sum[populated]
                block_min = float(np.min(coherence))
                min_coherence = block_min if min_coherence is None else min(min_coherence, block_min)
    return [digest.digest() for digest in digests], min_coherence


def _average_wf_attribute(groups: list[h5py.Group], name: str, output: h5py.Group) -> list[bytes]:
    """Uniform mean of one numeric attribute (kept bitwise where every input agrees)."""
    values = [np.asarray(group.attrs[name]) for group in groups]
    for value in values:
        if not np.all(np.isfinite(value)):
            leaf = _average_wf_leaf_name(groups[0].name, name, attribute=True)
            raise _AverageWfRefusal(f"{leaf} holds a non-finite value; a diverged step cannot be averaged.")
    work_dtype = np.complex128 if values[0].dtype.kind == "c" else np.float64
    total = values[0].astype(work_dtype)
    same = np.ones(values[0].shape, dtype=bool)
    for value in values[1:]:
        total = total + value.astype(work_dtype)
        same &= _average_wf_bitwise_equal(value, values[0])
    mean = np.where(same, values[0], (total / len(values)).astype(values[0].dtype))
    output.attrs[name] = mean[()] if mean.ndim == 0 else mean
    return [hashlib.sha256(value.tobytes()).digest() for value in values]


def _average_wf_layout(file: h5py.File, name: str) -> str:
    """Identify combined and single-Hamiltonian payloads by their groups."""
    from .hamiltonians import _is_twist_average_layout, _require_twist_average_layout

    if _is_twist_average_layout(file):
        try:
            _require_twist_average_layout(file)
        except ValueError as exc:
            raise _AverageWfRefusal(f"{name}: {exc}") from exc
        return "twist-average"
    if "structure_data" in file and "wavefunction_data" in file:
        return "single-twist"
    raise _AverageWfRefusal(f"{name} is not a Hamiltonian data file (neither a single nor combined twist payload).")


def _average_wf_pattern_warnings(digests: dict[str, list[bytes]], names: list[str]) -> list[str]:
    """Warn when the averaged leaves do not all change together between two inputs.

    Within one optimization every optimized block normally changes at every
    step, so every averaged leaf partitions the inputs in the same way.
    """
    warnings: list[str] = []
    partial: str | None = None
    leaves = list(digests)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            equal = [leaf for leaf in leaves if digests[leaf][i] == digests[leaf][j]]
            if leaves and len(equal) == len(leaves):
                warnings.append(
                    f"{names[i]} and {names[j]} hold identical parameters, so this state enters the mean twice "
                    "(for example a step repeated after an optimizer rollback)."
                )
            elif equal and partial is None:
                changed = next(leaf for leaf in leaves if leaf not in equal)
                partial = (
                    f"{equal[0]} is identical in {names[i]} and {names[j]} while {changed} differs between them. "
                    "Within one optimization every optimized block normally changes at every step; this happens when "
                    "a block was frozen for some steps (for example by the SR diagonal floor) and when steps of "
                    "optimizations with different optimized parameters are mixed. Check the inputs."
                )
    return warnings if partial is None else [*warnings, partial]


def _average_wf_validate(path: str, layout: str) -> None:
    """Load the written file through the loaders a jQMC run uses."""
    try:
        if layout == "twist-average":
            from ._twist_average import load_twist, load_twist_average_input

            for twist_index in range(load_twist_average_input(path).n_twist):
                load_twist(path, twist_index)
        else:
            Hamiltonian_data.load_from_hdf5(path)
    except Exception as exc:
        raise _AverageWfRefusal(f"the averaged Hamiltonian does not load ({type(exc).__name__}: {exc}).") from exc


def average_opt_step_wavefunctions(inputs: list[str], output: str) -> dict[str, Any]:
    """Uniformly average the variational parameters of optimization-step Hamiltonian files.

    "One optimization" is defined by what the files contain. The inputs must
    have the same layout (single-twist ``Hamiltonian_data`` or combined
    twist average), the same HDF5 tree (groups, datasets, attributes, link
    kinds, shapes and dtypes) and bitwise identical values everywhere except
    in the variational parameters (:data:`_AVERAGE_WF_PARAMETER_FIELDS`, the
    determinant MO coefficients and the per-twist MO / lambda payload). This
    fixes the structure, basis, ECP, twist set (count, k-points, weights),
    ansatz class, Jastrow channels and precision mode (the stored dtypes). The
    ``opt_*`` flags are not stored. A scalar J1/J2 parameter that the
    optimizer never updated is stored as an HDF5 attribute instead of a
    dataset, so that difference is refused as a structural mismatch. An array
    block frozen in one set of inputs and optimized in another cannot be told
    apart from a block frozen for some steps; it produces a warning.

    Each differing parameter leaf becomes the uniform arithmetic mean of the
    inputs (float64 / complex128 accumulation, cast to the stored dtype).
    Elements identical in every input, and all other leaves, stay bitwise
    those of the first input. Determinant MO rows whose coherence falls below
    :data:`jqmc._setting.OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE` (a sign or
    phase jump of an orbital between steps) are refused, and so is a non-finite
    variational parameter, whether or not it differs between the inputs. The averaged MOs are
    not re-orthonormalized. The output is written atomically, after it has
    loaded through :meth:`Hamiltonian_data.load_from_hdf5` or, for every
    twist, :func:`jqmc._twist_average.load_twist`.

    Args:
        inputs: Hamiltonian files of one optimization (at least one).
        output: Output path; it must not be one of the inputs.

    Returns:
        A summary with ``layout``, ``leaves`` (averaged leaf paths),
        ``min_mo_row_coherence`` and ``warnings``.

    Raises:
        _AverageWfRefusal: The inputs are inconsistent or cannot be averaged.
    """
    import shutil
    import tempfile

    from ._setting import OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE

    if not inputs:
        raise _AverageWfRefusal("no input file was given.")
    real_inputs = [os.path.realpath(path) for path in inputs]
    for index, path in enumerate(inputs):
        if not os.path.isfile(path):
            raise _AverageWfRefusal(f"{path} does not exist or is not a file.")
        if real_inputs.index(real_inputs[index]) != index:
            raise _AverageWfRefusal(f"{path} is given more than once; each step enters the mean once.")
    if os.path.realpath(output) in real_inputs:
        raise _AverageWfRefusal(f"the output {output} is one of the inputs.")

    files: list[h5py.File] = []
    temporary = None
    try:
        for path in inputs:
            try:
                files.append(h5py.File(path, "r"))
            except OSError as exc:
                raise _AverageWfRefusal(f"{path} is not a readable HDF5 file ({exc}).") from exc
        layouts = [_average_wf_layout(file, path) for file, path in zip(files, inputs, strict=True)]
        for path, layout in zip(inputs[1:], layouts[1:], strict=True):
            if layout != layouts[0]:
                raise _AverageWfRefusal(f"{path} is a {layout} file but {inputs[0]} is a {layouts[0]} file.")
        leaves, warnings = _average_wf_scan(files, inputs)

        directory = os.path.dirname(os.path.abspath(output))
        handle, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(output)}.", suffix=".tmp", dir=directory)
        os.close(handle)
        shutil.copyfile(inputs[0], temporary)
        digests: dict[str, list[bytes]] = {}
        min_coherence: float | None = None
        averaged = [_average_wf_leaf_name(path, name, attribute=attribute) for path, name, attribute, _ in leaves]
        with h5py.File(temporary, "r+") as out:
            for (group_path, name, is_attribute, mo_rows), leaf in zip(leaves, averaged, strict=True):
                if is_attribute:
                    digests[leaf] = _average_wf_attribute([file[group_path] for file in files], name, out[group_path])
                    continue
                digests[leaf], coherence = _average_wf_dataset(
                    [file[group_path][name] for file in files], inputs, leaf, out[group_path][name], mo_rows=mo_rows
                )
                if coherence is None:
                    continue
                if coherence < OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE:
                    raise _AverageWfRefusal(
                        f"the MO rows of {leaf} are not coherent across the inputs (min_k |sum_i c_ik| / sum_i |c_ik| = "
                        f"{coherence:.6f} < {OPT_STEP_AVERAGE_MIN_MO_ROW_COHERENCE}): an orbital changed sign or phase "
                        "between steps, and the arithmetic mean over such a gauge jump is not the mean of one "
                        "continuous optimization."
                    )
                min_coherence = coherence if min_coherence is None else min(min_coherence, coherence)
            warnings.extend(_average_wf_pattern_warnings(digests, inputs))
            info = build_info()
            steps = [int(match.group(1)) if (match := _AVERAGE_WF_STEP_PATTERN.search(path)) else -1 for path in inputs]
            prefix = _AVERAGE_WF_PROVENANCE_PREFIX
            out.attrs[f"{prefix}sources"] = np.asarray(inputs, dtype=h5py.string_dtype())
            out.attrs[f"{prefix}steps"] = np.asarray(steps, dtype=np.int64)
            out.attrs[f"{prefix}num_sources"] = len(inputs)
            out.attrs[f"{prefix}weighting"] = "uniform"
            out.attrs[f"{prefix}leaves"] = np.asarray(averaged, dtype=h5py.string_dtype())
            out.attrs[f"{prefix}jqmc_version"] = info.version
            out.attrs[f"{prefix}jqmc_git_commit"] = info.git_commit
        for file in files:
            file.close()
        files = []
        _average_wf_validate(temporary, layouts[0])
        os.replace(temporary, output)
        temporary = None
        return {"layout": layouts[0], "leaves": averaged, "min_mo_row_coherence": min_coherence, "warnings": warnings}
    finally:
        for file in files:
            file.close()
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)


@vmc_app.command("average-wf")
def vmc_average_wf(
    hamiltonian_files: list[str] = typer.Argument(
        ..., help="hamiltonian_data_opt_step_*.h5 files of ONE optimization, e.g. its last 20 steps."
    ),
    output: str = typer.Option(
        "hamiltonian_data_averaged.h5", "-o", "--output", help="Output Hamiltonian file (same format as the inputs)."
    ),
) -> None:
    """Average the wavefunction parameters of several VMC optimization steps with uniform weights.

    Only the variational parameters that differ between the inputs are
    averaged; everything else is copied bitwise from the first file. Inputs
    that are not steps of one optimization are refused with exit status 1,
    and nothing is written.
    """
    try:
        summary = average_opt_step_wavefunctions(list(hamiltonian_files), output)
    except _AverageWfRefusal as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for warning in summary["warnings"]:
        typer.echo(f"Warning: {warning}", err=True)
    typer.echo(f"Averaged {len(hamiltonian_files)} {summary['layout']} Hamiltonian file(s) with uniform weights.")
    if summary["leaves"]:
        typer.echo("Averaged variational parameters:")
        for leaf in summary["leaves"]:
            typer.echo(f"  {leaf}")
    else:
        typer.echo("The inputs hold identical parameters; nothing was averaged.")
    if summary["min_mo_row_coherence"] is not None:
        typer.echo(f"Smallest MO-row coherence |sum_i c_ik| / sum_i |c_ik| = {summary['min_mo_row_coherence']:.6f}.")
    typer.echo("All other data are copied bitwise from the first input.")
    typer.echo(f"Averaged Hamiltonian data is saved in {output}.")


typer_click_vmc = typer.main.get_command(vmc_app)

_cli.add_command(typer_click_vmc, "vmc")


# mcmc app
mcmc_app = typer.Typer(help="Pre- and Post-Processing for MCMC calculations.")

#: Checkpoint ``_meta.driver_type`` values each post-processing app reads.
_APP_DRIVER_TYPES = {"mcmc": ("MCMC",), "lrdmc": ("GFMC_t", "GFMC_n")}


def _require_checkpoint_driver(restart_chk: str, *, app: str, command: str) -> None:
    """Refuse a checkpoint (or twist directory) written by the other driver.

    ``mcmc`` and ``lrdmc`` apply different estimators to the same ``e_L`` /
    ``w_L`` datasets, so running one app on the other driver's checkpoint
    prints a plausible but different estimate (issue #588). A file whose
    ``_meta`` carries no ``driver_type``, or that cannot be opened, is left to
    the observable loader and its existing diagnostics.
    """
    accepted = _APP_DRIVER_TYPES[app]
    other_app = "lrdmc" if app == "mcmc" else "mcmc"
    if os.path.isdir(restart_chk):
        paths = [path for path, _weight in load_twist_checkpoint_manifest(restart_chk)]
    else:
        paths = [restart_chk]
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                meta = f.get("_meta")
                driver_type = None if meta is None else meta.attrs.get("driver_type")
        except OSError:
            continue
        if isinstance(driver_type, bytes):
            driver_type = driver_type.decode()
        if driver_type is not None and str(driver_type) not in accepted:
            typer.echo(
                f"Error: {path} was written by the {driver_type} driver, but `jqmc-tool {app} {command}` "
                f"reads {' / '.join(accepted)} checkpoints only. Use `jqmc-tool {other_app} {command}` instead.",
                err=True,
            )
            raise typer.Exit(code=1)


#: TOML section whose frozen sampler keys every twist of one run shares, per stored driver type.
_TWIST_SAMPLER_SECTIONS = {"MCMC": "mcmc", "GFMC_t": "lrdmc-tau", "GFMC_n": "lrdmc-bra"}
#: Driver-config keys that may differ between the twists of one run: observable switches (a
#: resume reuses a twist that stores more observables) and the E_scf that GFMC_n evolves
#: (its input value is compared through ``_meta/initial_E_scf``).
_TWIST_UNSHARED_CONFIG_KEYS = frozenset({"comput_position_deriv", "comput_stress", "comput_log_WF_param_deriv", "E_scf"})
#: Hamiltonian parts shared by all twists; the orbitals legitimately differ per twist.
_TWIST_SHARED_HAMILTONIAN_GROUPS = (
    "hamiltonian_data/structure_data",
    "hamiltonian_data/coulomb_potential_data",
    "hamiltonian_data/wavefunction_data/jastrow_data",
)


def _h5_values_equal(first: Any, second: Any) -> bool:
    """Exact equality of two stored HDF5 values (same dtype, shape and contents)."""
    first, second = np.asarray(first), np.asarray(second)
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    return bool(np.array_equal(first, second, equal_nan=first.dtype.kind in "fc"))


def _h5_trees_equal(first: h5py.Group | h5py.Dataset, second: h5py.Group | h5py.Dataset) -> bool:
    """Exact recursive equality of two stored HDF5 groups or datasets, attributes included."""
    if isinstance(first, h5py.Group) != isinstance(second, h5py.Group):
        return False
    if set(first.attrs) != set(second.attrs):
        return False
    if any(not _h5_values_equal(first.attrs[key], second.attrs[key]) for key in first.attrs):
        return False
    if isinstance(first, h5py.Dataset):
        return _h5_values_equal(first[()], second[()])
    if set(first) != set(second):
        return False
    return all(_h5_trees_equal(first[key], second[key]) for key in first)


def _check_twist_directory_consistency(restart_chk: str, manifest: list[tuple[str, float]]) -> None:
    """Refuse twist checkpoints that do not come from one twist-averaged run (issue #861).

    ``load_twist_checkpoint_manifest`` checks the twist indices, convergence
    and BZ weights only. A twist file copied in from another run (re-optimized
    Jastrow, other LRDMC tau, other walker count) or a duplicated twist would
    still be combined into one BZ energy. Every twist must therefore carry the
    same stored driver type, the same frozen sampler settings (the keys a
    twist-averaged resume compares, plus the MPC switch and cutoff and the
    GFMC_n input E_scf), the same structure, ECP and Jastrow, and a distinct
    k point. A twist without ``k_point`` but with ``ham_digest`` is the Gamma
    twist (the driver stores no k point for it). Metadata are compared
    exactly; a value stored by some twists only is a mismatch, and one stored
    by none is not compared. The seed, the step count and the per-twist
    orbitals may differ. Raises ``ValueError`` naming the offending files.
    """
    from ._restart_policy import TOML_TO_CFG

    def fail(path: str, reference_path: str, what: str) -> None:
        raise ValueError(
            f"{path}: {what} is missing or differs from {reference_path}; the twist checkpoints in {restart_chk} "
            "are not from one twist-averaged run, refusing to combine them into one BZ average."
        )

    with contextlib.ExitStack() as stack:
        files = [(path, stack.enter_context(h5py.File(path, "r"))) for path, _weight in manifest]
        metas = [f["_meta"].attrs if "_meta" in f else {} for _path, f in files]
        configs = [f["rank_0/driver_config"].attrs if "rank_0/driver_config" in f else {} for _path, f in files]

        def compare(values: list[Any], what: str, equal: Any = _h5_values_equal) -> None:
            present = [index for index, value in enumerate(values) if value is not None]
            if not present:
                return
            reference = present[0]
            for index, value in enumerate(values):
                if value is None or not equal(value, values[reference]):
                    fail(files[index][0], files[reference][0], what)

        compare([meta.get("driver_type") for meta in metas], "_meta/driver_type")
        driver_type = metas[0].get("driver_type")
        if isinstance(driver_type, bytes):
            driver_type = driver_type.decode()
        section = _TWIST_SAMPLER_SECTIONS.get(str(driver_type))
        config_keys = {"num_walkers", "random_discretized_mesh"}
        config_keys |= {cfg for (group, _key), cfg in TOML_TO_CFG.items() if group in (section, "coulomb")}
        for key in sorted(config_keys - _TWIST_UNSHARED_CONFIG_KEYS):
            compare([cfg.get(key) for cfg in configs], f"driver_config {key}")
        for key in ("initial_E_scf", "mpc_cutoff"):
            compare([meta.get(key) for meta in metas], f"_meta/{key}")
        for group in _TWIST_SHARED_HAMILTONIAN_GROUPS:
            compare([f[group] if group in f else None for _path, f in files], group, equal=_h5_trees_equal)

        seen: dict[tuple[float, ...], str] = {}
        for (path, _f), meta in zip(files, metas, strict=True):
            if "k_point" in meta:
                k_point = np.asarray(meta["k_point"], dtype=np.float64)
            elif "ham_digest" in meta:
                k_point = np.zeros(3)
            else:
                continue
            key = tuple(k_point.ravel().tolist())
            if key in seen:
                raise ValueError(
                    f"{path}: duplicate k_point {list(key)} (also in {seen[key]}); the twist checkpoints in "
                    f"{restart_chk} are not from one twist-averaged run, refusing to combine them into one BZ average."
                )
            seen[key] = path


def _require_consistent_twist_directory(restart_chk: str) -> None:
    """Exit 1 without a result when the twist directory mixes runs (see ``_check_twist_directory_consistency``)."""
    manifest = load_twist_checkpoint_manifest(restart_chk)
    try:
        _check_twist_directory_consistency(restart_chk, manifest)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _check_mcmc_binning_inputs(rank_data: list, *, num_mcmc_bin_blocks: int, num_mcmc_warmup_steps: int, source: str) -> None:
    """Refuse binning settings that would make the jackknife estimate meaningless.

    Without these checks an over-long warmup or too many bins produced empty
    bins, and an all-zero-weight leave-one-out set divided by zero; either
    case printed a number instead of an error.
    """
    if num_mcmc_bin_blocks < 1:
        raise typer.BadParameter(f"num_mcmc_bin_blocks = {num_mcmc_bin_blocks} must be >= 1.")
    if num_mcmc_warmup_steps < 0:
        raise typer.BadParameter(f"num_mcmc_warmup_steps = {num_mcmc_warmup_steps} must be >= 0.")
    if not rank_data:
        raise typer.BadParameter(f"{source}: no MCMC observables found in the checkpoint.")
    for rank, obs in enumerate(rank_data):
        n_total = int(np.shape(obs["e_L"])[0])
        n_kept = n_total - num_mcmc_warmup_steps
        if n_kept < num_mcmc_bin_blocks:
            raise typer.BadParameter(
                f"{source}: rank {rank} has {n_total} measurements; after discarding {num_mcmc_warmup_steps} "
                f"warmup steps {max(n_kept, 0)} remain, fewer than num_mcmc_bin_blocks = {num_mcmc_bin_blocks}."
            )


def _check_jackknife_inputs(w_L_binned: np.ndarray, *, source: str) -> None:
    """Every delete-one-bin weight sum must be positive and finite."""
    if w_L_binned.size < 2:
        raise typer.BadParameter(f"{source}: the jackknife needs at least 2 binned samples, got {w_L_binned.size}.")
    leave_one_out = np.real(np.sum(w_L_binned) - w_L_binned)
    if not np.all(np.isfinite(leave_one_out)) or np.any(leave_one_out <= 0.0):
        raise typer.BadParameter(f"{source}: a delete-one-bin weight sum is zero, negative or non-finite.")


def _check_energy_result(E_mean: float, E_std: float, *, source: str) -> None:
    """A non-finite final estimate is an error, never an output line."""
    if not (np.isfinite(E_mean) and np.isfinite(E_std)):
        raise typer.BadParameter(f"{source}: non-finite energy estimate E = {E_mean} +- {E_std}.")


def _check_lrdmc_analysis_floors(
    *,
    num_gfmc_bin_blocks: int,
    num_gfmc_warmup_steps: int,
    num_gfmc_collect_steps: int,
) -> None:
    """Refuse LRDMC analysis settings below the documented minima (exit status 1).

    Shared by ``lrdmc compute-energy``, ``compute-force`` and
    ``extrapolate-energy``; the minima are those enforced for the
    ``lrdmc-bra``/``lrdmc-tau`` inputs.
    """
    if num_gfmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
        typer.echo(
            f"Error: num_gfmc_warmup_steps ({num_gfmc_warmup_steps}) must be "
            f">= {GFMC_MIN_WARMUP_STEPS}. E_scf is unreliable before "
            f"step {GFMC_MIN_WARMUP_STEPS}.",
            err=True,
        )
        raise typer.Exit(code=1)
    if num_gfmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
        typer.echo(
            f"Error: num_gfmc_bin_blocks ({num_gfmc_bin_blocks}) must be >= {GFMC_MIN_BIN_BLOCKS}.",
            err=True,
        )
        raise typer.Exit(code=1)
    if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
        typer.echo(
            f"Error: num_gfmc_collect_steps ({num_gfmc_collect_steps}) must be >= {GFMC_MIN_COLLECT_STEPS}.",
            err=True,
        )
        raise typer.Exit(code=1)


def _check_lrdmc_binning_inputs(
    rank_data: list,
    *,
    num_gfmc_bin_blocks: int,
    num_gfmc_warmup_steps: int,
    num_gfmc_collect_steps: int,
    source: str,
) -> None:
    """LRDMC analogue of ``_check_mcmc_binning_inputs``.

    Every rank that holds samples (a merged checkpoint keeps the reduced
    history on rank 0 only) must keep at least one measurement per bin after
    the weight-history shift and the warmup, as ``_lrdmc_energy_estimate``
    requires for the energy.
    """
    if num_gfmc_bin_blocks < 1:
        raise typer.BadParameter(f"num_gfmc_bin_blocks = {num_gfmc_bin_blocks} must be >= 1.")
    if num_gfmc_warmup_steps < 0:
        raise typer.BadParameter(f"num_gfmc_warmup_steps = {num_gfmc_warmup_steps} must be >= 0.")
    if num_gfmc_collect_steps < 0:
        raise typer.BadParameter(f"num_gfmc_collect_steps = {num_gfmc_collect_steps} must be >= 0.")
    for rank, obs in enumerate(rank_data):
        raw_e_L = obs.get("e_L", np.empty(0))
        if raw_e_L.size == 0:
            continue
        n_total = int(np.shape(raw_e_L)[0])
        n_kept = n_total - num_gfmc_collect_steps - num_gfmc_warmup_steps
        if n_kept < num_gfmc_bin_blocks:
            raise typer.BadParameter(
                f"{source}: rank {rank} has {n_total} measurements; too few measurements after collect/warmup to form "
                f"nonempty bins ({max(n_kept, 0)} remain after {num_gfmc_collect_steps} collect and "
                f"{num_gfmc_warmup_steps} warmup steps, fewer than num_gfmc_bin_blocks = {num_gfmc_bin_blocks})."
            )


def _check_force_result(force_mean: np.ndarray, force_std: np.ndarray, *, source: str) -> None:
    """A non-finite force mean or error bar is an error, never a printed table."""
    bad = ~(np.isfinite(force_mean) & np.isfinite(force_std))
    if np.any(bad):
        raise typer.BadParameter(
            f"{source}: non-finite atomic force estimate in {int(np.count_nonzero(bad))} of {bad.size} components "
            "(a non-finite e_L, weight or force sample in the analysed window)."
        )


# Per-atom force observables, shape (steps, walkers, atoms, 3). Force
# post-processing reads them one bin block at a time (issue 949): holding them
# whole needs several times the checkpoint's observable size on long runs.
_PER_ATOM_FORCE_KEYS = ("force_HF", "force_PP", "E_L_force_PP")


def _streamed_weighted_force_bins(
    datasets: dict[str, h5py.Dataset],
    w_L: np.ndarray,
    *,
    shift: int,
    num_warmup_steps: int,
    num_bin_blocks: int,
) -> list[np.ndarray]:
    """Bin sums of ``w_L * F`` for every per-atom force observable of one rank.

    ``w_L`` is the analysed weight window, i.e. the rows ``[shift:][num_warmup_steps:]``
    of the stored history (``shift`` is 0 for MCMC and the collect length for
    LRDMC). For each observable ``F`` in ``_PER_ATOM_FORCE_KEYS`` the result is
    the ``(num_bin_blocks * walkers, atoms, 3)`` array of
    ``np.array_split(np.einsum("iw,iwjk->iwjk", w_L, F[shift:][num_warmup_steps:]), num_bin_blocks)``
    block sums over steps. Only one block of one observable is read from HDF5
    at a time: the blocks are those of ``np.array_split`` mapped back to
    absolute dataset rows, and each block gets the same einsum and step sum, so
    the values are bit-identical to the whole-array computation.
    """
    # np.array_split's section sizes: the first (rows % bins) blocks hold one extra row.
    num_rows = int(w_L.shape[0])
    each, extra = divmod(num_rows, num_bin_blocks)
    stops = list(itertools.accumulate([each + 1] * extra + [each] * (num_bin_blocks - extra)))
    bounds = list(zip([0, *stops[:-1]], stops, strict=True))
    binned = []
    for key in _PER_ATOM_FORCE_KEYS:
        dataset = datasets[key]
        rows = range(int(dataset.shape[0]))[shift:][num_warmup_steps:]
        if len(rows) != num_rows:
            raise ValueError(
                f"observables/{key} has {len(rows)} analysed rows but the weight window has {num_rows}; "
                "the checkpoint is inconsistent."
            )
        sums = np.asarray(
            [
                np.sum(np.einsum("iw,iwjk->iwjk", w_L[start:stop], dataset[rows.start + start : rows.start + stop]), axis=0)
                for start, stop in bounds
            ]
        )
        binned.append(sums.reshape((-1, *sums.shape[2:])))
    return binned


def _mcmc_energy_estimate(
    restart_chk: str,
    *,
    num_mcmc_bin_blocks: int,
    num_mcmc_warmup_steps: int,
) -> tuple[dict[str, float | None], int, int]:
    """Compute one merged-checkpoint MCMC energy estimate."""
    rank_data = load_observables_from_checkpoint(restart_chk, ["e_L", "w_L", "mpc_dV"])
    _check_mcmc_binning_inputs(
        rank_data, num_mcmc_bin_blocks=num_mcmc_bin_blocks, num_mcmc_warmup_steps=num_mcmc_warmup_steps, source=restart_chk
    )
    w_L_binned_list = []
    w_L_e_L_binned_list = []

    for obs in rank_data:
        e_L = obs["e_L"][num_mcmc_warmup_steps:]
        w_L = obs["w_L"][num_mcmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned_list += list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned_list += list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

    w_L_binned = np.asarray(w_L_binned_list)
    w_L_e_L_binned = np.asarray(w_L_e_L_binned_list)
    w_L_binned_sum = np.sum(w_L_binned)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)
    M = w_L_binned.size
    _check_jackknife_inputs(w_L_binned, source=restart_chk)
    E_jackknife_binned = np.asarray(
        [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )
    if np.iscomplexobj(E_jackknife_binned):
        E_jackknife_binned = np.real(E_jackknife_binned)
    _check_energy_result(
        float(np.average(E_jackknife_binned)), float(np.sqrt(M - 1) * np.std(E_jackknife_binned)), source=restart_chk
    )
    estimate: dict[str, float | None] = {
        "E": float(np.average(E_jackknife_binned)),
        "sigma_E": float(np.sqrt(M - 1) * np.std(E_jackknife_binned)),
        "dE_mpc": None,
        "sigma_dE_mpc": None,
        "E_mpc": None,
        "sigma_E_mpc": None,
    }

    mpc_dV_per_rank = [obs.get("mpc_dV") for obs in rank_data]
    if mpc_dV_per_rank and all(arr is not None and arr.size != 0 for arr in mpc_dV_per_rank):
        w_L_mpc_dV_binned_list = []
        for obs in rank_data:
            mpc_dV = obs["mpc_dV"][num_mcmc_warmup_steps:]
            w_L = obs["w_L"][num_mcmc_warmup_steps:]
            split = np.array_split(w_L * mpc_dV, num_mcmc_bin_blocks, axis=0)
            w_L_mpc_dV_binned_list += list(np.ravel([np.sum(arr, axis=0) for arr in split]))
        w_L_mpc_dV_binned = np.asarray(w_L_mpc_dV_binned_list)
        w_L_mpc_dV_binned_sum = np.sum(w_L_mpc_dV_binned)
        dE_mpc_jackknife_binned = np.asarray(
            [(w_L_mpc_dV_binned_sum - w_L_mpc_dV_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
        )
        if np.iscomplexobj(dE_mpc_jackknife_binned):
            dE_mpc_jackknife_binned = np.real(dE_mpc_jackknife_binned)
        estimate["dE_mpc"] = float(np.average(dE_mpc_jackknife_binned))
        estimate["sigma_dE_mpc"] = float(np.sqrt(M - 1) * np.std(dE_mpc_jackknife_binned))
        estimate["E_mpc"] = float(estimate["E"] + estimate["dE_mpc"])
        estimate["sigma_E_mpc"] = float(np.sqrt(M - 1) * np.std(E_jackknife_binned + dE_mpc_jackknife_binned))

    return estimate, len(rank_data), M


def _combine_twist_energy_estimates(estimates: list[dict[str, float | None]], weights: np.ndarray) -> dict[str, float | None]:
    """Combine independent per-twist estimates with normalized BZ weights."""

    def combine(mean_key: str, sigma_key: str) -> tuple[float | None, float | None]:
        means = [estimate[mean_key] for estimate in estimates]
        sigmas = [estimate[sigma_key] for estimate in estimates]
        if all(value is None for value in means) and all(value is None for value in sigmas):
            return None, None
        if any(value is None for value in means) or any(value is None for value in sigmas):
            raise ValueError(f"Per-twist observable {mean_key} is only partially present; refusing an inconsistent BZ average.")
        mean = float(np.dot(weights, np.asarray(means, dtype=np.float64)))
        sigma = float(np.sqrt(np.dot(weights**2, np.square(np.asarray(sigmas, dtype=np.float64)))))
        return mean, sigma

    E, sigma_E = combine("E", "sigma_E")
    dE_mpc, sigma_dE_mpc = combine("dE_mpc", "sigma_dE_mpc")
    E_mpc, sigma_E_mpc = combine("E_mpc", "sigma_E_mpc")
    return {
        "E": E,
        "sigma_E": sigma_E,
        "dE_mpc": dE_mpc,
        "sigma_dE_mpc": sigma_dE_mpc,
        "E_mpc": E_mpc,
        "sigma_E_mpc": sigma_E_mpc,
    }


def _echo_energy_estimate(estimate: dict[str, float | None]) -> None:
    """Print the stable jqmc-tool energy output consumed by workflows."""
    typer.echo(f"E = {estimate['E']} +- {estimate['sigma_E']} Ha.")
    if estimate["dE_mpc"] is not None:
        typer.echo(f"MPC correction: dE_MPC = {estimate['dE_mpc']} +- {estimate['sigma_dE_mpc']} Ha.")
        typer.echo(f"Total Energy (MPC-corrected): E_MPC = {estimate['E_mpc']} +- {estimate['sigma_E_mpc']} Ha.")


def _twist_force_estimate(
    checkpoint_path: str,
    *,
    num_bin_blocks: int,
    num_warmup_steps: int,
    num_collect_steps: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Force jackknife of one twist checkpoint with the requested analysis options.

    Same estimator and refusals as the single-file ``compute-force`` commands:
    ``num_collect_steps is None`` selects the MCMC window, otherwise the LRDMC
    window (``num_collect_steps`` rows shifted out, accumulated weights).
    """
    rank_data = load_observables_from_checkpoint(checkpoint_path, ["e_L", "w_L"])
    if num_collect_steps is None:
        _check_mcmc_binning_inputs(
            rank_data, num_mcmc_bin_blocks=num_bin_blocks, num_mcmc_warmup_steps=num_warmup_steps, source=checkpoint_path
        )
    else:
        _check_lrdmc_binning_inputs(
            rank_data,
            num_gfmc_bin_blocks=num_bin_blocks,
            num_gfmc_warmup_steps=num_warmup_steps,
            num_gfmc_collect_steps=num_collect_steps,
            source=checkpoint_path,
        )

    def binned(values: np.ndarray) -> np.ndarray:
        # (bins, walkers, ...) -> (bins * walkers, ...), the single-file replica order.
        sums = np.asarray([np.sum(arr, axis=0) for arr in np.array_split(values, num_bin_blocks, axis=0)])
        return sums.reshape((-1, *sums.shape[2:]))

    replicas: list[list[np.ndarray]] = [[], [], [], [], []]
    contributing = [rank for rank, obs in enumerate(rank_data) if obs.get("e_L", np.empty(0)).size != 0]
    with open_observable_datasets(checkpoint_path, _PER_ATOM_FORCE_KEYS) as force_datasets:
        for rank in contributing:
            if any(key not in force_datasets[rank] or force_datasets[rank][key].size == 0 for key in _PER_ATOM_FORCE_KEYS):
                raise typer.BadParameter(f"{checkpoint_path}: the checkpoint holds no force samples (atomic_force was off).")
        if num_collect_steps is not None:
            # One common power-of-two scale for every rank entering this estimate.
            lrdmc_w_L = compute_retained_accumulated_weights(
                [rank_data[rank]["w_L"] for rank in contributing], num_collect_steps, num_warmup_steps
            )
        for index, rank in enumerate(contributing):
            obs = rank_data[rank]
            raw_e_L = obs["e_L"]
            if num_collect_steps is None:
                shift = 0
                w_L = obs["w_L"][num_warmup_steps:]
            else:
                shift = num_collect_steps
                w_L = lrdmc_w_L[index]
            e_L = raw_e_L[shift:][num_warmup_steps:]
            replicas[0].append(binned(w_L))
            replicas[1].append(binned(w_L * e_L))
            force_binned = _streamed_weighted_force_bins(
                force_datasets[rank], w_L, shift=shift, num_warmup_steps=num_warmup_steps, num_bin_blocks=num_bin_blocks
            )
            for slot, values in zip(range(2, 5), force_binned, strict=True):
                replicas[slot].append(values)
    w_L_binned, w_L_e_L_binned, w_L_force_HF_binned, w_L_force_PP_binned, w_L_E_L_force_PP_binned = (
        np.concatenate(parts, axis=0) if parts else np.empty(0) for parts in replicas
    )
    _check_jackknife_inputs(w_L_binned, source=checkpoint_path)
    M = w_L_binned.size
    leave_one_out_w = (np.sum(w_L_binned) - w_L_binned)[:, None, None]
    mean_e_L = (np.sum(w_L_e_L_binned) - w_L_e_L_binned)[:, None, None] / leave_one_out_w
    mean_force_HF = (np.sum(w_L_force_HF_binned, axis=0) - w_L_force_HF_binned) / leave_one_out_w
    mean_force_PP = (np.sum(w_L_force_PP_binned, axis=0) - w_L_force_PP_binned) / leave_one_out_w
    mean_E_L_force_PP = (np.sum(w_L_E_L_force_PP_binned, axis=0) - w_L_E_L_force_PP_binned) / leave_one_out_w
    # Real projection per replica, as in the single-file commands and the drivers' get_aF.
    force_jn = np.real(-mean_force_HF - 2.0 * (mean_E_L_force_PP - mean_e_L * mean_force_PP))
    force_mean = np.average(force_jn, axis=0)
    force_std = np.sqrt(M - 1) * np.std(force_jn, axis=0)
    _check_force_result(force_mean, force_std, source=checkpoint_path)
    return force_mean, force_std


def _twist_force_pressure_summary(
    restart_chk: str,
    *,
    num_bin_blocks: int,
    num_warmup_steps: int,
    num_collect_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, list[str]]:
    """Combine per-twist force estimates and stored pressure summaries for a twist directory.

    The force of every twist is re-estimated from that twist's stored
    histories with the requested ``-b/-w`` (and LRDMC ``-c``) options and
    combined with the embedded normalized BZ weights, as ``compute-energy``
    does for the energy. The hydrostatic pressure has no tool-side estimator:
    it combines the per-twist summaries the run stored at its own settings,
    and non-finite ones are refused. A stress-only run (``atomic_force =
    false``, ``compute_stress = true``) stores pressure summaries without
    force summaries; it reports the pressure and returns ``None`` for the
    force arrays.
    """
    manifest = load_twist_checkpoint_manifest(restart_chk)
    summaries = [load_twist_checkpoint_summary(restart_chk, idx) for idx in range(len(manifest))]
    weights = np.asarray([weight for _path, weight in manifest], dtype=np.float64)
    required_force = {"F_k", "sigma_F_k"}
    has_force = [required_force <= summary.keys() for summary in summaries]
    if any(has_force) != all(has_force):
        raise ValueError(f"{restart_chk}: force summaries are only partially present.")
    force_mean = force_std = None
    if all(has_force):
        per_twist = [
            _twist_force_estimate(
                checkpoint_path,
                num_bin_blocks=num_bin_blocks,
                num_warmup_steps=num_warmup_steps,
                num_collect_steps=num_collect_steps,
            )
            for checkpoint_path, _weight in manifest
        ]
        per_twist_F = np.asarray([mean for mean, _std in per_twist], dtype=np.float64)
        per_twist_sigma_F = np.asarray([std for _mean, std in per_twist], dtype=np.float64)
        force_mean = np.tensordot(weights, per_twist_F, axes=(0, 0))
        force_std = np.sqrt(np.tensordot(weights**2, per_twist_sigma_F**2, axes=(0, 0)))
        _check_force_result(force_mean, force_std, source=restart_chk)

    pressure_keys = {"P_hydro_k", "sigma_P_hydro_k"}
    has_pressure = [pressure_keys <= summary.keys() for summary in summaries]
    if any(has_pressure) != all(has_pressure):
        raise ValueError(f"{restart_chk}: hydrostatic pressure summaries are only partially present.")
    pressure_mean = pressure_std = None
    if all(has_pressure):
        per_twist_P = np.asarray([summary["P_hydro_k"] for summary in summaries], dtype=np.float64)
        per_twist_sigma_P = np.asarray([summary["sigma_P_hydro_k"] for summary in summaries], dtype=np.float64)
        if not (np.all(np.isfinite(per_twist_P)) and np.all(np.isfinite(per_twist_sigma_P))):
            raise typer.BadParameter(f"{restart_chk}: non-finite per-twist hydrostatic pressure summary.")
        pressure_mean = float(np.dot(weights, per_twist_P))
        pressure_std = float(np.sqrt(np.dot(weights**2, per_twist_sigma_P**2)))
    if force_mean is None and pressure_mean is None:
        raise ValueError(
            f"{restart_chk}: the twists have neither force nor pressure summaries "
            "(run with atomic_force = true and/or compute_stress = true)."
        )

    atomic_labels: list[str] = []
    if force_mean is not None:
        hamiltonian_data = load_hamiltonian_from_checkpoint(manifest[0][0])
        atomic_labels = [str(label) for label in hamiltonian_data.structure_data.atomic_labels]
    return force_mean, force_std, pressure_mean, pressure_std, atomic_labels


def _echo_force_pressure_summary(
    force_mean: np.ndarray | None,
    force_std: np.ndarray | None,
    pressure_mean: float | None,
    pressure_std: float | None,
    atomic_labels: list[str],
) -> None:
    """Print force output (when present) plus the covariance-aware hydrostatic pressure."""
    if force_mean is not None:
        typer.echo("Atomic Forces:")
        sep = 16 * 3
        typer.echo("-" * sep)
        typer.echo("Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
        typer.echo("-" * sep)
        for atom_idx, atomic_label in enumerate(atomic_labels):
            row_values = [f"{ufloat(force_mean[atom_idx, axis], force_std[atom_idx, axis]):+2uS}" for axis in range(3)]
            typer.echo(atomic_label.ljust(8) + " ".join(value.ljust(12) for value in row_values))
        typer.echo("-" * sep)
    if pressure_mean is not None:
        typer.echo("Hydrostatic Pressure:")
        typer.echo("(per-twist pressure summaries stored by the run at its own warmup/bin settings; -b/-w/-c do not apply)")
        typer.echo(f"P_hydro = {pressure_mean} +- {pressure_std} Ha/bohr^3.")


@mcmc_app.command("compute-energy")
def mcmc_compute_energy(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. restart.h5"),
    num_mcmc_bin_blocks: int = typer.Option(
        1,
        "-b",
        "--num_mcmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_mcmc_warmup_steps: int = typer.Option(
        0, "-w", "--num_mcmc_warmup_steps", help="Number of observable measurement steps for warmup (i.e., discarded)."
    ),
) -> None:
    """VMC energy calculation."""
    typer.echo(f"Read restart checkpoint file(s) from {restart_chk}.")

    _require_checkpoint_driver(restart_chk, app="mcmc", command="compute-energy")
    if os.path.isdir(restart_chk):
        _require_consistent_twist_directory(restart_chk)
        manifest = load_twist_checkpoint_manifest(restart_chk)
        estimates = []
        for checkpoint_path, _weight in manifest:
            estimate, _num_ranks, _num_samples = _mcmc_energy_estimate(
                checkpoint_path,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            )
            estimates.append(estimate)
        weights = np.asarray([weight for _path, weight in manifest], dtype=np.float64)
        typer.echo(f"Found {len(manifest)} twist checkpoints.")
        _echo_energy_estimate(_combine_twist_energy_estimates(estimates, weights))
        return

    """Load observables from the HDF5 checkpoint."""
    rank_data = load_observables_from_checkpoint(restart_chk, ["e_L", "w_L", "mpc_dV"])
    num_ranks = len(rank_data)
    typer.echo(f"Found {num_ranks} MPI ranks.")
    _check_mcmc_binning_inputs(
        rank_data, num_mcmc_bin_blocks=num_mcmc_bin_blocks, num_mcmc_warmup_steps=num_mcmc_warmup_steps, source=restart_chk
    )

    w_L_binned_list = []
    w_L_e_L_binned_list = []

    for obs in rank_data:
        e_L = obs["e_L"][num_mcmc_warmup_steps:]
        w_L = obs["w_L"][num_mcmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
        w_L_binned_list += w_L_binned
        w_L_e_L_binned_list += w_L_e_L_binned

    w_L_binned = np.array(w_L_binned_list)
    w_L_e_L_binned = np.array(w_L_e_L_binned_list)

    # jackknife implementation
    w_L_binned_sum = np.sum(w_L_binned)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)

    M = w_L_binned.size
    typer.echo(f"Total number of binned samples = {M}")
    _check_jackknife_inputs(w_L_binned, source=restart_chk)

    E_jackknife_binned = np.array(
        [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )

    # Project complex ``E`` onto the physical real part before the mean/std
    # (the imaginary part is a contamination diagnostic, not an observable).
    # Without this projection the ``typer.echo`` would format complex numpy
    # scalars as ``(-44.20581+0j)``, which the workflow regex parser
    # in ``jqmc_workflow.mcmc_workflow`` cannot match.
    if np.iscomplexobj(E_jackknife_binned):
        E_jackknife_binned = np.real(E_jackknife_binned)

    E_mean = float(np.average(E_jackknife_binned))
    E_std = float(np.sqrt(M - 1) * np.std(E_jackknife_binned))
    _check_energy_result(E_mean, E_std, source=restart_chk)

    typer.echo(f"E = {E_mean} +- {E_std} Ha.")

    # MPC finite-size correction (energy-only observable, ``physical = no``):
    # E_MPC = E_Ewald + <dV> with dV = V_ee^MPC - V_ee^Ewald. The key is absent
    # (old checkpoints) or has zero rows when the run used ``mpc=False``; that is
    # the normal case and must stay silent. Auto-detected -- no CLI flag needed,
    # since the checkpoint already records whether MPC was accumulated.
    mpc_dV_per_rank = [obs.get("mpc_dV") for obs in rank_data]
    if mpc_dV_per_rank and all(arr is not None and arr.size != 0 for arr in mpc_dV_per_rank):
        # Same weights, same binning and same jackknife as ``E`` above: only the
        # numerator changes, so ``w_L_binned`` / ``w_L_binned_sum`` / ``M`` are reused.
        w_L_mpc_dV_binned_list = []

        for obs in rank_data:
            mpc_dV = obs["mpc_dV"][num_mcmc_warmup_steps:]
            w_L = obs["w_L"][num_mcmc_warmup_steps:]
            w_L_mpc_dV_split = np.array_split(w_L * mpc_dV, num_mcmc_bin_blocks, axis=0)
            w_L_mpc_dV_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_mpc_dV_split]))
            w_L_mpc_dV_binned_list += w_L_mpc_dV_binned

        w_L_mpc_dV_binned = np.array(w_L_mpc_dV_binned_list)
        w_L_mpc_dV_binned_sum = np.sum(w_L_mpc_dV_binned)

        dE_mpc_jackknife_binned = np.array(
            [(w_L_mpc_dV_binned_sum - w_L_mpc_dV_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
        )

        # ``dV`` is real at any twist (difference of two real e-e potentials), but
        # ``w_L`` may be complex, so project as for ``E`` before formatting.
        if np.iscomplexobj(dE_mpc_jackknife_binned):
            dE_mpc_jackknife_binned = np.real(dE_mpc_jackknife_binned)

        dE_mpc_mean = float(np.average(dE_mpc_jackknife_binned))
        dE_mpc_std = float(np.sqrt(M - 1) * np.std(dE_mpc_jackknife_binned))

        typer.echo(f"MPC correction: dE_MPC = {dE_mpc_mean} +- {dE_mpc_std} Ha.")
        # E and dE_MPC are jackknife ratios over the same walkers and weights.
        # Independent quadrature under-reports their correlated error, so form
        # E_jk + dV_jk and take its jackknife error directly.
        E_mpc_jackknife_binned = E_jackknife_binned + dE_mpc_jackknife_binned
        E_mpc_std = float(np.sqrt(M - 1) * np.std(E_mpc_jackknife_binned))
        typer.echo(f"Total Energy (MPC-corrected): E_MPC = {E_mean + dE_mpc_mean} +- {E_mpc_std} Ha.")


@mcmc_app.command("compute-force")
def mcmc_compute_force(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. restart.h5"),
    num_mcmc_bin_blocks: int = typer.Option(
        1,
        "-b",
        "--num_mcmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_mcmc_warmup_steps: int = typer.Option(
        0, "-w", "--num_mcmc_warmup_steps", help="Number of observable measurement steps for warmup (i.e., discarded)."
    ),
) -> None:
    """VMC atomic force calculation."""
    typer.echo(f"Read restart checkpoint file(s) from {restart_chk}.")

    _require_checkpoint_driver(restart_chk, app="mcmc", command="compute-force")
    if os.path.isdir(restart_chk):
        _require_consistent_twist_directory(restart_chk)
        _echo_force_pressure_summary(
            *_twist_force_pressure_summary(
                restart_chk, num_bin_blocks=num_mcmc_bin_blocks, num_warmup_steps=num_mcmc_warmup_steps
            )
        )
        return

    """Load the 2-D observables; the per-atom force observables are streamed by bin block below."""
    rank_data = load_observables_from_checkpoint(restart_chk, ["e_L", "w_L"])
    num_ranks = len(rank_data)
    typer.echo(f"Found {num_ranks} MPI ranks.")

    # Check that force computation was enabled
    cfg = load_driver_config_from_checkpoint(restart_chk, rank=0)
    if not cfg.get("comput_position_deriv", False):
        typer.echo("Error: atomic_force was not enabled during the MCMC run. Cannot compute forces.")
        raise typer.Exit(code=1)
    _check_mcmc_binning_inputs(
        rank_data, num_mcmc_bin_blocks=num_mcmc_bin_blocks, num_mcmc_warmup_steps=num_mcmc_warmup_steps, source=restart_chk
    )

    hamiltonian_data = load_hamiltonian_from_checkpoint(restart_chk)
    atomic_labels = hamiltonian_data.structure_data.atomic_labels

    w_L_binned_list = []
    w_L_e_L_binned_list = []
    w_L_force_HF_binned_list = []
    w_L_force_PP_binned_list = []
    w_L_E_L_force_PP_binned_list = []

    with open_observable_datasets(restart_chk, _PER_ATOM_FORCE_KEYS) as force_datasets:
        for obs, datasets in zip(rank_data, force_datasets, strict=True):
            e_L = obs["e_L"][num_mcmc_warmup_steps:]
            w_L = obs["w_L"][num_mcmc_warmup_steps:]

            # split and binning with multiple walkers
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)

            # binned sum
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_force_HF_binned, w_L_force_PP_binned, w_L_E_L_force_PP_binned = (
                list(values)
                for values in _streamed_weighted_force_bins(
                    datasets, w_L, shift=0, num_warmup_steps=num_mcmc_warmup_steps, num_bin_blocks=num_mcmc_bin_blocks
                )
            )

            w_L_binned_list += w_L_binned
            w_L_e_L_binned_list += w_L_e_L_binned
            w_L_force_HF_binned_list += w_L_force_HF_binned
            w_L_force_PP_binned_list += w_L_force_PP_binned
            w_L_E_L_force_PP_binned_list += w_L_E_L_force_PP_binned

    w_L_binned = np.array(w_L_binned_list)
    w_L_e_L_binned = np.array(w_L_e_L_binned_list)
    w_L_force_HF_binned = np.array(w_L_force_HF_binned_list)
    w_L_force_PP_binned = np.array(w_L_force_PP_binned_list)
    w_L_E_L_force_PP_binned = np.array(w_L_E_L_force_PP_binned_list)

    # jackknife implementation
    w_L_binned_sum = np.sum(w_L_binned, axis=0)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned, axis=0)
    w_L_force_HF_binned_sum = np.sum(w_L_force_HF_binned, axis=0)
    w_L_force_PP_binned_sum = np.sum(w_L_force_PP_binned, axis=0)
    w_L_E_L_force_PP_binned_sum = np.sum(w_L_E_L_force_PP_binned, axis=0)

    M = w_L_binned.size
    typer.echo(f"Total number of binned samples = {M}")
    _check_jackknife_inputs(w_L_binned, source=restart_chk)

    force_HF_jn = -1.0 * np.array(
        [(w_L_force_HF_binned_sum - w_L_force_HF_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )

    force_Pulay_jn = -2.0 * np.array(
        [
            (
                (w_L_E_L_force_PP_binned_sum - w_L_E_L_force_PP_binned[m]) / (w_L_binned_sum - w_L_binned[m])
                - (
                    (w_L_e_L_binned_sum - w_L_e_L_binned[m])
                    / (w_L_binned_sum - w_L_binned[m])
                    * (w_L_force_PP_binned_sum - w_L_force_PP_binned[m])
                    / (w_L_binned_sum - w_L_binned[m])
                )
            )
            for m in range(M)
        ]
    )

    # The atomic force is a real observable. At a complex Bloch twist e_L (and
    # so E_L * force_PP) is complex and each jackknife sample carries an
    # imaginary part that vanishes only in expectation. Project every sample
    # onto its real part BEFORE the mean and the variance, as the drivers'
    # get_aF do; otherwise the error includes the imaginary fluctuation.
    # Identity on the real (Gamma / open-boundary) path.
    force_jn = np.real(force_HF_jn + force_Pulay_jn)

    force_mean = np.average(force_jn, axis=0)
    force_std = np.sqrt(M - 1) * np.std(force_jn, axis=0)
    _check_force_result(force_mean, force_std, source=restart_chk)

    typer.echo("Atomic Forces:")
    sep = 16 * 3
    typer.echo("-" * sep)
    typer.echo("Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
    typer.echo("-" * sep)
    for i in range(len(atomic_labels)):
        atomic_label = str(atomic_labels[i])
        row_values = [f"{ufloat(force_mean[i, j], force_std[i, j]):+2uS}" for j in range(3)]
        row_str = atomic_label.ljust(8) + " ".join(val.ljust(12) for val in row_values)
        typer.echo(row_str)
    typer.echo("-" * sep)


@mcmc_app.command("generate-input")
def mcmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for MCMC calculations."),
    filename: str = typer.Option("mcmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
) -> None:
    """Generate an input file for MCMC calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                control_table[key] = str(value)
            else:
                control_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "mcmc"
        doc.add("control", control_table)

        mcmc_table = tomlkit.table()
        for key, value in cli_parameters["mcmc"].items():
            if value is None:
                mcmc_table[key] = str(value)
            else:
                mcmc_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                mcmc_table[key].comment(cli_parameters["mcmc_comments"][key])
        doc.add("mcmc", mcmc_table)

        coulomb_table = tomlkit.table()
        for key, value in cli_parameters["coulomb"].items():
            if value is None:
                coulomb_table[key] = str(value)
            else:
                coulomb_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                coulomb_table[key].comment(cli_parameters["coulomb_comments"][key])
        doc.add("coulomb", coulomb_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_mcmc = typer.main.get_command(mcmc_app)

_cli.add_command(typer_click_mcmc, "mcmc")


# LRDMC_app
lrdmc_app = typer.Typer(help="Pre- and Post-Processing for LRDMC calculations.")


def _lrdmc_energy_estimate(
    restart_chk: str,
    *,
    num_gfmc_bin_blocks: int,
    num_gfmc_warmup_steps: int,
    num_gfmc_collect_steps: int,
) -> tuple[dict[str, float | None], int, int]:
    """Compute one merged-checkpoint LRDMC energy estimate."""
    if num_gfmc_bin_blocks < 2 or num_gfmc_warmup_steps < 0 or num_gfmc_collect_steps < 1:
        raise ValueError("LRDMC energy requires at least two bins, nonnegative warmup, and positive collect steps.")
    rank_data = load_observables_from_checkpoint(restart_chk, ["e_L", "w_L", "mpc_dV"])
    w_L_binned_list = []
    w_L_e_L_binned_list = []
    k = num_gfmc_collect_steps

    contributing = [obs for obs in rank_data if obs.get("e_L", np.empty(0)).size != 0]
    for obs in contributing:
        raw_e_L = obs["e_L"]
        if raw_e_L.shape != obs["w_L"].shape or raw_e_L.ndim != 2:
            raise ValueError(f"{restart_chk}: e_L and w_L must have the same (steps, walkers) shape.")
        if raw_e_L.shape[0] - k - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
            raise ValueError(f"{restart_chk}: too few measurements after collect/warmup to form nonempty bins.")
    # One common power-of-two scale for every rank entering this estimate.
    retained_w_L = compute_retained_accumulated_weights([obs["w_L"] for obs in contributing], k, num_gfmc_warmup_steps)

    for obs, w_L in zip(contributing, retained_w_L, strict=True):
        e_L = obs["e_L"][k:][num_gfmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_gfmc_bin_blocks, axis=0)
        w_L_binned_list += list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        split = np.array_split(w_L * e_L, num_gfmc_bin_blocks, axis=0)
        w_L_e_L_binned_list += list(np.ravel([np.sum(arr, axis=0) for arr in split]))

    w_L_binned = np.asarray(w_L_binned_list)
    w_L_e_L_binned = np.asarray(w_L_e_L_binned_list)
    w_L_binned_sum = np.sum(w_L_binned)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned)
    M = w_L_binned.size
    if M < 2 or not np.all(np.isfinite(w_L_binned)) or np.any(w_L_binned_sum - w_L_binned <= 0):
        raise ValueError(f"{restart_chk}: insufficient finite, positive-weight jackknife samples.")
    E_jackknife_binned = np.asarray(
        [(w_L_e_L_binned_sum - w_L_e_L_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )
    if np.iscomplexobj(E_jackknife_binned):
        E_jackknife_binned = np.real(E_jackknife_binned)
    estimate: dict[str, float | None] = {
        "E": float(np.average(E_jackknife_binned)),
        "sigma_E": float(np.sqrt(M - 1) * np.std(E_jackknife_binned)),
        "dE_mpc": None,
        "sigma_dE_mpc": None,
        "E_mpc": None,
        "sigma_E_mpc": None,
    }
    if not np.isfinite(estimate["E"]) or not np.isfinite(estimate["sigma_E"]):
        raise ValueError(f"{restart_chk}: non-finite LRDMC energy or error bar.")

    # Restricted to ranks that actually hold e_L (see lrdmc_compute_energy):
    # a merged checkpoint stores the already-MPI-reduced observable history
    # under rank_0 only, so every other rank's dict is empty. Gating on ALL
    # ranks (unfiltered) always saw those empty dicts and silently reported
    # "no MPC", even for a genuine mpc=True twist.
    mpc_dV_per_rank = [obs.get("mpc_dV") for obs in rank_data if obs.get("e_L", np.empty(0)).size != 0]
    has_mpc = [arr is not None and arr.size != 0 for arr in mpc_dV_per_rank]
    if any(has_mpc) and not all(has_mpc):
        raise ValueError(f"{restart_chk}: missing mpc_dV on a rank contributing energy samples.")
    if any(has_mpc):
        w_L_mpc_dV_binned_list = []
        for obs, w_L in zip(contributing, retained_w_L, strict=True):
            if obs["mpc_dV"].shape != obs["e_L"].shape:
                raise ValueError(f"{restart_chk}: mpc_dV must have the same (steps, walkers) shape as e_L.")
            mpc_dV = obs["mpc_dV"][k:][num_gfmc_warmup_steps:]
            if not np.all(np.isfinite(mpc_dV)):
                raise ValueError(f"{restart_chk}: non-finite MPC correction samples.")
            split = np.array_split(w_L * mpc_dV, num_gfmc_bin_blocks, axis=0)
            w_L_mpc_dV_binned_list += list(np.ravel([np.sum(arr, axis=0) for arr in split]))
        w_L_mpc_dV_binned = np.asarray(w_L_mpc_dV_binned_list)
        w_L_mpc_dV_binned_sum = np.sum(w_L_mpc_dV_binned)
        dE_mpc_jackknife_binned = np.asarray(
            [(w_L_mpc_dV_binned_sum - w_L_mpc_dV_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
        )
        if np.iscomplexobj(dE_mpc_jackknife_binned):
            dE_mpc_jackknife_binned = np.real(dE_mpc_jackknife_binned)
        estimate["dE_mpc"] = float(np.average(dE_mpc_jackknife_binned))
        estimate["sigma_dE_mpc"] = float(np.sqrt(M - 1) * np.std(dE_mpc_jackknife_binned))
        estimate["E_mpc"] = float(estimate["E"] + estimate["dE_mpc"])
        estimate["sigma_E_mpc"] = float(np.sqrt(M - 1) * np.std(E_jackknife_binned + dE_mpc_jackknife_binned))
        if any(not np.isfinite(estimate[key]) for key in ("dE_mpc", "sigma_dE_mpc", "E_mpc", "sigma_E_mpc")):
            raise ValueError(f"{restart_chk}: non-finite MPC energy or error bar.")

    return estimate, len(rank_data), M


def _lrdmc_extrapolation_inputs(restart_chks: list[str], polynomial_order: int) -> list[tuple[float, list[tuple[str, float]]]]:
    """Validate one complete, comparable energy input per lattice spacing.

    Twist identity and weights are checkpoint metadata, not noisy estimates:
    compare them exactly, independent of file-index order. Legacy flat files
    remain supported, but must not be mixed with twist-averaged directories.
    """
    if polynomial_order < 0:
        raise ValueError("polynomial order must be nonnegative.")
    directory_flags = [os.path.isdir(path) for path in restart_chks]
    if any(directory_flags) and not all(directory_flags):
        raise ValueError("cannot mix twist checkpoint directories and single-checkpoint files in one extrapolation.")
    twist_mode = any(directory_flags)
    inputs = []
    reference_twists = None
    for restart_chk in restart_chks:
        manifest = load_twist_checkpoint_manifest(restart_chk) if twist_mode else [(restart_chk, 1.0)]
        alat = None
        twists = []
        for checkpoint_path, _weight in manifest:
            with h5py.File(checkpoint_path, "r") as f:
                ranks = [key for key in f if key.startswith("rank_")]
                if not ranks:
                    raise ValueError(f"{checkpoint_path}: no rank driver_config containing alat.")
                for rank in ranks:
                    cfg = f[f"{rank}/driver_config"].attrs
                    raw_alat = cfg.get("alat", np.nan)
                    if np.ndim(raw_alat) != 0 or np.iscomplexobj(raw_alat):
                        raise ValueError(f"{checkpoint_path}/{rank}: expected a positive finite alat.")
                    rank_alat = float(raw_alat)
                    squared_alat = rank_alat * rank_alat
                    if not np.isfinite(squared_alat) or squared_alat <= 0 or rank_alat <= 0:
                        raise ValueError(f"{checkpoint_path}/{rank}: expected a positive finite alat.")
                    if alat is not None and rank_alat != alat:
                        raise ValueError(f"{restart_chk}: all ranks and twists must have the same alat.")
                    alat = rank_alat
                if twist_mode:
                    raw_k_point = np.asarray(f["_meta"].attrs.get("k_point", np.empty(0)))
                    if (
                        raw_k_point.shape != (3,)
                        or not np.issubdtype(raw_k_point.dtype, np.number)
                        or np.iscomplexobj(raw_k_point)
                        or not np.all(np.isfinite(raw_k_point))
                    ):
                        raise ValueError(f"{checkpoint_path}: a finite three-component k_point is required for extrapolation.")
                    twists.append((*map(float, raw_k_point), float(f["_meta"].attrs["normalized_twist_weight"])))
        if twist_mode:
            signature = sorted(twists)
            # Normalize in k-point order, not file-index order. Summation
            # round-off must not make a reindexed copy look like a new grid.
            total_weight = sum(row[3] for row in signature)
            if not np.isfinite(total_weight) or total_weight <= 0:
                raise ValueError(f"{restart_chk}: twist weights must have a positive finite sum.")
            signature = [(*row[:3], row[3] / total_weight) for row in signature]
            if len({row[:3] for row in signature}) != len(signature):
                raise ValueError(f"{restart_chk}: duplicate k_point metadata in twist checkpoint set.")
            if reference_twists is not None and signature != reference_twists:
                raise ValueError(f"{restart_chk}: every lattice spacing must use the same twist set and BZ weights.")
            reference_twists = signature
        inputs.append((alat, manifest))

    spacings_squared = [alat**2 for alat, _manifest in inputs]
    if len(set(spacings_squared)) != len(spacings_squared):
        raise ValueError("duplicate lattice spacing: provide exactly one checkpoint file or twist directory per distinct a^2.")
    if len(inputs) < polynomial_order + 1:
        raise ValueError(
            f"polynomial order {polynomial_order} needs at least {polynomial_order + 1} distinct lattice "
            f"spacings, but only {len(inputs)} checkpoint(s) were given. Pass a lower "
            f"--polynomial-order (e.g. -p {max(len(inputs) - 1, 1)}) or add checkpoints."
        )
    return inputs


def _lrdmc_extrapolation_mpc_cutoff(inputs: list[tuple[float, list[tuple[str, float]]]]) -> float | None:
    """Require complete MPC histories and one cutoff, or an entirely non-MPC scan.

    A driver-config cutoff alone is not an enable flag: GFMC writes its default
    even when MPC is off. Empty, noncontributing MPI ranks need no histories.
    Missing legacy enable flags are allowed, but MPC extrapolation requires a
    cutoff in every rank's driver_config; optional twist-summary cutoffs must
    agree. Metadata are compared exactly, not with a numerical tolerance.
    """
    enabled = []
    cutoff = None
    for _a, manifest in inputs:
        for checkpoint_path, _weight in manifest:
            with h5py.File(checkpoint_path, "r") as f:
                ranks = [f[key] for key in f if key.startswith("rank_")]
                configs = [rank["driver_config"].attrs for rank in ranks]
                flags = []
                for cfg in configs:
                    if "mpc" in cfg:
                        flag = np.asarray(cfg["mpc"])
                        if flag.ndim != 0 or flag.dtype.kind not in "biu" or flag.item() not in (0, 1):
                            raise ValueError(f"{checkpoint_path}: mpc flag must be a scalar boolean.")
                        flags.append(bool(flag))
                histories = []
                for rank in ranks:
                    obs = rank.get("observables", {})
                    if "e_L" in obs and obs["e_L"].size:
                        histories.append("mpc_dV" in obs and obs["mpc_dV"].size > 0)
                meta = f["_meta"].attrs if "_meta" in f else {}
                has_summary = any(key in meta for key in ("mpc_cutoff", "dE_mpc_k", "E_mpc_k"))
                has_mpc = any(flags) or any(histories) or has_summary
                enabled.append(has_mpc)
                if not has_mpc:
                    continue
                if any(not flag for flag in flags):
                    raise ValueError(f"{checkpoint_path}: MPC data or enabled ranks conflict with mpc=False.")
                if not histories or not all(histories):
                    raise ValueError(f"{checkpoint_path}: missing mpc_dV on a rank contributing energy samples.")
                if any("mpc_cutoff" not in cfg for cfg in configs):
                    raise ValueError(f"{checkpoint_path}: missing driver_config mpc_cutoff for MPC extrapolation.")
                cutoffs = [cfg["mpc_cutoff"] for cfg in configs]
                if "mpc_cutoff" in meta:
                    cutoffs.append(meta["mpc_cutoff"])
                for raw_cutoff in cutoffs:
                    value = np.asarray(raw_cutoff)
                    if value.ndim != 0 or value.dtype.kind not in "fiu" or not np.isfinite(value) or value <= 0:
                        raise ValueError(f"{checkpoint_path}: mpc_cutoff must be a positive finite real scalar.")
                    value = float(value)
                    if cutoff is not None and value != cutoff:
                        raise ValueError("all ranks, twists, and lattice spacings must use the same mpc_cutoff.")
                    cutoff = value
    if any(enabled) and not all(enabled):
        raise ValueError("MPC data must be present at every lattice spacing and twist, or absent everywhere.")
    return cutoff


@lrdmc_app.command("compute-energy")
def lrdmc_compute_energy(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. restart.h5"),
    num_gfmc_bin_block: int = typer.Option(
        10,
        "-b",
        "--num_gfmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_gfmc_warmup_steps: int = typer.Option(
        25,
        "-w",
        "--num_gfmc_warmup_steps",
        help="Number of observable measurement steps for warmup (i.e., discarded).",
    ),
    num_gfmc_collect_steps: int = typer.Option(
        5,
        "-c",
        "--num_gfmc_collect_steps",
        help="Number of measurement (before binning) for collecting the weights.",
    ),
) -> None:
    """LRDMC energy calculation."""
    typer.echo(f"Read restart checkpoint file(s) from {restart_chk}.")
    _check_lrdmc_analysis_floors(
        num_gfmc_bin_blocks=num_gfmc_bin_block,
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
    )

    _require_checkpoint_driver(restart_chk, app="lrdmc", command="compute-energy")
    if os.path.isdir(restart_chk):
        _require_consistent_twist_directory(restart_chk)
        manifest = load_twist_checkpoint_manifest(restart_chk)
        estimates = []
        for checkpoint_path, _weight in manifest:
            estimate, _num_ranks, _num_samples = _lrdmc_energy_estimate(
                checkpoint_path,
                num_gfmc_bin_blocks=num_gfmc_bin_block,
                num_gfmc_warmup_steps=num_gfmc_warmup_steps,
                num_gfmc_collect_steps=num_gfmc_collect_steps,
            )
            estimates.append(estimate)
        weights = np.asarray([weight for _path, weight in manifest], dtype=np.float64)
        typer.echo(f"Found {len(manifest)} twist checkpoints.")
        _echo_energy_estimate(_combine_twist_energy_estimates(estimates, weights))
        return

    estimate, num_ranks, num_samples = _lrdmc_energy_estimate(
        restart_chk,
        num_gfmc_bin_blocks=num_gfmc_bin_block,
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
    )
    typer.echo(f"Found {num_ranks} MPI ranks.")
    typer.echo(f"Total number of binned samples = {num_samples}")
    _echo_energy_estimate(estimate)


@lrdmc_app.command("compute-force")
def lrdmc_compute_force(
    restart_chk: str = typer.Argument(..., help="Restart checkpoint file, e.g. restart.h5"),
    num_gfmc_bin_block: int = typer.Option(
        10,
        "-b",
        "--num_gfmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_gfmc_warmup_steps: int = typer.Option(
        25,
        "-w",
        "--num_gfmc_warmup_steps",
        help="Number of observable measurement steps for warmup (i.e., discarded).",
    ),
    num_gfmc_collect_steps: int = typer.Option(
        5,
        "-c",
        "--num_gfmc_collect_steps",
        help="Number of measurement (before binning) for collecting the weights.",
    ),
) -> None:
    """LRDMC atomic force calculation."""
    typer.echo(f"Read restart checkpoint file(s) from {restart_chk}.")
    _check_lrdmc_analysis_floors(
        num_gfmc_bin_blocks=num_gfmc_bin_block,
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
    )

    _require_checkpoint_driver(restart_chk, app="lrdmc", command="compute-force")
    if os.path.isdir(restart_chk):
        _require_consistent_twist_directory(restart_chk)
        _echo_force_pressure_summary(
            *_twist_force_pressure_summary(
                restart_chk,
                num_bin_blocks=num_gfmc_bin_block,
                num_warmup_steps=num_gfmc_warmup_steps,
                num_collect_steps=num_gfmc_collect_steps,
            )
        )
        return

    """Load the 2-D observables; the per-atom force observables are streamed by bin block below."""
    rank_data = load_observables_from_checkpoint(restart_chk, ["e_L", "w_L"])
    num_ranks = len(rank_data)
    typer.echo(f"Found {num_ranks} MPI ranks.")

    # Check that force computation was enabled
    cfg = load_driver_config_from_checkpoint(restart_chk, rank=0)
    if not cfg.get("comput_position_deriv", False):
        typer.echo("Error: atomic_force was not enabled during the LRDMC run. Cannot compute forces.")
        raise typer.Exit(code=1)
    _check_lrdmc_binning_inputs(
        rank_data,
        num_gfmc_bin_blocks=num_gfmc_bin_block,
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
        source=restart_chk,
    )

    hamiltonian_data = load_hamiltonian_from_checkpoint(restart_chk)
    atomic_labels = hamiltonian_data.structure_data.atomic_labels

    w_L_binned_list = []
    w_L_e_L_binned_list = []
    w_L_force_HF_binned_list = []
    w_L_force_PP_binned_list = []
    w_L_E_L_force_PP_binned_list = []

    num_mcmc_warmup_steps = num_gfmc_warmup_steps
    num_mcmc_bin_blocks = num_gfmc_bin_block
    k = num_gfmc_collect_steps
    contributing = [rank for rank, obs in enumerate(rank_data) if obs.get("e_L", np.empty(0)).size != 0]
    # One common power-of-two scale for every rank entering this estimate.
    retained_w_L = compute_retained_accumulated_weights(
        [rank_data[rank]["w_L"] for rank in contributing], k, num_mcmc_warmup_steps
    )

    with open_observable_datasets(restart_chk, _PER_ATOM_FORCE_KEYS) as force_datasets:
        for rank, w_L in zip(contributing, retained_w_L, strict=True):
            raw_e_L = rank_data[rank]["e_L"]
            datasets = force_datasets[rank]

            e_L = raw_e_L[k:][num_mcmc_warmup_steps:]

            # split and binning with multiple walkers
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)

            # binned sum
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_force_HF_binned, w_L_force_PP_binned, w_L_E_L_force_PP_binned = (
                list(values)
                for values in _streamed_weighted_force_bins(
                    datasets, w_L, shift=k, num_warmup_steps=num_mcmc_warmup_steps, num_bin_blocks=num_mcmc_bin_blocks
                )
            )

            w_L_binned_list += w_L_binned
            w_L_e_L_binned_list += w_L_e_L_binned
            w_L_force_HF_binned_list += w_L_force_HF_binned
            w_L_force_PP_binned_list += w_L_force_PP_binned
            w_L_E_L_force_PP_binned_list += w_L_E_L_force_PP_binned

    w_L_binned = np.array(w_L_binned_list)
    w_L_e_L_binned = np.array(w_L_e_L_binned_list)
    w_L_force_HF_binned = np.array(w_L_force_HF_binned_list)
    w_L_force_PP_binned = np.array(w_L_force_PP_binned_list)
    w_L_E_L_force_PP_binned = np.array(w_L_E_L_force_PP_binned_list)

    # jackknife implementation
    w_L_binned_sum = np.sum(w_L_binned, axis=0)
    w_L_e_L_binned_sum = np.sum(w_L_e_L_binned, axis=0)
    w_L_force_HF_binned_sum = np.sum(w_L_force_HF_binned, axis=0)
    w_L_force_PP_binned_sum = np.sum(w_L_force_PP_binned, axis=0)
    w_L_E_L_force_PP_binned_sum = np.sum(w_L_E_L_force_PP_binned, axis=0)

    M = w_L_binned.size
    typer.echo(f"Total number of binned samples = {M}")
    _check_jackknife_inputs(w_L_binned, source=restart_chk)

    force_HF_jn = -1.0 * np.array(
        [(w_L_force_HF_binned_sum - w_L_force_HF_binned[m]) / (w_L_binned_sum - w_L_binned[m]) for m in range(M)]
    )

    force_Pulay_jn = -2.0 * np.array(
        [
            (
                (w_L_E_L_force_PP_binned_sum - w_L_E_L_force_PP_binned[m]) / (w_L_binned_sum - w_L_binned[m])
                - (
                    (w_L_e_L_binned_sum - w_L_e_L_binned[m])
                    / (w_L_binned_sum - w_L_binned[m])
                    * (w_L_force_PP_binned_sum - w_L_force_PP_binned[m])
                    / (w_L_binned_sum - w_L_binned[m])
                )
            )
            for m in range(M)
        ]
    )

    # The atomic force is a real observable. At a complex Bloch twist e_L (and
    # so E_L * force_PP) is complex and each jackknife sample carries an
    # imaginary part that vanishes only in expectation. Project every sample
    # onto its real part BEFORE the mean and the variance, as the drivers'
    # get_aF do; otherwise the error includes the imaginary fluctuation.
    # Identity on the real (Gamma / open-boundary) path.
    force_jn = np.real(force_HF_jn + force_Pulay_jn)

    force_mean = np.average(force_jn, axis=0)
    force_std = np.sqrt(M - 1) * np.std(force_jn, axis=0)
    _check_force_result(force_mean, force_std, source=restart_chk)

    typer.echo("Atomic Forces:")
    sep = 16 * 3
    typer.echo("-" * sep)
    typer.echo("Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
    typer.echo("-" * sep)
    for i in range(len(atomic_labels)):
        atomic_label = str(atomic_labels[i])
        row_values = [f"{ufloat(force_mean[i, j], force_std[i, j]):+2uS}" for j in range(3)]
        row_str = atomic_label.ljust(8) + " ".join(val.ljust(12) for val in row_values)
        typer.echo(row_str)
    typer.echo("-" * sep)


@lrdmc_app.command("extrapolate-energy")
def lrdmc_extrapolate_energy(
    restart_chks: list[str] = typer.Argument(
        ...,
        help="LRDMC checkpoint files or complete twist checkpoint directories, one per lattice spacing. Do not mix formats.",
    ),
    polynomial_order: int = typer.Option(
        2,
        "-p",
        "--polynomial-order",
        help="Polynomial order with respect to a^2 for extrapolation. Default is 2. i.e., E_0 + a^2 * E_2 + a^4 * E_4.",
    ),
    plot_graph: bool = typer.Option(
        False,
        "-g",
        "--plot-graph",
        help="Open an interactive matplotlib window summarizing the result (skipped, with a message, when the matplotlib backend is non-interactive, e.g. MPLBACKEND=Agg on a headless machine; use -s to save the figure instead).",
    ),
    save_graph: str = typer.Option(None, "-s", "--save-graph", help="Specify a graph filename."),
    num_gfmc_bin_block: int = typer.Option(
        10,
        "-b",
        "--num_gfmc_bin_blocks",
        help="Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
    ),
    num_gfmc_warmup_steps: int = typer.Option(
        25,
        "-w",
        "--num_gfmc_warmup_steps",
        help="Number of observable measurement steps for warmup (i.e., discarded).",
    ),
    num_gfmc_collect_steps: int = typer.Option(
        5,
        "-c",
        "--num_gfmc_collect_steps",
        help="Number of measurement (before binning) for collecting the weights.",
    ),
) -> None:
    """Extrapolate E and, when present, E_MPC to a=0 after any BZ averaging."""
    sep = 72
    typer.echo("-" * sep)
    typer.echo(f"Read restart checkpoint files from {restart_chks}.")
    _check_lrdmc_analysis_floors(
        num_gfmc_bin_blocks=num_gfmc_bin_block,
        num_gfmc_warmup_steps=num_gfmc_warmup_steps,
        num_gfmc_collect_steps=num_gfmc_collect_steps,
    )

    alat_list = []
    energy_series = {"E": []}

    try:
        inputs = _lrdmc_extrapolation_inputs(restart_chks, polynomial_order)
        mpc_cutoff = _lrdmc_extrapolation_mpc_cutoff(inputs)
        if all(os.path.isdir(path) for path in restart_chks):
            # After the specific alat/k-point/MPC checks, so their diagnostics keep priority.
            for restart_chk, (_a, manifest) in zip(restart_chks, inputs, strict=True):
                _check_twist_directory_consistency(restart_chk, manifest)
        quantity_keys = {"E": ("E", "sigma_E")}
        if mpc_cutoff is not None:
            energy_series["E_MPC"] = []
            quantity_keys["E_MPC"] = ("E_mpc", "sigma_E_mpc")
            typer.echo(f"Extrapolating both E and E_MPC (mpc_cutoff = {mpc_cutoff} Ha).")
        for a, manifest in inputs:
            estimates = []
            for checkpoint_path, _weight in manifest:
                estimate, num_ranks, num_samples = _lrdmc_energy_estimate(
                    checkpoint_path,
                    num_gfmc_bin_blocks=num_gfmc_bin_block,
                    num_gfmc_warmup_steps=num_gfmc_warmup_steps,
                    num_gfmc_collect_steps=num_gfmc_collect_steps,
                )
                typer.echo(f"Found {num_ranks} MPI ranks in {checkpoint_path}.")
                typer.echo(f"  Total number of binned samples = {num_samples}")
                estimates.append(estimate)
            weights = np.asarray([weight for _path, weight in manifest], dtype=np.float64)
            combined = _combine_twist_energy_estimates(estimates, weights)
            alat_list.append(a)
            for label, (mean_key, sigma_key) in quantity_keys.items():
                mean, sigma = combined[mean_key], combined[sigma_key]
                if mean is None or sigma is None or not np.isfinite(mean) or not np.isfinite(sigma):
                    raise ValueError(f"a={a}: missing or non-finite BZ {label} energy or error bar.")
                energy_series[label].append((mean, sigma))
                typer.echo(f"  For a = {a} bohr: {label} = {mean} +- {sigma} Ha.")
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("-" * sep)

    typer.echo("Extrapolation of the energy with respect to a^2.")
    typer.echo(f"  Polynomial order = {polynomial_order}.")

    # Prepare data for polynomial fitting
    interpolate_num = 1000
    x = np.array(alat_list) ** 2
    xs = np.linspace(0.0, np.max(x) * 1.2, interpolate_num)

    # The unweighted least-squares fit is linear in the input energies,
    # w = P y with P = pinv(V) and V the a^2 Vandermonde matrix, so the
    # continuum value E0 = c . y (c = the constant-term row of P) and its
    # uncertainty for independent points, sqrt(sum_i (c_i sigma_i)^2), follow
    # in closed form. This is exactly the expectation that a Gaussian
    # resampling of the points estimates with finite random draws,
    # so the result is deterministic and free of resampling noise. E and
    # E_MPC are separate marginal fits; the E+dV covariance is already in
    # sigma_E_mpc from the paired jackknife, BEFORE twist averaging, and no
    # joint E/E_MPC covariance is reported.
    projector = np.linalg.pinv(np.vander(x, polynomial_order + 1))
    intercept_row = projector[-1]
    fits = {}
    for label, points in energy_series.items():
        y, y_err = np.asarray(points).T
        w = projector @ y
        E_0_mean = float(intercept_row @ y)
        E_0_std = float(np.sqrt(np.sum((intercept_row * y_err) ** 2)))
        fits[label] = (E_0_mean, E_0_std, np.polyval(w, xs))
        typer.echo(f"  For a -> 0 bohr: {label} = {E_0_mean} +- {E_0_std} Ha.")
    typer.echo("-" * sep)

    # plot graphs
    if plot_graph or save_graph is not None:
        plt.rcParams["font.size"] = 8
        plt.rcParams["font.family"] = "sans-serif"

        fig = plt.figure(figsize=(4, 4), facecolor="white", dpi=300, tight_layout=True)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.tick_params(axis="both", which="both", direction="in")
        for (label, points), color in zip(energy_series.items(), ("blue", "darkorange"), strict=False):
            y, y_err = np.asarray(points).T
            E_0_mean, E_0_std, y_mean_plot = fits[label]
            ax1.errorbar(x, y, yerr=y_err, fmt="o", markersize=4, capsize=3, color=color, label=f"{label} data")
            ax1.errorbar(
                0.0, E_0_mean, yerr=E_0_std, fmt="s", markersize=4, capsize=3, color=color, label=f"{label} extrapolated"
            )
            ax1.plot(xs, y_mean_plot, ls="--", color=color)
        ax1.set_xlabel(r"Lattice discretization $a^2$ (bohr$^2$)")
        ax1.set_ylabel("LRDMC Energy (Ha)")
        ax1.legend(loc="best")

        if save_graph is not None:
            plt.savefig(save_graph)
            typer.echo(f"Graph is saved in {save_graph}.")

        if plot_graph:
            _show_figure_if_interactive()
    typer.echo("-" * sep)
    typer.echo("Extrapolation is finished.")


@lrdmc_app.command("generate-input")
def lrdmc_generate_input(
    flag: bool = typer.Option(False, "-g", "--generate", help="Generate input file for LRDMC calculations."),
    filename: str = typer.Option("lrdmc.toml", "-f", "--filename", help="Filename for the input file."),
    exclude_comment: bool = typer.Option(False, "-nc", "--without-comment", help="Exclude comments in the input file."),
) -> None:
    """Generate an input file for LRDMC calculations."""
    if flag:
        doc = tomlkit.document()

        control_table = tomlkit.table()
        for key, value in cli_parameters["control"].items():
            if value is None:
                control_table[key] = str(value)
            else:
                control_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                control_table[key].comment(cli_parameters["control_comments"][key])
        control_table["job_type"] = "lrdmc-bra"
        doc.add("control", control_table)

        lrdmc_table = tomlkit.table()
        _omit = GENERATED_INPUT_OMITTED_KEYS.get("lrdmc-bra", ())
        for key, value in cli_parameters["lrdmc-bra"].items():
            if key in _omit:
                continue
            if value is None:
                lrdmc_table[key] = str(value)
            else:
                lrdmc_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                lrdmc_table[key].comment(cli_parameters["lrdmc-bra_comments"][key])
        doc.add("lrdmc-bra", lrdmc_table)

        coulomb_table = tomlkit.table()
        for key, value in cli_parameters["coulomb"].items():
            if value is None:
                coulomb_table[key] = str(value)
            else:
                coulomb_table[key] = value
            if not exclude_comment and not isinstance(value, bool):
                coulomb_table[key].comment(cli_parameters["coulomb_comments"][key])
        doc.add("coulomb", coulomb_table)

        with open(filename, "w") as f:
            f.write(tomlkit.dumps(doc))
        typer.echo(f"Input file is generated: {filename}")

    else:
        typer.echo("Activate the flag (-g) to generate an input file. See --help for more information.")


typer_click_lrdmc = typer.main.get_command(lrdmc_app)

_cli.add_command(typer_click_lrdmc, "lrdmc")


def _collect_exit_exception_classes() -> tuple[type[BaseException], ...]:
    """Return every ``Exit`` exception class a ``--help`` dispatch may raise.

    click's own ``Exit`` is always present. typer raises its own
    ``typer.exceptions.Exit`` from ``--help`` callbacks, and newer typer
    releases additionally vendor a private copy of click under
    ``typer._click``; every such class is a different class object and must
    be caught separately. The lookups use ``getattr`` so that a typer release
    without one of the attributes cannot break the console script.
    """
    classes: list[type[BaseException]] = [click.exceptions.Exit]
    candidate_modules: list[object] = [getattr(typer, "exceptions", None)]
    try:
        from typer._click import exceptions as _typer_click_exceptions  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        candidate_modules.append(_typer_click_exceptions)
    for module in candidate_modules:
        typer_exit = getattr(module, "Exit", None)
        if isinstance(typer_exit, type) and issubclass(typer_exit, BaseException) and typer_exit not in classes:
            classes.append(typer_exit)
    return tuple(classes)


_EXIT_EXCEPTION_CLASSES = _collect_exit_exception_classes()


def _cli_main() -> Any:
    """Console-script entry point.

    Starting in typer 0.16 typer vendors its own copy of click (under
    ``typer._click``). When the outer click group (``_cli``, defined with
    ``@click.group()``) dispatches to a typer-derived subcommand and the
    user passes ``--help``, typer's vendored show_help raises
    ``typer._click.exceptions.Exit(0)``. The outer click ``main()`` only
    catches its OWN ``click.exceptions.Exit`` (different class object), so
    the typer ``Exit`` leaks past ``sys.exit(_cli())`` with a traceback
    despite being a successful (exit code 0) termination.

    Wrap ``_cli()`` here and translate any such ``Exit`` to a clean
    ``sys.exit(code)`` so ``--help`` prints no spurious traceback.
    ``SystemExit`` raised by click itself is re-raised unchanged; the set of
    ``Exit`` classes is resolved once at import time by
    :func:`_collect_exit_exception_classes`, so a typer release that does not
    expose the vendored attribute cannot break the entry point.
    """
    try:
        return _cli()
    except SystemExit:
        # click's ``main()`` already turned its own ``Exit`` into ``SystemExit``
        # with the right code; let it through untouched.
        raise
    except _EXIT_EXCEPTION_CLASSES as e:
        sys.exit(int(getattr(e, "exit_code", 0)))


if __name__ == "__main__":
    _cli_main()
