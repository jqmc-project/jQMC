"""QE -> jQMC Hamiltonian conversion.

Library entry point of the conversion pipeline (QE plane-wave
reference -> chosen basis projection -> jQMC ``Hamiltonian_data``).
``jqmc-tool qe`` builds the ``qe_params`` / ``jastrow_params``
dicts from its options and calls :func:`run_qe2jqmc`;
:class:`jqmc_workflow.QE_Workflow` runs that command.
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

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from .jastrow_factor import DTN_DEFAULT_CUTOFF_RATIO

if TYPE_CHECKING:
    from .coulomb_potential import Coulomb_potential_data
    from .jastrow_factor import Jastrow_DTN_data
    from .pw_pgto_projection import ProjectionResult
    from .qe_wrapper import QEReference
    from .structure import Structure_data

logger = getLogger("jqmc").getChild(__name__)


def _build_jastrow_dtn(
    structure_data: Structure_data,
    *,
    jastrow_dtn: str | None,
    coulomb_potential_data: Coulomb_potential_data,
    dtn_c: int = 4,
    dtn_n_u: int = 8,
    dtn_n_chi: int = 8,
    dtn_n_eN: int = 4,
    dtn_n_ee: int = 4,
    dtn_cutoff_ratio_u: float = DTN_DEFAULT_CUTOFF_RATIO,
    dtn_cutoff_ratio_chi: float = DTN_DEFAULT_CUTOFF_RATIO,
    dtn_cutoff_ratio_f: float = DTN_DEFAULT_CUTOFF_RATIO,
) -> Jastrow_DTN_data | None:
    """Compatibility wrapper around the shared QE DTN option builder."""
    from .qe_to_jqmc import build_jastrow_dtn_from_options

    return build_jastrow_dtn_from_options(
        structure_data,
        jastrow_dtn=jastrow_dtn,
        coulomb_potential_data=coulomb_potential_data,
        dtn_c=dtn_c,
        dtn_n_u=dtn_n_u,
        dtn_n_chi=dtn_n_chi,
        dtn_n_eN=dtn_n_eN,
        dtn_n_ee=dtn_n_ee,
        dtn_cutoff_ratio_u=dtn_cutoff_ratio_u,
        dtn_cutoff_ratio_chi=dtn_cutoff_ratio_chi,
        dtn_cutoff_ratio_f=dtn_cutoff_ratio_f,
    )


def run_qe2jqmc(qe_params: dict, jastrow_params: dict, hamiltonian_h5: str) -> None:
    """Execute the QE -> jQMC conversion and write ``hamiltonian_h5``.

    Args:
        qe_params: Conversion options (qe_save_dir, kind, upf_paths,
            ecutwfc, alpha_min, ...), as built by ``jqmc-tool qe``.
        jastrow_params: Jastrow options (j1/j2/j3 + DTN params), as built
            by ``jqmc-tool qe``.
        hamiltonian_h5: Output HDF5 path; written by rank 0 only when
            running under MPI.
    """
    # Heavy imports done lazily so module import is cheap. The _setting imports
    # are also lazy so tests that monkeypatch the module-level
    # constants (e.g. PW_PGTO_AUTO_RESIDUAL_TARGET) see the patched
    # value on every call.
    from ._setting import (
        PGTO_EXP_TAIL_EPS_USER_DEFAULT,
        PW_PGTO_AUTO_MAX_ECUTWFC_RY,
        PW_PGTO_AUTO_MIN_ECUTWFC_RY,
        PW_PGTO_AUTO_RESIDUAL_TARGET,
        PW_PGTO_SVD_RCOND,
    )
    from .pw_pgto_projection import (
        PW_TAPER_KINDS,
        MixedBasisSweep,
        alpha_min_from_gcut,
        filter_shells_by_alpha_min,
        optimize_mixed_basis,
        project_pw_to_mixed,
        project_pw_to_pgto,
        project_pw_to_pgto_spline,
        project_pw_to_pw,
        project_pw_to_spline,
        taper_qe_reference,
    )
    from .pgto_exponent_optimizer import PGTOExponentBoundsError, optimize_pgto_exponents
    from .qe_to_jqmc import (
        _build_qe_coulomb_potential_data,
        _require_upf_for_every_species,
        _upf_species_key,
        build_hamiltonian_from_wavefunction,
        build_wavefunction_from_projection,
        check_binary_occupations,
    )
    from .qe_wrapper import _find_save_dir, load_qe_kpoint_list, load_qe_reference, qe_save_is_lsda, truncate_qe_reference_bands
    from .structure import _tiling_determinant_adjugate, normalize_tiling_matrix
    from .trexio_wrapper import _convert_from_atomic_labels_to_atomic_numbers
    from .upf_wrapper import ECPData, UPFExponentInit

    # --- conversion options --------------------------------------------
    qe_save_dir = str(qe_params["qe_save_dir"])
    kind = str(qe_params["kind"])
    upf_paths = list(qe_params.get("upf_paths") or [])
    twist_average = bool(qe_params.get("twist_average") or False)
    tiling_raw = qe_params.get("tiling")
    tiling_matrix = None if tiling_raw is None or len(tiling_raw) == 0 else normalize_tiling_matrix(tiling_raw)
    tiling_determinant = 1 if tiling_matrix is None else abs(_tiling_determinant_adjugate(tiling_matrix)[0])
    requested_supertwist = qe_params.get("supertwist")
    if requested_supertwist is not None and (tiling_matrix is None or twist_average):
        raise ValueError("supertwist requires tiling and selects a single sector; do not combine it with twist_average")

    # Auto-discover UPF from the QE save_dir when upf_paths is empty.
    # QE always copies the pseudopotentials it used into the .save/
    # directory and records the (element, pseudo_file) mapping in
    # data-file-schema.xml -- the authoritative source we read from
    # here. Users can still pass an explicit upf_paths to override
    # (e.g. when supplying a different UPF for the same element).
    if not upf_paths:
        import os
        import xml.etree.ElementTree as ET

        from .qe_wrapper import _find_save_dir

        # Resolve the ``*.save`` directory exactly as ``load_qe_reference``
        # does (the save itself, its parent holding one ``*.save``, or a file
        # inside it). Reading ``<qe_save_dir>/data-file-schema.xml`` verbatim
        # missed the XML for the parent-directory form, so pseudopotential
        # saves converted with no ECP at all (issue #99).
        upf_save_dir = _find_save_dir(qe_save_dir)
        xml_path = os.path.join(upf_save_dir, "data-file-schema.xml")
        discovered_pseudopotentials: dict[str, str] = {}
        if os.path.isfile(xml_path):
            try:
                xml_root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                xml_root = None
            if xml_root is not None:
                # QE records the same species in input and output sections.
                # Keep the first input declaration and avoid duplicate paths.
                for species in xml_root.iter("species"):
                    element = species.get("name")
                    pseudo_element = species.find("pseudo_file")
                    pseudo_filename = "" if pseudo_element is None else (pseudo_element.text or "").strip()
                    if element is not None and pseudo_filename and element not in discovered_pseudopotentials:
                        discovered_pseudopotentials[element] = pseudo_filename
        upf_paths = [
            f"{element}={os.path.join(upf_save_dir, pseudo_filename)}"
            for element, pseudo_filename in discovered_pseudopotentials.items()
        ]
        if upf_paths:
            logger.info(
                "Auto-discovered %d UPF entries from %s/data-file-schema.xml: %s",
                len(upf_paths),
                upf_save_dir,
                ", ".join(upf_paths),
            )
    ecutwfc = qe_params.get("ecutwfc")
    alpha_min = qe_params.get("alpha_min")
    # Exponent-tail bound of the exponent optimiser.  An absent key
    # takes the user-facing default (1e-2); None or 0 disables it.
    _exp_tail_eps = qe_params.get("exp_tail_eps", PGTO_EXP_TAIL_EPS_USER_DEFAULT)
    exp_tail_eps = None if _exp_tail_eps is None or float(_exp_tail_eps) == 0.0 else float(_exp_tail_eps)
    if exp_tail_eps is not None and not (0.0 < exp_tail_eps < 1.0):
        raise ValueError(f"exp_tail_eps must satisfy 0 < exp_tail_eps < 1, or be 0 / None to disable; got {exp_tail_eps}.")
    max_cond_number = qe_params.get("max_cond_number")
    # ``is None`` rather than ``or``: 0 is falsy, so ``or`` silently replaced an
    # explicitly supplied 0 with the default. A bogus 0 must be rejected loudly
    # (a mesh_factor of 0 would collapse the spline grid to the ``order + 1``
    # floor, i.e. a 6^3 cube, and quietly produce garbage orbitals), not
    # substituted. The same applies to spline_order and ewald_tol.
    _mesh_factor = qe_params.get("mesh_factor")
    # Default 0.75, not 0.5: 0.5 puts the spline Nyquist
    # exactly at the PW cutoff: values stay accurate (O(h^6)) but the Laplacian
    # (O(h^4)) does not, biasing the determinant kinetic energy (measured at
    # a 300 Ry cutoff: -9.5 mHa/twist; -0.44 at 0.75). QMCPACK's default grid is
    # ceil(4 * maxIndex), i.e. 2x Nyquist; 0.75 = 1.5x Nyquist keeps a safety
    # margin at 42% of the 1.0 memory. See the Nyquist-headroom warning in
    # project_pw_to_spline.
    mesh_factor = 0.75 if _mesh_factor is None else float(_mesh_factor)
    if not mesh_factor > 0.0:
        raise ValueError(f"mesh_factor must be > 0 (got {mesh_factor}); it scales the QE FFT grid.")
    _spline_order = qe_params.get("spline_order")
    spline_order = 5 if _spline_order is None else int(_spline_order)
    if spline_order not in (3, 5):
        raise ValueError(f"spline_order must be 3 (cubic) or 5 (quintic); got {spline_order}.")
    _ewald_tol = qe_params.get("ewald_tol")
    ewald_tol = 1.0e-10 if _ewald_tol is None else float(_ewald_tol)
    if not ewald_tol > 0.0:
        raise ValueError(f"ewald_tol must be > 0 (got {ewald_tol}).")
    ecp_z_cores = qe_params.get("ecp_z_cores") or []
    # Band-order Cholesky orthonormalization of pure-pGTO MOs in the
    # periodic AO metric. ``is None`` keeps an explicit False.
    _orthonormalize = qe_params.get("orthonormalize")
    orthonormalize = True if _orthonormalize is None else bool(_orthonormalize)

    # --- Jastrow options -----------------------------------------------
    j1_parameter = jastrow_params.get("j1_parameter")
    j1_type = str(jastrow_params.get("j1_type") or "exp")
    j2_parameter = jastrow_params.get("j2_parameter")
    j2_type = str(jastrow_params.get("j2_type") or "pade")
    j3_basis_type = str(jastrow_params.get("j3_basis_type") or "none")
    jastrow_dtn = str(jastrow_params.get("jastrow_dtn") or "none")

    def _dtn_int(name: str, default: int) -> int:
        # ``is None`` rather than ``or`` (as for mesh_factor above): an explicit
        # 0 must reach the DTN builder's validation, not become the default.
        value = jastrow_params.get(name)
        return default if value is None else int(value)

    dtn_c = _dtn_int("dtn_c", 4)
    dtn_n_u = _dtn_int("dtn_n_u", 8)
    dtn_n_chi = _dtn_int("dtn_n_chi", 8)
    dtn_n_eN = _dtn_int("dtn_n_eN", 4)
    dtn_n_ee = _dtn_int("dtn_n_ee", 4)

    def _ratio(name: str) -> float:
        value = jastrow_params.get(name)
        return DTN_DEFAULT_CUTOFF_RATIO if value is None else float(value)

    dtn_cutoff_ratio_u = _ratio("dtn_cutoff_ratio_u")
    dtn_cutoff_ratio_chi = _ratio("dtn_cutoff_ratio_chi")
    dtn_cutoff_ratio_f = _ratio("dtn_cutoff_ratio_f")

    if tiling_matrix is not None:
        if kind not in ("pw", "bspline", "pgto", "pgto+pw"):
            raise ValueError("QE orbital tiling supports kind='pw', 'bspline', 'pgto' or 'pgto+pw'.")
        if j3_basis_type.lower() not in {"none", "ao", "ao-full", "ao-small", "ao-medium", "ao-large"}:
            raise ValueError(
                "QE tiling requires an independent supercell J3 basis: use ao-small/medium/large/full or none, not mo."
            )

    # --- ecutwfc parsing: numeric, "auto", or omitted ------------------
    auto_pw_cutoff = False
    g_cut: float | None = None
    if ecutwfc is not None:
        ec_str = str(ecutwfc).strip()
        if ec_str.lower() == "auto":
            if kind != "pgto+pw":
                raise ValueError(
                    f"ecutwfc='auto' is only meaningful for kind='pgto+pw' "
                    f"(got kind={kind!r}); pass a number in Rydberg or omit."
                )
            auto_pw_cutoff = True
        else:
            try:
                ec_val = float(ec_str)
            except ValueError as exc:
                raise ValueError(f"ecutwfc must be a number in Rydberg or 'auto', got {ec_str!r}.") from exc
            if ec_val < 0.0:
                raise ValueError("ecutwfc must be non-negative (Rydberg).")
            g_cut = float(np.sqrt(ec_val))

    if max_cond_number is not None and float(max_cond_number) <= 0.0:
        raise ValueError("max_cond_number must be positive.")
    svd_rcond = PW_PGTO_SVD_RCOND if max_cond_number is None else 1.0 / float(max_cond_number)
    alpha_min_val = None if alpha_min is None else float(alpha_min)

    if twist_average and tiling_matrix is None:
        from .qe_to_jqmc import convert_qe_to_twist_average_input

        j2_param_twist = None if (j2_parameter is None or float(j2_parameter) < 0.0) else float(j2_parameter)
        j1_param_twist = None if j1_parameter is None else float(j1_parameter)
        convert_qe_to_twist_average_input(
            qe_save_dir=qe_save_dir,
            output_path=hamiltonian_h5,
            upf_paths=upf_paths,
            kind=kind,
            svd_rcond=svd_rcond,
            alpha_min=alpha_min_val,
            exp_tail_eps=exp_tail_eps,
            ecp_z_cores=tuple(float(z) for z in ecp_z_cores) if ecp_z_cores else None,
            j1_param=j1_param_twist,
            j1_type=j1_type,
            j2_param=j2_param_twist,
            j2_type=j2_type,
            j3_basis_type=j3_basis_type,
            jastrow_dtn=jastrow_dtn,
            dtn_c=dtn_c,
            dtn_n_u=dtn_n_u,
            dtn_n_chi=dtn_n_chi,
            dtn_n_eN=dtn_n_eN,
            dtn_n_ee=dtn_n_ee,
            dtn_cutoff_ratio_u=dtn_cutoff_ratio_u,
            dtn_cutoff_ratio_chi=dtn_cutoff_ratio_chi,
            dtn_cutoff_ratio_f=dtn_cutoff_ratio_f,
            ewald_tol=ewald_tol,
            ecutwfc=ecutwfc,
            mesh_factor=mesh_factor,
            spline_order=spline_order,
            orthonormalize=orthonormalize,
        )
        logger.info("Twist-average Hamiltonian data is saved in %s.", hamiltonian_h5)
        return

    # LSDA (nspin=2) saves carry two independent spin channels
    # (wfcup1.hdf5 / wfcdw1.hdf5). Canonical SD has no square paired-lambda
    # requirement: each channel is truncated and projected independently
    # through its highest occupied band. No minority-spin virtual rows are
    # generated merely to match the majority-spin row count.
    is_lsda = qe_save_is_lsda(qe_save_dir)
    reference_dn = None
    if is_lsda:
        ref_up_full = load_qe_reference(qe_save_dir, spin="up")
        ref_dn_full = load_qe_reference(qe_save_dir, spin="dw")
        # QMC fixes the electron numbers: accept only integer (N_up, N_dn),
        # i.e. binary (0/1) band occupations in BOTH channels. Fail here --
        # before the expensive projection -- with the QE-side remedy; the
        # geminal builder re-checks as a backstop.
        check_binary_occupations(ref_up_full.bands.occupations, channel="up")
        check_binary_occupations(ref_dn_full.bands.occupations, channel="dw")
        occ_idx_up = ref_up_full.bands.occupied_band_indices()
        occ_idx_dn = ref_dn_full.bands.occupied_band_indices()
        if occ_idx_up.size == 0 or occ_idx_dn.size == 0:
            raise ValueError("LSDA save has an empty spin channel; cannot build a determinant.")
        n_keep_up = int(occ_idx_up.max()) + 1
        n_keep_dn = int(occ_idx_dn.max()) + 1
        reference = truncate_qe_reference_bands(ref_up_full, n_keep_up)
        reference_dn = truncate_qe_reference_bands(ref_dn_full, n_keep_dn)
        logger.info(
            "LSDA (nspin=2) save detected: n_occ_up=%d, n_occ_dn=%d; "
            "occupied-only source ranges are %d up bands and %d down bands.",
            int(occ_idx_up.size),
            int(occ_idx_dn.size),
            n_keep_up,
            n_keep_dn,
        )
    else:
        reference = load_qe_reference(qe_save_dir)
        # Same integer-filling contract on the restricted path (metallic
        # smearing is not convertible until grand-canonical TABC lands).
        check_binary_occupations(reference.bands.occupations)

    tiled_sector = None
    if tiling_matrix is not None:
        from pathlib import Path

        from .orbital_tiling import group_tiling_k_points
        from .orbital_tiling_sources import convert_qe_tiled_twist_average, select_tiling_sector

        kpoint_entries = load_qe_kpoint_list(str(Path(_find_save_dir(qe_save_dir)) / "data-file-schema.xml"))
        fractional_k = np.asarray([k for k, _ in kpoint_entries]) @ np.linalg.inv(reference.cell.b)
        groups = group_tiling_k_points(tiling_matrix, fractional_k, [w for _, w in kpoint_entries])
        if twist_average:
            convert_qe_tiled_twist_average(qe_params, jastrow_params, hamiltonian_h5, groups)
            logger.info("Tiled twist-average Hamiltonian data is saved in %s.", hamiltonian_h5)
            return
        tiled_sector = select_tiling_sector(groups, requested_supertwist)

    # Vacuum cells only (bulk stays bit-identical): taper the PW coefficients
    # at the G-sphere edge before ANY plane-wave-carrying projection sees them,
    # so the fitted spline / PW block and its fidelity diagnostic agree on the
    # same (ringing-free) bands. See PW_TAPER_* in _setting.py.
    # A truncated ``pw`` basis keeps only |G+k| <= g_cut, so its edge is g_cut.
    pw_taper_meta: dict | None = None
    if kind in PW_TAPER_KINDS:
        _taper_edge = g_cut if (kind == "pw" and g_cut is not None) else None
        reference, pw_taper_meta = taper_qe_reference(reference, g_end=_taper_edge)
        if reference_dn is not None:
            reference_dn, _meta_dn = taper_qe_reference(reference_dn, g_end=_taper_edge, label="_dn")
            if pw_taper_meta is not None and _meta_dn is not None:
                # The dn channel's loss keys carry the ``_dn`` label; the geometry keys are identical.
                pw_taper_meta.update(_meta_dn)

    logger.info("")

    needs_upf_for_shells = kind in ("pgto", "pgto+pw", "pgto+bspline") or (
        tiling_matrix is not None and j3_basis_type.lower() != "none"
    )
    strategy: UPFExponentInit | None = None
    shells = None
    if upf_paths:
        upf_map: dict[str, str] = {}
        for spec in upf_paths:
            if "=" not in spec:
                raise ValueError(f"upf_paths entry expects 'element=path.upf', got {spec!r}.")
            elem, path = spec.split("=", 1)
            upf_map[elem.strip()] = path.strip()
        _require_upf_for_every_species(upf_map, list(reference.atom_labels), list(reference.atom_elements))
        g_target_from_ref = float(np.sqrt(2.0 * reference.ecutwfc))
        # Cell-aware diffuse floor for the AE bare-Coulomb pgto ladder
        # (PBC ill-conditioning guard, _setting.AE_PGTO_CELL_SAFETY).
        # ``cell_min_edge`` = shortest lattice-vector length. For non-AE
        # UPFs the PSWFC fit drives the exponents and this field is
        # ignored, so passing it unconditionally is safe.
        cell_min_edge_from_ref = float(np.min(np.linalg.norm(reference.cell.a, axis=1)))
        strategy = UPFExponentInit(
            upf_map,
            g_target=g_target_from_ref,
            cell_min_edge=cell_min_edge_from_ref,
        )
        if needs_upf_for_shells:
            logger.info(
                "UPFExponentInit: tight-extension g_target=%.3f Bohr^-1 (= sqrt(2 * ecutwfc_QE) = sqrt(2 * %.3f Ha)); "
                "AE diffuse floor cell_min_edge=%.3f Bohr (shortest lattice vector).",
                g_target_from_ref,
                reference.ecutwfc,
                cell_min_edge_from_ref,
            )
            shells = [
                strategy.shells_for_element(_upf_species_key(strategy, label, element))
                for label, element in zip(reference.atom_labels, reference.atom_elements, strict=True)
            ]
    elif needs_upf_for_shells:
        raise ValueError(f"kind={kind!r} requires upf_paths=['element=path.upf', ...] to seed the pGTO basis.")

    # A numeric mixed-basis cutoff is already the final deployment boundary.
    # Optimise the active exponent ladder once at the fixed cutoff.  The
    # differentiable spilling objective uses the bounded WFC representation;
    # its independent condition constraint is the raw full overlap of the
    # emitted AOs.  Adoption is audited with the FFT-complete projector.
    # Filtering the seed first ensures every optimised primitive is retained by
    # the final projection.  LSDA up/dn channels share one exponent set.
    if kind == "pgto+pw" and g_cut is not None and not auto_pw_cutoff:
        fixed_floor = float(alpha_min_val) if alpha_min_val is not None else alpha_min_from_gcut(g_cut)
        active_shells = filter_shells_by_alpha_min(shells, fixed_floor)
        if any(atom_shells for atom_shells in active_shells):
            opt_refs = [reference] if reference_dn is None else [reference, reference_dn]
            exponent_opt = optimize_pgto_exponents(
                opt_refs,
                active_shells,
                occ_only=True,
                svd_rcond=svd_rcond,
                g_cut=g_cut,
                exponent_floor=fixed_floor,
                fixed_mixed_overlap=False,
                deployed_fixed_mixed_overlap=True,
                exp_tail_eps=exp_tail_eps,
            )
            # A rejected search returns the already-filtered seed (the basis
            # project_pw_to_mixed would select at this cutoff), clipped to
            # the exponent-tail bound when one is active.
            shells = exponent_opt.shells_per_atom
            logger.info(
                "Fixed pgto+pw exponent optimisation at ecutwfc=%.3f Ry: "
                "accepted=%s, occupation-weighted spilling %.6e -> %.6e.",
                g_cut**2,
                exponent_opt.accepted,
                exponent_opt.deployed_initial_spilling,
                exponent_opt.deployed_optimized_spilling,
            )

    # ``ecutwfc='auto'`` first evaluates the seed ladder, then variationally
    # optimises the uncontracted exponents against each still-insufficient
    # mixed pGTO + low-G-PW trial, in ascending cutoff order.  This is crucial:
    # optimising the full pGTO-only span can favour diffuse primitives that the
    # auto trial subsequently removes via alpha_min, worsening rather than
    # lowering the selected cutoff.  ``PrimitiveShell.coefficients`` remain
    # inert PSWFC-fit metadata.  LSDA up/dn targets share one exponent set.
    def _auto_g_cut_values(ref: QEReference) -> np.ndarray:
        # The first trial is the cheap 7.5 Ry, independent of the upper cap;
        # below 30 Ry QE input it is a quarter of the upper cutoff instead.
        # No PW content beyond the QE sphere is requested.
        g_max = min(float(np.sqrt(2.0 * ref.ecutwfc)), float(np.sqrt(PW_PGTO_AUTO_MAX_ECUTWFC_RY)))
        g_min = min(float(np.sqrt(PW_PGTO_AUTO_MIN_ECUTWFC_RY)), 0.5 * g_max)
        return np.geomspace(g_min, g_max, 8)

    exponent_auto_sweep = None
    if auto_pw_cutoff:
        seed_shells = shells
        opt_refs = [reference] if reference_dn is None else [reference, reference_dn]
        auto_max_g_cut = float(np.sqrt(PW_PGTO_AUTO_MAX_ECUTWFC_RY))
        user_fixed_alpha = None if alpha_min_val is None else float(alpha_min_val)
        seed_sweep = optimize_mixed_basis(
            reference,
            seed_shells,
            residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
            drop_diffuse=True,
            fixed_alpha_min=user_fixed_alpha,
            max_g_cut=auto_max_g_cut,
            g_cut_values=_auto_g_cut_values(reference),
            svd_rcond=svd_rcond,
            occ_only=True,
        )
        exponent_auto_sweep = seed_sweep
        for seed_trial in seed_sweep.trials:
            if seed_trial.mean_residual <= PW_PGTO_AUTO_RESIDUAL_TARGET:
                break
            trial_floor = float(seed_trial.alpha_min if user_fixed_alpha is None else user_fixed_alpha)
            active_shells = filter_shells_by_alpha_min(seed_shells, trial_floor)
            if not any(atom_shells for atom_shells in active_shells):
                continue
            try:
                exponent_opt = optimize_pgto_exponents(
                    opt_refs,
                    active_shells,
                    occ_only=True,
                    svd_rcond=svd_rcond,
                    g_cut=float(seed_trial.g_cut),
                    exponent_floor=trial_floor,
                    exp_tail_eps=exp_tail_eps,
                )
            except PGTOExponentBoundsError as exc:
                if user_fixed_alpha is not None:
                    raise
                # The automatic floor can leave no feasible exponent ladder
                # for this trial. Retain the seed sweep and every other trial;
                # an unmet target still raises the normal auto-cutoff error.
                logger.info("Auto pgto+pw: exponent search infeasible at %.3f Ry: %s", seed_trial.g_cut**2, exc)
                continue
            if not exponent_opt.accepted:
                continue
            candidate_up = optimize_mixed_basis(
                reference,
                exponent_opt.shells_per_atom,
                residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
                g_cut_values=[float(seed_trial.g_cut)],
                fixed_alpha_min=trial_floor,
                svd_rcond=svd_rcond,
                occ_only=True,
            )
            candidate_dn_met = True
            if reference_dn is not None:
                candidate_dn = optimize_mixed_basis(
                    reference_dn,
                    exponent_opt.shells_per_atom,
                    residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
                    g_cut_values=[float(seed_trial.g_cut)],
                    fixed_alpha_min=trial_floor,
                    svd_rcond=svd_rcond,
                    occ_only=True,
                )
                candidate_dn_met = candidate_dn.target_met
            if candidate_up.target_met and candidate_dn_met:
                shells = exponent_opt.shells_per_atom
                exponent_auto_sweep = MixedBasisSweep(
                    best=candidate_up.best,
                    trials=candidate_up.trials,
                    target_met=True,
                )
                logger.info(
                    "Auto pgto+pw exponent optimisation lowered the selected cutoff from %.3f to %.3f Ry; mean residual %.3e.",
                    seed_sweep.best.g_cut**2,
                    candidate_up.best.g_cut**2,
                    candidate_up.best.mean_residual,
                )
                break

    # --- projection branch dispatch ------------------------------------
    # ``occ_only`` policy: both restricted and LSDA channels project only
    # occupied bands. The canonical SD builder performs an independent
    # occupation selection again as a representation-boundary assertion.
    def _project_reference(
        ref: QEReference,
        *,
        occ_only: bool = True,
        g_cut_fixed: float | None = None,
    ) -> ProjectionResult:
        """Project one reference (spin channel) with the resolved settings.

        ``g_cut_fixed`` (pgto+pw only) bypasses the auto sweep with an
        already-selected handoff cutoff, so the dn channel of an LSDA
        ``ecutwfc='auto'`` conversion reuses the up-channel g_cut and the
        two spin blocks share the same PW/pGTO split.
        """
        if kind == "pw":
            return project_pw_to_pw(ref, g_cut=g_cut, occ_only=occ_only)
        if kind == "bspline":
            return project_pw_to_spline(ref, mesh_factor=mesh_factor, order=spline_order, occ_only=occ_only)
        if kind == "pgto":
            return project_pw_to_pgto(ref, shells, occ_only=occ_only, svd_rcond=svd_rcond, alpha_min=alpha_min_val)
        if kind == "pgto+pw":
            if auto_pw_cutoff and g_cut_fixed is None:
                auto_max_g_cut = float(np.sqrt(PW_PGTO_AUTO_MAX_ECUTWFC_RY))
                user_fixed_alpha = None if alpha_min_val is None else float(alpha_min_val)
                if user_fixed_alpha is None:
                    logger.info(
                        "Auto pgto+pw: per-trial alpha_min via "
                        "alpha_min_from_gcut(g_cut, PW_PGTO_TAIL_TOL); ladder "
                        "capped at ecutwfc=%.1f Ry. Set alpha_min (jqmc-tool qe --alpha-min) "
                        "to override the per-trial floor with a fixed value.",
                        PW_PGTO_AUTO_MAX_ECUTWFC_RY,
                    )
                else:
                    logger.info(
                        "Auto pgto+pw: alpha_min=%.4f forces a fixed floor "
                        "across the sweep (overrides per-trial alpha_min_from_gcut).",
                        user_fixed_alpha,
                    )
                if ref is reference and exponent_auto_sweep is not None:
                    sweep = exponent_auto_sweep
                else:
                    sweep = optimize_mixed_basis(
                        ref,
                        shells,
                        residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
                        drop_diffuse=True,
                        fixed_alpha_min=user_fixed_alpha,
                        max_g_cut=auto_max_g_cut,
                        g_cut_values=_auto_g_cut_values(ref),
                        svd_rcond=svd_rcond,
                        occ_only=occ_only,
                    )
                if not sweep.target_met:
                    raise RuntimeError(
                        f"ecutwfc='auto': overlap residual did not reach the target "
                        f"{PW_PGTO_AUTO_RESIDUAL_TARGET:.2e} within the auto-sweep cap "
                        f"(largest tried = {PW_PGTO_AUTO_MAX_ECUTWFC_RY:.1f} Ry, "
                        f"achieved mean residual {sweep.best.mean_residual:.3e}). "
                        f"Set ecutwfc to a number larger than "
                        f"{PW_PGTO_AUTO_MAX_ECUTWFC_RY:.1f} Ry (manual mode)."
                    )
                proj = sweep.best
                logger.info(
                    "Auto pgto+pw cutoff: target=%.2e MET, selected g_cut=%.3f "
                    "Bohr^-1 (ecutwfc=%.3f Ry), n_pw=%d, mean residual=%.3e.",
                    PW_PGTO_AUTO_RESIDUAL_TARGET,
                    proj.g_cut,
                    proj.g_cut**2,
                    proj.n_pw,
                    proj.mean_residual,
                )
                return proj
            if auto_pw_cutoff and g_cut_fixed is not None:
                # LSDA dn channel: reuse the up-selected cutoff, but retain the
                # exact same auto-sweep overlap convention.  Falling through
                # to ``project_pw_to_mixed`` would switch S_gg to the full FFT
                # cube and make the dn residual incomparable with the auto
                # criterion used to select/optimise this shared split.
                fixed_sweep = optimize_mixed_basis(
                    ref,
                    shells,
                    residual_target=PW_PGTO_AUTO_RESIDUAL_TARGET,
                    g_cut_values=[float(g_cut_fixed)],
                    drop_diffuse=True,
                    fixed_alpha_min=(None if alpha_min_val is None else float(alpha_min_val)),
                    svd_rcond=svd_rcond,
                    occ_only=occ_only,
                )
                return fixed_sweep.best
            if g_cut_fixed is not None:
                g_cut_val = float(g_cut_fixed)
            elif g_cut is not None:
                g_cut_val = float(g_cut)
            else:
                g_max = float(np.sqrt(2.0 * ref.ecutwfc))
                g_cut_val = 0.5 * g_max
            return project_pw_to_mixed(
                ref,
                shells,
                g_cut=g_cut_val,
                drop_diffuse=True,
                occ_only=occ_only,
                svd_rcond=svd_rcond,
                alpha_min=alpha_min_val,
            )
        if kind == "pgto+bspline":
            return project_pw_to_pgto_spline(
                ref,
                shells,
                mesh_factor=mesh_factor,
                order=spline_order,
                alpha_min=(0.0 if alpha_min_val is None else alpha_min_val),
                svd_rcond=svd_rcond,
                occ_only=occ_only,
            )
        raise ValueError(f"Unknown kind {kind!r}; expected one of 'pw', 'bspline', 'pgto', 'pgto+pw', 'pgto+bspline'.")

    tiled_references: list[QEReference] | None = None
    tiled_projections: list[ProjectionResult] | None = None
    tiled_references_dn: list[QEReference] | None = None
    tiled_projections_dn: list[ProjectionResult] | None = None
    if tiling_matrix is None:
        if reference_dn is None:
            projection = _project_reference(reference)
            projection_dn = None
        else:
            projection = _project_reference(reference)
            # pgto+pw 'auto': the up-channel sweep already selected the
            # handoff g_cut; the dn channel reuses it so both spin blocks
            # share the same PW / pGTO split (uniform basis quality).
            _g_cut_shared = float(projection.g_cut) if (kind == "pgto+pw" and auto_pw_cutoff) else None
            projection_dn = _project_reference(reference_dn, g_cut_fixed=_g_cut_shared)
    else:
        assert tiled_sector is not None
        tiled_references = []
        tiled_projections = []
        if is_lsda:
            tiled_references_dn = []
            tiled_projections_dn = []

        for source_index in tiled_sector[1]:
            ik = source_index + 1
            if is_lsda:
                ref_up_full = load_qe_reference(qe_save_dir, ik=ik, spin="up")
                ref_dn_full = load_qe_reference(qe_save_dir, ik=ik, spin="dw")
                check_binary_occupations(ref_up_full.bands.occupations, channel=f"up k[{ik}]")
                check_binary_occupations(ref_dn_full.bands.occupations, channel=f"dw k[{ik}]")
                occ_idx_up = ref_up_full.bands.occupied_band_indices()
                occ_idx_dn = ref_dn_full.bands.occupied_band_indices()
                if occ_idx_up.size == 0 or occ_idx_dn.size == 0:
                    raise ValueError(f"LSDA k-point {ik} has an empty spin channel; cannot build a tiled determinant.")
                ref_up = truncate_qe_reference_bands(ref_up_full, int(occ_idx_up.max()) + 1)
                ref_dn = truncate_qe_reference_bands(ref_dn_full, int(occ_idx_dn.max()) + 1)
            else:
                ref_up = load_qe_reference(qe_save_dir, ik=ik)
                check_binary_occupations(ref_up.bands.occupations, channel=f"k[{ik}]")
                ref_dn = None

            if kind in PW_TAPER_KINDS:
                ref_up, taper_meta_up = taper_qe_reference(ref_up, g_end=_taper_edge, label=f"_k{ik}")
                if taper_meta_up is not None:
                    if pw_taper_meta is None:
                        pw_taper_meta = {}
                    pw_taper_meta.update(taper_meta_up)
                if ref_dn is not None:
                    ref_dn, taper_meta_dn = taper_qe_reference(ref_dn, g_end=_taper_edge, label=f"_dn_k{ik}")
                    if taper_meta_dn is not None:
                        if pw_taper_meta is None:
                            pw_taper_meta = {}
                        pw_taper_meta.update(taper_meta_dn)

            tiled_references.append(ref_up)
            tiled_projections.append(_project_reference(ref_up))
            if ref_dn is not None:
                assert tiled_references_dn is not None and tiled_projections_dn is not None
                tiled_references_dn.append(ref_dn)
                tiled_projections_dn.append(_project_reference(ref_dn))

        # Retain these names for the non-tiled assembly below; the tiled branch
        # consumes the complete lists instead.
        projection = tiled_projections[0]
        projection_dn = None if tiled_projections_dn is None else tiled_projections_dn[0]

    # --- Jastrow + Hamiltonian assembly --------------------------------
    j2_param = None if (j2_parameter is None or float(j2_parameter) < 0.0) else float(j2_parameter)
    j1_param = None if j1_parameter is None else float(j1_parameter)
    z_cores_tuple = tuple(float(z) for z in ecp_z_cores) if ecp_z_cores else None

    deduced_z_cores: tuple[float, ...] | None = None
    if strategy is not None:
        # Z from the element (labels may be tagged, e.g. 'O1'); z_valence from
        # the species UPF, keyed by label or element like every UPF lookup.
        z_nuc = _convert_from_atomic_labels_to_atomic_numbers(list(reference.atom_elements))
        deduced_z_cores = tuple(
            float(z) - strategy.z_valence_for_element(_upf_species_key(strategy, label, element))
            for z, label, element in zip(z_nuc, reference.atom_labels, reference.atom_elements, strict=True)
        )

    ecp_by_element: dict[str, ECPData] | None = None
    if strategy is not None:
        ecp_by_element = {}
        for atom_label, atom_element in zip(reference.atom_labels, reference.atom_elements, strict=True):
            key = _upf_species_key(strategy, atom_label, atom_element)
            if not strategy.is_coulomb_ae_for_element(key) and atom_label not in ecp_by_element:
                ecp_by_element[atom_label] = strategy.ecp_for_element(key)
        if not ecp_by_element:
            ecp_by_element = None
        elif z_cores_tuple is None:
            z_cores_tuple = deduced_z_cores
    if z_cores_tuple is not None and any(float(z) > 0.0 for z in z_cores_tuple) and strategy is None:
        raise ValueError(
            f"ECP atom(s) detected (z_cores={tuple(z_cores_tuple)}) but no upf_paths "
            f"supplied. The QE path extracts the analytic ccECP from each UPF's "
            f"[Gaussinfo] block; provide upf_paths=['element=path.upf', ...]."
        )

    logger.info("")

    if tiling_matrix is None:
        structure_data_qe, aos_data_qe, geminal_data_qe = build_wavefunction_from_projection(
            reference,
            projection,
            kind=kind,
            qe_reference_dn=reference_dn,
            projection_result_dn=projection_dn,
            orthonormalize=orthonormalize,
        )
        z_cores_qe = z_cores_tuple
    else:
        assert tiled_references is not None and tiled_projections is not None
        from .qe_orbital_tiling import build_tiled_wavefunction

        structure_data_qe, aos_data_qe, geminal_data_qe = build_tiled_wavefunction(
            tiled_references,
            tiled_projections,
            kind=kind,
            tiling_matrix=tiling_matrix,
            supertwist=tiled_sector[0],
            references_down=tiled_references_dn,
            projections_down=tiled_projections_dn,
            orthonormalize=orthonormalize,
        )
        # ``tile_structure_data`` orders atoms translation-major, so repeating
        # the primitive per-atom core list preserves atom alignment exactly.
        z_cores_qe = None if z_cores_tuple is None else z_cores_tuple * tiling_determinant
    mos_data_qe = getattr(geminal_data_qe, "orb_data_up_spin", None)

    j3_choice = j3_basis_type if str(j3_basis_type).lower() != "none" else None
    if j3_choice is None or str(j3_choice).lower() in ("none", ""):
        jastrow_three_body_data_qe = None
    elif tiling_matrix is not None:
        from .jqmc_tool import build_supercell_jastrow_three_body

        # Physical atoms are translation-major; the independent J3 shell list
        # follows the same order. It never consumes the Slater k blocks.
        jastrow_three_body_data_qe = build_supercell_jastrow_three_body(
            structure_data_qe, shells * tiling_determinant, str(j3_choice).lower()
        )
    else:
        # Keep this import local because jqmc_tool imports run_qe2jqmc lazily.
        from .jqmc_tool import build_jastrow_three_body_for_basis

        jastrow_three_body_data_qe = build_jastrow_three_body_for_basis(
            aos_data_qe,
            mos_data_qe,
            str(j3_choice).lower(),
        )
    coulomb_potential_data_qe, _ = _build_qe_coulomb_potential_data(
        structure_data_qe,
        int(geminal_data_qe.num_electron_up) + int(geminal_data_qe.num_electron_dn),
        z_cores_qe,
        ecp_by_element,
    )
    jastrow_dtn_data_qe = _build_jastrow_dtn(
        structure_data_qe,
        jastrow_dtn=jastrow_dtn,
        coulomb_potential_data=coulomb_potential_data_qe,
        dtn_c=dtn_c,
        dtn_n_u=dtn_n_u,
        dtn_n_chi=dtn_n_chi,
        dtn_n_eN=dtn_n_eN,
        dtn_n_ee=dtn_n_ee,
        dtn_cutoff_ratio_u=dtn_cutoff_ratio_u,
        dtn_cutoff_ratio_chi=dtn_cutoff_ratio_chi,
        dtn_cutoff_ratio_f=dtn_cutoff_ratio_f,
    )

    if jastrow_dtn_data_qe is not None and (
        (j1_param is not None) or (j2_param is not None) or (jastrow_three_body_data_qe is not None)
    ):
        raise ValueError(
            "jastrow_dtn != 'none' is mutually exclusive with j1_parameter, "
            "j2_parameter, and j3_basis_type. Disable the Pade J1/J2 + AO J3 "
            "family (or set jastrow_dtn='none')."
        )

    hamiltonian_data = build_hamiltonian_from_wavefunction(
        structure_data_qe,
        aos_data_qe,
        geminal_data_qe,
        j1_param=j1_param,
        j1_type=j1_type,
        j2_param=j2_param,
        j2_type=j2_type,
        ewald_tol=float(ewald_tol),
        ecp_z_cores=z_cores_qe,
        ecp_by_element=ecp_by_element,
        jastrow_three_body_data=jastrow_three_body_data_qe,
        jastrow_dtn_data=jastrow_dtn_data_qe,
    )
    logger.info("")

    hamiltonian_data.save_to_hdf5(hamiltonian_h5)
    logger.info("Hamiltonian data is saved in %s.", hamiltonian_h5)
    if pw_taper_meta is not None:
        # Recorded only when applied, so untapered (bulk) files are unchanged.
        import h5py

        with h5py.File(hamiltonian_h5, "a") as f:
            for key, value in pw_taper_meta.items():
                f.attrs[key] = value
