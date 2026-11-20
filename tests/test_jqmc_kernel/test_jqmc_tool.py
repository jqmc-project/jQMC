"""Unit tests for jqmc_tool trexio conversions."""

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

import dataclasses
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from click import unstyle
from typer import BadParameter

from jqmc import output_parser

pytestmark = [pytest.mark.with_chdir]
import tomlkit

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance
from jqmc._setting import (
    ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    ATOL_PYTHON_FP64_BZ_MEAN,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_TOOL_EXTRAPOLATED_ENERGY_DISPLAY,
    ATOL_TOOL_FORCE_TEXT_ROUNDTRIP,
    RTOL_BIT_EXACT,
    RTOL_PYTHON_FP64_AGGREGATOR_LOOSE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_ZERO,
    get_tol_hdf5_roundtrip,
)
from jqmc.jqmc_tool import (
    _J3_PERIOD_RANGES,
    build_jastrow_three_body_for_basis,
    hamiltonian_show_info,
    hamiltonian_to_xyz,
    lrdmc_generate_input,
    mcmc_generate_input,
    qe_convert_to,
    trexio_convert_to,
    trexio_show_detail,
    trexio_show_info,
    vmc_analyze_output,
    vmc_generate_input,
)
from jqmc.trexio_wrapper import read_trexio_file
from tests._test_data_cache import qe_example_files_root
from tests._fixed_sample_statistics import equal_bin_statistics
from tests._xyz_reference import assert_xyz_matches_trexio


def _dataclass_leaves(value, path=""):
    """Every non-dataclass leaf of a nested dataclass tree, keyed by its attribute path."""
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _dataclass_leaves(getattr(value, field.name), f"{path}.{field.name}")
    elif isinstance(value, (tuple, list)) and any(dataclasses.is_dataclass(item) for item in value):
        for index, item in enumerate(value):
            yield from _dataclass_leaves(item, f"{path}[{index}]")
    else:
        yield path, value


trexio_files = [
    "H2_ecp_ccpvtz_cart.h5",
    "N2_ecp_ccpvtz_cart.h5",
    "Cl2_ecp_ccpvtz_cart.h5",
    "Ti2_ecp_ccpvtz_cart.h5",
    "CuBr_ecp_ccpvtz_cart.h5",
]

# Inputs of the J3 distinct-primitive test (issue #112): general-contraction
# bases (shared primitives across contracted shells) in Cartesian and spherical
# form, molecular and periodic, plus a segmented control (N2).
_J3_DISTINCT_PRIMITIVE_FILES = [
    "CuBr_ecp_ccpvtz_cart.h5",
    "N_ae_ccpvdz_cart.h5",
    "Ar_ae_ccpvdz_cart.h5",
    "Ar_ae_ccpvdz_sphe.h5",
    "Li_pbc_ae_ccpvdz_cart.h5",
    "Li_pbc_ae_ccpvdz_sphe.h5",
    "N2_ecp_ccpvtz_cart.h5",
]


# Invalid Pade/exp J1/J2 options of ``trexio convert-to`` (issue #590):
# case -> (TREXIO fixture, extra CLI arguments, expected message fragment).
_INVALID_PADE_JASTROW_OPTIONS = {
    "j2-negative": ("H2_ae_ccpvdz_cart.h5", ["-j2", "-1.0"], "jastrow_2b_param = -1.0 must be non-negative"),
    "j1-negative": ("H2_ae_ccpvdz_cart.h5", ["-j1", "-1.0"], "jastrow_1b_param = -1.0 must be non-negative"),
    "j2-unknown-type": (
        "H2_ae_ccpvdz_cart.h5",
        ["-j2", "0.5", "--jastrow-2b-type", "foo"],
        "jastrow_2b_type = 'foo' must be 'exp' or 'pade'",
    ),
    "j1-unknown-type": (
        "H2_ae_ccpvdz_cart.h5",
        ["-j1", "0.5", "--jastrow-1b-type", "foo"],
        "jastrow_1b_type = 'foo' must be 'exp' or 'pade'",
    ),
    "j2-nan": ("H2_ae_ccpvdz_cart.h5", ["-j2", "nan"], "jastrow_2b_param = nan must be finite"),
    "j1-nan": ("H2_ae_ccpvdz_cart.h5", ["-j1", "nan"], "jastrow_1b_param = nan must be finite"),
    "j2-inf": ("H2_ae_ccpvdz_cart.h5", ["-j2", "inf"], "jastrow_2b_param = inf must be finite"),
    "twist-average-j2-nan": (
        "lithium_twist_222.h5",
        ["--twist-average", "-j2", "nan"],
        "jastrow_2b_param = nan must be finite",
    ),
}


class TestTrexioConvertTo:
    """Tests for trexio convert-to AO basis selection."""

    @pytest.mark.parametrize(
        "trexio_filename",
        trexio_files,
        ids=trexio_files,
    )
    @pytest.mark.light
    def test_ao_full_matches_uncontracted(self, tmp_path, trexio_filename):
        """Ensure ao-full keeps the full uncontracted AO basis."""
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_filename)
        hamiltonian_file = os.path.join(tmp_path, "hamiltonian_data.h5")

        _, aos_data, _, _, _, _ = read_trexio_file(trexio_file, store_tuple=True)
        uncontracted = aos_data._build_uncontracted_aos()

        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-full",
        )

        from jqmc.hamiltonians import Hamiltonian_data

        hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        orb_data = hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data

        assert orb_data.num_ao == uncontracted.num_ao
        assert orb_data.num_ao_prim == uncontracted.num_ao_prim

        uncontracted_exps = [None] * uncontracted.num_ao
        for p, orb in enumerate(uncontracted.orbital_indices):
            if uncontracted_exps[orb] is None:
                uncontracted_exps[orb] = float(uncontracted.exponents[p])

        expected_counts: dict[tuple[int, int, float], int] = {}
        for i in range(uncontracted.num_ao):
            key = (uncontracted.nucleus_index[i], uncontracted.angular_momentums[i], uncontracted_exps[i])
            expected_counts[key] = expected_counts.get(key, 0) + 1

        orb_exps = [None] * orb_data.num_ao
        for p, orb in enumerate(orb_data.orbital_indices):
            if orb_exps[orb] is None:
                orb_exps[orb] = float(orb_data.exponents[p])

        actual_counts: dict[tuple[int, int, float], int] = {}
        for i in range(orb_data.num_ao):
            key = (orb_data.nucleus_index[i], orb_data.angular_momentums[i], orb_exps[i])
            actual_counts[key] = actual_counts.get(key, 0) + 1
        assert actual_counts == expected_counts

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_filename", trexio_files, ids=trexio_files)
    @pytest.mark.parametrize("choice", ["ao-small", "ao-medium", "ao-large"])
    def test_ao_selection(self, tmp_path, trexio_filename, choice):
        """Ensure AO selection matches the expected shell-by-shell choice."""
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_filename)
        hamiltonian_file = os.path.join(tmp_path, f"hamiltonian_{choice}.h5")

        j3_ao_selection_table = {
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
        j3_transition_z_ranges = (
            (21, 30),
            (39, 48),
            (57, 71),
            (72, 80),
            (89, 103),
            (104, 112),
        )
        j3_shell_l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

        _, aos_data, _, _, _, _ = read_trexio_file(trexio_file, store_tuple=True)
        uncontracted = aos_data._build_uncontracted_aos()

        ao_exps = [None] * uncontracted.num_ao
        for p, orb in enumerate(uncontracted.orbital_indices):
            if ao_exps[orb] is None:
                ao_exps[orb] = float(uncontracted.exponents[p])

        expected_indices = []
        if choice in ("ao", "ao-full"):
            expected_indices = list(range(uncontracted.num_ao))
        else:
            for nucleus in set(uncontracted.nucleus_index):
                ao_idxs = [i for i, nuc in enumerate(uncontracted.nucleus_index) if nuc == nucleus]
                if not ao_idxs:
                    continue

                z = uncontracted.structure_data.atomic_numbers[nucleus]
                period = 7
                for idx, (zmin, zmax) in enumerate(_J3_PERIOD_RANGES, start=1):
                    if zmin <= z <= zmax:
                        period = idx
                        break

                if period == 1:
                    key = "period1"
                elif period == 2:
                    key = "period2"
                elif period == 3:
                    key = "period3"
                else:
                    is_transition = any(zmin <= z <= zmax for zmin, zmax in j3_transition_z_ranges)
                    key = "period4plus_transition" if is_transition else "period4plus_main"

                spec = j3_ao_selection_table[key][choice]
                shell_counts: dict[int, int] = {}
                for count_str, shell in re.findall(r"(\d+)([spdfgh])", spec):
                    l = j3_shell_l_map[shell]
                    shell_counts[l] = int(count_str)

                for l in set(uncontracted.angular_momentums):
                    ao_idxs_l = [i for i in ao_idxs if uncontracted.angular_momentums[i] == l]
                    if not ao_idxs_l:
                        continue

                    n_shells = shell_counts.get(l, 0)
                    if n_shells <= 0:
                        continue

                    basis_exps = sorted({float(ao_exps[i]) for i in ao_idxs_l})
                    if n_shells >= len(basis_exps):
                        sel_basis = basis_exps
                    else:
                        log_exps = np.log(basis_exps)
                        frac = n_shells / len(basis_exps)
                        margin = (1.0 - frac) / 2.5
                        cdf = np.linspace(0, 1, len(log_exps))
                        lo = np.interp(margin, cdf, log_exps)
                        hi = np.interp(1.0 - margin, cdf, log_exps)
                        targets = np.linspace(lo, hi, n_shells)
                        sel_basis = []
                        remaining = list(basis_exps)
                        log_remaining = list(log_exps)
                        for t in targets:
                            idx = int(np.argmin([abs(lr - t) for lr in log_remaining]))
                            sel_basis.append(remaining.pop(idx))
                            log_remaining.pop(idx)
                        sel_basis.sort()

                    expected_indices.extend([i for i in ao_idxs_l if float(ao_exps[i]) in sel_basis])

            expected_indices = sorted(set(expected_indices))

        expected_counts: dict[tuple[int, int, float], int] = {}
        for i in expected_indices:
            key = (uncontracted.nucleus_index[i], uncontracted.angular_momentums[i], ao_exps[i])
            expected_counts[key] = expected_counts.get(key, 0) + 1

        expected_shells: dict[int, str] = {}
        for nucleus in set(uncontracted.nucleus_index):
            shells: dict[int, set[float]] = {}
            for i in expected_indices:
                if uncontracted.nucleus_index[i] != nucleus:
                    continue
                shells.setdefault(uncontracted.angular_momentums[i], set()).add(float(ao_exps[i]))
            parts = []
            for l, label in [(0, "s"), (1, "p"), (2, "d"), (3, "f"), (4, "g"), (5, "h")]:
                count = len(shells.get(l, set()))
                if count > 0:
                    parts.append(f"{count}{label}")
            expected_shells[nucleus] = "".join(parts)

        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type=choice,
        )

        from jqmc.hamiltonians import Hamiltonian_data

        hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        orb_data = hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data

        orb_exps = [None] * orb_data.num_ao
        for p, orb in enumerate(orb_data.orbital_indices):
            if orb_exps[orb] is None:
                orb_exps[orb] = float(orb_data.exponents[p])

        actual_counts: dict[tuple[int, int, float], int] = {}
        for i in range(orb_data.num_ao):
            key = (orb_data.nucleus_index[i], orb_data.angular_momentums[i], orb_exps[i])
            actual_counts[key] = actual_counts.get(key, 0) + 1

        actual_shells: dict[int, str] = {}
        for nucleus in set(orb_data.nucleus_index):
            shells: dict[int, set[float]] = {}
            for i in range(orb_data.num_ao):
                if orb_data.nucleus_index[i] != nucleus:
                    continue
                shells.setdefault(orb_data.angular_momentums[i], set()).add(orb_exps[i])
            parts = []
            for l, label in [(0, "s"), (1, "p"), (2, "d"), (3, "f"), (4, "g"), (5, "h")]:
                count = len(shells.get(l, set()))
                if count > 0:
                    parts.append(f"{count}{label}")
            actual_shells[nucleus] = "".join(parts)

        for nucleus in sorted(expected_shells):
            expected = expected_shells[nucleus]
            actual = actual_shells.get(nucleus, "")
            print(f"nucleus {nucleus}: expected [{expected}] actual [{actual}]")
        assert actual_counts == expected_counts

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_filename", _J3_DISTINCT_PRIMITIVE_FILES, ids=_J3_DISTINCT_PRIMITIVE_FILES)
    @pytest.mark.parametrize("choice", ["ao", "ao-full", "ao-small", "ao-medium", "ao-large"])
    def test_j3_ao_basis_keeps_each_distinct_trexio_primitive_once(self, tmp_path, trexio_filename, choice):
        """A J3 ``ao*`` basis holds each distinct input primitive at most once.

        Independent reference: the TREXIO basis group read with the ``trexio``
        package (not through jqmc). Its distinct primitives are the distinct
        ``(nucleus, l, exponent)`` triples; each carries every angular component
        of ``l`` (Cartesian ``(nx, ny, nz)`` with ``nx + ny + nz = l``, or real
        spherical ``m = -l..l``). The general-contraction inputs (Dunning
        cc-pVDZ all-electron Ar / N / periodic Li, ccECP cc-pVTZ CuBr) list some
        primitives in several contracted shells; the segmented N2 input is a
        control. ``ao`` / ``ao-full`` must hold exactly the reference set;
        ``ao-small/medium/large`` must hold, for each atom and l,
        ``min(table count, distinct exponents)`` radial shells (the documented
        shell table, e.g. ``4s2p`` for period-3 ``ao-small``), each a reference
        exponent with all its components exactly once.
        """
        import trexio

        from jqmc.atomic_orbital import AOs_pGTO_base_data, AOs_sphe_gto_data
        from jqmc.hamiltonians import Hamiltonian_data, _detile_aos_data_for_save

        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_filename)

        # The documented shell table (doc/examples.md: ao-small is 3s1p for O,
        # 3s for H), written out per element for the elements of these inputs.
        shell_table = {
            "period2": {"ao-small": "3s1p", "ao-medium": "4s2p1d", "ao-large": "5s3p2d1f"},
            "period3": {"ao-small": "4s2p", "ao-medium": "5s3p1d", "ao-large": "6s4p3d1f"},
            "period4plus_main": {"ao-small": "4s2p1d", "ao-medium": "5s3p2d1f", "ao-large": "6s4p3d2f"},
            "period4plus_transition": {
                "ao-small": "4s2p2d1f",
                "ao-medium": "5s3p3d2f1g",
                "ao-large": "6s4p4d3f2g1h",
            },
        }
        element_row = {
            "Li": "period2",
            "N": "period2",
            "Ar": "period3",
            "Br": "period4plus_main",
            "Cu": "period4plus_transition",
        }
        shell_l = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

        # --- independent reference from the raw TREXIO basis tables ---
        tf = trexio.File(trexio_file, "r", trexio.TREXIO_AUTO)
        try:
            labels = [str(label).strip() for label in trexio.read_nucleus_label(tf)]
            shell_nucleus = np.asarray(trexio.read_basis_nucleus_index(tf))
            shell_ang_mom = np.asarray(trexio.read_basis_shell_ang_mom(tf))
            prim_shell = np.asarray(trexio.read_basis_shell_index(tf))
            prim_exponent = np.asarray(trexio.read_basis_exponent(tf), dtype=np.float64)
            is_spherical = trexio.read_ao_cartesian(tf) == 0
        finally:
            tf.close()

        # Basis parameters are fp64 storage in both precision modes. Evaluation
        # may round them later, but must not change which functions are stored.
        exponent_dtype = np.float64

        def _stored(exponent: float) -> float:
            return float(np.asarray(exponent, dtype=exponent_dtype))

        reference_exponents: dict[tuple[int, int], set[float]] = {}
        raw_exponents: dict[tuple[int, int], set[float]] = {}
        for p, shell in enumerate(prim_shell):
            nl = (int(shell_nucleus[shell]), int(shell_ang_mom[shell]))
            reference_exponents.setdefault(nl, set()).add(_stored(prim_exponent[p]))
            raw_exponents.setdefault(nl, set()).add(float(prim_exponent[p]))
        # The storage rounding must not merge two distinct input exponents.
        assert {nl: len(v) for nl, v in reference_exponents.items()} == {nl: len(v) for nl, v in raw_exponents.items()}

        def _components(l: int) -> set:
            if is_spherical:
                return set(range(-l, l + 1))
            return {(nx, ny, l - nx - ny) for nx in range(l + 1) for ny in range(l + 1 - nx)}

        # --- jqmc: trexio convert-to with the requested J3 basis ---
        hamiltonian_file = os.path.join(tmp_path, f"hamiltonian_{choice}.h5")
        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type=choice,
        )
        orb_data = Hamiltonian_data.load_from_hdf5(
            hamiltonian_file
        ).wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
        if isinstance(orb_data, AOs_pGTO_base_data):
            orb_data = _detile_aos_data_for_save(orb_data)
        assert isinstance(orb_data, AOs_sphe_gto_data) == is_spherical
        # An uncontracted basis: one primitive per AO.
        assert orb_data.num_ao_prim == orb_data.num_ao
        assert sorted(int(i) for i in orb_data.orbital_indices) == list(range(orb_data.num_ao))

        exponent_by_ao = [None] * orb_data.num_ao
        for p, orb in enumerate(orb_data.orbital_indices):
            exponent_by_ao[int(orb)] = _stored(orb_data.exponents[p])
        actual_keys = []
        for i in range(orb_data.num_ao):
            if is_spherical:
                component = int(orb_data.magnetic_quantum_numbers[i])
            else:
                component = (
                    int(orb_data.polynominal_order_x[i]),
                    int(orb_data.polynominal_order_y[i]),
                    int(orb_data.polynominal_order_z[i]),
                )
            actual_keys.append(
                (int(orb_data.nucleus_index[i]), int(orb_data.angular_momentums[i]), component, exponent_by_ao[i])
            )

        # (a) no basis function appears twice.
        assert len(actual_keys) == len(set(actual_keys)), (
            f"{trexio_filename} {choice}: {len(actual_keys) - len(set(actual_keys))} duplicate J3 basis functions "
            f"in {len(actual_keys)}"
        )

        # (b) per (nucleus, l): the selected exponents are reference exponents,
        # each with every component, and their number is the expected one.
        actual_by_nl: dict[tuple[int, int], dict[float, set]] = {}
        for nucleus, l, component, exponent in actual_keys:
            actual_by_nl.setdefault((nucleus, l), {}).setdefault(exponent, set()).add(component)

        expected_num_ao = 0
        for (nucleus, l), exponents in sorted(reference_exponents.items()):
            if choice in ("ao", "ao-full"):
                n_expected = len(exponents)
            else:
                spec = shell_table[element_row[labels[nucleus]]][choice]
                counts = {shell_l[name]: int(n) for n, name in re.findall(r"(\d+)([spdfgh])", spec)}
                n_expected = min(counts.get(l, 0), len(exponents))
            selected = actual_by_nl.get((nucleus, l), {})
            assert len(selected) == n_expected, (
                f"{trexio_filename} {choice}: nucleus {nucleus} ({labels[nucleus]}) l={l}: "
                f"{len(selected)} radial shells, expected {n_expected} of {len(exponents)} distinct"
            )
            assert set(selected) <= exponents
            for exponent, components in selected.items():
                assert components == _components(l)
            expected_num_ao += n_expected * len(_components(l))
        assert set(actual_by_nl) <= set(reference_exponents)
        assert orb_data.num_ao == expected_num_ao

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("trexio_filename", "merged_indices", "lowest_index"),
        [
            # N cc-pVDZ (Cartesian): the 0.7466 s exponent of the two general-
            # contraction shells (primitives 7 and 15) and the 0.2248 s shell (16).
            ("N_ae_ccpvdz_cart.h5", [7, 15], 16),
            # Ar cc-pVDZ (spherical twin): the 0.6504 s exponent of the three
            # general-contraction shells (10, 21, 32) and the 0.2337 s shell (33).
            ("Ar_ae_ccpvdz_sphe.h5", [10, 21, 32], 33),
        ],
        ids=["N_ae_ccpvdz_cart.h5", "Ar_ae_ccpvdz_sphe.h5"],
    )
    def test_j3_ao_full_basis_functions_distinct_in_fp64_storage(self, tmp_path, trexio_filename, merged_indices, lowest_index):
        """Distinct fp64 J3 ``ao-full`` parameters remain distinct in both modes.

        Boundary input: a temporary copy of a committed TREXIO file whose s
        exponents are rewritten to ``1`` (one shell) and ``1 + 2**-25`` (the
        other shells' shared primitive). They differ in fp64 but coincide after
        fp32 evaluation rounding. Storage and duplicate removal must use the
        exact fp64 parameters, independently of the active evaluation policy.

        Independent reference: the stored signatures (center, angular
        component, exponent and coefficient) must be pairwise distinct and
        equal the set counted directly from raw TREXIO tables using ``trexio``.
        Shared exact duplicates still occur only once.
        """
        import shutil

        import trexio

        from jqmc.atomic_orbital import AOs_pGTO_base_data, AOs_sphe_gto_data
        from jqmc.hamiltonians import Hamiltonian_data, _detile_aos_data_for_save

        source = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_filename)
        trexio_file = os.path.join(tmp_path, trexio_filename)
        shutil.copyfile(source, trexio_file)
        one = 1.0
        one_plus = 1.0 + 2.0**-25
        with h5py.File(trexio_file, "r+") as f:
            exponents = np.array(f["basis/basis_exponent"][()], dtype=np.float64)
            exponents[lowest_index] = one
            for index in merged_indices:
                exponents[index] = one_plus
            f["basis/basis_exponent"][...] = exponents

        # --- independent reference from the raw TREXIO tables ---
        tf = trexio.File(trexio_file, "r", trexio.TREXIO_AUTO)
        try:
            coords = np.asarray(trexio.read_nucleus_coord(tf), dtype=np.float64)
            shell_nucleus = np.asarray(trexio.read_basis_nucleus_index(tf))
            shell_ang_mom = np.asarray(trexio.read_basis_shell_ang_mom(tf))
            prim_shell = np.asarray(trexio.read_basis_shell_index(tf))
            prim_exponent = np.asarray(trexio.read_basis_exponent(tf), dtype=np.float64)
            is_spherical = trexio.read_ao_cartesian(tf) == 0
        finally:
            tf.close()
        # The rewrite is read back from the raw file: two fp64-distinct values.
        assert prim_exponent[lowest_index] == one
        assert all(prim_exponent[i] == one_plus for i in merged_indices)
        assert one != one_plus

        storage_dtype = np.float64

        def _stored(value: float) -> float:
            return float(np.asarray(value, dtype=storage_dtype))

        def _components(l: int) -> list:
            if is_spherical:
                return list(range(-l, l + 1))
            return [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]

        expected_signatures = set()
        for p, shell in enumerate(prim_shell):
            nucleus = int(shell_nucleus[shell])
            l = int(shell_ang_mom[shell])
            for component in _components(l):
                expected_signatures.add(
                    (tuple(float(c) for c in coords[nucleus]), l, component, _stored(prim_exponent[p]), _stored(1.0))
                )

        # --- jqmc: public trexio convert-to with the ao-full J3 basis ---
        hamiltonian_file = os.path.join(tmp_path, "hamiltonian_ao_full.h5")
        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-full",
        )
        orb_data = Hamiltonian_data.load_from_hdf5(
            hamiltonian_file
        ).wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
        if isinstance(orb_data, AOs_pGTO_base_data):
            orb_data = _detile_aos_data_for_save(orb_data)
        assert isinstance(orb_data, AOs_sphe_gto_data) == is_spherical
        assert orb_data.num_ao_prim == orb_data.num_ao
        assert np.asarray(orb_data.exponents).dtype == np.dtype(np.float64)
        assert np.asarray(orb_data.coefficients).dtype == np.dtype(np.float64)
        assert one in orb_data.exponents and one_plus in orb_data.exponents

        positions = np.asarray(orb_data.structure_data.positions, dtype=np.float64)
        prim_by_ao = {int(orb): p for p, orb in enumerate(orb_data.orbital_indices)}
        signatures = []
        for i in range(orb_data.num_ao):
            p = prim_by_ao[i]
            if is_spherical:
                component = int(orb_data.magnetic_quantum_numbers[i])
            else:
                component = (
                    int(orb_data.polynominal_order_x[i]),
                    int(orb_data.polynominal_order_y[i]),
                    int(orb_data.polynominal_order_z[i]),
                )
            signatures.append(
                (
                    tuple(float(c) for c in positions[int(orb_data.nucleus_index[i])]),
                    int(orb_data.angular_momentums[i]),
                    component,
                    float(orb_data.exponents[p]),
                    float(orb_data.coefficients[p]),
                )
            )

        duplicates = len(signatures) - len(set(signatures))
        assert duplicates == 0, (
            f"{trexio_filename}: {duplicates} identical stored J3 basis functions in {len(signatures)} "
            f"(basis storage dtype {np.dtype(storage_dtype).name})"
        )
        assert set(signatures) == expected_signatures

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_pbc_file", ["lithium.h5", "lithium_k025.h5"])
    @pytest.mark.parametrize("j3_basis_type", ["ao-small", "ao-full"])
    def test_trexio_convert_to_pbc_j3_load_roundtrip(self, tmp_path, trexio_pbc_file, j3_basis_type):
        """End-to-end PBC J3 convert + load roundtrip.

        ``jqmc-tool trexio convert-to ... -j3 ao-*`` must produce an h5
        that ``Hamiltonian_data.load_from_hdf5`` can read without
        ``OverflowError`` in ``_apply_image_expansion ->
        _build_image_expanded_arrays -> _primitive_image_radius``.

        Bug chain caught by this test:
          1. ``AOs_*_pgto_data._build_uncontracted_aos`` returns
             ``type(self)(...)`` without propagating pgto-only fields, so
             ``_image_truncation_tol`` falls back to its dataclass default
             ``0.0`` (and ``_n_image`` to ``1``, ``k_point`` to ``None``
             on the complex Bloch path).
          2. ``_detile_aos_data_for_save`` carries the buggy
             ``_image_truncation_tol=0.0`` onto the saved gto leaf.
          3. ``_apply_image_expansion_post_load`` reads it and feeds
             ``log_inv_tol = max(-log(0), 1.0) = +inf`` to
             ``_primitive_image_radius`` -> ``r_cut = inf`` ->
             ``int(ceil(r_cut / L_min)) + 1`` -> ``OverflowError``.

        The molecular variants of this class cannot catch this failure
        because their ``aos_data`` is gto, not pgto, so the pgto-only
        fields are not part of the contract.

        Counterpart unit coverage at the source:
        ``test_atomic_orbital.py::TestPGTOSemantics::test_build_uncontracted_aos_preserves_pgto_fields``.

        The loaded J3, structure and up-spin MO/AO data are compared exactly
        with the data built from the same TREXIO source before saving.
        """
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_pbc_file)
        hamiltonian_file = os.path.join(tmp_path, f"hamiltonian_pbc_j3_{j3_basis_type}.h5")

        trexio_convert_to(
            trexio_file=trexio_file,
            hamiltonian_file=hamiltonian_file,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type=j3_basis_type,
        )

        from jqmc.hamiltonians import Hamiltonian_data

        hamiltonian = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        # ``sanity_check`` confirms the J3 leaf survived the full
        # convert/save/load chain in a self-consistent state (catches the
        # latent ``_n_image`` mismatch even after a partial fix to ``tol``).
        hamiltonian.sanity_check()
        j3 = hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data
        assert j3 is not None
        assert j3.orb_data.num_ao > 0
        # The round trip preserves what it saved: the loaded J3 equals the J3
        # the converter builds from the same TREXIO input, and the loaded
        # structure and up-spin MO/AO data equal the TREXIO source, leaf by leaf.
        source_structure, source_aos, source_mos_up, _, _, _ = read_trexio_file(trexio_file)
        source_j3 = build_jastrow_three_body_for_basis(source_aos, source_mos_up, j3_basis_type)
        for loaded, source in (
            (j3, source_j3),
            (hamiltonian.wavefunction_data.geminal_data.orb_data_up_spin, source_mos_up),
            (hamiltonian.structure_data, source_structure),
        ):
            loaded_leaves, source_leaves = dict(_dataclass_leaves(loaded)), dict(_dataclass_leaves(source))
            assert loaded_leaves.keys() == source_leaves.keys()
            for path, value in loaded_leaves.items():
                np.testing.assert_array_equal(np.asarray(value), np.asarray(source_leaves[path]), err_msg=path)

    @pytest.mark.light
    @pytest.mark.parametrize("case", list(_INVALID_PADE_JASTROW_OPTIONS))
    def test_trexio_convert_to_rejects_invalid_pade_jastrow_options(self, tmp_path, case):
        """Invalid Pade/exp J1/J2 options are usage errors and leave no Hamiltonian file.

        ``Jastrow_one_body_data.sanity_check`` / ``Jastrow_two_body_data.sanity_check``
        (applied by ``Hamiltonian_data.load_from_hdf5``) require a finite,
        non-negative parameter and a form in {'exp', 'pade'}. The converter
        must refuse what its own loader refuses, and a non-finite parameter
        (ln|Psi| = NaN for nan, a silently vanishing term for inf), with a
        non-zero exit, the offending value in the message, no success line
        and no output file.
        """
        from click.testing import CliRunner

        from jqmc.jqmc_tool import _cli

        trexio_filename, args, message = _INVALID_PADE_JASTROW_OPTIONS[case]
        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_filename)
        hamiltonian_file = tmp_path / "hamiltonian_data.h5"
        # The ``jqmc-tool`` console group, so the disposition is the one a shell user sees.
        result = CliRunner().invoke(_cli, ["trexio", "convert-to", trexio_file, "-o", str(hamiltonian_file), *args])
        assert result.exit_code != 0, result.output
        assert message in _flatten_cli_output(result.output)
        assert "is saved" not in result.output
        assert not hamiltonian_file.exists()

    @pytest.mark.light
    def test_trexio_convert_to_valid_pade_j2_matches_analytic_term(self, tmp_path):
        """A valid ``-j2`` still converts, and the reloaded file adds the analytic Pade J2 to ln|Psi| (control).

        For H2 (one up, one down electron) the two-body Jastrow is the single
        antiparallel Pade term u(r) = r / (2 (1 + b r)) with b = ``-j2``
        (doc/api_reference_cli_tool.md). ln|Psi| of the ``-j2 1.0`` file
        must equal ln|Psi| of the Jastrow-free conversion of the same TREXIO
        file plus u(|r_up - r_dn|) at a fixed configuration.
        """
        from click.testing import CliRunner

        from jqmc._precision import get_tolerance_min
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_tool import _cli
        from jqmc.wavefunction import evaluate_ln_wavefunction

        trexio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
        with_j2 = tmp_path / "with_j2.h5"
        without_jastrow = tmp_path / "without_jastrow.h5"
        b = 1.0
        for path, extra in ((with_j2, ["-j2", str(b)]), (without_jastrow, [])):
            result = CliRunner().invoke(_cli, ["trexio", "convert-to", trexio_file, "-o", str(path), *extra])
            assert result.exit_code == 0, (result.output, result.exception)
            assert path.is_file()

        r_up = np.array([[0.3, 0.2, 0.1]])
        r_dn = np.array([[0.9, -0.2, 0.5]])
        r_ee = float(np.linalg.norm(r_up[0] - r_dn[0]))
        u_pade = r_ee / (2.0 * (1.0 + b * r_ee))

        hamiltonian_j2 = Hamiltonian_data.load_from_hdf5(str(with_j2))
        hamiltonian_none = Hamiltonian_data.load_from_hdf5(str(without_jastrow))
        assert hamiltonian_none.wavefunction_data.jastrow_data.jastrow_two_body_data is None
        ln_psi_j2 = float(evaluate_ln_wavefunction(hamiltonian_j2.wavefunction_data, r_up, r_dn))
        ln_psi_none = float(evaluate_ln_wavefunction(hamiltonian_none.wavefunction_data, r_up, r_dn))
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval", "jastrow_eval", "wf_eval"), "strict")
        np.testing.assert_allclose(ln_psi_j2, ln_psi_none + u_pade, atol=atol, rtol=rtol)


_TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


def _flatten_cli_output(text):
    """Strip color and join lines wrapped in a Rich usage box."""
    return " ".join(re.sub(r"[\u2500-\u257f]", " ", unstyle(text)).split())


def _make_mcmc_obj(
    num_steps=100,
    num_walkers=4,
    num_atoms=2,
    num_elec_up=1,
    num_elec_dn=1,
    e_L_value=1.0,
    w_L_value=1.0,
    with_force=True,
    atomic_labels=None,
    rng=None,
):
    """Return a SimpleNamespace that mimics a serialised MCMC object."""
    if rng is None:
        rng = np.random.default_rng(42)

    shape_iw = (num_steps, num_walkers)
    e_L = np.full(shape_iw, e_L_value, dtype=np.float64)
    w_L = np.full(shape_iw, w_L_value, dtype=np.float64)

    obj = SimpleNamespace(e_L=e_L, w_L=w_L)

    if with_force:
        obj.comput_position_deriv = True

        if atomic_labels is None:
            atomic_labels = [f"X{i}" for i in range(num_atoms)]
        obj.hamiltonian_data = SimpleNamespace(structure_data=SimpleNamespace(atomic_labels=atomic_labels))

        shape_iwj3 = (num_steps, num_walkers, num_atoms, 3)
        shape_iwk3_up = (num_steps, num_walkers, num_elec_up, 3)
        shape_iwk3_dn = (num_steps, num_walkers, num_elec_dn, 3)
        shape_iwjk_up = (num_steps, num_walkers, num_atoms, num_elec_up)
        shape_iwjk_dn = (num_steps, num_walkers, num_atoms, num_elec_dn)

        # Compute force products from random raw gradients (matches MCMC storage)
        de_L_dR = rng.standard_normal(shape_iwj3) * 0.01
        de_L_dr_up = rng.standard_normal(shape_iwk3_up) * 0.01
        de_L_dr_dn = rng.standard_normal(shape_iwk3_dn) * 0.01
        dln_Psi_dR = rng.standard_normal(shape_iwj3) * 0.01
        dln_Psi_dr_up = rng.standard_normal(shape_iwk3_up) * 0.01
        dln_Psi_dr_dn = rng.standard_normal(shape_iwk3_dn) * 0.01
        _omega_up = rng.standard_normal(shape_iwjk_up) * 0.01
        _omega_dn = rng.standard_normal(shape_iwjk_dn) * 0.01
        domega_dr_up = rng.standard_normal(shape_iwj3) * 0.01
        domega_dr_dn = rng.standard_normal(shape_iwj3) * 0.01

        obj.force_HF = (
            de_L_dR + np.einsum("iwjk,iwkl->iwjl", _omega_up, de_L_dr_up) + np.einsum("iwjk,iwkl->iwjl", _omega_dn, de_L_dr_dn)
        )
        force_PP = (
            dln_Psi_dR
            + np.einsum("iwjk,iwkl->iwjl", _omega_up, dln_Psi_dr_up)
            + np.einsum("iwjk,iwkl->iwjl", _omega_dn, dln_Psi_dr_dn)
            + 0.5 * (domega_dr_up + domega_dr_dn)
        )
        obj.force_PP = force_PP
        obj.E_L_force_PP = np.einsum("iw,iwjk->iwjk", e_L, force_PP)
    else:
        obj.comput_position_deriv = False

    return obj


def _make_lrdmc_obj(
    num_steps=100,
    num_walkers=4,
    num_atoms=2,
    num_elec_up=1,
    num_elec_dn=1,
    e_L_value=1.0,
    w_L_value=1.0,
    alat=0.5,
    with_force=True,
    atomic_labels=None,
    rng=None,
):
    """Return a SimpleNamespace that mimics a serialised GFMC_n (LRDMC) object."""
    obj = _make_mcmc_obj(
        num_steps=num_steps,
        num_walkers=num_walkers,
        num_atoms=num_atoms,
        num_elec_up=num_elec_up,
        num_elec_dn=num_elec_dn,
        e_L_value=e_L_value,
        w_L_value=w_L_value,
        with_force=with_force,
        atomic_labels=atomic_labels,
        rng=rng,
    )
    obj.alat = alat
    obj.num_gfmc_collect_steps = 5
    return obj


# Mapping from SimpleNamespace attribute names to HDF5 observable dataset names.
_OBS_KEY_MAP = {
    "e_L": "e_L",
    "w_L": "w_L",
    "mpc_dV": "mpc_dV",
    "force_HF": "force_HF",
    "force_PP": "force_PP",
    "E_L_force_PP": "E_L_force_PP",
}


def _write_chk(path, objs, driver_type=None):
    """Write a list of objects as an HDF5 merged checkpoint (one rank per obj).

    ``driver_type`` defaults to ``"GFMC_n"`` for LRDMC objects (they carry
    ``alat``) and ``"MCMC"`` otherwise, as the real drivers stamp it.
    """
    if driver_type is None:
        driver_type = "GFMC_n" if hasattr(objs[0], "alat") else "MCMC"
    with h5py.File(path, "w") as f:
        # _meta group
        meta = f.create_group("_meta")
        meta.attrs["driver_type"] = driver_type
        meta.attrs["mpi_size"] = len(objs)

        # hamiltonian_data -- minimal stub for force tests
        first = objs[0]
        if hasattr(first, "hamiltonian_data"):
            _write_minimal_hamiltonian(f, first.hamiltonian_data)

        for rank, obj in enumerate(objs):
            rgrp = f.create_group(f"rank_{rank}")

            # driver_config
            cfg = rgrp.create_group("driver_config")
            cfg.attrs["comput_position_deriv"] = getattr(obj, "comput_position_deriv", False)
            if hasattr(obj, "alat"):
                cfg.attrs["alat"] = obj.alat
            if hasattr(obj, "num_gfmc_collect_steps"):
                cfg.attrs["num_gfmc_collect_steps"] = obj.num_gfmc_collect_steps
            for key in ("mpc", "mpc_cutoff"):
                if hasattr(obj, key):
                    cfg.attrs[key] = getattr(obj, key)

            # observables
            obs = rgrp.create_group("observables")
            for attr_name, h5_name in _OBS_KEY_MAP.items():
                arr = getattr(obj, attr_name, None)
                if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
                    obs.create_dataset(h5_name, data=arr)


def _write_minimal_hamiltonian(f, ham_ns):
    """Write a minimal hamiltonian_data group that load_hamiltonian_from_checkpoint can read.

    Mimics the structure produced by _save_dataclass_to_hdf5 for Hamiltonian_data,
    but only fills the fields that compute-force actually uses (atomic_labels).
    """
    hg = f.create_group("hamiltonian_data")
    hg.attrs["_class_name"] = "Hamiltonian_data"
    hg.attrs["_module_name"] = "jqmc.hamiltonians"

    sd = hg.create_group("structure_data")
    sd.attrs["_class_name"] = "Structure_data"
    sd.attrs["_module_name"] = "jqmc.structure"
    labels = ham_ns.structure_data.atomic_labels
    sd.create_dataset("atomic_labels", data=np.array(labels, dtype="S"))
    n = len(labels)
    sd.create_dataset("positions", data=np.zeros((n, 3)))
    sd.attrs["pbc_flag"] = False
    sd.create_dataset("atomic_numbers", data=np.ones(n, dtype=int))
    sd.create_dataset("element_symbols", data=np.array(labels, dtype="S"))
    sd.create_dataset("vec_a", data=np.empty(0))
    sd.create_dataset("vec_b", data=np.empty(0))
    sd.create_dataset("vec_c", data=np.empty(0))


def _write_harmonic_only_trexio_copy(source, target):
    """Copy a Cartesian TREXIO file, keeping MO coefficients only on harmonic monomials.

    A monomial ``x^a y^b z^c`` with every exponent <= 1 (1, x, y, z, xy, xz, yz,
    xyz) has zero Laplacian, so a radial factor times it is a pure solid
    harmonic: the resulting MOs lie exactly in the spherical subspace of the
    same d/f-containing Cartesian basis. Coefficients on every monomial with an
    exponent >= 2 (xx, xxy, ...) are set to zero. The component order is the
    TREXIO Cartesian convention within each shell (xx, xy, xz, yy, yz, zz, ...).
    """
    import shutil

    shutil.copyfile(source, target)
    with h5py.File(target, "r+") as h5:
        shell_of_ao = np.asarray(h5["ao/ao_shell"][...], dtype=np.int64)
        shell_l = np.asarray(h5["basis/basis_shell_ang_mom"][...], dtype=np.int64)
        non_harmonic = []
        ao = 0
        while ao < shell_of_ao.size:
            l = int(shell_l[shell_of_ao[ao]])
            orders = [(a, b, l - a - b) for a in range(l, -1, -1) for b in range(l - a, -1, -1)]
            assert np.all(shell_of_ao[ao : ao + len(orders)] == shell_of_ao[ao])
            non_harmonic.extend(ao + k for k, order in enumerate(orders) if max(order) >= 2)
            ao += len(orders)
        assert non_harmonic, "the control must contain d or higher Cartesian shells"
        for name in ("mo/mo_coefficient", "mo/mo_coefficient_im"):
            if name in h5:
                coefficients = h5[name][...]
                coefficients[:, non_harmonic] = 0.0
                h5[name][...] = coefficients


_SPHE_PROJECTION_WARNING = re.compile(
    r"--ao-conv-to sphe discards the part of the Cartesian wavefunction outside the spherical-harmonic subspace: "
    r"spin-(up|dn) occupied MOs, maximum relative S-norm residual (\S+) \(MO (\d+);"
)


def _independent_s_metric_projection(source_mos, sphe_aos):
    """Project Cartesian MO rows onto the span of ``sphe_aos`` in the analytic AO overlap metric.

    The spherical -> native-Cartesian embedding ``T`` is assembled by the
    independent shell-matching oracle of the MATH-MO-BASIS-PROJECTION-001
    tests (exponent/contraction matching plus the analytic solid-harmonic
    table), not by the production conversion; ``S`` is the complete molecular
    or periodic Bloch overlap of the source Cartesian AOs. Returns the
    spherical row coefficients ``d`` with ``d^T = (T* S T^T)^-1 T* S c^T``
    (rows in the order of ``sphe_aos``), the same function as Cartesian rows
    ``d T``, and the per-MO relative S-norm of the discarded part ``c - d T``.
    Host-side fp64/complex128 in both modes.
    """
    from jqmc.atomic_orbital import AOs_pGTO_base_data, _detile_ragged_to_molecular, compute_overlap_matrix
    from tests.test_jqmc_kernel.test_molecular_orbital import _independent_native_cart_embedding

    cart_aos = source_mos.aos_data
    coefficients = np.asarray(source_mos.mo_coefficients)
    overlap = np.asarray(compute_overlap_matrix(cart_aos))
    if isinstance(cart_aos, AOs_pGTO_base_data):
        embedding = _independent_native_cart_embedding(
            _detile_ragged_to_molecular(cart_aos), _detile_ragged_to_molecular(sphe_aos)
        )
    else:
        embedding = _independent_native_cart_embedding(cart_aos, sphe_aos)
    embedding = np.asarray(embedding, dtype=np.result_type(overlap.dtype, np.float64))
    metric_rhs = embedding.conj() @ overlap
    spherical = np.linalg.solve(metric_rhs @ embedding.T, metric_rhs @ coefficients.T).T
    projected = spherical @ embedding
    discarded = coefficients - projected

    def s_norm_sq(rows):
        return np.einsum("ij,jk,ik->i", rows.conj(), overlap, rows).real

    return spherical, projected, np.sqrt(s_norm_sq(discarded) / s_norm_sq(coefficients))


def _random_configurations(structure, geminal, rng, count=3):
    """Electron configurations scattered around the nuclei (fixed seed)."""
    positions = np.asarray(structure.positions, dtype=np.float64)
    for _ in range(count):
        r_up = positions[rng.integers(len(positions), size=geminal.num_electron_up)] + rng.normal(
            scale=0.7, size=(geminal.num_electron_up, 3)
        )
        r_dn = positions[rng.integers(len(positions), size=geminal.num_electron_dn)] + rng.normal(
            scale=0.7, size=(geminal.num_electron_dn, 3)
        )
        yield r_up, r_dn


#: Mixed-precision upper bound on cond_2(G) of the draws at which the lossy projection's ln|Psi| is
#: compared (issue #716), the same cap as ``test_periodic_invariants._COND_MAX``. Mixed precision
#: evaluates real GTO/pGTO values in the float32 ``ao_eval`` zone, and ln|det| amplifies that
#: rounding by about cond_2(G): at the second Li_pbc draw of seed 591 (cond_2(G) = 3.9e4, a
#: same-spin near-node configuration) the written spherical and the Cartesian reference ln|Psi|
#: deviate from their own float64 values by 6.9e-3 and -1.6e-3, which no float32 AO evaluation can
#: bring inside the unchanged strict budget. Full precision keeps every draw (no cap).
_SPHE_PROJECTION_COND_G_MAX = 1.0e4
#: Proposals drawn from the one seeded stream before the screen fails loudly.
_SPHE_PROJECTION_MAX_DRAWS = 64


def _well_conditioned_configurations(structure, geminal, rng, cond_G_max, count=3):
    """The first ``count`` draws of :func:`_random_configurations` with cond_2(G) below ``cond_G_max``.

    Proposals continue the single declared rng stream; a rejected draw is
    never counted, and exhausting the proposal limit fails. ``G`` is the
    production geminal matrix of ``geminal`` at each proposal, and its
    spectral condition number is evaluated in float64 (complex128 for a
    complex G) before any comparison.
    Returns ``[(r_up, r_dn, cond_G), ...]``.
    """
    from jqmc.determinant import compute_geminal_all_elements

    accepted = []
    proposals = _random_configurations(structure, geminal, rng, count=_SPHE_PROJECTION_MAX_DRAWS)
    for r_up, r_dn in proposals:
        g = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
        g = g.astype(np.result_type(g.dtype, np.float64))
        cond_G = float(np.linalg.cond(g, p=2))
        if np.isfinite(cond_G) and cond_G < cond_G_max:
            accepted.append((r_up, r_dn, cond_G))
            if len(accepted) == count:
                return accepted
    raise AssertionError(f"only {len(accepted)} of {count} draws have cond_2(G) < {cond_G_max:g}")


def _assert_written_ln_psi(written_wavefunction, structure, geminal, coefficients_up, coefficients_dn, rng):
    """ln|Psi| of the written file equals a NumPy Slater determinant of the given Cartesian MO rows.

    No Jastrow is written, so ``Psi = det(up) det(dn)``; the MO values are the
    rows times the NumPy-oracle values of the source Cartesian AOs.
    """
    from jqmc._precision import get_tolerance_min
    from jqmc.wavefunction import evaluate_ln_wavefunction

    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval", "wf_eval"), "strict")
    for r_up, r_dn in _random_configurations(structure, geminal, rng):
        expected = sum(
            np.linalg.slogdet(rows @ np.asarray(orbitals.aos_data._compute_AOs_debug(r)))[1]
            for rows, orbitals, r in (
                (coefficients_up, geminal.orb_data_up_spin, r_up),
                (coefficients_dn, geminal.orb_data_dn_spin, r_dn),
            )
        )
        actual = float(evaluate_ln_wavefunction(written_wavefunction, r_up, r_dn))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


class TestTrexioConvertToSphericalProjection:
    """``trexio convert-to --ao-conv-to sphe`` writes the S-metric projection and reports any discarded part (#591)."""

    @pytest.mark.light
    @pytest.mark.parametrize(
        "trexio_filename",
        ["N2_ecp_ccpvtz_cart.h5", "Li_pbc_ae_ccpvdz_cart.h5", "lithium_k025_cart.h5"],
    )
    def test_trexio_convert_to_sphe_warns_and_writes_s_metric_projection(self, tmp_path, caplog, trexio_filename):
        """A lossy Cartesian-to-spherical conversion completes, warns, and writes the projected wavefunction.

        Native Cartesian SCF MOs carry s/p-like trace components of their d/f
        shells that the spherical basis cannot represent. The command exits 0
        and writes the file, and logs one WARNING per spin whose maximum
        relative S-norm residual and MO index equal an independent S-metric
        projection (shell-matched analytic embedding, analytic overlap). The
        written spherical MO coefficients equal that projection, and the
        written ln|Psi| equals the same projected function expanded in the
        source Cartesian AOs (evaluated through the Cartesian AO path, not the
        spherical one under test). ln|Psi| is compared at the first three
        seed-591 draws; in mixed precision only draws whose production
        geminal matrix has cond_2(G) < 1e4 count, because a
        near-node draw amplifies the float32 ``ao_eval`` rounding beyond the
        unchanged strict budget.
        """
        import logging

        from typer.testing import CliRunner

        from jqmc._precision import get_tolerance, get_tolerance_min, is_mixed_precision_enabled
        from jqmc._setting import AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.jqmc_tool import trexio_app
        from jqmc.molecular_orbital import MOs_data
        from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction

        trexio_file = os.path.join(_TREXIO_DIR, trexio_filename)
        output = tmp_path / "hamiltonian_sphe.h5"
        with caplog.at_level(logging.WARNING, logger="jqmc"):
            result = CliRunner().invoke(trexio_app, ["convert-to", trexio_file, "-o", str(output), "--ao-conv-to", "sphe"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert output.is_file()
        reported = {}
        for record in caplog.records:
            match = _SPHE_PROJECTION_WARNING.search(record.getMessage())
            if record.levelno == logging.WARNING and match:
                reported[match.group(1)] = (match.group(2), int(match.group(3)))

        written = Hamiltonian_data.load_from_hdf5(str(output)).wavefunction_data
        structure, _, _, _, geminal, _ = read_trexio_file(trexio_file, store_tuple=True)
        projected = {}
        expected = {}
        atol, rtol = get_tolerance("det_eval", "strict")
        for spin in ("up", "dn"):
            source = getattr(geminal, f"orb_data_{spin}_spin")
            written_mos = getattr(written.geminal_data, f"orb_data_{spin}_spin")
            spherical, cartesian_rows, residual = _independent_s_metric_projection(source, written_mos.aos_data)
            np.testing.assert_allclose(np.asarray(written_mos.mo_coefficients), spherical, atol=atol, rtol=rtol)
            projected[spin] = MOs_data(num_mo=source.num_mo, aos_data=source.aos_data, mo_coefficients=cartesian_rows)
            worst = int(np.argmax(residual))
            if residual[worst] > AO_SPHERICAL_PROJECTION_RESIDUAL_THRESHOLD:
                expected[spin] = (f"{residual[worst]:.3e}", worst)
        assert expected, "the lossy fixture must discard a non-negligible part"
        assert reported == expected

        reference = Wavefunction_data(
            jastrow_data=Jastrow_data(),
            geminal_data=geminal.replace(orb_data_up_spin=projected["up"], orb_data_dn_spin=projected["dn"]),
        )
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval", "wf_eval"), "strict")
        cond_G_max = _SPHE_PROJECTION_COND_G_MAX if is_mixed_precision_enabled() else np.inf
        draws = _well_conditioned_configurations(structure, reference.geminal_data, np.random.default_rng(591), cond_G_max)
        for r_up, r_dn, cond_G in draws:
            assert cond_G < cond_G_max
            np.testing.assert_allclose(
                float(evaluate_ln_wavefunction(written, r_up, r_dn)),
                float(evaluate_ln_wavefunction(reference, r_up, r_dn)),
                atol=atol,
                rtol=rtol,
            )

    @pytest.mark.light
    @pytest.mark.parametrize("source", ["H2_ae_ccpvdz_cart.h5", "N2_ecp_ccpvtz_cart.h5-harmonic"])
    def test_trexio_convert_to_sphe_lossless_source_keeps_ln_psi(self, tmp_path, caplog, source):
        """A Cartesian source inside the spherical subspace converts silently to the same wavefunction.

        Controls: H2 cc-pVDZ (s/p shells only) and a copy of N2 cc-pVTZ (d/f
        shells) whose MOs keep only harmonic-monomial coefficients. No
        projection warning is logged, and the written spherical Hamiltonian's
        ln|Psi| at fixed configurations equals an independent NumPy
        Slater-determinant evaluation of the Cartesian source MOs read from the
        same TREXIO file (no Jastrow, so Psi = det up * det dn).
        """
        import logging

        from typer.testing import CliRunner

        from jqmc.atomic_orbital import AOs_sphe_gto_data
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_tool import trexio_app

        if source.endswith("-harmonic"):
            trexio_file = tmp_path / "harmonic_source.h5"
            _write_harmonic_only_trexio_copy(os.path.join(_TREXIO_DIR, source.removesuffix("-harmonic")), trexio_file)
        else:
            trexio_file = Path(_TREXIO_DIR) / source
        output = tmp_path / "hamiltonian_sphe.h5"
        with caplog.at_level(logging.WARNING, logger="jqmc"):
            result = CliRunner().invoke(trexio_app, ["convert-to", str(trexio_file), "-o", str(output), "--ao-conv-to", "sphe"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert not any(_SPHE_PROJECTION_WARNING.search(record.getMessage()) for record in caplog.records)
        converted = Hamiltonian_data.load_from_hdf5(str(output))
        assert isinstance(converted.wavefunction_data.geminal_data.orb_data_up_spin.aos_data, AOs_sphe_gto_data)

        structure, _, _, _, geminal, _ = read_trexio_file(str(trexio_file), store_tuple=True)
        _assert_written_ln_psi(
            converted.wavefunction_data,
            structure,
            geminal,
            np.asarray(geminal.orb_data_up_spin.mo_coefficients),
            np.asarray(geminal.orb_data_dn_spin.mo_coefficients),
            np.random.default_rng(591),
        )


class TestGenerateInput:
    """Tests for mcmc / lrdmc / vmc generate-input commands."""

    @pytest.mark.light
    def test_mcmc_generate_input_creates_file(self, tmp_path):
        """MCMC generate input creates file."""
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        doc = tomlkit.loads(Path(outfile).read_text())
        assert "control" in doc
        assert "mcmc" in doc
        assert doc["control"]["job_type"] == "mcmc"

    @pytest.mark.light
    def test_lrdmc_generate_input_creates_file(self, tmp_path):
        """Lrdmc generate input creates file."""
        outfile = str(tmp_path / "lrdmc.toml")
        lrdmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        doc = tomlkit.loads(Path(outfile).read_text())
        assert "control" in doc
        assert "lrdmc-bra" in doc
        assert doc["control"]["job_type"] == "lrdmc-bra"

    @pytest.mark.light
    def test_vmc_generate_input_creates_file(self, tmp_path):
        """Vmc generate input creates file."""
        outfile = str(tmp_path / "vmc.toml")
        vmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        doc = tomlkit.loads(Path(outfile).read_text())
        assert "control" in doc
        assert "vmc" in doc
        assert doc["control"]["job_type"] == "vmc"

    @pytest.mark.light
    def test_generate_input_without_flag_does_not_create(self, tmp_path):
        """Generate input without flag does not create."""
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=False, filename=outfile, exclude_comment=False)
        assert not os.path.exists(outfile)

    @pytest.mark.light
    def test_generate_input_without_comment(self, tmp_path):
        """Generate input without comment."""
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=True, filename=outfile, exclude_comment=True)
        text = Path(outfile).read_text()
        # When comments are excluded, there should be no "# " comment lines
        # (tomlkit comments start with #)
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("#"), f"Unexpected comment line when exclude_comment=True: {line}"

    @pytest.mark.light
    def test_generate_input_with_comment(self, tmp_path):
        """Generate input with comment."""
        outfile = str(tmp_path / "mcmc.toml")
        mcmc_generate_input(flag=True, filename=outfile, exclude_comment=False)
        text = Path(outfile).read_text()
        assert "#" in text, "Expected comments in the generated TOML file"


def _reference_mcmc_jackknife(rank_data, num_warmup, num_bin_blocks):
    """Reference jackknife using np.sum (matches runtime get_E)."""
    w_all = []
    we_all = []
    for obs in rank_data:
        e_L = obs["e_L"][num_warmup:]
        w_L = obs["w_L"][num_warmup:]
        for arr in np.array_split(w_L, num_bin_blocks, axis=0):
            w_all.extend(np.ravel(np.sum(arr, axis=0)).tolist())
        for arr in np.array_split(w_L * e_L, num_bin_blocks, axis=0):
            we_all.extend(np.ravel(np.sum(arr, axis=0)).tolist())
    w = np.array(w_all)
    we = np.array(we_all)
    M = w.size
    S_w, S_we = np.sum(w), np.sum(we)
    E_jk = np.array([(S_we - we[m]) / (S_w - w[m]) for m in range(M)])
    E_mean = np.mean(E_jk)
    E_std = np.sqrt(M - 1) * np.std(E_jk)
    return E_mean, E_std


def _reference_lrdmc_jackknife(rank_data, num_warmup, num_bin_blocks, k):
    """Jackknife reference using the separately checked XLA historical weights.

    The tool uses the NumPy checkpoint accumulator. Both accumulators have
    hand-evaluated exact fixtures in test_checkpoint_weights; do not copy the
    checkpoint implementation into this oracle.
    """
    import jax.numpy as jnp

    from jqmc.jqmc_gfmc import _compute_G_L

    w_all = []
    we_all = []
    for obs in rank_data:
        raw_e_L = obs["e_L"]
        raw_w_L = obs["w_L"]
        e_L = raw_e_L[k:][num_warmup:]
        w_L = np.asarray(_compute_G_L(jnp.asarray(raw_w_L), k))[num_warmup:]
        for arr in np.array_split(w_L, num_bin_blocks, axis=0):
            w_all.extend(np.ravel(np.sum(arr, axis=0)).tolist())
        for arr in np.array_split(w_L * e_L, num_bin_blocks, axis=0):
            we_all.extend(np.ravel(np.sum(arr, axis=0)).tolist())
    w = np.array(w_all)
    we = np.array(we_all)
    M = w.size
    S_w, S_we = np.sum(w), np.sum(we)
    E_jk = np.array([(S_we - we[m]) / (S_w - w[m]) for m in range(M)])
    E_mean = np.mean(E_jk)
    E_std = np.sqrt(M - 1) * np.std(E_jk)
    return E_mean, E_std


class TestComputeEnergy:
    """Tests for mcmc and lrdmc compute-energy commands."""

    @pytest.mark.light
    @pytest.mark.parametrize("num_ranks", [1, 2])
    def test_mcmc_twist_checkpoint_directory_uses_stored_weights(self, tmp_path, num_ranks):
        """A sharded twist checkpoint is reduced from per-twist estimators.

        Each twist contains either one non-MPI rank or two MPI ranks while the
        tool itself is a single process. Unequal weights make this a check against both an
        ``IsADirectoryError`` and the scientifically-wrong shortcut
        of pooling all twist samples (or assigning equal BZ weights).
        """
        checkpoint_dir = tmp_path / "restart.h5"
        checkpoint_dir.mkdir()
        twist_weights = np.array([0.25, 0.75])
        twist_data = []
        for twist_idx, (weight, offset) in enumerate(zip(twist_weights, (-3.0, -7.0), strict=True)):
            objs = []
            rank_data = []
            for rank in range(num_ranks):
                rng = np.random.default_rng(910 + 10 * twist_idx + rank)
                obj = _make_mcmc_obj(num_steps=80, num_walkers=3, with_force=False)
                obj.e_L = offset + 0.2 * rng.standard_normal((80, 3))
                obj.w_L = rng.uniform(0.8, 1.2, (80, 3))
                objs.append(obj)
                rank_data.append({"e_L": obj.e_L, "w_L": obj.w_L})
            path = checkpoint_dir / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), objs)
            with h5py.File(path, "a") as f:
                f["_meta"].attrs["twist_index"] = twist_idx
                f["_meta"].attrs["normalized_twist_weight"] = weight
                f["_meta"].attrs["num_twists"] = len(twist_weights)
                f["_meta"].attrs["converged"] = True
            twist_data.append(rank_data)

        estimates = [_reference_mcmc_jackknife(data, 0, 4) for data in twist_data]
        expected_mean = float(np.dot(twist_weights, [estimate[0] for estimate in estimates]))
        expected_std = float(np.sqrt(np.dot(twist_weights**2, np.square([estimate[1] for estimate in estimates]))))

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        result = CliRunner().invoke(mcmc_app, ["compute-energy", str(checkpoint_dir), "-b", "4", "-w", "0"])
        assert result.exit_code == 0, result.output
        match = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert match is not None
        actual_mean, actual_std = map(float, match.groups())
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(actual_mean, expected_mean, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_std, expected_std, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["-b", "4", "-w", "40"], "fewer than num_mcmc_bin_blocks"),
            (["-b", "50", "-w", "0"], "fewer than num_mcmc_bin_blocks"),
            (["-b", "0", "-w", "0"], "must be >= 1"),
            (["-b", "4", "-w", "-1"], "must be >= 0"),
        ],
        ids=["warmup-eats-all", "too-many-bins", "zero-bins", "negative-warmup"],
    )
    def test_mcmc_compute_energy_rejects_vacuous_binning(self, tmp_path, args, message):
        """Settings that leave empty bins are usage errors, not a printed energy."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        chk = tmp_path / "mcmc.h5"
        _write_chk(str(chk), [_make_mcmc_obj(num_steps=40, num_walkers=2, e_L_value=-1.0, with_force=False)])
        result = CliRunner().invoke(mcmc_app, ["compute-energy", str(chk), *args])
        assert result.exit_code != 0
        assert "E = " not in result.output
        assert message in _flatten_cli_output(result.output)

    @pytest.mark.light
    def test_mcmc_compute_energy_rejects_zero_weights(self, tmp_path):
        """An all-zero-weight run cannot form delete-one-bin ratios."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        obj = _make_mcmc_obj(num_steps=40, num_walkers=2, e_L_value=-1.0, w_L_value=0.0, with_force=False)
        chk = tmp_path / "mcmc.h5"
        _write_chk(str(chk), [obj])
        result = CliRunner().invoke(mcmc_app, ["compute-energy", str(chk), "-b", "4", "-w", "0"])
        assert result.exit_code != 0
        assert "delete-one-bin weight sum" in _flatten_cli_output(result.output)

    @pytest.mark.light
    def test_mcmc_pre_upgrade_twist_checkpoint_is_rejected(self, tmp_path):
        """A twist directory without embedded BZ metadata is rejected."""
        checkpoint_dir = tmp_path / "restart.h5"
        checkpoint_dir.mkdir()
        for twist_idx, offset in enumerate((-2.0, -6.0)):
            obj = _make_mcmc_obj(num_steps=40, num_walkers=2, e_L_value=offset, with_force=False)
            path = checkpoint_dir / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), [obj])
            with h5py.File(path, "a") as f:
                f["_meta"].attrs["twist_index"] = twist_idx
                f["_meta"].attrs["converged"] = True

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        result = CliRunner().invoke(mcmc_app, ["compute-energy", str(checkpoint_dir), "-b", "2", "-w", "0"])
        assert result.exit_code != 0
        assert "pre-upgrade checkpoint directories are unsupported" in str(result.exception)

    @pytest.mark.light
    @pytest.mark.parametrize("num_ranks", [1, 2])
    def test_lrdmc_twist_checkpoint_directory_uses_stored_weights(self, tmp_path, num_ranks):
        """LRDMC sharded post-processing covers non-MPI and MPI checkpoints."""
        checkpoint_dir = tmp_path / "restart.h5"
        checkpoint_dir.mkdir()
        twist_weights = np.array([0.4, 0.6])
        estimates = []
        for twist_idx, (weight, offset) in enumerate(zip(twist_weights, (-4.0, -8.0), strict=True)):
            objs = []
            rank_data = []
            for rank in range(num_ranks):
                rng = np.random.default_rng(930 + 10 * twist_idx + rank)
                obj = _make_lrdmc_obj(num_steps=80, num_walkers=2, with_force=False)
                obj.e_L = offset + 0.1 * rng.standard_normal((80, 2))
                obj.w_L = rng.uniform(0.9, 1.1, (80, 2))
                objs.append(obj)
                rank_data.append({"e_L": obj.e_L, "w_L": obj.w_L})
            path = checkpoint_dir / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), objs)
            with h5py.File(path, "a") as f:
                f["_meta"].attrs["twist_index"] = twist_idx
                f["_meta"].attrs["normalized_twist_weight"] = weight
                f["_meta"].attrs["num_twists"] = len(twist_weights)
                f["_meta"].attrs["converged"] = True
            estimates.append(_reference_lrdmc_jackknife(rank_data, 30, 10, 5))

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        result = CliRunner().invoke(
            lrdmc_app,
            ["compute-energy", str(checkpoint_dir), "-b", "10", "-w", "30", "-c", "5"],
        )
        assert result.exit_code == 0, result.output
        match = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert match is not None
        expected_mean = float(np.dot(twist_weights, [estimate[0] for estimate in estimates]))
        expected_std = float(np.sqrt(np.dot(twist_weights**2, np.square([estimate[1] for estimate in estimates]))))
        actual_mean, actual_std = map(float, match.groups())
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(actual_mean, expected_mean, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_std, expected_std, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_mcmc_random_energy_jackknife(self, tmp_path):
        """jqmc-tool must match the np.sum-based reference jackknife for random data."""
        rng = np.random.default_rng(42)
        num_steps, num_walkers, num_bins, warmup = 103, 4, 7, 0
        chk_path = str(tmp_path / "mcmc.h5")
        obj = _make_mcmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False)
        obj.e_L = rng.standard_normal((num_steps, num_walkers)) - 5.0
        obj.w_L = rng.uniform(0.8, 1.2, (num_steps, num_walkers))
        _write_chk(chk_path, [obj])

        E_ref, std_ref = _reference_mcmc_jackknife(
            [{"e_L": obj.e_L, "w_L": obj.w_L}],
            warmup,
            num_bins,
        )

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk_path, "-b", str(num_bins), "-w", str(warmup)])
        assert result.exit_code == 0
        m = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert m is not None
        E_cli, std_cli = float(m.group(1)), float(m.group(2))
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(E_cli, E_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(std_cli, std_ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_mcmc_multi_rank_random(self, tmp_path):
        """Multiple MPI ranks with random data should match np.sum reference."""
        num_steps, num_walkers, num_bins, warmup = 97, 3, 5, 0
        chk_path = str(tmp_path / "mcmc.h5")
        objs = []
        rank_data = []
        for i in range(3):
            rng = np.random.default_rng(i + 100)
            obj = _make_mcmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False)
            obj.e_L = rng.standard_normal((num_steps, num_walkers)) - 3.0
            obj.w_L = rng.uniform(0.9, 1.1, (num_steps, num_walkers))
            objs.append(obj)
            rank_data.append({"e_L": obj.e_L, "w_L": obj.w_L})
        _write_chk(chk_path, objs)

        E_ref, std_ref = _reference_mcmc_jackknife(rank_data, warmup, num_bins)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk_path, "-b", str(num_bins), "-w", str(warmup)])
        assert result.exit_code == 0
        assert "Found 3 MPI ranks" in result.output
        m = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert m is not None
        E_cli, std_cli = float(m.group(1)), float(m.group(2))
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(E_cli, E_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(std_cli, std_ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_mcmc_warmup_discards_steps(self, tmp_path):
        """Warmup discard + random data post-warmup must match reference."""
        rng = np.random.default_rng(99)
        num_steps, num_walkers, num_bins, warmup = 100, 2, 3, 10
        chk_path = str(tmp_path / "mcmc.h5")
        obj = _make_mcmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False)
        obj.e_L = rng.standard_normal((num_steps, num_walkers)) - 5.0
        obj.w_L = rng.uniform(0.8, 1.2, (num_steps, num_walkers))
        # Poison first warmup steps to make warmup discard meaningful
        obj.e_L[:warmup, :] = 100.0
        _write_chk(chk_path, [obj])

        E_ref, std_ref = _reference_mcmc_jackknife(
            [{"e_L": obj.e_L, "w_L": obj.w_L}],
            warmup,
            num_bins,
        )

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk_path, "-b", str(num_bins), "-w", str(warmup)])
        assert result.exit_code == 0
        m = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert m is not None
        E_cli, std_cli = float(m.group(1)), float(m.group(2))
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(E_cli, E_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(std_cli, std_ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_lrdmc_random_energy_jackknife(self, tmp_path):
        """LRDMC jqmc-tool must match the np.sum-based reference jackknife for random data."""
        rng = np.random.default_rng(77)
        num_steps, num_walkers, num_bins, warmup, k = 203, 4, 10, 30, 5
        chk_path = str(tmp_path / "lrdmc.h5")
        obj = _make_lrdmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False)
        obj.e_L = rng.standard_normal((num_steps, num_walkers)) - 7.0
        obj.w_L = rng.uniform(0.9, 1.1, (num_steps, num_walkers))
        _write_chk(chk_path, [obj])

        E_ref, std_ref = _reference_lrdmc_jackknife(
            [{"e_L": obj.e_L, "w_L": obj.w_L}],
            warmup,
            num_bins,
            k,
        )

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-energy", chk_path, "-b", str(num_bins), "-w", str(warmup), "-c", str(k)])
        assert result.exit_code == 0
        m = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert m is not None
        E_cli, std_cli = float(m.group(1)), float(m.group(2))
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(E_cli, E_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(std_cli, std_ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_lrdmc_multi_rank_random(self, tmp_path):
        """Multiple LRDMC ranks with random data must match np.sum reference."""
        num_steps, num_walkers, num_bins, warmup, k = 203, 2, 10, 30, 5
        chk_path = str(tmp_path / "lrdmc.h5")
        objs = []
        rank_data = []
        for i in range(2):
            rng = np.random.default_rng(i + 200)
            obj = _make_lrdmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False)
            obj.e_L = rng.standard_normal((num_steps, num_walkers)) - 2.0
            obj.w_L = rng.uniform(0.9, 1.1, (num_steps, num_walkers))
            objs.append(obj)
            rank_data.append({"e_L": obj.e_L, "w_L": obj.w_L})
        _write_chk(chk_path, objs)

        E_ref, std_ref = _reference_lrdmc_jackknife(rank_data, warmup, num_bins, k)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-energy", chk_path, "-b", str(num_bins), "-w", str(warmup), "-c", str(k)])
        assert result.exit_code == 0
        assert "Found 2 MPI ranks" in result.output
        m = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert m is not None
        E_cli, std_cli = float(m.group(1)), float(m.group(2))
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(E_cli, E_ref, atol=atol, rtol=rtol)
        np.testing.assert_allclose(std_cli, std_ref, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_lrdmc_extrapolate_energy(self, tmp_path):
        """extrapolate-energy with two LRDMC checkpoints should report a->0 result."""
        rng1, rng2 = np.random.default_rng(300), np.random.default_rng(301)
        chk1 = str(tmp_path / "lrdmc1.h5")
        chk2 = str(tmp_path / "lrdmc2.h5")

        obj1 = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=0.5, with_force=False)
        obj1.e_L = rng1.standard_normal((100, 4)) - 5.0
        obj1.w_L = rng1.uniform(0.9, 1.1, (100, 4))
        obj2 = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=0.3, with_force=False)
        obj2.e_L = rng2.standard_normal((100, 4)) - 5.2
        obj2.w_L = rng2.uniform(0.9, 1.1, (100, 4))
        _write_chk(chk1, [obj1])
        _write_chk(chk2, [obj2])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["extrapolate-energy", chk1, chk2, "-b", "10", "-w", "25", "-c", "5", "-p", "1"])
        assert result.exit_code == 0
        assert "Extrapolation" in result.output
        assert "a -> 0" in result.output or "a = 0" in result.output or "For a ->" in result.output

    @pytest.mark.light
    def test_lrdmc_extrapolate_refuses_underdetermined_order(self, tmp_path):
        """Regression: order 2 with two lattice spacings must REFUSE.

        numpy.polyfit on an underdetermined system returns a minimum-norm
        solution whose a->0 intercept is garbage (demonstrated -4.85 vs the
        correct -5.31 Ha) behind nothing but a stderr RankWarning.
        """
        rng1, rng2 = np.random.default_rng(300), np.random.default_rng(301)
        chk1, chk2 = str(tmp_path / "l1.h5"), str(tmp_path / "l2.h5")
        obj1 = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=0.5, with_force=False)
        obj1.e_L = rng1.standard_normal((100, 4)) - 5.0
        obj1.w_L = rng1.uniform(0.9, 1.1, (100, 4))
        obj2 = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=0.3, with_force=False)
        obj2.e_L = rng2.standard_normal((100, 4)) - 5.2
        obj2.w_L = rng2.uniform(0.9, 1.1, (100, 4))
        _write_chk(chk1, [obj1])
        _write_chk(chk2, [obj2])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        # default polynomial order is 2 -> underdetermined for 2 spacings
        result = runner.invoke(lrdmc_app, ["extrapolate-energy", chk1, chk2, "-b", "10", "-w", "25", "-c", "5"])
        assert result.exit_code != 0
        blob = result.output + str(result.exception or "")
        assert "polynomial order" in blob and "lattice" in blob

    @pytest.mark.light
    def test_lrdmc_extrapolate_recovers_linear_intercept(self, tmp_path):
        """Numeric gate: -p 1 on exactly linear E(a^2) recovers the intercept.

        Two nearly noise-free checkpoints on E = E0 + c * a^2; the a->0 value
        printed by the CLI must be E0 to well inside the Monte-Carlo scatter.
        """
        E0, c = -5.3125, 1.0
        chk1, chk2 = str(tmp_path / "l1.h5"), str(tmp_path / "l2.h5")
        for chk, alat in ((chk1, 0.5), (chk2, 0.3)):
            obj = _make_lrdmc_obj(num_steps=400, num_walkers=4, alat=alat, with_force=False)
            rng = np.random.default_rng(int(alat * 1000))
            obj.e_L = (E0 + c * alat**2) + 1.0e-6 * rng.standard_normal((400, 4))
            obj.w_L = np.ones((400, 4))
            _write_chk(chk, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["extrapolate-energy", chk1, chk2, "-b", "10", "-w", "25", "-c", "5", "-p", "1"])
        assert result.exit_code == 0
        import re

        match = re.search(r"For a -> 0 bohr: E = (-?\d+\.\d+)", result.output)
        assert match, result.output
        assert abs(float(match.group(1)) - E0) < ATOL_TOOL_EXTRAPOLATED_ENERGY_DISPLAY, result.output

    @pytest.mark.light
    def test_lrdmc_extrapolate_energy_resampling_is_deterministic(self, tmp_path):
        """The a->0 resampling must not read the global numpy RNG.

        Two CLI invocations on byte-identical input checkpoints, with different
        global numpy seeds set between them, must print an identical a->0
        result. Monte Carlo resampling noise drawn from the unseeded global
        ``np.random`` state would make the printed E and its error bar differ
        from run to run (and depend on whatever global seed happens to be set,
        as forced here).
        """
        rng1, rng2 = np.random.default_rng(300), np.random.default_rng(301)
        chk1 = str(tmp_path / "lrdmc1.h5")
        chk2 = str(tmp_path / "lrdmc2.h5")

        obj1 = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=0.5, with_force=False)
        obj1.e_L = rng1.standard_normal((100, 4)) - 5.0
        obj1.w_L = rng1.uniform(0.9, 1.1, (100, 4))
        obj2 = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=0.3, with_force=False)
        obj2.e_L = rng2.standard_normal((100, 4)) - 5.2
        obj2.w_L = rng2.uniform(0.9, 1.1, (100, 4))
        _write_chk(chk1, [obj1])
        _write_chk(chk2, [obj2])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        args = ["extrapolate-energy", chk1, chk2, "-b", "10", "-w", "25", "-c", "5", "-p", "1"]

        # Deliberately different global-RNG states around each invocation: a
        # fix that still reads np.random (module-level) would pick up the
        # global state and diverge; a local, fixed-seed generator will not.
        np.random.seed(111)
        result_a = runner.invoke(lrdmc_app, args)
        assert result_a.exit_code == 0, result_a.output

        np.random.seed(222)
        result_b = runner.invoke(lrdmc_app, args)
        assert result_b.exit_code == 0, result_b.output

        pattern = r"For a -> 0 bohr: E = ([^\n]+) Ha\."
        match_a = re.search(pattern, result_a.output)
        match_b = re.search(pattern, result_b.output)
        assert match_a and match_b, (result_a.output, result_b.output)
        assert match_a.group(1) == match_b.group(1), (result_a.output, result_b.output)


class TestEstimatorAgainstAnalyticStatistics:
    """Pin estimator arithmetic to finite-sample identities and known values.

    The reference identities do not reuse the production leave-one-out formula.
    A fixed seed defines inputs, never a confidence band for accepting outputs.
    """

    @staticmethod
    def _run_energy(tmp_path, e_L, w_L, bins):
        chk = str(tmp_path / "mcmc.h5")
        obj = _make_mcmc_obj(num_steps=e_L.shape[0], num_walkers=e_L.shape[1], with_force=False)
        obj.e_L, obj.w_L = e_L, w_L
        _write_chk(chk, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        result = CliRunner().invoke(mcmc_app, ["compute-energy", chk, "-b", str(bins), "-w", "0"])
        assert result.exit_code == 0, result.output
        import re

        match = re.search(r"E = (-?[\d.]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match, result.output
        return float(match.group(1)), float(match.group(2))

    @pytest.mark.light
    def test_unweighted_iid_error_matches_sigma_over_sqrt_nbin(self, tmp_path):
        """With unit weights and equal bins the jackknife error is EXACTLY the
        standard error of the bin means, sd_sample / sqrt(N_bin).

        Proof: E_jk[m] = (S - y[m])/(N-1) over bin means y, so
        sqrt(N-1)*sd_pop(E_jk) = sd_pop(y)/sqrt(N-1) = sd_sample(y)/sqrt(N).
        Asserting the exact identity (not a percent band) is what makes this bite
        on a mis-normalisation such as sqrt(M) for sqrt(M-1), a 1.3% effect here.
        """
        rng = np.random.default_rng(2026)
        steps, walkers, bins = 200, 4, 10
        e_L = -1.0 + 0.25 * rng.standard_normal((steps, walkers))
        w_L = np.ones((steps, walkers))

        mean, std = self._run_energy(tmp_path, e_L, w_L, bins)

        bin_means = np.ravel([np.mean(a, axis=0) for a in np.array_split(e_L, bins, axis=0)])
        analytic_std = float(np.std(bin_means, ddof=1) / np.sqrt(bin_means.size))

        np.testing.assert_allclose(mean, float(np.mean(e_L)), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(std, analytic_std, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_AGGREGATOR_LOOSE)

    @pytest.mark.light
    @pytest.mark.parametrize(
        "profile,seed,short_steps,long_steps,walkers,bins",
        [
            ("independent-fixed-streams", 99, 100, 400, 4, 10),
            ("repeated-fixed-sample", 99, 100, 400, 4, 10),
            ("scaled-fixed-sample", 99, 100, 400, 4, 10),
        ],
    )
    def test_error_scales_as_inverse_sqrt_of_sample_count(
        self, tmp_path, profile, seed, short_steps, long_steps, walkers, bins
    ):
        """Each fixed sample's reported mean and SE equal their independent finite-data reference.

        Keep the original PCG64 seed-99, 100x4/400x4, ten-bin independent
        streams. Add fourfold row repetition (unchanged bin means/SE) and
        repetition with half-sized deviations (half-sized SE). A pairwise
        bin-mean-difference formula, not the production jackknife or np.std,
        independently evaluates each input. Repetition is not extra independent
        information: the node name does not assert a universal
        factor of two for four times as many rows.
        The realized arrays must have the declared shapes/dtype; the separate
        stream must remain distinguishable from both repeated-input fixtures.

        Synthetic unit-weight fp64 arrays are written to temporary HDF5
        checkpoints and passed through the real compute-energy CLI. No TREXIO,
        physical wavefunction, electron positions or cond(G) is involved.
        Full/mixed each check the same pure-fp64 postprocessor with the unchanged
        central Python-fp64 numerical pair. No sigma or empirical ratio band
        can substitute for the two pointwise output comparisons.
        """
        rng = np.random.default_rng(seed)
        short = -1.0 + 0.3 * rng.standard_normal((short_steps, walkers))
        if profile == "independent-fixed-streams":
            long = -1.0 + 0.3 * rng.standard_normal((long_steps, walkers))
        else:
            assert long_steps % short_steps == 0
            long = np.repeat(short, long_steps // short_steps, axis=0)
            if profile == "scaled-fixed-sample":
                long = -1.0 + (long + 1.0) / 2
            else:
                assert profile == "repeated-fixed-sample"

        assert short.shape == (short_steps, walkers) and long.shape == (long_steps, walkers), "fixed-sample input shape"
        assert short.dtype == long.dtype == np.float64, "fixed-sample input dtype"
        if profile == "independent-fixed-streams":
            repeated = np.repeat(short, long_steps // short_steps, axis=0)
            assert np.max(np.abs(long - repeated)) > ATOL_PYTHON_FP64_LOOSE, "independent input duplicates repeated fixture"
            scaled = -1.0 + (repeated + 1.0) / 2
            assert np.max(np.abs(long - scaled)) > ATOL_PYTHON_FP64_LOOSE, "independent input duplicates scaled fixture"

        repeated = np.repeat(short, 4, axis=0)
        scaled = -1.0 + (repeated + 1.0) / 2
        input_signature = np.asarray(
            [
                *short.shape,
                *long.shape,
                short.dtype == np.float64,
                long.dtype == np.float64,
                np.max(np.abs(long - repeated)) > ATOL_PYTHON_FP64_LOOSE,
                np.max(np.abs(long - scaled)) > ATOL_PYTHON_FP64_LOOSE,
            ]
        )
        required_signature = np.asarray(
            [100, 4, 400, 4, 1, 1]
            + {"independent-fixed-streams": [1, 1], "repeated-fixed-sample": [0, 1], "scaled-fixed-sample": [1, 0]}[profile]
        )
        np.testing.assert_array_equal(input_signature, required_signature)

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        actual = np.asarray(
            [
                self._run_energy(tmp_path / "a", short, np.ones_like(short), bins),
                self._run_energy(tmp_path / "b", long, np.ones_like(long), bins),
            ]
        )
        reference = np.asarray([equal_bin_statistics(short, bins), equal_bin_statistics(long, bins)])
        assert actual.shape == reference.shape == (2, 2)
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        assert np.all(reference[:, 1] > 0)
        ratio = actual[0, 1] / actual[1, 1]
        reference_ratio = {
            "independent-fixed-streams": reference[0, 1] / reference[1, 1],
            "repeated-fixed-sample": 1.0,
            "scaled-fixed-sample": 2.0,
        }[profile]
        np.testing.assert_allclose(actual, reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(ratio, reference_ratio, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_zero_variance_data_reports_zero_error(self, tmp_path):
        """A constant e_L is an exact-eigenstate sample: the error must vanish."""
        steps, walkers, bins = 120, 4, 10
        e_L = np.full((steps, walkers), -2.75)
        mean, std = self._run_energy(tmp_path, e_L, np.ones((steps, walkers)), bins)
        np.testing.assert_allclose(mean, -2.75, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)
        assert std < ATOL_PYTHON_FP64_LOOSE, std


def _write_lrdmc_extrapolation_series(
    tmp_path, *, num_twists=2, num_ranks=1, complex_values=False, with_mpc=False, spacings=(0.25, 0.5)
):
    """Write two lattice spacings with exactly known real bin-mean statistics.

    Only rank 0 holds the reduced GFMC history, including in the MPI layout.
    The first 35 samples are deliberately wrong: collect=5 and warmup=30 must
    discard them. The remaining ten equal-size bins have unit weights.
    """
    weights = {1: (1.0,), 2: (0.25, 0.75), 3: (0.1, 0.2, 0.7)}[num_twists]
    checkpoint_dirs = []
    for spacing_idx, alat in enumerate(spacings):
        directory = tmp_path / f"a{spacing_idx}" / "restart.h5"
        directory.mkdir(parents=True)
        checkpoint_dirs.append(str(directory))
        for twist_idx, weight in enumerate(weights):
            obj = SimpleNamespace(alat=alat, num_gfmc_collect_steps=5, mpc=with_mpc, mpc_cutoff=30.0)
            centered_bins = np.arange(10, dtype=np.float64) - 4.5
            fluctuations = np.repeat(centered_bins, 10) * (twist_idx + 1) / 8
            mean = -3.0 - 4.0 * twist_idx + (twist_idx + 1) * alat**2
            history = mean + fluctuations
            if complex_values:
                history = history + 100j * fluctuations
            obj.e_L = np.concatenate((np.full(35, 999.0), history))[:, None]
            obj.w_L = np.ones((135, 1))
            if with_mpc:
                obj.mpc_dV = 2.0 - np.real(obj.e_L)
            objs = [obj] + [SimpleNamespace(alat=alat, mpc=with_mpc, mpc_cutoff=30.0) for _ in range(num_ranks - 1)]
            path = directory / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), objs)
            with h5py.File(path, "a") as f:
                meta = f["_meta"].attrs
                meta["driver_type"] = "GFMC_t"
                meta["twist_index"] = twist_idx
                meta["normalized_twist_weight"] = weight
                meta["num_twists"] = num_twists
                meta["converged"] = True
                meta["k_point"] = (0.25 * twist_idx, 0.0, 0.0)
                if with_mpc:
                    meta["mpc_cutoff"] = 30.0
    return checkpoint_dirs


class TestLRDMCTwistExtrapolation:
    """Continuum extrapolation consumes one complete BZ estimate per spacing."""

    @pytest.mark.light
    @pytest.mark.parametrize("num_twists", [1, 2])
    @pytest.mark.parametrize("num_ranks", [1, 3])
    @pytest.mark.parametrize("complex_values", [False, True])
    @pytest.mark.parametrize("with_mpc", [False, True])
    def test_bz_points_and_intercept_match_analytic_statistics(self, tmp_path, num_twists, num_ranks, complex_values, with_mpc):
        """Unequal BZ weights and real-part bin statistics determine E(a) and E0.

        Unit-weight bins have SE = sample_sd(bin means) / sqrt(N_bin), an
        independent statistical identity, not a copy of the jackknife code.
        The two-spacing linear intercept and its propagated error are exact
        (E0 = (4 E1 - E2) / 3, sigma0 = sqrt(17) / 3 sigma) with no injected
        random draws: the continuum fit is deterministic closed-form algebra.
        MPC data must not change the uncorrected-E extrapolation.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(
            tmp_path, num_twists=num_twists, num_ranks=num_ranks, complex_values=complex_values, with_mpc=with_mpc
        )
        weights = np.array([1.0] if num_twists == 1 else [0.25, 0.75])
        offsets = np.array([-3.0, -7.0])[:num_twists]
        slopes = np.array([1.0, 2.0])[:num_twists]
        per_twist_error = slopes / 8 * np.std(np.arange(10), ddof=1) / np.sqrt(10)
        expected_error = np.sqrt(np.sum((weights * per_twist_error) ** 2))
        expected_intercept = weights @ offsets
        alat = np.array([0.25, 0.5])
        expected_energy = expected_intercept + (weights @ slopes) * alat**2
        runner = CliRunner()
        options = ["-b", "10", "-w", "30", "-c", "5"]
        result = runner.invoke(lrdmc_app, ["extrapolate-energy", *directories, "-p", "1", *options])
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(
            re.findall(r"For a = ([\d.eE+-]+) bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output), dtype=float
        )
        assert points.shape == (2, 3)
        np.testing.assert_allclose(
            points,
            np.column_stack((alat, expected_energy, np.full(2, expected_error))),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        match = re.search(r"For a -> 0 bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match is not None, result.output
        # E0 = (4 E(a=0.25) - E(a=0.5)) / 3 for independent runs.
        np.testing.assert_allclose(
            np.asarray(match.groups(), dtype=float),
            [expected_intercept, np.sqrt(17) / 3 * expected_error],
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        for directory, point in zip(directories, points, strict=True):
            energy_result = runner.invoke(lrdmc_app, ["compute-energy", directory, *options])
            assert energy_result.exit_code == 0, energy_result.output
            energy_match = re.search(r"^E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", energy_result.output, re.MULTILINE)
            assert energy_match is not None
            np.testing.assert_array_equal(np.asarray(energy_match.groups(), dtype=float), point[1:])
        if with_mpc:
            corrected = np.asarray(
                re.findall(r"For a (?:= [\d.eE+-]+|-> 0) bohr: E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output),
                dtype=float,
            )
            assert corrected.shape == (3, 2)
            # E + dV is exactly constant; adding their independent variances is wrong.
            np.testing.assert_allclose(corrected, [[2.0, 0.0]] * 3, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        else:
            # Production stores a default mpc_cutoff even with mpc=False.
            assert "E_MPC" not in result.output

    @pytest.mark.light
    def test_continuum_estimate_is_deterministic_and_independent_of_global_rng(self, tmp_path):
        """Repeated calls print bit-identical continuum lines whatever the global NumPy RNG state.

        A Gaussian resampling with unseeded ``np.random.randn`` columns would
        change E0 and its error bar on every call.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=2, with_mpc=True)
        args = ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "30", "-c", "5"]
        continuum_lines = []
        for global_seed in (1, 2, None):
            if global_seed is not None:
                np.random.seed(global_seed)
            result = CliRunner().invoke(lrdmc_app, args)
            assert result.exit_code == 0, result.output + str(result.exception)
            lines = re.findall(r"^  For a -> 0 bohr: .*$", result.output, re.MULTILINE)
            assert len(lines) == 2, result.output
            continuum_lines.append(lines)
        assert continuum_lines[1] == continuum_lines[0]
        assert continuum_lines[2] == continuum_lines[0]

    @pytest.mark.light
    @pytest.mark.parametrize(
        "polynomial_order, spacings",
        [(1, (0.2, 0.3, 0.4, 0.5)), (2, (0.2, 0.3, 0.4)), (2, (0.2, 0.3, 0.4, 0.5))],
    )
    @pytest.mark.parametrize("with_mpc", [False, True])
    def test_overdetermined_and_interpolating_fits_match_normal_equations(self, tmp_path, polynomial_order, spacings, with_mpc):
        """Non-polynomial E(a^2) points pin the least-squares intercept and its propagated error.

        Each spacing's energy history is shifted by a known offset after the
        warmup, so the points do not lie on the fitted polynomial. The reference
        solves the normal equations (V^T V) w = V^T y independently of the
        production pseudo-inverse: E0 = c . y with c = e0^T (V^T V)^-1 V^T and
        sigma0 = sqrt(sum_i (c_i sigma_i)^2). With MPC the corrected series is
        E_MPC = 2 + offset with zero error.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=1, with_mpc=with_mpc, spacings=spacings)
        offsets = np.array([0.011, -0.023, 0.017, -0.007])[: len(spacings)]
        for directory, offset in zip(directories, offsets, strict=True):
            with h5py.File(Path(directory) / "twist_000.h5", "a") as f:
                f["rank_0/observables/e_L"][:] = f["rank_0/observables/e_L"][:] + offset
        alat = np.asarray(spacings)
        sigma = np.std(np.arange(10), ddof=1) / np.sqrt(10) / 8
        energies = -3.0 + alat**2 + offsets
        vander = np.vander(alat**2, polynomial_order + 1)
        coefficients = np.linalg.solve(vander.T @ vander, vander.T)[-1]
        options = ["-p", str(polynomial_order), "-b", "10", "-w", "30", "-c", "5"]
        result = CliRunner().invoke(lrdmc_app, ["extrapolate-energy", *directories, *options])
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(
            re.findall(r"For a = [\d.eE+-]+ bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output), dtype=float
        )
        np.testing.assert_allclose(
            points,
            np.column_stack((energies, np.full(len(alat), sigma))),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        match = re.search(r"For a -> 0 bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match is not None, result.output
        np.testing.assert_allclose(
            np.asarray(match.groups(), dtype=float),
            [coefficients @ energies, sigma * np.sqrt(np.sum(coefficients**2))],
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        corrected = re.search(r"For a -> 0 bohr: E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
        if with_mpc:
            assert corrected is not None, result.output
            np.testing.assert_allclose(
                np.asarray(corrected.groups(), dtype=float),
                [coefficients @ (2.0 + offsets), 0.0],
                atol=ATOL_PYTHON_FP64_LOOSE,
                rtol=RTOL_PYTHON_FP64_ZERO,
            )
        else:
            assert corrected is None

    @pytest.mark.light
    def test_nonuniform_projection_weights_match_the_weighted_reference(self, tmp_path):
        """History-dependent GFMC weights reach each twist's ratio estimator."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_ranks=3, complex_values=True)
        expected = []
        for spacing_idx, directory in enumerate(directories):
            per_twist = []
            for twist_idx in range(2):
                with h5py.File(Path(directory) / f"twist_{twist_idx:03d}.h5", "a") as f:
                    obs = f["rank_0/observables"]
                    raw_e = np.real(obs["e_L"][:])
                    raw_w = np.linspace(0.85, 1.1 + twist_idx / 10 + spacing_idx / 20, len(raw_e))[:, None]
                    obs["w_L"][:] = raw_w
                per_twist.append(_reference_lrdmc_jackknife([{"e_L": raw_e, "w_L": raw_w}], 30, 10, 5))
            means, errors = np.asarray(per_twist).T
            expected.append([np.dot([0.25, 0.75], means), np.sqrt(np.sum((np.array([0.25, 0.75]) * errors) ** 2))])
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "30", "-c", "5"]
        )
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(
            re.findall(r"For a = [\d.eE+-]+ bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output), dtype=float
        )
        assert points.shape == (2, 2)
        np.testing.assert_allclose(points, expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    @pytest.mark.parametrize(
        "defect,diagnostic",
        [
            ("weights", "twist set and BZ weights"),
            ("k_point", "twist set and BZ weights"),
            ("missing_k_point", "k_point"),
            ("malformed_k_point", "k_point"),
            ("nonfinite_k_point", "k_point"),
            ("duplicate_k_point", "duplicate k_point"),
            ("twist_alat", "same alat"),
            ("rank_alat", "same alat"),
            ("missing_alat", "alat"),
            ("zero_alat", "positive finite alat"),
            ("negative_alat", "positive finite alat"),
            ("nan_alat", "positive finite alat"),
            ("overflow_alat", "positive finite alat"),
            ("missing_weight", "normalized_twist_weight"),
            ("missing_count", "num_twists"),
            ("incomplete", "incomplete twist checkpoint set"),
            ("different_twist_count", "twist set and BZ weights"),
            ("unconverged", "not converged"),
        ],
    )
    def test_inconsistent_twist_series_is_rejected(self, tmp_path, defect, diagnostic):
        """Invalid metadata must fail before producing a continuum estimate."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_ranks=3)
        path = Path(directories[1]) / "twist_001.h5"
        with h5py.File(path, "a") as f:
            meta = f["_meta"].attrs
            cfg = f["rank_0/driver_config"].attrs
            if defect == "weights":
                meta["normalized_twist_weight"] = 0.5
            elif defect == "k_point":
                meta["k_point"] = (0.375, 0.0, 0.0)
            elif defect == "missing_k_point":
                del meta["k_point"]
            elif defect == "malformed_k_point":
                meta["k_point"] = (0.25, 0.0)
            elif defect == "nonfinite_k_point":
                meta["k_point"] = (np.nan, 0.0, 0.0)
            elif defect == "duplicate_k_point":
                meta["k_point"] = (0.0, 0.0, 0.0)
            elif defect == "twist_alat":
                for rank in range(3):
                    f[f"rank_{rank}/driver_config"].attrs["alat"] = 0.375
            elif defect == "rank_alat":
                f["rank_1/driver_config"].attrs["alat"] = 0.375
            elif defect == "missing_alat":
                del cfg["alat"]
            elif defect in ("zero_alat", "negative_alat", "nan_alat", "overflow_alat"):
                cfg["alat"] = {"zero_alat": 0.0, "negative_alat": -0.5, "nan_alat": np.nan, "overflow_alat": 1.0e200}[defect]
            elif defect == "missing_weight":
                del meta["normalized_twist_weight"]
            elif defect == "missing_count":
                del meta["num_twists"]
            elif defect == "unconverged":
                meta["converged"] = False
        if defect in ("incomplete", "different_twist_count"):
            path.rename(path.with_suffix(".incomplete"))
        if defect == "different_twist_count":
            with h5py.File(Path(directories[1]) / "twist_000.h5", "a") as f:
                f["_meta"].attrs["num_twists"] = 1
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "30", "-c", "5"]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception.__context__, BadParameter)
        assert diagnostic in str(result.exception.__context__)
        assert "For a -> 0" not in result.output

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["directories", "flat_files", "mixed"])
    def test_ambiguous_or_duplicate_spacing_inputs_are_rejected(self, tmp_path, layout):
        """Repeated a^2 or mixing a BZ mean with one twist is not a valid fit."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path)
        if layout == "directories":
            paths = [directories[0], directories[0], directories[1]]
            diagnostic = "duplicate lattice spacing"
        elif layout == "flat_files":
            paths = [str(Path(directories[0]) / f"twist_{idx:03d}.h5") for idx in range(2)]
            diagnostic = "duplicate lattice spacing"
        else:
            paths = [directories[0], str(Path(directories[1]) / "twist_000.h5")]
            diagnostic = "cannot mix"
        result = CliRunner().invoke(lrdmc_app, ["extrapolate-energy", *paths, "-p", "1", "-b", "10", "-w", "30", "-c", "5"])
        assert result.exit_code != 0
        assert isinstance(result.exception.__context__, BadParameter)
        assert diagnostic in str(result.exception.__context__)
        assert "For a -> 0" not in result.output

    @pytest.mark.light
    @pytest.mark.parametrize("num_twists", [2, 3])
    def test_reindexed_twists_keep_the_same_bz_estimate(self, tmp_path, num_twists):
        """Twist identity is the k point and weight, not its directory index."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=num_twists)
        directory = Path(directories[1])
        first, second, temporary = (directory / name for name in ("twist_000.h5", f"twist_{num_twists - 1:03d}.h5", "swap.h5"))
        first.rename(temporary)
        second.rename(first)
        temporary.rename(second)
        for idx, path in ((0, first), (num_twists - 1, second)):
            with h5py.File(path, "a") as f:
                f["_meta"].attrs["twist_index"] = idx
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "30", "-c", "5"]
        )
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(re.findall(r"For a = ([\d.eE+-]+) bohr: E = ([\d.eE+-]+)", result.output), dtype=float)
        assert points.shape == (2, 2)
        expected_intercept, expected_slope = {2: (-6.0, 1.75), 3: (-9.4, 2.6)}[num_twists]
        np.testing.assert_allclose(
            points[:, 1],
            expected_intercept + expected_slope * points[:, 0] ** 2,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    @pytest.mark.light
    def test_legacy_flat_complex_checkpoints_use_real_energy(self, tmp_path):
        """Legacy flat checkpoints need no twist metadata and retain real E/SE."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=1, complex_values=True)
        files = [str(Path(directory) / "twist_000.h5") for directory in directories]
        for path in files:
            with h5py.File(path, "a") as f:
                for key in ("twist_index", "normalized_twist_weight", "num_twists", "converged", "k_point"):
                    del f["_meta"].attrs[key]
        result = CliRunner().invoke(lrdmc_app, ["extrapolate-energy", *files, "-p", "1", "-b", "10", "-w", "30", "-c", "5"])
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(
            re.findall(r"For a = ([\d.eE+-]+) bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output), dtype=float
        )
        assert points.shape == (2, 3)
        np.testing.assert_allclose(
            points[:, 1],
            -3.0 + points[:, 0] ** 2,
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            points[:, 2],
            np.std(np.arange(10), ddof=1) / (8 * np.sqrt(10)),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "defect,diagnostic",
        [
            ("empty", "jackknife samples"),
            ("short", "nonempty bins"),
            ("shape", "same (steps, walkers) shape"),
            ("zero_weight", "jackknife samples"),
            ("nan_energy", "non-finite LRDMC energy"),
        ],
    )
    def test_invalid_histories_are_rejected(self, tmp_path, defect, diagnostic):
        """An invalid per-twist history cannot turn into a NaN continuum estimate."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path)
        with h5py.File(Path(directories[1]) / "twist_001.h5", "a") as f:
            obs = f["rank_0/observables"]
            if defect in ("empty", "short", "shape"):
                length = {"empty": 0, "short": 35, "shape": 100}[defect]
                keys = ("e_L", "w_L") if defect != "shape" else ("w_L",)
                for key in keys:
                    data = obs[key][:length]
                    del obs[key]
                    obs.create_dataset(key, data=data)
            elif defect == "zero_weight":
                obs["w_L"][:] = 0.0
            elif defect == "nan_energy":
                obs["e_L"][36, 0] = np.nan
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "30", "-c", "5"]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception.__context__, BadParameter)
        assert diagnostic in str(result.exception.__context__)
        assert "For a -> 0" not in result.output

    @pytest.mark.light
    @pytest.mark.parametrize(
        "option,value,diagnostic",
        [
            ("-p", "2", "distinct lattice spacings"),
            ("-p", "-1", "nonnegative"),
            ("-b", "1", "num_gfmc_bin_blocks (1) must be >= 10"),
            ("-w", "-1", "num_gfmc_warmup_steps (-1) must be >= 25"),
            ("-c", "0", "num_gfmc_collect_steps (0) must be >= 5"),
        ],
    )
    def test_invalid_analysis_controls_are_rejected(self, tmp_path, option, value, diagnostic):
        """Too few distinct spacings and undefined jackknife controls are errors.

        Undefined jackknife controls (one bin, negative warmup, no weight
        history) lie below the documented LRDMC minima and are refused by them
        with exit status 1, as in compute-energy.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path)
        options = {"-p": "1", "-b": "10", "-w": "30", "-c": "5"}
        options[option] = value
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, *(item for pair in options.items() for item in pair)]
        )
        assert result.exit_code != 0
        if option == "-p":
            assert isinstance(result.exception.__context__, BadParameter)
            assert diagnostic in str(result.exception.__context__)
        else:
            assert result.exit_code == 1
            assert diagnostic in _flatten_cli_output(result.output)
        assert "For a -> 0" not in result.output


class TestLRDMCMPCExtrapolation:
    """The additional E_MPC series is complete, cutoff-compatible, and joint-binned."""

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["directories", "flat_files"])
    def test_disabled_mpc_with_default_cutoff_is_energy_only(self, tmp_path, layout):
        """A default cutoff in driver_config must not be mistaken for MPC enabled."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=1, with_mpc=False)
        paths = directories if layout == "directories" else [str(Path(d) / "twist_000.h5") for d in directories]
        result = CliRunner().invoke(lrdmc_app, ["extrapolate-energy", *paths, "-p", "1", "-b", "10", "-w", "30", "-c", "5"])
        assert result.exit_code == 0, result.output + str(result.exception)
        assert "E_MPC" not in result.output
        assert "For a -> 0 bohr: E =" in result.output

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["directories", "flat_files"])
    @pytest.mark.parametrize("num_ranks", [1, 3])
    @pytest.mark.parametrize("complex_values", [False, True])
    @pytest.mark.parametrize("correlation", [-1.0, -0.5, 0.5])
    def test_covariance_aware_points_and_intercept(self, tmp_path, layout, num_ranks, complex_values, correlation):
        """Known bin statistics pin both positive and negative covariance at a and zero."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        num_twists = 2 if layout == "directories" else 1
        directories = _write_lrdmc_extrapolation_series(
            tmp_path, num_twists=num_twists, num_ranks=num_ranks, complex_values=complex_values, with_mpc=True
        )
        alat = np.array([0.25, 0.5])
        for a, directory in zip(alat, directories, strict=True):
            for twist_idx in range(num_twists):
                with h5py.File(Path(directory) / f"twist_{twist_idx:03d}.h5", "a") as f:
                    obs = f["rank_0/observables"]
                    mean_e = -3.0 - 4.0 * twist_idx + (twist_idx + 1) * a**2
                    obs["mpc_dV"][:] = 0.5 + (twist_idx + 1) * a**2 + correlation * (np.real(obs["e_L"][:]) - mean_e)
                    if layout == "flat_files":
                        # Legacy files may lack enable/summary metadata; the
                        # histories and rank-config cutoffs still identify MPC.
                        del f["_meta"].attrs["mpc_cutoff"]
                        for rank in range(num_ranks):
                            del f[f"rank_{rank}/driver_config"].attrs["mpc"]
        paths = directories if layout == "directories" else [str(Path(d) / "twist_000.h5") for d in directories]
        weights = np.array([0.25, 0.75] if num_twists == 2 else [1.0])
        slopes = np.arange(1, num_twists + 1)
        intercept = weights @ (-2.5 - 4 * np.arange(num_twists))
        per_twist_sigma = (1 + correlation) * slopes / 8 * np.std(np.arange(10), ddof=1) / np.sqrt(10)
        expected_sigma = np.sqrt(np.sum((weights * per_twist_sigma) ** 2))
        expected_points = np.column_stack((alat, intercept + 2 * (weights @ slopes) * alat**2, np.full(2, expected_sigma)))
        runner = CliRunner()
        options = ["-b", "10", "-w", "30", "-c", "5"]
        result = runner.invoke(lrdmc_app, ["extrapolate-energy", *paths, "-p", "1", *options])
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(
            re.findall(r"For a = ([\d.eE+-]+) bohr: E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output), dtype=float
        )
        assert points.shape == (2, 3)
        np.testing.assert_allclose(points, expected_points, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        match = re.search(r"For a -> 0 bohr: E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match is not None, result.output
        np.testing.assert_allclose(
            np.asarray(match.groups(), dtype=float),
            [intercept, np.sqrt(17) / 3 * expected_sigma],
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        assert "For a -> 0 bohr: E =" in result.output
        for path, point in zip(paths, points, strict=True):
            energy_result = runner.invoke(lrdmc_app, ["compute-energy", path, *options])
            assert energy_result.exit_code == 0, energy_result.output
            energy_match = re.search(r": E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", energy_result.output)
            assert energy_match is not None
            np.testing.assert_array_equal(np.asarray(energy_match.groups(), dtype=float), point[1:])
            # jqmc.output_parser extracts every energy this output prints, MPC included (#308).
            parsed = output_parser.parse_tool_energy(energy_result.output)
            for name in ("energy", "energy_error", "dE_mpc", "dE_mpc_error", "energy_mpc", "energy_mpc_error"):
                assert parsed[name] is not None and np.isfinite(parsed[name]), name
            assert parsed["num_twists"] == (num_twists if layout == "directories" else None)

    @pytest.mark.light
    def test_nonuniform_weights_and_multiple_contributing_ranks(self, tmp_path):
        """Joint sample-level E+dV is the independent reference for weighted MPI data."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_ranks=3, complex_values=True, with_mpc=True)
        expected = []
        for spacing_idx, directory in enumerate(directories):
            estimates = []
            for twist_idx in range(2):
                ranks = []
                with h5py.File(Path(directory) / f"twist_{twist_idx:03d}.h5", "a") as f:
                    raw_e = f["rank_0/observables/e_L"][:]
                    for rank in range(2):
                        obs = f[f"rank_{rank}/observables"]
                        e = raw_e + rank / 4
                        dv = 0.25 - np.real(e) / 2 + np.arange(len(e))[:, None] / 100
                        w = np.linspace(0.9, 1.1 + (rank + twist_idx + spacing_idx) / 10, len(e))[:, None]
                        for key, data in (("e_L", e), ("mpc_dV", dv), ("w_L", w)):
                            if key in obs:
                                del obs[key]
                            obs.create_dataset(key, data=data)
                        ranks.append({"e_L": np.real(e) + dv, "w_L": w})
                estimates.append(_reference_lrdmc_jackknife(ranks, 30, 10, 5))
            means, sigmas = np.asarray(estimates).T
            expected.append([np.dot([0.25, 0.75], means), np.sqrt(np.sum((np.array([0.25, 0.75]) * sigmas) ** 2))])
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "30", "-c", "5"]
        )
        assert result.exit_code == 0, result.output + str(result.exception)
        points = np.asarray(
            re.findall(r"For a = [\d.eE+-]+ bohr: E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output), dtype=float
        )
        assert points.shape == (2, 2)
        np.testing.assert_allclose(points, expected, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_quadratic_fit_and_saved_graph_include_both_energies(self, tmp_path):
        """A known quartic in a reaches the default fit and both labelled plot series."""
        import matplotlib.pyplot as plt
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        spacings = (0.125, 0.25, 0.5)
        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=1, with_mpc=True, spacings=spacings)
        for a, directory in zip(spacings, directories, strict=True):
            with h5py.File(Path(directory) / "twist_000.h5", "a") as f:
                obs = f["rank_0/observables"]
                obs["mpc_dV"][:] = -2 + a**2 + 3 * a**4 - obs["e_L"][:]
        graph = tmp_path / "extrapolation.png"
        result = CliRunner().invoke(
            lrdmc_app, ["extrapolate-energy", *directories, "-b", "10", "-w", "30", "-c", "5", "-s", str(graph)]
        )
        assert result.exit_code == 0, result.output + str(result.exception)
        match = re.search(r"For a -> 0 bohr: E_MPC = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match is not None
        np.testing.assert_allclose(
            np.asarray(match.groups(), dtype=float), [-2, 0], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        assert graph.is_file() and graph.stat().st_size > 0
        labels = plt.gcf().axes[0].get_legend_handles_labels()[1]
        assert "E data" in labels and "E_MPC data" in labels
        assert "E extrapolated" in labels and "E_MPC extrapolated" in labels
        plt.close("all")

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["directories", "flat_files"])
    @pytest.mark.parametrize("mpc_marker", ["flag", "summary"])
    def test_enabled_mpc_cannot_lose_all_histories(self, tmp_path, layout, mpc_marker):
        """Losing all corrections cannot silently downgrade a recorded MPC scan to E-only."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=1, with_mpc=True)
        paths = directories if layout == "directories" else [str(Path(d) / "twist_000.h5") for d in directories]
        for directory in directories:
            with h5py.File(Path(directory) / "twist_000.h5", "a") as f:
                del f["rank_0/observables/mpc_dV"]
                if mpc_marker == "flag":
                    del f["_meta"].attrs["mpc_cutoff"]
                else:
                    del f["rank_0/driver_config"].attrs["mpc"]
        result = CliRunner().invoke(lrdmc_app, ["extrapolate-energy", *paths, "-p", "1", "-b", "10", "-w", "30", "-c", "5"])
        assert result.exit_code != 0
        assert isinstance(result.exception.__context__, BadParameter)
        assert "missing mpc_dV" in str(result.exception.__context__)
        assert "For a -> 0" not in result.output

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["directories", "flat_files"])
    @pytest.mark.parametrize(
        "defect,diagnostic",
        [
            ("missing_history", "missing mpc_dV"),
            ("empty_history", "missing mpc_dV"),
            ("missing_active_rank", "missing mpc_dV"),
            ("mpc_disabled_at_one_spacing", "MPC data must be present at every"),
            ("flag_false", "mpc=False"),
            ("empty_rank_flag_false", "mpc=False"),
            ("invalid_flag", "mpc flag"),
            ("missing_cutoff", "mpc_cutoff"),
            ("rank_cutoff", "same mpc_cutoff"),
            ("spacing_cutoff", "same mpc_cutoff"),
            ("meta_cutoff", "same mpc_cutoff"),
            ("zero_cutoff", "positive finite"),
            ("negative_cutoff", "positive finite"),
            ("nan_cutoff", "positive finite"),
            ("inf_cutoff", "positive finite"),
            ("complex_cutoff", "positive finite"),
            ("array_cutoff", "positive finite"),
            ("string_cutoff", "positive finite"),
            ("shape", "same (steps, walkers) shape"),
            ("broadcastable_shape", "same (steps, walkers) shape"),
            ("nan_history", "non-finite MPC"),
        ],
    )
    def test_inconsistent_mpc_series_is_rejected(self, tmp_path, layout, defect, diagnostic):
        """Never drop a partial correction or fit incompatible MPC Hamiltonians."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        num_twists = 2 if layout == "directories" else 1
        directories = _write_lrdmc_extrapolation_series(tmp_path, num_twists=num_twists, num_ranks=3, with_mpc=True)
        paths = directories if layout == "directories" else [str(Path(d) / "twist_000.h5") for d in directories]
        for twist_idx in range(num_twists if defect == "mpc_disabled_at_one_spacing" else 1):
            with h5py.File(Path(directories[1]) / f"twist_{twist_idx:03d}.h5", "a") as f:
                obs = f["rank_0/observables"]
                cfg = f["rank_0/driver_config"].attrs
                meta = f["_meta"].attrs
                if defect in ("missing_history", "empty_history", "mpc_disabled_at_one_spacing"):
                    del obs["mpc_dV"]
                    if defect == "empty_history":
                        obs.create_dataset("mpc_dV", data=np.empty((0, 1)))
                    if defect == "mpc_disabled_at_one_spacing":
                        del meta["mpc_cutoff"]
                        for rank in range(3):
                            f[f"rank_{rank}/driver_config"].attrs["mpc"] = False
                elif defect == "missing_active_rank":
                    for key in ("e_L", "w_L"):
                        f["rank_1/observables"].create_dataset(key, data=obs[key][:])
                elif defect == "flag_false":
                    cfg["mpc"] = False
                elif defect == "empty_rank_flag_false":
                    f["rank_1/driver_config"].attrs["mpc"] = False
                elif defect == "invalid_flag":
                    cfg["mpc"] = "false"
                elif defect == "missing_cutoff":
                    del cfg["mpc_cutoff"]
                elif defect == "rank_cutoff":
                    f["rank_1/driver_config"].attrs["mpc_cutoff"] = 40.0
                elif defect == "spacing_cutoff":
                    meta["mpc_cutoff"] = 40.0
                    for rank in range(3):
                        f[f"rank_{rank}/driver_config"].attrs["mpc_cutoff"] = 40.0
                elif defect == "meta_cutoff":
                    meta["mpc_cutoff"] = 40.0
                elif defect.endswith("_cutoff"):
                    cfg["mpc_cutoff"] = {
                        "zero_cutoff": 0.0,
                        "negative_cutoff": -30.0,
                        "nan_cutoff": np.nan,
                        "inf_cutoff": np.inf,
                        "complex_cutoff": 30.0 + 1j,
                        "array_cutoff": [30.0],
                        "string_cutoff": "30",
                    }[defect]
                elif defect in ("shape", "broadcastable_shape"):
                    data = obs["mpc_dV"][:100] if defect == "shape" else obs["mpc_dV"][:].T
                    del obs["mpc_dV"]
                    obs.create_dataset("mpc_dV", data=data)
                elif defect == "nan_history":
                    obs["mpc_dV"][36, 0] = np.nan
        result = CliRunner().invoke(lrdmc_app, ["extrapolate-energy", *paths, "-p", "1", "-b", "10", "-w", "30", "-c", "5"])
        assert result.exit_code != 0
        assert isinstance(result.exception.__context__, BadParameter)
        assert diagnostic in str(result.exception.__context__)
        assert "For a -> 0" not in result.output


class TestComputeEnergyMPC:
    """Regression: the MPC-corrected error bar must be covariance-aware."""

    @pytest.mark.light
    def test_mpc_total_error_uses_combined_jackknife(self, tmp_path):
        """E and dE_MPC share walkers and weights, so the combined bar is the
        jackknife std of E_jk + dV_jk -- NOT the independent quadrature, which
        under-reports by 25.5% on this fixture class."""
        rng = np.random.default_rng(7)
        num_steps, num_walkers, bins = 100, 4, 10
        obj = _make_mcmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False, rng=rng)
        e_L = rng.standard_normal((num_steps, num_walkers)) - 5.0
        # anti-correlated correction: quadrature OVER-reports here, combined is smaller
        mpc_dV = -0.5 * (e_L + 5.0) + 0.01 * rng.standard_normal((num_steps, num_walkers))
        w_L = rng.uniform(0.9, 1.1, (num_steps, num_walkers))
        obj.e_L, obj.w_L, obj.mpc_dV = e_L, w_L, mpc_dV
        chk = str(tmp_path / "mcmc.h5")
        _write_chk(chk, [obj])

        # reference combined jackknife (same binning as the CLI)
        w_split = np.array_split(w_L, bins, axis=0)
        w_binned = np.ravel([np.sum(a, axis=0) for a in w_split])
        we_split = np.array_split(w_L * e_L, bins, axis=0)
        we_binned = np.ravel([np.sum(a, axis=0) for a in we_split])
        wv_split = np.array_split(w_L * mpc_dV, bins, axis=0)
        wv_binned = np.ravel([np.sum(a, axis=0) for a in wv_split])
        M = w_binned.size
        E_jk = np.array([(we_binned.sum() - we_binned[m]) / (w_binned.sum() - w_binned[m]) for m in range(M)])
        dV_jk = np.array([(wv_binned.sum() - wv_binned[m]) / (w_binned.sum() - w_binned[m]) for m in range(M)])
        expected_bar = float(np.sqrt(M - 1) * np.std(E_jk + dV_jk))
        wrong_quadrature = float(np.sqrt((np.sqrt(M - 1) * np.std(E_jk)) ** 2 + (np.sqrt(M - 1) * np.std(dV_jk)) ** 2))

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-energy", chk, "-b", str(bins), "-w", "0"])
        assert result.exit_code == 0, result.output
        import re

        match = re.search(r"Total Energy .MPC-corrected.: E_MPC = (-?[\d.]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match, result.output
        reported_bar = float(match.group(2))
        assert abs(reported_bar - expected_bar) < ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT * max(1.0, abs(expected_bar)), (
            f"reported {reported_bar} vs combined {expected_bar}"
        )
        # the old independent-quadrature bar must NOT be what is reported
        assert abs(reported_bar - wrong_quadrature) > 0.05 * wrong_quadrature

    @pytest.mark.light
    def test_lrdmc_reports_mpc_when_only_rank0_holds_reduced_observables(self, tmp_path):
        """A merged multi-rank LRDMC checkpoint stores the already-MPI-
        reduced observable history under rank_0 only -- every other rank's
        HDF5 group is present but has no ``e_L``/``w_L``/``mpc_dV`` datasets
        at all (the real on-disk shape written by a production run, not a
        contrived corner case). The MPC gate considers only the ranks that
        actually contributed energy samples: requiring EVERY rank to carry a
        non-empty ``mpc_dV`` would make those empty placeholder ranks report
        "no MPC" even for a genuine ``mpc=True`` run, silently dropping the
        correction from ``jqmc-tool lrdmc compute-energy`` (and hence from
        ``workflow_state.toml``) while the live run's own printed output has it.

        (MCMC checkpoints always populate every rank's own ``e_L``/``w_L``/
        ``mpc_dV`` -- this "reduced onto rank_0" layout is LRDMC-specific,
        so there is no MCMC sibling of this test.)
        """
        rng = np.random.default_rng(13)
        num_steps, num_walkers, k, bins, warmup = 100, 4, 5, 10, 30
        obj = _make_lrdmc_obj(num_steps=num_steps, num_walkers=num_walkers, with_force=False, rng=rng)
        e_L = rng.standard_normal((num_steps, num_walkers)) - 4.0
        mpc_dV = 0.25 + 0.02 * rng.standard_normal((num_steps, num_walkers))
        w_L = rng.uniform(0.9, 1.1, (num_steps, num_walkers))
        obj.e_L, obj.w_L, obj.mpc_dV = e_L, w_L, mpc_dV
        chk = str(tmp_path / "lrdmc.h5")
        _write_chk(chk, [obj, SimpleNamespace(), SimpleNamespace()])

        expected_dE_mean, expected_dE_std = _reference_lrdmc_jackknife([{"e_L": mpc_dV, "w_L": w_L}], warmup, bins, k)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        result = CliRunner().invoke(lrdmc_app, ["compute-energy", chk, "-b", str(bins), "-w", str(warmup), "-c", str(k)])
        assert result.exit_code == 0, result.output
        assert "Found 3 MPI ranks." in result.output
        match = re.search(r"MPC correction: dE_MPC = (-?[\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
        assert match, f"MPC correction line missing from output:\n{result.output}"
        actual_mean, actual_std = map(float, match.groups())
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(actual_mean, expected_dE_mean, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_std, expected_dE_std, atol=atol, rtol=rtol)


class TestComputeForce:
    """Tests for mcmc and lrdmc compute-force commands."""

    @staticmethod
    def _write_twist_dir(checkpoint_dir, weights, *, pressure, force, driver_type="MCMC"):
        """Two twist checkpoints carrying only the requested internal summaries."""
        checkpoint_dir.mkdir()
        for twist_idx in range(2):
            obj = _make_mcmc_obj(num_steps=40, num_walkers=2, e_L_value=-1.0, with_force=False, atomic_labels=["H0", "H1"])
            path = checkpoint_dir / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), [obj], driver_type=driver_type)
            with h5py.File(path, "a") as f:
                meta = f["_meta"]
                meta.attrs["twist_index"] = twist_idx
                meta.attrs["normalized_twist_weight"] = weights[twist_idx]
                meta.attrs["num_twists"] = 2
                meta.attrs["converged"] = True
                meta.attrs["E_k"] = -1.0
                meta.attrs["sigma_k"] = 0.0
                if force:
                    meta.create_dataset("F_k", data=np.zeros((2, 3)))
                    meta.create_dataset("sigma_F_k", data=np.zeros((2, 3)))
                if pressure is not None:
                    meta.create_dataset("P_hydro_k", data=pressure[0][twist_idx])
                    meta.create_dataset("sigma_P_hydro_k", data=pressure[1][twist_idx])

    @pytest.mark.light
    @pytest.mark.parametrize("command", ["mcmc", "lrdmc"])
    def test_twist_stress_only_run_reports_pressure(self, tmp_path, command):
        """A stress-only twist run (no force summaries) prints its BZ pressure instead of failing."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        weights = np.array([0.25, 0.75])
        per_twist_P = np.array([0.01, 0.03])
        per_twist_sigma_P = np.array([0.001, 0.002])
        checkpoint_dir = tmp_path / "restart.h5"
        self._write_twist_dir(
            checkpoint_dir,
            weights,
            pressure=(per_twist_P, per_twist_sigma_P),
            force=False,
            driver_type="MCMC" if command == "mcmc" else "GFMC_n",
        )
        app = mcmc_app if command == "mcmc" else lrdmc_app
        args = ["compute-force", str(checkpoint_dir), "-b", "10", "-w", "30"]
        if command == "lrdmc":
            args = ["compute-force", str(checkpoint_dir), "-b", "10", "-w", "25", "-c", "5"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Atomic Forces:" not in result.output
        match = re.search(r"P_hydro = ([+-]?[\d.eE+-]+) \+- ([\d.eE+-]+) Ha/bohr\^3", result.output)
        assert match is not None, result.output
        assert float(match.group(1)) == float(np.dot(weights, per_twist_P))
        assert float(match.group(2)) == float(np.sqrt(np.dot(weights**2, per_twist_sigma_P**2)))

    @pytest.mark.light
    @pytest.mark.parametrize("missing", ["both", "partial-force"])
    def test_twist_force_summary_refusals(self, tmp_path, missing):
        """No force/pressure summaries at all, or a force summary on only some twists, is an explicit error."""
        import h5py as _h5py
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        checkpoint_dir = tmp_path / "restart.h5"
        self._write_twist_dir(checkpoint_dir, np.array([0.5, 0.5]), pressure=None, force=(missing == "partial-force"))
        if missing == "partial-force":
            with _h5py.File(checkpoint_dir / "twist_001.h5", "a") as f:
                del f["_meta"]["F_k"]
                del f["_meta"]["sigma_F_k"]
        result = CliRunner().invoke(mcmc_app, ["compute-force", str(checkpoint_dir), "-b", "10", "-w", "30"])
        assert result.exit_code != 0
        expected = "neither force nor pressure" if missing == "both" else "only partially present"
        assert expected in _flatten_cli_output(str(result.exception) + " " + result.output)

    @pytest.mark.light
    def test_twist_tool_matches_internal_energy_force_pressure_summaries(self, tmp_path):
        """Single-process tool output matches MPI-driver E/F/P summaries."""
        checkpoint_dir = tmp_path / "restart.h5"
        checkpoint_dir.mkdir()
        weights = np.array([0.25, 0.75])
        per_twist_E = np.array([-2.0, -6.0])
        per_twist_sigma_E = np.zeros(2)
        per_twist_P = np.array([0.01, 0.03])
        per_twist_sigma_P = np.array([0.001, 0.002])
        # Distinct O(1) per-twist forces with O(0.1) errors (a constant shift and a
        # scale of the Hellmann-Feynman samples). The driver's internal summary of a
        # twist is its force jackknife at the driver's settings, which this test
        # also passes to the tool (-b 2 -w 0).
        force_shift = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            ]
        )
        force_scale = np.array([100.0, 200.0])
        objs = []
        for twist_idx in range(2):
            obj = _make_mcmc_obj(
                num_steps=40,
                num_walkers=2,
                e_L_value=per_twist_E[twist_idx],
                with_force=True,
                atomic_labels=["H0", "H1"],
                rng=np.random.default_rng(42 + twist_idx),
            )
            obj.force_HF = force_scale[twist_idx] * obj.force_HF - force_shift[twist_idx]
            objs.append(obj)
        per_twist_force = [_reference_force_jackknife([obj], warmup=0, bins=2) for obj in objs]
        per_twist_F = np.asarray([mean for mean, _std in per_twist_force])
        per_twist_sigma_F = np.asarray([std for _mean, std in per_twist_force])

        for twist_idx, obj in enumerate(objs):
            path = checkpoint_dir / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), [obj])
            with h5py.File(path, "a") as f:
                meta = f["_meta"]
                meta.attrs["twist_index"] = twist_idx
                meta.attrs["normalized_twist_weight"] = weights[twist_idx]
                meta.attrs["num_twists"] = 2
                meta.attrs["converged"] = True
                meta.attrs["E_k"] = per_twist_E[twist_idx]
                meta.attrs["sigma_k"] = per_twist_sigma_E[twist_idx]
                meta.create_dataset("F_k", data=per_twist_F[twist_idx])
                meta.create_dataset("sigma_F_k", data=per_twist_sigma_F[twist_idx])
                meta.create_dataset("P_hydro_k", data=per_twist_P[twist_idx])
                meta.create_dataset("sigma_P_hydro_k", data=per_twist_sigma_P[twist_idx])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app
        from jqmc_workflow.forces_parser import parse_force_table

        runner = CliRunner()
        energy_result = runner.invoke(mcmc_app, ["compute-energy", str(checkpoint_dir), "-b", "2", "-w", "0"])
        assert energy_result.exit_code == 0, energy_result.output
        energy_match = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", energy_result.output)
        assert energy_match is not None

        force_result = runner.invoke(mcmc_app, ["compute-force", str(checkpoint_dir), "-b", "2", "-w", "0"])
        assert force_result.exit_code == 0, force_result.output
        parsed_force = parse_force_table(force_result.output)
        assert parsed_force is not None
        pressure_match = re.search(r"P_hydro = ([+-]?[\d.eE+-]+) \+- ([\d.eE+-]+) Ha/bohr\^3", force_result.output)
        assert pressure_match is not None

        internal_E = float(np.dot(weights, per_twist_E))
        internal_sigma_E = float(np.sqrt(np.dot(weights**2, per_twist_sigma_E**2)))
        internal_F = np.tensordot(weights, per_twist_F, axes=(0, 0))
        internal_sigma_F = np.sqrt(np.tensordot(weights**2, per_twist_sigma_F**2, axes=(0, 0)))
        internal_P = float(np.dot(weights, per_twist_P))
        internal_sigma_P = float(np.sqrt(np.dot(weights**2, per_twist_sigma_P**2)))

        np.testing.assert_allclose(
            np.asarray(energy_match.groups(), dtype=float),
            [internal_E, internal_sigma_E],
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        for atom_idx, row in enumerate(parsed_force):
            comparison_iteration = str(atom_idx)  # noqa: F841 - captured by the GDD iteration receipt
            np.testing.assert_allclose(
                [row["Fx"], row["Fy"], row["Fz"]],
                internal_F[atom_idx],
                atol=ATOL_TOOL_FORCE_TEXT_ROUNDTRIP,
                rtol=RTOL_PYTHON_FP64_ZERO,
            )
            np.testing.assert_allclose(
                [row["Fx_err"], row["Fy_err"], row["Fz_err"]],
                internal_sigma_F[atom_idx],
                atol=ATOL_TOOL_FORCE_TEXT_ROUNDTRIP,
                rtol=RTOL_PYTHON_FP64_ZERO,
            )
        np.testing.assert_allclose(
            np.asarray(pressure_match.groups(), dtype=float),
            [internal_P, internal_sigma_P],
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    _TWIST_WEIGHTS = np.array([0.25, 0.75])
    _TWIST_PRESSURE = (np.array([0.01, 0.03]), np.array([0.001, 0.002]))
    _DRIVER_SETTINGS = {"mcmc": {"bins": 3, "warmup": 5, "collect": None}, "lrdmc": {"bins": 3, "warmup": 5, "collect": 5}}

    @classmethod
    def _write_force_twist_dir(cls, checkpoint_dir, command, *, num_steps=40):
        """Two twists with fluctuating force histories plus the driver's stored summaries.

        The stored ``F_k``/``sigma_F_k`` are each twist's force jackknife at the
        driver's own settings (``_DRIVER_SETTINGS``), as the twist driver writes them.
        """
        checkpoint_dir.mkdir()
        lrdmc = command == "lrdmc"
        settings = cls._DRIVER_SETTINGS[command]
        objs = []
        for twist_idx in range(2):
            obj = _random_force_obj(num_steps, 2, 301 + twist_idx, lrdmc=lrdmc)
            # Distinct per-twist forces, so the BZ weights matter.
            obj.force_HF = obj.force_HF - (1.0 + 2.0 * twist_idx)
            objs.append(obj)
            stored_F, stored_sigma_F = _reference_force_jackknife(
                [obj], warmup=settings["warmup"], bins=settings["bins"], collect=settings["collect"]
            )
            path = checkpoint_dir / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), [obj])
            with h5py.File(path, "a") as f:
                meta = f["_meta"]
                meta.attrs["driver_type"] = "GFMC_n" if lrdmc else "MCMC"
                meta.attrs["twist_index"] = twist_idx
                meta.attrs["normalized_twist_weight"] = cls._TWIST_WEIGHTS[twist_idx]
                meta.attrs["num_twists"] = 2
                meta.attrs["converged"] = True
                meta.attrs["E_k"] = -1.0
                meta.attrs["sigma_k"] = 0.0
                meta.create_dataset("F_k", data=stored_F)
                meta.create_dataset("sigma_F_k", data=stored_sigma_F)
                meta.create_dataset("P_hydro_k", data=cls._TWIST_PRESSURE[0][twist_idx])
                meta.create_dataset("sigma_P_hydro_k", data=cls._TWIST_PRESSURE[1][twist_idx])
        return objs

    @classmethod
    def _bz_reference_rows(cls, objs, *, warmup, bins, collect):
        """Formatted BZ force rows from independent per-twist delete-one-replica references."""
        from uncertainties import ufloat

        per_twist = [_reference_force_jackknife([obj], warmup=warmup, bins=bins, collect=collect) for obj in objs]
        mean = np.tensordot(cls._TWIST_WEIGHTS, np.asarray([m for m, _s in per_twist]), axes=(0, 0))
        std = np.sqrt(np.tensordot(cls._TWIST_WEIGHTS**2, np.asarray([s for _m, s in per_twist]) ** 2, axes=(0, 0)))
        assert np.isfinite(mean).all() and np.all(std > 0)
        labels = ["H", "Li"]
        return [[label, *(f"{ufloat(mean[i, j], std[i, j]):+2uS}" for j in range(3))] for i, label in enumerate(labels)]

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("mcmc", {"-b": 4, "-w": 10}),
            ("mcmc", {"-b": 2, "-w": 0}),
            ("lrdmc", {"-b": 10, "-w": 30, "-c": 5}),
            ("lrdmc", {"-b": 10, "-w": 25, "-c": 7}),
        ],
        ids=["mcmc-longer-warmup", "mcmc-no-warmup", "lrdmc-longer-warmup", "lrdmc-other-collect"],
    )
    def test_twist_force_honours_analysis_options(self, tmp_path, command, args):
        """compute-force on a twist directory analyses every twist with the requested -b/-w/-c.

        The printed BZ force rows equal the normalized-weight combination of an
        independent delete-one-replica jackknife of each twist's histories at the
        requested options, not the summaries the driver stored at its own
        settings; the stored pressure summary is still printed. LRDMC twists
        hold 50 measurements, so both options respect the documented minima.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        checkpoint_dir = tmp_path / "restart.h5"
        objs = self._write_force_twist_dir(checkpoint_dir, command, num_steps=50 if command == "lrdmc" else 40)
        settings = self._DRIVER_SETTINGS[command]
        expected_rows = self._bz_reference_rows(objs, warmup=args["-w"], bins=args["-b"], collect=args.get("-c"))
        stored_rows = self._bz_reference_rows(
            objs, warmup=settings["warmup"], bins=settings["bins"], collect=settings["collect"]
        )
        assert expected_rows != stored_rows  # the requested options change the estimate
        cli_args = [item for key, value in args.items() for item in (key, str(value))]
        app = lrdmc_app if command == "lrdmc" else mcmc_app
        result = CliRunner().invoke(app, ["compute-force", str(checkpoint_dir), *cli_args])
        assert result.exit_code == 0, result.output
        rows = [line.split() for line in result.output.splitlines() if line.split() and line.split()[0] in ("H", "Li")]
        assert rows == expected_rows
        match = re.search(r"P_hydro = ([+-]?[\d.eE+-]+) \+- ([\d.eE+-]+) Ha/bohr\^3", result.output)
        assert match is not None, result.output
        np.testing.assert_allclose(
            np.asarray(match.groups(), dtype=float),
            [
                float(np.dot(self._TWIST_WEIGHTS, self._TWIST_PRESSURE[0])),
                float(np.sqrt(np.dot(self._TWIST_WEIGHTS**2, self._TWIST_PRESSURE[1] ** 2))),
            ],
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("command", "args", "defect", "message"),
        [
            ("mcmc", ["-b", "1000", "-w", "1000"], None, "fewer than num_mcmc_bin_blocks"),
            ("mcmc", ["-b", "3", "-w", "5"], "nan-force_HF", "non-finite atomic force estimate"),
            ("mcmc", ["-b", "3", "-w", "5"], "inf-P_hydro", "non-finite per-twist hydrostatic pressure"),
            ("mcmc", ["-b", "3", "-w", "5"], "nan-sigma_P_hydro", "non-finite per-twist hydrostatic pressure"),
            ("lrdmc", ["-b", "10", "-w", "40", "-c", "5"], None, "too few measurements after collect/warmup"),
            ("lrdmc", ["-b", "10", "-w", "25", "-c", "5"], "nan-force_HF", "non-finite atomic force estimate"),
            ("lrdmc", ["-b", "10", "-w", "25", "-c", "5"], "inf-P_hydro", "non-finite per-twist hydrostatic pressure"),
        ],
        ids=[
            "mcmc-warmup-eats-all",
            "mcmc-nan-force-sample",
            "mcmc-inf-pressure",
            "mcmc-nan-pressure-error",
            "lrdmc-warmup-eats-all",
            "lrdmc-nan-force-sample",
            "lrdmc-inf-pressure",
        ],
    )
    def test_twist_force_refuses_unusable_window_or_nonfinite_values(self, tmp_path, command, args, defect, message):
        """Twist-directory compute-force refuses an empty analysis window and non-finite values.

        Options that leave a twist with fewer measurements than bins, a non-finite
        force sample in the analysed window of one twist, and a non-finite stored
        per-twist pressure or pressure error exit non-zero with the reason and
        print neither a force table nor a pressure value.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        checkpoint_dir = tmp_path / "restart.h5"
        self._write_force_twist_dir(checkpoint_dir, command)
        with h5py.File(checkpoint_dir / "twist_001.h5", "a") as f:
            if defect == "nan-force_HF":
                force_HF = f["rank_0/observables/force_HF"]
                force_HF[-3, 0, 0, 0] = np.nan
            elif defect in ("inf-P_hydro", "nan-sigma_P_hydro"):
                key, value = ("P_hydro_k", np.inf) if defect == "inf-P_hydro" else ("sigma_P_hydro_k", np.nan)
                del f["_meta"][key]
                f["_meta"].create_dataset(key, data=value)
        app = lrdmc_app if command == "lrdmc" else mcmc_app
        result = CliRunner().invoke(app, ["compute-force", str(checkpoint_dir), *args])
        assert result.exit_code != 0, result.output
        assert "Atomic Forces:" not in result.output
        assert "P_hydro =" not in result.output
        assert message in _flatten_cli_output(str(result.exception) + " " + result.output)

    @pytest.mark.light
    def test_mcmc_compute_force_basic(self, tmp_path):
        """Smoke test: compute-force should print atomic force table."""
        chk_path = str(tmp_path / "mcmc.h5")
        obj = _make_mcmc_obj(
            num_steps=100,
            num_walkers=4,
            num_atoms=2,
            num_elec_up=1,
            num_elec_dn=1,
            atomic_labels=["H", "H"],
            with_force=True,
        )
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "2", "-w", "0"])
        assert result.exit_code == 0
        assert "Atomic Forces:" in result.output
        assert "H" in result.output
        assert "Fx" in result.output

    @pytest.mark.light
    def test_mcmc_compute_force_multi_rank(self, tmp_path):
        """Force computation with multiple MPI ranks."""
        chk_path = str(tmp_path / "mcmc.h5")
        objs = [
            _make_mcmc_obj(
                num_steps=100,
                num_walkers=2,
                num_atoms=2,
                num_elec_up=1,
                num_elec_dn=1,
                atomic_labels=["H", "H"],
                with_force=True,
                rng=np.random.default_rng(i),
            )
            for i in range(2)
        ]
        _write_chk(chk_path, objs)

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "1", "-w", "0"])
        assert result.exit_code == 0
        assert "Found 2 MPI ranks" in result.output
        assert "Atomic Forces:" in result.output

    @pytest.mark.light
    def test_mcmc_compute_force_without_deriv_raises(self, tmp_path):
        """If comput_position_deriv is False, compute-force should fail."""
        chk_path = str(tmp_path / "mcmc.h5")
        obj = _make_mcmc_obj(num_steps=100, num_walkers=4, with_force=False)
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "1", "-w", "0"])
        assert result.exit_code != 0

    @pytest.mark.light
    def test_lrdmc_compute_force_basic(self, tmp_path):
        """Smoke test for LRDMC force computation."""
        chk_path = str(tmp_path / "lrdmc.h5")
        obj = _make_lrdmc_obj(
            num_steps=100,
            num_walkers=4,
            num_atoms=2,
            num_elec_up=1,
            num_elec_dn=1,
            atomic_labels=["N", "N"],
            with_force=True,
        )
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-force", chk_path, "-b", "10", "-w", "25", "-c", "5"])
        assert result.exit_code == 0
        assert "Atomic Forces:" in result.output
        assert "N" in result.output

    @pytest.mark.light
    def test_lrdmc_compute_force_without_deriv_raises(self, tmp_path):
        """If comput_position_deriv is False, LRDMC compute-force should fail."""
        chk_path = str(tmp_path / "lrdmc.h5")
        obj = _make_lrdmc_obj(num_steps=100, num_walkers=4, with_force=False)
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        runner = CliRunner()
        result = runner.invoke(lrdmc_app, ["compute-force", chk_path, "-b", "10", "-w", "25", "-c", "5"])
        assert result.exit_code != 0
        assert "atomic_force was not enabled" in result.output

    @pytest.mark.light
    def test_mcmc_force_constant_values(self, tmp_path):
        """With zero random derivatives, all forces should be near zero."""
        chk_path = str(tmp_path / "mcmc.h5")
        obj = _make_mcmc_obj(
            num_steps=100,
            num_walkers=4,
            num_atoms=2,
            num_elec_up=1,
            num_elec_dn=1,
            atomic_labels=["H", "H"],
            with_force=True,
        )
        # Set force product arrays to zero
        for attr in [
            "force_HF",
            "force_PP",
            "E_L_force_PP",
        ]:
            setattr(obj, attr, np.zeros_like(getattr(obj, attr)))
        _write_chk(chk_path, [obj])

        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        runner = CliRunner()
        result = runner.invoke(mcmc_app, ["compute-force", chk_path, "-b", "2", "-w", "0"])
        assert result.exit_code == 0
        # With all derivatives zero, forces should be 0+/-0
        assert "Atomic Forces:" in result.output


def _random_force_obj(num_steps, num_walkers, seed, *, lrdmc=False):
    """Synthetic force checkpoint rank with random e_L, positive weights and force products.

    Unlike ``_make_mcmc_obj`` the local energy and the weights fluctuate, so the
    Pulay covariance term and the weighting both contribute to the estimate.
    """
    rng = np.random.default_rng(seed)
    num_atoms = 2
    e_L = -1.0 + 0.1 * rng.standard_normal((num_steps, num_walkers))
    w_L = rng.uniform(0.5, 1.5, (num_steps, num_walkers))
    force_HF = rng.standard_normal((num_steps, num_walkers, num_atoms, 3))
    force_PP = rng.standard_normal((num_steps, num_walkers, num_atoms, 3))
    obj = SimpleNamespace(
        e_L=e_L,
        w_L=w_L,
        force_HF=force_HF,
        force_PP=force_PP,
        E_L_force_PP=np.einsum("iw,iwjk->iwjk", e_L, force_PP),
        comput_position_deriv=True,
        hamiltonian_data=SimpleNamespace(structure_data=SimpleNamespace(atomic_labels=["H", "Li"])),
    )
    if lrdmc:
        obj.alat = 0.3
        obj.num_gfmc_collect_steps = 5
    return obj


def _reference_force_jackknife(objs, *, warmup, bins, collect=None):
    """Delete-one-replica force reference written from the estimator's definition.

    A replica is one (rank, walker, time bin) block; time bins follow the
    documented equal-split convention (the first ``n % bins`` bins hold one more
    measurement). For every deleted replica the weighted means over the other
    replicas give F = -<F_HF> - 2 (<E F_PP> - <E><F_PP>); the reported mean is the
    replica average and the error sqrt(M - 1) times the replica standard
    deviation. LRDMC weights are the XLA historical products of
    ``jqmc_gfmc._compute_G_L`` (separately checked; not the checkpoint accumulator).
    """
    import jax.numpy as jnp

    from jqmc.jqmc_gfmc import _compute_G_L

    replicas = []
    for obj in objs:
        if collect is None:
            sl = slice(warmup, None)
            w_L = np.asarray(obj.w_L)[sl]
            shift = 0
        else:
            w_L = np.asarray(_compute_G_L(jnp.asarray(obj.w_L), collect))[warmup:]
            shift = collect
        e_L = np.asarray(obj.e_L)[shift:][warmup:]
        f_hf = np.asarray(obj.force_HF)[shift:][warmup:]
        f_pp = np.asarray(obj.force_PP)[shift:][warmup:]
        ef_pp = np.asarray(obj.E_L_force_PP)[shift:][warmup:]
        n = e_L.shape[0]
        edges = np.concatenate([[0], np.cumsum([n // bins + (1 if b < n % bins else 0) for b in range(bins)])])
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            for walker in range(e_L.shape[1]):
                w = w_L[lo:hi, walker]
                replicas.append(
                    (
                        w.sum(),
                        (w * e_L[lo:hi, walker]).sum(),
                        (w[:, None, None] * f_hf[lo:hi, walker]).sum(axis=0),
                        (w[:, None, None] * f_pp[lo:hi, walker]).sum(axis=0),
                        (w[:, None, None] * ef_pp[lo:hi, walker]).sum(axis=0),
                    )
                )
    samples = []
    for m in range(len(replicas)):
        kept = [r for i, r in enumerate(replicas) if i != m]
        W = sum(r[0] for r in kept)
        mean_e = sum(r[1] for r in kept) / W
        mean_hf = sum(r[2] for r in kept) / W
        mean_pp = sum(r[3] for r in kept) / W
        mean_epp = sum(r[4] for r in kept) / W
        samples.append(np.real(-mean_hf - 2.0 * (mean_epp - mean_e * mean_pp)))
    samples = np.asarray(samples)
    M = samples.shape[0]
    return samples.mean(axis=0), np.sqrt(M - 1) * samples.std(axis=0)


def _assert_force_refused(result, message):
    """A refused force request exits non-zero, names the reason and prints no force table."""
    output = _flatten_cli_output(result.output)
    assert result.exit_code != 0, result.output
    assert "Atomic Forces:" not in result.output
    assert "nan(" not in result.output.lower()
    assert message in output, result.output


def _inject_force_defect(obj, defect):
    """Put one abnormal sample (or weight set) inside the analysed window.

    Row -3 is analysed by both drivers: it is past every warmup used below and,
    for LRDMC, its raw weight enters the historical products of later rows.
    """
    if defect == "zero-weights":
        obj.w_L = np.zeros_like(obj.w_L)
    elif defect == "nan-weight":
        obj.w_L[-3, 0] = np.nan
    elif defect == "nan-e_L":
        obj.e_L[-3, 0] = np.nan
    elif defect == "nan-force_HF":
        obj.force_HF[-3, 0, 0, 0] = np.nan
    elif defect == "inf-force_PP":
        obj.force_PP[-3, 0, 1, 2] = np.inf
    else:
        raise AssertionError(defect)


class TestComputeForceRefusals:
    """Single-checkpoint compute-force refuses settings and samples that leave the force undefined.

    The force jackknife has no defined value when a bin is empty, a delete-one
    weight sum is not positive, or a sample in the analysed window is not
    finite; the commands must fail with an explicit reason and print no table,
    as compute-energy does. Settings at the boundary still give the reference
    force.
    """

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("num_walkers", "args", "message"),
        [
            (2, ["-b", "4", "-w", "40"], "fewer than num_mcmc_bin_blocks"),
            (2, ["-b", "50", "-w", "0"], "fewer than num_mcmc_bin_blocks"),
            (2, ["-b", "0", "-w", "0"], "must be >= 1"),
            (2, ["-b", "4", "-w", "-1"], "must be >= 0"),
            (1, ["-b", "1", "-w", "0"], "at least 2 binned samples"),
        ],
        ids=["warmup-eats-all", "too-many-bins", "zero-bins", "negative-warmup", "single-replica"],
    )
    def test_mcmc_compute_force_rejects_vacuous_binning(self, tmp_path, num_walkers, args, message):
        """Empty bins or a single replica are usage errors, not a NaN force table."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        chk = str(tmp_path / "mcmc.h5")
        _write_chk(chk, [_random_force_obj(40, num_walkers, 7)])
        result = CliRunner().invoke(mcmc_app, ["compute-force", chk, *args])
        _assert_force_refused(result, message)

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("defect", "message"),
        [
            ("zero-weights", "delete-one-bin weight sum"),
            ("nan-weight", "delete-one-bin weight sum"),
            ("nan-e_L", "non-finite atomic force estimate"),
            ("nan-force_HF", "non-finite atomic force estimate"),
            ("inf-force_PP", "non-finite atomic force estimate"),
        ],
        ids=["zero-weights", "nan-weight", "nan-e_L", "nan-force_HF", "inf-force_PP"],
    )
    def test_mcmc_compute_force_rejects_invalid_samples(self, tmp_path, defect, message):
        """Zero or non-finite weights and non-finite samples are refused, never printed as forces."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import mcmc_app

        obj = _random_force_obj(40, 2, 11)
        _inject_force_defect(obj, defect)
        chk = str(tmp_path / "mcmc.h5")
        _write_chk(chk, [obj])
        result = CliRunner().invoke(mcmc_app, ["compute-force", chk, "-b", "10", "-w", "30"])
        _assert_force_refused(result, message)

    @pytest.mark.light
    @pytest.mark.parametrize(
        "args",
        [
            ["-b", "10", "-w", "40", "-c", "5"],
            ["-b", "10", "-w", "25", "-c", "40"],
            ["-b", "100", "-w", "25", "-c", "5"],
        ],
        ids=["warmup-eats-all", "collect-eats-all", "too-many-bins"],
    )
    def test_lrdmc_compute_force_rejects_vacuous_binning(self, tmp_path, args):
        """LRDMC settings that leave empty bins are usage errors, not a NaN force table."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        chk = str(tmp_path / "lrdmc.h5")
        _write_chk(chk, [_random_force_obj(40, 2, 13, lrdmc=True)])
        result = CliRunner().invoke(lrdmc_app, ["compute-force", chk, *args])
        _assert_force_refused(result, "too few measurements after collect/warmup")

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("defect", "message"),
        [
            ("zero-weights", "delete-one-bin weight sum"),
            ("nan-weight", "delete-one-bin weight sum"),
            ("nan-e_L", "non-finite atomic force estimate"),
            ("nan-force_HF", "non-finite atomic force estimate"),
            ("inf-force_PP", "non-finite atomic force estimate"),
        ],
        ids=["zero-weights", "nan-weight", "nan-e_L", "nan-force_HF", "inf-force_PP"],
    )
    def test_lrdmc_compute_force_rejects_invalid_samples(self, tmp_path, defect, message):
        """LRDMC: zero or non-finite weights and non-finite samples are refused."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        obj = _random_force_obj(50, 2, 17, lrdmc=True)
        _inject_force_defect(obj, defect)
        chk = str(tmp_path / "lrdmc.h5")
        _write_chk(chk, [obj])
        result = CliRunner().invoke(lrdmc_app, ["compute-force", chk, "-b", "10", "-w", "25", "-c", "5"])
        _assert_force_refused(result, message)

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("command", "num_steps", "num_ranks", "args"),
        [
            ("mcmc", 40, 1, {"-b": 10, "-w": 30}),
            ("mcmc", 43, 2, {"-b": 10, "-w": 30}),
            ("lrdmc", 40, 1, {"-b": 10, "-w": 25, "-c": 5}),
            ("lrdmc", 47, 1, {"-b": 10, "-w": 25, "-c": 7}),
        ],
        ids=["mcmc-one-per-bin", "mcmc-uneven-two-ranks", "lrdmc-one-per-bin", "lrdmc-uneven"],
    )
    def test_compute_force_boundary_settings_match_reference(self, tmp_path, command, num_steps, num_ranks, args):
        """Valid control: the smallest accepted window still prints the reference force table.

        One measurement per bin (and an uneven split) is accepted with exit 0
        and the printed rows equal the independent delete-one-replica reference
        in the specified two-significant-digit uncertainty format.
        """
        from typer.testing import CliRunner
        from uncertainties import ufloat

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        lrdmc = command == "lrdmc"
        objs = [_random_force_obj(num_steps, 2, 23 + rank, lrdmc=lrdmc) for rank in range(num_ranks)]
        chk = str(tmp_path / f"{command}.h5")
        _write_chk(chk, objs)
        cli_args = [item for key, value in args.items() for item in (key, str(value))]
        app = lrdmc_app if lrdmc else mcmc_app
        result = CliRunner().invoke(app, ["compute-force", chk, *cli_args])
        assert result.exit_code == 0, result.output
        assert "Atomic Forces:" in result.output
        mean, std = _reference_force_jackknife(objs, warmup=args["-w"], bins=args["-b"], collect=args.get("-c"))
        assert np.isfinite(mean).all() and np.all(std > 0)
        labels = ["H", "Li"]
        rows = [line.split() for line in result.output.splitlines() if line.split() and line.split()[0] in labels]
        expected_rows = [
            [label, *(f"{ufloat(mean[i, j], std[i, j]):+2uS}" for j in range(3))] for i, label in enumerate(labels)
        ]
        assert rows == expected_rows

    @staticmethod
    def _lrdmc_minima_route(tmp_path, route):
        """Command line (without -b/-w/-c) and source objects of one LRDMC analysis route.

        Every input leaves nonempty bins at the documented minima -b 10 -w 25 -c 5
        and also at the below-minimum settings, so only the minima can refuse them.
        """
        if route == "compute-force-twist":
            checkpoint_dir = tmp_path / "restart.h5"
            objs = TestComputeForce._write_force_twist_dir(checkpoint_dir, "lrdmc")
            return ["compute-force", str(checkpoint_dir)], objs
        if route == "extrapolate-energy":
            paths, objs = [], []
            for idx, alat in enumerate((0.5, 0.3)):
                rng = np.random.default_rng(700 + idx)
                obj = _make_lrdmc_obj(num_steps=100, num_walkers=4, alat=alat, with_force=False)
                obj.e_L = rng.standard_normal((100, 4)) - 5.0 - 0.2 * idx
                obj.w_L = rng.uniform(0.9, 1.1, (100, 4))
                path = str(tmp_path / f"lrdmc_{idx}.h5")
                _write_chk(path, [obj])
                paths.append(path)
                objs.append(obj)
            return ["extrapolate-energy", *paths, "-p", "1"], objs
        obj = _random_force_obj(40, 2, 29, lrdmc=True)
        chk = str(tmp_path / "lrdmc.h5")
        _write_chk(chk, [obj])
        return [route, chk], [obj]

    @pytest.mark.light
    @pytest.mark.parametrize("route", ["compute-energy", "compute-force", "compute-force-twist", "extrapolate-energy"])
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["-b", "10", "-w", "24", "-c", "5"], "num_gfmc_warmup_steps (24) must be >= 25"),
            (["-b", "9", "-w", "25", "-c", "5"], "num_gfmc_bin_blocks (9) must be >= 10"),
            (["-b", "10", "-w", "25", "-c", "4"], "num_gfmc_collect_steps (4) must be >= 5"),
            (["-b", "3", "-w", "2", "-c", "2"], "num_gfmc_warmup_steps (2) must be >= 25"),
            (["-w", "0"], "num_gfmc_warmup_steps (0) must be >= 25"),
        ],
        ids=["warmup-24", "bins-9", "collect-4", "b3-w2-c2", "w0"],
    )
    def test_lrdmc_analysis_below_documented_minima_is_refused(self, tmp_path, route, args, message):
        """Every LRDMC analysis command refuses settings below the documented minima.

        doc/api_reference_cli_tool.md: the LRDMC commands reject warm-up below
        25, fewer than 10 bins and a weight history shorter than 5, so an
        explicit -w 0 is rejected (the jqmc lrdmc-bra/lrdmc-tau input
        minima). compute-force (checkpoint file and twist directory) and
        extrapolate-energy must refuse exactly as compute-energy does: exit
        status 1, the violated minimum named, and no estimate printed. The
        inputs keep nonempty bins at these settings, so no other check refuses.
        """
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        command, _objs = self._lrdmc_minima_route(tmp_path, route)
        result = CliRunner().invoke(lrdmc_app, [*command, *args])
        assert result.exit_code == 1, result.output
        assert message in _flatten_cli_output(result.output), result.output
        for estimate in ("Atomic Forces:", "P_hydro =", "E =", "For a", "Extrapolation is finished"):
            assert estimate not in result.output, result.output

    @pytest.mark.light
    @pytest.mark.parametrize("route", ["compute-energy", "compute-force", "compute-force-twist", "extrapolate-energy"])
    def test_lrdmc_analysis_at_documented_minima_matches_reference(self, tmp_path, route):
        """Valid control: -b 10 -w 25 -c 5 is accepted and prints the independent reference.

        Force rows equal the delete-one-replica force jackknife written from the
        estimator definition (BZ-weighted per twist for a directory); energies
        equal the separately written LRDMC energy jackknife, and the two-spacing
        a -> 0 value is the closed-form linear intercept
        E0 = (x2 E1 - x1 E2) / (x2 - x1), x = a^2, with
        sigma0 = sqrt((x2 sigma1)^2 + (x1 sigma2)^2) / |x2 - x1|.
        """
        from typer.testing import CliRunner
        from uncertainties import ufloat

        from jqmc.jqmc_tool import lrdmc_app

        command, objs = self._lrdmc_minima_route(tmp_path, route)
        result = CliRunner().invoke(lrdmc_app, [*command, "-b", "10", "-w", "25", "-c", "5"])
        assert result.exit_code == 0, result.output + str(result.exception)
        atol, rtol = get_tolerance("local_energy", "strict")
        if route in ("compute-energy", "extrapolate-energy"):
            reference = np.asarray([_reference_lrdmc_jackknife([{"e_L": obj.e_L, "w_L": obj.w_L}], 25, 10, 5) for obj in objs])
            pattern = r"^E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha"
            if route == "extrapolate-energy":
                pattern = r"For a = [\d.eE+-]+ bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha"
            points = np.asarray(re.findall(pattern, result.output, re.MULTILINE), dtype=float)
            assert points.shape == reference.shape, result.output
            np.testing.assert_allclose(points, reference, atol=atol, rtol=rtol)
            if route == "extrapolate-energy":
                (e1, s1), (e2, s2) = reference
                x1, x2 = 0.5**2, 0.3**2
                expected = [(x2 * e1 - x1 * e2) / (x2 - x1), np.hypot(x2 * s1, x1 * s2) / abs(x2 - x1)]
                match = re.search(r"For a -> 0 bohr: E = ([\d.eE+-]+) \+- ([\d.eE+-]+) Ha", result.output)
                assert match is not None, result.output
                np.testing.assert_allclose(np.asarray(match.groups(), dtype=float), expected, atol=atol, rtol=rtol)
            return
        if route == "compute-force-twist":
            expected_rows = TestComputeForce._bz_reference_rows(objs, warmup=25, bins=10, collect=5)
        else:
            mean, std = _reference_force_jackknife(objs, warmup=25, bins=10, collect=5)
            assert np.isfinite(mean).all() and np.all(std > 0)
            expected_rows = [
                [label, *(f"{ufloat(mean[i, j], std[i, j]):+2uS}" for j in range(3))] for i, label in enumerate(["H", "Li"])
            ]
        rows = [line.split() for line in result.output.splitlines() if line.split() and line.split()[0] in ("H", "Li")]
        assert rows == expected_rows


class TestComputeDriverMismatch:
    """mcmc and lrdmc compute-* refuse a checkpoint written by the other driver (issue #588)."""

    @pytest.mark.light
    @pytest.mark.parametrize("layout", ["file", "twist-directory"])
    @pytest.mark.parametrize("command", ["compute-energy", "compute-force"])
    @pytest.mark.parametrize(("app_name", "stored_driver"), [("mcmc", "GFMC_t"), ("mcmc", "GFMC_n"), ("lrdmc", "MCMC")])
    def test_wrong_driver_checkpoint_is_refused(self, tmp_path, app_name, stored_driver, command, layout):
        """The other driver's checkpoint exits 1 naming its driver and the right command, without a result."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        if layout == "file":
            restart_chk = tmp_path / "restart.h5"
            obj = _random_force_obj(40, 2, 23, lrdmc=stored_driver != "MCMC")
            _write_chk(str(restart_chk), [obj], driver_type=stored_driver)
        else:
            restart_chk = tmp_path / "restart_dir"
            pressure = (np.array([0.01, 0.03]), np.array([0.001, 0.002]))
            TestComputeForce._write_twist_dir(
                restart_chk, np.array([0.25, 0.75]), pressure=pressure, force=False, driver_type=stored_driver
            )
        app, args, other_app = (
            (mcmc_app, ["-b", "10", "-w", "30"], "lrdmc")
            if app_name == "mcmc"
            else (lrdmc_app, ["-b", "10", "-w", "25", "-c", "5"], "mcmc")
        )
        result = CliRunner().invoke(app, [command, str(restart_chk), *args])
        output = _flatten_cli_output(result.output)
        assert result.exit_code == 1, result.output
        assert f"written by the {stored_driver} driver" in output, result.output
        assert f"Use `jqmc-tool {other_app} {command}` instead." in output, result.output
        for result_marker in ("E = ", "Atomic Forces:", "P_hydro = ", "Total number of binned samples"):
            assert result_marker not in result.output, result.output


#: Frozen sampler settings a real twist-averaged run stores in every twist's rank driver_config.
_TWIST_RUN_SAMPLER = {
    "MCMC": {
        "num_walkers": 2,
        "Dt": 2.0,
        "epsilon_AS": 0.0,
        "num_mcmc_per_measurement": 4,
        "use_swct": True,
        "random_discretized_mesh": True,
        "mpc": False,
    },
    "GFMC_t": {
        "num_walkers": 2,
        "tau": 0.1,
        "alat": 0.3,
        "non_local_move": "tmove",
        "num_gfmc_collect_steps": 5,
        "use_swct": True,
        "epsilon_PW": 0.0,
        "random_discretized_mesh": True,
        "mpc": False,
    },
}


def _scale_first_float_dataset(group, factor=1.5):
    """Change one stored floating-point array below ``group`` (a modified physics parameter)."""
    names = []
    group.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f" else None)
    dataset = group[next(name for name in names if group[name].size > 0)]
    dataset[...] = factor * dataset[()]


@pytest.fixture(scope="module")
def twist_run_hamiltonians(tmp_path_factory):
    """A real H2 Hamiltonian and the same system with a different Jastrow (a re-optimized run)."""
    from jqmc.hamiltonians import Hamiltonian_data

    out = {}
    for label, j2 in (("reference", 1.0), ("other_jastrow", 0.5)):
        path = tmp_path_factory.mktemp("twist_ham") / f"{label}.h5"
        trexio_convert_to(
            trexio_file=str(Path(_TREXIO_DIR) / "H2_ecp_ccpvtz_cart.h5"),
            hamiltonian_file=str(path),
            j1_parmeter=None,
            j2_parmeter=j2,
            j3_basis_type="ao-small",
        )
        out[label] = Hamiltonian_data.load_from_hdf5(str(path))
    return out


class TestTwistDirectoryConsistency:
    """compute-energy/compute-force refuse a twist directory assembled from different runs (issue #861)."""

    @staticmethod
    def _write_twist_run(directory, twist_run_hamiltonians, driver_type, *, gamma_k_point=False, alat=None):
        """Two twists of one run: twist 0 is Gamma (no stored k_point unless requested), twist 1 is k=(0.5,0,0).

        Both twists share structure, ECP, Jastrow and sampler settings; their orbitals,
        digests and seeds differ, as for the twists of one real twist-averaged run.
        """
        from jqmc.hamiltonians import _save_dataclass_to_hdf5

        directory.mkdir()
        sampler = dict(_TWIST_RUN_SAMPLER[driver_type])
        if alat is not None:
            sampler["alat"] = alat
        rank_data = []
        for twist_idx, k_point in enumerate(((0.0, 0.0, 0.0) if gamma_k_point else None, (0.5, 0.0, 0.0))):
            obj = _random_force_obj(40, 2, 861 + twist_idx, lrdmc=driver_type != "MCMC")
            path = directory / f"twist_{twist_idx:03d}.h5"
            _write_chk(str(path), [obj], driver_type=driver_type)
            with h5py.File(path, "a") as f:
                del f["hamiltonian_data"]
                _save_dataclass_to_hdf5(f.create_group("hamiltonian_data"), twist_run_hamiltonians["reference"])
                if twist_idx:
                    _scale_first_float_dataset(f["hamiltonian_data/wavefunction_data/geminal_data"])
                meta = f["_meta"].attrs
                meta["twist_index"] = twist_idx
                meta["converged"] = True
                meta["normalized_twist_weight"] = (0.25, 0.75)[twist_idx]
                meta["num_twists"] = 2
                meta["ham_digest"] = f"digest-of-twist-{twist_idx}"
                if k_point is not None:
                    meta["k_point"] = np.asarray(k_point)
                cfg = f["rank_0/driver_config"].attrs
                for key, value in sampler.items():
                    cfg[key] = value
                cfg["mcmc_seed"] = 12345 + twist_idx
            rank_data.append([{"e_L": obj.e_L, "w_L": obj.w_L}])
        return rank_data

    @staticmethod
    def _apply_defect(directory, twist_run_hamiltonians, defect):
        """Replace twist 1 by data of another run, as a hand-assembled directory would."""
        from jqmc.hamiltonians import _save_dataclass_to_hdf5

        path = directory / "twist_001.h5"
        if defect == "duplicate_twist":
            path.write_bytes((directory / "twist_000.h5").read_bytes())
            with h5py.File(path, "a") as f:
                f["_meta"].attrs["twist_index"] = 1
            return
        with h5py.File(path, "a") as f:
            meta = f["_meta"].attrs
            cfg = f["rank_0/driver_config"].attrs
            if defect == "jastrow":
                del f["hamiltonian_data"]
                _save_dataclass_to_hdf5(f.create_group("hamiltonian_data"), twist_run_hamiltonians["other_jastrow"])
            elif defect == "structure":
                _scale_first_float_dataset(f["hamiltonian_data/structure_data"], factor=1.01)
            elif defect == "ecp":
                _scale_first_float_dataset(f["hamiltonian_data/coulomb_potential_data"])
            elif defect == "duplicate_k_point":
                meta["k_point"] = np.zeros(3)
            elif defect == "driver_type":
                meta["driver_type"] = "GFMC_n"
            else:
                cfg[defect] = {"num_walkers": 4, "Dt": 1.0, "tau": 0.3, "alat": 0.2, "non_local_move": "dltmove"}[defect]

    @pytest.mark.light
    @pytest.mark.parametrize("command", ["compute-energy", "compute-force"])
    @pytest.mark.parametrize(
        ("driver_type", "defect", "diagnostic"),
        [
            ("MCMC", "jastrow", "hamiltonian_data/wavefunction_data/jastrow_data"),
            ("MCMC", "structure", "hamiltonian_data/structure_data"),
            ("MCMC", "ecp", "hamiltonian_data/coulomb_potential_data"),
            ("MCMC", "num_walkers", "driver_config num_walkers"),
            ("MCMC", "Dt", "driver_config Dt"),
            ("MCMC", "duplicate_twist", "duplicate k_point"),
            ("MCMC", "duplicate_k_point", "duplicate k_point"),
            ("GFMC_t", "jastrow", "hamiltonian_data/wavefunction_data/jastrow_data"),
            ("GFMC_t", "tau", "driver_config tau"),
            ("GFMC_t", "alat", "driver_config alat"),
            ("GFMC_t", "non_local_move", "driver_config non_local_move"),
            ("GFMC_t", "driver_type", "_meta/driver_type"),
            ("GFMC_t", "duplicate_twist", "duplicate k_point"),
        ],
    )
    def test_twists_from_different_runs_are_refused(
        self, tmp_path, twist_run_hamiltonians, driver_type, defect, diagnostic, command
    ):
        """A twist replaced by another run's data exits 1 naming the mismatch, and prints no result."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        directory = tmp_path / "restart_dir"
        self._write_twist_run(directory, twist_run_hamiltonians, driver_type)
        self._apply_defect(directory, twist_run_hamiltonians, defect)
        app, args = (
            (mcmc_app, ["-b", "4", "-w", "0"]) if driver_type == "MCMC" else (lrdmc_app, ["-b", "10", "-w", "25", "-c", "5"])
        )
        result = CliRunner().invoke(app, [command, str(directory), *args])
        output = _flatten_cli_output(result.output)
        assert result.exit_code == 1, result.output
        assert "not from one twist-averaged run" in output, result.output
        assert diagnostic in output, result.output
        for result_marker in ("E = ", "Atomic Forces:", "P_hydro = ", "Found 2 twist checkpoints"):
            assert result_marker not in result.output, result.output

    @pytest.mark.light
    @pytest.mark.parametrize("driver_type", ["MCMC", "GFMC_t"])
    def test_twists_of_one_run_give_the_weighted_twist_average(self, tmp_path, twist_run_hamiltonians, driver_type):
        """Per-twist orbitals, digests and seeds may differ; the BZ energy equals the weighted reference."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app, mcmc_app

        directory = tmp_path / "restart_dir"
        rank_data = self._write_twist_run(directory, twist_run_hamiltonians, driver_type)
        if driver_type == "MCMC":
            app, args = mcmc_app, ["-b", "4", "-w", "0"]
            estimates = [_reference_mcmc_jackknife(data, 0, 4) for data in rank_data]
        else:
            app, args = lrdmc_app, ["-b", "10", "-w", "25", "-c", "5"]
            estimates = [_reference_lrdmc_jackknife(data, 25, 10, 5) for data in rank_data]
        weights = np.array([0.25, 0.75])
        expected = (
            float(np.dot(weights, [mean for mean, _std in estimates])),
            float(np.sqrt(np.dot(weights**2, np.square([std for _mean, std in estimates])))),
        )
        result = CliRunner().invoke(app, ["compute-energy", str(directory), *args])
        assert result.exit_code == 0, result.output
        match = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", result.output)
        assert match is not None, result.output
        atol, rtol = get_tolerance("local_energy", "strict")
        np.testing.assert_allclose(np.asarray(match.groups(), dtype=float), expected, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_extrapolation_refuses_a_directory_mixing_tau(self, tmp_path, twist_run_hamiltonians):
        """extrapolate-energy applies the same check to every lattice-spacing directory."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import lrdmc_app

        directories = []
        for alat in (0.2, 0.3):
            directory = tmp_path / f"alat_{alat}"
            self._write_twist_run(directory, twist_run_hamiltonians, "GFMC_t", gamma_k_point=True, alat=alat)
            directories.append(str(directory))
        args = ["extrapolate-energy", *directories, "-p", "1", "-b", "10", "-w", "25", "-c", "5"]
        control = CliRunner().invoke(lrdmc_app, args)
        assert control.exit_code == 0, control.output + str(control.exception)
        assert "For a -> 0 bohr: E = " in control.output
        self._apply_defect(Path(directories[1]), twist_run_hamiltonians, "tau")
        result = CliRunner().invoke(lrdmc_app, args)
        assert result.exit_code != 0
        assert isinstance(result.exception.__context__, BadParameter)
        assert "driver_config tau" in str(result.exception.__context__)
        assert "not from one twist-averaged run" in str(result.exception.__context__)
        assert "For a -> 0" not in result.output


class TestHamiltonianCommands:
    """Tests for hamiltonian show-info and to-xyz commands."""

    @pytest.fixture(params=trexio_files, ids=trexio_files)
    def h5_file(self, tmp_path, request):
        """Create a Hamiltonian and return its path and TREXIO source path."""
        trexio_file = Path(_TREXIO_DIR) / request.param
        h5_path = str(tmp_path / "hamiltonian_data.h5")
        trexio_convert_to(
            trexio_file=str(trexio_file),
            hamiltonian_file=h5_path,
            j1_parmeter=None,
            j2_parmeter=1.0,
            j3_basis_type="ao-small",
        )
        return h5_path, trexio_file

    @pytest.mark.light
    def test_hamiltonian_show_info(self, h5_file, capsys):
        """show-info should print without error."""
        h5_path, _ = h5_file
        hamiltonian_show_info(hamiltonian_data=h5_path)
        captured = capsys.readouterr()
        # It should print *something* about the hamiltonian
        assert len(captured.out) > 0

    @pytest.mark.light
    def test_hamiltonian_to_xyz(self, h5_file, tmp_path):
        """to-xyz preserves the TREXIO geometry and writes coordinates in angstroms."""
        h5_path, trexio_file = h5_file
        xyz_path = tmp_path / "struct.xyz"
        hamiltonian_to_xyz(hamiltonian_data=h5_path, xyz_file=str(xyz_path))
        assert_xyz_matches_trexio(xyz_path, trexio_file)


class TestTrexioShowCommands:
    """Tests for trexio show-info and show-detail commands."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_filename", trexio_files, ids=trexio_files)
    def test_trexio_show_info(self, trexio_filename, capsys):
        """show-info should produce output without error."""
        trexio_show_info(filename=os.path.join(_TREXIO_DIR, trexio_filename))
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_filename", trexio_files, ids=trexio_files)
    def test_trexio_show_detail(self, trexio_filename, capsys):
        """show-detail should produce output without error."""
        trexio_show_detail(filename=os.path.join(_TREXIO_DIR, trexio_filename))
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestVmcAnalyzeOutput:
    """Tests for vmc analyze-output command."""

    _SAMPLE_LOG = (
        "Optimization step =   1/10\n"
        "E = -1.17 +- 0.05 Ha\n"
        "Max generalized force = 0.123 +- 0.045 Ha/a.u.\n"
        "Max generalized force signal-to-noise = 2.733.\n"
        "Optimization step =   2/10\n"
        "E = -1.18 +- 0.04 Ha\n"
        "Max generalized force = 0.100 +- 0.030 Ha/a.u.\n"
        "Max generalized force signal-to-noise = 3.333.\n"
    )

    def _write_log(self, path, text=None):
        Path(path).write_text(text or self._SAMPLE_LOG)

    @pytest.mark.light
    def test_analyze_output_table(self, tmp_path, capsys):
        """analyze-output should print a table with Iter / E / generalized force."""
        log_path = str(tmp_path / "vmc.log")
        self._write_log(log_path)
        vmc_analyze_output(filenames=[log_path], plot_graph=False, save_graph=None)
        captured = capsys.readouterr()
        assert "Iter" in captured.out
        assert "E (Ha)" in captured.out
        # Check that both iterations appear
        assert "1" in captured.out
        assert "2" in captured.out

    @pytest.mark.light
    def test_analyze_output_save_graph(self, tmp_path, capsys):
        """save-graph should create an image file."""
        log_path = str(tmp_path / "vmc.log")
        self._write_log(log_path)
        graph_path = str(tmp_path / "result.png")
        vmc_analyze_output(filenames=[log_path], plot_graph=False, save_graph=graph_path)
        assert os.path.exists(graph_path)

    @pytest.mark.light
    def test_analyze_output_multiple_files(self, tmp_path, capsys):
        """analyze-output should combine data from multiple files."""
        log1 = str(tmp_path / "vmc1.log")
        log2 = str(tmp_path / "vmc2.log")

        log_text_1 = (
            "Optimization step =   1/5\n"
            "E = -1.17 +- 0.05 Ha\n"
            "Max generalized force = 0.123 +- 0.045 Ha/a.u.\n"
            "Max generalized force signal-to-noise = 2.733.\n"
        )
        log_text_2 = (
            "Optimization step =   2/5\n"
            "E = -1.18 +- 0.04 Ha\n"
            "Max generalized force = 0.100 +- 0.030 Ha/a.u.\n"
            "Max generalized force signal-to-noise = 3.333.\n"
        )

        self._write_log(log1, log_text_1)
        self._write_log(log2, log_text_2)
        vmc_analyze_output(filenames=[log1, log2], plot_graph=False, save_graph=None)
        captured = capsys.readouterr()
        # Both iterations should appear in the combined output
        assert "1" in captured.out
        assert "2" in captured.out

    # Optimizer log blocks as ``jqmc_optimizer`` prints them. A diverged step prints ``E = nan +- nan Ha`` and
    # "Non-finite energy detected" and stops before its force lines; a non-finite force is printed as ``nan``.
    @staticmethod
    def _step_block(step, num_steps, energy=None, force=None, snr=None):
        lines = ["=" * 66, f"Optimization step = {step}/{num_steps}.", "=" * 66]
        if energy is not None:
            lines.append(f"E = {energy} Ha")
        if energy is not None and energy.startswith("nan"):
            lines.append(f"Non-finite energy detected (E={energy.split()[0]}). Rejecting the optimization step.")
        if force is not None:
            lines.append(f"Max generalized force = {force} Ha/a.u.")
        if snr is not None:
            lines.append(f"Max generalized force signal-to-noise = {snr}.")
        return "\n".join(lines) + "\n"

    def _healthy_log(self, first_bad=None, bad_block=None):
        blocks = [
            self._step_block(1, 4, "-1.31530 +- 0.17781", "0.011 +- 0.020", "0.536"),
            self._step_block(2, 4, "-1.18696 +- 0.07474", "0.0049 +- 0.0099", "0.488"),
            self._step_block(3, 4, "-1.25052 +- 0.11307", "-0.008 +- 0.012", "0.654"),
            self._step_block(4, 4, "-1.02254 +- 0.08718", "-0.0089 +- 0.0023", "3.909"),
        ]
        if first_bad is not None:
            blocks[first_bad - 1] = bad_block
        return "".join(blocks)

    # Rows of the healthy log above, written from its literal values (value(error) with two error digits).
    _HEALTHY_ROWS = {
        1: "1 -1.32(18) +0.011(20) 0.536",
        2: "2 -1.187(75) +0.0049(99) 0.488",
        3: "3 -1.25(11) -0.008(12) 0.654",
        4: "4 -1.023(87) -0.0089(23) 3.909",
    }

    @staticmethod
    def _table_rows(stdout):
        """Rows between the second and third dashed rule, whitespace-normalised."""
        lines = stdout.splitlines()
        rules = [i for i, line in enumerate(lines) if line and set(line) == {"-"}]
        assert len(rules) == 3, stdout
        return [" ".join(line.split()) for line in lines[rules[1] + 1 : rules[2]]]

    @pytest.mark.light
    @pytest.mark.parametrize("case", ["nan-force-step-2", "nan-energy-last-step", "diverged-restart"])
    def test_analyze_output_reports_nonfinite_steps_aligned(self, tmp_path, monkeypatch, case):
        """A non-finite step keeps its row and alignment and the command exits non-zero without a graph."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import vmc_app

        if case == "nan-force-step-2":
            bad = self._step_block(2, 4, "-1.18696 +- 0.07474", "nan +- nan", "0.488")
            files = {"vmc.out": self._healthy_log(2, bad)}
            expected = [self._HEALTHY_ROWS[1], "2 -1.187(75) +nan(nan) 0.488", self._HEALTHY_ROWS[3], self._HEALTHY_ROWS[4]]
            bad_steps = ["2 (vmc.out)"]
        elif case == "nan-energy-last-step":
            files = {"vmc.out": self._healthy_log(4, self._step_block(4, 4, "nan +- nan"))}
            expected = [self._HEALTHY_ROWS[1], self._HEALTHY_ROWS[2], self._HEALTHY_ROWS[3], "4 nan(nan) - -"]
            bad_steps = ["4 (vmc.out)"]
        else:
            files = {"vmc.out": self._healthy_log(), "vmc_cont.out": self._step_block(5, 8, "nan +- nan")}
            expected = [*self._HEALTHY_ROWS.values(), "5 nan(nan) - -"]
            bad_steps = ["5 (vmc_cont.out)"]
        paths = []
        for name, text in files.items():
            (tmp_path / name).write_text(text)
            paths.append(name)
        graph = tmp_path / "vmc.png"

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(vmc_app, ["analyze-output", *paths, "-s", str(graph)])

        assert result.exit_code != 0
        assert self._table_rows(result.stdout) == expected
        assert f"non-finite energy or generalized force in optimization step(s) {', '.join(bad_steps)}" in result.stderr
        assert not graph.exists()

    @pytest.mark.light
    @pytest.mark.parametrize("case", ["empty-file", "mcmc-log"])
    def test_analyze_output_refuses_input_without_steps(self, tmp_path, case):
        """A file without any optimization step is an error, not an empty table."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import vmc_app

        log = tmp_path / "vmc.out"
        log.write_text("" if case == "empty-file" else "E = -1.17 +- 0.05 Ha.\nVar(E) = 0.1 +- 0.01 Ha^2.\n")
        result = CliRunner().invoke(vmc_app, ["analyze-output", str(log)])
        assert result.exit_code != 0
        assert "Iter" not in result.stdout
        assert f"no VMC optimization step was found in {log}" in result.stderr

    @pytest.mark.light
    def test_analyze_output_keeps_exponent_and_unfinished_steps_aligned(self, tmp_path):
        """An exponent-notation energy stays on its step, and an unfinished last step is shown and reported."""
        from typer.testing import CliRunner

        from jqmc.jqmc_tool import vmc_app

        exponent = self._step_block(3, 4, "-1.25052e+00 +- 1.1307e-01", "-0.008 +- 0.012", "0.654")
        text = self._healthy_log(3, exponent).replace(
            self._step_block(4, 4, "-1.02254 +- 0.08718", "-0.0089 +- 0.0023", "3.909"), self._step_block(4, 4)
        )
        log = tmp_path / "vmc.out"
        log.write_text(text)
        result = CliRunner().invoke(vmc_app, ["analyze-output", str(log)])
        assert result.exit_code == 0
        assert self._table_rows(result.stdout) == [
            self._HEALTHY_ROWS[1],
            self._HEALTHY_ROWS[2],
            self._HEALTHY_ROWS[3],
            "4 - - -",
        ]
        assert f"(the step did not finish): 4 ({log})" in result.stderr


# ---------------------------------------------------------------------------
# QE convert-to (Quantum ESPRESSO -> hamiltonian_data.h5)
# ---------------------------------------------------------------------------

_QE_SAVE_DIR = qe_example_files_root() / "Li_gamma" / "out" / "li.save"
_QE_XML = _QE_SAVE_DIR / "data-file-schema.xml"
_QE_WFC = _QE_SAVE_DIR / "wfc1.hdf5"
_LI_UPF = _QE_SAVE_DIR / "Li.ccECP.upf"

_qe_available = _QE_XML.is_file() and _QE_WFC.is_file() and _LI_UPF.is_file()


@pytest.mark.skipif(not _qe_available, reason="QE Li_gamma / ccECP UPF sample not present")
class TestQEConvertTo:
    """Tests for ``jqmc-tool qe convert-to`` (QE -> hamiltonian_data.h5)."""

    @pytest.mark.light
    def test_cli_twist_average_flag_reaches_converter(self, monkeypatch):
        """The Typer surface forwards ``--twist-average`` to run_qe2jqmc."""
        from click.testing import CliRunner

        from jqmc.jqmc_tool import typer_click_qe

        captured = {}

        def _fake_run_qe2jqmc(*, qe_params, jastrow_params, hamiltonian_h5):
            captured["qe_params"] = qe_params
            captured["jastrow_params"] = jastrow_params
            captured["hamiltonian_h5"] = hamiltonian_h5

        monkeypatch.setattr("jqmc.jqmc_qe.run_qe2jqmc", _fake_run_qe2jqmc)
        result = CliRunner().invoke(
            typer_click_qe,
            [
                str(_QE_SAVE_DIR),
                "-k",
                "bspline",
                "--twist-average",
                "--jastrow-dtn",
                "u_chi_f",
                "--dtn-Nu",
                "6",
                "--dtn-Nee",
                "2",
                "-o",
                "twists.h5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["qe_params"]["twist_average"] is True
        assert captured["qe_params"]["kind"] == "bspline"
        assert captured["jastrow_params"]["jastrow_dtn"] == "u_chi_f"
        assert captured["jastrow_params"]["dtn_n_u"] == 6
        assert captured["jastrow_params"]["dtn_n_ee"] == 2
        assert captured["hamiltonian_h5"] == "twists.h5"

    @pytest.mark.light
    def test_cli_tiling_matrix_reaches_converter(self, monkeypatch):
        """Repeated ``--tiling`` values are forwarded in row/diagonal order."""
        from click.testing import CliRunner

        from jqmc.jqmc_tool import typer_click_qe

        captured = {}

        def _fake_run_qe2jqmc(*, qe_params, jastrow_params, hamiltonian_h5):
            captured["qe_params"] = qe_params

        monkeypatch.setattr("jqmc.jqmc_qe.run_qe2jqmc", _fake_run_qe2jqmc)
        result = CliRunner().invoke(
            typer_click_qe,
            [
                str(_QE_SAVE_DIR),
                "-k",
                "bspline",
                "--tiling",
                "2",
                "--tiling",
                "1",
                "--tiling",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["qe_params"]["tiling"] == [2, 1, 1]

    @pytest.mark.light
    def test_cli_dtn_cutoff_ratio_defaults_are_the_shared_constant(self, monkeypatch):
        """``jqmc-tool qe`` exposes 0.95 by default and honours explicit ratios."""
        from click.testing import CliRunner

        from jqmc.jastrow_factor import DTN_DEFAULT_CUTOFF_RATIO
        from jqmc.jqmc_tool import typer_click_qe

        assert DTN_DEFAULT_CUTOFF_RATIO == 0.95

        captured = {}

        def _fake_run_qe2jqmc(*, qe_params, jastrow_params, hamiltonian_h5):
            captured["jastrow_params"] = jastrow_params

        monkeypatch.setattr("jqmc.jqmc_qe.run_qe2jqmc", _fake_run_qe2jqmc)

        base_args = [str(_QE_SAVE_DIR), "-k", "pw", "--jastrow-dtn", "u_chi_f", "-o", "h.h5"]
        result = CliRunner().invoke(typer_click_qe, base_args)
        assert result.exit_code == 0, result.output
        for key in ("dtn_cutoff_ratio_u", "dtn_cutoff_ratio_chi", "dtn_cutoff_ratio_f"):
            assert captured["jastrow_params"][key] == 0.95

        # An explicit ratio must still win over the default.
        result = CliRunner().invoke(
            typer_click_qe,
            base_args
            + [
                "--dtn-cutoff-ratio-u",
                "0.75",
                "--dtn-cutoff-ratio-chi",
                "0.75",
                "--dtn-cutoff-ratio-f",
                "0.75",
            ],
        )
        assert result.exit_code == 0, result.output
        for key in ("dtn_cutoff_ratio_u", "dtn_cutoff_ratio_chi", "dtn_cutoff_ratio_f"):
            assert captured["jastrow_params"][key] == 0.75

    @pytest.mark.light
    def test_qe_toml_ratio_fallback_is_the_shared_constant(self):
        """An omitted ``[jastrow]`` ratio falls back to 0.95, not 0.75."""
        import inspect

        from jqmc.jastrow_factor import DTN_DEFAULT_CUTOFF_RATIO
        from jqmc.jqmc_qe import _build_jastrow_dtn

        signature = inspect.signature(_build_jastrow_dtn)
        for name in ("dtn_cutoff_ratio_u", "dtn_cutoff_ratio_chi", "dtn_cutoff_ratio_f"):
            assert signature.parameters[name].default == DTN_DEFAULT_CUTOFF_RATIO == 0.95

    def _run_qe_convert(self, hamiltonian_file, upf_paths):
        """Invoke ``qe_convert_to`` with the lightweight pw projection.

        ``kind='pw'`` is the exact, grid-free plane-wave projection -- the
        cheapest path that still exercises the full QE -> projection -> jqmc
        -> ECP -> Hamiltonian pipeline. The analytic ECP block comes from the
        UPF's ``[Gaussinfo]`` section regardless of the projection ``kind``.
        All typer Option arguments are passed explicitly because the command
        is invoked as a plain function (no CLI parsing applies the Option
        defaults).
        """
        qe_convert_to(
            qe_save_dir=str(_QE_SAVE_DIR),
            kind="pw",
            upf_paths=upf_paths,
            mesh_factor=1.0,
            spline_order=5,
            ecutwfc=None,
            alpha_min=None,
            max_cond_number=None,
            hamiltonian_file=hamiltonian_file,
            j1_parameter=None,
            j1_type="exp",
            j2_parameter=1.0,
            j2_type="pade",
            ewald_tol=1.0e-8,
            ecp_z_cores=None,
        )

    @pytest.mark.light
    def test_qe_convert_to_with_ccecp(self, tmp_path):
        """QE -> Hamiltonian with the analytic ECP read from UPF ``[Gaussinfo]``.

        Li.ccECP removes 2 core electrons and has l_max = 0 (a single
        non-local s channel), so each atom's attached ECP block carries
        z_core = 2, max_ang_mom_plus_1 = l_max + 1 = 1, and
        ``parse_gaussinfo(...).num_ecps`` terms. The bcc-Li fixture has 2
        atoms, so the merged ``coulomb_potential_data`` has 2 entries per
        per-atom array.
        """
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.upf_wrapper import parse_gaussinfo

        hamiltonian_file = os.path.join(tmp_path, "hamiltonian_data.h5")
        self._run_qe_convert(hamiltonian_file, [f"Li={_LI_UPF}"])

        assert os.path.isfile(hamiltonian_file)
        ham = Hamiltonian_data.load_from_hdf5(hamiltonian_file)
        cpd = ham.coulomb_potential_data

        n_atom = int(ham.structure_data.natom)
        assert cpd.ecp_flag is True
        # Per-atom z_core / max_ang_mom_plus_1 entries (one per atom of Li).
        assert tuple(cpd.z_cores) == (2.0,) * n_atom
        assert tuple(cpd.max_ang_mom_plus_1) == (1,) * n_atom

        expected = parse_gaussinfo(str(_LI_UPF))
        # The same ECP block is attached to every Li atom.
        assert cpd.num_ecps == expected.num_ecps * n_atom
        assert tuple(cpd.ang_moms) == expected.ang_moms * n_atom
        assert tuple(cpd.powers) == expected.powers * n_atom
        _atol_h5, _rtol_h5 = get_tol_hdf5_roundtrip("strict")
        np.testing.assert_allclose(np.asarray(cpd.exponents), np.tile(expected.exponents, n_atom), rtol=_rtol_h5, atol=_atol_h5)
        np.testing.assert_allclose(
            np.asarray(cpd.coefficients), np.tile(expected.coefficients, n_atom), rtol=_rtol_h5, atol=_atol_h5
        )
        # The first ``num_ecps`` ECP terms attach to atom 0, the next to atom 1, ...
        expected_nuclei = []
        for ia in range(n_atom):
            expected_nuclei.extend([ia] * expected.num_ecps)
        assert tuple(cpd.nucleus_index) == tuple(expected_nuclei)


@pytest.mark.light
def test_generated_lrdmc_input_survives_restart_against_an_evolved_checkpoint(tmp_path):
    """The GENERATED lrdmc input must still be a valid CONTINUATION input.

    The documented journey edits ``restart = true`` in the very file the
    generator wrote. ``E_scf`` is frozen on restart AND evolves during the run,
    so emitting the ``0.0`` placeholder made the frozen-key guard refuse that
    continuation outright -- the advertised fresh -> restart journey could not
    be completed at all. The guard is right; the generator was
    wrong. This pins the generated file against a checkpoint whose ``E_scf``
    has moved, which is what every real run produces.
    """
    import toml as _toml

    from jqmc._restart_policy import assert_restart_frozen_consistency

    path = os.path.join(tmp_path, "lrdmc_input.toml")
    lrdmc_generate_input(flag=True, filename=path, exclude_comment=True)
    dict_toml = _toml.load(path)
    # The user replaces the required placeholders before the fresh run.
    assert dict_toml["lrdmc-bra"]["num_mcmc_per_measurement"] == "None"
    dict_toml["lrdmc-bra"]["num_mcmc_steps"] = 100
    dict_toml["lrdmc-bra"]["num_mcmc_per_measurement"] = 40

    assert "E_scf" not in dict_toml["lrdmc-bra"], (
        "the generated continuation input carries a placeholder E_scf; a real checkpoint's "
        "E_scf has evolved away from it and the frozen-key guard will refuse the restart"
    )
    # A checkpoint from a finished run stores every frozen key (the writer
    # always does; an absent one is rejected): the generated values, except
    # E_scf, which has moved away from the 0.0 default.
    from jqmc._restart_policy import TOML_TO_CFG

    checkpoint_driver_config = {
        cfg_key: dict_toml[section][key]
        for (section, key), cfg_key in TOML_TO_CFG.items()
        if section in ("control", "lrdmc-bra", "coulomb") and key in dict_toml.get(section, {})
    }
    checkpoint_driver_config["E_scf"] = -1.10389
    assert_restart_frozen_consistency("lrdmc-bra", dict_toml, checkpoint_driver_config)


# ---------------------------------------------------------------------------
# LRDMC accumulated weights beyond the float64 range (issue #816)
# ---------------------------------------------------------------------------

_OVERFLOW_TAU = 0.1
# The command-line minima (25 warmup windows, 10 bins); 43 retained windows give unequal bins.
_OVERFLOW_WARMUP = 25
_OVERFLOW_RETAINED = 43
_OVERFLOW_BINS = 10


def _overflow_lrdmc_objs(collect, *, num_ranks=2, num_walkers=2):
    """Two-rank LRDMC checkpoint data of a ~-100 Ha molecule with unshifted weights exp(-tau e).

    Every raw weight is about e^10 (tau = 0.1), so a window of ``collect`` steps
    multiplies to about e^(10 collect). Each rank shifts its energies so that
    the two ranks' window weights differ by several orders of magnitude.
    """
    objs = []
    num_steps = collect + _OVERFLOW_WARMUP + _OVERFLOW_RETAINED
    for rank in range(num_ranks):
        rng = np.random.default_rng(816 + 10 * rank + collect)
        e_L = -100.0 - 0.4 * rank + 0.5 * rng.standard_normal((num_steps, num_walkers))
        w_L = np.exp(-_OVERFLOW_TAU * (e_L + 0.3 * rng.standard_normal((num_steps, num_walkers))))
        force_PP = rng.standard_normal((num_steps, num_walkers, 2, 3))
        obj = SimpleNamespace(
            e_L=e_L,
            w_L=w_L,
            force_HF=rng.standard_normal((num_steps, num_walkers, 2, 3)),
            force_PP=force_PP,
            E_L_force_PP=np.einsum("iw,iwjk->iwjk", e_L, force_PP) + rng.standard_normal((num_steps, num_walkers, 2, 3)),
            comput_position_deriv=True,
            hamiltonian_data=SimpleNamespace(structure_data=SimpleNamespace(atomic_labels=["H", "Li"])),
            alat=0.3,
            num_gfmc_collect_steps=collect,
        )
        objs.append(obj)
    return objs


def _lrdmc_reference_from_window_weights(objs, window_weights, *, warmup, bins, collect):
    """Delete-one-replica LRDMC energy and force from the estimator definitions.

    ``window_weights[r]`` holds the retained window weight of every step and
    walker of rank ``r`` (one common scale for all ranks). A replica is one
    (rank, time bin, walker) block, time bins follow the documented equal split.
    For each deleted replica the weighted means over the others give E = <e>
    and F = -<F_HF> - 2(<e F_PP> - <e><F_PP>); the reported value is the replica
    mean and the error sqrt(M - 1) times the replica standard deviation.
    """
    replicas = []
    for obj, weights in zip(objs, window_weights, strict=True):
        moments = [
            np.asarray(getattr(obj, name))[collect:][warmup:] for name in ("e_L", "force_HF", "force_PP", "E_L_force_PP")
        ]
        n = weights.shape[0]
        edges = np.concatenate([[0], np.cumsum([n // bins + (1 if b < n % bins else 0) for b in range(bins)])])
        for b in range(bins):
            for walker in range(weights.shape[1]):
                g = weights[edges[b] : edges[b + 1], walker]
                replicas.append([g.sum()] + [np.tensordot(g, m[edges[b] : edges[b + 1], walker], axes=(0, 0)) for m in moments])
    energies, forces = [], []
    for deleted in range(len(replicas)):
        kept = [replica for index, replica in enumerate(replicas) if index != deleted]
        total = sum(replica[0] for replica in kept)
        e_mean, hf_mean, pp_mean, epp_mean = (sum(replica[slot] for replica in kept) / total for slot in range(1, 5))
        energies.append(e_mean)
        forces.append(-hf_mean - 2.0 * (epp_mean - e_mean * pp_mean))
    energies, forces = np.asarray(energies), np.asarray(forces)
    M = energies.shape[0]
    return (
        (np.mean(energies), np.sqrt(M - 1) * np.std(energies)),
        (np.mean(forces, axis=0), np.sqrt(M - 1) * np.std(forces, axis=0)),
    )


def _log_space_window_weights(objs, collect, warmup):
    """Retained window weights exp(log G - max log G), one maximum over every rank and walker.

    ``log G`` is the math.fsum of the window's raw log-weights; no product is formed.
    """
    import math

    log_products = []
    for obj in objs:
        log_w = np.log(obj.w_L)
        steps, walkers = log_w.shape
        log_products.append(
            np.asarray(
                [[math.fsum(log_w[j : j + collect, walker]) for walker in range(walkers)] for j in range(steps - collect)]
            )[warmup:]
        )
    largest = max(float(np.max(values)) for values in log_products)
    return [np.exp(values - largest) for values in log_products], largest


def _run_lrdmc_tool(tmp_path, objs, collect):
    """Run lrdmc compute-energy, single-file compute-force and the twist force estimator on one checkpoint."""
    from typer.testing import CliRunner

    from jqmc.jqmc_tool import _twist_force_estimate, lrdmc_app

    chk = str(tmp_path / f"lrdmc_c{collect}.h5")
    _write_chk(chk, objs)
    options = ["-b", str(_OVERFLOW_BINS), "-w", str(_OVERFLOW_WARMUP), "-c", str(collect)]
    energy_result = CliRunner().invoke(lrdmc_app, ["compute-energy", chk, *options])
    force_result = CliRunner().invoke(lrdmc_app, ["compute-force", chk, *options])
    assert energy_result.exit_code == 0, energy_result.output
    assert force_result.exit_code == 0, force_result.output
    match = re.search(r"E\s*=\s*([+-]?[\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", energy_result.output)
    assert match is not None, energy_result.output
    force_rows = [line.split() for line in force_result.output.splitlines() if line.split() and line.split()[0] in ("H", "Li")]
    twist_force = _twist_force_estimate(
        chk, num_bin_blocks=_OVERFLOW_BINS, num_warmup_steps=_OVERFLOW_WARMUP, num_collect_steps=collect
    )
    return (float(match.group(1)), float(match.group(2))), force_rows, twist_force


def _assert_lrdmc_tool_matches(actual, expected):
    from uncertainties import ufloat

    (energy, force_rows, twist_force), (expected_energy, expected_force) = actual, expected
    atol, rtol = get_tolerance("local_energy", "strict")
    np.testing.assert_allclose(energy, expected_energy, atol=atol, rtol=rtol, equal_nan=False)
    for actual_value, expected_value in zip(twist_force, expected_force, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, atol=atol, rtol=rtol, equal_nan=False)
    mean, std = expected_force
    expected_rows = [
        [label, *(f"{ufloat(mean[i, j], std[i, j]):+2uS}" for j in range(3))] for i, label in enumerate(["H", "Li"])
    ]
    assert force_rows == expected_rows


@pytest.mark.light
@pytest.mark.parametrize("collect", [100, 200])
def test_lrdmc_tool_matches_log_space_reference_beyond_float64_products(tmp_path, collect):
    """lrdmc compute-energy/compute-force equal a log-space reference when window products exceed float64.

    Two-rank, two-walker merged checkpoint of a ~-100 Ha molecule (unshifted
    weights exp(-0.1 e), about e^10 per step) analysed with -c 100 and -c 200,
    25 warmup windows and 10 bins: the unscaled window products are about
    e^1000 and e^2000, beyond float64 (the input check states it). The reference
    sums each window's raw log-weights with math.fsum, subtracts the largest over
    both ranks and walkers and evaluates the delete-one-replica definitions. The
    printed energy and error, the twist force estimator (mean and error arrays)
    use the local_energy strict policy of the current mode; the single-file force
    table must print the reference rows in its two-significant-digit format.
    No statistical band.
    """
    objs = _overflow_lrdmc_objs(collect)
    window_weights, largest_log_product = _log_space_window_weights(objs, collect, _OVERFLOW_WARMUP)
    assert largest_log_product > np.log(np.finfo(np.float64).max)
    expected = _lrdmc_reference_from_window_weights(
        objs, window_weights, warmup=_OVERFLOW_WARMUP, bins=_OVERFLOW_BINS, collect=collect
    )
    _assert_lrdmc_tool_matches(_run_lrdmc_tool(tmp_path, objs, collect), expected)


@pytest.mark.light
@pytest.mark.parametrize("collect", [5, 10])
def test_lrdmc_tool_equals_unscaled_products_for_short_windows(tmp_path, collect):
    """For representable window products the tool estimates equal those of the unscaled products.

    Same two-rank ~-100 Ha checkpoint analysed with -c 5 (the command-line minimum)
    and -c 10 (products about e^50 and e^100). The reference weights are the explicit np.prod of the
    preceding raw weights of each rank and walker (the explicit-product definition),
    in the same delete-one-replica reference; local_energy strict policy.
    """
    objs = _overflow_lrdmc_objs(collect)
    window_weights = [
        np.asarray([np.prod(obj.w_L[j : j + collect], axis=0) for j in range(obj.w_L.shape[0] - collect)])[_OVERFLOW_WARMUP:]
        for obj in objs
    ]
    assert all(np.all(np.isfinite(weights)) for weights in window_weights)
    expected = _lrdmc_reference_from_window_weights(
        objs, window_weights, warmup=_OVERFLOW_WARMUP, bins=_OVERFLOW_BINS, collect=collect
    )
    _assert_lrdmc_tool_matches(_run_lrdmc_tool(tmp_path, objs, collect), expected)


def _whole_array_force_estimate(chk, *, warmup, bins, collect, jackknife):
    """The force estimator as it was before block streaming, on fully loaded arrays.

    Every observable of every rank is read whole, each force product is formed
    with one ``np.einsum`` over the full window and binned with
    ``np.array_split``. ``jackknife`` selects the delete-one form of the
    single-file commands (``"file"``) or of the twist estimator (``"twist"``),
    whose operation orders differ, so each route has its own exact reference.
    """
    from jqmc._checkpoint import compute_retained_accumulated_weights, load_observables_from_checkpoint

    force_keys = ("force_HF", "force_PP", "E_L_force_PP")
    ranks = [obs for obs in load_observables_from_checkpoint(chk) if obs.get("e_L", np.empty(0)).size != 0]
    if collect is None:
        shift, weights = 0, [obs["w_L"][warmup:] for obs in ranks]
    else:
        shift, weights = collect, compute_retained_accumulated_weights([obs["w_L"] for obs in ranks], collect, warmup)

    def binned(values):
        sums = np.asarray([np.sum(arr, axis=0) for arr in np.array_split(values, bins, axis=0)])
        return sums.reshape((-1, *sums.shape[2:]))

    parts = [[] for _ in range(5)]
    for obs, w_L in zip(ranks, weights, strict=True):
        parts[0].append(binned(w_L))
        parts[1].append(binned(w_L * obs["e_L"][shift:][warmup:]))
        for slot, key in zip((2, 3, 4), force_keys, strict=True):
            parts[slot].append(binned(np.einsum("iw,iwjk->iwjk", w_L, obs[key][shift:][warmup:])))
    w, we, whf, wpp, wepp = (np.concatenate(part, axis=0) for part in parts)
    M = w.size
    if jackknife == "file":
        hf_jn = -1.0 * np.array([(np.sum(whf, axis=0) - whf[m]) / (np.sum(w) - w[m]) for m in range(M)])
        pulay_jn = -2.0 * np.array(
            [
                (np.sum(wepp, axis=0) - wepp[m]) / (np.sum(w) - w[m])
                - ((np.sum(we) - we[m]) / (np.sum(w) - w[m]) * (np.sum(wpp, axis=0) - wpp[m]) / (np.sum(w) - w[m]))
                for m in range(M)
            ]
        )
        force_jn = np.real(hf_jn + pulay_jn)
    else:
        loo_w = (np.sum(w) - w)[:, None, None]
        mean_e = (np.sum(we) - we)[:, None, None] / loo_w
        mean_hf, mean_pp, mean_epp = ((np.sum(x, axis=0) - x) / loo_w for x in (whf, wpp, wepp))
        force_jn = np.real(-mean_hf - 2.0 * (mean_epp - mean_e * mean_pp))
    return np.average(force_jn, axis=0), np.sqrt(M - 1) * np.std(force_jn, axis=0)


@pytest.mark.light
@pytest.mark.parametrize("route", ["mcmc-two-ranks", "lrdmc", "lrdmc-twist"])
def test_force_postprocessing_streamed_bins_equal_whole_array_estimator(tmp_path, monkeypatch, route):
    """Force post-processing that reads per-atom observables one bin block at a time equals the whole-array estimator.

    The force mean and error arrays of mcmc compute-force (two ranks, -b 10 -w 7:
    36 analysed rows, uneven bins), lrdmc compute-force and the twist force
    estimator (one contributing rank plus an empty rank as a merged LRDMC
    checkpoint stores them, -b 10 -w 25 -c 7: 15 analysed rows, uneven bins,
    shifted by the collect length) are captured unrounded at the force check
    and must equal, bit for bit, the pre-streaming estimator evaluated on the
    fully loaded arrays. A block mapped to the wrong dataset rows changes the
    sums and fails the exact comparison.
    """
    from typer.testing import CliRunner

    from jqmc import jqmc_tool

    chk = str(tmp_path / "restart.h5")
    if route == "mcmc-two-ranks":
        _write_chk(chk, [_random_force_obj(43, 2, 949 + rank) for rank in range(2)])
        settings = {"warmup": 7, "bins": 10, "collect": None}
    else:
        empty_rank = SimpleNamespace(e_L=np.empty((0, 2)), w_L=np.empty((0, 2)), comput_position_deriv=True, alat=0.3)
        _write_chk(chk, [_random_force_obj(47, 2, 949, lrdmc=True), empty_rank])
        settings = {"warmup": 25, "bins": 10, "collect": 7}

    if route == "lrdmc-twist":
        actual = jqmc_tool._twist_force_estimate(
            chk,
            num_bin_blocks=settings["bins"],
            num_warmup_steps=settings["warmup"],
            num_collect_steps=settings["collect"],
        )
    else:
        captured = []
        original = jqmc_tool._check_force_result

        def capture(force_mean, force_std, *, source):
            captured.append((force_mean, force_std))
            original(force_mean, force_std, source=source)

        monkeypatch.setattr(jqmc_tool, "_check_force_result", capture)
        app, extra = (jqmc_tool.mcmc_app, []) if route == "mcmc-two-ranks" else (jqmc_tool.lrdmc_app, ["-c", "7"])
        cli_args = ["-b", str(settings["bins"]), "-w", str(settings["warmup"]), *extra]
        result = CliRunner().invoke(app, ["compute-force", chk, *cli_args])
        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        actual = captured[0]
    expected = _whole_array_force_estimate(chk, **settings, jackknife="twist" if route == "lrdmc-twist" else "file")
    assert np.all(expected[1] > 0)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.light
@pytest.mark.parametrize("command", ["compute-energy", "compute-force", "extrapolate-energy"])
def test_lrdmc_analysis_defaults_pass_the_command_floors(command):
    """The LRDMC analysis commands run without -b/-w/-c: their defaults pass the commands' own floor check."""
    from jqmc.jqmc_tool import _check_lrdmc_analysis_floors, typer_click_lrdmc

    defaults = {p.name: p.default for p in typer_click_lrdmc.commands[command].params}
    _check_lrdmc_analysis_floors(
        num_gfmc_bin_blocks=defaults["num_gfmc_bin_block"],
        num_gfmc_warmup_steps=defaults["num_gfmc_warmup_steps"],
        num_gfmc_collect_steps=defaults["num_gfmc_collect_steps"],
    )
