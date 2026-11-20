"""Hamiltonians whose spins are not an opposite-twist (+k, -k) pair are refused.

Every complex Bloch input pairs the spin channels at opposite twists: the
down-spin channel is a Bloch basis at ``-k`` when the up channel is at
``+k``. For an independent (LSDA / UHF) down channel that basis is beta's own
and need not equal alpha's. A down channel at the same twist -- the
``(+k, +k)`` layout, whether with beta's coefficients or their
conjugates -- and a periodic-Gaussian ``+k`` leaf merely relabelled ``-k``
(image phases of the wrong twist) must fail ``sanity_check``, on construction
and when a stored Hamiltonian is loaded, before any sampling. Independent-spin
twist-averaged inputs without the required ``/_meta/spin_twist_pairing``
attribute are refused by ``load_twist``.
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

import h5py
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from tests._test_data_cache import qe_example_files_root

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_ROOT = Path(__file__).resolve().parents[1]
_TREXIO = _ROOT / "trexio_example_files" / "oxygen_tiling_sectors_sphe.h5"
_QE = qe_example_files_root() / "O-lsda_k025" / "out" / "o_k025.save"
# A beta plane-wave cutoff below the full sphere: beta gets its own, smaller basis.
_BETA_G_CUT = 5.7


def _relabelled(leaf):
    """The leaf (or composite) with only its k label negated."""
    from jqmc.atomic_orbital import AOs_composite_data

    if isinstance(leaf, AOs_composite_data):
        return leaf.replace(children=tuple(_relabelled(child) for child in leaf.children))
    k_point = getattr(leaf, "k_point", None)
    return leaf if k_point is None else leaf.replace(k_point=tuple(-float(c) for c in k_point))


def _geminal(kind, beta_g_cut=None):
    """An independent-spin SD at a general twist from a committed source."""
    if kind == "pgto":
        from jqmc.trexio_wrapper import read_trexio_file

        _, _, _, _, geminal, _ = read_trexio_file(str(_TREXIO), store_tuple=True, twist_ik=2)
        return geminal
    from jqmc.pw_pgto_projection import project_pw_to_pw, project_pw_to_spline
    from jqmc.qe_to_jqmc import build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference

    up = load_qe_reference(str(_QE), spin="up")
    dn = load_qe_reference(str(_QE), spin="dw")
    if kind == "pw":
        projections = (project_pw_to_pw(up, occ_only=False), project_pw_to_pw(dn, g_cut=beta_g_cut, occ_only=False))
    else:
        projections = (project_pw_to_spline(up), project_pw_to_spline(dn))
    _, _, geminal = build_wavefunction_from_projection(
        up, projections[0], kind=kind, qe_reference_dn=dn, projection_result_dn=projections[1]
    )
    return geminal


def _with_down(geminal, construction):
    """Rebuild the down channel from beta's own +k leaf (the with_negated_k partner of the stored -k leaf)."""
    from jqmc.determinant import SD_data

    dn = geminal.orb_data_dn_spin
    beta_plus_k = dn.aos_data.with_negated_k()
    stored = np.asarray(dn.mo_coefficients)  # conj(C_beta)
    if construction == "same_twist":  # same-twist layout: beta at +k
        new = dn.replace(aos_data=beta_plus_k, mo_coefficients=np.conj(stored))
    elif construction == "conj_coefficients_only":  # conj(C_beta) left on the +k leaf
        new = dn.replace(aos_data=beta_plus_k)
    elif construction == "relabelled_leaf":  # +k leaf carrying only a -k label
        new = dn.replace(aos_data=_relabelled(beta_plus_k))
    else:
        raise ValueError(construction)
    return SD_data(
        num_electron_up=geminal.num_electron_up,
        num_electron_dn=geminal.num_electron_dn,
        orb_data_up_spin=geminal.orb_data_up_spin,
        orb_data_dn_spin=new,
        time_reversal=geminal.time_reversal,
    )


@pytest.mark.parametrize(
    "kind,construction",
    [
        ("pgto", "same_twist"),
        ("pgto", "conj_coefficients_only"),
        ("pgto", "relabelled_leaf"),
        ("pw", "same_twist"),
        ("pw", "conj_coefficients_only"),
        ("bspline", "same_twist"),
        ("bspline", "conj_coefficients_only"),
    ],
)
def test_sanity_check_refuses_non_opposite_spin_twists(kind, construction, tmp_path):
    """Same-twist, conjugated-coefficients-only and relabelled down channels fail sanity_check.

    The converter output passes. A relabelled leaf is a violation only for
    the periodic Gaussian, whose image phases belong to a definite twist; a
    plane-wave or spline leaf is a Bloch basis of whatever twist it carries.
    For the periodic Gaussian the valid Hamiltonian is written, refused by
    ``save_to_hdf5`` in broken form, and its stored down-leaf twist is then
    overwritten with the up twist -- a same-twist layout, whose image phases are rebuilt from the stored twist on load:
    ``Hamiltonian_data.load_from_hdf5`` must raise the regeneration error.
    """
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.wavefunction import Wavefunction_data

    geminal = _geminal(kind)
    assert not geminal.time_reversal
    geminal.sanity_check()
    broken = _with_down(geminal, construction)
    with pytest.raises(ValueError, match="opposite twist"):
        broken.sanity_check()
    if kind != "pgto" or construction != "same_twist":
        return
    from jqmc.trexio_wrapper import read_trexio_file

    structure, _, _, _, _, coulomb = read_trexio_file(str(_TREXIO), store_tuple=True, twist_ik=2)

    def hamiltonian(geminal_data):
        wavefunction = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_data)
        return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)

    path = tmp_path / "hamiltonian.h5"
    with pytest.raises(ValueError, match="opposite twist"):
        hamiltonian(broken).save_to_hdf5(str(path))
    hamiltonian(geminal).save_to_hdf5(str(path))
    Hamiltonian_data.load_from_hdf5(str(path))
    leaves = "wavefunction_data/geminal_data/orb_data_{}_spin/aos_data/k_point"
    with h5py.File(path, "r+") as f:
        f[leaves.format("dn")][...] = f[leaves.format("up")][...]
    with pytest.raises(ValueError, match="regenerate"):
        Hamiltonian_data.load_from_hdf5(str(path))


@pytest.mark.parametrize("layout", ["smaller_cutoff", "reordered_rows"])
def test_independent_beta_plane_wave_basis_is_accepted(layout):
    """An opposite-twist down channel on beta's own plane-wave basis passes.

    Beta is projected with a smaller cutoff than alpha (a different PW basis
    size), or beta's rows and coefficient columns are permuted. The
    converter's down channel -- conj(C_beta) on beta's with_negated_k() leaf
    -- must pass the pairing check and evaluate to the conjugate of beta's own
    +k orbitals.
    """
    from jqmc.determinant import SD_data
    from jqmc.molecular_orbital import compute_MOs

    geminal = _geminal("pw", beta_g_cut=_BETA_G_CUT if layout == "smaller_cutoff" else None)
    up_leaf = geminal.orb_data_up_spin.aos_data
    dn = geminal.orb_data_dn_spin
    if layout == "smaller_cutoff":
        assert dn.aos_data.num_ao < up_leaf.num_ao
    else:
        order = np.random.default_rng(7).permutation(dn.aos_data.num_ao)
        g = np.asarray(dn.aos_data._g_indices).reshape(-1, 3)[order]
        dn = dn.replace(
            aos_data=dn.aos_data.replace(_g_indices=tuple(int(m) for m in g.ravel())),
            mo_coefficients=np.asarray(dn.mo_coefficients)[:, order],
        )
        geminal = SD_data.build(
            num_electron_up=geminal.num_electron_up,
            num_electron_dn=geminal.num_electron_dn,
            mos_up=geminal.orb_data_up_spin,
            mos_dn=dn,
            time_reversal=False,
        )
    geminal.sanity_check()
    beta_plus_k = dn.replace(aos_data=dn.aos_data.with_negated_k(), mo_coefficients=np.conj(np.asarray(dn.mo_coefficients)))
    r = np.random.default_rng(5).uniform(0.0, 10.0, size=(6, 3))
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval"), "strict")
    np.testing.assert_allclose(
        np.asarray(compute_MOs(dn, r)), np.conj(np.asarray(compute_MOs(beta_plus_k, r))), atol=atol, rtol=rtol
    )


def test_load_twist_refuses_unmarked_independent_spin_input(tmp_path):
    """An independent-spin twist input without the required spin-twist pairing attribute is refused at a complex twist."""
    from click.testing import CliRunner

    from jqmc._twist_average import SPIN_TWIST_PAIRING_ATTR, load_twist
    from jqmc.jqmc_tool import _cli

    out = tmp_path / "twist.h5"
    result = CliRunner().invoke(_cli, ["trexio", "convert-to", str(_TREXIO), "-o", str(out), "--twist-average"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert not load_twist(str(out), 2).wavefunction_data.geminal_data.time_reversal
    with h5py.File(out, "r+") as f:
        del f["_meta"].attrs[SPIN_TWIST_PAIRING_ATTR]
    with pytest.raises(ValueError, match="Regenerate"):
        load_twist(str(out), 2)
