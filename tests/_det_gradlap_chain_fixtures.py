"""Shared inputs for the determinant grad/lap long-chain gates.

The XLA streaming-state gate (``tests/test_jqmc_kernel/test_det_gradlap_long_chain.py``)
and the Rust interval-cache gate (``tests/test_rs_kernel/test_det_gradlap_long_chain_rust.py``)
walk the same deterministic LRDMC-like hop schedule on the same wavefunctions:

* every hop moves one electron by ``HOP_ALAT`` along a lattice direction ``+-x/y/z``
  (the discretized-kinetic move of LRDMC), drawn from a fixed ``numpy`` PCG64 seed;
* the first electron of every spin channel that holds two or more electrons is
  FROZEN for the whole chain, so its rows of every carried table are never
  rewritten by a hop. A per-hop corruption of a carried quantity therefore
  accumulates over the whole chain instead of being washed out by the next
  rewrite -- the regime in which a drifting incremental update becomes visible
  only after many hops;
* molecular walks are kept near the nuclei: a hop that would leave the
  ``MOLECULAR_RADIUS`` ball around the nearest nucleus is reflected;
* every visited configuration satisfies the same input-only geminal
  conditioning predicate as ``draw_well_conditioned_cfg`` (``cond(G)`` below
  its default cap, evaluated by XLA before any derivative is computed): a hop
  that would cross into a near-nodal configuration is redrawn. This fixes the
  numerical domain the way the fixed-node walk itself avoids nodes, without
  selecting on the compared quantity. :func:`near_singular_schedule`
  deliberately breaks this predicate: it inserts near-coincident same-spin
  excursions to test the carried updates where the
  Sherman-Morrison factors are largest.

Wavefunctions: committed occupied-orbital SDs and their AGP rewrites, real and
complex, molecular and periodic, closed and open shell. The open-shell periodic
cases are the unrestricted triplet-oxygen QE saves (``O-lsda_gamma`` real,
``O-lsda_k025`` complex at k = 0.25) converted by the production converters;
their AO-basis AGP forms come from the production ``to_AGP``. The same two saves
converted to B-spline leaves give multi-electron B-spline
determinants (4 up / 2 dn, square 4 x 4 up block). The dense MO-basis
AGP of the molecular nitrogen atom carries more orbitals than electrons.
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

import os

import numpy as np

from tests._test_data_cache import qe_example_files_root

HOP_ALAT = 0.30
N_HOPS = 1000
SHORT_HOPS = 32
MOLECULAR_RADIUS = 2.5
_SEED = 20261002

# Near-singular excursions (near-coincident same-spin construction):
# every NEAR_SINGULAR_EVERY hops of a NEAR_SINGULAR_HOPS-hop walk, one up electron
# jumps to NEAR_SINGULAR_DELTA bohr of a same-spin electron and jumps back. The
# visited geminal is then far outside the draw_well_conditioned_cfg cap
# (cond(G) >= NEAR_SINGULAR_MIN_COND is asserted, so the case cannot pass
# vacuously) and the Sherman-Morrison factors of both hops are large.
NEAR_SINGULAR_HOPS = 300
NEAR_SINGULAR_EVERY = 50
NEAR_SINGULAR_DELTA = 1.0e-5
NEAR_SINGULAR_MIN_COND = 1.0e6

_HERE = os.path.dirname(os.path.abspath(__file__))
_TREXIO_DIR = os.path.join(_HERE, "trexio_example_files")
_QE_DIR = os.fspath(qe_example_files_root())
_O_K025_SAVE = os.path.join(_QE_DIR, "O-lsda_k025", "out", "o_k025.save")
_O_K025_UPF = os.path.join(_O_K025_SAVE, "O.ccECP.upf")
_O_GAMMA_SAVE = os.path.join(_QE_DIR, "O-lsda_gamma", "out", "o.save")


def _trexio_ham(stem: str, gem: str):
    """Committed TREXIO ham (no Jastrow) in the requested geminal representation."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    return _build_ham_for_cell(stem, False, None, gem)


def _o_lsda_twist_ham(tmp_dir: str):
    from jqmc._twist_average import load_twist
    from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

    out = os.path.join(tmp_dir, "o_lsda_k025_pgto.h5")
    if not os.path.exists(out):
        convert_qe_to_twist_average_input(_O_K025_SAVE, out, upf_paths={"O": _O_K025_UPF}, kind="pgto")
    return load_twist(out, 0)


def _o_lsda_gamma_ham(tmp_dir: str):
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jqmc_qe import run_qe2jqmc

    out = os.path.join(tmp_dir, "o_lsda_gamma_pgto.h5")
    if not os.path.exists(out):
        qe_params = {"qe_save_dir": _O_GAMMA_SAVE, "kind": "pgto", "upf_paths": [], "ecutwfc": None, "alpha_min": None}
        run_qe2jqmc(qe_params, {"j1_parameter": None, "j2_parameter": None}, out)
    return Hamiltonian_data.load_from_hdf5(out)


def _o_lsda_bspline_ham(tmp_dir: str, twist: bool):
    """The O-lsda QE saves converted by the production converters to a B-spline leaf.

    Multi-electron (4 up / 2 dn) B-spline determinants: the up block is a square
    4 x 4 orbital block, so an index or transpose fault in a per-electron
    contraction is not hidden by the 1 x 1 blocks of the Li B-spline case.
    """
    if twist:
        from jqmc._twist_average import load_twist
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input

        out = os.path.join(tmp_dir, "o_lsda_k025_bspline.h5")
        if not os.path.exists(out):
            convert_qe_to_twist_average_input(_O_K025_SAVE, out, upf_paths={"O": _O_K025_UPF}, kind="bspline")
        return load_twist(out, 0)
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jqmc_qe import run_qe2jqmc

    out = os.path.join(tmp_dir, "o_lsda_gamma_bspline.h5")
    if not os.path.exists(out):
        qe_params = {"qe_save_dir": _O_GAMMA_SAVE, "kind": "bspline", "upf_paths": [], "ecutwfc": None, "alpha_min": None}
        run_qe2jqmc(qe_params, {"j1_parameter": None, "j2_parameter": None}, out)
    return Hamiltonian_data.load_from_hdf5(out)


def _dense_mo_agp(sd, seed: int = _SEED, n_extra: int = 2):
    """A genuinely dense AGP over MO leaves that contain more orbitals than electrons.

    The up leaf is the SD's occupied rows plus ``n_extra`` seeded random AO
    combinations; the dn leaf is the SD's dn rows padded with the leading up
    rows to the same count (AGP needs equally many up and dn orbitals). With
    more orbitals than up electrons the up-move contraction weights of the dn
    spin genuinely change (for an exactly square orbital block they do not), so
    every carried table is live. ``lambda = [1 | 0] + 0.3 R`` (``R`` standard
    normal, complex for a complex SD) is dense, so every paired and unpaired
    entry enters ``G`` and the derivative tables.
    """
    from jqmc.determinant import AGP_data

    n_up, n_dn = int(sd.num_electron_up), int(sd.num_electron_dn)
    c_up = np.asarray(sd.orb_data_up_spin.mo_coefficients)
    c_dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
    is_complex = np.iscomplexobj(c_up) or np.iscomplexobj(c_dn)
    rng = np.random.default_rng(seed)
    extra = rng.normal(size=(n_extra, c_up.shape[1])) * np.max(np.abs(c_up))
    if is_complex:
        extra = extra + 1j * rng.normal(size=extra.shape) * np.max(np.abs(c_up))
    c_up_ext = np.vstack([c_up, extra])
    n_orb = c_up_ext.shape[0]
    c_dn_ext = np.vstack([c_dn, c_up_ext[: n_orb - n_dn]])
    mos_up = sd.orb_data_up_spin.replace(num_mo=n_orb, mo_coefficients=np.ascontiguousarray(c_up_ext))
    mos_dn = sd.orb_data_dn_spin.replace(num_mo=n_orb, mo_coefficients=np.ascontiguousarray(c_dn_ext))
    shape = (n_orb, n_orb + n_up - n_dn)
    noise = rng.normal(size=shape)
    if is_complex:
        noise = noise + 1j * rng.normal(size=shape)
    lam = np.hstack([np.eye(n_orb), np.zeros((n_orb, n_up - n_dn))]) + 0.3 * noise
    return AGP_data.build(
        num_electron_up=n_up,
        num_electron_dn=n_dn,
        orb_data_up_spin=mos_up,
        orb_data_dn_spin=mos_dn,
        lambda_matrix=lam,
        time_reversal=False,
    )


def _with_geminal(ham, geminal):
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))


def _drop_jastrow(ham):
    from jqmc.jastrow_factor import Jastrow_data

    jd = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


def build_case(case: str, tmp_dir: str):
    """Return the ``Hamiltonian_data`` of one named chain case (no Jastrow)."""
    from jqmc.determinant import to_AGP

    if case == "sd_real_mol_open":
        return _trexio_ham("N_ae_ccpvdz_cart.h5", "SD")
    if case == "agp_real_mol_open":
        return _trexio_ham("N_ae_ccpvdz_cart.h5", "AGP")
    if case == "agp_mo_real_mol_open":
        ham = _trexio_ham("N_ae_ccpvdz_cart.h5", "SD")
        return _with_geminal(ham, _dense_mo_agp(ham.wavefunction_data.geminal_data))
    if case == "sd_real_pbc_open":
        return _drop_jastrow(_o_lsda_gamma_ham(tmp_dir))
    if case == "agp_real_pbc_open":
        ham = _drop_jastrow(_o_lsda_gamma_ham(tmp_dir))
        return _with_geminal(ham, to_AGP(ham.wavefunction_data.geminal_data))
    if case == "sd_complex_twist_open":
        return _drop_jastrow(_o_lsda_twist_ham(tmp_dir))
    if case == "agp_complex_twist_open":
        ham = _drop_jastrow(_o_lsda_twist_ham(tmp_dir))
        return _with_geminal(ham, to_AGP(ham.wavefunction_data.geminal_data))
    if case == "sd_real_bspline_open":
        return _drop_jastrow(_o_lsda_bspline_ham(tmp_dir, twist=False))
    if case == "sd_complex_bspline_open":
        return _drop_jastrow(_o_lsda_bspline_ham(tmp_dir, twist=True))
    if case == "sd_complex_bspline_twist":
        from tests.test_rs_kernel._rs_fixtures import rs_ham

        return rs_ham("bspline_twist_jnone")
    raise KeyError(f"unknown det grad/lap chain case {case!r}")


def geminal_cond_max() -> float:
    """The conditioning cap of ``draw_well_conditioned_cfg`` (single source)."""
    from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

    return float(draw_well_conditioned_cfg.__kwdefaults__["cond_max"])


def start_configuration(ham, seed: int = _SEED):
    """Deterministic ion-centred, well-conditioned start ``(r_up, r_dn)``."""
    from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

    return draw_well_conditioned_cfg(ham, seed=seed)


def _geminal_cond(geminal, r_up, r_dn) -> float:
    from jqmc.determinant import compute_geminal_all_elements

    cond = float(np.linalg.cond(np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))))
    return cond if np.isfinite(cond) else np.inf


def frozen_electrons(n_up: int, n_dn: int) -> tuple[set[int], set[int]]:
    """Electrons never moved by :func:`hop_schedule` (index 0 of a spin with >= 2)."""
    return ({0} if n_up >= 2 else set()), ({0} if n_dn >= 2 else set())


def hop_schedule(ham, r_up0, r_dn0, n_hops: int = N_HOPS, seed: int = _SEED):
    """Deterministic LRDMC-like hop list ``[(is_up, electron, new_position)]``."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    frozen_up, frozen_dn = frozen_electrons(nu, nd)
    movable = [(True, i) for i in range(nu) if i not in frozen_up] + [(False, i) for i in range(nd) if i not in frozen_dn]
    periodic = bool(ham.structure_data.pbc_flag)
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    r_up = np.array(r_up0, dtype=np.float64)
    r_dn = np.array(r_dn0, dtype=np.float64)
    cond_max = geminal_cond_max()
    rng = np.random.default_rng(seed + 1)
    hops = []
    rejected = 0
    while len(hops) < n_hops:
        if rejected > n_hops:
            raise AssertionError("hop schedule cannot leave a near-nodal region; choose another seed")
        is_up, e = movable[int(rng.integers(len(movable)))]
        step = np.zeros(3)
        step[int(rng.integers(3))] = HOP_ALAT * (1.0 if rng.random() < 0.5 else -1.0)
        r = r_up if is_up else r_dn
        old = r[e].copy()
        new = old + step
        if not periodic and np.min(np.linalg.norm(nuclei - new, axis=1)) > MOLECULAR_RADIUS:
            new = old - step
        r[e] = new
        if _geminal_cond(gd, r_up, r_dn) >= cond_max:
            r[e] = old
            rejected += 1
            continue
        rejected = 0
        hops.append((bool(is_up), int(e), new.copy()))
    return hops


def near_singular_schedule(ham, r_up0, r_dn0, n_hops: int = NEAR_SINGULAR_HOPS, seed: int = _SEED):
    """:func:`hop_schedule` with near-coincident same-spin excursions inserted.

    Before every ``NEAR_SINGULAR_EVERY``-th regular hop, up electron 1 is moved to
    ``NEAR_SINGULAR_DELTA`` bohr of up electron 0 (the frozen one) and then back to
    where it was, so the walk returns to the well-conditioned schedule. Returns the
    hop list and the largest ``cond(G)`` met at the excursions.
    """
    gd = ham.wavefunction_data.geminal_data
    if int(gd.num_electron_up) < 2:
        raise ValueError("a near-coincident excursion needs two up electrons")
    base = hop_schedule(ham, r_up0, r_dn0, n_hops=n_hops, seed=seed)
    r_up = np.array(r_up0, dtype=np.float64)
    r_dn = np.array(r_dn0, dtype=np.float64)
    direction = np.array([0.6, 0.0, 0.8])
    hops = []
    cond_max = 0.0
    for k, (is_up, e, new) in enumerate(base):
        if k % NEAR_SINGULAR_EVERY == 0:
            home = r_up[1].copy()
            near = r_up[0] + NEAR_SINGULAR_DELTA * direction
            r_up[1] = near
            cond_max = max(cond_max, _geminal_cond(gd, r_up, r_dn))
            r_up[1] = home
            hops.append((True, 1, near.copy()))
            hops.append((True, 1, home.copy()))
        (r_up if is_up else r_dn)[e] = new
        hops.append((is_up, e, new))
    return hops, cond_max
