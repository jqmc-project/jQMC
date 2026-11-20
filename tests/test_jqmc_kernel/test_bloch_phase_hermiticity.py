"""Many-body Bloch phase and Hamiltonian Hermiticity at a k != Gamma twist.

Two exact finite-input identities, checked on fixed inputs in each precision mode
against that mode's own strict zone policy (no sampling, no sigma band):

1. Many-body Bloch boundary condition. Translating ONE electron by an integer
   lattice vector ``L = n . cell`` multiplies the signed complex wavefunction by
   ``exp(2 pi i k_frac . n)`` of that electron's spin channel, and leaves the complex
   local energy unchanged. The modulus and the unit phase are compared separately.
   The expected channel momentum comes from the fixture request (the canonical
   provider's k, or the TREXIO file's up k), not from the dn leaf under test:
   restricted / time-reversal pairs use ``-k`` for dn, unrestricted channels ``+k``.
   The canonical J3 terms remain active on their shared phase-free orbital basis;
   the determinant, J1, J2, J3, DTN and ECP parts are checked together.

2. Hermiticity of the production local-energy operator. For a Jastrow-free
   plane-wave Slater determinant at a twist, ``conj(Psi_a) (T Psi_b)`` is a
   trigonometric polynomial in the fractional electron coordinates (the Bloch
   factors cancel), so a uniform grid with more points per axis than its highest
   frequency integrates it exactly. The potential is real, so its matrix part is
   Hermitian on any grid. Hence ``H_ab = sum_grid conj(Psi_a) Psi_b e_L[b]`` must
   equal ``conj(H_ba)`` to rounding, and the kinetic matrix must not depend on the
   grid offset. This exact identity is limited to band-limited orbitals and no
   Jastrow: pGTO, B-spline and Jastrow integrands are not trigonometric
   polynomials. The complex VMC non-local ECP projector requires a separate
   quadrature analysis; it is not included in this all-electron finite-grid identity.

3. Single-orbital Bloch boundary condition: each complex
   pGTO, plane-wave and B-spline AO obeys chi(r + n.cell) = exp(2 pi i k.n) chi(r)
   on cubic, FCC, triclinic, hexagonal and monoclinic cells; a transposed inv_cell
   in the k map fails on the asymmetric cells only.

4. Hermiticity with J2 and pGTO: without an exact grid,
   the anti-Hermitian part of the local-energy matrix is shown to be the integral
   of the divergence of a lattice-periodic current, from two exact pointwise
   identities (see the section comment below).
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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_cancelling_mean, get_tolerance_min
from jqmc.hamiltonians import compute_local_energy
from jqmc.wavefunction import compute_kinetic_energy, evaluate_wavefunction
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# The canonical provider builds every complex leaf at this fractional twist; it is
# restated here as the independent expectation, and the built leaves must agree.
_CANONICAL_K = np.asarray((0.25, 0.125, 0.0))
_COMPLEX_CANONICAL = (
    "WF-PATTERN-005",
    "WF-PATTERN-006",
    "WF-PATTERN-008",
    "WF-PATTERN-010",
    "WF-PATTERN-012",
    "WF-PATTERN-014",
    "WF-PATTERN-016",
)
_TREXIO_TWISTS = {
    "h4_twist_j12": ("H4_pbc_ae_k025_cart.h5", None),
    "li_twist_jdtn": ("lithium_k025.h5", {"dtn": True}),
}
# Integer lattice translations (rows of n). None is a Bloch no-op at either fixture k.
_SHIFTS = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, -1], [2, -1, 1]])
_RT = np.eye(3)
_PSI_ZONES = ("ao_eval", "mo_eval", "jastrow_eval", "det_eval", "wf_eval")
_E_L_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_eval", "jastrow_grad_lap", "det_eval", "det_grad_lap", "wf_kinetic")
_E_L_ZONES += ("coulomb", "ewald", "local_energy")


def _leaf(orbital):
    return getattr(orbital, "aos_data", orbital)


def _leaf_k(orbital) -> np.ndarray:
    """Fractional k of every complex leaf in one spin channel (composites must agree)."""
    leaf = _leaf(orbital)
    leaves = getattr(leaf, "children", None) or (leaf,)
    ks = {tuple(np.round(np.asarray(x.k_point, dtype=float), 12)) for x in leaves if getattr(x, "k_point", None) is not None}
    assert len(ks) == 1, f"channel carries {len(ks)} distinct k points: {ks}"
    return np.asarray(ks.pop())


def _channels(geminal):
    up = getattr(geminal, "orb_data_up_spin", None) or geminal.mos_up
    dn = getattr(geminal, "orb_data_dn_spin", None) or geminal.mos_dn
    return up, dn


@functools.cache
def _case(case: str):
    """Return (ham, k_up, k_dn, r_up, r_dn) with k from the fixture request."""
    rng = np.random.default_rng(864)
    if case in _TREXIO_TWISTS:
        from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham

        stem, spec = _TREXIO_TWISTS[case]
        ham = _build_ham(stem, True, spec)
        geminal = ham.wavefunction_data.geminal_data
        assert geminal.time_reversal, f"{case}: TREXIO twist fixture is expected to be a time-reversal pair"
        k_up = _leaf_k(_channels(geminal)[0])
        k_dn = -k_up
        R = np.asarray(ham.structure_data.positions, float)
        r_up = R[rng.integers(0, len(R), geminal.num_electron_up)] + rng.normal(scale=0.8, size=(geminal.num_electron_up, 3))
        r_dn = R[rng.integers(0, len(R), geminal.num_electron_dn)] + rng.normal(scale=0.8, size=(geminal.num_electron_dn, 3))
    else:
        ham = build_canonical_hamiltonian(case)
        geminal = ham.wavefunction_data.geminal_data
        # Restricted and independent channels alike pair at opposite twists.
        k_up = _CANONICAL_K
        k_dn = -_CANONICAL_K
        r_up = rng.normal(scale=0.6, size=(geminal.num_electron_up, 3))
        r_dn = rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3))
    return ham, k_up, k_dn, r_up, r_dn


@functools.cache
def _psi_fn():
    return jax.jit(evaluate_wavefunction)


@functools.cache
def _e_l_fn():
    return jax.jit(compute_local_energy)


def _bloch_violations(ham, k_up, k_dn, r_up, r_dn):
    """Return per-shift messages for every violated Bloch or e_L invariant (empty = pass)."""
    wf = ham.wavefunction_data
    cell = np.asarray(ham.structure_data.cell, float)
    psi_atol, psi_rtol = get_tolerance_min(_PSI_ZONES, "strict")
    e_atol, e_rtol = get_tolerance_min(_E_L_ZONES, "strict")
    psi0 = complex(np.asarray(_psi_fn()(wf, jnp.asarray(r_up), jnp.asarray(r_dn))))
    e0 = complex(np.asarray(_e_l_fn()(ham, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT))))
    assert np.isfinite(psi0) and abs(psi0) > 0.0 and np.isfinite(e0)
    failures = []
    channels = [("up", i, k_up) for i in range(len(r_up))] + [("dn", i, k_dn) for i in range(len(r_dn))]
    # one electron of each spin (the first) and the last up electron
    selected = [c for c in channels if c[1] == 0] + ([channels[len(r_up) - 1]] if len(r_up) > 1 else [])
    for spin, index, k in selected:
        for n in _SHIFTS:
            expected_phase = np.exp(2j * np.pi * float(k @ n))
            # Non-vacuity: the chosen translation must carry a nontrivial Bloch phase.
            assert abs(expected_phase - 1.0) > 0.5, f"vacuous shift n={n} for k={k}"
            up, dn = r_up.copy(), r_dn.copy()
            (up if spin == "up" else dn)[index] += n @ cell
            psi = complex(np.asarray(_psi_fn()(wf, jnp.asarray(up), jnp.asarray(dn))))
            ratio = psi / psi0
            tag = f"{spin}{index} n={tuple(int(x) for x in n)}"
            if not np.isclose(abs(ratio), 1.0, atol=psi_atol, rtol=psi_rtol):
                failures.append(f"{tag}: modulus |Psi(r+L)/Psi(r)|={abs(ratio):.12g}")
            # Unit-phase comparison: a relative error rtol in Psi moves the phase by <= rtol.
            unit = ratio / abs(ratio)
            if not abs(unit - expected_phase) <= psi_atol + psi_rtol:
                failures.append(f"{tag}: phase {np.angle(unit):.9f} != expected {np.angle(expected_phase):.9f}")
            e = complex(np.asarray(_e_l_fn()(ham, jnp.asarray(up), jnp.asarray(dn), jnp.asarray(_RT))))
            if not np.isclose(e, e0, atol=e_atol, rtol=e_rtol):
                failures.append(f"{tag}: e_L {e!r} != {e0!r}")
    return failures, psi0, e0


@pytest.mark.parametrize("case", _COMPLEX_CANONICAL + tuple(_TREXIO_TWISTS))
def test_many_body_bloch_phase_and_local_energy_invariance(case):
    """Translating one electron by a lattice vector multiplies Psi by its channel Bloch phase.

    Inputs: the seven complex k != Gamma canonical patterns (orthorhombic, monoclinic,
    rhombohedral, cubic, hexagonal and triclinic cells; pGTO, plane-wave, B-spline and
    composite leaves; SD, AGP, AGPn; restricted and unrestricted; J1/J2/J3/DTN with
    the shared phase-free J3 basis; a nonzero synthetic s non-local ECP) and the committed
    H4 J1J2 and Li ECP J1J2DTN twist
    TREXIO fixtures. Seeded PCG64(864) positions, no conditioning screen. For the first
    up, last up and first dn electron and four lattice translations with nontrivial
    phase, the modulus ratio must be one and the unit phase must equal
    exp(2 pi i k.n) with the channel k taken from the fixture request. The complex
    local energy must be unchanged. Full and mixed each use their strict zone policy.
    """
    ham, k_up, k_dn, r_up, r_dn = _case(case)
    geminal = ham.wavefunction_data.geminal_data
    up, dn = _channels(geminal)
    # The built leaves must carry the requested momenta (drift guard, not the oracle).
    np.testing.assert_array_equal(_leaf_k(up), k_up)
    if geminal.num_electron_dn:
        np.testing.assert_array_equal(_leaf_k(dn), k_dn)
    failures, _, _ = _bloch_violations(ham, k_up, k_dn, r_up, r_dn)
    assert failures == [], f"{case}: " + "; ".join(failures)
    # Non-vacuity: each channel's shifts distinguish +k from -k (a conjugated leaf).
    for k in (k_up, k_dn) if geminal.num_electron_dn else (k_up,):
        assert np.max(np.abs(np.sin(2 * np.pi * (_SHIFTS @ k)))) > 0.5, f"{case}: shifts cannot see the sign of k={k}"
    jax.clear_caches()


def test_bloch_phase_activates_the_nonlocal_ecp_and_jastrow():
    """The translated e_L of the monoclinic ECP pattern consumes a live non-local projector and Jastrow.

    Canonical WF-PATTERN-006 (monoclinic, unrestricted AGPn, J1/J2/J3/DTN, synthetic
    s-channel non-local ECP) at the seeded positions. Zeroing only the non-local
    coefficient, or removing the whole Jastrow, must change e_L outside the strict
    policy, so the translation-invariance verdict above is not vacuous for either.
    """
    ham, _, _, r_up, r_dn = _case("WF-PATTERN-006")
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    args = (jnp.asarray(r_up), jnp.asarray(r_dn), jnp.asarray(_RT))
    e0 = complex(np.asarray(_e_l_fn()(ham, *args)))
    cp = ham.coulomb_potential_data
    assert tuple(cp.ang_moms) == (1, 1, 0) and float(cp.coefficients[2]) != 0.0
    local_only = ham.replace(coulomb_potential_data=cp.replace(coefficients=(*cp.coefficients[:2], 0.0)))
    e_local = complex(np.asarray(_e_l_fn()(local_only, *args)))
    assert not np.isclose(e_local, e0, atol=atol, rtol=rtol), "non-local ECP is inert at the Bloch configuration"
    jas = ham.wavefunction_data.jastrow_data
    no_jastrow = ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(
            jastrow_data=jas.replace(
                jastrow_one_body_data=None,
                jastrow_two_body_data=None,
                jastrow_three_body_data=None,
                jastrow_dtn_data=None,
            )
        )
    )
    e_nj = complex(np.asarray(_e_l_fn()(no_jastrow, *args)))
    assert not np.isclose(e_nj, e0, atol=atol, rtol=rtol), "Jastrow is inert at the Bloch configuration"
    jax.clear_caches()


@pytest.mark.parametrize("fault", ["drop-dn-restore-phase", "conjugate-plane-wave-k"])
def test_bloch_phase_check_rejects_a_dropped_or_conjugated_channel_phase(fault, monkeypatch):
    """The Bloch-phase verdict turns red when one orbital family loses or conjugates its phase.

    drop-dn-restore-phase: on triclinic restricted AGP WF-PATTERN-016, the pGTO
    wrap-restore factor exp(2 pi i k.floor(r inv_cell)) is replaced by one for the
    dn (-k) leaf only. conjugate-plane-wave-k: on rhombohedral WF-PATTERN-008 the
    complex plane-wave leaf evaluates at -k instead of +k. The unmodified inputs pass
    the same check (the parametrized test above); each fault must produce a phase
    violation while the modulus check alone stays satisfied, showing that the
    modulus-only translation tests could not see this class.
    """
    from jqmc import atomic_orbital as ao

    jax.clear_caches()
    if fault == "drop-dn-restore-phase":
        case = "WF-PATTERN-016"
        original = ao._ComplexBlochPGTOMixin._restore_phase

        def restore(self, r_carts, zone):
            phase = original(self, r_carts, zone)
            return jnp.ones_like(phase) if float(self.k_frac_np[0]) < 0.0 else phase

        monkeypatch.setattr(ao._ComplexBlochPGTOMixin, "_restore_phase", restore)
    else:
        case = "WF-PATTERN-008"
        original = ao.AOs_plane_wave_complex_data._gk_vectors_jnp.fget

        def gk(self):
            recip = 2.0 * jnp.pi * ao._inv_3x3(self.structure_data._cell_jnp).T
            k_cart = jnp.asarray(self.k_frac_np, dtype=jnp.float64) @ recip
            return original(self) - 2.0 * k_cart[None, :]

        monkeypatch.setattr(ao.AOs_plane_wave_complex_data, "_gk_vectors_jnp", property(gk))
    ham, k_up, k_dn, r_up, r_dn = _case(case)
    _psi_fn.cache_clear()
    _e_l_fn.cache_clear()
    failures, _, _ = _bloch_violations(ham, k_up, k_dn, r_up, r_dn)
    monkeypatch.undo()
    _psi_fn.cache_clear()
    _e_l_fn.cache_clear()
    jax.clear_caches()
    assert any(": phase " in f for f in failures), f"{fault}: phase fault not detected: {failures}"
    assert not any(": modulus " in f for f in failures), f"{fault}: unexpected modulus change: {failures}"
    if fault == "drop-dn-restore-phase":
        assert all(f.startswith("dn") for f in failures if ": phase " in f), failures


# --------------------------------------------------------------------------- #
# Single-orbital Bloch boundary condition on skewed cells
# --------------------------------------------------------------------------- #
_LEAF_K = (0.25, -0.1, 0.4)
_LEAF_CELLS = ("cubic_10bohr", "primitive_fcc_si", "triclinic_skewed", "hexagonal", "monoclinic")
_LEAF_KINDS = ("pgto-cart", "pgto-sphe", "plane-wave", "bspline")
# Rows of n: the three lattice vectors and a1 + a2 - a3. Each has a nontrivial phase at _LEAF_K.
_LEAF_SHIFTS = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, -1]])
# Far shifts (issue #36): an unwrapped walker can drift 1e3 - 1e5 bohr from the home cell.
# k_frac.n = 54.15 and 4802.55, so both phases are nontrivial.
_LEAF_FAR_SHIFTS = np.asarray([[101, -37, 63], [10001, -7003, 4005]])


def _leaf_cell(name):
    from tests.conftest import CELL_FIXTURES
    from tests.test_jqmc_kernel._noncubic_oracles import HEXAGONAL_CELL, MONOCLINIC_CELL

    return {"hexagonal": HEXAGONAL_CELL, "monoclinic": MONOCLINIC_CELL}.get(name, CELL_FIXTURES.get(name))


@functools.cache
def _complex_leaf(kind, cell_name):
    """One complex Bloch AO leaf of the requested family at _LEAF_K on the named cell."""
    from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
    from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_plane_wave_complex_data, _apply_image_expansion_complex
    from jqmc.pw_pgto_projection import bspline_prefilter_periodic
    from tests.test_jqmc_kernel import _noncubic_oracles as ref

    cell = _leaf_cell(cell_name)
    structure = ref.build_structure(cell, ref.ATOM_FRACTIONS @ cell)
    if kind.startswith("pgto-"):
        base = ref.build_aos(structure, kind.removeprefix("pgto-"))
        return _apply_image_expansion_complex(base, structure, k_point=_LEAF_K, tol=AO_IMAGE_TRUNCATION_ATOL)
    if kind == "plane-wave":
        millers = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [0, 1, 1]], dtype=float)
        return AOs_plane_wave_complex_data.build(structure, millers @ structure.recip_cell, k_point=_LEAF_K)
    shape = (8, 8, 8)
    x, y, z = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
    two_pi = 2 * np.pi
    bands = np.stack(
        (
            1 + 0.2 * np.cos(two_pi * x) + 0.3j * np.sin(two_pi * z),
            np.sin(two_pi * x) + 0.3 * np.cos(two_pi * y) + 0.2j * np.cos(two_pi * (x + z)),
            np.sin(two_pi * y) + 0.4j * np.cos(two_pi * z),
        )
    )
    controls = bspline_prefilter_periodic(bands.astype(np.complex128), 5).reshape(len(bands), -1)
    return AOs_bspline_complex_data.build(structure, controls, shape, 5, k_point=_LEAF_K)


def _leaf_points(cell):
    """Fractional points next to all six faces and outside the home cell, seeded PCG64(3003)."""
    rng = np.random.default_rng(3003)
    near_low = rng.uniform(0.005, 0.08, size=(3, 3))
    near_high = rng.uniform(0.92, 0.995, size=(3, 3))
    outside = rng.uniform(-0.3, 1.3, size=(4, 3))
    return np.concatenate((near_low, near_high, outside)) @ cell


def _single_orbital_bloch_violations(kind, cell_name, shifts=_LEAF_SHIFTS):
    """Messages for every AO / shift with chi(r + n.cell) != exp(2 pi i k.n) chi(r) (empty = pass)."""
    from jqmc.atomic_orbital import compute_AOs

    cell = _leaf_cell(cell_name)
    leaf = _complex_leaf(kind, cell_name)
    np.testing.assert_array_equal(np.asarray(leaf.k_point, dtype=float), np.asarray(_LEAF_K))
    r = _leaf_points(cell)
    atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    chi = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(r)))
    assert np.all(np.isfinite(chi)) and np.max(np.abs(chi)) > 1.0e-2, f"{kind}/{cell_name}: vacuous AO values"
    failures = []
    for n in shifts:
        expected = np.exp(2j * np.pi * float(np.asarray(_LEAF_K) @ n))
        assert abs(expected - 1.0) > 0.5, f"vacuous shift n={n}"
        shifted = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(r + n @ cell)))
        assert np.all(np.isfinite(shifted))
        if not np.allclose(shifted, expected * chi, atol=atol, rtol=rtol, equal_nan=False):
            err = np.max(np.abs(shifted - expected * chi))
            failures.append(f"{kind}/{cell_name} n={tuple(int(v) for v in n)}: max|chi(r+L)-phase chi(r)|={err:.3e}")
    return failures


@pytest.mark.parametrize("cell_name", _LEAF_CELLS)
@pytest.mark.parametrize("kind", _LEAF_KINDS)
def test_single_orbital_bloch_boundary_condition_on_skewed_cells(kind, cell_name):
    """Each complex AO leaf obeys chi(r + n.cell) = exp(2 pi i k_frac.n) chi(r) on skewed cells.

    Leaves: periodic Cartesian and spherical pGTO (two H centres, s/p/d/f shells, the
    non-orthogonal-cell oracle basis), a six-plane-wave leaf and a three-band complex
    quintic B-spline leaf, all at k_frac = (0.25, -0.1, 0.4). Cells: cubic_10bohr,
    primitive_fcc_si, triclinic_skewed, hexagonal and monoclinic. Ten seeded PCG64(3003)
    fractional points lie next to every face or outside the home cell; the shifts
    a1, a2, a3 and a1 + a2 - a3 move each point across faces. The expected phase is
    written from k_frac and the integer shift, never from a production Cartesian k.
    Each AO value is compared with the ao_eval strict zone policy of the active mode.

    Two far shifts, n = (101, -37, 63) and (10001, -7003, 4005), up to about 1e5 bohr on
    the cubic cell, check that the law holds at the same zone policy independently of |n|
    (an fp32 phase formed before reduction modulo 2 pi loses accuracy in proportion
    to |r|).
    """
    failures = _single_orbital_bloch_violations(kind, cell_name, np.concatenate((_LEAF_SHIFTS, _LEAF_FAR_SHIFTS)))
    assert failures == [], "; ".join(failures)
    jax.clear_caches()


def _exact_bloch_phase(n):
    """exp(2 pi i k_frac.n) with k_frac.n reduced modulo 1 in exact rational arithmetic."""
    from fractions import Fraction

    turns = sum(Fraction(str(k)) * int(m) for k, m in zip(_LEAF_K, n)) % 1
    return np.exp(2j * np.pi * float(turns))


@pytest.mark.parametrize("cell_name", _LEAF_CELLS)
@pytest.mark.parametrize("kind", _LEAF_KINDS)
def test_complex_leaf_bloch_phase_far_from_home_cell(kind, cell_name):
    """A complex AO leaf picks up exp(2 pi i k_frac.n) for a far lattice translation.

    MCMC walkers are not wrapped, so an electron can sit 1e3 - 1e5 bohr away from the
    home cell. For the far shifts n = (101, -37, 63) and (10001, -7003, 4005) every
    complex pGTO (Cartesian and spherical), plane-wave and B-spline leaf at
    k_frac = (0.25, -0.1, 0.4) must satisfy chi(r + n.cell) = exp(2 pi i k_frac.n) chi(r)
    on the cubic and skewed cells. The expected phase is formed from the literal k_frac
    and the integer n in exact rational arithmetic, reduced modulo 1 before the
    exponential, so it never uses a production Cartesian k or an fp32 product. The
    home-cell values chi(r) are the leaf's own values at r; the law is judged with the
    ao_eval strict zone policy of the active mode. In mixed precision this requires the
    leaf to reduce its Bloch phase in fp64 before the fp32 cast.
    """
    from jqmc.atomic_orbital import compute_AOs

    cell = _leaf_cell(cell_name)
    leaf = _complex_leaf(kind, cell_name)
    np.testing.assert_array_equal(np.asarray(leaf.k_point, dtype=float), np.asarray(_LEAF_K))
    r = _leaf_points(cell)
    atol, rtol = get_tolerance_min(("ao_eval",), "strict")
    chi = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(r)))
    assert np.all(np.isfinite(chi)) and np.max(np.abs(chi)) > 1.0e-2, f"{kind}/{cell_name}: vacuous AO values"
    failures = []
    for n in _LEAF_FAR_SHIFTS:
        expected = _exact_bloch_phase(n)
        assert abs(expected - 1.0) > 0.5, f"vacuous shift n={n}"
        shifted = np.asarray(compute_AOs(aos_data=leaf, r_carts=jnp.asarray(r + n @ cell)))
        assert np.all(np.isfinite(shifted))
        if not np.allclose(shifted, expected * chi, atol=atol, rtol=rtol, equal_nan=False):
            err = np.max(np.abs(shifted - expected * chi))
            failures.append(f"{kind}/{cell_name} n={tuple(int(v) for v in n)}: max|chi(r+n.cell)-phase chi(r)|={err:.3e}")
    assert failures == [], "; ".join(failures)
    jax.clear_caches()


def _transpose_inv_cell_fault(monkeypatch, kind):
    """Install ``inv_cell -> inv_cell.T`` in the fractional-k map of one leaf family."""
    from jqmc import atomic_orbital as ao

    if kind.startswith("pgto-"):

        def restore(self, r_carts, zone):
            inv_cell_t = ao._inv_3x3(self.structure_data._cell_jnp.astype(jnp.float64)).T
            k_frac = jnp.asarray(self.k_frac_np, dtype=jnp.float64)
            n0 = jnp.floor(jnp.asarray(r_carts, dtype=jnp.float64) @ inv_cell_t)
            angle = jnp.asarray(2.0 * jnp.pi * (n0 @ k_frac), dtype=ao.get_dtype_jnp(zone))
            return jnp.exp(jnp.asarray(1j, dtype=ao.get_complex_dtype_jnp(zone)) * angle)

        monkeypatch.setattr(ao._ComplexBlochPGTOMixin, "_restore_phase", restore)
    elif kind == "plane-wave":
        original = ao.AOs_plane_wave_complex_data._gk_vectors_jnp.fget

        def gk(self):
            inv = ao._inv_3x3(self.structure_data._cell_jnp)
            k_frac = jnp.asarray(self.k_frac_np, dtype=jnp.float64)
            return original(self) + (k_frac @ (2.0 * jnp.pi * inv) - k_frac @ (2.0 * jnp.pi * inv.T))[None, :]

        monkeypatch.setattr(ao.AOs_plane_wave_complex_data, "_gk_vectors_jnp", property(gk))
    else:

        def k_cart(self):
            inv = ao._inv_3x3(self.structure_data._cell_jnp)
            return jnp.asarray(self.k_frac_np, dtype=jnp.float64) @ (2.0 * jnp.pi * inv)

        monkeypatch.setattr(ao.AOs_bspline_complex_data, "_k_cart_jnp", property(k_cart))


@pytest.mark.parametrize("kind", _LEAF_KINDS)
def test_single_orbital_bloch_check_rejects_a_transposed_inverse_cell(kind, monkeypatch):
    """``inv_cell -> inv_cell.T`` in the k map fails on every asymmetric cell and passes on symmetric ones.

    pGTO: the wrap-restore factor uses floor(r inv_cell.T). Plane wave and B-spline: the
    Cartesian twist is k_frac (2 pi inv_cell) instead of k_frac (2 pi inv_cell.T). The
    fault is the identity when the cell matrix is symmetric (cubic_10bohr, and also the
    primitive FCC cell as written), so the Bloch check must stay satisfied there and
    must fail on the triclinic, hexagonal and monoclinic cells. This shows that the
    asymmetric skewed cells are what make the check sensitive to this slip.
    """
    jax.clear_caches()
    _transpose_inv_cell_fault(monkeypatch, kind)
    results = {name: _single_orbital_bloch_violations(kind, name) for name in _LEAF_CELLS}
    monkeypatch.undo()
    jax.clear_caches()
    for name in _LEAF_CELLS:
        cell = _leaf_cell(name)
        if np.array_equal(cell, cell.T):
            assert results[name] == [], f"{kind}: fault must be invisible on symmetric {name}: {results[name]}"
        else:
            assert results[name], f"{kind}: transposed inv_cell not detected on {name}"
    assert sum(not np.array_equal(_leaf_cell(n), _leaf_cell(n).T) for n in _LEAF_CELLS) == 3


# --------------------------------------------------------------------------- #
# Hermiticity on an exact periodic quadrature
# --------------------------------------------------------------------------- #
_HERM_K = (0.25, -0.1, 0.4)
_HERM_MILLERS = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float)
_HERM_GRID = 2  # max per-axis Miller difference is 1 < 2: the grid sum is exact
_HERM_STATES = 3
_HERM_UP, _HERM_DN = 2, 1


@functools.cache
def _hermiticity_states():
    """Three Jastrow-free unrestricted PW Slater determinants on a triclinic cell at k."""
    from jqmc.atomic_orbital import AOs_plane_wave_complex_data
    from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
    from jqmc.determinant import SD_data
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.molecular_orbital import MOs_data
    from jqmc.structure import Structure_data
    from jqmc.wavefunction import Wavefunction_data

    cell = np.asarray([[6, 0, 0], [1, 7, 0], [0.5, 1.5, 8]], dtype=float)
    structure = Structure_data.build(
        pbc_flag=True,
        positions=np.asarray([[0.3, 0.2, 0.1]]),
        atomic_numbers=(3,),
        element_symbols=("Li",),
        atomic_labels=("Li1",),
        vec_a=cell[0],
        vec_b=cell[1],
        vec_c=cell[2],
    )
    coulomb = Coulomb_potential_data(structure_data=structure)
    coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, n_electron=_HERM_UP + _HERM_DN))
    leaf = AOs_plane_wave_complex_data.build(structure, _HERM_MILLERS @ structure.recip_cell, k_point=_HERM_K)
    rng = np.random.default_rng(2718)
    hams = []
    for _ in range(_HERM_STATES):
        mos = []
        # Independent spin channels at opposite twists: the down basis is the -k leaf.
        for n, spin_leaf in ((_HERM_UP, leaf), (_HERM_DN, leaf.with_negated_k())):
            c = rng.normal(size=(n, spin_leaf.num_ao)) + 1j * rng.normal(size=(n, spin_leaf.num_ao))
            mos.append(MOs_data.build(aos_data=spin_leaf, mo_coefficients=c))
        geminal = SD_data.build(
            num_electron_up=_HERM_UP, num_electron_dn=_HERM_DN, mos_up=mos[0], mos_dn=mos[1], time_reversal=False
        )
        wf = Wavefunction_data(
            geminal_data=geminal,
            jastrow_data=Jastrow_data(
                jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
            ),
        )
        hams.append(Hamiltonian_data.build(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wf))
    return cell, hams


def _grid(cell, offsets):
    """All (up, dn) configurations of a uniform fractional grid, one offset per electron."""
    axis = np.arange(_HERM_GRID) / _HERM_GRID
    one = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    n_e = _HERM_UP + _HERM_DN
    index = np.stack(np.meshgrid(*([np.arange(len(one))] * n_e), indexing="ij"), axis=-1).reshape(-1, n_e)
    frac = one[index] + offsets[None]
    carts = frac @ cell
    return carts[:, :_HERM_UP], carts[:, _HERM_UP:]


@functools.cache
def _batched():
    psi = jax.jit(jax.vmap(evaluate_wavefunction, in_axes=(None, 0, 0)))
    kin = jax.jit(jax.vmap(compute_kinetic_energy, in_axes=(None, 0, 0)))
    e_l = jax.jit(jax.vmap(compute_local_energy, in_axes=(None, 0, 0, None)))
    return psi, kin, e_l


def _quadrature_matrices(hams, r_up, r_dn):
    """Grid means of conj(Psi_a) Psi_b O_b for O = e_L and T, plus their summand scales."""
    psi_f, kin_f, e_f = _batched()
    ju, jd = jnp.asarray(r_up), jnp.asarray(r_dn)
    psi = np.stack([np.asarray(psi_f(h.wavefunction_data, ju, jd)) for h in hams])
    kin = np.stack([np.asarray(kin_f(h.wavefunction_data, ju, jd)) for h in hams])
    e_l = np.stack([np.asarray(e_f(h, ju, jd, jnp.asarray(_RT))) for h in hams])
    assert np.all(np.isfinite(psi)) and np.all(np.isfinite(kin)) and np.all(np.isfinite(e_l))
    pair = np.conj(psi)[:, None, :] * psi[None, :, :]
    h_terms = pair * e_l[None, :, :]
    t_terms = pair * kin[None, :, :]
    return h_terms.mean(-1), t_terms.mean(-1), np.abs(h_terms).max(), np.abs(t_terms).max(), e_l


_OFFSETS = (
    np.asarray([[0.11, 0.07, 0.03], [0.37, 0.29, 0.41], [0.23, 0.13, 0.31]]),
    np.asarray([[0.19, 0.43, 0.05], [0.02, 0.17, 0.27], [0.33, 0.09, 0.14]]),
)


def _hermiticity_errors(hams, cell):
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    errors, results = [], []
    for offsets in _OFFSETS:
        r_up, r_dn = _grid(cell, offsets)
        h, t, h_scale, t_scale, e_l = _quadrature_matrices(hams, r_up, r_dn)
        h_tol, _ = get_tolerance_cancelling_mean(atol, rtol, h_scale)
        t_tol, _ = get_tolerance_cancelling_mean(atol, rtol, t_scale)
        if not np.max(np.abs(h - h.conj().T)) <= h_tol:
            errors.append(f"H not Hermitian: max|H-H^+|={np.max(np.abs(h - h.conj().T)):.3e} > {h_tol:.3e}")
        if not np.max(np.abs(t - t.conj().T)) <= t_tol:
            errors.append(f"T not Hermitian: max|T-T^+|={np.max(np.abs(t - t.conj().T)):.3e} > {t_tol:.3e}")
        results.append((h, t, t_tol, e_l, h_tol))
    t_diff = np.max(np.abs(results[0][1] - results[1][1]))
    if not t_diff <= max(results[0][2], results[1][2]):
        errors.append(f"kinetic quadrature depends on the grid offset: {t_diff:.3e}")
    return errors, results


def test_local_energy_operator_is_hermitian_on_exact_periodic_quadrature():
    """At k != Gamma the production local-energy matrix is Hermitian on an exact quadrature.

    Input: three Jastrow-free, spin-unrestricted Slater determinants (2 up, 1 dn) of
    six complex plane waves at k=(0.25,-0.1,0.4) on a triclinic all-electron Li cell
    with Ewald Coulomb, with seeded PCG64(2718) complex MO coefficients. Every
    electron runs over a 2x2x2 fractional grid with its own offset, so electrons never
    coincide; the integrand conj(Psi_a) T Psi_b has per-axis frequencies <= 1 and is
    summed exactly. Two independent offset sets are used. H_ab from production
    compute_local_energy and T_ab from compute_kinetic_energy must equal their
    conjugate transposes, and T must not depend on the offsets, each within the strict
    zone policy rebased on the summand scale. Non-vacuity: Im(e_L) and the off-diagonal
    Im(H_ab) are nonzero. Full and mixed are judged separately.
    """
    cell, hams = _hermiticity_states()
    errors, results = _hermiticity_errors(hams, cell)
    assert errors == [], "; ".join(errors)
    h, _, _, e_l, h_tol = results[0]
    off = h[~np.eye(len(h), dtype=bool)]
    # A Re-projected H would differ from the actual one by more than the verdict bound.
    assert np.max(np.abs(off.imag)) > h_tol, f"off-diagonal Im H is inside the verdict bound {h_tol}: {off}"
    assert np.median(np.abs(e_l.imag)) > 1.0e-3 * np.median(np.abs(e_l.real)), "pointwise e_L is effectively real"
    jax.clear_caches()


def test_hermiticity_check_rejects_a_non_hermitian_kinetic_cross_term(monkeypatch):
    """An i-slip in the plane-wave Laplacian cross term must break the Hermiticity verdict.

    Same inputs as the Hermiticity test. The complex plane-wave Laplacian
    -(|G|^2 + 2 G.k + |k|^2) is replaced by -(|G|^2 + 2i G.k + |k|^2) in every
    Laplacian entry point: a plausible algebra slip that makes the kinetic
    operator's diagonal complex in the plane-wave basis. The same verdict and strict
    policy must now report a non-Hermitian H and T.
    """
    from jqmc import atomic_orbital as ao

    cls = ao.AOs_plane_wave_complex_data
    original_lap = cls._compute_AOs_laplacian
    original_vgl = cls._compute_AOs_value_grad_lap

    def slip(self, lap_times_val_over_g2):
        recip = 2.0 * jnp.pi * ao._inv_3x3(self.structure_data._cell_jnp).T
        m = jnp.asarray(np.asarray(self._g_indices, dtype=np.float64).reshape(-1, 3))
        g = m @ recip
        k = jnp.asarray(self.k_frac_np, dtype=jnp.float64) @ recip
        return (2.0 - 2.0j) * (g @ k)

    def lap(self, r_carts):
        val = self._value_in_zone(r_carts, "ao_grad_lap")
        return original_lap(self, r_carts) + slip(self, None)[:, None] * val

    def vgl(self, r_carts):
        v, gx, gy, gz, lp = original_vgl(self, r_carts)
        return v, gx, gy, gz, lp + slip(self, None)[:, None] * v.astype(lp.dtype)

    jax.clear_caches()
    _batched.cache_clear()
    monkeypatch.setattr(cls, "_compute_AOs_laplacian", lap)
    monkeypatch.setattr(cls, "_compute_AOs_value_grad_lap", vgl)
    cell, hams = _hermiticity_states()
    errors, results = _hermiticity_errors(hams, cell)
    monkeypatch.undo()
    _batched.cache_clear()
    jax.clear_caches()
    h, t, t_tol, _, h_tol = results[0]
    assert np.max(np.abs(h - h.conj().T)) > h_tol, errors
    assert np.max(np.abs(t - t.conj().T)) > t_tol, errors


# --------------------------------------------------------------------------- #
# Hermiticity with a Jastrow and pGTO orbitals: exact divergence form
# --------------------------------------------------------------------------- #
# With J2 (and pGTO), conj(Psi_a) T Psi_b is not a trigonometric polynomial, so no
# finite grid integrates it exactly. The same Hermiticity statement is proven here
# from two exact pointwise identities instead of a quadrature:
#   (i)  V_s(r) := e_L[s](r) + (1/2) lap Psi_s / Psi_s is real and the same for every
#        state s (it is the local potential), with lap Psi_s taken by forward-mode AD
#        of the production evaluate_wavefunction. Then, pointwise,
#        conj(Psi_a) Psi_b (e_L[b] - conj(e_L[a])) = -(1/2) div J_ab,
#        J_ab = conj(Psi_a) grad Psi_b - Psi_b grad conj(Psi_a);
#   (ii) J_ab is lattice periodic: conj(Psi_a) Psi_b and
#        g_ab = grad ln Psi_b - conj(grad ln Psi_a) are unchanged when one electron
#        moves by a lattice vector (states a and b share each channel's k).
# The divergence of a periodic field integrates to zero over the cell, so (i) and
# (ii) give H_ab = conj(H_ba) exactly for any Jastrow, basis and cell. A non-local
# ECP is excluded because this pointwise identity assumes a local potential;
# the canonical case includes its phase-free J3 correlation.
_DIV_CASES = ("plane-wave-triclinic-j2", "pgto-cart-h4-twist-j12", "pgto-sphe-triclinic-agp-j12")
_DIV_SHIFTS = np.asarray([[1, 0, 0], [0, 1, -1]])


def _perturbed_state(ham, seed):
    """Same Hamiltonian with seeded complex perturbations of every channel's MO coefficients."""
    from jqmc.molecular_orbital import MOs_data

    rng = np.random.default_rng(seed)
    geminal = ham.wavefunction_data.geminal_data
    changes = {}
    for name in ("orb_data_up_spin", "orb_data_dn_spin"):
        orb = getattr(geminal, name)
        assert isinstance(orb, MOs_data), f"{name} is expected to be an MO channel"
        c = np.asarray(orb.mo_coefficients)
        noise = rng.normal(size=c.shape) + 1j * rng.normal(size=c.shape)
        changes[name] = orb.replace(mo_coefficients=c + 0.3 * np.max(np.abs(c)) * noise)
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal.replace(**changes)))


@functools.cache
def _divergence_case(case):
    """Return (two states, cell, configurations) for a Hermiticity divergence case."""
    from jqmc.jastrow_factor import Jastrow_one_body_data, Jastrow_two_body_data

    if case == "plane-wave-triclinic-j2":
        cell, hams = _hermiticity_states()
        structure = hams[0].structure_data
        j2 = Jastrow_two_body_data(jastrow_2b_param=0.8, jastrow_2b_type="pade", structure_data=structure)
        states = [
            h.replace(
                wavefunction_data=h.wavefunction_data.replace(
                    jastrow_data=h.wavefunction_data.jastrow_data.replace(jastrow_two_body_data=j2)
                )
            )
            for h in hams[:2]
        ]
        rng = np.random.default_rng(4242)
        configs = [
            (rng.uniform(-0.3, 1.3, size=(_HERM_UP, 3)) @ cell, rng.uniform(-0.3, 1.3, size=(_HERM_DN, 3)) @ cell)
            for _ in range(3)
        ]
        return tuple(states), cell, tuple(configs)
    if case == "pgto-cart-h4-twist-j12":
        from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham

        ham = _build_ham("H4_pbc_ae_k025_cart.h5", True, None)
        jas = ham.wavefunction_data.jastrow_data
        assert isinstance(jas.jastrow_one_body_data, Jastrow_one_body_data)
        assert isinstance(jas.jastrow_two_body_data, Jastrow_two_body_data)
    else:
        ham = build_canonical_hamiltonian("WF-PATTERN-016")
        assert ham.wavefunction_data.jastrow_data.jastrow_two_body_data is not None
    assert not ham.coulomb_potential_data.ecp_flag, f"{case}: the divergence identity needs a local potential"
    cell = np.asarray(ham.structure_data.cell, float)
    geminal = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(5151)
    R = np.asarray(ham.structure_data.positions, float)
    configs = []
    for hop in (0, 1, 2):
        up = R[rng.integers(0, len(R), geminal.num_electron_up)] + rng.normal(scale=0.7, size=(geminal.num_electron_up, 3))
        dn = R[rng.integers(0, len(R), geminal.num_electron_dn)] + rng.normal(scale=0.7, size=(geminal.num_electron_dn, 3))
        up[0] += hop * cell[0]
        dn[-1] -= hop * cell[2]
        configs.append((up, dn))
    return (ham, _perturbed_state(ham, 77)), cell, tuple(configs)


def _psi_derivatives(wf, up, dn):
    """Psi, grad Psi (electrons x 3) and lap Psi by forward-mode AD of evaluate_wavefunction."""
    n_up = up.shape[0]
    x = jnp.asarray(np.concatenate((up, dn)))

    def psi(points):
        return evaluate_wavefunction(wf, points[:n_up], points[n_up:])

    value = complex(np.asarray(psi(x)))
    grad = np.asarray(jax.jacfwd(psi)(x))
    lap = complex(np.asarray(jnp.einsum("ijij->", jax.jacfwd(jax.jacfwd(psi))(x))))
    return value, grad, lap


def _divergence_violations(case):
    """Messages for every violated pointwise identity (i) or (ii) (empty = pass), plus measurements."""
    states, cell, configs = _divergence_case(case)
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    psi_atol, psi_rtol = get_tolerance_min(_PSI_ZONES, "strict")
    failures = []
    im_e_l, im_g, worst = [], [], [0.0, 0.0, 0.0]
    for c, (up, dn) in enumerate(configs):
        values = []
        for ham in states:
            e_l = complex(np.asarray(compute_local_energy(ham, jnp.asarray(up), jnp.asarray(dn), jnp.asarray(_RT))))
            psi, grad, lap = _psi_derivatives(ham.wavefunction_data, up, dn)
            assert np.isfinite(e_l) and np.isfinite(psi) and abs(psi) > 0.0 and np.isfinite(lap) and np.all(np.isfinite(grad))
            values.append((e_l, psi, grad, -0.5 * lap / psi))
            im_e_l.append(abs(e_l.imag))
        potential = values[0][0] - values[0][3]
        for s, (e_l, _psi, _grad, kinetic_ad) in enumerate(values):
            want = kinetic_ad + potential.real
            worst[0] = max(worst[0], abs(e_l - want))
            if not np.isclose(e_l, want, atol=atol, rtol=rtol, equal_nan=False):
                failures.append(f"config {c} state {s}: e_L={e_l!r} != T_AD + V={want!r} (Im V_0={potential.imag:.3e})")
        (_, psi_a, grad_a, _), (_, psi_b, grad_b, _) = values
        pair = np.conj(psi_a) * psi_b
        g_ab = grad_b / psi_b - np.conj(grad_a / psi_a)
        im_g.append(float(np.max(np.abs(g_ab.imag))))
        n_up = up.shape[0]
        for index in (0, n_up):
            for n in _DIV_SHIFTS:
                su, sd = up.copy(), dn.copy()
                (su if index < n_up else sd)[0] += n @ cell
                (psi_sa, grad_sa, _), (psi_sb, grad_sb, _) = (_psi_derivatives(ham.wavefunction_data, su, sd) for ham in states)
                pair_s = np.conj(psi_sa) * psi_sb
                g_s = grad_sb / psi_sb - np.conj(grad_sa / psi_sa)
                worst[1] = max(worst[1], abs(pair_s / pair - 1.0))
                worst[2] = max(worst[2], float(np.max(np.abs(g_s - g_ab))))
                tag = f"config {c} {'up' if index < n_up else 'dn'}0 n={tuple(int(v) for v in n)}"
                if not abs(pair_s / pair - 1.0) <= psi_atol + psi_rtol:
                    failures.append(f"{tag}: conj(Psi_a) Psi_b not periodic, ratio={pair_s / pair!r}")
                if not np.allclose(g_s, g_ab, atol=atol, rtol=rtol, equal_nan=False):
                    failures.append(f"{tag}: current direction not periodic, max diff {np.max(np.abs(g_s - g_ab)):.3e}")
    return failures, (atol, rtol, max(im_e_l), max(im_g), worst)


@pytest.mark.parametrize("case", _DIV_CASES)
def test_local_energy_antihermitian_part_is_a_periodic_divergence(case):
    """With J2 and pGTO at k != Gamma, H_ab - conj(H_ba) is the integral of a periodic divergence.

    Cases: the triclinic plane-wave system of the quadrature test with a Pade J2
    (a = 0.8) added (states 0 and 1, three seeded PCG64(4242) configurations over
    fractional [-0.3, 1.3)); the committed all-electron H4_pbc_ae_k025_cart.h5
    (Cartesian pGTO, J1 + J2, time-reversal SD); and canonical WF-PATTERN-016
    (triclinic spherical pGTO, restricted AGP, J1 + J2 + phase-free J3).
    For the TREXIO and canonical cases the second state perturbs every MO
    coefficient channel by seeded PCG64(77) complex noise; configurations are seeded
    PCG64(5151) nucleus-centred draws with one up and one dn electron moved 0, 1 and
    2 lattice vectors out. At each configuration: (i) production e_L of both states
    equals the AD kinetic energy -(1/2) lap Psi / Psi plus one real, state-independent
    potential; (ii) conj(Psi_a) Psi_b and grad ln Psi_b - conj(grad ln Psi_a) are
    unchanged when an up or a dn electron moves by a1 or a2 - a3. Together these make
    the anti-Hermitian part of the local-energy matrix the integral of the
    divergence of a lattice-periodic current, which is zero. Strict zone policy of
    the active mode; full and mixed are judged separately. Non-vacuity: Im(e_L) and
    Im(g_ab) exceed the tolerance, and removing J2 changes e_L outside it.
    """
    failures, (atol, rtol, im_e_l, im_g, _worst) = _divergence_violations(case)
    assert failures == [], f"{case}: " + "; ".join(failures)
    assert im_e_l > atol, f"{case}: e_L is effectively real ({im_e_l:.3e})"
    assert im_g > atol, f"{case}: the current has no imaginary direction ({im_g:.3e})"
    states, _cell, configs = _divergence_case(case)
    ham = states[0]
    jas = ham.wavefunction_data.jastrow_data
    no_j2 = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jas.replace(jastrow_two_body_data=None)))
    up, dn = configs[0]
    args = (jnp.asarray(up), jnp.asarray(dn), jnp.asarray(_RT))
    e0 = complex(np.asarray(compute_local_energy(ham, *args)))
    e1 = complex(np.asarray(compute_local_energy(no_j2, *args)))
    assert not np.isclose(e0, e1, atol=atol, rtol=rtol), f"{case}: J2 is inert"
    jax.clear_caches()


@pytest.mark.parametrize("fault", ["re-projected-kinetic", "conjugated-drift-square"])
@pytest.mark.parametrize("case", _DIV_CASES)
def test_antihermitian_divergence_rejects_a_real_projection_or_conjugation(case, fault, monkeypatch):
    """A real-projected kinetic energy, or conj(v).v in place of v.v, breaks identity (i).

    re-projected-kinetic: the kinetic energy used by compute_local_energy keeps only
    its real part (the Re-projection class). conjugated-drift-square: the drift term
    sum (grad J + grad ln D).(grad J + grad ln D) is replaced by the modulus square
    sum |grad J + grad ln D|^2, the conjugation slip the production comment warns
    against. Same cases and policy as the positive test; each fault must report an
    identity (i) violation, and the unmodified inputs pass (positive test above).
    """
    from jqmc import hamiltonians
    from jqmc.determinant import compute_grads_and_laplacian_ln_Det
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    original = hamiltonians.compute_kinetic_energy

    if fault == "re-projected-kinetic":

        def kinetic(wavefunction_data, r_up_carts, r_dn_carts):
            t = original(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
            return jnp.real(t).astype(t.dtype)

    else:

        def kinetic(wavefunction_data, r_up_carts, r_dn_carts):
            t = original(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
            gj_up, gj_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(
                wavefunction_data.jastrow_data, r_up_carts, r_dn_carts
            )
            gd_up, gd_dn, _, _ = compute_grads_and_laplacian_ln_Det(wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
            v = jnp.concatenate((gj_up + gd_up, gj_dn + gd_dn))
            return t + (0.5 * (jnp.sum(v * v) - jnp.sum(jnp.conj(v) * v))).astype(t.dtype)

    jax.clear_caches()
    monkeypatch.setattr(hamiltonians, "compute_kinetic_energy", kinetic)
    failures, _ = _divergence_violations(case)
    monkeypatch.undo()
    jax.clear_caches()
    assert any("!= T_AD + V" in f for f in failures), f"{case}/{fault}: fault not detected: {failures}"
