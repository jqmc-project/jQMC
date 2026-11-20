"""Mixed-precision AO value range precondition (issue #111).

In mixed precision the standalone AO value kernels (XLA ``compute_AOs`` and the Rust f32
value kernel ``jqmc_rs.compute_AOs`` of the mixed MCMC value path) evaluate every Gaussian
primitive in the ``ao_eval`` dtype (float32) as the product of the normalization N, the radial
factor N exp(-a r^2) and the angular polynomial (Cartesian x^nx y^ny z^nz, or the real solid
harmonic, bounded by r^l). PR #115 / #129 form N in float64 and round it once, but for very
high l or extreme exponents N itself, or the polynomial / radial factor where the AO is
significant, still leaves the float32 range; the kernels then return Inf, NaN, 0 or a wrong
finite value silently (the x^82 example of #111: a finite but wrong mixed XLA local energy).

The precondition is loud: in mixed precision a basis whose values the ``ao_eval``
dtype cannot represent is rejected with ``AOEvalRangeError`` where it is configured into a
wavefunction (``MOs_data.sanity_check``: MO build, geminal / Hamiltonian validation, TREXIO and
HDF5 loads, the MCMC / GFMC driver constructors; and ``Jastrow_three_body_data.sanity_check``).
A bare AO leaf stays constructible (its float64 gradient / Laplacian / fused kernels are fine).
Full precision never raises.

Independent reference (no jqmc code): the analytic range of every float32 factor of one
primitive, computed here with ``math.lgamma`` and a bisection for the significant shell:

* N, from N^-2 = prod_i Gamma(n_i + 1/2) / (2a)^(n_i + 1/2) (Cartesian) or
  N_n^2 = 2 (2a)^(l + 3/2) / Gamma(l + 3/2), N = N_n sqrt((2l+1)/(4 pi)) (spherical);
* the peak AO value max_r |N r^l exp(-a r^2)| (at x_i^2 = n_i / (2a));
* the significant shell t_in <= a r^2 <= t_out where (l/2) ln(t/t0) - (t - t0) >= ln(eps),
  t0 = l/2 (envelope within eps of its peak): the radial factor N exp(-t_out), r^l at t_out and
  the polynomial on the peak ray at t_in;
* the AO value an underflowing factor can drop: tiny * max(N, max_r N r^l exp(-a r^2)) and
  tiny * r_u^l at the radius where N exp(-a r_u^2) reaches tiny, against the strict ao_eval
  absolute tolerance;

with the limits of ``np.finfo`` of the ``ao_eval`` dtype. A primitive is representable when
every quantity is a normal finite value (the dropped values below the absolute tolerance).
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
from math import exp, lgamma, log, pi, sqrt

import h5py
import numpy as np
import pytest
from numpy.polynomial import legendre

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_dtype_np, get_tolerance, get_tolerance_min, precision_policy_snapshot
from jqmc.atomic_orbital import (
    AOEvalRangeError,
    AOs_cart_gto_data,
    AOs_cart_pgto_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    build_AOs_sphe_pgto_complex_data,
    check_ao_eval_value_range,
    compute_AOs,
)
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, _load_dataclass_from_hdf5
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
#: The committed basis fixtures of tests/trexio_example_files registered as GDD fixture sources
#: (TREXIO files and J3 Jastrow dumps). The one unregistered file there, the 4x4x4 Li twist grid,
#: carries the same Li basis as the registered 2x2x2 grid.
_FIXTURES = (
    "Ar_ae_ccpvdz_cart.h5",
    "Ar_ae_ccpvdz_sphe.h5",
    "Cl2_ecp_ccpvtz_cart.h5",
    "CuBr_ecp_ccpvtz_cart.h5",
    "H2_ae_ccpvdz_cart.h5",
    "H2_ae_ccpvdz_sphe.h5",
    "H2_ae_ccpvqz.h5",
    "H2_ae_ccpvtz_cart.h5",
    "H2_ecp_ccpvdz_cart.h5",
    "H2_ecp_ccpvtz.h5",
    "H2_ecp_ccpvtz_cart.h5",
    "H2_pbc_ae_cart.h5",
    "H2_pbc_ae_k025_cart.h5",
    "H2_pbc_ae_k025_sphe.h5",
    "H2_pbc_ae_sphe.h5",
    "H4_pbc_ae_cart.h5",
    "H4_pbc_ae_k025_cart.h5",
    "H_ae_ccpvdz_cart.h5",
    "H_ecp_ccpvdz_cart.h5",
    "H_ecp_ccpvqz.h5",
    "Li_ae_ccpvdz_cart.h5",
    "Li_ecp_ccpvdz_cart.h5",
    "Li_pbc_ae_ccpvdz_cart.h5",
    "Li_pbc_ae_ccpvdz_sphe.h5",
    "N2_ecp_ccpvtz_cart.h5",
    "N_ae_ccpvdz_cart.h5",
    "Ti2_ecp_ccpvtz_cart.h5",
    "lithium.h5",
    "lithium_cart.h5",
    "lithium_k025.h5",
    "lithium_k025_cart.h5",
    "lithium_twist_222.h5",
    "silicon.h5",
    "water_ccecp_ccpvqz.h5",
    "jastrow_data_w_1b_2b_1b3b_ae.h5",
    "jastrow_data_w_2b_1b3b_w_ecp.h5",
    "jastrow_data_w_2b_3b_w_ecp.h5",
)


def _ao_eval_is_narrow(zone="ao_eval"):
    """Whether the selected value zone is narrower than the float64 basis storage."""
    return np.finfo(get_dtype_np(zone)).bits < np.finfo(np.float64).bits


def _shell(l_tot):
    return [(nx, ny, l_tot - nx - ny) for nx in range(l_tot, -1, -1) for ny in range(l_tot - nx, -1, -1)]


# ----------------------------------------------------------------------------------------------
# Independent analytic reference (plain Python, no jqmc code)
# ----------------------------------------------------------------------------------------------


def _significant_shell(l_tot, ln_depth):
    """(t_in, t_out): a r^2 where the envelope r^l exp(-a r^2) is exp(-ln_depth) below its peak."""
    if l_tot == 0:
        return 0.0, ln_depth
    t0 = 0.5 * l_tot

    def drop(t):
        return t0 * log(t / t0) - (t - t0) + ln_depth

    def bisect(lo, hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (drop(mid) > 0) == (drop(lo) > 0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    hi = t0
    while drop(hi) > 0:
        hi *= 2.0
    return bisect(t0 * 1e-300, t0), bisect(t0, hi)


def _range_limits(zone="ao_eval"):
    info = np.finfo(get_dtype_np(zone))
    atol, _ = get_tolerance(zone, "strict")
    return log(float(info.tiny)), log(float(info.max)), -log(float(info.eps)), log(atol)


def _reference_violations(kind, comp, a, zone="ao_eval"):
    """Return the value-zone range violations of one primitive (empty = representable).

    ``kind`` is ``"cart"`` (``comp`` = (nx, ny, nz)) or ``"sphe"`` (``comp`` = l). ``a`` is the
    exponent rounded to the selected value-zone dtype, as the kernels use it.
    """
    ln_min, ln_max, ln_depth, ln_negl = _range_limits(zone)
    if kind == "cart":
        powers = comp
        l_tot = sum(powers)
        ln_n = -0.5 * sum(lgamma(k + 0.5) - (k + 0.5) * log(2.0 * a) for k in powers)
        ln_norms = [ln_n]
        ln_poly_peak = sum(0.5 * k * log(k / (2.0 * a)) for k in powers if k > 0)
    else:
        l_tot = comp
        powers = None
        ln_nn = 0.5 * (log(2.0) + (l_tot + 1.5) * log(2.0 * a) - lgamma(l_tot + 1.5))
        ln_n = ln_nn + 0.5 * log((2 * l_tot + 1) / (4.0 * pi))
        ln_norms = [ln_nn, ln_n]
        ln_poly_peak = 0.5 * l_tot * log(l_tot / (2.0 * a)) if l_tot > 0 else 0.0
    t_in, t_out = _significant_shell(l_tot, ln_depth)
    ln_peak = ln_n + ln_poly_peak - 0.5 * l_tot
    ln_env_peak = ln_n + (0.5 * l_tot * log(l_tot / (2.0 * a)) - 0.5 * l_tot if l_tot > 0 else 0.0)
    out = []
    if any(v < ln_min or v > ln_max for v in ln_norms):
        out.append("N")
    if ln_peak > ln_max or ln_peak - ln_depth < ln_min:
        out.append("peak value")
    if ln_n - t_out < ln_min:
        out.append("radial factor in shell")
    if l_tot > 0:
        if 0.5 * l_tot * log(t_out / a) > ln_max:
            out.append("r^l in shell")
        if kind == "cart":
            scale = log(t_in / (0.5 * l_tot))
            factors = [0.5 * k * (log(k / (2.0 * a)) + scale) for k in powers if k > 0]
            poly_in = min(min(factors), sum(factors))
        else:
            poly_in = 0.5 * l_tot * log(t_in / a)
        if poly_in < ln_min:
            out.append("polynomial in shell")
    if max(ln_n, ln_env_peak) + ln_min > ln_negl:
        out.append("dropped value (polynomial underflow)")
    t_u = ln_n - ln_min
    if l_tot > 0 and t_u > 0 and 0.5 * l_tot * log(max(t_u, t_out) / a) + ln_min > ln_negl:
        out.append("dropped value (radial underflow)")
    return out


def _stored(a, zone="ao_eval"):
    return float(get_dtype_np(zone)(a))


# ----------------------------------------------------------------------------------------------
# Builders and closed forms
# ----------------------------------------------------------------------------------------------


def _structure(pbc=False):
    if pbc:
        cell = 12.0
        return Structure_data.build(
            positions=np.array([[0.5 * cell] * 3]),
            pbc_flag=True,
            atomic_numbers=(7,),
            element_symbols=("N",),
            atomic_labels=("N1",),
            vec_a=(cell, 0.0, 0.0),
            vec_b=(0.0, cell, 0.0),
            vec_c=(0.0, 0.0, cell),
        )
    return Structure_data.build(
        positions=np.zeros((1, 3)), pbc_flag=False, atomic_numbers=(7,), element_symbols=("N",), atomic_labels=("N1",)
    )


def _cart_kwargs(comps, a, structure):
    k = len(comps)
    return dict(
        structure_data=structure,
        nucleus_index=(0,) * k,
        num_ao=k,
        num_ao_prim=k,
        orbital_indices=tuple(range(k)),
        exponents=np.full(k, a, dtype=np.float64),
        coefficients=np.ones(k, dtype=np.float64),
        angular_momentums=tuple(sum(n) for n in comps),
        polynominal_order_x=tuple(n[0] for n in comps),
        polynominal_order_y=tuple(n[1] for n in comps),
        polynominal_order_z=tuple(n[2] for n in comps),
    )


def _sphe_kwargs(l_tot, a, structure, ms=None):
    ms = tuple(range(-l_tot, l_tot + 1)) if ms is None else tuple(ms)
    k = len(ms)
    return dict(
        structure_data=structure,
        nucleus_index=(0,) * k,
        num_ao=k,
        num_ao_prim=k,
        orbital_indices=tuple(range(k)),
        exponents=np.full(k, a, dtype=np.float64),
        coefficients=np.ones(k, dtype=np.float64),
        angular_momentums=(l_tot,) * k,
        magnetic_quantum_numbers=ms,
    )


def _cart_value(n, a, p):
    """c N x^nx y^ny z^nz exp(-a r^2), c = 1 (no jqmc code)."""
    ln_n = -0.5 * sum(lgamma(k + 0.5) - (k + 0.5) * log(2.0 * a) for k in n)
    val = exp(ln_n - a * float(p @ p))
    for xi, k in zip(p, n):
        val *= float(xi) ** k
    return val


def _sphe_m0_value(l_tot, a, p):
    """N_n sqrt((2l+1)/(4 pi)) r^l P_l(cos theta) exp(-a r^2) (Racah solid harmonic, no jqmc code)."""
    ln_n = 0.5 * (log(2.0) + (l_tot + 1.5) * log(2.0 * a) - lgamma(l_tot + 1.5))
    r = float(np.linalg.norm(p))
    val = exp(ln_n - a * r * r) * sqrt((2 * l_tot + 1) / (4.0 * pi))
    if l_tot == 0:
        return val
    return val * r**l_tot * float(legendre.legval(float(p[2]) / r, [0.0] * l_tot + [1.0]))


# ----------------------------------------------------------------------------------------------
# 1. Out-of-range bases are rejected loudly in mixed precision (never in full)
# ----------------------------------------------------------------------------------------------

#: case -> (kind, components / l, exponent, periodic). The x^82 example of #111, the first
#: failing degree at a = 1 after #115 (l = 53), the l = 8 complete shell above its post-#115
#: window (a <= 5.6e7), diffuse high l, and spherical shells whose N exceeds float32 (molecular
#: and periodic, l <= 6), plus a periodic Cartesian shell.
_REJECT_CASES = {
    "cart_x82_a1": ("cart", [(82, 0, 0)], 1.0, False),
    "cart_axis_l53_a1": ("cart", [(53, 0, 0)], 1.0, False),
    "cart_shell_l8_a1e8": ("cart", _shell(8), 1.0e8, False),
    "cart_axis_l24_a1e-3": ("cart", [(24, 0, 0)], 1.0e-3, False),
    "cart_pgto_shell_l8_a1e9": ("cart", _shell(8), 1.0e9, True),
    "sphe_l6_a1e11": ("sphe", 6, 1.0e11, False),
    "sphe_l4_a1e15": ("sphe", 4, 1.0e15, False),
    "sphe_pgto_l6_a1e11": ("sphe", 6, 1.0e11, True),
}


def _build_leaf(kind, comps, a, periodic):
    structure = _structure(pbc=periodic)
    if kind == "cart":
        cls = AOs_cart_pgto_data if periodic else AOs_cart_gto_data
        return cls.build(**_cart_kwargs(comps, a, structure))
    cls = AOs_sphe_pgto_data if periodic else AOs_sphe_gto_data
    return cls.build(**_sphe_kwargs(comps, a, structure))


def _build_orbitals(aos):
    """Configure the leaf into a one-MO orbital set (validated by ``MOs_data.build``)."""
    return MOs_data.build(aos_data=aos, mo_coefficients=np.ones((1, aos.num_ao)))


@pytest.mark.light
@pytest.mark.parametrize("case", list(_REJECT_CASES))
def test_ao_eval_range_precondition_rejects_unrepresentable_mixed_basis(case):
    """Mixed precision rejects an AO basis whose float32 factors leave the range; full builds it.

    The independent analytic reference must find at least one float32-range violation for every
    case in mixed precision. In mixed precision configuring the AO leaf into an orbital set
    (``MOs_data.build``) and ``check_ao_eval_value_range`` must raise ``AOEvalRangeError`` naming the atom, l, component and
    exponent of an offending primitive and suggesting full precision; before the precondition
    it returned a leaf whose AO values were silently wrong (Inf, NaN, 0 or rescaled). In full
    precision the same leaf builds and validates and ``check_ao_eval_value_range`` is a no-op.
    """
    kind, comps, a, periodic = _REJECT_CASES[case]
    if _ao_eval_is_narrow():
        refs = (
            [_reference_violations("cart", n, _stored(a)) for n in comps]
            if kind == "cart"
            else [_reference_violations("sphe", comps, _stored(a))]
        )
        assert any(refs), f"{case}: the analytic reference finds the primitive representable"
        aos = _build_leaf(kind, comps, a, periodic)
        with pytest.raises(AOEvalRangeError):
            check_ao_eval_value_range(aos)
        with pytest.raises(AOEvalRangeError) as excinfo:
            _build_orbitals(aos)
        message = str(excinfo.value)
        l_tot = sum(comps[0]) if kind == "cart" else comps
        assert "atom N1" in message
        assert f"l={l_tot}," in message
        assert f"exponent={_stored(a):.6g}" in message
        assert "full" in message
        assert isinstance(excinfo.value, ValueError)
    else:
        aos = _build_leaf(kind, comps, a, periodic)
        check_ao_eval_value_range(aos)
        _build_orbitals(aos)


def _x82_hamiltonian():
    """The #111 driver model: bare Z = 7 nucleus, one up electron, single AO x^82 exp(-r^2).

    Assembled from directly constructed dataclasses (no ``build`` validation), so that the
    driver constructor is the first place where the basis is validated.
    """
    structure = _structure()
    aos = AOs_cart_gto_data._from_full_basis(**_cart_kwargs([(82, 0, 0)], 1.0, structure))
    up = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=np.ones((1, 1)))
    dn = MOs_data(num_mo=0, aos_data=aos, mo_coefficients=np.empty((0, 1)))
    geminal = SD_data(num_electron_up=1, num_electron_dn=0, orb_data_up_spin=up, orb_data_dn_spin=dn)
    return Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=Coulomb_potential_data(structure_data=structure),
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
    )


@pytest.mark.light
def test_ao_eval_range_precondition_guards_driver_and_j3_setup():
    """The x^82 example is rejected at MCMC construction and as a J3 basis in mixed precision.

    In mixed precision the MCMC constructor (which validates the Hamiltonian after the precision
    mode is configured, and serves the XLA and Rust backends alike) must raise
    ``AOEvalRangeError`` instead of producing the silently wrong local energies of the x^82
    driver probe, and a J3 factor on the same basis must fail its ``sanity_check``. In full
    precision both validate.
    """
    hamiltonian = _x82_hamiltonian()
    aos = hamiltonian.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos)
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    if _ao_eval_is_narrow():
        with pytest.raises(AOEvalRangeError):
            MCMC(hamiltonian_data=hamiltonian, num_walkers=2, mcmc_seed=34, Dt=0.3, epsilon_AS=0.0)
        with pytest.raises(AOEvalRangeError):
            j3.sanity_check()
        with pytest.raises(AOEvalRangeError):
            hamiltonian.sanity_check()
    else:
        hamiltonian.sanity_check()
        j3.sanity_check()
        MCMC(hamiltonian_data=hamiltonian, num_walkers=2, mcmc_seed=34, Dt=0.3, epsilon_AS=0.0)


# ----------------------------------------------------------------------------------------------
# 2. No false positives: every committed basis fixture validates in both modes
# ----------------------------------------------------------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize("fixture", _FIXTURES)
def test_ao_eval_range_precondition_accepts_committed_fixtures(fixture):
    """Every committed TREXIO / Jastrow basis fixture loads and validates in full and mixed.

    The realistic bases (all-electron up to Ar with s exponents ~1.5e5, ECP bases up to g
    functions, periodic Gamma / twisted / multi-k, Cartesian and spherical) must not trip the
    precondition: the TREXIO reader (which validates its AO leaf), the MO / geminal
    ``sanity_check`` and an explicit ``check_ao_eval_value_range`` all succeed, and the
    independent analytic reference finds every primitive representable.
    """
    path = os.path.join(_FIXTURE_DIR, fixture)
    if fixture.startswith("jastrow_data_"):
        with h5py.File(path, "r") as f:
            jastrow = _load_dataclass_from_hdf5(Jastrow_data, f["jastrow_data"])
        jastrow.sanity_check()
        orb = jastrow.jastrow_three_body_data.orb_data
        aos = orb.aos_data if isinstance(orb, MOs_data) else orb
    else:
        _, aos, mos_up, mos_dn, geminal, _ = read_trexio_file(trexio_file=path, store_tuple=True)
        mos_up.sanity_check()
        mos_dn.sanity_check()
        geminal.sanity_check()
    check_ao_eval_value_range(aos)
    is_cart = isinstance(aos, AOs_cart_gto_data)
    ao_of_prim = np.asarray(aos.orbital_indices)
    seen = set()
    for p, a in enumerate(np.asarray(aos.exponents)):
        ao = int(ao_of_prim[p])
        if is_cart:
            key = ("cart", (aos.polynominal_order_x[ao], aos.polynominal_order_y[ao], aos.polynominal_order_z[ao]))
        else:
            key = ("sphe", int(aos.angular_momentums[ao]))
        key = key + (_stored(a),)
        if key in seen:
            continue
        seen.add(key)
        assert _reference_violations(*key) == [], f"{fixture}: reference finds {key} out of range"


# ----------------------------------------------------------------------------------------------
# 3. Either rejected or correct: accepted bases near the range boundary evaluate within tolerance
# ----------------------------------------------------------------------------------------------

#: family -> (kind, components or l, exponent grid). Exponent grids straddle the mixed bound.
_BOUNDARY_FAMILIES = {
    "cart_shell_l0": ("cart", _shell(0), [10.0**k for k in range(-4, 15)]),
    "cart_shell_l2": ("cart", _shell(2), [10.0**k for k in range(-4, 15)]),
    "cart_shell_l4": ("cart", _shell(4), [10.0**k for k in range(-4, 15)]),
    "cart_shell_l6": ("cart", _shell(6), [10.0**k for k in range(-4, 15)]),
    "cart_shell_l8": ("cart", _shell(8), [10.0 ** (0.5 * k) for k in range(-8, 21)]),
    "cart_axis_l13": ("cart", [(13, 0, 0)], [10.0**k for k in range(-4, 9)]),
    "cart_axis_l20": ("cart", [(20, 0, 0)], [10.0 ** (0.5 * k) for k in range(-6, 9)]),
    "cart_axis_l28": ("cart", [(28, 0, 0)], [10.0 ** (0.5 * k) for k in range(-6, 7)]),
    "cart_axis_l40": ("cart", [(40, 0, 0)], [10.0 ** (0.5 * k) for k in range(-6, 7)]),
    "cart_axis_l53": ("cart", [(53, 0, 0)], [10.0 ** (0.5 * k) for k in range(-6, 7)]),
    "sphe_l2": ("sphe", 2, [10.0**k for k in range(4, 17)]),
    "sphe_l4": ("sphe", 4, [10.0**k for k in range(4, 17)]),
    "sphe_l6": ("sphe", 6, [10.0 ** (0.5 * k) for k in range(8, 31)]),
}


def _shell_points(l_tot, a, direction_sets, ln_depth):
    """Points across the significant shell t_in <= a r^2 <= t_out and at a r^2 = 2 t_out, on several directions."""
    t_in, t_out = _significant_shell(l_tot, ln_depth)
    ts = list(np.linspace(t_in, t_out, 7)) + [2.0 * t_out]
    pts = []
    for t in ts:
        for d in direction_sets:
            u = np.asarray(d, dtype=np.float64)
            pts.append(sqrt(t / a) * u / np.linalg.norm(u))
    return np.array(pts)


@pytest.mark.light
@pytest.mark.parametrize("family", list(_BOUNDARY_FAMILIES))
def test_ao_eval_range_precondition_accepted_basis_is_evaluated_within_tolerance(family):
    """A basis is either rejected loudly (mixed only) or evaluated within the ao_eval tolerance.

    For each exponent of the grid (straddling the mixed-precision bound), configuring the leaf
    into an orbital set (``MOs_data.build``) must raise
    ``AOEvalRangeError`` exactly when the independent analytic reference finds a float32-range
    violation (never in full precision). Every accepted basis is evaluated with XLA
    ``compute_AOs`` and Rust ``jqmc_rs.compute_AOs`` at points across the significant shell
    and out to a r^2 = 2 t_out, on the peak ray, near-axis and generic directions, and must be finite
    and equal the closed form within the strict ao_eval tolerance of the active mode.
    """
    kind, comps, grid = _BOUNDARY_FAMILIES[family]
    narrow = _ao_eval_is_narrow()
    ln_min, ln_max, ln_depth, _ = _range_limits()
    atol, rtol = get_tolerance("ao_eval", "strict")
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    structure = _structure()
    accepted = 0
    for a in grid:
        if kind == "cart":
            refs = [_reference_violations("cart", n, _stored(a)) for n in comps]
            l_tot = sum(comps[0])
        else:
            refs = [_reference_violations("sphe", comps, _stored(a))]
            l_tot = comps
        expect_reject = narrow and any(refs)
        if kind == "cart":
            aos = AOs_cart_gto_data.build(**_cart_kwargs(comps, a, structure))
        else:
            aos = AOs_sphe_gto_data.build(**_sphe_kwargs(comps, a, structure, ms=(0,)))
        try:
            _build_orbitals(aos)
        except AOEvalRangeError:
            assert expect_reject, f"{family} a={a:.3g}: rejected but the reference finds it representable"
            continue
        assert not expect_reject, f"{family} a={a:.3g}: accepted but the reference finds {refs}"
        accepted += 1
        if kind == "cart":
            peak_dirs = [tuple(sqrt(k) for k in n) for n in comps if sum(n) > 0][:1] or [(1.0, 0.0, 0.0)]
            dirs = peak_dirs + [(1.0, 0.2, 0.1), (1.0, 1e-2, 1e-2), (1e-2, 1.0, 0.3)]
            pts = _shell_points(l_tot, a, dirs, ln_depth)
            expected = np.array([[_cart_value(n, a, p) for p in pts] for n in comps])
        else:
            dirs = [(0.0, 0.0, 1.0), (0.3, 0.2, 1.0), (1.0, 1e-2, 1e-2)]
            pts = _shell_points(l_tot, a, dirs, ln_depth)
            expected = np.array([[_sphe_m0_value(comps, a, p) for p in pts]])
        assert np.all(np.isfinite(expected))
        for what, actual in (
            ("XLA compute_AOs", compute_AOs(aos, pts)),
            ("Rust compute_AOs", jqmc_rs.compute_AOs(aos, pts)),
        ):
            actual = np.asarray(actual, dtype=np.float64)
            assert actual.shape == expected.shape, what
            assert np.all(np.isfinite(actual)), f"{family} a={a:.3g}: {what} non-finite"
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"{family} a={a:.3g}: {what}")
    # The boundary grids always contain representable exponents.
    assert accepted > 0


# ----------------------------------------------------------------------------------------------
# 4. Either rejected or correct for periodic and complex-Bloch spherical leaves (after PR #123)
# ----------------------------------------------------------------------------------------------

#: Cubic cell edge (bohr) of the periodic spherical leaves. The atom sits at the origin, so the
#: displacement of each evaluation point is exact in float64; the grids start at a = 1e2, where
#: every lattice image is below the float64 range at the evaluation points.
_PBC_SPHE_CELL = 40.0
#: Fractional twist of the complex-Bloch leaf.
_PBC_SPHE_TWIST = (0.25, 0.0, 0.0)

#: family -> (l, exponent grid). Exponent grids straddle the mixed bound.
_PBC_SPHE_FAMILIES = {
    "l2": (2, [10.0**k for k in range(2, 20)]),
    "l4": (4, [10.0 ** (0.5 * k) for k in range(8, 33)]),
    "l6": (6, [10.0 ** (0.5 * k) for k in range(6, 31)]),
}


@pytest.mark.light
@pytest.mark.parametrize("leaf", ["periodic", "complex"])
@pytest.mark.parametrize("family", list(_PBC_SPHE_FAMILIES))
def test_ao_eval_range_precondition_periodic_spherical_basis_rejected_or_correct(family, leaf):
    """A periodic or complex-Bloch spherical basis is either rejected loudly (mixed only) or evaluated within tolerance.

    Same contract as the molecular boundary test, on one m = 0 primitive of an
    ``AOs_sphe_pgto_data`` leaf or a complex-Bloch leaf (``build_AOs_sphe_pgto_complex_data``,
    twist (1/4, 0, 0), complex MO coefficients). Configuring it into an orbital set
    (``MOs_data.build``) must raise ``AOEvalRangeError`` exactly when the independent analytic
    reference finds a range violation in the leaf's value zone (never in full precision).
    Complex Bloch pGTO values use ao_bloch_eval, which remains float64 in mixed;
    real periodic pGTO values use ao_eval. Every accepted basis
    is evaluated with XLA ``compute_AOs`` and Rust ``jqmc_rs.compute_AOs`` across the
    significant shell and at a r^2 = 2 t_out on three directions, and must be finite and equal
    the closed form within the strict ao_eval tolerance of the active mode (the complex value
    with a vanishing imaginary part: only the home image contributes). The normalization N_n of a periodic leaf is assembled in
    float64, so an accepted periodic leaf never returns inf in mixed.
    """
    l_tot, grid = _PBC_SPHE_FAMILIES[family]
    # Select the expected zone independently of the production range checker.
    zone = "ao_bloch_eval" if leaf == "complex" else "ao_eval"
    narrow = _ao_eval_is_narrow(zone)
    # Retain the original shell sample points and AO comparison tolerance.
    _, _, ln_depth, _ = _range_limits()
    atol, rtol = get_tolerance("ao_eval", "strict")
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    cell = _PBC_SPHE_CELL
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=True,
        atomic_numbers=(7,),
        element_symbols=("N",),
        atomic_labels=("N1",),
        vec_a=(cell, 0.0, 0.0),
        vec_b=(0.0, cell, 0.0),
        vec_c=(0.0, 0.0, cell),
    )
    mo_dtype = np.complex128 if leaf == "complex" else np.float64
    accepted = 0
    for a in grid:
        expect_reject = narrow and bool(_reference_violations("sphe", l_tot, _stored(a, zone), zone))
        kwargs = _sphe_kwargs(l_tot, a, structure, ms=(0,))
        if leaf == "periodic":
            aos = AOs_sphe_pgto_data.build(**kwargs)
        else:
            kwargs.pop("structure_data")
            aos = build_AOs_sphe_pgto_complex_data(structure, k_point=_PBC_SPHE_TWIST, **kwargs)
        try:
            MOs_data.build(aos_data=aos, mo_coefficients=np.ones((1, aos.num_ao), dtype=mo_dtype))
        except AOEvalRangeError:
            assert expect_reject, f"{leaf} {family} a={a:.3g}: rejected but the reference finds it representable"
            continue
        assert not expect_reject, f"{leaf} {family} a={a:.3g}: accepted but the reference finds a violation"
        accepted += 1
        pts = _shell_points(l_tot, a, [(0.0, 0.0, 1.0), (0.3, 0.2, 1.0), (1.0, 1e-2, 1e-2)], ln_depth)
        expected = np.array([[_sphe_m0_value(l_tot, a, p) for p in pts]])
        assert np.all(np.isfinite(expected))
        for what, actual in (
            ("XLA compute_AOs", compute_AOs(aos, pts)),
            ("Rust compute_AOs", jqmc_rs.compute_AOs(aos, pts)),
        ):
            actual = np.asarray(actual)
            assert actual.shape == expected.shape, what
            if what == "XLA compute_AOs":
                # The native real-value export promotes its result to float64;
                # its return dtype does not describe its arithmetic precision.
                expected_dtype = np.complex128 if leaf == "complex" else get_dtype_np(zone)
                assert actual.dtype == np.dtype(expected_dtype), f"{leaf} {family}: {what} value-zone dtype"
            assert np.all(np.isfinite(actual)), f"{leaf} {family} a={a:.3g}: {what} non-finite"
            if leaf == "complex":
                np.testing.assert_allclose(
                    actual.imag, 0.0, atol=atol, rtol=rtol, err_msg=f"{leaf} {family} a={a:.3g}: {what} imag"
                )
                actual = actual.real
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float64),
                expected,
                atol=atol,
                rtol=rtol,
                err_msg=f"{leaf} {family} a={a:.3g}: {what}",
            )
    # The boundary grids always contain representable exponents.
    assert accepted > 0


# ----------------------------------------------------------------------------------------------
# 5. Contraction coefficients: the kernels multiply c into the radial product
# ----------------------------------------------------------------------------------------------
#
# Every value kernel forms the radial factor with the contraction coefficient c before the
# angular polynomial: exp(-a r^2), then c exp(-a r^2), then N (c exp(-a r^2)) (or the
# prefactor N c rounded once, then (N c) exp(-a r^2)), summed over the primitives of the AO.
# A finite c and a finite final AO value do not make these partial products representable:
# for x^53 at a = 100 and c = 1e15, N c exp(-9) overflows float32 at a r^2 = 9, inside the
# significant shell, while c phi = 2.8e11 there.


def _reference_coefficient_violations(kind, comp, a, c):
    """Float32-range violations of one primitive with contraction coefficient ``c`` (no jqmc code).

    The unit-coefficient conditions of ``_reference_violations`` are kept; this adds, from the
    same lgamma normalization and the bisected significant shell: c itself, the radial product
    |N c| (its value at r = 0), the peak |c| max_r |phi| and |c| exp(-ln eps) below it, the
    radial partial products exp(-t), |c| exp(-t) and |N c| exp(-t) inside the shell, and the
    AO value dropped where the polynomial (l > 0) or the first radial partial product
    underflows, against the strict ao_eval absolute tolerance. ``c = 0`` contributes nothing.
    """
    out = list(_reference_violations(kind, comp, a))
    if c == 0.0:
        return out
    ln_min, ln_max, ln_depth, ln_negl = _range_limits()
    ln_c = log(abs(c))
    if kind == "cart":
        l_tot = sum(comp)
        ln_n = -0.5 * sum(lgamma(k + 0.5) - (k + 0.5) * log(2.0 * a) for k in comp)
        ln_poly_peak = sum(0.5 * k * log(k / (2.0 * a)) for k in comp if k > 0)
    else:
        l_tot = comp
        ln_nn = 0.5 * (log(2.0) + (l_tot + 1.5) * log(2.0 * a) - lgamma(l_tot + 1.5))
        ln_n = ln_nn + 0.5 * log((2 * l_tot + 1) / (4.0 * pi))
        ln_poly_peak = 0.5 * l_tot * log(l_tot / (2.0 * a)) if l_tot > 0 else 0.0
    _, t_out = _significant_shell(l_tot, ln_depth)
    ln_nc = ln_n + ln_c
    ln_peak = ln_n + ln_poly_peak - 0.5 * l_tot + ln_c
    ln_env_peak = ln_n + ln_c + (0.5 * l_tot * log(l_tot / (2.0 * a)) - 0.5 * l_tot if l_tot > 0 else 0.0)
    if ln_c < ln_min or ln_c > ln_max:
        out.append("coefficient")
    if ln_nc > ln_max:
        out.append("radial product N c")
    if ln_peak > ln_max or ln_peak - ln_depth < ln_min:
        out.append("peak value c phi")
    if min(0.0, ln_c, ln_nc) - t_out < ln_min:
        out.append("radial partial product in shell")
    if l_tot > 0 and max(ln_nc, ln_env_peak) + ln_min > ln_negl:
        out.append("dropped c phi (polynomial underflow)")
    t_u = max(min(-ln_min, ln_c - ln_min, ln_nc - ln_min), t_out)
    if ln_nc + 0.5 * l_tot * log(t_u / a) - t_u > ln_negl:
        out.append("dropped c phi (radial underflow)")
    return out


def _pbc_sphe_structure():
    cell = _PBC_SPHE_CELL
    return Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=True,
        atomic_numbers=(7,),
        element_symbols=("N",),
        atomic_labels=("N1",),
        vec_a=(cell, 0.0, 0.0),
        vec_b=(0.0, cell, 0.0),
        vec_c=(0.0, 0.0, cell),
    )


def _coefficient_leaf(kind, comps, a, coefficients, periodic=False):
    """One AO per primitive (Cartesian components, or spherical m = 0), with the given coefficients."""
    if kind == "cart":
        kwargs = _cart_kwargs(comps, a, _structure())
        kwargs["coefficients"] = np.asarray(coefficients, dtype=np.float64)
        return AOs_cart_gto_data.build(**kwargs)
    kwargs = _sphe_kwargs(comps, a, _pbc_sphe_structure() if periodic else _structure(), ms=(0,))
    kwargs["coefficients"] = np.asarray(coefficients, dtype=np.float64)
    return (AOs_sphe_pgto_data if periodic else AOs_sphe_gto_data).build(**kwargs)


def _unit_values(kind, comps, a, pts):
    """Closed-form unit-coefficient AO values, shape (num_ao, n_pts) (no jqmc code)."""
    if kind == "cart":
        return np.array([[_cart_value(n, a, p) for p in pts] for n in comps])
    return np.array([[_sphe_m0_value(comps, a, p) for p in pts]])


#: case -> (kind, components or l, exponent, coefficient, periodic). The x^53 witness
#: (a = 100, c = 1e15), the same AO contracted with an ordinary c = 1 primitive at a = 50, and
#: spherical l = 4 at a = 1e6 with c = 1e25 (N c overflows; N and c phi are representable),
#: molecular and periodic.
_CONTRACTED_PARTNER_EXPONENT = 50.0
_COEFFICIENT_REJECT_CASES = {
    "cart_x53_a1e2_c1e15": ("cart", [(53, 0, 0)], 100.0, 1.0e15, False),
    "cart_x53_contracted_c1e15": ("cart", [(53, 0, 0)], 100.0, 1.0e15, False),
    "sphe_l4_a1e6_c1e25": ("sphe", 4, 1.0e6, 1.0e25, False),
    "sphe_pgto_l4_a1e6_c1e25": ("sphe", 4, 1.0e6, 1.0e25, True),
}


@pytest.mark.light
@pytest.mark.parametrize("case", list(_COEFFICIENT_REJECT_CASES))
def test_ao_eval_range_precondition_rejects_overflowing_coefficient_radial_product(case):
    """A non-unit contraction coefficient that overflows the float32 radial product is rejected.

    The witness bases have a representable normalization, coefficient and final AO value, and
    pass every unit-coefficient condition; only the radial product N c exp(-a r^2), formed by
    the kernels before the angular polynomial, leaves float32 inside the significant shell.
    In mixed precision the independent reference must flag the coefficient, and both
    ``check_ao_eval_value_range`` and configuring the leaf into an orbital set
    (``MOs_data.build``, MO coefficient 1/c) must raise ``AOEvalRangeError``; without the
    coefficient-aware precondition both would accept it and XLA / Rust ``compute_AOs`` and
    ``compute_MOs`` would return inf. In full precision the same orbital set builds and the AO
    values (XLA and Rust) and the MO values equal c phi and phi of the closed form at the witness
    point (0.3, 0, 0) and across the significant shell.
    """
    from jqmc.molecular_orbital import compute_MOs

    kind, comps, a, c, periodic = _COEFFICIENT_REJECT_CASES[case]
    contracted = case.endswith("contracted_c1e15")
    l_tot = sum(comps[0]) if kind == "cart" else comps
    comp = comps[0] if kind == "cart" else comps
    assert _reference_violations(kind, comp, _stored(a)) == []
    if contracted:
        # Two primitives in one AO: the witness and an ordinary (a = 50, c = 1) primitive.
        kwargs = _cart_kwargs(comps, a, _structure())
        kwargs.update(
            num_ao_prim=2,
            orbital_indices=(0, 0),
            exponents=np.array([a, _CONTRACTED_PARTNER_EXPONENT]),
            coefficients=np.array([c, 1.0]),
        )
        aos = AOs_cart_gto_data.build(**kwargs)
        assert _reference_coefficient_violations(kind, comp, _stored(_CONTRACTED_PARTNER_EXPONENT), 1.0) == []
    else:
        aos = _coefficient_leaf(kind, comps, a, [c], periodic)
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    mo_coefficients = np.full((1, aos.num_ao), 1.0 / c)
    if _ao_eval_is_narrow():
        assert _reference_coefficient_violations(kind, comp, _stored(a), c), f"{case}: reference finds it representable"
        with pytest.raises(AOEvalRangeError) as excinfo:
            check_ao_eval_value_range(aos)
        assert f"l={l_tot}," in str(excinfo.value)
        assert "radial product N c" in str(excinfo.value)
        with pytest.raises(AOEvalRangeError):
            MOs_data.build(aos_data=aos, mo_coefficients=mo_coefficients)
        return
    check_ao_eval_value_range(aos)
    mos = MOs_data.build(aos_data=aos, mo_coefficients=mo_coefficients)
    _, _, ln_depth, _ = _range_limits()
    if kind == "cart":
        # The witness point (0.3, 0, 0), a r^2 = 9, and the shell on the x axis.
        pts = np.vstack([np.array([[0.3, 0.0, 0.0]]), _shell_points(l_tot, a, [(1.0, 0.0, 0.0)], ln_depth)])
    else:
        pts = _shell_points(l_tot, a, [(0.0, 0.0, 1.0), (0.3, 0.2, 1.0)], ln_depth)
    unit = _unit_values(kind, comps, a, pts)
    expected_ao = c * unit
    if contracted:
        expected_ao = expected_ao + _unit_values(kind, comps, _CONTRACTED_PARTNER_EXPONENT, pts)
    atol, rtol = get_tolerance("ao_eval", "strict")
    for what, actual in (("XLA compute_AOs", compute_AOs(aos, pts)), ("Rust compute_AOs", jqmc_rs.compute_AOs(aos, pts))):
        actual = np.asarray(actual, dtype=np.float64)
        assert np.all(np.isfinite(actual)), f"{case}: {what} non-finite"
        np.testing.assert_allclose(actual, expected_ao, atol=atol, rtol=rtol, err_msg=f"{case}: {what}")
    mo_atol, mo_rtol = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
    mo = np.asarray(compute_MOs(mos, pts), dtype=np.float64)
    assert np.all(np.isfinite(mo)), f"{case}: compute_MOs non-finite"
    np.testing.assert_allclose(mo, expected_ao / c, atol=mo_atol, rtol=mo_rtol, err_msg=f"{case}: compute_MOs")


#: family -> (kind, components or l, exponent, periodic). Each family scans the contraction
#: coefficient c = 1e-40 ... 1e40 (_COEFFICIENT_GRID) on a basis that is representable at c = 1.
_COEFFICIENT_FAMILIES = {
    "cart_axis_l53_a1e2": ("cart", [(53, 0, 0)], 100.0, False),
    "cart_shell_l0_a1e4": ("cart", _shell(0), 1.0e4, False),
    "cart_shell_l2_a1": ("cart", _shell(2), 1.0, False),
    "cart_shell_l6_a1e-2": ("cart", _shell(6), 1.0e-2, False),
    "sphe_l0_a1": ("sphe", 0, 1.0, False),
    "sphe_l4_a1e6": ("sphe", 4, 1.0e6, False),
    "sphe_pgto_l4_a1e6": ("sphe", 4, 1.0e6, True),
}
_COEFFICIENT_GRID = [10.0 ** (5 * k) for k in range(-8, 9)] + [-3.0e12, 2.5e-33]


@pytest.mark.light
@pytest.mark.parametrize("family", list(_COEFFICIENT_FAMILIES))
def test_ao_eval_range_precondition_coefficient_scan_rejected_or_correct(family):
    """Over a contraction-coefficient scan a basis is either rejected loudly (mixed only) or evaluated correctly.

    For each c of the scan (1e-40 ... 1e40, plus a negative and a near-tiny value), configuring
    the leaf into an orbital set (``MOs_data.build``, MO coefficients 1/c) must raise
    ``AOEvalRangeError`` exactly when the independent coefficient-aware reference finds a
    float32-range violation (never in full precision). Every accepted basis is evaluated with
    XLA ``compute_AOs`` and Rust ``jqmc_rs.compute_AOs`` across the significant shell and at
    a r^2 = 2 t_out on several directions, and must be finite and equal c phi of the closed
    form within the strict ao_eval tolerance; the production MO consumer ``compute_MOs``
    (MO = AO / c) must equal phi in AO units, i.e. c MO = c phi within the ao_eval / mo_eval
    tolerance. The scan contains accepted and (in mixed) rejected coefficients on both sides.
    """
    from jqmc.molecular_orbital import compute_MOs

    kind, comps, a, periodic = _COEFFICIENT_FAMILIES[family]
    narrow = _ao_eval_is_narrow()
    _, _, ln_depth, _ = _range_limits()
    atol, rtol = get_tolerance("ao_eval", "strict")
    mo_atol, mo_rtol = get_tolerance_min(["ao_eval", "mo_eval"], "strict")
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    if kind == "cart":
        l_tot = sum(comps[0])
        components = comps
        peak_dirs = [tuple(sqrt(k) for k in n) for n in comps if sum(n) > 0][:1] or [(1.0, 0.0, 0.0)]
        dirs = peak_dirs + [(1.0, 0.2, 0.1), (1e-2, 1.0, 0.3)]
    else:
        l_tot = comps
        components = [comps]
        dirs = [(0.0, 0.0, 1.0), (0.3, 0.2, 1.0), (1.0, 1e-2, 1e-2)]
    pts = _shell_points(l_tot, a, dirs, ln_depth)
    unit = _unit_values(kind, comps, a, pts)
    assert np.all(np.isfinite(unit))
    assert not any(_reference_violations(kind, n, _stored(a)) for n in components)
    accepted = rejected = 0
    for c in _COEFFICIENT_GRID:
        refs = [_reference_coefficient_violations(kind, n, _stored(a), c) for n in components]
        expect_reject = narrow and any(refs)
        aos = _coefficient_leaf(kind, comps, a, [c] * len(components), periodic)
        try:
            mos = MOs_data.build(aos_data=aos, mo_coefficients=np.eye(aos.num_ao) / c)
        except AOEvalRangeError:
            assert expect_reject, f"{family} c={c:.3g}: rejected but the reference finds it representable"
            rejected += 1
            continue
        assert not expect_reject, f"{family} c={c:.3g}: accepted but the reference finds {refs}"
        accepted += 1
        expected = c * unit
        for what, actual in (("XLA compute_AOs", compute_AOs(aos, pts)), ("Rust compute_AOs", jqmc_rs.compute_AOs(aos, pts))):
            actual = np.asarray(actual, dtype=np.float64)
            assert actual.shape == expected.shape, what
            assert np.all(np.isfinite(actual)), f"{family} c={c:.3g}: {what} non-finite"
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"{family} c={c:.3g}: {what}")
        mo = np.asarray(compute_MOs(mos, pts), dtype=np.float64)
        assert np.all(np.isfinite(mo)), f"{family} c={c:.3g}: compute_MOs non-finite"
        np.testing.assert_allclose(c * mo, expected, atol=mo_atol, rtol=mo_rtol, err_msg=f"{family} c={c:.3g}: compute_MOs")
    assert accepted > 0
    assert rejected > 0 or not narrow
