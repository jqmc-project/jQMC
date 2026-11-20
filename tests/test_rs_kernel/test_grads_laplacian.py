"""Layer-1 equivalence: analytic grad/Laplacian of ln|det G| and the Jastrow
part, plus the assembled kinetic energy, Rust vs jqmc. Foundations of e_L.

Fixtures are the committed hams. The (SD/AGP, sphe/cart, AE/ECP, jnone/j12/
j123ao) axes are mapped to committed patterns flowing the SAME code paths:
  - H2 ae sphe jnone            -> mol_sphe_sd_jnone
  - N2 ecp cart jnone           -> mol_ecp_cart_jnone (proxy: ECP+cart+SD)
  - water ecp sphe j12          -> mol_ecp_sphe_j12
  - water ecp agp sphe jnone    -> mol_sphe_agp_jnone (AGP coverage; kinetic
    energy does not touch the ECP potential at all, so AE-vs-ECP is
    immaterial for this quantity -- only the AGP determinant branch matters)
  - N2/H2/water *_j123ao        -> mol_cart_sd_j123 (already carries an ACTIVE
    pgto-basis J3, see _rs_fixtures) for the AE case, and a freshly-attached
    random J3 on the ECP
    cart/sphe patterns for ECP+J3 coverage.
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

import numpy as np
import pytest

import jqmc_rs

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.determinant import compute_grads_and_laplacian_ln_Det
from jqmc.jastrow_factor import (
    Jastrow_three_body_data,
    compute_grads_and_laplacian_Jastrow_part,
    compute_grads_and_laplacian_Jastrow_three_body,
)
from jqmc.wavefunction import compute_kinetic_energy

from tests.test_rs_kernel._rs_fixtures import rs_ham

_ZONES_LNDET = ("ao_eval", "ao_grad_lap", "det_grad_lap")
_ZONES_JASTROW = ("jastrow_grad_lap",)
_ZONES_J3 = _ZONES_JASTROW + ("ao_eval", "ao_grad_lap")
_ZONES_KINETIC = ("wf_kinetic", "ao_eval", "ao_grad_lap", "det_grad_lap", "jastrow_grad_lap")


def _config(ham, seed=0):
    gem = ham.wavefunction_data.geminal_data
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    r_up = R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.6, size=(nu, 3))
    r_dn = R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.6, size=(nd, 3))
    return r_up, r_dn


# Small, multi-axis corpus: SD/AGP, sphe/cart, AE/ECP, jnone/j12.
_LNDET_PATTERNS = ["mol_sphe_sd_jnone", "mol_ecp_cart_jnone", "mol_ecp_sphe_j12", "mol_sphe_agp_jnone"]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _LNDET_PATTERNS)
def test_grads_laplacian_ln_det(pattern):
    """Grads laplacian ln det."""
    ham = rs_ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    # The spherical water default point is close enough to a determinant node
    # to amplify a one-ULP fp32 AO difference.  seed=20 retains the same route
    # with cond(G)=1.8e1.
    r_up, r_dn = _config(ham, seed=20 if pattern == "mol_ecp_sphe_j12" else 0)
    gu, gd, lu, ld = jqmc_rs.compute_grads_and_laplacian_ln_det(gem, r_up, r_dn)
    Gu, Gd, Lu, Ld = (np.asarray(x, float) for x in compute_grads_and_laplacian_ln_Det(gem, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNDET, level="strict")
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol)
    assert np.allclose(gd, Gd, atol=atol, rtol=rtol)
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol)
    assert np.allclose(ld, Ld, atol=atol, rtol=rtol)


# J1+J2 only (j123ao/J3 + DTN out of the autodiff-free Rust scope).
#
# The ``_pade`` / ``_p1e2`` / ``_e1p2`` rows carry the PRODUCTION-DEFAULT radial
# form (every other fixture in this suite pins the
# non-default ``exp``, so without them the pade branch of ``j1_grad_lap_one_spin`` /
# ``j2_pair`` -- the branch that feeds the kinetic energy and the MCMC drift --
# would have zero coverage). ``_p1e2`` / ``_e1p2`` split the J1 x J2 form
# cross-product, so a form bug confined to one channel cannot hide behind the
# other channel's agreement.
_J12_PATTERNS = [
    "mol_ecp_cart_j12",
    "mol_ecp_sphe_j12",
    "mol_cart_sd_j12",
    "mol_cart_sd_j12_pade",
    "mol_cart_sd_j12_p1e2",
    "mol_cart_sd_j12_e1p2",
    "mol_ecp_cart_j12_pade",
    "mol_ecp_sphe_j12_pade",
]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _J12_PATTERNS)
def test_grads_laplacian_jastrow_part(pattern):
    """Grads laplacian jastrow part."""
    ham = rs_ham(pattern)
    jd = ham.wavefunction_data.jastrow_data
    r_up, r_dn = _config(ham, seed=1)
    gu, gd, lu, ld = jqmc_rs.compute_grads_and_laplacian_jastrow_part(jd, r_up, r_dn)
    Gu, Gd, Lu, Ld = (np.asarray(x, float) for x in compute_grads_and_laplacian_Jastrow_part(jd, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_JASTROW, level="strict")
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol)
    assert np.allclose(gd, Gd, atol=atol, rtol=rtol)
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol)
    assert np.allclose(ld, Ld, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# Liveness guard for the pade rows above.
#
# A parity row is only worth its runtime if the branch it is supposed to select
# is actually selected. If the ``jastrow_{1b,2b}_type`` token were dropped
# anywhere on the way to Rust (the PyO3 ``getattr`` in ``data.rs``, or a future
# refactor that hardcodes the form), the ``_pade`` rows would collapse into
# duplicates of their ``exp`` twins and STILL pass -- Rust and XLA would simply
# agree on the wrong form together. This test pins the discrimination directly:
# the two forms must produce measurably DIFFERENT Jastrow derivatives on
# identical positions, on the RUST side, so the pade rows cannot go vacuous.
#
# The separation asserted is a physical one (the two radial kernels differ at
# O(1) for a=1), compared against the same named parity tolerance the rows above
# use -- there is no independent literal here.
_FORM_TWINS = [
    ("mol_cart_sd_j12", "mol_cart_sd_j12_pade"),
    ("mol_cart_sd_j12", "mol_cart_sd_j12_p1e2"),  # J1 alone switched
    ("mol_cart_sd_j12", "mol_cart_sd_j12_e1p2"),  # J2 alone switched
    ("mol_ecp_sphe_j12", "mol_ecp_sphe_j12_pade"),
]


@pytest.mark.light
@pytest.mark.parametrize(("exp_pattern", "pade_pattern"), _FORM_TWINS)
def test_rust_jastrow_form_token_is_honoured(exp_pattern, pade_pattern):
    """Rust must SEE the functional-form token: the exp and pade twins of the
    same cell, on identical electron positions, must not agree to within the
    parity tolerance. Guards every ``_pade`` parity row in this suite against
    silently degenerating into a second copy of its exp twin.
    """
    ham_exp, ham_pade = rs_ham(exp_pattern), rs_ham(pade_pattern)
    r_up, r_dn = _config(ham_exp, seed=1)
    a = jqmc_rs.compute_grads_and_laplacian_jastrow_part(ham_exp.wavefunction_data.jastrow_data, r_up, r_dn)
    b = jqmc_rs.compute_grads_and_laplacian_jastrow_part(ham_pade.wavefunction_data.jastrow_data, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_JASTROW, level="strict")
    exp_components = np.concatenate([np.asarray(component).reshape(-1) for component in a])
    pade_components = np.concatenate([np.asarray(component).reshape(-1) for component in b])
    assert not np.allclose(exp_components, pade_components, atol=atol, rtol=rtol), (
        f"Rust gives IDENTICAL Jastrow grad/lap for {exp_pattern} and {pade_pattern} -- "
        "the jastrow_1b_type / jastrow_2b_type token is not reaching the Rust kernel, "
        "so every pade parity row in this suite is vacuous"
    )
    # The negative control cannot replace the backend-parity oracle: wrong but
    # distinct Rust results must also fail, for both functional forms.
    for form, ham, actual in (("exp", ham_exp, exp_components), ("pade", ham_pade, pade_components)):
        expected = compute_grads_and_laplacian_Jastrow_part(ham.wavefunction_data.jastrow_data, r_up, r_dn)
        reference = np.concatenate([np.asarray(component).reshape(-1) for component in expected])
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _J12_PATTERNS + ["mol_sphe_agp_jnone"])
def test_kinetic_energy(pattern):
    """Kinetic energy."""
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    r_up, r_dn = _config(ham, seed=2)
    k_rust = jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)
    k_jqmc = float(compute_kinetic_energy(wf, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    assert np.isclose(k_rust, k_jqmc, rtol=rtol, atol=atol), (k_rust, k_jqmc)


def _attach_random_j3(ham, seed=0, scale=0.2):
    """Attach an ACTIVE three-body Jastrow on the geminal's own AO leaf.

    The committed ECP cart/sphe fixtures carry no j3 block, so a random one is
    built from scratch on the same AO leaf the geminal already uses.
    """
    wf = ham.wavefunction_data
    orb = wf.geminal_data.orb_data_up_spin
    aos_data = getattr(orb, "aos_data", orb)  # unwrap MOs_data if MO-rep
    n = aos_data.num_ao
    jm = np.random.default_rng(seed).normal(scale=scale, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    j3d = Jastrow_three_body_data(orb_data=aos_data, j_matrix=jm)
    jd2 = wf.jastrow_data.replace(jastrow_three_body_data=j3d)
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jd2))


# J3 (three-body): mol_cart_sd_j123 already carries an active pgto-basis J3
# (see _rs_fixtures); the ECP cart/sphe patterns get one freshly attached.
_J123_PATTERNS = ["mol_cart_sd_j123", "mol_ecp_cart_jnone", "mol_ecp_sphe_jnone"]


def _ham_with_j3(pattern, seed):
    ham = rs_ham(pattern)
    if ham.wavefunction_data.jastrow_data.jastrow_three_body_data is not None:
        return ham
    return _attach_random_j3(ham, seed=seed)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _J123_PATTERNS)
def test_grads_laplacian_jastrow_three_body(pattern):
    """Grads laplacian jastrow three body."""
    ham = _ham_with_j3(pattern, seed=7)
    jd2 = ham.wavefunction_data.jastrow_data
    # This random spherical J3 contraction has cancellation at the seed-7
    # electron point.  Keep the random J3 matrix fixed and use a stable point.
    r_up, r_dn = _config(ham, seed=14 if pattern == "mol_ecp_sphe_jnone" else 7)
    gu, gd, lu, ld = jqmc_rs.compute_grads_and_laplacian_jastrow_three_body(jd2, r_up, r_dn)
    Gu, Gd, Lu, Ld = (
        np.asarray(x, float) for x in compute_grads_and_laplacian_Jastrow_three_body(jd2.jastrow_three_body_data, r_up, r_dn)
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_J3, level="strict")
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol)
    assert np.allclose(gd, Gd, atol=atol, rtol=rtol)
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol)
    assert np.allclose(ld, Ld, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("pattern", _J123_PATTERNS)
def test_kinetic_energy_with_j3(pattern):
    """Kinetic energy with j3."""
    ham = _ham_with_j3(pattern, seed=7)
    wf2 = ham.wavefunction_data
    r_up, r_dn = _config(ham, seed=7)
    k_rust = jqmc_rs.compute_kinetic_energy(wf2.geminal_data, wf2.jastrow_data, r_up, r_dn)
    k_jqmc = float(compute_kinetic_energy(wf2, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    assert np.isclose(k_rust, k_jqmc, rtol=rtol, atol=atol), (k_rust, k_jqmc)


# =========================================================================== #
# physically NON-ZERO ``core_electrons``
# x composite AO.
#
# THE GAP. ``core_electrons`` is the per-nucleus count of electrons removed by
# the pseudopotential; the one-body Jastrow uses it as
# ``Z_eff = Z - core_electrons`` and, through ``c = (2 Z_eff)^(1/4)``
# (jqmc/rs/src/jastrow_factor.rs:647), it sets the electron-nucleus cusp
# constant that the J1 value, gradient, Laplacian, the kinetic energy and every
# force/param-derivative built on them inherit.
#
# A fixture with ``core_electrons == 0``, i.e. ``Z_eff == Z``, is the one value
# at which the whole subtraction is a no-op: Rust could read the wrong field,
# drop it, or hardcode zero, and every such test would stay green. The 38
# builder call sites therefore read the value from each fixture's own
# pseudopotential via ``tests/_ecp_core_electrons.py``, so the pseudized
# fixtures carry their real core counts (Li 2, Si 10, Cl 10). Note that pseudization is not
# visible in a stem name -- ``H2_ecp_ccpvdz_cart.h5`` carries ``ecp_flag=True``
# with ``z_cores == 0`` (hydrogen ccECP removes no core), while ``lithium.h5``
# and ``silicon.h5`` carry no ``_ecp_`` in their names and remove 2 and 10.
#
# This row stays because it asserts something the builders cannot: that Rust
# READS the field. It sets BOTH arms explicitly (0 vs z_cores) instead of
# leaning on a builder default -- see ``_composite_core_ham``.
#
# THE FIXTURE. Built programmatically from the committed QE ``Li_gamma``
# pgto(+)pw composite (``_rs_fixtures`` pattern ``pbc_composite_j12``) by writing
# the fixture's OWN ``coulomb_potential_data.z_cores`` -- the physically correct
# value, 2 removed core electrons per Li under ccECP, so Z_eff = 3 - 2 = 1 --
# into the J1 block. No new asset is added; the number comes from the committed
# pseudopotential the same ham already carries (``qe_to_jqmc.py:1109`` threads
# exactly this ``z_cores -> core_electrons`` link in production).
# =========================================================================== #
import os  # noqa: E402

from jqmc._setting import MCMC_MIN_STEPS  # noqa: E402
from jqmc.atomic_orbital import AOs_composite_data  # noqa: E402
from jqmc.determinant import compute_geminal_all_elements  # noqa: E402
from jqmc.hamiltonians import compute_local_energy_fast  # noqa: E402

from tests.test_rs_kernel._route_spy import assert_native_ran, install_ctx_spy  # noqa: E402

_ZONES_EL_COMPOSITE = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")
_COMPOSITE_CORE_PATTERN = "pbc_composite_j12"


def _composite_core_ham(with_core: bool):
    """The composite-AO ham with ``core_electrons`` set to the fixture's own ``z_cores``
    (``with_core=True``) or to zero (``with_core=False``).

    BOTH arms are set explicitly: a zero arm taken from the shared builder's ham would
    make the discriminator depend on a builder default it does not own (the builders read
    ``core_electrons`` from each fixture's own pseudopotential). Setting each arm from here
    keeps the contrast a property of the test rather than of whatever the builders happen
    to default to.
    """
    ham = rs_ham(_COMPOSITE_CORE_PATTERN)
    n_atom = len(ham.structure_data.positions)
    core = tuple(float(z) for z in ham.coulomb_potential_data.z_cores) if with_core else tuple([0.0] * n_atom)
    jd = ham.wavefunction_data.jastrow_data
    jd = jd.replace(jastrow_one_body_data=jd.jastrow_one_body_data.replace(core_electrons=core))
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


def _assert_core_and_composite_live(ham):
    """LIVENESS, three independent claims, all measured:

    1. ``core_electrons`` is physically NON-ZERO on every nucleus and equals the
       fixture's own pseudopotential ``z_cores`` -- so ``Z_eff != Z`` and the
       subtraction the Rust J1 performs is not a no-op;
    2. the removal is not the whole charge (``Z_eff > 0``), i.e. the cusp
       constant ``(2 Z_eff)^(1/4)`` stays a real, non-degenerate number;
    3. the determinant's AO leaf really is the COMPOSITE (pGTO (+) PW) container
       with more than one child of more than one kind -- if a fixture swap ever
       collapsed it to a single-family AO, this row would silently stop being the
       "composite" cross and must go red.
    """
    j1 = ham.wavefunction_data.jastrow_data.jastrow_one_body_data
    core = np.asarray(j1.core_electrons, dtype=float)
    z_cores = np.asarray(ham.coulomb_potential_data.z_cores, dtype=float)
    z = np.asarray(ham.structure_data.atomic_numbers, dtype=float)
    assert core.size == z.size and core.size > 0, (core.shape, z.shape)
    assert np.all(core > 0.0), ("core_electrons is not physically non-zero -- Z_eff == Z, the no-op case", core)
    assert np.array_equal(core, z_cores), ("core_electrons does not match the fixture's own z_cores", core, z_cores)
    assert np.all(z - core > 0.0), ("Z_eff is non-positive -- the J1 cusp constant is degenerate", z, core)

    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert isinstance(aos, AOs_composite_data), ("AO leaf is not composite", type(aos).__name__)
    kinds = {type(c).__name__ for c in aos.children}
    assert len(aos.children) >= 2 and len(kinds) >= 2, ("composite AO degenerated to a single AO family", kinds)


def _composite_core_cfg(ham, seed):
    gd = ham.wavefunction_data.geminal_data
    cell = np.asarray(ham.structure_data.cell, dtype=float)
    rng = np.random.default_rng(seed)
    return (
        np.ascontiguousarray(rng.random((int(gd.num_electron_up), 3)) @ cell),
        np.ascontiguousarray(rng.random((int(gd.num_electron_dn), 3)) @ cell),
    )


@pytest.mark.light
def test_rust_reads_core_electrons_on_composite_ao():
    """LIVENESS DISCRIMINATOR. The two hams differ ONLY in
    ``core_electrons`` (0 vs z_cores). If Rust ignored the field -- read the wrong
    attribute, hardcoded 0, or dropped it in ``data.rs`` -- it would return
    IDENTICAL Jastrow derivatives for both, and every parity row below would be
    vacuous while still passing. Compared against the same named parity tolerance
    the rows below use; no independent literal.
    """
    ham0 = _composite_core_ham(with_core=False)
    ham1 = _composite_core_ham(with_core=True)
    _assert_core_and_composite_live(ham1)
    r_up, r_dn = _composite_core_cfg(ham1, seed=11)
    a = jqmc_rs.compute_grads_and_laplacian_jastrow_part(ham0.wavefunction_data.jastrow_data, r_up, r_dn)
    b = jqmc_rs.compute_grads_and_laplacian_jastrow_part(ham1.wavefunction_data.jastrow_data, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_JASTROW)
    assert not all(np.allclose(np.asarray(x, float), np.asarray(y, float), atol=atol, rtol=rtol) for x, y in zip(a, b)), (
        "Rust gives IDENTICAL Jastrow grad/lap for core_electrons=0 and core_electrons=z_cores -- "
        "the field is not reaching the Rust J1 kernel, so every core_electrons parity row is vacuous"
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_grads_laplacian_jastrow_part_composite_core_electrons(seed):
    """ROW 7 comparison (a): the J1+J2 gradient / Laplacian on a composite-AO ham
    with a physical ``core_electrons``, Rust vs XLA."""
    ham = _composite_core_ham(with_core=True)
    _assert_core_and_composite_live(ham)
    jd = ham.wavefunction_data.jastrow_data
    r_up, r_dn = _composite_core_cfg(ham, seed)
    gu, gd, lu, ld = jqmc_rs.compute_grads_and_laplacian_jastrow_part(jd, r_up, r_dn)
    Gu, Gd, Lu, Ld = (np.asarray(x, float) for x in compute_grads_and_laplacian_Jastrow_part(jd, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_JASTROW, level="strict")
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol), np.abs(np.asarray(gu) - Gu).max()
    assert np.allclose(gd, Gd, atol=atol, rtol=rtol), np.abs(np.asarray(gd) - Gd).max()
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol), np.abs(np.asarray(lu) - Lu).max()
    assert np.allclose(ld, Ld, atol=atol, rtol=rtol), np.abs(np.asarray(ld) - Ld).max()


@pytest.mark.light
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_kinetic_energy_composite_core_electrons(seed):
    """ROW 7 comparison (b): the ASSEMBLED kinetic energy -- where the
    core-corrected J1 derivative meets the composite AO determinant."""
    ham = _composite_core_ham(with_core=True)
    _assert_core_and_composite_live(ham)
    wf = ham.wavefunction_data
    r_up, r_dn = _composite_core_cfg(ham, seed)
    k_rust = jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)
    k_jqmc = float(compute_kinetic_energy(wf, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    assert np.isclose(k_rust, k_jqmc, rtol=rtol, atol=atol), (seed, k_rust, k_jqmc)


@pytest.mark.light
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_local_energy_fast_composite_core_electrons(seed):
    """ROW 7 comparison (c): the full production ``e_L`` (the quantity the driver
    actually accumulates) on the same cell."""
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    ham = _composite_core_ham(with_core=True)
    _assert_core_and_composite_live(ham)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _composite_core_cfg(ham, seed)
    RT = np.eye(3)
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=float)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=float) @ RT
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn)))
    native = float(jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, w, gp, g_inv))
    jax_eL = float(compute_local_energy_fast(ham, r_up, r_dn, RT, g_inv, None))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL_COMPOSITE, level="strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (seed, native, jax_eL)


@pytest.mark.heavy
def test_route_parity_composite_core_electrons(monkeypatch):
    """ROW 7 route parity: a PRODUCTION VMC run on the composite-AO ham carrying
    the physical ``core_electrons`` reaches the NATIVE fused energy chunk -- the
    call site that consumes the J1 block this row's formulas were verified on."""
    from jqmc.jqmc_mcmc import MCMC

    ham = _composite_core_ham(with_core=True)
    _assert_core_and_composite_live(ham)
    spy = install_ctx_spy(monkeypatch)
    os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        m = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        m.run(num_mcmc_steps=MCMC_MIN_STEPS)
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    ctx = "composite/core_electrons"
    assert_native_ran(spy, "run_energy_chunk_batch", context=ctx)
    assert bool(spy.build_arg("pbc")) is True, f"{ctx}: build_obs_ctx was not asked for the periodic family"
    assert bool(spy.build_arg("complex")) is False, f"{ctx}: build_obs_ctx asked for the complex family"
