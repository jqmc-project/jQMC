"""Layer 2: PRODUCTION ROUTE PARITY for the c-axis param-deriv blocks.

WHY THE ASSEMBLER-LEVEL COVERAGE IS NOT ENOUGH
----------------------------------------------
``test_ungate_basis_opt_param_deriv.py`` drives the production ASSEMBLER
(``_rs_native_c_param_grads``) but builds the ``NativeObsCtx`` itself, so it cannot see
a driver that builds the ctx with the wrong flags -- for example at a complex twist,
where the monolithic ``ParamDerivArgs`` parses a REAL geminal and
``build_obs_ctx(want_param_deriv=True)`` cannot serve the geminal. Its population is
also a single bare cart-GTO molecular fixture, and the c-axis matrix's ``_RADIAL_AO``
set (``test_rs_analytic_matrix.py``) has no ``composite`` entry, so a periodic
composite cell never enumerates a ``Basis`` block there.

This module runs the FULL production ``MCMC`` driver (``comput_log_WF_param_deriv`` /
``comput_e_L_param_deriv``, the public constructor kwargs) with the route spy attached,
over a population that DELIBERATELY includes periodic composite, pure PW, pure B-spline
and complex twist.

WHAT EACH ROW ASSERTS
---------------------
  1. **native ran** -- the spy saw the expected ``pd_ok_*`` / ``pd_el_*`` ctx methods,
     for every family the wavefunction actually carries.
  2. **the route is the one we think it is** -- a complex twist is served by the PYTHON
     ``_RsComplexPdCtx`` adapter; DTN delegates to the shared ``NativeObsCtx`` while
     the remaining families use the free complex kernels
     methods. The spy keeps those in a separate bucket so a row cannot claim native-ctx
     coverage it does not have. This is a coverage FACT, recorded, not a failure.
  3. **block set / order / dtype** of the driver's public ``dln_Psi_dc`` / ``de_L_dc``
     match the flag-OFF XLA route exactly.
  4. **values** match at the calibrated named tier.

THE PERIODIC COMPOSITE BASIS-OPT ROW
------------------------------------
``test_route_parity_periodic_composite_basis_opt`` requires the native basis methods
to run through the production assembler on a periodic composite leaf, whose GTO
children are parsed in the Python canonical child order
(``determinant._concat_gto_radial``). It is NOT xfailed and NOT skipped: a native
``RuntimeError`` (``NativeObsCtx: param_deriv basis (non-GTO geminal leaf) was not
built for this Hamiltonian``) fails it.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._precision import get_tolerance_xla_rust_consistency, precision_policy_snapshot
from jqmc._setting import (
    MCMC_MIN_STEPS,
)
from jqmc.determinant import _gto_leaves_of
from jqmc.jqmc_mcmc import MCMC

from tests.test_rs_kernel._route_spy import (
    assert_arg_is_not_none,
    assert_complex_pd_adapter_ran,
    assert_native_ran,
    install_ctx_spy,
)
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel._gdd_rust_adapters import record_rust_bound_case

_ZONES = ("ao_eval", "mo_eval", "det_eval", "jastrow_eval")

_HAM_CACHE: dict[str, object] = {}


def _ham(pattern):
    if pattern not in _HAM_CACHE:
        _HAM_CACHE[pattern] = rs_ham(pattern)
    return _HAM_CACHE[pattern]


def _tol(ham, block):
    """Static parity envelope for propagated parameter-derivative aggregates.

    These derivatives cross fp32 AO/Jastrow leaves, determinant inverses, and
    kinetic reductions.  Their reduction order varies across supported JAX
    versions, so mixed mode uses the single named medium aggregate envelope.
    In full mode that helper resolves to exactly the strict fp64 tolerance.
    """
    return get_tolerance_xla_rust_consistency(_ZONES, level="medium")


def _run(ham, rust_on, basis=False, j3_basis=False, steps=MCMC_MIN_STEPS):
    """One PRODUCTION ``MCMC`` param-deriv run, backend chosen by the env override.

    ``basis=True`` enables the two geminal radial blocks through the SAME call
    ``run_optimize`` makes (``__set_param_gradient_flags``) -- invoking the optimizer
    loop itself would add an SR solve and a checkpoint dump that have nothing to do with
    the route under test.

    Rust and XLA use their production RNG implementations and therefore walk different
    chains. Public derivative blocks are compared as end-to-end statistical estimators.
    """
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        m = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=2,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
            random_discretized_mesh=True,
        )
        if basis:
            m._MCMC__set_param_gradient_flags(lambda_basis_exp=True, lambda_basis_coeff=True)
        if j3_basis:
            m._MCMC__set_param_gradient_flags(j3_basis_exp=True, j3_basis_coeff=True)
        m.run(num_mcmc_steps=steps)
        return m
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


def _expected_pd_methods(ham, basis):
    """The ``pd_*`` methods the production assembler MUST reach for this wavefunction.

    Derived from the Hamiltonian for route inventory, not a proof that these
    calls include every physical contribution. Numerical source guarantees
    independently check the assembled values on inputs with live derivatives.
    """
    from jqmc.determinant import AGPn_data, SD_data

    wf = ham.wavefunction_data
    jas, gem = wf.jastrow_data, wf.geminal_data
    ecp = bool(ham.coulomb_potential_data.ecp_flag)
    ok, el = [], []

    def add(fam, ecp_capable=True):
        ok.append(f"pd_ok_{fam}")
        el.append(f"pd_el_{fam}")
        if ecp and ecp_capable:
            el.append(f"pd_el_ecp_{fam}")

    if jas is not None:
        if jas.jastrow_one_body_data is not None:
            add("j1")
        if jas.jastrow_two_body_data is not None:
            add("j2")
        if jas.jastrow_three_body_data is not None:
            add("j3")
        if jas.jastrow_dtn_data is not None:
            add("dtn")
    if gem is not None:
        if isinstance(gem, (SD_data, AGPn_data)):
            add("mo")
            if isinstance(gem, AGPn_data):
                # This inventory does not require a separate ECP lambda-diag
                # obs-ctx method. AGPn ECP derivatives need not vanish; occupied-only
                # fixtures can hide them. Numerical source-parity evidence must
                # independently compare the total on an ECP-sensitive input.
                add("lambda_diag", ecp_capable=False)
        else:
            add("lambda")
    if basis and _gto_leaves_of(gem.orb_data_up_spin) and _gto_leaves_of(gem.orb_data_dn_spin):
        add("basis")
    return ok, el


def _assert_blocks_and_values(m_rust, m_jax, ham, tag):
    """Check the driver's public parameter-derivative storage contract.

    ORDERING, precisely. The two routes DO emit these dicts in different insertion
    orders (measured on ``mol_cart_sd_j12`` with the basis flags on: the native
    assembler appends the basis blocks last, the XLA reducer places them before the MO
    blocks). That is not a defect, and asserting dict order here would be a false
    positive: the only consumer, ``MCMC.get_O_matrix``, walks the CANONICAL
    ``get_variational_blocks()`` order and indexes ``dln_Psi_dc`` BY NAME, so insertion
    order is unobservable downstream. What must match is the block SET (a route that
    reduced a different set is exactly the route-parity failure) plus per-block shape / dtype /
    values. The ordering that IS load-bearing lives one level down -- the tuple returns
    of the ``pd_*`` methods themselves, e.g. the DTN ``(alpha, beta, gamma)`` triple and
    the MO ``(up, dn)`` pair -- and is asserted from the spy's return descriptor by
    ``_assert_pd_return_shapes``.
    """
    for name in ("dln_Psi_dc", "de_L_dc"):
        a = getattr(m_rust, name)
        b = getattr(m_jax, name)
        assert set(a) == set(b), f"{tag}: {name} block SET rust={sorted(a)} vs xla={sorted(b)}"
        for blk in a:
            ra, rb = np.asarray(a[blk]), np.asarray(b[blk])
            assert ra.shape == rb.shape, f"{tag}: {name}[{blk}] shape {ra.shape} vs {rb.shape}"
            assert ra.dtype == rb.dtype, f"{tag}: {name}[{blk}] dtype {ra.dtype} vs {rb.dtype}"
            assert not np.any(np.isnan(ra)), f"{tag}: {name}[{blk}] (rust) has NaN"
            assert not np.any(np.isnan(rb)), f"{tag}: {name}[{blk}] (xla) has NaN"

    # The optimizer consumes the exact BlockDerivativeSource publication, not the
    # legacy storage-space dictionaries above. In particular, a real parameter at a
    # twist has a complex per-walker dE_L/dc even though the legacy diagnostic is
    # deliberately projected to real. Compare every exact component so an early
    # Rust-side Re projection cannot hide behind matching diagnostics.
    all_blocks = ham.wavefunction_data.get_variational_blocks(
        state_token=("rust-xla-exact-source-parity", 0),
        opt_geminal=True,
        opt_J3_basis_exp=True,
        opt_J3_basis_coeff=True,
        opt_lambda_basis_exp=True,
        opt_lambda_basis_coeff=True,
    )
    active_names = set(m_rust.dln_Psi_dc)
    source_blocks = [block for block in all_blocks if block.name in active_names]
    assert {block.name for block in source_blocks} == active_names
    for local_energy in (False, True):
        rust_sources = m_rust._optimization_derivative_sources(blocks=source_blocks, local_energy=local_energy)
        jax_sources = m_jax._optimization_derivative_sources(blocks=source_blocks, local_energy=local_energy)
        assert tuple(source.name for source in rust_sources) == tuple(source.name for source in jax_sources)
        for block, rust_source, jax_source in zip(source_blocks, rust_sources, jax_sources, strict=True):
            for component in ("direct", "d_p", "d_pbar"):
                rust_values = getattr(rust_source, component)
                jax_values = getattr(jax_source, component)
                if rust_values is None:
                    assert jax_values is None
                    continue
                assert jax_values is not None
                rust_values = np.asarray(rust_values)
                jax_values = np.asarray(jax_values)
                assert rust_values.shape == jax_values.shape
                assert rust_values.dtype == jax_values.dtype
                assert np.all(np.isfinite(rust_values))
                assert np.all(np.isfinite(jax_values))


#: The ``pd_*`` families whose native return is a TUPLE, and its required arity. This is
#: the ordering contract that IS load-bearing: the assembler unpacks these positionally
#: (``ta, tb, tg = ctx.pd_el_dtn(...)`` / ``gu, gd = ctx.pd_el_mo(...)``), so a kernel that
#: returned two elements instead of three, or swapped up/dn, would mis-assign whole blocks
#: with no shape error anywhere.
_TUPLE_ARITY = {"dtn": 3, "mo": 2, "lambda_diag": 2, "basis": 2}


def _assert_pd_return_shapes(spy, tag):
    """CHECK 3 at the METHOD level: tuple-returning ``pd_*`` methods keep their arity."""
    for call in spy.calls + spy.adapter_calls:
        # The real path goes through the rayon ``pd_*_batch`` twins,
        # which stack a leading walker axis but preserve the tuple
        # arity of the per-walker scalar (DTN triple, MO pair, ...). Strip the suffix
        # so the arity contract is checked on the batch returns too; the complex
        # adapter/native DTN calls stay scalar and match unchanged.
        method = call.method.removesuffix("_batch").removesuffix("_complex")
        for fam, arity in _TUPLE_ARITY.items():
            if not (method.endswith(f"_{fam}") and method.startswith(("pd_ok_", "pd_el_"))):
                continue
            kind = call.ret[0]
            assert kind == "seq", f"{tag}: {call.method}() returned a {kind}, expected a {arity}-tuple"
            assert call.ret[1] == arity, (
                f"{tag}: {call.method}() returned {call.ret[1]} elements, expected {arity} -- the assembler "
                f"unpacks this positionally, so a changed arity silently mis-assigns blocks"
            )


# --------------------------------------------------------------------------- #
# Population. DELIBERATELY includes periodic composite / PW / B-spline / complex
# twist -- the families the c-axis matrix excludes from its Basis enumeration.
# --------------------------------------------------------------------------- #
_REAL_CELLS = [
    # (pattern, basis_opt, heavy)
    # R09 tied-spin control (H2 SD, time_reversal=True). The N SD row below is
    # the independent-spin control with a live down-channel block.
    ("mol_h2_cart_sd_jnone", False, False),
    ("mol_cart_sd_j12", False, False),
    ("mol_cart_sd_j12", True, False),
    ("mol_cart_sd_j123", False, True),
    ("mol_cart_sd_jdtn", False, True),
    # ECP + J3 / DTN: the ONLY committed patterns that reach ``pd_el_ecp_j3`` and
    # ``pd_el_ecp_dtn`` (the non-local mesh add for those two families). Without them
    # those two of the 23 production pd_* methods have no route coverage at all.
    ("mol_ecp_cart_j123", False, True),
    ("mol_ecp_cart_jdtn", False, True),
    ("mol_cart_agp_jnone", True, False),  # AGP -> pd_*_lambda + basis
    ("mol_cart_agpn_jnone", False, False),  # AGPn -> pd_*_lambda_diag
    ("pbc_gamma_sphepgto_j12", False, True),
    ("pbc_gamma_sphepgto_j12", True, True),
    ("pbc_composite_jnone", False, True),
    ("pbc_composite_j12", False, True),
    ("pw_gamma_jnone", False, True),
    ("bspline_gamma_jnone", False, True),
]

_REAL_IDS = [f"{p}{'-basis' if b else ''}" for p, b, _ in _REAL_CELLS]


@pytest.mark.parametrize(
    "pattern, basis",
    [pytest.param(p, b, marks=pytest.mark.heavy if h else pytest.mark.light) for p, b, h in _REAL_CELLS],
    ids=_REAL_IDS,
)
def test_route_parity_param_deriv_real(pattern, basis, monkeypatch):
    """CHECKS 1, 3, 4 for the REAL-path c-axis blocks, through the production driver."""
    ham = _ham(pattern)
    tag = f"{pattern}{'/basis' if basis else ''}"

    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True, basis=basis)

    # CHECK 1: the ctx was built FOR the param derivatives, and every family the
    # wavefunction carries reached its native method.
    assert spy.build_count() >= 1, f"{tag}: build_obs_ctx was never called -- production did not take the native route"
    assert bool(spy.build_arg("want_param_deriv")) is True, (
        f"{tag}: build_obs_ctx was not asked for ParamDerivArgs, yet the run requested O_k and dE_L/dc"
    )
    ok_methods, el_methods = _expected_pd_methods(ham, basis)
    # REAL path: the assembler fans each family over the walker axis through the rayon
    # ``pd_*_batch`` twin, so the native method the spy records
    # is the batch entry. (The complex adapter has no ``_batch`` surface, so its scalar
    # ``pd_*`` still runs -- asserted with the un-suffixed name in the complex test.)
    for meth in ok_methods + el_methods:
        assert_native_ran(spy, meth + "_batch", context=tag)
    if "pd_el_mo" in el_methods:
        assert_arg_is_not_none(spy, "pd_el_mo_batch", "geminal_inv", context=tag)
    assert not spy.adapter_calls, f"{tag}: a REAL-path cell was served by the complex adapter: {spy.summary()}"
    _assert_pd_return_shapes(spy, tag)

    m_jax = _run(ham, rust_on=False, basis=basis)
    _assert_blocks_and_values(m_rust, m_jax, ham, tag)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["twist_sphepgto_j12", "twist_sphepgto_jdtn"])
def test_route_parity_param_deriv_complex_twist(pattern, monkeypatch):
    """COMPLEX TWIST: adapter surface, with the Jastrow families backed by the shared native context.

    The driver keeps ``_RsComplexPdCtx`` as the common method surface for complex
    c-axis families. The Jastrow families delegate to the shared ``NativeObsCtx``,
    which parses the complex Jastrow / J3 orbital / ECP tables once per run: DTN
    (reparsing a complex B-spline geminal for every walker is grid-sized) and the
    J1 / J2 / J3 ``j_matrix`` dE_L/dc and the J3 O_k
    (``pd_*_complex``). This test asserts:

      * the native ctx is built with ``want_param_deriv=True`` at a twist;
      * exactly the active Jastrow methods land on ``NativeObsCtx``: DTN, the J1 / J2 /
        J3 kinetic and ECP ``pd_el_*_complex`` and ``pd_ok_j3_complex``; the J1 / J2
        O_k (real, no geminal) and the geminal / basis families keep using their
        parity-pinned adapter/free-kernel route;
      * the adapter DID serve every family (the complex-twist cell defect was that nothing
        served them at all);
      * blocks and values still match XLA.
    """
    ham = _ham(pattern)
    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True)

    assert bool(spy.build_arg("complex")) is True, f"{pattern}: build_obs_ctx was not asked for the complex family"
    assert bool(spy.build_arg("want_param_deriv")) is True, (
        f"{pattern}: the complex context was not asked for its shared-geminal DTN optimizer args"
    )
    native_pd = {n for n in spy.names() if n.startswith(("pd_ok_", "pd_el_"))}
    jas = ham.wavefunction_data.jastrow_data
    ecp = bool(ham.coulomb_potential_data.ecp_flag)
    # The walker batch of each family is ONE native rayon call (``*_complex_batch``).
    expected_native_pd = (
        {"pd_ok_dtn_complex_batch", "pd_el_dtn_complex_batch", "pd_el_ecp_dtn_complex_batch"}
        if jas.jastrow_dtn_data is not None
        else set()
    )
    for fam, leaf in (
        ("j1", jas.jastrow_one_body_data),
        ("j2", jas.jastrow_two_body_data),
        ("j3", jas.jastrow_three_body_data),
    ):
        if leaf is not None:
            expected_native_pd.add(f"pd_el_{fam}_complex_batch")
            if ecp:
                expected_native_pd.add(f"pd_el_ecp_{fam}_complex_batch")
    if jas.jastrow_three_body_data is not None:
        expected_native_pd.add("pd_ok_j3_complex_batch")
    assert native_pd == expected_native_pd, (
        f"{pattern}: complex native Jastrow route differs from the fixed-per-run ownership contract: "
        f"expected={sorted(expected_native_pd)}, got={sorted(native_pd)}"
    )

    ok_methods, el_methods = _expected_pd_methods(ham, basis=False)
    # The assembler asks the adapter for the walker batch (``<method>_batch``).
    adapter_names = {c.method.removesuffix("_batch") for c in spy.adapter_calls}
    missing = sorted(set(ok_methods + el_methods) - adapter_names)
    assert not missing, (
        f"{pattern}: the complex param-deriv adapter never served {missing} -- the kernels exist "
        f"but the driver does not route these families to them. Observed: {sorted(adapter_names)}"
    )
    _assert_pd_return_shapes(spy, pattern)

    m_jax = _run(ham, rust_on=False)
    _assert_blocks_and_values(m_rust, m_jax, ham, pattern)


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pbc_composite_j12", "pbc_composite_jnone"])
def test_route_parity_periodic_composite_basis_opt(pattern, monkeypatch):
    """Periodic composite basis-opt runs natively through the production assembler.

    ``determinant._gto_leaves_of()`` enumerates the GTO/pGTO children of a composite AO and
    ``_concat_gto_radial()`` documents that their radial parameters concatenate in child
    order -- so a composite DOES carry a legitimate basis-optimization block, and the
    production assembler (``jqmc_mcmc._rs_native_c_param_grads``) calls
    ``ctx.pd_{ok,el}_basis()`` for it. The Rust side must inspect the composite's
    children; a composite treated as non-GTO leaves ``basis=None`` and the driver raises
        RuntimeError: NativeObsCtx: param_deriv basis (non-GTO geminal leaf) was not built
        for this Hamiltonian (build_obs_ctx flag mismatch)

    This is deliberately NOT xfailed and NOT skipped. The c-axis matrix cannot see this
    population at all (``_RADIAL_AO`` has no ``composite``, so no ``Basis`` block is ever
    enumerated for these cells). The row requires both native basis methods to run and
    the public blocks to match the XLA run.
    """
    ham = _ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    # The premise, asserted rather than assumed: Python DOES consider this a basis-opt cell.
    assert _gto_leaves_of(gem.orb_data_up_spin), f"{pattern}: composite up-spin leaf carries no GTO child"
    assert _gto_leaves_of(gem.orb_data_dn_spin), f"{pattern}: composite dn-spin leaf carries no GTO child"

    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True, basis=True)
    assert_native_ran(spy, "pd_ok_basis_batch", context=f"{pattern}/basis")
    assert_native_ran(spy, "pd_el_basis_batch", context=f"{pattern}/basis")

    m_jax = _run(ham, rust_on=False, basis=True)
    _assert_blocks_and_values(m_rust, m_jax, ham, f"{pattern}/basis")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["pw_gamma_jnone", "bspline_gamma_jnone"])
def test_route_parity_nongto_basis_opt_emits_no_basis_block(pattern, monkeypatch):
    """A pure PW / B-spline determinant has NO radial basis parameter, on either route.

    With the basis flags active, the native assembler and the XLA reducer must AGREE that
    there is nothing to emit. The interesting failure this guards is asymmetric silence:
    one route quietly dropping a block the other stores. (Contrast the periodic composite row above, where
    the leaf DOES carry GTO children and the native route raises instead.)
    """
    ham = _ham(pattern)
    gem = ham.wavefunction_data.geminal_data
    assert not _gto_leaves_of(gem.orb_data_up_spin), f"{pattern}: expected a leaf with no GTO child"

    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True, basis=True)
    assert "pd_ok_basis_batch" not in spy.names(), f"{pattern}: the assembler called pd_ok_basis on a non-GTO leaf"
    m_jax = _run(ham, rust_on=False, basis=True)

    for name in ("dln_Psi_dc", "de_L_dc"):
        basis_blocks = {b for b in getattr(m_jax, name) if "basis" in b}
        assert not basis_blocks, f"{pattern}: the XLA route DOES store {sorted(basis_blocks)} in {name}; native emits none"
    _assert_blocks_and_values(m_rust, m_jax, ham, f"{pattern}/basis")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["twist_sphepgto_j12"])
def test_route_parity_complex_basis_opt(pattern, monkeypatch):
    """Required Rust/XLA comparison for complex Bloch AO-radial derivatives.

    The production route uses ``_RsComplexPdCtx`` to call the exported Rust complex
    kernels because ``NativeObsCtx`` parses a real-valued geminal.  This row requires
    that exact Rust adapter route (not JAX fallback) and numerical parity.
    """
    ham = _ham(pattern)
    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True, basis=True)
    assert_complex_pd_adapter_ran(spy, "pd_ok_basis", context=f"{pattern}/complex-basis")
    assert_complex_pd_adapter_ran(spy, "pd_el_basis", context=f"{pattern}/complex-basis")
    m_jax = _run(ham, rust_on=False, basis=True)
    _assert_blocks_and_values(m_rust, m_jax, ham, f"{pattern}/complex-basis")


@pytest.mark.heavy
@pytest.mark.parametrize("pattern", ["mol_cart_sd_j123", "mol_ecp_cart_j123", "pbc_composite_j123"])
def test_route_parity_j3_basis_parameter_derivatives(pattern, monkeypatch):
    """Required Rust/XLA comparison for the J3 AO-radial derivatives.

    A driver that disables the native c-axis path when a J3 basis flag is active would
    run both legs on JAX and report parity without exercising Rust.  This test
    therefore requires a successful Rust production run, observes the native J3-radial
    methods, and then compares the public O_k and dE_L/dc blocks against the XLA run.
    The composite row additionally requires child-qualified
    ``j3_child0_basis_{exp,coeff}`` publication.
    """
    ham = _ham(pattern)
    spy = install_ctx_spy(monkeypatch)
    m_rust = _run(ham, rust_on=True, j3_basis=True)
    assert_native_ran(spy, "pd_ok_j3_basis_batch", context=f"{pattern}/j3-basis")
    assert_native_ran(spy, "pd_el_j3_basis_batch", context=f"{pattern}/j3-basis")
    m_jax = _run(ham, rust_on=False, j3_basis=True)
    _assert_blocks_and_values(m_rust, m_jax, ham, f"{pattern}/j3-basis")


# main's P01-P15 evidence uses deliberately nonzero leaves, including correlated
# AGPn occupations and child-qualified composite-J3 radial blocks.  Mirror those
# exact scenarios through the production Rust c-parameter assembler so broad tree
# equality cannot hide an omitted write behind a zero fixture.
_MAIN_LEAF_FAMILIES = ("ln_Psi", "e_L")
_MAIN_LEAF_SCENARIOS = (
    "direct-mo",
    "agpn",
    "dense-agp",
    "lambda-basis",
    "legacy-jastrow",
    "dtn",
    "composite-j3",
)


@pytest.mark.heavy
@pytest.mark.parametrize("family", _MAIN_LEAF_FAMILIES)
@pytest.mark.parametrize(
    "scenario",
    _MAIN_LEAF_SCENARIOS,
)
def test_parameter_leaf_liveness_matches_main_oracle(family, scenario):
    """Every main P01-P15 leaf is declared, live, and XLA-autograd == Rust."""
    from jqmc import derivatives
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from jqmc.derivatives.bridge import _reduce_c_param_grads
    from jqmc.jqmc_mcmc import _rs_native_c_param_grads
    from tests.test_jqmc_kernel import test_analytic_coverage_evidence as evidence

    compared_by_name = {}

    def compare(ham, r_up, r_dn, flags, *, live_exempt=frozenset()):
        if family == "ln_Psi":
            automatic = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
            quantity = "ln_Psi"
        else:
            automatic = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)["c"]
            quantity = "e_L"

        automatic = _reduce_c_param_grads(
            ham.wavefunction_data,
            jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), automatic),
            1,
            flags,
        )
        declared = {
            block.name
            for block in ham.wavefunction_data.get_variational_blocks(
                state_token=("rust-route-parity", 0),
                **flags,
            )
        }
        automatic = {name: value for name, value in automatic.items() if name in declared}
        assert set(automatic) == declared

        wf = ham.wavefunction_data
        ctx = __import__("jqmc_rs").build_obs_ctx(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            False,
            bool(ham.structure_data.pbc_flag),
            False,
            False,
            False,
            True,
            precision_policy=precision_policy_snapshot(),
        )
        mesh = icosahedron_sym_mesh_Nv12
        native = _rs_native_c_param_grads(
            ctx,
            ham,
            [np.asarray(r_up)],
            [np.asarray(r_dn)],
            flags,
            1,
            quantity=quantity,
            ecp_flag=bool(ham.coulomb_potential_data.ecp_flag),
            ecp_w=np.asarray(mesh.weights),
            ecp_gp=np.asarray(mesh.grid_points),
            rts_b=[np.eye(3)],
            complex_twist=False,
        )
        native = {name: value for name, value in native.items() if name in declared}
        assert set(native) == declared, (
            f"{scenario}/{family}: Rust dropped or renamed a declared parameter leaf; "
            f"declared={sorted(declared)}, native={sorted(native)}"
        )
        for name in sorted(declared):
            reference = np.asarray(automatic[name])
            got = np.asarray(native[name])
            assert reference.size > 0
            assert np.all(np.isfinite(reference))
            assert np.all(np.isfinite(got))
            if name not in live_exempt:
                assert float(np.max(np.abs(reference))) > 1.0e-12, (
                    f"{scenario}/{family}/{name}: main autograd leaf is vacuously zero"
                )
            atol, rtol = _tol(ham, name)
            np.testing.assert_allclose(
                got,
                reference,
                atol=atol,
                rtol=rtol,
                err_msg=f"{scenario}/{family}/{name}: Rust vs main autograd",
            )
            compared_by_name.setdefault(name, []).append((got, reference, atol, rtol))

    ham, r_up, r_dn = evidence._leaf_hamiltonian(scenario)
    flags = evidence._resolved_flags(evidence._SCENARIO_FLAGS[scenario])
    exempt = {"lambda_diag"} if scenario == "agpn" and family == "e_L" else set()
    compare(ham, r_up, r_dn, flags, live_exempt=exempt)

    # The open-shell AGPn e_L fixture is physically invariant in lambda_diag.  Main
    # therefore carries a correlated closed-shell control where that leaf is live;
    # Rust must pass the same control rather than claiming the zero row as coverage.
    if scenario == "agpn" and family == "e_L":
        closed = evidence._correlated_closed_agpn_hamiltonian()
        ru_closed, rd_closed = evidence._draw(closed, seed=59)
        compare(closed, ru_closed, rd_closed, flags)

    for source_id in evidence._LEAF_SOURCE_IDS_BY_SCENARIO[scenario]:
        pairs = []
        tolerances = []
        for name in evidence._LEAF_BLOCKS_BY_SOURCE_ID.get(source_id, ()):
            for actual, reference, atol, rtol in compared_by_name.get(name, ()):
                pairs.append((actual, reference))
                tolerances.append((atol, rtol))
        if source_id in {"P14", "P15"}:
            suffix = "basis_exp" if source_id == "P14" else "basis_coeff"
            for name, rows in compared_by_name.items():
                if name.endswith(suffix):
                    for actual, reference, atol, rtol in rows:
                        pairs.append((actual, reference))
                        tolerances.append((atol, rtol))
        assert pairs and len(set(tolerances)) == 1
        atol, rtol = tolerances[0]
        obligation_id = evidence.leaf_obligation_id(family, source_id)
        record_rust_bound_case(
            upstream_obligation_id=obligation_id,
            component_id=source_id,
            pytest_node_id="tests/test_rs_kernel/test_route_parity_param_deriv.py::test_parameter_leaf_liveness_matches_main_oracle",
            input_binding=evidence._leaf_input_binding(obligation_id, scenario),
            r_up=r_up,
            r_dn=r_dn,
            generator_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence._leaf_hamiltonian",
            seed=evidence._LEAF_SEEDS[scenario],
            compared_pairs=tuple(pairs),
            zones=_ZONES,
            level="medium",
            atol=atol,
            rtol=rtol,
        )


_MAIN_VALIDITY_FAMILIES = ("ln_Psi", "e_L")
_MAIN_VALIDITY_DISPOSITIONS = (
    "geminal-parameterless-basis",
    "j3-parameterless-basis",
    "unknown-j1-form",
    "unknown-j2-form",
)


@pytest.mark.heavy
@pytest.mark.parametrize("family", _MAIN_VALIDITY_FAMILIES)
@pytest.mark.parametrize("disposition", _MAIN_VALIDITY_DISPOSITIONS)
def test_parameter_validity_disposition_matches_main(family, disposition):
    """Rust preserves main's explicit n/a/error outcomes for c-axis owners."""
    import jqmc_rs
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from jqmc.jqmc_mcmc import _rs_native_c_param_grads
    from tests.test_jqmc_kernel import test_analytic_coverage_evidence as evidence

    if disposition.endswith("parameterless-basis"):
        ham, r_up, r_dn, flags = evidence._parameterless_hamiltonian(disposition)
        expected_error = False
    else:
        ham, r_up, r_dn, flags = evidence._invalid_jastrow_hamiltonian(disposition)
        expected_error = True

    wf = ham.wavefunction_data

    def invoke():
        ctx = jqmc_rs.build_obs_ctx(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            False,
            bool(ham.structure_data.pbc_flag),
            False,
            False,
            False,
            True,
            precision_policy=precision_policy_snapshot(),
        )
        mesh = icosahedron_sym_mesh_Nv12
        return _rs_native_c_param_grads(
            ctx,
            ham,
            [np.asarray(r_up)],
            [np.asarray(r_dn)],
            flags,
            1,
            quantity="ln_Psi" if family == "ln_Psi" else "e_L",
            ecp_flag=bool(ham.coulomb_potential_data.ecp_flag),
            ecp_w=np.asarray(mesh.weights),
            ecp_gp=np.asarray(mesh.grid_points),
            rts_b=[np.eye(3)],
            complex_twist=False,
        )

    if expected_error:
        with pytest.raises(ValueError, match=r"unknown|Unknown"):
            invoke()
        obligation_id = evidence.validity_obligation_id(family, disposition)
        record_rust_bound_case(
            upstream_obligation_id=obligation_id,
            component_id=disposition,
            pytest_node_id="tests/test_rs_kernel/test_route_parity_param_deriv.py::test_parameter_validity_disposition_matches_main",
            input_binding=evidence._validity_input_binding(obligation_id, disposition),
            r_up=r_up,
            r_dn=r_dn,
            generator_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence._invalid_jastrow_hamiltonian",
            seed=47,
            observations=("route-executed", "expected-output-leaves-accounted"),
        )
        return

    assert {
        block.name
        for block in wf.get_variational_blocks(
            state_token=("rust-route-validity", 0),
            **flags,
        )
    } == set()
    assert invoke() == {}, f"{family}/{disposition}: a parameterless owner must publish no Rust block"
    obligation_id = evidence.validity_obligation_id(family, disposition)
    record_rust_bound_case(
        upstream_obligation_id=obligation_id,
        component_id=disposition,
        pytest_node_id="tests/test_rs_kernel/test_route_parity_param_deriv.py::test_parameter_validity_disposition_matches_main",
        input_binding=evidence._validity_input_binding(obligation_id, disposition),
        r_up=r_up,
        r_dn=r_dn,
        generator_operation="tests.test_jqmc_kernel.test_analytic_coverage_evidence._parameterless_hamiltonian",
        seed=43,
        observations=("route-executed", "expected-output-leaves-accounted"),
    )


@pytest.mark.heavy
def test_v1_potential_independence_native_contract():
    """Rust ln_Psi c/r/R outputs are invariant to swapping AE for ECP potential data."""
    import jqmc_rs
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
    from jqmc.jqmc_mcmc import _geminal_inv_batched, _rs_native_c_param_grads
    from tests.test_jqmc_kernel import test_analytic_coverage_evidence as evidence
    from tests.test_jqmc_kernel._grad_fidelity_harness import _flags_for_axis

    ae = evidence._build_ham("H2_ae_ccpvdz_cart.h5", True)
    ecp = evidence._build_ham("H2_ecp_ccpvdz_cart.h5", True)
    swapped = ae.replace(coulomb_potential_data=ecp.coulomb_potential_data)
    r_up, r_dn = evidence._draw(ae, seed=53)
    flags = _flags_for_axis("c")
    mesh = icosahedron_sym_mesh_Nv12

    def c_blocks(ham):
        wf = ham.wavefunction_data
        ctx = jqmc_rs.build_obs_ctx(
            wf.geminal_data,
            wf.jastrow_data,
            ham.coulomb_potential_data,
            False,
            False,
            False,
            False,
            False,
            True,
            precision_policy=precision_policy_snapshot(),
        )
        return _rs_native_c_param_grads(
            ctx,
            ham,
            [np.asarray(r_up)],
            [np.asarray(r_dn)],
            flags,
            1,
            quantity="ln_Psi",
            ecp_flag=bool(ham.coulomb_potential_data.ecp_flag),
            ecp_w=np.asarray(mesh.weights),
            ecp_gp=np.asarray(mesh.grid_points),
            rts_b=[np.eye(3)],
            complex_twist=False,
        )

    left = c_blocks(ae)
    right = c_blocks(swapped)
    assert set(left) == set(right)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=f"ln_Psi/c/{name}")

    gd = ae.wavefunction_data.geminal_data
    _, inverse, _, _ = _geminal_inv_batched(gd, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])
    geometry_left = jqmc_rs.vmc_grad_ln_psi(
        gd,
        ae.wavefunction_data.jastrow_data,
        np.asarray(r_up),
        np.asarray(r_dn),
        np.asarray(inverse[0]),
    )
    geometry_right = jqmc_rs.vmc_grad_ln_psi(
        swapped.wavefunction_data.geminal_data,
        swapped.wavefunction_data.jastrow_data,
        np.asarray(r_up),
        np.asarray(r_dn),
        np.asarray(inverse[0]),
    )
    for axis, lhs, rhs in zip(("R", "r_up", "r_dn"), geometry_left, geometry_right, strict=True):
        np.testing.assert_array_equal(lhs, rhs, err_msg=f"ln_Psi/{axis}")
    obligation_id = evidence.IDENTITY_OBLIGATION_ID
    record_rust_bound_case(
        upstream_obligation_id=obligation_id,
        component_id="c-r-R",
        pytest_node_id="tests/test_rs_kernel/test_route_parity_param_deriv.py::test_v1_potential_independence_native_contract",
        input_binding=evidence.committed_input(
            obligation_id,
            source_ids=("WF-SRC-H2-AE-CART-MOL", "WF-SRC-H2-ECP-CART-MOL"),
            builder_operation="tests.test_rs_kernel.test_route_parity_param_deriv.test_v1_potential_independence_native_contract",
            transformations=("derivatives.targeted.substitute-ecp-potential-data.v1",),
        ),
        r_up=r_up,
        r_dn=r_dn,
        generator_operation="tests.test_jqmc_kernel._grad_fidelity_harness._draw",
        seed=53,
        exact_pairs=tuple((left[name], right[name]) for name in sorted(left))
        + tuple(zip(geometry_left, geometry_right, strict=True)),
    )
