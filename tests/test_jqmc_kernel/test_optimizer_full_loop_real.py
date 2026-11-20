"""C12 full-loop real-coordinate optimization matrix.

The sample payload is deterministic, but every wave-function object, block
publisher, coordinate solve, decode, and ansatz retraction is production code.
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
import weakref
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from jqmc._setting import atol_consistency, rtol_consistency
from jqmc.atomic_orbital import compute_overlap_matrix
from jqmc.determinant import AGPn_data, SD_data, to_AGP, to_AGPn
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    EnergyEstimate,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    OptimizationConfig,
    ParameterSelection,
    run_optimize,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

from .complex_opt_ground_truth_harness import _build_ham


pytestmark = [pytest.mark.light, pytest.mark.with_chdir]
_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")


class _Comm:
    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0

    def Barrier(self):
        return None

    def allreduce(self, value, op=None):
        return value

    def Allreduce(self, send, receive, op=None):
        np.copyto(np.asarray(receive[0]), np.asarray(send[0]))

    def allgather(self, value):
        return [value]

    def bcast(self, value, root=0):
        return value

    def Allgatherv(self, send, receive):
        np.asarray(receive[0]).reshape(-1)[:] = np.asarray(send[0]).reshape(-1)


def _source(block, components):
    if block.derivative_policy == "direct_real":
        return BlockDerivativeSource.build(block=block, direct=components[0])
    if block.derivative_policy == "holomorphic":
        return BlockDerivativeSource.build(block=block, d_p=components[0])
    return BlockDerivativeSource.build(block=block, d_p=components[0], d_pbar=components[1])


def _component_values(block, *, sample_count, seed):
    rng = np.random.default_rng(seed)
    shape = (sample_count, *block.shape)
    primary = rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)
    if block.derivative_policy == "wirtinger_pair":
        secondary = 0.3 * (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape))
        return primary, secondary
    return (primary,)


def _binned_component(values, weights, energies, *, conjugate=False, with_energy=False):
    value = np.conj(values) if conjugate else values
    expansion = (slice(None),) + (None,) * (value.ndim - 1)
    weighted = weights[expansion] * value
    if with_energy:
        weighted = weighted * energies[expansion]
    return np.stack([np.sum(part, axis=0) for part in np.split(weighted, 4)], axis=0)


def _batch(blocks, token):
    sample_count = 8
    weights = np.asarray([1.0, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0])
    energies = np.asarray([-1.2 + 0.1j, -0.7 - 0.2j, -1.0 + 0.3j, -0.4, -1.4, -0.9 + 0.2j, -0.6, -1.1])
    sources = []
    moments = []
    for index, block in enumerate(blocks):
        components = _component_values(block, sample_count=sample_count, seed=100 + index)
        source = _source(block, components)
        sources.append(source)
        sums = tuple(_binned_component(value, weights, energies) for value in components)
        conjugate_energy_sums = tuple(
            _binned_component(value, weights, energies, conjugate=True, with_energy=True) for value in components
        )
        moments.append(
            JackknifeBlockMoments.build(
                block=block,
                source_sum=_source(block, sums),
                conjugate_source_energy_sum=_source(block, conjugate_energy_sums),
            )
        )
    batch_token = "c12-full-loop:1"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=batch_token,
        bin_sample_counts=np.full(4, 2, dtype=np.int64),
        bin_weight_sums=np.asarray([np.sum(part) for part in np.split(weights, 4)]),
        bin_weighted_energy_sums=np.asarray([np.sum(part) for part in np.split(weights * energies, 4)]),
        log_source_moments=tuple(moments),
    )
    return OptimizationBatch.build(
        blocks=blocks,
        batch_token=batch_token,
        state_token=token,
        weights=weights,
        local_energies=energies,
        log_derivative_sources=tuple(sources),
        local_energy_derivative_sources=None,
        jackknife_bin_metadata=metadata,
    )


@dataclass
class _Sampler:
    hamiltonian_data: Hamiltonian_data
    comm: _Comm = field(default_factory=_Comm)
    world_comm: _Comm = field(default_factory=_Comm)
    rank: int = 0
    size: int = 1
    optimization_step_index: int = 0
    wavefunction_state_token: tuple[str, int] = ("c12-full-loop", 0)

    def derivative_capabilities(self):
        return True, True

    def configure_optimization(self, *, method, hyperparameters):
        return None

    def select_parameters(self, flags):
        return None

    def sample(self, *, num_steps, max_time):
        return None

    def estimate_energy(self, *, num_warmup_steps, num_bin_blocks):
        return EnergyEstimate(mean=-1.0, standard_error=0.05, variance=0.2, variance_standard_error=0.01)

    def variational_blocks(self, parameters):
        return self.hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=self.wavefunction_state_token,
            **parameters.variational_block_kwargs(),
        )

    def optimization_batch(self, *, num_warmup_steps, num_bin_blocks, blocks, need_local_energy_derivatives):
        assert not need_local_energy_derivatives
        return _batch(tuple(blocks), self.wavefunction_state_token)

    def validate_wavefunction_candidate(self, candidate):
        # This sampler supplies an exact synthetic estimator batch rather
        # than walkers. Keep the candidate's actual model validation live.
        candidate.sanity_check()

    def advance_wavefunction_revision(self):
        lineage, revision = self.wavefunction_state_token
        self.wavefunction_state_token = (lineage, revision + 1)

    def advance_optimization_steps(self, count):
        self.optimization_step_index += int(count)


def _disable_side_effects(monkeypatch):
    monkeypatch.setattr("jqmc.jqmc_optimizer._check_mpi4py_jax_distribution_consistency", lambda: None)
    monkeypatch.setattr("jqmc.jqmc_optimizer._init_external_control_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("jqmc.jqmc_optimizer._remove_external_control_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("jqmc.jqmc_optimizer._log_memory_usage", lambda *args, **kwargs: None)


def _config(*, fix_rank=None, num_opt_steps=1):
    return OptimizationConfig(
        num_sampling_steps=8,
        num_opt_steps=num_opt_steps,
        num_bin_blocks=4,
        use_device_optimizer=False,
        fix_geminal_rank=fix_rank,
        parameters=ParameterSelection(
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=True,
            opt_JDTN_u_param=False,
            opt_JDTN_chi_param=False,
            opt_JDTN_f_param=False,
        ),
        optimizer_kwargs={"method": "sr", "delta": 2.0e-4, "epsilon": 0.2, "cg_flag": False},
    )


def _assert_mo_invariants(geminal):
    for orbitals in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        coefficients = np.asarray(orbitals.mo_coefficients)
        overlap = np.asarray(compute_overlap_matrix(orbitals.aos_data))
        gram = coefficients.conj() @ overlap @ coefficients.T
        np.testing.assert_allclose(
            gram,
            np.eye(coefficients.shape[0]),
            rtol=rtol_consistency,
            atol=atol_consistency,
        )


class _BatchLifetimeSampler(_Sampler):
    """Assert the previous production batch dies before the next sampling."""

    def sample(self, *, num_steps, max_time):
        del num_steps, max_time
        if getattr(self, "batch_reference", None) is not None:
            assert self.batch_reference() is None

    def optimization_batch(self, **kwargs):
        batch = super().optimization_batch(**kwargs)
        self.batch_reference = weakref.ref(batch)
        return batch


def test_full_loop_releases_previous_batch_before_next_sampling(monkeypatch):
    """Full loop releases previous batch before next sampling."""
    _disable_side_effects(monkeypatch)
    sampler = _BatchLifetimeSampler(_build_ham("twist"))

    run_optimize(sampler, _config(num_opt_steps=2))

    assert sampler.wavefunction_state_token == ("c12-full-loop", 2)


def test_full_loop_consumes_the_aggregated_step_energy(monkeypatch):
    """Full loop consumes the aggregated step energy."""
    _disable_side_effects(monkeypatch)
    sampler = _Sampler(_build_ham("twist"))
    aggregated = EnergyEstimate(
        mean=-2.5,
        standard_error=0.0125,
        variance=0.3,
        variance_standard_error=0.02,
    )
    seen = []

    def aggregate(local):
        seen.append(local)
        return aggregated

    config = replace(_config(), energy_aggregator=aggregate)
    run_optimize(sampler, config)

    assert seen == [EnergyEstimate(mean=-1.0, standard_error=0.05, variance=0.2, variance_standard_error=0.01)]


def _open_shell_ham():
    structure, _aos, _up, _dn, geminal, coulomb = read_trexio_file(
        os.path.join(_TREXIO, "Li_ae_ccpvdz_cart.h5"), store_tuple=True
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=0.7,
        jastrow_2b_type="exp",
        structure_data=structure,
    )
    return Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(jastrow_two_body_data=j2),
            geminal_data=geminal,
        ),
        coulomb_potential_data=coulomb,
    )


def test_gamma_agp_is_converted_and_retracted_as_time_reversal_agpn(monkeypatch):
    """Gamma agp is converted and retracted as time reversal agpn."""
    _disable_side_effects(monkeypatch)
    ham = _build_ham("gamma")
    source = ham.wavefunction_data.geminal_data
    source = source.replace(
        time_reversal=True,
        orb_data_dn_spin=source.orb_data_dn_spin.replace(mo_coefficients=np.conj(source.orb_data_up_spin.mo_coefficients)),
    )
    agp = to_AGP(source)
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=agp))
    sampler = _Sampler(ham)

    run_optimize(sampler, _config(fix_rank=1))

    geminal = sampler.hamiltonian_data.wavefunction_data.geminal_data
    assert isinstance(geminal, AGPn_data)
    assert geminal.time_reversal is True
    np.testing.assert_array_equal(
        geminal.orb_data_dn_spin.mo_coefficients,
        np.conj(geminal.orb_data_up_spin.mo_coefficients),
    )
    _assert_mo_invariants(geminal)
    assert sampler.wavefunction_state_token == ("c12-full-loop", 1)


def test_non_gamma_mixed_jastrow_complex_mo_and_real_lambda_full_step(monkeypatch):
    """Non gamma mixed jastrow complex MO and real lambda full step."""
    _disable_side_effects(monkeypatch)
    ham = _build_ham("twist")
    agpn = to_AGPn(ham.wavefunction_data.geminal_data)
    agpn = agpn.replace(
        time_reversal=True,
        orb_data_dn_spin=agpn.orb_data_dn_spin.replace(mo_coefficients=np.conj(agpn.orb_data_up_spin.mo_coefficients)),
    )
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=agpn))
    before = ham.wavefunction_data.get_variational_blocks(
        state_token=("c12-full-loop", 0), **_config().parameters.variational_block_kwargs()
    )
    sampler = _Sampler(ham)

    run_optimize(sampler, _config())

    after_wf = sampler.hamiltonian_data.wavefunction_data
    after = after_wf.get_variational_blocks(state_token=("c12-full-loop", 1), **_config().parameters.variational_block_kwargs())
    names = [block.name for block in after]
    assert names == ["j2_param", "mo_coefficients_up", "lambda_diag"]
    assert any(not np.array_equal(np.asarray(a.values), np.asarray(b.values)) for a, b in zip(after, before, strict=True))
    assert np.iscomplexobj(after[names.index("mo_coefficients_up")].values)
    assert not np.iscomplexobj(after[names.index("lambda_diag")].values)
    _assert_mo_invariants(after_wf.geminal_data)


@pytest.mark.parametrize("agp_to_agpn", [True, False], ids=["agp-to-agpn", "sd"])
def test_non_gamma_rank_manifold_full_step(monkeypatch, agp_to_agpn):
    """Cover both rank conversion and native SD on the complex twist fixture."""
    _disable_side_effects(monkeypatch)
    ham = _build_ham("twist")
    source_sd = ham.wavefunction_data.geminal_data
    before_up = np.asarray(source_sd.orb_data_up_spin.mo_coefficients).copy()
    before_dn = np.asarray(source_sd.orb_data_dn_spin.mo_coefficients).copy()
    if agp_to_agpn:
        ham = ham.replace(
            wavefunction_data=ham.wavefunction_data.replace(geminal_data=to_AGP(ham.wavefunction_data.geminal_data))
        )
    sampler = _Sampler(ham)

    run_optimize(sampler, _config(fix_rank=1 if agp_to_agpn else None))

    geminal = sampler.hamiltonian_data.wavefunction_data.geminal_data
    assert isinstance(geminal, AGPn_data if agp_to_agpn else SD_data)
    assert geminal.time_reversal is False
    assert np.iscomplexobj(geminal.orb_data_up_spin.mo_coefficients)
    assert np.iscomplexobj(geminal.orb_data_dn_spin.mo_coefficients)
    assert not np.array_equal(np.asarray(geminal.orb_data_up_spin.mo_coefficients), before_up)
    assert not np.array_equal(np.asarray(geminal.orb_data_dn_spin.mo_coefficients), before_dn)
    _assert_mo_invariants(geminal)


@pytest.mark.parametrize("promote_to_agpn", [False, True], ids=["sd", "agpn"])
def test_open_shell_independent_spin_full_step(monkeypatch, promote_to_agpn):
    """Open shell independent spin full step."""
    from .optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd

    _disable_side_effects(monkeypatch)
    ham = _open_shell_ham()
    if promote_to_agpn:
        ham = ham.replace(
            wavefunction_data=ham.wavefunction_data.replace(
                geminal_data=build_explicit_common_rank_agpn_from_sd(ham.wavefunction_data.geminal_data)
            )
        )
    before_geminal = ham.wavefunction_data.geminal_data
    before_up = np.asarray(before_geminal.orb_data_up_spin.mo_coefficients).copy()
    before_dn = np.asarray(before_geminal.orb_data_dn_spin.mo_coefficients).copy()
    sampler = _Sampler(ham)

    run_optimize(sampler, _config())

    geminal = sampler.hamiltonian_data.wavefunction_data.geminal_data
    assert isinstance(geminal, AGPn_data if promote_to_agpn else SD_data)
    assert geminal.time_reversal is False
    assert not np.array_equal(np.asarray(geminal.orb_data_up_spin.mo_coefficients), before_up)
    assert not np.array_equal(np.asarray(geminal.orb_data_dn_spin.mo_coefficients), before_dn)
    if promote_to_agpn:
        blocks = sampler.variational_blocks(_config().parameters)
        assert [block.name for block in blocks][-2:] == ["lambda_diag", "lambda_unpaired"]
    _assert_mo_invariants(geminal)


class _FakeClock:
    """Deterministic stand-in for ``time.perf_counter`` in the optimizer module."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class _TimedSampler(_Sampler):
    """Advance the fake clock by fixed amounts in sampling and batch construction."""

    clock: _FakeClock | None = None

    def sample(self, *, num_steps, max_time):
        del num_steps, max_time
        self.clock.now += 1.0

    def optimization_batch(self, **kwargs):
        batch = super().optimization_batch(**kwargs)
        self.clock.now += 3.0
        return batch


def _logged_seconds(caplog, label):
    prefix = f"Time for {label} = "
    values = [rec.getMessage().strip() for rec in caplog.records if rec.getMessage().strip().startswith(prefix)]
    assert len(values) == 1, values
    return float(values[0][len(prefix) :].split()[0])


def test_step_timing_books_each_interval_once(monkeypatch, caplog):
    """The per-step timing summary books the fused batch+solver call once (gF + SR, misc >= 0)."""
    import logging
    import types

    import jqmc.jqmc_optimizer as optimizer_module

    _disable_side_effects(monkeypatch)
    clock = _FakeClock()
    monkeypatch.setattr(optimizer_module, "time", types.SimpleNamespace(perf_counter=clock.perf_counter))
    production_target = optimizer_module.compute_real_coordinate_optimization_target

    targets = []

    def timed_target(*args, **kwargs):
        target = production_target(*args, **kwargs)
        targets.append(target)
        clock.now += 5.0
        return target

    monkeypatch.setattr(optimizer_module, "compute_real_coordinate_optimization_target", timed_target)
    sampler = _TimedSampler(_build_ham("twist"))
    sampler.clock = clock

    with caplog.at_level(logging.INFO, logger="jqmc"):
        run_optimize(sampler, _config(num_opt_steps=1))

    assert sampler.optimization_step_index == 1
    assert _logged_seconds(caplog, "sampling") == 1.0
    assert _logged_seconds(caplog, "computing generalized forces (gF)") == 3.0
    assert _logged_seconds(caplog, "optimizer (SR)") == 5.0
    assert _logged_seconds(caplog, "misc. (others)") == 0.0
    # The logged max force keeps its error: the printed values are the computed
    # ones rounded to enough decimals for two significant digits of the error.
    (line,) = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Max generalized force = ")]
    (target,) = targets
    force = np.asarray(target.coordinate_force)
    argmax = int(np.argmax(np.abs(force)))
    error = float(np.asarray(target.coordinate_force_std)[argmax])
    assert error > 0.0
    _assert_value_and_error_text(line.split("=")[1].split("Ha/a.u.")[0], float(force[argmax]), error)
    # The same formatter at errors this fixture does not reach: a fixed %.3f printed
    # every error below 5e-4 as 0.000; from 0.01 up the previous text is kept.
    for small_or_large in (3.4e-6, 1.2e-4, 4.9e-4, 5.0e-3, 0.012, 0.33):
        text = optimizer_module._format_value_and_error(-0.0291234, small_or_large)
        _assert_value_and_error_text(text, -0.0291234, small_or_large)
        if small_or_large >= 0.01:
            assert text == f"{-0.0291234:.3f} +- {small_or_large:.3f}"


def _assert_value_and_error_text(text: str, value: float, error: float) -> None:
    """``"value +- error"`` is value and error rounded alike, with two or more significant digits of the error."""
    printed_value, printed_error = (token.strip() for token in text.split("+-"))
    decimals = len(printed_error.split(".")[1])
    assert len(printed_value.split(".")[1]) == decimals
    significant_digits = printed_error.replace(".", "").lstrip("0")
    assert significant_digits[1:], f"fewer than two significant digits: {printed_error}"
    assert float(printed_error) == round(error, decimals)
    assert float(printed_value) == round(value, decimals)
