"""Independent pressure/stress estimator reference on real periodic driver runs.

``get_aP`` returns ``(-diag(sigma_mean), diag(sigma_std))`` and
``get_P_hydrostatic`` the trace-first jackknife of ``-tr(sigma)/3``. The older
``test_aT_aP_diagonal_consistency_*`` nodes compare ``get_aP`` with a slice of
``get_aT`` -- the same computation re-indexed. Here the pressures are
recomputed from the STORED per-step components of a real periodic run
(lithium.h5, 2-atom Li ccECP cell) with a plain delete-one (time block, walker)
jackknife written from the documented estimator, and compared with the public
accessors.

Documented estimator (``MCMC.get_aT`` docstring, mirrored by GFMC):

    sigma = < sigma_HF > + 2 ( < E_L sigma_PP > - < E_L > < sigma_PP > ),

weighted averages over the retained samples, real part, symmetrized, divided
by the cell volume |a . (b x c)|. ``P_i = -sigma_ii``; ``P_hydro = -tr(sigma)/3``.
Errors are jackknife errors ``sqrt((n - 1)/n * sum_j (x_j - mean)^2)`` over the
n replicates; the hydrostatic error is taken over the replicate traces.

The GFMC weights are rebuilt from the stored bare branching weights as the
product over the previous ``num_gfmc_collect_steps`` steps; the GFMC
observables drop the first ``num_gfmc_collect_steps`` rows, as documented for
the ``w_L`` / ``e_L`` accessors. The cell volume is computed from ``vec_a/b/c``.

The comparison is on fixed stored data (no sampling-error band): each scalar
at the same-mode ``local_energy`` strict tolerance. Sensitivity controls show
that the same tolerance rejects a reference with the Pulay factor 2 replaced
by 1, a reference without the jackknife (n - 1) factor, and the quadrature
hydrostatic error that ignores the covariance of the three normal stresses.
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

import os

import jax
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_PERIODIC_FIXTURE = "lithium.h5"
_SEED = 34356
_NUM_WALKERS = 2
_MCMC_STEPS = 8
_GFMC_STEPS = 12
_GFMC_COLLECT = 2
_WARMUP = 1
_BINS = 3


def _build_hamiltonian() -> Hamiltonian_data:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", _PERIODIC_FIXTURE)
    structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=path, store_tuple=True
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)
    return Hamiltonian_data.build(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
    )


def _run(driver: str):
    hamiltonian_data = _build_hamiltonian()
    if driver == "mcmc":
        run = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=_SEED,
            num_walkers=_NUM_WALKERS,
            comput_position_deriv=False,
            comput_stress=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            epsilon_AS=1.0e-2,
        )
        run.run(num_mcmc_steps=_MCMC_STEPS)
        return run
    common = dict(
        hamiltonian_data=hamiltonian_data,
        num_walkers=_NUM_WALKERS,
        num_gfmc_collect_steps=_GFMC_COLLECT,
        mcmc_seed=_SEED,
        alat=0.30,
        non_local_move="tmove",
        comput_position_deriv=False,
        comput_stress=True,
    )
    if driver == "gfmc_t":
        run = GFMC_t(**common, tau=0.10)
    else:
        run = GFMC_n(**common, num_mcmc_per_measurement=2, E_scf=0.0)
    run.run(num_mcmc_steps=_GFMC_STEPS)
    return run


def _stored_components(run, driver: str):
    """Per-step (w, e_L, sigma_HF, sigma_PP, E_L sigma_PP) after warmup, from the stored buffers."""
    if driver == "mcmc":
        w = np.asarray(run.w_L)
        e = np.asarray(run.e_L)
        hf = np.asarray(run.stress_HF_stored)
        pp = np.asarray(run.stress_PP_stored)
        elpp = np.asarray(run.E_L_stress_PP_stored)
    else:
        collect = run.num_gfmc_collect_steps
        bare = np.asarray(run._stored_w_L, dtype=np.float64)
        w = np.stack([np.prod(bare[n - collect : n], axis=0) for n in range(collect, bare.shape[0])])
        e = np.asarray(run._stored_e_L)[collect:]
        hf = np.asarray(run._stored_stress_HF)[collect:]
        pp = np.asarray(run._stored_stress_PP)[collect:]
        elpp = np.asarray(run._stored_E_L_stress_PP)[collect:]
    return tuple(np.asarray(a)[_WARMUP:] for a in (w, e, hf, pp, elpp))


def _cell_volume(run) -> float:
    structure = run.hamiltonian_data.structure_data
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (structure.vec_a, structure.vec_b, structure.vec_c))
    return float(abs(np.dot(a, np.cross(b, c))))


def _reference(components, volume, bins, *, pulay_factor=2.0, jackknife_factor=True):
    """Delete-one (time block, walker) jackknife of the documented stress estimator."""
    w, e, hf, pp, elpp = (np.asarray(x) for x in components)
    w = w.astype(np.float64)
    replicates = []
    for block in np.array_split(np.arange(w.shape[0]), bins):
        for walker in range(w.shape[1]):
            keep = np.ones(w.shape, dtype=bool)
            keep[block, walker] = False
            kept_w = w[keep]
            total = np.sum(kept_w)
            mean_hf = np.einsum("n,nij->ij", kept_w, hf[keep]) / total
            mean_pp = np.einsum("n,nij->ij", kept_w, pp[keep]) / total
            mean_elpp = np.einsum("n,nij->ij", kept_w, elpp[keep]) / total
            mean_e = np.dot(kept_w, e[keep]) / total
            sigma = np.real(mean_hf + pulay_factor * (mean_elpp - mean_e * mean_pp))
            replicates.append(0.5 * (sigma + sigma.T) / volume)
    replicates = np.asarray(replicates)
    count = replicates.shape[0]
    factor = (count - 1) / count if jackknife_factor else 1.0 / count

    def summarize(values):
        mean = np.mean(values, axis=0)
        return mean, np.sqrt(factor * np.sum((values - mean) ** 2, axis=0))

    sigma_mean, sigma_std = summarize(replicates)
    pressure_mean, pressure_std = summarize(-np.diagonal(replicates, axis1=1, axis2=2))
    hydro_mean, hydro_std = summarize(-np.trace(replicates, axis1=1, axis2=2) / 3.0)
    return {
        "sigma": (sigma_mean, sigma_std),
        "pressure": (pressure_mean, pressure_std),
        "hydrostatic": (float(hydro_mean), float(hydro_std)),
    }


def _assert_pressure_matches(observed, reference, label):
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    for part, name in enumerate(("mean", "error")):
        np.testing.assert_allclose(
            np.asarray(observed[part], dtype=np.float64),
            np.asarray(reference[part], dtype=np.float64),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"{label} {name} differs from the independent stored-component jackknife",
        )


def _public_estimates(run):
    return {
        "sigma": run.get_aT(_WARMUP, _BINS),
        "pressure": run.get_aP(_WARMUP, _BINS),
        "hydrostatic": run.get_P_hydrostatic(_WARMUP, _BINS),
    }


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["mcmc", "gfmc_t", "gfmc_n"])
def test_get_aP_and_hydrostatic_pressure_match_stored_component_jackknife(driver):
    """get_aP, get_P_hydrostatic and get_aT equal an independent jackknife of the stored stress components.

    One short periodic run per driver (lithium.h5, no Jastrow, 2 walkers,
    warmup 1, 3 bins; GFMC with 2 collect steps). The reference uses only the
    stored per-step weights, energies and stress components, rebuilt GFMC
    weights and the cell volume from the lattice vectors. Means and errors of
    the three normal pressures, the hydrostatic pressure and the full tensor
    are compared at the local_energy strict tolerance of the running mode.

    Non-vacuity: every normal pressure and its error are larger than the
    tolerance. Sensitivity (same data, same tolerance): a reference with
    Pulay factor 1, one without the (n - 1) jackknife factor, and the
    quadrature hydrostatic error each disagree with production.
    """
    run = _run(driver)
    assert bool(run.hamiltonian_data.structure_data.pbc_flag)
    components = _stored_components(run, driver)
    volume = _cell_volume(run)
    reference = _reference(components, volume, _BINS)
    public = _public_estimates(run)

    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    pressure_mean, pressure_std = reference["pressure"]
    assert np.all(np.abs(pressure_mean) > atol + rtol * np.abs(pressure_mean)), pressure_mean
    assert np.all(pressure_std > atol + rtol * np.abs(pressure_mean)), pressure_std

    for key in ("pressure", "hydrostatic", "sigma"):
        _assert_pressure_matches(public[key], reference[key], f"{driver} {key}")

    no_factor_two = _reference(components, volume, _BINS, pulay_factor=1.0)
    with pytest.raises(AssertionError, match="pressure mean differs"):
        _assert_pressure_matches(public["pressure"], no_factor_two["pressure"], f"{driver} pressure")
    no_jackknife_factor = _reference(components, volume, _BINS, jackknife_factor=False)
    with pytest.raises(AssertionError, match="pressure error differs"):
        _assert_pressure_matches(public["pressure"], no_jackknife_factor["pressure"], f"{driver} pressure")
    quadrature = (reference["hydrostatic"][0], float(np.sqrt(np.sum(pressure_std**2)) / 3.0))
    with pytest.raises(AssertionError, match="hydrostatic error differs"):
        _assert_pressure_matches(public["hydrostatic"], quadrature, f"{driver} hydrostatic")
