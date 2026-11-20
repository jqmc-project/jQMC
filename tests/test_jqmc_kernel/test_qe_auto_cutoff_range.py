"""Automatic mixed-basis cutoff selection on exact, tiny Bloch references."""

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

import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc.qe_wrapper import QEBands, QECell, QEReference, QEWavefunction
from jqmc.upf_wrapper import PrimitiveShell

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _antisymmetric_bloch_reference(miller, qe_cutoff_ry=100.0):
    """An s Gaussian cannot represent this x/y-antisymmetric band.

    With reciprocal lattice I and k=(1/4,1/4,1/4), the two selected
    wavevectors have equal length. Their opposite coefficients cancel
    against every origin-centred s-Gaussian Fourier column, at every
    exponent. Before their PW rows enter, the exact residual is one;
    afterward it is zero. No fitted residual or projection is mocked.
    """
    return QEReference(
        cell=QECell(a=2.0 * np.pi * np.eye(3)),
        atom_labels=("He",),
        atom_elements=("He",),
        atom_positions=np.zeros((1, 3)),
        ecutwfc=qe_cutoff_ry / 2.0,
        fft_grid=(32, 32, 32),
        bands=QEBands(eigenvalues=np.zeros(1), occupations=np.ones(1), n_electrons=2.0, fermi_energy=0.0),
        wfc=QEWavefunction(
            millers=np.array([[0, 0, 0], [miller, 0, 0], [0, miller, 0]], dtype=np.int64),
            coeffs=np.array([[0.0, 1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0),
            gamma_only=False,
            k_point=np.full(3, 0.25),
        ),
    )


def _convert_reference(monkeypatch, tmp_path, reference):
    """Replace file parsing and the seed only; execute the real converter."""
    from jqmc import pw_pgto_projection, qe_wrapper, upf_wrapper
    from jqmc.jqmc_qe import run_qe2jqmc

    class OneShellSeed:
        def __init__(self, *args, **kwargs):
            pass

        def shells_for_element(self, label):
            return [PrimitiveShell(angular_l=0, exponents=(10.0,), coefficients=(1.0,), rel_error=0.0, overlap_cond=1.0)]

        def z_valence_for_element(self, label):
            return 2.0

        def is_coulomb_ae_for_element(self, label):
            return True

    monkeypatch.setattr(qe_wrapper, "qe_save_is_lsda", lambda path: False)
    monkeypatch.setattr(qe_wrapper, "load_qe_reference", lambda path: reference)
    monkeypatch.setattr(upf_wrapper, "UPFExponentInit", OneShellSeed)
    sweeps = []
    original_sweep = pw_pgto_projection.optimize_mixed_basis

    def observe_sweep(*args, **kwargs):
        sweep = original_sweep(*args, **kwargs)
        sweeps.append(sweep)
        return sweep

    monkeypatch.setattr(pw_pgto_projection, "optimize_mixed_basis", observe_sweep)
    output = tmp_path / "auto.h5"
    run_qe2jqmc(
        qe_params={"qe_save_dir": str(tmp_path), "kind": "pgto+pw", "ecutwfc": "auto", "upf_paths": ["He=unused.upf"]},
        jastrow_params={},
        hamiltonian_h5=str(output),
    )
    assert output.is_file()
    # The ordinary public assembly checks the supported complex-twist SD.
    from jqmc.hamiltonians import Hamiltonian_data

    hamiltonian = Hamiltonian_data.load_from_hdf5(str(output))
    hamiltonian.sanity_check()
    return sweeps


def test_auto_cutoff_represents_a_band_requiring_more_than_30_ry(monkeypatch, tmp_path):
    """Actual projection fails below 39.1875 Ry and succeeds at the new cap."""
    from jqmc._setting import PW_PGTO_AUTO_RESIDUAL_TARGET

    reference = _antisymmetric_bloch_reference(6)
    sweeps = _convert_reference(monkeypatch, tmp_path, reference)
    sweep = sweeps[0]
    assert len(sweep.trials) == 8
    cutoffs = np.array([trial.g_cut**2 for trial in sweep.trials])
    np.testing.assert_allclose(cutoffs[[0, -1]], [7.5, 40.0], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    # Equal-ratio spacing is an algorithm contract, not an expected residual
    # copied from the production projector.
    np.testing.assert_allclose(
        cutoffs[1:] / cutoffs[:-1],
        np.full(7, (40.0 / 7.5) ** (1.0 / 7.0)),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )
    assert cutoffs[-2] < 39.1875 < cutoffs[-1]
    np.testing.assert_allclose(
        [trial.mean_residual for trial in sweep.trials[:-1]],
        np.ones(7),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )
    assert sweep.target_met
    assert sweep.best is sweep.trials[-1]
    assert sweep.best.mean_residual <= PW_PGTO_AUTO_RESIDUAL_TARGET
    assert sweep.best.n_pw == 3


@pytest.mark.parametrize(
    "miller,qe_cutoff_ry,expected_first_ry,expected_trials,expected_last_ry",
    [(2, 100.0, 7.5, 1, 7.5), (2, 24.0, 6.0, 1, 6.0), (4, 20.0, 5.0, 8, 20.0)],
)
def test_auto_cutoff_preserves_the_first_sufficient_trial(
    monkeypatch, tmp_path, miller, qe_cutoff_ry, expected_first_ry, expected_trials, expected_last_ry
):
    """Keep cheap successes and the original lower-QE-cutoff boundary."""
    from jqmc._setting import PW_PGTO_AUTO_RESIDUAL_TARGET

    reference = _antisymmetric_bloch_reference(miller, qe_cutoff_ry)
    sweeps = _convert_reference(monkeypatch, tmp_path, reference)
    sweep = sweeps[0]
    assert len(sweep.trials) == expected_trials
    assert sweep.target_met
    assert sweep.best is sweep.trials[-1]
    assert all(trial.mean_residual > PW_PGTO_AUTO_RESIDUAL_TARGET for trial in sweep.trials[:-1])
    np.testing.assert_allclose(
        sweep.trials[0].g_cut ** 2, expected_first_ry, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE
    )
    assert sweep.best.mean_residual <= PW_PGTO_AUTO_RESIDUAL_TARGET
    assert sweep.best.g_cut <= np.sqrt(qe_cutoff_ry)
    np.testing.assert_allclose(sweep.best.g_cut**2, expected_last_ry, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)


def test_auto_cutoff_does_not_hide_other_optimizer_errors(monkeypatch, tmp_path):
    """Only infeasible tail bounds may reject an automatic exponent attempt."""
    from jqmc import pgto_exponent_optimizer

    def invalid_seed(*args, **kwargs):
        raise ValueError("invalid seed")

    monkeypatch.setattr(pgto_exponent_optimizer, "optimize_pgto_exponents", invalid_seed)
    with pytest.raises(ValueError, match="invalid seed"):
        _convert_reference(monkeypatch, tmp_path, _antisymmetric_bloch_reference(6))
    assert not (tmp_path / "auto.h5").exists()
