"""Executed zone boundaries for molecular ECP and complex periodic wavefunctions."""

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

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import atomic_orbital, coulomb_potential, determinant, hamiltonians, molecular_orbital, wavefunction
from jqmc._precision import get_complex_dtype_np, get_dtype_np
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data, compute_Jastrow_two_body
from jqmc.trexio_wrapper import read_trexio_file


pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.parametrize("filename", ["H2_ecp_ccpvtz.h5", "lithium_k025.h5"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_real_ecp_and_complex_periodic_kernel_zone_dtypes(filename):
    """Execute molecular ECP and non-Gamma periodic ECP kernels and check each output zone.

    Committed H2_ecp_ccpvtz.h5 supplies real molecular orbitals; lithium_k025.h5
    supplies complex periodic orbitals at nonzero k. Use one PCG64 seed-2025
    configuration, scale 0.5 Bohr, the file's electron counts and no cond(G)
    selection. Add a conventional exp J2 with b=0.5 using the actual cell.
    Run AO/MO values and derivatives, geminal and row/column updates, log Det,
    Det gradients/Laplacian, J2, bare/local/nonlocal ECP, kinetic energy,
    log wavefunction and local energy. The supported nearest-neighbor nonlocal
    ECP route uses six fixed Cartesian directions and identity rotation, as
    opposed to the molecular-only all-pairs route. Log magnitudes and projected
    nonlocal energy are real by definition even on the complex fixture.

    Every floating output is evaluated, not just eval_shape-traced. An exact
    four-integer descriptor checks dtype kind, byte width, nonempty and finite
    status against the configured zone. GDD requires all 26 named outputs, not
    merely one successful assertion from the loop. Full and mixed each run against their own policy;
    neither values nor derivatives are compared across precisions. These are
    dtype/safety contracts, not numerical-accuracy or physical-oracle claims.
    """
    trexio_file = Path(__file__).resolve().parents[1] / "trexio_example_files" / filename
    structure, aos, mos_up, _, geminal, coulomb = read_trexio_file(trexio_file=str(trexio_file), store_tuple=True)
    rng = np.random.default_rng(2025)
    r_up = jnp.asarray(rng.standard_normal((geminal.num_electron_up, 3)) * 0.5)
    r_dn = jnp.asarray(rng.standard_normal((geminal.num_electron_dn, 3)) * 0.5)
    j2 = Jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp", structure_data=structure)
    wf = wavefunction.Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data(jastrow_two_body_data=j2))
    ham = hamiltonians.Hamiltonian_data(structure_data=structure, wavefunction_data=wf, coulomb_potential_data=coulomb)
    ham.sanity_check()
    complex_orbitals = filename == "lithium_k025.h5"
    assert bool(structure.pbc_flag) == complex_orbitals
    nonlocal_result = coulomb_potential.compute_ecp_non_local_parts_nearest_neighbors(coulomb, wf, r_up, r_dn, jnp.eye(3), Nv=6)
    outputs = [
        (
            "ao-value",
            "ao_bloch_eval" if complex_orbitals else "ao_eval",
            complex_orbitals,
            atomic_orbital.compute_AOs(aos, r_up),
        ),
        ("ao-laplacian", "ao_grad_lap", complex_orbitals, atomic_orbital.compute_AOs_laplacian(aos, r_up)),
        ("mo-value", "mo_eval", complex_orbitals, molecular_orbital.compute_MOs(mos_up, r_up)),
        ("mo-laplacian", "mo_lap", complex_orbitals, molecular_orbital.compute_MOs_laplacian(mos_up, r_up)),
        ("geminal", "det_eval", complex_orbitals, determinant.compute_geminal_all_elements(geminal, r_up, r_dn)),
        ("row-update", "det_ratio", complex_orbitals, determinant.compute_geminal_up_one_row_elements(geminal, r_up[:1], r_dn)),
        (
            "column-update",
            "det_ratio",
            complex_orbitals,
            determinant.compute_geminal_dn_one_column_elements(geminal, r_up, r_dn[:1]),
        ),
        ("log-det", "det_eval", False, determinant.compute_ln_det_geminal_all_elements(geminal, r_up, r_dn)),
        ("jastrow", "jastrow_eval", False, compute_Jastrow_two_body(j2, r_up, r_dn)),
        ("bare-coulomb", "coulomb", False, coulomb_potential.compute_bare_coulomb_potential(coulomb, r_up, r_dn)),
        ("local-ecp", "coulomb", False, coulomb_potential.compute_ecp_local_parts_all_pairs(coulomb, r_up, r_dn)),
        ("nonlocal-ecp-grid", "coulomb", False, nonlocal_result[2]),
        ("nonlocal-ecp-total", "coulomb", False, nonlocal_result[3]),
        ("kinetic", "wf_kinetic", complex_orbitals, wavefunction.compute_kinetic_energy(wf, r_up, r_dn)),
        ("log-wavefunction", "wf_eval", False, wavefunction.evaluate_ln_wavefunction(wf, r_up, r_dn)),
        ("local-energy", "local_energy", complex_orbitals, hamiltonians.compute_local_energy(ham, r_up, r_dn, jnp.eye(3))),
    ]
    for axis, value in enumerate(atomic_orbital.compute_AOs_grad(aos, r_up)):
        outputs.append((f"ao-gradient-{axis}", "ao_grad_lap", complex_orbitals, value))
    for axis, value in enumerate(molecular_orbital.compute_MOs_grad(mos_up, r_up)):
        outputs.append((f"mo-gradient-{axis}", "mo_grad", complex_orbitals, value))
    for index, value in enumerate(determinant.compute_grads_and_laplacian_ln_Det(geminal, r_up, r_dn)):
        outputs.append((f"det-derivative-{index}", "det_grad_lap", complex_orbitals, value))
    for label, zone, is_complex, value in outputs:
        array = np.asarray(value)
        expected_dtype = np.dtype(get_complex_dtype_np(zone) if is_complex else get_dtype_np(zone))
        actual_descriptor = np.asarray(
            [ord(array.dtype.kind), array.dtype.itemsize, array.size > 0, np.all(np.isfinite(array))], dtype=np.int64
        )
        expected_descriptor = np.asarray([ord(expected_dtype.kind), expected_dtype.itemsize, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(actual_descriptor, expected_descriptor, err_msg=f"{filename}: {label}")


@pytest.mark.parametrize("kernel", ["ao", "geminal"])
@pytest.mark.parametrize("fault", ["component-width", "complex-kind"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_zone_dtype_verdict_rejects_complex_kernel_drift(monkeypatch, kernel, fault):
    """A real non-Gamma kernel result cannot silently change component width or lose complex kind."""
    filename = "lithium_k025.h5"
    test_real_ecp_and_complex_periodic_kernel_zone_dtypes(filename)
    if kernel == "ao":
        module, name, zone = atomic_orbital, "compute_AOs", "ao_bloch_eval"
        label = "ao-value"
    else:
        module, name, zone = determinant, "compute_geminal_all_elements", "det_eval"
        label = "geminal"
    original = getattr(module, name)

    def corrupted(*args, **kwargs):
        value = original(*args, **kwargs)
        if fault == "complex-kind":
            return jnp.real(value)
        wrong_dtype = jnp.complex64 if get_complex_dtype_np(zone) == np.complex128 else jnp.complex128
        return value.astype(wrong_dtype)

    monkeypatch.setattr(module, name, corrupted)
    with pytest.raises(AssertionError, match=label):
        test_real_ecp_and_complex_periodic_kernel_zone_dtypes(filename)
