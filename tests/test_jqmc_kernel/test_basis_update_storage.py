"""Lossless optimizer publication into the raw GTO model storage."""

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

import numpy as np
import pytest

from jqmc.atomic_orbital import AOs_GTO_base_data
from jqmc.determinant import Geminal_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.trexio_wrapper import read_trexio_file
from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _block(name, values):
    # Keep the oracle independent even if a faulty writer mutates its input.
    return make_test_block(name, np.array(values, copy=True), field="real", derivative_policy="direct_real", sync_scope="world")


def _check_raw_storage(filename, parameter, owner):
    trexio_file = Path(__file__).resolve().parents[1] / "trexio_example_files" / filename
    _, aos, _, _, geminal, _ = read_trexio_file(trexio_file=str(trexio_file), store_tuple=True)
    field = "exponents" if parameter == "exp" else "coefficients"
    original = np.asarray(getattr(aos.radial, field))
    expected_up = original * 0.95
    expected_dn = original * 1.05
    if owner == "geminal":
        model = Geminal_data.convert_from_MOs_to_AOs(geminal)
        expected_values = np.concatenate([expected_up, expected_dn])
        updated = model.apply_block_update(_block(f"lambda_basis_{parameter}", expected_values))
        leaves = [updated.orb_data_up_spin, updated.orb_data_dn_spin]
    elif owner == "j3":
        model = Jastrow_data(jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos))
        expected_values = expected_up
        updated = model.apply_block_update(_block(f"j3_basis_{parameter}", expected_values))
        leaves = [updated.jastrow_three_body_data.orb_data]
    else:
        assert owner == "ao"
        expected_values = expected_up
        leaves = [aos.apply_basis_block_update(_block(f"basis_{parameter}", expected_values))]
    arrays = [getattr(leaf.radial, field) for leaf in leaves]
    # Inspect raw leaves: the public AO properties upcast and could hide a
    # quantized fp32 array. Check both dtype and exact published values.
    actual_dtypes = np.asarray([[isinstance(a, np.ndarray), ord(a.dtype.kind), a.dtype.itemsize] for a in arrays])
    expected_dtypes = np.tile(np.asarray([1, ord("f"), np.dtype(np.float64).itemsize]), (len(arrays), 1))
    np.testing.assert_array_equal(actual_dtypes, expected_dtypes)
    actual_values = np.concatenate(arrays)
    np.testing.assert_array_equal(actual_values, expected_values)


@pytest.mark.parametrize("filename", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
@pytest.mark.parametrize("parameter", ["exp", "coeff"])
@pytest.mark.parametrize("owner", ["ao", "geminal", "j3"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Publishes stored basis parameters without evaluating any electron positions."
)
def test_basis_update_preserves_raw_fp64_storage(filename, parameter, owner):
    """Cartesian/spherical H2 exponent and coefficient updates retain exact fp64 model values.

    Load the committed H2 AE cc-pVDZ TREXIO file. Scale the raw radial
    parameters by 0.95 (and 1.05 for the geminal's separate down-spin leaf),
    then publish through the direct AO, geminal, or J3 optimizer route.
    Compare raw storage dtype and every value exactly. No electron positions,
    random draws, condition-number selection, or full/mixed cross-comparison
    is involved: both modes must retain their own supplied fp64 parameters.
    """
    _check_raw_storage(filename, parameter, owner)


@pytest.mark.parametrize("parameter", ["exp", "coeff"])
@pytest.mark.parametrize("owner", ["ao", "geminal", "j3"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Checks stored parameter publication and lossy delegation without electron positions."
)
def test_basis_storage_contract_rejects_lossy_delegation(monkeypatch, parameter, owner):
    """Reintroducing fp32 rounding at any of the three update boundaries fails exact publication."""
    filename = "H2_ae_ccpvdz_cart.h5"
    _check_raw_storage(filename, parameter, owner)
    if owner == "ao":
        cls, name = AOs_GTO_base_data, "apply_basis_block_update"
    elif owner == "geminal":
        cls, name = Geminal_data, "_apply_basis_block_delegated"
    else:
        cls, name = Jastrow_data, "_apply_j3_basis_block_delegated"
    original = getattr(cls, name)

    def lossy(receiver, block, *args, **kwargs):
        rounded = block.replace(values=np.asarray(block.values, dtype=np.float32))
        return original(receiver, rounded, *args, **kwargs)

    monkeypatch.setattr(cls, name, staticmethod(lossy) if owner == "j3" else lossy)
    with pytest.raises(AssertionError):
        _check_raw_storage(filename, parameter, owner)
