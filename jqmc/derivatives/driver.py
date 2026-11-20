"""Legacy analytic parameter-gradient closures used by the parity oracle.

The unmasked parameter path feeds the closure output into
``collect_param_grads`` + ``flatten_param_grads`` (then drops inactive blocks by
its parameter flags). The two closures therefore return the raw pytree shapes
expected at that boundary:

  * O_k  : a batched Wavefunction_data-shaped grad pytree consumed by
           ``wavefunction_data.collect_param_grads``.
  * dE_L : a (batched) Hamiltonian_data-shaped carrier whose ``.wavefunction_data``
           is the grad pytree.

Both closures batch over walkers through :mod:`jqmc.derivatives.bridge`. They use
the strict analytic dispatch: an unsupported axis or leaf raises
``NotImplementedError`` rather than silently falling back to autograd.
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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .bridge import batched_grad_e_L, batched_grad_ln_Psi

if TYPE_CHECKING:
    from ..hamiltonians import Hamiltonian_data
    from ..wavefunction import Wavefunction_data


def make_param_grad_closures(
    base_hamiltonian_data: Hamiltonian_data,
    *,
    mask: Mapping[str, bool] | None = None,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return analytic closures for unmasked parameter gradients.

        o_k_closure(wf, r_up_batch, r_dn_batch, geminal_inv) -> wf-grad pytree
        de_L_dc_closure(ham, r_up_batch, r_dn_batch, rotation_matrix_batch) -> ham-carrier

    ``geminal_inv`` is accepted for signature parity and ignored (the analytic O_k
    recomputes G^-1 internally). ``base_hamiltonian_data`` supplies the structure /
    Coulomb context for the O_k closure, which receives only wave-function data.
    """
    batched_log_wavefunction_param_gradient = batched_grad_ln_Psi(wrt="c", mask=mask)
    batched_local_energy_param_gradient = batched_grad_e_L(wrt="c", mask=mask)

    def o_k_closure(
        wavefunction_data: Wavefunction_data,
        r_up_batch: Any,
        r_dn_batch: Any,
        geminal_inv: Any | None = None,
    ) -> Any:
        hamiltonian_with_wavefunction = base_hamiltonian_data.replace(wavefunction_data=wavefunction_data)
        return batched_log_wavefunction_param_gradient(hamiltonian_with_wavefunction, r_up_batch, r_dn_batch)

    def de_L_dc_closure(
        hamiltonian_data: Hamiltonian_data,
        r_up_batch: Any,
        r_dn_batch: Any,
        rotation_matrix_batch: Any,
    ) -> Hamiltonian_data:
        wavefunction_gradient = batched_local_energy_param_gradient(
            hamiltonian_data,
            r_up_batch,
            r_dn_batch,
            rotation_matrix_batch,
        )
        # Preserve the carrier shape expected by parameter-gradient consumers.
        return hamiltonian_data.replace(wavefunction_data=wavefunction_gradient)

    return o_k_closure, de_L_dc_closure
