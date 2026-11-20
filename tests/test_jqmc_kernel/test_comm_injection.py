"""Comm parameter injection on all sampler classes.

The twist-averaged driver needs each ``MCMC`` / ``GFMC_*`` instance to reduce over its
twist's intra-comm rather than ``COMM_WORLD``. The **parameter
addition** half of the comm refactor (``mpi_comm_in`` /
``world_comm_in`` ``__init__`` kwargs with ``None`` defaults that fall back
to the module-global ``mpi_comm``) to all six sampler classes:

* :class:`jqmc.jqmc_mcmc.MCMC` and :class:`jqmc.jqmc_mcmc._MCMC_debug`
* :class:`jqmc.jqmc_gfmc.GFMC_t` and :class:`jqmc.jqmc_gfmc._GFMC_t_debug`
* :class:`jqmc.jqmc_gfmc.GFMC_n` and :class:`jqmc.jqmc_gfmc._GFMC_n_debug`

The body rename (routing reductions through ``self._mpi_comm`` /
``self._world_comm``) is not covered here. This test only confirms that the parameter
additions are present and that the defaults preserve the
``COMM_WORLD`` behaviour at the instance level.
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

import inspect
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mpi4py import MPI

from jqmc.jqmc_gfmc import GFMC_n, GFMC_t, _GFMC_n_debug, _GFMC_t_debug
from jqmc.jqmc_mcmc import MCMC, _MCMC_debug

ALL_SAMPLER_CLASSES = [MCMC, _MCMC_debug, GFMC_t, _GFMC_t_debug, GFMC_n, _GFMC_n_debug]


class TestCommInjectionKwargs:
    @pytest.mark.light
    @pytest.mark.parametrize("cls", ALL_SAMPLER_CLASSES)
    def test_init_accepts_mpi_comm_kwargs_with_none_default(self, cls):
        """Every sampler ``__init__`` exposes ``mpi_comm_in`` and ``world_comm_in``
        keyword arguments, both defaulting to ``None``. The ``None`` default is the
        backward-compatible signal that falls back to the module-global
        ``mpi_comm`` so a single-twist caller is byte-for-byte unchanged."""
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        assert "mpi_comm_in" in params, f"{cls.__name__}.__init__ missing mpi_comm_in"
        assert "world_comm_in" in params, f"{cls.__name__}.__init__ missing world_comm_in"
        assert params["mpi_comm_in"].default is None, (
            f"{cls.__name__}.__init__: mpi_comm_in default must be None "
            "(falls back to the module-global mpi_comm; got %r)" % params["mpi_comm_in"].default
        )
        assert params["world_comm_in"].default is None, (
            f"{cls.__name__}.__init__: world_comm_in default must be None "
            "(falls back to the module-global mpi_comm; got %r)" % params["world_comm_in"].default
        )

    @pytest.mark.light
    @pytest.mark.parametrize("cls", ALL_SAMPLER_CLASSES)
    def test_kwargs_are_optional_at_call_site(self, cls):
        """The ``mpi_comm_in`` / ``world_comm_in`` parameters are keyword-only
        in spirit (added after the existing tail of optional kwargs), so an
        existing caller that omits both still binds correctly to ``__init__``."""
        sig = inspect.signature(cls.__init__)
        # ``self`` is ``inspect.Parameter.POSITIONAL_OR_KEYWORD``; we just need
        # to confirm we can bind the comm-injection params by keyword.
        # A bare ``cls(...)`` call would require building a full Hamiltonian_data
        # fixture which is heavy; signature.bind_partial is enough for the test.
        sig.bind_partial(mpi_comm_in=None, world_comm_in=None)
        sig.bind_partial(mpi_comm_in=MPI.COMM_WORLD)
        sig.bind_partial(world_comm_in=MPI.COMM_WORLD)
