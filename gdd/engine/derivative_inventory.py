"""Reviewed B/C/P/R derivative inventories (GDD engine, requirement side).

Implements the inventory half of ``derivatives.production_inventory.v1``: the
reviewed branch (B01-B31), validity (C01-C12), parameter-leaf (P01-P15), and
numerical-regime (R01-R10) tables that the guarantee contract's ``source_id``
references resolve to.  ``tests/test_jqmc_kernel/gen_coverage_readme.py``
renders the human-readable ledger from these rows; the completion gates verify
their ID ranges and family sets.  This module must stay importable without
pytest, jqmc, JAX, or any test module.
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

from gdd.engine.derivative_matrix import _PRODUCTION_AO_FAMILY_TOKENS

# Declared production type registries (verified against introspection by the
# structural gate in tests/test_jqmc_kernel/test_gdd_derivatives_contract.py).
PRODUCTION_AO_LEAF_CLASSES = tuple(f"AOs_{token}_data" for token in _PRODUCTION_AO_FAMILY_TOKENS)
PRODUCTION_GEMINAL_CLASSES = ("AGP_data", "AGPn_data", "SD_data")
PRODUCTION_JASTROW_CHANNEL_FIELDS = (
    "jastrow_dtn_data",
    "jastrow_one_body_data",
    "jastrow_three_body_data",
    "jastrow_two_body_data",
)
PRODUCTION_DTN_CHANNEL_FLAGS = ("use_chi", "use_f", "use_u")


PRODUCTION_BRANCH_INVENTORY = (
    ("B01", "ln_Psi,e_L", "derivative target", "`c`; `r`; `R`; `eps`", "four explicit assemble dispatch branches"),
    (
        "B02",
        "ln_Psi,e_L",
        "c-axis parameter flag",
        "geminal; geminal basis exp/coeff; J1; J2; J3; J3 basis exp/coeff; DTN u/chi/f",
        "the eleven `_PARAM_FLAG_KEYS`; concrete output leaves are listed separately",
    ),
    ("B03", "ln_Psi,e_L,e_L_t,e_L_n", "boundary", "molecule; PBC Gamma; complex twist", "`pbc_flag` and `k_point`"),
    (
        "B04",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "AO leaf",
        "; ".join(f"`{name}`" for name in PRODUCTION_AO_LEAF_CLASSES),
        "all source-discovered concrete `AOs_*_data` classes",
    ),
    (
        "B05",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "composite AO topology",
        "ordered child families and real/complex regime",
        "`AOs_composite_data.children`; composite is not one homogeneous leaf",
    ),
    (
        "B06",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "orbital representation",
        "direct AO; `MOs_data`",
        "AGP permits AO or MO representation; SD/AGPn require MO representation",
    ),
    (
        "B07",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "geminal class",
        "; ".join(f"`{name}`" for name in PRODUCTION_GEMINAL_CLASSES),
        "all source-discovered production `_GeminalBase` subclasses",
    ),
    (
        "B08",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "electron-count topology",
        "closed shell; partially polarized/open shell; fully polarized (zero spin channel)",
        "`num_electron_up/down` and unpaired columns",
    ),
    (
        "B09",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "spin pairing",
        "time-reversal tied; unrestricted independent spin channels",
        "`time_reversal` changes conjugate folding and published MO leaves",
    ),
    (
        "B10",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "real/complex parameter leaves",
        "real Gamma/molecule; complex Bloch orbitals/lambda/MOs with real AGPn `lambda_diag`",
        "selective real projection and outermost conjugate folding",
    ),
    (
        "B11",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "Jastrow channel subset",
        "; ".join(f"`{name}`" for name in PRODUCTION_JASTROW_CHANNEL_FIELDS) + "; low-level Pade/AO+DTN composition",
        "four optional fields are additive; production tools choose the Pade/AO family or DTN, but the public "
        "container does not enforce that policy",
    ),
    ("B12", "ln_Psi,e_L,e_L_t,e_L_n", "J1 functional form", "exp; pade", "`jastrow_1b_type`"),
    ("B13", "ln_Psi,e_L,e_L_t,e_L_n", "J2 functional form", "exp; pade", "`jastrow_2b_type`"),
    (
        "B14",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "J3 orbital representation",
        "direct AO; MO-backed; composite child topology; real/complex",
        "J3 `orb_data` is independent of the geminal orbital representation",
    ),
    (
        "B15",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "DTN active channels",
        f"all {2 ** len(PRODUCTION_DTN_CHANNEL_FLAGS)} subsets of "
        + ", ".join(f"`{name}`" for name in PRODUCTION_DTN_CHANNEL_FLAGS)
        + ", including the empty subset",
        "source-discovered `use_*` fields independently gate derivative bodies and leaves",
    ),
    (
        "B16",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "DTN constraint regime",
        "electron-nucleus cusp policy; cutoff smoothness/order/species shapes",
        "ECP-aware electron-nucleus cusp policy, `cutoff_smoothness`, orders and constrained maps",
    ),
    (
        "B17",
        "ln_Psi",
        "coordinate component",
        "r: determinant + all Jastrows; R: determinant + J1 + J3 + DTN; eps: molecular/PBC compositions",
        "different channels are structurally absent on different axes",
    ),
    (
        "B18",
        "ln_Psi,e_L",
        "strain implementation",
        "molecular affine virial; periodic full-cell strain",
        "`pbc_flag` eps branch",
    ),
    (
        "B19",
        "ln_Psi,e_L",
        "basis-parameter capability",
        "single radial leaf; parameterless PW/B-spline; composite child blocks/raw leaves; unsupported future radial leaf",
        "geminal and J3 publication contracts differ; every flag must yield a gradient, n/a, or an intentional error",
    ),
    ("B20", "e_L,e_L_t,e_L_n", "Hamiltonian", "all-electron; ECP", "`ecp_flag`; ln_Psi is potential-independent"),
    (
        "B21",
        "e_L",
        "energy component",
        "kinetic; molecular bare/PBC Ewald Coulomb; ECP local; ECP non-local",
        "axis-specific additive branches in `assemble.compute_grad_e_L`",
    ),
    ("B22", "e_L", "ECP quadrature", "identity; nontrivial orthogonal `RT`", "non-local ECP mesh input"),
    (
        "B23",
        "e_L_t,e_L_n",
        "LRDMC hopping mesh",
        "identity; nontrivial orthogonal `RT`",
        "discretized kinetic shifts for AE/ECP; also ECP quadrature when present",
    ),
    ("B24", "e_L_t,e_L_n", "derivative entry", "positions (`R,r_up,r_dn`); strain (`eps`)", "public estimator wrappers"),
    (
        "B25",
        "e_L_t,e_L_n",
        "non-local move",
        "AE: move-independent value; ECP: tmove/dltmove; unknown values are errors",
        "all four public analytical wrappers validate the vocabulary before JIT",
    ),
    (
        "B26",
        "e_L_t,e_L_n",
        "dltmove correction activity",
        "inactive without ECP/Jastrow; active with ECP and any Jastrow",
        "`_dltmove_delta_active`",
    ),
    (
        "B27",
        "e_L_t,e_L_n",
        "S-prime membership predicates",
        "sign-flip false/true crossed with `ei_used > zv` false/true",
        "both predicates in `_sprime_data` determine membership per electron",
    ),
    (
        "B28",
        "e_L_t,e_L_n",
        "continuum Coulomb measurement derivative",
        "molecular AE; periodic AE; ECP bare/local contributions",
        "measurement derivatives contain no propagation-only `eid-ei` S-prime correction",
    ),
    (
        "B29",
        "e_L_t,e_L_n",
        "continuum measurement geometry",
        "molecular; periodic",
        "continuous kinetic/Coulomb/Ewald/ECP branches; no propagation mesh derivative",
    ),
    (
        "B30",
        "e_L_t,e_L_n",
        "complex measurement algebra",
        "real Gamma/molecule; complex twist",
        "holomorphic wave-function/ECP derivative with real projection only at the observable boundary",
    ),
    (
        "B31",
        "e_L_t,e_L_n",
        "estimator relation",
        "direct `e_L_t`; `e_L_n == e_L_t` when the differentiated n-forward uses `use_fast_update=False`",
        "public wrappers delegate to the same shared derivative; the identity precondition must be pinned",
    ),
)

PRODUCTION_VALIDITY_CONSTRAINTS = (
    (
        "C01",
        "reachability",
        "the valid universe is the union of QE/TREXIO loaders and the public data/build API; missing "
        "committed fixtures are debt, not n/a",
    ),
    (
        "C02",
        "AO/boundary",
        "validity comes from data-class sanity/build contracts; do not infer n/a solely from an AO "
        "class label or current fixture absence",
    ),
    ("C03", "spin sectors", "up/down orbital leaves must agree on the real/complex twist regime"),
    ("C04", "geminal representation", "SD/AGPn require MOs; AGP permits AO or MO representation"),
    (
        "C05",
        "alpha-major convention",
        "time_reversal=True requires `N_up >= N_dn`; unpaired columns equal `N_up-N_dn`",
    ),
    (
        "C06",
        "AGPn leaves",
        "`lambda_diag` is always real; `lambda_unpaired` follows the real/complex regime",
    ),
    (
        "C07",
        "Jastrow construction",
        "QE/TREXIO tools select the Pade/AO family or DTN; `Jastrow_data` permits their additive composition, "
        "so it is low-level reachable, not invalid",
    ),
    ("C08", "J1/J2 vocabulary", "only exp/pade are valid and unknown values must raise"),
    (
        "C09",
        "basis parameters",
        "PW/B-spline are n/a; GTO/pGTO produce flat blocks; composite J3 requires per-child flat plus raw accounting; "
        "unsupported future radial leaves raise intentionally",
    ),
    (
        "C10",
        "ECP move",
        "public GFMC derivative entries accept only tmove/dltmove and reject unknown values explicitly",
    ),
    (
        "C11",
        "dense AGP conditioning",
        "conditioning-based test exclusions are coverage policy, not production invalidity",
    ),
    (
        "C12",
        "k-average",
        "per-twist kernels are covered here; aggregation weights/spin reconstruction are a separate contract",
    ),
)

PARAMETER_LEAF_INVENTORY = (
    (
        "P01",
        "`opt_geminal`",
        "SD",
        "`mo_coefficients_up`; plus `mo_coefficients_dn` when unrestricted",
        "outermost time-reversal folding",
    ),
    (
        "P02",
        "`opt_geminal`",
        "AGPn",
        "`mo_coefficients_up`; unrestricted `mo_coefficients_dn`; `lambda_diag`; open-shell `lambda_unpaired`",
        "all stored leaves must be present and live",
    ),
    (
        "P03",
        "`opt_geminal`",
        "AGP",
        "`lambda_matrix`",
        "AO- and MO-representation AGP are both production scope",
    ),
    ("P04", "`opt_lambda_basis_exp`", "geminal radial basis", "`lambda_basis_exp`", "GTO/pGTO children only"),
    (
        "P05",
        "`opt_lambda_basis_coeff`",
        "geminal radial basis",
        "`lambda_basis_coeff`",
        "GTO/pGTO children only",
    ),
    ("P06", "`opt_J1_param`", "J1", "`j1_param`", "only when J1 is present"),
    ("P07", "`opt_J2_param`", "J2", "`j2_param`", "only when J2 is present"),
    ("P08", "`opt_J3_param`", "J3", "`j3_matrix`", "only when J3 is present"),
    ("P09", "`opt_J3_basis_exp`", "single-leaf J3 radial basis", "`j3_basis_exp`", "GTO/pGTO leaf only"),
    (
        "P10",
        "`opt_J3_basis_coeff`",
        "single-leaf J3 radial basis",
        "`j3_basis_coeff`",
        "GTO/pGTO leaf only",
    ),
    ("P11", "`opt_JDTN_u_param`", "DTN u", "`j3_poly_alpha`", "only when `use_u` is true"),
    ("P12", "`opt_JDTN_chi_param`", "DTN chi", "`j3_poly_beta`", "only when `use_chi` is true"),
    ("P13", "`opt_JDTN_f_param`", "DTN f", "`j3_poly_gamma`", "only when `use_f` is true"),
    (
        "P14",
        "`opt_J3_basis_exp`",
        "composite J3 GTO/pGTO child",
        "raw `children[i].radial.exponents`; variational block `j3_child{i}_basis_exp`",
        "raw equality, flat publication, and optimizer update routing are all required",
    ),
    (
        "P15",
        "`opt_J3_basis_coeff`",
        "composite J3 GTO/pGTO child",
        "raw `children[i].radial.coefficients`; variational block `j3_child{i}_basis_coeff`",
        "raw equality, flat publication, and optimizer update routing are all required",
    ),
)

PRODUCTION_NUMERICAL_REGIMES = (
    ("R01", "ln_Psi,e_L,e_L_t,e_L_n", "ordinary nonsingular geminal", "baseline AD comparison"),
    (
        "R02",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "moderately conditioned geminal with the active-precision running inverse",
        "same-precision fast-route sensitivity without a near-node robustness claim",
    ),
    ("R03", "ln_Psi,e_L", "every expected parameter leaf is nonzero in at least one valid cell", "detect missing writes"),
    ("R04", "e_L,e_L_t,e_L_n", "non-local ECP ratio has nonzero real and imaginary parts", "detect premature real projection"),
    (
        "R05",
        "e_L_t,e_L_n",
        "all four S-prime predicate truth pairs in the propagation helper",
        "pin propagation membership while measurement derivatives remain continuum-only",
    ),
    (
        "R06",
        "e_L_t,e_L_n",
        "fixed-node/clamp inputs on both sides, away from the nondifferentiable boundary",
        "branch selection",
    ),
    (
        "R07",
        "e_L_t,e_L_n",
        "at least two positive `alat` values",
        "prove continuum measurement-derivative invariance to propagation lattice spacing",
    ),
    (
        "R08",
        "e_L,e_L_t,e_L_n",
        "identity plus nontrivial deterministic mesh rotations",
        "exercise `RT` covariance for e_L and AE continuum-measurement derivative invariance; ECP quadrature may depend on `RT`",
    ),
    (
        "R09",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "tied and independent spin channels with nonzero down-channel signal",
        "catch folding loss",
    ),
    (
        "R10",
        "ln_Psi,e_L,e_L_t,e_L_n",
        "each additive component live alone and in composition",
        "catch cancellation-hidden omissions",
    ),
)
