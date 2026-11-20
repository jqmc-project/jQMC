"""Unit tests for jqmc.upf_wrapper (UPF parsing + primitive-GTO fit)."""

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

pytestmark = [pytest.mark.with_chdir]
from jqmc._setting import (
    ATOL_TEXT_FORMAT_PARSE,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_UPF_ANALYTIC_GAUSSIAN_FIT_RECOVERY,
    RTOL_UPF_OVERLAP_QUADRATURE_REFERENCE,
)
from jqmc.upf_wrapper import (
    ECPData,
    EvenTemperedExponentInit,
    PrimitiveShell,
    PSWFChannel,
    UPFData,
    UPFExponentInit,
    fit_pswfc_channel,
    fit_pswfc_channel_auto,
    generate_ae_even_tempered_shells,
    parse_gaussinfo,
    parse_upf,
)
from tests._test_data_cache import qe_example_files_root

_LI_UPF = qe_example_files_root() / "Li.ccECP.upf"

_li_available = _LI_UPF.is_file()
requires_li = pytest.mark.skipif(not _li_available, reason="Li.ccECP.upf not present")


class TestUPFWrapper:
    @pytest.mark.light
    @requires_li
    def test_parse_upf_v1_text_li(self):
        """Li UPF v1 preserves the mesh and both radial channels at six independently pinned file samples."""
        upf = parse_upf(str(_LI_UPF))
        assert upf.element == "Li"
        assert upf.z_valence == pytest.approx(1.0)
        # Li ccECP keeps only s and p channels (no d for a Z=3 atom).
        assert upf.angular_ls == (0, 1)
        assert len(upf.channels) == 2
        # Mesh and chi share length and are finite.
        for ch in upf.channels:
            assert ch.chi.shape == upf.r_mesh.shape
            assert np.all(np.isfinite(ch.chi))
        # Literal decimal tokens from committed Li.ccECP.upf, not production
        # parser output. PP_R starts at line 80, S at 1206, P at 1483; each
        # data line has four samples. Check near-origin, interior and tail.
        indices = [0, 1, 512, 768, 1000, 1103]
        assert upf.r_mesh.shape == (1104,)
        np.testing.assert_array_equal(
            upf.r_mesh[indices],
            [
                6.9336127435063426e-05,
                7.0243381465794050e-05,
                5.3904330115524264e-02,
                1.5029886088516713e00,
                3.0675231327309181e01,
                1.1703295148869513e02,
            ],
        )
        np.testing.assert_array_equal(
            upf.channels[0].chi[indices],
            [
                4.0493410439536535e-06,
                4.1023261354179152e-06,
                3.1681706782937602e-03,
                2.7836069313063028e-01,
                3.9738419292574463e-06,
                0.0,
            ],
        )
        np.testing.assert_array_equal(
            upf.channels[1].chi[indices],
            [
                9.2266738547924540e-10,
                9.4697131951453512e-10,
                5.5754769530912845e-04,
                2.1858495371244344e-01,
                2.9330790245544748e-04,
                0.0,
            ],
        )

    # Cu-only d-channel fit test deleted in the Li migration -- Li.ccECP
    # has no d channel (max_ang_mom_plus_1 = 2 -> only s + p), so the
    # specific d-channel-fit accuracy guarantee is fixture-irrelevant.
    # The general fit-quality contract is exercised by
    # test_fit_auto_selects_reasonable_n below on the s channel.

    @pytest.mark.light
    @requires_li
    def test_fit_auto_selects_reasonable_n(self):
        """Fit auto selects reasonable n."""
        upf = parse_upf(str(_LI_UPF))
        s_ch = next(c for c in upf.channels if c.angular_l == 0)
        shell = fit_pswfc_channel_auto(upf.r_mesh, s_ch.chi, angular_l=0, rel_target=2.0e-2, cond_max=1.0e6, n_min=2, n_max=6)
        assert 2 <= len(shell.exponents) <= 6
        assert np.isfinite(shell.overlap_cond)

    @pytest.mark.light
    def test_unsuccessful_least_squares_fit_is_never_used(self, monkeypatch):
        """least_squares without success (here: the max_nfev limit, status 0) raises; its exponents are not used."""
        from scipy.optimize import OptimizeResult

        from jqmc import upf_wrapper

        def _unsuccessful(fun, x0, **kwargs):
            return OptimizeResult(x=np.asarray(x0), success=False, status=0, nfev=4000, message="max_nfev exceeded")

        monkeypatch.setattr(upf_wrapper, "least_squares", _unsuccessful)
        r = np.exp(np.linspace(np.log(1.0e-4), np.log(40.0), 300))
        with pytest.raises(upf_wrapper.PSWFCFitError, match="status=0, nfev=4000: max_nfev exceeded"):
            fit_pswfc_channel(r, r * np.exp(-0.5 * r**2), 0, 2)

    @pytest.mark.light
    @pytest.mark.parametrize("angular_l", [0, 1, 2])
    def test_fit_pswfc_channel_recovers_analytic_gaussian_sum(self, angular_l):
        """The PSWFC fit recovers an exact two-Gaussian radial function for s, p and d.

        Reference: R_l(r) = r^l (1.0 exp(-0.5 r^2) + 2.0 exp(-3.0 r^2)), given to
        the fit as chi_l = r R_l on a logarithmic UPF-like mesh. It lies exactly
        in the documented model space sum_p c_p r^l exp(-alpha_p r^2), so the
        auto fit must stop at two primitives and return exponents (0.5, 3.0)
        and coefficients (1.0, 2.0) with a vanishing residual. The r^l
        prefactor is what makes this exact for l > 0.
        """
        r = np.exp(np.linspace(np.log(1.0e-4), np.log(40.0), 1500))
        chi = r * r**angular_l * (1.0 * np.exp(-0.5 * r**2) + 2.0 * np.exp(-3.0 * r**2))
        shell = fit_pswfc_channel_auto(r, chi, angular_l=angular_l)
        assert shell.angular_l == angular_l
        assert shell.exponents == pytest.approx((0.5, 3.0), rel=RTOL_UPF_ANALYTIC_GAUSSIAN_FIT_RECOVERY)
        assert shell.coefficients == pytest.approx((1.0, 2.0), rel=RTOL_UPF_ANALYTIC_GAUSSIAN_FIT_RECOVERY)
        assert shell.rel_error <= RTOL_UPF_ANALYTIC_GAUSSIAN_FIT_RECOVERY

    @pytest.mark.light
    @requires_li
    def test_upf_exponent_init_strategy_caches(self):
        # g_target is required in production (typically derived from
        # ref.ecutwfc). Tests without a QE reference pass a fixed Bohr^-1
        # value; 25 is far past any PSWFC alpha_max so extension fires.
        """Upf exponent init strategy caches."""
        strat = UPFExponentInit({"li": str(_LI_UPF)}, n_max=5, g_target=25.0)
        shells = strat.shells_for_element("Li")
        # Li ccECP has only s and p shells.
        assert [s.angular_l for s in shells] == [0, 1]
        # Case-insensitive element lookup + caching returns the same object.
        assert strat.shells_for_element("LI") is shells

    @pytest.mark.light
    def test_upf_exponent_init_missing_element_raises(self):
        """Upf exponent init missing element raises."""
        strat = UPFExponentInit({}, g_target=25.0)
        with pytest.raises(KeyError):
            strat.shells_for_element("Fe")

    @pytest.mark.light
    def test_ae_hardcoded_h_basis_is_ccpvdz_5s1p(self):
        """H AE bare-Coulomb basis is the cc-pVDZ 3s + 1p set extended with two
        tight even-tempered s exponents (75, 39).

        The single real consumer of the bare-Coulomb AE path is
        ``H.coulomb-ae.UPF``. Earlier versions tried to invent a Z-scaled
        tempered ladder that systematically lost most of its primitives to
        ``optimize_mixed_basis`` pre-filtering; the hardcoded approach side-
        steps that entirely by sitting in the natural chemistry-grade range
        where the auto-LS keeps the basis intact. The two tight exponents
        (75, 39; even-tempered ratio ~2 up from cc-pVDZ's 13.01) were added
        for the bare-Coulomb cusp, which needs alpha well above 13.01 to
        resolve the nuclear singularity (see ``_AE_H_S_EXPONENTS`` in
        ``upf_wrapper.py``).
        """
        shells = generate_ae_even_tempered_shells(g_target=14.14, upf_lmax=0, atomic_number=1)
        assert [s.angular_l for s in shells] == [0, 1]
        # 5s: cc-pVDZ uncontracted exponents for H, plus two tight tempered
        # exponents (75, 39) for the bare-Coulomb cusp.
        assert shells[0].exponents == (75.0, 39.0, 13.01, 1.962, 0.4446)
        # 1p: cc-pVDZ polarisation exponent.
        assert shells[1].exponents == (0.7270,)

    @pytest.mark.light
    def test_ae_hardcoded_basis_ignores_g_target_and_upf_lmax(self):
        """``g_target`` and ``upf_lmax`` are accepted for API compatibility
        but do not drive the exponent choice."""
        # Wildly different g_target / upf_lmax still returns the same H basis.
        shells_a = generate_ae_even_tempered_shells(g_target=2.0, upf_lmax=0, atomic_number=1)
        shells_b = generate_ae_even_tempered_shells(g_target=50.0, upf_lmax=3, atomic_number=1)
        assert shells_a[0].exponents == shells_b[0].exponents
        assert shells_a[1].exponents == shells_b[1].exponents

    @pytest.mark.light
    def test_ae_hardcoded_basis_rejects_non_hydrogen(self):
        """Any atomic_number != 1 raises NotImplementedError.

        All other AE / ECP UPFs ship PSWFC channels and go through
        ``fit_pswfc_channel_auto``; this branch is reserved for hydrogen.
        """
        for Z in (2, 3, 6, 14):
            with pytest.raises(NotImplementedError, match=r"only H \(Z=1\)"):
                generate_ae_even_tempered_shells(g_target=14.14, upf_lmax=0, atomic_number=Z)

    @pytest.mark.light
    def test_ae_hardcoded_basis_cell_warning_does_not_trim(self, caplog):
        """A pathologically small cell triggers a WARNING but the hardcoded
        exponents are NOT trimmed -- the basis is the basis."""
        import logging

        # Very small cell -> alpha_floor = 4 / 1^2 = 4. The smallest hardcoded
        # exponent 0.4446 is below this, so a warning fires. The shells are
        # returned unchanged.
        with caplog.at_level(logging.WARNING):
            shells = generate_ae_even_tempered_shells(g_target=14.14, upf_lmax=0, atomic_number=1, cell_min_edge=1.0)
        assert any(record.levelno >= logging.WARNING for record in caplog.records)
        assert shells[0].exponents == (75.0, 39.0, 13.01, 1.962, 0.4446)
        assert shells[1].exponents == (0.7270,)

    @pytest.mark.light
    def test_ae_hardcoded_basis_no_warning_at_realistic_cell(self, caplog):
        """For any realistic cell (cell_min_edge >= ~3 Bohr) the cell-aware
        safety floor sits below the hardcoded smallest exponent, so the
        warning never fires."""
        import logging

        with caplog.at_level(logging.WARNING):
            shells = generate_ae_even_tempered_shells(g_target=14.14, upf_lmax=0, atomic_number=1, cell_min_edge=10.0)
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)
        assert shells[0].exponents == (75.0, 39.0, 13.01, 1.962, 0.4446)

    @pytest.mark.light
    def test_ae_hardcoded_basis_validates_inputs(self):
        """Negative ``cell_min_edge`` / non-positive ``cell_safety`` are
        still rejected."""
        with pytest.raises(ValueError, match="cell_min_edge"):
            generate_ae_even_tempered_shells(g_target=14.14, upf_lmax=0, atomic_number=1, cell_min_edge=-1.0)
        with pytest.raises(ValueError, match="cell_safety"):
            generate_ae_even_tempered_shells(
                g_target=14.14,
                upf_lmax=0,
                atomic_number=1,
                cell_min_edge=5.0,
                cell_safety=0.0,
            )

    @pytest.mark.light
    def test_even_tempered_strategy_shape(self):
        """Even tempered strategy shape."""
        strat = EvenTemperedExponentInit(alpha_min=0.2, alpha_max=10.0, n=4, l_max=2)
        shells = strat.shells_for_element("Si")
        assert [s.angular_l for s in shells] == [0, 1, 2]
        for s in shells:
            assert len(s.exponents) == 4
            assert list(s.exponents) == sorted(s.exponents)
            assert s.exponents[0] == pytest.approx(0.2)
            assert s.exponents[-1] == pytest.approx(10.0)

    @pytest.mark.light
    def test_even_tempered_n1(self):
        """Even tempered n1."""
        strat = EvenTemperedExponentInit(alpha_min=0.2, alpha_max=10.0, n=1, l_max=0)
        shells = strat.shells_for_element("H")
        assert shells[0].exponents == (10.0,)

    @pytest.mark.light
    @requires_li
    def test_parse_gaussinfo_li_ccecp(self):
        """[Gaussinfo] in the Li ccECP UPF round-trips to ECPData.

        Li.ccECP has z_core = 2 (Z_atomic 3 - Z_val 1), l_max = 0 (a single
        non-local s channel, so the local tag is l_max + 1 = 1), and 4 terms
        total (1 s + 3 local). ``l_max`` counts the NON-LOCAL channels only --
        the local one is what they are projected against, not an l value of
        its own (getting this backwards, i.e. counting the local channel too,
        is an off-by-one: it made every ccECP UPF here claim
        max_ang_mom_plus_1 = 2 where PySCF's own dump of the same elements
        says 1). The local channel's first term is the
        charge-recovery (coef = Z_val = 1, n_pow = 1); this invariant is what
        makes the format unambiguous.
        """
        ecp = parse_gaussinfo(str(_LI_UPF))
        assert isinstance(ecp, ECPData)
        assert ecp.element == "Li"
        assert ecp.z_core == 2
        assert ecp.max_ang_mom_plus_1 == 1
        assert ecp.num_ecps == 4
        # Local-first ordering: ang_moms[0..2] = local (tag = l_max+1 = 1),
        # then 1 s term (tag 0).
        assert tuple(ecp.ang_moms) == (1, 1, 1, 0)
        # Charge-recovery local term (Opium ccECP convention).
        assert ecp.powers[0] == 1
        assert ecp.coefficients[0] == pytest.approx(1.0)
        assert ecp.exponents[0] == pytest.approx(15.0, abs=ATOL_TEXT_FORMAT_PARSE)

    @pytest.mark.light
    def test_parse_gaussinfo_missing_section_raises(self, tmp_path):
        """A UPF without [Gaussinfo] is rejected with an Opium-rpt hint."""
        minimal = (
            "  <PP_INFO>\n"
            "[Atom]\n"
            "Si\n"
            "  </PP_INFO>\n"
            "<PP_HEADER>\n"
            "   Si        Element\n"
            "   4.0       Z valence\n"
            "</PP_HEADER>\n"
            "<PP_R>0.0 1.0</PP_R>\n"
            "<PP_PSWFC>\n"
            "S    0   2.00   Wavefunction\n"
            "0.0 1.0\n"
            "</PP_PSWFC>\n"
        )
        p = tmp_path / "Si_no_gaussinfo.upf"
        p.write_text(minimal)
        with pytest.raises(ValueError, match=r"\[Gaussinfo\].*Opium.*rpt"):
            parse_gaussinfo(str(p))

    @pytest.mark.light
    def test_parse_gaussinfo_two_nonlocal_channels_keep_their_l(self, tmp_path):
        """Each [Gaussinfo] non-local channel keeps its own l when there are two of them.

        The committed ccECP fixtures (Li, O) carry a single non-local channel,
        so any permutation of the non-local tags is invisible there. This
        hand-written block has an s channel (one term), a p channel (two
        terms) and the local channel (two terms), each with distinct
        exponents, so the expected TREXIO layout is written down by hand:
        local terms first with tag l_max + 1 = 2, then the s term with tag 0,
        then the two p terms with tag 1.
        """
        text = (
            "  <PP_INFO>\n"
            "[Gaussinfo]\n"
            "8   3\n"
            "1   2   2\n"
            "2   11.0   21.0\n"
            "2   12.0   22.0\n"
            "2   13.0   23.0\n"
            "1   14.0    8.0\n"
            "3   15.0   25.0\n"
            "\n"
            "[Configs]\n"
            "  </PP_INFO>\n"
            "<PP_HEADER>\n"
            "   Ne        Element\n"
            "   8.0       Z valence\n"
            "</PP_HEADER>\n"
            "<PP_R>0.1 1.0</PP_R>\n"
            "<PP_PSWFC>\n"
            "S    0   2.00   Wavefunction\n"
            "0.1 0.2\n"
            "</PP_PSWFC>\n"
        )
        p = tmp_path / "Ne_two_nonlocal.upf"
        p.write_text(text)
        ecp = parse_gaussinfo(str(p))
        assert ecp.element == "Ne"
        assert ecp.z_core == 2
        assert ecp.max_ang_mom_plus_1 == 2
        assert ecp.ang_moms == (2, 2, 0, 1, 1)
        assert ecp.exponents == (14.0, 15.0, 11.0, 12.0, 13.0)
        assert ecp.coefficients == (8.0, 25.0, 21.0, 22.0, 23.0)
        assert ecp.powers == (1, 3, 2, 2, 2)

    @pytest.mark.light
    def test_parse_upf_v2_xml_synthetic(self, tmp_path):
        # Minimal synthetic v2 UPF exercising the XML branch.
        """UPF v2 preserves every serialized radial mesh and wavefunction value in its original order."""
        r = np.linspace(0.0, 4.0, 50)
        chi = (r * np.exp(-0.8 * r**2)).astype(np.float64)
        r_txt = " ".join(f"{x:.10e}" for x in r)
        chi_txt = " ".join(f"{x:.10e}" for x in chi)
        xml = f"""<UPF version="2.0.1">
  <PP_HEADER element="Si" z_valence="4.0"/>
  <PP_MESH>
    <PP_R type="real" size="{r.size}">{r_txt}</PP_R>
  </PP_MESH>
  <PP_PSWFC>
    <PP_CHI.1 l="0" label="3S" occupation="2.0" size="{r.size}">{chi_txt}</PP_CHI.1>
  </PP_PSWFC>
</UPF>
"""
        p = tmp_path / "Si.upf"
        p.write_text(xml)
        upf = parse_upf(str(p))
        assert upf.element == "Si"
        assert upf.z_valence == pytest.approx(4.0)
        assert upf.angular_ls == (0,)
        assert upf.channels[0].chi.shape == upf.r_mesh.shape
        # Compare with the decimal values actually written, not the original
        # unrounded arrays: formatting to .10e is intentionally lossy.
        np.testing.assert_array_equal(upf.r_mesh, [float(token) for token in r_txt.split()])
        np.testing.assert_array_equal(upf.channels[0].chi, [float(token) for token in chi_txt.split()])

    @pytest.mark.light
    def test_parse_upf_v2_bare_coulomb_ae_without_pswfc(self, tmp_path):
        """A UPF v2 header with is_coulomb="true" parses as all-electron bare Coulomb with no channels.

        This is the layout of QE's H.coulomb-ae.UPF: the header declares
        is_coulomb="true" and PP_PSWFC is empty (number_of_wfc="0"), so there
        is nothing to fit and UPFExponentInit must take the hard-coded AE
        branch. A pseudopotential header (no is_coulomb attribute, or
        is_coulomb="false") in the same layout is a hard error.
        """
        r_txt = "1.0e-3 1.0e-2 1.0e-1 1.0e0 1.0e1"
        layout = """<UPF version="2.0.1">
  <PP_HEADER element="H " pseudo_type="1/r" {flag} z_valence="1.000000000000e0" l_max="0" number_of_wfc="0"/>
  <PP_MESH>
    <PP_R type="real" size="5">{r_txt}</PP_R>
  </PP_MESH>
  <PP_PSWFC>
  </PP_PSWFC>
</UPF>
"""
        p = tmp_path / "H_bare_coulomb_ae.UPF"
        p.write_text(layout.format(flag='is_coulomb="true"', r_txt=r_txt))
        upf = parse_upf(str(p))
        assert upf.is_coulomb_ae is True
        assert upf.element == "H"
        assert upf.z_valence == 1.0
        assert upf.channels == ()
        assert upf.l_max_upf == 0
        np.testing.assert_array_equal(upf.r_mesh, [1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1])
        assert UPFExponentInit({"H": str(p)}, g_target=25.0).is_coulomb_ae_for_element("H") is True

        for flag in ("", 'is_coulomb="false"'):
            q = tmp_path / f"H_pseudo_{len(flag)}.UPF"
            q.write_text(layout.format(flag=flag, r_txt=r_txt))
            with pytest.raises(ValueError, match="no PP_CHI channels"):
                parse_upf(str(q))

    # ---------------------------------------------------------------------------
    # Tight-primitive extension (always-on in UPFExponentInit)
    # ---------------------------------------------------------------------------
    @pytest.mark.light
    def test_gaussian_radial_overlap_cond_matches_fit_pswfc_convention(self):
        """Helper must reproduce the cond number formula used by fit_pswfc_channel."""
        from jqmc.upf_wrapper import _gaussian_radial_overlap_cond

        alphas = (0.5, 1.2, 4.0, 11.0)
        l = 1
        # Replicate the closed-form S_pq propto (alpha_p + alpha_q)^{-(l + 3/2)}
        # used inside fit_pswfc_channel (constants cancel in cond).
        a_sum = np.add.outer(np.asarray(alphas), np.asarray(alphas))
        overlap = a_sum ** (-(l + 1.5))
        expected = float(np.linalg.cond(overlap))
        got = _gaussian_radial_overlap_cond(alphas, l)
        assert got == pytest.approx(expected, rel=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_gaussian_radial_overlap_cond_matches_quadrature(self):
        """The overlap condition number is cond of the radial overlap integrals, computed here by quadrature.

        The reference matrix is S_pq = int_0^inf r^(2l+2) exp(-(alpha_p + alpha_q) r^2) dr,
        i.e. <r^l e^{-alpha_p r^2} | r^l e^{-alpha_q r^2}> in the radial measure
        r^2 dr, integrated numerically with scipy.integrate.quad (no closed form
        is used). Both the extension helper and the ``overlap_cond`` that
        fit_pswfc_channel_auto reports for a fitted shell must equal cond(S),
        for s, p and d.
        """
        from scipy.integrate import quad

        from jqmc.upf_wrapper import _gaussian_radial_overlap_cond

        def quadrature_cond(alphas, l):
            s = np.empty((len(alphas), len(alphas)))
            for p, a_p in enumerate(alphas):
                for q, a_q in enumerate(alphas):
                    s[p, q], _ = quad(
                        lambda x: x ** (2 * l + 2) * np.exp(-(a_p + a_q) * x * x),
                        0.0,
                        np.inf,
                        epsabs=0.0,
                        epsrel=RTOL_PYTHON_FP64_LOOSE,
                    )
            return float(np.linalg.cond(s))

        alphas = (0.5, 1.2, 4.0, 11.0)
        r = np.exp(np.linspace(np.log(1.0e-4), np.log(40.0), 1500))
        for l in (0, 1, 2):
            got = _gaussian_radial_overlap_cond(alphas, l)
            assert got == pytest.approx(quadrature_cond(alphas, l), rel=RTOL_UPF_OVERLAP_QUADRATURE_REFERENCE)
            chi = r * r**l * (np.exp(-0.5 * r**2) + 2.0 * np.exp(-3.0 * r**2))
            shell = fit_pswfc_channel_auto(r, chi, angular_l=l)
            assert shell.overlap_cond == pytest.approx(
                quadrature_cond(shell.exponents, l), rel=RTOL_UPF_OVERLAP_QUADRATURE_REFERENCE
            )

    @pytest.mark.light
    def test_extend_shells_tight_geometric_progression(self):
        """Even-tempered extension: appends alpha_max*ratio^k until alpha_target / max cap."""
        from jqmc.upf_wrapper import PrimitiveShell, _extend_shells_tight

        s = PrimitiveShell(
            angular_l=0,
            exponents=(0.5, 1.5, 4.5),
            coefficients=(1.0, 1.0, 1.0),
            rel_error=1e-3,
            overlap_cond=10.0,
        )
        # G_target=25, tail_tol=1e-3 -> alpha_target = 25^2 / (4 * ~6.9) ~ 22.6
        out = _extend_shells_tight([s], G_target=25.0, tail_tol=1e-3, ratio=2.5, max_per_shell=3)
        assert len(out) == 1
        new = out[0]
        # 4.5 * 2.5 = 11.25  (kept; <22.6)
        # 11.25 * 2.5 = 28.125 (rejected; >22.6)
        assert new.exponents == pytest.approx((0.5, 1.5, 4.5, 11.25), rel=RTOL_PYTHON_FP64_LOOSE)
        assert new.coefficients == (1.0, 1.0, 1.0, 1.0)
        assert new.angular_l == 0
        assert new.rel_error == 1e-3  # preserved
        # overlap_cond recomputed on the extended set; differs from the old 10.0.
        assert new.overlap_cond != 10.0

    @pytest.mark.light
    def test_extend_shells_tight_max_per_shell_cap(self):
        """Extension stops at max_per_shell even if alpha_target is larger."""
        from jqmc.upf_wrapper import PrimitiveShell, _extend_shells_tight

        s = PrimitiveShell(
            angular_l=0,
            exponents=(4.0,),
            coefficients=(1.0,),
            rel_error=0.0,
            overlap_cond=1.0,
        )
        # G_target chosen huge so alpha_target ~ 1e4; max_per_shell=2 binds first.
        out = _extend_shells_tight(
            [s],
            G_target=500.0,
            tail_tol=1e-3,
            ratio=2.5,
            max_per_shell=2,
        )
        # 4 * 2.5^k sequence (k=1, 2): 10, 25 -> both kept (alpha_target ~ 9050 >> 25).
        assert out[0].exponents == pytest.approx((4.0, 10.0, 25.0), rel=RTOL_PYTHON_FP64_LOOSE)

    @pytest.mark.light
    def test_upf_exponent_init_extends_basis_by_default(self):
        """UPFExponentInit always applies the tight-extension logic.

        Compares against the un-extended PSWFC fit (re-invokes
        fit_pswfc_channel_auto). For each shell, either:
          (a) the extension appended at least one primitive, OR
          (b) the un-extended ``alpha_max * ratio`` already exceeds
              ``alpha_target`` -- i.e. the shell was already saturated at
              the test's ``g_target`` / RATIO setting, so the helper
              correctly added nothing.
        Either branch is a pass; both confirm the extension logic is wired
        correctly. The Li fixture happens to fall in branch (b) at ratio=2.5,
        while methane's H polarisation p-shell falls in (a).
        """
        from pathlib import Path
        from jqmc.upf_wrapper import (
            UPFExponentInit,
            fit_pswfc_channel_auto,
            parse_upf,
        )
        from jqmc._setting import (
            PGTO_TIGHT_RATIO,
            PW_PGTO_TAIL_TOL,
        )

        upf_path = qe_example_files_root() / "Li.ccECP.upf"
        if not upf_path.is_file():
            pytest.skip(f"Li UPF fixture not present at {upf_path}")
        g_target_test = 25.0  # explicit; production code derives this from ref.ecutwfc
        init = UPFExponentInit({"Li": str(upf_path)}, g_target=g_target_test)
        extended = init.shells_for_element("Li")

        upf = parse_upf(str(upf_path))
        raw = [fit_pswfc_channel_auto(upf.r_mesh, ch.chi, ch.angular_l) for ch in upf.channels]

        alpha_target = g_target_test**2 / (4.0 * abs(np.log(PW_PGTO_TAIL_TOL)))

        assert len(extended) == len(raw)
        for ext, r in zip(extended, raw):
            assert max(ext.exponents) >= max(r.exponents)
            assert ext.overlap_cond > 0.0 and np.isfinite(ext.overlap_cond)
            if len(ext.exponents) == len(r.exponents):
                # No extension applied -- must be because the next step would
                # exceed alpha_target (else the helper would have added it).
                next_alpha = max(r.exponents) * PGTO_TIGHT_RATIO
                assert next_alpha > alpha_target, (
                    f"shell l={r.angular_l} alpha_max={max(r.exponents):.3f} "
                    f"NOT extended, but next_alpha={next_alpha:.3f} <= "
                    f"alpha_target={alpha_target:.3f} -- extension logic broken"
                )

    @pytest.mark.light
    def test_upf_exponent_init_li_fit_quality_and_default_extension(self):
        """UPFExponentInit fits every Li PSWFC channel and then always appends the tight ladder.

        Independent references only. (1) Fit quality: the committed
        Li.ccECP.upf radial functions R_l = chi_l / r (tracked in git, so a
        missing fixture is a hard failure, not a skip). The fitted primitives,
        evaluated as the documented model R_l(r) = sum_p c_p r^l exp(-alpha_p r^2),
        must reproduce them within ``rel_target`` in the radial L2 norm, with
        the norm computed here by trapezoid quadrature over r^2 dr. (2) Default
        extension: two Fourier-reach limits whose outcome does not depend on
        the alpha_target formula. A reach of 0.1 Bohr^-1 lies far below every
        fitted exponent, so nothing is appended; a reach of 1000 Bohr^-1 lies
        far above the fitted range, so exactly PGTO_TIGHT_MAX_PER_SHELL
        primitives alpha_max * PGTO_TIGHT_RATIO^k (k = 1, 2, ...) with unit
        coefficients are appended to the unchanged fitted shell.
        """
        from jqmc._setting import PGTO_TIGHT_MAX_PER_SHELL, PGTO_TIGHT_RATIO

        rel_target = 1.0e-2
        upf = parse_upf(str(_LI_UPF))
        fitted = UPFExponentInit({"Li": str(_LI_UPF)}, g_target=0.1, rel_target=rel_target).shells_for_element("Li")
        extended = UPFExponentInit({"Li": str(_LI_UPF)}, g_target=1000.0, rel_target=rel_target).shells_for_element("Li")

        r = upf.r_mesh
        assert [s.angular_l for s in fitted] == [ch.angular_l for ch in upf.channels] == [0, 1]
        assert [s.angular_l for s in extended] == [0, 1]
        for ch, fit, ext in zip(upf.channels, fitted, extended):
            l = ch.angular_l
            reference = ch.chi / r
            model = sum(c * r**l * np.exp(-a * r**2) for a, c in zip(fit.exponents, fit.coefficients))
            rel_l2 = np.sqrt(np.trapezoid((model - reference) ** 2 * r**2, r) / np.trapezoid(reference**2 * r**2, r))
            assert rel_l2 <= rel_target, f"l={l}: fitted shell misses the Li PSWFC by {rel_l2:.3e}"

            n_fit = len(fit.exponents)
            assert ext.exponents[:n_fit] == fit.exponents
            assert ext.coefficients[:n_fit] == fit.coefficients
            ladder = [max(fit.exponents) * PGTO_TIGHT_RATIO**k for k in range(1, PGTO_TIGHT_MAX_PER_SHELL + 1)]
            assert ext.exponents[n_fit:] == pytest.approx(ladder, rel=RTOL_PYTHON_FP64_LOOSE)
            assert ext.coefficients[n_fit:] == (1.0,) * PGTO_TIGHT_MAX_PER_SHELL
            assert np.isfinite(ext.overlap_cond) and ext.overlap_cond > fit.overlap_cond
