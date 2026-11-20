# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
APIDOC_DIR   = ./api_reference_mod

# Regenerate the per-package module pages from the docstrings. The table of
# contents ``modules.rst`` is maintained by hand (``--no-toc``) so that both
# packages appear on one page, and ``-e`` gives every module its own page so
# that the modules are listed in the navigation sidebar. The Rust crate under
# ``jqmc/rs`` is not a Python package and is excluded.
apidoc:
	sphinx-apidoc -f -e --no-toc -o ./$(APIDOC_DIR)/ ../jqmc ../jqmc/rs
	sphinx-apidoc -f -e --no-toc -o ./$(APIDOC_DIR)/ ../jqmc_workflow/
	@python3 -c 'import pathlib;f=pathlib.Path("$(APIDOC_DIR)/jqmc.rst");s=f.read_text("utf-8");i=s.index("Module contents");f.write_text(s[:i]+"Module contents\n---------------\n\nThe names re-exported by the ``jqmc`` package itself (its public API) are\ndocumented in :ref:`api_reference_link`.\n","utf-8")'

html: apidoc
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../docs_build $(SPHINXOPTS) $(O)

%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
