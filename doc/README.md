# How to write jQMC documentation

This directory contains `python-sphinx` documentation source.

## How to compile

Install the documentation dependencies first (from the repository root):

```bash
pip install ".[docs]"
```

Then, in this directory:

```bash
make html
```

`make html` regenerates the API pages and writes HTML to `../docs_build/`;
any other Sphinx target (`make latexpdf`, `make linkcheck`, ...) writes to
`_build/`.

For a fresh build that fails on warnings, run from the repository root:

```bash
make -C doc html SPHINXOPTS="-W --keep-going -E -a"
```

This regenerates the API pages and writes HTML to `docs_build/`. `-E` ignores
the saved Sphinx environment and `-a` rewrites every output page. When
reporting a build problem, include the checkout and Sphinx version:

```bash
git branch --show-current
git rev-parse --short HEAD
sphinx-build --version
```

The same build generates the GDD section from authored YAML and the
guarantee/evidence renderer. Open `docs_build/gdd_audit/index.html`, or use
the GDD entry in the manual navigation. Guarantee statements, tests, comparison
criteria and wavefunction fixtures retain their links.
Generated files under `doc/gdd_audit/`, `gdd/ledger/` and `gdd/derived/` are
local outputs and are not committed.

## Source files

* `Makefile` Makefile.
* `conf.py` contains the Sphinx configuration.
* `api_reference_mod/*.rst` are the module API pages that `make apidoc` (run by
  `make html`) regenerates from the docstrings.
* `*.md` are markdown documentation source files.

## Release documentation checks

From the repository root with the supported Python and documentation dependencies:

```bash
python examples/sync_examples.py
python scripts/render_examples_doc.py
python scripts/check_docs_examples.py
pytest -m light tests/test_jqmc_kernel/test_docs_examples.py
make -C doc html SPHINXOPTS="-W --keep-going -E -a"
python scripts/check_docs_html.py docs_build
pre-commit run --all-files
```

The static gate includes linear CLI staging traces (calculators modeled, not
executed), workflow output identity, all include markers, workflow docstring
parameters, and copyable shell syntax. Named `#configuration` include regions
keep workflow README settings synchronized with their executable scripts.
The HTML gate checks inherited driver methods and local links. These do not
replace full scientific runs with QE/PySCF, MPI, accelerators, and convergence
studies.

## How to deploy the documentation

A push to the `rc-gh-pages` branch of the public jQMC repository launches
`public-deploy-gh-pages.yml`, which builds the Sphinx manual and publishes it to
GitHub Pages.
