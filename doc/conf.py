"""Sphinx configuration of jQMC documentation."""

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

import pathlib
import sys
from datetime import datetime

# Make jqmc and jqmc_workflow importable for autodoc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.render_examples_doc import write_examples_page
from scripts.render_gdd_manual import write_manual_pages

# Keep the generated page current for local Sphinx builds. The pre-commit hook
# and the docs deploy workflow run ``python scripts/render_examples_doc.py
# --check`` so drift fails without mutating the checkout.
write_examples_page()
# Build the complete GDD reference as ordinary, untracked manual pages.
write_manual_pages()

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.mathjax",
    "myst_parser",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]
myst_enable_extensions = ["linkify", "dollarmath", "amsmath", "colon_fence"]
# Link only scheme-prefixed URLs (``https://...``). Fuzzy matching would turn
# identifiers such as ``logger.info`` or ``conftest.py`` into HTTP links.
myst_linkify_fuzzy_links = False
# Auto-generate anchors for Markdown headings up to depth 4 so that
# in-document links like ``[text](#phase-management)`` resolve.
myst_heading_anchors = 4

# Napoleon: parse Google/NumPy style docstrings (used in this project) and
# render attribute/parameter tables in autodoc pages.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_attr_annotations = True
napoleon_preprocess_types = True
# Custom section "Output Values" used by Workflow docstrings -- render as
# a parameter-style list (same as "Returns").
napoleon_custom_sections = [("Output Values", "params_style")]

# Move type hints from signatures into the description so Attributes/Parameters
# tables remain readable and avoid duplicated type-only attribute listings.
autodoc_typehints = "description"

# The public-API page (``api_reference.md``) documents the re-exported names
# such as ``jqmc.Hamiltonian_data`` while the module reference documents the
# same classes under their defining module (``jqmc.hamiltonians``). A type
# reference to ``Hamiltonian_data`` therefore has two valid targets; Sphinx
# picks the first and warns. That situation is intentional, so the ambiguity
# warning is silenced.
suppress_warnings = ["ref.python"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "jQMC"
author = "Kosuke Nakano"
copyright = f"{datetime.now().year}, {author}"

# The release this manual describes; doc/overview.md and doc/install.md name
# the same release in their text.
release = "0.3.0a1"
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# Paths, relative to this directory, that are not documentation sources.
exclude_patterns = [
    "_build",
    "README.md",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------------

html_theme_options = {
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/jqmc-project/jQMC", "icon": "fa-brands fa-github"},
    ],
}

html_theme = "sphinx_book_theme"

html_title = f"jQMC v{release}"

html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "jQMCdoc"

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "jQMC.tex", "jQMC manual", "Kosuke Nakano", "manual"),
]
