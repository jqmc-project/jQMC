"""Check rendered local links and inherited public driver API headings."""

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

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.add(attrs[key])


def check_html(directory: Path) -> list[str]:
    directory = directory.resolve()
    pages = {}
    for path in directory.rglob("*.html"):
        if "_static" in path.relative_to(directory).parts:
            continue
        page = Page()
        page.feed(path.read_text("utf-8"))
        pages[path] = page
    errors = []
    for path, page in pages.items():
        for link in sorted(page.links):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or link.startswith("/"):
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                errors.append(f"{path.name}: missing target {link}")
            elif parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
                errors.append(f"{path.name}: missing anchor {link}")
    api = pages.get(directory / "api_reference.html")
    for driver in ("MCMC", "GFMC_n", "GFMC_t"):
        for method in ("get_E", "get_aF", "get_aT", "get_aP", "load_from_hdf5"):
            anchor = f"jqmc.{driver}.{method}"
            if api is None or anchor not in api.ids:
                errors.append(f"api_reference.html: missing public method {anchor}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_directory", type=Path)
    errors = check_html(parser.parse_args().html_directory)
    if errors:
        print("\n".join(errors))
        return 1
    print("Rendered documentation links and public driver APIs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
