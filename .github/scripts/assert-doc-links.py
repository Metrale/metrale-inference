#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Relative and site-root markdown links must resolve to something that exists.

Found by sweeping: `docs/lora-implementation-status.md` linked to
`lora-mvp-proposal.md` and `lora-codebase-brief.md`, neither of which has ever
existed in this repository's history. The links were dead the day the file was
committed, and nothing noticed for as long as the file has been there.

PRECISION MATTERS MORE THAN COVERAGE HERE, and the reason is first-hand: the
throwaway sweep that found those two links reported nineteen, and seventeen were
its own bugs. It stripped the leading dot from `.github`, and it resolved
site-root URLs like `/images/...` against the filesystem. A link checker that
cries wolf is worse than none, because it will be muted or removed. So this one
is explicit about every root it knows and silent about everything else:

  * a link starting with `/` is a SITE-ROOT url. No tree in this repository
    publishes a static root, so every such link is refused rather than resolved
    against the repo root.
  * `book/**` links to `/api/...` are rustdoc output, placed under /api/ beside
    the book at deploy time (docs.metrale.ai/api/). They cannot exist in the
    tree and are not a defect.
  * external schemes, anchors, and template placeholders are not links to files.

Everything else is checked. If a root is added later and this refuses, the fix
is to teach it that root -- not to widen the ignore list.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(\s*([^)\s]+?)\s*\)|!\[[^\]]*\]\(\s*([^)\s]+?)\s*\)")
EXTERNAL = ("http://", "https://", "mailto:", "tel:", "data:", "#", "<")

# paths that are generated at deploy time and cannot exist in the tree
GENERATED = ("/api/",)
SKIP_DIRS = {"node_modules", "target", ".git", "build", "vendor"}


def main() -> None:
    broken: list[tuple[str, str, str]] = []
    checked = 0
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in md.relative_to(ROOT).parts):
            continue
        rel = str(md.relative_to(ROOT))
        for m in LINK.finditer(md.read_text(errors="ignore")):
            url = m.group(1) or m.group(2) or ""
            if not url or url.startswith(EXTERNAL):
                continue
            # `{{ }}`/`$(...)` are templates, not paths
            if "{" in url or "$" in url:
                continue
            target = url.split("#", 1)[0]
            if not target:
                continue
            if target.startswith("/"):
                if target.startswith(GENERATED):
                    continue
                broken.append((rel, url, "site-root link in a tree with no known static root"))
                continue
            path = md.parent / target
            checked += 1
            if not path.exists():
                broken.append((rel, url, "no such file"))

    if broken:
        for f, u, why in broken:
            print(f"REFUSE: {f} links to {u} -- {why}", file=sys.stderr)
        print(f"\n{len(broken)} broken link(s) of {checked} checked.", file=sys.stderr)
        sys.exit(1)
    print(f"ok: {checked} in-repo markdown links all resolve")


if __name__ == "__main__":
    main()
