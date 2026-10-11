#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-11: The code corpora must partition the code: no file in none, none in two.

`coderag.yml` builds the engine's source as several corpora (`metrale-code-*`),
each a pair of include/exclude globs, because one corpus came out over the
publish cap. The split is only as good as the globs. The two failure modes are
both silent:

  * a GAP    -- a file one corpus includes but its exclude removes for the sake
                of another corpus, and that other corpus does not include. The
                site's chat then cannot answer questions about that file, and
                every build is still green;
  * an OVERLAP -- a file in two corpora is embedded twice, shows up twice in
                retrieval, and eats the headroom the split was cut for.

WHAT THIS CHECKS. It reads the globs out of coderag.yml (the single source; none
is restated here), expands them over `git ls-files` with the same glob
semantics AutoRepoRAG v2.0.0 uses (`src/util.mjs`: `**/` is zero or more whole
segments, `*` stays inside one segment, `?` is one non-separator character,
`{a,b}` alternates), and requires every tracked file that any code corpus
includes to be in exactly one of them -- except files that match an exclude
carried by ALL the code corpora, which are the deliberate drops (build output,
test files). The shared excludes are derived as that intersection, not listed.

WHAT IT CANNOT SEE: a file no corpus includes by extension (a new language), and
whether a corpus fits the byte cap -- the build's own size check does that.

Stdlib + PyYAML. Run from the repo root, or pass the root as argv[1].
"""
import pathlib
import re
import subprocess
import sys

import yaml

CORPUS_PREFIX = "metrale-code-"
REUSABLE = "./.github/workflows/coderag-corpus.yml"
REGEX_SPECIALS = set("\\^$.|+()[]{}")


def expand_braces(glob):
    m = re.search(r"\{([^{}]*)\}", glob)
    if not m:
        return [glob]
    out = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(glob[: m.start()] + alt + glob[m.end():]))
    return out


def glob_to_regex(glob):
    out, i = "", 0
    while i < len(glob):
        c = glob[i]
        if c == "*":
            if glob[i + 1 : i + 2] == "*":
                if (i == 0 or glob[i - 1] == "/") and glob[i + 2 : i + 3] == "/":
                    out += "(?:[^/]+/)*"
                    i += 3
                else:
                    out += ".*"
                    i += 2
            else:
                out += "[^/]*"
                i += 1
        elif c == "?":
            out += "[^/]"
            i += 1
        else:
            out += "\\" + c if c in REGEX_SPECIALS else c
            i += 1
    return re.compile("^" + out + "$")


def matcher(globs):
    regexes = [glob_to_regex(g) for glob in globs for g in expand_braces(glob)]
    return lambda path: any(r.match(path) for r in regexes)


def lines(value):
    return [ln.strip() for ln in (value or "").splitlines() if ln.strip()]


def code_corpora(workflow):
    corpora = {}
    for job_id, job in (workflow.get("jobs") or {}).items():
        if job.get("uses") != REUSABLE:
            continue
        w = job.get("with") or {}
        base = w.get("base", "")
        if base.startswith(CORPUS_PREFIX):
            corpora[base] = (lines(w.get("paths")), lines(w.get("exclude")))
    return corpora


def main(root="."):
    root = pathlib.Path(root)
    workflow = yaml.safe_load((root / ".github/workflows/coderag.yml").read_text(encoding="utf-8"))
    corpora = code_corpora(workflow)
    if len(corpora) < 2:
        print(f"REFUSE: found {len(corpora)} {CORPUS_PREFIX}* corpora in coderag.yml; a partition needs at least two", file=sys.stderr)
        return 1

    shared = set.intersection(*(set(ex) for _, ex in corpora.values()))
    drop = matcher(sorted(shared))
    includes = {b: matcher(inc) for b, (inc, _) in corpora.items()}
    excludes = {b: matcher(ex) for b, (_, ex) in corpora.items()}

    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout.split("\0")
    gaps, overlaps, counts = [], [], dict.fromkeys(corpora, 0)
    for path in filter(None, tracked):
        wanted = [b for b in corpora if includes[b](path)]
        if not wanted or drop(path):
            continue
        holders = [b for b in wanted if not excludes[b](path)]
        for b in holders:
            counts[b] += 1
        if not holders:
            gaps.append(f"{path} (included by {', '.join(wanted)}, kept by none)")
        elif len(holders) > 1:
            overlaps.append(f"{path} (in {', '.join(holders)})")

    empty = [b for b, n in counts.items() if n == 0]
    for b in empty:
        print(f"REFUSE: {b} selects no tracked file", file=sys.stderr)
    for g in gaps:
        print(f"REFUSE: gap: {g}", file=sys.stderr)
    for o in overlaps:
        print(f"REFUSE: overlap: {o}", file=sys.stderr)
    if empty or gaps or overlaps:
        return 1
    sizes = ", ".join(f"{b}={n}" for b, n in counts.items())
    print(f"ok: {sum(counts.values())} code files, each in exactly one of {len(corpora)} corpora ({sizes})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
