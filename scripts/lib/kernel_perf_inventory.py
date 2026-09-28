#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The kernel inventory KERNEL-PERF.md is generated from, derived from the tree.

1. Targets: every (hardware, model, quant) `scripts/lib/kernel_layout.py` walks —
   the Python mirror of THE resolver (`crates/closure/src/layout.rs`), so `[sources]
   use`, `[hardware] inherits`, `[model] kernel_source` and `[shadow]` are applied
   exactly as the build applies them. A file several targets compile is one source.
2. Modules: the stem of each compiled file, renamed by the `[modules]` tables of
   the target's KERNEL.tomls merged least-specific first (the build's order).
3. Entry points: `kernel_perf_scan` over each compiled source and the local headers
   it includes.
4. Call sites: every string literal in `crates/**/*.rs` equal to an entry-point
   name, or a `{}`-templated literal that matches one (`format!` names).

I/O is limited to reading the tree; `build()` returns plain data.
"""
from __future__ import annotations

import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_layout  # noqa: E402
import kernel_perf_scan as scan  # noqa: E402

_STR = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


@dataclass
class Kernel:
    name: str
    file: str  # repo-relative file holding the definition (a .cu/.metal or a header)
    line: int
    hardware: str  # the kernels/<hw> tree the file lives in
    via_macro: str | None
    modules: set[str] = field(default_factory=set)
    targets: set[tuple[str, str, str]] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)  # compiled files (differs from `file` for header bodies)
    sites: list[tuple[str, int, str, bool]] = field(default_factory=list)  # (file, line, kind, exact)

    @property
    def key(self) -> str:
        return f"{self.file}::{self.name}"


@dataclass
class Inventory:
    kernels: dict[str, Kernel]
    targets: list[tuple[str, str, str]]
    internal: dict[str, list[str]]  # file -> non-loader-visible __global__ names
    hf_ids: dict[tuple[str, str], str]  # (hw, model dir) -> checkpoint id


def _rel(root: Path, p: Path) -> str:
    return os.path.relpath(p, root)


def _merged_modules(configs: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in configs:
        with open(c, "rb") as f:
            out.update(tomllib.load(f).get("modules", {}))
    return out


def _resolver(lay: kernel_layout.Layout, role: str, cache: dict[Path, str]):
    """Quoted-include resolution as the staged compile sees it: the role directory
    (its entries and subdirectories), the other role, then the includer's own directory."""
    order = [(lay.leaf, lay.leaf_subdirs), (lay.common, lay.common_subdirs)]
    if role == "common":
        order.reverse()

    def read(p: Path) -> str:
        if p not in cache:
            cache[p] = p.read_text(errors="replace")
        return cache[p]

    def include(from_file: str, inc: str):
        head, _, rest = inc.partition("/")
        for entries, subdirs in order:
            if not rest and inc in entries:
                p = entries[inc].source
                return str(p), read(p)
            if rest and head in subdirs and (subdirs[head] / rest).is_file():
                p = subdirs[head] / rest
                return str(p), read(p)
        p = Path(from_file).parent / inc
        if p.is_file():
            return str(p), read(p)
        return None

    return include, read


def _hf_ids(root: Path, targets) -> dict[tuple[str, str], str]:
    out = {}
    for hw, model, _q in targets:
        path = root / "kernels" / hw / model / "MODEL.toml"
        with open(path, "rb") as f:
            out[(hw, model)] = tomllib.load(f).get("model", {}).get("hf_id", model)
    return out


def scan_tree(root: Path) -> Inventory:
    targets = kernel_layout.walk(root)
    kernels: dict[str, Kernel] = {}
    internal: dict[str, list[str]] = {}
    cache: dict[Path, str] = {}
    parsed: dict[tuple[str, str], tuple] = {}
    for t in targets:
        lay = kernel_layout.discover(root, *t)
        renames = _merged_modules(lay.configs())
        for stem, entry in lay.modules().items():
            module = renames.get(stem, stem)
            role = "leaf" if entry.source in {e.source for e in lay.leaf.values()} else "common"
            include, read = _resolver(lay, role, cache)
            src = entry.source
            key = (str(src), role)
            if key not in parsed:
                text = read(src)
                if lay.hardware.source_ext == "metal":
                    parsed[key] = (scan.metal_entries(str(src), text), [])
                else:
                    parsed[key] = scan.cuda_entries(str(src), text, include)
            visible, hidden = parsed[key]
            for e in hidden:
                internal.setdefault(_rel(root, Path(e.defined_in)), [])
                if e.name not in internal[_rel(root, Path(e.defined_in))]:
                    internal[_rel(root, Path(e.defined_in))].append(e.name)
            for e in visible:
                file = _rel(root, Path(e.defined_in))
                k = kernels.setdefault(
                    f"{file}::{e.name}",
                    Kernel(e.name, file, e.line, Path(file).parts[1], e.via_macro),
                )
                k.modules.add(module)
                k.targets.add(t)
                k.sources.add(_rel(root, src))
    return Inventory(kernels, targets, internal, _hf_ids(root, targets))


_TEST_NAME = re.compile(r"(^|_)tests?(_|\.rs$)")
_CONST_STR = re.compile(r'\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"')


def _site_kind(rel: str) -> str:
    if "/examples/" in rel or "/benches/" in rel:
        return "example"
    if "/tests/" in rel or _TEST_NAME.search(Path(rel).name):
        return "test"
    return "engine"


def _test_lines(lines: list[str]) -> set[int]:
    """1-based line numbers inside an inline `#[cfg(test)] mod x { ... }` block."""
    out: set[int] = set()
    for i, line in enumerate(lines):
        if line.strip() != "#[cfg(test)]":
            continue
        if not re.match(r"\s*(pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{", " ".join(lines[i + 1:i + 3])):
            continue
        depth, opened = 0, False
        for j in range(i + 1, len(lines)):
            code = _STR.sub('""', lines[j].split("//")[0])
            depth += code.count("{") - code.count("}")
            opened = opened or "{" in code
            out.add(j + 1)
            if opened and depth <= 0:
                break
    return out


def _module_arg(window: str, name: str, consts: dict[str, str]) -> str | None:
    """The module a `.kernel(module, "name")` call names, when the window shows it."""
    m = re.search(r'kernel\(\s*([^,()]+?)\s*,\s*"' + re.escape(name) + '"', window)
    if not m:
        return None
    expr = m.group(1).strip().lstrip("&")
    if expr.startswith('"') and expr.endswith('"'):
        return expr[1:-1]
    return consts.get(expr.split("::")[-1])


def _rust_files(root: Path):
    for dp, dn, fn in os.walk(root / "crates"):
        dn[:] = sorted(d for d in dn if d != "target")
        for f in sorted(fn):
            if f.endswith(".rs"):
                yield Path(dp) / f


def attach_call_sites(root: Path, inv: Inventory) -> None:
    by_name: dict[str, list[Kernel]] = {}
    for k in inv.kernels.values():
        by_name.setdefault(k.name, []).append(k)
    names = set(by_name)
    files = [(p, _rel(root, p), p.read_text(errors="replace").splitlines()) for p in _rust_files(root)]
    seen_consts: dict[str, set[str]] = {}
    for _p, _rel_path, lines in files:
        for m in _CONST_STR.finditer("\n".join(lines)):
            seen_consts.setdefault(m.group(1), set()).add(m.group(2))
    consts = {k: next(iter(v)) for k, v in seen_consts.items() if len(v) == 1}
    templates: list[tuple[re.Pattern, str, int, str]] = []
    for _p, rel, lines in files:
        kind = _site_kind(rel)
        test_lines = _test_lines(lines)
        for lineno, line in enumerate(lines, 1):
            if line.lstrip().startswith("//"):
                continue
            line_kind = "test" if lineno in test_lines else kind
            for m in _STR.finditer(line):
                s = m.group(1)
                if s in names:
                    cands = by_name[s]
                    module = _module_arg(" ".join(lines[max(0, lineno - 4):lineno]), s, consts)
                    narrowed = [k for k in cands if module in k.modules] if module else []
                    for k in narrowed or cands:
                        k.sites.append((rel, lineno, line_kind, True))
                elif "{" in s and len(_PLACEHOLDER.split(s)[0]) >= 4:
                    pat = "".join(
                        re.escape(p) if i % 2 == 0 else "[A-Za-z0-9_]+"
                        for i, p in enumerate(_split_keep(s))
                    )
                    templates.append((re.compile(pat + r"\Z"), rel, lineno, line_kind))
    for k in inv.kernels.values():
        if any(exact for *_x, exact in k.sites):
            continue
        for pat, rel, lineno, kind in templates:
            if pat.match(k.name):
                k.sites.append((rel, lineno, kind, False))


def _split_keep(s: str) -> list[str]:
    """'a{x}b{y}' -> ['a', '{x}', 'b', '{y}', ''] (literals at even indexes)."""
    out, last = [], 0
    for m in _PLACEHOLDER.finditer(s):
        out += [s[last:m.start()], m.group(0)]
        last = m.end()
    out.append(s[last:])
    return out


def build(root: Path) -> Inventory:
    inv = scan_tree(root)
    attach_call_sites(root, inv)
    return inv
