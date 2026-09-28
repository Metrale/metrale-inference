#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Loader-visible kernel entry points of one kernel source, read statically.

An entry point is what the engine can look up by name: an `extern "C"`
`__global__` function in a CUDA/HIP source (or in a local header the source
includes), or a `kernel` function in a Metal source. Names produced by
function-like macros (`#define ENTRY(NAME, ...) extern "C" __global__ void
NAME(...)`, `gated_delta_rule_wy##K`) are expanded for every invocation,
including a macro that invokes another.

`static __global__` and non-`extern "C"` template kernels are not
loader-visible (their names are mangled or internal) and are returned
separately so the document can count them without listing them as rows.

Pure: callers pass text and an include resolver; nothing here touches the
filesystem.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_DEFINE = re.compile(rf"^[ \t]*#[ \t]*define[ \t]+({IDENT})\(([^)]*)\)(.*)$", re.M)
_DEFINE_SPAN = re.compile(r"^[ \t]*#[ \t]*define\b(?:[^\n]*\\\n)*[^\n]*", re.M)
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"', re.M)
_LAUNCH_BOUNDS = re.compile(r"__launch_bounds__\s*\(")
_METAL = re.compile(rf"(?:\[\[\s*kernel\s*\]\]|\bkernel)\s+void\s+({IDENT})\s*\(")


@dataclass(frozen=True)
class Entry:
    name: str
    line: int
    defined_in: str  # the file (as the resolver named it) whose text holds the definition
    via_macro: str | None


def strip_comments(text: str) -> str:
    """Blank comments while keeping line numbers and string literals."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _join_continuations(text: str) -> str:
    """Join backslash-continued lines (used where only macro bodies, not line numbers, matter)."""
    return re.sub(r"\\\n", " ", text)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _balanced(text: str, open_pos: int) -> int:
    """Index just past the parenthesis that closes the one at open_pos."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def _split_args(s: str) -> list[str]:
    args, depth, cur = [], 0, []
    for ch in s:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        depth += ch in "([{"
        depth -= ch in ")]}"
        cur.append(ch)
    args.append("".join(cur).strip())
    return args


_OBJ_DEFINE = re.compile(rf"^[ \t]*#[ \t]*define[ \t]+({IDENT})(?![A-Za-z0-9_(])[ \t]*(.*)$", re.M)


def _declarator(text: str, pos: int, funcs: dict[str, "_Macro"]) -> str | None:
    """The name expression of the `__global__` at pos: every token between the return
    type and the parameter list, where a known function-like macro keeps its argument
    group (`PAGED_CONCAT(KERNEL_NAME, _64)`) and `__launch_bounds__(...)` is skipped."""
    i = pos + len("__global__")
    last: tuple[int, int] | None = None
    while i < len(text):
        m = re.compile(rf"\s*({IDENT}(?:\s*##\s*{IDENT})*)").match(text, i)
        if not m:
            break
        tok_end = m.end()
        j = tok_end
        while j < len(text) and text[j].isspace():
            j += 1
        ident = m.group(1)
        if j < len(text) and text[j] == "(":
            if ident == "__launch_bounds__":
                i = _balanced(text, j)
                continue
            if ident in funcs:
                return text[m.start(1):_balanced(text, j)]
            return ident
        last = (m.start(1), tok_end)
        i = tok_end
    return text[last[0]:last[1]] if last else None


def _expand(expr: str, params: dict[str, str], pre: "_Pre", depth: int = 0) -> str:
    """Just enough of the C preprocessor to spell a kernel name: parameter
    substitution with `##` pasting, then a rescan for object- and function-like
    macros (arguments pre-expanded)."""
    if depth > 16:
        return expr
    pieces = [p.strip() for p in expr.split("##")]
    if len(pieces) > 1:
        expr = "".join(params.get(p, p) for p in pieces)
    else:
        expr = re.sub(IDENT, lambda m: params.get(m.group(0), m.group(0)), expr)
    out: list[str] = []
    i = 0
    while i < len(expr):
        m = re.compile(IDENT).match(expr, i)
        if not m:
            out.append(expr[i])
            i += 1
            continue
        ident, i = m.group(0), m.end()
        k = i
        while k < len(expr) and expr[k].isspace():
            k += 1
        if ident in pre.funcs and k < len(expr) and expr[k] == "(":
            end = _balanced(expr, k)
            args = [_expand(a, {}, pre, depth + 1) for a in _split_args(expr[k + 1:end - 1])]
            mac = pre.funcs[ident]
            out.append(_expand(mac.body.strip(), dict(zip(mac.params, args)), pre, depth + 1))
            i = end
        elif ident in pre.objs and pre.objs[ident] != ident:
            out.append(_expand(pre.objs[ident], {}, pre, depth + 1))
        else:
            out.append(ident)
    return "".join(out).strip()


@dataclass
class _Macro:
    name: str
    params: list[str]
    body: str
    line: int
    file: str


@dataclass
class _Pre:
    funcs: dict[str, _Macro]
    objs: dict[str, str]


def _defines(text: str, file: str, pre: _Pre) -> None:
    """Add the file's macros to pre; the first definition in include order wins, as a
    `#define X` before `#include` does against the header's `#ifndef X` default."""
    joined = _join_continuations(text)
    for m in _DEFINE.finditer(joined):
        params = [p.strip() for p in m.group(2).split(",") if p.strip()]
        pre.funcs.setdefault(m.group(1), _Macro(m.group(1), params, m.group(3), _line_of(joined, m.start()), file))
    for m in _OBJ_DEFINE.finditer(joined):
        pre.objs.setdefault(m.group(1), m.group(2).strip())


def _macro_names(mac: _Macro, args: list[str], pre: _Pre, depth: int = 0) -> list[str]:
    """Every kernel name one invocation of mac(args) defines."""
    if depth > 8:
        return []
    subst = dict(zip(mac.params, args))
    names: list[str] = []
    body = mac.body
    for g in re.finditer(r"\b__global__\b", body):
        pre_text = body[max(0, g.start() - 40):g.start()].split()
        if pre_text and pre_text[-1] == "static":
            continue
        expr = _declarator(body, g.start(), pre.funcs)
        if expr:
            names.append(_expand(expr, subst, pre))
    for inner in pre.funcs.values():
        if inner.name == mac.name or "__global__" not in _flatten(inner, pre):
            continue
        for call in re.finditer(rf"\b{inner.name}\s*\(", body):
            end = _balanced(body, call.end() - 1)
            inner_args = [_expand(a, subst, pre) for a in _split_args(body[call.end():end - 1])]
            names.extend(_macro_names(inner, inner_args, pre, depth + 1))
    return names


def _flatten(mac: _Macro, pre: _Pre, depth: int = 0) -> str:
    """mac's body with the bodies of the function-like macros it invokes appended."""
    if depth > 8:
        return mac.body
    extra = [
        _flatten(inner, pre, depth + 1)
        for inner in pre.funcs.values()
        if inner.name != mac.name and re.search(rf"\b{inner.name}\s*\(", mac.body)
    ]
    return " ".join([mac.body, *extra])


def _extern_c(text: str, pos: int) -> bool:
    """Whether the `__global__` at pos is declared `extern "C"` (in the same declaration
    or inside an `extern "C" {` block)."""
    decl_start = max(text.rfind(";", 0, pos), text.rfind("}", 0, pos), text.rfind("{", 0, pos)) + 1
    if 'extern "C"' in text[decl_start:pos]:
        return True
    block = text.rfind('extern "C"', 0, pos)
    if block < 0:
        return False
    brace = text.find("{", block)
    return 0 <= brace < pos and text[block + len('extern "C"'):brace].strip() == "" and _balanced_brace(text, brace) > pos


def _balanced_brace(text: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def cuda_entries(
    name: str,
    text: str,
    include: Callable[[str, str], tuple[str, str] | None],
) -> tuple[list[Entry], list[Entry]]:
    """(loader-visible, internal) entry points of CUDA source `name` with `text`.
    `include(from_file, path)` returns (resolved name, text) of a quoted local include,
    or None when it is not a file of this tree (a toolkit header)."""
    files: list[tuple[str, str]] = []
    seen: set[str] = set()

    def visit(fname: str, ftext: str) -> None:
        if fname in seen:
            return
        seen.add(fname)
        clean = strip_comments(ftext)
        files.append((fname, clean))
        for inc in _INCLUDE.finditer(clean):
            got = include(fname, inc.group(1))
            if got is not None:
                visit(*got)

    visit(name, text)
    pre = _Pre({}, {})
    for fname, clean in files:
        _defines(clean, fname, pre)
    kernel_macros = {n: m for n, m in pre.funcs.items() if "__global__" in _flatten(m, pre)}
    visible: list[Entry] = []
    internal: list[Entry] = []
    for fname, clean in files:
        # Direct definitions: outside #define bodies.
        no_defs = _DEFINE_SPAN.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), clean)
        for g in re.finditer(r"\b__global__\b", no_defs):
            expr = _declarator(no_defs, g.start(), pre.funcs)
            if not expr:
                continue
            nm = _expand(expr, {}, pre)
            if not re.fullmatch(IDENT, nm):
                continue
            pre_text = no_defs[max(0, g.start() - 60):g.start()]
            templ = re.search(r"template\s*<[^;{}]*>\s*$", pre_text) is not None
            words = pre_text.split()
            is_static = bool(words) and words[-1] == "static"
            entry = Entry(nm, _line_of(no_defs, g.start()), fname, None if nm == expr else expr)
            if is_static or templ or not _extern_c(no_defs, g.start()):
                internal.append(entry)
            else:
                visible.append(entry)
        # Invocations of kernel-defining macros at top level.
        for mac in kernel_macros.values():
            for call in re.finditer(rf"(?<![A-Za-z0-9_#])\b{mac.name}\s*\(", no_defs):
                end = _balanced(no_defs, call.end() - 1)
                args = [_expand(a, {}, pre) for a in _split_args(no_defs[call.end():end - 1])]
                for nm in _macro_names(mac, args, pre):
                    if re.fullmatch(IDENT, nm):
                        visible.append(Entry(nm, _line_of(no_defs, call.start()), fname, mac.name))
    return _dedup(visible), _dedup(internal)


def metal_entries(name: str, text: str) -> list[Entry]:
    clean = strip_comments(text)
    return _dedup([Entry(m.group(1), _line_of(clean, m.start()), name, None) for m in _METAL.finditer(clean)])


def _dedup(entries: list[Entry]) -> list[Entry]:
    out: dict[str, Entry] = {}
    for e in entries:
        out.setdefault(e.name, e)
    return list(out.values())
