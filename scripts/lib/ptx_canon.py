# SPDX-License-Identifier: MIT OR Apache-2.0
"""PTX canonical form for kernel_neutrality.py: split a module into entries and declarations, and
reduce one entry plus everything it reaches to the text whose hash says "same kernel".

Every rule removes something two compiles of one source may differ in, or that moves when unrelated
code moves; none removes an instruction. The rules are listed in kernel_neutrality.py's docstring.
"""
from __future__ import annotations

import hashlib
import re

_TOP = re.compile(r"^(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)")
_DECL = re.compile(r"^(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(global|const|shared)\b.*?([A-Za-z_$][\w$]*)\s*(?:\[[^\]]*\])*\s*(?:=|;)")
_IDENT = re.compile(r"[A-Za-z_$%][\w$]*")
_NUMBERED = re.compile(r"^(\$str(\$\d+)?|__unnamed_\d+)$")
_INTERNAL = re.compile(r"[0-9a-f]{8}_\d+_[A-Za-z0-9_]*?_(?:cu|cuh|hip)_[0-9a-f]{8}")
# A mangled TU-local name `<len>_INTERNAL_<hex>_<n>_<file>_cu_<hex>_<digits>`: the trailing digits are
# per-compile (two builds of one file differ there), and only the length prefix says where they end.
_MANGLED_TU = re.compile(r"(\d+)(_INTERNAL_[0-9a-f]{8}_)")
_TU_WHOLE = re.compile(r"_INTERNAL_[0-9a-f]{8}_\d+_\w*?_(?:cu|cuh|hip)_[0-9a-f]{8}_\d+")
# Virtual register classes nvcc emits; special registers (%clock64, %envreg3, %pm0) are not renamed.
_VREG = re.compile(r"%(rd|rs|rq|r|fd|f|p|hh|h)(\d+)\b")
_LOCAL_SHARED = re.compile(r"\s+\.shared\s")
_BYTES = re.compile(r"=\s*\{([\d,\s]+)\}")
_PATH_EXT = (".cu", ".cuh", ".h", ".hip", ".metal")


def split_module(text: str):
    """Top-level items of one PTX module: {name: (kind, lines)} plus entry order."""
    lines = [re.sub(r"\s*//.*$", "", ln) for ln in text.splitlines()
             if not ln.lstrip().startswith((".file", ".loc"))]
    lines = [ln for ln in lines if ln.strip()]
    items: dict[str, tuple[str, list[str]]] = {}
    entries: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _TOP.match(ln)
        if m:
            kind, name = m.group(1), m.group(2)
            j = i
            body = [ln]
            if not ln.rstrip().endswith(";"):
                depth = 0
                opened = False
                while j < len(lines):
                    cur = lines[j] if j > i else ln
                    if j > i:
                        body.append(cur)
                    depth += cur.count("{") - cur.count("}")
                    opened = opened or "{" in cur
                    if opened and depth == 0:
                        break
                    if not opened and cur.rstrip().endswith(";"):
                        break
                    j += 1
            prev = items.get(name)
            if prev is None or len(body) > len(prev[1]):
                items[name] = ("entry" if kind == "entry" else "func", body)
            if kind == "entry" and ln.lstrip().startswith(".visible") and name not in entries:
                entries.append(name)
            i = j + 1
            continue
        d = _DECL.match(ln)
        if d and not ln.startswith(("\t", " ")):
            items[d.group(2)] = ("decl", [ln])
        i += 1
    return items, entries


def _mask_bytes(line: str) -> str:
    m = _BYTES.search(line)
    if not m:
        return line
    try:
        s = bytes(int(x) for x in m.group(1).split(",") if x.strip()).rstrip(b"\0").decode()
    except (ValueError, UnicodeDecodeError):
        return line
    if "/" in s and s.endswith(_PATH_EXT):
        return re.sub(r"\[\d+\]", "[]", line[: m.start()]) + "= <PATH>;"
    return line


def _mask_mangled_tu(ln: str) -> str:
    out, pos = [], 0
    for m in _MANGLED_TU.finditer(ln):
        if m.start() < pos:
            continue
        digits, at = m.group(1), m.start(2)
        for k in range(len(digits)):
            n = int(digits[k:])
            if n and _TU_WHOLE.fullmatch(ln, at, at + n):
                out.append(ln[pos:m.start(1) + k] + "<TU>")
                pos = at + n
                break
    return "".join(out) + ln[pos:]


def _normalise_body(body: list[str]) -> list[str]:
    """Drop what two compiles of one source may differ in: label, depot and virtual register
    numbers (registers renamed by first use) and TU ids."""
    out, regs = [], {}

    def reg(m: re.Match) -> str:
        return "%" + m.group(1) + "#" + str(regs.setdefault((m.group(1), m.group(2)), len(regs)))
    for ln in body:
        ln = re.sub(r"\$L__BB\d+_", "$L__BB_", ln)
        ln = re.sub(r"__local_depot\d+", "__local_depot", ln)
        ln = re.sub(r"\$L__tmp\d+", "$L__tmp", ln)
        ln = _INTERNAL.sub("<TU>", _mask_mangled_tu(ln))
        if re.match(r"\s*\.reg\s", ln):
            ln = re.sub(r"<\d+>", "<N>", ln)
        else:
            ln = _VREG.sub(reg, ln)
        out.append(ln.rstrip())
    return out


def _mask_assert_lines(body: list[str]) -> tuple[list[str], list[int]]:
    """Mask the line-number immediate (param2) of each `__assertfail` call block.

    Returns the masked body and the masked values, so a diff can still report them."""
    out, vals, start = list(body), [], None
    for i, ln in enumerate(body):
        t = ln.strip()
        if t == "{":
            start = i
        elif t == "}" and start is not None:
            block = body[start:i]
            if any("__assertfail" in b for b in block):
                for j in range(start, i):
                    m = re.match(r"(\s*st\.param\.b32\s+\[param2(?:\+0)?\],\s*)(\d+);", out[j])
                    if m:
                        vals.append(int(m.group(2)))
                        out[j] = m.group(1) + "<L>;"
            start = None
    return out, vals


def _canonical(items: dict, names: set[str], e: str) -> tuple[list[str], list]:
    """The lines an entry's hash covers: its body, then every declaration it reaches, renamed."""
    order: list[str] = []
    seen = {e}
    stack = [e]
    while stack:
        cur = stack.pop()
        for ln in items[cur][1]:
            for tok in _IDENT.findall(ln):
                if tok in names and tok not in seen:
                    seen.add(tok)
                    order.append(tok)
                    stack.append(tok)
    rename = {}
    for tok in [e] + order:
        if _NUMBERED.match(tok):
            rename[tok] = f"$G{len(rename)}"

    def sub(ln: str) -> str:
        return _IDENT.sub(lambda m: rename.get(m.group(0), m.group(0)), ln)

    def label(tok: str) -> str:
        return rename.get(tok) or _INTERNAL.sub("<TU>", _mask_mangled_tu(tok))

    # nvcc declares a static __shared__ of an inlined helper inside the entry when one kernel uses
    # it and at module scope when several do; both forms go to the declaration list.
    decls: dict[str, list[str]] = {}

    def hoist(lines: list[str]) -> list[str]:
        kept = []
        for ln in lines:
            d = _DECL.match(ln.strip()) if _LOCAL_SHARED.match(ln) else None
            if d:
                decls[label(d.group(2))] = [ln.strip()]
            else:
                kept.append(ln)
        return kept

    masked, alines = _mask_assert_lines(_normalise_body(items[e][1]))
    out = [sub(ln) for ln in hoist(masked)]
    for tok in order:
        lines = _mask_assert_lines(_normalise_body(items[tok][1]))[0]
        decls[label(tok)] = [ln.strip() if items[tok][0] == "decl" else ln for ln in hoist(lines)]
    for lab in sorted(decls):
        out.append(f"\0{lab}\0")
        out.extend(_mask_bytes(sub(ln)) for ln in decls[lab])
    return out, alines


def kernel_hashes(text: str) -> dict[str, dict]:
    items, entries = split_module(text)
    names = set(items)
    result = {}
    for e in entries:
        lines, alines = _canonical(items, names, e)
        h = hashlib.sha256()
        for ln in lines:
            h.update(ln.encode() + (b"" if ln.startswith("\0") else b"\n"))
        result[e] = {"hash": h.hexdigest()[:24], "assert_line": alines}
    return result


def canonical_text(text: str, entry: str) -> list[str]:
    items, _ = split_module(text)
    return _canonical(items, set(items), entry)[0]
