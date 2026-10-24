#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Is a kernel-tree change behaviour-neutral? Snapshot every kernel, then diff two snapshots.

A snapshot records, per (target, module):
  (a) the set of `extern "C"` entry points;
  (b) per entry, a SHA-256 over its normalised PTX body plus every module-level
      declaration (.global/.const/.shared/.func) it reaches, transitively;
  (c) per entry, `ptxas -v` registers, spill stores/loads, stack frame, barriers
      and shared memory.

Two sources feed a snapshot:
  ptx     the PTX one `cargo build -p metrale-kernels` embeds (read via the
          METRALE_PTX_DIR of the newest build-script output under a profile
          dir). Gives (a), (b) and (c). Build once per hardware set and
          snapshot after each build: one target dir holds one hardware at a time.
  source  the repo's resolver mirror (scripts/lib) over every target of every
          hardware, compiled or not. Gives (a) only. It is the only view of
          hardware with no compiler on this host (strix, strix-hip, metal).

Normalisation, so a move or a deletion elsewhere in the module does not show
as a change. Each rule removes a name the compiler derives from position or
path, never an instruction:
  - per-function counters: `$L__BB<f>_<n>`, `__local_depot<f>`, `$L__tmp<n>`;
  - module-numbered symbols (`$str`, `$str$<n>`, `__unnamed_<n>`) renamed in
    first-use order within the entry's closure;
  - `_INTERNAL_<hash>_<len>_<file>_<hash>` TU ids (they encode the path);
  - byte-array initialisers that decode to a path ending in a kernel source
    extension (device asserts embed the staged path) become `<PATH>`;
  - `.file`/`.loc` debug lines and comments.
  - the `__assertfail` line-number immediate (`__LINE__`): deleting lines
    above an assert changes it. The values are kept beside the hash and the
    diff reports a change as `assert line arg changed`.
Trailing `// callseq N` comments (module-numbered) are dropped with all comments.

Usage:
  kernel_neutrality.py snapshot-ptx --profile-dir target/release --out gb10.json
  kernel_neutrality.py snapshot-source --root . --out src.json
  kernel_neutrality.py diff BASE.json HEAD.json [--expect-deleted names.txt]

`--expect-deleted` lists `module::kernel` (or `kernel`) lines. The diff passes
only if the removed entries are exactly those, nothing is added, and every
surviving entry has the same hash and the same ptxas numbers. Exit 0 = pass.

ptxas runs only while no GPU benchmark holds BENCH_ACTIVE / BENCH_ACTIVE.d
(the cargo-guard `held` rule); it waits otherwise.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH_LOCK = Path(os.environ.get("METRALE_BENCH_LOCK", "/workspace/claude-state/glm/agentic/BENCH_ACTIVE"))

_TOP = re.compile(r"^(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)")
_DECL = re.compile(r"^(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(global|const|shared)\b.*?([A-Za-z_$][\w$]*)\s*(?:\[[^\]]*\])*\s*(?:=|;)")
_IDENT = re.compile(r"[A-Za-z_$%][\w$]*")
_NUMBERED = re.compile(r"^(\$str(\$\d+)?|__unnamed_\d+)$")
_INTERNAL = re.compile(r"[0-9a-f]{8}_\d+_[A-Za-z0-9_]*?_(?:cu|cuh|hip)_[0-9a-f]{8}")
_BYTES = re.compile(r"=\s*\{([\d,\s]+)\}")
_PATH_EXT = (".cu", ".cuh", ".h", ".hip", ".metal")
_PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
_PTXAS_PROPS = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_PTXAS_USED = re.compile(r"Used (\d+) registers(.*)")


# ── lock ──────────────────────────────────────────────────────────────────
def held() -> bool:
    if BENCH_LOCK.exists():
        return True
    d = Path(str(BENCH_LOCK) + ".d")
    for f in d.glob("*") if d.is_dir() else []:
        pid = f.name.rsplit(".", 1)[-1]
        if pid.isdigit():
            try:
                os.kill(int(pid), 0)
                return True
            except ProcessLookupError:
                continue
            except PermissionError:
                return True
    return False


def wait_clear() -> None:
    announced = False
    while held():
        if not announced:
            print("kernel_neutrality: waiting, a GPU bench holds the lock", file=sys.stderr)
            announced = True
        time.sleep(30)


# ── PTX parsing ───────────────────────────────────────────────────────────
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


def _normalise_body(body: list[str]) -> list[str]:
    out = []
    for ln in body:
        ln = re.sub(r"\$L__BB\d+_", "$L__BB_", ln)
        ln = re.sub(r"__local_depot\d+", "__local_depot", ln)
        ln = re.sub(r"\$L__tmp\d+", "$L__tmp", ln)
        ln = _INTERNAL.sub("<TU>", ln)
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
                    m = re.match(r"(\s*st\.param\.b32\s+\[param2\],\s*)(\d+);", out[j])
                    if m:
                        vals.append(int(m.group(2)))
                        out[j] = m.group(1) + "<L>;"
            start = None
    return out, vals


def kernel_hashes(text: str) -> dict[str, dict]:
    items, entries = split_module(text)
    names = set(items)
    result = {}
    for e in entries:
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

        h = hashlib.sha256()
        masked, alines = _mask_assert_lines(_normalise_body(items[e][1]))
        for ln in masked:
            h.update(sub(ln).encode() + b"\n")
        for tok in sorted(order, key=lambda t: rename.get(t, t)):
            h.update(b"\0" + rename.get(tok, tok).encode() + b"\0")
            for ln in _mask_assert_lines(_normalise_body(items[tok][1]))[0]:
                h.update(_mask_bytes(sub(ln)).encode() + b"\n")
        result[e] = {"hash": h.hexdigest()[:24], "assert_line": alines}
    return result


# ── ptxas ─────────────────────────────────────────────────────────────────
def ptxas_stats(ptx: Path, arch: str, ptxas: str) -> dict[str, dict]:
    wait_clear()
    with tempfile.NamedTemporaryFile(suffix=".cubin") as tmp:
        p = subprocess.run([ptxas, "-v", f"-arch={arch}", str(ptx), "-o", tmp.name],
                           capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ptxas failed on {ptx}: {p.stderr[-2000:]}")
    stats: dict[str, dict] = {}
    cur = None
    for ln in p.stderr.splitlines():
        m = _PTXAS_ENTRY.search(ln)
        if m:
            cur = m.group(1)
            stats[cur] = {}
            continue
        if cur is None:
            continue
        m = _PTXAS_PROPS.search(ln)
        if m:
            stats[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = _PTXAS_USED.search(ln)
        if m:
            stats[cur]["regs"] = int(m.group(1))
            stats[cur]["used"] = m.group(2).strip(", ")
    return stats


def _ptx_dir(profile_dir: Path) -> Path:
    outs = sorted(profile_dir.glob("build/metrale-kernels-*/output"), key=lambda p: p.stat().st_mtime)
    for out in reversed(outs):
        for ln in out.read_text(errors="replace").splitlines():
            if ln.startswith("cargo:rustc-env=METRALE_PTX_DIR="):
                d = Path(ln.split("=", 2)[2])
                if (d / "target_ptx.rs").is_file():
                    return d
    raise SystemExit(f"no metrale-kernels build output with a target_ptx.rs under {profile_dir}")


def parse_target_ptx(rs: str) -> dict[str, dict[str, str]]:
    """target label -> {module name -> ptx file name}, from the generated target_ptx.rs."""
    consts = dict(re.findall(r'pub const (\w+): &\[u8\] = include_bytes!\(concat!\(env!\("METRALE_PTX_DIR"\), "/([^"]+)"\)\);', rs))
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"/// Kernel modules for target: \(([^,]+), ([^,]+), ([^)]+)\)\.\npub fn \w+\(\)[^{]*\{\n\s*vec!\[\n(.*?)\n\s*\]", rs, re.S):
        label = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        out[label] = {mod: consts[c] for mod, c in re.findall(r'\("([^"]+)", (\w+)\)', m.group(4))}
    if not out:
        raise SystemExit("target_ptx.rs lists no targets (a METRALE_SKIP_BUILD stub?)")
    return out


def snapshot_ptx(profile_dir: Path, jobs: int, ptxas: str) -> dict:
    d = _ptx_dir(profile_dir)
    targets = parse_target_ptx((d / "target_ptx.rs").read_text())
    by_sha: dict[str, Path] = {}
    file_sha: dict[str, str] = {}
    for mods in targets.values():
        for f in mods.values():
            if f not in file_sha:
                sha = hashlib.sha256((d / f).read_bytes()).hexdigest()
                file_sha[f] = sha
                by_sha.setdefault(sha, d / f)
    print(f"kernel_neutrality: {d}: {len(targets)} targets, {len(file_sha)} files, {len(by_sha)} unique",
          file=sys.stderr)

    def one(item):
        sha, path = item
        text = path.read_text()
        arch = re.search(r"^\.target\s+(\S+)", text, re.M).group(1).rstrip(",")
        hashes = kernel_hashes(text)
        stats = ptxas_stats(path, arch, ptxas)
        for k in hashes:
            hashes[k]["ptxas"] = stats.get(k)
        return sha, arch, hashes

    per_sha: dict[str, dict] = {}
    arches = set()
    with cf.ThreadPoolExecutor(jobs) as ex:
        for n, (sha, arch, h) in enumerate(ex.map(one, sorted(by_sha.items())), 1):
            per_sha[sha] = h
            arches.add(arch)
            if n % 200 == 0:
                print(f"  {n}/{len(by_sha)}", file=sys.stderr)
    snap = {"kind": "ptx", "ptx_dir": str(d), "arch": sorted(arches), "targets": {}}
    for label, mods in sorted(targets.items()):
        snap["targets"][label] = {mod: per_sha[file_sha[f]] for mod, f in sorted(mods.items())}
    return snap


def snapshot_source(root: Path) -> dict:
    sys.path.insert(0, str(root / "scripts" / "lib"))
    import kernel_perf_inventory as inventory  # noqa: E402

    inv = inventory.scan_tree(root)
    snap: dict = {"kind": "source", "targets": {"/".join(t): {} for t in inv.targets}}
    for k in inv.kernels.values():
        for t in k.targets:
            for mod in k.modules:
                snap["targets"]["/".join(t)].setdefault(mod, {})[k.name] = {}
    return snap


# ── diff ──────────────────────────────────────────────────────────────────
def diff(base: dict, head: dict, expect: set[str] | None) -> int:
    removed, added, changed, regs, aline, gone_t, new_t = [], [], [], [], [], [], []
    for t in sorted(set(base["targets"]) | set(head["targets"])):
        if t not in head["targets"]:
            gone_t.append(t)
            continue
        if t not in base["targets"]:
            new_t.append(t)
            continue
        bm, hm = base["targets"][t], head["targets"][t]
        for mod in sorted(set(bm) | set(hm)):
            bk, hk = bm.get(mod, {}), hm.get(mod, {})
            for k in sorted(set(bk) - set(hk)):
                removed.append((t, mod, k))
            for k in sorted(set(hk) - set(bk)):
                added.append((t, mod, k))
            for k in sorted(set(bk) & set(hk)):
                a, b = bk[k], hk[k]
                if a.get("hash") != b.get("hash"):
                    changed.append((t, mod, k))
                if a.get("ptxas") != b.get("ptxas"):
                    regs.append((t, mod, k, a.get("ptxas"), b.get("ptxas")))
                if a.get("assert_line") != b.get("assert_line"):
                    aline.append((t, mod, k, a.get("assert_line"), b.get("assert_line")))
    surviving = sum(len(set(m) & set(head["targets"][t].get(mod, {})))
                    for t, ms in base["targets"].items() if t in head["targets"] for mod, m in ms.items())
    print(f"targets: base {len(base['targets'])}, head {len(head['targets'])}; surviving entries compared: {surviving}")
    print(f"removed entries: {len(removed)} ({len({(m, k) for _, m, k in removed})} distinct module::kernel)")
    print(f"added entries: {len(added)}")
    print(f"hash changed: {len(changed)}")
    print(f"ptxas changed: {len(regs)}")
    print(f"assert line arg changed (informational): {len(aline)}")
    for t in gone_t:
        print(f"  TARGET GONE {t}")
    for t in new_t:
        print(f"  TARGET NEW {t}")
    for row in added[:50]:
        print("  ADDED", *row)
    for row in changed[:50]:
        print("  HASH", *row)
    for row in regs[:50]:
        print("  PTXAS", *row)
    for row in aline[:20]:
        print("  ASSERT-LINE", *row)
    ok = not (added or changed or regs or gone_t or new_t)
    if expect is not None:
        got = {(m, k) for _, m, k in removed}
        def wanted(m: str, k: str) -> bool:
            return f"{m}::{k}" in expect or k in expect
        unexpected = sorted(x for x in got if not wanted(*x))
        named = {e for e in expect}
        hit = {f"{m}::{k}" for m, k in got} | {k for _, k in got}
        missing = sorted(e for e in named if e not in hit)
        for x in unexpected:
            print("  UNEXPECTED REMOVAL", "::".join(x))
        for x in missing:
            print("  EXPECTED BUT STILL PRESENT", x)
        ok = ok and not unexpected and not missing
    elif removed:
        for row in removed[:50]:
            print("  REMOVED", *row)
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("snapshot-ptx")
    a.add_argument("--profile-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a.add_argument("--ptxas", default=os.environ.get("PTXAS", "ptxas"))
    b = sp.add_parser("snapshot-source")
    b.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    b.add_argument("--out", type=Path, required=True)
    c = sp.add_parser("diff")
    c.add_argument("base", type=Path)
    c.add_argument("head", type=Path)
    c.add_argument("--expect-deleted", type=Path)
    ns = ap.parse_args(argv)
    if ns.cmd == "snapshot-ptx":
        ns.out.write_text(json.dumps(snapshot_ptx(ns.profile_dir, ns.jobs, ns.ptxas), sort_keys=True))
        return 0
    if ns.cmd == "snapshot-source":
        ns.out.write_text(json.dumps(snapshot_source(ns.root.resolve()), sort_keys=True))
        return 0
    expect = None
    if ns.expect_deleted:
        expect = {ln.strip() for ln in ns.expect_deleted.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")}
    return diff(json.loads(ns.base.read_text()), json.loads(ns.head.read_text()), expect)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
