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

Normalisation (scripts/lib/ptx_canon.py), so a move or a deletion elsewhere
in the module, or a second compile of the same source, does not show as a
change. Each rule removes a name, never an instruction:
  - per-function counters: `$L__BB<f>_<n>`, `__local_depot<f>`, `$L__tmp<n>`;
  - virtual registers (`%r`, `%rd`, `%f`, `%p`, ...) renamed in first-use
    order per function, and the `.reg` counts; nvcc numbers them differently
    from one compile of an unchanged file to the next;
  - module-numbered symbols (`$str`, `$str$<n>`, `__unnamed_<n>`) renamed in
    first-use order within the entry's closure;
  - `_INTERNAL_<hash>_<len>_<file>_<hash>` TU ids (they encode the path), and
    in a mangled name the per-compile digits after them, cut by the length
    prefix;
  - byte-array initialisers that decode to a path ending in a kernel source
    extension (device asserts embed the staged path) become `<PATH>`;
  - `.file`/`.loc` debug lines and comments.
  - the `__assertfail` line-number immediate (`__LINE__`): deleting lines
    above an assert changes it. The values are kept beside the hash and the
    diff reports a change as `assert line arg changed`.
Trailing `// callseq N` comments (module-numbered) are dropped with all comments.

Usage:
  kernel_neutrality.py snapshot-ptx --profile-dir target/release --out gb10.json [--keep-dir D]
  kernel_neutrality.py explain BASE.json HEAD.json TARGET MODULE KERNEL
  kernel_neutrality.py rehash SNAP.json --out SNAP2.json
  kernel_neutrality.py snapshot-source --root . --out src.json
  kernel_neutrality.py diff BASE.json HEAD.json [--expect-deleted names.txt] [--noise BASE2.json]

`--expect-deleted` lists `module::kernel` (or `kernel`) lines. The diff passes
only if the removed entries are exactly those, nothing is added, and every
surviving entry has the same hash and the same ptxas numbers. Exit 0 = pass.
`explain` prints the diff of one entry's hashed text (both snapshots need --keep-dir).

ptxas runs only while no GPU benchmark holds BENCH_ACTIVE / BENCH_ACTIVE.d
(the cargo-guard `held` rule); it waits otherwise.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from ptx_canon import canonical_text, kernel_hashes  # noqa: E402

BENCH_LOCK = Path(os.environ.get("METRALE_BENCH_LOCK", "/workspace/claude-state/glm/agentic/BENCH_ACTIVE"))

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


# ── ptxas ─────────────────────────────────────────────────────────────────
def _ptxas(ptx: Path, arch: str, ptxas: str, entry: str | None):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as tmp:
        argv = [ptxas, "-v", f"-arch={arch}", str(ptx), "-o", tmp.name]
        if entry:
            argv[2:2] = ["-e", entry]
        return subprocess.run(argv, capture_output=True, text=True)


def _parse_ptxas(stderr: str) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    cur = None
    for ln in stderr.splitlines():
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


def ptxas_stats(ptx: Path, arch: str, ptxas: str, entries: list[str]) -> dict[str, dict]:
    """ptxas -v numbers per entry. A module ptxas rejects (some baselines carry entries over the
    static shared-memory limit) is redone one entry at a time; a rejected entry records its
    error text instead of numbers, so the diff still compares it."""
    wait_clear()
    p = _ptxas(ptx, arch, ptxas, None)
    if p.returncode == 0:
        return _parse_ptxas(p.stderr)
    stats: dict[str, dict] = {}
    for e in entries:
        wait_clear()
        q = _ptxas(ptx, arch, ptxas, e)
        if q.returncode == 0:
            stats.update(_parse_ptxas(q.stderr))
        else:
            err = [ln.split(":", 1)[-1].strip() for ln in q.stderr.splitlines() if "error" in ln]
            stats[e] = {"error": err}
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


def snapshot_ptx(profile_dir: Path, jobs: int, ptxas: str, keep: Path | None) -> dict:
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
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
        for sha, path in by_sha.items():
            if not (keep / f"{sha}.ptx.gz").exists():
                (keep / f"{sha}.ptx.gz").write_bytes(gzip.compress(path.read_bytes()))

    def one(item):
        sha, path = item
        text = path.read_text()
        arch = re.search(r"^\.target\s+(\S+)", text, re.M).group(1).rstrip(",")
        hashes = kernel_hashes(text)
        stats = ptxas_stats(path, arch, ptxas, list(hashes))
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
    snap = {"kind": "ptx", "ptx_dir": str(d), "arch": sorted(arches), "targets": {}, "files": {},
            "keep_dir": str(keep.resolve()) if keep else None}
    for label, mods in sorted(targets.items()):
        snap["targets"][label] = {mod: per_sha[file_sha[f]] for mod, f in sorted(mods.items())}
        snap["files"][label] = {mod: file_sha[f] for mod, f in sorted(mods.items())}
    return snap


def explain(base: dict, head: dict, target: str, module: str, kernel: str) -> int:
    """Print the unified diff of one kernel's canonical (hashed) text; needs both snapshots' --keep-dir."""
    texts = []
    for snap in (base, head):
        sha = snap["files"][target][module]
        ptx = gzip.decompress((Path(snap["keep_dir"]) / f"{sha}.ptx.gz").read_bytes()).decode()
        texts.append(canonical_text(ptx, kernel))
    out = list(difflib.unified_diff(texts[0], texts[1], "base", "head", lineterm="", n=2))
    print("\n".join(out) if out else "canonical texts identical")
    return 0


def rehash(snap: dict) -> dict:
    """Recompute every entry hash of a --keep-dir snapshot with the current canonical form (ptxas kept)."""
    cache: dict[str, dict] = {}
    for t, mods in snap["files"].items():
        for mod, sha in mods.items():
            if sha not in cache:
                ptx = gzip.decompress((Path(snap["keep_dir"]) / f"{sha}.ptx.gz").read_bytes()).decode()
                cache[sha] = kernel_hashes(ptx)
            for k, v in snap["targets"][t][mod].items():
                v.update(hash=cache[sha][k]["hash"], assert_line=cache[sha][k]["assert_line"])
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
def diff(base: dict, head: dict, expect: set[str] | None, noise: list[dict]) -> int:
    """`noise` holds rebuilds of the base: an entry whose hash already differs between two builds of
    the base source is compiler nondeterminism, so its head change is reported but not failed."""
    removed, added, changed, regs, aline, gone_t, new_t, noisy = [], [], [], [], [], [], [], []
    varies = {(t, mod, k) for n in noise for t, ms in base["targets"].items() for mod, ks in ms.items()
              for k, v in ks.items() if n["targets"].get(t, {}).get(mod, {}).get(k, v)["hash"] != v["hash"]}
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
                    (noisy if (t, mod, k) in varies else changed).append((t, mod, k))
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
    if noise:
        print(f"entries that differ between base builds: {len(varies)}; of those changed in head "
              f"(tolerated): {len(noisy)}")
    print(f"ptxas changed: {len(regs)}")
    print(f"assert line arg changed (informational): {len(aline)}")
    for t in gone_t:
        print(f"  TARGET GONE {t}")
    for t in new_t:
        print(f"  TARGET NEW {t}")
    for row in added[:50]:
        print("  ADDED", *row)
    for row in noisy[:20]:
        print("  NOISY", *row)
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
        def keys(pairs: set[tuple[str, str]]) -> set[str]:
            return {f"{m}::{k}" for m, k in pairs} | {k for _, k in pairs}
        built = keys({(m, k) for ms in base["targets"].values() for m, ks in ms.items() for k in ks})
        hit = keys(got)
        missing = sorted(e for e in expect if e in built and e not in hit)
        not_built = sorted(e for e in expect if e not in built)
        print(f"expected removals: {len(expect)}; not built for these targets (informational): {len(not_built)}")
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
    a.add_argument("--keep-dir", type=Path, help="store each unique PTX file (gzip, by sha) for `explain`")
    b = sp.add_parser("snapshot-source")
    b.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    b.add_argument("--out", type=Path, required=True)
    c = sp.add_parser("diff")
    c.add_argument("base", type=Path)
    c.add_argument("head", type=Path)
    c.add_argument("--expect-deleted", type=Path)
    c.add_argument("--noise", type=Path, action="append", default=[],
                   help="a second snapshot of the base build (repeatable); see diff()")
    r = sp.add_parser("rehash")
    r.add_argument("snapshot", type=Path)
    r.add_argument("--out", type=Path, required=True)
    e = sp.add_parser("explain")
    for arg in ("base", "head"):
        e.add_argument(arg, type=Path)
    for arg in ("target", "module", "kernel"):
        e.add_argument(arg)
    ns = ap.parse_args(argv)
    if ns.cmd == "rehash":
        ns.out.write_text(json.dumps(rehash(json.loads(ns.snapshot.read_text())), sort_keys=True))
        return 0
    if ns.cmd == "explain":
        return explain(json.loads(ns.base.read_text()), json.loads(ns.head.read_text()),
                       ns.target, ns.module, ns.kernel)
    if ns.cmd == "snapshot-ptx":
        ns.out.write_text(json.dumps(snapshot_ptx(ns.profile_dir, ns.jobs, ns.ptxas, ns.keep_dir), sort_keys=True))
        return 0
    if ns.cmd == "snapshot-source":
        ns.out.write_text(json.dumps(snapshot_source(ns.root.resolve()), sort_keys=True))
        return 0
    expect = None
    if ns.expect_deleted:
        expect = {ln.strip() for ln in ns.expect_deleted.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")}
    return diff(json.loads(ns.base.read_text()), json.loads(ns.head.read_text()), expect,
                [json.loads(p.read_text()) for p in ns.noise])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
