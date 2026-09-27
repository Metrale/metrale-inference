#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check every launch recipe against this tree's own `met`, and build the
release bundle from the recipes that passed.

    recipes.py check --met target/debug/met --out recipes-dist [--commit SHA]

Checks (the rules are in recipes_lib.py, which says where each comes from):

  1. every recipes/**/*.yaml parses, with the engine's required keys;
  2. every key and value is on `met dump-serve-options`' flag surface, and
     every METRALE_* name in `env:` is a declared lever;
  3. every runtime: metrale recipe, rendered as the engine renders it, is
     accepted by `met serve` (clap, the lever check, `validate_serve_args`);
  4. every `recipe = "..."` in kernels/**/BENCH.toml names a recipe here;
  5. the engine's own reader (`met doctor` over the bundle's index.json) keeps
     every recipe;
  6. ADVISORY: gpu_memory_utilization above the box class's ceiling. Warns,
     never fails.

Writes to --out: recipes.tar.gz, serve-options.json, index.json, each with a
`.sha256` sidecar. Exit 1 when any check other than 6 fails.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import recipes_lib as lib  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_recipes(recipes_dir: pathlib.Path) -> tuple[list[lib.Recipe], list[lib.Finding]]:
    recipes, findings = [], []
    for path in sorted(recipes_dir.rglob("*.yaml")):
        rel = path.relative_to(recipes_dir).as_posix()
        try:
            # Bytes, not read_text(): no newline translation, so `sha256` is the file's.
            recipes.append(lib.parse(rel, path.read_bytes().decode("utf-8")))
        except lib.RecipeError as e:
            findings.append(lib.Finding("parse", rel, str(e)))
    return recipes, findings


def load_bench_refs(kernels_dir: pathlib.Path) -> dict[str, list[str]]:
    refs = {}
    for path in sorted(kernels_dir.rglob("BENCH.toml")):
        with path.open("rb") as f:
            found = lib.bench_refs(tomllib.load(f))
        if found:
            refs[path.relative_to(kernels_dir.parent).as_posix()] = found
    return refs


def clean_env(home: str, extra: dict[str, str]) -> dict[str, str]:
    """No inherited METRALE_*: CI's own build switches are not levers, and
    `met` refuses to start under an undeclared one."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, **extra}


def run(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout, p.stderr


def dump_manifest(met: str, home: str) -> bytes:
    code, out, err = run([met, "dump-serve-options"], clean_env(home, {}))
    if code != 0:
        raise SystemExit(f"`{met} dump-serve-options` failed (exit {code}): {err.strip()}")
    json.loads(out)
    return out.encode()


def check_met(met: str, recipes: list[lib.Recipe], table: dict, home: str) -> list[lib.Finding]:
    findings = []
    for r in recipes:
        if not r.is_metrale:
            continue
        cmd = lib.met_command(met, lib.render_argv(r, table))
        code, out, err = run(cmd, clean_env(home, r.env))
        ok, why = lib.met_verdict(code, out + err)
        if not ok:
            findings.append(lib.Finding("met-refused", r.id, why))
    return findings


def check_doctor(met: str, index: bytes, expected: int) -> list[lib.Finding]:
    with tempfile.TemporaryDirectory() as home:
        cache = pathlib.Path(home, ".metrale", "metrale-recipes")
        cache.mkdir(parents=True)
        (cache / "index.json").write_bytes(index)
        _, out, _ = run([met, "doctor"], clean_env(home, {}))
    ok, why = lib.doctor_verdict(out, expected)
    return [] if ok else [lib.Finding("engine-reader", "index.json", why)]


def tarball(recipes_dir: pathlib.Path) -> bytes:
    """Byte-reproducible: sorted members, zero mtimes and owners, gzip without
    a timestamp. The same tree always hashes the same."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in recipes_dir.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(path.relative_to(recipes_dir.parent).as_posix())
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def write_asset(out: pathlib.Path, name: str, data: bytes) -> None:
    (out / name).write_bytes(data)
    (out / f"{name}.sha256").write_text(lib.sidecar(name, data))


def commit_info(commit: str | None) -> tuple[str, int]:
    rev = commit or "HEAD"
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", rev], capture_output=True, text=True, check=True)
    ts = subprocess.run(["git", "-C", str(ROOT), "show", "-s", "--format=%ct", rev],
                        capture_output=True, text=True, check=True)
    return sha.stdout.strip(), int(ts.stdout.strip())


def report(findings: list[lib.Finding], github: bool) -> None:
    for f in findings:
        print(f.render())
        if github:
            level = "error" if f.error else "warning"
            print(f"::{level} title=recipes: {f.kind}::{f.subject}: {f.message}")


def check(args: argparse.Namespace) -> int:
    recipes_dir, out = pathlib.Path(args.recipes), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipes, findings = load_recipes(recipes_dir)
    with tempfile.TemporaryDirectory() as home:
        manifest_bytes = dump_manifest(args.met, home)
        manifest = json.loads(manifest_bytes)
        table, levers = lib.flag_table(manifest), lib.lever_names(manifest)
        for r in recipes:
            findings += lib.check_surface(r, table, levers)
        findings += check_met(args.met, recipes, table, home)
    ids = {r.id for r in recipes}
    findings += lib.check_bench_refs(load_bench_refs(pathlib.Path(args.kernels)), ids)
    for r in recipes:
        findings += lib.check_util(r)

    tar = tarball(recipes_dir)
    commit, committed_at = commit_info(args.commit)
    index = lib.dumps(lib.index_document(recipes, commit, committed_at,
                                         {"recipes.tar.gz": tar, "serve-options.json": manifest_bytes},
                                         manifest["engine_version"]))
    findings += check_doctor(args.met, index, len(list(recipes_dir.rglob("*.yaml"))))

    errors = [f for f in findings if f.error]
    warnings = [f for f in findings if not f.error]
    report(errors + warnings, args.github)
    skipped = sorted(r.id for r in recipes if not r.is_metrale)
    print(f"{len(recipes)} recipes read; {len(recipes) - len(skipped)} served by `met` checked against "
          f"{manifest['engine_version']} ({len(manifest['flags'])} flags, {len(manifest['levers'])} levers); "
          f"not launchable by `met`, so parse-only: {', '.join(skipped) or 'none'}")
    print(f"{len(errors)} error(s), {len(warnings)} advisory warning(s)")
    if errors:
        print("no bundle written: the recipes did not pass")
        return 1
    write_asset(out, "recipes.tar.gz", tar)
    write_asset(out, "serve-options.json", manifest_bytes)
    write_asset(out, "index.json", index)
    print(f"bundle for {commit} written to {out}/")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="validate every recipe and write the bundle")
    c.add_argument("--met", required=True, help="the `met` binary built from this tree")
    c.add_argument("--out", required=True, help="directory for the bundle")
    c.add_argument("--recipes", default=str(ROOT / "recipes"))
    c.add_argument("--kernels", default=str(ROOT / "kernels"))
    c.add_argument("--commit", help="commit the bundle is stamped with (default HEAD)")
    c.add_argument("--github", action="store_true", help="also emit ::error/::warning annotations")
    args = ap.parse_args()
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
