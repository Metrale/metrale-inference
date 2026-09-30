#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-test for recipes.py / recipes_lib.py.

    recipes-selftest.py               the checks, on fixtures, with no binary
    recipes-selftest.py --met PATH    also through a real `met`

Three kinds of case, and why each is needed:

  * the POSITIVE control: the fixture corpus passes with no finding. Without
    it, a checker that refused everything would pass every negative control.
  * a NEGATIVE control per check: one defect, and the finding it must raise.
  * a MUTATION per check: the check is replaced by one that finds nothing, and
    its negative control must then FAIL. A control that still passes against a
    broken check is not testing that check.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recipes  # noqa: E402
import recipes_lib as lib  # noqa: E402

FIX = HERE / "recipes-fixtures"
RESULTS: list[tuple[str, bool, str]] = []


def case(name: str):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except AssertionError as e:
            RESULTS.append((name, False, str(e) or "assertion failed"))
        return fn
    return wrap


def corpus() -> list[lib.Recipe]:
    rs, bad = recipes.load_recipes(FIX / "recipes")
    assert not bad, [f.render() for f in bad]
    return rs


def good() -> lib.Recipe:
    return next(r for r in corpus() if r.id == "fam/good")


def manifest() -> dict:
    return json.loads((FIX / "serve-options.json").read_text())


def surface(r: lib.Recipe, m: dict | None = None) -> list[str]:
    m = m or manifest()
    return [f.kind for f in lib.check_surface(r, lib.flag_table(m), lib.lever_names(m))]


def edited(**defaults: str) -> lib.Recipe:
    r = copy.deepcopy(good())
    r.defaults.update(defaults)
    return r


def bench_kinds(refs: dict[str, list[str]]) -> list[str]:
    return [f.kind for f in lib.check_bench_refs(refs, {r.id for r in corpus()})]


# ── controls, as plain functions so a mutation can re-run them ────────────────


def c_positive() -> None:
    rs = corpus()
    assert [r.id for r in rs] == ["fam/good", "fam/other"], [r.id for r in rs]
    for r in rs:
        assert surface(r) == [], (r.id, surface(r))
        assert lib.check_util(r) == [], (r.id, lib.check_util(r))
    refs = recipes.load_bench_refs(FIX / "kernels")
    assert refs == {"kernels/gb10/model/BENCH.toml": ["fam/good", "fam/good"]}, refs
    assert bench_kinds(refs) == []


def c_render() -> None:
    argv = lib.render_argv(good(), lib.flag_table(manifest()))
    assert argv == ["met", "serve", "example/model", "--ep-size", "2", "--gpu-memory-utilization", "0.85",
                    "--bind", "0.0.0.0", "--max-seq-len", "4096", "--port", "8888", "--scheduler", "slai",
                    "--speculative", "--world-size", "2"], argv
    cmd = lib.met_command("met", argv)
    assert cmd[:2] == ["met", "serve"] and "example/model" not in cmd and cmd[-1] == "--no-tui", cmd
    assert lib.met_command("met", argv + ["--no-tui"]).count("--no-tui") == 1


def c_unknown_key() -> None:
    assert surface(edited(bogus_key="1")) == ["unknown-key"]


def c_bad_value() -> None:
    assert surface(edited(scheduler="fcfs")) == ["bad-value"]


def c_presence_only() -> None:
    assert surface(edited(speculative="false")) == ["presence-only"]
    assert surface(edited(speculative="1")) == ["presence-only"]


def c_unknown_lever() -> None:
    r = copy.deepcopy(good())
    r.env["METRALE_NOT_A_LEVER"] = "1"
    assert surface(r) == ["unknown-lever"]


def c_bench_missing() -> None:
    assert bench_kinds({"kernels/x/BENCH.toml": ["fam/good", "fam/nope"]}) == ["bench-missing"]
    with tempfile.TemporaryDirectory() as d:
        k = pathlib.Path(d, "kernels", "gb10", "m")
        k.mkdir(parents=True)
        (k / "BENCH.toml").write_text('[[baseline]]\nrecipe = "fam/nope"\n')
        assert bench_kinds(recipes.load_bench_refs(pathlib.Path(d, "kernels"))) == ["bench-missing"]


def c_util() -> None:
    over = lib.check_util(edited(gpu_memory_utilization="0.90"))
    assert [(f.kind, f.error) for f in over] == [("util-ceiling", False)], over
    assert lib.check_util(edited(gpu_memory_utilization="0.85")) == []
    other = next(r for r in corpus() if r.id == "fam/other")
    assert lib.check_util(other) == [], "a container with no hardware class is not judged"


def c_parse() -> None:
    text = (FIX / "recipes/fam/good.yaml").read_text()
    for broken, why in [(text.replace("model: example/model\n", ""), "model"),
                        (text.replace("min_nodes: 2", "min_nodes: two"), "min_nodes"),
                        (text.replace("  scheduler: slai\n", "  scheduler:\n    - slai\n"), "scalar")]:
        try:
            lib.parse("fam/x.yaml", broken)
        except lib.RecipeError as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"parse accepted a recipe with a broken {why}")


def c_met_verdict() -> None:
    table = [((1, f"Error: {lib.SENTINEL}.\n"), True), ((0, ""), False), ((2, "error: unexpected argument"), False),
             ((1, "Error: Metrale Engine CLI: 1 invalid flag combination"), False), ((1, ""), False),
             ((-9, lib.SENTINEL), False)]
    for (code, out), want in table:
        assert lib.met_verdict(code, out)[0] is want, (code, out, want)


def c_doctor_verdict() -> None:
    line = "      ok  recipes   2 cached in /h/.metrale/metrale-recipes/index.json\n"
    assert lib.doctor_verdict(line, 2)[0]
    assert not lib.doctor_verdict(line, 3)[0]
    assert not lib.doctor_verdict(" PROBLEM  recipes   /x has never been written\n", 2)[0]


def c_bundle() -> None:
    a, b = recipes.tarball(FIX / "recipes"), recipes.tarball(FIX / "recipes")
    assert a == b, "the tarball is not byte-reproducible"
    doc = lib.index_document(corpus(), "c" * 40, 7, {"recipes.tar.gz": a}, "fixture")
    assert [x["id"] for x in doc["recipes"]] == ["fam/good", "fam/other"]
    raw = (FIX / "recipes/fam/good.yaml").read_bytes()
    assert doc["recipes"][0]["sha256"] == lib.sha256(raw) and doc["files"]["fam/good"] == raw.decode()
    assert doc["tree_sha"] == "c" * 40 and doc["fetched_at"] == 7
    assert lib.sidecar("x.json", b"") == f"{lib.sha256(b'')}  x.json\n"


def c_mirror() -> None:
    same = b'{"flags": []}'
    assert lib.mirror_verdict([], same, same)[0] is False, "an unchanged release was mirrored"
    assert lib.mirror_verdict(["recipes/a/b.yaml"], same, same)[0] is True, "a recipe change was not mirrored"
    assert lib.mirror_verdict([], b"{}", same)[0] is True, "a serve-options change was not mirrored"
    assert lib.mirror_verdict(None, same, same)[0] is True, "an unknown diff did not fail open"
    assert lib.mirror_verdict([], None, same)[0] is True, "an unread snapshot did not fail open"


def c_mirror_cli() -> None:
    # Through the subcommand and a real git: an unknown pinned commit fails open, and the
    # same commit on both sides with identical serve options is not mirrored.
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        so = pathlib.Path(d) / "so.json"
        so.write_bytes(b"{}")
        head = subprocess.run(["git", "-C", str(recipes.ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()

        def verdict(pinned: str, pinned_file: pathlib.Path) -> str:
            out = subprocess.run([sys.executable, str(HERE / "recipes.py"), "mirror-needed",
                                  "--pinned-commit", pinned, "--head", head,
                                  "--pinned-serve-options", str(pinned_file),
                                  "--release-serve-options", str(so)],
                                 capture_output=True, text=True, check=True).stdout
            return out.splitlines()[0]
        assert verdict(head, so) == "dispatch=false", "the same commit and snapshot were mirrored"
        assert verdict("0" * 40, so) == "dispatch=true", "an unknown pinned commit did not fail open"
        assert verdict(head, pathlib.Path(d) / "absent.json") == "dispatch=true", "a missing snapshot"


PURE = {"positive": c_positive, "render": c_render, "unknown-key": c_unknown_key, "bad-value": c_bad_value,
        "presence-only": c_presence_only, "unknown-lever": c_unknown_lever, "bench-missing": c_bench_missing,
        "util-ceiling": c_util, "parse": c_parse, "met-verdict": c_met_verdict,
        "doctor-verdict": c_doctor_verdict, "bundle": c_bundle,
        "mirror": c_mirror, "mirror-cli": c_mirror_cli}

# Each check replaced by one that finds nothing (or accepts everything). The
# named controls must then fail.
MUTATIONS = {
    "check_surface finds nothing": ("check_surface", lambda *a, **k: [],
                                    ["unknown-key", "bad-value", "presence-only", "unknown-lever"]),
    "check_bench_refs finds nothing": ("check_bench_refs", lambda *a, **k: [], ["bench-missing"]),
    "check_util finds nothing": ("check_util", lambda *a, **k: [], ["util-ceiling"]),
    "mirror_verdict never mirrors": ("mirror_verdict", lambda *a, **k: (False, "no"), ["mirror"]),
    "met_verdict accepts everything": ("met_verdict", lambda *a, **k: (True, "accepted"), ["met-verdict"]),
    "doctor_verdict accepts everything": ("doctor_verdict", lambda *a, **k: (True, "ok"), ["doctor-verdict"]),
    "render_argv renders no flag": ("render_argv", lambda r, table: ["met", "serve", r.model], ["render"]),
    "parse accepts anything": ("parse", lambda rel, text: lib.Recipe(
        id="fam/x", path="recipes/fam/x.yaml", text=text, version="2", model="m", container="c",
        runtime=None, min_nodes=1), ["parse"]),
}


# ── through a real `met` ──────────────────────────────────────────────────────


def with_met(met: str) -> dict:
    def refused(r: lib.Recipe) -> list[str]:
        with tempfile.TemporaryDirectory() as home:
            real = json.loads(recipes.dump_manifest(met, home))
            return [f.kind for f in recipes.check_met(met, [r], lib.flag_table(real), home)]

    def m_positive() -> None:
        assert refused(good()) == [], refused(good())

    def m_unknown_flag() -> None:
        assert refused(edited(bogus_key="1")) == ["met-refused"]

    def m_bad_value() -> None:
        assert refused(edited(scheduler="fcfs")) == ["met-refused"]

    def m_cross_flag() -> None:
        # Out of range: clap parses it; only `validate_serve_args` refuses it.
        assert refused(edited(gpu_memory_utilization="1.5")) == ["met-refused"]

    def m_unknown_lever() -> None:
        r = copy.deepcopy(good())
        r.env["METRALE_NOT_A_LEVER"] = "1"
        assert refused(r) == ["met-refused"]

    def m_engine_reader() -> None:
        rs = corpus()
        idx = lib.dumps(lib.index_document(rs, "c" * 40, 0, {}, "fixture"))
        assert recipes.check_doctor(met, idx, 2) == [], "the engine's reader dropped a fixture recipe"
        doc = json.loads(idx)
        doc["files"]["fam/good"] = doc["files"]["fam/good"].replace("container:", "no_container:")
        kinds = [f.kind for f in recipes.check_doctor(met, lib.dumps(doc), 2)]
        assert kinds == ["engine-reader"], kinds

    def m_real_manifest() -> None:
        with tempfile.TemporaryDirectory() as home:
            real = json.loads(recipes.dump_manifest(met, home))
        assert surface(good(), real) == [], "the fixture uses a flag the engine no longer has"
        assert surface(edited(bogus_key="1"), real) == ["unknown-key"]

    return {"met: positive": m_positive, "met: unknown flag": m_unknown_flag, "met: bad value": m_bad_value,
            "met: validate_serve_args": m_cross_flag, "met: unknown lever": m_unknown_lever,
            "met: engine reader": m_engine_reader, "met: real manifest": m_real_manifest}


def run_mutations(controls: dict) -> None:
    for label, (attr, broken, names) in MUTATIONS.items():
        original = getattr(lib, attr)
        setattr(lib, attr, broken)
        try:
            for name in names:
                if name not in controls:
                    continue
                try:
                    controls[name]()
                except AssertionError:
                    RESULTS.append((f"mutation [{label}] fails {name}", True, ""))
                else:
                    RESULTS.append((f"mutation [{label}] fails {name}", False,
                                    "the control still passed against the broken check"))
        finally:
            setattr(lib, attr, original)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--met", help="also run the controls through this `met` binary")
    args = ap.parse_args()
    controls = dict(PURE)
    if args.met:
        controls.update(with_met(args.met))
        MUTATIONS["met_verdict accepts everything"][2].extend(
            ["met: unknown flag", "met: bad value", "met: validate_serve_args", "met: unknown lever"])
        MUTATIONS["render_argv renders no flag"][2].extend(["met: unknown flag", "met: validate_serve_args"])
        MUTATIONS["doctor_verdict accepts everything"][2].append("met: engine reader")
    for name, fn in controls.items():
        case(name)(fn)
    run_mutations(controls)
    for name, ok, why in RESULTS:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f" -- {why}"))
    failed = [r for r in RESULTS if not r[1]]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed"
          + ("" if args.met else " (no --met: the controls through a real binary did not run)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
