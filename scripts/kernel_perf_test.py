#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-test of scripts/kernel_perf.py on a fixture tree built in a temporary directory.

Every positive control is paired with a negative one, and each mutation of a
committed input must turn `--check` red; a check that stays green under its
mutation is not testing anything.

    python3 scripts/kernel_perf_test.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import kernel_perf_scan as scan  # noqa: E402

CLI = HERE / "kernel_perf.py"
GENERATED = "<!-- kernel_perf.py: BEGIN GENERATED (edit the inputs, then run scripts/kernel_perf.py) -->\n<!-- kernel_perf.py: END GENERATED -->\n"

FILES = {
    "kernels/gb10/HARDWARE.toml": '[hardware]\nname = "gb10"\nvendor = "nvidia"\narch = "sm_121f"\n',
    "kernels/hopper/HARDWARE.toml": '[hardware]\nname = "hopper"\nvendor = "nvidia"\narch = "sm_90a"\ninherits = "gb10"\n',
    "kernels/gb10/common/KERNEL.toml": '[modules]\nnorm_ops = "norm"\n',
    "kernels/gb10/common/norm_ops.cu": """
        #include "body.cuh"
        extern "C" __global__ void rms_norm(float* x) {}
        // extern "C" __global__ void commented_out(float* x) {}
        static __global__ void internal_helper(float* x) {}
        template <int N> __global__ void templ_kernel(float* x) {}
        #define WY(K) extern "C" __global__ void wy##K(float* h) {}
        WY(2)
        WY(3)
        """,
    "kernels/gb10/common/body.cuh": """
        #ifndef ENTRY
        #define ENTRY attn_body
        #endif
        #define CAT_(a, b) a##b
        #define CAT(a, b) CAT_(a, b)
        extern "C" __global__ void ENTRY(float* o) {}
        extern "C" __global__ void CAT(ENTRY, _64)(float* o) {}
        """,
    "kernels/gb10/common/mamba_scan.cu": """
        extern "C" __global__ __launch_bounds__(128) void mamba2_scan(float* s) {}
        extern "C" __global__ void mamba2_state(float* s) {}
        """,
    "kernels/gb10/dense-a/MODEL.toml": '[model]\nname = "dense-a"\nhf_id = "org/Dense-A"\n',
    "kernels/gb10/dense-a/nvfp4/KERNEL.toml": "",
    "kernels/gb10/ssm-b/MODEL.toml": '[model]\nname = "ssm-b"\nhf_id = "org/SSM-B"\n',
    "kernels/gb10/ssm-b/nvfp4/KERNEL.toml": "",
    "kernels/hopper/ssm-b/MODEL.toml": '[model]\nname = "ssm-b"\nhf_id = "org/SSM-B"\n',
    "crates/layers/src/attn.rs": """
        const NORM: &str = "norm";
        fn load(g: &G) { g.kernel(NORM, "rms_norm"); g.kernel("norm_ops", "attn_body");
                         g.kernel("mamba_scan", "mamba2_state"); }
        #[cfg(test)]
        mod tests {
            fn t(g: &G) { g.kernel("norm", "wy2"); }
        }
        """,
    "crates/layers/src/mamba/scan.rs": 'fn l(g: &G) { g.kernel("mamba_scan", "mamba2_scan"); g.kernel("norm", "wy3"); }\n',
    "crates/layers/examples/bench.rs": 'fn b(g: &G) { g.kernel("norm_ops", "attn_body_64"); }\n',
    "docs/kernel-perf/taxonomy.toml": """
        [[family]]
        id = "dense"
        label = "Dense"
        name = "Dense attention"
        models = ["dense-a"]
        components = []
        shared_engine = true

        [[family]]
        id = "ssm"
        label = "SSM"
        name = "Mamba hybrid"
        models = ["ssm-b"]
        components = ["mamba2"]
        shared_engine = true

        [[component]]
        id = "attention"
        name = "Attention"
        generic = true
        sites = ["crates/layers/src/attn.rs"]

        [[component]]
        id = "mamba2"
        name = "Mamba2"
        generic = false
        sites = ["crates/layers/src/mamba/"]

        [[component]]
        id = "norm"
        name = "Norm"
        generic = true
        sites = []

        [[rule]]
        component = "mamba2"
        kind = "scan"
        names = ["mamba2_*", "wy*"]

        [[rule]]
        component = "attention"
        kind = "prefill"
        names = ["attn_*"]

        [[rule]]
        component = "norm"
        kind = "rms"
        files = ["*/norm_ops.cu"]
        """,
    "docs/kernel-perf/tradeoffs.toml": """
        [[t]]
        match = "kernels/gb10/common/norm_ops.cu::rms_norm"
        text = "FP32 accumulation; one CTA per row."
        prs = [7]
        source = "fixture"
        """,
    "docs/kernel-perf/measurements.toml": """
        [[m]]
        kernel = "norm::rms_norm"
        file = "kernels/gb10/common/norm_ops.cu"
        hardware = "gb10"
        model = "org/Dense-A"
        regime = "decode C=1"
        time_us = 10.0
        bytes = 1000
        flops = 0
        bound = "memory"
        floor_us = 4.0
        pct_of_floor = 40.0
        source = "fixture"
        notes = ""
        """,
    "KERNEL-PERF.md": "# Fixture\n\n" + GENERATED,
}


def write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n"))


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), "--root", str(root), *args], capture_output=True, text=True)


class ScanTest(unittest.TestCase):
    def test_entry_points_macros_and_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_tree(root, FILES)
            src = root / "kernels/gb10/common/norm_ops.cu"
            hdr = root / "kernels/gb10/common/body.cuh"
            vis, hidden = scan.cuda_entries(
                str(src), src.read_text(), lambda _f, inc: (str(hdr), hdr.read_text()) if inc == "body.cuh" else None)
        names = {e.name for e in vis}
        self.assertEqual(names, {"rms_norm", "wy2", "wy3", "attn_body", "attn_body_64"})
        self.assertEqual({e.name for e in hidden}, {"internal_helper", "templ_kernel"})
        self.assertNotIn("commented_out", names)

    def test_includer_define_wins_over_header_default(self):
        body = '#ifndef ENTRY\n#define ENTRY attn_body\n#endif\nextern "C" __global__ void ENTRY(float* o) {}\n'
        src = '#define ENTRY attn_custom\n#include "body.cuh"\n'
        vis, _ = scan.cuda_entries("x.cu", src, lambda _f, inc: ("body.cuh", body))
        self.assertEqual([e.name for e in vis], ["attn_custom"])


class GeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_tree(self.root, FILES)
        out = run(self.root)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.doc = (self.root / "KERNEL-PERF.md").read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self) -> dict[str, dict]:
        import json
        out = run(self.root, "--json")
        self.assertEqual(out.returncode, 0, out.stderr)
        return {r["kernel"].split("::")[1] + "@" + r["file"]: r for r in json.loads(out.stdout)}

    def test_check_is_green_after_generate(self):
        out = run(self.root, "--check")
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_shared_file_is_one_row_with_every_user(self):
        rows = self.rows()
        r = rows["rms_norm@kernels/gb10/common/norm_ops.cu"]
        self.assertEqual(r["targets"], ["gb10/dense-a/nvfp4", "gb10/ssm-b/nvfp4", "hopper/ssm-b/nvfp4"])
        self.assertEqual(r["modules"], ["norm"])
        self.assertEqual(sum(1 for k in rows if k.startswith("rms_norm@")), 1)

    def test_header_body_row_names_the_header_and_its_compiled_source(self):
        r = self.rows()["attn_body@kernels/gb10/common/body.cuh"]
        self.assertEqual(r["compiled_from"], ["kernels/gb10/common/norm_ops.cu"])

    def test_family_attribution(self):
        rows = self.rows()
        f = lambda n, file="kernels/gb10/common/norm_ops.cu": rows[f"{n}@{file}"]["families"]  # noqa: E731
        self.assertEqual(f("rms_norm"), ["dense", "ssm"])            # generic kernel, generic site
        self.assertEqual(f("mamba2_scan", "kernels/gb10/common/mamba_scan.cu"), ["ssm"])  # component filter
        self.assertEqual(f("wy3"), ["ssm"])                          # site component filter
        self.assertEqual(f("mamba2_state", "kernels/gb10/common/mamba_scan.cu"), ["ssm"])  # primary filter at a generic site
        self.assertEqual(f("wy2"), [])                               # inline test module only
        self.assertEqual(f("attn_body_64", "kernels/gb10/common/body.cuh"), [])  # examples only
        self.assertIn("attn_body_64", self.doc.split("## Compiled but not launched")[1])

    def test_measured_and_not_measured(self):
        row = next(line for line in self.doc.splitlines() if "`rms_norm`" in line and line.startswith("| norm::"))
        self.assertIn("40%", row)
        row = next(line for line in self.doc.splitlines() if "mamba2_scan" in line and line.startswith("|"))
        self.assertIn("not measured", row)

    def assert_mutation_fails(self, rel: str, text: str, expect: str):
        write_tree(self.root, {rel: text})
        out = run(self.root, "--check")
        self.assertNotEqual(out.returncode, 0, f"--check stayed green after mutating {rel}")
        self.assertIn(expect, out.stderr)

    def test_mutation_new_kernel_makes_doc_stale(self):
        self.assert_mutation_fails(
            "kernels/gb10/common/mamba_scan.cu",
            'extern "C" __global__ void mamba2_scan(float* s) {}\nextern "C" __global__ void mamba2_state(float* s) {}\n'
            'extern "C" __global__ void mamba2_new(float* s) {}\n',
            "KERNEL-PERF.md is stale")

    def test_mutation_unclassified_kernel_is_refused(self):
        self.assert_mutation_fails(
            "kernels/gb10/common/mamba_scan.cu",
            'extern "C" __global__ void mamba2_scan(float* s) {}\nextern "C" __global__ void mamba2_state(float* s) {}\n'
            'extern "C" __global__ void mystery(float* s) {}\n',
            "no rule classifies")

    def test_mutation_stale_tradeoff_is_refused(self):
        self.assert_mutation_fails(
            "docs/kernel-perf/tradeoffs.toml",
            '[[t]]\nmatch = "kernels/gb10/common/gone.cu"\ntext = "x"\nprs = []\nsource = "s"\n',
            "names no kernel file")

    def test_mutation_inconsistent_pct_is_refused(self):
        text = FILES["docs/kernel-perf/measurements.toml"].replace("pct_of_floor = 40.0", "pct_of_floor = 55.0")
        self.assert_mutation_fails("docs/kernel-perf/measurements.toml", text, "pct_of_floor 55.0")

    def test_mutation_unknown_measured_kernel_is_refused(self):
        text = FILES["docs/kernel-perf/measurements.toml"].replace('"norm::rms_norm"', '"norm::rms_gone"')
        self.assert_mutation_fails("docs/kernel-perf/measurements.toml", text, "no kernel rms_gone")

    def test_mutation_hand_edit_of_generated_block_is_caught(self):
        self.assert_mutation_fails("KERNEL-PERF.md", self.doc.replace("not measured", "99%", 1), "stale")

    def test_mutation_orphan_model_directory_is_refused(self):
        write_tree(self.root, {"kernels/gb10/new-model/nvfp4/KERNEL.toml": ""})
        self.assert_mutation_fails("kernels/gb10/new-model/MODEL.toml", '[model]\nname = "new-model"\n',
                                   "belongs to no family")


if __name__ == "__main__":
    unittest.main(verbosity=1)
