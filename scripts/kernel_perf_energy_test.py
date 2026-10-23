#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Self-test of the measurements.toml energy group (scripts/lib/kernel_perf_energy.py) on the
kernel_perf_test fixture tree plus one energy row and a gb10 KERNEL_FAMILIES.toml [roofline].

Every negative test mutates one thing of a row that passes and must turn `--check` red with
its own message.

    python3 scripts/kernel_perf_energy_test.py
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kernel_perf_test import FILES, run, write_tree

MEAS = "docs/kernel-perf/measurements.toml"
ROOFLINE = """
    [roofline]
    dram_gbps = 249.0
    bf16_tflops = 123.7
    fp8_tflops = 243.6
    nvfp4_tflops = 490.8
    context_tokens = 4096
    dram_pj_per_byte = 23.2
    cuda_core_fp32_pj_per_flop = 1.7
    static_w = 12.5
    energy_label = "preliminary"
    energy_source = "fixture"
    """
# 2026-10-22: 0.08 J x 3000 launches / 13.5 s = 17.78 W (the units identity); floor dram_only
# = 1e9 B x 23.2 pJ/B = 0.0232 J = 92.8 % of the 0.025 J dynamic.
ENERGY_ROW = """
    [[m]]
    kernel = "norm::rms_norm"
    file = "kernels/gb10/common/norm_ops.cu"
    hardware = "gb10"
    model = "org/Dense-A"
    regime = "steady-state microbench R=1"
    time_us = 10.0
    bytes = 1000000000
    flops = 0
    bound = "memory"
    floor_us = 4.0
    pct_of_floor = 40.0
    source = "fixture"
    notes = ""
    energy_j = 0.08
    energy_dyn_j = 0.025
    avg_w = 17.78
    static_w = 12.4
    energy_ci_j = [0.079, 0.081]
    energy_dyn_ci_j = [0.023, 0.027]
    avg_w_ci = [17.6, 17.9]
    energy_ci_level = 0.95
    energy_ci_method = "varcomp-satterthwaite"
    energy_method = "steady_state"
    energy_samples = 3
    energy_boxes = 1
    energy_launches = 3000
    energy_window_ms = 13500.0
    energy_ess = 41.5
    energy_outliers = 0
    energy_seed = 42
    energy_order = "ABBA"
    energy_label = "measured"
    sm_mhz = 2435.0
    temp_c = 52.0
    throttle_pct = 0.0
    power_source = "nvml_gpu_rail_counter"
    power_rail = "gpu_core"
    energy_by_source = { nvml_gpu_rail_counter = 0.08, nvml_gpu_rail_sensor = 0.0832 }
    energy_dyn_by_source = { nvml_gpu_rail_counter = 0.025, nvml_gpu_rail_sensor = 0.026 }
    sensor_delta_pct = { "nvml_gpu_rail_counter~nvml_gpu_rail_sensor" = 4.0 }
    energy_flags = []
    energy_agreement = { "nvml_gpu_rail_counter~nvml_gpu_rail_sensor" = { deming_slope = 1.04, deming_intercept_w = 0.1, ba_bias_w = 0.7, ba_loa_w = [0.2, 1.2], n = 120 } }
    energy_fingerprint = "GB10 fixture"
    energy_source = "fixture"
    trace_sha256 = "1b74b8af243fd0f94f6adbb2474f9e9cfbff2d0a1d1f0ef3c511993f8ff8d3aa"
    energy_floor_class = "dram_only"
    energy_floor_j = 0.0232
    pct_of_energy_floor = 92.8
    """


class EnergyGroupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = dict(FILES)
        files[MEAS] = FILES[MEAS] + ENERGY_ROW
        files["kernels/gb10/common/KERNEL_FAMILIES.toml"] = ROOFLINE
        write_tree(self.root, files)
        out = run(self.root)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.meas = (self.root / MEAS).read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def mutate(self, old: str, new: str, expect: str):
        self.assertIn(old, self.meas, "the mutation must change the passing row")
        (self.root / MEAS).write_text(self.meas.replace(old, new))
        out = run(self.root, "--check")
        self.assertNotEqual(out.returncode, 0, f"--check stayed green after {old!r} -> {new!r}")
        self.assertIn(expect, out.stderr)

    def test_a_good_energy_row_passes_and_renders_its_cells(self):
        out = run(self.root, "--check")
        self.assertEqual(out.returncode, 0, out.stderr)
        md = (self.root / "docs/kernel-perf/MEASUREMENTS.md").read_text()
        self.assertIn("| 80.000 | 17.8 | 92.8% |", md)
        # 2026-10-22: the row without energy renders blank cells, never 0.
        self.assertIn("| 40.0% |  |  |  |", md)

    def test_half_a_group_names_the_missing_keys(self):
        self.mutate("energy_dyn_j = 0.025\n", "", "missing energy_dyn_j")

    def test_wrong_pct_of_energy_floor(self):
        self.mutate("pct_of_energy_floor = 92.8", "pct_of_energy_floor = 50.0", "pct_of_energy_floor 50.0")

    def test_wrong_energy_floor_j(self):
        self.mutate("energy_floor_j = 0.0232", "energy_floor_j = 0.03", "energy_floor_j 0.03")

    def test_floor_without_class(self):
        self.mutate('energy_floor_class = "dram_only"\n', "", "without energy_floor_class")

    def test_clamped_clock(self):
        self.mutate("sm_mhz = 2435.0", "sm_mhz = 513.0", "below the 2000 MHz clock floor")

    def test_measured_with_two_repeats(self):
        self.mutate("energy_samples = 3", "energy_samples = 2", "is 'preliminary'")

    def test_preliminary_with_three_repeats_on_one_box(self):
        self.mutate('energy_label = "measured"', 'energy_label = "preliminary"', "is 'measured'")

    def test_certified_needs_two_boxes(self):
        self.mutate('energy_label = "measured"', 'energy_label = "certified"', "is 'measured'")

    def test_units_identity_off_by_ten_percent(self):
        self.mutate("avg_w = 17.78", "avg_w = 19.56", "units:")

    def test_headline_source_mismatch(self):
        self.mutate("nvml_gpu_rail_counter = 0.08,", "nvml_gpu_rail_counter = 0.081,",
                    "energy_by_source[nvml_gpu_rail_counter] must equal energy_j")

    def test_sensor_delta_beyond_ten_percent_needs_its_flag(self):
        self.mutate('"nvml_gpu_rail_counter~nvml_gpu_rail_sensor" = 4.0', '"nvml_gpu_rail_counter~nvml_gpu_rail_sensor" = 31.0',
                    "without a 'sensor_delta>10%' flag")

    def test_sensor_delta_with_its_flag_passes(self):
        text = self.meas.replace('"nvml_gpu_rail_counter~nvml_gpu_rail_sensor" = 4.0',
                                 '"nvml_gpu_rail_counter~nvml_gpu_rail_sensor" = 31.0').replace(
            "energy_flags = []", 'energy_flags = ["sensor_delta>10%:nvml_gpu_rail_counter~nvml_gpu_rail_sensor"]')
        (self.root / MEAS).write_text(text)
        out = run(self.root, "--check")
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_ci_that_misses_its_estimate(self):
        self.mutate("energy_ci_j = [0.079, 0.081]", "energy_ci_j = [0.081, 0.082]", "does not bracket")

    def test_dynamic_above_total(self):
        self.mutate("energy_dyn_j = 0.025", "energy_dyn_j = 0.09", "0 < energy_dyn_j <= energy_j")

    def test_unknown_source(self):
        self.mutate('power_source = "nvml_gpu_rail_counter"', 'power_source = "rapl"', "not a known source")

    def test_bad_trace_hash(self):
        self.mutate('trace_sha256 = "1b74', 'trace_sha256 = "XX74', "64 lowercase hex")


if __name__ == "__main__":
    unittest.main(verbosity=1)
