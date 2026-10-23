#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The optional energy group of a measurements.toml [[m]] row (DESIGN-kernel-energy.md §4.1).

Pure: `check(row, where, roofline)` returns the problems it finds, `roofline` being the
row's hardware [roofline] table from kernels/<hw>/common/KERNEL_FAMILIES.toml (or None).
Physical quantities are floats (J, W, ms); BU64 governs byte sizes, not physics.

Rules: the REQUIRED keys are all-or-none; positivity; every CI brackets its estimate;
the units identity energy_j * energy_launches / (energy_window_ms / 1000) == avg_w within
UNITS_TOL; energy_by_source[power_source] == energy_j; the label is exactly the n/boxes
rule's output; sm_mhz >= SM_MHZ_FLOOR; a sensor delta beyond SENSOR_FLAG_PCT carries its
flag; energy_floor_j / pct_of_energy_floor are re-derived from the roofline.
"""
from __future__ import annotations

import re

NUM = (int, float)
UNITS_TOL = 0.05
SM_MHZ_FLOOR = 2000
SENSOR_FLAG_PCT = 10.0
FLAG_PREFIX = "sensor_delta>10%"
PCT_TOL = 0.5
SOURCES = {"nvml_gpu_rail_counter", "nvml_gpu_rail_sensor", "nvidia_smi_power_draw_average",
           "hwmon", "amdsmi", "apple_ioreport", "external_meter"}
RAILS = {"gpu_core", "gpu_memory", "board", "module", "soc", "cpu_package", "dram", "platform", "wall"}
METHODS = {"steady_state", "in_serve_regression"}
LABELS = {"preliminary", "measured", "certified"}
# energy_floor_class -> the [roofline] key of its pJ/FLOP (None: no FLOP term).
FLOOR_CLASSES = {"dram_only": None, "cuda_core_fp32": "cuda_core_fp32_pj_per_flop",
                 "bf16_mma": "bf16_mma_pj_per_flop", "fp8_mma": "fp8_mma_pj_per_flop",
                 "nvfp4_mma": "nvfp4_mma_pj_per_flop"}
PAIR = object()  # sentinel type: a [lo, hi] list of two numbers
REQUIRED = {
    "energy_j": NUM, "energy_dyn_j": NUM, "avg_w": NUM, "static_w": NUM,
    "energy_ci_j": PAIR, "energy_dyn_ci_j": PAIR, "avg_w_ci": PAIR,
    "energy_ci_level": NUM, "energy_ci_method": str, "energy_method": str,
    "energy_samples": int, "energy_boxes": int, "energy_launches": int, "energy_window_ms": NUM,
    "energy_ess": NUM, "energy_outliers": int, "energy_seed": int, "energy_order": str,
    "energy_label": str, "sm_mhz": NUM, "temp_c": NUM, "throttle_pct": NUM,
    "power_source": str, "power_rail": str,
    "energy_by_source": dict, "energy_dyn_by_source": dict, "sensor_delta_pct": dict,
    "energy_flags": list, "energy_agreement": dict,
    "energy_fingerprint": str, "energy_source": str, "trace_sha256": str,
}
DERIVED = {"energy_floor_class": str, "energy_floor_j": NUM, "pct_of_energy_floor": NUM}
KEYS = {**REQUIRED, **DERIVED}
AGREEMENT_KEYS = {"deming_slope": NUM, "deming_intercept_w": NUM, "ba_bias_w": NUM, "ba_loa_w": PAIR, "n": int}
SHA = re.compile(r"[0-9a-f]{64}")


def label_for(samples: int, boxes: int) -> str:
    """Owner decision 4: n >= 3 on two boxes certified, on one box measured, else preliminary."""
    if samples >= 3 and boxes >= 2:
        return "certified"
    if samples >= 3 and boxes == 1:
        return "measured"
    return "preliminary"


def floor_j(row: dict, roofline: dict) -> float | None:
    """(bytes * dram_pj_per_byte + flops * <class>_pj_per_flop) * 1e-12, or None when the
    roofline lacks a constant the class needs."""
    per_flop_key = FLOOR_CLASSES[row["energy_floor_class"]]
    e_b = roofline.get("dram_pj_per_byte")
    e_f = 0.0 if per_flop_key is None else roofline.get(per_flop_key)
    if not isinstance(e_b, NUM) or not isinstance(e_f, NUM):
        return None
    return (row["bytes"] * e_b + row["flops"] * e_f) * 1e-12


def _pair(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(isinstance(x, NUM) and not isinstance(x, bool) for x in v)


def _typed(v, ty) -> bool:
    if ty is PAIR:
        return _pair(v)
    if isinstance(v, bool):
        return False
    return isinstance(v, ty)


def check(row: dict, where: str, roofline: dict | None) -> list[str]:
    present = [k for k in KEYS if k in row]
    if not present:
        return []
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        return [f"{where}: energy group is all-or-none; missing {', '.join(missing)}"]
    bad = [k for k in present if not _typed(row[k], KEYS[k])]
    if bad:
        return [f"{where}: energy keys mistyped: {', '.join(bad)}"]
    p: list[str] = []
    e, dyn, w, s = row["energy_j"], row["energy_dyn_j"], row["avg_w"], row["static_w"]
    if not (e > 0 and 0 < dyn <= e and w > 0 and s > 0 and w > s):
        p.append(f"{where}: need energy_j > 0, 0 < energy_dyn_j <= energy_j, avg_w > static_w > 0")
    for ci, est in (("energy_ci_j", e), ("energy_dyn_ci_j", dyn), ("avg_w_ci", w)):
        lo, hi = row[ci]
        if not lo <= est <= hi:
            p.append(f"{where}: {ci} [{lo}, {hi}] does not bracket its estimate {est}")
    if not 0 < row["energy_ci_level"] < 1:
        p.append(f"{where}: energy_ci_level must lie in (0, 1)")
    if row["energy_method"] not in METHODS:
        p.append(f"{where}: energy_method must be one of {sorted(METHODS)}")
    n, boxes = row["energy_samples"], row["energy_boxes"]
    if n < 1 or boxes < 1 or row["energy_launches"] < 1 or row["energy_window_ms"] <= 0:
        p.append(f"{where}: energy_samples, energy_boxes, energy_launches and energy_window_ms must be positive")
    elif e > 0:
        implied = e * row["energy_launches"] / (row["energy_window_ms"] / 1000.0)
        if abs(implied - w) > UNITS_TOL * w:
            p.append(f"{where}: units: energy_j * energy_launches / window_s = {implied:.4g} W, "
                     f"not avg_w {w} (tolerance {UNITS_TOL:.0%})")
    want = label_for(n, boxes)
    if row["energy_label"] != want:
        p.append(f"{where}: energy_label {row['energy_label']!r} but n={n} on {boxes} box(es) is {want!r}")
    if row["sm_mhz"] < SM_MHZ_FLOOR:
        p.append(f"{where}: sm_mhz {row['sm_mhz']} is below the {SM_MHZ_FLOOR} MHz clock floor")
    if row["power_source"] not in SOURCES:
        p.append(f"{where}: power_source {row['power_source']!r} is not a known source")
    if row["power_rail"] not in RAILS:
        p.append(f"{where}: power_rail {row['power_rail']!r} is not a known rail")
    for table in ("energy_by_source", "energy_dyn_by_source"):
        unknown = [k for k in row[table] if k not in SOURCES]
        if unknown or not all(_typed(v, NUM) for v in row[table].values()):
            p.append(f"{where}: {table} must map known sources to numbers (bad: {unknown})")
    if row["energy_by_source"].get(row["power_source"]) != e:
        p.append(f"{where}: energy_by_source[{row['power_source']}] must equal energy_j {e}")
    if row["energy_dyn_by_source"].get(row["power_source"]) != dyn:
        p.append(f"{where}: energy_dyn_by_source[{row['power_source']}] must equal energy_dyn_j {dyn}")
    if not all(isinstance(f, str) for f in row["energy_flags"]):
        p.append(f"{where}: energy_flags must be strings")
    for pair, pct in row["sensor_delta_pct"].items():
        if not _typed(pct, NUM):
            p.append(f"{where}: sensor_delta_pct[{pair}] must be a number")
        elif abs(pct) > SENSOR_FLAG_PCT and not any(
                isinstance(f, str) and f.startswith(FLAG_PREFIX) for f in row["energy_flags"]):
            p.append(f"{where}: sensor_delta_pct[{pair}] = {pct} exceeds {SENSOR_FLAG_PCT}% "
                     f"without a {FLAG_PREFIX!r} flag")
    for pair, a in row["energy_agreement"].items():
        if not isinstance(a, dict) or set(a) != set(AGREEMENT_KEYS) or not all(
                _typed(a[k], t) for k, t in AGREEMENT_KEYS.items()):
            p.append(f"{where}: energy_agreement[{pair}] must hold exactly {sorted(AGREEMENT_KEYS)}")
    if not SHA.fullmatch(row["trace_sha256"]):
        p.append(f"{where}: trace_sha256 must be 64 lowercase hex characters")
    p += _floor(row, where, roofline)
    return p


def _floor(row: dict, where: str, roofline: dict | None) -> list[str]:
    cls = row.get("energy_floor_class")
    has = [k for k in ("energy_floor_j", "pct_of_energy_floor") if k in row]
    if cls is None:
        return [f"{where}: {', '.join(has)} without energy_floor_class"] if has else []
    if cls not in FLOOR_CLASSES:
        return [f"{where}: energy_floor_class {cls!r} is not one of {sorted(FLOOR_CLASSES)}"]
    derived = floor_j(row, roofline) if roofline is not None else None
    if derived is None:
        return [f"{where}: energy_floor_class {cls!r} needs [roofline] energy constants this hardware lacks"] \
            if has else []
    if len(has) != 2:
        return [f"{where}: energy_floor_j and pct_of_energy_floor go together"]
    p = []
    if abs(row["energy_floor_j"] - derived) > 1e-6 * max(derived, 1e-30) + 1e-15:
        p.append(f"{where}: energy_floor_j {row['energy_floor_j']} != bytes/flops x roofline = {derived:.6g}")
    pct = derived / row["energy_dyn_j"] * 100.0
    if abs(pct - row["pct_of_energy_floor"]) > PCT_TOL:
        p.append(f"{where}: pct_of_energy_floor {row['pct_of_energy_floor']} != energy_floor_j/energy_dyn_j*100 = {pct:.2f}")
    return p


def cells(row: dict) -> list[str]:
    """The MEASUREMENTS.md cells `mJ | W | % E-floor`: blank when absent, never 0."""
    if "energy_j" not in row:
        return ["", "", ""]
    pct = row.get("pct_of_energy_floor")
    return [f"{row['energy_j'] * 1e3:.3f}", f"{row['avg_w']:.1f}", "" if pct is None else f"{pct:.1f}%"]
