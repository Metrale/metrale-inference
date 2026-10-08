#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Keep the strix-hip setup doc and the setup scripts in step.

`docs/porting/strix-hip-setup.md` carries the validated version table;
`scripts/strix-hip/wsl-setup.sh` and `scripts/strix-hip/linux-setup.sh`
enforce the rows marked "pinned". This test fails when:
  - a pinned doc row disagrees with the script that enforces it,
  - a "pinned" row has no check here,
  - the ROCm pins the two scripts share have drifted apart,
  - the native kernel parameters in linux-setup.sh, the GTT arithmetic, and
    the copy preflight.sh warns against disagree,
  - either sample preflight JSON in the doc contradicts the pins.
It also runs itself against deliberately drifted copies and requires each to
fail, so a parser that silently matches nothing cannot pass.

Usage: python3 scripts/strix-hip/pins_test.py   (from anywhere; no deps)
"""
import ast
import json
import pathlib
import re
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/porting/strix-hip-setup.md"
WSL_SETUP = ROOT / "scripts/strix-hip/wsl-setup.sh"
LINUX_SETUP = ROOT / "scripts/strix-hip/linux-setup.sh"
PREFLIGHT = ROOT / "scripts/strix-hip/preflight.sh"
TOOLCHAIN = ROOT / "rust-toolchain.toml"

# Pins both setup scripts must carry with identical values.
SHARED = ("UBUNTU_VERSION_ID", "ROCM_VERSION", "AMDGPU_INSTALL_VERSION",
          "AMDGPU_INSTALL_PKG_VERSION", "AMDGPU_INSTALL_URL", "AMDGPU_INSTALL_SHA256")


def script_pins(text: str) -> dict[str, str]:
    """Top-level `NAME="value"` assignments of a setup script."""
    pins = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(\S.*)$", line)
        if m and not m.group(2).startswith("("):
            pins[m.group(1)] = shlex.split(m.group(2))[0]
    return pins


def cmdline_params(pins: dict[str, str]) -> dict[str, str]:
    """linux-setup.sh CMDLINE_PARAMS as {key: value}."""
    return dict(p.split("=", 1) for p in pins.get("CMDLINE_PARAMS", "").split() if "=" in p)


def preflight_native_cmdline(text: str) -> dict:
    m = re.search(r"NATIVE_CMDLINE = (\{.*?\})", text, re.S)
    return ast.literal_eval(m.group(1)) if m else {}


def version_table(doc: str) -> dict[str, tuple[str, str]]:
    """`Component -> (validated version, enforced)` from the version table."""
    sec = doc.split("## Validated versions", 1)
    if len(sec) != 2:
        raise SystemExit("doc has no '## Validated versions' section")
    body = sec[1].split("\n## ", 1)[0]
    rows = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("Component", "") or set(cells[0]) <= {"-", " "}:
            continue
        rows[cells[0]] = (cells[1], cells[2])
    return rows


def sample_json(doc: str, heading: str) -> dict:
    m = re.search(r"## " + re.escape(heading) + r"\n.*?```json\n(.*?)\n```", doc, re.S)
    if not m:
        raise SystemExit(f"doc has no JSON block under '## {heading}'")
    return json.loads(m.group(1))


def check(doc: str, wsl: str, linux: str, preflight: str, toolchain: str) -> list[str]:
    wp, lp = script_pins(wsl), script_pins(linux)
    rows = version_table(doc)
    errs = []

    for name in SHARED:
        if not wp.get(name) or wp.get(name) != lp.get(name):
            errs.append(f"shared pin {name}: wsl-setup.sh {wp.get(name)!r} vs linux-setup.sh {lp.get(name)!r}")

    params = cmdline_params(lp)
    gtt, pages = lp.get("GTT_MIB", ""), lp.get("TTM_PAGES", "")
    if not (gtt.isdigit() and pages.isdigit() and int(gtt) * 256 == int(pages)):
        errs.append(f"linux-setup.sh TTM_PAGES={pages!r} is not GTT_MIB={gtt!r} * 256")
    want_params = {"amd_iommu": "off", "amdgpu.gttsize": gtt,
                   "ttm.pages_limit": pages, "ttm.page_pool_size": pages}
    if params != want_params:
        errs.append(f"linux-setup.sh CMDLINE_PARAMS {params} does not match GTT_MIB/TTM_PAGES {want_params}")
    if preflight_native_cmdline(preflight) != params:
        errs.append(f"preflight.sh NATIVE_CMDLINE {preflight_native_cmdline(preflight)} != linux-setup.sh {params}")

    channel = re.search(r'^channel\s*=\s*"([^"]+)"', toolchain, re.M)
    ver = wp.get("ROCDXG_VERSION", "")
    expected = {
        "Ubuntu": wp.get("UBUNTU_VERSION_ID", ""),
        "ROCm userspace": wp.get("ROCM_VERSION", ""),
        "amdgpu-install package file": wp.get("AMDGPU_INSTALL_URL", "").rsplit("/", 1)[-1],
        "amdgpu-install dpkg version": wp.get("AMDGPU_INSTALL_PKG_VERSION", ""),
        "amdgpu-install sha256": wp.get("AMDGPU_INSTALL_SHA256", ""),
        "WSL: librocdxg (rocdxg-roct)": ver,
        "WSL: librocdxg sha256": wp.get("ROCDXG_SHA256", ""),
        "rustc": channel.group(1) if channel else "",
    }
    for key, value in params.items():
        expected[f"Native: {key}"] = value
    for component, value in expected.items():
        if not value:
            errs.append(f"no script/toolchain pin found for '{component}'")
        got = rows.get(component, (None, None))[0]
        if got != value:
            errs.append(f"doc row '{component}' is {got!r}, script pins {value!r}")

    for component, (_, enforced) in rows.items():
        if enforced.startswith("pinned") and component not in expected:
            errs.append(f"doc row '{component}' says pinned but pins_test.py does not check it")

    # The URLs must name the pinned versions, not just sit next to them.
    for name, p in (("wsl-setup.sh", wp), ("linux-setup.sh", lp)):
        if p.get("AMDGPU_INSTALL_VERSION", "?") not in p.get("AMDGPU_INSTALL_URL", ""):
            errs.append(f"{name}: AMDGPU_INSTALL_URL does not contain AMDGPU_INSTALL_VERSION")
    if f"/v{ver}/rocdxg-roct_{ver}_amd64.deb" not in wp.get("ROCDXG_URL", ""):
        errs.append("ROCDXG_URL does not name ROCDXG_VERSION")
    hipcc = rows.get("hipcc", ("", ""))[0]
    if f"roc-{wp.get('ROCM_VERSION', '?')}" not in hipcc:
        errs.append(f"doc hipcc row {hipcc!r} is not roc-{wp.get('ROCM_VERSION')}")

    samples = {"native": sample_json(doc, "Sample preflight output: native Linux"),
               "wsl": sample_json(doc, "Sample preflight output: Windows + WSL")}
    for kind, sample in samples.items():
        rocm = sample.get("rocm", {})
        checks = [("version", "ROCM_VERSION"), ("amdgpu_install", "AMDGPU_INSTALL_PKG_VERSION")]
        if kind == "wsl":
            checks.append(("rocdxg_roct", "ROCDXG_VERSION"))
        for key, pin in checks:
            if rocm.get(key) != wp.get(pin):
                errs.append(f"{kind} sample rocm.{key}={rocm.get(key)!r} but {pin}={wp.get(pin)!r}")
        if sample.get("rocminfo", {}).get("gfx_name") != "gfx1151" or sample.get("ok") is not True:
            errs.append(f"{kind} sample preflight is not an ok gfx1151 run")
        if sample.get("wsl") is not (kind == "wsl"):
            errs.append(f"{kind} sample has wsl={sample.get('wsl')!r}")

    native = samples["native"]
    nb = native.get("native") or {}
    if native.get("host") != "native" or native.get("warnings") != []:
        errs.append("native sample is not host=native with no warnings")
    if native.get("rocm", {}).get("rocdxg_roct") is not None:
        errs.append("native sample has librocdxg installed")
    if nb.get("cmdline") != params:
        errs.append(f"native sample cmdline {nb.get('cmdline')} != linux-setup.sh {params}")
    gtt_bytes = int(gtt) * 2**20 if gtt.isdigit() else -1
    if (nb.get("amdgpu") or {}).get("gtt_total") != gtt_bytes:
        errs.append(f"native sample gtt_total {(nb.get('amdgpu') or {}).get('gtt_total')} != GTT_MIB * 2^20 = {gtt_bytes}")
    return errs


def main() -> int:
    doc, wsl, linux = DOC.read_text(), WSL_SETUP.read_text(), LINUX_SETUP.read_text()
    preflight, toolchain = PREFLIGHT.read_text(), TOOLCHAIN.read_text()

    # Known-bad controls. Each drifted input must be rejected with the named error.
    bad_doc = doc.replace("| ROCm userspace | 7.2.1 |", "| ROCm userspace | 7.2.2 |", 1)
    bad_doc = re.sub(r"(\| WSL: librocdxg sha256 \| )[0-9a-f]{64}", r"\g<1>" + "0" * 64, bad_doc, count=1)
    bad_doc = bad_doc.replace("| Native: amdgpu.gttsize | 126976 |", "| Native: amdgpu.gttsize | 65536 |", 1)
    controls = [
        ("doc rows", (bad_doc, wsl, linux, preflight, toolchain),
         ("'ROCm userspace'", "'WSL: librocdxg sha256'", "'Native: amdgpu.gttsize'")),
        ("linux-setup.sh ROCm", (doc, wsl, linux.replace('ROCM_VERSION="7.2.1"', 'ROCM_VERSION="7.2.2"', 1),
                                 preflight, toolchain), ("shared pin ROCM_VERSION",)),
        ("linux-setup.sh TTM", (doc, wsl, linux.replace('TTM_PAGES="32505856"', 'TTM_PAGES="32505855"', 1),
                                preflight, toolchain), ("is not GTT_MIB",)),
        ("preflight.sh cmdline", (doc, wsl, linux, preflight.replace('"amd_iommu": "off"', '"amd_iommu": "on"', 1),
                                  toolchain), ("preflight.sh NATIVE_CMDLINE",)),
    ]
    for label, args, needles in controls:
        if args == (doc, wsl, linux, preflight, toolchain):
            print(f"FAIL: known-bad control '{label}' did not change its input")
            return 1
        bad_errs = check(*args)
        missing = [n for n in needles if not any(n in e for e in bad_errs)]
        if missing:
            print(f"FAIL: known-bad control '{label}' not rejected for {missing}: {bad_errs}")
            return 1

    errs = check(doc, wsl, linux, preflight, toolchain)
    for e in errs:
        print(f"FAIL: {e}")
    if errs:
        return 1
    print("strix-hip pins: doc table, both samples, wsl-setup.sh, linux-setup.sh and preflight.sh agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
