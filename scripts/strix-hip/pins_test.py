#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Keep the strix-hip setup doc and the setup script in step.

`docs/porting/strix-hip-setup.md` carries the validated version table;
`scripts/strix-hip/wsl-setup.sh` enforces the rows marked "pinned". This test
fails when they disagree, when a "pinned" row has no check here, or when the
doc's sample preflight JSON contradicts the pins. It also runs itself against
a deliberately drifted copy of the doc and requires that copy to fail, so a
parser that silently matches nothing cannot pass.

Usage: python3 scripts/strix-hip/pins_test.py   (from anywhere; no deps)
"""
import json
import pathlib
import re
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/porting/strix-hip-setup.md"
SETUP = ROOT / "scripts/strix-hip/wsl-setup.sh"
TOOLCHAIN = ROOT / "rust-toolchain.toml"


def script_pins(text: str) -> dict[str, str]:
    """Top-level `NAME="value"` assignments of wsl-setup.sh."""
    pins = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(\S.*)$", line)
        if m and not m.group(2).startswith("("):
            pins[m.group(1)] = shlex.split(m.group(2))[0]
    return pins


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


def sample_json(doc: str) -> dict:
    m = re.search(r"## Sample preflight output.*?```json\n(.*?)\n```", doc, re.S)
    if not m:
        raise SystemExit("doc has no sample preflight JSON block")
    return json.loads(m.group(1))


def check(doc: str, setup: str, toolchain: str) -> list[str]:
    pins = script_pins(setup)
    rows = version_table(doc)
    errs = []

    def want(component: str, expected: str):
        got = rows.get(component, (None, None))[0]
        if got != expected:
            errs.append(f"doc row '{component}' is {got!r}, script pins {expected!r}")

    channel = re.search(r'^channel\s*=\s*"([^"]+)"', toolchain, re.M)
    ver = pins.get("ROCDXG_VERSION", "")
    expected = {
        "Ubuntu": pins.get("UBUNTU_VERSION_ID", ""),
        "ROCm userspace": pins.get("ROCM_VERSION", ""),
        "amdgpu-install package file": pins.get("AMDGPU_INSTALL_URL", "").rsplit("/", 1)[-1],
        "amdgpu-install dpkg version": pins.get("AMDGPU_INSTALL_PKG_VERSION", ""),
        "amdgpu-install sha256": pins.get("AMDGPU_INSTALL_SHA256", ""),
        "librocdxg (rocdxg-roct)": ver,
        "librocdxg sha256": pins.get("ROCDXG_SHA256", ""),
        "rustc": channel.group(1) if channel else "",
    }
    for component, value in expected.items():
        if not value:
            errs.append(f"no script/toolchain pin found for '{component}'")
        want(component, value)

    for component, (_, enforced) in rows.items():
        if enforced.startswith("pinned") and component not in expected:
            errs.append(f"doc row '{component}' says pinned but pins_test.py does not check it")

    # The URLs must name the pinned versions, not just sit next to them.
    if pins.get("AMDGPU_INSTALL_VERSION", "?") not in pins.get("AMDGPU_INSTALL_URL", ""):
        errs.append("AMDGPU_INSTALL_URL does not contain AMDGPU_INSTALL_VERSION")
    if f"/v{ver}/rocdxg-roct_{ver}_amd64.deb" not in pins.get("ROCDXG_URL", ""):
        errs.append("ROCDXG_URL does not name ROCDXG_VERSION")
    hipcc = rows.get("hipcc", ("", ""))[0]
    if f"roc-{pins.get('ROCM_VERSION', '?')}" not in hipcc:
        errs.append(f"doc hipcc row {hipcc!r} is not roc-{pins.get('ROCM_VERSION')}")

    sample = sample_json(doc)
    rocm = sample.get("rocm", {})
    for key, pin in (("version", "ROCM_VERSION"), ("rocdxg_roct", "ROCDXG_VERSION"),
                     ("amdgpu_install", "AMDGPU_INSTALL_PKG_VERSION")):
        if rocm.get(key) != pins.get(pin):
            errs.append(f"sample preflight rocm.{key}={rocm.get(key)!r} but {pin}={pins.get(pin)!r}")
    if sample.get("rocminfo", {}).get("gfx_name") != "gfx1151" or sample.get("ok") is not True:
        errs.append("sample preflight is not an ok gfx1151 run")
    return errs


def main() -> int:
    doc, setup, toolchain = DOC.read_text(), SETUP.read_text(), TOOLCHAIN.read_text()

    # Known-bad control: drift the ROCm row and the librocdxg sha; both must be caught.
    bad = doc.replace("| ROCm userspace | 7.2.1 |", "| ROCm userspace | 7.2.2 |", 1)
    bad = re.sub(r"(\| librocdxg sha256 \| )[0-9a-f]{64}", r"\g<1>" + "0" * 64, bad, count=1)
    bad_errs = check(bad, setup, toolchain)
    if bad == doc or len([e for e in bad_errs if "ROCm userspace" in e or "librocdxg sha256" in e]) != 2:
        print("FAIL: drifted control doc was not rejected on both rows:", bad_errs)
        return 1

    errs = check(doc, setup, toolchain)
    for e in errs:
        print(f"FAIL: {e}")
    if errs:
        return 1
    print("strix-hip pins: doc table, sample preflight and wsl-setup.sh agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
