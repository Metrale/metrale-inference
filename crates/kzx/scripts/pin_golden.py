#!/usr/bin/env python3
"""Re-pin tests/golden.rs from the current codec output.

Run after an intentional model or format change: `python3 scripts/pin_golden.py`. It runs the
golden tests, reads each failure's reported `len`, `head` and `fnv`, and rewrites the pins.
A format change must also bump the CHANGELOG entry that explains it.
"""
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
test = root / "tests" / "golden.rs"
out = subprocess.run(
    ["cargo", "test", "--release", "--test", "golden", "--", "--test-threads=1"],
    cwd=root, capture_output=True, text=True,
).stdout + subprocess.run(
    ["cargo", "test", "--release", "--test", "golden", "--", "--test-threads=1"],
    cwd=root, capture_output=True, text=True,
).stderr
pat = re.compile(r"golden `(\w+)` changed: len (\d+) head \[([0-9a-f, ]+)\] fnv (0x[0-9a-f]+)")
found = {m.group(1): m for m in pat.finditer(out)}
if not found:
    print("all golden pins already match")
    sys.exit(0)
src = test.read_text()
for name, m in found.items():
    head = ", ".join(f"0x{b.strip()}" for b in m.group(3).split(","))
    block = re.compile(
        r'(name: "' + name + r'",\n\s+len: )\d+(,\n\s+head: \[)[^\]]*(\],\n\s+fnv: )(?:0x[0-9a-fA-F_]+|\d+)',
    )
    fnv = m.group(4)
    fnv = fnv[:2] + "_".join(fnv[2:][i:i + 4] for i in range(0, 16, 4))
    src, n = block.subn(lambda mm: f"{mm.group(1)}{m.group(2)}{mm.group(2)}{head}{mm.group(3)}{fnv}", src)
    print(f"{name}: len {m.group(2)} fnv {fnv} ({n} pin updated)")
test.write_text(src)
