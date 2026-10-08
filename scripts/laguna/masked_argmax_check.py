# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-07: Constructed CUDA gate for argmax_bf16_batch_masked_host, run against
# masked_argmax_check.cu compiled as probe.so. PyTorch is only the input/output oracle;
# no model weights are loaded. Usage: python3 masked_argmax_check.py <directory holding
# probe.so and the exact argmax_feed.cu it was built from>.
import argparse
import ctypes
import hashlib
import json
import random
from pathlib import Path

import torch

ABSENT = 2**32 - 1

parser = argparse.ArgumentParser(description="Constructed masked greedy CUDA gate; no model weights")
parser.add_argument("directory", type=Path)
root = parser.parse_args().directory
lib = ctypes.CDLL(str(root / "probe.so"))
fn = lib.run
fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 3
random.seed(77)
torch.manual_seed(77)


def constructed_row(width, case):
    """2026-10-07: Row `case` of the twelve constructed cases at one vocabulary width."""
    x = torch.randn(width, dtype=torch.bfloat16, device="cuda")
    m = []
    if case == 0:
        x.zero_()
    if case == 1:
        x.fill_(-3e30)
    if case == 2:
        x[-1] = x[0] = 100
    if case == 3:
        m = [int(x.argmax())]
    if case == 4:
        x[0] = float("nan")
    if case == 5:
        x[0] = float("inf")
    if case == 6:
        x[0] = -float("inf")
    if case == 7:
        x[0] = float("nan")
        m = [0]
    if case == 8:
        m = list(range(min(width, 8)))
    if case == 9:
        m = [0, 0, ABSENT, width + 1]
    if case == 10:
        x.zero_()
        x[0] = -0.0
    if case == 11:
        x.fill_(-float("inf"))
    return x, m


rows = []
for width in [1, 2, 7, 255, 256, 257, 100352]:
    for case in range(12):
        x, m = constructed_row(width, case)
        mask = torch.tensor((m + [ABSENT] * 8)[:8], device="cuda", dtype=torch.uint32)
        out = torch.empty(1, device="cuda", dtype=torch.uint32)
        fn(x.data_ptr(), mask.data_ptr(), out.data_ptr(), width, 1, width)
        y = x.float().cpu()
        valid = [i for i in range(width) if i not in m]
        if not valid or not bool(torch.isfinite(y[valid]).all()):
            expected = ABSENT
        else:
            expected = max(valid, key=lambda i: (float(y[i]), i))
        actual = out.item()
        assert actual == expected, (width, case, actual, expected)
        rows.append({"width": width, "case": case, "actual": actual, "expected": expected})

# 2026-10-07: The known-wrong first-tie and unmasked policies must differ.
assert max(range(7), key=lambda i: (0.0, i)) != 0
assert max([0, 2], key=lambda i: ([4, 9, 9][i], i)) != 1

# 2026-10-07: Mixed rows and padded stride prove row-local masking and no padding reads.
x = torch.tensor(
    [[4, 9, 9, 1000], [4, 9, 9, 1000], [float("nan"), 2, 3, 1000]],
    device="cuda",
    dtype=torch.bfloat16,
)
m = torch.tensor(
    [[1] + [ABSENT] * 7, [1, 2] + [ABSENT] * 6, [0] + [ABSENT] * 7],
    device="cuda",
    dtype=torch.uint32,
)
out = torch.empty(3, device="cuda", dtype=torch.uint32)
fn(x.data_ptr(), m.data_ptr(), out.data_ptr(), 3, 3, 4)
assert out.cpu().tolist() == [2, 0, 2]

with open(root / "result.json", "w") as f:
    json.dump(
        {
            "cases": len(rows),
            "rows": rows,
            "mixed": [2, 0, 2],
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "source_sha256": hashlib.sha256((root / "argmax_feed.cu").read_bytes()).hexdigest(),
        },
        f,
        indent=2,
    )
print("PASS", len(rows), "plus mixed stride rows")
