# SPDX-License-Identifier: MIT OR Apache-2.0

"""2026-10-07: Emit four bounded post-soak request fixtures; performs no HTTP calls."""

import hashlib
import json
from pathlib import Path
import sys

from corpus import CORPUS

if len(sys.argv) != 2:
    raise SystemExit("usage: prepare_requests.py NEW_OUTPUT_JSON")
output = Path(sys.argv[1])
rows = []
for task in CORPUS:
    prompt = task["requirements"]
    if "starter" in task:
        prompt += "\n\nCurrent file:\n```python\n" + task["starter"] + "```"
    request = {
        "model": "poolside/Laguna-XS-2.1-NVFP4",
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    rows.append(
        {
            "id": task["id"],
            "role": task["role"],
            "request": request,
            "request_sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest(),
        }
    )
with output.open("x") as handle:
    json.dump(
        {
            "model_revision": "d32afde8b09af1539b49ff96ff5551c674485f8e",
            "corpus_sha256": hashlib.sha256(
                (Path(__file__).parent / "corpus.py").read_bytes()
            ).hexdigest(),
            "model_requests_per_repeat": 4,
            "executed_model_requests": 0,
            "requests": rows,
        },
        handle,
        indent=2,
    )
    handle.write("\n")
print(
    json.dumps(
        {"requests_prepared": len(rows), "requests_sent": 0, "output": str(output)}
    )
)
