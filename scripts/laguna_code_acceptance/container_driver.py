# SPDX-License-Identifier: MIT OR Apache-2.0

"""2026-10-07: Trusted container-only driver; never run this module on the host."""

import copy
import io
import json
import sys

bundle = json.loads(sys.stdin.buffer.read(131073))
output = sys.stdout
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
namespace = {}
results = []
try:
    exec(compile(bundle["source"], "candidate.py", "exec"), namespace)
    function = namespace["solve"]
    for args in bundle["arguments"]:
        original = copy.deepcopy(args)
        try:
            value = function(*args)
            result = {"value": value, "error": None}
        except Exception as error:
            result = {"value": None, "error": type(error).__name__}
        result["mutated"] = args != original
        results.append(result)
    output.write(
        json.dumps(
            {
                "schema": 1,
                "nonce": bundle["nonce"],
                "completed_cases": len(results),
                "results": results,
            },
            allow_nan=False,
        )
        + "\n"
    )
except BaseException as error:
    output.write(json.dumps({"load_error": type(error).__name__}) + "\n")
