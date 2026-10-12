#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Compile actual shim query bodies against host HIP-call witnesses."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/kernels/hip/libcuda_hip_shim.cpp"
FIXTURE = ROOT / "crates/kernels/tests/hip_device_query/forwarding.cpp"


def body(source, name):
    marker = "int " + name + "("
    if source.count(marker) != 1:
        raise ValueError(f"expected exactly one definition: {name}")
    start = source.index(marker)
    brace = source.index("{", start)
    if ";" in source[start:brace]:
        raise ValueError(f"declaration instead of definition: {name}")
    depth = 1
    end = brace + 1
    while depth:
        if end == len(source):
            raise ValueError("unterminated definition")
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    source = SOURCE.read_text()
    definitions = "\n".join(body(source, name) for name in
                            ("cuDeviceGetName", "cuDeviceGetAttribute"))
    fixture = FIXTURE.read_text()
    assert fixture.count("// PRODUCTION_DEFINITIONS") == 1
    compiler = os.environ.get("CXX", "c++")
    print(subprocess.check_output([compiler, "--version"], text=True).splitlines()[0])
    print("source_sha256=" + hashlib.sha256(source.encode()).hexdigest())
    with tempfile.TemporaryDirectory(prefix="metrale-hip-query-") as tmp:
        tmp = Path(tmp)

        def run(code, name):
            cpp, exe = tmp / (name + ".cpp"), tmp / name
            cpp.write_text(fixture.replace("// PRODUCTION_DEFINITIONS", code))
            argv = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unused-parameter", "-Wno-unused-function",
                    str(cpp), "-o", str(exe)]
            print("compile:", " ".join(argv), flush=True)
            subprocess.run(argv, check=True)
            return subprocess.run([str(exe)], capture_output=True, text=True)

        result = run(definitions, "production")
        print(result.stdout + result.stderr, end="")
        if result.returncode:
            raise SystemExit(result.returncode)
        old = "case 16: mapped = hipDeviceAttributeMultiprocessorCount;"
        assert definitions.count(old) == 1, "wrong-mapping control anchor changed"
        mutant = definitions.replace(old, "case 16: mapped = hipDeviceAttributeWarpSize;")
        wrong = run(mutant, "wrong_mapping")
        assert wrong.returncode != 0 and "last_attribute == hip[i]" in wrong.stderr
        print("PASS wrong-case control rejected:", wrong.stderr.strip())
        name_forward = "return hipDeviceGetName(name, len, device);"
        assert definitions.count(name_forward) == 1
        fixed_name = definitions.replace(name_forward,
            'if (name && len > 0) { std::snprintf(name, len, "AMD-gfx1151"); }\n    return 0;')
        wrong_name = run(fixed_name, "fixed_name")
        assert wrong_name.returncode != 0 and "strcmp(name, expected)" in wrong_name.stderr
        print("PASS fixed-name control rejected:", wrong_name.stderr.strip())
        fixed_count = definitions.replace(old + " break;",
            "case 16: *value = 40; return 0;")
        assert fixed_count != definitions
        wrong_count = run(fixed_count, "fixed_count")
        assert wrong_count.returncode != 0 and "calls == 1" in wrong_count.stderr
        print("PASS fixed-count control rejected:", wrong_count.stderr.strip())
        for malformed in (source + "\n" + body(source, "cuDeviceGetName"), ""):
            try:
                body(malformed, "cuDeviceGetName")
            except ValueError:
                pass
            else:
                raise AssertionError("nonunique/missing production definition accepted")
        print("PASS missing/duplicate extraction controls")


if __name__ == "__main__":
    main()
