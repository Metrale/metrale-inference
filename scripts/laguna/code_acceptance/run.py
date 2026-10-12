# SPDX-License-Identifier: MIT OR Apache-2.0

"""2026-10-07: Offline semantic grader; candidate code only executes inside Docker."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import uuid

from corpus import CORPUS

ROOT = Path(__file__).resolve().parent
IMAGE = "sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3"
MAX_SOURCE = 32768
MAX_INPUT = 131072
MAX_OUTPUT = 65536
WALL_SECONDS = 12


def command(name):
    return [
        "docker",
        "create",
        "--pull=never",
        "--name",
        name,
        "-i",
        "--network",
        "none",
        "--ipc",
        "none",
        "--read-only",
        "--log-driver",
        "none",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "32",
        "--memory",
        "128m",
        "--memory-swap",
        "128m",
        "--cpus",
        "1",
        "--ulimit",
        "cpu=2:2",
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        "fsize=1048576:1048576",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--workdir",
        "/tmp",
        "--entrypoint",
        "/usr/bin/env",
        IMAGE,
        "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "HOME=/tmp",
        "PYTHONDONTWRITEBYTECODE=1",
        "python3",
        "-I",
        "-c",
        (ROOT / "container_driver.py").read_text(),
    ]


def run_source(source, arguments):
    if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE:
        raise ValueError("source exceeds32KiB or is not text")
    nonce = uuid.uuid4().hex
    payload = json.dumps(
        {"source": source, "arguments": arguments, "nonce": nonce}
    ).encode()
    if len(payload) > MAX_INPUT:
        raise ValueError("input exceeds128KiB")
    name = "laguna-code-" + uuid.uuid4().hex
    argv = command(name)
    # 2026-10-07: No shell, host mount, GPU, inherited secret env or Docker socket in container.
    started = time.monotonic()
    subprocess.run(
        argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10
    )
    try:
        metadata = json.loads(
            subprocess.check_output(["docker", "container", "inspect", name], timeout=5)
        )[0]
        host = metadata["HostConfig"]
        if not (
            host["NetworkMode"] == "none"
            and host["ReadonlyRootfs"]
            and host["LogConfig"]["Type"] == "none"
            and host["Memory"] == 128 * 1024 * 1024
            and host["MemorySwap"] == host["Memory"]
            and host["NanoCpus"] == 1000000000
            and host["PidsLimit"] == 32
            and host["IpcMode"] == "none"
            and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"}
            and host["CapDrop"] == ["ALL"]
            and not host.get("Binds")
            and not host.get("DeviceRequests")
            and not metadata["Mounts"]
            and "no-new-privileges:true" in host["SecurityOpt"]
            and metadata["Config"]["User"] == "65534:65534"
        ):
            raise ValueError("container isolation configuration mismatch")
        process = subprocess.Popen(
            ["docker", "start", "-a", "-i", name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except BaseException:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        raise
    result = {
        "image": IMAGE,
        "container": name,
        "nonce": nonce,
        "isolation_verified_before_start": True,
        "isolation": {
            key: host[key]
            for key in [
                "NetworkMode",
                "ReadonlyRootfs",
                "Memory",
                "MemorySwap",
                "NanoCpus",
                "PidsLimit",
                "CapDrop",
                "SecurityOpt",
                "LogConfig",
                "Tmpfs",
                "Ulimits",
            ]
        },
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    failure = None
    try:
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        input_offset = 0
        for kind, stream in [("stdout", process.stdout), ("stderr", process.stderr)]:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, kind)
        while selector.get_map():
            if time.monotonic() - started >= WALL_SECONDS:
                failure = "wall_timeout"
                break
            for key, _ in selector.select(timeout=0.1):
                if key.data == "stdin":
                    try:
                        input_offset += os.write(
                            key.fd, payload[input_offset : input_offset + 8192]
                        )
                    except BrokenPipeError:
                        input_offset = len(payload)
                    if input_offset == len(payload):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = streams[key.data]
                remaining = MAX_OUTPUT - len(target)
                target.extend(chunk[: max(remaining, 0)])
                if len(chunk) > remaining:
                    failure = "output_limit"
                    break
            if failure:
                break
        if failure:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        failure = type(error).__name__
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        process.kill()
        process.wait(timeout=5)
    finally:
        selector.close()
        # 2026-10-07: Preserve the owned container's exit cause before cleanup.
        # Exit 137 alone does not distinguish an OOM kill from another signal.
        try:
            state = json.loads(
                subprocess.check_output(
                    ["docker", "inspect", "--format", "{{json .State}}", name],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            )
            result["container_exit_state"] = {
                key: state.get(key)
                for key in ("ExitCode", "OOMKilled", "Error", "Status")
            }
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            TypeError,
            AttributeError,
        ) as error:
            result["container_exit_state_unavailable"] = type(error).__name__
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    result.update(
        {
            "exit_code": process.returncode,
            "failure": failure,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": streams["stdout"].decode("utf-8", errors="replace"),
            "stderr": streams["stderr"].decode("utf-8", errors="replace"),
        }
    )
    return result


def canonical(value):
    # 2026-10-07: JSON serialization distinguishes bool from integer; rejects NaN.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def grade(task, receipt):
    if receipt["failure"] or receipt["exit_code"] != 0:
        return {
            "passed": False,
            "kind": "infrastructure_failure"
            if receipt["exit_code"] in (125, 126, 127)
            else "execution_failure",
            "reason": receipt["failure"] or "container_exit",
        }
    try:
        parsed = json.loads(receipt["stdout"])
        if (
            parsed.get("schema") != 1
            or parsed.get("nonce") != receipt["nonce"]
            or parsed.get("completed_cases") != len(task["tests"])
            or not task["tests"]
        ):
            raise ValueError("missing expected test observations")
        outputs = parsed["results"]
        if len(outputs) != len(task["tests"]):
            raise ValueError("wrong result count")
        checks = []
        for test, actual in zip(task["tests"], outputs, strict=True):
            checks.append(
                actual.get("mutated") is False
                and actual.get("error") == test["error"]
                and (
                    test["error"] is not None
                    or canonical(actual["value"]) == canonical(test["expected"])
                )
            )
        return {
            "passed": all(checks),
            "kind": "semantic_pass" if all(checks) else "semantic_failure",
            "cases": checks,
        }
    except (ValueError, TypeError, KeyError):
        return {
            "passed": False,
            "kind": "invalid_candidate_execution",
            "reason": "invalid_driver_result",
        }


def evaluate(task, source):
    receipt = run_source(source, [test["args"] for test in task["tests"]])
    receipt.update({"task": task["id"], "grade": grade(task, receipt)})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--task", choices=[task["id"] for task in CORPUS])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.controls == bool(args.candidate) or (args.candidate and not args.task):
        parser.error("choose --controls OR --candidate FILE --task TASK")
    # 2026-10-07: Inspect the pinned local image; never pull an unreviewed tag automatically.
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", IMAGE], timeout=5)
    )[0]
    if image["Id"] != IMAGE:
        raise ValueError("image identity mismatch")
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipts = []
    if args.controls:
        for task in CORPUS:
            for name, source, expected in [("positive", task["good"], True)] + [
                (name, value, False) for name, value in task["bad"].items()
            ]:
                receipt = evaluate(task, source)
                receipt.update({"control": name, "expected_pass": expected})
                receipts.append(receipt)
                (args.output / (task["id"] + "-" + name + ".json")).write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                if receipt["grade"]["passed"] != expected:
                    raise AssertionError("control failed: " + task["id"] + "/" + name)
    else:
        if args.candidate.stat().st_size > MAX_SOURCE:
            raise ValueError("candidate too large")
        task = next(task for task in CORPUS if task["id"] == args.task)
        receipt = evaluate(task, args.candidate.read_text())
        receipts.append(receipt)
        (args.output / "candidate.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    summary = {
        "image_id": IMAGE,
        "image_repo_digests": image.get("RepoDigests"),
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["corpus.py", "container_driver.py", "run.py"]
        },
        "model_requests": 0,
        "executions": len(receipts),
        "passed": sum(r["grade"]["passed"] for r in receipts),
        "scope": "authored controls only"
        if args.controls
        else "offline candidate semantics; no speed/agent claim",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
