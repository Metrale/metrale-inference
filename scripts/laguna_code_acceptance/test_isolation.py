# SPDX-License-Identifier: MIT OR Apache-2.0

"""2026-10-07: Run only authored isolation controls in owned ephemeral containers."""

import argparse
import json
from pathlib import Path
import subprocess

from run import MAX_OUTPUT, grade, run_source

PROBE = """def solve():
    import os, socket
    import pathlib
    blocked = False
    try:
        pathlib.Path('/laguna-host-write-control').write_text('forbidden')
    except OSError:
        blocked = True
    pathlib.Path('/tmp/allowed').write_text('temporary')
    sock = socket.socket()
    sock.settimeout(0.25)
    no_network = False
    try:
        sock.connect(('198.51.100.1', 9))
    except OSError:
        no_network = True
    finally:
        sock.close()
    status = pathlib.Path('/proc/self/status').read_text()
    caps_zero = any(line.startswith('CapEff:') and int(line.split()[1],16)==0 for line in status.splitlines())
    no_new_privileges = any(line.startswith('NoNewPrivs:') and line.split()[1]=='1' for line in status.splitlines())
    interfaces = [line.split(':')[0].strip() for line in pathlib.Path('/proc/net/dev').read_text().splitlines()[2:]]
    return {'uid': os.getuid(), 'root_write_blocked': blocked, 'tmp_write': pathlib.Path('/tmp/allowed').read_text(),
            'network_blocked': no_network, 'no_ipv4_routes': len(pathlib.Path('/proc/net/route').read_text().splitlines()) <= 1,
            'caps_zero': caps_zero, 'no_new_privileges': no_new_privileges,
            'no_socket': not pathlib.Path('/var/run/docker.sock').exists(),
            'clean_env': set(os.environ)<= {'PATH','HOME','PYTHONDONTWRITEBYTECODE','LC_CTYPE'}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipts = []
    receipt = run_source(PROBE, [[]])
    expected = {
        "uid": 65534,
        "root_write_blocked": True,
        "tmp_write": "temporary",
        "network_blocked": True,
        "no_ipv4_routes": True,
        "caps_zero": True,
        "no_new_privileges": True,
        "no_socket": True,
        "clean_env": True,
    }
    verdict = grade({"tests": [{"expected": expected, "error": None}]}, receipt)
    assert verdict["passed"], receipt
    receipt["control"] = "isolation_facts"
    receipts.append(receipt)
    for name, source, expected_failure in [
        (
            "stdout_bound",
            "def solve():\n import os\n os.write(1,b'x'*131072)\n",
            "output_limit",
        ),
        (
            "stderr_bound",
            "def solve():\n import os\n os.write(2,b'x'*131072)\n",
            "output_limit",
        ),
        ("exit_without_tests", "import os\nos._exit(0)\n", None),
        (
            "forged_success",
            'import os\nos.write(1,b\'{"passed":true,"results":[{"value":null,"error":null,"mutated":false}]}\')\nos._exit(0)\n',
            None,
        ),
        ("wall_bound", "def solve():\n import time\n time.sleep(60)\n", "wall_timeout"),
        ("cpu_bound", "def solve():\n while True: pass\n", None),
        (
            "memory_bound",
            "def solve():\n return len(bytearray(512 * 1024 * 1024))\n",
            None,
        ),
    ]:
        receipt = run_source(source, [[]])
        receipt["control"] = name
        assert not grade({"tests": [{"expected": None, "error": None}]}, receipt)[
            "passed"
        ]
        if expected_failure:
            assert receipt["failure"] == expected_failure, receipt
        elif name == "memory_bound":
            assert receipt["exit_code"] != 0, receipt
            assert receipt["container_exit_state"]["OOMKilled"] is True, receipt
        elif name == "cpu_bound":
            assert receipt["exit_code"] != 0 and receipt["elapsed_seconds"] < 12, (
                receipt
            )
            assert receipt["container_exit_state"]["OOMKilled"] is False, receipt
        else:
            assert receipt["exit_code"] == 0 and receipt["failure"] is None, receipt
        assert len(receipt["stdout"].encode()) <= MAX_OUTPUT
        assert len(receipt["stderr"].encode()) <= MAX_OUTPUT
        receipts.append(receipt)
    for receipt in receipts:
        probe = subprocess.run(
            ["docker", "container", "inspect", receipt["container"]],
            capture_output=True,
            check=False,
            timeout=5,
        )
        assert probe.returncode != 0, "owned container leaked"
        (args.output / (receipt["control"] + ".json")).write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    try:
        run_source("x" * 32769, [])
    except ValueError:
        pass
    else:
        raise AssertionError("oversized source accepted")
    print(
        json.dumps(
            {
                "isolation_controls": len(receipts),
                "source_size_refusal": True,
                "model_requests": 0,
                "owned_containers_remaining": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
