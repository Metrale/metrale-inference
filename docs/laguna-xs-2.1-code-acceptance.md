# Laguna: bounded coding acceptance

This authored corpus tests code semantics separately from the arithmetic
formatting investigation. It is four small Python tasks, not SWE-bench, a broad coding qualification, or an
agent/tool-loop qualification. Checkpoint remains
[poolside/Laguna-XS-2.1-NVFP4 at d32afde8b09af1539b49ff96ff5551c674485f8e](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e).

| Task | Form | Semantic coverage |
|---|---|---|
| Merge intervals | Generate | Touching/overlap, containment, duplicates, negative endpoints, empty input, invalid ranges/types, input immutability |
| Retry delay | Edit supplied bug | Status eligibility, ASCII-only header, zero, whitespace, clamp, invalid/Unicode headers, huge attempts, strict attempt type |
| Stable dependency order | Generate | Lexicographically smallest ready node, duplicate edges, unknown dependencies, disconnected cycles, self-cycle, immutability |
| Idempotent ingestion | Edit supplied bug | Arrival order, exact retries, duplicate existing events, conflicting fields, equal sequence numbers across different IDs, immutability |

There are 31 semantic cases. Tests compare exact structured results and exception
types, including boolean/integer distinctions; they do not grade substrings.
Four authored correct implementations pass. Eight deliberate incorrect variants
fail: missed touching boundary, input mutation, Unicode header acceptance, lost
zero delay, wrong ready ordering, ignored cycles, ignored conflicts and sequence
sorting. These results validate the grader, not Laguna's ability. The compact source-bound
receipt is [laguna-code-authored-controls.json](model-evidence/laguna-code-authored-controls.json).

## Execution boundary

The existing `agentic-webserver` shell runner executes `sh -c` in a host working
directory. It is not an isolation boundary and is not used here. Existing MLPerf
agentic coding metrics compare normalized executable names; they do not prove
functional correctness for these tasks.

`run.py` treats generated source as bytes. Only `container_driver.py`, inside an
owned disposable Docker container, compiles/executes that source. The pinned
locally available image is
`python@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3`
(Python 3.12.15). No automatic image pull occurs. The host verifies the actual
container configuration before starting it:

- Network and IPC disabled; no mounts, host bind paths, GPU, or Docker socket.
- Read-only root, nonroot UID 65534, all capabilities dropped, no-new-privileges.
- Only a 16 MiB `/tmp` tmpfs is writable. No host credentials/env are forwarded;
  the Python process runs under a cleared environment with explicit basic values.
- One CPU, 128 MiB memory with no extra swap, 32 PIDs, CPU limit two seconds,
  wall deadline 12 seconds, bounded file size/open descriptors, no Docker log storage.
- Source at most 32 KiB, input 128 KiB, stdout/stderr 64 KiB each. Nonblocking pipe I/O
  keeps output floods or stalled input from bypassing the wall deadline.
- Timeout/overflow removes only the UUID-named owned container; cleanup verifies
  those containers no longer exist. It never kills unrelated Docker processes.

Seven authored isolation controls pass: environment/filesystem/network facts;
stdout and stderr overflow; wall and CPU bounds; exit 0 without tests; and forged
success JSON. (The exit-cause section below adds an eighth.) A nonce-bound driver envelope and complete expected observations
are required; process exit 0 or a printed `passed:true` is not semantic success.
This is not adversarial attestation: candidate code and the Python driver share
an interpreter inside the container, so a deliberately hostile introspective
submission requires stronger out-of-process verification before any trust claim.
Infrastructure failures, resource/execution failures, invalid driver results and
semantic failures remain distinct, with raw exit codes and capped output saved.

The first isolation probe incorrectly expected only a loopback interface. The
Docker host also exposed inactive tunnel interfaces. The revised probe verifies the
actual Docker `NetworkMode=none`, no IPv4 routes and failed external connection,
rather than treating interface names as isolation evidence. No candidate/model
result is inferred from the failed probe.

## Reproduce authored controls locally

```sh
python3 scripts/laguna/code_acceptance/run.py --controls --output /absolute/NEW-controls
python3 scripts/laguna/code_acceptance/test_isolation.py --output /absolute/NEW-isolation
python3 scripts/laguna/code_acceptance/prepare_requests.py /absolute/NEW-requests.json
```

The request preparer writes four fixtures and makes zero HTTP calls. It sends
requirements and starter code only, without hidden tests or reference solutions.
Every control receipt records image identity, source hashes, candidate hash,
resource configuration, timing, observed cases and grading outcome. The committed
receipt hashes `run.py` and `test_isolation.py` as they were before the exit-cause
control was added. These short
container timings are grader overhead, not model inference performance.

## Model run protocol

The model run below used a frozen localhost native-server identity and the
pinned checkpoint, after the six-hour soak. The four prepared requests were sent
sequentially, temperature 0, at most 1,024 output tokens, no tools, logprobs or
schema, three repeats: 12 requests. Raw HTTP responses, actual usage, finish reason, model/source/binary identity and
request hashes are saved. A truncated response counts as incomplete, not as a
failed semantic test, and the budget is never raised silently.

Instruction formatting is graded separately from semantics: raw Python satisfies
the requested format; a single unambiguous Python fence is extracted into a new
candidate file and recorded as a format violation. Prose, multiple competing
snippets or an ambiguous patch are never guessed into a passing implementation.
No response runs in the host process. Each extracted candidate is graded with
`run.py --candidate /absolute/candidate.py --task TASK --output /absolute/NEW-grade`.
Semantic pass/fail stays separate from formatting, container failure and
incomplete generation, and arithmetic's exact-format and explanatory-response
results are reported separately.

These one-shot code-edit requests do not establish agentic capability: there is no
edit/test feedback loop or tool-call validation inside the isolation boundary.

## First native model execution (2026-10-07)

After the soak finished, the unchanged original server answered all 12 requests:
four tasks across three repeats. Docker-isolated semantic grading passed 9/12.
Merge intervals, dependency ordering and idempotent ingestion passed every case
on every repeat. Retry-delay edits failed on all repeats for three explicit
requirements: 5,000 leading zeros, ASCII-only digits, and rejecting Boolean
attempts. The failures remain recorded rather than weakening the corpus.

All 12 responses used a single code fence. That format violation is separate
from semantics; the unambiguous code body was graded only inside the pinned
container. No generated code ran in the host interpreter. This establishes a
bounded one-shot coding result, not broad coding or agentic qualification.
See [`laguna-post-soak-U.json`](model-evidence/laguna-post-soak-U.json) for the
exact checkpoint, executable identity and raw evidence hashes.

### Explicit-feedback repair (2026-10-07)

On fixed native source `020057a` using the default async route, three repair
turns supply the original retry-delay requirements, each original generated
solution, and the three failing counterexamples. All return identical plain code
without Markdown fences. The isolated grader now passes 10/11 cases for each:
bool attempts and Unicode headers are corrected, but 5,000 leading zeros still
trigger Python's integer-string conversion limit. Therefore **0/3 repaired task
passes**, distinct from the original 9/12 aggregate task result. This partial
repair evidence does not establish general coding/agent capability. Source and
raw-receipt hashes are in `model-evidence/laguna-eos-capture-followup.json`.

### Exit-cause evidence

The grader now records the owned container's Docker exit state before cleanup.
An exit code alone is insufficient to distinguish an OOM kill from another
signal. Eight isolation controls pass, including a bounded memory-allocation
control with `OOMKilled=true` and a CPU-bound control with `OOMKilled=false`;
source-size refusal and cleanup checks remain intact. Missing exit-state data is
reported explicitly, and semantic admission is unchanged.

An initial optimized reference serving engine on the same pinned checkpoint also
passes 9/12 tasks, with three retry-delay failures. This is a separate W4A4
execution policy, not numerical parity evidence for native execution. One retry
fails the same three semantic cases; two generate an unbounded huge power and
exit 137 before observations. Replaying those identical source bytes in the
same isolated grader confirms `OOMKilled=true` for both. Initial and replay
receipts remain separate; infrastructure success never turns missing semantic
observations into a pass.
