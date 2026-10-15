# metprof

Profiling analysis over nsys sqlite exports, CPU only. It is a standalone
Cargo workspace (its own `Cargo.lock`); build and test it with
`--manifest-path scripts/metprof/Cargo.toml`. It reads captures with its own
read-only SQLite reader and never runs a GPU workload.

Commands (`metprof --help` lists every flag): `summarize` (per-device busy
union and family breakdown), `families` (kernel families and the coverage
gate), `segment requests|decode|spec` (requests and pinned decode steps),
`diff`, `envdiff`, `verdict` (two captures, two environments, a recorded
verdict), `tree` / `tree-diff` (the kernel tree below), and `ttft log` /
`ttft compare` / `ttft client` (time to first token, further below).

Every threshold is a flag or a config key. A missing one is an error, never a
default.

## Kernel tree (`metprof tree`)

```
metprof tree --capture CAP.sqlite --device N --segment all|K [--gap-ms MS --expect N]
             --config families.toml --provenance PROV.toml --out tree.json
             [--model NAME]
             [--circuit ARCH --mode MODE --rows R --policy RECIPE --kernels-dir kernels/]
             [--baseline older.json]
metprof tree-diff older.json newer.json [--out diff.json]
```

The output (`"schema": "metprof.kernel_tree/1"`, `"time_unit": "us"`):

```json
{
  "schema": "metprof.kernel_tree/1",
  "time_unit": "us",
  "provenance": { "capture_sha256": "...", "binary_sha256": "...", "git_sha": "...", "...": "..." },
  "source": "circuit",
  "total_time": 1130.5,
  "coverage": 0.997,
  "kernels": {
    "<node>": {
      "fuses": ["..."], "total_time": 0.0, "self_time": 0.0, "launches": 0,
      "mean_us": 0.0, "min_us": 0.0, "max_us": 0.0,
      "p10_us": 0.0, "median_us": 0.0, "p90_us": 0.0,
      "kernels": { "<child>": { "...": "..." } }
    },
    "<leaf>": { "...": "...", "variants": [ { "demangled": "...", "config_sha256": "...", "kernel_id": "...", "...": "..." } ], "kernels": {} }
  },
  "unattributed": {
    "<demangled kernel name>": {
      "total_time": 0.0, "launches": 0, "p10_us": 0.0, "median_us": 0.0, "p90_us": 0.0,
      "variants": [ "..." ]
    }
  }
}
```

### Invariants

- Every launch in the segment is counted exactly once: either under one leaf
  or in `unattributed`. `total_time` is the sum of both.
- A node's `total_time` is its `self_time` plus the `total_time` of its
  children. Only leaves (kernels) have launches of their own.
- Times are sums of kernel durations, not a union of intervals. Overlapping
  kernels on different streams each count in full (`busy` gives the union).
- `coverage` is attributed time over total time. The run fails, and writes
  nothing, when it is below `[tree] coverage_min` of the config. The key has
  no default.
- `fuses` names what a node absorbed. Those constituents are never given a
  time of their own: a fused kernel has one measured time.
- Node keys are non-empty and contain no `/`, so a node's path (its keys
  joined by `/`) is unambiguous. Paths are the join key for baselines.
- A `.json` tree is read back exactly; a tree whose totals do not add up is
  refused as a baseline.

### Provenance file

`--provenance` is required. Every key is required. Any value may be the
explicit string `"unknown"`, but no key may be left out.

```toml
binary = "path/to/serve-binary"   # hashed; or binary_sha256 = "<64 hex>"
git_sha = "<7-64 hex>"
git_dirty = false
env = "path/to/env-dump"          # hashed; or env_sha256 = "<64 hex>"
recipe = "org/recipe-id"
isl = 1024
osl = 128
concurrency = 1
rank = 0
```

The tree's `provenance` adds `capture_sha256`, `device`, `segment` (`all` or
`request.K`), the `circuit` block (circuit trees only), `metprof_version` and
`window`: `{"start_ns", "end_ns"}`, the first launch start and the last launch
end of the segment on the capture's own clock (`null` for an empty segment).

### Kernel identity and launch distribution

- Every leaf, and every `unattributed` entry, lists `variants`: one per
  distinct identity among its launches, in first-seen order. An identity is
  the demangled name plus the launch configuration CUPTI records: `grid`,
  `block`, `registers_per_thread`, `static_shared_bytes`,
  `dynamic_shared_bytes`. Interior nodes have no `variants`.
- `config_sha256` is the SHA-256 of the configuration's canonical text,
  exactly
  `grid=X,Y,Z;block=X,Y,Z;registers_per_thread=R;static_shared_bytes=S;dynamic_shared_bytes=D`.
  `kernel_id` is the SHA-256 of the demangled name, a newline (`\n`) and that
  same text. Neither depends on the capture, so they join one kernel across
  captures; a code change that keeps the name and configuration keeps the
  id (pair it with `binary_sha256`). A variant's `launches` and `total_time` add up
  to its owner's.
- `p10_us`, `median_us`, `p90_us` are per-launch durations over every launch
  under the node (a variant: its own launches), by linear interpolation
  between closest ranks (`h = (n - 1) p`). They are `null` exactly when there
  are no launches.
- There is no PTX or SASS hash: the nsys export records no per-kernel code
  identity. `binary_sha256` in the provenance covers the code that ran.
- The launch-configuration columns (`blockX/Y/Z`, `registersPerThread`,
  `staticSharedMemory`, `dynamicSharedMemory`) are required of the kernel
  table; an export without them is refused.

### Families source (any model)

Without `--circuit`, each kernel is placed by the family table, the single
source for name matching:

- path = `<overlay node or family group>/<family>/<kernel short name>`.
- `--model NAME` applies `[tree.model.NAME]`: `[[tree.model.NAME.node]]`
  entries move the families their regexes match under `path` (a `/`-separated
  parent path), with optional `fuses`. `fuses = { <family> = [..] }` states
  what a family's kernels absorbed, e.g. a grouped gate/up GEMM that fuses
  `gate_gemm` and `up_gemm`. A pattern that matches no family, or a family
  claimed by two nodes, is an error.
- Unclassified kernels go to `unattributed` under their demangled names.

### Circuit source (models with a circuit)

With `--circuit`, the tree follows the circuit compiler's fusion plan for the
`--policy` instance (a recipe id from `circuits/INSTANCES.toml`), which must
be an instance of `--circuit`. The plan is built offline from the
`kernels/` texts with `metrale-circuit`, with every rule-named kernel
treated as available.

- path = `<arch>/<mode>/<layer>/<group key>/<module>::<func>`, where
  `<layer>` is `layer.N` or the prologue/epilogue block, and `<group key>` is
  `<rule>[<member locals>]`, suffixed `#n` on a repeat within the layer.
- Each planned group node carries `members` (member node locals, in pattern
  order). `fuses` (the member ops) is set only on groups with two or more
  members. Host groups and groups with zero repeats launch nothing and are
  left out; a planned kernel that never ran stays as a leaf with 0 launches.
- Attribution: each pinned decode step (one CUDA graph launch, all
  graph-node kernels of one correlation id) is aligned to the plan's expected
  launch sequence by longest common subsequence on the kernel's short name.
  The short name is `func`, or the CUPTI name given by
  `[tree.circuit_names] "<module>::<func>" = "<name>"`. A launch the
  alignment does not match (a renamed, extra or reordered kernel) and every
  launch outside the pinned steps goes to `unattributed`. The provenance
  `circuit` block reports `matched`, `missed`, `unmatched` and
  `outside_steps`.
- Eager segments (no graph-node kernels) and the per-run and verify-batch
  modes are refused.

#### Join contract with the circuit compiler

A circuit group node's `circuit` object is the join key back to the plan:

```json
"circuit": { "group": 412, "rule": "<rule name>", "plan_digest": "<64 hex>" }
```

- `group` is the group's index in the fusion plan (`gNNNN` in a printed plan),
- `rule` is the rule that formed the group,
- `plan_digest` is the plan's digest (`digest:` in a printed plan).

A join is valid only when all three agree with the plan being visualised. The
provenance `circuit` block also records `arch`, `mode`, `rows`, `recipe` and
the rules digest. The tree reads the circuit crate; it never changes it.

### Baselines and diffs

- `--baseline older.json` adds `baseline_time` and `delta` to every node,
  matched by path. A node absent from the baseline has `null` for both. Root
  `fused_away` lists baseline nodes with no counterpart, and `new` lists
  nodes the baseline lacks; a missing subtree is listed once, at its top
  node.
- `tree-diff` prints the node delta table, largest `|delta|` first, with each
  node's status (`both`, `new`, `fused_away`). `--out` also writes the full
  diff. Trees from different sources are not compared.

### Maintenance

`metprof` links the circuit planner (`crates/circuit`, pure, no GPU code) as
a path dependency, and CI builds it `--locked`. When the planner's
dependencies change, refresh this lock with
`cargo update --manifest-path scripts/metprof/Cargo.toml -p metrale-circuit`.
The circuit tests read the repo's `kernels/` and compare the offline plan with
the checked-in golden `kernels/circuits/plans/qwen3_5-decode-n1.txt`, so a
plan change shows up here when the golden is regenerated.

## Time to first token (`metprof ttft`)

```
metprof ttft log --log RANK0.log --config ttft.toml [--client RUN.jsonl]
metprof ttft compare --a CONTROL.log --b TREATMENT.log --config ttft.toml --min-reps N
metprof ttft client --url http://HOST:PORT --config ttft.toml --cell NAME \
    --prompt PROMPT.txt --cold N --warm M --out RUN.jsonl
```

`config/ttft.example.toml` documents every key; a missing key is an error.

### The one definition

- **TTFT_server** is the rank-0 log's `Done: ... TTFT=` value for the
  request, nothing else. A request runs from its `Chunked prefill start` line
  to the next `Done` line.
- **TTFT_client** is the time from writing the request to the first SSE frame
  that carries reasoning, content or a tool call. A chunked frame is timed
  when its HTTP chunk has fully arrived.
- **cold / warm** is decided by the server: a request is warm when a
  `Marconi SSM cache hit` line falls inside it. No latency threshold and no
  "first request" rule decides it, and a cache-hit line in an unknown format
  is an error.
- **First-token class**: `step` when the first token is `[class]
  step_token_id` (a token that streams no delta, so the client sees it one
  decode step later), `other` otherwise. Both are reported per cell.
- **Statistics**: the true median with n, the range, and p90 only at n >= 10
  (labelled `p90_interpolated`).

A request is measured when its `max_tokens` is `[select] max_tokens` and its
prompt token count is a cell's. A measured request that overlaps another
request, or has not finished when the log ends, is an error: without a
request id in the log its `Done` line cannot be attributed.

### Compare

`ttft compare` reports, per (cell, mode) present in both logs, the median
delta (B - A), both ranges and a verdict by range separation: `win` when every
B rep is below every A rep, `loss` when every one is above, `overlap`
otherwise, and `too-few` when either side has fewer than `--min-reps` reps.
The verdict says nothing about whether a lever was engaged; that is
`metprof verdict`.

### Client and join contract

`ttft client` sends `[client] warmup_requests` unmeasured requests (with
`warmup_max_tokens`, which must differ from the measured `max_tokens` so the
log never counts them), then `--cold` requests with a fresh salt each, then
`--warm` re-sends of the last cold prompt. The prompt file carries `{salt}`
exactly once; the salt is `salt_digits` decimal digits. A response whose
`usage.prompt_tokens` is not the cell's is refused.

`ttft log --client` joins the client records to the server's measured
requests in order: each record (warm-ups excluded) takes the next measured
request with the same prompt token count and a `Done` TTFT within 0.05 ms of
the response's `usage.time_to_first_token_ms` (the log prints one decimal).
The usage time is only the join key. A record that finds no request, has no
usage time, lands in another cell, or whose cold/warm intent disagrees with
the server's cache-hit marker is an error.
