# Measurement discipline

Every rule here exists because breaking it once produced a wrong number that was believed.
They apply to every timing, energy, TTFT or accuracy number a bring-up produces, on any class.

## Comparisons

- **Same-box A/B only.** Two boxes of one class differ (driver, firmware, cooling): on the GB10
  reference fleet one box ran 32k warm TTFT about 2x slower and drew 7-12 % more J/tok than
  another on identical code. Before attributing a delta, read each record's box; if they
  differ, run a same-box A/B before deciding anything.
- **n = 3 interleaved fresh serves** per arm (A B A B A B), one binary per arm, the arms built in
  separate build directories. A difference inside the control spread is no difference.
- **Die-temperature gate**: wait for the die to cool to the class's ceiling before every serve.
  Use the harness's own reading; an ad-hoc probe of the hottest sensor catches an instant in a
  duty cycle and reads up to tens of degrees hot.
- **One fresh serve per rung at high concurrency**, with a watchdog, and no request timeout at
  C >= 64 (a default request timeout cuts wide rungs mid-generation and reports
  `finish_reason="timeout"`, which reads as a throughput plateau). Read the finish reasons,
  not just tok/s.
- **A rung that goes backwards** (C128 below C64) is the engine struggling there: treat it as a
  warning, and on a shared box budget for the possibility that the run takes the box down.

## Energy

- Use the **NVML cumulative energy counter** through the API
  (`nvmlDeviceGetTotalEnergyConsumption`), not an integral of sampled `power.draw`, which
  reads a few percent high under load. Label the result "GPU-rail energy" when the device's
  NVML covers only the GPU rail. Compare engines measured the same way only.
- **Energy from a mock is not trusted**: synthetic weights draw more power than trained ones.
- On a unified-memory device, host "used" memory includes the GPU's allocations and is
  attributable to no process: not a leak, but real pressure.

## Proving a lever

- **Prove the lever moved**: confirm from the runtime predicate (and the live process's
  environment), not the startup log line, that the arm actually turned it on; and that the two
  arms differ in exactly one variable. A flag's value is not what runs at a given width (a
  per-concurrency ladder may override it); grep the serve log for what ran.
- **A passing test may not have run**: an orphan test module, an env-gated early return, a
  mutation that did not compile or did not apply, a `--lib` run that skips the binary's tests,
  a check whose verdict is printed unconditionally. Prove a check can fail (a negative control
  that fails, naming the test) before trusting its pass.
- **Byte identity** is strict prefill bits (byte-identical logits) PLUS greedy transcripts over
  a fixed prompt set, both against a control run of the unchanged binary in the same session.
- **Stale PTX**: a build directory shared between worktrees ships stale kernels with no error.
  One worktree, one `CARGO_TARGET_DIR`; prove a kernel edit reached the binary.
- **Never build a gate value from a debug build**, and never time one.

## Configurations

- **Recipes set every variable explicitly.** An inherited CLI default (a maximum batch size of
  8) once throttled every rung above C8 of a published measurement with no visible change to
  the recipe.
- **Precision follows the checkpoint's declared formats**, per layer, weights and activations.
  Running above it silently is not a default; going below it is a flagged opt-in with an
  accuracy bar, disclosed on records.
- **Record the vLLM version and image digest** (`docker image inspect`), the launch command
  and the effective engine args from its log, beside every vLLM number. A tag such as `latest`
  is not a version.
- **The PARITY-O.R.A.C.L.E** rules on both sides' resolved configs before any vLLM-vs-Metrale
  number is recorded, published or used in the loop.
- **A measurement-definition change** (prompt content, usage accounting, a clock, a scorer
  input) rides its own PR and its own campaign: a gate that goes red for a definition reason
  cannot be attributed inside a kernel campaign.
- **Bounds are measure-then-declare.** An unmeasured gate variant cannot bootstrap itself, and a
  bound is never ratcheted in the PR that first records its baseline.
- **Competitive bounds**: where a gate's bound is "at least as fast as vLLM", it is vLLM's
  number on the same instrument and box, measured one-shot; parity is the gate, winning is the
  objective.
- **BFCL**: a score is meaningless without its draw (N, category sample pct, SHA-256 of the
  ordered sample ids). Samples that fail on context length must stay in the denominator.

## vLLM specifics

- A vLLM stall or hang at a rung is a result, not missing data: capture the launch command,
  image digest, time at zero tok/s with requests running (from its metrics), completed vs
  issued, logs; then a one-change-at-a-time mitigation ladder. Re-run once on a fresh serve
  before calling it reproducible.
- vLLM with speculative decoding at C >= 32 has taken a unified-memory box down even at the
  safe memory utilization. On shared hardware, get the number once, record it, and do not
  re-run it to tidy a table.
- vLLM may lack a working path for a checkpoint's format on the new class (a W4A16 fallback, a
  shape constraint on the kernel). If it cannot load the checkpoint, record that with the error,
  and agree with the campaign owner which baseline replaces it.
