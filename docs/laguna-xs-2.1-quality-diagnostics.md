# Laguna output variability: observation and diagnostic controls

Checkpoint: [poolside/Laguna-XS-2.1-NVFP4 at d32afde8b09af1539b49ff96ff5551c674485f8e](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e).
This supplements the [native qualification record](laguna-xs-2.1-native-bringup.md).
It changes no production sampling policy and does not qualify batched inference.

## Observed responses

The frozen snapshot with SHA-256
`3d94c6a7087c2023a8ca5e1fc58f9767718b4cd16af4cffead3201e10beededa`
contains 2,695 responses in completed groups: 1,748 exact integer answers and
947 explanations. A separate offline audit found supported literal addition
claims/final-answer markers in all 947 explanations, with no contradiction among
recognized claims. That limited parser is not complete semantic grading: the
947 responses remain `unscored_format`, not numeric passes. The full ongoing
soak and its later snapshots are tracked separately.

Observed text variation at client concurrency two/four does not identify a
race, actual scheduler batch membership, or the responsible arithmetic kernel.
The fixed vanilla Q/K norm dispatch defect and its earlier repeated-digit
failure are distinct from explanatory output-format failures.

## Static source finding and bounded local proof

Explicit request temperature zero overrides the model preset and uses greedy
sampling. Random seeds and top-k defaults are not sufficient explanations.
However, requesting `top_logprobs` is not a transparent observation:

- `scheduler/decode_logits_step.rs::argmax_readback_eligible` considers all active
  rows. One row requesting logprobs forces host-logit readback for its whole batch.
- The production host sampler takes the last equal maximum. The plain device
  argmax follows a 1,024-thread reduction tree. Verify uses the first maximum.
- Those distinct contracts are documented and tested in `metrale-sampling`.
  There is no universal tie-policy contract that can safely replace all three.

`crates/sampling/tests/laguna_sampling_path_diagnostic.rs` compares the real
host sampler with the existing CPU oracle for the GPU kernel. Exactly
BF16-representable tied logits select different IDs; a wide case yields
first/tree/last IDs 1/2/1025. Eight unique-maximum controls agree. Both tests and
scoped Clippy pass locally. This is not a new GPU capture and **does not establish
that ties caused any observed Laguna response**.

## Prepared post-soak diagnosis

Keep the live soak unchanged. After completion, freeze its actual server identity
and run bounded identical/mixed/permuted request groups without logprobs. Retain
slot-specific variants, finish reasons, exact formatting, wrong integers and
unscored prose separately. A JSON-schema test is an independent constrained
output test; passing it does not repair unconstrained instruction-following.

A separate, default-disabled diagnostic should capture unchanged-path selected
IDs and same-step logits with actual scheduler membership, padded batch size,
row order, sequence position and prefix-cache status. Select only explicit test
requests and bounded token positions; do not indiscriminately save prompts or
loaded tensors. Preserve an uninstrumented before/after control because copies
can perturb scheduling. Compare identical teacher-forced prefixes at the first
divergence, separating equal-logit tie choices from changed logits. Existing
host-only dump hooks cannot be assumed to observe the fast device route.

Only after that capture should a compatibility-reviewed fix be proposed. If
logprobs transparency is the problem, preserve the eligible device path's token
choice while extracting probabilities independently. Do not silently normalize
all host/device/verify policies or declare output variability solved from these
constructed tests.

### Bounded post-soak capture candidate

The optional server feature `laguna-diagnostic-capture` adds a hook immediately
**after the synchronous router's existing native argmax** and before its next
forward can reuse the logits buffer. Normal builds exclude that hook. Even a
feature-enabled build performs no copies or writes without
`METRALE_LAGUNA_CAPTURE_PLAN=/absolute/plan.json`.

This is a separate **timing intervention**, not an observation of the original
async soak. Preserve the uninstrumented results and run an explicit sync/no-mix
baseline first. Startup refuses async, multiple ranks, batch sizes above four,
speculation, and prefill codispatch/varlen; it requires
both `METRALE_BISECT_NO_MIX=1` and `METRALE_BISECT_Q12_DISABLE=1`
(the second also disables the separate batched-mixed lane). Only BF16 native-argmax decode rows are captured;
host/logprobs or masked sampling is refused for selected fixtures. Prefill's first
generated token is outside this hook. Device IDs are pre-postprocessing
candidates, not a claim about the final streamed token.

The JSON plan has exactly these fields:

```json
{
  "output": "/absolute/new-private-directory",
  "model_revision": "d32afde8b09af1539b49ff96ff5551c674485f8e",
  "asserted_source_commit": "FULL_40_CHARACTER_COMMIT",
  "prompt_sha256": ["SHA256_OF_EXACT_PROMPT_U32_LITTLE_ENDIAN_BYTES"],
  "generated_positions": [1, 2, 3],
  "max_records": 16
}
```

Replace the explicit placeholders before use. The output directory must not
exist. There may be at most eight prompt hashes, sixteen positions (1–128), and
64 records, bounding logits payloads to 51,380,224 bytes. A position means the
number of generated tokens **already processed after this decode forward**.
Every live batch row must match an allowlisted prompt; an unrelated row skips
the whole capture without copying it. Receipt row order is the router's actual
order, including slot, sequence length, prompt hash/length and prefix-lookup
flag. It does not infer padded rows from HTTP concurrency. Reused slots are
qualified by ticket, prompt hash and sequence position. No raw prompt is saved.

Each successful capture saves unchanged BF16 bytes, their SHA-256, unchanged
selected IDs and its exact router label. Startup also saves the running binary's
actual SHA-256. Source/checkpoint fields are explicitly operator assertions;
the surrounding immutable deployment receipt must verify them. Files without a
matching JSON receipt are incomplete captures, not evidence. The directory is
owner-only on Unix. This implementation does not change any tie policy.

The local test harness uses synthetic row bytes and callback counters to verify
same-step preservation, unchanged selected IDs, live membership order, unknown
batch/position/cap exclusion, duplicate-slot and precision/mode refusals, and
new-directory enforcement. **No live GPU capture or Spark2 restart has been
performed for this candidate.** It must remain unused until the soak completes
and the controlled diagnostic deployment is reviewed.

Validation uses `METRALE_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p
metrale-server --no-default-features --features metal,laguna-diagnostic-capture
--test laguna_capture_diagnostic` on the local Mac; six controls pass. Scoped
Clippy covers that test and the feature-enabled `met` binary. The initial
backend-free invocation failed because the existing server binary imports GPU
initialization symbols without a backend; enabling its normal Metal backend
resolved that build configuration. These are host/mock checks, not CUDA evidence.
