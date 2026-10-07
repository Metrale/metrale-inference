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
