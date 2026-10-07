# GPT-OSS-20B native C1 performance diagnostics

October 7, 2026. These are bounded GB10 diagnostics for the explicit experimental
route, not a certification, vLLM comparison, or general model-quality claim.
The checkpoint is [OpenAI GPT-OSS-20B revision
6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
Native weights remain packed MXFP4; KV and the logical-vocabulary LM head are BF16.
[Quality results](gpt-oss-native-quality.md) and
[numerical/serving limits](gpt-oss-native-bringup.md) remain separate gates.

## Grouping four selected experts

The original C1 path launched gate GEMV, bias, SwiGLU, down GEMV and bias separately
for each of four experts: twenty launches per layer. The grouped path launches
those same five stages over four independent slots. It keeps the existing FP32
FMA/reduction order, each BF16 rounding boundary, and the final expert-ID-ordered
weighted reduction. Scratch for gate/up and activation grows fourfold; selected
output storage and model weights do not change.

The host still reads and validates the four router IDs once per layer. Grouped
kernels additionally reject invalid/duplicate IDs before indexing weights.
Contiguous 32-expert storage is checked before deriving strides. This does not
remove the 24 per-token synchronizations, enable CUDA graphs, batching, tensor
parallelism, prefix reuse, or disk swapping. Prefill remains scalar.

Correctness evidence:

- Twelve constructed comparisons cover tail and full gate/down dimensions,
  shared versus per-slot inputs, and bias/no-bias. Every output BF16 bit matches
  the incumbent serial operators. Three wrong-input-stride controls are detected;
  nine invalid/duplicate-ID cases avoid weight indexing and produce NaNs. Actual
  serving retains the earlier host errors, rather than relying on NaN sampling.
- The 251-token Harmony/window-boundary fixture produces byte-identical hidden
  traces and full logits against the original native baseline. Their SHA-256
  values are respectively `430071770446087b65764368131c994f061e1d37caf8c9947f67d2a4243837ce`
  and `04e60cbfc0c833f64e8e7a6c1cd3e0f715afd57f8feb751f04953f9c6398b7e6`.
  This preserves the native baseline; it does not erase its reference differences.
- Scoped storage tests, Linux CUDA Clippy, and SM90/SM100 compilation pass.
  The latter is compilation evidence, not measured correctness/performance on
  those devices.

## Release workload

The incumbent is commit `43598c80e026475f570d5d24c4c40067383d576c`, RELEASE binary
SHA-256 `2c7b215d57bbf321358d754789ce6d1ccaace1fe60df35f603b07576944ae08e`.
The grouped binary adds only the recorded runtime/kernel grouping overlays:
`13c3f3a93cc41fe41990eef3854e84393b395faa64bf2447645ea306011d7e24`.
The baseline archive, exact overlays, binaries/PTX, raw SSE events, source hashes,
constructed arrays and full-model traces are retained in the private run receipts.

Both use localhost, C1, a 512-token limit, GPU memory fraction 0.85, greedy decoding
and explicit low reasoning effort. The pinned checkpoint template produces
81/87/304 prompt IDs, rechecked before the grouped run. There is one arithmetic
warmup followed by three interleaved repetitions per case. No other CPU builds,
GPU jobs or artifact copies run during measurement. All nine measured requests
pass final-text, stop, framing and token-count checks.

Values below are medians. The incumbent repeat agrees with its original run
within 0.4% total latency. First visible means nonempty SSE `delta.content`, not
the role event or hidden analysis; first generated is the server's engine TTFT.

| Workload | Generated tokens | Incumbent TTFT | Grouped TTFT | Incumbent total | Grouped total | Grouped first visible |
|---|---:|---:|---:|---:|---:|---:|
| Integer arithmetic | 20 | 1.742 s | 1.585 s | 2.253 s | 2.055 s | 2.030 s |
| Count 1–16 | 63 | 1.871 s | 1.703 s | 3.531 s | 3.232 s | 2.099 s |
| Early identifier retrieval | 24 | 6.506 s | 5.937 s | 7.128 s | 6.509 s | 6.435 s |

The geometric mean total-latency speedup is **1.0947×** (about **8.65% less
latency**); decode is **40.54–40.73 tokens/s**, versus **37.21–37.47** in the
incumbent repeat. The predeclared gate required identical quality/counts, no
per-case TTFT/total regression above 2%, and geometric mean speedup above 1.02×.
This small workload passes that diagnostic gate; there is no energy measurement,
C2–C128 evidence, statistical certification or automatic promotion to qualified
support.

A matching three-request Nsight trace observes **332,169 kernel launches** versus
539,529 before grouping: exactly 360 fewer launches for each of 576 token
forwards. Total expert-GEMV GPU time drops from 7.132 s to 6.621 s; host launch
API time drops from about 1.044 s to 0.648 s. Router-ID readback count stays
unchanged. Readback API time includes waiting for prior GPU work and must not be
added to GPU duration as a separate cost. Profiling overhead is excluded from
the latency table above; this is not a per-kernel roofline certification.

An earlier unpack-bit optimization was rejected. It was bit-exact and about
8.6% faster on warm isolated gate GEMVs, but full-model latency regressed
5.5–6.2%. Its gate kernel averaged 96.27 µs versus 82.42 µs in the incumbent
profile. The candidate was removed from production, with both the passing micro
results and failing full-model receipts preserved. Warm microbenchmark wins
alone do not justify a dispatch change.

## Pinned eager reference context

A separate [Transformers v4.55.0 reference](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py)
uses dequantized BF16 weights, eager attention and batched prefill. It is an
operation/quality reference, not an optimized serving competitor or an
equivalent-precision native baseline. Model load is excluded. Peak allocated
CUDA memory is 46.92 GB. All nine final texts pass.

| Workload | Reference generated tokens | First generated | Local first visible | Total |
|---|---:|---:|---:|---:|
| Integer arithmetic | 20 | 0.253 s | 6.910 s | 7.280 s |
| Count 1–16 | 63 | 0.262 s | 6.173 s | 23.206 s |
| Early identifier retrieval | 19 | 0.454 s | 6.016 s | 7.128 s |

Reference decode is approximately 2.70 tokens/s. Its prefill is much faster than
the current scalar native path. Reference visibility is local decoded final-body
availability, not HTTP/SSE arrival. Retrieval generates 19 tokens instead of the
native 24, so its total latency is not token-count equivalent. These differences
must remain visible in comparisons; none establishes a vLLM speedup.

## Reproduce the bounded checks

On an idle CUDA host, build the selected-operator harness from the repository:

```bash
nvcc -shared -Xcompiler -fPIC -arch=sm_121f -O3 --fmad=false \
  -DTQ_PLUS_SIGNS crates/model-layers/tests/cuda/gpt_oss_selected_experts_test.cu \
  -o /path/to/fresh-fixture/selected.so
python scripts/gpt_oss_selected_parity.py /path/to/fresh-fixture
```

The fixture directory must include the compiled source tree if source hashes are
to be recorded. The Python harness writes raw selected weights/scales, inputs,
outputs, known-bad output and SHA-256 receipts under a new `parity-run1` directory.
It does not download models or call external services.

With the explicitly admitted C1 server already running and its binary/source/
model-revision identity saved in a JSON receipt:

```bash
python scripts/gpt_oss_c1_latency.py --self-test
python scripts/gpt_oss_c1_latency.py --base http://127.0.0.1:18842 \
  --evidence /path/to/server-evidence.json --output /path/to/fresh-run
```

Keep the host idle through both arms; preserve failures and do not replace these
small diagnostics with a certification claim. Batched prefill, safe deferred
router-error reporting/device-resident selection, broader quality, concurrency,
and energy qualification remain outstanding.
