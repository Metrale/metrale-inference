# Minimum-token GPU greedy sampling

The opt-in `METRALE_MIN_TOKENS_GPU_GREEDY=1` path keeps minimum-output-length EOS masking on the GPU for neutral, greedy BF16 decode rows. The default remains the existing host path. Serving correctness, actual dispatch and the bounded opposite-order workload gate are verified. The switch remains off by default.

The existing host eligibility rule copies the entire vocabulary to the CPU while any request has an active minimum-token floor. The new entry masks each row's EOS IDs and, where applicable, the same post-thinking tokens as the host pipeline. It selects the highest token index on equal finite maxima, matching the host greedy policy. It does not reuse the older device argmax's different tie policy.

Admission excludes grammar, tools, log probabilities, active thinking, adaptive sampling, non-neutral penalties or bias, and diagnostic observers that need the original host logits. At the floor boundary, ordinary existing sampling eligibility resumes. Unsupported models use the trait's default `None` result and the original host readback. More than eight unique in-range masks, nonfinite unmasked logits, or an all-masked row also fall back for the entire batch. Out-of-range IDs are ignored and duplicate IDs are removed, matching host mask writes.

The model uses bounded existing scratch and its default forward stream; it does not allocate GPU storage per token or mutate logits. The synchronous readback completes before that scratch is reused. The asynchronous router delegates this variant to its synchronous inner router. Existing device-feed behavior is unchanged.

Validation so far:

- The model-side packing test checks duplicate/out-of-range IDs and the eight-ID capacity boundary.
- The new CUDA primitive passed 84 constructed rows across vocabulary widths 1 through 100352 plus three mixed-mask rows with padded stride. Cases include masked maxima, high-index ties, signed-zero ties, very negative finite values, masked and unmasked NaNs, infinities, and all-masked refusal.
- The same source compiles for SM90, SM100 and SM121. Both pre-existing feed entry bodies remain byte-identical on all three targets. These are compilation checks, not device execution claims for SM90/SM100.
- The server tests exercise actual host-pipeline mask equality, floor boundaries, mixed rows and admission refusals.

The constructed CUDA gate is `scripts/laguna/masked_argmax_check.py`, using `masked_argmax_check.cu` compiled as `probe.so` with the common-kernel include path, `-O3 --fmad=false -shared -Xcompiler -fPIC`, and the local device architecture. It requires PyTorch only as a constructed input/output oracle; it loads no model weights. Pass a directory containing the shared library and the exact `argmax_feed.cu` source to the script.

The checkpoint used for subsequent serving qualification must remain [poolside/Laguna-XS-2.1-NVFP4 at d32afde8b09af1539b49ff96ff5551c674485f8e](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e). Primitive correctness alone establishes neither model quality nor a performance improvement.

## Frozen serving correctness

Candidate executable SHA-256 `3aeaa2f04ff185b4fd624ba6aa9ee6df8949d1a183b10b4d55d63bc98c7091d8` embeds the exact new argmax PTX (`18e81d9a985dec515ff5c75b2b9155c1e56d933978d7dcb4616d9fecdd0e7778`). Its qualified dense and expert projection PTX remain unchanged from the preceding LUT candidate; the rejected private paired-expert experiment is absent.

With the option enabled, six structured cases, four EOS/tool controls, six concurrent structured cases, unequal-length draining, cancellation survivors and a subsequent single request passed. All twelve collected coding candidates retain the preceding candidate's exact source hashes. Isolated semantic grading remains **9/12**, with the same three retry-delay failures; this is not a clean coding qualification. Structured/tool requests intentionally keep their host route.

A separate C4 fixed-count Nsight capture records 64 calls to the new masked kernel, proving that ordinary eligible requests actually execute it. Profiled timings are diagnostic and are excluded from the uninstrumented workload verdict. Local server tests pass 334 scheduler cases (one ignored) plus five lever-registry tests; the model packing test and scoped server/model-engine Clippy also pass.

## Opposite-order workload gate

Four quiet sessions ran incumbent A, candidate B, candidate B2, incumbent A2 with the same GPU, checkpoint, FP8 KV, batch limit four, graph policy and request fixtures. The incumbent is the already qualified LUT executable `a9f01a9389ee456dc83f8aafcfc99316b0069fadb9649f102cda0a7fe23ef7f8`; the candidate is the executable above. All **160 cohorts / 400 requests** passed fixed-count admission, including warmups. Measured repetitions are retained without outlier removal.

| Workload | Total client latency change, first / reverse order |
|---|---:|
| C1, 64 input / 64 output | +0.21% / +0.17% (flat) |
| C2, 64 input / 64 output | −3.59% / −2.52% |
| C4, 64 input / 64 output | −3.62% / −3.34% |
| C4, four distinct prompts / 128 output | −3.16% / −3.55% |

Short and long prefill controls are essentially flat. Every fixed-workload text-hash set matches across arms. Diverse outputs vary within the incumbent as well as the candidate, and some cross-arm hash sets differ; all are retained and the diverse workload is a count/latency diagnostic, not a semantic-quality pass. Its actual tokenizer fixture is identical across all four sessions.

These measurements include prefill and final stream drain. Client TPOT is not isolated kernel throughput, and client concurrency does not prove batch membership. The preceding native-versus-Marlin comparison uses older binaries and must not be relabeled as a comparison of this new candidate. A broad competitiveness claim, clean coding qualification and energy measurements remain outstanding.
