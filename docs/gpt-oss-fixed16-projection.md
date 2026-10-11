# GPT-OSS fixed-row dense projection

Updated 2026-10-07. This is a bounded performance improvement to the existing experimental native path, not numerical qualification or benchmark certification. The checkpoint remains [openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).

The FP32-output dense batch kernel now specializes complete 16-row tiles. Its predicate requires both a batch divisible by 16 and exactly 16 rows per grid-Y tile; partial tiles keep the existing path. Column accumulation, warp reduction, FP32 output and subsequent bias/BF16 rounding are unchanged. Entry-owned shared scratch lets both template branches reuse one allocation. The BF16 entry retains its existing arithmetic and launch ABI.

On GB10, the FP32 entry's local stack fell from 64 bytes to zero. Registers increased from 48 to 56; shared storage stayed at 17,920 bytes. An initial prototype duplicated shared storage between template branches; that version was corrected before request timing. These resource measurements explain the experiment's motivation, not a claim of measured hardware-counter attribution.

## Validation and measured result

The primitive gate passed 65 random shape comparisons against scalar FP32 execution, legacy BF16 comparisons, split batches 17/32/47, nine independent exact-sum/truncated-grid controls, and 12 direct empty/overprovisioned-grid cases. Full-model widths 0/16/31/64/127/128 passed exact hidden/cache comparisons over the 251-token prompt and following decode; all 12 retained trace hashes match the immutable accepted baseline.

Real-server gates passed 12 padded text cases, nine blocking-tool cases, five streamed-tool cases, seven streaming lifecycle cases, and exact prompt boundaries 128/129/255/256. Cancellation/reuse and response graders were unchanged.

Idle A→B then B→A request timing used three measured repetitions after warmup. Prompt/generated counts matched in every arm: **81/20, 87/63 and 304/24**. Geometric total latency was **2.3105% and 2.2569% lower**, passing the predeclared 2% minimum in both sessions. First-generated latency improved approximately 3.2–6.2%; decode remained about 40.0–40.5 tokens/s. Neither model admission nor chunk-capacity defaults changed.

The shared scratch refactor preserved BF16 output bits and resource usage, but not identical machine instructions. A separate bounded, opposite-order BF16 screen across M1/16/17/32/47/128 measured new/old latency ratios of 0.981–1.015. This is not a blanket no-regression claim for every shape or device. CHKI reports 53 targets across GB10, B200, B300 and Hopper; SM121/90/100a/103a compile checks passed, with runtime measurements limited to GB10.

## Artifact boundaries

The accepted baseline executable SHA-256 is `e52eef8199a0afbb84dd5ea04b0891c5780b93b81bbaa12e08d22e26f1f82fbc`; the measured candidate is `829dbf3a9b048dbc4370cc126d5c6f0d250e77f7e47500869b67c81d2313086f`. The tested kernel source SHA-256 is `b57998ca96c096aab861d70ed9a22dbc58324e41cd19132bab0ffcded6629786`. Final source changes only explanatory comments relative to that tested kernel. The timing comparison receipt SHA-256 is `fd08bb6a00f15c6b405762979c9d9ee2f33c0f2811e6b7101efb5c1058f820d4`.

The frozen candidate also includes the production transitive-source watcher fix from `51716c7`. A prelaunch staged-source check caught stale PTX in the cloned older build before any server started; that failure and the subsequent verified rebuild are retained. Raw request/response, resource, source, compiler and trace receipts remain available privately.

The original 244/251 sequential-reference limitation remains open. This bit-preserving optimization does not qualify the alternate tensor-core expert path or establish competitiveness with the optimized reference engine.
