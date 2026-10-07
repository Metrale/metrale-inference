# Working log: QCI Gemma 4 and Nemotron 3.5 Lightning

This is the note the next agent should read before adding a measurement.
The dump entry is deterministic. An absent weight path is a blocker. Two
different absent paths print the same receipt.

Branch base: `main` at `3e954ac`.
Method example: https://github.com/Metrale/metrale-inference/pull/126
Issues: https://github.com/Metrale/investor-mvp/issues/44 and
https://github.com/Metrale/investor-mvp/issues/58

## Commands

From a checkout of this branch:

```
cargo test -p metrale-qci-dump
cargo run -p metrale-qci-dump --bin qci-dump -- --case gemma-4-26b-a4b --hardware rtx-3090
cargo run -p metrale-qci-dump --bin qci-dump -- --case gemma-4-26b-a4b --hardware strix-halo
cargo run -p metrale-qci-dump --bin qci-dump -- --case nemotron-3.5-lightning-30b-a3b --hardware rtx-3090
cargo run -p metrale-qci-dump --bin qci-dump -- --case nemotron-3.5-lightning-30b-a3b --hardware strix-halo
```

A second hardware id, including one this machine is not, is the same binary:

```
cargo run -p metrale-qci-dump --bin qci-dump -- --case gemma-4-26b-a4b --hardware h100-sxm
```

A missing weight file is still exit 0. The text says `weights: absent` and
`WEIGHTS_ABSENT`. The path is not printed, so a relocated missing path matches.

```
cargo run -p metrale-qci-dump --bin qci-dump -- --case gemma-4-26b-a4b --hardware rtx-3090 --weights C:\no\such\a.gguf
cargo run -p metrale-qci-dump --bin qci-dump -- --case gemma-4-26b-a4b --hardware rtx-3090 --weights D:\other\missing.gguf
```

Chat admission, same binary, same gate the server calls:

```
cargo run -p metrale-qci-dump --bin qci-dump -- --case gemma-4-26b-a4b --request request.json
```

`request.json` is a chat-completions body. A video part or a bad image returns
JSON with `error.type` `invalid_request_error`. Exit 0 means the tool handled
it. It does not mean the model ran.

When `met` itself is built, `crates/server/src/api/chat/mod.rs` calls
`metrale_qci_dump::gate` for the two pinned model ids before auto-swap.
Other model ids do not enter the gate.

## Pins checked 2026-10-06

Gemma 4, issue #44, P0. Original `google/gemma-4-26B-A4B-it`.
Repository `unsloth/gemma-4-26B-A4B-it-GGUF`.
Revision `c099eb48e663fd284577b04978a94ffccb261841` (the repo sha that day).
Artifact `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, HTTP 200, 16947541728 bytes.
Vision component `mmproj-F16.gguf`, HTTP 200, 1193058784 bytes.
Concurrency 1. Context modest (`max_model_len` 8192, `max_images` 4,
`max_image_bytes` 2000000, `max_text_chars` 8192 on the recipe).
HF license tag `apache-2.0`. Card link
`https://ai.google.dev/gemma/docs/gemma_4_license`.
Blocker: `LICENSE_GOVERNING_TEXT_UNDECIDED`.
Pipeline tag on the card: `image-text-to-text`. That is a card fact, not a
native serve result.

Nemotron 3.5 Lightning, issue #58, P1. Does not block the four-model P0 exit.
Original `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` revision
`a9904d24bcc1d289a1950fa9d2b978c47cf903b9`, which matches `.src_sha` PRIMARY
on the GGUF repo.
Repository `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`.
Revision `8a08a1c81dadcc75d35dbb96016cfd344b632e67`.
Artifact `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, HTTP 200,
18898091584 bytes.
License `openmdw-1.1`, link `https://openmdw.ai/license/1-1/`.
The GGUF card's license tag is `other`, which is how that license is tagged.
Tokenizer bos `<s>`, eos `<|im_end|>`.
`tokenizer.json` sha256
`623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7`.
Chat template blob `d85b0c772f8fe585063847c5f6bf5ec48eb210be`.
`TEMPLATE_BYTE_IDENTITY_UNMEASURED` against the template embedded in the GGUF
metadata. MTP is off. The repo also contains
`mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`. This pin does not load it.
Do not assume NVFP4 kernels on the RTX 3090.
Distinct from `NVIDIA-Nemotron-3-Nano` and from the H100 and H200 Nano numbers.

## Blockers a receipt is allowed to clear only with a real run

- `WEIGHTS_ABSENT` / `VISION_WEIGHTS_ABSENT` — the file was not passed or was not a file.
- `PARITY_UNMEASURED` — no tier 1, 2, or 3 record. tok/s and J/tok stay withheld.
- `VRAM_UNMEASURED` — a file length is not memory.
- `COHERENCE_UNMEASURED` — the admission text says generation did not run.
- `ENGINE_NOT_BUILT` — pass `--runtime-commit` only for a binary you built.
- `TARGET_NOT_MEASURED` — `--hardware strix-halo` with no Strix log.
- `TOPOLOGY_NOT_IN_ISSUE` — any other hardware id. Still a receipt, still not a claim.
- `LICENSE_GOVERNING_TEXT_UNDECIDED` — Gemma only, until a product decision names the grant.
- `NATIVE_GEMMA4_CIRCUIT_REFUSED`, `NO_VISION_LOADER`, `NO_AMPERE_KERNEL_CLASS`.
- `NOT_THE_NANO_BENCHMARK`, `NATIVE_GGUF_Q4_0_UNSERVED`, `TEMPLATE_BYTE_IDENTITY_UNMEASURED`.
- `contract.41`, `contract.42`, `contract.46` — `BLOCKER UNFROZEN` when the issue requires them.

`support_label` stays `withheld`. `live_qci_acceptance` stays `not-claimed`.

## What the next agent should not do

- Do not mark investor-mvp #44 or #58 done from this log.
- Do not merge this beachhead on a 3090 file listing.
- Do not copy a GB10 or H100 tok/s or J/tok into these rows.
- Do not turn the GGUF recipe into `runtime: metrale` without a serve that gets past the flag check and a parity record.
- Do not decode video inside this gate. The ffmpeg flag on other models stays where it is.
