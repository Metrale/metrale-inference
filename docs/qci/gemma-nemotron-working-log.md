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
The closed Q4_0 file's `tokenizer.chat_template` is 9867 bytes. `git hash-object`
of those bytes, with no added newline, is that same blob. Template byte
identity is measured. MTP is off. The repo also contains
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
- `NOT_THE_NANO_BENCHMARK`, `NATIVE_GGUF_Q4_0_UNSERVED`.
  `TEMPLATE_BYTE_IDENTITY_UNMEASURED` was cleared by the git-blob comparison above.
- `contract.41`, `contract.42`, `contract.46` — `BLOCKER UNFROZEN` when the issue requires them.

`support_label` stays `withheld`. `live_qci_acceptance` stays `not-claimed`.

## Host attempts after the first dump

### Weight search

Looked for `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, `mmproj-F16.gguf`, and
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` under `O:\Metrale`,
`O:\CDrive_Archive\huggingface`, `D:\`, `C:\Users\alexa\.cache`,
`C:\Users\alexa\AppData\Local`, `Documents`, `Downloads`, `Desktop`, and `V:\`.
Every name was a miss. No download was started. `--weights` was not pointed
at a real file, so the receipts stay on `WEIGHTS_ABSENT`.

### `met circuit venn`

One attempt per pinned id, against the closest recipe already in this tree:

```
met circuit venn --target unsloth/gemma-4-26B-A4B-it-GGUF --against gemma4/gemma-4-26b-a4b-nvfp4 --out kernels/circuits/venn/gemma-pin-vs-closest.md
met circuit venn --target ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --against nemotron-3-nano/nemotron-3-nano-30b-a3b-nvfp4 --out kernels/circuits/venn/nemotron-pin-vs-closest.md
```

The binary was not produced. With `METRALE_SKIP_BUILD=1` and
`CUDARC_CUDA_VERSION=13000`, `cargo build -p metrale-server --bin met --offline`
stopped at the link:

```
LINK : fatal error LNK1181: cannot open input file 'cuda.lib'
```

Blocker: `MET_LINK_CUDA_LIB`. No venn report was written. The in-tree
Nemotron 3.5 Lightning NVFP4 instance is a different artifact from the Q4_0
pin and was not substituted for it.

### CUDA 12.6 link

`cuda.lib` is present at
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\cuda.lib`.
`nvcc` there is release 12.6, V12.6.85. This host has no `nccl.lib`, so the
link was:

```
METRALE_SKIP_BUILD=1
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
cargo build -p metrale-server --bin met --offline --no-default-features --features cuda
```

`CUDARC_CUDA_VERSION` was not set. Exit 0. `target\debug\met.exe` was
produced. It was not started as a serve. Port 8888 was already refusing
connections before the build and still refused afterward. Nothing was
started there.

The binary as first linked overflowed the default 1MB Windows main stack
(`thread 'main' has overflowed its stack`, exit `0xC00000FD`). A one-off
`editbin` is not how the tree builds. `crates/server/build.rs` now passes
`/STACK:16777216` only when linking the `met` binary on Windows. A fresh
link with that script, and no edit after it, ran `met circuit venn --help`
twice. Both exited 0 and both printed usage. Port 8888 was down before the
link and was still down after. Nothing was started there.

### In-tree Lightning circuit, not the Q4_0 pin

```
met circuit venn --check --target nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4 --against qwen3.6/qwen3.6-35b-a3b-fp8-bf16head,qwen3.8/qwen3.8-27b-nvfp4-unsloth --mode decode,multi_seq,verify,draft --rows 1,16,128 --verify-rows 2 --out kernels/circuits/venn/nemotron-3.5-lightning-vs-qwen3.6-35b-a3b.md
```

The first `--check` exited 1: the file was stale at the end. Regenerating
with the same flags and no `--check` exited 0 and wrote that path. A second
`--check` exited 0 and printed `current`. The regenerated file's git blob
is `874bef9b4f580ca13986cef6d4576ae9a7e11c4c`, the same as `HEAD`, so there
is no report delta to commit. `cargo test -p metrale-circuit --test venn`
passes after the test lists paths with `/`, the same way the CLI does.
This result
is the NVFP4 instance `nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4`.
It is not the Q4_0 pin `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`.
No tok/s, J/tok, LKB, or LAB number from it was copied onto the GGUF rows.

### Gemma recipe

```
met circuit show --recipe gemma4/gemma-4-26b-a4b-nvfp4
```

Exit 1. The tool said there is no circuit instance for that recipe.
`kernels/circuits/INSTANCES.toml` was not given a row for
`unsloth/gemma-4-26B-A4B-it-GGUF`. Blocker: `NATIVE_GEMMA4_CIRCUIT_REFUSED`.
No diagram was written.

### Venn after the link

Same two pins, same closest recipes:

```
met circuit venn --target unsloth/gemma-4-26B-A4B-it-GGUF --against gemma4/gemma-4-26b-a4b-nvfp4 --out kernels/circuits/venn/gemma-4-26b-a4b-gguf-vs-nvfp4.md
met circuit venn --target ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --against nemotron-3-nano/nemotron-3-nano-30b-a3b-nvfp4 --out kernels/circuits/venn/nemotron-3.5-lightning-gguf-vs-nano.md
```

Both exited 1. Neither wrote a report. The tool said each target is no
recipe, checkpoint, or arch in `kernels/circuits/INSTANCES.toml`. The
instances it listed are the two Qwen 3.8 rows, the two Qwen 3.6 rows, and
`nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4`. That last id is the
NVFP4 circuit, not the Q4_0 GGUF pin, and it was not substituted.

Blocker: `VENN_PIN_NOT_AN_INSTANCE`.

### Checks on the first head

`Build mdBook + rustdoc` failed because the public docs for `DumpInput`
linked the private function `render` (`crates/qci-dump/src/dump.rs`). The
link was removed. Receipt text did not change.

`PR benchmark gate` is red because `STAMPED=false`. That hold stays. Do not
comment `/stamp`.

`comments` is red on undated lines in
`kernels/gb10/common/moe_nvfp4_grouped_tc.cu` (`Tile order: chunk c...`).
That file is already on `origin/main` and is not in this branch's diff.

### This host on 2026-10-07

`nvidia-smi` reported `NVIDIA GeForce RTX 3090, 24576 MiB, driver_version 616.92`.
`http://127.0.0.1:8888/health` refused the connection. Nothing was started
on that port.

The weight search above missed the three pins, and no download had been
started then. A fetch was started after that onto `O:\Metrale\qci-pins`, at
the revisions already in this log. The files are not in git.

`mmproj-F16.gguf` closed at 1193058784 bytes, the pinned vision length.
The header is GGUF version 3, `general.architecture=clip`,
`general.type=mmproj`, `general.name=Gemma-4-26B-A4B-It`,
`general.file_type=1`. In llama.cpp `aedb2a5e9` that integer is
`LLAMA_FTYPE_MOSTLY_F16`.

Shipped `qci-dump --case gemma-4-26b-a4b --hardware rtx-3090 --vision-weights`
on that file printed `vision_weights: present`,
`vision_file_bytes: 1193058784`, `weights: absent`,
`support_label: withheld`, and no tok/s or J/tok number. It still names
`VRAM_UNMEASURED`, `PARITY_UNMEASURED`, and `COHERENCE_UNMEASURED`.

Both language files later closed at the pinned lengths. Gemma
`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` is 16947541728 bytes. Nemotron
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` is 18898091584 bytes.
The headers already recorded above match those closed files. Neither file
is in git.

Shipped `qci-dump` was run twice on `rtx-3090` for each case, with
`--weights` pointed at the closed file. The Gemma runs also passed
`--vision-weights` for the closed `mmproj-F16.gguf`. The two Gemma runs
matched each other. The two Nemotron runs matched each other. Neither
receipt printed the path. Gemma printed `weights: present`,
`weights_file_bytes: 16947541728`, `vision_weights: present`, and
`vision_file_bytes: 1193058784`. Nemotron printed `weights: present`,
`weights_file_bytes: 18898091584`, and
`vision_weights: not-part-of-this-pin`. Both kept
`support_label: withheld` and printed no tok/s or J/tok number. Both still
name `VRAM_UNMEASURED`, `PARITY_UNMEASURED`, and `COHERENCE_UNMEASURED`.

#### First session, this host

1. The dump sources are `4b2bda9`. Local
   `cargo test -p metrale-qci-dump --offline` on that code exited 0:
   9 passed, 0 failed. The shipped binary is `target\debug\qci-dump.exe`.
   The closed-length receipts in this section are commit `de6a666`.
   That commit changes this log and the ledger. It does not change the
   dump sources.
2. Missing-path receipts for both cases match and do not print the path.
   `support_label` is `withheld`.
3. Vision file and both language files: the receipts above. A present
   file records its byte length and does not clear VRAM, parity, or
   coherence.
4. An already-installed `llama-cli` is version 9637 (`aedb2a5e9`), built
   with Clang 20.1.8 for Windows x86_64. No installer was launched.
   `--list-devices` printed
   `Vulkan0: NVIDIA GeForce RTX 3090 (24540 MiB, 23755 MiB free)`.
   It was invoked once against the closed Gemma pin, with `-n 0`,
   `-c 512`, `-ngl 99`, `--no-warmup`, and `--no-perf`. The binary said
   `--no-conversation` is not supported by `llama-cli` and to use
   `llama-completion` instead, then printed the Gemma file name and
   `modalities : text` and entered interactive mode. That process was
   stopped. It was not a server and it was not left running. Any timing
   line it printed is not a beachhead result and was not copied onto a
   receipt or into this log. The attempt belongs on the reference row
   (`ggml-gguf`), not the native row. Recipe `runtime` stays
   `gguf-reference`. No `INSTANCES.toml` row was added for either GGUF pin.
5. Strix stays `TARGET_NOT_MEASURED`. This host is not a Strix Halo.
6. Bit parity stays `PARITY_UNMEASURED`. tok/s and J/tok stay withheld.
   No number was copied from another device or from the NVFP4 Lightning
   instance.
7. Shipped `qci-dump --request` admitted an inline PNG for the Gemma case.
   `generation` is `not-run`, coherence is `BLOCKER COHERENCE_UNMEASURED`,
   and the body does not say the model is supported. A body with
   `video_url` printed `video_outside_runtime` and did not echo the
   payload. The unit test `video_decoding_stays_outside_the_runtime`
   asserts HTTP 400.
8. Nemotron stays P1. It does not block the four-model P0 exit. It was
   not reprioritized.

`NATIVE_GEMMA4_CIRCUIT_REFUSED` is unchanged.

On head `4b2bda9`, `Build mdBook + rustdoc`, `cargo test --workspace`,
and `recipes` passed. The concluded failure was `PR benchmark gate`
because the pull request is unstamped. The closed-length note is commit
`de6a666`. A check read of that head showed the same unstamped
`PR benchmark gate` failure. `cargo test --workspace` was still pending
on that read, so the workspace pass is not claimed for `de6a666`.
Do not comment `/stamp`. Jobs still queued with an empty runner name on
`metrale-macos-burst` or `self-hosted,macOS,ARM64,metal-gpu` were not
waited out. The name for that state is `RUNNER_LABEL_OFFLINE`.

### Reference runtime memory on this 3090

`llama-completion` from the already-installed llama.cpp build 9637
(`aedb2a5e9`) was the recorded run for each closed language pin. No
installer was launched. Flags were `-c 512`, `-ngl 99`, `-fit off`,
`-n 1`, `-no-cnv`, `--no-warmup`. Context 512 is the modest context for this sample. Gemma's
training context is 262144 and Nemotron's is 1048576. This sample does not
use those.

`nvidia-smi` `memory.used` on the RTX 3090 (24576 MiB):

- Gemma, before 1272 MiB, peak while the process was alive 18080 MiB, after
  exit 1272 MiB.
- Nemotron, before 1272 MiB, peak while the process was alive 19258 MiB,
  after exit 1271 MiB.

Both processes exited on their own. Port 8888 was down after they exited.
The device-free `qci-dump` receipt still prints `VRAM_UNMEASURED`, because
that program does not observe the GPU. The numbers above are the
reference-row serve record only. The native row stays `VRAM_UNMEASURED`.
`PARITY_UNMEASURED` and `COHERENCE_UNMEASURED` stay. Any timing line the
runner printed was not copied.

## What the next agent should not do

- Do not mark investor-mvp #44 or #58 done from this log.
- Do not merge this beachhead on a 3090 file listing.
- Do not copy a GB10 or H100 tok/s or J/tok into these rows.
- Do not turn the GGUF recipe into `runtime: metrale` without a serve that gets past the flag check and a parity record.
- Do not decode video inside this gate. The ffmpeg flag on other models stays where it is.
