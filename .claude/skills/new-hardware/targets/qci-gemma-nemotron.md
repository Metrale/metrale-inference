# Hardware Beachhead Campaign: QCI Gemma 4 and Nemotron 3.5 Lightning

- **Campaign PR:** opened from this branch. Not for merge until a Strix host and a reference runtime have proved the rows.
- **Branch / base:** `feat/qci-gemma-nemotron-dump` on `main` at `3e954ac`.
- **Devices:** `rtx-3090` (supplement) and `strix-halo` (target). Neither id is in `kernels/DEVICES.toml`.
- **Kernel class:** there is no `kernels/ampere/`. `kernels/strix/` and `kernels/strix-hip/` exist and do not contain these models.
- **Status:** code-free prep plus a device-free dump. Nothing here has run on Strix or on a loaded weight file.
- **Done means:** for each model, vendor, repository, revision, license, and quantization are pinned (Nemotron also pins tokenizer and template); text and images for Gemma share one API and one recipe; video decoding stays outside that path; each topology has a receipt with memory, coherence, parameters, and limitations, or a named blocker; bit parity is recorded before tok/s or J/tok; reference and native rows stay separate; 3090 results stay a supplement; Nemotron stays P1 and off the four-model P0 critical path. Live QCI acceptance is a different record.
- **Ledger:** `ledger/qci-gemma-nemotron.toml`.
- **LKB baseline:** unverified. `met circuit lkb` was not run for these checkpoints on this host. Do not copy the H100 ledger.
- **LAB baseline:** unverified on this host. The circuit refusal for `gemma4` is cited below. It is not a coverage percentage.
- **Loop budget per model:** not opened. Parity is unmeasured, so the improvement loop has not started.

The issues are [investor-mvp #44](https://github.com/Metrale/investor-mvp/issues/44) and [investor-mvp #58](https://github.com/Metrale/investor-mvp/issues/58). The method example is [metrale-inference #126](https://github.com/Metrale/metrale-inference/pull/126). Dependencies #41, #42, and #46 were open on 2026-10-06. Their schemas are not invented here.

## 1. The tree as it is today

| Fact | Value | Source |
|---|---|---|
| Device ids the planner knows | `h100-sxm`, `h100-nvl`, `h200-sxm`, `b200`, `gb200-b200`, `gb300`, `hgx-b300`, `gb10` | `kernels/DEVICES.toml` lines 114, 194, 271, 350, 432, 513, 593, 674 |
| RTX 3090 / sm_86 | absent from that file | same |
| Strix Halo class file | `vendor = "amd"`, `arch = "gfx1151"`, descriptive memory keys | `kernels/strix/HARDWARE.toml` lines 1-17 |
| Strix model leaves | `qwen3.6-27b`, `qwen3.6-35b-a3b` only | `kernels/strix/` |
| Ampere kernel class | no `kernels/ampere/` directory on this commit | `kernels/` |
| Gemma 4 kernels that exist | gb10 `gemma-4-26b-a4b` and `gemma-4-31b`, not yet measured, no audio tower | `KERNEL_ARCH_ROADMAP.md` line 155 |
| Gemma 4 loader | text layers only. The module docs do not mention vision, audio, or video | `crates/model-arch/src/weight_loader/gemma4.rs` lines 3-7 |
| Circuit and `gemma4` | the NVFP4 Gemma 4 checkpoints are on the refusal path, reason `model_type gemma4` | `crates/circuit/tests/checkpoint_refusals.rs` lines 71-74 |
| Existing Gemma recipe | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16`, NVFP4, GB10 image | `recipes/gemma4/gemma-4-26b-a4b-nvfp4.yaml` lines 2-7 |
| Nemotron kernels the tree serves | Nano, Super, Puzzle. Not Lightning | `KERNEL_ARCH_ROADMAP.md` line 156 |
| Lightning NVFP4 fixture | circuit arch `nemotron_h`, MTP dimension 1. That fixture is not the GGUF Q4_0 pin | `crates/circuit/tests/checkpoints.rs` lines 100-148 |
| Lightning gaps the roadmap names | hidden-size pin, MTP layer | `KERNEL_ARCH_ROADMAP.md` line 204 |
| Existing Nano recipe | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `recipes/nemotron-3-nano/nemotron-3-nano-30b-a3b-nvfp4.yaml` line 2 |
| Optional ffmpeg video decode | a serve flag, off unless `--video-allow-ffmpeg` | `crates/server/src/main_modules/app_state.rs` lines 65-69 |
| QCI chat gate | campaign model ids only; other models fall through | `crates/server/src/api/chat/mod.rs` lines 107-114 |

## 2. Audit

1. **No Ampere class.** `kernels/` has no `ampere` directory, and `DEVICES.toml` has no 3090. A 3090 result cannot be a native kernel plan. Consequence: the 3090 row is a consumer supplement with blockers, not a certification.
2. **Strix has no leaf for either model.** `kernels/strix/` builds Qwen 3.6 leaves. Consequence: `TARGET_NOT_MEASURED` on every Strix row. Issue #42 (the Strix runtime) is still open, so the software chain is `CONTRACT_42` unfrozen for Gemma.
3. **The pinned Gemma artifact is not the NVFP4 recipe.** The recipe at `recipes/gemma4/gemma-4-26b-a4b-nvfp4.yaml` names a different repository and NVFP4. Using it would be a substitution. It is not used. The pin is the GGUF below.
4. **The circuit refuses `model_type gemma4`.** `checkpoint_refusals.rs` lines 71-74. The gb10 kernels and the loader exist beside that refusal. Native serve of this pin is `NATIVE_GEMMA4_CIRCUIT_REFUSED`.
5. **The Gemma loader has no vision path.** `gemma4.rs` lines 3-7. Images are admitted on the API so order, malformed input, pressure, and video refusal can be tested. That admission does not decode pixels and does not claim the vision tower runs.
6. **Lightning is not Nano.** The served Nemotron row is Nano / Super / Puzzle (`KERNEL_ARCH_ROADMAP.md` line 156). The Lightning NVFP4 fixture has an MTP head (`checkpoints.rs` line 141). This campaign pins GGUF Q4_0 and sets MTP off. Those are different artifacts. `NOT_THE_NANO_BENCHMARK`.
7. **NVFP4 is not an Ampere plan.** The dump says native limitations include not assuming NVFP4 kernels on the 3090. No measurement is copied from GB10 or Hopper.
8. **Video decode exists elsewhere and is outside this path.** `app_state.rs` lines 65-69. `gate` returns `video_outside_runtime` for the two campaign ids and does not call ffmpeg.
9. **#41 and #46 are open.** Receipt field names in this dump are a beachhead text, not the frozen `ModelRecipe` / approval schema from those issues. `contract.41` and the required sibling stay `BLOCKER UNFROZEN`.
10. **LKB and LAB numbers were not taken.** `met circuit lkb` was not run. Coverage cells in the ledger are empty on purpose.

## 3. What transfers, and what does not

| Work | Transfers? | Why |
|---|---|---|
| gb10 Gemma 4 kernel sources | code only, and only on gb10 | they are not an Ampere or Strix build, and they are not measured |
| gb10 Nemotron-H Mamba2 / ReLU² kernels | code only, for the NVFP4 Nano shape | Lightning's GGUF Q4_0 and its MTP head are not that shape |
| Qwen `q4k_mmq` | no | the roadmap names it under Qwen3 dense, not under these pins |
| H100 tok/s or J/tok | no | another box, another model, and parity here is unmeasured |
| Optional ffmpeg video path | no | this campaign keeps video outside the runtime |

## 4. Models, in issue order

| Model | Pin | Why this one | What runs here |
|---|---|---|---|
| Gemma 4 26B-A4B | `unsloth/gemma-4-26B-A4B-it-GGUF` @ `c099eb48e663fd284577b04978a94ffccb261841`, artifact `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (16947541728 bytes) plus `mmproj-F16.gguf` (1193058784 bytes). Original `google/gemma-4-26B-A4B-it`. | P0 multimodal. HF `pipeline_tag` is `image-text-to-text`. Card license tag `apache-2.0` and `license_link` `https://ai.google.dev/gemma/docs/gemma_4_license`. Both recorded. Governing text is `LICENSE_GOVERNING_TEXT_UNDECIDED`. | Device-free admission and a blocker receipt. No weight load. |
| Nemotron 3.5 Lightning 30B-A3B | `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` @ `8a08a1c81dadcc75d35dbb96016cfd344b632e67`, artifact `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (18898091584 bytes). Original `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` @ `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` (the conversion `.src_sha` PRIMARY). License `openmdw-1.1`. Tokenizer bos `<s>`, eos `<|im_end|>`, `tokenizer.json` sha256 `623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7`, chat template blob `d85b0c772f8fe585063847c5f6bf5ec48eb210be`. Template bytes against the GGUF-embedded copy are `TEMPLATE_BYTE_IDENTITY_UNMEASURED`. | P1 agent worker. Not Nano. MTP file on the GGUF repo is not loaded. | Device-free receipt. No weight load. |

Checked with HTTP HEAD on 2026-10-06. Those byte sizes match the issue text (16.948 GB and 1.193 GB; 18.898 GB). Gated was false on both GGUF repos. The files were not on this machine.

## 5. Parameterization

No kernel was copied for this beachhead. The duplication this campaign should not add:

- A second Gemma or Nemotron leaf under `kernels/strix/` before a Strix host shows which gb10 sources actually compile there.
- An Ampere tree cloned from gb10. #126's rule is that a measured win does not transfer with the source.
- A frozen recipe schema beside #41. The YAML files use `Recipe::parse`'s required keys and a `qci:` block the launcher ignores.

## 6. First session, in order

- [ ] Check out this PR. Build `qci-dump` with `cargo run -p metrale-qci-dump --bin qci-dump`. A private `CARGO_TARGET_DIR` if you also build `met`.
- [ ] Read the working log and run the two dump commands. Confirm the pins and that `support_label` is `withheld`.
- [ ] On a machine that has the GGUF files, pass `--weights` and `--vision-weights`. A present file records its byte length and still does not clear VRAM, parity, or coherence.
- [ ] Bring up an Ampere-compatible reference runtime for the 3090. Record it on the reference row. Do not label that row native.
- [ ] On Strix, follow issue #42's software chain once #42 freezes it. Replace `TARGET_NOT_MEASURED` only with a log from that host.
- [ ] Bit parity before any tok/s or J/tok. Keep the metrics in separate fields.
- [ ] Gemma images: the same `/v1/chat/completions` body the gate already admits. Video stays a 400 `video_outside_runtime`.
- [ ] Nemotron stays scheduled after the P0 slice unless the owner reprioritizes it.

## 7. Log

| Date | Step | Result | Evidence |
|---|---|---|---|
| 2026-10-06 | HF HEAD of both starting artifacts and the Gemma vision component | sizes match the issues; revisions match; Gemma license text is two links; Nemotron license is openmdw-1.1; weights not on disk | `docs/qci/gemma-nemotron-working-log.md` |
| 2026-10-06 | Device-free dump and admission tests | receipts and API errors; no serve of weights | `cargo test -p metrale-qci-dump` |
