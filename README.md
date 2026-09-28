<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/readme/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/readme/banner-light.svg">
    <img alt="Metrale Engine: Next-Generation LLM Inference Engine in Rust" src="docs/readme/banner-light.svg" width="640">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Metrale/metrale-inference/actions/workflows/ci.yml"><img alt="CI status of the main branch" src="https://github.com/Metrale/metrale-inference/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="#licence"><img alt="Licence: MIT OR Apache-2.0" src="https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-8A76CC"></a>
</p>

# Metrale Engine: Next-Generation LLM Inference Engine in Rust

Metrale Engine is an LLM inference engine written in Rust and CUDA. One
binary, `met`, serves an OpenAI-compatible HTTP API (and the Anthropic
Messages API) with no Python in the serving path. Every `(hardware, model,
quantization)` target has its own CUDA kernel set, compiled to PTX at build
time. Kernel sets cover NVIDIA GB10 (DGX Spark), Hopper and Blackwell
(B200, B300), AMD Strix Halo and Apple Metal, serving NVFP4 and FP8
checkpoints with speculative decoding (MTP, DFlash, n-gram). The certified
results below were measured on the GB10.

On one DGX Spark serving `unsloth/Qwen3.8-27B-NVFP4`, it matches or exceeds
vLLM 0.27.1 with vLLM's own MTP speculative decoding in aggregate decode
throughput at every concurrency from 1 to 128, with every workload axis
matched, and is clearly ahead at C=1, 2, 64 and 128:

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Metrale Engine (tok/s) | 23.59 | 41.02 | 74.21 | 125.95 | 203.36 | 291.01 | 386.63 | 478.11 |
| vLLM 0.27.1 + MTP (tok/s) | 19.72 | 37.11 | 71.61 | 124.48 | 197.03 | 283.48 | 361.39 | 358.57 |
| Ratio | 1.196x | 1.105x | 1.036x | 1.012x | 1.032x | 1.027x | 1.070x | 1.333x |
| Metrale Engine, certified gate record at `aa5d059438` (tok/s) | 25.06 | 46.33 | 78.07 | 129.62 | 216.68 | 308.88 | 399.59 | 459.31 |
| Ratio, certified record / vLLM + MTP | 1.271x | 1.249x | 1.090x | 1.041x | 1.100x | 1.090x | 1.106x | 1.281x |

Source: [`bench/ladder38/published.json`](bench/ladder38/published.json) and
the raw per-rung files it names, plus the gate record
[`.benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json`](.benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json).
The first two rows were measured on 2026-08-17 and 2026-08-18 on a
pre-release build, on `spark-43fa`. The row for `aa5d059438` is the newest
gate record this tree carries, measured 2026-09-27 on the same instrument
on the second certification box, `spark-28c2`, and the only Metrale
Engine row you can rebuild from this repository (see
[Step 6](#step-6-run-the-concurrency-ladder-gate); it serves with W4A4
activation downcast and a different MTP K ladder, which is why it differs
from the August row). Each August cell is the mean of 3 timed reps after 1
discarded warmup; the gate row is one batch per rung after one warmup. The
instrument: ISL 128, OSL 1024, temperature 0, seed 42, context 2048, batch
cap 128, GPU memory utilization 0.85, fp8 KV cache, prefix caching on,
thinking off, MTP K=4 on both engines in the August rows, same box and same
client. In the August pair the C=4 to C=32 margins (1.2% to 3.6%) are no
larger than the rep-to-rep spread recorded in those files (up to 3.5%), so
read those rungs as parity; C=1, C=2, C=64 and C=128 are clear of that
spread. Against the same vLLM leg the certified row is ahead by 4% to 28% at
every rung. The campaign log is
[`bench/ladder38/RESULTS.md`](bench/ladder38/RESULTS.md).

<a id="moe-vs-vllm"></a>
On the 35B mixture-of-experts flagship, `Qwen/Qwen3.6-35B-A3B-FP8`, the
certified configuration is ahead of vLLM 0.27.1 with MTP from C=1 to C=8
and behind it at C=16:

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Metrale Engine, certified gate record at `aa5d059438` (tok/s) | 84.99 | 127.12 | 180.24 | 235.82 | 311.18 |
| vLLM 0.27.1 + MTP, one-shot (tok/s) | 52.11 | 93.16 | 145.94 | 222.08 | 329.93 |
| Ratio, certified record / vLLM + MTP | 1.631x | 1.364x | 1.235x | 1.062x | 0.943x |
| Metrale Engine, GPU-rail energy (J/token) | 0.570 | 0.409 | 0.313 | 0.261 | 0.215 |

Source: the gate record
[`.benchmarks/concurrency-sweep-moe/2026-09-27-aa5d059438.json`](.benchmarks/concurrency-sweep-moe/2026-09-27-aa5d059438.json)
and [`bench/baselines/qwen36-35b-a3b/published.json`](bench/baselines/qwen36-35b-a3b/published.json)
with the raw file it names, `vllm_moe_c1_16.json` (`tok_s_mean` per rung).
Both sides ran the same instrument: ISL 128, OSL 1024, essay request,
temperature 0, seed 42, context 2048, batch cap 128, GPU memory utilization
0.85, bf16 KV cache, thinking off. The Metrale Engine row is the default
serve configuration, with no serve lever and no opt-in precision flag,
measured 2026-09-27 on `spark-28c2` as one batch per rung after one warmup.
The vLLM row was measured once, on 2026-09-19 on `spark-43fa`, as the mean
of 3 reps after 1 warmup, and has not been re-run. The two engines do not
speculate alike: vLLM drafts 3 tokens per step (MTP K=4), while the gate
serves `num_drafts=1` with the MTP throughput gate on `auto` (accept length
1.65 to 1.71 at every rung). vLLM runs its Marlin FP8 MoE kernel, forced
because its default DeepGEMM path fails at weight load on GB10. At C=1, 2
and 4 the lead (23% to 63%) is several times vLLM's rep-to-rep spread at
those rungs (10.9%, 5.0%, 3.3%). At C=8 the gate figure is above all three
vLLM reps, but its 6.2% lead is about the size of that spread (6.4%), so
read C=8 as a narrow lead. At C=16 vLLM is 6.0% ahead. The vLLM manifest
records no energy, so this model has no J/token comparison.
[The MoE ladder](#the-moe-ladder) has the rest.

The tree at commit `aa5d059438` is certified: all 13 required gates pass,
backed by 23 Ed25519-signed records in [`.benchmarks/`](.benchmarks) dated
2026-09-27. The previous certification's records (2026-09-26, at
`68dd6bea35`) remain beside them.
[Reproduce the benchmarks](#reproduce-the-benchmarks) walks through
re-running them on your own hardware, and the vLLM side too.

## Contents

- <img src="docs/readme/icons/overview.svg" width="16" height="16" alt="Overview icon"> [What it is](#what-it-is)
- <img src="docs/readme/icons/hardware.svg" width="16" height="16" alt="Chip icon"> [Requirements](#requirements)
- <img src="docs/readme/icons/terminal.svg" width="16" height="16" alt="Terminal icon"> [Quick start](#quick-start)
- <img src="docs/readme/icons/gauge.svg" width="16" height="16" alt="Gauge icon"> [Reproduce the benchmarks](#reproduce-the-benchmarks)
  - [Step 1. Get the certified tree](#step-1-get-the-certified-tree)
  - [Step 2. Install the toolchain](#step-2-install-the-toolchain)
  - [Step 3. Build met as the gates build it](#step-3-build-met-as-the-gates-build-it)
  - [Step 4. Sync the recipe index](#step-4-sync-the-recipe-index)
  - [Step 5. Download the checkpoints](#step-5-download-the-checkpoints)
  - [Step 6. Run the concurrency ladder gate](#step-6-run-the-concurrency-ladder-gate)
  - [Step 7. Compare with the committed record](#step-7-compare-with-the-committed-record)
  - [The MoE ladder](#the-moe-ladder)
  - [The single-stream decode floor](#the-single-stream-decode-floor)
  - [Verify the signed records](#verify-the-signed-records)
  - [The vLLM baseline on the same box](#the-vllm-baseline-on-the-same-box)
  - [Advanced: the full 17-gate certification](#advanced-the-full-17-gate-certification)
- <img src="docs/readme/icons/layers.svg" width="16" height="16" alt="Layers icon"> [Architecture at a glance](#architecture-at-a-glance)
- <img src="docs/readme/icons/shield.svg" width="16" height="16" alt="Shield icon"> [Accuracy and correctness gates](#accuracy-and-correctness-gates)
- <img src="docs/readme/icons/lock.svg" width="16" height="16" alt="Lock icon"> [Security](#security)
- <img src="docs/readme/icons/licence.svg" width="16" height="16" alt="Document icon"> [Licence](#licence)
- <img src="docs/readme/icons/people.svg" width="16" height="16" alt="People icon"> [Appendix: people](#appendix-people)
- <img src="docs/readme/icons/link.svg" width="16" height="16" alt="Link icon"> [Links](#links)

<a id="what-it-is"></a>
## <img src="docs/readme/icons/overview.svg" width="20" height="20" alt="Overview icon"> What it is

- **One binary.** `met` (crate `metrale-server`, version `1.0.0-beta-preview`)
  loads Hugging Face safetensors checkpoints, runs the scheduler and serves
  `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`
  and the Anthropic `/v1/messages` endpoint. It binds
  `127.0.0.1:8888` by default; `--bind 0.0.0.0` exposes it, with
  `--require-auth` for bearer tokens.
- **Kernels per target.** `crates/kernels/build.rs` compiles
  `kernels/<hardware>/<model>/<quant>/*.cu` over `kernels/<hardware>/common/`
  into PTX and embeds it. At startup the server picks the target from the
  checkpoint's `model_type` and `hidden_size`. There is no per-request kernel
  selection. `kernels/gb10/` carries 30 model directories.
- **Speculative decoding.** MTP heads (`--speculative`), DFlash block
  diffusion with a drafter checkpoint (`--dflash`), self-speculation and an
  n-gram proposer. A throughput gate switches MTP off while it is slower.
- **Hybrid models.** GDN and Mamba recurrent state is managed like the KV
  cache: per-token snapshots so a rejected draft can roll back, and prefix
  caching that restores SSM state as well as KV.
- **Signed benchmarks.** `met bench run <id> --pull-request-gate` measures a
  checkpoint under a pinned recipe and writes a signed record into
  `.benchmarks/`. The `PR benchmark gate` CI job fails a pull request unless
  every required gate has a passing record that still covers the tree.

The [book](https://docs.metrale.ai) covers the design in depth; the API
reference is at [docs.metrale.ai/api](https://docs.metrale.ai/api/).

<a id="requirements"></a>
## <img src="docs/readme/icons/hardware.svg" width="20" height="20" alt="Chip icon"> Requirements

### Hardware and software

| | Requirement | Where it is stated |
|---|---|---|
| GPU | NVIDIA GB10 (DGX Spark), compute capability 12.1. Kernels are compiled for `sm_121f`. | [`kernels/gb10/HARDWARE.toml`](kernels/gb10/HARDWARE.toml) |
| Memory | Unified LPDDR5X shared by CPU and GPU. Both certification boxes report 121.7 GiB (`mem_total_kb` 127,601,452 and 127,600,752). A gate refuses to start its server unless at least 85% of it is available, so benchmark on an idle box. | 2026-09-27 records; `[benchmarks.limits.memory]` in `HARDWARE.toml` |
| OS | Ubuntu 24.04 (DGX OS). The GB10 image and the release tarball are built on Ubuntu 24.04, glibc 2.39. | [`docker/gb10/Dockerfile`](docker/gb10/Dockerfile), [`docs/GB10_DEPLOYMENT_GUIDE.md`](docs/GB10_DEPLOYMENT_GUIDE.md) |
| NVIDIA driver | 580 or newer. The certification ran on 580.126.09 (`spark-43fa`) and 580.159.03 (`spark-28c2`). | `GB10_DEPLOYMENT_GUIDE.md`; `hardware.driver` in the records |
| CUDA toolkit | 13.0 with `nvcc` on `PATH`. The certified kernels were compiled by nvcc 13.0.88 (`cuda_13.0.r13.0/compiler.36424714_0` in each record's `closure`). | 2026-09-27 records |
| Rust | 1.93.1, pinned in [`rust-toolchain.toml`](rust-toolchain.toml); rustup installs it on the first `cargo` call. | `rust-toolchain.toml` |
| Build packages | `build-essential pkg-config git cmake libclang-dev libibverbs-dev` | build stage of `docker/gb10/Dockerfile` |
| Containers | Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), for `metralectl` and the vLLM baseline. | [Metrale/metralectl README](https://github.com/Metrale/metralectl) |
| Python | `python3` and `python3-venv`, with `aiohttp` for the ladder harness, `cryptography` to check signatures without `met`, and the `hf` CLI from `huggingface_hub` for downloads. | [`bench/ladder38/harness_w55_conc_ladder.py`](bench/ladder38/harness_w55_conc_ladder.py) |
| Network | `huggingface.co` (checkpoints); `api.github.com` and `raw.githubusercontent.com` (`met sync-recipes` reads the recipe library in [Metrale/metralectl](https://github.com/Metrale/metralectl)); Docker Hub (images); crates.io (the build, and the agentic gate's generated projects); `metrale.ai` (installer). | [`crates/server/src/recipe/fetch.rs`](crates/server/src/recipe/fetch.rs) |

### Disk

Checkpoint sizes are the Hugging Face repository totals (GB = 10^9 bytes).
None of the repositories is gated.

| Item | Size | Used by |
|---|---:|---|
| `unsloth/Qwen3.8-27B-NVFP4` | 23.44 GB | concurrency ladder, decode floor, vLLM baseline; also `bfcl-subset`, `kat-equality-gate`, `concurrency-sweep-dflash2` |
| `Qwen/Qwen3.6-35B-A3B-FP8` | 37.49 GB | MoE ladder, quick start; also `agentic-webserver`, `vision-fidelity`, both TTFT gates, `bfcl-subset-echolp`, `ssm-state-poisoning-gate` |
| `unsloth/Qwen3.6-27B-NVFP4` | 23.44 GB | `video-fidelity` (full certification only) |
| `incoai/Qwen3.8-27B-DFlash2` | 3.85 GB | drafter for `concurrency-sweep-dflash2` (full certification only) |
| `vllm/vllm-openai:v0.27.1`, arm64 | 10.53 GB compressed download | vLLM baseline |
| `target/release` | about 5 GB | the `met` build |

The three benchmarks walked through below need 60.93 GB of checkpoints; the
full certification needs 88.22 GB. With the build, the vLLM image unpacked
and room for run artifacts, keep at least 150 GB free.

### Other hardware targets

The tree also builds kernel sets for other hardware. They are built from
source and are not covered by the GB10 certification.

| Target | Directory | Architecture | Model sets |
|---|---|---|---:|
| NVIDIA H100 / H200 | `kernels/hopper` | `sm_90a` | 8 |
| NVIDIA B200 | `kernels/b200` | `sm_100a` | 7 |
| NVIDIA B300 | `kernels/b300` | `sm_103a` | 1 (Kimi K3 bring-up) |
| AMD Strix Halo, through SCALE | `kernels/strix` | `gfx1151` | 2 |
| AMD Strix Halo, native HIP | `kernels/strix-hip` | `gfx1151` | 2 |
| Apple Silicon | `kernels/metal` | `metal3.1` | 2 |

[`docs/HARDWARE.md`](docs/HARDWARE.md) describes each target and how to add
one. Select a target with `METRALE_TARGET_HW=<directory>` at build time.

<a id="quick-start"></a>
## <img src="docs/readme/icons/terminal.svg" width="20" height="20" alt="Terminal icon"> Quick start

`metralectl` launches validated recipes (container image, checkpoint and
serve settings). The installer puts it in `~/.local/bin`, refuses a download
whose checksum is not in the release's `SHA256SUMS`, and sets up its
background agent.

```bash
curl -fsSL https://metrale.ai/install.sh | sh
metralectl list                                     # the recipes
metralectl run qwen3.6-35b-a3b-fp8-mtp --print      # print the docker command without running it
metralectl run qwen3.6-35b-a3b-fp8-mtp              # serve Qwen3.6-35B-A3B-FP8 on port 8888
```

`uvx metralectl list` runs it without installing. `metralectl run` needs
Docker with the NVIDIA Container Toolkit; this recipe pulls
`metrale/metrale-inference-gb10:latest` and the 37.49 GB checkpoint.

From source, with the [requirements](#requirements) above in place
(`met serve` reads the Hugging Face cache and does not download; the first
build takes 15 to 30 minutes):

```bash
git clone https://github.com/Metrale/metrale-inference.git
cd metrale-inference
export PATH=/usr/local/cuda/bin:$PATH
cargo build --release --bin met
pip install -U huggingface_hub                     # provides the `hf` CLI
hf download Qwen/Qwen3.6-35B-A3B-FP8               # 37.49 GB into ~/.cache/huggingface/hub
target/release/met serve Qwen/Qwen3.6-35B-A3B-FP8 --max-seq-len 16384
```

Either way, send a request:

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

On a terminal `met serve` opens a dashboard; `--no-tui` keeps the plain log.
[`QUICKSTART.md`](QUICKSTART.md) has per-model commands and `met serve --help`
lists every flag.

<a id="reproduce-the-benchmarks"></a>
## <img src="docs/readme/icons/gauge.svg" width="20" height="20" alt="Gauge icon"> Reproduce the benchmarks

This section re-measures the certified numbers with the same runner that
produced them, `met bench run … --pull-request-gate`, and then measures vLLM
on the same box. `met bench` is short for `met benchmark`; both spellings work
everywhere. Nothing here needs access to our machines.

A gate run starts its own server from the benchmark's recipe on a free port,
applies the pins in the model's `BENCH.toml` (serve overrides, request
parameters, floors), measures, stops the server and writes a signed record.
The concurrency ladder's pins live in
[`kernels/gb10/qwen3.8-27b/BENCH.toml`](kernels/gb10/qwen3.8-27b/BENCH.toml).

### Step 1. Get the certified tree

```bash
git clone https://github.com/Metrale/metrale-inference.git
cd metrale-inference
git checkout --detach certified-2026-09-27      # tag on aa5d05943868049527c4cb4e48513bacd134e726
```

`aa5d059438` is the commit every 2026-09-27 record names in its `git_sha`;
the tag `certified-2026-09-27` points at it. If your clone lacks the tag,
`git fetch origin refs/pull/34/head` brings the commit in. The records were
committed on top of it and reached `main` by squash merge;
`git diff --stat certified-2026-09-27 origin/main` shows what `main` adds.
The commands below read the records from `main` with
`git show origin/main:<path>`. At the measured commit itself no record is
present yet, so every gate is owed again, which is what you want when
re-measuring.

### Step 2. Install the toolchain

```bash
sudo apt-get install -y build-essential pkg-config git cmake libclang-dev libibverbs-dev curl python3 python3-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
nvcc --version      # release 13.0; V13.0.88 compiled the certified kernels
rustc --version     # rustc 1.93.1, selected by rust-toolchain.toml
python3 -m venv ~/metrale-venv && . ~/metrale-venv/bin/activate
pip install -U huggingface_hub aiohttp cryptography
```

The later steps assume this shell: CUDA on `PATH` and the virtual
environment active.

### Step 3. Build met as the gates build it

The bench nodes build with `cargo build --release --bin met` from the
repository root. With `METRALE_TARGET_HW`, `METRALE_TARGET_MODEL` and
`METRALE_TARGET_QUANT` unset, `crates/kernels/build.rs` compiles its
defaults: hardware `gb10`, every model directory, quantization `nvfp4`. That
is the kernel set the 2026-09-27 records list in their `closure` field.

```bash
env -u METRALE_TARGET_HW -u METRALE_TARGET_MODEL -u METRALE_TARGET_QUANT \
    -u METRALE_SKIP_BUILD -u METRALE_EXTRA_NVCC_FLAGS \
  cargo build --release --bin met
target/release/met --version        # met 1.0.0-beta-preview
```

[`QUICKSTART.md`](QUICKSTART.md) puts a first build at 15 to 30 minutes; PTX
compilation for every GB10 target dominates. Step 7 checks that your kernels
hash to the certified ones.

### Step 4. Sync the recipe index

```bash
target/release/met sync-recipes
target/release/met doctor
```

A gate resolves its recipe (for example `qwen3.8/qwen3.8-27b-nvfp4-throughput`)
from a local index, and `sync-recipes` fills it from the `recipes/` tree of
[Metrale/metralectl](https://github.com/Metrale/metralectl). It is a separate
command so that a benchmark never reaches the network mid-run. It prints the
path of the index, the recipe count and the tree sha it read. The same recipes
are in this repository under [`recipes/`](recipes/), where CI checks each one
against the `met` built from the same commit. `met doctor`
checks the box. On a new machine its `identity` line reads `no signing key
yet — one is minted on this box's first gate record`; after your first gate
run it will say the key is not committed in `.github/record-signers/`, which
matters only for records you intend to merge.

### Step 5. Download the checkpoints

```bash
hf download unsloth/Qwen3.8-27B-NVFP4      # 23.44 GB: concurrency ladder, decode floor, vLLM baseline
hf download Qwen/Qwen3.6-35B-A3B-FP8       # 37.49 GB: MoE ladder
```

Both land in `~/.cache/huggingface/hub`, which `met` reads by default
(`--cache-dir`, `$HF_HUB_CACHE` and `$HF_HOME` override it) and which the vLLM
container mounts below.

### Step 6. Run the concurrency ladder gate

The gate's `[benchmarks.serve_env]` table in `BENCH.toml` declares three
serve levers. A gate that serves in-process cannot set them after it has
started, so pass them on the command line, exactly:

```bash
env METRALE_FP8_ROWWISE=1 \
    METRALE_MTP_DCUT_RATIO=1.0 \
    METRALE_MTP_K_LADDER=1:3,2:2,4:1,8:1,16:1 \
  target/release/met bench run concurrency-sweep \
    --pull-request-gate --hardware gb10 --checkpoint unsloth/Qwen3.8-27B-NVFP4
```

The runner refuses any other `METRALE_*` serve lever in its environment, and
refuses these three at any other value, so start from a clean shell. Nothing
else may hold the GPU: the certified run's server held 103,308 MiB. With
`--serve-reuse` instead, the runner starts the server as a child process and
hands it the declared levers itself.

What the runner applies from `BENCH.toml` without being asked:

- **Serve overrides** on the `qwen3.8/qwen3.8-27b-nvfp4-throughput` recipe:
  `kv_cache_dtype=fp8`, `max_batch_size=128`, `max_model_len=2048`,
  `prefill_codispatch=true`, `w4a4_downcast=true`, `w4a4_downcast_wide=true`.
- **Request parameters:** concurrencies 1, 2, 4, 8, 16, 32, 64, 128; ISL
  128; OSL 1024; `prompt_mode=essay`, which sends the published ladder's
  request byte for byte.
- **Floors** per rung in aggregate tok/s, and energy ceilings in joules per
  token.

This is the certified gate configuration. It is not the command behind the
published ladder above: it adds W4A4 activation downcast (a numerics change,
opt-in per recipe) and uses a different MTP K ladder. Its floors guard
against regression. The ratio against vLLM comes from the ladder files.

**How long.** The 2026-09-27 run measured for 1,663 s (27.7 minutes, from the
record's `hardware_state`), plus the model load. `met bench list` gives the
range as 25 to 90 minutes. The widest rungs are slow to report: a C=128 rung
takes several minutes (its energy window in the record is 285 s) and its
TTFT p50 was 33 s, so long gaps between progress lines there are not a hang.

**What it prints.** Progress goes to stderr. The record path is printed as
soon as it is written:

```text
gate record written as <repo>/.benchmarks/concurrency-sweep/<YYYY-MM-DD>-aa5d059438.json
                  and <repo>/.benchmarks/concurrency-sweep/<YYYY-MM-DD>-aa5d059438.json.sig
```

The report follows on stdout. For the certified run it opened with:

```text
  Peak throughput          459.3tok/s
  at concurrency           128
  Best TTFT p50            399ms
  Cells                    8/8
```

then a latency and throughput table with one row per rung (columns `ISL`,
`Conc`, `TTFT p50`, `p90`, `p99`, `TPOT p50`, `p90`, `E2E p50`, `tok/s`,
`min tok`, `min cache%`, `err`), and closed with the verdict:

```text
  Pass: 8 cells, zero errors, zero vacuous — every populated floor met (C1 25.1/22.0 · C2 46.3/38.0 · C4 78.1/67.0 · C8 129.6/110.0 · C16 216.7/180.0 · C32 308.9/260.0 · C64 399.6/360.0 · C128 459.3/440.0 · peak 459.3/440.0)
```

These lines are rebuilt from the committed record with the print formats in
`crates/server/src/cli/bench_print.rs` and
`crates/bench/src/benchmarks/concurrency_report.rs`; the verdict line is the
record's `verdict_reason`. The per-rung values in the record,
[`.benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json`](.benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cN_aggregate_tok_s` | 25.06 | 46.33 | 78.07 | 129.62 | 216.68 | 308.88 | 399.59 | 459.31 |
| Floor in `BENCH.toml` | 22 | 38 | 67 | 110 | 180 | 260 | 360 | 440 |
| `cN_tpot_p50_ms` | 39.5 | 42.7 | 49.8 | 56.9 | 67.9 | 94.1 | 143.9 | 246.0 |
| `cN_ttft_p50_ms` | 399 | 534 | 1,063 | 2,112 | 4,202 | 8,403 | 16,679 | 33,232 |
| `cN_accept_len` | 2.27 | 2.02 | 1.67 | 1.63 | 1.62 | 1.63 | 1.00 | 1.00 |
| `cN_gpu_rail_joules_per_token` | 1.46 | 0.81 | 0.50 | 0.32 | 0.21 | 0.17 | 0.15 | 0.13 |

All requests of a rung arrive at once, so TTFT at high concurrency includes
queueing behind the other prefills. An accept length of 1.00 at C=64 and
C=128 means no draft token was accepted at those rungs.

The run also writes its history to `~/.metrale/runs` (or `$METRALE_HOME/runs`).
The record carries the metrics, the verdict, the command line, the serve
overrides and levers, the hardware fingerprint with before and after thermal
captures, the kernel closure hashes and the commit sha. The `.sig` beside it
is an Ed25519 signature over the record bytes and that sha, made with a key
generated on first use under `~/.metrale/identity/`. The first record also
writes that key's public half to `.github/record-signers/<fingerprint>.pub`.

### Step 7. Compare with the committed record

```bash
git show origin/main:.benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json > certified.json
python3 - certified.json .benchmarks/concurrency-sweep/<your-record>.json <<'EOF'
import json, sys
ref, new = (json.load(open(p)) for p in sys.argv[1:3])
for c in (1, 2, 4, 8, 16, 32, 64, 128):
    k = f"c{c}_aggregate_tok_s"
    a, b = ref["metrics"][k], new["metrics"][k]
    print(f"C={c:<4} committed {a:7.1f}   yours {b:7.1f}   {b / a:.3f}x")
t = "gb10/qwen3.8-27b/nvfp4"
same = ref["closure"][t]["hash"] == new["closure"][t]["hash"]
print(f"{t} kernel closure: {'identical' if same else 'DIFFERENT'}; verdict {new['verdict']}")
EOF
```

The closure hash covers every source file the target's device code is
compiled from, the manifests, the nvcc flags, the architecture and the
compiler string (`crates/closure`). `identical` means your binary runs the
kernels that were certified. A throughput difference then comes from the
box: its clocks, temperature and background load are in your record's
`hardware_state`, beside the certified one's.

### The MoE ladder

`concurrency-sweep-moe` runs the same driver on the 35B mixture-of-experts
flagship, `Qwen/Qwen3.6-35B-A3B-FP8`, on the published instrument (ISL 128,
OSL 1024, essay request) at C=1 to 16. Its pins are in
[`kernels/gb10/qwen3.6-35b-a3b/BENCH.toml`](kernels/gb10/qwen3.6-35b-a3b/BENCH.toml)
and it declares no serve levers.

```bash
target/release/met bench run concurrency-sweep-moe \
  --pull-request-gate --hardware gb10 --checkpoint Qwen/Qwen3.6-35B-A3B-FP8
```

The certified run measured for 277 s. It serves recipe
`qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head` with `kv_cache_dtype=bf16`,
`max_model_len=2048`, `max_batch_size=128`, `gpu_memory_utilization=0.85`,
`scheduler=fifo`, `num_drafts=1`, `disable_thinking=true` and
`ssm_cache_slots=32`.

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Metrale Engine, gate record at `aa5d059438` (tok/s) | 84.99 | 127.12 | 180.24 | 235.82 | 311.18 |
| Floor in `BENCH.toml` | 69.47 | 80.88 | 92.72 | 101.71 | 102.63 |
| `cN_accept_len` | 1.71 | 1.68 | 1.68 | 1.65 | 1.65 |
| `cN_gpu_rail_joules_per_token` | 0.570 | 0.409 | 0.313 | 0.261 | 0.215 |
| Metrale Engine, gate record at `68dd6bea35` (tok/s) | 76.15 | 94.21 | 106.81 | 113.17 | 117.79 |
| vLLM 0.27.1 + MTP, one-shot (tok/s) | 52.11 | 93.16 | 145.94 | 222.08 | 329.93 |

Sources:
[`.benchmarks/concurrency-sweep-moe/2026-09-27-aa5d059438.json`](.benchmarks/concurrency-sweep-moe/2026-09-27-aa5d059438.json),
[`.benchmarks/concurrency-sweep-moe/2026-09-26-68dd6bea35.json`](.benchmarks/concurrency-sweep-moe/2026-09-26-68dd6bea35.json)
and [`bench/baselines/qwen36-35b-a3b/published.json`](bench/baselines/qwen36-35b-a3b/published.json)
(raw file `vllm_moe_c1_16.json`, measured once on 2026-09-19, with vLLM
forced onto its Marlin FP8 MoE path because its default DeepGEMM path fails
at weight load on GB10).

The two Metrale Engine records serve the same recipe with the same
overrides, on different boxes (`spark-43fa` on 2026-09-26, `spark-28c2` on
2026-09-27). The engine changes between their commits are these four, now
the default for FP8 MoE checkpoints:

- **Canonical row tiers.** The W8A16 projections, the NVFP4 LM head and the
  FP8 MoE (grouped kernels with a per-row router) each sum a row in one
  order at every batch width, so greedy output does not depend on how many
  rows share a launch. `--no-canonical-tiers` restores the row-count tiers.
- **Grouped FP8 expert decode.** One block per active expert reads that
  expert's weights once for every row routed to it, byte-identical to the
  per-row kernels.
- **GDN carried-state MTP verify.** The batched verify reads and writes each
  sequence's recurrent state once.
- **A width-aware MTP gate.** The throughput gate's refresh interval scales
  with batch width, and it lets a width change settle before it switches.

Against vLLM that puts Metrale Engine ahead from C=1 to C=8 and at 0.94x at
C=16; the [MoE table at the top](#moe-vs-vllm) has the ratios and how to
read the C=8 margin. The floors still guard regression, not parity: they
were cut on 2026-09-23 from Metrale Engine's own curve, as the `BENCH.toml`
note says directly (its comparison with vLLM predates this certification).
No Metrale Engine leg
has been run under the vLLM manifest's exact serve profile (its parity note
requires MTP K=4; the gate serves `num_drafts=1`), so the manifest itself
scores no pair; the ratios read a gate record against the one-shot, rung by
rung.

`--moe-nvfp4-experts` is an opt-in and not part of the certified
configuration. It adds an NVFP4 copy of the routed experts at load and
decodes with it, so a decode step reads half the routed-expert bytes, and
the model's answers change. Its help text in
[`crates/server/src/cli/serve_args.rs`](crates/server/src/cli/serve_args.rs)
and the recipe
[`recipes/qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head-nvfp4experts.yaml`](recipes/qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head-nvfp4experts.yaml)
state what was measured (2026-09-27, GB10, canonical tiers): on one BFCL
echolp shard (N=253), overall/normalized 86.56/88.86 against 85.38/87.72
with FP8 experts; `agentic-webserver` passed 10/10 but took 168 turns and
774 s of summed wall against 128 turns and 545 s, over that gate's 700 s
ceiling. No gate record measures it, so this README gives no speed for it.

### The single-stream decode floor

`decode-floor` makes three timed streaming runs of one fixed code prompt and
gates the median server-reported decode rate.

```bash
target/release/met bench run decode-floor \
  --pull-request-gate --hardware gb10 --checkpoint unsloth/Qwen3.8-27B-NVFP4
```

The certified record,
[`.benchmarks/decode-floor/2026-09-27-aa5d059438.json`](.benchmarks/decode-floor/2026-09-27-aa5d059438.json),
reads `server_decode_tok_s` 27.64 with an MTP accept length of 2.72 and 818
output tokens, against a floor of 25.5 with a declared noise of 0.5 (the
verdict prints the noise-adjusted bar, 25.0); its
verdict reads "median decode 27.6 tok/s over 3 pinned runs (accept_len_mean
2.72) — clears the 25.0 tok/s floor". It measured for 107 s.

### Verify the signed records

The gate check is what CI runs on every pull request. It needs no GPU and no
server. The check diffs each record's commit against `HEAD`, so `aa5d059438`
must be present in the clone (Step 1 brings it in); in a clone without it,
every gate reads `NONE` with `git cannot diff that commit`. Run it where the
records are, on `main`:

```bash
git worktree add ../metrale-inference-main origin/main
cd ../metrale-inference-main
../metrale-inference/target/release/met benchmark --pull-request-gate-check
```

At the commit that added the 2026-09-27 records it prints:

```text
gate check for <sha> (<path>)
  PASS  agentic-webserver
  PASS  vision-fidelity
  PASS  video-fidelity
  PASS  ttft-warm-gate
  PASS  ttft-cold-gate
  PASS  bfcl-subset
  PASS  bfcl-subset-echolp
  PASS  ssm-state-poisoning-gate
  PASS  decode-floor
  PASS  concurrency-sweep
  PASS  concurrency-sweep-dflash2
  PASS  kat-equality-gate
  PASS  concurrency-sweep-moe

intent: not evaluated (no --pr)
all 13 required gates pass
```

For each gate it looks for a record whose commit differs from `HEAD` only
outside that gate's invalidation paths, requires a completed run with a
passing verdict (for the BFCL groups, a complete partition of shard
records), and verifies the signature against a public key committed in
[`.github/record-signers/`](.github/record-signers). The 2026-09-27 records
are signed by `efe157d6f6360027` (host `spark-43fa`) and `02156264cbf75bd7`
(host `spark-28c2`).

To check a signature without `met`, from the repository root:

```bash
python3 - .benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json <<'EOF'
import base64, hashlib, json, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
rec = sys.argv[1]
body = open(rec, "rb").read()
sig = json.load(open(rec + ".sig"))
key_line = next(l for l in open(f".github/record-signers/{sig['key']}.pub")
                if l.strip() and not l.startswith("#"))
pub = base64.b64decode(key_line.strip())
assert hashlib.sha256(pub).hexdigest()[:16] == sig["key"]
sha = json.loads(body)["git_sha"]
Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(sig["sig"]), body + sha.encode())
print(f"OK  {rec}  signed by {sig['key']} for commit {sha}")
EOF
```

For that record it prints:

```text
OK  .benchmarks/concurrency-sweep/2026-09-27-aa5d059438.json  signed by 02156264cbf75bd7 for commit aa5d059438
```

A signature proves that the record file and the commit it names are the ones
the key holder signed: editing either breaks it. It does not prove that the
benchmark ran, or ran at that speed. Keys are generated on the benchmark box,
and no signature scheme can witness a wall clock. Timing numbers are signed
claims that an independent re-run can check, which is why this section
exists. [`docs/provable-benchmark-work.md`](docs/provable-benchmark-work.md)
sets out that threat model.

### The vLLM baseline on the same box

[`bench/ladder38/published.json`](bench/ladder38/published.json) records the
`vllm-mtp` series as the image digest (`vllm/vllm-openai:latest @
sha256:0a51ea5b…`, tag `v0.27.1`), the environment (`HF_HUB_OFFLINE=1`) and
the `vllm serve` arguments. The `metrale` series' `cli` field is the August
command line for that build and is kept as recorded; four of its flags
(`--host`, `--scheduling-policy`, `--disable-tool-grammar`,
`--ssm-tail-midchunk`) are not accepted by this tree's `met serve`, so use
the gate in [Step 6](#step-6-run-the-concurrency-ladder-gate) to serve the
Metrale Engine side rather than that string. The image's entrypoint is
`vllm serve`, so as a container command the vLLM arguments read:

```bash
docker run --rm --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967 \
  --model unsloth/Qwen3.8-27B-NVFP4 --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8001 --max-model-len 2048 --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`HF_HUB_OFFLINE=1` means the checkpoint must already be in the mounted cache
([step 5](#step-5-download-the-checkpoints)). Stop every other GPU process
first. When the server answers on port 8001, drive it with the published
ladder's harness:

```bash
python3 bench/ladder38/harness_w55_conc_ladder.py \
  --url http://127.0.0.1:8001 --model unsloth/Qwen3.8-27B-NVFP4 \
  --label vllm_mtp --out vllm_mtp.json \
  --concs 1,2,4,8,16,32,64,128 --reps 3 --isl 128 --osl 1024 --warmup 1
```

Each rep prints one line (`[vllm_mtp] C=  1 rep0  tok/s= …`); the reference
leg's `tok_s_mean` per rung is the vLLM row of the dense table at the top. Expect
roughly an hour for all eight rungs.

Every measurement argument is required; the harness defaults nothing. It
prints its own sha256 first and writes it into the output as
`driver_sha256`, together with each rep's raw values and each rung's
`tok_s_mean`. The copy in this tree hashes `4649c9b001…`, a later revision
than the one that drove the published vLLM legs (`6412b12d4d`). It sends
`presence_penalty` and `frequency_penalty` of 0.0 explicitly; vLLM defaults
both to 0.0, so its sampling is unchanged (`harness_shas.equivalence` in
`published.json`). The reference vLLM leg took
1 h 05 min (`started_utc` to `finished_utc` in `vllm_fp8_mtp_reference.json`).

`published.json` also records that vLLM + MTP has taken a GB10 down at the
widest rungs, with power cycles; its energy re-run stopped at C=16 for that
reason. Run `--concs 1,2,4,8,16` first if the box matters to you.

**Reading the two sides together.** The `vllm-mtp` series declares its
instrument (ISL 128, OSL 1024, essay prompt, context 2048, batch cap 128,
fp8 KV, temperature 0, seed 42) so that a `concurrency-sweep` gate record
can be read against it rung by rung. The harness reports the mean of three
reps; the gate measures one batch per rung after one warmup; both send the
same request. To drive the certified Metrale Engine configuration with the
harness as well, keep the gate's server up and point the harness at it:

```bash
target/release/met bench run concurrency-sweep --serve-reuse \
  --pull-request-gate --hardware gb10 --checkpoint unsloth/Qwen3.8-27B-NVFP4
PORT=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/.metrale/serve-lease.json")))["port"])')
python3 bench/ladder38/harness_w55_conc_ladder.py \
  --url http://127.0.0.1:$PORT --model unsloth/Qwen3.8-27B-NVFP4 \
  --label metrale_gate --out metrale_gate.json \
  --concs 1,2,4,8,16,32,64,128 --reps 3 --isl 128 --osl 1024 --warmup 1
target/release/met bench serve-release
```

`--serve-reuse` leaves the server running after the gate and records its
port in `serve-lease.json` under `~/.metrale` (or `$METRALE_HOME`);
`serve-release` stops it.

### Advanced: the full 17-gate certification

`met bench certify` runs every required gate the current commit does not yet
have a passing record for, then applies the same check CI applies.

```bash
sudo apt-get install -y ffmpeg      # video-fidelity decodes its MP4 clips through ffmpeg
git add .github/record-signers/     # the key your first gate run wrote; certify refuses an unregistered signer
target/release/met bench certify --hardware gb10 --no-guard --dry-run    # the plan and the preflight
target/release/met bench certify --hardware gb10 --no-guard --yes
```

- **What it runs.** The gates listed under
  [Accuracy and correctness gates](#accuracy-and-correctness-gates) that
  have no passing record at this commit yet; the ones you ran in step 6
  onwards are skipped. Four checkpoints (88.22 GB) cover all 17. The two
  BFCL groups run their whole draw on one box, or split into shards across
  several with `--with-nodes`.
  `--yes` confirms `agentic-webserver`, which executes model-authored shell
  commands in a sandbox. `--no-guard` turns off the watch on an upstream
  branch, which a detached checkout does not have.
- **Preflight.** Before spending any GPU time it checks that `HEAD` is the
  anchor, no invalidation-path file is uncommitted, the signing key is
  registered, `METRALE_HOME` is writable, no other `met` is running and
  enough memory is free.
- **How long.** The 2026-09-27 campaign ran on two GB10s in 3 h 04 min of
  wall clock, first to last hardware capture across its 23 records; the
  campaign driver itself ran 3 h 07 min (17:31 to 20:38 UTC), preflight and
  final check included. Its measurement windows sum to 5.9 hours, which is
  roughly what one box needs, plus model loads.
- **What "certified" means.** Exit code 0: every required gate has a passing
  record at the anchor commit, and the records the campaign added agree (one
  commit, and one signer across the speed-class gates). Exit code 2 means a
  verdict failed; 3 means the campaign was aborted. Commit the new files in
  `.benchmarks/` and `.github/record-signers/` to keep them.

The [certification chapter](book/src/operations/certify.md) of the book
([online](https://docs.metrale.ai/operations/certify.html)) covers
multi-node campaigns, thermal parking, sharding and the lockfile.

<a id="architecture-at-a-glance"></a>
## <img src="docs/readme/icons/layers.svg" width="20" height="20" alt="Layers icon"> Architecture at a glance

A Cargo workspace of 21 crates:

| Layer | Crates |
|---|---|
| Surface | `server` (the `met` binary: HTTP API, scheduler loop, tokenizer and chat templates, tool-call parsers, dashboard, CLI), `bench` (benchmark registry, gate records, signing, certification) |
| Model | `model-engine` (the `Model` trait; prefill, decode, verify, SSM state), `model-arch` (per-family architectures and loaders), `model-layers` (attention, SSM, MoE, FFN, MTP heads, vision, LoRA), `model-weights` (weight store, fast loader, preflight) |
| Serving | `scheduler` (step plans and the driver loop), `cache` (paged KV, radix-tree prefix cache), `speculative` (MTP gate, DFlash, n-gram), `sampling`, `grammar` (pure-Rust grammar-constrained decoding), `storage` (GDS and RDMA tiers, expert paging) |
| GPU | `kernels` (PTX compiled from `kernels/<hw>/<model>/<quant>/`), `gpu-runtime` (backend, streams, buffers, kernel registry), `gpu-sys` (raw FFI: cuFile, NVML, NCCL, RDMA verbs), `comm` (collectives), `telemetry` (metrics, GPU spans, energy attribution) |
| Foundation | `core` (shared types), `config` (model config tree), `closure` (kernel source hashing for records), `governance` (the PR journey ledger) |

A chat completion takes this path:

1. Axum receives `POST /v1/chat/completions` (`crates/server/src/api/chat/`).
2. The chat template is rendered, the prompt tokenized, and sampling, stops
   and grammar resolved.
3. The request joins the scheduler queue.
4. Each scheduler step admits and prefills new sequences, drafts (MTP,
   n-gram or DFlash) and verifies, and emits accepted tokens over SSE.
5. Each layer launches its kernels through handles resolved once at startup
   from the embedded PTX: KV cache write, paged attention, SSM state update,
   MoE routing and expert GEMV.
6. Logits are sampled and the token is streamed back.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is a five-minute tour; the
[book](https://docs.metrale.ai) has one chapter per crate and deep dives on
NVFP4, MTP, SSM layers and attention.

<a id="accuracy-and-correctness-gates"></a>
## <img src="docs/readme/icons/shield.svg" width="20" height="20" alt="Shield icon"> Accuracy and correctness gates

A merge needs a passing record for each of these 17 gates, the `REQUIRED`
list in [`crates/bench/src/gate/coverage.rs`](crates/bench/src/gate/coverage.rs).
A record stays valid until a change touches that gate's invalidation paths.
Results are from the 2026-09-27 records at `aa5d059438`.

| Gate | What it checks | Checkpoint | 2026-09-27 result |
|---|---|---|---|
| `agentic-webserver` | Ten agentic runs, each building a working Axum web server with tools, then verified | Qwen3.6-35B-A3B-FP8 | 10/10 `webserver_ok`, 10/10 `followed_directions`, 4.211 s per turn (limit 8.5), 455 s total (limit 700) |
| `vision-fidelity` | Exact vision-token geometry across a resolution ladder, capability probes, a no-image control | Qwen3.6-35B-A3B-FP8 | 14 geometry cells matched, 3/3 probes, control held |
| `video-fidelity` | Temporal order of a colour sequence, group-count geometry, MP4/GIF decoder parity, a no-video control | Qwen3.6-27B-NVFP4 | 14/14 legs passed, control held, none skipped |
| `ttft-warm-gate` | Cached-prefix time to first token against the stored same-box baseline | Qwen3.6-35B-A3B-FP8 | median −51.8% (limit +3%), p90 −52.8% (limit +5%) |
| `ttft-cold-gate` | Uncached prefill time to first token against the stored baseline | Qwen3.6-35B-A3B-FP8 | median −15.1% (limit +3%), p90 −7.6% (limit +5%) |
| `bfcl-subset` | Function-calling accuracy on the fixed n=995 MLPerf-edge BFCL draw, AST-scored, run as shards | Qwen3.8-27B-NVFP4 | overall 82.91 (floor 82.6), normalized 83.95 (floor 82.56), 995 samples |
| `bfcl-subset-echolp` | The same on the n=1004 draw | Qwen3.6-35B-A3B-FP8 | overall 84.86 (floor 84.16), normalized 85.93 (floor 85.37), 1004 samples |
| `ssm-state-poisoning-gate` | Replayed conversations must come back byte-identical to the reference | Qwen3.6-35B-A3B-FP8 | 12 of 12 replays byte-identical |
| `decode-floor` | Median single-stream decode rate over three pinned runs | Qwen3.8-27B-NVFP4 | 27.6 tok/s, floor 25.0 |
| `concurrency-sweep` | Aggregate throughput floors at C=1 to 128, no errors, no vacuous cells | Qwen3.8-27B-NVFP4 | all 8 rungs above floor, peak 459.3 tok/s |
| `concurrency-sweep-dflash2` | The same at C=1 to 16 with the DFlash2 drafter | Qwen3.8-27B-NVFP4 + `incoai/Qwen3.8-27B-DFlash2` | all 5 rungs above floor, peak 72.1 tok/s |
| `kat-equality-gate` | The same sample must get the same answer whatever ran before it (hermetic serve) | Qwen3.8-27B-NVFP4 | 257 samples byte-identical across 2 request orders |
| `concurrency-sweep-moe` | Aggregate throughput floors at C=1 to 16 on the MoE, published instrument | Qwen3.6-35B-A3B-FP8 | all 5 rungs above floor, peak 311.2 tok/s |
| `high-isl-ttft-cold` | Uncached 32k-token prefill TTFT on the dense flagship | Qwen3.8-27B-NVFP4 | added after this certification; ceiling 24115.2 ms |
| `high-isl-ttft-warm` | Cached 32k-token prefix TTFT on the dense flagship | Qwen3.8-27B-NVFP4 | added after this certification; ceiling 2132.3 ms |
| `high-isl-ttft-cold-moe` | Uncached 32k-token prefill TTFT on the 35B MoE | Qwen3.6-35B-A3B-FP8 | added after this certification; ceiling 9231.9 ms |
| `high-isl-ttft-warm-moe` | Cached 32k-token prefix TTFT on the 35B MoE | Qwen3.6-35B-A3B-FP8 | added after this certification; ceiling 579.4 ms |

The BFCL figures are the aggregate over six shard records, as
`met benchmark aggregate bfcl-subset --sha aa5d059438` prints them; every
other row is the record's `verdict_reason`. The descriptions follow
`met benchmark list`. The four high-ISL gates were added after `aa5d059438`
was certified. Their ceilings are vLLM 0.27.1's time to first token on the
same prompt and box, so a pass means at least as fast as vLLM;
[`bench/baselines/qwen36-35b-a3b/ttft/published.json`](bench/baselines/qwen36-35b-a3b/ttft/published.json)
and
[`bench/baselines/qwen38-27b/ttft/published.json`](bench/baselines/qwen38-27b/ttft/published.json)
hold the measurements.

<a id="security"></a>
## <img src="docs/readme/icons/lock.svg" width="20" height="20" alt="Lock icon"> Security

Report vulnerabilities privately to **security@metrale.ai**, not in a public
issue. [`SECURITY.md`](SECURITY.md) has the scope, the response times and the
disclosure policy.

The server binds `127.0.0.1` by default, and `--require-auth` enables bearer
tokens. Fetching remote images and decoding video through ffmpeg are both off
until a flag enables them. `cargo deny` checks advisories, licences,
sources and banned crates on every pull request
([`.github/workflows/security.yml`](.github/workflows/security.yml)).

<a id="licence"></a>
## <img src="docs/readme/icons/licence.svg" width="20" height="20" alt="Document icon"> Licence

Metrale Engine is licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option. Third-party code keeps its own
licence; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`CITATIONS.md`](CITATIONS.md).

The benchmark records in `.benchmarks/` are dated, and each can be checked
against its signature and the commit it names.

<a id="appendix-people"></a>
## <img src="docs/readme/icons/people.svg" width="20" height="20" alt="People icon"> Appendix: people

**Thomas Braun** ([@tbraun96](https://github.com/tbraun96)) is the founder
and principal engineer of [Metrale](https://metrale.ai).

**Tom Turney** ([@TheTom](https://github.com/TheTom)) is a core contributor.
He contributed the TurboQuant+ KV-cache quantization work (the `turbo2`,
`turbo3`, `turbo4` and `turbo8` KV cache types and their asymmetric
variants; see [`docs/turboquant-plus.md`](docs/turboquant-plus.md)), whose
research lives in
[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) and
whose llama.cpp reference implementation is
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant).
He also contributed Hopper performance work, including the C=1 improvements
in the certified commit `aa5d059438`.

<a id="links"></a>
## <img src="docs/readme/icons/link.svg" width="20" height="20" alt="Link icon"> Links

- Website: [metrale.ai](https://metrale.ai)
- Engine overview: [metrale.ai/engine](https://metrale.ai/engine)
- Benchmark dashboard: [metrale.ai/benchmarks](https://metrale.ai/benchmarks)
- Verification steps for reviewers: [metrale.ai/diligence](https://metrale.ai/diligence)
- Documentation (the book): [docs.metrale.ai](https://docs.metrale.ai)
- API reference: [docs.metrale.ai/api](https://docs.metrale.ai/api/)
- Engineering blog: [blog.metrale.ai](https://blog.metrale.ai)
- Launch recipes: [`recipes/`](recipes/)
- Launcher: [Metrale/metralectl](https://github.com/Metrale/metralectl)
- Issues and discussions: [GitHub issues](https://github.com/Metrale/metrale-inference/issues), [GitHub discussions](https://github.com/Metrale/metrale-inference/discussions)
- Contact: [metrale.ai/contact](https://metrale.ai/contact) and the [Metrale Discord](https://discord.gg/RQcGakU2jW)
- Security reports: security@metrale.ai
