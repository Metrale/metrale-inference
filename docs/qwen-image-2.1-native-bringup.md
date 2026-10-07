# Qwen Image 2.1 integration

This is the cumulative integration plan for `Qwen/Qwen-Image-2.1`.
Add implementation, regressions and qualification evidence to this model branch
and draft PR. Status: checkpoint and verified backup complete; two external-reference images
generated and visually inspected. Native generation remains unimplemented. Image API routes currently return unsupported responses.

## Inputs and use boundary

- [Official model repository](https://github.com/QwenLM/Qwen-Image-2.1).
- [Checkpoint and component descriptions](https://huggingface.co/Qwen/Qwen-Image-2.1).
- [Model license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE).

The reviewed license permits research/evaluation and requires separate permission
for commercial use. Verify the license at the chosen immutable revision before
retrieval or distribution. Commercial product availability is blocked pending
appropriate permission. Do not treat the engine source license as the weight license.

## Implementation and evidence gates

- [x] Pin the full checkpoint revision, tokenizer/text encoder, VAE, scheduler and
      image processing components in the [component manifest](model-manifests/qwen-image-2.1.json).
      Local active files and the complete backup have now passed SHA-256 checks.
- [x] SHA-256 verify a complete TrueNAS backup and BACKUP-MANIFEST.json after
      download, retaining local active components. A transformer-only backup is incomplete.
- [x] Pin and execute the official reference pipeline to establish expected image
      behavior and resource needs. Label reference results explicitly.
- [ ] Describe the native model circuit and compare reusable primitives before
      adding kernels. Record unsupported lowering and component boundaries.
- [ ] Implement native loading and generation with intermediate numerical parity;
      keep reference-backend and native-engine qualification separate.
- [ ] Implement image generation/editing API validation, bounded job queues,
      cancellation, errors and result retrieval. Never return placeholder success.
- [ ] Validate text rendering, reference-image editing, RGBA and advertised size
      limits with saved prompts, seeds, components, outputs and human image review.
- [ ] Exercise invalid dimensions, missing components, overload, cancellation,
      disconnected clients and memory pressure with known-bad controls.
- [ ] Expose per-image latency, peak memory, generation settings and measured
      cost per image; do not reuse text-token economics for image jobs.
- [ ] Complete native stability and required certification before marking support.

Start with one device and explicit memory limits. Do not interrupt the GPT-OSS or
Laguna campaigns to stage this model. Multi-device operation and product UI
availability require separate measured evidence. This PR remains draft while
these gates are open; the reference observations below are not native benchmarks.

## Pinned component inventory (2026-10-06)

The manifest pins checkpoint `d26bb61231c349cf6b7896fa83353113880e1ba3`
and every serving file, plus the README and license. Selected download size is
33,131,615,131 bytes (30.86 GiB); weights account for 33,115,613,408 bytes.
This is disk size, not a measured GPU memory requirement. The repository's QR
asset and Git attributes are excluded from that serving inventory. The complete
backup includes both: 28 files, 33,134,949,561 bytes, with a verified manifest.

| Component | Pinned identity | Download bytes |
| --- | --- | ---: |
| Joint text/image encoder | Qwen3VLForConditionalGeneration | 17,534,409,013 |
| Visual transformer | QwenImage21Transformer2DModel | 14,230,315,061 |
| RGBA autoencoder | AutoencoderKLQwenImage21 | 1,350,991,591 |
| Tokenizer and preprocessing | Qwen3VLProcessor | 15,884,996 |
| Flow scheduler | FlowMatchEulerDiscreteScheduler | 485 |

The visual transformer has 32 layers, 32 attention heads of width 128 and
64 input/output channels. Its 7B description excludes the joint encoder and VAE.
The encoder includes 36 text layers (hidden width 4096, 32 query/8 KV heads)
and a 27-layer vision tower. The VAE has four input/output channels and 64 latent
channels. Preserve the checkpoint's latent normalization arrays and scheduler
configuration; a generic RGB VAE or text-only encoder is not interchangeable.

The [pinned license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE)
allows research/evaluation only without a separate commercial license. Permission
for commercial use has not been verified. Preserve the agreement and required
Notice when distributing materials. This does not block inspecting metadata or
planning research qualification; it does block claiming commercial readiness.

## Retrieval and reference verification

1. Retrieve only manifest paths from the immutable checkpoint revision. Reject
   missing files, unexpected component substitutions, wrong sizes or SHA-256
   mismatches. Check both safetensors indexes: every referenced shard must exist,
   and every indexed tensor must be found in its assigned shard header. All local file hashes were verified before reference loading; adversarial
   shard-index integrity controls remain to be implemented.
2. Copy the complete selected set to TrueNAS and independently hash destination
   bytes. Write `BACKUP-MANIFEST.json` only after all checks pass, with repository,
   revision, each file's size/hash and completion time. Retain active local weights.
   Reserve space for the entire checkpoint and temporary download files on both
   destinations; a transformer-only archive cannot restore this pipeline.
3. Install the manifest's exact Diffusers and Transformers source revisions in
   an isolated reference environment, then freeze the resolved dependency lock.
   These source pins ran successfully in the isolated environment described below;
   broader reference qualification remains pending. The model index's `0.37.0.dev0` is provenance, not a
   reproducible dependency pin. Record Torch/CUDA, driver and binary identities.
4. The pinned official pipeline performs joint text/image conditioning and
   optional prefix KV reuse; it samples without classifier-free guidance by
   default. Capture actual settings explicitly. First compare cache on/off at a
   fixed seed, prompt, resolution, dtype and step count, saving intermediate
   embeddings, denoising latents and decoded output for later native parity.
5. Keep first-run load time separate from warm generation. Measure complete
   request latency and encoder, denoising and VAE stages; report images/minute,
   peak memory and joules/image at fixed quality settings. Start at one request
   and qualify bounded concurrent queues before increasing GPU parallelism.
   A reference result is not a native Metrale pass.

Integrity controls must include a missing encoder shard, a corrupted VAE byte,
an index pointing to the wrong shard and a mismatched processor config, each
rejected alongside an unchanged valid checkpoint. Numerical controls need a
deliberately wrong latent scale or scheduler shift to demonstrate that comparison
detects errors. Image review covers RGB, actual alpha-channel transparency,
reference-image editing, typography and repeatability; merely writing a PNG does
not establish correctness. Set numerical tolerances before evaluating results,
and record any precision-dependent differences rather than relaxing them to pass.

## Reference observations (2026-10-06)

Single NVIDIA GB10, Torch 2.13.0+cu130, CUDA 13.0, BF16, memory fraction 0.85.
Diffusers `c6df88a511a98740646ee55577b590c9852650ce` and Transformers
`14e738b5d0cc69aa27a95dde272aea41fde44f2f`; offline local checkpoint loading.
Initial pipeline load took 210.11 seconds, excluded from generation below.
Both cases used 40 steps and guidance scale 1.0. These are individual samples,
not repeated throughput or native-engine measurements.

| Case | Seed | Size | Generation seconds | Peak allocated / reserved bytes | Visual inspection |
| --- | ---: | --- | ---: | --- | --- |
| Red cube left, blue sphere right | 426 | 512×512 | 12.77 | 34,276,423,168 / 34,661,728,256 | Correct colors and positions; red object has polygonal/beveled shape rather than a strict cube. Prompt fidelity incomplete. |
| METRALE / INFERENCE GPU poster | 427 | 1024×1024 | 53.47 | 39,566,055,424 / 41,204,842,496 | Both words legible and correctly spelled; green chip centered. |

Output SHA-256: geometry `5c037c4d2c7ada8a73e29f8178802c539152a976ed1ed98d77ca1315d83cb141`;
poster `44e568efbdfc0368351ee7d3e73513097cc6d0e20a6e4ce3d82447b8e2994b04`.
RGBA file mode alone does not prove useful alpha transparency. Editing, alpha,
repeatability, size limits, component parity, native API and native performance
remain unqualified. The geometry miss is retained as a quality counterexample.
