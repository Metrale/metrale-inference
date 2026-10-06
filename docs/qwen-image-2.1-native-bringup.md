# Qwen Image 2.1 integration

This is the cumulative integration plan for `Qwen/Qwen-Image-2.1`.
Add implementation, regressions and qualification evidence to this model branch
and draft PR. Status: discovery only; no downloaded checkpoint, native generation
or image-quality pass. Image API routes currently return unsupported responses.

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
      Weight hashes are upstream declarations, not verified local downloads.
- [ ] SHA-256 verify a complete TrueNAS backup and BACKUP-MANIFEST.json after
      download, retaining local active components. A transformer-only backup is incomplete.
- [ ] Pin and execute the official reference pipeline to establish expected image
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
these gates are open; no generated image or benchmark is claimed by this plan.

## Pinned component inventory (2026-10-06)

The manifest pins checkpoint `d26bb61231c349cf6b7896fa83353113880e1ba3`
and every serving file, plus the README and license. Selected download size is
33,131,615,131 bytes (30.86 GiB); weights account for 33,115,613,408 bytes.
This is disk size, not a measured GPU memory requirement. The repository's QR
asset and Git attributes are excluded. No weights have been downloaded.

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
   and every indexed tensor must be found in its assigned shard header. The seven
   weight files have not yet undergone these local integrity checks.
2. Copy the complete selected set to TrueNAS and independently hash destination
   bytes. Write `BACKUP-MANIFEST.json` only after all checks pass, with repository,
   revision, each file's size/hash and completion time. Retain active local weights.
   Reserve space for the entire checkpoint and temporary download files on both
   destinations; a transformer-only archive cannot restore this pipeline.
3. Install the manifest's exact Diffusers and Transformers source revisions in
   an isolated reference environment, then freeze the resolved dependency lock.
   These source pins were inspected, but their combined environment is not yet
   executed or qualified. The model index's `0.37.0.dev0` is provenance, not a
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
