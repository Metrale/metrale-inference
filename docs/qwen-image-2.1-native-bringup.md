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

- [ ] Pin the full checkpoint revision, tokenizer/text encoder, VAE, scheduler and
      image processing components. Record versions, sizes, hashes and license.
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
