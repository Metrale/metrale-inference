# Qwen Image 2.1 scheduler evidence

The native host scheduler implements the checkpoint's deterministic exponential
flow schedule and staged BF16 Euler update. It performs no I/O; the caller owns
latent buffers and transfers. This is a correctness path, not a GPU throughput
claim or a complete image pipeline.

Actual checkpoint: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
[immutable revision](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3).
Reference Diffusers commit: `c6df88a511a98740646ee55577b590c9852650ce`.
The replay script refuses a scheduler source hash other than
`5448bbfe15324ea8034470e742c7782b1de3864c0836d98ef3dade65e30df6a7`.

The risks are scalar promotion, lost rounding boundaries, incorrect timestep
normalization and silently accepting a different scheduler policy. The independent
oracle is the pinned scheduler plus actual Torch operations, with constructed
operands rather than checkpoint weights. Tests reject unsupported policy, unknown
config keys, one-step terminal-stretch degeneracy, invalid indices, shape mismatch,
nonfinite inputs and positive denoising deltas. A deliberately wrong FP32 product
provides a detection control; it is not accepted as an equivalent implementation.

Twelve schedules (2, 4, 40 and 50 steps at 256, 1024 and 8192 image tokens) match
all sigma, raw timestep and normalized BF16 model-timestep bits. Four step cases
each span all 65,280 finite BF16 prediction encodings with a repeating sample
pattern. Native Rust output hashes match the reference. The same reference cases
were run on CPU and CUDA; schedule arrays and output hashes were identical.

The zero-dimensional FP32 delta first converts to BF16 before multiplication.
The product rounds to BF16, then adds to the FP32-upcast sample before the final
BF16 cast. Keeping the delta or product in FP32 changes results. The pipeline
also casts raw time to BF16 before dividing by 1000 in BF16. These boundaries
are preserved explicitly.

Reproduce the reference receipt with the pinned environment:

```sh
python scripts/qwen_image21/scheduler_reference.py --checkpoint /path/to/checkpoint --device cpu
python scripts/qwen_image21/scheduler_reference.py --checkpoint /path/to/checkpoint --device cuda
```

Three scoped Rust tests and scoped Clippy pass. The initial test build exposed
an unsupported SHA digest formatting trait; it was corrected before numerical
execution. This increment stops at the verified scheduler component. Custom
sigmas, stochastic sampling, inverted schedules, per-token timesteps, image
quality, device-resident update performance and full pipeline integration remain
outside this evidence.
