# Native text-only prompt contract

The optional `qwen-image-text` feature provides native Rust tokenization for the
pinned [Qwen/Qwen-Image-2.1 checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3).
The caller supplies tokenizer and template bytes through its I/O boundary. Their
hashes must match the existing checkpoint manifest; no alternate-template fallback
is accepted.

The image pipeline uses a raw system/user/assistant layout. An empty prompt becomes
one space; other whitespace, Unicode and newlines are preserved. The system-prefix
token count is derived from the pinned tokenizer. It matches the independently
captured processor system-template count of 14, rather than hardcoding that number.
The returned IDs include the whole prefix for the native encoder; `drop_prefix`
identifies how many leading hidden-state positions to remove afterward.

Five actual-tokenizer cases match the pinned processor exactly. Three controls
reject a changed tokenizer asset, insufficient capacity and image-conditioned
input. Two isolated unit controls and scoped Clippy pass. A broader Metal lib-test
build fails in unrelated existing backend test helpers; that failure is retained
and is not reported as a suite pass.

This path is one unpadded text prompt. It does not implement image-conditioned
processing, batched left padding, vision/deepstack, or image generation. The capture
example reads assets but loads no model weights and executes no GPU work:

```sh
cargo run -p metrale-model-arch --no-default-features --features metal,qwen-image-text \
  --example qwen_image21_prompt_capture -- /path/to/checkpoint/processor
```
