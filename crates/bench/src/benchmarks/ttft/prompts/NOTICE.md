# High-ISL TTFT prompt fixtures

## `long-32k.txt`

The opening of Herman Melville's *Moby-Dick; or, The Whale* (1851), cut so that
the request the high-ISL TTFT gates send is exactly 32,768 prompt tokens on the
Qwen3.6-35B-A3B MoE as this engine serves it.

### Source

- **Work**: *Moby Dick; Or, The Whale*, Herman Melville.
- **Edition**: Project Gutenberg eBook #2701, release date July 1, 2001, "Most
  recently updated: September 7, 2026" (from the file's header).
- **Download**: <https://www.gutenberg.org/cache/epub/2701/pg2701.txt>, fetched
  2026-09-27; the ebook page is <https://www.gutenberg.org/ebooks/2701>.
- **Download sha256**:
  `907420db6c4b68c70e2988cd2ad9c8cf79138667a01b63376d18dd17fef1a18b`
  (1,276,267 bytes, UTF-8, CRLF line endings).

### Status

The novel was first published in 1851 and is in the public domain in the United
States. The fixture holds only Melville's text: the Project Gutenberg header,
footer, licence and trademark text are removed, as is the transcriber's note
that follows the table of contents. No Project Gutenberg trademark or licence
term applies to what remains.

### How the file was made

`scripts/make_long_prompt.py` rebuilds the file byte for byte from the download
and fails on any other download:

1. Keep the text strictly between the `*** START OF THE PROJECT GUTENBERG
   EBOOK …` and `*** END OF THE PROJECT GUTENBERG EBOOK …` lines.
2. Normalise CRLF to LF, strip trailing whitespace from every line, remove the
   transcriber's note, and strip leading and trailing whitespace.
3. Cut at the last word boundary where the whole request renders to 32,768
   tokens, and end the file with one newline. The request is the user message
   `[<tag>] <file>\n<task line>` (`long_prompt::content`), rendered through the
   engine's `jinja-templates/qwen3_5_moe.jinja` override with thinking disabled
   and tokenized by the checkpoint's `tokenizer.json`. The tags have one shape
   (`cold-32k-` or `warm-32k-`, then 16 decimal digits) and render to the same
   count; the script checks this for the warm tag and several cold tags. The
   tags, task line and target are read from `ttft/long_prompt.rs`.

The cut falls inside chapter 9, "The Sermon", after "and all the watery world
of woe".

```sh
python3 scripts/make_long_prompt.py --source pg2701.txt --check
```

### Result

| | |
|---|---|
| Bytes | 126,709 |
| sha256 | `4f961eabd9433e18166e795052b59239760d4e3cc7fa78c258d0bd1b623a1993` |
| Lines / words | 2,485 / 21,772 |
| Warm user message sha256 | `c943228f01a771c1ddce0cf5c0cb0eb6f6bd8a495b19a19b82f2fd400fc31b26` |

`long_prompt_tests.rs` pins the file's sha256 and length and the warm message's
sha256.

### Token counts

Counted offline with `tokenizers` 0.22.2 and `jinja2` 3.1.2; the gates record
the server's own `usage.prompt_tokens` on every sample.

| Checkpoint (tokenizer.json sha256) | Rendering | Prompt tokens |
|---|---|---|
| Qwen/Qwen3.6-35B-A3B-FP8 (`5f9e4d49…`) | this engine: `qwen3_5_moe.jinja` override, `enable_thinking: false` | **32,768** |
| Qwen/Qwen3.6-35B-A3B-FP8 | vLLM: the checkpoint's own template, `enable_thinking: false` (adds the empty think block) | 32,772 |
| unsloth/Qwen3.8-27B-NVFP4 (`06b95093…`) | this engine and vLLM: the checkpoint's own template, `enable_thinking: false` | 32,772 |
| either tokenizer | the user message alone, no template | 32,725 |

Both engines receive the same bytes. The 4-token difference is the
`<think>\n\n</think>\n\n` block that the checkpoint templates add and the
engine's MoE override does not, so every rendering is at least the gates'
default `min_prompt_tokens` of 32,768.
