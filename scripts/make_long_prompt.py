#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Rebuild the high-ISL TTFT prompt fixture from Project Gutenberg eBook #2701.

Writes `crates/bench/src/benchmarks/ttft/prompts/long-32k.txt` byte for byte
(see the `NOTICE.md` beside it):

1. Take the plain-text download (`--source`, or fetch `SOURCE_URL`) and refuse
   it unless its sha256 is `SOURCE_SHA256`: Project Gutenberg re-issues its
   files, and a different edition would cut at a different word.
2. Keep only the text between the START and END markers, so no Project
   Gutenberg header, footer, licence or trademark text survives, and drop the
   transcriber's note after the contents (it names Project Gutenberg and is
   not Melville's). Normalise CRLF to LF, strip trailing whitespace from each
   line and the whole text.
3. Cut at the last word boundary where the request the high-ISL gates send
   (a cold tag, the text, the task line; `ttft/long_prompt.rs`) renders to
   exactly `LONG_32K_TOKENS` tokens through the Qwen3.6-35B-A3B tokenizer and
   the chat template this engine serves that checkpoint with
   (`jinja-templates/qwen3_5_moe.jinja`, thinking disabled). Among cuts with
   that count, the last one that ends a sentence wins.

The tag, task line and target come from `long_prompt.rs`, which is their only
definition. Needs `tokenizers`, `jinja2` and the tokenizers in the local
Hugging Face cache; nothing is downloaded except the book, and only without
`--source`.

    python3 scripts/make_long_prompt.py --source pg2701.txt           # write
    python3 scripts/make_long_prompt.py --source pg2701.txt --check   # compare
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
RUST = REPO / "crates/bench/src/benchmarks/ttft/long_prompt.rs"
FIXTURE = REPO / "crates/bench/src/benchmarks/ttft/prompts/long-32k.txt"
MOE_OVERRIDE = REPO / "jinja-templates/qwen3_5_moe.jinja"

SOURCE_URL = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
SOURCE_SHA256 = "907420db6c4b68c70e2988cd2ad9c8cf79138667a01b63376d18dd17fef1a18b"
START = "*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK; OR, THE WHALE ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK; OR, THE WHALE ***"
TRANSCRIBER_NOTE = re.compile(r"\nOriginal Transcriber\u2019s Notes:\n\n.*?\n\n", re.S)

MOE = "Qwen/Qwen3.6-35B-A3B-FP8"
DENSE = "unsloth/Qwen3.8-27B-NVFP4"


def rust_const(src: str, name: str) -> str:
    m = re.search(rf"const {name}: [^=]+= (.+?);", src)
    if not m:
        sys.exit(f"{RUST}: no const {name}")
    return m.group(1)


def rust_str(src: str, name: str) -> str:
    raw = rust_const(src, name)
    if not (raw.startswith('"') and raw.endswith('"')) or "\\" in raw:
        sys.exit(f"{RUST}: {name} must be a plain string literal without escapes")
    return raw[1:-1]


def novel(download: bytes) -> str:
    text = download.decode("utf-8").lstrip("\ufeff").replace("\r\n", "\n")
    start = text.index(START) + len(START)
    body = "\n".join(line.rstrip() for line in text[start : text.index(END, start)].split("\n"))
    body, notes = TRANSCRIBER_NOTE.subn("\n", body)
    if notes != 1:
        sys.exit(f"expected one transcriber's note, found {notes}")
    return body.strip()


def snapshot(repo_id: str) -> pathlib.Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    pattern = os.path.join(hub, f"models--{repo_id.replace('/', '--')}", "snapshots", "*")
    for snap in sorted(glob.glob(pattern)):
        if os.path.exists(os.path.join(snap, "tokenizer.json")):
            return pathlib.Path(snap)
    sys.exit(f"{repo_id}: no tokenizer.json under {pattern}")


def template(source: str):
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


class Renderer:
    """One (tokenizer, template, context) triple: how one engine serves one checkpoint."""

    def __init__(self, repo_id: str, template_path: pathlib.Path, context: dict):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(snapshot(repo_id) / "tokenizer.json"))
        self.template = template(template_path.read_text(encoding="utf-8"))
        self.context = context

    def render(self, content: str) -> str:
        return self.template.render(
            messages=[{"role": "user", "content": content}],
            add_generation_prompt=True,
            **self.context,
        )

    def count(self, content: str) -> int:
        return len(self.tokenizer.encode(self.render(content), add_special_tokens=False).ids)


# 2026-09-27: The context each engine renders with `chat_template_kwargs: {enable_thinking: false}`:
# this engine's `tokenizer::chat_render::render_chat`, and vLLM, which passes the kwargs only.
SERVED = {"enable_thinking": False, "reasoning_effort": "none", "add_vision_id": False,
          "disable_tool_steering": False}
VLLM = {"enable_thinking": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=pathlib.Path, help=f"a saved copy of {SOURCE_URL}")
    ap.add_argument("--check", action="store_true", help="compare with the committed file")
    args = ap.parse_args()

    download = args.source.read_bytes() if args.source else urllib.request.urlopen(SOURCE_URL).read()
    got = hashlib.sha256(download).hexdigest()
    if got != SOURCE_SHA256:
        sys.exit(f"download sha256 {got} is not {SOURCE_SHA256}: a different edition")

    src = RUST.read_text(encoding="utf-8")
    task = rust_str(src, "TASK_LINE")
    warm = rust_str(src, "WARM_TAG")
    cold_prefix = rust_str(src, "COLD_TAG_PREFIX")
    digits = int(rust_const(src, "NONCE_DIGITS"))
    target = int(rust_const(src, "LONG_32K_TOKENS").replace("_", ""))
    colds = [cold_prefix + d * digits for d in "0159"] + [cold_prefix + ("1234567890" * 4)[:digits]]

    def content(text: str, tag: str) -> str:
        # 2026-09-27: Mirrors `long_prompt::content`; `long_prompt_tests.rs` pins its digest.
        return f"[{tag}] {text}\n{task}"

    moe_ckpt = snapshot(MOE)
    if json.loads((moe_ckpt / "config.json").read_text())["model_type"] != MOE_OVERRIDE.stem:
        sys.exit(f"{MOE} is no longer served through {MOE_OVERRIDE.name}")
    served = Renderer(MOE, MOE_OVERRIDE, SERVED)
    moe_vllm = Renderer(MOE, moe_ckpt / "chat_template.jinja", VLLM)
    dense_ckpt = snapshot(DENSE)
    dense_served = Renderer(DENSE, dense_ckpt / "chat_template.jinja", SERVED)
    dense_vllm = Renderer(DENSE, dense_ckpt / "chat_template.jinja", VLLM)
    # 2026-09-27: The whole rendering, so a template change that adds a system turn or drops the
    # empty think block is a failure here rather than a silent change of count.
    frame = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    for r, tail in [(served, ""), (moe_vllm, "<think>\n\n</think>\n\n"),
                    (dense_served, "<think>\n\n</think>\n\n"), (dense_vllm, "<think>\n\n</think>\n\n")]:
        if r.render("x") != frame.format("x") + tail:
            sys.exit(f"unexpected rendering {r.render('x')!r}")

    text = novel(download)
    ends = [m.end() for m in re.finditer(r"\S+", text)]

    def fixture(k: int) -> str:
        return text[: ends[k - 1]] + "\n"

    def count(k: int) -> int:
        return served.count(content(fixture(k), colds[-1]))

    lo, hi = 1, len(ends)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    if count(lo) != target:
        sys.exit(f"no word boundary renders to {target} tokens (best {count(lo)}); change TASK_LINE")
    best = lo
    k = lo
    while k > 1 and count(k) == target:
        if re.search(r"[.!?][\"'\u2019\u201d)]*$", fixture(k).rstrip("\n")):
            best = k
            break
        k -= 1
    out = fixture(best).encode("utf-8")

    body = out.decode("utf-8")
    counts = {}
    for label, r in [("moe_served", served), ("moe_vllm", moe_vllm),
                     ("dense_served", dense_served), ("dense_vllm", dense_vllm)]:
        per_tag = {r.count(content(body, t)) for t in [warm, *colds]}
        if len(per_tag) != 1:
            sys.exit(f"{label}: the warm and cold tags render to different counts {per_tag}")
        counts[label] = per_tag.pop()
        counts[label.split("_")[0] + "_text_only"] = len(
            r.tokenizer.encode(body, add_special_tokens=False).ids
        )
    if counts["moe_served"] != target:
        sys.exit(f"cut renders to {counts['moe_served']}, not {target}")

    report = {
        "source_sha256": SOURCE_SHA256,
        "fixture_bytes": len(out),
        "fixture_sha256": hashlib.sha256(out).hexdigest(),
        "fixture_words": best,
        "fixture_ends_with": body.rstrip("\n")[-60:],
        "warm_content_sha256": hashlib.sha256(content(body, warm).encode("utf-8")).hexdigest(),
        "tokens": counts,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.check:
        if FIXTURE.read_bytes() != out:
            print(f"{FIXTURE} differs from the rebuilt fixture", file=sys.stderr)
            return 1
        print(f"{FIXTURE.relative_to(REPO)} matches byte for byte")
        return 0
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
