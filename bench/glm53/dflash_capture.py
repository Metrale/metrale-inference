#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-10: Capture side of the DFlash2 acceptance replay, over the OpenAI-compatible API.

  generate --url U --model M --max-tokens N --out gen.json
      One greedy chat request (thinking off) on the fixed prose prompt, streamed with
      `return_token_ids`, so the generated token ids are exact. Records the ids, the text and
      the usage block.
  teacher  --url U --model M --gen gen.json --template T --tokenizer J --max-tokens N --out tf.json
      Re-sends prompt + generated ids of a `generate` capture as one raw `/v1/completions`
      prompt of token ids: the target then computes every position's hidden states in a
      single prefill, which the replay compares with the hidden states the decode/verify
      path produced for the same positions.

The prompt ids are rendered offline (chat template + tokenizer.json) and checked against the
server's `usage.prompt_tokens`; a mismatch is an error, never a warning.
Run it as the only client of a freshly started serve: the engine's one-shot dumps fire at the
first propose at or past the armed position, whichever request reaches it.
"""
import argparse
import json
import sys
import urllib.request

PROSE = ("The quick brown fox jumped over the lazy dog near a river bank. Mountains rise above the "
         "clouds while birds sing their morning songs. Using the text above only as a starting point, "
         "write a long, richly detailed essay that keeps introducing new specifics, examples and "
         "vocabulary. Never repeat a sentence.")
PREFIX = "[000000] "


def messages():
    return [{"role": "user", "content": PREFIX + PROSE}]


def post(url, body, stream):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=3600)
    if not stream:
        return json.loads(resp.read())
    return resp


def render_prompt_ids(template_path, tokenizer_path):
    import jinja2
    import tokenizers
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    text = env.from_string(open(template_path).read()).render(
        messages=messages(), add_generation_prompt=True, enable_thinking=False)
    tok = tokenizers.Tokenizer.from_file(tokenizer_path)
    return tok.encode(text, add_special_tokens=False).ids


def generate(a):
    # 2026-10-10: min_tokens = max_tokens, so the sequence reliably reaches the armed dump position.
    body = {"model": a.model, "messages": messages(), "max_tokens": a.max_tokens,
            "min_tokens": a.max_tokens, "temperature": 0, "stream": True, "return_token_ids": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    ids, text, usage, finish = [], [], None, None
    for line in post(a.url.rstrip("/") + "/v1/chat/completions", body, True):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            ids.extend(ch.get("token_ids") or [])
            d = ch.get("delta") or {}
            text.append((d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or ""))
            finish = ch.get("finish_reason") or finish
    if usage is None or usage.get("completion_tokens") != len(ids):
        sys.exit(f"token ids incomplete: {len(ids)} ids vs usage {usage}")
    out = {"messages": messages(), "completion_ids": ids, "text": "".join(text),
           "usage": usage, "finish_reason": finish, "max_tokens": a.max_tokens}
    json.dump(out, open(a.out, "w"))
    print(f"generated {len(ids)} tokens after a {usage['prompt_tokens']}-token prompt -> {a.out}")


def teacher(a):
    gen = json.load(open(a.gen))
    prompt_ids = render_prompt_ids(a.template, a.tokenizer)
    if len(prompt_ids) != gen["usage"]["prompt_tokens"]:
        sys.exit(f"offline prompt render has {len(prompt_ids)} tokens, the server counted "
                 f"{gen['usage']['prompt_tokens']}: template or tokenizer differ")
    ids = prompt_ids + gen["completion_ids"]
    body = {"model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0}
    d = post(a.url.rstrip("/") + "/v1/completions", body, False)
    if d["usage"]["prompt_tokens"] != len(ids):
        sys.exit(f"server saw {d['usage']['prompt_tokens']} prompt tokens, sent {len(ids)}")
    out = {"prompt_ids": prompt_ids, "completion_ids": gen["completion_ids"],
           "forced_len": len(ids), "continuation": d["choices"][0]["text"], "usage": d["usage"]}
    json.dump(out, open(a.out, "w"))
    print(f"teacher-forced {len(ids)} tokens ({len(prompt_ids)} prompt) -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    t = sub.add_parser("teacher")
    for p in (g, t):
        p.add_argument("--url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--max-tokens", type=int, required=True)
        p.add_argument("--out", required=True)
    t.add_argument("--gen", required=True)
    t.add_argument("--template", required=True)
    t.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    {"generate": generate, "teacher": teacher}[a.cmd](a)


if __name__ == "__main__":
    main()
