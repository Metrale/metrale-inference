#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-08: Long-context accuracy probes for the GLM-5.3-Flash campaign, engine-neutral.
#
# Owner: bench, GLM campaign.
# Invariants: every request body is built deterministically from (probe, length, depth, seed),
# so two engines receive byte-identical requests; each body's sha256 is recorded with its verdict.
"""Two long-context probes against any OpenAI-compatible chat endpoint (standard library only).

needle      A distinctive fact buried at several depths in numbered filler; the model must
            return it. Verdict: the answer string appears in `content` (reasoning is recorded
            separately and never counted).
transcript  An agentic-coding transcript: many read_file tool calls and their results (code
            files) fill the context; the FIRST file read defines the port the task needs. The
            final turn asks the model to start the server through the run_command tool.
            Verdict: the reply is a tool call to run_command whose arguments parse as JSON and
            contain the port. This checks long context, the tool-call parser and agent
            behaviour together.

Sizing: prompt length is calibrated per endpoint from `usage.prompt_tokens` of probe requests,
then verified on the real request (`usage.prompt_tokens` is recorded). A prompt over the
endpoint's context fails as an ERROR and stays in the denominator.

    python3 long_context.py --url http://HOST:PORT --model NAME --out result.json
"""
import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request

NEEDLE = "The access code for the Vermilion vault is 7741-KESTREL."
ANSWER = "7741-KESTREL"
QUESTION = ("\n\nQuestion: what is the access code for the Vermilion vault? "
            "Reply with the code alone.")
FILLER = ("Maintenance note {n}: the west pump was serviced and the intake filter replaced; "
          "readings held within tolerance for the shift.\n")
PORT = "48213"
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the repository root.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]
SYSTEM = ("You are a coding agent working in a Python repository. Use the tools to inspect "
          "files and run commands. Call one tool at a time.")
TASK = ("Read the repository to learn how the service is configured, then start it. "
        "Begin with config/settings.py.")
FINAL = ("You have read enough. Start the HTTP service now with run_command: run "
         "`python -m service.server --port <PORT>` with the default port that "
         "config/settings.py defines. Do not read any more files.")


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def code_file(rng, idx):
    """A synthetic but code-shaped module, deterministic in (seed, idx)."""
    words = ["parse", "render", "client", "buffer", "retry", "cache", "token", "route",
             "schema", "worker", "queue", "event", "handler", "session", "metric", "config"]
    lines = [f'"""Module {idx}: helpers for the {rng.choice(words)} subsystem."""', "import os",
             "import json", ""]
    for f in range(rng.randint(10, 16)):
        a, b = rng.choice(words), rng.choice(words)
        lines += [f"def {a}_{b}_{idx}_{f}(value, limit={rng.randint(2, 900)}):",
                  f'    """Return the {a} {b} for value, clamped to limit."""',
                  f"    total = sum(ord(c) for c in str(value)) % {rng.randint(7, 97)}",
                  f"    if total > limit:", f"        return limit",
                  f"    return {{'{a}': total, '{b}': json.dumps(value)[:{rng.randint(8, 64)}]}}", ""]
    return "\n".join(lines)


SETTINGS = ('"""Service settings."""\nimport os\n\nDEFAULT_HOST = "0.0.0.0"\n'
            f'DEFAULT_PORT = {PORT}\nLOG_LEVEL = os.environ.get("LOG_LEVEL", "info")\n')


def transcript_messages(n_files, seed):
    rng = random.Random(seed)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    files = [("config/settings.py", SETTINGS)] + [
        (f"service/mod_{i:04d}.py", code_file(rng, i)) for i in range(n_files)]
    for k, (path, text) in enumerate(files):
        cid = f"call_{k:05d}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": text})
    msgs.append({"role": "user", "content": FINAL})
    return msgs


def needle_prompt(lines, depth):
    body = [FILLER.format(n=i) for i in range(lines)]
    body.insert(min(len(body), int(len(body) * depth)), NEEDLE + "\n")
    return "".join(body) + QUESTION


def request(probe, model, size, depth, seed, max_tokens):
    if probe == "needle":
        msgs = [{"role": "user", "content": needle_prompt(size, depth)}]
        body = {"model": model, "messages": msgs}
    else:
        body = {"model": model, "messages": transcript_messages(size, seed), "tools": TOOLS}
    body.update({"max_tokens": max_tokens, "temperature": 0, "seed": seed, "stream": False})
    return body


def prompt_tokens(base, body, timeout):
    probe = dict(body, max_tokens=1)
    return post(base + "/v1/chat/completions", probe, timeout)["usage"]["prompt_tokens"]


def calibrate(base, probe, model, target, seed, timeout):
    """Smallest size unit count whose prompt is about `target` tokens (two-point linear fit)."""
    lo, hi = (200, 2000) if probe == "needle" else (2, 20)
    tlo = prompt_tokens(base, request(probe, model, lo, 0.5, seed, 1), timeout)
    thi = prompt_tokens(base, request(probe, model, hi, 0.5, seed, 1), timeout)
    per = (thi - tlo) / (hi - lo)
    return max(1, int(lo + (target - tlo) / per))


def verdict(probe, msg):
    content = msg.get("content") or ""
    if probe == "needle":
        return ANSWER in content, {"content": content[:400]}
    calls = msg.get("tool_calls") or []
    detail = {"content": content[:400], "tool_calls": calls[:3]}
    if len(calls) != 1 or calls[0]["function"]["name"] != "run_command":
        return False, detail
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (ValueError, TypeError):
        return False, detail
    return PORT in str(args.get("command", "")), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", default="needle,transcript")
    ap.add_argument("--lengths", required=True, help="comma-separated target prompt tokens")
    ap.add_argument("--depths", default="0.1,0.5,0.9", help="needle depths")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--timeout-s", type=int, required=True)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    results = []
    for probe in a.probes.split(","):
        for target in [int(x) for x in a.lengths.split(",")]:
            size = calibrate(base, probe, a.model, target, a.seed, a.timeout_s)
            depths = [float(d) for d in a.depths.split(",")] if probe == "needle" else [0.0]
            for depth in depths:
                body = request(probe, a.model, size, depth, a.seed, a.max_tokens)
                raw = json.dumps(body, sort_keys=True).encode()
                row = {"probe": probe, "target_tokens": target, "size_units": size,
                       "depth": depth, "body_sha256": hashlib.sha256(raw).hexdigest()}
                t0 = time.time()
                try:
                    resp = post(base + "/v1/chat/completions", body, a.timeout_s)
                    msg = resp["choices"][0]["message"]
                    ok, detail = verdict(probe, msg)
                    row.update(ok=ok, usage=resp.get("usage"), detail=detail,
                               finish_reason=resp["choices"][0].get("finish_reason"),
                               reasoning_chars=len(msg.get("reasoning_content") or msg.get("reasoning") or ""))
                except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
                    row.update(ok=False, error=repr(exc)[:400])
                row["wall_s"] = round(time.time() - t0, 2)
                results.append(row)
                print(f"{probe:10s} {target:>7} depth={depth:.1f} ok={row['ok']} "
                      f"ptok={(row.get('usage') or {}).get('prompt_tokens')} wall={row['wall_s']}s "
                      f"{row.get('error', '')}", flush=True)
    passed = sum(r["ok"] for r in results)
    with open(__file__, "rb") as f:
        me = hashlib.sha256(f.read()).hexdigest()
    json.dump({"url": a.url, "model": a.model, "seed": a.seed, "driver_sha256": me,
               "passed": passed, "total": len(results), "results": results},
              open(a.out, "w"), indent=2)
    print(f"# {passed}/{len(results)} passed -> {a.out}")


if __name__ == "__main__":
    main()
