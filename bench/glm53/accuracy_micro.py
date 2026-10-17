#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-09: Fast accuracy microtest for one serve config, engine-neutral over the
OpenAI-compatible HTTP API (`/v1/completions` with `echo` + `logprobs`).

  dump    --url U --model M --out run.json
  compare --ref ref.json --ref-sha256 S --test run.json --mode exact|numerics [thresholds]

`dump` records three legs:
  tf      teacher-forced prompt logprobs (top-20 per position) over the fixed CORPUS: the
          prefill path at the declared reference's tokens.
  dec1    greedy decode, one request at a time: 8 prompts x 128 tokens, top-20 logprobs per
          generated token: the decode path at one row.
  dec4    the same prompts four at a time: the decode path at four rows.

`compare --mode exact` (bit-identical levers) requires every leg to equal the reference
byte for byte: tokens, chosen-token logprobs and top-20 lists. `compare --mode numerics`
(precision-changing levers) reports, per leg, top-1 agreement, a top-20 KL(ref || test), and
|delta logprob| of the forced or shared token. Decode legs are compared up to each prompt's
first divergence, and the reference's top-1/top-2 margin at that point is reported. A run
passes when every metric is inside the thresholds given on the command line; the thresholds
are set from measured good and bad arms (STATUS of the campaign), never defaulted here.

The reference is pinned: `--ref-sha256` must equal the SHA-256 of the reference file, and the
reference must carry this script's CORPUS_SHA256, so a drifting reference fails instead of
re-baselining.
"""
import argparse, hashlib, json, math, sys, threading, urllib.request

CORPUS = [
    "The river leaves the mountains as a narrow torrent, cutting through granite and carrying gravel that grinds the bed smooth. Further down, the valley widens and the current slows; silt settles in long banks where willows take root. Farmers have used these banks for centuries, planting barley in spring and flax in early summer, and the floods that once ruined harvests are now held back by a chain of low earthen dams built by hand.",
    "def merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n\n# The loop runs at most len(a) + len(b) times, so merging is linear in the total length.\n",
    "Let f(x) = 3x^2 - 12x + 7. The derivative is f'(x) = 6x - 12, which is zero at x = 2. Since f''(x) = 6 > 0, the point x = 2 is a minimum, and the minimum value is f(2) = 12 - 24 + 7 = -5. The parabola therefore never reaches below -5, and it crosses zero where x = 2 plus or minus the square root of 5/3.",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Lisbon\", \"unit\": \"celsius\", \"days\": 3}}\nThe tool returned: {\"forecast\": [{\"day\": 1, \"high\": 24, \"low\": 17}, {\"day\": 2, \"high\": 22, \"low\": 16}, {\"day\": 3, \"high\": 25, \"low\": 18}]}. Over the next three days Lisbon stays mild, with highs between 22 and 25 degrees.",
    "長江是中國最長的河流，全長約六千三百公里，流經青海、西藏、四川、湖北、江西、安徽和江蘇等地，最後在上海附近注入東海。沿岸城市眾多，航運發達，自古以來就是重要的交通要道。",
    "Q: A train leaves at 9:40 and the trip takes 2 hours and 35 minutes. When does it arrive?\nA: Adding 2 hours gives 11:40, and adding 35 more minutes gives 12:15. The train arrives at 12:15.\nQ: If it is delayed by 50 minutes, when does it arrive?\nA: 12:15 plus 50 minutes is 13:05.",
]
PROMPTS = [
    "Explain how a heat pump moves heat from a cold place to a warm one.",
    "Write a Python function that checks whether a string is a palindrome, then explain it.",
    "List five causes of the French Revolution and explain each in one sentence.",
    "Describe the life cycle of a star like the Sun.",
    "What are the tradeoffs between TCP and UDP?",
    "Translate into French: The library opens at nine and closes at six.",
    "Solve step by step: 17 * 23 + 144 / 12.",
    "Write a short story opening about a lighthouse keeper during a storm.",
]
CORPUS_SHA256 = hashlib.sha256(json.dumps([CORPUS, PROMPTS], ensure_ascii=False).encode()).hexdigest()
TOPK = 20
DEC_TOKENS = 128


def post(url, body):
    r = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def lp_block(choice):
    lp = choice.get("logprobs") or {}
    return {"tokens": lp.get("tokens", []), "lp": lp.get("token_logprobs", []),
            "top": lp.get("top_logprobs", [])}


def dump(a):
    out = {"corpus_sha256": CORPUS_SHA256, "tf": [], "dec1": [None] * len(PROMPTS), "dec4": [None] * len(PROMPTS)}
    for text in CORPUS:
        d = post(a.url, {"model": a.model, "prompt": text, "max_tokens": 1, "echo": True,
                         "logprobs": TOPK, "temperature": 0})
        out["tf"].append(lp_block(d["choices"][0]))

    def gen(i, leg):
        d = post(a.url, {"model": a.model, "prompt": PROMPTS[i], "max_tokens": DEC_TOKENS,
                         "logprobs": TOPK, "temperature": 0, "ignore_eos": True})
        out[leg][i] = lp_block(d["choices"][0])

    for i in range(len(PROMPTS)):
        gen(i, "dec1")
    for s in range(0, len(PROMPTS), 4):
        th = [threading.Thread(target=gen, args=(i, "dec4")) for i in range(s, min(len(PROMPTS), s + 4))]
        [t.start() for t in th]; [t.join() for t in th]
    json.dump(out, open(a.out, "w"))
    print(f"dumped {a.out} sha256={hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")


def top1_set(top):
    """The tokens tied at the top logprob. Ties are common with BF16 logits, and a server may
    list the top-k in any order, so agreement is an intersection of the tied sets."""
    if not top:
        return set()
    m = max(top.values())
    return {t for t, v in top.items() if v == m}


def pos_metrics(rtop, ttop, rlp, tlp):
    """Per position: top-1 agreement, top-20 KL(ref||test) (a token missing from the test's
    top-20 takes the test's smallest listed logprob, so the KL is a lower bound), |dlogprob|."""
    agree = bool(top1_set(rtop) & top1_set(ttop))
    floor = min(ttop.values()) if ttop else -1e9
    kl = 0.0
    for tok, l in rtop.items():
        kl += math.exp(l) * (l - ttop.get(tok, floor))
    d = abs(rlp - tlp) if (rlp is not None and tlp is not None) else 0.0
    return agree, max(kl, 0.0), d


def compare(a):
    raw = open(a.ref, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != a.ref_sha256:
        sys.exit(f"FAIL: reference sha256 {sha} != pinned {a.ref_sha256}")
    ref = json.loads(raw); test = json.load(open(a.test))
    if ref.get("corpus_sha256") != CORPUS_SHA256 or test.get("corpus_sha256") != CORPUS_SHA256:
        sys.exit("FAIL: corpus sha256 mismatch (reference or run made from another corpus)")
    if a.mode == "exact":
        bad = [leg for leg in ("tf", "dec1", "dec4") if ref[leg] != test[leg]]
        for leg in bad:
            for i, (r, t) in enumerate(zip(ref[leg], test[leg])):
                if r != t:
                    n = next((k for k in range(min(len(r["tokens"]), len(t["tokens"])))
                              if r["tokens"][k] != t["tokens"][k] or r["lp"][k] != t["lp"][k]), None)
                    print(f"  {leg}[{i}] differs (first token/logprob difference at {n})")
        print("EXACT", "PASS" if not bad else f"FAIL ({', '.join(bad)})")
        sys.exit(0 if not bad else 1)
    ok = True
    for leg in ("tf", "dec1", "dec4"):
        agree = kls = n = 0; ds = []; margins = []; unmeasured = 0
        for r, t in zip(ref[leg], test[leg]):
            L = min(len(r["tokens"]), len(t["tokens"]))
            for k in range(L):
                if leg != "tf" and r["tokens"][k] != t["tokens"][k]:
                    rt = sorted((r["top"][k] or {}).values(), reverse=True)
                    if len(rt) > 1:
                        margins.append(rt[0] - rt[1])
                    else:
                        unmeasured += 1
                    break
                if not r["top"][k] or not t["top"][k]:
                    continue
                g, kl, d = pos_metrics(r["top"][k], t["top"][k], r["lp"][k], t["lp"][k])
                agree += g; kls += kl; ds.append(d); n += 1
        ds.sort()
        m = {"positions": n, "top1": agree / max(n, 1), "kl_mean": kls / max(n, 1),
             "dlp_p99": ds[int(0.99 * (len(ds) - 1))] if ds else 0.0, "dlp_max": ds[-1] if ds else 0.0,
             "diverged": len(margins) + unmeasured, "unmeasured_divergences": unmeasured,
             "max_margin_at_divergence": max(margins) if margins else None}
        if n == 0:
            checks = [False]
        elif leg == "tf":
            checks = [m["top1"] >= a.tf_min_top1, m["kl_mean"] <= a.tf_max_kl, m["dlp_p99"] <= a.tf_max_dlp_p99]
        else:
            checks = [m["kl_mean"] <= a.dec_max_kl, m["dlp_p99"] <= a.dec_max_dlp_p99,
                      unmeasured <= a.max_unmeasured_divergences,
                      (max(margins) if margins else 0.0) <= a.max_divergence_margin]
        leg_ok = all(checks); ok &= leg_ok
        print(f"{leg:5s} {'PASS' if leg_ok else 'FAIL'} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}))
    print("NUMERICS", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="cmd", required=True)
    d = s.add_parser("dump"); d.add_argument("--url", required=True); d.add_argument("--model", required=True); d.add_argument("--out", required=True)
    c = s.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("--ref-sha256", required=True)
    c.add_argument("--test", required=True); c.add_argument("--mode", choices=["exact", "numerics"], required=True)
    NUM = ("--tf-min-top1", "--tf-max-kl", "--tf-max-dlp-p99", "--dec-max-kl", "--dec-max-dlp-p99",
           "--max-divergence-margin", "--max-unmeasured-divergences")
    for f in NUM:
        c.add_argument(f, type=float, default=None)
    a = p.parse_args()
    if a.cmd == "dump":
        dump(a)
    else:
        if a.mode == "numerics" and any(getattr(a, f[2:].replace("-", "_")) is None for f in NUM):
            sys.exit("numerics mode needs every threshold: " + " ".join(NUM) + " (set from measured arms)")
        compare(a)


if __name__ == "__main__":
    main()
