#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-10: Offline DFlash2 acceptance replay and drafter parity from one serve's dumps.

Inputs (one greedy single-stream request, see dflash_capture.py):
  --ctx-dir D   ctx rows the drafter attended at the dump's propose: D/metrale_ctx_parity.bin
                (BF16 [ctx_len, n_taps * hidden]) and D/metrale_ctx_parity.json (ctx_positions, ...)
  --gen G       the capture: completion token ids and usage.prompt_tokens
  --log L       the serve log with one `DFLASH TRACE drafts: token_in=.. position=.. drafts_pre_cap=[..]`
                line per propose (row 0 first) and the `DFLASH K=γ verify: .. accepted=a/k` lines
  --block-dir B optional: metrale_block_drafts.json + metrale_block_logits.bin of one propose

Reports, with the reference math of dflash2_ref.py:
  consistency  ctx positions contiguous; every traced anchor equals the generated token at its
               position (the position convention); logged accept counts equal the ones recomputed
               from the traced drafts against the generated tokens (the replay's ground truth).
  parity       per traced propose: engine drafts vs reference drafts with the same anchor,
               position and ctx prefix; first diverging row and the reference selector margin there
               (a numerics flip has a small margin, a defect a large one).
  acceptance   per-position acceptance (accepted at row i / proposes) of the engine's drafts and of
               the reference drafts at the same proposes, and the reference's at EVERY generated
               position (no dependence on where the serve's steps happened to land).
  taps         with --ctx-dir-b: per-tap relative error between two dumps of the same token sequence
               (e.g. decode/verify rows vs a teacher-forced prefill of the same ids).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash2_ref as ref  # noqa: E402

TRACE = re.compile(r"DFLASH TRACE drafts: token_in=(\d+) position=(\d+) .*drafts_pre_cap=\[([\d, ]*)\]")
VERIFY = re.compile(r"DFLASH K=.* verify: γ=(\d+) accepted=(\d+)/(\d+)")


def load_ctx(d):
    meta = json.load(open(os.path.join(d, "metrale_ctx_parity.json")))
    raw = np.fromfile(os.path.join(d, "metrale_ctx_parity.bin"), dtype=np.uint16)
    rows = ref.bf16_to_f32(raw).reshape(meta["ctx_len"], -1)
    return rows, np.array(meta["ctx_positions"], dtype=np.int64), meta


def parse_log(path):
    steps, verifies = [], []
    for line in open(path, errors="replace"):
        m = TRACE.search(line)
        if m:
            pre = [int(x) for x in m.group(3).split(",") if x.strip()]
            steps.append({"anchor": int(m.group(1)), "position": int(m.group(2)), "drafts": pre[1:]})
            continue
        m = VERIFY.search(line)
        if m:
            verifies.append(int(m.group(2)))
    return steps, verifies


def accept_len(drafts, truth):
    n = 0
    for d, t in zip(drafts, truth):
        if d != t:
            break
        n += 1
    return n


def per_position(acc, width):
    acc = np.array(acc)
    return [float(np.mean(acc >= i)) for i in range(1, width + 1)] if len(acc) else []


def fmt(v):
    return " ".join(f"{x:.3f}" for x in v)


def taps_compare(a, pa, b, pb, n_taps, prompt_len):
    common = np.intersect1d(pa, pb)
    ia = np.searchsorted(pa, common)
    ib = np.searchsorted(pb, common)
    A = a[ia].reshape(len(common), n_taps, -1)
    B = b[ib].reshape(len(common), n_taps, -1)
    err = np.linalg.norm(A - B, axis=-1) / np.maximum(np.linalg.norm(B, axis=-1), 1e-30)
    print(f"TAPS {len(common)} shared positions (prompt < {prompt_len}); relative L2 error per tap")
    for name, sel in (("prompt", common < prompt_len), ("generated", common >= prompt_len)):
        if not sel.any():
            continue
        e = err[sel]
        print(f"  {name:9s} rows {int(sel.sum()):4d}  median " + fmt(np.median(e, 0))
              + "  p99 " + fmt(np.quantile(e, 0.99, axis=0)) + "  max " + fmt(e.max(0)))
    worst = np.argsort(-err.max(1))[:5]
    print("  worst positions: " + ", ".join(f"{int(common[w])}:{err[w].max():.3f}" for w in worst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ctx-dir", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--log")
    ap.add_argument("--block-dir")
    ap.add_argument("--ctx-dir-b", help="second ctx dump of the same ids, for the taps comparison")
    ap.add_argument("--weights", choices=["bf16", "fp8row"], required=True)
    ap.add_argument("--width", type=int, required=True, help="block rows (anchor + drafts)")
    ap.add_argument("--all-positions", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    gen = json.load(open(a.gen))
    P = gen["usage"]["prompt_tokens"]
    comp = gen["completion_ids"]
    end = P + len(comp)

    def tok(pos):
        return comp[pos - P] if P <= pos < end else None

    rows, pos, meta = load_ctx(a.ctx_dir)
    n_taps = len(json.load(open(os.path.join(a.drafter, "config.json")))["dflash_config"]["target_layer_ids"])
    gaps = np.nonzero(np.diff(pos) != 1)[0]
    print(f"CTX rows {len(pos)} positions {pos[0]}..{pos[-1]} (dump at position {meta['position']}), "
          f"prompt {P}, generated {len(comp)}; gaps {len(gaps)}"
          + (f" first at rows {gaps[:8].tolist()} (pos {pos[gaps[:8]].tolist()})" if len(gaps) else ""))
    if a.ctx_dir_b:
        rb, pb, _ = load_ctx(a.ctx_dir_b)
        taps_compare(rows, pos, rb, pb, n_taps, P)

    # 2026-10-10: A propose at position p attends ctx rows at positions < p, so only positions
    # up to one past the dump's last row have their full ctx in the dump.
    last = int(pos[-1]) + 1
    d = ref.RefDrafter(a.drafter, a.target, a.weights)
    d.set_ctx(rows, pos)
    W = a.width

    def run(anchors, positions):
        out = []
        counts = [int(np.searchsorted(pos, p)) for p in positions]
        for s in range(0, len(anchors), a.batch):
            out += d.propose(anchors[s:s + a.batch], positions[s:s + a.batch], counts[s:s + a.batch], W)
        return out

    if a.block_dir:
        blk = json.load(open(os.path.join(a.block_dir, "metrale_block_drafts.json")))
        lg = ref.bf16_to_f32(np.fromfile(os.path.join(a.block_dir, "metrale_block_logits.bin"),
                                         dtype=np.uint16)).reshape(blk["gamma"], -1)
        r = run([blk["last_token"]], [blk["position"]])[0]
        print(f"BLOCK at position {blk['position']}: engine {blk['drafts'][1:]} ref {r['drafts']}")
        for t in range(W - 1):
            ours = lg[t + 1]
            kept = ours > -1e29
            diff = np.abs(ours[kept] - r["logits"][t][kept])
            print(f"  row {t + 1}: draft {'==' if blk['drafts'][t + 1] == r['drafts'][t] else '!='} "
                  f"masked {int((~kept).sum())} | |dlogit| unmasked mean {diff.mean():.4f} max {diff.max():.4f} "
                  f"| ref top1 {int(r['top_ids'][t][0])} engine top1 {int(np.argmax(ours))}")

    if a.log:
        steps, verifies = parse_log(a.log)
        print(f"LOG {len(steps)} traced proposes, {len(verifies)} K=γ verify lines")
        if len(verifies) in (len(steps), len(steps) - 1):
            for s, v in zip(steps, verifies):
                s["logged"] = v
        steps = [s for s in steps if P <= s["position"] <= last and len(s["drafts"]) >= W - 1]
        bad_anchor = [s for s in steps if s["anchor"] != tok(s["position"])]
        print(f"TRACE {len(steps)} proposes in the generated span; anchor != generated token at its "
              f"position: {len(bad_anchor)}" + (f" e.g. {bad_anchor[:3]}" if bad_anchor else ""))
        truth = [[tok(s["position"] + 1 + i) for i in range(W - 1)] for s in steps]
        eng_acc = [accept_len(s["drafts"][:W - 1], t) for s, t in zip(steps, truth)]
        logged = [(s["logged"], e) for s, e in zip(steps, eng_acc) if "logged" in s]
        print(f"VERIFY recomputed == logged accept count on {sum(x == y for x, y in logged)}/{len(logged)} proposes")
        refs = run([s["anchor"] for s in steps], [s["position"] for s in steps])
        ref_acc = [accept_len(r["drafts"], t) for r, t in zip(refs, truth)]
        agree = np.array([[x == y for x, y in zip(s["drafts"][:W - 1], r["drafts"])] for s, r in zip(steps, refs)])
        first = [next((i for i, ok in enumerate(row) if not ok), None) for row in agree]
        div = [(i, r["margins"][i]) for i, r in zip(first, refs) if i is not None]
        print(f"PARITY per-row draft agreement (engine vs reference): {fmt(agree.mean(0))}")
        print(f"  proposes identical {sum(f is None for f in first)}/{len(first)}; first-divergence row histogram "
              f"{np.bincount([i for i, _ in div], minlength=W - 1).tolist()}")
        if div:
            mg = np.array([m for _, m in div])
            print(f"  reference margin at first divergence: median {np.median(mg):.3f} p90 {np.quantile(mg, 0.9):.3f} max {mg.max():.3f}")
        print(f"ACCEPT engine drafts    mean {np.mean(eng_acc):.3f} per-pos {fmt(per_position(eng_acc, W - 1))}")
        print(f"ACCEPT reference drafts mean {np.mean(ref_acc):.3f} per-pos {fmt(per_position(ref_acc, W - 1))}")

    if a.all_positions:
        ps = list(range(P, min(end - 1, last + 1)))
        refs = run([tok(p) for p in ps], ps)
        acc = [accept_len(r["drafts"], [tok(p + 1 + i) for i in range(W - 1)]) for r, p in zip(refs, ps)]
        print(f"ACCEPT reference at every position ({len(ps)}): mean {np.mean(acc):.3f} per-pos {fmt(per_position(acc, W - 1))}")


if __name__ == "__main__":
    main()
