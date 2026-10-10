#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-10: Reference math of a DFlash2 block-diffusion drafter on the CPU (numpy, f32), for
offline parity and acceptance replay against an engine's dumps.

Semantics (the drafter checkpoint's published architecture, independent of any engine kernel):
  ctx row j  c_j = hidden_norm(fc(concat of the tapped target layers at position j))
             per layer: K_j = rope(k_norm(k_proj c_j), pos_j), V_j = v_proj c_j
  block      embed([anchor, mask x (w-1)]) at positions p .. p+w-1
  layer      h = input_ln(r); (h, d) = attn_conv.prepare(h); a = attn(h) over ctx + block, NOT causal,
             keys at distance >= sliding_window masked; r += attn_conv.finish(a, d)
             h = post_ln(r); (h, d) = mlp_conv.prepare(h); r += mlp_conv.finish(mlp(h), d)
  conv       per row t of each block: out_t = (base[0] + delta_t[0]) * x_t + (base[1] + delta_t[1]) * x_{t-1},
             the tap-1 term absent on the block's first row; delta is per channel group
  output     out = norm(r); logits = lm_head(out); top-k per row; greedy walk of the candidate
             selector from the anchor: score(prev -> c) = logit_c + succ[c] . (pred[prev] * hp(out_t))
`weights="fp8row"` rounds the seven per-layer GEMM weights and the lm_head to FP8 E4M3 with one
absmax scale per output row, to emulate a drafter served with row-scaled FP8 weights.
"""
import json
import os
import struct

import numpy as np


def bf16_to_f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


class SafeTensors:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        h = self.header[name]
        if h["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, found {h['dtype']}")
        a, b = h["data_offsets"]
        return self.mm[self.base + a:self.base + b].view(np.uint16).reshape(h["shape"])

    def f32(self, name):
        return bf16_to_f32(self.raw(name))


def checkpoint_tensor(model_dir, name):
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        shard = json.load(open(index))["weight_map"][name]
    else:
        shard = "model.safetensors"
    return SafeTensors(os.path.join(model_dir, shard)).raw(name)


def find_tensor(model_dir, suffixes):
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    for s in suffixes:
        hits = [k for k in index if k == s or k.endswith("." + s)]
        if len(hits) == 1:
            return checkpoint_tensor(model_dir, hits[0])
    raise KeyError(f"none of {suffixes} found exactly once in {model_dir}")


def e4m3_round(x):
    """Round f32 values (already scaled into [-448, 448]) to the nearest FP8 E4M3 value."""
    a = np.minimum(np.abs(x), 448.0)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    step = 2.0 ** (np.maximum(e, -6.0) - 3.0)
    return (np.sign(x) * np.round(a / step) * step).astype(np.float32)


def fp8_row(w):
    scale = np.max(np.abs(w), axis=1, keepdims=True) / 448.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    return e4m3_round(w / scale) * scale


def mm(x, w):
    """x [..., k] @ w [k, n] as one 2-D GEMM (a batched 3-D matmul runs one small GEMM per block)."""
    return (x.reshape(-1, x.shape[-1]) @ w).reshape(*x.shape[:-1], w.shape[1])


def rms_norm(x, w, eps):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, pos, theta):
    """x [n, heads, d], rotate-half (neox) layout; pos [n]."""
    d = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def grouped_conv(x, delta, base, group):
    """x [b, w, h] (b blocks of w rows); delta [b, w, taps, groups]; base [taps, h]."""
    gidx = np.arange(x.shape[-1]) // group
    out = (base[0] + delta[:, :, 0, :][..., gidx]) * x
    for tap in range(1, base.shape[0]):
        c = base[tap] + delta[:, :, tap, :][..., gidx]
        out[:, tap:] += c[:, tap:] * x[:, :-tap]
    return out


class RefDrafter:
    def __init__(self, drafter_dir, target_dir, weights="bf16"):
        if weights not in ("bf16", "fp8row"):
            raise ValueError(f"weights must be bf16 or fp8row, got {weights}")
        cfg = json.load(open(os.path.join(drafter_dir, "config.json")))
        d = cfg["dflash_config"]
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.group = int(d["conv_group_size"])
        self.mask_id = int(d["mask_token_id"])
        self.top_k = int(d["selector_top_k"])
        self.layer_ids = list(d["target_layer_ids"])
        self.embed_scale = float(d.get("input_embedding_scale", 1.0))
        if float(d.get("output_multiplier", 1.0)) != 1.0 or d.get("final_logit_softcapping"):
            raise ValueError("output_multiplier / final_logit_softcapping are not implemented here")
        self.window = cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None
        self.n_layers = cfg["num_hidden_layers"]
        self.vocab = cfg["vocab_size"]
        st = SafeTensors(os.path.join(drafter_dir, "model.safetensors"))
        q = fp8_row if weights == "fp8row" else (lambda w: w)
        self.fc = st.f32("fc.weight").T.copy()
        self.hidden_norm = st.f32("hidden_norm.weight")
        self.norm = st.f32("norm.weight")
        self.layers = []
        for li in range(self.n_layers):
            p = f"layers.{li}."
            g = lambda n: st.f32(p + n)
            self.layers.append({
                "in_ln": g("input_layernorm.weight"), "post_ln": g("post_attention_layernorm.weight"),
                "q": q(g("self_attn.q_proj.weight")).T.copy(), "k": q(g("self_attn.k_proj.weight")).T.copy(),
                "v": q(g("self_attn.v_proj.weight")).T.copy(), "o": q(g("self_attn.o_proj.weight")).T.copy(),
                "k_ctx": g("self_attn.k_proj.weight").T.copy(), "v_ctx": g("self_attn.v_proj.weight").T.copy(),
                "q_norm": g("self_attn.q_norm.weight"), "k_norm": g("self_attn.k_norm.weight"),
                "gate": q(g("mlp.gate_proj.weight")).T.copy(), "up": q(g("mlp.up_proj.weight")).T.copy(),
                "down": q(g("mlp.down_proj.weight")).T.copy(),
                "a_proj": g("attention_conv.kernel_projection.weight").T.copy(),
                "a_base": g("attention_conv.base_kernel"),
                "m_proj": g("mlp_conv.kernel_projection.weight").T.copy(),
                "m_base": g("mlp_conv.base_kernel"),
            })
        self.hp = st.f32("candidate_selector.hidden_projection.weight").T.copy()
        self.pred = st.raw("candidate_selector.predecessor_codebook")
        self.succ = st.raw("candidate_selector.successor_codebook")
        self.embed = find_tensor(target_dir, ["embed_tokens.weight"])
        lm = find_tensor(target_dir, ["lm_head.weight"])
        self.lm_head = q(bf16_to_f32(lm[: self.vocab])).T.copy()
        self.ctx_k = self.ctx_v = self.ctx_pos = None

    def set_ctx(self, taps, positions):
        """taps [n, n_layers_tapped * hidden] (f32), positions [n]: K/V of every ctx row."""
        c = rms_norm(taps @ self.fc, self.hidden_norm, self.eps)
        self.ctx_pos = np.asarray(positions, dtype=np.int64)
        self.ctx_k, self.ctx_v = [], []
        n = len(self.ctx_pos)
        for L in self.layers:
            k = rms_norm((c @ L["k_ctx"]).reshape(n, self.nkv, self.hd), L["k_norm"], self.eps)
            self.ctx_k.append(rope(k, self.ctx_pos, self.theta))
            self.ctx_v.append((c @ L["v_ctx"]).reshape(n, self.nkv, self.hd))

    def forward(self, anchors, positions, ctx_counts, width):
        """`len(anchors)` independent blocks of `width` rows (anchor + masks); block b sits at
        positions[b] .. positions[b]+width-1 and attends the first ctx_counts[b] ctx rows given
        to `set_ctx`. Returns the final normed rows [b, w, h]."""
        b = len(anchors)
        ids = np.array([[a] + [self.mask_id] * (width - 1) for a in anchors])
        r = (bf16_to_f32(self.embed[ids.reshape(-1)]) * self.embed_scale).reshape(b, width, -1)
        q_pos = np.asarray(positions)[:, None] + np.arange(width)[None, :]
        rep = self.nh // self.nkv
        for li, L in enumerate(self.layers):
            h = rms_norm(r, L["in_ln"], self.eps)
            dyn = mm(h, L["a_proj"]).reshape(b, width, 2, L["a_base"].shape[1], -1)
            h1 = grouped_conv(h, dyn[:, :, 0], L["a_base"][0], self.group)
            q = rms_norm(mm(h1, L["q"]).reshape(b, width, self.nh, self.hd), L["q_norm"], self.eps)
            k = rms_norm(mm(h1, L["k"]).reshape(b, width, self.nkv, self.hd), L["k_norm"], self.eps)
            v = mm(h1, L["v"]).reshape(b, width, self.nkv, self.hd)
            att = np.empty((b, width, self.nh * self.hd), dtype=np.float32)
            for i in range(b):
                n = int(ctx_counts[i])
                qi = rope(q[i], q_pos[i], self.theta)
                K = np.repeat(np.concatenate([self.ctx_k[li][:n], rope(k[i], q_pos[i], self.theta)], 0), rep, axis=1)
                V = np.repeat(np.concatenate([self.ctx_v[li][:n], v[i]], 0), rep, axis=1)
                s = np.einsum("qhd,khd->hqk", qi, K) / np.sqrt(self.hd)
                if self.window:
                    k_pos = np.concatenate([self.ctx_pos[:n], q_pos[i]])
                    s = np.where(((q_pos[i][:, None] - k_pos[None, :]) >= self.window)[None], -np.inf, s)
                s = s - s.max(-1, keepdims=True)
                pr = np.exp(s)
                pr /= pr.sum(-1, keepdims=True)
                att[i] = np.einsum("hqk,khd->qhd", pr, V).reshape(width, -1)
            r = r + grouped_conv(mm(att, L["o"]), dyn[:, :, 1], L["a_base"][1], self.group)
            h = rms_norm(r, L["post_ln"], self.eps)
            dyn = mm(h, L["m_proj"]).reshape(b, width, 2, L["m_base"].shape[1], -1)
            h1 = grouped_conv(h, dyn[:, :, 0], L["m_base"][0], self.group)
            g = mm(h1, L["gate"])
            m = mm(g * 0.5 * (1.0 + np.tanh(0.5 * g)) * mm(h1, L["up"]), L["down"])
            r = r + grouped_conv(m, dyn[:, :, 1], L["m_base"][1], self.group)
        return rms_norm(r, self.norm, self.eps)

    def propose(self, anchors, positions, ctx_counts, width):
        """Per block: drafts for rows 1..width-1, per-row top-k ids/logits, and the selector's
        margin (best minus second-best edge score) at each row."""
        out = self.forward(anchors, positions, ctx_counts, width)[:, 1:]
        all_logits = mm(out, self.lm_head)
        all_hp = mm(out, self.hp)
        res = []
        for i, anchor in enumerate(anchors):
            logits = all_logits[i]
            top = np.argpartition(-logits, self.top_k, axis=1)[:, : self.top_k]
            order = np.argsort(-np.take_along_axis(logits, top, 1), axis=1)
            top = np.take_along_axis(top, order, 1)
            vals = np.take_along_axis(logits, top, 1)
            hp = all_hp[i]
            drafts, margins, prev = [], [], int(anchor)
            for t in range(width - 1):
                gate = bf16_to_f32(self.pred[prev]) * hp[t]
                sc = vals[t] + bf16_to_f32(self.succ[top[t]]) @ gate
                o = np.argsort(-sc)
                prev = int(top[t][o[0]])
                drafts.append(prev)
                margins.append(float(sc[o[0]] - sc[o[1]]))
            res.append({"drafts": drafts, "top_ids": top, "top_vals": vals, "margins": margins,
                        "logits": logits})
        return res
