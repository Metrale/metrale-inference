#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Full-row selected-expert replay; exact gate, retain failures.

Native original PTX vs pinned CUDA BF16 B32 reference geometry; unused experts
have zero weights solely to preserve batch geometry without loading their data.
B4 is a diagnostic control, never a substitute for the pinned B32 comparison.
Learned tensors remain on the execution host and are recorded by hash only.
"""
import argparse
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--modules', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', default='215:6,49:9', help='Explicit position:layer pairs from retained snapshots')
    a = p.parse_args()
    cases = [tuple(int(part) for part in item.split(':')) for item in a.cases.split(',')]
    if not cases or any(len(case) != 2 or case[0] < 0 or not 0 <= case[1] < 24 for case in cases):
        p.error('cases require nonnegative position and layer0..23')
    a.output.mkdir(exist_ok=False)
    assert transformers.__version__ == '4.55.0'
    reference_hashes = {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for name, fn in [('unpack', convert_moe_packed_tensors), ('experts', GptOssExperts.forward)]}
    assert reference_hashes == {'unpack': '9c9e8facbc0eebfa943a6a4142dd8a745cb85a68accc88547757265d1c5729d8', 'experts': 'f9965c8654b80c241df8ab3d3bbf3018a488c0563071de3e64959ad0869cb574'}
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = dict(gate='exact BF16 bits; preserve failures, no tolerance widening', model='openai/gpt-oss-20b', expected_revision='6cee5e81ee83917806bbde320786a8fb61efebee', torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), reference_hashes=reference_hashes, source_sha256=digest(__file__), comparisons=[], controls=[], weight_hashes={}, snapshot_hashes={}, ptx_hashes={})
    driver = ctypes.CDLL('libcuda.so.1')
    anchor = torch.empty(1, device='cuda')
    modules = json.loads(a.modules.read_text())['modules']
    handles, functions = [], {}
    def call(name, *args):
        code = getattr(driver, name)(*args)
        if code:
            raise RuntimeError(f'{name}: CUDA {code}')
    for name, symbol in [('gpt_oss_mxfp4_gemv', 'gpt_oss_mxfp4_gemv_bf16'), ('nllb_encoder', 'nllb_bias_bf16'), ('gpt_oss_expert_ops', 'gpt_oss_swiglu_bf16')]:
        module = next(m for m in modules if m['name'] == name)
        path = a.modules.parent / module['ptx']
        assert digest(path) == module['ptx_sha256']
        report['ptx_hashes'][name] = digest(path)
        handle, function = ctypes.c_void_p(), ctypes.c_void_p()
        call('cuModuleLoad', ctypes.byref(handle), ctypes.c_char_p(str(path).encode()))
        call('cuModuleGetFunction', ctypes.byref(function), handle, ctypes.c_char_p(symbol.encode()))
        handles.append(handle)
        functions[name] = function
    def launch(module, grid, block, args):
        packed = (ctypes.c_void_p * len(args))(*(ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in args))
        call('cuLaunchKernel', functions[module], grid, 1, 1, block, 1, 1, 0, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), packed, None)
        torch.cuda.synchronize()
    def ptr(t):
        return ctypes.c_void_p(t.data_ptr())
    def gemv(blocks, scales, x, rows):
        y = torch.empty(rows, dtype=torch.bfloat16, device='cuda')
        launch('gpt_oss_mxfp4_gemv', (rows+3)//4, 128, [ptr(blocks), ptr(scales), ptr(x), ptr(y), ctypes.c_uint(rows), ctypes.c_uint(2880)])
        return y
    def bias(x, b):
        y = x.clone()
        launch('nllb_encoder', (x.numel()+255)//256, 256, [ptr(y), ptr(b), ctypes.c_uint(1), ctypes.c_uint(x.numel())])
        return y
    def activation(x):
        y = torch.empty(2880, dtype=torch.bfloat16, device='cuda')
        launch('gpt_oss_expert_ops', 12, 256, [ptr(x), ptr(y), ctypes.c_uint(2880)])
        return y
    def reference_activation(x):
        gate = x[..., ::2].clamp(max=7)
        up = x[..., 1::2].clamp(min=-7, max=7)
        return (up + 1) * (gate * torch.sigmoid(gate * 1.702))
    def compare(name, native, reference):
        native, reference = native.reshape(-1), reference.reshape(-1)
        bits = int((native.view(torch.int16) != reference.view(torch.int16)).sum())
        delta = native.float() - reference.float()
        report['comparisons'].append(dict(name=name, role='diagnostic_batch_shape' if 'bmm4-vs32' in name else 'exact_gate', values=native.numel(), mismatches=bits, max_abs=float(delta.abs().max())))
        (a.output/(name+'-native.bin')).write_bytes(raw(native))
        (a.output/(name+'-reference.bin')).write_bytes(raw(reference))
    index = json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def weight(layer, suffix, expert):
        name = f'model.layers.{layer}.mlp.experts.{suffix}'
        with safe_open(a.checkpoint/index[name], framework='pt', device='cpu') as f:
            x = f.get_slice(name)[expert].contiguous()
        report['weight_hashes'][f'{name}[{expert}]'] = hashlib.sha256(raw(x)).hexdigest()
        return x.cuda()
    def snapshot(position, layer, name, shape, dtype='<i2'):
        path = a.snapshots/f'p{position}-l{layer}-{name}.bin'
        report['snapshot_hashes'][path.name] = digest(path)
        x = torch.from_numpy(np.fromfile(path, dtype=dtype).copy())
        if dtype == '<i2':
            x = x.view(torch.bfloat16)
        return x.reshape(shape).cuda()
    try:
        for position, layer in cases:
            prefix = f'p{position}-l{layer}'
            x = snapshot(position, layer, 'post_attention_norm', (2880,))
            ids = snapshot(position, layer, 'router_ids', (4,), '<i4').tolist()
            captured = snapshot(position, layer, 'selected_experts', (4, 2880))
            gate_weights = torch.zeros((32, 2880, 5760), dtype=torch.bfloat16, device='cuda')
            down_weights = torch.zeros((32, 2880, 2880), dtype=torch.bfloat16, device='cuda')
            report.setdefault('geometry', []).append(dict(position=position, layer=layer, gate_shape=list(gate_weights.shape), gate_stride=list(gate_weights.stride()), down_shape=list(down_weights.shape), down_stride=list(down_weights.stride()), batch32_unused_experts='zero weights, shape control only'))
            packed, biases = [], []
            for expert in ids:
                gb, gs = weight(layer, 'gate_up_proj_blocks', expert), weight(layer, 'gate_up_proj_scales', expert)
                db, ds = weight(layer, 'down_proj_blocks', expert), weight(layer, 'down_proj_scales', expert)
                gate_weights[expert].copy_(convert_moe_packed_tensors(gb, gs).T)
                down_weights[expert].copy_(convert_moe_packed_tensors(db, ds).T)
                packed.append((gb, gs, db, ds))
                biases.append((weight(layer, 'gate_up_proj_bias', expert), weight(layer, 'down_proj_bias', expert)))
            repeated = x.repeat(32, 1).reshape(32, 1, 2880)
            gate32 = torch.bmm(repeated, gate_weights).reshape(32, 5760)
            gate4 = torch.bmm(repeated[:4], gate_weights[ids]).reshape(4, 5760)
            native_acts, reference_acts, native_downs = [], [], []
            for slot, expert in enumerate(ids):
                name = f'{prefix}-e{expert}'
                gb, gs, db, ds = packed[slot]
                gpre = gemv(gb, gs, x, 5760)
                compare(name+'-gate-bmm32', gpre, gate32[expert])
                compare(name+'-gate-bmm4-vs32', gate4[slot], gate32[expert])
                g = bias(gpre, biases[slot][0])
                compare(name+'-bias-isolated', g, gpre + biases[slot][0])
                act = activation(g)
                compare(name+'-activation-isolated', act, reference_activation(g))
                native_acts.append(act)
                reference_acts.append(reference_activation(gate32[expert] + biases[slot][0]))
                down = bias(gemv(db, ds, act, 2880), biases[slot][1])
                native_downs.append(down)
                compare(name+'-captured-identity', down, captured[slot])
                if slot == 0:
                    wrong_layout = reference_activation(torch.cat([g[1::2], g[::2]]))
                    wrong_rounding = ((g[1::2].float().clamp(-7, 7)+1) * (g[::2].float().clamp(max=7) * torch.sigmoid(g[::2].float().clamp(max=7)*1.702))).bfloat16()
                    control = dict(case=name, wrong_layout=int((wrong_layout != act).sum()), omitted_bf16_rounding=int((wrong_rounding != act).sum()))
                    assert control['wrong_layout'] and control['omitted_bf16_rounding']
                    report['controls'].append(control)
            input32 = torch.zeros((32, 1, 2880), dtype=torch.bfloat16, device='cuda')
            input32[ids, 0] = torch.stack(native_acts)
            down32 = torch.bmm(input32, down_weights).reshape(32, 2880)
            down4 = torch.bmm(torch.stack(native_acts)[:, None], down_weights[ids]).reshape(4, 2880)
            input32[ids, 0] = torch.stack(reference_acts)
            full_reference = torch.bmm(input32, down_weights).reshape(32, 2880)
            for slot, expert in enumerate(ids):
                name = f'{prefix}-e{expert}'
                compare(name+'-down-bmm4-vs32', down4[slot], down32[expert])
                _, _, db, ds = packed[slot]
                compare(name+'-down-raw-bmm32', gemv(db, ds, native_acts[slot], 2880), down32[expert])
                compare(name+'-down-bmm32-isolated', native_downs[slot], down32[expert]+biases[slot][1])
                compare(name+'-full-reference', native_downs[slot], full_reference[expert]+biases[slot][1])
            # 2026-10-07: Isolate selected-output reduction from upstream expert arithmetic.
            scores = snapshot(position, layer, 'router_scores', (32,))
            native32 = torch.zeros((32, 2880), dtype=torch.bfloat16, device='cuda')
            reference32 = torch.zeros_like(native32)
            native32[ids] = torch.stack(native_downs)
            reference32[ids] = torch.stack([full_reference[expert]+biases[slot][1] for slot, expert in enumerate(ids)])
            native_reduced = (native32 * scores[:, None]).sum(0)
            reference_reduced = (reference32 * scores[:, None]).sum(0)
            captured_moe = snapshot(position, layer, 'moe', (2880,))
            compare(prefix+'-weighted-reduce-isolated', captured_moe, native_reduced)
            compare(prefix+'-full-moe-reference', captured_moe, reference_reduced)
            del native32, reference32, native_reduced, reference_reduced, captured_moe
            del gate_weights, down_weights, packed, biases, gate32, gate4, native_acts, reference_acts, native_downs, input32, down32, down4, full_reference
            torch.cuda.empty_cache()
        report['passed'] = all(row['mismatches'] == 0 for row in report['comparisons'] if row['role'] == 'exact_gate')
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        (a.output/'comparison.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    finally:
        for handle in handles:
            call('cuModuleUnload', handle)


if __name__ == '__main__':
    raise SystemExit(main())
