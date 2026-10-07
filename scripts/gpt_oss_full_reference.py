#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Pinned eager reference only; dequantized BF16 does not establish native packed support.
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import types

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, Mxfp4Config
from transformers.models.gpt_oss import modeling_gpt_oss
from transformers.integrations import mxfp4

REVISION = '6cee5e81ee83917806bbde320786a8fb61efebee'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backup-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-ids', default='1,2,3,4')
    parser.add_argument('--input-ids-file', type=Path)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--attention-policy', choices=['pinned-bf16','diagnostic-fp32'], default='pinned-bf16')
    args = parser.parse_args()
    if transformers.__version__ != '4.55.0':
        raise RuntimeError('reference requires Transformers 4.55.0')
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if (config.model_type, config.num_hidden_layers, config.hidden_size, config.vocab_size) != ('gpt_oss',24,2880,201088):
        raise RuntimeError('unexpected checkpoint architecture')
    ids = json.loads(args.input_ids_file.read_text())['input_ids'] if args.input_ids_file else [int(x) for x in args.input_ids.split(',')]
    if not ids or any(x < 0 or x >= config.vocab_size for x in ids):
        raise ValueError('input token outside vocabulary')
    manifest = json.loads(args.backup_manifest.read_text())
    if manifest['repo'] != 'openai/gpt-oss-20b' or manifest['revision'] != REVISION:
        raise RuntimeError('unexpected checkpoint revision')
    provenance = dict(scope='Transformers eager reference; MXFP4 weights explicitly dequantized to BF16',
                      revision=REVISION, torch=torch.__version__, transformers=transformers.__version__,
                      input_ids=ids, attention_implementation='eager', attention_policy=args.attention_policy, gpu_memory_fraction=0.85,
                      source_hashes={str(p):sha(p) for p in [Path(__file__),Path(inspect.getfile(modeling_gpt_oss)),Path(inspect.getfile(mxfp4))]},
                      backup_manifest_sha256=sha(args.backup_manifest), layer_trace='after complete decoder layer including post-MoE residual, before final norm',
                      logits_trace='LM head after final norm, stored BF16', inspection_only=args.inspect_only)
    if not args.inspect_only and any((args.output/name).exists() for name in ['reference.layers.bf16','reference.logits.bf16','reference.json']):
        raise FileExistsError('reference output exists; use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.inspect_only:
        print(json.dumps(provenance,indent=2))
        return
    # 2026-10-07: Verify every loaded checkpoint component against the completed backup manifest before allocation.
    index = json.loads((args.checkpoint/'model.safetensors.index.json').read_text())
    required = sorted(set(index['weight_map'].values()) | {'config.json','model.safetensors.index.json'})
    records = {f['path']:f for f in manifest['files']}
    provenance['checkpoint_sha256'] = {}
    for name in required:
        actual = sha(args.checkpoint/name)
        if name not in records or actual != records[name]['sha256']:
            raise RuntimeError(f'checkpoint digest mismatch: {name}')
        provenance['checkpoint_sha256'][name] = actual
    # 2026-10-07: Optional reference-only ablation promotes inputs to the unchanged pinned eager body.
    # Dense FP32 reduction is not the native online reduction; this isolates dtype effects only.
    if args.attention_policy == 'diagnostic-fp32':
        original = modeling_gpt_oss.eager_attention_forward
        def fp32_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            proxy = types.SimpleNamespace(num_key_value_groups=module.num_key_value_groups,
                                          sinks=module.sinks.float(), training=module.training)
            output, weights = original(proxy, query.float(), key.float(), value.float(),
                                       None if attention_mask is None else attention_mask.float(),
                                       scaling, dropout=dropout, **kwargs)
            return output.to(query.dtype), weights
        modeling_gpt_oss.eager_attention_forward = fp32_attention
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    budget = int(torch.cuda.get_device_properties(0).total_memory * 0.85)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True), device_map={'':'cuda:0'},
        max_memory={0:budget}, low_cpu_mem_usage=True, attn_implementation='eager',
    ).eval()
    provenance['load_seconds'] = time.perf_counter()-start
    provenance['device'] = torch.cuda.get_device_name(0)
    if len(model.model.layers) != 24 or any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('reference unexpectedly offloaded or incomplete')
    layer_rows = []
    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(lambda module, inputs, output: layer_rows.append(output.detach().cpu().contiguous())))
    cache = None
    next_ids = []
    elapsed = []
    try:
        with (args.output/'reference.layers.bf16').open('xb') as layers_file, (args.output/'reference.logits.bf16').open('xb') as logits_file, torch.inference_mode():
            for token in ids:
                layer_rows.clear()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                out = model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True,return_dict=True)
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-begin)
                cache = out.past_key_values
                if len(layer_rows) != 24:
                    raise RuntimeError(f'incomplete layer trace: {len(layer_rows)}')
                hidden = torch.cat(layer_rows,dim=0).contiguous()
                logits = out.logits[0,-1].to(torch.bfloat16).cpu().contiguous()
                if not torch.isfinite(hidden).all() or not torch.isfinite(logits).all():
                    raise RuntimeError('reference produced nonfinite values')
                layers_file.write(hidden.view(torch.int16).numpy().astype('<i2',copy=False).tobytes())
                logits_file.write(logits.view(torch.int16).numpy().astype('<i2',copy=False).tobytes())
                next_ids.append(int(logits.float().argmax()))
                print(json.dumps(dict(token=token,next_id=next_ids[-1],seconds=elapsed[-1])),flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    provenance.update(layer_shape=[len(ids),24,2880],logits_shape=[len(ids),201088],next_ids=next_ids,
                      step_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      output_hashes={name:sha(args.output/name) for name in ['reference.layers.bf16','reference.logits.bf16']})
    (args.output/'reference.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    main()
