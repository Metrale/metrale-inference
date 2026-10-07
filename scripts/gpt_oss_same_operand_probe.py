#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Isolate native/Transformers primitives using identical captured operands and pinned small weight slices.
import argparse
import ctypes
import hashlib
import json
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gpt_oss import modeling_gpt_oss as ref


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--modules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    config = AutoConfig.from_pretrained(a.checkpoint,local_files_only=True)
    index = json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    receipt = dict(scope='same-operand layer9 diagnostic, no full-model qualification',comparisons=[],source_sha256=digest(__file__),weights={},snapshots={})
    def weight(name):
        full='model.layers.9.'+name
        with safe_open(a.checkpoint/index[full],framework='pt',device='cpu') as f:
            x=f.get_tensor(full).contiguous()
        raw=x.view(torch.uint8).numpy().tobytes()
        receipt['weights'][full]=dict(shape=list(x.shape),sha256=hashlib.sha256(raw).hexdigest())
        return x.cuda()
    def snapshot(position,name,shape):
        path=a.snapshots/f'p{position}-l9-{name}.bin'
        receipt['snapshots'][str(path)]=digest(path)
        x=torch.from_numpy(np.fromfile(path,dtype='<i2').copy()).view(torch.bfloat16).reshape(shape).cuda()
        return x
    def save(name,x):
        raw=x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        (a.output/(name+'.bin')).write_bytes(raw)
    def compare(name,x,y):
        save(name+'-reference',x);save(name+'-native',y)
        d=x.float()-y.float()
        receipt['comparisons'].append(dict(name=name,shape=list(x.shape),bit_mismatches=int((x.view(torch.int16)!=y.view(torch.int16)).sum()),
                                          max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())))
    modules=json.loads(a.modules.read_text())
    norm=next(m for m in modules['modules'] if m['name']=='rms_norm_vanilla')
    ptx=a.modules.parent/norm['ptx']
    if digest(ptx)!=norm['ptx_sha256']:raise RuntimeError('PTX digest mismatch')
    receipt['norm_ptx_sha256']=digest(ptx)
    driver=ctypes.CDLL('libcuda.so.1')
    def call(name,*args):
        code=getattr(driver,name)(*args)
        if code:raise RuntimeError(f'{name}: CUDA{code}')
    # 2026-10-07: Torch has established this thread's primary context; load original production PTX into it.
    context_anchor = torch.empty(1,device='cuda')
    module=ctypes.c_void_p();fn=ctypes.c_void_p()
    call('cuModuleLoad',ctypes.byref(module),ctypes.c_char_p(str(ptx).encode()))
    call('cuModuleGetFunction',ctypes.byref(fn),module,ctypes.c_char_p(b'rms_norm_vanilla'))
    def native_norm(x,w):
        out=torch.empty_like(x)
        args=[ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(w.data_ptr()),ctypes.c_void_p(out.data_ptr()),ctypes.c_uint(2880),ctypes.c_float(config.rms_norm_eps)]
        ptrs=(ctypes.c_void_p*len(args))(*(ctypes.cast(ctypes.pointer(v),ctypes.c_void_p) for v in args))
        call('cuLaunchKernel',fn,1,1,1,1024,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),ptrs,None)
        torch.cuda.synchronize()
        return out
    input_w=weight('input_layernorm.weight');post_w=weight('post_attention_layernorm.weight')
    linears={n:(weight(n+'.weight'),weight(n+'.bias')) for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.router']}
    rope=ref.GptOssRotaryEmbedding(config,device='cuda')
    sinks=weight('self_attn.sinks')
    if config.layer_types[9] != 'full_attention':raise RuntimeError('layer9 full-attention contract changed')
    def rms(x,w):
        # 2026-10-07: Execute the pinned class method with its actual weight and epsilon.
        holder=type('Norm',(),{'weight':w,'variance_epsilon':config.rms_norm_eps})()
        return ref.GptOssRMSNorm.forward(holder,x)
    try:
        for position in [49,215]:
            x=snapshot(position,'incoming_hidden',(1,2880))
            nr=native_norm(x,input_w);rr=rms(x,input_w)
            compare(f'p{position}-input_norm',rr,nr)
            save(f'p{position}-incoming_hidden',x)
            # 2026-10-07: Use native norm replay output for both projection paths, removing accumulated-input ambiguity.
            projected={n:F.linear(nr,*linears['self_attn.'+n+'_proj']) for n in ['q','k','v']}
            q=projected['q'].reshape(1,1,64,64).transpose(1,2)
            k=projected['k'].reshape(1,1,8,64).transpose(1,2)
            cos,sin=rope(q,torch.tensor([[position]],device='cuda'))
            q,k=ref.apply_rotary_pos_emb(q,k,cos,sin)
            compare(f'p{position}-q_post_rope',q.reshape(64,64),snapshot(position,'q_post_rope',(64,64)))
            compare(f'p{position}-k_post_rope',k.reshape(8,64),snapshot(position,'k_post_rope',(8,64)))
            compare(f'p{position}-v',projected['v'].reshape(8,64),snapshot(position,'v',(8,64)))
            pre=snapshot(position,'attention_pre_o',(1,4096));post=snapshot(position,'attention_post_o',(1,2880))
            compare(f'p{position}-o_projection',F.linear(pre,*linears['self_attn.o_proj']),post)
            native_q=snapshot(position,'q_post_rope',(1,64,1,64))
            cache_k=snapshot(position,'cache_keys',(position+1,8,64)).permute(1,0,2).unsqueeze(0)
            cache_v=snapshot(position,'cache_values',(position+1,8,64)).permute(1,0,2).unsqueeze(0)
            proxy=types.SimpleNamespace(num_key_value_groups=8,sinks=sinks,training=False)
            bf16_attention,_=ref.eager_attention_forward(proxy,native_q,cache_k,cache_v,None,64**-0.5)
            compare(f'p{position}-attention_bf16',bf16_attention.reshape(64,64),pre.reshape(64,64))
            proxy.sinks=sinks.float()
            fp32_attention,_=ref.eager_attention_forward(proxy,native_q.float(),cache_k.float(),cache_v.float(),None,64**-0.5)
            compare(f'p{position}-attention_fp32',fp32_attention.to(torch.bfloat16).reshape(64,64),pre.reshape(64,64))
            proxy.sinks=torch.full_like(sinks.float(),-torch.inf)
            missing_sink,_=ref.eager_attention_forward(proxy,native_q.float(),cache_k.float(),cache_v.float(),None,64**-0.5)
            compare(f'p{position}-attention_missing_sink_control',missing_sink.to(torch.bfloat16).reshape(64,64),pre.reshape(64,64))
            residual=x+post
            captured_norm=snapshot(position,'post_attention_norm',(1,2880))
            compare(f'p{position}-post_attention_norm',rms(residual,post_w),captured_norm)
            compare(f'p{position}-post_norm_native_replay',native_norm(residual,post_w),captured_norm)
            logits=F.linear(captured_norm,*linears['mlp.router'])
            compare(f'p{position}-router_logits',logits,snapshot(position,'router_logits',(1,32)))
            top,ids=torch.topk(logits,4,dim=-1);scores=torch.zeros_like(logits).scatter_(1,ids,F.softmax(top,dim=-1))
            compare(f'p{position}-router_scores',scores,snapshot(position,'router_scores',(1,32)))
    finally:
        call('cuModuleUnload',module)
    receipt['outputs']={f.name:digest(f) for f in sorted(a.output.glob('*.bin'))}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['comparisons'],indent=2))


if __name__=='__main__':main()
