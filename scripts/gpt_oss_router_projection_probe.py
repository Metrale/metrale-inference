#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Cross native/reference router operands through both exact operator implementations at first decision crossings.
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--modules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    anchor=torch.empty(1,device='cuda')
    modules=json.loads(a.modules.read_text())
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    ref={(x['position'],x['layer']):x for x in json.loads(a.reference.read_text()) if x['stage']=='router'}
    report=dict(scope='identical router operands, two implementations; no tie-policy or runtime changes',source_sha256=sha(__file__),reference_sha256=sha(a.reference),torch=torch.__version__,device=torch.cuda.get_device_name(),cases=[],modules={})
    driver=ctypes.CDLL('libcuda.so.1');handles=[]
    def call(name,*args):
        code=getattr(driver,name)(*args)
        if code:raise RuntimeError(f'{name}: CUDA{code}')
    def kernel(module_name,symbol):
        m=next(m for m in modules['modules'] if m['name']==module_name);path=a.modules.parent/m['ptx']
        if sha(path)!=m['ptx_sha256']:raise RuntimeError('PTX digest mismatch')
        report['modules'][module_name]=m['ptx_sha256']
        handle=ctypes.c_void_p();fn=ctypes.c_void_p()
        call('cuModuleLoad',ctypes.byref(handle),ctypes.c_char_p(str(path).encode()));handles.append(handle)
        call('cuModuleGetFunction',ctypes.byref(fn),handle,ctypes.c_char_p(symbol.encode()));return fn
    gemv=kernel('gemv','dense_gemv_bf16_fp32out');bias=kernel('projection_bias','projection_bias_bf16')
    def launch(fn,args,grid):
        ptrs=(ctypes.c_void_p*len(args))(*(ctypes.cast(ctypes.pointer(v),ctypes.c_void_p) for v in args))
        call('cuLaunchKernel',fn,grid,1,1,256,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),ptrs,None)
    def native_linear(x,w,b):
        accum=torch.empty((1,32),device='cuda',dtype=torch.float32);out=torch.empty((1,32),device='cuda',dtype=torch.bfloat16)
        launch(gemv,[ctypes.c_void_p(t.data_ptr()) for t in [x,w,accum]]+[ctypes.c_uint(32),ctypes.c_uint(2880)],8)
        launch(bias,[ctypes.c_void_p(t.data_ptr()) for t in [accum,b,out]]+[ctypes.c_uint(32),ctypes.c_uint(32)],1)
        torch.cuda.synchronize();return out
    def save(name,t):
        (a.output/(name+'.bin')).write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    def tensor(bits):return torch.from_numpy(np.asarray(bits,dtype=np.int16).copy()).view(torch.bfloat16).reshape(1,-1).cuda()
    def compare(x,y):
        return dict(bit_mismatches=int((x.view(torch.int16)!=y.view(torch.int16)).sum()),max_abs=float((x.float()-y.float()).abs().max()))
    def decision(x):
        values,ids=torch.topk(x,5,dim=-1)
        return dict(ids=ids[0,:4].cpu().tolist(),cutoff_gap=float(values[0,3].float()-values[0,4].float()),logits_bf16_bits=x.view(torch.int16).cpu().reshape(-1).tolist())
    try:
        for position,layer in [(215,6),(49,9),(248,23)]:
            label=f'p{position}-l{layer}';r=ref[position,layer]
            prefix=f'model.layers.{layer}.mlp.router.';weights=[]
            for suffix in ['weight','bias']:
                name=prefix+suffix
                with safe_open(a.checkpoint/index[name],framework='pt',device='cpu') as f:t=f.get_tensor(name).cuda()
                save(label+'-'+suffix,t);weights.append(t)
            w,b=weights
            path=a.native/(label+'-post_attention_norm.bin')
            nx=tensor(np.fromfile(path,dtype='<i2'));rx=tensor(r['input_bf16_bits'])
            actual=tensor(np.fromfile(a.native/(label+'-router_logits.bin'),dtype='<i2'))
            expected=tensor(r['router_logits_bf16_bits'])
            item=dict(position=position,layer=layer,input_difference=compare(nx,rx),native_input_sha256=sha(path),reference_ids=r['indices'],branches={})
            for tag,x,target in [('native_input',nx,actual),('reference_input',rx,expected)]:
                nl=native_linear(x,w,b);tl=F.linear(x,w,b)
                for name,t in [('input',x),('native_linear',nl),('torch_linear',tl),('captured_logits',target)]:save(label+'-'+tag+'-'+name,t)
                item['branches'][tag]=dict(native_vs_torch=compare(nl,tl),native_vs_captured=compare(nl,target),torch_vs_captured=compare(tl,target),native_decision=decision(nl),torch_decision=decision(tl))
            report['cases'].append(item)
    finally:
        for h in handles:call('cuModuleUnload',h)
    report['outputs']={f.name:sha(f) for f in sorted(a.output.glob('*.bin'))}
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k!='branches'} | {'branches':{tag:{k:v for k,v in br.items() if not k.endswith('decision')} for tag,br in c['branches'].items()}} for c in report['cases']],indent=2))


if __name__=='__main__':main()
