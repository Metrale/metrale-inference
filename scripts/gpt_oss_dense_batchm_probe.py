#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Exact scalar-order and legacy-entry gates, not model qualification.
import argparse, ctypes, hashlib, json
from pathlib import Path
import torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(8173)
    f=ctypes.CDLL(str(a.library.resolve())).compare_batch;f.argtypes=[ctypes.c_void_p]*6+[ctypes.c_uint]*4;f.restype=ctypes.c_int
    report=dict(scope='constructed scalar-order FP32 and frozen BF16-entry parity; no prefill support claim',source_sha256=sha(a.source),scalar_source_sha256=sha(a.source.with_name('dense_gemv_bf16.cu')),harness_sha256=sha(a.frozen.with_name('dense_batchm_fp32_test.cu')),frozen_sha256=sha(a.frozen),library_sha256=sha(a.library),probe_sha256=sha(__file__),torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),cases=[])
    for m in [1,2,8,16,17,31,32,33,63,64,65,127,128]:
      for n,k in [(32,2880),(512,2880),(4096,2880),(2880,4096),(4,8)]:
        stride=n+4;x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        batch=torch.full((m,stride),123.,device='cuda');scalar=batch.clone();old=torch.full((m,stride),123.,device='cuda',dtype=torch.bfloat16);new=old.clone()
        assert f(*(ctypes.c_void_p(v.data_ptr()) for v in [x,w,batch,scalar,old,new]),m,n,k,stride)==0;torch.cuda.synchronize()
        bias=torch.randn(n,device='cuda',dtype=torch.bfloat16)
        expected=(scalar[:,:n]+bias.float()).bfloat16();bad=(scalar[:,:n].bfloat16()+bias)
        row=dict(m=m,n=n,k=k,fp32_mismatches=int((batch.view(torch.int32)!=scalar.view(torch.int32)).sum()),legacy_mismatches=int((old.view(torch.int16)!=new.view(torch.int16)).sum()),padding_ok=bool((batch[:,n:]==123).all() and (new[:,n:]==123).all()),early_round_control=int((expected.view(torch.int16)!=bad.view(torch.int16)).sum()))
        if n==4 and m==17:
            for name,tensor in [('input',x),('weight',w),('fp32_batch',batch),('fp32_scalar',scalar),('bf16_frozen',old),('bf16_current',new),('bias',bias),('bias_expected',expected),('bias_early_bad',bad)]:
                data=tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes();(a.output/(name+'.bin')).write_bytes(data)
        report['cases'].append(row)
    split=ctypes.CDLL(str(a.library.resolve())).compare_bf16_split;split.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint]*5;split.restype=ctypes.c_int
    report['legacy_split_cases']=[]
    for m,y in [(17,2),(32,2),(47,3)]:
        x=torch.randn(m,2880,device='cuda',dtype=torch.bfloat16);w=torch.randn(32,2880,device='cuda',dtype=torch.bfloat16);old=torch.full((m,36),123.,device='cuda',dtype=torch.bfloat16);new=old.clone()
        assert split(*(ctypes.c_void_p(v.data_ptr()) for v in [x,w,old,new]),m,32,2880,36,y)==0;torch.cuda.synchronize()
        report['legacy_split_cases'].append(dict(m=m,y=y,mismatches=int((old.view(torch.int16)!=new.view(torch.int16)).sum())))
    report['split_controls']=[]
    bad_fn=ctypes.CDLL(str(a.library.resolve())).truncated_batch
    bad_fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint]*4;bad_fn.restype=ctypes.c_int
    for m in [17,31,32,33,63,64,65,127,128]:
        # 2026-10-07: Integer products/sums bounded below2^24 give an independent exact oracle.
        x_int=torch.arange(m*64,dtype=torch.int64).reshape(m,64)%15-7
        w_int=torch.arange(8*64,dtype=torch.int64).reshape(8,64)%11-5
        expected=(x_int@w_int.T).float()/256
        x=(x_int.float()/16).to(device='cuda',dtype=torch.bfloat16)
        w=(w_int.float()/16).to(device='cuda',dtype=torch.bfloat16)
        batch=torch.full((m,12),123.,device='cuda');scalar=batch.clone();bad=batch.clone()
        old=torch.full((m,12),123.,device='cuda',dtype=torch.bfloat16);new=old.clone()
        assert f(*(ctypes.c_void_p(v.data_ptr()) for v in [x,w,batch,scalar,old,new]),m,8,64,12)==0
        assert bad_fn(*(ctypes.c_void_p(v.data_ptr()) for v in [x,w,bad]),m,8,64,12)==0
        torch.cuda.synchronize()
        row=dict(m=m,integer_oracle_mismatches=int((batch[:,:8].cpu()!=expected).sum()),truncation_detected=int((bad[:,:8].cpu()!=expected).sum()),padding_ok=bool((batch[:,8:]==123).all()))
        report['split_controls'].append(row)
        if m==17:
            for name,tensor in [('integer_input',x),('integer_weight',w),('integer_expected',expected),('integer_actual',batch),('truncated_bad',bad)]:
                (a.output/(name+'.bin')).write_bytes(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    report['refusals']=[]
    for m,n,k,stride in [(0,8,64,12),(129,8,64,12),(17,7,64,12),(17,8,63,12),(17,8,64,7)]:
        # 2026-10-07: Null pointers are never dereferenced when geometry refuses before launch.
        code=f(*([ctypes.c_void_p(0)]*6),m,n,k,stride)
        report['refusals'].append(dict(m=m,n=n,k=k,stride=stride,refused=code!=0))
    report['passed']=all(r['refused'] for r in report['refusals']) and all(r['integer_oracle_mismatches']==0 and r['truncation_detected']>0 and r['padding_ok'] for r in report['split_controls']) and all(r['mismatches']==0 for r in report['legacy_split_cases']) and all(r['fp32_mismatches']==r['legacy_mismatches']==0 and r['padding_ok'] for r in report['cases']) and sum(r['early_round_control'] for r in report['cases'])>0
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
