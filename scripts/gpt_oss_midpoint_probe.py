#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Diagnostic midpoint retry; exact rational gates, no production switch.

Constructed operands are saved for replay. Learned rows are read only from private
exact-dot directories and represented by hashes in receipts; never copied out.
An explicit wide-exponent diagnostic demonstrates FP64 is not universally exact.
"""
import argparse,ctypes,hashlib,json,math,struct
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def value(bits):
    number=struct.unpack('<f',struct.pack('<I',int(bits)<<16))[0]
    if not math.isfinite(number):raise ValueError('oracle supports finite operands only')
    return Fraction(number)
def nearest(exact):
    approx=struct.unpack('<I',struct.pack('<f',float(exact)))[0]>>16
    choices=[n for n in range(max(0,approx-2),min(65535,approx+2)+1) if n&0x7f80!=0x7f80]
    return min(choices,key=lambda n:(abs(value(n)-exact),n&1,n))
def oracle(blocks,scales,x):
    magnitude=[Fraction(0),Fraction(1,2),Fraction(1),Fraction(3,2),Fraction(2),Fraction(3),Fraction(4),Fraction(6)]
    terms=[]
    for i,b in enumerate(x):
        code=(int(blocks[i//2])>>(4*(i%2)))&15
        weight=magnitude[code&7]*Fraction(2)**(int(scales[i//32])-127)
        if code&8:weight=-weight
        # Pinned unpack includes BF16 conversion before the multiply.
        weight=value(nearest(weight))
        terms.append(weight*value(b))
    exact=sum(terms,Fraction())
    return exact,nearest(exact)
def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--kernel-source',type=Path,required=True);p.add_argument('--incumbent-source',type=Path,required=True);p.add_argument('--private-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.cuda.set_per_process_memory_fraction(.85)
    library=ctypes.CDLL(str(a.library));run=library.run_midpoint;run.argtypes=[ctypes.c_void_p]*9+[ctypes.c_uint]*2;run.restype=ctypes.c_int
    report={'model':'openai/gpt-oss-20b','revision':'6cee5e81ee83917806bbde320786a8fb61efebee','checkpoint_url':'https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee','scope':'isolated diagnostic midpoint retry; no production/default change or speed qualification','gate':'exact rational BF16 on bounded constructed and retained learned rows; non-midpoint output identity; wide-exponent case is explicitly diagnostic','torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'source_sha256':sha(__file__),'kernel_sha256':sha(a.kernel_source),'incumbent_sha256':sha(a.incumbent_source),'library_sha256':sha(a.library),'cases':[]}
    assert nearest(Fraction(257,256))==0x3f80 and nearest(Fraction(259,256))==0x3f82
    assert nearest(Fraction(-257,256))==0xbf80 and nearest(Fraction(-259,256))==0xbf82
    def case(name,blocks,scales,x,constructed,diagnostic=False,recorded=None,recorded_expected=None):
        exact,expected=oracle(blocks,scales,x)
        if recorded_expected is not None:assert expected==recorded_expected, "independent rational oracles disagree"
        db=torch.from_numpy(blocks.copy()).cuda();ds=torch.from_numpy(scales.copy()).cuda();dx=torch.from_numpy(x.copy()).view(torch.bfloat16).cuda()
        legacy=torch.empty(1,dtype=torch.bfloat16,device='cuda');repaired=torch.empty_like(legacy);bad=torch.empty_like(legacy)
        accum=torch.empty(1,dtype=torch.float32,device='cuda');precise=torch.empty(1,dtype=torch.float64,device='cuda');flag=torch.empty(1,dtype=torch.int32,device='cuda')
        torch.cuda.synchronize();code=run(*(ctypes.c_void_p(t.data_ptr()) for t in [db,ds,dx,legacy,repaired,bad,accum,precise,flag]),1,len(x));assert code==0,code;torch.cuda.synchronize()
        old,new,wrong=[int(t.view(torch.int16).item())&65535 for t in [legacy,repaired,bad]];trigger=bool(flag.item())
        if recorded is not None:assert old==recorded,'incumbent does not reproduce retained operand result'
        row={'name':name,'kind':'constructed' if constructed else 'learned-private','diagnostic_only':diagnostic,'exact_numerator':str(exact.numerator),'exact_denominator':str(exact.denominator),'expected_bf16':expected,'legacy_bf16':old,'repaired_bf16':new,'double_round_bad_bf16':wrong,'retry':trigger,'legacy_fp32_bits':int(accum.view(torch.int32).item())&0xffffffff,'retry_fp64_bits':int(precise.view(torch.int64).item())&0xffffffffffffffff,'exact_match':new==expected,'non_midpoint_identity':trigger or new==old,'operand_hashes':{key:hashlib.sha256(data).hexdigest() for key,data in [('blocks',blocks.tobytes()),('scales',scales.tobytes()),('input',x.tobytes())]}}
        if constructed:
            folder=a.output/name;folder.mkdir()
            for key,data in [('blocks',blocks.tobytes()),('scales',scales.tobytes()),('input',x.tobytes()),('legacy',raw(legacy)),('repaired',raw(repaired)),('double-round-bad',raw(bad)),('expected',struct.pack('<H',expected)),('accum',raw(accum)),('precise',raw(precise)),('retry',raw(flag))]:(folder/(key+'.bin')).write_bytes(data)
        report['cases'].append(row)
    tiny=Fraction(1,2**30)
    cases=[('positive-even-above',[1,Fraction(1,256),tiny]),('positive-even-below',[1,Fraction(1,256),-tiny]),('negative-even-below',[-1,Fraction(-1,256),-tiny]),('positive-odd-below',[1,Fraction(3,256),-tiny]),('negative-odd-above',[-1,Fraction(-3,256),tiny]),('tie-even-lower',[1,Fraction(1,256),0]),('tie-even-upper',[1,Fraction(3,256),0]),('negative-tie-lower',[-1,Fraction(-1,256),0]),('negative-tie-upper',[-1,Fraction(-3,256),0]),('non-midpoint',[1,Fraction(1,128),0]),('wide-exponent-limit',[1,Fraction(1,256),Fraction(1,2**80)])]
    for name,values in cases:
        x=np.zeros(32,dtype='<u2');blocks=np.zeros(16,dtype=np.uint8);scales=np.array([127],dtype=np.uint8)
        for i,v in enumerate(values):x[i]=nearest(Fraction(v));blocks[i//2]|=2<<(4*(i%2))
        case(name,blocks,scales,x,True,name=='wide-exponent-limit')
    for folder in ['gpt-expert-exact-dot-row231-v3','gpt-layer0-exact-p215-e29-gate','gpt-layer0-exact-p248-e6-gate','gpt-layer0-exact-p248-e5-down','gpt-layer0-exact-p248-e6-down']:
        path=a.private_root/folder;original=json.loads((path/'exact-dot.json').read_text())
        case(folder,np.fromfile(path/'blocks.bin',dtype=np.uint8),np.fromfile(path/'scales.bin',dtype=np.uint8),np.fromfile(path/'input.bf16',dtype='<u2'),False,recorded=original['native_bf16_bits'],recorded_expected=original['correctly_rounded_bf16_bits'])
    report['bounded_gate_passed']=all(r['exact_match'] and r['non_midpoint_identity'] for r in report['cases'] if not r['diagnostic_only'])
    report['known_bad_double_round_detected']=sum(r['repaired_bf16']!=r['double_round_bad_bf16'] for r in report['cases'] if not r['diagnostic_only'])
    report['wide_exponent_limit_observed']=any(r['diagnostic_only'] and not r['exact_match'] for r in report['cases'])
    report['all_rows_exact']=all(r['exact_match'] for r in report['cases'])
    report['experiment_completed_as_predicted']=report['bounded_gate_passed'] and report['known_bad_double_round_detected']>0 and report['wide_exponent_limit_observed']
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return 0 if report['experiment_completed_as_predicted'] else 1
if __name__=='__main__':raise SystemExit(main())
