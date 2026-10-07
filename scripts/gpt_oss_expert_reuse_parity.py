#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Constructed expert-weight reuse parity and complete-plan controls.
import ctypes, hashlib, json, pathlib, sys
import torch
root=pathlib.Path(sys.argv[1]).resolve();out=root/(sys.argv[2] if len(sys.argv)>2 else 'parity-run1');out.mkdir(exist_ok=False)
libpath=root/'reuse.so';lib=ctypes.CDLL(str(libpath));fn=lib.token_run
fn.argtypes=[ctypes.c_uint64]*6+[ctypes.c_uint]*3+[ctypes.c_int]*3;fn.restype=ctypes.c_int
reduce=lib.token_reduce;reduce.argtypes=[ctypes.c_uint64]*4+[ctypes.c_uint]*2;reduce.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(80917)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,t):
    p.write_bytes(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes());return sha(p)
results=[]
for n,k in [(35,96),(5760,2880),(2880,2880)]:
    b=torch.randint(0,256,(32,n,k//2),dtype=torch.uint8,device='cuda')
    s=torch.randint(121,128,(32,n,k//32),dtype=torch.uint8,device='cuda')
    bias=torch.randn((32,n),dtype=torch.bfloat16,device='cuda')
    for tokens in [1,2,3,4,5,15,16]:
        # 2026-10-07: Expert31 repeats across tokens; all four are unique within each token.
        ids=torch.tensor([[31,(t*3)%31,(t*3+7)%31,(t*3+19)%31] for t in range(tokens)],dtype=torch.int32,device='cuda')
        for per in [0,1]:
            x=torch.randn((4 if per else 1,tokens,k),device='cuda',dtype=torch.bfloat16)
            for add in [0,1]:
                old=torch.empty((4,tokens,n),dtype=torch.bfloat16,device='cuda');new=torch.empty_like(old)
                for mode,y in [(0,old),(1,new)]:
                    assert fn(b.data_ptr(),s.data_ptr(),x.data_ptr(),ids.data_ptr(),y.data_ptr(),bias.data_ptr(),n,k,tokens,per,mode,add)==0
                differences=int((old.view(torch.int16)!=new.view(torch.int16)).sum())
                name=f'n{n}-k{k}-t{tokens}-per{per}-bias{add}';record={'case':name,'differences':differences};results.append(record)
                d=out/name;d.mkdir();record['hashes']={}
                for key,t in [('input',x),('ids',ids),('serial',old),('token_grid',new)]:record['hashes'][key]=save(d/(key+'.bin'),t)
                if n==35:
                    for key,t in [('blocks',b),('scales',s),('bias',bias)]:record['hashes'][key]=save(d/(key+'.bin'),t)
                print(name,differences,flush=True);assert differences==0
                reverse=torch.empty_like(old)
                assert fn(b.data_ptr(),s.data_ptr(),x.data_ptr(),ids.data_ptr(),reverse.data_ptr(),bias.data_ptr(),n,k,tokens,per,3,add)==0
                assert bool((old.view(torch.int16)==reverse.view(torch.int16)).all())
                record['reversed_plan_exact']=True
                for malformed in [4,5]:
                    assert fn(b.data_ptr(),s.data_ptr(),x.data_ptr(),ids.data_ptr(),reverse.data_ptr(),bias.data_ptr(),n,k,tokens,per,malformed,add)==-3
                record['omitted_duplicate_plan_refusals']=True
                if tokens>1 and per and not add:
                    bad=torch.empty_like(old)
                    assert fn(b.data_ptr(),s.data_ptr(),x.data_ptr(),ids.data_ptr(),bad.data_ptr(),bias.data_ptr(),n,k,tokens,per,2,add)==0
                    delta=int((old.view(torch.int16)!=bad.view(torch.int16)).sum());assert delta>0
                    record['wrong_slot_stride_differences']=delta;record['wrong_slot_stride_sha256']=save(d/'wrong-slot-stride.bin',bad)
                # 2026-10-07: Independent staged-product, ascending-expert-ID reduction oracle.
                scores=torch.randn((tokens,32),dtype=torch.bfloat16,device='cuda')
                observed=torch.empty((tokens,n),dtype=torch.bfloat16,device='cuda')
                assert reduce(new.data_ptr(),scores.data_ptr(),ids.data_ptr(),observed.data_ptr(),tokens,n)==0
                order=ids.argsort(dim=1);expected=torch.zeros((tokens,n),dtype=torch.float32,device='cuda')
                for slot_order in range(4):
                    chosen=order[:,slot_order];values=new[chosen,torch.arange(tokens,device='cuda')]
                    weights=scores[torch.arange(tokens,device='cuda'),ids.long().gather(1,chosen[:,None]).squeeze(1)]
                    expected=expected+(values*weights[:,None]).float()
                expected=expected.to(torch.bfloat16);delta=int((observed.view(torch.int16)!=expected.view(torch.int16)).sum());assert delta==0
                record['reduction_differences']=delta
                if tokens>1 and add:
                    wrong_scores=scores[0:1].expand(tokens,32).contiguous();bad_reduce=torch.empty_like(observed)
                    assert reduce(new.data_ptr(),wrong_scores.data_ptr(),ids.data_ptr(),bad_reduce.data_ptr(),tokens,n)==0
                    changed=int((bad_reduce.view(torch.int16)!=observed.view(torch.int16)).sum());assert changed>0
                    record['wrong_score_token_stride_differences']=changed
                    record['wrong_score_token_stride_sha256']=save(d/'wrong-score-token-stride.bin',bad_reduce)
                for key,t in [('scores',scores),('reduced',observed),('reduction_oracle',expected)]:record['hashes'][key]=save(d/(key+'.bin'),t)
        # 2026-10-07: Only the poisoned token becomes NaN; repeated experts across tokens remain valid.
        for bad_ids in [[32,0,1,2],[0,0,1,2],[4294967295,2,3,4]]:
            poisoned=ids.clone();poisoned[-1]=torch.tensor(bad_ids,device='cuda',dtype=torch.int64).to(torch.int32)
            y=torch.empty((4,tokens,n),dtype=torch.bfloat16,device='cuda')
            assert fn(b.data_ptr(),s.data_ptr(),x.data_ptr(),poisoned.data_ptr(),y.data_ptr(),bias.data_ptr(),n,k,tokens,1,1,1)==-2
            results.append({'shape':[n,k,tokens],'invalid_last_token':bad_ids,'host_invalid_ids_refused':True})
# 2026-10-07: Adversarial cancellation makes a wrong slot-order sum observably wrong.
ids=torch.tensor([[31,0,17,2],[2,17,0,31]],dtype=torch.int32,device='cuda')
selected=torch.empty((4,2,3),dtype=torch.bfloat16,device='cuda');scores=torch.ones((2,32),dtype=torch.bfloat16,device='cuda')
values={31:2**-20,0:256,17:-256,2:2**-17}
for t in range(2):
    for slot,e in enumerate(ids[t].cpu().tolist()):selected[slot,t].fill_(values[e])
observed=torch.empty((2,3),dtype=torch.bfloat16,device='cuda')
assert reduce(selected.data_ptr(),scores.data_ptr(),ids.data_ptr(),observed.data_ptr(),2,3)==0
expected=torch.full_like(observed,2**-20);wrong=torch.zeros((2,3),dtype=torch.float32,device='cuda')
for slot in range(4):wrong=wrong+selected[slot].float()
wrong=wrong.to(torch.bfloat16)
assert bool((observed.view(torch.int16)==expected.view(torch.int16)).all())
changed=int((wrong.view(torch.int16)!=observed.view(torch.int16)).sum());assert changed>0
d=out/'reduction-order-control';d.mkdir()
results.append({'case':'reduction-order-control','wrong_slot_order_differences':changed,'hashes':{key:save(d/(key+'.bin'),t) for key,t in [('selected',selected),('scores',scores),('ids',ids),('observed',observed),('expected',expected),('known_bad',wrong)]}})
receipt={'scope':'same-operand token-grid/expert-reuse equality; independent staged weighted reduction; not full-model or speed qualification','cases':results,'passed':True,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'library_sha256':sha(libpath),'script_sha256':sha(pathlib.Path(__file__)),'source_hashes':{str(p.relative_to(root)):sha(p) for p in root.rglob('*.cu')},'peak_bytes':torch.cuda.max_memory_allocated()}
(out/'comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
