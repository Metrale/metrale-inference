# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Exact grouped/serial parity; synthetic operands, no speed claims.
import ctypes,hashlib,json,pathlib,sys
import torch
root=pathlib.Path(sys.argv[1]);out=root/'parity-run1';out.mkdir(exist_ok=False)
libpath=root/'selected.so';lib=ctypes.CDLL(str(libpath));fn=lib.selected_run
fn.argtypes=[ctypes.c_uint64]*6+[ctypes.c_uint]*3+[ctypes.c_int]*2;fn.restype=ctypes.c_int
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(73451)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_tensor(path,t):
 a=t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();path.write_bytes(a);return hashlib.sha256(a).hexdigest()
rows=[]
for n,k in [(35,96),(5760,2880),(2880,2880)]:
 blocks=torch.randint(0,256,(32,n,k//2),dtype=torch.uint8,device='cuda')
 scales=torch.randint(120,130,(32,n,k//32),dtype=torch.uint8,device='cuda')
 bias=torch.randn((32,n),device='cuda',dtype=torch.bfloat16)
 for stride in [0,k]:
  x=torch.randn((1 if stride==0 else 4,k),device='cuda',dtype=torch.bfloat16)
  ids=torch.tensor([31,0,17,2],dtype=torch.int32,device='cuda')
  for add in [0,1]:
   old=torch.empty((4,n),device='cuda',dtype=torch.bfloat16);new=torch.empty_like(old)
   for group,y in [(0,old),(1,new)]:
    code=fn(blocks.data_ptr(),scales.data_ptr(),x.data_ptr(),ids.data_ptr(),y.data_ptr(),bias.data_ptr(),n,k,stride,group,add);assert code==0,code
   diff=int((old.view(torch.int16)!=new.view(torch.int16)).sum());name=f'n{n}-k{k}-stride{stride}-bias{add}'
   d=out/name;d.mkdir();hashes={}
   for key,t in [('input',x),('ids',ids),('selected_blocks',blocks[ids.long()]),('selected_scales',scales[ids.long()]),('selected_bias',bias[ids.long()]),('serial',old),('grouped',new)]:hashes[key]=save_tensor(d/(key+'.bin'),t)
   rows.append({'case':name,'mismatches':diff,'shape':[n,k],'input_stride':stride,'bias':add,'hashes':hashes});print(name,diff,flush=True)
   assert diff==0
   if stride and not add:
    bad=torch.empty_like(old)
    assert fn(blocks.data_ptr(),scales.data_ptr(),x.data_ptr(),ids.data_ptr(),bad.data_ptr(),bias.data_ptr(),n,k,0,1,0)==0
    changes=int((bad.view(torch.int16)!=old.view(torch.int16)).sum());assert changes>0
    rows[-1]['known_bad_shared_input_mismatches']=changes
    rows[-1]['known_bad_sha256']=save_tensor(d/'known_bad_shared_input.bin',bad)
    del bad
 # 2026-10-07: Invalid IDs must not address checkpoint memory, even bypassing host validation.
 for bad in [[32,0,1,2],[0,0,1,2],[4294967295,2,3,4]]:
  ids=torch.tensor(bad,dtype=torch.int64,device='cuda').to(torch.int32);y=torch.empty((4,n),device='cuda',dtype=torch.bfloat16)
  code=fn(blocks.data_ptr(),scales.data_ptr(),x.data_ptr(),ids.data_ptr(),y.data_ptr(),bias.data_ptr(),n,k,k,1,1)
  assert code==0 and bool(torch.isnan(y).all());rows.append({'shape':[n,k],'invalid_ids':bad,'nan_guard':True})
 del blocks,scales,bias,x,ids,old,new,y
identity={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'library_sha256':sha(libpath),'harness_sha256':sha(pathlib.Path(__file__)),'source_hashes':{str(p.relative_to(root)):sha(p) for p in root.rglob('*.cu')},'cases':rows,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'passed':True,'scope':'Grouped versus existing serial native arithmetic, not Transformers numerical qualification.'}
(out/'comparison.json').write_text(json.dumps(identity,indent=2)+'\n')
