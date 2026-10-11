#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Unchanged native attention/gather, production Rust visibility plan.
REPO CHECKPOINT PLAN_DIR ROTARY_RESULTS PRELUDE_RESULTS NEW_OUTPUT. Diagnostic,
exact gate is reported, never widened to accommodate existing attention math.
"""
import ctypes,hashlib,inspect,json,pathlib,subprocess,sys
import torch
from safetensors import safe_open
import diffusers.models.transformers.transformer_qwenimage21 as reference
repo,model,plans,rotary,prelude,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7'
sources=[]
for name in ['nllb_encoder.cu','embed_from_argmax.cu','dense_gemm_bf16.cu']:
 p=out/name;p.write_bytes((repo/'kernels/gb10/common'/name).read_bytes());sources.append(p)
wrapper=out/'wrapper.cu';wrapper.write_text(''.join('#include "'+str(p.resolve())+'"\n' for p in sources)+r'''
extern "C" int gather(void* ids,void* x,void* y,unsigned rows) {
 batched_embed<<<rows,256>>>((unsigned*)ids,(__nv_bfloat16*)x,(__nv_bfloat16*)y,4096);return cudaDeviceSynchronize();
}
extern "C" int attend(void*q,void*k,void*v,void*out,unsigned keys) {
 nllb_attn_kv_bf16<<<32,128,(keys+128)*4>>>((__nv_bfloat16*)q,(__nv_bfloat16*)k,(__nv_bfloat16*)v,(__nv_bfloat16*)out,1,keys,32,128,1.0f/sqrtf(128.0f),0);return cudaDeviceSynchronize();
}
extern "C" int project(void*x,void*w,void*y,unsigned rows) {
 dense_gemm_bf16_pipelined<<<dim3(32,(rows+127)/128),256>>>((__nv_bfloat16*)x,(__nv_bfloat16*)w,(__nv_bfloat16*)y,rows,4096,4096);return cudaDeviceSynchronize();
}
''')
command=['nvcc','-shared','-Xcompiler','-fPIC','-O3','--fmad=false','-arch=sm_121',str(wrapper),'-o',str(out/'attention.so')]
build=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(build.stdout+build.stderr);build.check_returncode()
lib=ctypes.CDLL(str(out/'attention.so'))
for name,pointers in [('gather',3),('attend',4),('project',3)]:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p]*pointers+[ctypes.c_uint];fn.restype=ctypes.c_int
component=model/'transformer';index=json.loads((component/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map'];wn='transformer_blocks.0.attn.to_out.0.weight'
with safe_open(component/index[wn],framework='pt',device='cpu') as f: wc=f.get_tensor(wn)
assert wc.shape==(4096,4096) and wc.dtype==torch.bfloat16
torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(2129);weight=wc.cuda()
results=[];controls={'causal_only':0,'dense':0,'drop_query_padding':0,'transposed_output_weight':0};leaks=[]
def save(name,x): (out/name).write_bytes(x.cpu().contiguous().view(torch.uint16).numpy().tobytes())
def metrics(a,b): return dict(elements=a.numel(),bf16_bit_mismatches=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),max_abs_error=float((a.float()-b.float()).abs().max()),relative_l2=float(torch.linalg.vector_norm(a.float()-b.float())/torch.linalg.vector_norm(b.float()).clamp_min(1e-30)))
for name in ['saved','mixed']:
 pf=plans/f'{name}-plan.json';data=json.loads(pf.read_text());(out/pf.name).write_bytes(pf.read_bytes())
 ids=torch.tensor(data['image_ids'],device='cuda');valid=torch.tensor(data['key_valid'],device='cuda');batch,seq=valid.shape
 pos=torch.arange(seq,device='cuda');same=(ids[:,None]==ids[None,:])&(ids[:,None]>=0)
 mask=((pos[:,None]>=pos[None,:])|same)[None,None]&valid[:,None,None,:]
 # Independent oracle verifies the actual Rust gather spans before CUDA execution.
 for query,(start,count) in enumerate(data['spans']):
  sample,local=divmod(query,seq)
  expected=torch.where(mask[sample,0,local])[0].cpu().tolist()
  assert data['gather'][start:start+count]==[sample*seq+k for k in expected]
 gather=torch.tensor(data['gather'],dtype=torch.uint32,device='cuda')
 def native(q,k,v):
  ck,cv=[torch.empty((len(data['gather']),32,128),device='cuda',dtype=torch.bfloat16) for _ in range(2)];actual=torch.zeros_like(q)
  if len(data['gather']):
   for inp,compact in [(k,ck),(v,cv)]: assert lib.gather(gather.data_ptr(),inp.data_ptr(),compact.data_ptr(),len(data['gather']))==0
  for query,(start,count) in enumerate(data['spans']):
   if count: assert lib.attend(q.data_ptr()+query*8192,ck.data_ptr()+start*8192,cv.data_ptr()+start*8192,actual.data_ptr()+query*8192,count)==0
  return actual
 def sdpa(q,k,v,m): return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=m).transpose(1,2).contiguous()
 if name=='saved':
  def read(path): return torch.frombuffer(bytearray(path.read_bytes()),dtype=torch.bfloat16).reshape(batch,seq,32,128).cuda()
  cases={'real_projection':(read(rotary/'saved-q-native.bf16'),read(rotary/'saved-k-native.bf16'),read(prelude/'v-native.bf16'))}
 else:
  q,k,v=[torch.randn(batch,seq,32,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
  cases={'random':(q,k,v),'high_logits':(q*8,k*8,v),'uniform':(torch.zeros_like(q),k,v)}
 for case,(q,k,v) in cases.items():
  expected=sdpa(q,k,v,mask);actual=native(q,k,v)
  expected_projection=torch.nn.functional.linear(expected.flatten(2),weight);projection=torch.empty_like(expected_projection)
  assert lib.project(actual.data_ptr(),weight.data_ptr(),projection.data_ptr(),batch*seq)==0
  local_projection=torch.nn.functional.linear(actual.flatten(2),weight)
  results.append(dict(layout=name,case=case,attention=metrics(actual,expected),projection_same_input=metrics(projection,local_projection),composed_projection=metrics(projection,expected_projection),fully_masked_nonzero=int(torch.count_nonzero(actual[~mask.any(-1).squeeze(1)]))))
  for label,t in [('q',q),('k',k),('v',v),('native',actual),('reference',expected),('projection',projection),('reference-projection',expected_projection)]: save(f'{name}-{case}-{label}.bf16',t)
  for label,m in [('causal_only',(pos[:,None]>=pos[None,:])[None,None]&valid[:,None,None,:]),('dense',valid[:,None,None,:].expand(batch,1,seq,seq))]: controls[label]+=metrics(sdpa(q,k,v,m),expected)['bf16_bit_mismatches']
  controls['drop_query_padding']+=metrics(expected*valid[:,:,None,None],expected)['bf16_bit_mismatches']
  controls['transposed_output_weight']+=metrics(torch.nn.functional.linear(actual.flatten(2),weight.T.contiguous()),local_projection)['bf16_bit_mismatches']
  vk=v.clone();vk[~valid]=1234;masked=native(q,k,vk)
  vk=v.clone();vk[1]*=7;other=native(q,k,vk)
  leaks.append(dict(layout=name,case=case,padded_key_changes=metrics(masked,actual)['bf16_bit_mismatches'],sample0_changes=metrics(other[0],actual[0])['bf16_bit_mismatches']))
receipt=dict(seed=2129,checkpoint_revision='d26bb61231c349cf6b7896fa83353113880e1ba3',reference_source_sha256=refsha,torch=torch.__version__,cuda=torch.version.cuda,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},weight=wn,weight_sha256=hashlib.sha256(wc.view(torch.uint16).numpy().tobytes()).hexdigest(),compile_command=command,criterion='exact BF16 diagnostic gate',results=results,known_bad_controls=controls,isolation=leaks,qualified=False,scope='unchanged scalar attention/gather and real output projection; no speed or full-block qualification')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
assert all(controls.values()) and all(not r['padded_key_changes'] and not r['sample0_changes'] for r in leaks)
assert all(not r['fully_masked_nonzero'] for r in results)
assert all(r['attention']['bf16_bit_mismatches']==0 and r['projection_same_input']['bf16_bit_mismatches']==0 for r in results),'existing attention/GEMM precision gate differs'
