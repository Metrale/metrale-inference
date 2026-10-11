#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Pinned actual-weight visual block stack and same-input comparisons.
CHECKPOINT FIXTURE_DIR NATIVE_STACK_DIR NEW_OUTPUT. This is not image generation.
"""
import hashlib,inspect,json,pathlib,sys,time
import torch
from safetensors import safe_open
import diffusers.models.transformers.transformer_qwenimage21 as reference
model,fixture,native,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7'
torch.cuda.set_per_process_memory_fraction(.85);torch.set_num_threads(4)
receipt=json.loads((native/'receipt.json').read_text());component=model/'transformer';index=json.loads((component/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
def read(path):
 data=path.read_bytes();assert len(data)==2*3*4096*2
 return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(2,3,4096).cuda()
hidden=read(fixture/'input.bf16');modbytes=(fixture/'modulation.bf16').read_bytes();assert hashlib.sha256(modbytes).hexdigest()==receipt['modulation_sha256']
assert hashlib.sha256((fixture/'input.bf16').read_bytes()).hexdigest()==receipt['input_sha256']
mod=torch.frombuffer(bytearray(modbytes),dtype=torch.bfloat16).reshape(3,16384).cuda();mask=torch.tensor([False,True,True],device='cuda')
cis=reference.QwenImage21Rope(10000,[16,56,56])([[1,1,1]],torch.tensor([False,True,False]),torch.device('cpu')).cuda()
results=[]
def metrics(a,b):
 finite=torch.isfinite(a)&torch.isfinite(b);af,bf=a[finite].double(),b[finite].double();error=af-bf
 return dict(elements=a.numel(),bf16_bit_mismatches=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),nonfinite_pairs=int((~finite).sum()),max_abs_error_finite=float(error.abs().max()) if error.numel() else 0.0,relative_l2_finite=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(bf).clamp_min(1e-30)))
for expected in receipt['results']:
 layer=expected['layer'];state={}
 for name,expected_sha in expected['weight_sha256'].items():
  with safe_open(component/index[name],framework='pt',device='cpu') as f: cpu=f.get_tensor(name)
  assert cpu.dtype==torch.bfloat16 and hashlib.sha256(cpu.view(torch.uint16).numpy().tobytes()).hexdigest()==expected_sha
  state[name.removeprefix(f'transformer_blocks.{layer}.')]=cpu.cuda()
 with torch.device('meta'): block=reference.QwenImage21TransformerBlock(4096,32,128,3,1e-6)
 block.load_state_dict(state,assign=True);block.eval()
 native_input=read(fixture/'input.bf16' if layer==0 else native/f'layer-{layer-1:02}.bf16')
 kwargs=dict(rotary_emb=cis,target_token_mask=mask,segments=[(0,1,True),(1,2,False),(2,3,True)])
 torch.cuda.synchronize();started=time.monotonic()
 with torch.no_grad(): hidden=block(hidden,mod,**kwargs)
 torch.cuda.synchronize();elapsed=time.monotonic()-started
 with torch.no_grad(): same=block(native_input,mod,**kwargs)
 actual=read(native/f'layer-{layer:02}.bf16')
 assert hashlib.sha256((native/f'layer-{layer:02}.bf16').read_bytes()).hexdigest()==expected['output_sha256']
 result=dict(layer=layer,cumulative=metrics(actual,hidden),same_input=metrics(actual,same),reference_forward_seconds=elapsed);results.append(result)
 for label,t in [('reference',hidden),('same-input-reference',same)]: (out/f'{label}-layer-{layer:02}.bf16').write_bytes(t.cpu().view(torch.uint16).numpy().tobytes())
 print(json.dumps(result,allow_nan=False),flush=True)
 del block,state,native_input,actual,same
result=dict(checkpoint_revision=receipt['checkpoint_revision'],native_receipt_sha256=hashlib.sha256((native/'receipt.json').read_bytes()).hexdigest(),reference_source_sha256=refsha,torch=torch.__version__,results=results,qualified=False,scope='32 visual blocks over fixed2x3 BF16 fixture; no embeddings/encoder/VAE/denoising/image or throughput qualification')
(out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert all(not r['cumulative']['bf16_bit_mismatches'] for r in results),'exact native/reference stack gate remains failed'
