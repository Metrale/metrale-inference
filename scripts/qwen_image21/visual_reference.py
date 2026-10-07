#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Independent pinned full visual transformer replay; external encoder fixture.
CHECKPOINT FIXTURE NATIVE_OUTPUT NEW_OUT. Returns only trailing target for comparison.
"""
import hashlib,inspect,json,pathlib,sys
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
model,fixture,native,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
source=pathlib.Path(inspect.getfile(QwenImage21Transformer2DModel));assert hashlib.sha256(source.read_bytes()).hexdigest()=='0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7'
torch.cuda.set_per_process_memory_fraction(.85)
def read(path,shape):return torch.frombuffer(bytearray(path.read_bytes()),dtype=torch.bfloat16).reshape(shape).cuda()
image=read(fixture/'image-input.bf16',(2,8,64));text=read(fixture/'text-input.bf16',(2,3,4096))
engine=QwenImage21Transformer2DModel.from_pretrained(model/'transformer',torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
traces=[]
def hook(index):
 def capture(module,args,output):
  value=output[0] if isinstance(output,tuple) else output
  (out/f'layer-{index:02}.bf16').write_bytes(value.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
 return capture
for i,block in enumerate(engine.transformer_blocks):block.register_forward_hook(hook(i))
with torch.inference_mode():
 output=engine(image,text,torch.tensor([.731,.019],device='cuda'),[[[1,2,2],[1,2,2]]]*2,torch.tensor([[False,True,False,True]]*2,device='cuda'),encoder_hidden_states_mask=torch.tensor([[True,True,False],[False,True,True]],device='cuda'),return_dict=False)[0]
target=output[:,-4:].contiguous();actual=read(native/'target-latents.bf16',(2,4,64));finite=torch.isfinite(actual)&torch.isfinite(target);a,b=actual[finite].double(),target[finite].double();err=a-b
receipt=dict(scope='full pinned visual transformer reference only; external encoder embeddings and no VAE/scheduler',exact_gate_pass=bool(torch.equal(actual.view(torch.int16),target.view(torch.int16))),elements=target.numel(),bit_mismatches=int((actual.view(torch.int16)!=target.view(torch.int16)).sum()),nonfinite_pairs=int((~finite).sum()),relative_l2_finite=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(b).clamp_min(1e-30)),max_abs_error_finite=float(err.abs().max()),torch=torch.__version__,reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256={n:hashlib.sha256((fixture/n).read_bytes()).hexdigest() for n in ['image-input.bf16','text-input.bf16']})
(out/'target-reference.bf16').write_bytes(target.cpu().view(torch.uint16).numpy().tobytes());(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
