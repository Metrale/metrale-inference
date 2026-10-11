#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Independent original-FP32 complete VAE reference.
MODEL INPUT_F32 NATIVE_OUT NEW_OUT H W; no globally BF16 pipeline substitution.
"""
import hashlib,inspect,json,pathlib,sys
import numpy as np
import torch
import diffusers.models.autoencoders.autoencoder_kl_qwenimage21 as reference
model,source,native,out=map(pathlib.Path,sys.argv[1:5]);h,w=map(int,sys.argv[5:7]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='1e29f252dbddd044b84c8b794589001ce94bb125a8f13eb5b226ef30c1e2ea11'
torch.cuda.set_per_process_memory_fraction(.85);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
vae=reference.AutoencoderKLQwenImage21.from_pretrained(model/'vae',torch_dtype=torch.float32,local_files_only=True).cuda().eval();results=[]
def record(name,value):
 value=value.detach().cpu().contiguous();(out/(name+'.f32')).write_bytes(value.numpy().tobytes());other=torch.from_numpy(np.fromfile(native/(name+'.f32'),dtype=np.float32).copy()).reshape(value.shape);finite=torch.isfinite(value)&torch.isfinite(other);a,b=value[finite].double(),other[finite].double();d=a-b
 results.append(dict(stage=name,elements=value.numel(),bit_mismatches=int((value.view(torch.int32)!=other.view(torch.int32)).sum()),nonfinite_pairs=int((~finite).sum()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a).clamp_min(1e-30)),max_abs=float(d.abs().max())))
def hook(name):return lambda module,args,value:record(name,value)
handles=[]
for name,module in [('post_quant',vae.post_quant_conv),('conv_in',vae.decoder.conv_in),('mid_res0',vae.decoder.mid_block.resnets[0]),('mid_attn',vae.decoder.mid_block.attentions[0]),('mid_res1',vae.decoder.mid_block.resnets[1]),*[(f'up{i}',m) for i,m in enumerate(vae.decoder.up_blocks)],('decoded',vae.decoder.conv_out)]:handles.append(module.register_forward_hook(hook(name)))
x=torch.from_numpy(np.fromfile(source,dtype=np.float32).copy()).reshape(1,64,1,h,w).cuda()
with torch.no_grad(): decoded=vae.decode(x).sample
record('clamped',decoded)
receipt=dict(reference_source_sha256=refsha,precision='original checkpoint FP32; TF32 disabled',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=results,scope='same-input single-frame decoder reference, no native pipeline qualification');(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
