# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-06: Independent pinned Diffusers schedule and Torch BF16 step receipts.
import argparse, hashlib, inspect, json
from pathlib import Path
import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--device',choices=['cpu','cuda'],required=True)
args=parser.parse_args()
root=args.checkpoint
source_sha=hashlib.sha256(Path(inspect.getfile(FlowMatchEulerDiscreteScheduler)).read_bytes()).hexdigest()
if source_sha!='5448bbfe15324ea8034470e742c7782b1de3864c0836d98ef3dade65e30df6a7':
 raise SystemExit('scheduler source does not match pinned reference')
config=json.loads((root/'scheduler/scheduler_config.json').read_text())
result={'model':'https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3','device':args.device,'source_sha256':hashlib.sha256(Path(inspect.getfile(FlowMatchEulerDiscreteScheduler)).read_bytes()).hexdigest(),'torch':torch.__version__,'numpy':np.__version__,'config':config,'schedules':[],'steps':[]}
for count in (2,4,40,50):
 for tokens in (256,1024,8192):
  m=(config['max_shift']-config['base_shift'])/(config['max_image_seq_len']-config['base_image_seq_len'])
  mu=tokens*m+(config['base_shift']-m*config['base_image_seq_len'])
  scheduler=FlowMatchEulerDiscreteScheduler.from_config(config)
  scheduler.set_timesteps(sigmas=np.linspace(1.,1/count,count),mu=mu,device=args.device)
  result['schedules'].append({'count':count,'image_tokens':tokens,'sigma_bits':scheduler.sigmas.view(torch.int32).tolist(),'time_bits':scheduler.timesteps.view(torch.int32).tolist(),'model_time_bits':(scheduler.timesteps.bfloat16()/1000).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()})
bits=torch.arange(65536,dtype=torch.int32,device=args.device).to(torch.uint16)
x=bits.view(torch.bfloat16);valid=torch.isfinite(x);x=x[valid]
sample=torch.tensor([0.,1.,-1.,.5],dtype=torch.bfloat16,device=args.device).repeat((len(x)+3)//4)[:len(x)]
for dt in (-0.020000040531158447,-0.028576254844665527,-0.1234567,-1.):
 scalar=torch.tensor(dt,dtype=torch.float32,device=args.device)
 product=scalar*x
 out=(sample.float()+product).bfloat16()
 fused=(sample.float()+scalar*x.float()).bfloat16()
 result['steps'].append({'dt':dt,'elements':len(x),'sha256':hashlib.sha256(out.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),'wrong_fp32_product_differences':int((out.view(torch.int16)!=fused.view(torch.int16)).sum()),'product_dtype':str(product.dtype),'scalar_cast_oracle_differences':int((product.view(torch.int16)!=(x.float()*scalar.bfloat16().float()).bfloat16().view(torch.int16)).sum())})
print(json.dumps(result,indent=2))
