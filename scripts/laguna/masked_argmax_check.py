# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-07: Run against masked_argmax_check.cu compiled as probe.so.
import torch, ctypes, json, random, hashlib
from pathlib import Path
import argparse
p = argparse.ArgumentParser(description='Constructed masked greedy CUDA gate; no model weights')
p.add_argument('directory', type=Path)
root = p.parse_args().directory
lib=ctypes.CDLL(str(root/'probe.so')); fn=lib.run
fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint]*3
random.seed(77); torch.manual_seed(77)
rows=[]; masks=[]
for width in [1,2,7,255,256,257,100352]:
 for case in range(12):
  x=torch.randn(width,dtype=torch.bfloat16,device='cuda'); m=[]
  if case==0:x.zero_()
  if case==1:x.fill_(-3e30)
  if case==2:x[-1]=x[0]=100
  if case==3:m=[int(x.argmax())]
  if case==4:x[0]=float('nan')
  if case==5:x[0]=float('inf')
  if case==6:x[0]=-float('inf')
  if case==7:x[0]=float('nan');m=[0]
  if case==8:m=list(range(min(width,8)))
  if case==9:m=[0,0,2**32-1,width+1]
  if case==10:x.zero_();x[0]=-0.
  if case==11:x.fill_(-float('inf'))
  mask=torch.tensor((m+[2**32-1]*8)[:8],device='cuda',dtype=torch.uint32)
  out=torch.empty(1,device='cuda',dtype=torch.uint32)
  fn(x.data_ptr(),mask.data_ptr(),out.data_ptr(),width,1,width)
  y=x.float().cpu(); valid=[i for i in range(width) if i not in m]
  expected=2**32-1 if not valid or not bool(torch.isfinite(y[valid]).all()) else max(valid,key=lambda i:(float(y[i]),i))
  actual=out.item();assert actual==expected,(width,case,actual,expected)
  rows.append({'width':width,'case':case,'actual':actual,'expected':expected})
# 2026-10-07: The known-wrong first-tie and unmasked policies must differ.
assert max(range(7), key=lambda i: (0.0, i)) != 0
assert max([0, 2], key=lambda i: ([4,9,9][i], i)) != 1
# 2026-10-07: Mixed rows and padded stride prove row-local masking and no padding reads.
x=torch.tensor([[4,9,9,1000],[4,9,9,1000],[float('nan'),2,3,1000]],device='cuda',dtype=torch.bfloat16)
m=torch.tensor([[1]+[2**32-1]*7,[1,2]+[2**32-1]*6,[0]+[2**32-1]*7],device='cuda',dtype=torch.uint32)
out=torch.empty(3,device='cuda',dtype=torch.uint32);fn(x.data_ptr(),m.data_ptr(),out.data_ptr(),3,3,4)
assert out.cpu().tolist()==[2,0,2]
json.dump({'cases':len(rows),'rows':rows,'mixed':[2,0,2],'torch':torch.__version__,'device':torch.cuda.get_device_name(),'source_sha256':hashlib.sha256((root/'argmax_feed.cu').read_bytes()).hexdigest()},open(root/'result.json','w'),indent=2)
print('PASS',len(rows),'plus mixed stride rows')
