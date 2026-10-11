#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Integer-exact BF16 dot oracle for visual output projection.
WEIGHT_BF16 IO_DIAGNOSTIC_DIRECTORY NEW_RECEIPT. No CUDA or Torch dependency.
Checks every row with a native/reference difference, preserving the original gate.
"""
import hashlib,json,pathlib,struct,sys
weight,directory,receipt=map(pathlib.Path,sys.argv[1:])
def words(p):return struct.unpack('<'+'H'*(p.stat().st_size//2),p.read_bytes())
def scalar(bits):
 exponent=(bits>>7)&255;assert exponent!=255
 mantissa=bits&127
 return (-1 if bits&32768 else 1)*(mantissa+(128 if exponent else 0)), (exponent-134 if exponent else -133)
def rounded(accumulator):
 if not accumulator:return 0
 sign=32768 if accumulator<0 else 0;n=abs(accumulator);power=n.bit_length()-1-266
 # Round exact sum to nearest-even BF16, including subnormals and carry.
 shift=max(power-7,-133)+266;q,r=divmod(n,1<<shift);half=1<<(shift-1)
 q+=r>half or (r==half and q&1)
 if q==256:q=128;power+=1
 if power < -126:return sign+q
 if power>127:return sign+32640
 return sign+((power+127)<<7)+(q-128)
# Independent exact values, ties, subnormals and cancellation controls.
assert rounded(1<<266)==0x3f80
assert rounded((1<<266)+(1<<258))==0x3f80
assert rounded((1<<266)+3*(1<<258))==0x3f82
assert rounded(-(1<<266))==0xbf80
assert rounded(1<<132)==0
assert rounded(3*(1<<132))==2
assert rounded((1<<300)-(1<<300))==0
w=words(weight);x=words(directory/'output_scale-native.bf16');native=words(directory/'image_out-native.bf16');reference=words(directory/'image_out-reference.bf16')
assert len(w)==64*4096 and len(x)%4096==0 and len(native)==len(reference)==len(x)//4096*64
mismatches=[]
for offset,(a,b) in enumerate(zip(native,reference)):
 if a==b:continue
 row,col=divmod(offset,64);total=0
 for left,right in zip(x[row*4096:(row+1)*4096],w[col*4096:(col+1)*4096]):
  lm,le=scalar(left);rm,re=scalar(right);total+=(lm*rm)<<(le+re+266)
 oracle=rounded(total);mismatches.append(dict(row=row,column=col,native_bits=a,reference_bits=b,exact_dot_rounded_bits=oracle,native_exact=a==oracle,reference_exact=b==oracle))
result=dict(scope='integer-exact same-input dot on native/reference-disagreeing outputs only; original exact-reference gate unchanged',weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((directory/'output_scale-native.bf16').read_bytes()).hexdigest(),disagreements=mismatches)
with receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
