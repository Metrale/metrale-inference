#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""2026-10-07: Pinned dense text encoder comparison at pre-final-norm boundary.
CHECKPOINT IDS_JSON NATIVE_DIRECTORY NEW_OUT. No vision or prompt-template claim.
"""
import hashlib,inspect,json,pathlib,sys
import torch
from transformers import Qwen3VLForConditionalGeneration
import transformers.models.qwen3_vl.modeling_qwen3_vl as reference
model,idsfile,native,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
refsha=hashlib.sha256(pathlib.Path(inspect.getfile(reference)).read_bytes()).hexdigest();assert refsha=='1a02b852d3b113c1664ff5e7ba6f7600e28da2ddf9d479e873de723ff1dc2e60'
torch.cuda.set_per_process_memory_fraction(.85);ids=torch.tensor(json.loads(idsfile.read_text()),device='cuda').reshape(1,-1)
model=Qwen3VLForConditionalGeneration.from_pretrained(model/'text_encoder',torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();text=model.model.language_model
handle=text.norm.register_forward_hook(lambda m,args,output:args[0])
results=[]
def capture(index):
 def hook(module,args,output):
  value=output[0] if isinstance(output,tuple) else output
  actual=torch.frombuffer(bytearray((native/f'layer-{index:02}.bf16').read_bytes()),dtype=torch.bfloat16).reshape_as(value).cuda()
  finite=torch.isfinite(value)&torch.isfinite(actual);a,b=actual[finite].double(),value[finite].double();error=a-b
  results.append(dict(layer=index,elements=value.numel(),bit_mismatches=int((actual.view(torch.int16)!=value.view(torch.int16)).sum()),nonfinite_pairs=int((~finite).sum()),relative_l2_finite=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(b).clamp_min(1e-30)),max_abs_error_finite=float(error.abs().max())))
  (out/f'layer-{index:02}.bf16').write_bytes(value.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
 return hook
for i,layer in enumerate(text.layers):layer.register_forward_hook(capture(i))
with torch.inference_mode():result=text(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
handle.remove();(out/'pre-final-norm.bf16').write_bytes(result.cpu().contiguous().view(torch.uint16).numpy().tobytes())
receipt=dict(reference_source_sha256=refsha,torch=torch.__version__,input_ids_sha256=hashlib.sha256(idsfile.read_bytes()).hexdigest(),results=results,scope='pinned dense text encoder pre-final-norm reference; no vision/tokenizer/pipeline qualification',exact_gate_pass=all(r['bit_mismatches']==0 for r in results));(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
