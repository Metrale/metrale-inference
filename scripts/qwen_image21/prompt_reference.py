# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-06: Text-only processor reference; never loads model weights.
import argparse,json,hashlib
from pathlib import Path
from transformers import AutoProcessor
parser=argparse.ArgumentParser()
parser.add_argument('--processor',type=Path,required=True)
root=parser.parse_args().processor
p=AutoProcessor.from_pretrained(root,local_files_only=True)
system='Comprehend and analyze the provided prompt.'
message=[{'role':'system','content':[{'type':'text','text':system}]}]
sys_ids=p.apply_chat_template(message,tokenize=True,return_dict=False)[0]
prefix=f'<|im_start|>system\n{system}<|im_end|>\n'
print(json.dumps({'system_ids':sys_ids,'raw_system_ids':p.tokenizer.encode(prefix),'template_sha256':hashlib.sha256((root/'chat_template.jinja').read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest(),'cases':[{'prompt':s,'ids':p(text=prefix+'<|im_start|>user\n'+(s or ' ')+'<|im_end|>\n<|im_start|>assistant\n',padding=True,padding_side='left',return_tensors='pt').input_ids[0].tolist()} for s in ['', 'A red cube on a white table.', 'café 日本 😀', 'first line\nsecond line', '  ']]},indent=2))
