# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Load exact v4.55 reference function bodies; never replace environment packages.
import argparse, ast, ctypes, hashlib, json, math, types
from pathlib import Path
from typing import Optional
import torch

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--reference-dir', required=True)
p.add_argument('--config', required=True)
p.add_argument('--parameters', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
ns = dict(torch=torch, math=math, Optional=Optional, PretrainedConfig=object)
ref = Path(a.reference_dir)
for name, digest in {
    'modeling_rope_utils.py': '391180db0d5e779692db59c1d920d1bd864551332b7a6fb6dd5ad8984a3a9a36',
    'modeling_gpt_oss.py': '3bdb6f15154f5555c264720e021b4ef1f868dde9fbe476bb3510650030ebfbf5',
}.items():
    assert hashlib.sha256((ref/name).read_bytes()).hexdigest() == digest, name

for filename, names in [('modeling_rope_utils.py', ['_compute_yarn_parameters']),
                        ('modeling_gpt_oss.py', ['_apply_rotary_emb'])]:
    tree = ast.parse((ref / filename).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), ns)
# 2026-10-07: Remove only decorators from the pinned forward body to avoid importing unrelated models.
tree = ast.parse((ref / 'modeling_gpt_oss.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GptOssRotaryEmbedding')
forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
forward.decorator_list = []
exec(compile(ast.Module(body=[forward], type_ignores=[]), 'modeling_gpt_oss.py', 'exec'), ns)
config = types.SimpleNamespace(**json.loads(Path(a.config).read_text()))
params = json.loads(Path(a.parameters).read_text())
lib = ctypes.CDLL(a.library)
lib.table.argtypes = [ctypes.c_void_p] + [ctypes.c_float]*4
lib.rotate.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint]*3 + [ctypes.c_float]
freq = torch.empty(32, device='cuda', dtype=torch.float32)
assert lib.table(freq.data_ptr(), params['base'], params['factor'], params['low'], params['span']) == 0
expected_freq, scale = ns['_compute_yarn_parameters'](config, torch.device('cuda'))
positions = torch.tensor([0,1,127,128,4095,4096,8192,131071], device='cuda', dtype=torch.int32)
# 2026-10-07: Deterministic BF16 operands include signed tiny values and cancellation-sensitive mantissas.
torch.manual_seed(419)
q = (torch.randn(8,64,64,device='cuda') * 3).to(torch.bfloat16)
k = (torch.randn(8,8,64,device='cuda') * .25).to(torch.bfloat16)
q[0,0,:] = torch.linspace(-.01,.01,64,device='cuda').to(torch.bfloat16)
qi,ki = q.clone(),k.clone()
cos,sin = ns['forward'](types.SimpleNamespace(inv_freq=expected_freq,attention_scaling=scale), q, positions.to(torch.int64)[None,:])
cos,sin = cos[0,:,None,:],sin[0,:,None,:]
eq,ek = ns['_apply_rotary_emb'](q,cos,sin),ns['_apply_rotary_emb'](k,cos,sin)
assert lib.rotate(q.data_ptr(),k.data_ptr(),positions.data_ptr(),freq.data_ptr(),8,64,8,params['attention_factor']) == 0
badconfig = types.SimpleNamespace(**vars(config))
badconfig.rope_scaling = dict(config.rope_scaling, truncate=True)
badfreq,badscale = ns['_compute_yarn_parameters'](badconfig,torch.device('cuda'))
bc,bs = ns['forward'](types.SimpleNamespace(inv_freq=badfreq,attention_scaling=badscale),qi,positions.to(torch.int64)[None,:])
truncated = ns['_apply_rotary_emb'](qi,bc[0,:,None,:],bs[0,:,None,:])
interleaved = torch.stack((qi[...,::2]*cos-qi[...,1::2]*sin,qi[...,1::2]*cos+qi[...,::2]*sin),dim=-1).flatten(-2)
def bits(x):
    return x.contiguous().view(torch.int32 if x.dtype==torch.float32 else torch.int16).cpu().reshape(-1).tolist()
def mismatch(x,y):
    return int((x!=y).sum().item())
result = dict(torch_version=torch.__version__,device=torch.cuda.get_device_name(),parameters=params,
              positions=positions.cpu().tolist(),q_shape=list(q.shape),k_shape=list(k.shape),
              frequency_bits=bits(freq),reference_frequency_bits=bits(expected_freq),
              q_input_bits=bits(qi),k_input_bits=bits(ki),q_output_bits=bits(q),k_output_bits=bits(k),
              q_reference_bits=bits(eq),k_reference_bits=bits(ek),truncated_bits=bits(truncated),interleaved_bits=bits(interleaved),
              frequency_mismatches=mismatch(freq,expected_freq),q_mismatches=mismatch(q,eq),k_mismatches=mismatch(k,ek),
              truncated_detections=mismatch(truncated,eq),interleaved_detections=mismatch(interleaved,eq),
              source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ref/'modeling_rope_utils.py',ref/'modeling_gpt_oss.py',Path(a.library),Path(__file__)]})
Path(a.output).write_text(json.dumps(result))
print(json.dumps({k:v for k,v in result.items() if not k.endswith('_bits')}))
assert result['frequency_mismatches']==result['q_mismatches']==result['k_mismatches']==0
assert result['truncated_detections']>0 and result['interleaved_detections']>0
