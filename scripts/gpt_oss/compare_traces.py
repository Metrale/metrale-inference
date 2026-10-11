#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Report native/reference differences without changing tolerances or declaring support.
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path, shape):
    bits = np.fromfile(path, dtype='<u2')
    if bits.size != np.prod(shape):
        raise ValueError(f'{path}: expected {shape}, found {bits.size} BF16 values')
    return bits.reshape(shape), (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def metrics(a, b, abits, bbits):
    delta = a.astype(np.float64)-b.astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(b.astype(np.float64)**2)))
    rms = float(np.sqrt(np.mean(delta*delta)))
    return dict(bit_mismatches=int(np.count_nonzero(abits != bbits)),values=int(a.size),
                max_abs=float(np.max(np.abs(delta))),rms=rms, normalized_rms=rms/reference_rms if reference_rms else None,
                reference_rms=reference_rms,
                native_nonfinite=int(np.count_nonzero(~np.isfinite(a))),reference_nonfinite=int(np.count_nonzero(~np.isfinite(b))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    receipt = json.loads((a.reference/'reference.json').read_text())
    native = json.loads((a.native/'receipt.json').read_text())
    if native['tokens'] != receipt['input_ids']:
        raise ValueError('teacher-forcing token mismatch')
    rows = len(receipt['input_ids'])
    nb,n = read(a.native/'receipt.layers.bf16',(rows,24,2880))
    rb,r = read(a.reference/'reference.layers.bf16',(rows,24,2880))
    lb,l = read(a.native/'receipt.logits.bf16',(rows,201088))
    eb,e = read(a.reference/'reference.logits.bf16',(rows,201088))
    data = dict(scope='raw numeric comparison; no native correctness/performance acceptance inferred',input_ids=receipt['input_ids'],
                layers=[[metrics(n[t,i],r[t,i],nb[t,i],rb[t,i]) for i in range(24)] for t in range(rows)],
                logits=[metrics(l[t],e[t],lb[t],eb[t]) for t in range(rows)],
                native_argmax=np.argmax(l,axis=1).tolist(),reference_argmax=np.argmax(e,axis=1).tolist(),
                reference_top1_margin=(np.sort(e,axis=1)[:,-1]-np.sort(e,axis=1)[:,-2]).tolist(),
                sliding_boundary_positions=[i for i in [127,128,129,130] if i < rows],
                source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.native/'receipt.layers.bf16',a.native/'receipt.logits.bf16',a.reference/'reference.layers.bf16',a.reference/'reference.logits.bf16']})
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ['layers','source_hashes']}))
