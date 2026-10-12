#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# 2026-10-07: Diagnostic C1 latency; not certification or numerical qualification.
import argparse, hashlib, json, pathlib, statistics, time, urllib.request

CASES = [
    {'id': 'short-arithmetic', 'prompt': 'What is 17 times 6? Reply with only the integer.', 'max_tokens': 64, 'expected': '102'},
    {'id': 'modest-decode', 'prompt': 'Write the integers 1 through 16 in ascending order, separated by commas. No other text.', 'max_tokens': 128, 'expected': ','.join(map(str, range(1,17)))},
    {'id': 'long-retrieval', 'prompt': 'The archive identifier is ZX-204. Keep it unchanged.\n' + '\n'.join(f'Record {i}: routine status is green.' for i in range(24)) + '\nReturn only the archive identifier from the beginning.', 'max_tokens': 64, 'expected': 'ZX-204'},
]

def grade(events, expected):
    text = ''.join(e['data']['choices'][0].get('delta', {}).get('content', '') or '' for e in events if isinstance(e['data'], dict) and e['data'].get('choices'))
    finished = [e['data']['choices'][0].get('finish_reason') for e in events if isinstance(e['data'], dict) and e['data'].get('choices') and e['data']['choices'][0].get('finish_reason')]
    normalized = text.strip().replace(' ', '')
    return {'text': text, 'finish_reasons': finished, 'content_correct': normalized == expected, 'clean_terminal': finished == ['stop'] and events[-1]['data'] == '[DONE]', 'no_framing': all(x not in text for x in ['<|', 'analysis', 'assistant'])}

def controls():
    good=[{'data':{'choices':[{'delta':{'content':'102'},'finish_reason':None}]}},{'data':{'choices':[{'delta':{},'finish_reason':'stop'}]}},{'data':'[DONE]'}]
    assert all(grade(good,'102')[x] for x in ['content_correct','clean_terminal','no_framing'])
    assert not grade(good,'103')['content_correct']
    assert not grade(good[:-1],'102')['clean_terminal']
    bad=json.loads(json.dumps(good));bad[1]['data']['choices'][0]['finish_reason']='length';assert not grade(bad,'102')['clean_terminal']
    bad=json.loads(json.dumps(good));bad[0]['data']['choices'][0]['delta']['content']='<|analysis|>102';assert not grade(bad,'102')['no_framing']


def run_case(case, base):
    payload={'model':'gpt-oss-20b','messages':[{'role':'user','content':case['prompt']}],'temperature':0,'reasoning_effort':'low','max_tokens':case['max_tokens'],'stream':True,'stream_options':{'include_usage':True}}
    request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic();events=[]; first_visible=None;usage=None
    with urllib.request.urlopen(request,timeout=180) as response:
        for raw in response:
            now=time.monotonic()-start
            line=raw.decode('utf-8').strip()
            if not line.startswith('data:'): continue
            value=line[5:].strip(); data=value if value=='[DONE]' else json.loads(value)
            events.append({'elapsed_seconds':now,'data':data})
            if isinstance(data,dict):
                if data.get('usage'):usage=data['usage']
                if data.get('choices') and data['choices'][0].get('delta',{}).get('content') and first_visible is None:first_visible=now
    if not events:raise ValueError('empty SSE stream')
    return {'request':payload,'events':events,'grade':grade(events,case['expected']),'first_visible_seconds':first_visible,'total_seconds':time.monotonic()-start,'usage':usage}


def main():
    controls()
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--evidence');parser.add_argument('--base',default='http://127.0.0.1:18842');parser.add_argument('--self-test',action='store_true');a=parser.parse_args()
    if a.self_test:print('5 grading controls passed');return
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
    evidence=pathlib.Path(a.evidence)
    plan={'scope':'C1 diagnostic only; warm sequential requests; no timing acceptance threshold; all failures retained; no energy/certification claim','source_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'server_evidence':json.loads(evidence.read_text()),'repetitions':3,'cases':CASES,'order':['warmup']+[c['id'] for _ in range(3) for c in CASES],'quality_gate':'exact expected final after removing spaces; stop+[DONE]; no Harmony framing','max_seq_len':512,'memory_utilization':0.85,'first_visible_definition':'first nonempty SSE delta.content arrival; excludes role event and hidden analysis','first_generated_definition':'server usage.time_to_first_token_ms; engine boundary, not user-visible'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for index,case in enumerate([CASES[0]]+[c for _ in range(3) for c in CASES]):
        row=run_case(case,a.base);row.update(case_id=case['id'],warmup=index==0,index=index)
        (out/f'{index:02d}-{case["id"]}.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(index,case['id'],row['grade'],row['first_visible_seconds'],flush=True)
    summary={}
    for c in CASES:
        selected=[r for r in rows if not r['warmup'] and r['case_id']==c['id']]
        summary[c['id']]={'all_quality_checks_pass':all(all(r['grade'][key] for key in ['content_correct','clean_terminal','no_framing']) for r in selected),'median_first_visible_seconds':statistics.median(r['first_visible_seconds'] for r in selected),'median_total_seconds':statistics.median(r['total_seconds'] for r in selected),'raw_usage':[r['usage'] for r in selected]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
