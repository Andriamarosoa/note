"""S35 attribution audit of actually SELECTED heads, with honest limitations.

Source S35 saves final predictions and each event's selected head sequence.
Without intermediate states, selected-head usage is observable but cannot
establish that individual head's output caused the final correction.
Never mistake observational correlations for independent head ablations.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def investigate(args):
    if args.output.exists():raise ValueError('do not replace an existing audit')
    orig=load(args.input/'predictions.npz')
    flow=load(args.input/'routing.npz')
    source=json.loads((args.input/'report.json').read_text())
    ids=np.asarray(orig['global_index'])
    if not np.array_equal(flow['global_index'],ids):raise ValueError('event identity drift')
    names=list(map(str,orig['variant_ids']))
    rules=list(map(str,flow['variant_ids']))
    heads=list(map(str,flow['action_names']))
    assert len(ids)==59309 and len(rules)==9 and len(heads)==19
    assert not any('H9' in name or 'YourMT3' in name for name in heads)
    baseline=orig['predictions'][:,names.index('series18_parent')]
    truth=orig['true_K']
    output={}
    for name in rules:
        path=flow['selected_head'][:,rules.index(name),:]
        last=np.where(path[:,2]>=0,path[:,2],
                      np.where(path[:,1]>=0,path[:,1],path[:,0]))
        chosen=orig['predictions'][:,names.index(name)]
        changed=chosen!=baseline
        fixed=changed & (chosen==truth)&(baseline!=truth)
        broken=changed & (chosen!=truth)&(baseline==truth)
        neutral=changed & (chosen!=truth)&(baseline!=truth)
        active=(path>=0).sum(1)
        rec={
            'events':int(len(ids)), 'head_calls':int((path>=0).sum()),
            'events_with_recheck':int((active>=2).sum()),
            'events_with_three_heads':int((active==3).sum()),
            'events_never_modified':int((active==0).sum()),
            'corrected':int(fixed.sum()),'regressed':int(broken.sum()),
            'neutral':int(neutral.sum()),
            'by_head':{},
            'by_true_K':{
                str(k):{'total':int((truth==k).sum()),
                  'corrected':int((fixed&(truth==k)).sum()),
                  'regressed':int((broken&(truth==k)).sum()),
                  'neutral':int((neutral&(truth==k)).sum())}
                for k in range(7)}
        }
        for j,head in enumerate(heads):
            assigned=(last==j)
            appears=np.any(path==j,axis=1)
            rec['by_head'][head]={
                'called':int((path==j).sum()),
                'distinct_event_uses':int(appears.sum()),
                'last_selected_for_event':int(assigned.sum()),
                'last_head_on_corrected':int((assigned&fixed).sum()),
                'last_head_on_regressed':int((assigned&broken).sum()),
                'last_head_on_neutral':int((assigned&neutral).sum()),
                'appears_on_fixed_event_path':int((appears&fixed).sum()),
                'appears_on_broken_event_path':int((appears&broken).sum()),
                'event_selected_true_K':{
                    str(k):int((appears&(truth==k)).sum()) for k in range(7)}
            }
        if rec['corrected']!=source['audits'][name]['corrected'] or rec['regressed']!=source['audits'][name]['regressed']:
            raise ValueError('bad paired S18 audit: '+name)
        if any(rec['by_head'][h]['called']!=source['audits'][name]['route']['head_usage'][h] for h in heads):
            raise ValueError('head call count differs from independently verified result')
        assert rec['by_head']['H8_pitch_shift']['called']==0
        output[name]=rec
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps({
        'status':'verified_descriptive','source':'S35',
        'H9_excluded':True,'H8_no_unverified_transform':True,
        'individual_causal_effect_verified':False,
        'no_intermediate_state_stored':True,
        'no_production_promotion':True,
        'policies':output},indent=2,sort_keys=True)+'\n')
    selected=max(rules,key=lambda p:output[p]['corrected']-output[p]['regressed'])
    rows=output[selected]['by_head']
    with (args.output/'head_usage.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['head','calls','distinct_events','last_head_on_corrected',
           'last_head_on_regressed','last_head_on_neutral',
           'appears_on_fixed_event_path','appears_on_broken_event_path'])
        for h,x in rows.items():
            w.writerow([h,x['called'],x['distinct_event_uses'],
                x['last_head_on_corrected'],x['last_head_on_regressed'],
                x['last_head_on_neutral'],
                x['appears_on_fixed_event_path'],
                x['appears_on_broken_event_path']])
    lines=['# S35 head-by-head outcome audit (descriptive)','',
           '**Usage on a corrected event does NOT prove individual head fixed it.**',
           'Intermediate K states were not stored, so this is NOT an isolated head ablation.',
           'H9 forbidden; H8 not called without transformed WAV provenance.',
           f"Selected exploratory variant by *net global* improvement: {selected}.",
           f"Corrected {output[selected]['corrected']}, regressed {output[selected]['regressed']}, neutral {output[selected]['neutral']}.",
           f"Events with two or three selected heads: {output[selected]['events_with_recheck']} / {output[selected]['events_with_three_heads']}.",
           '', '| Head | Calls | Last head on corrected event | Last head on regressed event | Appeared anywhere on corrected event | Appeared anywhere on regressed event |',
           '|---|---:|---:|---:|---:|---:|']
    for h,x in sorted(rows.items(),key=lambda v:-v[1]['called']):
        lines.append(f"| {h} | {x['called']} | "
          f"{x['last_head_on_corrected']} | {x['last_head_on_regressed']} | "
          f"{x['appears_on_fixed_event_path']} | {x['appears_on_broken_event_path']} |")
    lines+=['','## Paired K0–K6 mistakes on selected candidate',
         '| True K | Corrections | Regressions | Neutral changes |',
         '|---|---:|---:|---:|']
    for k,d in output[selected]['by_true_K'].items():
        lines.append(f"| K{k} | {d['corrected']} | {d['regressed']} | {d['neutral']} |")
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    investigate(parser.parse_args())
