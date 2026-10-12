"""S194 independently compare prelabel-frozen adapted source and H7 on disjoint XML cohorts.

Only artifact reports from S190 and S193. S190 exposed labels motivated H7
comparison, S193 disjoint tracks evaluate it. Does not learn a gate.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
def verify(x,m):
    if not x:raise ValueError(m)
def exact(d):
    out=d['S179_new_parent_adapted_selection']
    h7=d['original_7_expert_individual_comparators']['H7_attack_morphology']
    return {'matched':out['total'],
      'poly_truth':out['poly_true_rows'],
      'adapted_total_correct':out['correct'],
      'adapted_poly_correct':sum(out['per_K_correct'][2:]),
      'H7_total_correct':h7['correct'],
      'H7_poly_correct':sum(h7['per_K_correct'][2:]),
      'H7_global_ExactK':h7['global_ExactK'],
      'H7_poly_ExactK':h7['poly_ExactK']}
def run(a,b,out):
    verify(not out.exists(),'immutable disjoint diagnostic')
    first=json.loads(a.read_text());second=json.loads(b.read_text())
    verify(first['status'].startswith('S190_FROZEN_') and
           second['status'].startswith('S193_FROZEN_'),
           'must read original benchmark and third new fixed cohort independently')
    fa=set(p['wav_name_SHA256'] for p in first['source_XML_checksum_proofs'])
    fb=set(p['wav_name_SHA256'] for p in second['source_XML_checksum_proofs'])
    verify(len(fa)==len(fb)==16 and not (fa & fb),
           'source WAV overlap or event leakage')
    verify(first['S189_event_index_sha256']!=second['S192_event_index_sha256'],
           'identical sound event index')
    x=exact(first);y=exact(second)
    verify(x['matched']==62 and y['matched']==65 and x['poly_truth']==11
            and y['poly_truth']==14,
            'held-out fixed comparison event count drift')
    total=x['matched']+y['matched'];poly=x['poly_truth']+y['poly_truth']
    result={'status':'S194_DISJOINT_PRE_FROZEN_SOURCE_H7_VALIDATION_NOT_S80',
      'exposed_development_XML_S190_SHA256':hashlib.sha256(a.read_bytes()).hexdigest(),
      'untouched_at_prior_H7_selection_S193_SHA256':hashlib.sha256(b.read_bytes()).hexdigest(),
      'identical_wav_hashes_across_benchmarks':0,
      'prior_development_S190':x,
      'fresh_disjoint_evaluation_S193':y,
      'fresh_global_ExactK_adapted':y['adapted_total_correct']/y['matched'],
      'fresh_global_ExactK_H7':y['H7_total_correct']/y['matched'],
      'fresh_poly_ExactK_adapted':y['adapted_poly_correct']/y['poly_truth'],
      'fresh_poly_ExactK_H7':y['H7_poly_correct']/y['poly_truth'],
      'fresh_H7_minus_adapted_global_correct':y['H7_total_correct']-y['adapted_total_correct'],
      'fresh_H7_minus_adapted_poly_correct':y['H7_poly_correct']-y['adapted_poly_correct'],
      'two_cohort_total_matched':total,
      'two_cohort_poly_truth_count':poly,
      'pooled_H7_global_ExactK':(x['H7_total_correct']+y['H7_total_correct'])/total,
      'pooled_H7_poly_ExactK':(x['H7_poly_correct']+y['H7_poly_correct'])/poly,
      'pooled_adapted_global_ExactK':(x['adapted_total_correct']+y['adapted_total_correct'])/total,
      'pooled_adapted_poly_ExactK':(x['adapted_poly_correct']+y['adapted_poly_correct'])/poly,
      'any_model_retrained_on_XML':False,
      'comparison_is_official_S80_reference':False,
      'original_16_S45_193_S46_restored':False,'S80_promoted':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
def selftest():
    r={'S179_new_parent_adapted_selection':{'total':3,'correct':2,'poly_true_rows':1,'per_K_correct':[0,2,0,0,0,0,0]},
       'original_7_expert_individual_comparators':{'H7_attack_morphology':{'correct':3,'per_K_correct':[0,2,1,0,0,0,0],'global_ExactK':1.,'poly_ExactK':1.}}}
    x=exact(r);verify(x['H7_poly_correct']==1 and x['adapted_poly_correct']==0,'paired metric proof')
    print('S194 PASS paired original vs new disjoint evaluation; no refit')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dev',type=Path);p.add_argument('--fresh',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        verify(args.dev and args.fresh and args.out,'both genuinely disjoint labeled source reports required')
        run(args.dev,args.fresh,args.out)
