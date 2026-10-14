"""S366 read-only 12+24 prospective IDMT Dataset1 pooled class-ledger.

Both cohorts had original official, both original-trained S80 models SHA
sealed BEFORE first label/XML read. This script aggregates SOURCE reports
only; it fits nothing and adjusts no labels or thresholds. K0 labels
conditional on XML transcription exhaustiveness; song-composition
independence never claimed. Dataset2 counterexample must remain visible.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def p_exact(fixes,reg):
    n=fixes+reg
    if not n:return 1.
    x=min(fixes,reg)
    return min(1.,2*sum(math.comb(n,k) for k in range(x+1))/2**n)
def combined(a,b):
    keys=['events','official_correct','S80_correct','S80_corrections',
        'S80_regressions','changed_both_wrong','diff_correct']
    s={key:int(a[key])+int(b[key]) for key in keys}
    need(s['S80_correct']-s['official_correct']==s['diff_correct']
         and s['S80_corrections']-s['S80_regressions']==s['diff_correct'],
         'pooled paired K correctness ledger invalid')
    s['baseline_rate']=s['official_correct']/s['events'] if s['events'] else None
    s['S80_rate']=s['S80_correct']/s['events'] if s['events'] else None
    s['paired_gain_points']=100*s['diff_correct']/s['events'] if s['events'] else None
    s['event_level_exact_McNemar_two_sided_p']=p_exact(s['S80_corrections'],s['S80_regressions'])
    return s
def signflips(gains):
    counts=Counter({0:1})
    for gain in gains:
        nxt=Counter()
        for delta,c in counts.items():
            nxt[delta+gain]+=c
            nxt[delta-gain]+=c
        counts=nxt
    n=2**len(gains)
    need(sum(counts.values())==n,'exact independent WAV-unit sign permutation incorrect')
    net=sum(gains)
    return dict(wav_count=len(gains),net_gain=net,
       two_sided_exact_wav_signflip_p=sum(v for k,v in counts.items()
           if abs(k)>=abs(net))/n,
       conditional_on_each_WAV_being_an_exchangeable_independent_piece=True,
       composition_disjointness_proven=False)
def run(first,second,out):
    need(not out.exists(),'immutable S366 pooled two-prospective-cohort audit')
    a=json.loads(first.read_text());b=json.loads(second.read_text())
    need(a['status']=='S356_READ_ONLY_PROSPECTIVE_CHORD_GAIN_DISTRIBUTION_AND_UNCERTAINTY'
         and b['status']=='S364_READ_ONLY_PROSPECTIVE_CHORD_GAIN_DISTRIBUTION_AND_UNCERTAINTY'
         and len(a['per_WAV'])==12 and len(b['per_WAV'])==24
         and a['paired_all_native_if_XML_complete']['events']==477
         and b['paired_all_native_if_XML_complete']['events']==967
         and not a['reference_promotion_authorized']
         and not b['reference_promotion_authorized'],
         'source S356/S364 original independent-comparison reports mismatched')
    per_k={f'K{k}':combined(a['by_K'][str(k)],b['by_K'][str(k)]) for k in range(7)}
    all_=combined(a['paired_all_native_if_XML_complete'],b['paired_all_native_if_XML_complete'])
    pos=combined(a['paired_positive_attack_only'],b['paired_positive_attack_only'])
    poly=combined(a['paired_true_poly_attack_only'],b['paired_true_poly_attack_only'])
    need(all_['events']==sum(x['events'] for x in per_k.values())==1444
         and pos['events']==sum(per_k[f'K{k}']['events'] for k in range(1,7))
         and poly['events']==sum(per_k[f'K{k}']['events'] for k in range(2,7)),
         'pooled true class K0..K6 ledgers do not sum to 1444')
    allwav=a['per_WAV']+b['per_WAV']
    need(len({x['WAV_filename_SHA256'] for x in allwav})==36,
         'two cohorts share at least one WAV, invalid prospective disjoint pooling')
    rpt=dict(status='S366_READONLY_POOLED_12_AND_24_PROSPECTIVE_CHORDS_CONDITIONAL_K_LEDGER',
       source_S356_SHA256=sha(first),source_S364_SHA256=sha(second),
       file_count=36,native_V88_event_count=1444,
       all_native_if_XML_complete=all_,
       positive_new_attacks_K1to6_if_XML_complete=pos,
       true_polyphonic_new_attacks_K2to6_if_XML_complete=poly,
       by_true_K_if_XML_complete=per_k,
       original_first_cohort_12_WAV=a['paired_all_native_if_XML_complete'],
       original_second_cohort_24_WAV=b['paired_all_native_if_XML_complete'],
       all_filename_group_signflip=signflips([x['global_conditional']['diff_correct']
                                              for x in allwav]),
       poly_filename_group_signflip=signflips([x['poly']['diff_correct']
                                               for x in allwav]),
       first_cohort_real_poly_count=a['paired_true_poly_attack_only']['events'],
       second_cohort_real_poly_count=b['paired_true_poly_attack_only']['events'],
       source_original_trained_S80_unmodified=True,
       full_XML_K0_negative_exhaustiveness_not_certified=True,
       independent_composition_groups_not_certified=True,
       previous_IDMT_Dataset2_counterexample_original_S80_vs_freeze=dict(
           reference_correct=761,candidate_correct=760,native_events=964),
       promoted_over_freeze_local_combo=False,
       reason='Pooled global gain does not establish reproducibility of poly K2–K6, K5/K6 sample support, annotation-negative exhaustiveness, or cross-dataset improvement.')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(rpt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rpt.items() if k not in ('by_true_K_if_XML_complete','original_first_cohort_12_WAV','original_second_cohort_24_WAV')},indent=2))
    print('PER-K',json.dumps(per_k,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('first','second','out'):
        p.add_argument('--'+key,type=Path,required=True)
    q=p.parse_args();run(q.first,q.second,q.out)
