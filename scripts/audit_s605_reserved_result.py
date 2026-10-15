"""Post-test verification and loss attribution; no network, fitting or tuning."""
import argparse
import collections
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.s592_s597_urmp_prospective import read
from scripts.s591_score_state_note_predictions import match_ledger
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times
from scripts.s601_bp_candidate_gate import counts


def run(root,out):
    need(not out.exists(),'Immutable offline verification')
    frozen=root/'S604';scored=root/'S605';f=read(frozen/'report.json');r=read(scored/'report.json')
    pred=read(frozen/'predictions.json');truth=read(scored/'truth_after_freeze.json')
    need(sha(frozen/'report.json')==r['freeze_report_sha256']
        and sha(frozen/'predictions.json')==r['prediction_sha256']==f['prediction_sha256']
        and all(sha(scored/name)==h for name,h in r['audit_hashes'].items()),'All frozen artifacts must match')
    replay=metrics(pred,truth,{s:dict(composer=v['folder']) for s,v in r['source_scope'].items()})
    need(replay==r['scores'],'Full report exact replay')
    checks=0
    for name in pred:
        for sid in truth:
            for ms in (30,50,80):
                c=counts(pred[name][sid],truth[sid],ms/1000)
                need(all(c[k]==r['scores'][name]['per_song'][sid]['scores'][str(ms)][k] for k in ('TP','FP','FN')),'Independent count')
                checks+=1
    losses=[];fn_reasons=collections.Counter();by_source={}
    for sid in truth:
        ev=read(frozen/(sid+'.candidates.json'));prob=np.load(frozen/(sid+'.head.npz'),allow_pickle=False)['probability']
        old,_=match_ledger(pred['BasicPitch_default'][sid],truth[sid])
        new,_=match_ledger(pred[r['primary_policy']][sid],truth[sid])
        by_source[sid]=dict(folder=r['source_scope'][sid]['folder'],gained=len(new-old),lost=len(old-new))
        for pitch,index,t in sorted(old-new):
            actual=sorted(x['time'] for x in truth[sid] if pitch in x['notes'])
            guessed=sorted(x['time'] for x in pred['BasicPitch_default'][sid] if pitch in x['notes'])
            i=next(i for i,j,d in match_times(np.array(guessed),np.array(actual),.08) if j==index)
            candidates=[k for k,x in enumerate(ev) if x['pitch']==pitch and abs(x['time']-guessed[i])<1e-9 and x['origin'][0]==1.]
            need(len(candidates)==1,'Every default BP note is present in the candidate bank')
            probability=float(prob[candidates[0]])
            losses.append(dict(source_id=sid,folder=r['source_scope'][sid]['folder'],pitch=pitch,
                truth_time=t,baseline_matched_time=guessed[i],candidate_probability=probability,
                reason='below_frozen_0.5_gate' if probability<.5 else 'NMS_or_matching_competition'))
        for pitch in range(40,89):
            for index,t in enumerate(sorted(x['time'] for x in truth[sid] if pitch in x['notes'])):
                if (pitch,index,t) in new:continue
                near=[float(prob[k]) for k,x in enumerate(ev) if x['pitch']==pitch and abs(x['time']-t)<=.08]
                reason='no_candidate_within80ms' if not near else ('all_candidates_below_0.5' if max(near)<.5 else 'NMS_or_matching_competition')
                fn_reasons[reason]+=1
    need(len(losses)==r['comparison_to_default_BP'][r['primary_policy']]['lost'],'All lost baseline true notes accounted')
    need(sum(fn_reasons.values())==r['scores'][r['primary_policy']]['fullstream']['80']['FN'],'All missed notes accounted')
    save(out,dict(retrospective_verification_not_another_test=True,full_score_exact_replay=True,
        independent_maximum_cardinality_checks=checks,by_source=by_source,
        baseline_true_note_losses=losses,primary_FN_categories=dict(fn_reasons),
        lost_baseline_categories=dict(collections.Counter(x['reason'] for x in losses)),
        no_model_threshold_or_prediction_changes=True))
    print('S605 OFFLINE AUDIT PASS',checks,'independent checks;',len(losses),'baseline losses;',dict(fn_reasons),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('model'))
    p.add_argument('--out',type=Path,default=Path('model/S605/offline_verification.json'))
    a=p.parse_args();run(a.root,a.out)
