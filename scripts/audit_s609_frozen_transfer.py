"""Offline replay and individual loss attribution; never fit or tune."""
import argparse
import collections
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.s592_s597_urmp_prospective import read
from scripts.s591_score_state_note_predictions import match_ledger
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times
from scripts.s601_bp_candidate_gate import counts, emit
from scripts.s606_s609_frozen_cross_corpus import checked_freezes, CORPORA


def run(root):
    c,m,freezes = checked_freezes(root/'S608')
    summary = {}
    for corpus in CORPORA:
        frozen = root/'S608'/corpus
        scored = root/'S609'/corpus
        out = scored/'offline_verification.json'
        need(not out.exists(),'Immutable offline verification')
        r = read(scored/'report.json')
        pred = read(frozen/'predictions.json')
        truth = read(scored/'truth_after_freeze.json')
        need(sha(frozen/'report.json')==r['freeze_report_sha256']
             and sha(frozen/'predictions.json')==r['prediction_sha256']
             and all(sha(scored/name)==h for name,h in r['audit_hashes'].items()),'Artifact identity')
        replay = metrics(pred,truth,{sid:dict(composer=s['label']) for sid,s in r['source_scope'].items()})
        need(replay==r['scores'],'Full report exact replay')
        checks = 0
        for name in pred:
            for sid in truth:
                for ms in (30,50,80):
                    q = counts(pred[name][sid],truth[sid],ms/1000)
                    need(all(q[k]==r['scores'][name]['per_song'][sid]['scores'][str(ms)][k] for k in ('TP','FP','FN')),
                         'Independent maximum cardinality')
                    checks += 1
        losses = []; fn_reasons = collections.Counter(); by_source = {}
        for sid in truth:
            ev = read(frozen/(sid+'.candidates.json'))
            prob = np.load(frozen/(sid+'.head.npz'),allow_pickle=False)['probability']
            seconds = r['source_scope'][sid]['seconds']
            decoded = [x for x in emit(ev,prob,**c['learned_decoder']) if x['time']<seconds]
            need(decoded==pred[r['primary_policy']][sid],'All frozen S601 decisions exactly replay')
            old,_ = match_ledger(pred['BasicPitch_default'][sid],truth[sid])
            new,_ = match_ledger(pred[r['primary_policy']][sid],truth[sid])
            by_source[sid] = dict(label=r['source_scope'][sid]['label'],gained=len(new-old),lost=len(old-new),
                F1_BP=r['scores']['BasicPitch_default']['per_song'][sid]['scores']['80']['F1'],
                F1_S601=r['scores'][r['primary_policy']]['per_song'][sid]['scores']['80']['F1'])
            for pitch,index,t in sorted(old-new):
                actual = sorted(x['time'] for x in truth[sid] if pitch in x['notes'])
                guessed = sorted(x['time'] for x in pred['BasicPitch_default'][sid] if pitch in x['notes'])
                i = next(i for i,j,d in match_times(np.array(guessed),np.array(actual),.08) if j==index)
                where = [k for k,x in enumerate(ev) if x['pitch']==pitch and abs(x['time']-guessed[i])<1e-9 and x['origin'][0]==1.]
                need(len(where)==1,'Every lost default BP note exists in frozen candidate bank')
                probability = float(prob[where[0]])
                losses.append(dict(source_id=sid,pitch=pitch,truth_time=t,baseline_matched_time=guessed[i],
                    candidate_probability=probability,
                    reason='below_frozen_0.5_gate' if probability<.5 else 'NMS_or_matching_competition'))
            for pitch in range(40,89):
                for index,t in enumerate(sorted(x['time'] for x in truth[sid] if pitch in x['notes'])):
                    if (pitch,index,t) in new:
                        continue
                    near = [float(prob[k]) for k,x in enumerate(ev) if x['pitch']==pitch and abs(x['time']-t)<=.08]
                    reason = 'no_candidate_within80ms' if not near else ('all_candidates_below_0.5' if max(near)<.5 else 'NMS_or_matching_competition')
                    fn_reasons[reason] += 1
        need(len(losses)==r['comparison_to_default_BP'][r['primary_policy']]['lost'],'All baseline losses accounted')
        need(sum(fn_reasons.values())==r['scores'][r['primary_policy']]['fullstream']['80']['FN'],'All false negatives accounted')
        save(out,dict(retrospective_verification_not_another_test=True,full_score_exact_replay=True,
            all_S601_decisions_exactly_replayed=True,independent_maximum_cardinality_checks=checks,
            by_source=by_source,baseline_true_note_losses=losses,primary_FN_categories=dict(fn_reasons),
            lost_baseline_categories=dict(collections.Counter(x['reason'] for x in losses)),
            F1_improved_sources=sum(x['F1_S601']>x['F1_BP'] for x in by_source.values()),
            F1_regressed_sources=sum(x['F1_S601']<x['F1_BP'] for x in by_source.values()),
            no_model_threshold_or_prediction_changes=True))
        summary[corpus] = dict(checks=checks,lost_notes=len(losses),FN_categories=dict(fn_reasons),sha256=sha(out))
        print('S609 OFFLINE PASS',corpus,summary[corpus],flush=True)
    save(root/'S609'/'offline_verification.json',summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('model'))
    run(p.parse_args().root)
