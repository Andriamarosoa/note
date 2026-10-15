"""S591 prospective scoring only after a separate S590 freeze artifact exists.

All 49 pitches, all false births, all missed notes; fixed +/-30/50/80ms.
Reports individual matched true notes gained/lost and false emissions. No tune.
"""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need,sha,save,metrics,POLICIES
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import fetch_midi,midi_events,onset_groups,match_times
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes
from scripts.s534_independent_real_repeat_notes_dedup_tolerance_audit import repeat_truth_intervals

def match_ledger(groups,truth):
    matched=set();false=[]
    for pitch in range(40,89):
        guess=sorted(float(x['time']) for x in groups if pitch in x['notes'])
        real=sorted(float(x['time']) for x in truth if pitch in x['notes'])
        pairs=match_times(np.asarray(guess),np.asarray(real),.08)
        used={i for i,j,d in pairs}
        matched.update((pitch,j,real[j]) for i,j,d in pairs)
        false.extend(dict(pitch=pitch,time=t) for i,t in enumerate(guess) if i not in used)
    return matched,false

def run(root,out):
    need(not out.exists(),'S591 independent scoring output immutable')
    m=json.loads((root/'S586/report.json').read_text())
    freeze=json.loads((root/'S590/report.json').read_text())
    proof=json.loads((root/'S587/report.json').read_text())
    features=root/'S589/S589_ALL_V88_K0_K6_ORIGINAL_WAV_49_PITCH_FEATURES_PREMIDI.jsonl'
    need(freeze['status']=='S590_THREE_POLICIES_SEALED_BEFORE_SIXTH_MIDI'
         and freeze['MIDI_NOT_OPENED'] and freeze['policies']==list(POLICIES)
         and freeze['notes_sha256']==sha(root/'S590/predictions.json')
         and freeze['probabilities_sha256']==sha(root/'S590/probabilities.npz')
         and freeze['feature_sha256']==sha(features)
         and all(freeze['input_hashes'][f'S{s}']==sha(root/f'S{s}/report.json') for s in (585,586,587,588,589))
         and m['previous_source_count']==33 and m['source_MIDI_files_not_read']
         and proof['preregistered_S586_SHA']==sha(root/'S586/report.json'),
         'S591 MIDI embargo: previous immutable predictions and all source proofs required')
    pred=json.loads((root/'S590/predictions.json').read_text())
    chosen={x['filename_SHA256']:x for x in m['chosen_metadata_only']}
    need(set(pred)==set(POLICIES) and len(chosen)==5 and all(set(x)==set(chosen) for x in pred.values()),
         'S591 every policy must score all predeclared sources, including empty predictions')
    for p in POLICIES:
        for src,g in pred[p].items():
            need(sum(len(x['notes']) for x in g)==freeze['note_counts'][p][src],'S591 predictions dropped')
            need(all(0<=x['time']<12 and len(x['notes'])==len(set(x['notes'])) and
                all(40<=q<=88 for q in x['notes']) for x in g),'S591 malformed predictions')
    # FIRST ACCESS TO TARGET ANNOTATIONS. This script runs in a later CI job.
    truth={}; provenance=[]
    for src,item in sorted(chosen.items()):
        blob=fetch_midi(m['source_audio_mirror'],item['MIDI_filename_path_ONLY_not_opened'])
        ev=midi_events(blob);off=float(item['audio_excerpt_start_seconds'])
        groups=onset_groups(ev,off,float(item['audio_excerpt_duration_seconds']))
        gt=[dict(time=float(g['onset']),notes=[int(p) for p in new_notes(ev,off,float(g['onset'])) if 40<=int(p)<=88]) for g in groups]
        truth[src]=[x for x in gt if x['notes']]
        provenance.append(dict(source_sha=src,composer=item['composer'],MIDI_sha256=hashlib.sha256(blob).hexdigest(),
            actual_true_notes=sum(len(x['notes']) for x in truth[src]),
            real_restrikes_under100ms=repeat_truth_intervals(truth[src],.100)))
    result=metrics(pred,truth,chosen)
    ledger=[];changes={}
    for name in POLICIES[1:]:
        changes[name]=collections.Counter()
        for src,gt in truth.items():
            old,fp0=match_ledger(pred[POLICIES[0]][src],gt)
            new,fp1=match_ledger(pred[name][src],gt)
            for label,diff in (('true_note_recovered',new-old),('true_note_lost',old-new)):
                for pitch,index,t in sorted(diff):
                    ledger.append(dict(policy=name,source_sha=src,composer=chosen[src]['composer'],
                        change=label,pitch=pitch,truth_time=t,truth_index=index))
                    changes[name][label]+=1
            for label,events in (('control_false_positive',fp0),('candidate_false_positive',fp1)):
                for event in events:ledger.append(dict(policy=name,source_sha=src,change=label,**event))
            need(len(old)==result[POLICIES[0]]['per_song'][src]['scores']['80']['TP'] and
                 len(new)==result[name]['per_song'][src]['scores']['80']['TP'], 'Independent note ledger/score divergence')
    comparison={};rng=np.random.default_rng(59101);srcs=sorted(truth)
    for name in POLICIES[1:]:
        base=result[POLICIES[0]]['fullstream']['80'];now=result[name]['fullstream']['80']
        need(changes[name]['true_note_recovered']-changes[name]['true_note_lost']==now['TP']-base['TP'],
             'S591 exact corrected/lost TP reconciliation')
        def f1(z):return 2*z[0]/max(1,2*z[0]+z[1]+z[2])
        aa=np.array([[result[POLICIES[0]]['per_song'][s]['scores']['80'][k] for k in ('TP','FP','FN')] for s in srcs])
        bb=np.array([[result[name]['per_song'][s]['scores']['80'][k] for k in ('TP','FP','FN')] for s in srcs])
        deltas=[]
        for _ in range(5000):
            ix=rng.integers(0,len(srcs),len(srcs));deltas.append(100*(f1(bb[ix].sum(0))-f1(aa[ix].sum(0))))
        comparison[name]=dict(delta_F1_points=100*(now['F1']-base['F1']),delta_TP=now['TP']-base['TP'],
            delta_FP=now['FP']-base['FP'],**dict(changes[name]),
            five_source_bootstrap_interval95_points=np.percentile(deltas,[2.5,97.5]).tolist(),
            interval_is_descriptive_small_sample_not_universal_significance=True,
            songs_improved=sum(result[name]['per_song'][s]['scores']['80']['F1']>result[POLICIES[0]]['per_song'][s]['scores']['80']['F1'] for s in srcs),
            songs_regressed=sum(result[name]['per_song'][s]['scores']['80']['F1']<result[POLICIES[0]]['per_song'][s]['scores']['80']['F1'] for s in srcs))
    out.mkdir(parents=True);save(out/'truth_after_freeze.json',truth);save(out/'individual_note_audit.json',ledger)
    report=dict(status='S591_SIXTH_SOURCE_PROSPECTIVE_FOUR_STATE_NOTEON_VERDICT',
        source_count=5,official_split='MAESTRO_train_NOT_official_test',
        true_note_count=sum(sum(len(g['notes']) for g in x) for x in truth.values()),
        original_MIDI_read_only_after_separate_SHA_frozen_prediction_job=True,
        freeze_SHA=sha(root/'S590/report.json'),MIDI_provenance=provenance,
        scores=result,comparisons=comparison,individual_audit_sha256=sha(out/'individual_note_audit.json'),
        full_S80_ExactK_NOT_REPLAYED=True,live_NOT_modified=True,multi_instrument_NOT_tested=True,
        no_promotion=True)
    save(out/'report.json',report)
    for p in POLICIES:print('S591 PROSPECTIVE',p,json.dumps(result[p]['fullstream']['80']),flush=True)
    print('S591 PAIRED AUDIT',json.dumps(comparison),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
