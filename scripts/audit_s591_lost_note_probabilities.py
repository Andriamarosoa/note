"""Post-label diagnosis of frozen S591 note losses; never a new test or tuning."""
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need,sha,save
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times

def run(root,out):
    score=json.loads((root/'S591/report.json').read_text())
    freeze=json.loads((root/'S590/report.json').read_text())
    feature=root/'S589/S589_ALL_V88_K0_K6_ORIGINAL_WAV_49_PITCH_FEATURES_PREMIDI.jsonl'
    files={'score':root/'S591/report.json','truth':root/'S591/truth_after_freeze.json',
        'ledger':root/'S591/individual_note_audit.json','features':feature,
        'probabilities':root/'S590/probabilities.npz','predictions':root/'S590/predictions.json'}
    need(sha(files['predictions'])==freeze['notes_sha256'] and
        sha(files['probabilities'])==freeze['probabilities_sha256'] and
        sha(feature)==freeze['feature_sha256'] and sha(files['ledger'])==score['individual_audit_sha256'],
        'Post-label audit must use original SHA-frozen probability and note ledgers')
    ledger=json.loads(files['ledger'].read_text());pred=json.loads(files['predictions'].read_text())
    truth=json.loads(files['truth'].read_text())
    rows=[json.loads(s) for s in feature.read_text().splitlines()]
    name='S585_state_strict100';details=[]
    old={(x['source_sha'],x['pitch'],x['time']) for x in ledger if x['policy']==name and x['change']=='control_false_positive'}
    new={(x['source_sha'],x['pitch'],x['time']) for x in ledger if x['policy']==name and x['change']=='candidate_false_positive'}
    with np.load(files['probabilities'],allow_pickle=False) as z:
        for x in ledger:
            if x['policy']!=name or not x['change'].startswith('true'):continue
            src=x['source_sha'];p=x['pitch']
            selected='S578_global28' if x['change']=='true_note_lost' else name
            guess=sorted(float(e['time']) for e in pred[selected][src] if p in e['notes'])
            real=sorted(float(e['time']) for e in truth[src] if p in e['notes'])
            pairs=match_times(np.asarray(guess),np.asarray(real),.08)
            times=[guess[i] for i,j,d in pairs if j==x['truth_index']]
            need(len(times)==1,'Every changed true note must retain its 1:1 matched prediction')
            ii=[i for i,r in enumerate(rows) if r['source_recording_sha']==src and
                abs(float(r['original_V88_candidate_start_s'])-times[0])<1e-8]
            need(len(ii)==1,'Changed prediction must come from one original WAV event')
            i=ii[0];q=z['state4'][i,p-40]
            details.append(dict(x,predicted_time=times[0],
                S578_probability=float(z['global28'][i,p-40]),state_probabilities=q.tolist(),
                new_attack_probability=float(q[2:].sum()),
                is_original_BP_birth=p in rows[i]['old_basicpitch_birth_pitches']))
    c=collections.Counter(x['change'] for x in details)
    comp=score['comparisons'][name]
    need(c['true_note_recovered']==comp['true_note_recovered'] and
        c['true_note_lost']==comp['true_note_lost'] and
        len(new-old)-len(old-new)==comp['delta_FP'],'Post-label audit must reconcile score')
    save(out,dict(retrospective_diagnostic_not_new_test=True,thresholds_and_weights_NOT_changed=True,
        source_SHA256={k:sha(p) for k,p in files.items()},FP_removed=len(old-new),FP_added=len(new-old),
        FP_retained=len(old&new),notes=details))
    print('S591 POSTLABEL AUDIT',dict(c), 'FP removed/added',len(old-new),len(new-old))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
