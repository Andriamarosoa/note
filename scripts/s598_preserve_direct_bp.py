"""Retrospective ablation: preserve BP times, add S585 candidates. No fitting."""
import argparse
from pathlib import Path
import numpy as np
from scripts.s585_attack_sustain_restrike import need, sha, save, metrics
from scripts.s592_s597_urmp_prospective import read, rows
from scripts.s591_score_state_note_predictions import match_ledger


def run(root, out):
    need(not out.exists(), 'Immutable S598 report')
    out.mkdir(parents=True)
    result = {}
    for label, freeze, features, bp, truthfile in (
        ('URMP_exposed5', 'S596', 'S595/features.jsonl', 'S594/notes.jsonl', 'S597/truth_after_freeze.json'),
        ('piano_exposed2', 'S590', 'S589/S589_ALL_V88_K0_K6_ORIGINAL_WAV_49_PITCH_FEATURES_PREMIDI.jsonl',
         'S588/S588_FIXED_BASIC_PITCH_PRETRAINED_PIANO_NOTES_PRELABEL.jsonl', 'S591/truth_after_freeze.json')):
        raw = rows(root/features); truth = read(root/truthfile)
        with np.load(root/freeze/'probabilities.npz', allow_pickle=False) as z:
            prob = z['state4'][:,:,2:].sum(axis=2); sources = z['source'].astype(str)
        need(len(raw) == len(sources), 'Feature/probability ordering')
        need(all(r['source_recording_sha'] == s for r,s in zip(raw,sources)), 'Source ordering')
        base = {s:[] for s in truth}
        for n in rows(root/bp):
            base[n['WAV_filename_SHA256']].append(dict(time=n['predicted_onset_s'],notes=[n['pitch_MIDI']]))
        pred = {'BasicPitch_direct':base}
        audit = {}
        for threshold in (.7,.8,.9):
            for gap in (.08,.10,.16):
                name = f'BP_plus_S585_p{threshold:.1f}_gap{int(gap*1000)}'
                pred[name] = {}
                audit[name] = {}
                for sid in truth:
                    candidates = []
                    bt = {p:[g['time'] for g in base[sid] if p in g['notes']] for p in range(40,89)}
                    for i in np.flatnonzero(sources == sid):
                        t = raw[i]['original_V88_candidate_start_s']
                        for j in np.flatnonzero(prob[i] >= threshold):
                            p = int(j+40)
                            if not any(abs(t-v) < gap for v in bt[p]):
                                candidates.append((float(prob[i,j]),float(t),p))
                    chosen = []
                    for score,t,p in sorted(candidates,reverse=True):
                        if not any(q == p and abs(t-u) < .1 for _,u,q in chosen):
                            chosen.append((score,t,p))
                    pred[name][sid] = sorted(base[sid] + [dict(time=t,notes=[p]) for _,t,p in chosen],key=lambda g:g['time'])
                    old,oldfp = match_ledger(base[sid],truth[sid])
                    new,newfp = match_ledger(pred[name][sid],truth[sid])
                    audit[name][sid] = dict(gained=sorted(new-old),lost=sorted(old-new),
                        old_FP=len(oldfp),new_FP=len(newfp),added_predictions=len(chosen))
        scores = metrics(pred,truth,{s:dict(composer=s) for s in truth})
        result[label] = dict(scores=scores,audit=audit,input_hashes={p:sha(root/p) for p in
            (features,bp,truthfile,freeze+'/probabilities.npz')})
        for name,s in scores.items():
            print('S598 DEVELOPMENT',label,name,s['fullstream']['80'],flush=True)
    save(out/'report.json',dict(development_only=True,not_new_independent_test=True,
        baseline_notes_never_removed_or_retimed=True,S80_and_Live_unchanged=True,cohorts=result))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('model'))
    p.add_argument('--out',type=Path,default=Path('model/S598'));a=p.parse_args();run(a.root,a.out)
