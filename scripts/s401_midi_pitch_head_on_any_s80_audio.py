"""S401: generic production-SHAPED MIDI note head for ANY already-scored S80 WAV.

Input acoustic NPZ: genuine (row_id, sequence42x49) 49 MIDI 40..88
harmonic activations (S97 original_s45_audio_only or S258 original).
Input S80 NPZ: genuine (row_id, s80_K). K supplied by existing S80.
No labels, no JAMS, no true K. All notes have MIDI IDs and printed names.
Does NOT apply the S397 guarded S70 change itself; that must happen in the
upstream S80 inference chain. Doesn't claim independent pitch accuracy.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    harmonic_pitch_scores,choose_pitches,note_name,sha)
def need(x,m):
    if not x:raise ValueError(m)
def run(audio,s80,out):
    need(not out.exists(),'immutable S401 note transcript output only')
    with np.load(audio,allow_pickle=False) as z:
        need({'row_id','sequence42x49'}.issubset(z.files),
             'S97 acoustic pitch activation input row_id/sequence42x49 absent')
        ids=z['row_id'].astype(str)
        seq=z['sequence42x49']
    with np.load(s80,allow_pickle=False) as z:
        need({'row_id','s80_K'}.issubset(z.files),
             'S80 must supply original event row_id and s80_K class')
        ref_ids=z['row_id'].astype(str)
        predicted_K=z['s80_K'].astype(np.int8)
    n=len(ids)
    need(n>0 and len(set(ids))==n and np.array_equal(ids,ref_ids),
         'S401 audio and S80 predicted event IDs must match exactly')
    need(seq.shape==(n,42,49) and predicted_K.shape==(n,)
         and np.isin(predicted_K,np.arange(7)).all(),
         'original S97 42x49 and S80 K0..6 contract violated')
    p=np.full((n,6),-1,np.int16)
    q=np.zeros((n,6),np.float32)
    for i in range(0,n,256):
        j=min(n,i+256)
        scores=harmonic_pitch_scores(seq[i:j])
        p[i:j],q[i:j]=choose_pitches(scores,predicted_K[i:j])
    need(np.array_equal((p>=0).sum(1),predicted_K),
         'not exactly S80 K candidate notes per audio event')
    out.mkdir(parents=True)
    npz=out/'notes_midi_predictions.npz'
    np.savez_compressed(npz,row_id=ids,s80_K=predicted_K,
                        midi_pitch_slots=p,chosen_note_scores=q)
    csvpath=out/'notes.csv'
    with csvpath.open('w',newline='',encoding='utf-8') as h:
        w=csv.writer(h)
        w.writerow(('row_id','S80_ExactK','MIDI_notes','note_names','scores'))
        for i in range(n):
            midi=[int(v) for v in p[i] if v>=0]
            w.writerow((ids[i],int(predicted_K[i]),
                        ' '.join(map(str,midi)),' '.join(note_name(v) for v in midi),
                        ' '.join(f'{v:.4f}' for v in q[i,:len(midi)])))
    rpt=dict(status='S401_GENERIC_S80_TO_MIDI_NOTE_TRANSCRIPTION',
             note_engine='S399_HARMONIC_MIDI_PITCH_RANKER',
             n_events=n,
             pitch_output_vocabulary='MIDI 40..88',
             K_range=[0,6],original_audio_event_ids_preserved=True,
             exact_K_cardinality_invariant=True,
             source_audio_sha256=sha(audio),source_S80_predictions_sha256=sha(s80),
             note_npz_sha256=sha(npz),note_csv_sha256=sha(csvpath),
             labels_read=False,pitch_model_trained=False,
             S80_inference_model_unchanged=True,
             S397_S70_guard_applied_by_this_module=False,
             external_note_accuracy_unverified=True)
    (out/'report.json').write_text(json.dumps(rpt,indent=2)+'\n')
    print(json.dumps(rpt,indent=2))
    print('S401 PASS MIDI pitch notes emitted for all S80-predicted event groups')
def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)
        z=np.zeros((3,42,49),np.float32)
        z[0,14:21,0]=np.log1p(9.)
        z[0,14:21,12]=np.log1p(8.)
        z[2,14:21,4]=np.log1p(6.)
        ids=np.array(['songX:0','songX:1764','songX:9000'])
        np.savez_compressed(p/'audio.npz',row_id=ids,sequence42x49=z)
        np.savez_compressed(p/'s80.npz',row_id=ids,s80_K=np.array([2,0,1]))
        run(p/'audio.npz',p/'s80.npz',p/'out')
        with np.load(p/'out/notes_midi_predictions.npz') as f:
            need((f['midi_pitch_slots'][0,:2]==[40,52]).all() and
                 (f['midi_pitch_slots'][1]==-1).all() and
                 int(f['midi_pitch_slots'][2,0])==44,
                 'S401 synthetic note/label output failure')
        lines=(p/'out/notes.csv').read_text().splitlines()
        need(len(lines)==4 and 'E2 E3' in lines[1] and 'G#2' in lines[3],
             'S401 MIDI note-name CSV incorrect')
        print('S401 SELFTEST PASS: K0/K1/K2 and note names, native IDs, exact-K source unchanged')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for k in ('audio','s80','out'):
        p.add_argument('--'+k,type=Path)
    v=p.parse_args()
    if v.self_test:selftest()
    else:
        for k in ('audio','s80','out'):
            need(getattr(v,k) is not None,'missing '+k)
        run(v.audio,v.s80,v.out)
