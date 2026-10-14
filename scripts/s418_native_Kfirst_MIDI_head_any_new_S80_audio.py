"""S418 — native S416 K-locked MIDI pitch INFERENCE on genuinely new audio.

No JAMS, no true K, no YourMT3 checkpoint or YourMT3 predictions at
inference. Inputs: acoustic 42x49 NPZ and independently precomputed S80 K.
Four frozen S416 CNN checkpoints are an ensemble trained on GuitarSet.
DO NOT report its performance when replayed on GuitarSet itself: three
of four models have seen each original composition. Use frozen S416
fold-held source outputs for evaluation, never this inference ensemble.
S418 is a trained pitch/NoteOn head, NOT an end-to-end raw WAV -> S80.
Note-off requires independently trained S417 full-WAV duration models.
"""
from __future__ import annotations
import argparse,csv,json,tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.s416_native_Kfirst_temporal_MIDI_note_decoder import NativeKTimePitchNet
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import choose_pitches,note_name,sha

def need(x,msg):
    if not x:raise ValueError(msg)

def infer(acoustic,s80,models,out,batch=192):
    need(not out.exists(),'S418 immutable native output path')
    with np.load(acoustic,allow_pickle=False) as a:
        need({'row_id','sequence42x49'}.issubset(a.files),'S418 input lacks true original acoustic features')
        ids=a['row_id'].astype(str);seq=a['sequence42x49'].astype(np.float32)
    with np.load(s80,allow_pickle=False) as kfile:
        need({'row_id','s80_K'}.issubset(kfile.files),
             'S418 user-precomputed S80 count output missing')
        source=kfile['row_id'].astype(str);K=kfile['s80_K'].astype(np.int8)
    need(len(ids)>0 and len(set(ids))==len(ids) and
         seq.shape==(len(ids),42,49) and
         np.array_equal(ids,source) and
         np.isin(K,range(7)).all() and np.isfinite(seq).all()
         and seq.min()>=0,
         'S418 source row IDs, S80 K or actual 42x49 audio malformed')
    torch.set_num_threads(2)
    pretrained=[]
    hashes={}
    for f in range(4):
        name=models/f'S416_native_Kfirst_temporal_note_fold{f}.pt'
        need(name.is_file(),'S416 native checkpoint missing '+str(name))
        cp=torch.load(name,map_location='cpu',weights_only=True)
        need(cp.get('architecture')=='NativeKTimePitchNet'
            and cp.get('K_model_input') is True
            and cp.get('K_output_locked') is True
            and cp.get('training_uses_YourMT3') is False
            and int(cp.get('fold'))==f,
            'Wrong S416 native/no-teacher model provenance')
        model=NativeKTimePitchNet().eval()
        model.load_state_dict(cp['model'],strict=True)
        pretrained.append(model)
        hashes[str(f)]=sha(name)
    output=np.full((len(ids),6),-1,np.int16)
    strength=np.zeros((len(ids),6),np.float32)
    for i in range(0,len(ids),batch):
        j=min(len(ids),i+batch)
        ac=torch.from_numpy(seq[i:j])
        kk=torch.from_numpy(K[i:j].astype(np.int64))
        with torch.no_grad():
            scores=sum(m(ac,kk).sigmoid() for m in pretrained)/len(pretrained)
        pitches,confidence=choose_pitches(scores.numpy(),K[i:j])
        output[i:j]=pitches;strength[i:j]=confidence
    need(np.array_equal((output>=0).sum(1),K),
         'CRITICAL S418 fails per-row S80 K invariance')
    out.mkdir(parents=True)
    npz=out/'S418_NATIVE_Kfirst_MIDI_note_predictions.npz'
    np.savez_compressed(npz,row_id=ids,s80_K=K,
         midi_pitch_slots=output,chosen_pitch_confidence=strength)
    csvname=out/'new_midi_notes.csv'
    with csvname.open('w',newline='',encoding='utf8') as h:
        w=csv.writer(h)
        w.writerow(['row_id','S80_K','MIDI','notes'])
        for r,k,slots in zip(ids,K,output):
            pitches=[int(x) for x in slots if x>=0]
            w.writerow([r,int(k),' '.join(map(str,pitches)),
                        ' '.join(note_name(x) for x in pitches)])
    # Real second independent recount from csv, not only output memory.
    with csvname.open(newline='',encoding='utf8') as h:
        entries=list(csv.DictReader(h))
    need(len(entries)==len(ids)
         and all(int(entry['S80_K'])==int(k)
           and len(entry['MIDI'].split())==int(k)
           for entry,k in zip(entries,K)),
         'S418 independent exported MIDI CSV Exact-K regression')
    report=dict(status='S418_NATIVE_Kfirst_MIDI_HEAD_ARBITRARY_S80_AUDIO_INFERENCE',
       new_events=len(ids),note_on_events=int(K.sum()),
       K_input_from_existing_S80_only=True,
       K0_emits_zero_new_notes=True,
       no_YourMT3_or_GT_at_inference=True,
       S80_predictions_untouched=True,
       full_note_Offset_not_predicted_here=True,
       four_model_GuitarSet_ensemble_is_NOT_song_OOF_evaluation=True,
       model_checkpoint_SHA256=hashes,
       S80_input_SHA256=sha(s80),acoustic_SHA256=sha(acoustic),
       note_predictions_SHA256=sha(npz),CSV_SHA256=sha(csvname),
       official_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S418 PASS native MIDI K-locked inference with zero teacher and zero labels',flush=True)

def selftest():
    torch.manual_seed(418)
    with tempfile.TemporaryDirectory() as t:
        d=Path(t);(d/'models').mkdir()
        for f in range(4):
            model=NativeKTimePitchNet()
            torch.save(dict(model=model.state_dict(),
               architecture='NativeKTimePitchNet',K_model_input=True,
               K_output_locked=True,fold=f,
               training_uses_YourMT3=False),
               d/'models'/f'S416_native_Kfirst_temporal_note_fold{f}.pt')
        ids=np.array(['a','b','c','d']);k=np.array([0,1,3,6],np.int8)
        np.savez_compressed(d/'a.npz',row_id=ids,
            sequence42x49=np.random.default_rng(418).random((4,42,49)).astype(np.float32))
        np.savez_compressed(d/'k.npz',row_id=ids,s80_K=k)
        infer(d/'a.npz',d/'k.npz',d/'models',d/'out')
        with np.load(d/'out/S418_NATIVE_Kfirst_MIDI_note_predictions.npz') as z:
            need(np.array_equal((z['midi_pitch_slots']>=0).sum(1),k),
                 'S418 native model failed K0..K6 synthetic real outputs')
        need((d/'out/new_midi_notes.csv').is_file(),
             'S418 independent source row export absent')
    print('S418 SELFTEST PASS real native 4-model K0 K1 K3 K6 inference on new acoustic samples')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--self-test',action='store_true')
    for name in ('audio','s80','models','out'):a.add_argument('--'+name,type=Path)
    b=a.parse_args()
    if b.self_test:selftest()
    else:
        for name in ('audio','s80','models','out'):
            need(getattr(b,name)is not None,'Missing --'+name)
        infer(b.audio,b.s80,b.models,b.out)
