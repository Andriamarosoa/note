"""S403 — deployable-shaped S402 K-conditioned neural MIDI note inference.

Inputs are acoustic S97 (row_id, sequence42x49) and precomputed, aligned
S80 guarded reference K (row_id,s80_K). Four prefit S402 pitch networks
are used as an unweighted ensemble on previously unseen audio. No JAMS,
no labels, no current true K and no learn/fit in inference.

WARNING: ensemble ON GuitarSet is not an OOF experiment since three or four
models may have trained on that composition. Do not score ensemble on those
same data or confuse with S402 song-OOF evaluation.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import joblib,numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import feature_bank,need
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import choose_pitches,note_name,sha

def run(audio,s80,models,out):
    need(not out.exists(),'S403 may not overwrite earlier output')
    with np.load(audio,allow_pickle=False) as z:
        need({'row_id','sequence42x49'}.issubset(z.files),
             'original S97 42x49 acoustic source missing')
        row=z['row_id'].astype(str);activation=z['sequence42x49']
    with np.load(s80,allow_pickle=False) as z:
        need({'row_id','s80_K'}.issubset(z.files),
             'active S80 predicted K not supplied')
        ids=z['row_id'].astype(str);k=z['s80_K'].astype(np.int8)
    n=len(row)
    need(n>0 and len(set(row))==n and np.array_equal(row,ids)
         and activation.shape==(n,42,49)
         and np.isin(k,range(7)).all(),
         'S403 MIDI note input event alignment or S80 K bad')
    loaded=[]
    checks={}
    for fold in range(4):
        path=models/f'S402_pitch_head_held_song_fold{fold}.joblib'
        need(path.is_file(),'Missing original S402 prefit pitch model: '+str(path))
        d=joblib.load(path)
        need(d['S80_K_fed_to_model_input'] is True and d['feature_width']==352
          and d['held_fold']==fold and hasattr(d['network'],'predict_proba'),
          'S402 checkpoint provenance or K input contract mismatch')
        loaded.append(d)
        checks[str(fold)]=sha(path)
    pitches=np.full((n,6),-1,np.int16)
    conf=np.zeros((n,6),np.float32)
    p_scores=np.zeros((n,49),np.float32)
    # Inference-time K is part of the neural input features, prior to the
    # MLP's predicted MIDI probabilities and before exact-K note selection.
    for i in range(0,n,512):
        j=min(n,i+512)
        x=feature_bank(activation[i:j],k[i:j])
        pooled=np.zeros((j-i,49),np.float64)
        for d in loaded:
            probs=d['network'].predict_proba(d['transform'].transform(x))
            need(probs.shape==(j-i,49) and np.isfinite(probs).all(),
                 'S403 S402-network MIDI probability shape drift')
            pooled+=probs/len(loaded)
        p_scores[i:j]=pooled.astype(np.float32)
        pitches[i:j],conf[i:j]=choose_pitches(pooled,k[i:j])
    need(np.array_equal((pitches>=0).sum(1),k.astype(np.int64)),
         'S403 did not preserve S80 exact-K')
    out.mkdir(parents=True)
    predictions=out/'S403_S80_K_CONDITIONED_note_predictions.npz'
    np.savez_compressed(predictions,row_id=row,s80_K=k,
          midi_pitch_slots=pitches,chosen_note_scores=conf,
          note_probabilities_midi40_to_88=p_scores)
    names=out/'notes.csv'
    with names.open('w',newline='',encoding='utf-8') as h:
        writer=csv.writer(h)
        writer.writerow(['row_id','S80_K','MIDI_pitches','note_names','confidence'])
        for i in range(n):
            midi=[int(v) for v in pitches[i] if v>=0]
            writer.writerow([row[i],int(k[i]),
                ' '.join(map(str,midi)),
                ' '.join(note_name(m) for m in midi),
                ' '.join(f'{v:.5f}' for v in conf[i,:len(midi)])])
    report=dict(status='S403_FROZEN_S402_K_FIRST_MIDI_NEURAL_RUNTIME',
       n_events=n,
       source_reference='S80_S397_K5_GUARDED',
       model='S402_K_CONDITIONED_352D_MLP',
       S80_K_at_model_INPUT=True,
       only_select_exact_K_notes=True,
       no_ground_truth_loaded=True,no_fit_or_tuning_during_runtime=True,
       source_audio_sha256=sha(audio),
       source_S80_K_sha256=sha(s80),
       S402_model_sha256=checks,
       prediction_sha256=sha(predictions),
       note_csv_sha256=sha(names),
       runtime_ensemble_not_same_as_song_OOF_measurement=True,
       source_reference_S80_untouched=True,
       new_WAV_exactK_must_be_computed_upstream=True,
       limitations='Unique pitch labels MIDI40..88 only; current S80 K may be wrong; per-note independent probabilities with K-conditioned shared representation.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print('S403 PASS K enters note model before MIDI logits on every event')
def selftest():
    import tempfile
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    x=np.zeros((6,42,49),np.float32)
    x[:,14:23,7]=np.log1p(4.)
    y=np.zeros((6,49),np.int8)
    y[0,7]=1;y[1,7]=1;y[2,7]=1;y[3,8]=1;y[4,8]=1;y[5,9]=1
    k0=np.array([0,1,2,3,4,5],np.int8)
    ft=feature_bank(x,k0)
    need(ft.shape==(6,352) and not np.array_equal(ft[0],ft[1]),
         'K first features must change before MIDI probability prediction')
    with tempfile.TemporaryDirectory() as p:
        d=Path(p);(d/'models').mkdir()
        for fold in range(4):
            ss=StandardScaler().fit(ft)
            nn=MLPClassifier(hidden_layer_sizes=(8,),max_iter=5,
             solver='adam',random_state=fold,batch_size=6)
            nn.fit(ss.transform(ft),y)
            joblib.dump(dict(S80_K_fed_to_model_input=True,
              feature_width=352,held_fold=fold,network=nn,transform=ss),
              d/'models'/f'S402_pitch_head_held_song_fold{fold}.joblib')
        np.savez_compressed(d/'audio.npz',row_id=np.arange(6).astype(str),
            sequence42x49=x)
        np.savez_compressed(d/'s80.npz',row_id=np.arange(6).astype(str),
            s80_K=k0)
        run(d/'audio.npz',d/'s80.npz',d/'models',d/'out')
        with np.load(d/'out/S403_S80_K_CONDITIONED_note_predictions.npz') as z:
            need(np.array_equal((z['midi_pitch_slots']>=0).sum(1),k0)
                and z['note_probabilities_midi40_to_88'].shape==(6,49),
                'runtime K-conditioned note output mismatch')
        need(len((d/'out/notes.csv').read_text().splitlines())==7,
             'S403 missing note names in output')
    print('S403 SELFTEST PASS: K0..K5, 4 prefit models, genuine K feature input')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for key in ('audio','s80','models','out'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ('audio','s80','models','out'):
            need(getattr(a,key)is not None,'missing --'+key)
        run(a.audio,a.s80,a.models,a.out)
