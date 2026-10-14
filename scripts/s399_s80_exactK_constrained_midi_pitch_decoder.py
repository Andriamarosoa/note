"""S399 — S80 active K5-guarded reference + first REAL pitch identity decoder.

Zero-shot, label-blind upstream harmonic activation -> MIDI pitch 40..88.
It does NOT infer cardinality from spectral peak counts. S80 S397 supplies K.
The source 42x49 activation bank is fitted six-partial harmonic templates,
not six independent physical notes; note pitches are proposals only.
All prediction outputs are SHA-sealed before any ground truth is opened.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
MIDI_MIN=40
MIDI_MAX=88
SLOTS=6
N=59309
HOP=256
RATE=44100
TIMES=(np.arange(42,dtype=np.float64)*HOP*1000./RATE)-80.0
PRE=(TIMES>=-65)&(TIMES< -18)
ON=(TIMES>=-12)&(TIMES<=42)
LATE=(TIMES>70)&(TIMES<140)
assert all(int(x.sum())>=5 for x in (PRE,ON,LATE))
def require(ok,why):
    if not ok:raise ValueError(why)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fp:
        for chunk in iter(lambda:fp.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def note_name(midi):
    labels=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
    return labels[int(midi)%12]+str(int(midi)//12-1)
def harmonic_pitch_scores(sequence):
    """Novel onset-relative excitation plus physical persistence and magnitude.

    No MIDI target, K truth, XML/JAMS, audio filename, fold or model-training
    metadata enters this function. All constants are fixed BEFORE evaluation.
    """
    z=np.asarray(sequence,np.float32)
    require(z.ndim==3 and z.shape[1:]==(42,49),
            'genuine original 42-frame by 49-MIDI-harmonic trajectory required')
    require(np.isfinite(z).all() and z.min()>=0,
            'nonnegative log1p harmonic activation required')
    a=np.expm1(z.astype(np.float64))
    pre=a[:,PRE].mean(axis=1)
    on=a[:,ON].mean(axis=1)
    late=a[:,LATE].mean(axis=1)
    excess=np.maximum(on-pre,0)
    # Per-note positive birth evidence; avoids relying on absolute volume
    # that changes drastically across tracks.
    birth=excess/(np.maximum(on,pre)+1e-6)
    # Presence at the onset / continuation into immediate late frames
    # prevents pure pre-existing sustain from masquerading as a new attack.
    prominence=on/(np.max(on,axis=1,keepdims=True)+1e-6)
    sustain=np.minimum(late,on)/(np.maximum(late,on)+1e-6)
    # Fixed coefficients (not fit on GuitarSet annotations).
    s=(0.60*birth+0.30*prominence+0.10*sustain) * np.sqrt(np.clip(prominence,0,1))
    require(np.isfinite(s).all() and s.shape==(len(z),49),'pitch scores invalid')
    return s.astype(np.float32)
def choose_pitches(scores,k):
    p=np.asarray(scores,np.float32)
    k=np.asarray(k,np.int64)
    require(p.ndim==2 and p.shape[1]==49 and
            k.shape==(len(p),) and np.isin(k,np.arange(7)).all(),
            'S80 K0..6 and genuine 49-source acoustic scores necessary')
    # Fixed deterministic tie order. Distinct MIDI pitch proposals only.
    order=np.argsort(-p,axis=1,kind='stable')[:,:SLOTS]
    out=np.full((len(p),SLOTS),-1,np.int16)
    conf=np.zeros((len(p),SLOTS),np.float32)
    for j in range(SLOTS):
        active=k>j
        out[active,j]=MIDI_MIN+order[active,j].astype(np.int16)
        conf[active,j]=p[np.flatnonzero(active),order[active,j]]
    require(np.array_equal((out>=0).sum(axis=1),k),
            'pitch decoder must ALWAYS emit exactly S80 K notes, never count itself')
    return out,conf
def demo_test():
    x=np.zeros((4,42,49),np.float32)
    # Inject three controlled temporal pitch activations.
    x[0,ON,7]=np.log1p(10.0);x[0,ON,16]=np.log1p(8.0)
    x[1,ON,4]=np.log1p(12.0)
    x[2,ON,2]=np.log1p(7.0);x[2,ON,14]=np.log1p(6.0);x[2,ON,26]=np.log1p(5.0)
    # a strong sustained pitch without new attack must not block genuine onset
    x[3,:,9]=np.log1p(5.0);x[3,ON,20]=np.log1p(8.0)
    s=harmonic_pitch_scores(x)
    pred,conf=choose_pitches(s,[2,0,3,1])
    require(pred[0,:2].tolist()==[47,56] and pred[1].tolist()==[-1]*6
      and set(pred[2,:3])=={42,54,66} and pred[3,0]==60,
      'S399 synthetic note identity reconstruction failed')
    require(np.all(conf>=0) and all(len(set(x[x>=0]))==len(x[x>=0]) for x in pred),
      'note rank strictness invalid')
    print('S399 SELFTEST PASS: K0 K1 K2 K3, stable pitch matching, no target dependence')
def run(audio,s80,source_report,out):
    require(not out.exists(),'cannot overwrite label-blind S399 evidence')
    report=json.loads(Path(source_report).read_text())
    require(report['status']=='S397_SOURCE_S70_K5_DOWNCAST_GUARD_FULL_S80_REPLAY_LABEL_FREE'
      and report['labels_used'] is False
      and report['S80_reference_model_weights_unchanged']
      and sha(s80)==report['source_sha256']['replayed_S80'],
      'S399 must consume exactly S397 SHA-frozen ACTIVE reference')
    with np.load(audio,allow_pickle=False) as z:
        require({'row_id','sequence42x49'}.issubset(z.files),
                'original harmonic audio source absent')
        rows=z['row_id'].astype(str);seq=z['sequence42x49']
    with np.load(s80,allow_pickle=False) as z:
        require({'row_id','s80_K'}.issubset(z.files),'S397 guarded S80 native rows absent')
        kid=z['row_id'].astype(str);k=z['s80_K']
    require(len(rows)==N and np.array_equal(rows,kid)
      and seq.shape==(N,42,49) and k.shape==(N,),
      'S399 original native event ID or source shape mismatch')
    pitches=np.full((N,SLOTS),-1,np.int16)
    confidence=np.zeros((N,SLOTS),np.float32)
    full=np.empty((N,49),np.float32)
    for i in range(0,N,512):
        j=min(N,i+512)
        evidence=harmonic_pitch_scores(seq[i:j])
        p,c=choose_pitches(evidence,k[i:j])
        pitches[i:j]=p;confidence[i:j]=c;full[i:j]=evidence
    require(np.array_equal((pitches>=0).sum(1),k.astype(int)),
            'S399 violation: notes cardinality must equal guarded S80')
    out.mkdir(parents=True)
    path=out/'S399_S80_K5_GUARDED_MIDI_note_predictions_N59309.npz'
    np.savez_compressed(path,row_id=rows,s80_K=k.astype(np.int8),
       midi_pitch_slots=pitches,note_candidate_scores=full,
       chosen_note_scores=confidence)
    saved=sha(path)
    samples=[dict(row_id=str(rows[i]),K=int(k[i]),
      notes=[note_name(m) for m in pitches[i] if m>=0],
      midi=[int(m) for m in pitches[i] if m>=0])
      for i in range(len(k)) if 2<=k[i]<=5][:12]
    obj=dict(status='S399_LABEL_BLIND_REAL_PITCH_SOURCE_S80_K5_GUARD_REFERENCE',
      reference_id='S80_S397_K5_GUARDED',
      rows=N,pitches_midi_min=MIDI_MIN,pitches_midi_max=MIDI_MAX,
      input_harmonic_frames=42,input_harmonic_bins=49,
      predicted_cardinality_eq_reference_all_rows=True,
      number_predicted_K0=int((k==0).sum()),
      output_sha256=saved,
      source_audio_sha256=sha(audio),source_S80_reference_sha256=sha(s80),
      predictor_uses_ground_truth=False,labels_read=False,
      pretrained_pitch_model=False,feature_bank='nonnegative 6-harmonic template activations',
      training_overlap_of_S80_reference=True,
      no_independent_note_accuracy_yet=True,
      duplicate_same_pitch_voices_not_representable=True,
      outside_pitch_40_88_not_representable=True,
      examples=samples,notes_claim='candidate MIDI pitch identities, not verified musical transcription',
      reference_S80_unchanged=True)
    (out/'report.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({k:v for k,v in obj.items() if k!='examples'},indent=2),flush=True)
    print('S399 PASS: all S80 K0..6 cardinalities protected; source notes sealed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for x in ('audio','s80','source-report','out'):
        p.add_argument('--'+x,required=False,type=Path)
    v=p.parse_args()
    if v.self_test:demo_test()
    else:
        for x in ('audio','s80','source_report','out'):
            require(getattr(v,x) is not None,'missing required --'+x.replace('_','-'))
        run(v.audio,v.s80,v.source_report,v.out)
