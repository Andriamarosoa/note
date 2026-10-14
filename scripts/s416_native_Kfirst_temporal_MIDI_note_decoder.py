"""S416 — NATIVE K-first time/pitch neural MIDI note decoder, no YourMT inference.

Four song-disjoint models trained from scratch on ORIGINAL GuitarSet acoustic
trajectories and TRAIN-SONG JAMS MIDI note identities. No teacher / external
YourMT3 weights, notes, hidden logits or predictions used even in training.

S80's original per-event K is an INPUT to temporal and pitch attention,
and ALWAYS a hard constraint on the exported note count. K0 => zero NoteOn,
even if the audio / harmonic frame is nonzero.

Prototype: local 42x49 harmonic bank, CNN over time-frequency, attention
across MIDI pitches; unlike YourMT3 this does NOT yet transcribe complete
NoteOff. Compare to S402 note identities and S415 teacher-assisted upper
target, but do not promote this trained model unless it wins without K loss.

GuitarSet had prior exposure to S80. Held-composition training of THIS
new native head does not make the whole pipeline independent.
"""
from __future__ import annotations
import argparse,collections,json,random
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold,labels,load,need)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    choose_pitches,sha)
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import scoped
from scripts.s408_train_pitch_conditioned_held_song_note_offsets import get_gt
from scripts.yourmt3_exactk_common import assign_onsets
import zipfile
from pathlib import PurePosixPath

N=59309
SEED=416
MIDI_MIN=40
MIDI_MAX=88
FOLDS=(0,1,2,3)
SRC_DIM=49
MAX_K=6
EXACT_GLOBAL=53529
EXACT_POLY=6054

class NativeKTimePitchNet(nn.Module):
    """Each note logit learns a temporal pattern and cross-pitch relation.
    K embeds before temporal attention and also conditions each pitch query.
    """
    def __init__(self,dim=40,nhead=4):
        super().__init__()
        self.k=nn.Embedding(7,dim)
        self.front=nn.Sequential(
            nn.Conv2d(1,12,kernel_size=(5,5),padding=(2,2)),
            nn.GELU(),
            nn.Conv2d(12,20,kernel_size=(3,3),padding=(1,1)),
            nn.GELU())
        self.pitch_proj=nn.Linear(20*3+6,dim)
        self.pitch_embedding=nn.Embedding(49,dim)
        self.k_scale=nn.Linear(dim,2*dim)
        self.cross=nn.MultiheadAttention(dim,nhead,batch_first=True,dropout=0)
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),
                                nn.Linear(dim,1))

    def forward(self,acoustic,pre_k):
        if acoustic.ndim!=3 or acoustic.shape[1:]!=(42,49):
            raise ValueError('S416 requires native 42 x 49 audio trajectories')
        if pre_k.ndim!=1 or len(pre_k)!=len(acoustic):
            raise ValueError('S416 requires one S80 K per event')
        if pre_k.dtype!=torch.long:
            raise TypeError('S80 K input dtype must be torch.long')
        if not torch.all((pre_k>=0)&(pre_k<=6)).item():
            raise ValueError('Invalid precomputed S80 K')
        z=self.front(acoustic.unsqueeze(1))
        # Three physically distinct time regions from S399's native 42 frames.
        feats=torch.cat([z[:,:,0:13].mean(2),z[:,:,13:25].mean(2),
                         z[:,:,25:42].mean(2)],dim=1).transpose(1,2)
        raw=torch.expm1(acoustic)
        pre=raw[:,0:13].mean(1)
        on=raw[:,13:25].mean(1)
        late=raw[:,25:42].mean(1)
        born=torch.relu(on-pre)
        strength=torch.stack([torch.log1p(pre),torch.log1p(on),
            torch.log1p(late),torch.log1p(born),
            torch.log1p(torch.relu(late-on)),
            torch.log1p(torch.relu(on-late))],dim=-1)
        x=self.pitch_proj(torch.cat([feats,strength],dim=-1))
        x=x+self.pitch_embedding(torch.arange(49,device=x.device))[None,:,:]
        k_embedding=self.k(pre_k)
        scale,bias=self.k_scale(k_embedding).chunk(2,dim=-1)
        x=x*(1+torch.tanh(scale[:,None,:]))+bias[:,None,:]
        attended,_=self.cross(x,x,x,need_weights=False)
        h=self.norm(x+attended)
        return self.head(h).squeeze(-1)

def pitch_loss(logits,target):
    # Top-K ranking is the evaluated objective; unlike independent per-pitch
    # BCE, sum over target notes is normalized to avoid K6 overdominance.
    y=target.float()
    positives=y.sum(1).clamp(min=1)
    ranking=-(F.log_softmax(logits,dim=1)*y).sum(1)/positives
    bce=F.binary_cross_entropy_with_logits(logits,y,
       pos_weight=torch.tensor(7.,device=logits.device))
    return .58*ranking.mean()+.42*bce

def count_guard(rows,k,slots):
    return bool(len(k)==len(rows) and slots.shape==(len(k),6)
        and np.all(np.isin(k,np.arange(7)))
        and np.array_equal((slots>=0).sum(axis=1),k)
        and np.all((slots[slots>=0]>=40)&(slots[slots>=0]<=88))
        and len(set(rows))==len(rows))

def metric_by_truth_K(truth,k,old,new):
    report={}
    for cls in range(7):
        rows=np.flatnonzero(truth['true_K']==cls)
        oldok=newok=0
        for j in rows:
            target=collections.Counter(int(x) for x in truth['midi_slots'][j] if x>=0)
            orig=collections.Counter(int(x) for x in old[j] if x>=0)
            now=collections.Counter(int(x) for x in new[j] if x>=0)
            oldok+=int(orig==target)
            newok+=int(now==target)
        report[str(cls)]=dict(n=len(rows),S402_correct=oldok,
            S416_correct=newok,delta=newok-oldok,Exact_K_regressions=0)
    return report

def run(audio,s80,original_S402,annotation,truth_path,out,epochs=14):
    need(not out.exists(),'S416 immutable audit cannot overwrite prior results')
    torch.set_num_threads(3)
    data=load(audio);g=load(s80);s=load(original_S402);t=load(truth_path)
    row=data['row_id'].astype(str)
    k=g['s80_K'].astype(np.int8)
    need(len(row)==N and len(set(row))==N
        and np.array_equal(row,g['row_id'].astype(str))
        and np.array_equal(row,s['row_id'].astype(str))
        and np.array_equal(row,t['row_id'].astype(str))
        and np.array_equal(k,s['s80_K'])
        and np.array_equal((s['midi_pitch_slots']>=0).sum(1),k)
        and int((k==t['true_K']).sum())==EXACT_GLOBAL
        and int(((k==t['true_K'])&(t['true_K']>=2)).sum())==EXACT_POLY,
        'Immutable original S80 S402 identity or count drift')
    members=data['member'].astype(str)
    comps=np.array([composition_key(m) for m in members])
    fold=np.array([composition_fold(x) for x in comps],np.int8)
    # Training true pitch identities can only influence samples in other
    # compositions. Held test labels are never passed to training arrays.
    multihot,original_true_slots,integrity=labels(data,annotation,t)
    need(multihot.shape==(N,49) and original_true_slots.shape==(N,6),
         'Original label geometry mismatch')
    true={'true_K':t['true_K'],'midi_slots':original_true_slots}
    seq=data['sequence42x49'].astype(np.float32)
    need(seq.shape==(N,42,49) and np.isfinite(seq).all()
         and np.min(seq)>=0,'Original acoustic 42x49 bank changed')
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    predictions=np.full((N,6),-1,np.int16)
    probabilities=np.zeros((N,49),np.float16)
    proofs=[]
    out.mkdir(parents=True)
    for fold_id in FOLDS:
        val=fold==fold_id
        train=(fold!=fold_id)&(k>0)
        require_no_overlap=not(set(comps[train])&set(comps[val]))
        need(require_no_overlap and train.sum()>9000
            and val.sum()>1000,'S416 song-disjoint fold overlap or small fold')
        train_ids=np.flatnonzero(train)
        test_ids=np.flatnonzero(val)
        gen=torch.Generator().manual_seed(SEED+fold_id)
        ds=TensorDataset(torch.from_numpy(seq[train_ids]),
                torch.from_numpy(k[train_ids].astype(np.int64)),
                torch.from_numpy(multihot[train_ids].astype(np.float32)))
        loader=DataLoader(ds,batch_size=224,shuffle=True,generator=gen,
                          num_workers=0,drop_last=False)
        model=NativeKTimePitchNet()
        optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        # Fixed architecture and epochs. No test-label-driven early stopping.
        losses=[]
        for ep in range(epochs):
            model.train();acc=cnt=0
            for z,kk,yy in loader:
                optim.zero_grad(set_to_none=True)
                lg=model(z,kk)
                loss=pitch_loss(lg,yy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optim.step()
                acc+=float(loss.detach())*len(z);cnt+=len(z)
            losses.append(acc/cnt)
            print(f'S416 TRAIN fold={fold_id} epoch={ep+1}/{epochs} loss={losses[-1]:.5f}',flush=True)
        model.eval()
        for off in range(0,len(test_ids),256):
            ids=test_ids[off:off+256]
            with torch.no_grad():
                scores=model(torch.from_numpy(seq[ids]),
                       torch.from_numpy(k[ids].astype(np.int64))).sigmoid().numpy()
            chosen,_=choose_pitches(scores,k[ids])
            predictions[ids]=chosen
            probabilities[ids]=scores.astype(np.float16)
        weights=out/f'S416_native_Kfirst_temporal_note_fold{fold_id}.pt'
        torch.save({'model':model.state_dict(),'architecture':'NativeKTimePitchNet',
            'K_model_input':True,'K_output_locked':True,
            'fold':fold_id,'training_uses_YourMT3':False,
            'trained_epochs':epochs},weights)
        proofs.append(dict(fold=fold_id,train_songs=int(len(set(comps[train]))),
            held_songs=int(len(set(comps[val]))),
            composition_overlap=0,training_events=int(train.sum()),
            held_events=int(val.sum()),loss_last=float(losses[-1]),
            checkpoint_SHA256=sha(weights)))
    need(count_guard(row,k,predictions),'S416 CRITICAL K LOCK BROKEN')
    byclass=metric_by_truth_K(true,k,s['midi_pitch_slots'],predictions)
    candidate_positive=t['true_K']>=2
    # Compare full chord MIDI identity with fixed S402 note comparator.
    oldpoly=sum(int(collections.Counter(int(q) for q in s['midi_pitch_slots'][i] if q>=0)
             ==collections.Counter(int(q) for q in original_true_slots[i] if q>=0))
             for i in np.flatnonzero(candidate_positive))
    newpoly=sum(int(collections.Counter(int(q) for q in predictions[i] if q>=0)
             ==collections.Counter(int(q) for q in original_true_slots[i] if q>=0))
             for i in np.flatnonzero(candidate_positive))
    npz=out/'S416_NATIVE_Kfirst_MIDI_NOTE_59309_SONG_OOF.npz'
    np.savez_compressed(npz,row_id=row,s80_K=k,midi_pitch_slots=predictions,
       note_probabilities_float16=probabilities,
       composition_fold=fold)
    output=dict(status='S416_NATIVE_KFIRST_TEMPORAL_NOTE_DECODER_SONGHELD',
       ORIGINAL_S80_K_REFERENCE_UNCHANGED=True,
       original_S80_K_correct_global=EXACT_GLOBAL,
       new_S416_K_correct_global=EXACT_GLOBAL,
       original_S80_K_correct_poly=EXACT_POLY,
       new_S416_K_correct_poly=EXACT_POLY,
       exactK_regressions=0,
       no_external_YourMT_weights_or_predictions_at_train_or_test=True,
       full_audio_input_only_native_42x49_harmonic_sequences=True,
       true_K_NEVER_used_as_model_input=True,
       source_labels_only_used_for_other_songs_training=True,
       trained_models=4,epochs_per_model=epochs,
       original_rows=N,output_notes=int(k.sum()),
       frozen_S80_count_sha256=sha(s80),
       baseline_S402_sha256=sha(original_S402),
       original_audio_sha256=sha(audio),
       original_JAMS_sha256=sha(annotation),
       original_true_K_sha256=sha(truth_path),
       performance=dict(original_S402_poly_chord_exact=oldpoly,
         S416_poly_chord_exact=newpoly,
         total_true_poly_events=int(candidate_positive.sum()),
         net_chords=newpoly-oldpoly),
       by_true_K=byclass,
       song_fold_proofs=proofs,
       validation_entire_S80_pipeline_independent=False,
       official_promotion=False,
       output_sha256=sha(npz))
    (out/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print('S416 RESULT',json.dumps(output['performance']),flush=True)
    print('S416 BY_K',json.dumps(byclass),flush=True)
    print('S416 PASS all 59309 original S80 event K exact, no YourMT at inference',flush=True)

def test():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    net=NativeKTimePitchNet().eval()
    wave=torch.rand(4,42,49)
    k=torch.tensor([0,1,3,6],dtype=torch.long)
    with torch.no_grad():
        a=net(wave,k)
        b=net(wave,torch.tensor([4,1,3,6],dtype=torch.long))
    assert a.shape==(4,49) and torch.isfinite(a).all()
    assert not torch.allclose(a[0],b[0]),'S416 K must affect native MIDI logits BEFORE top K'
    p,_=choose_pitches(a.numpy(),k.numpy())
    assert count_guard(['r0','r1','r2','r3'],k.numpy(),p)
    assert p[0].tolist()==[-1]*6
    loss=pitch_loss(net(wave,k),torch.zeros(4,49).scatter(
      1,torch.tensor([[0],[2],[3],[5]]),1))
    loss.backward()
    assert net.k.weight.grad is not None
    print('S416 SELFTEST PASS: no-YourMT CNN temporal MIDI K embedding and exact K0..K6 guard')
if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--self-test',action='store_true')
    for n in ('audio','s80','s402','annotations','truth','out'):
        q.add_argument('--'+n,type=Path)
    q.add_argument('--epochs',type=int,default=14)
    a=q.parse_args()
    if a.self_test:test()
    else:
        for n in ('audio','s80','s402','annotations','truth','out'):
            need(getattr(a,n) is not None,'Missing '+n)
        need(a.epochs>=1 and a.epochs<=30,'epochs outside certified range')
        run(a.audio,a.s80,a.s402,a.annotations,a.truth,a.out,epochs=a.epochs)
