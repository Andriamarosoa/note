"""S402 — K is a FIRST-CLASS INPUT of supervised MIDI pitch prediction.

Architecture: [original acoustic 42x49 history -> 294D event features,
               S80 predicted K -> 7D one-hot + K/6,
               acoustic x K interaction -> K-conditional feature block]
  -> train-only standardized shared MLP -> 49 simultaneous MIDI probabilities
  -> joint K-budget note selection (same K given at INPUT).

The note model is fit on JAMS note identities from OTHER COMPOSITIONS only,
with hash-based composition-disjoint 4 folds (all players of same composition
always in same fold). K input is the PREDICTED S80 K, NEVER true K.
Correctness is measured only after all fold-held note predictions are built.
This is a development experiment; S80 S397 itself has training overlap.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re
from pathlib import Path,PurePosixPath
import zipfile
import joblib,numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    MIDI_MIN,MIDI_MAX,PRE,ON,LATE,choose_pitches,note_name,sha)
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import (
    midi_value,read_note_events,scoped)
from scripts.yourmt3_exactk_common import assign_onsets

FOLDS=(0,1,2,3)
N=59309
D=49
SEED=402
def need(x,m):
    if not x:raise ValueError(m)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def composition_key(member):
    name=PurePosixPath(str(member)).name
    match=re.fullmatch(r'(\d{2})_(.+)\.jams',name)
    need(match is not None,'unknown GuitarSet composition pattern '+name)
    return match.group(2)
def composition_fold(key):
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16)%4
def feature_bank(seq,k):
    z=np.asarray(seq,np.float32);k=np.asarray(k,np.int64)
    need(z.ndim==3 and z.shape[1:]==(42,49)
         and k.shape==(len(z),) and np.isin(k,range(7)).all(),
         'S402 acoustic+S80_K JOINT input contract invalid')
    a=np.expm1(z.astype(np.float64))
    pre=a[:,PRE].mean(1);on=a[:,ON].mean(1);late=a[:,LATE].mean(1)
    pos=np.maximum(on-pre,0)
    flux=np.maximum(np.diff(a,axis=1),0).sum(axis=1)
    base=np.column_stack([
       np.log1p(pre),np.log1p(on),np.log1p(late),
       np.log1p(pos),pos/(on+pre+1e-4),
       np.log1p(flux)]).astype(np.float32)
    # Keep feature amplitudes comparable across recordings.
    # Fixed per-event global normalization has no label dependency.
    base[:,49:98] /= (np.max(base[:,49:98],axis=1,keepdims=True)+.05)
    base[:,147:196] /= (np.max(base[:,147:196],axis=1,keepdims=True)+.05)
    k_hot=np.eye(7,dtype=np.float32)[k]
    k_real=k[:,None].astype(np.float32)/6.0
    # Actual K-conditioned computations occur BEFORE the first learned
    # layer. The K-specific raw-activation interaction changes each pitch
    # feature at input; no top-K-only trick is possible.
    conditioned=np.log1p(on).astype(np.float32)*(1.0+k_real)
    features=np.column_stack((base,k_hot,k_real,conditioned)).astype(np.float32)
    need(features.shape==(len(z),352) and np.isfinite(features).all(),
         'S402 feature width or values drifted')
    return features
def labels(audio,jams,truth):
    members=audio['member'].astype(str)
    starts=audio['start_samples'].astype(np.int64)
    ends=audio['end_samples'].astype(np.int64)
    truek=truth['true_K'].astype(int)
    n=len(members)
    target=np.zeros((n,D),np.uint8)
    observed=np.zeros((n,6),np.int16)-1
    counts=np.zeros(n,np.int8)
    invalid=np.zeros(n,bool)
    total=unassigned=duplicates=out_range=0
    with zipfile.ZipFile(jams) as z:
        names=z.namelist()
        byname=collections.defaultdict(list)
        for name in names:
            if name.endswith('.jams'):
                byname[PurePosixPath(name).name].append(name)
        for member in sorted(set(members)):
            options=byname.get(PurePosixPath(member).name,[])
            need(len(options)==1,'missing/ambiguous JAMS member '+str(member))
            ix=np.flatnonzero(members==member)
            order=np.argsort(starts[ix],kind='stable')
            group=ix[order]
            ev=read_note_events(z.read(options[0]))
            total+=len(ev)
            _,assignment=assign_onsets(np.array([i[0] for i in ev],np.int64),starts[group],ends[group])
            byrow=collections.defaultdict(list)
            for (_,pitch,slot),j in zip(ev,assignment):
                if j<0:unassigned+=1;continue
                byrow[int(group[j])].append((slot,pitch))
            for i,items in byrow.items():
                items.sort()
                counts[i]=len(items)
                if len(items)>6:
                    invalid[i]=True;continue
                for j,(_,pitch) in enumerate(items):
                    observed[i,j]=pitch
                    if MIDI_MIN<=pitch<=MIDI_MAX:
                        target[i,pitch-MIDI_MIN]=1
                    else:invalid[i]=True;out_range+=1
                if len(set(p for _,p in items))<len(items):
                    duplicates+=1
    need(int((counts!=truek).sum())==0
         and int(invalid.sum())==0
         and int((truek>=2).sum())==7385,
         'original MIDI labels and K mismatch or unrepresentable source')
    return target,observed,dict(
        JAMS_note_events=total,unassigned_note_events=unassigned,
        original_K_annotation_drift=int((counts!=truek).sum()),
        duplicate_pitch_events=duplicates,out_of_MIDI_40_88=out_range)
def pair_score(truth,refK,base,new,mask):
    ix=np.flatnonzero(mask)
    exact_old=exact_new=old_tp=new_tp=truth_total=old_total=new_total=correct_K=0
    wins=losses=0
    for i in ix:
        target=collections.Counter(int(p) for p in truth[i] if p>=0)
        a=collections.Counter(int(p) for p in base[i] if p>=0)
        b=collections.Counter(int(p) for p in new[i] if p>=0)
        oc=target==a;nc=target==b
        exact_old+=int(oc);exact_new+=int(nc)
        wins+=int(nc and not oc);losses+=int(oc and not nc)
        old_tp+=sum((target&a).values());new_tp+=sum((target&b).values())
        truth_total+=sum(target.values())
        old_total+=sum(a.values());new_total+=sum(b.values())
        correct_K+=int(refK[i]==sum(target.values()))
    def f1(tp,npred,ngt):
        return 2*tp/(npred+ngt) if npred+ngt else 1.
    return dict(n=len(ix),S399_chord_correct=exact_old,
       S402_chord_correct=exact_new,
       S399_chord_rate=exact_old/len(ix) if len(ix) else None,
       S402_chord_rate=exact_new/len(ix) if len(ix) else None,
       S402_corrections=wins,S402_regressions=losses,
       S402_net=exact_new-exact_old,
       S399_midi_TP=old_tp,S402_midi_TP=new_tp,
       true_midi_notes=truth_total,
       S399_micro_F1=f1(old_tp,old_total,truth_total),
       S402_micro_F1=f1(new_tp,new_total,truth_total),
       unchanged_S80_ExactK_correct=correct_K)
def run(audio,s80,s399,annotations,truth,out):
    need(not out.exists(),'S402 experiment artifact cannot overwrite')
    data=load(audio);s=load(s80);b=load(s399);t=load(truth)
    ids=data['row_id'].astype(str);k=s['s80_K'].astype(np.int8)
    yK=t['true_K'].astype(int)
    old=b['midi_pitch_slots']
    need(len(ids)==N and len(set(ids))==N
      and np.array_equal(ids,s['row_id'].astype(str))
      and np.array_equal(ids,b['row_id'].astype(str))
      and np.array_equal(ids,t['row_id'].astype(str))
      and np.array_equal(k,b['s80_K'])
      and int((k==yK).sum())==53529
      and old.shape==(N,6),
      'S80 guarded baseline S397 / original S399 / S258 identities drift')
    member=data['member'].astype(str)
    compositions=np.array([composition_key(m) for m in member])
    folds=np.array([composition_fold(c) for c in compositions],dtype=np.int8)
    pos=k>0
    target,truth_midi,integrity=labels(data,annotations,t)
    x=np.empty((N,352),np.float32)
    x_ablation=np.empty((N,352),np.float32)
    for i in range(0,N,256):
        j=min(N,i+256)
        x[i:j]=feature_bank(data['sequence42x49'][i:j],k[i:j])
        x_ablation[i:j]=feature_bank(data['sequence42x49'][i:j],np.zeros(j-i,np.int8))
    out.mkdir(parents=True)
    new=np.full((N,6),-1,np.int16)
    selected_conf=np.zeros((N,6),np.float32)
    pred_abl=np.full((N,6),-1,np.int16)
    foldproof=[]
    for fold in FOLDS:
        mask=(folds==fold)
        training=(~mask)&pos
        need(training.sum()>=1000 and mask.sum()>=1000
          and not (set(compositions[training])&set(compositions[mask])),
          'S402 composition-disjoint train/test grouping failure')
        std=StandardScaler()
        xtrain=std.fit_transform(x[training])
        clf=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',
           solver='adam',batch_size=512,learning_rate_init=.001,
           alpha=.003,random_state=SEED+fold,
           max_iter=24,n_iter_no_change=6,
           early_stopping=False,tol=1e-4,shuffle=True,verbose=False)
        clf.fit(xtrain,target[training])
        # Conditioning K must alter the ACTUAL 49 pitch probabilities,
        # not only the final count.
        ix=np.flatnonzero(mask)
        new_prob=clf.predict_proba(std.transform(x[ix]))
        abl_prob=clf.predict_proba(std.transform(x_ablation[ix]))
        need(new_prob.shape==(len(ix),49) and
             abl_prob.shape==(len(ix),49) and np.isfinite(new_prob).all(),
             'S402 neural MIDI pitch probability width mismatch')
        p,conf=choose_pitches(new_prob,k[ix])
        ap,_=choose_pitches(abl_prob,k[ix])
        new[ix]=p;selected_conf[ix]=conf;pred_abl[ix]=ap
        fp=out/f'S402_pitch_head_held_song_fold{fold}.joblib'
        joblib.dump(dict(transform=std,network=clf,feature_width=352,
              S80_K_fed_to_model_input=True,training_positive_only=True,
              held_fold=fold),fp,compress=3)
        foldproof.append(dict(fold=fold,
            compositions_train=int(len(set(compositions[training]))),
            compositions_val=int(len(set(compositions[mask]))),
            composition_overlap=0,
            train_events=int(training.sum()),test_events=int(mask.sum()),
            train_true_K5=int(np.count_nonzero(yK[training]==5)),
            tested_true_K5=int(np.count_nonzero(yK[mask]==5)),
            model_epochs=int(clf.n_iter_),
            network_loss=float(clf.loss_),
            ablated_K_changes_pitch_predictions=int(np.count_nonzero(np.any(ap!=p,axis=1))),
            ablated_K_mean_abs_pitch_prob_delta=float(np.mean(np.abs(new_prob-abl_prob))),
            weights_sha256=sha(fp)))
    need(np.array_equal((new>=0).sum(1),k)
         and np.array_equal((pred_abl>=0).sum(1),k),
         'S402 must preserve active S80 exact-K on every event')
    npz=out/'S402_K_FIRST_S80_MIDI_NOTE_59309_SONG_OOF.npz'
    np.savez_compressed(npz,row_id=ids,s80_K=k,midi_pitch_slots=new,
         chosen_note_scores=selected_conf,
         composition_fold=folds)
    # Report correctness for all groups; no hyperparameter/threshold search.
    scopes={'all':np.ones(N,bool),'positive_K1_6':yK>=1,
       'poly_K2_6':yK>=2}
    scopes.update({f'K{p}':yK==p for p in range(7)})
    metrics={name:pair_score(truth_midi,k,old,new,m)
             for name,m in scopes.items()}
    ablation={name:pair_score(truth_midi,k,old,pred_abl,m)
              for name,m in scopes.items()}
    report=dict(status='S402_K_AS_DIRECT_PITCH_NETWORK_INPUT_SONG_CROSSFIT',
       reference='S80_S397_K5_GUARDED',
       note_head='S402_K_CONDITIONED_352D_MLP',
       original_events=N,
       composition_fold_count=4,
       source_label_uses_real_midi_note_JAMS=True,
       target_K_for_note_training='S80 predicted K only, NEVER true K',
       S80_K_at_start_of_pitch_prediction=True,
       S80_K_one_hot_width=7,
       K_interacts_with_acoustic_pitch_features_BEFORE_MLP=True,
       K_input_ablation_without_changing_final_budget=True,
       labels_not_exposed_to_same_composition_model=True,
       fold_proofs=foldproof,
       number_new_pitch_networks=4,
       source_note_annotations=integrity,
       metrics=metrics,
       ablation_K_zero_input_metrics=ablation,
       prediction_sha256=sha(npz),
       source_audio_sha256=sha(audio),
       reference_S80_sha256=sha(s80),
       original_S399_sha256=sha(s399),
       prior_GuitarSet_development_exposed=True,
       S80_reconstructed_reference_training_overlap=True,
       external_unseen_generalization_unproven=True,
       note_head_promoted=False,
       S80_reference_unchanged=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('S402 PROOFS',json.dumps(foldproof),flush=True)
    print('S402 PAIR SCORES',json.dumps(metrics),flush=True)
    print('S402 NO-K ablation',json.dumps(ablation),flush=True)
    print('S402 PASS label-supervised K-first neural note predictions song-held across 59309; no reference mutation',flush=True)
def selftest():
    z=np.zeros((3,42,49),np.float32)
    z[:,ON,2]=np.log1p(3)
    k=np.array([0,1,5],np.int8)
    x=feature_bank(z,k)
    x2=feature_bank(z,np.zeros(3,np.int8))
    need(x.shape==(3,352) and not np.array_equal(x,x2),
         'K MUST ENTER NEURAL PITCH FEATURES before notes, no post-only K')
    need(not np.array_equal(x[0],x[1]),
         'Same audio with K0 vs K1 MUST have different predictor inputs')
    need(composition_key('00_BN1-129-E_comp.jams')==
      composition_key('04_BN1-129-E_comp.jams') and
      composition_fold(composition_key('00_BN1-129-E_comp.jams'))==
      composition_fold(composition_key('04_BN1-129-E_comp.jams')),
      'same-song across players disjoint grouping required')
    print('S402 SELFTEST PASS: K feeds predictor features at input; same audio different K => different features')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for nm in ('audio','s80','s399','annotations','truth','out'):
        p.add_argument('--'+nm,type=Path)
    u=p.parse_args()
    if u.self_test:selftest()
    else:
        for nm in ('audio','s80','s399','annotations','truth','out'):
            need(getattr(u,nm)is not None,'missing '+nm)
        run(u.audio,u.s80,u.s399,u.annotations,u.truth,u.out)
