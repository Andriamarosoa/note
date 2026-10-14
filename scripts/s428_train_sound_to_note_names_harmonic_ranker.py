"""S428 — train own acoustic/harmonic MIDI pitch ranker, S80 K locked.

Task: detected source event -> one or more NOTE NAMES (E3, G3, B3).
No note durations, no MIDI-file requirement, no Live modifications.
Uses S80 predicted K and the real 42x49 onset-resolved harmonic
source. Baseline is frozen S419. The four new note-pitch rankers train
on OTHER guitar compositions; any inner policy choices are made on
other compositions only. True K never affects predictions.

Limitations: S80's original GuitarSet development overlap remains, and
cross-fitted S402/S416 predictions used as auxiliary features are not
fully nested inside this outer gate. This is development, NOT an
independent blind benchmark. Never silently promote original references.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    load,labels,composition_key,composition_fold,need)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    PRE,ON,LATE,choose_pitches,note_name,sha)

N=59309
EXACT_GLOBAL=53529
EXACT_POLY=6054
POLY=7385
MIDI0=40
DIM=49
SEED=428
FEATURES=28

def shift(x,steps):
    """Never wrap MIDI range; shifted acoustic energy outside 40..88 is zero."""
    ret=np.zeros_like(x)
    if steps>0:ret[:,steps:]=x[:,:-steps]
    elif steps<0:ret[:,:steps]=x[:,-steps:]
    else:ret[:]=x
    return ret

def build_features(seq,k,s416p,base,s402):
    """Ground truth NEVER enters candidate features. 49 pitches per event."""
    n=len(k)
    need(seq.shape==(n,42,49) and s416p.shape==(n,49)
         and base.shape==(n,6) and s402.shape==(n,6)
         and np.isfinite(seq).all() and np.min(seq)>=0,
         "S428 need original 42x49 harmonic trajectories and frozen scores")
    a=np.expm1(seq.astype(np.float32))
    pre=np.mean(a[:,PRE],axis=1);attack=np.mean(a[:,ON],axis=1)
    tail=np.mean(a[:,LATE],axis=1)
    growing=np.maximum(attack-pre,0)
    downward=np.maximum(pre-attack,0)
    flux=np.maximum(np.diff(a,axis=1),0).sum(1)
    denom=attack+pre+.01
    norm=attack/(np.max(attack,axis=1,keepdims=True)+.01)
    prob=np.asarray(s416p,np.float32)
    bflag=np.zeros((n,49),np.float32)
    sflag=np.zeros_like(bflag)
    for j in range(6):
        keep=base[:,j]>=40
        bflag[np.flatnonzero(keep),base[keep,j]-40]=1
        keep=s402[:,j]>=40
        sflag[np.flatnonzero(keep),s402[keep,j]-40]=1
    # Spectral morphology and music-physics: octave harmonic links,
    # fifth, neighboring semitone and 19-semitone harmonic.
    # MIDI/pitch position remains available to the classifier but
    # no pitch class, musical key or actual labels are used.
    harmonic12=shift(attack,12)
    harmonic19=shift(attack,19)
    up12=shift(attack,-12)
    up19=shift(attack,-19)
    neighboring=np.maximum(shift(attack,-1),shift(attack,1))
    fifth=np.maximum(shift(attack,7),shift(attack,-7))
    high=shift(prob,-12)
    low=shift(prob,12)
    feats=np.stack([
      np.log1p(pre),np.log1p(attack),np.log1p(tail),
      np.log1p(growing),np.log1p(downward),np.log1p(flux),
      growing/denom,tail/(attack+.05),norm,
      np.log1p(harmonic12),np.log1p(harmonic19),
      np.log1p(up12),np.log1p(up19),
      np.log1p(neighboring),np.log1p(fifth),
      (attack-harmonic12)/(attack+harmonic12+.05),
      (attack-up12)/(attack+up12+.05),
      (attack-neighboring)/(attack+neighboring+.05),
      prob, high,low,
      bflag,sflag,
      np.broadcast_to(k[:,None]/6.,attack.shape),
      np.broadcast_to(np.arange(49,dtype=np.float32)[None,:]/49.,attack.shape),
      (prob>np.max(prob,axis=1,keepdims=True)*.55).astype(np.float32),
      (attack>np.max(attack,axis=1,keepdims=True)*.32).astype(np.float32),
      (growing>np.max(growing,axis=1,keepdims=True)*.40).astype(np.float32)
    ],axis=-1).astype(np.float32)
    need(feats.shape==(n,49,FEATURES)
         and np.isfinite(feats).all(),"S428 physical feature shape invalid")
    return feats

def exact(a,b):
    return bool(np.array_equal(np.sort(a[a>=0]),np.sort(b[b>=0])))

def accuracy_rows(target,old,new,ix):
    wins=losses=oldhits=newhits=0
    tp_old=tp_new=gt_count=0
    for z in ix:
        a=collections.Counter(int(x) for x in old[z] if x>=0)
        b=collections.Counter(int(x) for x in new[z] if x>=0)
        t=collections.Counter(int(x) for x in target[z] if x>=0)
        aa=a==t;bb=b==t
        oldhits+=aa;newhits+=bb;wins+=bb and not aa;losses+=aa and not bb
        tp_old+=sum((a&t).values());tp_new+=sum((b&t).values())
        gt_count+=sum(t.values())
    return dict(n=len(ix),S419_chord_correct=int(oldhits),
       S428_chord_correct=int(newhits),
       corrections=int(wins),regressions=int(losses),
       net=int(newhits-oldhits),
       baseline_note_TP=int(tp_old),candidate_note_TP=int(tp_new),
       actual_note_instances=int(gt_count))

def train_model(feats,prob,b_flag,truth,train_event_indices,seed):
    """Only target notes from training compositions may affect a fit."""
    batches=[];outcome=[]
    for x in np.array_split(train_event_indices,max(1,int(np.ceil(len(train_event_indices)/512)))):
        if not len(x):continue
        q=prob[x]
        # Acoustic candidate set is ALWAYS label blind. Add train true
        # notes as supervised positive examples, never to validation.
        acoustic=feats[x,:,1]
        t=np.zeros((len(x),49),bool)
        for j in range(6):
            p=truth[x,j]
            ok=p>=40
            t[np.flatnonzero(ok),p[ok]-40]=True
        idx=np.arange(49)[None,:]
        ptop=np.argsort(-q,axis=1,kind="stable")[:,:12]
        at=np.argsort(-acoustic,axis=1,kind="stable")[:,:8]
        choose=np.zeros((len(x),49),bool)
        choose[np.arange(len(x))[:,None],ptop]=True
        choose[np.arange(len(x))[:,None],at]=True
        choose|=b_flag[x,:,21]>0
        choose|=b_flag[x,:,22]>0
        choose|=t
        candidate=feats[x][choose]
        label=t[choose].astype(np.uint8)
        batches.append(candidate);outcome.append(label)
    X=np.concatenate(batches)
    y=np.concatenate(outcome)
    need(len(y)>25000 and int(y.sum())>3000
        and int((1-y).sum())>20000,
        "S428 insufficient labeled pitch positives/negatives")
    model=HistGradientBoostingClassifier(
        learning_rate=.065,max_iter=95,max_leaf_nodes=20,
        min_samples_leaf=70,l2_regularization=7.,
        early_stopping=False,random_state=seed)
    model.fit(X,y,sample_weight=np.where(y==1,3.,1.))
    return model,dict(training_pitch_rows=int(len(y)),
         positives=int(y.sum()),negatives=int(len(y)-y.sum()))

def rank_predict(model,X,k,ix):
    predictions=np.full((len(ix),6),-1,np.int16)
    for off in range(0,len(ix),384):
        batch=X[ix[off:off+384]]
        raw=model.predict_proba(batch.reshape(-1,FEATURES))[:,1].reshape(-1,49)
        pred,_=choose_pitches(raw,k[ix[off:off+384]])
        predictions[off:off+384]=pred
    return predictions

def run(audio,reference,original_s402,s416,prior_s419,annotations,truth,out):
    need(not out.exists(),"S428 must not replace any historical result")
    a=load(audio);r=load(reference);s=load(original_s402)
    net=load(s416);base=load(prior_s419);t=load(truth)
    ids=a["row_id"].astype(str)
    K=r["s80_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
        and all(np.array_equal(ids,z["row_id"].astype(str))
                for z in (r,s,net,base,t))
        and all(np.array_equal(K,z["s80_K"]) for z in (s,net,base))
        and np.array_equal((base["midi_pitch_slots"]>=0).sum(1),K)
        and int((K==t["true_K"]).sum())==EXACT_GLOBAL
        and int(((K==t["true_K"])&(t["true_K"]>=2)).sum())==EXACT_POLY,
        "S428 source ID or S80 K0..K6 reference drift")
    prob=net["note_probabilities_float16"].astype(np.float32)
    baseline=base["midi_pitch_slots"].astype(np.int16)
    positive=np.flatnonzero(K>0)
    features=build_features(a["sequence42x49"][positive],K[positive],
           prob[positive],baseline[positive],s["midi_pitch_slots"][positive])
    row_to_pos=np.full(N,-1,np.int32)
    row_to_pos[positive]=np.arange(len(positive))
    names=a["member"].astype(str)
    comp=np.array([composition_key(m) for m in names])
    fold=np.array([composition_fold(c) for c in comp],np.int8)
    target_multi,targets,meta=labels(a,annotations,t)
    need(targets.shape==(N,6) and int((t["true_K"]>=2).sum())==POLY,
         "S428 original GuitarSet pitch annotation mismatch")

    out.mkdir(parents=True)
    pred=baseline.copy()
    folds=[]
    for f in range(4):
        train=(fold!=f)&(K>0)
        held=(fold==f)&(K>0)
        val_comp=np.array([
            int(hashlib.sha256(("S428-inner-"+v).encode()).hexdigest()[:8],16)%5==0
            for v in comp],bool)
        inner_train=train&~val_comp
        inner_val=train&val_comp
        need(inner_train.sum()>8000 and inner_val.sum()>1000
           and not(set(comp[train])&set(comp[fold==f]))
           and not(set(comp[inner_train])&set(comp[inner_val])),
           "S428 song split leak")
        inner_model,innercounts=train_model(
             features,prob[positive],features,targets,
             row_to_pos[np.flatnonzero(inner_train)],SEED+f)
        val_ix=np.flatnonzero(inner_val)
        val_est=rank_predict(inner_model,features,K[positive],row_to_pos[val_ix])
        # Predeclared selection per predicted K based only OTHER songs.
        gate={}
        for kk in range(1,7):
            v=val_ix[K[val_ix]==kk]
            if not len(v):gate[str(kk)]=dict(active=False,n=0,net=0);continue
            g=accuracy_rows(targets,baseline,
                dict_placeholder(val_ix,val_est,N),v) if False else None
            # Intentionally evaluate ONLY the inner validation subset.
            old_count=sum(exact(baseline[i],targets[i]) for i in v)
            new_count=sum(exact(val_est[np.searchsorted(val_ix,i)],targets[i]) for i in v)
            net_gain=new_count-old_count
            activate=bool(len(v)>=55 and net_gain>=3)
            gate[str(kk)]=dict(active=activate,n=len(v),
                old_exact=int(old_count),new_exact=int(new_count),
                net=int(net_gain),training_only=True)
        final_model,fullcounts=train_model(
            features,prob[positive],features,targets,
            row_to_pos[np.flatnonzero(train)],SEED+40+f)
        held_ix=np.flatnonzero(held)
        held_est=rank_predict(final_model,features,K[positive],row_to_pos[held_ix])
        for kk in range(1,7):
            if gate[str(kk)]["active"]:
                mask=K[held_ix]==kk
                pred[held_ix[mask]]=held_est[mask]
        fn=out/f"S428_acoustic_harmonic_pitch_ranker_fold{f}.joblib"
        joblib.dump(dict(model=final_model,feature_width=FEATURES,
            MIDI_min=MIDI0,original_S80_K_required=True,
            no_YourMT3_at_train_or_test=True,
            held_composition_fold=f),fn,compress=3)
        folds.append(dict(held_fold=f,inner_fit=innercounts,
            full_fit=fullcounts,
            train_songs=len(set(comp[train])),
            held_songs=len(set(comp[fold==f])),
            train_test_song_overlap=0,inner_song_overlap=0,
            note_head_gate=gate,
            model_SHA256=sha(fn)))
        print("S428 FOLD",json.dumps(folds[-1]),flush=True)
    need(np.array_equal((pred>=0).sum(1),K)
         and int((pred>=0).sum())==int(K.sum())==33475,
         "S428 CRITICAL per-row Exact-K regression")
    # Actual exported note-name file: 'son -> E3, G3, B3'.
    filename=out/"S428_NOTE_NAMES_per_sound_event.jsonl"
    with filename.open("w",encoding="utf8") as h:
        for row,k,notes in zip(ids,K,pred):
            h.write(json.dumps(dict(source_row_id=str(row),K=int(k),
                notes=[note_name(int(m)) for m in notes if m>=0],
                pitches=[int(m) for m in notes if m>=0]))+"\n")
    # Independently re-read actual output, compare each K even for K0.
    amount=0
    with filename.open(encoding="utf8") as h:
        for i,line in enumerate(h):
            record=json.loads(line)
            note_ids=record["pitches"]
            need(record["source_row_id"]==ids[i] and
                record["K"]==int(K[i]) and
                len(note_ids)==int(K[i]) and
                len(record["notes"])==int(K[i]) and
                all(note_name(int(m))==n for m,n in zip(note_ids,record["notes"])),
                "S428 exported note-name list deviates from S80 exact K")
            amount+=1
    need(amount==N,"S428 missing exported events")
    poly_ix=np.flatnonzero(t["true_K"]>=2)
    prior_report=accuracy_rows(targets,baseline,pred,poly_ix)
    need(prior_report["S419_chord_correct"]==2775 and
         prior_report["n"]==POLY,
         "S428 S419 pitch baseline not independently reproduced")
    byK={}
    for kk in range(7):
        byK[str(kk)]=accuracy_rows(targets,baseline,pred,
                    np.flatnonzero(t["true_K"]==kk))
    named=out/"S428_NOTE_NAMES_S80_K_LOCKED_59309.npz"
    np.savez_compressed(named,row_id=ids,s80_K=K,
       midi_pitch_slots=pred,composition_fold=fold)
    result=dict(status="S428_ACOUSTIC_HARMONIC_NOTE_NAMES_SONGHELD_K_LOCKED",
      input_is_real_harmonic_sound_42x49=True,
      output_is_musical_NOTE_NAMES_not_a_MIDI_file=True,
      no_YourMT3_at_training_or_inference=True,
      no_True_K_at_inference=True,
      full_S80_59309_exactK_predictions_unmodified=True,
      original_S80_correct_global=53529,candidate_S80_correct_global=53529,
      original_S80_correct_poly=6054,candidate_S80_correct_poly=6054,
      exactK_regressions=0,
      original_s419_chord_correct=2775,
      S428_chord_correct=prior_report["S428_chord_correct"],
      chord_comparison_poly_true_K2_to_K6=prior_report,
      by_true_K=byK,training_folds=folds,
      labels_composition_held_for_new_ranker=True,
      S402_S416_upstream_models_not_fully_nested=True,
      old_S80_has_GuitarSet_training_overlap=True,
      original_active_references_and_Live_UNTOUCHED=True,
      independent_outside_GuitarSet_validation=False,
      promoted=False,
      output_notes_hash=sha(filename),output_npz_hash=sha(named),
      source_hashes=dict(original_S80=sha(reference),
        original_audio=sha(audio),S419=sha(prior_s419),
        S416=sha(s416),annotation_JAMS=sha(annotations)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S428 PITCH RESULT",json.dumps(prior_report),flush=True)
    print("S428 PITCH BY TRUE K",json.dumps(byK),flush=True)
    print("S428 PASS all 59309 original S80 K unchanged from exported musical note names",flush=True)

def selftest():
    x=np.zeros((2,42,49),np.float32)
    k=np.array([0,3],np.int8)
    src=np.zeros((2,6),np.int16)-1
    src[1,:3]=[52,55,59]
    x[:,ON,12]=np.log1p(5.)
    q=np.random.default_rng(SEED).random((2,49)).astype(np.float32)
    z=build_features(x,k,q,src,src)
    need(z.shape==(2,49,FEATURES) and np.isfinite(z).all(),
         "S428 harmonic fundamental/overtone features malformed")
    pred,_=choose_pitches(q,k)
    need(sum(pred[0]>=0)==0 and sum(pred[1]>=0)==3
         and note_name(52)=="E3" and note_name(55)=="G3"
         and note_name(59)=="B3",
         "S428 musical note names or K budget failed")
    print("S428 SELFTEST PASS: raw harmonic source -> E3 G3 B3, exactly S80 K0/3")
if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for arg in ("audio","s80","s402","s416","s419","truth","annotations","out"):
        q.add_argument("--"+arg,type=Path)
    p=q.parse_args()
    if p.self_test:selftest()
    else:
        for field in ("audio","s80","s402","s416","s419","truth","annotations","out"):
            need(getattr(p,field) is not None,"missing --"+field)
        run(p.audio,p.s80,p.s402,p.s416,p.s419,p.annotations,p.truth,p.out)
