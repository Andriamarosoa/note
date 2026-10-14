"""S431 — native octave-disambiguation note names, ZERO changes to S80 K.

Input is precomputed sound event acoustic 42x49 and frozen S428 musical
pitch outputs, not MIDI file or Live. For each *predicted K2/K3* event,
try ONE octave ±12 swap of ONE selected note, preserving number of pitches.
Model and optional activation threshold learned on OTHER guitar
compositions; disable entirely when inner-train compositions show
no robust gain. No ground-truth labels or true K enter held inference.

S80 and S428 upstream GuitarSet exposure -> DEVELOPMENT ONLY, not
independent end-to-end proof. S428 frozen baseline is never overwritten.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    load,labels,composition_key,composition_fold,need)
from scripts.s428_train_sound_to_note_names_harmonic_ranker import (
    build_features,FEATURES,accuracy_rows,exact)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    note_name,sha)

N=59309
BASE_POLY=3025
REAL_POLY=7385
RATE_GLOBAL=53529
RATE_POLY=6054
MIDI_MIN=40
MIDI_MAX=88
SEED=431
WIDTH=2*FEATURES+9
THRESHOLDS=(.60,.70,.80,.90,.95,.975,.99)

def candidate_bank(sources,notes,K,s416,head419,head402):
    """Create only octave alternatives, never use GT or labels."""
    sel=np.flatnonzero(np.isin(K,[2,3]))
    features=build_features(sources[sel],K[sel],s416[sel],
                            head419[sel],head402[sel])
    cand_source=[];cand_slot=[];cand_newpitch=[];X=[]
    for loc,idx in enumerate(sel):
        old=notes[idx]
        members=[int(p) for p in old if p>=0]
        assert len(members)==int(K[idx])
        for slot,pitch in enumerate(members):
            for delta in (-12,12):
                substitute=pitch+delta
                if substitute<MIDI_MIN or substitute>MIDI_MAX:
                    continue
                if substitute in members:
                    continue
                src=features[loc,pitch-MIDI_MIN]
                dst=features[loc,substitute-MIDI_MIN]
                hints=np.asarray([
                  K[idx]/6.,
                  delta/12.,
                  pitch/88.,
                  substitute/88.,
                  float(src[18]-dst[18]),
                  float(src[3]-dst[3]),
                  float(src[8]-dst[8]),
                  float(src[15]-dst[15]),
                  float(src[22]-dst[22])
                ],np.float32)
                X.append(np.concatenate([src,dst,hints]))
                cand_source.append(int(idx))
                cand_slot.append(slot)
                cand_newpitch.append(substitute)
    need(bool(X),"S431 no acoustic octave alternatives built")
    arr=np.asarray(X,np.float32)
    need(arr.shape==(len(cand_source),WIDTH) and np.isfinite(arr).all(),
         "S431 octave candidate acoustic feature failure")
    return (np.asarray(cand_source,np.int32),
            np.asarray(cand_slot,np.int8),
            np.asarray(cand_newpitch,np.int16),arr)

def candidate_chord(notes,slot,alt):
    before=notes.copy()
    before[slot]=alt
    return before

def choose_model(X,old,candidate_source,slots,pitches,truth,train_events,seed):
    """Discordant training only: true correction vs true regression."""
    in_train=train_events[candidate_source]
    which=np.flatnonzero(in_train)
    chosen=[];y=[]
    for j in which:
        i=int(candidate_source[j])
        prior=exact(old[i],truth[i])
        after=exact(candidate_chord(old[i],int(slots[j]),int(pitches[j])),truth[i])
        if prior==after:
            continue
        chosen.append(int(j));y.append(int(after))
    y=np.asarray(y,np.uint8)
    need(len(y)>180 and int(y.sum())>=18 and int((y==0).sum())>=80,
         "S431 octave corrections/regressions too rare for fit")
    model=HistGradientBoostingClassifier(
        max_iter=90,max_leaf_nodes=10,min_samples_leaf=15,
        learning_rate=.07,l2_regularization=5.,
        early_stopping=False,random_state=seed)
    model.fit(X[chosen],y,sample_weight=np.where(y==1,2.,1.))
    return model,dict(training_discordant=len(y),
            training_corrections=int(y.sum()),training_regressions=int((y==0).sum()))

def apply(model,X,cand_source,slot,pitch,orig,eligible,threshold):
    """At most ONE per-event octave substitution, exact same K."""
    output=orig.copy()
    ix=np.flatnonzero(eligible[cand_source])
    if len(ix)==0:
        return output,0
    probs=np.zeros(len(cand_source),np.float32)
    for start in range(0,len(ix),8192):
        j=ix[start:start+8192]
        probs[j]=model.predict_proba(X[j])[:,1]
    pick={}
    for j in ix:
        i=int(cand_source[j])
        p=float(probs[j])
        if p<threshold:
            continue
        if i not in pick or p>pick[i][0]:
            pick[i]=(p,int(j))
    for i,(_,j) in pick.items():
        output[i,int(slot[j])]=int(pitch[j])
    return output,len(pick)

def run(audio,s80,s402,s416,s419,s428,annotations,truth,out):
    need(not out.exists(),"S431 may not overwrite frozen S428 evidence")
    a=load(audio);r=load(s80);b=load(s428);t=load(truth)
    h419=load(s419);h402=load(s402);h416=load(s416)
    ids=a["row_id"].astype(str);K=r["s80_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and all(np.array_equal(ids,z["row_id"].astype(str))
                 for z in (r,b,t,h419,h402,h416))
         and np.array_equal((b["midi_pitch_slots"]>=0).sum(1),K)
         and int((K==t["true_K"]).sum())==RATE_GLOBAL
         and int(((K==t["true_K"])&(t["true_K"]>=2)).sum())==RATE_POLY,
         "S431 S80 or S428 frozen event source has changed")
    old=b["midi_pitch_slots"].astype(np.int16)
    seq=a["sequence42x49"]
    probabilities=h416["note_probabilities_float16"].astype(np.float32)
    first,slot,alternative,X=candidate_bank(
          seq,old,K,probabilities,h419["midi_pitch_slots"],
          h402["midi_pitch_slots"])
    members=a["member"].astype(str)
    comps=np.array([composition_key(x) for x in members])
    folds=np.array([composition_fold(x) for x in comps],np.int8)
    # All candidates are built without labels. Annotations are now read
    # exclusively for TRAIN OTHER COMPOSITIONS and final scoring.
    _,gt,meta=labels(a,annotations,t)
    need(int((t["true_K"]>=2).sum())==REAL_POLY,
         "S431 original GT poly denominator changed")
    proposed=old.copy()
    out.mkdir(parents=True)
    reports=[]
    for f in range(4):
        train=folds!=f;held=folds==f
        inner_marker=np.array([
            int(hashlib.sha256(("S431-inner-"+x).encode()).hexdigest()[:8],16)%5==0
            for x in comps],bool)
        inner_val=train&inner_marker&np.isin(K,[2,3])
        inner_train=train&~inner_marker&np.isin(K,[2,3])
        need(not (set(comps[train])&set(comps[held]))
             and not (set(comps[inner_val])&set(comps[inner_train]))
             and int(inner_val.sum())>100,
             "S431 composition holdout violated")
        enabled=False;threshold=None;target_net=0;held_changed=0
        try:
            cal,counts=choose_model(X,old,first,slot,alternative,
                      gt,inner_train,SEED+f)
        except ValueError as e:
            reports.append(dict(fold=f,training_skipped=str(e),
                                activated=False,held_changed=0,
                                composition_overlap=0))
            continue
        ratings=[]
        val_ix=np.flatnonzero(inner_val)
        for th in THRESHOLDS:
            val_pred,changed=apply(cal,X,first,slot,alternative,
                                  old,inner_val,th)
            count_before=sum(exact(old[i],gt[i]) for i in val_ix)
            count_after=sum(exact(val_pred[i],gt[i]) for i in val_ix)
            ratings.append(dict(threshold=th,
                val_old_exact=int(count_before),
                val_new_exact=int(count_after),
                val_net=int(count_after-count_before),
                val_changed=int(changed)))
        best=max(ratings,key=lambda v:(v["val_net"],v["threshold"]))
        target_net=int(best["val_net"])
        enabled=bool(target_net>=3)
        checkpoint=None
        if enabled:
            final,stats=choose_model(X,old,first,slot,alternative,
                        gt,train&np.isin(K,[2,3]),SEED+40+f)
            held_output,held_changed=apply(
                final,X,first,slot,alternative,
                old,held,best["threshold"])
            proposed[held]=held_output[held]
            checkpoint=out/f"S431_octave_corrector_fold{f}.joblib"
            joblib.dump(dict(model=final,held_fold=f,
                source_notes="S428_frozen",K_S80_locked=True,
                feature_width=WIDTH,no_YourMT_teacher=True,
                chosen_threshold=best["threshold"]),checkpoint,compress=3)
            threshold=float(best["threshold"])
        reports.append(dict(fold=f,train_compositions=len(set(comps[train])),
            held_compositions=len(set(comps[held])),composition_overlap=0,
            inner_training_discordant=counts,
            inner_validation_results=ratings,activated=enabled,
            threshold=threshold,inner_gain=target_net,
            held_changed=int(held_changed),
            trained_model_sha256=sha(checkpoint) if checkpoint else None))
        print("S431 FOLD",json.dumps(reports[-1]),flush=True)
    need(np.array_equal((proposed>=0).sum(1),K)
         and np.all(proposed[K==0]==-1)
         and int(np.sum(proposed>=0))==int(K.sum())==33475,
         "S431 CRITICAL: output cardinality not equal S80 for all rows")
    namepath=out/"S431_NOTE_NAMES_E3_FROM_AUDIO_59309.jsonl"
    with namepath.open("w",encoding="utf8") as h:
        for rid,k,ps in zip(ids,K,proposed):
            mids=[int(x) for x in ps if x>=0]
            h.write(json.dumps(dict(source_row_id=str(rid),K=int(k),
                notes=[note_name(p) for p in mids],
                pitches=mids))+"\n")
    with namepath.open(encoding="utf8") as h:
        count=0
        for i,line in enumerate(h):
            v=json.loads(line);m=v["pitches"]
            need(v["source_row_id"]==ids[i] and v["K"]==int(K[i])
               and len(m)==int(K[i])
               and v["notes"]==[note_name(p) for p in m],
               "S431 independent JSONL musical output violates S80 K")
            count+=1
    need(count==N,"S431 missing an original sound event")
    ix=np.flatnonzero(t["true_K"]>=2)
    comparison=accuracy_rows(gt,old,proposed,ix)
    need(comparison["S419_chord_correct"]==BASE_POLY,
         "S431 original S428 reference score not replicated")
    by_true_K={str(k):accuracy_rows(gt,old,proposed,
             np.flatnonzero(t["true_K"]==k)) for k in range(7)}
    npz=out/"S431_NOTE_NAMES_K_LOCKED_59309.npz"
    np.savez_compressed(npz,row_id=ids,s80_K=K,
                       midi_pitch_slots=proposed,composition_fold=folds)
    result=dict(status="S431_OCTAVE_DISAMBIGUATION_NOTE_NAMES_K_LOCKED",
        benchmark="GuitarSet_59309_original_native_events",
        baseline="S428_ACOUSTIC_HARMONIC_NOTE_NAMES_SONGHELD_K_LOCKED",
        original_baseline_poly_chords=BASE_POLY,
        new_chords_poly=comparison["S428_chord_correct"],
        true_poly_events=REAL_POLY,
        relative_gain_poly=comparison["net"],
        corrections_vs_S428=comparison["corrections"],
        regressions_vs_S428=comparison["regressions"],
        by_true_K=by_true_K,fold_train_only_policies=reports,
        original_S80_total_K_correct=53529,new_S80_total_K_correct=53529,
        original_S80_poly_K_correct=6054,new_S80_poly_K_correct=6054,
        exported_note_names_replayed=59309,
        no_K_regressions=True,
        no_note_duration_or_Live_changed=True,
        no_YourMT_teacher_or_prediction_used=True,
        frozen_S428_preserved=True,
        original_S80_S402_S408_references_unchanged=True,
        no_external_independent_whole_chain_validation=True,
        original_S80_GuitarSet_overlap=True,
        S428_upstream_head_not_fully_nested=True,
        not_officially_promoted=True,
        outputs=dict(notes_SHA256=sha(namepath),
                     NPZ_SHA256=sha(npz)),
        inputs=dict(audio=sha(audio),S80=sha(s80),S428=sha(s428),
                    JAMS=sha(annotations)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S431 OCTAVE RESULT",json.dumps(comparison),flush=True)
    print("S431 BY K",json.dumps(by_true_K),flush=True)
    print("S431 PASS S80 exactK 0 regression from actual exported E3 names",flush=True)

def test():
    old=np.array([52,55,-1,-1,-1,-1],np.int16)
    after=candidate_chord(old,0,64)
    need(note_name(old[0])=="E3" and note_name(after[0])=="E4"
         and len(after[after>=0])==2 and not exact(old,after),
         "S431 octave candidate must replace, never change K")
    note=np.array([[52,55,-1,-1,-1,-1],
                   [52,64,-1,-1,-1,-1]],np.int16)
    q=np.array([[.6]*WIDTH,[.2]*WIDTH],np.float32)
    class Fake:
        def predict_proba(self,x):
            p=np.full(len(x),.98)
            return np.column_stack([1-p,p])
    chosen,n=apply(Fake(),q,np.array([0,1]),np.array([0,0]),
                np.array([64,76]),note,np.array([True,False]),.95)
    need(n==1 and chosen[0,0]==64 and np.array_equal(chosen[1],note[1]),
         "S431 cannot touch held K class/overwrite excluded rows")
    print("S431 SELFTEST PASS: E3/E4 octave swap exactly preserves S80 K")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for name in ("audio","s80","s402","s416","s419","s428",
                 "annotations","truth","out"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.self_test:test()
    else:
        for name in ("audio","s80","s402","s416","s419","s428",
                     "annotations","truth","out"):
            need(getattr(a,name)is not None,"missing --"+name)
        run(a.audio,a.s80,a.s402,a.s416,a.s419,a.s428,
            a.annotations,a.truth,a.out)
