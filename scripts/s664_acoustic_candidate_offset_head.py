"""S664 — acoustic evidence head for NoteOff candidate ranking, GuitarSet.

Combine frozen S425 candidate NoteOff predictions with S411 real microphone
STFT trajectories (10ms frames, available historically as 59D trajectory
summaries per NOTE predicted note). Match audio row by source_row_id+MIDI,
not by index alone (S408 vs S420 can change pitch). Candidate-specific
acoustics are selected ONLY from horizons <= proposed NoteOff age; avoid
using 100ms+ acoustic if a candidate would end before that age.

Research validation: composition-disjoint four folds; real GuitarSet 190,
no ground truth or YourMT at held-song inference. Original 59,309 K, 33,475
MIDI pitches/onsets, original S425 output and Live all unchanged.
This is NOT guaranteed causal Live: original S411 microphone STFT is centered,
and S425 upstream NoteOff can use up to 5 seconds future audio. Do not deploy.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.s425_songheld_select_S422_S420_offsets_no_K_regression import (
    read,labels_for_other_songs_only,need,SR,N_MATCHED,
)
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,composition_fold,
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s662_80pct_offset_feasibility_and_residual import (
    evidence,candidates,build_features,stats,TOL,
)
from scripts.s663_candidate_offset_ranker import (
    PENALTIES, get_matrix,candidate_matrix_features,pick_from_prob,
)

AGES=np.asarray((100,200,350,500,750,1000,1400,1800,2500,3500),np.float32)
ACOUSTIC_DIM=59


def align_audio(p,source):
    """Join S411 -> S425 without labels. Repeated identical MIDI pitch in a
    source row gets its next available S411 occurrence, deterministically.
    """
    s=read(source)
    need(len(s["source_row_id"])==len(p["source_row_id"]) and
         s["note_features"].shape==(len(p["MIDI"]),ACOUSTIC_DIM),
         "S664 S411 source incompatible size / feature schema")
    matches=collections.defaultdict(list)
    for j,(row,pitch) in enumerate(zip(
        s["source_row_id"].astype(str),s["notes_midi"].astype(int))):
        matches[(row,int(pitch))].append(j)
    used=collections.Counter()
    feats=np.zeros((len(p["MIDI"]),ACOUSTIC_DIM),np.float32)
    known=np.zeros(len(p["MIDI"]),bool)
    for i,(row,pitch) in enumerate(zip(
        p["source_row_id"].astype(str),p["MIDI"].astype(int))):
        key=(row,int(pitch))
        take=used[key]
        if take>=len(matches[key]):
            continue
        j=matches[key][take]
        used[key]+=1
        if not bool(s["valid_note_mask"][j]):
            continue
        feats[i]=s["note_features"][j]
        known[i]=True
    need(known.sum()>30000 and np.isfinite(feats).all(),
         "S664 pitch matched S411 acoustics missing unexpectedly")
    return feats,known


def acoustic_candidate_features(acoustics,known,C,on,indices):
    """Only past-relative age horizons chosen for the candidate.
    This does NOT erase S411 center-window lookahead or S425 lookahead.
    """
    n=len(indices);m=C.shape[1]
    a=acoustics[indices]
    shaped=a[:,9:].reshape(n,10,5)
    duration_ms=np.maximum(1,(C[indices]-on[indices,None])/SR*1000.)
    at=np.searchsorted(AGES,duration_ms,side="right")-1
    eligible=at>=0
    at=np.clip(at,0,len(AGES)-1)
    prev=np.maximum(at-1,0)
    rows=np.arange(n)[:,None]
    current=shaped[rows,at]
    previous=shaped[rows,prev]
    # If candidate is earlier than first 100ms anchor, forbid seeing
    # the 100ms future spectrum; use only the onset summary.
    validity=eligible & known[indices,None] & (current[...,0]>.5)
    current=np.where(validity[:,:,None],current,0)
    previous=np.where(validity[:,:,None],previous,0)
    delta=current[:,:,1:]-previous[:,:,1:]
    initial=np.broadcast_to(a[:,:9][:,None,:],(n,m,9))
    extra=np.concatenate((
        initial,current,previous,delta,
        validity[:,:,None].astype(np.float32),
        np.broadcast_to(np.minimum(AGES[at],duration_ms)[:,:,None]/1000.,
                        (n,m,1)),
    ),axis=-1).reshape(n*m,-1)
    need(extra.shape==(n*m,25) and np.isfinite(extra).all(),
         "S664 candidate acoustic feature shape invalid")
    return extra.astype(np.float32)


def joined_features(X,C,on,old,surv,base,acoustics,known,indices):
    basic=candidate_matrix_features(X,C,on,old,surv,base,indices)
    acoustic=acoustic_candidate_features(acoustics,known,C,on,indices)
    out=np.concatenate((basic,acoustic),axis=-1)
    need(np.isfinite(out).all(),"S664 invalid fused acoustic NOTE feature")
    return out


def learn_ranker(X,C,on,old,surv,base,acoustics,known,target,idx,seed):
    input=joined_features(X,C,on,old,surv,base,acoustics,known,idx)
    correct=(np.abs(C[idx]-target[idx,None])<=TOL).reshape(-1)
    need(int(correct.sum())>7000 and int((~correct).sum())>10000,
         "S664 insufficient acoustic-candidate training labels")
    model=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=.057,max_iter=155,
        max_leaf_nodes=23,min_samples_leaf=65,
        l2_regularization=18.,early_stopping=False,
        random_state=seed)
    model.fit(input,correct)
    return model,dict(candidate_rows=int(len(correct)),
                      positive=int(correct.sum()),
                      negative=int((~correct).sum()))


def run(frozen,source,annotations,out):
    need(not out.exists(),"S664 immutable evidence destination already exists")
    p=read(frozen)
    on,old,surv,base=evidence(p)
    target,_=labels_for_other_songs_only(p,annotations)
    eligible=target>0
    need(int(eligible.sum())==N_MATCHED and
         int((np.abs(base[eligible]-target[eligible])<=TOL).sum())==16543,
         "S664 baseline S425 changed")
    acoustic,known=align_audio(p,source)
    X=build_features(p,on,old,surv,base)
    names,C=get_matrix(candidates(on,old,surv,base))
    comps=np.array([composition_key(q) for q in
                    p["track_member"].astype(str)])
    folds=np.array([composition_fold(c) for c in comps])
    need(len(set(folds))==4,"S664 invalid song folds")
    output={f"S664_acoustic_penalty_{a}":base.copy() for a in PENALTIES}
    per_fold=[]
    for f in range(4):
        held=(folds==f)
        train=(folds!=f)&eligible
        tr=np.flatnonzero(train)
        va=np.flatnonzero(held)
        need(len(tr)>12000 and len(va)>4000 and
             not(set(comps[train])&set(comps[held])),
             "S664 held composition exposure")
        model,proof=learn_ranker(X,C,on,old,surv,base,acoustic,known,
                                 target,tr,664+f)
        score=model.predict_proba(joined_features(
            X,C,on,old,surv,base,acoustic,known,va))[:,1]
        score=score.reshape(len(va),len(names))
        for penalty in PENALTIES:
            k=f"S664_acoustic_penalty_{penalty}"
            output[k][va]=pick_from_prob(score,C[va],names,penalty)
        subset=held&eligible
        item=dict(fold=f,train_correct_onset_notes=len(tr),
            held_notes=len(va),held_evaluable_notes=int(subset.sum()),
            acoustic_feature_mapped_notes=int(known[va].sum()),
            acoustic_coverage=float(known[va].mean()),
            train_proof=proof,composition_leakage=0,
            scores={k:int((np.abs(o[subset]-target[subset])<=TOL).sum())
                    for k,o in output.items()},
            baseline_correct=int(
                (np.abs(base[subset]-target[subset])<=TOL).sum()))
        per_fold.append(item)
        print("S664 FOLD",json.dumps(item),flush=True)
    scores=dict(S425=stats(base,target,eligible))
    for k,v in output.items():
        scores[k]=stats(v,target,eligible)
    report=dict(status="S664_ACOUSTIC_NOTE_OFFSET_CANDIDATE_SELECTION",
        samples_original_S80_K_frozen=59309,
        emitted_pitch_onset_notes_unchanged=33475,
        acoustic_source="S411 real MIC STFT (previously computed)",
        S411_note_alignment_by_row_and_pitch_not_index=True,
        acoustic_feature_note_coverage=int(known.sum()),
        acoustic_feature_coverage_fraction=float(known.mean()),
        no_held_annotation_used_as_inference_feature=True,
        held_composition_validation=True,
        upstream_S425_models_not_fully_nested=True,
        guitarset_historically_exposed=True,
        strict_live_not_modified=True,
        S411_centered_STFT_not_strictly_causal=True,
        S425_upstream_noteoff_can_use_future_audio=True,
        target=.80,
        reached_80=bool(max(x["accuracy"] for x in scores.values())>=.8),
        original_reference_not_changed=True,
        no_promotion=True,
        by_fold=per_fold,scores=scores,
        original_hashes=dict(S425=sha(frozen),S411=sha(source),
                             JAMS=sha(annotations)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S664 SCORES",json.dumps(scores),flush=True)
    print("S664 FINAL",json.dumps(dict(
        baseline=scores["S425"]["accuracy"],
        best_experimental=max(x["accuracy"] for k,x in scores.items()
                              if k!="S425"),
        achieved_80=report["reached_80"])),flush=True)


def self_test():
    # The 100 ms acoustic anchor must NOT leak to a note closing at 30ms.
    n=2;m=3
    x=np.ones((n,59),np.float32)
    x[:,9:]=2
    C=np.array([[1300,2000,3500],[1500,5000,10000]],np.int64)
    on=np.zeros(n,np.int64)
    out=acoustic_candidate_features(x,np.ones(n,bool),C,on,np.arange(n))
    assert out.shape==(n*m,25)
    assert np.all(out[:3,9:19]==0) # 29..79ms candidates
    print("S664 SELFTEST PASS physical offset candidate ages and no early acoustic lookahead")


if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--frozen",type=Path)
    q.add_argument("--s411",type=Path)
    q.add_argument("--annotations",type=Path)
    q.add_argument("--out",type=Path)
    q.add_argument("--self-test",action="store_true")
    a=q.parse_args()
    if a.self_test:self_test()
    else:
        need(a.frozen and a.s411 and a.annotations and a.out,
             "Need --frozen --s411 --annotations --out")
        run(a.frozen,a.s411,a.annotations,a.out)
