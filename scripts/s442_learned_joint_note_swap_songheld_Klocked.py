"""S442 -- learned JOINT note-set swap model, with S80 Exact-K locked.

This is a DEV-only stacked challenger to the frozen S428 OOF pitch sets.
For each event, candidates and pairwise/contextual features are label-blind.
Only OTHER compositions train a per-fold model and choose a conservative
decision threshold on an additional disjoint inner validation partition.
A one-note substitution preserves S80 K exactly. No Live, MIDI file or Offset.

The S428/S416 OOF auxiliary predictions are not fully nested in S442, and
S80 historically saw GuitarSet. Therefore these results are NOT independent
generalization and must not be promoted without a separate external audit.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key, composition_fold, labels, load, need,
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import (
    MIDI_MIN, PRE, ON, LATE, harmonic_pitch_scores, note_name, sha,
)
from scripts.s428_train_sound_to_note_names_harmonic_ranker import (
    accuracy_rows, exact, N, POLY, EXACT_GLOBAL, EXACT_POLY,
)

SEED=442
REQUIRED=(2,3,4)
MAX_CANDIDATES=14
FEATURE_COUNT=31
THRESHOLDS=(0.55,0.65,0.75,0.85,0.92)
MIN_VAL_CORRECTIONS=3

def proposals(sequence, scores416, baseline, k):
    """Pure acoustics + frozen external pitch proposals, no target notes."""
    z=np.asarray(sequence,dtype=np.float32)
    probs=np.asarray(scores416,dtype=np.float32)
    base=sorted(int(p) for p in baseline if p>=MIDI_MIN)
    need(z.shape==(42,49) and probs.shape==(49,) and len(base)==k
         and len(set(base))==k and k in REQUIRED,
         "S442 invalid K-locked event or training-free pitch proposal")
    evidence=harmonic_pitch_scores(z[None])[0].astype(float)
    a=np.expm1(z.astype(np.float64))
    pre=a[PRE].mean(0);on=a[ON].mean(0);late=a[LATE].mean(0)
    nov=np.maximum(on-pre,0)
    birth=nov/(np.maximum(pre,on)+1e-6)
    attack=on/(max(float(np.max(on)),1e-6))
    pm=probs/(max(float(np.max(probs)),1e-6))
    raw=evidence/(max(float(np.max(evidence)),1e-6))
    traj=np.maximum(a-np.mean(a[PRE],axis=0)[None],0)
    norm=np.linalg.norm(traj,axis=0)+1e-7
    cosine=np.clip((traj.T@traj)/(norm[:,None]*norm[None,:]),0,1)
    options=set(base)
    options.update(int(MIDI_MIN+i) for i in
                   np.argsort(-pm,kind="stable")[:7])
    options.update(int(MIDI_MIN+i) for i in
                   np.argsort(-raw,kind="stable")[:7])
    for p in base:
        for d in (-19,-12,-7,-2,-1,1,2,7,12,19):
            q=p+d
            if 40<=q<=88:options.add(q)
    # Preserve all baseline pitches. No training labels are added to pool.
    best=sorted((x for x in options if x not in base),
                key=lambda p:(-float(.6*pm[p-40]+.4*raw[p-40]),p))
    pool=set(base)|set(best[:MAX_CANDIDATES-k])
    examples=[];swaps=[]
    for old in base:
        oi=old-40
        others=[p for p in base if p!=old]
        for new in sorted(pool-set(base)):
            ni=new-40
            d=new-old
            distances=[abs(new-p) for p in others]
            nearest=min(distances)
            nearest_old=min(abs(old-p) for p in others)
            max_corr=max(float(cosine[ni,p-40]) for p in others)
            octave=max(1e-8,float(on[ni-12])) if ni>=12 else 0.
            octave_hi=float(on[ni+12]) if ni+12<49 else 0.
            other_energy=np.mean([float(on[p-40]) for p in others])
            # Pair + complete surviving chord context, features vary by
            # replacing each note rather than ranking pitches independently.
            feats=[
                k/6., oi/48.,ni/48.,d/24.,abs(d)/24.,
                float(abs(d)==12),float(abs(d)==19),float(abs(d)==7),
                float(abs(d)<=2),
                pm[oi],pm[ni],pm[ni]-pm[oi],
                raw[oi],raw[ni],raw[ni]-raw[oi],
                attack[oi],attack[ni],attack[ni]-attack[oi],
                birth[oi],birth[ni],birth[ni]-birth[oi],
                float(late[oi]/(on[oi]+.1)),
                float(late[ni]/(on[ni]+.1)),
                float(cosine[oi,ni]),max_corr,
                nearest/24.,nearest_old/24.,
                float(nearest==12 or nearest==19),
                float((octave+octave_hi)/(on[ni]+.1)),
                float(on[ni]/(other_energy+.1)),
                float(sum(abs(new-p) in (7,12,19) for p in others))/k,
            ]
            need(len(feats)==FEATURE_COUNT,"S442 chord feature dimension mismatch")
            examples.append(feats);swaps.append((old,new))
    return np.asarray(examples,np.float32),np.asarray(swaps,np.int16)

def make_training_bank(audio,prob,original,k,truth):
    """Build prospective candidate bank; truth controls LABEL only."""
    ix=np.flatnonzero(np.isin(k,REQUIRED))
    parts=[];swaps=[];labels_out=[];starts=np.zeros(len(k),np.int64)
    stops=np.zeros(len(k),np.int64);cursor=0
    positives=negatives=0
    for i in ix:
        x,s=proposals(audio[i],prob[i],original[i],int(k[i]))
        need(len(s)>0,"S442 empty pitch-swap set")
        starts[i]=cursor;cursor+=len(s);stops[i]=cursor
        true=set(int(m) for m in truth[i] if m>=MIDI_MIN)
        # One swap improves matched-note count iff newly inserted pitch
        # is present and the removed pitch absent. No true K gating.
        y=np.asarray([(int(add in true and old not in true)) for old,add in s],np.uint8)
        parts.append(x);swaps.append(s);labels_out.append(y)
        positives+=int(y.sum());negatives+=int(len(y)-y.sum())
        if len(parts)%2000==0:
            print("S442 INPUT",json.dumps(dict(events=len(parts),
                candidate_swaps=cursor,positives=positives)),flush=True)
    X=np.concatenate(parts);pair=np.concatenate(swaps);y=np.concatenate(labels_out)
    need(X.shape==(cursor,FEATURE_COUNT) and pair.shape==(cursor,2)
         and y.shape==(cursor,) and np.isfinite(X).all()
         and positives>100 and negatives>1000,
         "S442 insufficient actual learning samples")
    return dict(X=X,pair=pair,label=y,starts=starts,stops=stops,
                ix=ix,positives=positives,negatives=negatives)

def bank_indices(bank,event_indices):
    offsets=[np.arange(bank["starts"][i],bank["stops"][i]) for i in event_indices]
    return np.concatenate(offsets) if offsets else np.array([],np.int64)

def score_events(model,bank,indices):
    """One probability per eligible swap, indexed back to its event."""
    picks=bank_indices(bank,indices)
    if not len(picks):return {}
    proba=model.predict_proba(bank["X"][picks])[:,1]
    result={}
    cursor=0
    for i in indices:
        n=int(bank["stops"][i]-bank["starts"][i])
        values=proba[cursor:cursor+n]
        best=int(np.argmax(values))
        pair_id=int(bank["starts"][i]+best)
        result[int(i)]=(float(values[best]),tuple(map(int,bank["pair"][pair_id])))
        cursor+=n
    need(cursor==len(picks),"S442 paired probability alignment corruption")
    return result

def apply(original,proposed,threshold):
    """Guarantee no cardinality change; no note is duplicated or destroyed."""
    row=original.copy()
    for i,(p,(old,new)) in proposed.items():
        if p<threshold:continue
        live=[int(v) for v in row[i] if v>=MIDI_MIN]
        need(old in live and new not in live and old!=new,
             "S442 swap not admissible for frozen K")
        slot=next(j for j,v in enumerate(row[i]) if v==old)
        row[i,slot]=new
    need(np.array_equal((row>=MIDI_MIN).sum(1),
             (original>=MIDI_MIN).sum(1)),
         "S442 changed Exact-K")
    return row

def per_set_stats(target,original,candidate,indices):
    corrections=regressions=0
    tp_new=tp_old=0
    for i in indices:
        t=collections.Counter(int(m) for m in target[i] if m>=MIDI_MIN)
        a=collections.Counter(int(m) for m in original[i] if m>=MIDI_MIN)
        b=collections.Counter(int(m) for m in candidate[i] if m>=MIDI_MIN)
        corrections+=int(b==t and a!=t)
        regressions+=int(a==t and b!=t)
        tp_old+=sum((a&t).values())
        tp_new+=sum((b&t).values())
    return dict(n=int(len(indices)),corrections=corrections,
                regressions=regressions,net=corrections-regressions,
                note_tp_delta=tp_new-tp_old)

def train(bank,ids,seed):
    sel=bank_indices(bank,ids)
    need(len(sel)>4000,"S442 too few training substitutions")
    X=bank["X"][sel];y=bank["label"][sel]
    count=int(y.sum())
    need(count>=100 and len(y)-count>1000,"S442 degenerate swap labels")
    # Positive swap improvements are rare; choose only on train samples.
    weight=np.where(y==1, min(12.,max(2.,(len(y)-count)/count)),1.)
    model=HistGradientBoostingClassifier(
        max_iter=95,learning_rate=.07,max_leaf_nodes=15,
        min_samples_leaf=50,l2_regularization=10.,
        early_stopping=False,random_state=seed)
    model.fit(X,y,sample_weight=weight)
    return model,dict(candidates=int(len(y)),positive_swaps=count,
                      negative_swaps=int(len(y)-count))

def run(audio,s80,s416,s428,truth,annotations,out):
    need(not out.exists(),"S442 never overwrites preserved experiments")
    a=load(audio);kdata=load(s80);n=load(s416);old=load(s428);t=load(truth)
    ids=a["row_id"].astype(str)
    K=kdata["s80_K"].astype(np.int8)
    truthK=t["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and all(np.array_equal(ids,z["row_id"].astype(str))
                 for z in (kdata,n,old,t))
         and all(np.array_equal(K,z["s80_K"]) for z in (n,old))
         and int((K==truthK).sum())==EXACT_GLOBAL
         and int(((K==truthK)&(truthK>=2)).sum())==EXACT_POLY,
         "S442 original S80 source and historical Exact-K drift")
    baseline=old["midi_pitch_slots"].astype(np.int16)
    need(np.array_equal((baseline>=0).sum(1),K)
         and int(K.sum())==33475
         and int((truthK>=2).sum())==POLY,
         "S442 S428 genuine chord reference malformed")
    seq=a["sequence42x49"]
    prob=n["note_probabilities_float16"].astype(np.float32)
    need(seq.shape==(N,42,49) and prob.shape==(N,49),
         "S442 incompatible acoustic/probability model ABI")
    comps=np.asarray([composition_key(v) for v in a["member"].astype(str)])
    folds=np.asarray([composition_fold(x) for x in comps])
    inner=np.asarray([
        int(hashlib.sha256(("S442-inner-"+v).encode()).hexdigest()[:8],16)%5==0
        for v in comps],bool)
    _,target,integrity=labels(a,annotations,t)
    need(target.shape==(N,6),"S442 full musical targets invalid")
    bank=make_training_bank(seq,prob,baseline,K,target)
    pred=baseline.copy()
    foldreports=[]
    out.mkdir(parents=True)
    for f in range(4):
        outertrain=bank["ix"][folds[bank["ix"]]!=f]
        held=bank["ix"][folds[bank["ix"]]==f]
        train_ix=outertrain[~inner[outertrain]]
        val_ix=outertrain[inner[outertrain]]
        need(len(train_ix)>1500 and len(val_ix)>250 and len(held)>500
             and not (set(comps[train_ix])&set(comps[val_ix]))
             and not (set(comps[outertrain])&set(comps[held])),
             "S442 guitar composition train/held overlap")
        model,fit=train(bank,train_ix,SEED+f)
        val_proposed=score_events(model,bank,val_ix)
        chooser=[]
        for threshold in THRESHOLDS:
            proposed=apply(baseline,val_proposed,threshold)
            row=per_set_stats(target,baseline,proposed,val_ix)
            chooser.append(dict(threshold=float(threshold),**row))
        safe=[z for z in chooser if
              z["corrections"]>=MIN_VAL_CORRECTIONS and
              z["net"]>=3 and z["corrections"]>=2*z["regressions"]
              and z["note_tp_delta"]>=0]
        selected=max(safe,key=lambda z:(z["net"],-z["regressions"],
                                        z["threshold"])) if safe else None
        if selected is not None:
            held_proposed=score_events(model,bank,held)
            changed=apply(baseline,held_proposed,selected["threshold"])
            pred[held]=changed[held]
        else:
            changed=baseline
        # No held-out truth used for tuning, training, or gating.
        filename=out/f"S442_joint_note_swap_fold{f}.joblib"
        import joblib
        joblib.dump(dict(model=model,held_fold=f,feature_count=FEATURE_COUNT,
                         uses_S80_predicted_K=True,threshold=(
                             selected["threshold"] if selected else None)),
                    filename,compress=3)
        foldreports.append(dict(fold=f,fit=fit,
            inner_train_events=len(train_ix),inner_validation_events=len(val_ix),
            held_events=len(held),inner_candidates=chooser,
            active=selected is not None,selected_validation=selected,
            model_sha256=sha(filename),held_truth_not_used_for_threshold=True))
        print("S442 FOLD",json.dumps(foldreports[-1]),flush=True)
    need(np.array_equal((pred>=0).sum(1),K)
         and int((pred>=0).sum())==int(K.sum())==33475
         and np.array_equal(pred[~np.isin(K,REQUIRED)],
                            baseline[~np.isin(K,REQUIRED)]),
         "S442 CRITICAL original S80 K0..K6 or unmodified K0/K1/K5/K6 lost")
    path=out/"S442_FROZEN_NOTE_NAMES_KLOCKED_59309.npz"
    np.savez_compressed(path,row_id=ids,s80_K=K,midi_pitch_slots=pred)
    path_names=out/"S442_FROZEN_NOTE_NAMES_KLOCKED_59309.jsonl"
    with path_names.open("w") as h:
        for rid,kk,slots in zip(ids,K,pred):
            ps=[int(p) for p in slots if p>=MIDI_MIN]
            h.write(json.dumps(dict(source_row_id=str(rid),K=int(kk),
                   notes=[note_name(p) for p in ps],MIDI_pitches=ps))+"\n")
    need(sum(len(json.loads(line)["notes"]) for line in path_names.open())
         ==int(K.sum()),"S442 exported named musical notes differ from K")
    # Evaluation occurs only AFTER outputs are frozen. No promotion decision
    # depends on target-held data. This remains a development validation.
    poly_ix=np.flatnonzero(truthK>=2)
    overall=per_set_stats(target,baseline,pred,poly_ix)
    perK={str(kk):per_set_stats(
        target,baseline,pred,np.flatnonzero(truthK==kk)) for kk in range(7)}
    full=accuracy_rows(target,baseline,pred,poly_ix)
    result=dict(status="S442_LEARNED_JOINT_NOTE_SET_SONGHELD_K_LOCKED_DEV",
      output_is_note_names_not_MIDI_files=True,
      original_S80_predicted_K_used_as_input=True,
      original_S80_K_regressions=0,
      original_S80_global_correct=EXACT_GLOBAL,candidate_S80_global_correct=EXACT_GLOBAL,
      original_S80_poly_correct=EXACT_POLY,candidate_S80_poly_correct=EXACT_POLY,
      baseline_S428_poly_correct=int(full["S419_chord_correct"]),
      challenger_S442_poly_correct=int(full["S428_chord_correct"]),
      development_poly_comparison=overall,by_true_K=perK,
      unchanged_K0_K1_K5_K6=True,
      composition_disjoint_S442_folds=True,
      S428_S416_auxiliary_OOF_not_fully_nested=True,
      historical_S80_GuitarSet_training_overlap=True,
      no_independent_new_instrument_test=True,no_promotion=True,
      inner_gates_only_other_compositions=True,
      no_YourMT3_teacher_in_training=True,
      no_Live_or_Offset_changes=True,
      source_SHA256=dict(S80=sha(s80),audio=sha(audio),
                         S416=sha(s416),S428=sha(s428)),
      output_npz_sha256=sha(path),output_names_sha256=sha(path_names),
      per_fold=foldreports,source_note_annotation_integrity=integrity)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S442 RESULT",json.dumps(dict(poly=overall,
          baseline_poly_chords=full["S419_chord_correct"],
          challenger_poly_chords=full["S428_chord_correct"],
          K_regressions=0)),flush=True)

def selftest():
    z=np.zeros((42,49),np.float32)
    z[ON,12]=np.log1p(3.)
    base=np.array([52,55,-1,-1,-1,-1],np.int16)
    features,swaps=proposals(z,np.linspace(0,1,49),base,2)
    need(features.shape[1]==FEATURE_COUNT
         and len(features)==len(swaps)>5,"S442 pair context feature selftest")
    original=np.array([base],np.int16)
    row=apply(original,{0:(.91,(52,64))},.9)
    need(row[0,0]==64 and set(int(p) for p in row[0] if p>=0)=={55,64}
         and note_name(52)=="E3" and note_name(55)=="G3",
         "S442 must enforce physical notes and lock K2")
    need(np.array_equal(apply(original,{0:(.5,(52,64))},.9),original),
         "S442 threshold must abstain")
    print("S442 SELFTEST PASS K2 locked, son→E3, acoustic joint candidates")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for flag in ("audio","s80","s416","s428","truth","annotations","out"):
        ap.add_argument("--"+flag,type=Path)
    args=ap.parse_args()
    if args.self_test:selftest()
    else:
        for arg in ("audio","s80","s416","s428","truth","annotations","out"):
            need(getattr(args,arg) is not None,"S442 missing --"+arg)
        run(args.audio,args.s80,args.s416,args.s428,args.truth,args.annotations,args.out)
