"""S523 — separated ADD_NEW_PITCH vs DELETE_FALSE_PITCH learned heads.

Root cause: S519/S521 fourteen-head sigmoid selector recovered ZERO true
new pitches absent from BP, although the 14th acoustic generator could
emit MIDI40–88. S523 learns ADD and DELETE on DISTINCT training examples
and feeds them S516-style actual audio probabilities + the old 13-head
acoustic votes/context. One music birth event may have several notes.

CRITICAL NESTING: 8 outer leave-one-original-recording-out folds. For
each held recording, generate seven TRAIN recordings' S516 probabilities
from models fitted on six OTHER outer-training recordings (56 inner
fits total), fit the two action heads only from those TRAIN examples,
and generate held scores using a S516 model fit on seven TRAIN records.

No held MIDI truth leaks into outer training or thresholds. The eight
recordings were historically inspected, thus results are exploratory
and not a new independent blind generalization benchmark. All metrics
conditional on known positive true onsets; piano still has 102 omitted
canonical onsets and real stream negatives are not verified. No official
S80 / V88 / Live modification or promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def joint_features(d,X,rows,gen):
    rows=np.asarray(rows,int)
    n=len(rows)
    need(gen.shape==(n,49),"S523 nested generator scores shape must be events x 49")
    raw=X[rows].astype(np.float32)
    old=d["head_note_bits"][rows,
         d["selected_BP_baseline_head"][rows].astype(int),40:89].astype(bool)
    available=d["head_mask"][rows].astype(bool)
    votes=d["head_note_bits"][rows,:,40:89].astype(np.float32)
    notes= np.sum(votes*available[:,:,None],axis=1)/13.
    prior=d["source_K"][rows].astype(np.float32)
    extra=np.stack([gen,old.astype(np.float32),notes,
                    np.broadcast_to(np.minimum(prior[:,None],6)/6.,(n,49)),
                    np.maximum(raw[:,:,2]-raw[:,:,4],0),
                    raw[:,:,9]],axis=-1).astype(np.float32)
    feats=np.concatenate([raw,extra],axis=-1)
    need(feats.shape==(n,49,34) and np.isfinite(feats).all(),
         "S523 independent ADD/REMOVE heads must use only 34 real measured acoustic features")
    return feats,old
def fit_generator(X,y,rows,seed):
    from scripts.s519_train_nested_S509_attention_with_S516_raw_generator_14head import fit_gen
    return fit_gen(X,y,None,rows,seed)
def predict_generator(m,X,rows):
    from scripts.s519_train_nested_S509_attention_with_S516_raw_generator_14head import predict_gen
    return predict_gen(m,X,rows)
def train_two_actions(feats,old,y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    X=feats.reshape(-1,34)
    baseline=old.reshape(-1)
    target=y.reshape(-1)
    model={}
    n={}
    for name,sel in (("ADD_NEW",~baseline),("KEEP_OLD",baseline)):
        xx=X[sel];yy=target[sel]
        positive=int(np.sum(yy));negative=int(np.sum(~yy))
        need(positive>=5 and negative>=5,
             "S523 separate ADD_NEW vs KEEP_OLD supervision lacks both true classes")
        clf=make_pipeline(StandardScaler(),LogisticRegression(
                solver="liblinear",class_weight="balanced",
                C=.25,max_iter=1000,random_state=1523))
        clf.fit(xx,yy)
        model[name]=clf
        n[name]=dict(positive_pitch_labels=positive,negative_pitch_labels=negative)
    return model,n
def out_actions(model,feats,old):
    flat=feats.reshape(-1,34);flag=old.reshape(-1)
    p_add=np.zeros(len(flat),np.float32);p_keep=np.zeros(len(flat),np.float32)
    p_add[~flag]=model["ADD_NEW"].predict_proba(flat[~flag])[:,1]
    p_keep[flag]=model["KEEP_OLD"].predict_proba(flat[flag])[:,1]
    return p_add.reshape(old.shape),p_keep.reshape(old.shape)
def decode(p_add,p_keep,g,baseline,policy):
    old=baseline[:,40:89]
    output=old.copy()
    if policy in ("add_090","joint_090_015"):
        output[(~old)&(p_add>=.90)&(g>=.50)]=True
    elif policy=="joint_080_020":
        output[(~old)&(p_add>=.80)&(g>=.60)]=True
    elif policy=="delete_015":
        pass
    else:raise ValueError("S523 unsupported train-only-predeclared action policy")
    if policy in ("delete_015","joint_090_015"):
        output[old&(p_keep<=.15)&(g<=.35)]=False
    elif policy=="joint_080_020":
        output[old&(p_keep<=.20)&(g<=.30)]=False
    zero=~output.any(axis=1)
    output[zero]=old[zero]
    result=baseline.copy()
    result[:,40:89]=output
    return result
def score(y,base,out,sourceK):
    K=y.sum(1);Kold=base.sum(1);Knew=out.sum(1)
    old=(y==base).all(1);new=(y==out).all(1)
    change=np.any(base!=out,axis=1)
    def metrics(mask):
        tp=int(np.sum((y&out)[mask]));fp=int(np.sum((~y&out)[mask]))
        fn=int(np.sum((y&~out)[mask]))
        return dict(n=int(mask.sum()),BP_complete_chords=int(np.sum(mask&old)),
            candidate_complete_chords=int(np.sum(mask&new)),
            fixes=int(np.sum(mask&~old&new&change)),
            regressions=int(np.sum(mask&old&~new&change)),
            wrong_to_wrong=int(np.sum(mask&~old&~new&change)),
            modified=int(np.sum(mask&change)),
            BP_exactK=int(np.sum(mask&(Kold==K))),
            candidate_exactK=int(np.sum(mask&(Knew==K))),
            source_original_exactK=int(np.sum(mask&(sourceK==K))),
            TP_note_pitch=tp,FP_note_pitch=fp,FN_note_pitch=fn,
            F1_pitches_conditional_on_known_positive_onsets=2*tp/max(1,2*tp+fp+fn),
            true_missing_BP_pitch_added=int(np.sum((~base&out&y)[mask])),
            false_extra_pitch_added=int(np.sum((~base&out&~y)[mask])),
            true_BP_pitch_removed=int(np.sum((base&~out&y)[mask])),
            wrong_BP_pitch_removed=int(np.sum((base&~out&~y)[mask])))
    return dict(overall=metrics(np.ones(len(y),bool)),
                poly=metrics(K>=2),by_trueK={str(k):metrics(K==k) for k in range(7)})
def run(a):
    need(not a.out.exists(),"S523 real split source weights and MIDI audit immutable")
    from scripts.s516_train_real_RAW_audio_49_pitch_expert_songheld import load_sources
    import joblib
    original,d,bp,y,X,true=load_sources(a)
    recording=d["recording_sha"].astype(str)
    corpus=d["corpus"].astype(str)
    sourceK=d["source_K"].astype(int)
    N=len(recording)
    need(N==265 and len(set(recording))==8,
         "S523 original eight-source 265 real annotated positive music events missing")
    results=np.full((N,49,3),np.nan,np.float32)  # [generator,add,keep]
    a.out.mkdir(parents=True)
    folds=[]
    for fold,held in enumerate(sorted(set(recording))):
        train_rows=np.flatnonzero(recording!=held)
        test_rows=np.flatnonzero(recording==held)
        train_rec=sorted(set(recording[train_rows]))
        need(len(train_rec)==7 and len(test_rows)>0 and
             held not in train_rec,"S523 outer held source used to fit experts")
        nested_scores=np.full((len(train_rows),49),np.nan,np.float32)
        for k,inner in enumerate(train_rec):
            inn=np.flatnonzero(recording[train_rows]!=inner)
            out=np.flatnonzero(recording[train_rows]==inner)
            need(len(set(recording[train_rows[inn]]))==6
                 and held not in set(recording[train_rows[inn]]),
                 "S523 generator must be trained on six OTHER outer-train sources")
            m=fit_generator(X,true,train_rows[inn],seed=152300+fold*11+k)
            nested_scores[out]=predict_generator(m,X,train_rows[out])
        need(np.isfinite(nested_scores).all(),
             "S523 missing original nested train-only generator output")
        final=fit_generator(X,true,train_rows,seed=152390+fold)
        pgen=predict_generator(final,X,test_rows)
        trfeat,trold=joint_features(d,X,train_rows,nested_scores)
        tfeat,told=joint_features(d,X,test_rows,pgen)
        experts,counts=train_two_actions(trfeat,trold,true[train_rows])
        add,keep=out_actions(experts,tfeat,told)
        results[test_rows,:,0]=pgen
        results[test_rows,:,1]=add
        results[test_rows,:,2]=keep
        gen_file=a.out/f"S523_fold{fold}_S516_NESTED_RAW_GENERATOR.joblib"
        add_file=a.out/f"S523_fold{fold}_ADD_NEW_NOTE_ACTION.joblib"
        keep_file=a.out/f"S523_fold{fold}_KEEP_OR_REMOVE_NOTE_ACTION.joblib"
        for filename,model in ((gen_file,final),(add_file,experts["ADD_NEW"]),
                               (keep_file,experts["KEEP_OLD"])):
            joblib.dump(model,filename,compress=3)
        fold_record=dict(fold=fold,held_recording_SHA=held,
            outer_train_recording_count=7,
            inner_train_recording_count_per_generator=6,
            nested_generator_inner_models=7,
            ADD_head_supervised_training_counts=counts["ADD_NEW"],
            KEEP_head_supervised_training_counts=counts["KEEP_OLD"],
            genuinely_new_pitches_allowed_outside_BasicPitch=True,
            no_outer_held_recording_used_in_any_training=True,
            generator_sha256=sha(gen_file),
            add_action_sha256=sha(add_file),
            keep_action_sha256=sha(keep_file))
        folds.append(fold_record)
        print("S523 2 DISTINCT MUSIC NOTE ACTION HEADS TRAINED",json.dumps(fold_record),flush=True)
    need(np.isfinite(results).all() and
         np.all((results>=0)&(results<=1)),
         "S523 no genuine held-source note action predictions")
    by_source={}
    for name in ("guitar_S80","piano_V88"):
        ix=np.flatnonzero(corpus==name)
        by_source[name]={}
        for policy in ("add_090","delete_015","joint_090_015","joint_080_020"):
            proposed=decode(results[ix,:,1],results[ix,:,2],
                            results[ix,:,0],bp[ix],policy)
            audit=score(y[ix],bp[ix],proposed,sourceK[ix])
            need(audit["overall"]["BP_complete_chords"]==
                 original["by_source"][name]["raw_BP_base_full_note_sets_correct"],
                 "S523 original real source note baseline unexpectedly changed")
            audit["true_attacks_without_BP_canonical_group"]=original["by_source"][name][
                 "annotated_true_attacks_NOT_matched"]
            audit["all_original_true_positive_attacks"]=original["by_source"][name][
                 "truth_original_positive_attacks"]
            audit["full_stream_false_positive_onsets_not_measured"]=True
            by_source[name][policy]=audit
        print("S523 SEPARATE ADD AND REMOVE FULL TRUE PITCH VERDICT",json.dumps(dict(
          corpus=name,policies={k:{"overall":v["overall"],"poly":v["poly"]}
                                for k,v in by_source[name].items()})),flush=True)
    preds=a.out/"S523_SEPARATE_TRUE_NOTE_ADDITION_AND_DELETION_PROBABILITIES.jsonl"
    preds.write_text("".join(json.dumps(dict(
        canonical_event_id=str(d["event_id"][i]),
        source_recording_SHA=str(recording[i]),
        source_instrument=str(corpus[i]),
        raw_generator_pitch_40to88_prob=results[i,:,0].round(6).tolist(),
        ADD_NEW_NOTE_pitch_40to88_prob=results[i,:,1].round(6).tolist(),
        KEEP_BP_NOTE_pitch_40to88_prob=results[i,:,2].round(6).tolist(),
        source_label_never_used_in_held_inference=True))+"\n" for i in range(N)))
    report=dict(status="S523_2_SEPARATE_LEARNED_ADD_REMOVE_PITCH_HEADS_NESTED_EIGHT_RECORDINGS",
       original_S509_selection_and_S516_generator_problems_separated=True,
       genuinely_learned_ADD_NEW_NOTE_and_KEEP_OR_REMOVE_NOTE_heads=True,
       held_source_excluded_from_all_nested_inner_model_and_outer_model_training=True,
       train_song_group_count=7,inner_acoustic_training_songs=6,
       real_acoustic_pitches_40to88_not_limited_to_existing_BP_notes=True,
       fixed_policies_not_fit_to_held_MIDI=True,
       same_eight_historically_exposed_sources_not_blind=True,
       no_verified_negative_model_onset_events=True,
       missing_102_original_true_piano_attacks_not_recovered=True,
       no_official_S80_V88_Live_modification=True,
       no_promotion=True,
       folds=folds,by_source=by_source,
       frozen_source_S505_SHA=sha(a.heads),
       raw_original_S515_SHA=sha(a.acoustic),
       predicted_action_scores_SHA=sha(preds))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    bp=np.zeros((1,128),bool);bp[0,52]=True
    a=np.zeros((1,49),float);k=np.zeros((1,49),float);g=np.zeros((1,49),float)
    a[0,55-40]=.94;g[0,55-40]=.9
    k[0,52-40]=.98
    found=decode(a,k,g,bp,"joint_090_015")
    need(set(np.flatnonzero(found[0]))=={52,55},
         "S523 separate ADD head must add G3 while KEEP head preserves E3")
    print("S523 SELFTEST PASS independent true ADD and DELETE music note action heads")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("acoustic","acoustic-report","heads","head-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("acoustic","acoustic_report","heads","head_report","out"):
            need(getattr(a,k)is not None,"S523 missing --"+k)
        run(a)
