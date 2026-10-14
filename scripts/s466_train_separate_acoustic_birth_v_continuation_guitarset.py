"""S466: learned newborn-note vs non-birth event discriminator on GuitarSet ONLY.

A dedicated, source-K-agnostic binary head on 1090 real original V88/S97
features, trained on 59309 original GuitarSet candidate events labelled by
their true onset K. K=0 means no annotated onset assigned to this candidate;
it does NOT mean silent audio. Train four composition-disjoint heads, each with
separate whole-composition inner-validation threshold selection. Select a
conservative event suppression confidence gate ONLY on GuitarSet held songs
and protect recall of true poly and the source-S80 missed-K0 births.

This exports an EXPERIMENT, never mutates original source K, note names or
event timing. IDMT XML is never opened/used to train or choose thresholds.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    need,load,composition_key,composition_fold)
from scripts.s444_balanced_joint_note_swap_songheld_Klocked import balanced_inner_partition
from scripts.s458_train_independent_audio_only_K_blind_songheld import inputs
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

N=59309
SEED=466
DIM=1090
CUTS=(.01,.02,.035,.05,.075,.10,.125,.15,.20,.25,.30,.40)
FOLDS=4

def make_model(seed):
    return ExtraTreesClassifier(
        n_estimators=36,min_samples_leaf=16,max_depth=12,
        max_features="sqrt",class_weight="balanced_subsample",
        n_jobs=2,random_state=seed)

def predict_birth(est,X):
    need(np.array_equal(est.classes_,[0,1]),
         "S466 train-only original note-birth binary labels absent")
    prob=est.predict_proba(X)[:,1]
    need(len(prob)==len(X) and np.isfinite(prob).all()
         and np.all((prob>=0)&(prob<=1)),
         "S466 birth probability not a finite calibrated model output")
    return prob

def recall_stats(prob,cut,birth,oldK,trueK):
    drop=prob<cut
    positive=birth
    critical=positive&(oldK==0)
    poly=(trueK>=2)
    def fraction(mask):
        n=int(np.sum(mask))
        return float(np.mean(~drop[mask])) if n else None
    return dict(cut=float(cut),
        true_onset_recall=fraction(positive),
        true_poly_onset_recall=fraction(poly),
        source_K0_true_onset_recall=fraction(critical),
        true_birth_event_count=int(positive.sum()),
        source_K0_true_birth_count=int(critical.sum()),
        poly_true_birth_count=int(poly.sum()),
        nonbirth_dropped=int(np.sum(drop&~birth)),
        nonbirth_total=int(np.sum(~birth)),
        original_true_birth_lost=int(np.sum(drop&birth)),
        original_poly_birth_lost=int(np.sum(drop&poly)),
        original_missed_by_S80_birth_lost=int(np.sum(drop&critical)))

def select_gate(p,oldK,trueK):
    birth=trueK>0
    # Select thresholds ONLY using GuitarSet composition-disjoint inner
    # validation. Never move this threshold after seeing external IDMT truth.
    options=[recall_stats(p,cut,birth,oldK,trueK) for cut in CUTS]
    acceptable=[]
    for x in options:
        if (x["true_onset_recall"]>=.995
            and x["true_poly_onset_recall"]>=.995
            and (x["source_K0_true_onset_recall"] is None
                 or x["source_K0_true_onset_recall"]>=.995)
            and x["nonbirth_dropped"]>=max(20,round(.01*x["nonbirth_total"]))):
            acceptable.append(x)
    # No safe gate = source K unchanged; do not improvise on IDMT.
    chosen=max(acceptable,key=lambda a:a["cut"]) if acceptable else None
    return (chosen["cut"] if chosen else None),options,chosen

def run(audio,s80,truth,out):
    need(not out.exists(),"S466 immutable source count/attack head outputs exist")
    a=load(audio);o=load(s80);t=load(truth)
    ids=a["row_id"].astype(str)
    base=o["s80_K"].astype(np.int8)
    trueK=t["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and np.array_equal(ids,o["row_id"].astype(str))
         and np.array_equal(ids,t["row_id"].astype(str))
         and np.array_equal(a["incumbent_freeze_K"],o["incumbent_freeze_K"])
         and int(np.sum(base==trueK))==53529
         and int(np.sum((base==trueK)&(trueK>=2)))==6054
         and np.isin(trueK,range(7)).all(),
         "S466 genuine historical original S80 and annotated GuitarSet source changed")
    X=inputs(a,base)
    comps=np.asarray([composition_key(k) for k in a["member"].astype(str)])
    folds=np.asarray([composition_fold(v) for v in comps],np.int8)
    birth=(trueK>0).astype(np.int8)
    need(100<int(np.sum(birth==0))<N-100
         and 100<int(np.sum(birth==1))<N-100,
         "S466 genuine S258 JAMS truth lacks birth or nonbirth examples")
    allscores=np.full(N,np.nan,np.float32)
    decision=np.full(N,False,bool)
    threshold_by_fold=[]
    out.mkdir(parents=True)
    for fold in range(FOLDS):
        train=np.flatnonzero(folds!=fold)
        held=np.flatnonzero(folds==fold)
        innertrain,innerval,partition=balanced_inner_partition(train,comps)
        need(len(held)>1000 and len(innertrain)>10000
             and len(innerval)>1000
             and not(set(comps[held])&set(comps[train]))
             and not(set(comps[innertrain])&set(comps[innerval])),
             "S466 actual train/calibration/held composition overlap")
        calibrator=make_model(SEED+fold)
        calibrator.fit(X[innertrain],birth[innertrain])
        pv=predict_birth(calibrator,X[innerval])
        cut,all_choices,selected=select_gate(pv,base[innerval],trueK[innerval])
        del calibrator
        model=make_model(SEED+100+fold)
        model.fit(X[train],birth[train])
        ph=predict_birth(model,X[held])
        allscores[held]=ph.astype(np.float32)
        # Suppression is a DIAGNOSTIC proposed flag. Critical rule:
        # original K>0 events are always kept; no actual events discarded.
        selected_keep=((base[held]>0)|(ph>=cut)) if cut is not None else np.ones(len(held),bool)
        decision[held]=selected_keep
        pack=dict(
            status="S466_SOURCE_GUITARSET_TRAINED_ONLY_ONSET_BIRTH",
            model=model,fold=int(fold),
            train_feature_width=1090,
            no_original_K_feature=True,
            original_K_used_only_for_retention_safety=True,
            inner_selected_cut=cut,
            nonbirth_can_be_annotated_resonance=True,
            no_training_on_IDMT=True)
        file=out/f"S466_birth_vs_nonbirth_fold{fold}.joblib"
        joblib.dump(pack,file,compress=3)
        threshold_by_fold.append(dict(
           fold=fold,whole_composition_split=partition,
           inner_val_count=len(innerval),held_count=len(held),
           inner_selected_threshold=cut,
           inner_selected_validation_report=selected,
           all_inner_diagnostic_thresholds=all_choices,
           model_SHA256=sha(file),
           source_test=recall_stats(ph,cut if cut is not None else 0,
                 birth[held].astype(bool),base[held],trueK[held]),
           source_no_label_leakage=True))
        print("S466 FOLD",json.dumps(dict(fold=fold,
            calibrated_threshold=cut,
            held_label_count=len(held),
            held_birth_recall=threshold_by_fold[-1]["source_test"]["true_onset_recall"],
            held_poly_recall=threshold_by_fold[-1]["source_test"]["true_poly_onset_recall"],
            original_K0_birth_missed=threshold_by_fold[-1]["source_test"]["original_missed_by_S80_birth_lost"],
            recommended_drop=int(np.sum(~selected_keep)))),flush=True)
    need(np.isfinite(allscores).all() and np.array_equal(
        np.sort(np.concatenate([np.flatnonzero(folds==i) for i in range(4)])),
        np.arange(N)),"S466 held-composition predictions not complete")
    pred=out/"S466_GUITARSET_S80_UNCHANGED_BINARY_BIRTH_OOF.npz"
    np.savez_compressed(pred,row_id=ids,original_S80_K=base,true_K_ONLY_FOR_AUDIT=trueK,
        birth_prob_OOF=allscores,retention_proposed_OOF=decision,
        fold=folds)
    report=dict(
        status="S466_GUITARSET_TRAIN_ONLY_COMPOSITION_DISJOINT_BINARY_BIRTH_HEAD",
        source_GuitarSet_original_events=N,
        positive_true_birth_count=int(np.sum(birth)),
        negative_unmatched_birth_count=int(np.sum(~birth.astype(bool))),
        trained_four_fold_heads=4,all_folds=threshold_by_fold,
        original_S80_K_global_correct=53529,
        original_S80_K_true_poly_correct=6054,
        candidate_event_retain=int(decision.sum()),
        candidate_event_drop=int(np.sum(~decision)),
        hypothetical_true_birth_lost=int(np.sum(~decision&(birth==1))),
        hypothetical_poly_birth_lost=int(np.sum(~decision&(trueK>=2))),
        hypothetical_source_K0_real_birth_lost=int(np.sum(
            ~decision&(trueK>0)&(base==0))),
        original_S80_K_and_note_predictions_UNCHANGED=True,
        dataset_GuitarSet_historical_training_exposure=True,
        IDMT_not_read_during_training_or_threshold_selection=True,
        model_no_original_S80_K_as_input=True,
        model_retention_no_reference_changes=True,
        threshold_selection_nested_whole_composition=True,
        no_promotion=True,
        SHA256=dict(S258_audio=sha(audio),S397_K=sha(s80),
            S258_held_annotated_K=sha(truth),OOF=sha(pred)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S466 SUMMARY",json.dumps(dict(
        positive_birth=report["positive_true_birth_count"],
        nonbirth=report["negative_unmatched_birth_count"],
        candidate_drop=report["candidate_event_drop"],
        real_birth_lost=report["hypothetical_true_birth_lost"],
        real_poly_lost=report["hypothetical_poly_birth_lost"],
        source_K0_real_birth_lost=report["hypothetical_source_K0_real_birth_lost"])),
        flush=True)

def selftest():
    p=np.asarray([.01,.05,.15,.9])
    oldK=np.asarray([0,0,2,0])
    real=np.asarray([0,2,3,1])
    chosen,options,report=select_gate(p,oldK,real)
    need(chosen is None and len(options)==len(CUTS),
        "S466 training-only gate must abstain if true births threatened")
    print("S466 SELFTEST PASS true-K0 vs annotated true note, safe gate abstention")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for x in ("audio","s80","truth","out"):p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for x in ("audio","s80","truth","out"):
            need(getattr(a,x)is not None,"S466 missing --"+x)
        run(a.audio,a.s80,a.truth,a.out)
