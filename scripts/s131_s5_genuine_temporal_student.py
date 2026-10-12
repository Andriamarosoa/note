"""S131: S5 teacher distillation with ORIGINAL 245D flow and 490D morphology.

Strict teacher-only fidelity benchmark: train folds 1,2,4, compare to frozen
original S5 predictions on fold 0, then use audio-only S97+S116 to infer 57
new GOAT events. These are NEW student models, never historic S5/S80.
No GuitarSet true K and no GOAT MIDI/labels read; this is not true accuracy.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import joblib
import numpy as np
from scripts.s129_s5_111d_student import archived_fold,classifier,probabilities,require,FOLDS
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s130_s5_student_rare_k import measure

VARIANTS=("control111_flow245","control111_flow245_morph490")

def historical(features_root,teacher_root,fold):
    candidates=list((features_root/f"fold{fold}").rglob(f"features-fold-{fold}.npz"))
    require(len(candidates)==1,"missing original source fold")
    ids,control,teacher=archived_fold(candidates[0],teacher_root,fold)
    with np.load(candidates[0],allow_pickle=False) as z:
        require(np.array_equal(ids,z["global_index"]),"original temporal alignment")
        seq=z["sequence"].astype(np.float32)
    require(seq.shape==(len(ids),42,49),"original GuitarSet trajectory shape")
    flow=flow_profile(seq)
    morph,_=audio_temporal_views(seq)
    require(flow.shape==(len(ids),245) and morph.shape==(len(ids),490),
            "original temporal feature contract")
    return ids,teacher,{"control111_flow245":np.column_stack((control,flow)).astype(np.float32),
                         "control111_flow245_morph490":
                          np.column_stack((control,flow,morph)).astype(np.float32)}

def external(s116,s97):
    with np.load(s116,allow_pickle=False) as z:
        require(set(z.files)=={"row_id","control111","freeze_K","geometry46"},
                "GOAT original111 input contract")
        ids=z["row_id"].copy()
        control=z["control111"].astype(np.float32)
    with np.load(s97,allow_pickle=False) as z:
        require({"row_id","flow245","morphology490"}.issubset(z.files),
                "GOAT genuine temporal features missing")
        require(not set(z.files).intersection(
            {"true_K","truth","target","label","labels","midi","y"}),
            "GOAT labels forbidden in runtime")
        require(np.array_equal(ids,z["row_id"]),"GOAT control/temporal alignment")
        flow=z["flow245"].astype(np.float32)
        morph=z["morphology490"].astype(np.float32)
    require(len(ids)==57 and len(set(map(str,ids)))==57,"GOAT native groups")
    require(control.shape==(57,111) and flow.shape==(57,245) and
            morph.shape==(57,490),"GOAT feature dimensions")
    views={"control111_flow245":np.column_stack((control,flow)).astype(np.float32),
           "control111_flow245_morph490":
             np.column_stack((control,flow,morph)).astype(np.float32)}
    for view in views.values():
        require(np.isfinite(view).all(),"nonfinite genuine GOAT audio features")
    return ids,views

def run(original_features,original_s5,s116,s97,s129,out):
    require(not out.exists(),"immutable evaluation outputs")
    hist={f:historical(original_features,original_s5,f) for f in FOLDS}
    held_ids,held_truth,held_views=hist[0]  # 'truth' here is ONLY teacher K
    with np.load(s129,allow_pickle=False) as z:
        require(set(z.files)=={"global_index","student_predicted_K",
                                "student_probabilities"},"S129 source schema")
        require(np.array_equal(held_ids,z["global_index"]),
                "S129 fold0 identity drift")
        baseline=z["student_predicted_K"].copy()
    require(len(held_truth)==15952 and int(np.sum(baseline==held_truth))==14557,
            "frozen S129 teacher-fidelity source drift")
    goat_ids,goat_views=external(s116,s97)
    unweighted=measure(held_truth,baseline,baseline)
    report={"status":"S131_REAL_TEMPORAL_FEATURE_STUDENT_ABLATION",
            "original_historic_S5_recovered":False,
            "models_are_retrained_new_students":True,
            "train_folds":[1,2,4],"heldout_fold":0,
            "original_teacher_rows":len(held_truth),
            "unweighted_111D_baseline":{
                "fidelity":unweighted["heldout_teacher_agreement"],
                "poly_macro_teacher_recall":unweighted["macro_K2_K3_K4_teacher_recall"]},
            "GOAT_new_audio_groups":len(goat_ids),"variants":{},
            "truth_K_read":False,"GOAT_MIDI_or_labels_read":False,
            "true_Exact_K_scored":False,"S80_independent_score":None,
            "S80_promoted":False}
    out.mkdir(parents=True)
    for name in VARIANTS:
        x=np.concatenate([hist[f][2][name] for f in (1,2,4)])
        teacher=np.concatenate([hist[f][1] for f in (1,2,4)])
        require(x.shape[0]==len(teacher) and x.shape[1] in (356,846),
                "student feature shape")
        model=classifier()
        model.fit(x,teacher)
        pred=probabilities(model,held_views[name]).argmax(1).astype(np.int8)
        goat=probabilities(model,goat_views[name])
        metrics=measure(held_truth,baseline,pred)
        metrics["input_dimensions"]=x.shape[1]
        metrics["GOAT_K_histogram"]=np.bincount(goat.argmax(1),minlength=7).tolist()
        joblib.dump(model,out/(name+".joblib"),compress=3)
        np.savez_compressed(out/(name+"_GOAT.npz"),row_id=goat_ids,
                            predicted_K=goat.argmax(1).astype(np.int8),
                            probabilities=goat.astype(np.float32))
        metrics["model_sha256"]=hashlib.sha256(
            (out/(name+".joblib")).read_bytes()).hexdigest()
        report["variants"][name]=metrics
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],"baseline":report["unweighted_111D_baseline"],
        "variants":{name:{k:d[k] for k in
            ("heldout_teacher_agreement","macro_K2_K3_K4_teacher_recall",
             "paired_fixes_vs_s129","paired_breaks_vs_s129",
             "paired_net_teacher_fidelity","GOAT_K_histogram")}
            for name,d in report["variants"].items()},
        "GOAT_MIDI_or_labels_read":False,"S80_promoted":False},indent=2))
    return report

def selftest():
    seq=np.zeros((2,42,49),np.float32)
    f=flow_profile(seq);m,_=audio_temporal_views(seq)
    assert f.shape==(2,245) and m.shape==(2,490)
    assert np.column_stack((np.zeros((2,111)),f,m)).shape==(2,846)
    print("S131 PASS: genuine flow/morphology shapes and no label input")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for item in ("original-features","original-s5","s116","s97","s129","out"):
        ap.add_argument("--"+item,type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        require(all(x is not None for x in
            (a.original_features,a.original_s5,a.s116,a.s97,a.s129,a.out)),
            "pinned historical teacher and fresh audio sources required")
        run(a.original_features,a.original_s5,a.s116,a.s97,a.s129,a.out)
