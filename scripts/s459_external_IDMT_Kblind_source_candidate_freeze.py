"""S459 -- deploy the S458 NEW K candidate on genuine 506 external IDMT V88.

NO TRAINING, NO XML, NO ADAPTIVE EXTERNAL THRESHOLDS. Input is exact
S283 acoustic bank and immutable S297 original source S80 decisions.
All four S458 GuitarSet-only fold models vote separately; candidate
K change requires agreement by at least 3 separate fold checkpoints.
Original S80 predictions remain available unchanged. This does not
silently mutate the K-locked S450/S446 musical-note predictions.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import joblib,numpy as np
from scripts.s458_train_K_source_multiclass_morphology_songheld import (
    inputs,proba_model,candidate)
from scripts.s402_k_input_first_supervised_midi_pitch_network import need,load
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

N=506
MIN_VOTES=3

def vote(original,hypotheses):
    counts=collections.Counter(int(v) for v in hypotheses if v!=original)
    if not counts:return int(original),0
    k,n=counts.most_common(1)[0]
    return (int(k),n) if n>=MIN_VOTES else (int(original),n)

def run(source,source_report,s80,s80_report,heads,trained_report,out):
    need(not out.exists(),"S459 original external inferred-K archive immutable")
    a=load(source);z=load(s80)
    r=json.loads(source_report.read_text())
    s=json.loads(s80_report.read_text())
    teacher=json.loads(trained_report.read_text())
    need(r["status"]=="S283_EXTERNAL_ORIGINAL_S97_NATIVE_506_ONLY"
         and r["labels_MIDI_XML_read"] is False and r["output_SHA256"]==sha(source)
         and s["status"]=="S297_COMPLETE_GENUINE_ORIGINAL_TRAINED_S80_UNSEEN_NATIVE_IDMT"
         and teacher["status"]=="S458_K_BLIND_1090D_ACOUSTIC_ABLATION_COMP_DISJOINT_CANDIDATE"
         and teacher["no_external_IDMT_labels_or_weights"]
         and teacher["no_promotion"] and teacher["dimensions"]==1090
         and teacher["source_K_not_a_network_feature"],
         "S459 must use only frozen original IDMT sources and train-only heads")
    ids=a["row_id"].astype(str);K=z["s80_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N
         and np.array_equal(ids,z["row_id"].astype(str))
         and np.isin(K,range(7)).all()
         and len(set(a["member"].astype(str)))==5,
         "S459 original S80 native 506 IDMT K schema or source identity")
    X=inputs(a,K)
    proposed=np.tile(K[None],(4,1)).astype(np.int8)
    modelproof=[]
    for f in range(4):
        spec=teacher["all_model_folds"][f]
        path=heads/f"S458_acoustic_K_head_fold{f}.joblib"
        need(spec["fold"]==f and sha(path)==spec["model_SHA256"],
             "S459 cannot substitute or edit the original GuitarSet source model")
        obj=joblib.load(path)
        need(obj["held_fold"]==f and obj["original_S80_K_required"]
             and obj["feature_width"]==1090 and obj["no_external_IDMT_training_or_labels"],
             "S459 original morphology K-model ABI mismatch")
        thresholds=obj["threshold_by_predicted_source_K"]
        need(all(int(k) in (0,1,2,3,4) and 0<=float(v)<=1
             for k,v in thresholds.items()),
             "S459 source-K gate selected on unknown external labels")
        scores=proba_model(obj["model"],X)
        proposed[f]=candidate(K,scores,thresholds)
        modelproof.append(dict(fold=f,source_GuitarSet_model_SHA256=sha(path),
              new_source_label_free_actions=int(np.sum(proposed[f]!=K)),
              training_only_thresholds=thresholds))
    updated=K.copy()
    agreements=np.zeros(N,np.int8)
    for i in range(N):
        selected,n=vote(int(K[i]),proposed[:,i])
        updated[i]=selected
        agreements[i]=n
    need(np.isin(updated,range(7)).all()
         and np.array_equal(K,z["s80_K"])
         and np.array_equal(updated,updated.astype(np.int8)),
         "S459 original true pretrained S80 output modified or K out of bounds")
    out.mkdir(parents=True)
    path=out/"S459_external_IDMT_sourceK_frozen_candidate.npz"
    np.savez_compressed(path,row_id=ids,member=a["member"],
        start_samples=a["start_samples"],
        original_S80_K=K,
        S458_candidate_K=updated,
        four_fold_candidate_K=proposed,
        consensus_votes=agreements)
    change=[]
    for i in np.flatnonzero(updated!=K):
        change.append(dict(row_id=str(ids[i]),
            original_S80_K=int(K[i]),candidate_K=int(updated[i]),
            votes=int(agreements[i]),song_SHA=str(a["member"][i])))
    report=dict(status="S459_REAL_EXTERNAL_IDMT_MORPHOLOGY_K_CANDIDATE_FROZEN_NO_XML",
        source_native_events=N,recordings=5,
        original_S80_K_predictions_preserved_separately=True,
        nonofficial_candidate_K_changes=len(change),
        candidate_K_changes_by_pair=dict(collections.Counter(
            f"{r['original_S80_K']}->{r['candidate_K']}" for r in change)),
        original_K_hist=np.bincount(K,minlength=7).tolist(),
        candidate_K_hist=np.bincount(updated,minlength=7).tolist(),
        consensus_min_distinct_trained_models=MIN_VOTES,
        original_note_name_decoder_S450_unchanged=True,
        K0_background_truth_not_assumed=True,
        independent_acoustic_network_has_no_original_source_K_feature=True,
        annotations_XML_MIDI_never_read=True,
        no_new_music_labels_used_for_training_or_thresholds=True,
        source_IDMT_dataset3_annotations_previously_exposed_in_S299=True,
        candidate_not_official_ExactK=True,no_promotion=True,
        all_model_proofs=modelproof,
        changed_source_event_ledger=change,
        checksums=dict(S283=sha(source),S297=sha(s80),
         training_report=sha(trained_report),frozen_output=sha(path)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S459 EXTERNAL FREEZE",json.dumps(dict(
      original_K_hist=report["original_K_hist"],
      candidate_K_hist=report["candidate_K_hist"],
      candidate_changes=len(change),
      transitions=report["candidate_K_changes_by_pair"],
      original_S80_untouched=True)),flush=True)

def selftest():
    need(vote(2,[3,3,3,2])==(3,3),"S459 3 fold consensus required")
    need(vote(2,[3,3,4,4])==(2,2),"S459 2 fold votes cannot alter S80")
    need(vote(0,[0,0,0,0])==(0,0),"S459 no candidate preserves silence")
    print("S459 SELFTEST PASS complete abstention, original S80 K0/K2 K locks")

if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for x in ("source","source-report","s80","s80-report",
        "heads","trained-report","out"):
        q.add_argument("--"+x,type=Path)
    a=q.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("source","source_report","s80","s80_report",
                    "heads","trained_report","out"):
            need(getattr(a,key)is not None,"S459 missing --"+key)
        run(a.source,a.source_report,a.s80,a.s80_report,
            a.heads,a.trained_report,a.out)
