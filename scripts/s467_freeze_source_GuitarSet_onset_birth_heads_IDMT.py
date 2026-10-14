"""S467 — transfer four GuitarSet-only attack-born event networks to true IDMT.
Freeze 506 independent probabilities before opening XML; source S80 and source
event list *never* modified. Only emit experimental optional retain flags.
No threshold adapted to IDMT; all cuts come from S466 nested held songs.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import need,load
from scripts.s458_train_independent_audio_only_K_blind_songheld import inputs
from scripts.s466_train_separate_acoustic_birth_v_continuation_guitarset import predict_birth
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(audio,audio_report,s80,s80_report,models,models_report,out):
    need(not out.exists(),"S467 predictions already frozen")
    a=load(audio);p=load(s80)
    ar=json.loads(audio_report.read_text())
    pr=json.loads(s80_report.read_text())
    mr=json.loads(models_report.read_text())
    need(ar["status"]=="S283_EXTERNAL_ORIGINAL_S97_NATIVE_506_ONLY"
         and ar["output_SHA256"]==sha(audio)
         and ar["labels_MIDI_XML_read"] is False
         and pr["status"]=="S297_COMPLETE_GENUINE_ORIGINAL_TRAINED_S80_UNSEEN_NATIVE_IDMT"
         and mr["status"]=="S466_GUITARSET_TRAIN_ONLY_COMPOSITION_DISJOINT_BINARY_BIRTH_HEAD"
         and mr["IDMT_not_read_during_training_or_threshold_selection"]
         and mr["model_no_original_S80_K_as_input"]
         and mr["no_promotion"],
         "S467 source IDMT no-label original K or S466 train provenance changed")
    ids=a["row_id"].astype(str)
    K=p["s80_K"].astype(np.int8)
    need(len(ids)==506 and len(set(ids))==506
         and np.array_equal(ids,p["row_id"].astype(str))
         and np.bincount(K,minlength=7).tolist()==[456,26,11,7,6,0,0],
         "S467 original pretrained S80 K native array drift")
    X=inputs(a,K)
    scores=np.zeros((4,len(ids)),np.float32)
    cuts=np.full(4,np.nan,np.float32)
    proofs=[]
    for fold in range(4):
        path=models/f"S466_birth_vs_nonbirth_fold{fold}.joblib"
        record=mr["all_folds"][fold]
        need(sha(path)==record["model_SHA256"] and record["fold"]==fold,
             "S467 source trained fold checkpoint modified")
        obj=joblib.load(path)
        need(obj["status"]=="S466_SOURCE_GUITARSET_TRAINED_ONLY_ONSET_BIRTH"
             and obj["fold"]==fold
             and obj["no_original_K_feature"]
             and obj["no_training_on_IDMT"]
             and obj["train_feature_width"]==1090,
             "S467 acoustic-only original note-birth checkpoint unsupported")
        prob=predict_birth(obj["model"],X)
        scores[fold]=prob
        if record["inner_selected_threshold"] is not None:
            cut=float(record["inner_selected_threshold"])
            need(cut==obj["inner_selected_cut"] and 0<cut<1,
                 "S467 source safety threshold used without original GuitarSet calibration")
            cuts[fold]=cut
        proofs.append(dict(fold=fold,model_SHA256=sha(path),
                           original_GuitarSet_inner_threshold=(
                               float(cuts[fold]) if np.isfinite(cuts[fold]) else None)))
    # Source K>0 always retained. Source K0 is considered for *provisional*
    # removal only if ALL four independently trained models had safe
    # calibration, and ALL four predict no new birth.
    all_calibrated=bool(np.isfinite(cuts).all())
    suppress=np.zeros(len(K),bool)
    if all_calibrated:
        suppress=(K==0)&np.all(scores<cuts[:,None],axis=0)
    keep=~suppress
    need(np.all(keep[K>0]) and len(ids)==506,
         "S467 no actual original K>0 note can be suppressed or mutated")
    out.mkdir(parents=True)
    file=out/"S467_ORIGINAL_S80_IDMT_506_SEPARATE_BIRTH_PROB_FROZEN.npz"
    np.savez_compressed(file,row_id=ids,member=a["member"],
      original_S80_K=K,start_samples=a["start_samples"],
      individual_fold_birth_probabilities=scores,
      original_trainonly_thresholds=cuts,
      gated_retain_suggestion=keep)
    rep=dict(status="S467_FROZEN_SOURCE_K_BLIND_BIRTH_GATE_IDMT_NO_XML",
      source_native_events=len(K),source_recording_count=5,
      original_K_hist=np.bincount(K,minlength=7).tolist(),
      four_distinct_songheld_GuitarSet_models=len(proofs),
      all_four_had_safe_original_GuitarSet_calibration=all_calibrated,
      original_true_S80_K_changed=0,
      original_S80_K_nonzero_events_protected=int(np.sum(K>0)),
      proposed_K0_event_suppress_count=int(suppress.sum()),
      source_S80_K0_proposed_retain_count=int(np.sum(keep&(K==0))),
      experimental_retention_ONLY_not_applied_to_note_decoder=True,
      IDMT_XML_never_read_for_inference_or_thresholds=True,
      no_external_label_supervision=True,
      no_new_source_K_or_note_predictions=True,
      model_proofs=proofs,
      source_SHA=dict(original_IDMT_acoustic=sha(audio),
                      original_S80_predictions=sha(s80),
                      GuitarSet_binary_model_report=sha(models_report),
                      frozen_predictions=sha(file)),
      no_promotion=True)
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("S467 NOXML FREEZE",json.dumps(dict(
        original_506_K0=456,
        calibrated_all_four=all_calibrated,
        proposed_suppressed=rep["proposed_K0_event_suppress_count"],
        original_K_changed=0)),flush=True)

def selftest():
    k=np.array([0,1,0,3],np.int8)
    scores=np.array([[.01,.01,.20,.01]]*4,dtype=float)
    keep=~((k==0)&np.all(scores<.05,axis=0))
    need(keep.tolist()==[False,True,True,True],
         "S467 original K-nonzero and high-score true attack protection")
    print("S467 SELFTEST PASS original S80 positive notes cannot be removed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for x in ("audio","audio-report","s80","s80-report","models","models-report","out"):
        p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for x in ("audio","audio_report","s80","s80_report","models","models_report","out"):
            need(getattr(a,x)is not None,"S467 missing --"+x)
        run(a.audio,a.audio_report,a.s80,a.s80_report,
            a.models,a.models_report,a.out)
