"""S479: freeze independent IDMT V88 K0 rescue, *never* load original XML.
The four original S478 GuitarSet models, full 1090D S283 source
acoustic bank, and original V88 S277 identities are frozen before later
S480 paired true-attack scoring. Two fixed ensemble consensus rules (4/4,
3/4) are reported; none is selected based on IDMT labels.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import numpy as np,joblib
from scripts.s402_k_input_first_supervised_midi_pitch_network import need,load
from scripts.s458_train_independent_audio_only_K_blind_songheld import inputs
from scripts.s478_fit_original_V88_K0_rescue_3class_songheld import distribution
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidates(K,p,thresholds,vote):
    original=np.asarray(K,np.int8)
    need(len(p)==4 and p.shape==(4,len(K),3) and len(thresholds)==4,
         "S479 original 4 GuitarSet independent K0 source probability contract changed")
    endorses=np.zeros((4,len(K)),np.int8)
    for fold in range(4):
        cut=thresholds[fold]
        if cut is None:continue
        labels=np.argmax(p[fold],axis=1)
        strength=np.max(p[fold],axis=1)
        endorses[fold]=np.where((original==0)&(labels>=1)&(strength>=cut),labels,0)
    out=original.copy()
    # Consensus cannot resolve a split between K1 and K2+ without new
    # data. Abstain on ties to preserve both source and true poly attacks.
    for i in np.flatnonzero(original==0):
        counts=np.bincount(endorses[:,i],minlength=3)
        winning=int(np.argmax(counts[1:])+1)
        if counts[winning]>=vote and counts[winning]>counts[3-winning]:
            out[i]=winning
    need(np.array_equal(out[K>0],K[K>0]),"S479 source V88 K1+ modified")
    return out, endorses
def run(audio,orig,orig_report,models,models_report,out):
    need(not out.exists(),"S479 source-IDMT external predictions already sealed")
    z=load(audio);o=load(orig)
    meta=json.loads(orig_report.read_text())
    training=json.loads(models_report.read_text())
    ids=z["row_id"].astype(str);ref=o["incumbent_K"].astype(np.int8)
    need(meta["status"]=="S277_TRUE_OFFICIAL_FREEZE_ON_DATASET3_FIVE_WAV_UNLABELLED"
         and meta["sealed_original_predictions_SHA256"]==sha(orig)
         and training["status"]=="S478_GUITARSET_ONLY_V88_K0_SOURCE_RESIDUAL_4FOLD"
         and training["no_external_IDMT_or_MAESTRO_truth_used"]
         and training["no_promotion"] and len(training["four_GuitarSet_train_only_outerfolds"])==4
         and np.array_equal(ids,o["row_id"].astype(str))
         and np.array_equal(z["member"].astype(str),o["member"].astype(str))
         and len(ids)==506
         and np.array_equal(z["incumbent_freeze_K"].astype(np.int8),ref),
         "S479 truly original IDMT V88 and GuitarSet only K0 rescue proof failed")
    X=inputs(z,ref)
    foldp=np.zeros((4,len(ids),3),np.float32)
    cuts=[]
    for i in range(4):
        path=models/f"S478_source_V88_K0_rescue_fold{i}.joblib"
        spec=training["four_GuitarSet_train_only_outerfolds"][i]
        need(spec["fold"]==i and sha(path)==spec["original_model_SHA256"],
             "S479 trained source fold checkpoint is not SHA-original")
        pack=joblib.load(path)
        need(pack["status"]=="S478_GUITARSET_K0_SPECIALIST"
             and pack["no_IDMT_or_MAESTRO_training"]
             and pack["outer_fold"]==i and pack["original_source_class"]==0,
             "S479 risk of unseen-source labels entering pretrained model")
        foldp[i]=distribution(pack["model"],X)
        cuts.append(pack["chosen_guitarset_only_calibration_threshold"])
    candidates3,votes3=candidates(ref,foldp,cuts,3)
    candidates4,votes4=candidates(ref,foldp,cuts,4)
    need(np.array_equal(ref[candidates3==ref],candidates3[candidates3==ref])
         and np.array_equal(ref[ref>0],candidates3[ref>0])
         and np.array_equal(ref[ref>0],candidates4[ref>0]),
         "S479 original V88 non-K0 events not locked")
    out.mkdir(parents=True)
    file=out/"S479_FROZEN_IDMT_506_K0_RESIDUAL_3AND4VOTES_PREXML.npz"
    np.savez_compressed(file,row_id=ids,member=o["member"],
        start_samples=o["start_samples"],original_V88_K=ref,
        proposed_V88_K_consensus3=candidates3,
        proposed_V88_K_consensus4=candidates4,
        original_GuitarSet_trainonly_cut=np.asarray([
            v if v is not None else np.nan for v in cuts],np.float32),
        vote_by_fold=votes3,original_GuitarSet_models_prob=foldp)
    report=dict(status="S479_IDMT_ORIGINAL_V88_K0_RESCUE_AFTER_GUITARSET_TRAIN_NOXML",
        original_external_events=len(ids),
        original_native_5WAV=5,
        original_source_V88_K_hist=np.bincount(ref,minlength=7).tolist(),
        original_V88_K0_count=int(np.sum(ref==0)),
        GuitarSet_only_thresholds=cuts,
        three_vote_K0_to_K1=int(np.sum((ref==0)&(candidates3==1))),
        three_vote_K0_to_K2=int(np.sum((ref==0)&(candidates3==2))),
        four_vote_K0_to_K1=int(np.sum((ref==0)&(candidates4==1))),
        four_vote_K0_to_K2=int(np.sum((ref==0)&(candidates4==2))),
        original_V88_K1toK6_untouched=True,
        original_S80_complete_NOT_TESTED_here=True,
        IDMT_XML_or_labels_NEVER_OPENED=True,
        no_new_training_or_external_threshold_selection=True,
        no_live_onset_or_offset_modifications=True,
        experimental_predictions_only_not_applied=True,
        no_promotion=True,
        frozen_predictions_SHA256=sha(file),
        SHA256=dict(original_V88=sha(orig),original_S97_audio=sha(audio),
            S478_source_model_report=sha(models_report)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S479 ORIGINAL IDMT FROZEN",json.dumps(dict(
        total=506,original_K0=report["original_V88_K0_count"],
        threeVote_K1=report["three_vote_K0_to_K1"],
        threeVote_K2=report["three_vote_K0_to_K2"],
        fourVote_K1=report["four_vote_K0_to_K1"],
        fourVote_K2=report["four_vote_K0_to_K2"],
        MIDI_read=False)),flush=True)
def selftest():
    old=np.array([0,0,3],np.int8)
    prob=np.zeros((4,3,3))
    prob[:,0,1]=1
    prob[:3,1,2]=1;prob[3,1,0]=1
    prob[:,:,0]+=np.array([0.,0.,1.])[None]
    a,_=candidates(old,prob,[.5]*4,3)
    b,_=candidates(old,prob,[.5]*4,4)
    need(a.tolist()==[1,2,3] and b.tolist()==[1,0,3],
         "S479 consensus source K0 3/4 and 4/4 labels wrong")
    print("S479 SELFTEST PASS protected original nonzero V88 K")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("audio","orig","orig-report","models","models-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("audio","orig","orig_report","models","models_report","out"):
            need(getattr(a,k)is not None,"S479 missing "+k)
        run(a.audio,a.orig,a.orig_report,a.models,a.models_report,a.out)
