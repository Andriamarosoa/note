"""S517 independent replay of trained 49-note acoustic-generator note decisions.

Verify EIGHT training-only original-song model SHA outputs, per-pitch
probabilities frozen in S516, and metrics with independently implemented
note-set Exact-K and note-level F1. Use only original S505 correct MIDI
truth for post-evaluation, NEVER in acoustic head inference.

Reject source promotion unless notes/no false new onsets are proved on a
genuinely unseen multi-instrument corpus, unavailable in this cohort.
"""
from __future__ import annotations
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reassemble(prob,original,policy):
    previous=original[:,40:89]
    if policy=="threshold050":propose=(prob>=.5)
    elif policy=="threshold070":propose=(prob>=.7)
    elif policy=="BP_guarded_add085_remove015":
        propose=previous.copy()
        propose[(~previous)&(prob>=.85)]=True
        propose[previous&(prob<=.15)]=False
    else:raise ValueError("unknown frozen S516 policy")
    empty=np.flatnonzero(~propose.any(axis=1))
    propose[empty]=previous[empty]
    out=np.zeros_like(original)
    out[:,40:89]=propose
    return out
def score(y,old,new,srcK):
    trueK=y.sum(1);oldK=old.sum(1);newK=new.sum(1)
    base=np.all(y==old,axis=1);after=np.all(y==new,axis=1)
    changed=np.any(old!=new,axis=1)
    def single(mask):
        t=int(np.sum((y&new)[mask]));p=int(np.sum((~y&new)[mask]))
        n=int(np.sum((y&~new)[mask]))
        return dict(n=int(mask.sum()),
            baseline_full_chord_correct=int(np.sum(mask&base)),
            proposed_full_chord_correct=int(np.sum(mask&after)),
            fixes=int(np.sum(mask&~base&after&changed)),
            regressions=int(np.sum(mask&base&~after&changed)),
            changed=int(np.sum(mask&changed)),
            wrong_to_wrong=int(np.sum(mask&~base&~after&changed)),
            BP_ExactK=int(np.sum(mask&(oldK==trueK))),
            proposed_ExactK=int(np.sum(mask&(newK==trueK))),
            source_original_K_exact=int(np.sum(mask&(srcK==trueK))),
            true_note_TP=t,note_FP_on_known_true_attacks=p,
            note_FN_on_known_true_attacks=n,
            note_F1_on_known_true_attacks=float(2*t/max(1,2*t+p+n)))
    return dict(overall=single(np.ones(len(y),bool)),
           true_poly=single(trueK>=2),
           trueK={str(i):single(trueK==i) for i in range(7)})
def run(a):
    need(not a.out.exists(),"S517 no overwrite for source independent replay")
    source=json.loads(a.source_report.read_text())
    study=json.loads(a.study_report.read_text())
    need(source["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
         and source["audio_proposal_truth_table_SHA256"]==sha(a.source)
         and study["status"]==
             "S516_NEW_TRAINED_RAW_WAV_MULTILABEL_NOTE_GENERATOR_EIGHT_RECORDING_OOF"
         and study["original_head_labels_sha256"]==sha(a.source)
         and study["predictions_sha256"]==sha(a.predictions)
         and study["full_8_recording_crossfit"]
         and study["no_promotion"] and len(study["folds"])==8,
         "S517 original real SHA source and model outputs must match source before truth")
    for fold in study["folds"]:
        f=a.models/f"S516_fold{fold['fold']}_ACTUAL_RAW_NOTE_MULTILABEL_EXPERT.joblib"
        need(f.exists() and sha(f)==fold["checkpoint_SHA256"]
             and fold["train_original_recording_count"]==7,
             "S517 original actual trained held-song model fingerprint changed")
    with np.load(a.source,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    ids=data["event_id"].astype(str)
    lookup={x:i for i,x in enumerate(ids)}
    need(len(ids)==len(lookup),"S517 original music canonical ids duplicated")
    N=len(ids)
    pieces=data["corpus"].astype(str)
    grouped=data["recording_sha"].astype(str)
    truth=data["truth_note_bits"].astype(bool)
    before=data["head_note_bits"][np.arange(N),
              data["selected_BP_baseline_head"].astype(int)].astype(bool)
    sourceK=data["source_K"].astype(int)
    predictions=[json.loads(x) for x in a.predictions.read_text().splitlines()]
    need(len(predictions)==N,"S517 original 265 match positive events must be present")
    matrix=np.zeros((N,49),np.float32);used=np.zeros(N,bool)
    for p in predictions:
        id=p["canonical_event_id"]
        need(id in lookup,"S517 frozen predicted event absent original source")
        i=lookup[id];need(not used[i] and p["source_rec"]==grouped[i],
                          "S517 duplicate neural prediction or source recording changed")
        used[i]=True
        q=np.asarray(p["probabilities_by_MIDI_40to88"],np.float32)
        need(q.shape==(49,) and np.isfinite(q).all()
             and np.all((q>=0)&(q<=1)),"S517 invalid learned MIDI note probabilities")
        matrix[i]=q
    need(used.all(),"S517 incomplete independent source note predictions")
    output={}
    for corpus in ("guitar_S80","piano_V88"):
        ids_=np.flatnonzero(pieces==corpus)
        output[corpus]={}
        for policy in ("threshold050","threshold070","BP_guarded_add085_remove015"):
            changed=reassemble(matrix[ids_],before[ids_],policy)
            actual=score(truth[ids_],before[ids_],changed,sourceK[ids_])
            reference=study["by_source"][corpus][policy]["overall"]
            ours=actual["overall"]
            for key,m in (("old_exact_full_note_sets","baseline_full_chord_correct"),
                ("new_exact_full_note_sets","proposed_full_chord_correct"),
                ("fixes","fixes"),("regressions","regressions"),
                ("note_F1_conditional","note_F1_on_known_true_attacks"),
                ("new_ExactK","proposed_ExactK")):
                expected=reference[key]
                got=ours[m]
                need(abs(float(expected)-float(got))<1e-10,
                     f"S517 independently recomputed {corpus}/{policy}/{key} differs")
            actual["unmatched_true_attacks"]=(source["by_source"][corpus][
                    "annotated_true_attacks_NOT_matched"])
            actual["all_original_true_attacks"]=(source["by_source"][corpus][
                    "truth_original_positive_attacks"])
            actual["not_entire_audio_event_F1"]=True
            output[corpus][policy]=actual
        print("S517 INDEPENDENT NOTE GENERATOR VERDICT",json.dumps(
            dict(corpus=corpus,policies={
                k:{"overall":v["overall"],"poly":v["true_poly"]}
                for k,v in output[corpus].items()})),flush=True)
    a.out.mkdir(parents=True)
    anyreg=any(v["overall"]["regressions"]>0 for d in output.values() for v in d.values())
    final=dict(status="S517_EIGHT_SHA_VERIFIED_RAW_WAV_GENERATOR_NOTE_REPLAY",
        trained_original_8_fold_model_SHA_verified=True,
        independently_recomputed_all_full_chords_and_pitch_F1=True,
        separate_exactK_and_actual_pitch_identifications=True,
        all_missing_piano_attacks_remain_in_original_denominator=True,
        previously_exposed_not_new_blind=True,
        cannot_measure_false_note_onsets_on_unpaired_model_groups=True,
        conditional_positive_onset_note_F1_not_event_F1=True,
        observed_any_full_chord_regressions=anyreg,
        source_S80_Live_not_modified=True,no_promotion=True,
        by_source=output,S516_SHA256=sha(a.study_report))
    (a.out/"report.json").write_text(json.dumps(final,indent=2)+"\n")
def selftest():
    y=np.zeros((2,128),bool);y[:,52]=True
    b=y.copy();b[1,52]=False;b[1,55]=True
    x=score(y,y,b,np.array([1,1]))
    need(x["overall"]["regressions"]==1 and
         x["overall"]["BP_ExactK"]==x["overall"]["proposed_ExactK"],
         "S517 wrong pitch despite correct K still a regression")
    print("S517 SELFTEST PASS independent full-note regression parity")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("source","source-report","study-report","predictions","models","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("source","source_report","study_report","predictions","models","out"):
            need(getattr(a,k)is not None,"S517 missing --"+k)
        run(a)
