"""S520 truly independent (no TF, no re-fit) verification of S519:
14-head S509 attention with separately nested S516 audio generator.

Open checkpoint SHA and original S505 MIDI note truth ONLY after verifying
S519 frozen probabilities. Recompute note-by-note and polyphonic edits,
Exact-K and micro pitch F1 for each original source and true K0..K6.
No scores on 102 original piano notes lacking a canonical BP event.
Do NOT treat unmatched acoustic groups as known false alarms.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(p,old,g,policy):
    prior=old[:,40:89].copy()
    if policy=="attention050":decoded=p>=.50
    elif policy=="attention070":decoded=p>=.70
    elif policy=="protected_085":
        decoded=prior.copy()
        decoded[(~prior)&(p>=.85)&(g>=.75)]=True
        decoded[prior&(p<=.15)&(g<=.15)]=False
    else:raise ValueError("S520 unknown policy (cannot tune after reading MIDI)")
    empty=~decoded.any(1)
    decoded[empty]=prior[empty]
    out=old.copy()
    out[:,40:89]=decoded
    return out
def score(truth,old,output,sourceK):
    trueK=truth.sum(1);K0=old.sum(1);K1=output.sum(1)
    correct0=(truth==old).all(1);correct1=(truth==output).all(1)
    change=np.any(old!=output,axis=1)
    def count(mask):
        tp=int(np.sum((truth&output)[mask]))
        fp=int(np.sum((~truth&output)[mask]))
        fn=int(np.sum((truth&~output)[mask]))
        return dict(n=int(mask.sum()),
             old_full_chords=int(np.sum(mask&correct0)),
             new_full_chords=int(np.sum(mask&correct1)),
             fixes=int(np.sum(mask&~correct0&correct1&change)),
             regressions=int(np.sum(mask&correct0&~correct1&change)),
             wrong_to_wrong=int(np.sum(mask&~correct0&~correct1&change)),
             changed=int(np.sum(mask&change)),
             S80orV88_original_ExactK=int(np.sum(mask&(sourceK==trueK))),
             BP_baseline_ExactK=int(np.sum(mask&(K0==trueK))),
             selected_ExactK=int(np.sum(mask&(K1==trueK))),
             correct_note_pitches=tp,extra_note_pitches=fp,missed_note_pitches=fn,
             note_F1_conditional_true_attacks=2*tp/max(1,2*tp+fp+fn),
             note_precision_conditional_true_attacks=tp/max(1,tp+fp),
             note_recall_conditional_true_attacks=tp/max(1,tp+fn))
    return dict(overall=count(np.ones(len(truth),bool)),
         poly=count(trueK>=2),
         by_true_K={str(k):count(trueK==k) for k in range(7)})
def run(a):
    need(not a.out.exists(),"S520 do not overwrite independent original music note audit")
    src=json.loads(a.source_report.read_text())
    trained=json.loads(a.trained_report.read_text())
    need(src["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
      and trained["status"]=="S519_NESTED_S516_RAW_GENERATOR_INSIDE_S509_14HEAD_NEURAL_SELECTOR"
      and src["audio_proposal_truth_table_SHA256"]==sha(a.source)
      and trained["source_SHA256"]["S505_original_supervised_source"]==sha(a.source)
      and trained["source_SHA256"]["OOF_frozen_generator_neural_prediction_SHA256"]==sha(a.preds)
      and trained["no_promotion"] and len(trained["folds"])==8,
      "S520 all original independently trained note/generator predictions require SHA verification")
    for group in trained["folds"]:
        f=int(group["fold"])
        raw=a.models/f"S519_fold{f}_S516_INSIDE_14HEAD_RAW_GENERATOR.joblib"
        att=a.models/f"S519_fold{f}_S509_NEURAL_14HEAD_ATTENTION.weights.h5"
        need(raw.is_file() and att.is_file()
          and sha(raw)==group["generator_model_SHA256"]
          and sha(att)==group["attention_model_SHA256"]
          and group["training_recording_count"]==7
          and group["inner_train_recording_count_per_generator"]==6
          and group["nested_train_OOF_generator_models"]==7
          and group["outer_held_source_excluded_from_BOTH_models"],
          "S520 model checkpoint or nested data split source provenance changed")
    with np.load(a.source,allow_pickle=False) as z:
        d={key:z[key].copy() for key in z.files}
    N=len(d["event_id"])
    index={v:i for i,v in enumerate(d["event_id"].astype(str))}
    need(N==265 and len(index)==N,
         "S520 original authentic positive onset pairing changed")
    recordings=d["recording_sha"].astype(str)
    corpus=d["corpus"].astype(str)
    p=np.full((N,49),np.nan,np.float32)
    g=np.full((N,49),np.nan,np.float32)
    used=np.zeros(N,bool)
    for line in a.preds.read_text().splitlines():
        row=json.loads(line)
        ev=row["canonical_event_id"]
        need(ev in index,"S520 new neural recording predicted unknown musical event")
        i=index[ev]
        need(not used[i] and row["original_source_recording_SHA"]==recordings[i]
             and row["source_corpus"]==corpus[i],
             "S520 original note onset repeated or source shifted between folds")
        used[i]=True
        p[i]=np.asarray(row["per_MIDI_pitch_40to88_attention_probability"],np.float32)
        g[i]=np.asarray(row["per_MIDI_pitch_40to88_generator_probability"],np.float32)
    need(used.all() and np.isfinite(p).all() and np.isfinite(g).all()
         and np.all((p>=0)&(p<=1)) and np.all((g>=0)&(g<=1)),
         "S520 trained 14th-head probabilities not complete or invalid")
    bp=d["head_note_bits"][np.arange(N),d["selected_BP_baseline_head"].astype(int)].astype(bool)
    truth=d["truth_note_bits"].astype(bool)
    sourceK=d["source_K"].astype(int)
    result={};changes=[]
    for sc in ("guitar_S80","piano_V88"):
        q=np.flatnonzero(corpus==sc);out={}
        for policy in ("attention050","attention070","protected_085"):
            proposed=decode(p[q],bp[q],g[q],policy)
            z=score(truth[q],bp[q],proposed,sourceK[q])
            expected=trained["by_source"][sc][policy]
            for field in ("old_full_chords","new_full_chords","fixes","regressions",
                "selected_ExactK","note_F1_conditional_true_attacks"):
                got=z["overall"][field];want=expected["overall"][field]
                need(abs(float(got)-float(want))<1e-12,
                     f"S520 independent SHA parity fails for {sc}/{policy}/{field}")
            out[policy]=z
            for loc,i in enumerate(q):
                old=bp[i];new=proposed[loc]
                if np.array_equal(old,new):continue
                before=bool(np.array_equal(old,truth[i]))
                after=bool(np.array_equal(new,truth[i]))
                changes.append(dict(source=sc,policy=policy,
                   original_event_id=str(d["event_id"][i]),
                   source_recording_SHA=str(recordings[i]),
                   original_pitch_MIDI=np.flatnonzero(old).tolist(),
                   proposed_pitch_MIDI=np.flatnonzero(new).tolist(),
                   pitches_added=np.flatnonzero(~old&new).tolist(),
                   pitches_removed=np.flatnonzero(old&~new).tolist(),
                   true_pitch_MIDI_FOR_AUDIT_ONLY=np.flatnonzero(truth[i]).tolist(),
                   trueK_for_audit_only=int(truth[i].sum()),
                   corrected=int(not before and after),
                   regressed=int(before and not after)))
        result[sc]=out
        print("S520 INDEPENDENT NESTED 14-HEAD CHORD AND PITCH AUDIT",
          json.dumps(dict(corpus=sc,policies={p:{
             "overall":s["overall"],"poly":s["poly"]} for p,s in out.items()})),flush=True)
    a.out.mkdir(parents=True)
    f=a.out/"S520_TRUE_MUSICAL_NOTE_ADDITIONS_DELETIONS_AND_REGRESSIONS.jsonl"
    f.write_text("".join(json.dumps(c)+"\n" for c in changes))
    report=dict(status="S520_INDEPENDENT_REPLAY_NESTED_S509_S516_14HEAD_ATTENTION",
       original_trained_model_checkpoint_SHA_verified_16=True,
       original_nested_inner_train_source_only_checks_verified=True,
       no_retraining_or_threshold_fitting_on_true_eval_notes=True,
       independently_verified_exact_full_note_sets_and_midi_note_F1=True,
       all_regressions_accounted_for=True,
       original_source_S80_and_Live_unchanged=True,
       retrospective_sources_not_blind=True,
       missing_102_MAESTRO_true_onsets_not_implicitly_recovered=True,
       event_F1_and_negative_onset_precision_still_unmeasured=True,
       no_promotion=True,
       by_source=result,named_note_edits_sha256=sha(f),
       original_nested_predictions_SHA256=sha(a.preds))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    true=np.zeros((1,128),bool);true[0,52]=True
    old=true.copy()
    wrong=old.copy();wrong[0,52]=False;wrong[0,55]=True
    s=score(true,old,wrong,np.array([1]))
    need(s["overall"]["regressions"]==1
         and s["overall"]["selected_ExactK"]==1,
         "S520 same correct K with incorrect note identity is still a regression")
    print("S520 SELFTEST PASS independent identical-K pitch regression")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for arg in ("source","source-report","trained-report","preds","models","out"):
        ap.add_argument("--"+arg,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for name in ("source","source_report","trained_report","preds","models","out"):
            need(getattr(a,name)is not None,"S520 missing --"+name)
        run(a)
