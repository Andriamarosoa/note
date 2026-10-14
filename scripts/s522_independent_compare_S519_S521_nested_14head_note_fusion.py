"""S522 independent comparison S519 vs S521 true trained neural 14-heads.

S521 changed class imbalance weights based ONLY on training pitch labels.
S519 and S521 have same true positive source, source audio features and
8 original held-recording folds. Recompute ALL true note named-pitch edits
from SHA-frozen note probabilities (not from printed GitHub summaries).

This retrospective ablation is NOT a new independent blind trial:
S519 historical test scores were examined before S521 was conceived.
No F1-event on original full stream: unpaired candidate note attacks
lack confirmed true negative labels; 102 pianist onsets unrepresented.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s520_independent_replay_nested_14head_pitch_selection import decode,score
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_version(study,prob,weights,version,original_sha):
    rep=json.loads(study.read_text())
    need(rep["status"]==f"{version}_NESTED_S516_RAW_GENERATOR_INSIDE_S509_14HEAD_NEURAL_SELECTOR"
         and rep["source_SHA256"]["S505_original_supervised_source"]==original_sha
         and rep["source_SHA256"]["OOF_frozen_generator_neural_prediction_SHA256"]==sha(prob)
         and len(rep["folds"])==8 and rep["no_promotion"],
         f"S522 {version} genuine matched 8-song frozen neural source changed")
    folds={x["held_recording_SHA"] for x in rep["folds"]}
    for fold in rep["folds"]:
        f=fold["fold"]
        a=weights/f"{version}_fold{f}_S516_INSIDE_14HEAD_RAW_GENERATOR.joblib"
        b=weights/f"{version}_fold{f}_S509_NEURAL_14HEAD_ATTENTION.weights.h5"
        need(a.is_file() and b.is_file() and sha(a)==fold["generator_model_SHA256"]
             and sha(b)==fold["attention_model_SHA256"]
             and fold["training_recording_count"]==7
             and fold["inner_train_recording_count_per_generator"]==6,
             f"S522 {version} two source checkpoints or nesting changed")
    raw=[json.loads(x) for x in prob.read_text().splitlines()]
    return rep,raw,folds
def run(a):
    need(not a.out.exists(),"S522 do not overwrite frozen independent comparison")
    original=json.loads(a.source_report.read_text())
    need(original["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
         and original["audio_proposal_truth_table_SHA256"]==sha(a.source),
         "S522 original note identity supervision cohort corrupted")
    with np.load(a.source,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    N=len(d["event_id"])
    need(N==265 and len(set(d["recording_sha"].astype(str)))==8,
         "S522 original 5 guitar and 3 piano note sources changed")
    index={x:i for i,x in enumerate(d["event_id"].astype(str))}
    recordings=d["recording_sha"].astype(str)
    corpus=d["corpus"].astype(str)
    true=d["truth_note_bits"].astype(bool)
    bp=d["head_note_bits"][np.arange(N),d["selected_BP_baseline_head"].astype(int)].astype(bool)
    sourceK=d["source_K"].astype(int)
    result={}
    for ver,report,pred,weights in (
       ("S519",a.r519,a.p519,a.weights519),
       ("S521",a.r521,a.p521,a.weights521)):
        rep,raw,groups=read_version(report,pred,weights,ver,sha(a.source))
        need(groups==set(recordings) and len(raw)==N,
             f"S522 {ver} whole recorded source SHA set changed")
        probs=np.full((N,49),np.nan,np.float32)
        gen=np.full((N,49),np.nan,np.float32)
        used=set()
        for x in raw:
            name=x["canonical_event_id"]
            need(name in index and name not in used,
                 f"S522 {ver} missing or duplicate observed musical onset")
            i=index[name];used.add(name)
            need(x["original_source_recording_SHA"]==recordings[i] and
                 x["source_corpus"]==corpus[i],
                 f"S522 {ver} original recording identity mismatch")
            probs[i]=np.asarray(x["per_MIDI_pitch_40to88_attention_probability"],np.float32)
            gen[i]=np.asarray(x["per_MIDI_pitch_40to88_generator_probability"],np.float32)
        need(np.isfinite(probs).all() and np.isfinite(gen).all()
             and np.all((0<=probs)&(probs<=1))
             and np.all((0<=gen)&(gen<=1)),
             f"S522 {ver} one pitch probability invalid/missing")
        bysource={}
        for src in ("guitar_S80","piano_V88"):
            ix=np.flatnonzero(corpus==src)
            bysource[src]={}
            for policy in ("attention050","attention070","protected_085"):
                prediction=decode(probs[ix],bp[ix],gen[ix],policy)
                stats=score(true[ix],bp[ix],prediction,sourceK[ix])
                ref=rep["by_source"][src][policy]
                for field in ("new_full_chords","fixes","regressions","selected_ExactK",
                              "note_F1_conditional_true_attacks"):
                    need(abs(float(stats["overall"][field])-
                             float(ref["overall"][field]))<1.e-11,
                         f"S522 {ver} independent note identity parity failed "+field)
                recovered_missing=int(np.sum((~bp[ix]&prediction&true[ix])[:,40:89]))
                added_wrong=int(np.sum((~bp[ix]&prediction&~true[ix])[:,40:89]))
                lost_true=int(np.sum((bp[ix]&~prediction&true[ix])[:,40:89]))
                stats["genuinely_missing_BP_pitches_recovered"]=recovered_missing
                stats["newly_added_wrong_pitches"]=added_wrong
                stats["original_true_pitches_deleted"]=lost_true
                stats["unknown_note_onset_negatives_NOT_scored"]=True
                stats["all_original_annotated_positive_attacks"]=(
                    original["by_source"][src]["truth_original_positive_attacks"])
                stats["true_attacks_without_canonical_group"]=(
                    original["by_source"][src]["annotated_true_attacks_NOT_matched"])
                bysource[src][policy]=stats
            print("S522 INDEPENDENT 14 HEAD REPLAY",json.dumps(
                dict(model=ver,source=src,policies={
                    k:{"overall":v["overall"],
                       "added_true_new":v["genuinely_missing_BP_pitches_recovered"],
                       "added_false_new":v["newly_added_wrong_pitches"],
                       "deleted_true":v["original_true_pitches_deleted"]}
                    for k,v in bysource[src].items()})),flush=True)
        result[ver]=bysource
    a.out.mkdir(parents=True)
    rep=dict(status="S522_INDEPENDENT_TRUE_NOTE_S519_VS_S521_14HEAD_ATTENTION_AUDIT",
       eight_true_trained_S519_neural_and_acoustic_checkpoint_pairs_verified=True,
       eight_true_trained_S521_neural_and_acoustic_checkpoint_pairs_verified=True,
       original_S505_guitar_and_piano_pitch_baseline_unchanged=True,
       every_genuinely_added_missing_note_and_false_addition_independently_counted=True,
       no_model_retraining_or_test_threshold_tuning_in_replay=True,
       S521_informed_by_seen_S519_scores_so_NOT_prospective_blind=True,
       no_full_event_onset_F1_or_verified_negative_onsets=True,
       full_original_reference_S80_V88_Live_unchanged=True,
       no_promotion=True,
       by_version=result,
       source_SHA256=dict(original_true_notes=sha(a.source),
           S519_attention_predictions=sha(a.p519),
           S521_attention_predictions=sha(a.p521)))
    (a.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    y=np.zeros((1,128),bool);y[0,52]=True
    old=y.copy()
    new=y.copy();new[0,52]=False;new[0,55]=True
    data=score(y,old,new,np.array([1]))
    need(data["overall"]["regressions"]==1 and
         data["overall"]["selected_ExactK"]==1,
         "S522 exact-K changes cannot hide wrong note pitch regression")
    print("S522 SELFTEST PASS K-correct pitch-incorrect chord is regression")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for arg in ("source","source-report","r519","p519","weights519",
                "r521","p521","weights521","out"):
        p.add_argument("--"+arg,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for name in ("source","source_report","r519","p519","weights519",
                     "r521","p521","weights521","out"):
            need(getattr(a,name)is not None,"S522 missing "+name)
        run(a)
