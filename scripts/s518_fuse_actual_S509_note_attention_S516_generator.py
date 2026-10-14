"""S518: REAL integration of S516 generator as FOURTEENTH pitch HEAD into
S509's per-pitch note-selector evidence, SHA-frozen, without retraining.

S509 predictions live only where original 13 heads observed a pitch.
S516 supplies 49 MIDI pitch probabilities, even when ALL 13 lacked it.
Four predeclared selector policies, no true MIDI notes in decisions:
 * generator_reference: original S516 p>=.70
 * confidence_fusion: average p_S516, p_S509 when both available;
   for NEW pitches absent S509, use S516 alone.
 * anchor_fusion: S516 p>=.70 + S509 p>=.80 (existing candidates),
   but protect original BP note if both sources are uncertain.
 * conservative_joint: KEEP original BP unless S516 and available
   S509 agree strongly (for removals), allow new S516-only pitch with
   p>=.85; suppress empty emitted music group.

This is an inference-only *shadow* fusion, not yet trained S509 attention
on 14 heads. It does NOT justify any 80% full-stream note-event claim.
S80 GuitarSet historical K and Live, V88 original and offsets preserved.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(a):
    r505=json.loads(a.s505_report.read_text())
    r509=json.loads(a.s509_report.read_text())
    r516=json.loads(a.s516_report.read_text())
    need(r505["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
        and r509["status"]=="S509_REAL_13_HEAD_ATTENTION_NOTE_BY_NOTE_EIGHT_RECORDING_OOF"
        and r516["status"]=="S516_NEW_TRAINED_RAW_WAV_MULTILABEL_NOTE_GENERATOR_EIGHT_RECORDING_OOF"
        and r505["audio_proposal_truth_table_SHA256"]==sha(a.s505)
        and r509["source_S505_sha256"]==sha(a.s505)
        and r516["original_head_labels_sha256"]==sha(a.s505)
        and r509["probability_table_sha256"]==sha(a.s509)
        and r516["predictions_sha256"]==sha(a.s516)
        and r509["eight_holdout_recording_models_fitted"]
        and r516["full_8_recording_crossfit"]
        and r509["no_promotion"] and r516["no_promotion"],
        "S518 source S509/S516 probability stream changed or is not held-recording OOF")
    with np.load(a.s505,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    N=len(d["event_id"])
    events=d["event_id"].astype(str)
    index={event:i for i,event in enumerate(events)}
    need(N==len(index)==r505["rows"] and N==265, "S518 musical group identity changed")
    recordings=d["recording_sha"].astype(str)
    corpus=d["corpus"].astype(str)
    p509=np.full((N,128),np.nan,np.float32)
    seen=set()
    for ln in a.s509.read_text().splitlines():
        x=json.loads(ln)
        ev=x["canonical_event_id"]
        pitch=int(x["real_acoustic_candidate_MIDI_pitch"])
        need(ev in index and 0<=pitch<=127 and
             x["original_recording_SHA"]==recordings[index[ev]]
             and (ev,pitch) not in seen,
             "S518 S509 old head evidence cannot be reassigned to another recording/pitch")
        seen.add((ev,pitch));p509[index[ev],pitch]=float(x["learned_probability"])
    p516=np.full((N,128),np.nan,np.float32)
    seen=set()
    for ln in a.s516.read_text().splitlines():
        x=json.loads(ln)
        ev=x["canonical_event_id"]
        need(ev in index and ev not in seen
             and x["source_rec"]==recordings[index[ev]]
             and x["corpus"]==corpus[index[ev]],
             "S518 S516 previously unseen generator cannot be reassigned to a recording")
        seen.add(ev);p516[index[ev],40:89]=np.asarray(
            x["probabilities_by_MIDI_40to88"],np.float32)
    need(len(seen)==N and np.all(np.isfinite(p516[:,40:89])) and
         np.all((p516[:,40:89]>=0)&(p516[:,40:89]<=1)),
         "S518 missing independent generator pitch scores")
    need(np.all(np.isfinite(p509[np.isfinite(p509)])) and
         len([z for z in np.flatnonzero(np.isfinite(p509))])>N,
         "S518 no truly trained 13-head evidence")
    bp=d["head_note_bits"][np.arange(N),
        d["selected_BP_baseline_head"].astype(int)].astype(bool)
    return d,p509,p516,bp,r505,r509,r516

def decode(p509,p516,baseline,policy):
    """No truth, true K, or instrument-specific fitted thresholds here."""
    valid=np.isfinite(p509)
    old=baseline[:,40:89]
    left=np.nan_to_num(p509[:,40:89],nan=0.)
    new=p516[:,40:89]
    if policy=="generator_reference":
        predicted=new>=.70
    elif policy=="confidence_fusion":
        prob=np.where(valid[:,40:89],.5*left+.5*new,new)
        predicted=prob>=.70
    elif policy=="anchor_fusion":
        predicted=(new>=.70)|(valid[:,40:89]&(left>=.80))
        safeguard=old&(new>=.25)&(~valid[:,40:89] | (left>=.25))
        predicted|=safeguard
    elif policy=="conservative_joint":
        predicted=old.copy()
        predicted[(~old)&(new>=.85)]=True
        remove=old&(new<=.15)&valid[:,40:89]&(left<=.15)
        predicted[remove]=False
    else:raise ValueError("S518 no retrospective new test-tuned selector policy")
    # The event exists in the canonical BP graph; a 49-way score of
    # 0 should not erase a known event without an explicit K0 head audit.
    zero=~predicted.any(axis=1)
    predicted[zero]=old[zero]
    out=np.zeros_like(baseline)
    out[:,40:89]=predicted
    # Original outside-range BP pitches cannot be synthesized by this
    # generator; KEEP them rather than silently removing known audio.
    out[:,:40]=baseline[:,:40];out[:,89:]=baseline[:,89:]
    return out

def measure(y,old,new,sourceK):
    yg=y.sum(axis=1);bg=old.sum(axis=1);ng=new.sum(axis=1)
    b=np.all(old==y,axis=1);v=np.all(new==y,axis=1)
    changed=np.any(old!=new,axis=1)
    def score(mask):
        tp=int(np.sum((new&y)[mask]))
        fp=int(np.sum((new&~y)[mask]))
        fn=int(np.sum((~new&y)[mask]))
        add=int(np.sum((~old&new)[mask]))
        removed=int(np.sum((old&~new)[mask]))
        return dict(n=int(mask.sum()),source_full_correct=int(np.sum(mask&b)),
            candidate_full_correct=int(np.sum(mask&v)),
            corrections=int(np.sum(mask&(~b)&v&changed)),
            regressions=int(np.sum(mask&b&(~v)&changed)),
            wrong_to_wrong=int(np.sum(mask&(~b)&(~v)&changed)),
            modified_chords=int(np.sum(mask&changed)),
            added_pitches=add,removed_pitches=removed,
            original_Source_ExactK=int(np.sum(mask&(sourceK==yg))),
            original_BP_ExactK=int(np.sum(mask&(bg==yg))),
            selected_ExactK=int(np.sum(mask&(ng==yg))),
            true_pitch_TP=tp,extra_pitch_FP=fp,missed_pitch_FN=fn,
            pitch_precision_on_positive_attacks=tp/max(1,tp+fp),
            pitch_recall_on_positive_attacks=tp/max(1,tp+fn),
            pitch_F1_on_positive_attacks=2*tp/max(1,2*tp+fp+fn))
    return dict(all=score(np.ones(len(y),bool)),poly=score(yg>=2),
                by_true_K={str(k):score(yg==k) for k in range(7)})
def selftest():
    baseline=np.zeros((1,128),bool);baseline[0,52]=True
    p509=np.full((1,128),np.nan,np.float32);p509[0,52]=.92
    p516=np.zeros((1,128),np.float32);p516[0,52]=.80;p516[0,55]=.95
    a=decode(p509,p516,baseline,"conservative_joint")
    need(set(np.flatnonzero(a[0]))=={52,55},
         "S518 must enable generator E3/G3 combination when original 13 heads missed G3")
    b=decode(p509,p516,baseline,"confidence_fusion")
    need(set(np.flatnonzero(b[0]))=={52,55},
         "S518 new pitch outside old 13 heads must survive S509 availability gap")
    print("S518 SELFTEST PASS independent 14th generator can add new G3 to existing E3")
def run(a):
    need(not a.out.exists(),"S518 original joint S509+S516 evaluation immutable")
    d,p509,p516,bp,r505,r509,r516=load(a)
    N=len(bp);sources=d["corpus"].astype(str)
    truth=d["truth_note_bits"].astype(bool)
    originalK=d["source_K"].astype(int)
    result={};raw=[]
    for policy in ("generator_reference","confidence_fusion",
                   "anchor_fusion","conservative_joint"):
        notes=decode(p509,p516,bp,policy)
        result[policy]={}
        for corpus in ("guitar_S80","piano_V88"):
            ix=np.flatnonzero(sources==corpus)
            r=measure(truth[ix],bp[ix],notes[ix],originalK[ix])
            need(r["all"]["source_full_correct"]==
                 r505["by_source"][corpus]["raw_BP_base_full_note_sets_correct"],
                 "S518 original note baseline shifted")
            r["unmatched_original_positive_attacks"]=r505["by_source"][corpus][
                "annotated_true_attacks_NOT_matched"]
            r["all_true_annotated_attacks"]=r505["by_source"][corpus][
                "truth_original_positive_attacks"]
            r["unpaired_prediction_truth_unknown"]=True
            result[policy][corpus]=r
        print("S518 FOURTEENTH HEAD JOINT SHADOW",json.dumps(dict(policy=policy,
            guitar=result[policy]["guitar_S80"]["all"],
            piano=result[policy]["piano_V88"]["all"],
            guitar_poly=result[policy]["guitar_S80"]["poly"],
            piano_poly=result[policy]["piano_V88"]["poly"])),flush=True)
        for i in range(N):
            if np.array_equal(bp[i],notes[i]):continue
            raw.append(dict(canonical_event_id=str(d["event_id"][i]),
                source=sources[i],policy=policy,
                chosen_pitches=np.flatnonzero(notes[i]).tolist(),
                original_pitches=np.flatnonzero(bp[i]).tolist(),
                source_recording_sha=str(d["recording_sha"][i]),
                edited_with_only_original_audio_prelabel_inputs=True))
    a.out.mkdir(parents=True)
    edits=a.out/"S518_ACTUAL_PITCH_EDITS_PER_PREDECLARED_SHADOW_SELECTOR.jsonl"
    edits.write_text("".join(json.dumps(x)+"\n" for x in raw))
    result_report=dict(
       status="S518_S509_NEURAL_HEADS_PLUS_S516_NEW_PITCH_GENERATOR_JOINT_SHADOW",
       original_S509_evidence_13_heads_plus_S516_14th_generator=True,
       raw_generator_new_pitches_allowed_outside_S509_support=True,
       fixed_untrained_combination_policies_not_a_new_learned_selector=True,
       eight_recordingheld_S509_and_S516_own_predictions=True,
       separate_upstream_crossfit_not_fully_nested_end_to_end=True,
       existing_8_sources_already_seen_historically_not_blind=True,
       missing_original_true_piano_onsets_102_not_removed=True,
       no_full_stream_false_positive_onset_truth=True,
       not_equivalent_to_note_onset_pitch_full_F1=True,
       no_model_or_live_reference_promotion=True,no_promotion=True,
       by_policy=result,edits_sha256=sha(edits),
       evidence_SHA=dict(S505=sha(a.s505),S509=sha(a.s509),S516=sha(a.s516)))
    (a.out/"report.json").write_text(json.dumps(result_report,indent=2)+"\n")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for p in ("s505","s505-report","s509","s509-report","s516","s516-report","out"):
        ap.add_argument("--"+p,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for p in ("s505","s505_report","s509","s509_report","s516","s516_report","out"):
            need(getattr(a,p)is not None,"S518 missing --"+p)
        run(a)
