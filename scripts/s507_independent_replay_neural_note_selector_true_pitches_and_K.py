"""S507 independent replay of the frozen eight-song neural selector.

No TF import and NO retraining. Independently check all eight per-recording
held predictions against S505 original true note pitch sets and S506
SHA-audited model outputs. Audit by instrument, original predicted K,
TRUE K0..K6, exact note identities and Exact-K, including losses.
This cannot score stream false positives because unmatched model events
have no confirmed negative labels. Retrospective external source, NO PROMOTION.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measure(old,new,truth,sourceK):
    b=np.all(old==truth,axis=-1)
    p=np.all(new==truth,axis=-1)
    changed=np.any(old!=new,axis=-1)
    true_K=truth.sum(1);base_K=old.sum(1);new_K=new.sum(1)
    intersections=np.logical_and(new,truth).sum()
    pr=int(new.sum());total=int(truth.sum())
    return dict(n=len(b),full_notes_old=int(b.sum()),full_notes_new=int(p.sum()),
        corrections=int(np.sum(~b&p&changed)),
        regressions=int(np.sum(b&~p&changed)),
        wrong_to_wrong=int(np.sum(~b&~p&changed)),
        changed=int(changed.sum()),
        original_source_ExactK=int(np.sum(sourceK==true_K)),
        original_BP_ExactK=int(np.sum(base_K==true_K)),
        neural_ExactK=int(np.sum(new_K==true_K)),
        recovered_true_pitches=int(intersections),
        emitted_note_total=pr,true_note_total=total,
        note_precision=float(intersections/pr) if pr else None,
        note_recall=float(intersections/total) if total else None)

def run(a):
    need(not a.out.exists(),"S507 original independent replay is immutable")
    base=json.loads(a.data_report.read_text())
    report=json.loads(a.selector_report.read_text())
    need(base["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
         and base["audio_proposal_truth_table_SHA256"]==sha(a.data)
         and report["status"]=="S506_TRUE_NEURAL_13_HEADS_RECORDING_HELD_EXPLORATORY_NO_PROMOTION"
         and report["eight_recordings_held_out_in_outer_selector_training"]
         and report["source_S505_sha256"]==sha(a.data)
         and report["predictions_sha256"]==sha(a.predictions)
         and len(report["songheld_folds"])==8
         and report["no_promotion"],"S507 original eight held-song neural model provenance drift")
    rows=[json.loads(x) for x in a.predictions.read_text().splitlines()]
    with np.load(a.data,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    ids=data["event_id"].astype(str)
    index={v:i for i,v in enumerate(ids)}
    need(len(index)==len(ids)==len(rows),"S507 original canonical event rows inconsistent")
    ordered=[None]*len(ids)
    for row in rows:
        need(row["canonical_event_id"] in index,"S507 model predicted event absent from source")
        idx=index[row["canonical_event_id"]]
        need(ordered[idx] is None,"S507 duplicate prediction event")
        need(row["source_recording_sha"]==str(data["recording_sha"][idx])
             and row["corpus"]==str(data["corpus"][idx]),
             "S507 recording identity leaked/shifted between model and source")
        ordered[idx]=row
    need(all(row is not None for row in ordered),"S507 incomplete real model output")
    recordings=set(data["recording_sha"].astype(str))
    need(recordings=={x["held_recording_sha"] for x in report["songheld_folds"]},
         "S507 OOF recording set changed")
    for fold in report["songheld_folds"]:
        need(fold["train_recording_count"]==7 and
             fold["held_recording_absent_from_training"] and
             fold["training_only_audits_exclude_training_row_source"] and
             fold["held_audits_only_include_other_recordings"],
             "S507 held source was not excluded from fit or training audit")
        chk=a.model/f"S506_fold{fold['fold']}_TRAINED_NEURAL_SELECTOR.weights.h5"
        need(chk.is_file() and sha(chk)==fold["trained_checkpoint_sha256"],
             "S507 trained original neural fold weights SHA mismatch")
    truth=data["truth_note_bits"].astype(bool)
    proposals=data["head_note_bits"].astype(bool)
    baselinehead=data["selected_BP_baseline_head"].astype(int)
    corpus=data["corpus"].astype(str);sourceK=data["source_K"].astype(int)
    trueK=truth.sum(1)
    results={}
    for src in ("guitar_S80","piano_V88"):
        idx=np.flatnonzero(corpus==src)
        origin=proposals[idx,baselinehead[idx]]
        bypolicy={}
        for label,key in (
            ("fixed_guarded","selected_neural_head_guarded"),
            ("ungated_diagnostic","selected_neural_head_ungated")):
            selection=np.asarray([int(ordered[i][key]) for i in idx],int)
            need(np.all((selection>=-1)&(selection<13)),
                 "S507 learned head index outside original 13")
            new=np.stack([proposals[k,h] if h>=0 else proposals[k,baselinehead[k]]
                          for k,h in zip(idx,selection)])
            summary=measure(origin,new,truth[idx],sourceK[idx])
            summary["poly"]=measure(origin[trueK[idx]>=2],new[trueK[idx]>=2],
                             truth[idx][trueK[idx]>=2],
                             sourceK[idx][trueK[idx]>=2])
            summary["trueK"]={str(k):measure(origin[trueK[idx]==k],
                        new[trueK[idx]==k],truth[idx][trueK[idx]==k],
                        sourceK[idx][trueK[idx]==k]) for k in range(7)}
            summary["original_predicted_sourceK"]={str(k):measure(
                        origin[sourceK[idx]==k],new[sourceK[idx]==k],
                        truth[idx][sourceK[idx]==k],sourceK[idx][sourceK[idx]==k])
                        for k in range(7)}
            summary["chosen_note_heads"]={str(h):int(np.sum(selection==h))
                                          for h in range(-1,13)}
            summary["missing_known_true_attacks_without_detected_music_group"]=(
               base["by_source"][src]["annotated_true_attacks_NOT_matched"])
            summary["true_annotated_attacks_total"]=(
               base["by_source"][src]["truth_original_positive_attacks"])
            summary["unknown_unmatched_model_groups_are_not_negative_labels"]=True
            recorded=report["by_source"][src][label]["all"]
            need(summary["full_notes_old"]==recorded["BP_base_correct"]
                 and summary["full_notes_new"]==recorded["proposed_correct"]
                 and summary["corrections"]==recorded["fixes"]
                 and summary["regressions"]==recorded["regressions"]
                 and summary["neural_ExactK"]==recorded["candidate_K_correct"],
                 "S507 independent replay disagrees with original model audit")
            bypolicy[label]=summary
        results[src]=bypolicy
        print("S507 INDEPENDENT NEURAL REPLAY",json.dumps(dict(
            source=src,policies={k:{z:v for z,v in x.items()
                if z not in ("trueK","original_predicted_sourceK")}
                for k,x in bypolicy.items()})),flush=True)
    unsafe=any(m["regressions"]>0 or
            m["poly"]["regressions"]>0 for src in results.values()
            for m in src.values())
    a.out.mkdir(parents=True)
    audit=dict(status="S507_INDEPENDENT_REPLAY_OF_EIGHT_TRUE_TRAINED_NEURAL_SELECTOR_FOLDS",
       original_true_neural_models_sha_checked=8,
       no_training_or_model_tuning_in_independent_replay=True,
       all_original_source_head_groups_preserved=True,
       known_positives_only_cannot_score_false_model_beginnings=True,
       previous_exposure_makes_test_not_blind=True,
       all_trueK_and_original_predictedK_audits=True,
       any_regression_observed=unsafe,
       no_production_promotion=True,
       source_guitar_trueK_active_piano_trueK_new=True,
       by_source=results,
       source_SHA256=dict(S505=sha(a.data),S506_model_predictions=sha(a.predictions)))
    (a.out/"report.json").write_text(json.dumps(audit,indent=2)+"\n")

def selftest():
    a=np.zeros((2,128),bool);a[0,52]=True;a[1,52]=True
    b=a.copy();b[1,52]=False;b[1,55]=True
    y=a.copy()
    m=measure(a,b,y,np.array([1,1]))
    need(m["regressions"]==1 and m["full_notes_new"]==1,
        "S507 changing only a pitch at SAME K must register regression")
    print("S507 SELFTEST PASS same-K incorrect note must be audited as regression")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("data","data-report","selector-report","predictions","model","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("data","data_report","selector_report","predictions","model","out"):
            need(getattr(a,k)is not None,"S507 missing --"+k)
        run(a)
