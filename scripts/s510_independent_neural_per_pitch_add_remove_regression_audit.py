"""S510 SHA-independent replay and individual note edit autopsy of S509.

No TensorFlow, NO training, NO threshold tuning, no access to MIDI labels
until after checking frozen S509 note probabilities and checkpoint hashes.
Every pitch addition/deletion is audited by source and true K separately.
No candidate acoustic event outside the paired true-positive attack cohort
is labelled a genuine negative or true K0.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
def must(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def named(p):
    name=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    return name[int(p)%12]+str(int(p)//12-1)
def audit(truth,old,new,sourceK):
    before=np.all(truth==old,axis=1)
    after=np.all(truth==new,axis=1)
    edit=np.any(old!=new,axis=1)
    yK=truth.sum(1);poly=yK>=2
    def one(mask):
        return dict(n=int(mask.sum()),baseline_exact_notes=int(np.sum(mask&before)),
            candidate_exact_notes=int(np.sum(mask&after)),
            fixes=int(np.sum(mask&~before&after&edit)),
            regressions=int(np.sum(mask&before&~after&edit)),
            wrong_to_wrong=int(np.sum(mask&~before&~after&edit)),
            changed=int(np.sum(mask&edit)),
            original_source_ExactK=int(np.sum(mask&(yK==sourceK))),
            original_BP_ExactK=int(np.sum(mask&(yK==old.sum(1)))),
            candidate_ExactK=int(np.sum(mask&(yK==new.sum(1)))))
    return dict(all=one(np.ones(len(yK),bool)),poly=one(poly),
                by_true_K={str(k):one(yK==k) for k in range(7)},
                by_source_K={str(k):one(sourceK==k) for k in range(7)})

def run(a):
    base=json.loads(a.data_report.read_text())
    study=json.loads(a.study_report.read_text())
    must(not a.out.exists() and
        base["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
        and base["audio_proposal_truth_table_SHA256"]==sha(a.data)
        and study["status"]=="S509_REAL_13_HEAD_ATTENTION_NOTE_BY_NOTE_EIGHT_RECORDING_OOF"
        and study["truly_trained_per_pitch_neural_attention"]
        and study["source_S505_sha256"]==sha(a.data)
        and study["probability_table_sha256"]==sha(a.predictions)
        and len(study["folds"])==8 and study["no_promotion"],
        "S510 original eight trained source notes or frozen inference results changed")
    for fold in study["folds"]:
        checkpoint=a.models/f"S509_fold{fold['fold']}_REAL_PITCH_ATTENTION.weights.h5"
        must(checkpoint.is_file() and sha(checkpoint)==fold["trained_checkpoint_sha256"]
              and fold["evaluation_recording_excluded_from_training"],
              "S510 learned per-pitch model checkpoint SHA or training fold provenance invalid")
    with np.load(a.data,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    events=d["event_id"].astype(str);source=d["corpus"].astype(str)
    recordings=d["recording_sha"].astype(str)
    y=d["truth_note_bits"].astype(bool);head=d["head_note_bits"].astype(bool)
    baseheads=d["selected_BP_baseline_head"].astype(int)
    old=head[np.arange(len(y)),baseheads].astype(bool)
    available=d["head_mask"].astype(bool)
    index={e:i for i,e in enumerate(events)}
    must(len(index)==len(events),"S510 source musical canonical IDs must be unique")
    modelrows=[json.loads(line) for line in a.predictions.read_text().splitlines()]
    seen=set();proposals=collections.defaultdict(dict)
    for row in modelrows:
        event=row["canonical_event_id"]
        must(event in index and row["original_recording_SHA"]==recordings[index[event]],
             "S510 predicted pitches not from real held recording audio")
        pitch=int(row["real_acoustic_candidate_MIDI_pitch"])
        must(0<=pitch<128 and (event,pitch) not in seen and
             float(row["learned_probability"])>=0 and
             float(row["learned_probability"])<=1 and
             bool(np.any(head[index[event],:,pitch]&available[index[event]])),
             "S510 model emitted duplicated, unsupported or invalid musical pitch")
        seen.add((event,pitch))
        proposals[event][pitch]=float(row["learned_probability"])
    must(len(proposals)==len(events),"S510 all matched real note events must have at least one prediction")
    source_report={};details=[]
    for key,diagnostic in (("per_pitch_diagnostic",True),
                            ("per_pitch_guarded",False)):
        reconstructed=old.copy()
        if diagnostic:reconstructed[:]=False
        for e,pitches in proposals.items():
            i=index[e]
            for midi,prob in pitches.items():
                if diagnostic:
                    if prob>=.50:reconstructed[i,midi]=True
                else:
                    if old[i,midi] and prob<=.15:reconstructed[i,midi]=False
                    elif not old[i,midi] and prob>=.85:reconstructed[i,midi]=True
        empty=np.flatnonzero(~reconstructed.any(axis=1))
        reconstructed[empty]=old[empty]
        bysource={}
        for src in ("guitar_S80","piano_V88"):
            idx=np.flatnonzero(source==src)
            calc=audit(y[idx],old[idx],reconstructed[idx],
                       d["source_K"][idx].astype(int))
            expected=study["policies"][key][src]
            for field in ("baseline_full_chord_correct","predicted_full_chord_correct",
                          "corrected","regressed","changes","baseline_ExactK","new_ExactK"):
                old_field={"baseline_full_chord_correct":"baseline_exact_notes",
                           "predicted_full_chord_correct":"candidate_exact_notes",
                           "corrected":"fixes","regressed":"regressions",
                           "changes":"changed",
                           "baseline_ExactK":"original_BP_ExactK",
                           "new_ExactK":"candidate_ExactK"}[field]
                must(calc["all"][old_field]==expected["overall"][field],
                     "S510 independent replay disagrees with S509 source note metric: "+src+field)
            bysource[src]=calc
        source_report[key]=bysource
        changed=np.flatnonzero(np.any(old!=reconstructed,axis=1))
        for i in changed:
            removed=sorted(np.flatnonzero(old[i]&~reconstructed[i]).tolist())
            added=sorted(np.flatnonzero(~old[i]&reconstructed[i]).tolist())
            status=("fix" if not np.array_equal(y[i],old[i]) and np.array_equal(y[i],reconstructed[i])
               else "regression" if np.array_equal(y[i],old[i]) and not np.array_equal(y[i],reconstructed[i])
               else "wrong_to_wrong")
            details.append(dict(policy=key,canonical_event_id=str(events[i]),
                source=str(source[i]),recording_sha=str(recordings[i]),
                true_K=int(y[i].sum()),original_source_K=int(d["source_K"][i]),
                source_note_names=[named(v) for v in np.flatnonzero(old[i])],
                proposed_note_names=[named(v) for v in np.flatnonzero(reconstructed[i])],
                ground_truth_notes_FOR_AUDIT_ONLY=[named(v) for v in np.flatnonzero(y[i])],
                pitches_added=[named(v) for v in added],
                pitches_removed=[named(v) for v in removed],
                edit_result=status))
        print("S510 INDEPENDENT PER-NOTE REPLAY",json.dumps(dict(
            policy=key,source={k:v["all"] for k,v in bysource.items()},
            true_poly={k:v["poly"] for k,v in bysource.items()})),flush=True)
    a.out.mkdir(parents=True)
    dfile=a.out/"S510_EACH_REAL_NOTE_ADD_REMOVE_K_CORRECTION_REGRESSION.jsonl"
    dfile.write_text("".join(json.dumps(v)+"\n" for v in details))
    veto=any(x["edit_result"]=="regression" for x in details)
    report=dict(status="S510_INDEPENDENT_SOURCE_PITCH_BY_PITCH_S509_REPLAY",
       eight_neural_per_pitch_checkpoint_SHAs_verified=True,
       each_changed_full_chord_and_K_independently_scored=True,
       each_edit_has_named_added_removed_and_audit_truth_notes=True,
       previously_exposed_corpora_not_new_blind=True,
       no_true_K_was_available_to_neural_predictions=True,
       unpaired_model_events_NOT_true_zero_labels=True,
       number_of_regression_events=int(sum(x["edit_result"]=="regression" for x in details)),
       any_observed_regression_vetoes_S80_promotion=bool(veto),
       no_production_changes=True,no_promotion=True,
       by_policy=source_report,all_edit_ledger_SHA256=sha(dfile),
       source_S509_sha256=sha(a.study_report))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    z=np.zeros((2,128),bool);z[0,52]=True;z[1,52]=True
    w=z.copy();w[1,52]=False;w[1,55]=True
    q=audit(z,z,w,np.array([1,1]))
    must(q["all"]["regressions"]==1 and q["all"]["candidate_ExactK"]==2,
         "S510 wrong note identity at exact same K must veto promotion")
    print("S510 SELFTEST PASS wrong musical note always counts as regression even when Exact-K unchanged")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true")
    for k in ("data","data-report","study-report","predictions","models","out"):
        ap.add_argument("--"+k,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("data","data_report","study_report","predictions","models","out"):
            must(getattr(a,k)is not None,"S510 missing "+k)
        run(a)
