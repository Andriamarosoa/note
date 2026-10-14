"""S501 audit whether Exact-K improves ACTUAL polyphonic note identities.

Read SHA-frozen S500 K-constrained named pitches, true MIDI/XML only
after predictions fixed; evaluate 45 guitar and 322 piano original
positive attacks with same Hungarian 80ms one-to-one matching used
in S499. Compare to raw pretrained named-note baseline, count losses
and recoveries of COMPLETE pitch sets, and K accuracy separately.
No source model training, no tuning on these already-exposed corpora.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s499_score_canonical_musical_onsets_against_true_MIDI_XML import pairing
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def paired_metrics(true_sets,old_K,reference_sets,candidate_sets):
    y=[set(v) for v in true_sets]
    raw=[set(v) for v in reference_sets]
    prop=[set(v) for v in candidate_sets]
    truthK=np.asarray([len(x) for x in y],int)
    original=np.asarray(old_K,int)
    k_new=np.asarray([len(x) for x in prop],int)
    poly=truthK>=2
    before=np.asarray([a==b for a,b in zip(y,raw)],bool)
    after=np.asarray([a==b for a,b in zip(y,prop)],bool)
    bpK=np.asarray([len(x) for x in raw],int)
    old_good=original==truthK
    new_good=k_new==truthK
    return dict(n=len(y),true_poly=int(poly.sum()),
      source_original_K_correct=int(old_good.sum()),
      new_K_correct=int(new_good.sum()),
      K_fixes_against_source=int(np.sum(~old_good&new_good)),
      K_regressions_against_source=int(np.sum(old_good&~new_good)),
      BP_raw_K_correct=int(np.sum(bpK==truthK)),
      BP_raw_note_sets_correct=int(before.sum()),
      K_constrained_full_note_sets_correct=int(after.sum()),
      full_note_set_fixes_against_BP=int(np.sum(~before&after)),
      full_note_set_regressions_against_BP=int(np.sum(before&~after)),
      poly_BP_raw_note_sets_correct=int(np.sum(poly&before)),
      poly_K_constrained_note_sets_correct=int(np.sum(poly&after)),
      poly_full_note_fixes_against_BP=int(np.sum(poly&~before&after)),
      poly_full_note_regressions_against_BP=int(np.sum(poly&before&~after)),
      K_constrained_poly_exactK=int(np.sum(poly&new_good)),
      true_notes_total=sum(len(x) for x in y),
      predicted_notes_total=sum(len(x) for x in prop),
      true_pitches_recovered=sum(len(a&b) for a,b in zip(y,prop)))
def run(graph,report,guitar,guitar_report,piano,piano_report,out):
    need(not out.exists(),"S501 immutable paired source note sets already scored")
    frozen=json.loads(report.read_text())
    g=json.loads(guitar_report.read_text());p=json.loads(piano_report.read_text())
    need(frozen["status"]=="S500_TWO_SOURCE_JOINT_EXACTK_CONSTRAINED_REAL_NOTE_NAMES_PRELABEL"
         and frozen["no_training_no_XML_MIDI_label_access"]
         and frozen["four_precommitted_decoding_policies_untuned"]
         and g["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and g["matched_rows_SHA256"]==sha(guitar)
         and p["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and p["SHA256"]["paired_label_comparison"]==sha(piano),
         "S501 frozen notes must predate all true acoustic note set labels")
    sources=[
     ("guitar_S80",guitar,"source_wav","original_XML_anchor",
      "true_active","original_S80_K",45,25,21,27),
     ("piano_V88",piano,None,"onset_at_excerpt_s",
      "true_new_onset_MIDI_pitches","original_V88_K",322,42,73,125)]
    output={}
    for key,truthfile,source_key,tkey,nameskey,kkey,n,correct,polycount,baseline_notes in sources:
        candidate_path=graph/f"S500_{key}_K_CONSTRAINED_ACTUAL_NOTE_NAMES_PRELABEL.jsonl"
        need(frozen["frozen_output_shas"][key]==sha(candidate_path),
             "S501 cannot score altered K-constrained note predictions")
        candidates=[json.loads(s) for s in candidate_path.read_text().splitlines()]
        truth=[json.loads(s) for s in truthfile.read_text().splitlines()]
        need(len(truth)==n and len({v["canonical_event_id"] for v in candidates})==len(candidates),
             "S501 true attack sample or canonical unique group drifted")
        by_rec=collections.defaultdict(list)
        for j,c in enumerate(candidates):
            by_rec[c["source_recording_sha"]].append(j)
        assignment={}
        for source in sorted({(v[source_key] if source_key else
                                v["row_id"].split(":")[0]) for v in truth}):
            tid=[i for i,r in enumerate(truth) if
                (r[source_key] if source_key else r["row_id"].split(":")[0])==source]
            cid=by_rec.get(source,[])
            paired=pairing([truth[i][tkey] for i in tid],
                           [candidates[j]["onset_seconds"] for j in cid])
            for i,j,d in paired:assignment[tid[i]]=cid[j]
        y=[set(int(i) for i in item[nameskey]) for item in truth]
        old=[int(item[kkey]) for item in truth]
        need(sum(int(len(ys)==original) for ys,original in zip(y,old))==correct
             and sum(len(ys)>=2 for ys in y)==polycount,
             "S501 original S80 guitar or V88 piano true K source baseline changed")
        results={}
        for kind,baselinekey in (("new","detected_birth_note_MIDI"),
                                 ("active","detected_active_note_MIDI")):
            raw=[set(candidates[assignment[i]][baselinekey]) if i in assignment
                 else set() for i in range(n)]
            for method in ("confidence","harmonic_discount"):
                field="joint_"+kind+"_"+method+"_pitch_MIDI"
                generated=[set(candidates[assignment[i]][field]) if i in assignment
                           else set() for i in range(n)]
                results[kind+"_"+method]=paired_metrics(y,old,raw,generated)
        rawrelevant="active" if key=="guitar_S80" else "new"
        need(results[rawrelevant+"_confidence"]["BP_raw_note_sets_correct"]==baseline_notes,
             "S501 pretrained note set baseline 27/45 IDMT or 125/322 piano mismatch")
        output[key]=dict(canonical_predicted_groups=len(candidates),
            true_attack_count=n,musical_events_matched=len(assignment),
            missing_canonical_note_group_for_true_attack=n-len(assignment),
            previously_exposed_external_truth=True,policies=results)
        print("S501 TRUE NOTE SETS",json.dumps(dict(
            corpus=key,baseline=baseline_notes,
            decoded=results)),flush=True)
    out.mkdir(parents=True)
    rep=dict(status="S501_CANONICAL_JOINT_EXACTK_FULL_NOTESETS_REAL_IDMT_MAESTRO_AUDIT",
      source_IDMT_final_S80=True,
      source_MAESTRO_V88_not_final_S80=True,
      no_external_training_or_threshold_tuning=True,
      source_original_ExactK_and_live_untouched=True,
      all_four_note_rank_policies_evaluated_not_selected=True,
      all_UNKNOWN_unpaired_canonical_groups_not_assumed_false=True,
      independently_S500_pretruth_note_sets_SHA_verified=True,
      no_promotion=True,
      comparisons=output,
      sourceSHA=dict(S500=sha(report),original_guitar_real_notes=sha(guitar),
                     original_piano_real_notes=sha(piano)))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    z=paired_metrics([{52,55},{59}],[2,1],[{52,64},{59}],[{52,55},{59,64}])
    need(z["full_note_set_fixes_against_BP"]==1
         and z["full_note_set_regressions_against_BP"]==1,
         "S501 true pitch identity regressions must not disappear behind Exact K gains")
    print("S501 SELFTEST PASS whole-note set fixes/regressions independently from K")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--self-test",action="store_true")
    for k in ("graph","report","guitar","guitar-report","piano","piano-report","out"):
        a.add_argument("--"+k,type=Path)
    p=a.parse_args()
    if p.self_test:selftest()
    else:
        for k in ("graph","report","guitar","guitar_report","piano","piano_report","out"):
            need(getattr(p,k)is not None,"S501 missing --"+k)
        run(p.graph,p.report,p.guitar,p.guitar_report,p.piano,p.piano_report,p.out)
