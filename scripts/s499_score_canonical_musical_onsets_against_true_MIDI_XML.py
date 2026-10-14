"""S499 — score S498 canonical musical attacks, not per-V88 pseudo-onsets.

Uses original IDMT 45 XML known positive onsets + MAESTRO 322 MIDI
onsets from historical S488/S476, both AFTER verifying that S498
frozen canonical graph never opened labels. Each 20ms musical birth
group matches at most one true attack within 80ms via Hungarian.
Missing matched predictions count K0, as a *missed true attack*.
Unpaired canonical groups are UNKNOWN, not assumed false attacks.
Score independent BP Knew/Kactive alongside soft-source-K joint variants
and legacy source Exact-K, with event-level corrections and regressions.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
def check(c,msg):
    if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pairing(truth,guess,limit=.080):
    n=len(truth);m=len(guess)
    if not n or not m:return []
    diff=np.abs(np.asarray(truth,float)[:,None]-np.asarray(guess,float)[None,:])
    c=np.concatenate((np.where(diff<=limit,diff/limit*.5,1e6),
        np.ones((n,n))),axis=1)
    ii,jj=linear_sum_assignment(c)
    result=[(int(i),int(j),float(truth[i]-guess[j]))
            for i,j in zip(ii,jj) if j<m and diff[i,j]<=limit]
    check(len(set(j for i,j,d in result))==len(result)
          and len(set(i for i,j,d in result))==len(result),
          "S499 one music note birth matched to two true attacks")
    return result
def tally(trueK,oldK,predK,truemusic,predmusic):
    trueK=np.asarray(trueK,int);old=np.asarray(oldK,int);new=np.asarray(predK,int)
    good=old==trueK;after=new==trueK
    poly=trueK>=2
    note_correct=np.asarray([a==b for a,b in zip(truemusic,predmusic)],bool)
    return dict(n=int(len(trueK)),true_poly=int(poly.sum()),
        original_exactK_correct=int(good.sum()),canonical_exactK_correct=int(after.sum()),
        K_fixes=int(np.sum(~good&after)),K_regressions=int(np.sum(good&~after)),
        net_gain=int(after.sum()-good.sum()),
        original_true_poly_correct=int(np.sum(good&poly)),
        canonical_true_poly_correct=int(np.sum(after&poly)),
        poly_K_fixes=int(np.sum(~good&after&poly)),
        poly_K_regressions=int(np.sum(good&~after&poly)),
        canonical_true_notes_exact=int(note_correct.sum()),
        poly_canonical_true_notes_exact=int(np.sum(poly&note_correct)),
        true_attacks_not_predicted_as_notes=int(np.sum(new==0)))
def run(graph,report,idmt,idmt_report,piano,piano_report,out):
    check(not out.exists(),"S499 canonical postfreeze results are immutable")
    source=json.loads(report.read_text())
    a=json.loads(idmt_report.read_text());b=json.loads(piano_report.read_text())
    check(source["status"]=="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
          and source["no_original_MIDI_XML_annotations_read_in_canonical_model"]
          and source["one_independent_pitch_birth_group_exactly_one_canonical_note_event"]
          and source["no_training_no_promotion"]
          and a["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
          and a["matched_rows_SHA256"]==sha(idmt)
          and b["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
          and b["SHA256"]["paired_label_comparison"]==sha(piano),
          "S499 labels may only be opened after original external acoustic graph is frozen")
    trueg=[json.loads(x) for x in idmt.read_text().splitlines()]
    truep=[json.loads(x) for x in piano.read_text().splitlines()]
    check(len(trueg)==45 and len(truep)==322,
          "S499 genuine original matched MIDI/XML evaluation cohort changed")
    result={}
    for key,truthrows,sourceK,timekey,truthkey,identity,expected_base,polyN in (
        ("guitar_S80",trueg,"original_S80_K","original_XML_anchor","true_active",
         "source_wav",25,21),
        ("piano_V88",truep,"original_V88_K","onset_at_excerpt_s","true_new_onset_MIDI_pitches",
         "source_SHA",42,73)):
        path=graph/f"S498_{key}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        check(source["frozen_group_sha256"][key]==sha(path),
              "S499 prelabel canonical events were modified after reading annotation")
        gs=[json.loads(x) for x in path.read_text().splitlines()]
        check(len(gs)==source["source_summaries"][key]["canonical_musical_onset_groups"]
              and len({z["canonical_event_id"] for z in gs})==len(gs),
              "S499 one BP note onset duplicated")
        grouper=collections.defaultdict(list)
        for j,g in enumerate(gs):grouper[g["source_recording_sha"]].append(j)
        original=[int(x[sourceK]) for x in truthrows]
        y=[len(x[truthkey]) for x in truthrows]
        check(sum(i==j for i,j in zip(y,original))==expected_base
              and sum(a==b and a>=2 for a,b in zip(y,original))==(
                   7 if key=="guitar_S80" else 7)
              and sum(x>=2 for x in y)==polyN,
              "S499 original S80 IDMT or V88 MAESTRO paired exactK baseline drift")
        assignment={};lag=[]
        for rec in sorted({x[identity] if key=="guitar_S80" else
                           x["row_id"].split(":")[0] for x in truthrows}):
            idx=[i for i,t in enumerate(truthrows) if
                    (t[identity] if key=="guitar_S80" else
                     t["row_id"].split(":")[0])==rec]
            cand=grouper.get(rec,[])
            pairs=pairing([float(truthrows[i][timekey]) for i in idx],
                [float(gs[j]["event_time_seconds"]) for j in cand])
            for i,j,delta in pairs:
                assignment[idx[i]]=cand[j];lag.append(delta*1000)
        policies={}
        truth_notes=[set(map(int,x[truthkey])) for x in truthrows]
        for policy,Kfield,pitches in (
           ("BP_birth","K_new","bp_note_pitches"),
           ("BP_active","K_active","bp_active_pitches_at_birth_plus30"),
           ("joint_Knew_softprior","joint_count_newK","bp_note_pitches"),
           ("joint_Kactive_softprior","joint_count_activeK","bp_active_pitches_at_birth_plus30")):
            proposed=[int(gs[assignment[i]][Kfield]) if i in assignment else 0
                      for i in range(len(y))]
            notes=[set(map(int,gs[assignment[i]][pitches])) if i in assignment else set()
                   for i in range(len(y))]
            metrics=tally(y,original,proposed,truth_notes,notes)
            metrics["musical_groups_matched_to_true_onsets"]=len(assignment)
            metrics["musical_groups_unmatched_to_any_evaluated_true_onset"]=len(gs)-len(assignment)
            metrics["unknown_unpaired_musical_groups_are_not_assumed_false_positives"]=True
            metrics["same_source_model_original_reference"]=sourceK
            policies[policy]=metrics
        result[key]=dict(groundtruth_evaluated_positive_note_onsets=len(y),
            original_native_baseline_exactK=expected_base,
            canonical_independent_music_attack_groups=len(gs),
            music_groups_uniquely_matched=len(assignment),
            MIDI_onsets_without_canonical_match=len(y)-len(assignment),
            median_signed_BP_minus_true_onset_ms=(
                float(np.median([-z for z in lag])) if lag else None),
            group_policies=policies)
        print("S499 CANONICAL MUSICAL SCORE",json.dumps(dict(
             source=key,**result[key])),flush=True)
    out.mkdir(parents=True)
    rep=dict(status="S499_GUITAR_S80_PIANO_V88_CANONICAL_MUSICAL_K_AND_NOTES_PAIRED",
        original_S80_complete_only_on_guitar=True,
        original_V88_only_on_piano=True,
        note_onset_matching_Hungarian_one_to_one_80ms=True,
        source_native_candidate_aliases_not_extra_note_events=True,
        source_ExactK_a_soft_prior_to_Knew_and_Kactive=True,
        zero_label_based_training_or_threshold_selection=True,
        original_source_models_and_Live_and_offsets_unchanged=True,
        unknown_external_musical_groups_not_groundtruth_negatives=True,
        prospective_new_corpus_needed_before_any_promotion=True,
        no_promotion=True,
        by_source=result,
        evidence_sha256=dict(S498_graph=sha(report),true_original_IDMT=sha(idmt),
                             true_original_MAESTRO=sha(piano)))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
def selftest():
    assert len(pairing([.1,.12],[.11]))==1
    r=tally([2,1],[1,1],[2,2],[{52,55},{59}],[{52,55},{59,64}])
    check(r["K_fixes"]==1 and r["K_regressions"]==1,
         "S499 canonical note event Exact-K paired correction/regression invariant")
    check(r["canonical_true_notes_exact"]==1,"S499 exact note set must compare MIDI pitches")
    print("S499 SELFTEST PASS one-to-one true onsets and joint count/name loss comparison")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for k in ("graph","report","idmt","idmt-report","piano","piano-report","out"):
        ap.add_argument("--"+k,type=Path)
    v=ap.parse_args()
    if v.self_test:selftest()
    else:
        for k in ("graph","report","idmt","idmt_report","piano","piano_report","out"):
            check(getattr(v,k)is not None,"S499 missing --"+k)
        run(v.graph,v.report,v.idmt,v.idmt_report,v.piano,v.piano_report,v.out)
