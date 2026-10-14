"""S494 alternate SHADOW K0 evidence association, frozen without true labels.

S493 root-cause: BP true pitches lie <80ms of all 7 original guitar K0
events but strict one-to-one native-event matching S491 gives them to
competing original V88 events. Proposed experiment: *source-K0-FIRST*
one-to-one matching of Basic Pitch onset groups, exclusively among K0
events; allow active-pitch measurement at BP group onset +30ms instead
of the earlier native V88 timestamp.

This is NOT a deployable proof: allowing source-K0 priority can steal
birth evidence from already-correct K>=1 nearby events, and unmatched
original V88 459 events do not have verified negative truth labels.
Report explicit potential duplicate-note competitor collisions.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s491_freeze_K0_only_BasicPitch_birth_gate_two_sources import (
    group_birth,one_to_one,need)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_one(source_path,note_path):
    with np.load(source_path,allow_pickle=False) as data:
        ids=data["row_id"].astype(str)
        wav=data["member"].astype(str)
        starts=data["start_samples"].astype(np.int64)
        old=data["source_K"].astype(np.int8)
    notes=[json.loads(line) for line in note_path.read_text().splitlines()]
    groups=group_birth(notes)
    events_by_source=collections.defaultdict(list)
    for ev in notes:
        events_by_source[str(ev["WAV_filename_SHA256"])].append(ev)
    birth=old.copy()
    active=old.copy()
    collisions=[]
    matches=0
    for key in sorted(set(wav)):
        indices=np.flatnonzero(wav==key)
        k0indices=indices[old[indices]==0]
        if not len(k0indices) or not groups.get(key):continue
        candidates=groups[key]
        rowtime=starts[k0indices].astype(float)/44100.
        grouptime=np.array([q["time"] for q in candidates],float)
        pairing=one_to_one(rowtime,grouptime,.080)
        matches+=len(pairing)
        for i,j,dist in pairing:
            idx=int(k0indices[i])
            group=candidates[j]
            pitches=set(group["pitches"])
            activep={int(ev["pitch_MIDI"]) for ev in events_by_source[key]
                     if (float(ev["predicted_onset_s"])<=group["time"]+.030
                         <float(ev["predicted_offset_s"]))}
            if 1<=len(pitches)<=6:birth[idx]=len(pitches)
            if 1<=len(activep)<=6:active[idx]=len(activep)
            other=indices[(old[indices]>0)&
                      (abs(starts[indices]/44100.-group["time"])<=.080)]
            closer=sum(abs(starts[x]/44100.-group["time"])<dist for x in other)
            collisions.append(dict(row_id=str(ids[idx]),
                matched_BP_birth_onset_s=float(group["time"]),
                original_native_event_s=float(starts[idx]/44100.),
                proposed_birth_K=int(birth[idx]),
                proposed_active_birthplus30_K=int(active[idx]),
                competing_original_source_nonzero_K_events_within80ms=int(len(other)),
                other_original_nonzero_K_strictly_closer_to_BP_onset=int(closer),
                projected_duplicate_note_risk_without_true_labels=bool(closer)))
    need(np.array_equal(birth[old>0],old[old>0])
         and np.array_equal(active[old>0],old[old>0]),
         "S494 source K>=1 not allowed to change")
    return dict(row_id=ids,source_K=old,
        proposed_K0_priority_BP_birth_K=birth,
        proposed_K0_priority_BP_activeAtBPonsetPlus30_K=active,
        candidates=collisions,one_to_one_groups_paired=matches)
def run(s491,s491report,s487,s487report,s485,s485report,out):
    need(not out.exists(),"S494 source label-free priority proposals already frozen")
    r=json.loads(s491report.read_text())
    a=json.loads(s487report.read_text())
    b=json.loads(s485report.read_text())
    need(r["status"]=="S491_TWO_SOURCE_K0_ONLY_ONE_TO_ONE_EXTERNAL_PITCH_BIRTH_FROZEN_NO_TRUTH"
         and r["no_midi_or_xml_label_read_during_freeze"]
         and r["source_output_SHA256"]["guitar_S80"]["sha256"]==
             sha(s491/"S491_guitar_S80_SOURCE_K0_ONLY_FROZEN.npz")
         and r["source_output_SHA256"]["piano_V88"]["sha256"]==
             sha(s491/"S491_piano_V88_SOURCE_K0_ONLY_FROZEN.npz")
         and a["all_output_note_events_sha256"]==sha(s487)
         and b["frozen_notes_SHA256"]==sha(s485)
         and a["original_XML_NOT_READ_before_note_freeze"]
         and b["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False,
         "S494 original independent pretrained notes and K0 source predictions must remain unchanged")
    result={}
    out.mkdir(parents=True)
    for key,path,notes in (
        ("guitar_S80",s491/"S491_guitar_S80_SOURCE_K0_ONLY_FROZEN.npz",s487),
        ("piano_V88",s491/"S491_piano_V88_SOURCE_K0_ONLY_FROZEN.npz",s485)):
        z=run_one(path,notes)
        pred=out/f"S494_{key}_K0_FIRST_PRIORITY_SHADOW_PRETRUTH.npz"
        np.savez_compressed(pred,
            row_id=z["row_id"],source_K=z["source_K"],
            proposed_K0_priority_BP_birth_K=z["proposed_K0_priority_BP_birth_K"],
            proposed_K0_priority_BP_activeAtBPonsetPlus30_K=
                z["proposed_K0_priority_BP_activeAtBPonsetPlus30_K"])
        detail=out/f"S494_{key}_potential_competing_source_K_nonzero.jsonl"
        detail.write_text("".join(json.dumps(x)+"\n" for x in z["candidates"]))
        src=z["source_K"]
        p1=z["proposed_K0_priority_BP_birth_K"]
        p2=z["proposed_K0_priority_BP_activeAtBPonsetPlus30_K"]
        result[key]=dict(source_native_events=len(src),
            original_K0=int(np.sum(src==0)),
            matched_independent_BP_attack_groups=int(z["one_to_one_groups_paired"]),
            proposed_K0_birth_count=int(np.sum((src==0)&(p1>0))),
            proposed_K0_active_count=int(np.sum((src==0)&(p2>0))),
            competing_source_K_ge1_near_BP_attacks=int(sum(
                q["competing_original_source_nonzero_K_events_within80ms"]>0
                for q in z["candidates"])),
            competing_source_K_ge1_strictly_closer_to_BP_onset_n=int(sum(
                q["projected_duplicate_note_risk_without_true_labels"]
                for q in z["candidates"])),
            original_nonzero_K_modified=0,
            SHA256=dict(pretruth_policy_file=sha(pred),
                        competing_native_event_analysis=sha(detail)))
        print("S494 K0-PRIORITY",json.dumps(dict(corpus=key,**result[key])),flush=True)
    report=dict(status="S494_SOURCE_K0_FIRST_PRETRAINED_ONSET_EVIDENCE_PRELABEL_SHADOW",
        alternative_to_S491_no_model_refit=True,
        original_S80_and_V88_unchanged=True,
        read_no_XML_or_MIDI_truth_during_inference=True,
        already_exposed_corpus_cannot_claim_blindness=True,
        potential_duplicate_note_sourceKnonzero_conflicts_explicitly_audited=True,
        no_safety_promotion_eligible_without_true_false_event_annotations=True,
        original_Live_and_onset_offset_untouched=True,
        two_birth_and_activeAtBPonset_policies_fixed_before_scoring=True,
        no_promotion=True,
        by_source=result,
        input_sha256=dict(original_source_K0_frozen=sha(s491report),
            original_guitar_BP_note_events=sha(s487),
            original_piano_BP_note_events=sha(s485)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    # K0-first one-to-one cannot assign one real BP attack to >1 K0 event.
    native=np.array([.1,.101,.4])
    BP=np.array([.102])
    assert len(one_to_one(native,BP))==1
    print("S494 SELFTEST PASS source K0-first evidence still one-to-one")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for k in ("s491","s491report","s487","s487report","s485","s485report","out"):
        ap.add_argument("--"+k,type=Path)
    x=ap.parse_args()
    if x.self_test:selftest()
    else:
        for k in ("s491","s491report","s487","s487report","s485","s485report","out"):
            need(getattr(x,k)is not None,"S494 missing --"+k)
        run(x.s491,x.s491report,x.s487,x.s487report,x.s485,x.s485report,x.out)
