"""S500 — cardinality-constrained *actual note names* on canonical musical events.

Read SHA-frozen S498 canonical onset graph, source-independent pretrained
BP named pitch evidence, and source K as a SOFT count prior. When proposed K
is smaller than the number of detected pitches, select the strongest
supported fundamental candidates, not arbitrary frequencies. Never invent
a note for a K larger than available evidence. Pitch-onset harmonics
(+12,+19,+24 semitones) are a documented optional penalty, NOT a hard
octave ban. No XML/MIDI note truth is opened; no original S80/Live edits.

Two FIXED decoding rules ("confidence", "harmonic_discount") ×
("new", "active") yield four shadow note-set policies before scoring.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
def need(c,msg):
    if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def note_name(x):
    names=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    return names[int(x)%12]+str(int(x)//12-1)
def supported(events):
    p={}
    for e in events:
        k=int(e["pitch_MIDI"])
        a=float(e.get("pretrained_note_score",0))
        if k not in p or a>p[k]:p[k]=a
    return p
def choose(scored,desired,method):
    """Constrained top-K, abstain if K > count of candidate fundamentals."""
    evidence=dict(scored)
    available=len(evidence)
    if available==0:return [],dict(reason="no_pitch_evidence",requested_K=desired,available_K=0)
    if desired<1 or desired>6 or desired>available:
        return sorted(evidence),dict(reason="K_disagrees_with_available_candidates_abstain",
                          requested_K=desired,available_K=available)
    def harmonic_discount(pitch):
        if method!="harmonic_discount":return 0.
        prev=[q for q in evidence if pitch-q in (12,19,24)]
        if not prev:return 0.
        # Penalize uncertain potential harmonic when lower independent
        # fundamental is at least as well supported; not a universal ban.
        return .09 if any(evidence[q]>=evidence[pitch] for q in prev) else 0.
    order=sorted(evidence,key=lambda p:(-(evidence[p]-harmonic_discount(p)),-evidence[p],p))
    chosen=sorted(order[:desired])
    return chosen,dict(reason="count_constrained_rank",
         requested_K=desired,available_K=available)
def run(graph,graph_report,guitar_notes,guitar_report,piano_notes,piano_report,out):
    need(not out.exists(),"S500 named-note selection predictions immutable")
    g=json.loads(graph_report.read_text())
    a=json.loads(guitar_report.read_text());b=json.loads(piano_report.read_text())
    need(g["status"]=="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
        and g["no_original_MIDI_XML_annotations_read_in_canonical_model"]
        and a["all_output_note_events_sha256"]==sha(guitar_notes)
        and a["original_XML_NOT_READ_before_note_freeze"]
        and b["frozen_notes_SHA256"]==sha(piano_notes)
        and b["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False,
        "S500 true source independent notes / canonical graph must be frozen before any truth")
    out.mkdir(parents=True)
    summaries={};output_hash={}
    for source,note_file in (("guitar_S80",guitar_notes),("piano_V88",piano_notes)):
        canonical_file=graph/f"S498_{source}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        need(g["frozen_group_sha256"][source]==sha(canonical_file),
             "S500 canonical one-note birth group changed after capture")
        events=[json.loads(line) for line in canonical_file.read_text().splitlines()]
        raw=[json.loads(line) for line in note_file.read_text().splitlines()]
        bysource=collections.defaultdict(list)
        for e in raw:bysource[str(e["WAV_filename_SHA256"])].append(e)
        rendered=[];abstentions=collections.Counter();changed=collections.Counter()
        for q in events:
            rows=bysource[q["source_recording_sha"]]
            st=float(q["event_time_seconds"])
            # Fixed group of note births by original onset ID and 20ms
            # grouping. Rebuild scores from frozen raw notes, NOT labels.
            born=[e for e in rows if st-.0002<=float(e["predicted_onset_s"])<=st+.0202]
            active=[e for e in rows if
                    float(e["predicted_onset_s"])<=st+.030<
                    float(e["predicted_offset_s"])]
            base_birth=supported(born);base_active=supported(active)
            # Defensive: compare to the set of frozen canonical observed
            # notes; numerical 6-digit rounding can create <.2ms offsets.
            need(set(base_birth)==set(q["bp_note_pitches"]) and
                 set(base_active)==set(q["bp_active_pitches_at_birth_plus30"]),
                 "S500 cannot join original BP notes to frozen canonical event within 20ms")
            record=dict(canonical_event_id=q["canonical_event_id"],
               source_recording_sha=q["source_recording_sha"],
               onset_seconds=q["event_time_seconds"],
               source_K_prior=q["original_source_nearest_positive_K_prior"],
               detected_birth_K=q["K_new"],detected_active_K=q["K_active"],
               detected_birth_note_MIDI=q["bp_note_pitches"],
               detected_active_note_MIDI=q["bp_active_pitches_at_birth_plus30"],
               original_model_prediction_modified=False)
            for kind,evidence,target in (
                ("new",base_birth,int(q["joint_count_newK"])),
                ("active",base_active,int(q["joint_count_activeK"]))):
                for method in ("confidence","harmonic_discount"):
                    pitches,reason=choose(evidence,target,method)
                    tag=kind+"_"+method
                    record["joint_"+tag+"_pitch_MIDI"]=pitches
                    record["joint_"+tag+"_note_names"]=[note_name(v) for v in pitches]
                    record["joint_"+tag+"_requested_K"]=target
                    record["joint_"+tag+"_actual_K"]=len(pitches)
                    record["joint_"+tag+"_reason"]=reason["reason"]
                    abstentions[tag+":"+reason["reason"]]+=1
                    changed[tag]+=int(set(pitches)!=set(evidence))
            rendered.append(record)
        path=out/f"S500_{source}_K_CONSTRAINED_ACTUAL_NOTE_NAMES_PRELABEL.jsonl"
        path.write_text("".join(json.dumps(v,ensure_ascii=False)+"\n" for v in rendered))
        output_hash[source]=sha(path)
        summaries[source]=dict(original_canonical_musical_birth_groups=len(events),
            ranked_musical_note_groups=len(rendered),
            candidate_groups_with_K_note_subselection_by_policy=dict(changed),
            explicit_unsatisfied_K_abstentions_by_policy=dict(abstentions),
            each_event_notes_are_real_pretrained_MIDI_pitches_not_hallucinations=True,
            no_original_source_ExactK_or_onset_offset_changed=True)
        print("S500 NOTE NAMES",json.dumps(dict(corpus=source,**summaries[source])),flush=True)
    report=dict(status="S500_TWO_SOURCE_JOINT_EXACTK_CONSTRAINED_REAL_NOTE_NAMES_PRELABEL",
       count_Knew_vs_Kactive_distinct=True,
       four_precommitted_decoding_policies_untuned=True,
       minimum_pitch_evidence_not_invented_for_large_K=True,
       octave_harmonic_penalty_soft_and_optional=True,
       source_guitar_original_full_S80_and_piano_only_V88=True,
       original_S80_freeze_and_Live_untouched=True,
       no_training_no_XML_MIDI_label_access=True,
       external_corpora_previously_exposed_not_new_blind=True,
       no_promotion=True,
       by_source=summaries,
       frozen_output_shas=output_hash,
       source_shas=dict(S498=sha(graph_report),
                        guitar_real_BP_notes=sha(guitar_notes),
                        piano_real_BP_notes=sha(piano_notes)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    z={52:.95,64:.60,55:.90}
    same,_=choose(z,2,"confidence")
    reduce,_=choose(z,2,"harmonic_discount")
    need(same==[52,55] and reduce==[52,55],
         "S500 do not select unreliable octave artifact E4 instead of E3+G3")
    no,why=choose({52:.9},3,"confidence")
    need(no==[52] and why["reason"].startswith("K_disagrees"),
         "S500 never invent two extra notes to satisfy K=3 without pitch evidence")
    print("S500 SELFTEST PASS distinct pitch confidence, harmonic penalty and no note hallucinations")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("graph","graph-report","guitar-notes","guitar-report",
              "piano-notes","piano-report","out"):
        p.add_argument("--"+k,type=Path)
    x=p.parse_args()
    if x.self_test:selftest()
    else:
        for k in ("graph","graph_report","guitar_notes","guitar_report",
                  "piano_notes","piano_report","out"):
            need(getattr(x,k)is not None,"S500 missing --"+k)
        run(x.graph,x.graph_report,x.guitar_notes,x.guitar_report,
            x.piano_notes,x.piano_report,x.out)
