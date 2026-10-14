"""S498: canonical *musical onset* as the object of joint K, pitches, offsets.

A single pretrained Basic Pitch onset group is ONE canonical event, never
one per V88 spectral candidate. All native S80/V88 candidates +/-80ms
become evidence attached to that musical event (not extra output notes).
The native K is a SOFT count prior; independent named pitches give K_new,
K_active (+30ms after BP onset), and offset per note. An assignment is
NOT a new training label and no true XML/MIDI is read.

All parameters are fixed *before* S499 scoring. The original production
events/Exact-K/Live/onset/offset are unchanged and exported separately.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s496_joint_onset_note_graph_frozen_no_labels import note_groups,require,sha

SR=44100
RADIUS=.080
MIN_EVIDENCE_CONF=.55
def name(p):
    tones=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    return tones[int(p)%12]+str(int(p)//12-1)
def canonical(ids,members,samples,srcK,notes,corpus):
    require(len(ids)==len(members)==len(samples)==len(srcK),
            "S498 authentic frozen native ID counts changed")
    native=np.asarray(srcK,int)
    bysource=note_groups(notes)
    raw=collections.defaultdict(list)
    for ev in notes:raw[str(ev["WAV_filename_SHA256"])].append(ev)
    result=[]
    for recording in sorted(set(members)|set(bysource)):
        original_indices=np.flatnonzero(np.asarray(members)==recording)
        local_t=np.asarray(samples)[original_indices].astype(float)/SR
        groups=bysource.get(recording,[])
        for j,g in enumerate(groups):
            anchor=float(g["onset"])
            near=original_indices[np.flatnonzero(np.abs(local_t-anchor)<=RADIUS)]
            distances=np.abs(np.asarray(samples)[near]/SR-anchor) if len(near) else np.array([])
            # Every acoustic candidate in this window is EVIDENCE only;
            # canonical note groups are created ONCE, not cloned per candidate.
            nearK=native[near].astype(int) if len(near) else np.asarray([],int)
            pos=np.flatnonzero(nearK>0)
            nearest_positive=None
            if len(pos):
                idx=min(pos,key=lambda z:(distances[z],int(near[z])))
                nearest_positive=int(nearK[idx])
            # Source count acts only as an uncalibrated prior to the musical
            # count; if the independent note detector is confident, its
            # count can override that prior in this research-only branch.
            events=g["events"]
            confidences=[float(x.get("pretrained_note_score",0.)) for x in events]
            allpitches=sorted(g["pitches"])
            minimum=float(min(confidences)) if confidences else 0.
            beginning=len(allpitches)
            t=anchor+.030
            active=sorted({int(ev["pitch_MIDI"]) for ev in raw[recording]
                          if float(ev["predicted_onset_s"])<=t<float(ev["predicted_offset_s"])})
            fullK=len(active)
            # Fixed two-mode soft-K decision: if all independent notes have
            # >= 0.55 score, prefer count from actual evidence. Otherwise
            # prefer nearest source K>0, without inventing a source K0 as
            # a negative ground truth.
            def fuse(independent_K):
                bp=independent_K if 1<=independent_K<=6 else None
                if minimum>=MIN_EVIDENCE_CONF and bp is not None:return bp
                if nearest_positive is not None:return nearest_positive
                return bp if bp is not None else 0
            # Do not assume repeated same pitch is duplicate: reattack at
            # the same pitch can be legitimate even while a previous string
            # or piano pedal sustains. Record overlap as ambiguous.
            same_pitch_onsets={int(q):[
                float(v["predicted_onset_s"]) for v in raw[recording]
                if int(v["pitch_MIDI"])==int(q) and
                float(v["predicted_onset_s"])<anchor] for q in allpitches}
            reattack_on_sustained_pitch=any(
                any(float(e["predicted_onset_s"])<anchor<
                    float(e["predicted_offset_s"])
                    for e in raw[recording] if int(e["pitch_MIDI"])==q)
                for q in allpitches)
            root=nearest_positive
            result.append(dict(
                source=corpus,source_recording_sha=recording,
                group_number_in_source=j,
                canonical_event_id=recording+":BP:"+str(j),
                event_time_seconds=round(anchor,6),
                K_new=beginning,K_active=fullK,
                joint_count_newK=fuse(beginning),
                joint_count_activeK=fuse(fullK),
                original_source_nearest_positive_K_prior=root,
                source_original_native_candidates_within80ms=len(near),
                source_original_native_positive_candidates_within80ms=int(len(pos)),
                source_original_native_K0_candidates_within80ms=int(np.sum(nearK==0)),
                source_original_native_candidate_ids=[str(ids[k]) for k in near],
                native_candidate_time_offsets_ms=[
                    round(float(np.asarray(samples)[k]/SR-anchor)*1000,3) for k in near],
                all_native_candidates_evidence_only=True,
                bp_note_pitches=allpitches,
                bp_note_names=[name(k) for k in allpitches],
                bp_active_pitches_at_birth_plus30=active,
                bp_active_note_names_at_birth_plus30=[name(k) for k in active],
                note_offsets_seconds=[
                    dict(pitch_MIDI=int(q),offset_seconds=round(max(float(v["predicted_offset_s"])
                       for v in events if int(v["pitch_MIDI"])==q),6))
                    for q in allpitches],
                lowest_independent_note_score=round(minimum,5),
                reattack_during_existing_same_pitch_sustain=bool(reattack_on_sustained_pitch),
                source_alias_candidate_count=max(0,len(near)-1),
                K_truth_unknown_at_inference=True))
    idsout=[q["canonical_event_id"] for q in result]
    require(len(idsout)==len(set(idsout)),
            "S498 one physical onset group accidentally expanded to two canonical notes")
    require(all(z["K_new"]>0 and len(z["bp_note_names"])==z["K_new"]
                for z in result),
            "S498 canonical note birth count must equal distinct named pitches")
    return result

def run(s491,s491report,s487,s487report,s485,s485report,out):
    require(not out.exists(),"S498 canonical musical onset graph cannot be overwritten")
    src=json.loads(s491report.read_text())
    guitar=json.loads(s487report.read_text())
    piano=json.loads(s485report.read_text())
    require(src["status"]=="S491_TWO_SOURCE_K0_ONLY_ONE_TO_ONE_EXTERNAL_PITCH_BIRTH_FROZEN_NO_TRUTH"
            and src["no_midi_or_xml_label_read_during_freeze"]
            and guitar["all_output_note_events_sha256"]==sha(s487)
            and guitar["original_XML_NOT_READ_before_note_freeze"]
            and piano["frozen_notes_SHA256"]==sha(s485)
            and piano["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False,
            "S498 independent notes and historical source events must be SHA frozen")
    out.mkdir(parents=True)
    summaries={};files={}
    for source,note_file in (("guitar_S80",s487),("piano_V88",s485)):
        native=s491/f"S491_{source}_SOURCE_K0_ONLY_FROZEN.npz"
        require(sha(native)==src["source_output_SHA256"][source]["sha256"],
                "S498 original native source Exact-K must be bit-identical")
        with np.load(native,allow_pickle=False) as z:
            ids=z["row_id"].astype(str)
            members=z["member"].astype(str)
            samples=z["start_samples"].astype(np.int64)
            K=z["source_K"].astype(np.int8)
        notes=[json.loads(v) for v in note_file.read_text().splitlines()]
        groups=canonical(ids,members,samples,K,notes,source)
        file=out/f"S498_{source}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        file.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in groups))
        names=set(z["canonical_event_id"] for z in groups)
        require(len(names)==len(groups),"S498 one group becomes two musical events")
        summaries[source]=dict(native_source_candidate_events=len(ids),
            original_native_K0=int(np.sum(K==0)),
            canonical_musical_onset_groups=len(groups),
            canonical_groups_with_no_native_source_support=sum(
                e["source_original_native_candidates_within80ms"]==0 for e in groups),
            source_native_candidate_edges_total=sum(
                e["source_original_native_candidates_within80ms"] for e in groups),
            candidate_aliases_not_extra_output_notes=sum(
                e["source_alias_candidate_count"] for e in groups),
            independent_canonical_Knew_poly=sum(e["K_new"]>=2 for e in groups),
            canonical_events_changed_in_K_by_soft_source_prior_new=sum(
                e["joint_count_newK"]!=e["K_new"] for e in groups),
            canonical_events_changed_in_K_by_soft_source_prior_active=sum(
                e["joint_count_activeK"]!=e["K_active"] for e in groups),
            canon_birth_event_ids_unique=True,no_original_native_K_modified=True)
        files[source]=sha(file)
        print("S498 ORIGINAL MUSICAL EVENTS",json.dumps(dict(
            source=source,**summaries[source])),flush=True)
    report=dict(
        status="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL",
        source_guitar_original_full_S80=True,
        source_piano_original_V88_only=True,
        one_independent_pitch_birth_group_exactly_one_canonical_note_event=True,
        native_V88_S80_spectral_events_are_evidence_NOT_another_musical_attack=True,
        source_ExactK_used_as_soft_prior_to_independent_named_notes=True,
        Knew_and_Kactive_independently_reported=True,
        prior_confidence_threshold_fixed_without_external_truth=MIN_EVIDENCE_CONF,
        source_no_original_S80_or_V88_changes=True,
        no_original_MIDI_XML_annotations_read_in_canonical_model=True,
        no_training_no_promotion=True,
        previously_exposed_external_corpora_require_new_prospective_validation=True,
        source_summaries=summaries,
        frozen_group_sha256=files,
        original_source_input_sha256=dict(S491=sha(s491report),guitar_notes=sha(s487),
                                          piano_notes=sha(s485)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    notes=[
      dict(WAV_filename_SHA256="x",predicted_onset_s=.100,
           predicted_offset_s=.5,pitch_MIDI=52,pretrained_note_score=.9),
      dict(WAV_filename_SHA256="x",predicted_onset_s=.105,
           predicted_offset_s=.5,pitch_MIDI=55,pretrained_note_score=.9)]
    a=canonical(np.array(["x:80","x:90"]),np.array(["x","x"]),
                np.array([int(.080*SR),int(.090*SR)]),
                np.array([0,2],np.int8),notes,"test")
    require(len(a)==1 and a[0]["K_new"]==2 and
        a[0]["bp_note_names"]==["E3","G3"] and
        a[0]["source_alias_candidate_count"]==1,
        "S498 multiple source V88 candidates MUST form one acoustic note event")
    # A genuine second attack of a previously sounding E3 must NOT be
    # removed simply because the first E3 offset has not yet elapsed.
    notes.append(dict(WAV_filename_SHA256="x",predicted_onset_s=.260,
          predicted_offset_s=.6,pitch_MIDI=52,pretrained_note_score=.92))
    second=canonical(np.array(["x:80","x:90"]),np.array(["x","x"]),
                np.array([int(.08*SR),int(.09*SR)]),
                np.array([0,2],np.int8),notes,"test")
    require(len(second)==2 and second[1]["reattack_during_existing_same_pitch_sustain"],
            "S498 repeated pitch with separate real attack must remain separate")
    print("S498 SELFTEST PASS native duplicates deduplicated, true same-pitch reattack preserved")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s491","s491report","s487","s487report","s485","s485report","out"):
        p.add_argument("--"+k,type=Path)
    x=p.parse_args()
    if x.self_test:selftest()
    else:
        for k in ("s491","s491report","s487","s487report","s485","s485report","out"):
            require(getattr(x,k)is not None,"S498 missing "+k)
        run(x.s491,x.s491report,x.s487,x.s487report,x.s485,x.s485report,x.out)
