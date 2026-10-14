"""S496 — experimental joint temporal graph: original Exact-K + note births.

Every independent Basic Pitch note-on group can belong to at most ONE
native acoustic event (no double assignment), even if many V88 candidates
claim the same harmonic. Distinguish K_new at the note birth from K_active
30ms later. Original S80/V88 are soft priors for association, never edited.

Three *fixed before external truth* risk policies:
  "guarded": one-to-one matching, source K>=1 immutable, K0 may be
     rescued only if no source-K>=1 competes within ±80ms.
  "joint_birth": one-to-one assignment, shadow revisions permitted
     on uniquely supported onsets using number of new notes.
  "joint_active": one-to-one assignment, shadow revisions permitted
     on uniquely supported onsets using active notes at BP time+30ms.
The last two are DIAGNOSTIC ONLY and not safe to promote. Source
annotations/label files are never read in this module.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

WINDOW=.080
SAMPLE_RATE=44100
ASSIGNMENT_K0_PENALTY=.20
UNMATCH_COST=1.35
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def note_groups(events):
    by_source=collections.defaultdict(list)
    for e in events:by_source[str(e["WAV_filename_SHA256"])].append(e)
    result={}
    for source,rows in sorted(by_source.items()):
        groups=[]
        for e in sorted(rows,key=lambda x:(float(x["predicted_onset_s"]),int(x["pitch_MIDI"]))):
            st=float(e["predicted_onset_s"])
            require(0<=st<15 and float(e["predicted_offset_s"])>st,
                    "S496 invalid frozen note onset or offset")
            if not groups or st-groups[-1]["onset"]>.020:
                groups.append(dict(onset=st,events=[],pitches=set()))
            groups[-1]["events"].append(e)
            groups[-1]["pitches"].add(int(e["pitch_MIDI"]))
        result[source]=groups
    return result
def assign(times,sourceK,groups):
    """One-to-one penalized joint graph matching with unmatched dummy nodes.

    Original K influences matching *softly* but never forbids correction.
    Deterministic tie break by native event time/index, not ground truth.
    """
    n=len(times);m=len(groups)
    if n==0 or m==0:return []
    time=np.asarray(times,float)
    birth=np.asarray([x["onset"] for x in groups],float)
    dist=np.abs(time[:,None]-birth[None,:])
    cost=dist/WINDOW
    cost+=np.where(np.asarray(sourceK)[:,None]==0,ASSIGNMENT_K0_PENALTY,0.)
    # Penalize improbable origin: more notes born but near-zero original K.
    # Small K-dependent mismatch is a *soft* constraint, never the output.
    newK=np.asarray([len(x["pitches"]) for x in groups],int)
    cost+=.07*np.minimum(4,np.abs(np.asarray(sourceK)[:,None]-newK[None,:]))
    cost=np.where(dist<=WINDOW,cost,1e5)
    cost+=np.arange(n)[:,None]*1e-10+np.arange(m)[None,:]*1e-12
    c=np.concatenate((cost,np.full((n,n),UNMATCH_COST)),axis=1)
    rows,cols=linear_sum_assignment(c)
    matched=[(int(i),int(j),float(dist[i,j]))
             for i,j in zip(rows,cols) if j<m and dist[i,j]<=WINDOW
             and cost[i,j]<UNMATCH_COST]
    require(len({i for i,j,d in matched})==len(matched)
            and len({j for i,j,d in matched})==len(matched),
            "S496 one original pitch birth assigned to multiple native events")
    return matched
def route(ids,members,samples,source_K,notes,source_name):
    N=len(ids)
    require(N==len(members)==len(samples)==len(source_K) and
            len(set(ids))==N,"S496 authentic source event identity contract")
    original=np.asarray(source_K,np.int8)
    require(np.isin(original,np.arange(7)).all(),"S496 original source K out of range")
    all_groups=note_groups(notes)
    notes_by_source=collections.defaultdict(list)
    for e in notes:notes_by_source[str(e["WAV_filename_SHA256"])].append(e)
    guarded=original.copy();joint_birth=original.copy();joint_active=original.copy()
    owner=np.full(N,-1,np.int32)
    event_type=np.full(N,"unmatched",dtype="<U24")
    uncertain=np.zeros(N,bool)
    ambiguous_pitch=np.zeros(N,bool)
    source_nonzero_competitor=np.zeros(N,bool)
    graph=[];groups_assigned=0;total_groups=0
    for src in sorted(set(members)):
        chosen=np.flatnonzero(np.asarray(members)==src)
        local=chosen[np.argsort(np.asarray(samples)[chosen],kind="stable")]
        times=np.asarray(samples)[local]/SAMPLE_RATE
        candidates=all_groups.get(src,[])
        total_groups+=len(candidates)
        attached=assign(times,original[local],candidates)
        groups_assigned+=len(attached)
        for i,j,lag in attached:
            idx=int(local[i]);g=candidates[j]
            k_new=len(g["pitches"])
            t=float(g["onset"])+.030
            active={int(e["pitch_MIDI"]) for e in notes_by_source[src]
                    if float(e["predicted_onset_s"])<=t<float(e["predicted_offset_s"])}
            k_active=len(active)
            near=np.flatnonzero(np.abs(times-g["onset"])<=WINDOW)
            positive=near[original[local[near]]>0]
            positives_else=[int(local[z]) for z in positive if int(local[z])!=idx]
            # This graph did not prove that neighbouring native nonzero K
            # describes the SAME pitch. Mark conflict; DO NOT remove it.
            competition=len(positives_else)>0
            positive_closer=any(abs(float(samples[x]/SAMPLE_RATE)-g["onset"])<lag
                                for x in positives_else)
            raw_scores=np.asarray([float(e.get("pretrained_note_score",0.))
                                  for e in g["events"]],float)
            strong=bool(len(raw_scores) and np.min(raw_scores)>=.65)
            # Score and temporal windows are fixed a priori, no label tuning.
            # A second native event within 15 ms of the assigned onset means
            # the mapping itself is unstable; do not change the official K.
            n_native_close=int(np.sum(np.abs(times-g["onset"])<=.015))
            distinctive=n_native_close==1
            ambiguous=bool(len(set(g["pitches"]))!=len(g["events"]))
            qualified=strong and distinctive and not ambiguous
            owner[idx]=j
            event_type[idx]="matched_conflict" if competition else "matched_clear"
            source_nonzero_competitor[idx]=competition
            ambiguous_pitch[idx]=ambiguous
            uncertain[idx]=not qualified or competition
            # Conservative rescue is allowed only for an original K0
            # with unambiguous, strong, and nonduplicated independent birth.
            if original[idx]==0 and not competition and qualified and 1<=k_new<=6:
                guarded[idx]=k_new
            # Joint inferred revisions of positive K require independent
            # evidence and margin, and are kept SHADOW only.
            if qualified and 1<=k_new<=6:
                joint_birth[idx]=k_new
            if qualified and 1<=k_active<=6:
                joint_active[idx]=k_active
            graph.append(dict(corpus=source_name,row_id=str(ids[idx]),
                native_K=int(original[idx]),source_audio_id=src,
                native_onset_s=round(float(times[i]),6),
                inferred_birth_onset_s=round(float(g["onset"]),6),
                temporal_distance_ms=round(1000*lag,3),
                original_competing_nonzero_K=int(len(positives_else)),
                nonzero_competitor_closer_than_assignee=bool(positive_closer),
                nearby_native_count_within15ms=n_native_close,
                independent_note_confidence_min=round(float(np.min(raw_scores)),4),
                independent_note_pitches=sorted(g["pitches"]),
                active_pitches_at_BP_onset_plus30ms=sorted(active),
                K_new=int(k_new),K_active=int(k_active),
                assignment_confident=bool(qualified),
                ambiguous_duplicate_pitch=bool(ambiguous),
                guarded_inferred_K=int(guarded[idx]),
                joint_newK_shadow=int(joint_birth[idx]),
                joint_activeK_shadow=int(joint_active[idx]),
                no_annotations_used=True))
    require(np.array_equal(guarded[original>0],original[original>0]),
            "S496 guarded graph erroneously overwrote original S80/V88 positive K")
    require(np.isin(guarded,np.arange(7)).all() and
            np.isin(joint_birth,np.arange(7)).all() and
            np.isin(joint_active,np.arange(7)).all(),
            "S496 graph inferred K outside supported 0-6")
    require(len({z["row_id"] for z in graph})==groups_assigned,
            "S496 graph assigned duplicate native event ID")
    return dict(source_K=original,
        guarded_sourceK=guarded,
        joint_Knew_shadow=joint_birth,joint_Kactive_shadow=joint_active,
        note_group_assigned_to_native=owner,
        duplicated_pitches_among_BP_group=ambiguous_pitch,
        ambiguous_event_conflict=uncertain,
        competing_original_positiveK=source_nonzero_competitor,
        graph=graph,
        total_BP_note_birth_groups=total_groups,
        unique_assigned_note_birth_groups=groups_assigned)

def run(s491,s491report,s487,s487report,s485,s485report,out):
    require(not out.exists(),"S496 frozen shadow event graph cannot overwrite")
    source=json.loads(s491report.read_text())
    guitar=json.loads(s487report.read_text())
    piano=json.loads(s485report.read_text())
    require(source["status"]=="S491_TWO_SOURCE_K0_ONLY_ONE_TO_ONE_EXTERNAL_PITCH_BIRTH_FROZEN_NO_TRUTH"
            and source["no_midi_or_xml_label_read_during_freeze"]
            and guitar["all_output_note_events_sha256"]==sha(s487)
            and guitar["original_XML_NOT_READ_before_note_freeze"]
            and piano["frozen_notes_SHA256"]==sha(s485)
            and piano["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False,
            "S496 pre-label source K and named pitch original SHA changed")
    out.mkdir(parents=True)
    summary={};verified_files={};links=[]
    for key,note_file in (("guitar_S80",s487),("piano_V88",s485)):
        native=s491/f"S491_{key}_SOURCE_K0_ONLY_FROZEN.npz"
        require(source["source_output_SHA256"][key]["sha256"]==sha(native),
                "S496 actual originally sealed S80/V88 source file changed")
        with np.load(native,allow_pickle=False) as z:
            ids=z["row_id"].astype(str);member=z["member"].astype(str)
            start=z["start_samples"].astype(np.int64)
            old=z["source_K"].astype(np.int8)
        ev=[json.loads(row) for row in note_file.read_text().splitlines()]
        output=route(ids,member,start,old,ev,key)
        raw=out/f"S496_{key}_JOINT_SHADOW_FROZEN.npz"
        np.savez_compressed(raw,row_id=ids,member=member,start_samples=start,
             source_K=old,guarded_sourceK=output["guarded_sourceK"],
             joint_Knew_shadow=output["joint_Knew_shadow"],
             joint_Kactive_shadow=output["joint_Kactive_shadow"],
             ambiguous_event_conflict=output["ambiguous_event_conflict"],
             competing_original_positiveK=output["competing_original_positiveK"],
             note_group_assigned_to_native=output["note_group_assigned_to_native"])
        log=out/f"S496_{key}_EVENT_NOTE_GRAPH.jsonl"
        log.write_text("".join(json.dumps(e,ensure_ascii=False)+"\n" for e in output["graph"]))
        summary[key]=dict(events=len(ids),
             original_K0=int(np.sum(old==0)),
             original_positive_K=int(np.sum(old>0)),
             source_K0_to_positive_guarded=int(np.sum((old==0)&(output["guarded_sourceK"]>0))),
             source_positive_K_changed_guarded=int(np.sum((old>0)&(output["guarded_sourceK"]!=old))),
             source_K0_to_positive_joint_birth=int(np.sum((old==0)&(output["joint_Knew_shadow"]>0))),
             original_nonzero_K_modified_joint_birth=int(np.sum((old>0)&(output["joint_Knew_shadow"]!=old))),
             original_nonzero_K_modified_joint_active=int(np.sum((old>0)&(output["joint_Kactive_shadow"]!=old))),
             independent_BP_groups=output["total_BP_note_birth_groups"],
             independent_BP_groups_assigned_once=output["unique_assigned_note_birth_groups"],
             uncertain_matched_rows=int(np.sum(output["ambiguous_event_conflict"])),
             matched_rows_near_other_source_K_nonzero=int(np.sum(output["competing_original_positiveK"])),
             no_true_K_used_during_decisions=True)
        verified_files[key]=dict(frozen_source_SHA256=sha(raw),graph_SHA256=sha(log))
        print("S496 GRAPH FROZEN",json.dumps(dict(corpus=key,**summary[key])),flush=True)
    doc=dict(status="S496_TWO_SOURCE_JOINT_GRAPH_K_ONSET_PITCH_SHADOW_PRELABEL",
         original_source_guitar_final_S80_and_piano_V88_distinct=True,
         comparison_truth_definitions_not_interchangeable=True,
         K_new_and_K_active_reported_separately=True,
         all_BP_pitch_birth_groups_unique_owner=True,
         original_Source_K_soft_prior_in_graph_assignment=True,
         original_source_S80_or_V88_files_unchanged=True,
         fixed_nonnative_pitch_strength_min=.65,
         fixed_temporal_onset_match_window_ms=80,
         guarded_policy_no_source_Kpositive_changes=True,
         joint_nonzero_K_revision_is_shadow_only=True,
         experimental_not_live_not_promoted=True,
         true_note_annotations_not_read_in_S496=True,
         known_unannotated_guitar_events_not_negatives=True,
         no_promotion=True,
         by_source=summary,output_files=verified_files,
         source_shas=dict(S491=sha(s491report),
              original_guitar_independent_notes=sha(s487),
              original_piano_independent_notes=sha(s485)))
    (out/"report.json").write_text(json.dumps(doc,indent=2)+"\n")
def selftest():
    groups=[dict(onset=.12,pitches={52},events=[
        dict(predicted_onset_s=.12,predicted_offset_s=.4,pitch_MIDI=52,
             pretrained_note_score=.98,WAV_filename_SHA256="a")])]
    assert len(assign(np.array([.10,.12]),np.array([0,2]),groups))==1
    def m(t,p,k):return dict(WAV_filename_SHA256="a",predicted_onset_s=t,
        predicted_offset_s=t+.3,pitch_MIDI=p,pretrained_note_score=k)
    notes=[m(.12,52,.95),m(.129,55,.97)]
    ids=np.asarray(["a:100","a:110"])
    member=np.asarray(["a","a"]);starts=np.array([int(.10*SAMPLE_RATE),
                                               int(.12*SAMPLE_RATE)])
    base=np.array([0,2],np.int8)
    z=route(ids,member,starts,base,notes,"synthetic")
    assert len(z["graph"])==1 and z["guarded_sourceK"].tolist()==[0,2]
    assert z["unique_assigned_note_birth_groups"]==1
    # No positive neighbor: one acoustic birth can rescue original K0 once.
    z2=route(np.array(["a:100"]),np.array(["a"]),np.array([int(.1*SAMPLE_RATE)]),
        np.array([0],np.int8),[m(.10,52,.98)],"synthetic")
    assert z2["guarded_sourceK"].tolist()==[1]
    # K_active includes old sustained notes; K_new only counts a new birth.
    z3=route(np.array(["a:300"]),np.array(["a"]),np.array([int(.30*SAMPLE_RATE)]),
        np.array([0],np.int8),[m(.10,52,.98),m(.30,55,.96)],"synthetic")
    assert z3["joint_Knew_shadow"].tolist()==[1]
    assert z3["joint_Kactive_shadow"].tolist()==[2]
    print("S496 SELFTEST PASS joint graph unique BP owner, protected K, birth/active distinct")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for key in ("s491","s491report","s487","s487report","s485","s485report","out"):
        ap.add_argument("--"+key,type=Path)
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("s491","s491report","s487","s487report","s485","s485report","out"):
            require(getattr(a,key)is not None,"S496 missing --"+key)
        run(a.s491,a.s491report,a.s487,a.s487report,a.s485,a.s485report,a.out)
