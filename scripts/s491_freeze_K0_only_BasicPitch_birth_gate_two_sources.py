"""S491: label-free SHADOW fusion; only original K0 can become named-note evidence.

Two genuinely distinct previous model stacks:
  - IDMT original S80 native 506: source K from S297 (true S80 final).
  - MAESTRO original V88 native 575: source K from S470 (NOT full S80).
  - Multi-instrument Basic Pitch note evidence from already frozen S487/S485.

NEVER read XML/MIDI, their scored S488/S486 ledgers, or labels here.
For each corpus: pair original native EVENT TIMES with pretrained Basic
Pitch NOTE ATTACK GROUPS (20ms grouping, one-to-one matching <=80ms).
Only when original K is 0 and an independently detected pitch onset group
matches this same native event, propose K=number of new notes (K_birth),
or K=number of active notes at native +30ms (K_active). Export both
predefined policies, not tuned against existing test truth.
Source K>=1 and 0..6 original classes immutable; nothing deployed.
This cannot guarantee no false positives on UNLABELLED IDMT candidates.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
def need(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def group_birth(events):
    bysource=collections.defaultdict(list)
    for ev in events:
        bysource[str(ev["WAV_filename_SHA256"])].append(ev)
    groups={}
    for key,rows in bysource.items():
        gs=[]
        for ev in sorted(rows,key=lambda e:(float(e["predicted_onset_s"]),int(e["pitch_MIDI"]))):
            t=float(ev["predicted_onset_s"])
            if not gs or t-gs[-1]["time"]>.020:
                gs.append(dict(time=t,pitches=set(),events=[]))
            gs[-1]["pitches"].add(int(ev["pitch_MIDI"]))
            gs[-1]["events"].append(ev)
        groups[key]=gs
    return groups
def one_to_one(time_native,time_note,seconds=.080):
    if not len(time_native) or not len(time_note):return []
    dist=np.abs(time_native[:,None]-time_note[None,:])
    # Always prefer valid bipartite matches to any dummy at cost 1.
    costs=np.concatenate([
        np.where(dist<=seconds,dist/(seconds+1e-9)*.5,1e6),
        np.ones((len(time_native),len(time_native)),float)
    ],axis=1)
    i,j=linear_sum_assignment(costs)
    result=[(int(a),int(b),float(dist[a,b]))
            for a,b in zip(i,j) if b<len(time_note) and dist[a,b]<=seconds]
    need(len(set(a for a,b,d in result))==len(result) and
         len(set(b for a,b,d in result))==len(result),
         "S491 a note birth cannot justify multiple original event candidates")
    return result
def apply(ids,member,samples,source,notes,name):
    need(len(ids)==len(source)==len(member)==len(samples),
         "S491 source event count must exactly match frozen prediction IDs")
    bymember=group_birth(notes)
    raw_by_source=collections.defaultdict(list)
    for ev in notes:raw_by_source[str(ev["WAV_filename_SHA256"])].append(ev)
    original=source.astype(np.int8)
    birth=original.copy();active=original.copy()
    matched=np.zeros(len(original),bool)
    recent=np.zeros(len(original),bool)
    groupsmatched=0
    details=[]
    for key in sorted(set(member)):
        chosen=np.flatnonzero(member==key)
        groups=bymember.get(key,[])
        if not groups:continue
        local_times=samples[chosen].astype(np.float64)/44100.
        note_times=np.asarray([x["time"] for x in groups],float)
        pairs=one_to_one(local_times,note_times)
        groupsmatched+=len(pairs)
        for i,j,dist in pairs:
            globalidx=int(chosen[i])
            g=groups[j]
            matched[globalidx]=True
            if source[globalidx]!=0:
                continue
            pitches=set(g["pitches"])
            born=len(pitches)
            absolute_time=float(local_times[i])+.030
            # Source active-note inference always anchored at the NATIVE
            # event time; never at the annotation time (which is unavailable).
            alive={int(ev["pitch_MIDI"]) for ev in raw_by_source[key]
                   if (float(ev["predicted_onset_s"])<=absolute_time
                       <float(ev["predicted_offset_s"]))}
            if 1<=born<=6:
                birth[globalidx]=born
                recent[globalidx]=True
            if 1<=len(alive)<=6:
                active[globalidx]=len(alive)
                recent[globalidx]=True
            details.append(dict(row_id=str(ids[globalidx]),
                source=name,source_K0=True,
                paired_external_note_birth_time_s=float(g["time"]),
                event_time_s=float(local_times[i]),
                distance_ms=round(1000*dist,3),
                proposal_new_birth_K=int(birth[globalidx]),
                proposal_active_30ms_K=int(active[globalidx]),
                notes_new_birth_MIDI=sorted(pitches),
                notes_active_30ms_MIDI=sorted(alive)))
    need(np.array_equal(birth[original!=0],original[original!=0])
         and np.array_equal(active[original!=0],original[original!=0])
         and np.all((birth>=0)&(birth<=6)) and
         np.all((active>=0)&(active<=6)),
         "S491 protected source original S80/V88 K nonzero mutated")
    return birth,active,matched,recent,details,groupsmatched


def run(s277,s297,s487,s487report,s470,s485,s485report,out):
    need(not out.exists(),"S491 immutable pretruth shadow routing already produced")
    gr=json.loads(s487report.read_text())
    pr=json.loads(s485report.read_text())
    need(gr["status"]=="S487_FIVE_GUITAR_IDMT_REAL_WAV_PRETRAINED_BASIC_PITCH_FROZEN_NOXML"
         and gr["all_output_note_events_sha256"]==sha(s487)
         and gr["original_XML_NOT_READ_before_note_freeze"]
         and pr["status"]=="S485_PRETRAINED_BASIC_PITCH_NOTE_EVENTS_ON_THIRD_SOURCE_MAESTRO_PREMIDI"
         and pr["frozen_notes_SHA256"]==sha(s485)
         and pr["original_MAESTRO_MIDI_ANNOTATIONS_OPENED"] is False,
         "S491 cannot train or route if source guitar/piano pretrained note freezes changed")
    with np.load(s277,allow_pickle=False) as a:
        ids_g=a["row_id"].astype(str)
        members_g=a["member"].astype(str)
        times_g=a["start_samples"].astype(np.int64)
    with np.load(s297,allow_pickle=False) as a:
        source_g=a["s80_K"].astype(np.int8)
        need(np.array_equal(ids_g,a["row_id"].astype(str)),
             "S491 genuine source original S297 S80 IDs differ from V88 candidates")
    with np.load(s470,allow_pickle=False) as a:
        ids_p=a["row_id"].astype(str)
        members_p=a["member"].astype(str)
        times_p=a["start_samples"].astype(np.int64)
        source_p=a["original_V88_K"].astype(np.int8)
    need(len(ids_g)==506 and len(ids_p)==575 and
         np.bincount(source_g,minlength=7).tolist()==[456,26,11,7,6,0,0] and
         np.bincount(source_p,minlength=7).tolist()==[473,69,25,8,0,0,0] and
         len(set(ids_g))==506 and len(set(ids_p))==575,
         "S491 original frozen historical S80 guitar/V88 piano must remain untouched")
    scores={}
    collected=[]
    array={}
    for key,ids,members,starts,ref,path in [
        ("guitar_S80",ids_g,members_g,times_g,source_g,s487),
        ("piano_V88",ids_p,members_p,times_p,source_p,s485)]:
        notes=[json.loads(line) for line in path.read_text().splitlines()]
        proposedbirth,proposedactive,matched,recent,info,npairs=apply(
            ids,members,starts,ref,notes,key)
        array[key]=dict(row_id=ids,member=members,start_samples=starts,
             source_K=ref,proposed_only_K0_birth=proposedbirth,
             proposed_only_K0_active30=proposedactive,
             independent_BP_note_onset_group_matched=matched)
        scores[key]=dict(source_native_event_count=len(ids),
             source_K0_count=int((ref==0).sum()),
             external_onset_groups_paired_to_native=int(npairs),
             original_K0_became_K1_through_birth=int(np.sum((ref==0)&(proposedbirth==1))),
             original_K0_became_poly_through_birth=int(np.sum((ref==0)&(proposedbirth>=2))),
             original_K0_became_K1_through_active=int(np.sum((ref==0)&(proposedactive==1))),
             original_K0_became_poly_through_active=int(np.sum((ref==0)&(proposedactive>=2))),
             source_protected_K1toK6_changed=0,
             model_or_threshold_fitting=False)
        collected.extend(info)
        print("S491 FROZEN",json.dumps(dict(corpus=key,**scores[key])),flush=True)
    out.mkdir(parents=True)
    files={}
    for key,bank in array.items():
        filename=out/f"S491_{key}_SOURCE_K0_ONLY_FROZEN.npz"
        np.savez_compressed(filename,**bank)
        files[key]=dict(file=filename.name,sha256=sha(filename))
    logfile=out/"S491_unlabelled_BasicPitch_K0_candidate_support.jsonl"
    logfile.write_text("".join(json.dumps(x)+"\n" for x in collected))
    rep=dict(status="S491_TWO_SOURCE_K0_ONLY_ONE_TO_ONE_EXTERNAL_PITCH_BIRTH_FROZEN_NO_TRUTH",
        original_IDMT_source_is_final_S80=True,
        original_MAESTRO_source_is_V88_NOT_S80=True,
        original_note_times_not_annotation_times=True,
        independent_external_note_birth_group_max_distance_ms=80,
        original_native_events_and_timing_unchanged=True,
        only_original_source_K0_may_change=True,
        exact_notes_and_Live_not_deployed=True,
        no_midi_or_xml_label_read_during_freeze=True,
        previously_exposed_external_sources_cannot_prove_blindness=True,
        no_source_specific_supervised_thresholds=True,
        two_policies_predeclared_no_score_based_selection=True,
        no_model_training=True,no_promotion=True,
        by_corpus=scores,
        source_output_SHA256=files,
        acoustic_candidate_ledger_SHA256=sha(logfile),
        upstream_SHA256=dict(genuine_IDMT_V88=sha(s277),
             original_S80_final=sha(s297),external_IDMT_notes=sha(s487),
             original_V88_piano=sha(s470),external_piano_notes=sha(s485)))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")

def selftest():
    n=np.array([.1,.2,.4]);p=np.array([.11,.21])
    assert len(one_to_one(n,p))==2
    assert len(set(j for _,j,_ in one_to_one(n,p)))==2
    assert len(one_to_one(np.array([.100,.101]),np.array([.1])))==1
    print("S491 SELFTEST PASS one independent onset birth group can only justify one native event")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s277","s297","s487","s487report","s470","s485","s485report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s277","s297","s487","s487report","s470","s485","s485report","out"):
            need(getattr(a,k)is not None,"S491 missing --"+k)
        run(a.s277,a.s297,a.s487,a.s487report,a.s470,a.s485,a.s485report,a.out)
