"""S536 repeated-note re-strike aware MIDI pitch candidate loops (no training).

Retain every original Basic Pitch note birth. During a possible duplicate
within 80/100ms, look for genuine renewed per-pitch WAV onset burst AND
novelty relative to the previously EMITTED pitch. Allow genuine strong
re-articulation; suppress weak repeated resonances. Independent optional
weak-harmonic veto only for newly added pitches. Fixed ratio hypotheses,
no MIDI truth as inference features. Score full original 330 onsets.

Historical MAESTRO test compositions already exposed; exploratory only,
never new independent promotion score. Original S80 Exact-K and Live safe.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
from scripts.s530_full_maestro_stream_noteon_pitch_event_F1_audit import (
  load_sources,score_event_notes,score_event_birth,proposed_notes,need,sha,digest)
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import (
  fetch_midi,midi_events,onset_groups)
from scripts.s476_compare_MAESTRO_newnote_K_vs_sounding_K_same_V88 import new_notes

def emit(rows,policy):
    groups=[];state={};sc=collections.Counter()
    for r,p in sorted(rows,key=lambda z:float(z[0]["original_V88_candidate_start_s"])):
        t=float(r["original_V88_candidate_start_s"])
        arr=np.asarray(r["per_midi_features_49x11"],float)
        bp=proposed_notes(r,p,"BP_birth")
        if bp:
            groups.append(dict(time=t,notes=sorted(bp)))
            for pitch in bp:
                j=pitch-40
                state[pitch]=(t,float(arr[j,3]),float(arr[j,5]))
            continue
        conf=np.asarray(p["raw_only_49_pitch_prob"],float)
        selected=np.flatnonzero(conf>=.7)
        keep=set()
        for j in sorted(selected):
            pitch=int(j+40)
            existing=state.get(pitch)
            if existing is not None and 0 <= 1000*(t-existing[0]) <= policy["dedup_ms"]:
                prev_time,prev_burst,prev_novelty=existing
                good=False
                if policy["rearticulation_ratio"]>0:
                    good=(float(arr[j,3])>=policy["rearticulation_ratio"]*max(prev_burst,.05)
                        and float(arr[j,5])>=policy["rearticulation_ratio"]*max(prev_novelty,.05)
                        and float(arr[j,10])>=.5)
                if not good:
                    sc["duplicate_candidates_dropped"]+=1
                    continue
                sc["rearticulation_candidates_released_by_audio_evidence"]+=1
            # Only newly emitted notes can be harmonic duplicates.
            harm=False
            if policy["harmonic_ratio"]:
                for interval in (12,19,24):
                    k=j-interval
                    if k<0 or k not in keep:continue
                    if (arr[j,3] < policy["harmonic_ratio"]*max(.001,arr[k,3])
                        and arr[j,5] < policy["harmonic_ratio"]*max(.001,arr[k,5])):
                        harm=True;break
            if harm:sc["weak_harmonics_dropped"]+=1
            else:keep.add(int(j))
        if keep:
            names=sorted(40+j for j in keep)
            groups.append(dict(time=t,notes=names))
            for j in keep:
                state[40+j]=(t,float(arr[j,3]),float(arr[j,5]))
    return groups,dict(sc)

def run(a):
    need(not a.out.exists(),"S536 fixed policy experiment immutable")
    meta,s471,s527,s528,s529,source,proof,raw,learned=load_sources(a)
    bysource=collections.defaultdict(list)
    for x,p in zip(raw,learned):
        bysource[x["source_recording_sha"]].append((x,p))
    truth={}
    for ident,item in sorted(source.items()):
        blob=fetch_midi(meta["audio_mirror_directory"],
              item["MIDI_filename_path_ONLY_not_opened"])
        need(digest(blob)==proof[ident]["MIDI_bytes_SHA256"],
             "S536 same 330 genuine MIDI onsets not preserved")
        midi=midi_events(blob)
        offset=float(item["audio_excerpt_start_seconds"])
        true=onset_groups(midi,offset,float(item["audio_excerpt_duration_seconds"]))
        truth[ident]=[dict(time=float(z["onset"]),
                         notes=new_notes(midi,offset,float(z["onset"]))) for z in true]
        need(len(truth[ident])==proof[ident]["MIDI_attack_groups_in_excerpt"],
             "S536 full MIDI truth changed")
    specs=[
      ("strict80",80,0.,0.),("strict100",100,0.,0.),
      ("strict80_harm80",80,0.,.8),("strict100_harm80",100,0.,.8),
      ("reinstate80_ratio100",80,1.,0.),
      ("reinstate80_ratio150",80,1.5,0.),
      ("reinstate100_ratio100",100,1.,0.),
      ("reinstate100_ratio150",100,1.5,0.),
      ("reinstate80_ratio100_harm80",80,1.,.8),
      ("reinstate100_ratio100_harm80",100,1.,.8)]
    res={}
    for n,window,ratio,harm in specs:
        policy=dict(name=n,dedup_ms=window,
                    rearticulation_ratio=ratio,harmonic_ratio=harm)
        total=collections.Counter();records={}
        for ident in sorted(truth):
            prediction,stats=emit(bysource[ident],policy)
            target=truth[ident]
            note=score_event_notes(prediction,target)
            onset=score_event_birth(prediction,target)
            total.update(TP=note["TP"],FP=note["FP"],FN=note["FN"],
                  original_true_groups=onset["note_groups_true"],
                  emitted_groups=onset["note_groups_predicted"],
                  chord_exact=onset["matched_exact_full_note_chords"],
                  exactK=onset["matched_exactK"],**stats)
            records[ident]=dict(note=note,onset=onset,stats=stats)
        result=dict(total)
        result["F1_NoteOn"]=2*result["TP"]/max(1,2*result["TP"]+result["FP"]+result["FN"])
        result["precision_NoteOn"]=result["TP"]/max(1,result["TP"]+result["FP"])
        result["recall_NoteOn"]=result["TP"]/max(1,result["TP"]+result["FN"])
        need(result["original_true_groups"]==330,"S536 must evaluate all true original notes")
        res[n]=dict(parameters=policy,summary=result,per_recording=records)
        print("S536 AUDIO-BASED RE-STRIKE PROTECTION LOOP",
              json.dumps(dict(policy=n,metrics=result)),flush=True)
    audit=json.loads(a.s533_report.read_text())
    base=audit["tests"]["dedup80"]["summary"]
    check=res["strict80"]["summary"]
    need((check["TP"],check["FP"],check["FN"])
         ==(base["note_tp"],base["note_fp"],base["note_fn"]),
         "S536 strict80 original prior policy must independently match S533")
    a.out.mkdir(parents=True)
    report=dict(status="S536_TRUE_REARTICULATION_AWARE_NOTEON_DEDUP_HARMONIC_LOOPS",
      ten_fixed_rearticulation_condition_hypotheses=True,
      no_test_MIDI_used_to_choose_rearticulation_release=True,
      every_genuine_330_MAESTRO_original_true_noteon_groups_evaluated=True,
      original_80ms_dedup_F1_reproduced=True,
      BP_original_birth_notes_preserved=True,
      original_V88_K0_never_hard_restricts_pitch=True,
      only_old_exposed_three_MAESTRO_recordings_not_prospective=True,
      original_S80_ExactK_Live_unchanged=True,no_promotion=True,
      tests=res,sha_source=dict(S527=sha(a.audio),S528=sha(a.trained),
                                S533=sha(a.s533_report)))
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    base=np.zeros((49,11),float)
    base[12,3]=base[12,5]=1; base[12,10]=1
    one=dict(original_V88_candidate_start_s=.1,old_basicpitch_birth_pitches=[],
             per_midi_features_49x11=base.tolist())
    two=dict(one,original_V88_candidate_start_s=.15)
    p=np.zeros(49,float);p[12]=.9
    pred={"raw_only_49_pitch_prob":p.tolist()}
    policy=dict(dedup_ms=80,rearticulation_ratio=0,harmonic_ratio=0)
    a,st=emit([(one,pred),(two,pred)],policy)
    need(len(a)==1 and st["duplicate_candidates_dropped"]==1,
         "S536 repeated acoustic release cannot be unconditional")
    ratio=dict(dedup_ms=80,rearticulation_ratio=1,harmonic_ratio=0)
    b,st=emit([(one,pred),(two,pred)],ratio)
    need(len(b)==2 and st["rearticulation_candidates_released_by_audio_evidence"]==1,
         "S536 strong renewed physical note attack must be eligible as genuine retrigger")
    first=dict(one,old_basicpitch_birth_pitches=[52,64])
    c,st=emit([(first,pred),(two,pred)],policy)
    need(c[0]["notes"]==[52,64],"S536 cannot modify protected old true polyphonic BP note")
    print("S536 SELFTEST PASS new attack physical evidence allows repeats, source BP protected")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for z in ("manifest","s471-report","audio","audio-report","trained",
              "trained-report","protect-report","s533-report","out"):
        p.add_argument("--"+z,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for z in ("manifest","s471_report","audio","audio_report","trained",
                  "trained_report","protect_report","s533_report","out"):
            need(getattr(a,z)is not None,"S536 missing "+z)
        run(a)
