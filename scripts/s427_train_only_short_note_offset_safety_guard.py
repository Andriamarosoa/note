"""S427 — train-only short-note safety guard for frozen S425 NoteOff.

S426: S425 gains +231 on 100..499ms but loses -46 true <100ms.
This follow-up tests a PREDECLARED tiny rule family operating on
*PREDICTED* S420/S422/S425 durations only, never true duration at infer.
Select the policy on OTHER song compositions; abstain by default.
Zero modifications to MIDI pitch, NoteOn or original S80 per-row Exact-K.
Development-only: upstream meta models not fully nested.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s425_songheld_select_S422_S420_offsets_no_K_regression import (
    read,labels_for_other_songs_only,need,SR,N_EVENTS,N_NOTES,N_GT)
from scripts.s414_compare_official_yourmt3_full_notes_vs_S408_S412_guitarset import score
from scripts.s402_k_input_first_supervised_midi_pitch_network import composition_key,composition_fold
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha

CUTS=(70,100,140,190)
POLICIES=("S422_if_425_short","S422_if_422_short",
          "S420_if_425_short","S420_if_420_short",
          "S422_if_425_and_422_short","S420_if_both_425_420_short")
OPTIONS=[(policy,cut) for policy in POLICIES for cut in CUTS]
def offset_policy(base,survival,chosen,on,policy,cut):
    d0=(base-on)/SR*1000
    d1=(survival-on)/SR*1000
    dc=(chosen-on)/SR*1000
    if policy=="S422_if_425_short":
        apply=(dc<cut)&(chosen!=survival);other=survival
    elif policy=="S422_if_422_short":
        apply=(d1<cut)&(chosen!=survival);other=survival
    elif policy=="S420_if_425_short":
        apply=(dc<cut)&(chosen!=base);other=base
    elif policy=="S420_if_420_short":
        apply=(d0<cut)&(chosen!=base);other=base
    elif policy=="S422_if_425_and_422_short":
        apply=(np.maximum(dc,d1)<cut)&(chosen!=survival);other=survival
    elif policy=="S420_if_both_425_420_short":
        apply=(np.maximum(dc,d0)<cut)&(chosen!=base);other=base
    else:raise ValueError("invalid preregistered S427 rule")
    return np.where(apply,other,chosen),int(apply.sum())
def score_diffs(target,old,proposed,mask):
    eligible=mask&(target>0)
    original=int((np.abs(old[eligible]-target[eligible])<=2205).sum())
    tested=int((np.abs(proposed[eligible]-target[eligible])<=2205).sum())
    return tested-original
def run(s425,s80,annotations,out):
    need(not out.exists(),"S427 immutable source report exists")
    p=read(s425);r=read(s80)
    expected=("source_row_id","MIDI","track_member","onset_samples",
      "original_S80_row_id","original_S80_K","S420_offset_samples",
      "S422_offset_samples","S425_offset_samples")
    need(set(expected).issubset(p),"S427 frozen MIDI source files absent")
    ids=p["original_S80_row_id"].astype(str)
    k=p["original_S80_K"].astype(np.int8)
    need(len(ids)==N_EVENTS and len(set(ids))==N_EVENTS
       and np.array_equal(ids,r["row_id"].astype(str))
       and np.array_equal(k,r["s80_K"].astype(np.int8))
       and int(k.sum())==N_NOTES,
       "S427 original S80 exact K reference altered")
    on=p["onset_samples"].astype(np.int64)
    base=p["S420_offset_samples"].astype(np.int64)
    survival=p["S422_offset_samples"].astype(np.int64)
    chosen=p["S425_offset_samples"].astype(np.int64)
    track=p["track_member"].astype(str)
    pitch=p["MIDI"].astype(np.int16)
    source_ids=p["source_row_id"].astype(str)
    counters=collections.Counter(source_ids)
    need(len(source_ids)==len(on)==N_NOTES
      and all(counters[row]==int(value) for row,value in zip(ids,k))
      and np.all(chosen>on),"S427 source MIDI note count or NoteOff drift")
    comp=np.array([composition_key(m) for m in track])
    fold=np.array([composition_fold(c) for c in comp])
    # FROZEN candidate outputs before any annotation access.
    target,gt=labels_for_other_songs_only(
        dict(track_member=track,onset_samples=on,MIDI=pitch),annotations)
    need(sum(np.abs(chosen[target>0]-target[target>0])<=2205)==16543,
         "S427 frozen S425 result drift")
    final=chosen.copy()
    audit=[]
    for f in range(4):
        train=(fold!=f)&(target>0)
        held=fold==f
        need(not(set(comp[train])&set(comp[held])),
             "S427 train-test composition overlap")
        prior_score=score_diffs(target,chosen,chosen,train)
        need(prior_score==0,"S427 broken source scoring")
        rankings=[]
        for name,cut in OPTIONS:
            cand,count=offset_policy(base,survival,chosen,on,name,cut)
            delta=score_diffs(target,chosen,cand,train)
            rankings.append(dict(policy=name,threshold_ms=cut,
                training_gain=delta,changed_predictions=count))
        best=max(rankings,key=lambda x:(x["training_gain"],
                        -x["threshold_ms"],x["policy"]))
        # Conservative abstain if train improvement is weaker than
        # approximately 15 net correct notes.
        enabled=best["training_gain"]>=15
        affected=0
        if enabled:
            cand,_=offset_policy(base[held],survival[held],chosen[held],
                      on[held],best["policy"],best["threshold_ms"])
            affected=int(np.sum(cand!=chosen[held]))
            final[held]=cand
        audit.append(dict(held_fold=f,train_notes=int(train.sum()),
          held_notes=int(held.sum()),train_compositions=len(set(comp[train])),
          held_compositions=len(set(comp[held])),composition_overlap=0,
          preregistered_policies=len(OPTIONS),
          top_training_policies=sorted(rankings,key=lambda x:-x["training_gain"])[:4],
          selected_policy=best["policy"],selected_cut_ms=best["threshold_ms"],
          selected_training_net=best["training_gain"],
          enabled=bool(enabled),held_changed_notes=affected))
        print("S427 FOLD",json.dumps(audit[-1]),flush=True)
    need(np.all(final>on),"S427 invalid MIDI duration")
    out.mkdir(parents=True)
    exported=out/"S427_NATIVE_K_LOCKED_full_MIDI_notes.jsonl"
    bytrack=collections.defaultdict(list)
    for t,row,pp,oo,ff in zip(track,source_ids,pitch,on,final):
        bytrack[t].append(dict(source_row_id=str(row),pitch=int(pp),
                  onset_sample=int(oo),offset_sample=int(ff),
                  source="S427_native_S425_short_note_guard"))
    with exported.open("w") as file:
        for t in sorted(bytrack):
            file.write(json.dumps(dict(member=t,notes=bytrack[t]))+"\n")
    actual=collections.Counter()
    for line in exported.read_text().splitlines():
        for event in json.loads(line)["notes"]:
            need(event["offset_sample"]>event["onset_sample"],
                 "S427 invalid exported note time")
            actual[event["source_row_id"]]+=1
    need(sum(actual.values())==N_NOTES
      and all(actual[row]==int(kk) for row,kk in zip(ids,k)),
      "S427 CRITICAL actual emitted MIDI file changed S80 exact-K")
    def compute(offs):
        z=collections.defaultdict(list)
        for t,pp,oo,ff in zip(track,pitch,on,offs):
            z[t].append((int(oo),int(ff),int(pp),True))
        scored,_=score(z,gt)
        return {key:value for key,value in scored.items() if key!="by_track"}
    reference=compute(chosen)
    challenger=compute(final)
    need(reference["correct_offsets_conditional_on_matched_pitch_onset"]==16543,
      "S427 original S425 report does not independently reproduce")
    npz=out/"S427_frozen_all_native_MIDI_notes.npz"
    np.savez_compressed(npz,source_row_id=source_ids,
       original_S80_row_id=ids,original_S80_K=k,
       MIDI=pitch,track_member=track,onset_samples=on,
       S425_offset_samples=chosen,S427_offset_samples=final,composition_fold=fold)
    detail={}
    for f in range(4):
        selected=(fold==f)&(target>0)
        detail[str(f)]=dict(matched=int(selected.sum()),
          S425_correct=int((np.abs(chosen[selected]-target[selected])<=2205).sum()),
          S427_correct=int((np.abs(final[selected]-target[selected])<=2205).sum()))
    result=dict(status="S427_TRAIN_ONLY_SHORT_NOTE_GUARD_NATIVE_S80_K_LOCKED",
      protected_S80_exactK_count_N=N_EVENTS,
      native_NoteOn_count=N_NOTES,
      exactK_regressions=0,
      independently_verified_exported_midi_K_per_row=True,
      unchanged_MIDI_pitch_and_onset=True,
      no_YourMT3_at_training_or_inference=True,
      best_previous_S425_F1=reference["full_note_F1_50ms"],
      new_S427_F1=challenger["full_note_F1_50ms"],
      S425_correct_complete_notes=16543,
      S427_correct_complete_notes=challenger["correct_offsets_conditional_on_matched_pitch_onset"],
      net_gain=challenger["correct_offsets_conditional_on_matched_pitch_onset"]-16543,
      scores=dict(S425=reference,S427=challenger),
      by_held_fold=detail,
      train_only_selection=audit,
      possible_upstream_meta_training_overlap=True,
      original_S80_GuitarSet_train_overlap=True,
      no_official_promotion=True,
      Live_unchanged=True,
      source_hashes=dict(S425=sha(s425),original_S80=sha(s80),
           JAMS=sha(annotations),MIDI_export=sha(exported),NPZ=sha(npz)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S427 RESULT",json.dumps(dict(
      original=16543,new=result["S427_correct_complete_notes"],
      net=result["net_gain"],F1=challenger["full_note_F1_50ms"])),flush=True)
    print("S427 PASS: 59309 original K unchanged and independently replayed MIDI exports",flush=True)

def test():
    old=np.array([10000,10000,10000],np.int64)
    new=np.array([15000,12000,8000],np.int64)
    chosen=np.array([16000,10000,9000],np.int64)
    on=np.array([0,0,0],np.int64)
    selected,n=offset_policy(old,new,chosen,on,"S422_if_425_short",250)
    need(n==3 and np.array_equal(selected,new),
         "S427 preregistered short-note guard incorrect")
    print("S427 SELFTEST PASS: short-note guard exact selection")
if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for x in ("s425","s80","annotations","out"):
        q.add_argument("--"+x,type=Path)
    v=q.parse_args()
    if v.self_test:test()
    else:
        for x in ("s425","s80","annotations","out"):
            need(getattr(v,x)is not None,"missing --"+x)
        run(v.s425,v.s80,v.annotations,v.out)
