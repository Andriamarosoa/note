"""S685: native NOTE attack proposal vetos, without manufacturing notes.

Train on the ACTUAL source-emitted native Onset candidates (not 49x all
frames). Label success as same MIDI pitch and +/-80 ms truth on TRAIN only.
At inference the model receives native past-only 75D features, and may
suppress low-quality NoteOn proposals. It never creates new pitches,
touches native 7-way K or frozen S675 release heads, or sees future.

TRAIN 5 composition groups -> train; 4 held TRAIN -> select exactly one
threshold with source K and native true-positive locks. DEV is an evaluation,
not a source of tuning or promotion. Native default always untouched.
"""
from __future__ import annotations
import argparse,json,tempfile,zipfile,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    ROOT,MODEL,raw_guitar_audio,native_posteriors,
    native_rhythm_clock,note_instances
)
from scripts.s683_causal_onset_residual_head import onset_feature_rows
from scripts.s682_onset_four_lock_guard import (
    native_metrics,aggregate,score_release,full_aggregate,safe_full
)

GATES=(.08,.12,.18,.25,.32,.42,.53,.64)


def candidate_rows(feat,post,k,events,ends):
    on=[e for e in events if e["type"]=="note_on"]
    if not on:return np.empty((0,75),np.float32)
    frames=np.asarray([
        min(len(ends)-1,int(np.searchsorted(ends,int(e["decision_sample"]),"left")))
        for e in on],dtype=np.int64)
    pitches=np.asarray([int(e["pitch"])-40 for e in on],np.int64)
    return onset_feature_rows(feat,post,k,frames,pitches)


def filter_native_events(events,accepted):
    """Conservative emission gate: does NOT change native selector state.

    Thus no new attacks are invented, even after a veto. All NoteOffs
    corresponding to suppressed attacks are also suppressed. Kept event
    timelines and native model confidences remain byte-identical.
    """
    out=[];alive={}
    ons=0
    for e in events:
        pitch=int(e["pitch"])
        if e["type"]=="note_on":
            allow=bool(accepted[ons]);ons+=1
            alive[pitch]=allow
            if allow:out.append(e)
        elif pitch in alive:
            if alive[pitch]:out.append(e)
            # A restrike's NoteOff closes prior NoteOn before its own
            # NoteOn at the same time, handled in source order.
            del alive[pitch]
    if ons!=len(accepted):raise ValueError("Native Onset candidate count changed")
    return out


def record(spec,root,archive,heads,selection):
    audio,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,feat=feature_track(audio,block_size=4096)
    p,k=native_posteriors(feat,heads)
    events=decode(ends,p,k,selection,len(audio))
    score,pairs=event_score(events,truth,tolerance=.080)
    X=candidate_rows(feat,p,k,events,ends)
    label=np.zeros(len(X),np.int8)
    for i,_ in pairs:label[i]=1
    if len(X)!=score["TP"]+score["FP"]:raise AssertionError("Native Onset count changed")
    return dict(name=spec["name"],composition=spec["composition"],
                split=spec["split"],truth=truth,ends=ends,feats=feat,
                p=p,k=k,events=events,X=X,y=label,sha=sha)


def train(examples):
    X=np.vstack([x["X"] for x in examples])
    y=np.concatenate([x["y"] for x in examples])
    if y.sum()<900 or (1-y).sum()<300:raise AssertionError("Insufficient native candidate labels")
    model=HistGradientBoostingClassifier(
        learning_rate=.05,max_iter=130,max_leaf_nodes=13,
        min_samples_leaf=28,l2_regularization=17,
        early_stopping=False,random_state=685)
    model.fit(X,y)
    return model,dict(rows=int(len(y)),TP_labels=int(y.sum()),
                      FP_labels=int((1-y).sum()),
                      features=int(X.shape[1]))


def evaluate_native(item,prob):
    results={}
    for gate in (0.,)+GATES:
        name="native" if gate==0 else f"veto_{gate:.2f}"
        events=(item["events"] if gate==0 else
                filter_native_events(item["events"],prob>=gate))
        results[name]=native_metrics(events,item["truth"],item["ends"])
    return results


def guarded_choice(rows):
    options=list(rows[0]["stats"])
    sums={p:aggregate([row["stats"] for row in rows],p) for p in options}
    base=sums["native"]
    composers=sorted({r["composition"] for r in rows})
    proposals=[]
    for name in options[1:]:
        value=sums[name]
        good=sum(aggregate([r["stats"] for r in rows
                             if r["composition"]==comp],name)["onset_F1"]>=
                 aggregate([r["stats"] for r in rows
                             if r["composition"]==comp],"native")["onset_F1"]
                 for comp in composers)
        # No loss of a genuine onset or decoded Exact-K is tolerated.
        if (value["tp"]>=base["tp"] and value["K_frames_correct"]>=base["K_frames_correct"]
            and value["onset_F1"]>base["onset_F1"]+.0005 and good>=2):
            proposals.append(name)
    selected=max(proposals,key=lambda n:sums[n]["onset_F1"]) if proposals else "native"
    return selected,sums,proposals


def offset_score(item,prob,gate,release_head):
    eventset=(item["events"] if gate==0 else
              filter_native_events(item["events"],prob>=gate))
    onset,pairs=event_score(eventset,item["truth"],tolerance=.080)
    scene=dict(truth=item["truth"],ends=item["ends"],
               feats=item["feats"],post=item["p"],k=item["k"],
               events=eventset,onset=onset,pairs=pairs,
               clock=native_rhythm_clock(eventset,item["ends"]),
               notes=note_instances(eventset,item["ends"]))
    return score_release(scene,release_head)


def run(out,checkpoint):
    if out.exists():raise FileExistsError("Immutable S685 report exists")
    start=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted({s["composition"] for s in train})
    assert len(comps)==9 and len(train)==36 and len(dev)==12
    calibrated=set(comps[:5])
    native=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selector=Selection(**native.metadata["selection"])
    from causal_note.s672_live_offset import S672TrialNoteOff
    release=S672TrialNoteOff.load(checkpoint,native)
    report=dict(schema="S685_NATIVE_ONSET_EVENT_QUALITY_VETO",
                causal_past_only=True,no_new_or_backdated_notes=True,
                K_head_unchanged=True,pitch_head_unchanged=True,
                S675_release_head_unchanged=True,
                original_LIVE_source_unchanged=True,
                TEST_unopened=True,DEV_previously_exposed=True,
                train_compositions=sorted(calibrated),
                train_held_compositions=sorted(set(comps)-calibrated))
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"data").mkdir()
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/name,src["size"],src["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio:
            examples=[]
            for i,s in enumerate(train,1):
                if s["composition"] not in calibrated:continue
                row=record(s,root,audio,native,selector)
                examples.append(dict(X=row["X"],y=row["y"]))
                del row
                print("S685 TRAIN",len(examples),"/20",s["name"],flush=True)
            model,proof=train_head=globals()["train"](examples)
            report["training"]=proof
            del examples
            held_records=[]
            for i,s in enumerate([x for x in train if x["composition"] not in calibrated],1):
                row=record(s,root,audio,native,selector)
                probability=model.predict_proba(row["X"])[:,1]
                stat=evaluate_native(row,probability)
                held_records.append(dict(composition=s["composition"],name=s["name"],stats=stat))
                del row
                print("S685 HELD",i,"/16",s["name"],flush=True)
            selected,held_metrics,passed=guarded_choice(held_records)
            report["TRAIN_held_selection"]=selected
            report["TRAIN_held_onset_K"]=held_metrics
            report["TRAIN_held_passing_gates"]=passed
            print("S685 TRAIN-FROZEN",selected,json.dumps(held_metrics[selected]),flush=True)
            gate=0. if selected=="native" else float(selected.split("_")[1])
            held_offset=[]
            for i,s in enumerate([x for x in train if x["composition"] not in calibrated],1):
                row=record(s,root,audio,native,selector)
                prob=model.predict_proba(row["X"])[:,1]
                held_offset.append(dict(
                    native=offset_score(row,prob,0.,release.model),
                    candidate=offset_score(row,prob,gate,release.model)))
                del row
                print("S685 HELD S675",i,"/16",s["name"],flush=True)
            held_full={key:full_aggregate(held_offset,key) for key in ("native","candidate")}
            release_pass=selected!="native" and safe_full(held_full["native"],held_full["candidate"])
            report["TRAIN_held_S675"]=held_full
            report["TRAIN_held_four_locks_passed"]=release_pass
            rows=[];notes=[]
            for i,s in enumerate(dev,1):
                row=record(s,root,audio,native,selector)
                prob=model.predict_proba(row["X"])[:,1]
                baseline=filter_native_events(row["events"],np.ones(len(prob),bool))
                candidate=filter_native_events(row["events"],prob>=gate)
                stats=dict(
                    native=native_metrics(baseline,row["truth"],row["ends"]),
                    candidate=native_metrics(candidate,row["truth"],row["ends"]))
                full=dict(
                    native=offset_score(row,prob,0.,release.model),
                    candidate=offset_score(row,prob,gate,release.model))
                rows.append(dict(name=s["name"],composition=s["composition"],stats=stats))
                notes.append(full)
                del row
                print("S685 DEV",i,"/12",s["name"],flush=True)
    counts={name:aggregate([r["stats"] for r in rows],name)
            for name in ("native","candidate")}
    full={name:full_aggregate(notes,name) for name in ("native","candidate")}
    report["DEV_native_K_Onset"]=counts
    report["DEV_S675_offset_full_note"]=full
    report["DEV_by_recording"]=rows
    report["strict_no_regression_satisfied"]=bool(
        release_pass and
        counts["candidate"]["onset_F1"]>counts["native"]["onset_F1"] and
        counts["candidate"]["K_frames_correct"]>=counts["native"]["K_frames_correct"] and
        safe_full(full["native"],full["candidate"]))
    report["promoted"]=False
    report["elapsed_s"]=round(time.perf_counter()-start,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S685 FINAL",json.dumps(dict(
        selected=selected,
        original_onset=counts["native"],candidate_onset=counts["candidate"],
        original_offset=full["native"],candidate_offset=full["candidate"],
        four_locks_passed=report["strict_no_regression_satisfied"])),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.out,args.checkpoint)
