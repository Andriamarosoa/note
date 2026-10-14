"""S440 -- blinded-first paired score of the two frozen S439 joint note challengers.

Read true notes ONLY after requiring SHA-sealed S439 outputs and unchanged
S80 predicted K. This is DEVELOPMENT ONLY, not unseen-instrument validation.
"""
from __future__ import annotations
import argparse
import collections
import json
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import read_note_events
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s402_k_input_first_supervised_midi_pitch_network import need,composition_key
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import note_name,sha

def evaluate(base, proposals, truth):
    def note_multiset(v):
        return collections.Counter(v)
    full={}
    details=[]
    for label,rows in {"original_S437":base,**proposals}.items():
        exact={}
        true_by_k=collections.defaultdict(list)
        matched=true_total=pred_total=0
        for row in rows:
            rid=row["original_row_id"]
            expected=truth[rid]
            got=[next(p for p in range(40,89) if note_name(p)==n) for n in row["notes"]]
            hit=note_multiset(got)==note_multiset(expected)
            exact[rid]=hit
            true_by_k[len(expected)].append(hit)
            matched+=sum((note_multiset(got)&note_multiset(expected)).values())
            true_total+=len(expected);pred_total+=len(got)
        perK={str(k):{"events":len(v),"correct":int(sum(v)),
                     "accuracy":round(sum(v)/len(v),6)}
              for k,v in sorted(true_by_k.items())}
        ids_poly=[row["original_row_id"] for row in rows if len(truth[row["original_row_id"]])>=2]
        full[label]=dict(
            all_events=len(rows),whole_chords_correct=int(sum(exact.values())),
            true_K=perK,
            poly_K2plus_events=len(ids_poly),
            poly_K2plus_whole_chords_correct=int(sum(exact[r] for r in ids_poly)),
            per_note_F1=(2*matched/(true_total+pred_total)
                         if true_total+pred_total else 1.))
        if label!="original_S437":
            old=full["original_S437"]
            baseline_exact={rid:collections.Counter(
                next(p for p in range(40,89) if note_name(p)==n)
                for n in b["notes"])==collections.Counter(truth[rid])
                for rid,b in ((r["original_row_id"],r) for r in base)}
            changes=collections.defaultdict(lambda:{"corrections":0,"regressions":0,
                                                     "changes":0})
            for row,b in zip(rows,base):
                rid=row["original_row_id"]
                k=len(truth[rid])
                changed=row["notes"]!=b["notes"]
                group=changes[str(k)]
                group["changes"]+=int(changed)
                group["corrections"]+=int(hit if False else exact[rid] and not baseline_exact[rid])
                group["regressions"]+=int(baseline_exact[rid] and not exact[rid])
                if changed:
                    details.append(dict(variant=label,row_id=rid,
                       original_predicted_K=b["original_S80_K"],
                       true_K=k,baseline=b["notes"],new=row["notes"],
                       ground_truth=[note_name(p) for p in truth[rid]],
                       original_exact=bool(baseline_exact[rid]),
                       new_exact=bool(exact[rid])))
            corr=sum(x["corrections"] for x in changes.values())
            reg=sum(x["regressions"] for x in changes.values())
            full[label]["paired_vs_frozen_S437"]=dict(
                corrections=corr,regressions=reg,net=corr-reg,
                by_true_K=dict(sorted(changes.items(),key=lambda x:int(x[0]))))
    return full,details

def run(names,s437,s439_dir,source,s80,groundtruth,annotations,out):
    need(not out.exists(),"S440 immutable paired audit exists")
    parent=json.loads(s437.read_text())
    challenger=json.loads((s439_dir/"report.json").read_text())
    need(parent["status"]=="S437_FROZEN_MULTI_GUITAR_REAL_MIC_WAV_S80_SOURCE_NOTE_NAMES"
         and challenger["status"]=="S439_K_LOCKED_LABEL_BLIND_REAL_MIC_JOINT_NOTE_SET_CHALLENGERS"
         and challenger["fixed_weights_before_S440_labels"]
         and challenger["all_Original_S80_K_unchanged"]
         and challenger["no_JAMS_annotations_opened"]
         and challenger["original_frozen_names_SHA256"]==sha(names)
         and parent["output_SHA256"]==sha(names),
         "S440 prospective S437/S439 freeze prerequisite not proved")
    base=[json.loads(x) for x in names.read_text().splitlines()]
    need(len(base)==174,"S440 original event count changed")
    proposals={}
    for label in ("conservative","exploratory"):
        spec=challenger["candidates"][label]
        file=s439_dir/spec["path"]
        need(sha(file)==spec["sha256"],"S440 S439 candidate SHA changed")
        obj=[json.loads(x) for x in file.read_text().splitlines()]
        need(len(obj)==len(base),"S440 challenger sample size drift")
        for old,new in zip(base,obj):
            need(all(old[key]==new[key] for key in
                 ("original_row_id","member","start_sample","end_sample","original_S80_K"))
                 and len(new["notes"])==len(old["notes"]),
                 "S440 source row/S80 K event identity drift")
            need(new["S437_baseline_notes"]==old["notes"],
                 "S440 S437 original notes changed under challenger")
        proposals[label]=obj

    with np.load(source,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        member=z["member"].astype(str)
        start=z["start_samples"].astype(np.int64)
        end=z["end_samples"].astype(np.int64)
    with np.load(s80,allow_pickle=False) as z:
        need(np.array_equal(z["row_id"].astype(str),ids),
             "S440 source K IDs invalid")
        predicted_K=z["s80_K"]
    with np.load(groundtruth,allow_pickle=False) as z:
        need(np.array_equal(z["row_id"].astype(str),ids),
             "S440 true K ids invalid")
        true_K=z["true_K"]
    need(len(ids)==59309 and int(predicted_K.sum())==33475,
         "S440 protected original source altered")
    idmap={name:i for i,name in enumerate(ids)}
    selected=set(r["member"] for r in base)
    need(len(selected)==12 and len(set(composition_key(s) for s in selected))==12,
         "S440 unseen composition leakage among original 12")
    truths=collections.defaultdict(list)
    with zipfile.ZipFile(annotations) as z:
        index=collections.defaultdict(list)
        for path in z.namelist():
            if path.endswith(".jams"):
                index[PurePosixPath(path).name].append(path)
        for m in sorted(selected):
            matches=index[PurePosixPath(m).name]
            need(len(matches)==1,"S440 ambiguous official GuitarSet annotations")
            ii=np.flatnonzero(member==m)
            ii=ii[np.argsort(start[ii],kind="stable")]
            notes=read_note_events(z.read(matches[0]))
            _,mapping=assign_onsets(np.array([n[0] for n in notes],np.int64),
                                    start[ii],end[ii])
            for (_,pitch,string),pos in zip(notes,mapping):
                if pos>=0:
                    truths[int(ii[int(pos)])].append(int(pitch))
            need(all(len(truths[i])==int(true_K[i]) for i in ii),
                 "S440 official JAMS full-piece reconstruction invalid")
    truth={}
    for row in base:
        i=idmap[row["original_row_id"]]
        need(member[i]==row["member"] and
             int(start[i])==int(row["start_sample"]) and
             int(end[i])==int(row["end_sample"]) and
             int(predicted_K[i])==int(row["original_S80_K"]),
             "S440 frozen original timing/count mismatch")
        truth[row["original_row_id"]]=truths[i]
    scores,changes=evaluate(base,proposals,truth)
    need(scores["original_S437"]["whole_chords_correct"]==78
         and scores["original_S437"]["poly_K2plus_whole_chords_correct"]==27,
         "S440 independent original S438 audit not reproduced")
    out.mkdir(parents=True)
    report=dict(status="S440_PAIRED_REAL_MIC_12_GUITAR_JOINT_PITCH_UNPROMOTED_AUDIT",
      preregistered_predictions_before_any_labels=True,
      original_S80_unchanged_every_event=True,
      original_exactK_regressions=0,
      S439_preset_weights_not_chosen_from_this_score=True,
      GuitarSet_historical_training_overlap=True,
      no_independent_unseen_instrument_validation=True,
      no_promotion=True,score=scores,
      original_SHA256=sha(names),
      S439_report_SHA256=sha(s439_dir/"report.json"),
      official_annotation_SHA256=sha(annotations))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"paired_changes.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in changes))
    print("S440 PAIRED SCORE",json.dumps(scores),flush=True)
    print("S440 PASS original S80 K locked; prospective corrections/regressions measured",flush=True)

def selftest():
    base=[dict(original_row_id="e",notes=["E3"],original_S80_K=1),
          dict(original_row_id="f",notes=["G3"],original_S80_K=1)]
    variant=[dict(original_row_id="e",notes=["G3"],original_S80_K=1),
             dict(original_row_id="f",notes=["E3"],original_S80_K=1)]
    result,_=evaluate(base,{"test":variant},{"e":[55],"f":[55]})
    assert result["test"]["paired_vs_frozen_S437"]["corrections"]==1
    assert result["test"]["paired_vs_frozen_S437"]["regressions"]==1
    assert result["test"]["paired_vs_frozen_S437"]["net"]==0
    print("S440 SELFTEST PASS paired corrections/regressions and exact K")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for name in ("names","s437","s439-dir","source","s80",
                 "truth","annotations","out"):
        ap.add_argument("--"+name,type=Path)
    opts=ap.parse_args()
    if opts.self_test:
        selftest()
    else:
        for flag in ("names","s437","s439_dir","source","s80",
                     "truth","annotations","out"):
            need(getattr(opts,flag) is not None,"S440 missing "+flag)
        run(opts.names,opts.s437,opts.s439_dir,opts.source,
            opts.s80,opts.truth,opts.annotations,opts.out)
