"""S710: diagnose CAL errors, then search one causal joint state filter.

Frozen S706/S709 acoustic models, unchanged data split. Six targets are always
computed from one emitted stream. Shadow head diagnostics are never a score
of the emitted system. TEST labels are never opened.
"""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import shutil
import zipfile
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from causal_note.strict_live import SR
from causal_note.s704_joint_heads import JointPolicy,JointSelector,decode_joint
from causal_note.s708_cardinality_selection import GuidedPolicy,GuidedSelector,decode_guided
from causal_note.s710_joint_state_filter import StatePolicy,decode_state
from causal_note.s705_temporal_heads import predict_track
from scripts.s706_train_sharp_temporal_loops import (
    setup,record,load_model as load706,CAL_COMPS,HELD_COMPS)
from scripts.s709_train_string_heads import load_model as load709
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_train_joint_loops import targets,save,short,report_split
from scripts.s704_joint_metrics import metric,aggregate,target_rank,SCORES
from scripts.s704_verify_emitted_artifact import verify

CHECKPOINTS={
    "706":"3b953ae77f975522cb7a48c7f478e78e55023b97cd35dd49aa16417e7489a1e0",
    "709":"68327c6b5e44ad4497af91b25259f454905d8d3801e68113a565b18d7b4c01a5",
}

def load_assets(args,download_audio=True):
    (args.work/"data").mkdir(parents=True,exist_ok=True)
    protocol,heads,selection,offset=setup(args.work,args.offset_checkpoint)
    if download_audio:
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=protocol["guitar_archives"][key]
            download(source["url"],args.work/"data"/name,source["size"],source["sha256"])
    models={};bundles={}
    for family,path,loader in (("706",args.s706,load706),("709",args.s709,load709)):
        assert digest(path.read_bytes())==CHECKPOINTS[family],family
        models[family],bundles[family]=loader(path)
    return protocol,heads,selection,offset,models,bundles

def control(item,family,bundle):
    p,k=item["models"][family]
    if family=="709":
        return decode_guided(item["decision_ends"],p,k,GuidedPolicy(**bundle["policy"]),item["samples"])
    return decode_joint(item["decision_ends"],p,k,JointPolicy(**bundle["policy"]),item["samples"])

def predict_item(spec,archive,heads,selection,offset,models):
    item=record(spec,archive,heads,selection,offset)
    result={}
    for family,model in models.items():
        p,k=predict_track(model,item["x"])
        result[family]=(.7*p+.3*item["native_post"],k)
    item["models"]=result
    item.pop("x");item.pop("cx",None)
    return item

def diagnose(item,family,bundle):
    p,k=item["models"][family];truth=item["truth"]
    y,true_k=targets(item["decision_ends"],truth)
    predicted_k=k.argmax(1)
    active=true_k>0
    events=control(item,family,bundle)
    score=metric(events,truth,item["ends"])
    cls=GuidedSelector if family=="709" else JointSelector
    policy=GuidedPolicy(**bundle["policy"]) if family=="709" else JointPolicy(**bundle["policy"])
    selector=cls(policy);armed=[];age=[];occupancy=[]
    for end,post,count in zip(item["decision_ends"],p,k):
        armed.append((selector.armed | (post[:,0]<policy.onset*.4)).copy())
        age.append((end-selector.last_on).copy())
        occupancy.append(int(selector.active.sum()))
        selector.step(post,count,int(end))
    armed=np.array(armed);age=np.array(age);occupancy=np.array(occupancy)
    matched=set(score["matched_truth_ids"])
    causes={key:0 for key in ("head_gate_closed","rearm_blocked","refractory","capacity_or_state","other_overlap")}
    head_opportunity=presence_opportunity=0
    times=item["decision_ends"]/SR
    for j,note in enumerate(truth):
        within=np.abs(times-note["time"])<=.080
        pitch=note["pitch"]-40
        strong=within&(p[:,pitch,0]>=policy.onset)&(p[:,pitch,1]>=policy.sustain_on)
        head_opportunity+=bool(strong.any())
        presence_opportunity+=bool((within&(p[:,pitch,1]>=.5)).any())
        if j in matched:continue
        ready=strong&armed[:,pitch]
        refractory=ready&(age[:,pitch]>=.065*SR)
        if not strong.any():causes["head_gate_closed"]+=1
        elif not ready.any():causes["rearm_blocked"]+=1
        elif not refractory.any():causes["refractory"]+=1
        elif np.all(occupancy[refractory]>=6):causes["capacity_or_state"]+=1
        else:causes["other_overlap"]+=1
    intervals=[];opened={}
    for event in events:
        pitch=event["pitch"]
        if event["type"]=="note_on":
            opened[pitch]=len(intervals);intervals.append([event,None])
        elif pitch in opened:intervals[opened.pop(pitch)][1]=event
    _,pairs=event_score(events,truth)
    errors=[];panic=0
    for i,j in pairs:
        expected=truth[j]["time"]+truth[j]["duration"]
        if expected>=item["ends"][-1]/SR:continue
        off=intervals[i][1]
        if off is None or off["reason"]=="stream_end":panic+=1;continue
        errors.append((off["time"]-expected)*1000)
    return dict(name=item["spec"]["name"],texture=item["spec"]["name"].rsplit("_",1)[1],
        frames=len(true_k),active_frames=int(active.sum()),
        raw_count_correct=int((predicted_k==true_k).sum()),
        raw_active_count_correct=int(((predicted_k==true_k)&active).sum()),
        raw_count_under=int((predicted_k<true_k).sum()),raw_count_over=int((predicted_k>true_k).sum()),
        reference_notes=len(truth),missed_notes=score["fn"],head_gate_opportunity=head_opportunity,
        presence_opportunity=presence_opportunity,missed_causes=causes,
        offset_early=sum(e< -50 for e in errors),offset_within=sum(abs(e)<=50 for e in errors),
        offset_late=sum(e>50 for e in errors),offset_panic=panic,offset_errors_ms=errors,
        emitted=score)

def summarize_diagnostics(rows):
    total={}
    keys=("frames","active_frames","raw_count_correct","raw_active_count_correct","raw_count_under",
          "raw_count_over","reference_notes","missed_notes","head_gate_opportunity","presence_opportunity",
          "offset_early","offset_within","offset_late","offset_panic")
    for key in keys:total[key]=sum(r[key] for r in rows)
    total["missed_causes"]={key:sum(r["missed_causes"][key] for r in rows) for key in rows[0]["missed_causes"]}
    total["raw_count_accuracy"]=total["raw_count_correct"]/total["frames"]
    total["raw_active_count_accuracy"]=total["raw_active_count_correct"]/max(1,total["active_frames"])
    total["head_gate_opportunity_ratio"]=total["head_gate_opportunity"]/total["reference_notes"]
    total["presence_opportunity_ratio"]=total["presence_opportunity"]/total["reference_notes"]
    errors=[e for row in rows for e in row["offset_errors_ms"]]
    total["offset_error_ms_p10_p50_p90"]=np.percentile(errors,[10,50,90]).tolist() if errors else []
    total["emitted"]=aggregate([r["emitted"] for r in rows])
    return total

def prepare(args):
    args.cache.mkdir(parents=True,exist_ok=False)
    protocol,heads,selection,offset,models,bundles=load_assets(args)
    originals=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    diagnostic={family:[] for family in models}
    with zipfile.ZipFile(args.work/"data/GuitarSet_audio.zip") as archive,zipfile.ZipFile(args.work/"data/GuitarSet_annotations.zip") as ann:
        expanded=expanded_train_specs(originals,archive,ann)
        cal=[s for s in expanded if s["composition"] in CAL_COMPS]
        held=[s for s in expanded if s["composition"] in HELD_COMPS]
        assert (len(cal),len(held),len(dev))==(32,32,12)
        for i,spec in enumerate(cal):
            item=predict_item(spec,archive,heads,selection,offset,models)
            # Only now use CAL labels for analysis/calibration.
            item["truth"]=real_notes(labels(spec,args.work)["notes"],item["samples"]/SR)
            for family in models:diagnostic[family].append(diagnose(item,family,bundles[family]))
            joblib.dump(item,args.cache/f"cal_{i:03d}.joblib",compress=3)
            print("S710 PREPARE",i+1,len(cal),spec["name"],flush=True)
    summary={family:dict(all=summarize_diagnostics(rows),
        by_texture={texture:summarize_diagnostics([r for r in rows if r["texture"]==texture])
                    for texture in ("comp","solo")}) for family,rows in diagnostic.items()}
    save(args.cache/"manifest.json",dict(held=held,dev=dev,checkpoint_sha256=CHECKPOINTS,
        cal_compositions=CAL_COMPS,held_compositions=HELD_COMPS,TEST_opened=False,
        diagnostic_note="Raw count and gate opportunities are shadow diagnostics, not emitted-stream performance."))
    save(args.cache/"diagnosis.json",summary)
    save(args.cache/"diagnosis_per_source.json",diagnostic)
    print("S710 DIAGNOSIS "+json.dumps(summary),flush=True)

def choices():
    result={}
    for family in CHECKPOINTS:
        result[family+"_control"]=dict(family=family,kind="control")
        for opening,stay,count,release,closing in itertools.product((0.,1.,2.),(0.,1.),(0.,1.,2.),(0.,1.),(1,2)):
            policy=StatePolicy(opening_penalty=opening,stay_bonus=stay,count_weight=count,
                               release_weight=release,close_steps=closing)
            name=f"{family}_open{opening}_stay{stay}_k{count}_rel{release}_close{closing}"
            result[name]=dict(family=family,kind="state",policy=asdict(policy))
    return result

def decode_choice(item,choice,bundles):
    family=choice["family"]
    if choice["kind"]=="control":return control(item,family,bundles[family])
    p,k=item["models"][family]
    return decode_state(item["decision_ends"],p,k,StatePolicy(**choice["policy"]),item["samples"])

def search(args):
    args.out.mkdir(parents=True,exist_ok=False)
    protocol,heads,selection,offset,models,bundles=load_assets(args)
    manifest=json.loads((args.cache/"manifest.json").read_text())
    assert manifest["checkpoint_sha256"]==CHECKPOINTS and not manifest["TEST_opened"]
    policies=choices();scores={name:[] for name in policies}
    for i,path in enumerate(sorted(args.cache.glob("cal_*.joblib"))):
        item=joblib.load(path)
        for name,choice in policies.items():
            events=decode_choice(item,choice,bundles)
            scores[name].append(metric(events,item["truth"],item["ends"]))
        print("S710 CAL",i+1,32,flush=True)
    scores={name:aggregate(rows) for name,rows in scores.items()}
    best=max(scores,key=lambda n:target_rank(scores[n]))
    choice=policies[best]
    source=args.s706 if choice["family"]=="706" else args.s709
    shutil.copyfile(source,args.out/"s710_model.pt")
    report=dict(schema="S710_JOINT_CAUSAL_STATE_FILTER",promoted=False,official_LIVE_changed=False,
        TEST_opened=False,DEV_previously_exposed=True,HELD_previously_exposed=True,
        checkpoint_sha256=digest(source.read_bytes()),source_checkpoints=CHECKPOINTS,
        cal_compositions=CAL_COMPS,held_compositions=HELD_COMPS,
        frozen_candidate=dict(name=best,choice=choice,CAL=scores[best]),rounds=[],
        thresholds={s:.8 for s in SCORES},batch_hops=4,
        note="Frozen acoustics, causal emitted state from joint pitch-presence/count score. No label input to inference. All six metrics from same events.")
    for family in CHECKPOINTS:
        sub={name:value for name,value in scores.items() if policies[name]["family"]==family}
        report["rounds"].append(dict(source_family=family,best=max(sub,key=lambda n:target_rank(sub[n])),scores=sub))
    save(args.out/"report.json",report);save(args.out/"selection_frozen.json",report["frozen_candidate"])
    shutil.copyfile(args.cache/"diagnosis.json",args.out/"cal_diagnosis.json")
    print("S710 FROZEN "+json.dumps(report["frozen_candidate"]),flush=True)
    with zipfile.ZipFile(args.work/"data/GuitarSet_audio.zip") as archive:
        for phase in ("HELD","DEV"):
            specs=manifest[phase.lower()];rows=[]
            with (args.out/(phase.lower()+"_events.jsonl")).open("w") as fh:
                for i,spec in enumerate(specs):
                    item=predict_item(spec,archive,heads,selection,offset,models)
                    events=decode_choice(item,choice,bundles)
                    event_dict=dict(item["baseline"],candidate=events)
                    save(args.out/"latest_prediction_before_labels.json",dict(name=spec["name"],events=event_dict))
                    truth=real_notes(labels(spec,args.work)["notes"],item["samples"]/SR)
                    metrics={name:metric(e,truth,item["ends"]) for name,e in event_dict.items()}
                    base,trial=metrics["native_s675_causal"],metrics["candidate"]
                    paired=dict(recovered=len(set(trial["matched_truth_ids"])-set(base["matched_truth_ids"])),
                        lost=len(set(base["matched_truth_ids"])-set(trial["matched_truth_ids"])),
                        lost_long=len(set(base["long_onset_truth_ids"])-set(trial["long_onset_truth_ids"])),
                        lost_correct_offsets=len(set(base["correct_offset_truth_ids"])-set(trial["correct_offset_truth_ids"])))
                    rows.append(dict(name=spec["name"],composition=spec["composition"],metrics=metrics,paired=paired))
                    fh.write(json.dumps(dict(name=spec["name"],sha256=item["sha256"],samples=item["samples"],truth=truth,events=event_dict))+"\n");fh.flush()
                    print("S710",phase,i+1,len(specs),spec["name"],json.dumps(short(trial)),flush=True)
            report[phase]=report_split(rows,("native","native_s675_causal","candidate"))
            report[phase]["paired"]={key:sum(row["paired"][key] for row in rows) for key in rows[0]["paired"]}
            base,trial=report[phase]["native_s675_causal"],report[phase]["candidate"]
            report[phase]["non_regression"]=bool(all(trial[s]>=base[s] for s in SCORES)
                and trial["tp"]>=base["tp"] and report[phase]["paired"]["lost_long"]==0
                and report[phase]["paired"]["lost_correct_offsets"]==0)
            save(args.out/(phase.lower()+"_audit.json"),rows);save(args.out/"report.json",report)
    report["all_targets_above_80"]=all(report[p]["candidate"]["all_targets_above_80"] for p in ("HELD","DEV"))
    report["non_regression_passed"]=all(report[p]["non_regression"] for p in ("HELD","DEV"))
    save(args.out/"report.json",report)
    archive_path=args.out/"raw_recount_inputs.zip"
    with zipfile.ZipFile(archive_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("report.json","held_events.jsonl","dev_events.jsonl"):z.write(args.out/name,name)
    verified=verify(archive_path)
    assert verified["all_raw_event_counts_match_report"]
    assert all(verified[p]["candidate"]["max_active"]<=6 for p in ("HELD","DEV"))
    save(args.out/"independent_recount.json",verified)
    print("S710 VERIFIED "+json.dumps(verified,separators=(",",":")),flush=True)
    print("S710 FINAL "+json.dumps({k:report[k] for k in ("frozen_candidate","HELD","DEV","all_targets_above_80","promoted")}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("prepare","search"))
    ap.add_argument("--work",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--s706",type=Path,required=True)
    ap.add_argument("--s709",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):globals()[args.mode](args)
