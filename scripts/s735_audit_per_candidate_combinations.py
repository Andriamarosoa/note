"""S735 diagnostic: all 32 H0-anchored 6-head combinations PER pitch/frame.

Uses FROZEN trained S734 selector weights: no fitting or thresholds chosen from
CAL truth. Each of 32 masks is a distinct counterfactual decision policy,
not a separately retrained network. One additional per-row dynamic-mask arm
uses only causal acoustic evidence. Re-decode true MIDI events with frozen
S708 + S707, score all six metrics on CAL, then freeze and replay HELD/DEV.
Historical HELD/DEV is already exposed, not a blind external test.
No LIVE deployment, no promotion, no TEST.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.strict_live import SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s733_one_target_selector import OneTargetClassConditionalSelector
from scripts.s613_prepare_strict_live import download,labels,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import (
    load_model,FIT_COMPS,CAL_COMPS,HELD_COMPS)
from scripts.s704_joint_metrics import metric,aggregate,SCORES
from scripts.s731_auditory_scene_calibration import emit
from scripts.s733_train_onset_only import baseline_input,head_schema

NHEADS=6
COMBOS=tuple(range(32))
BASE="S708_frozen"
DYNAMIC="per_row_dynamic"
NAMES=tuple("mask_%02d"%i for i in COMBOS)+(DYNAMIC,)
# Fixed pre-declared acoustic gate, not optimized on CAL/HELD.
MIN_GAIN=.035
MIN_POST=.18
MIN_NOVELTY=.055
MIN_RISE=.035
MIN_SUSTAIN=.50
MAX_RELEASE=.52
PROMOTED=False


def masks():
    # H0 mandatory for EVERY combination. H5 is the learned K-conditioned
    # per-pitch head, never a replacement for frozen global Exact-K.
    out=np.zeros((32,6),np.float32)
    out[:,0]=1.
    for i in range(32):
        for b in range(5):
            out[i,b+1]=float(bool(i&(1<<b)))
    assert len({tuple(row) for row in out})==32
    return out


def audit_selector_rows(model,candidates,ctx,eligible,k,post,batch_size=1024):
    """Calculate full-S734 contextual attention ONCE for each eligible pitch."""
    n_pitch=eligible.shape[1]
    inds=np.flatnonzero(eligible.ravel())
    frame,pitch=np.divmod(inds,n_pitch)
    n=len(inds)
    heads=np.empty((n,6),np.float32)
    weights=np.empty((n,6),np.float32)
    if n==0:return inds,frame,pitch,heads,weights
    model.eval()
    with torch.inference_mode():
        for st in range(0,n,batch_size):
            sub=slice(st,min(st+batch_size,n))
            fi,pi=frame[sub],pitch[sub]
            num=len(fi)
            htype,hmask,kmask=head_schema(num)
            def fl(a):return torch.from_numpy(np.asarray(a,dtype=np.float32))
            p=candidates[fi,pi].astype(np.float32)
            x=(fl(p),fl(np.zeros((num,5,21),np.float32)),
               fl(htype),torch.from_numpy(hmask),torch.from_numpy(kmask),
               fl(ctx[fi,pi]),fl(k[fi]),fl(p[:,0]))
            out=model(*x)
            heads[sub,:5]=p
            heads[sub,5]=out["k_linked_onset"].cpu().numpy()
            weights[sub]=out["selection_weights"].cpu().numpy()
    if not np.all(np.isfinite(heads)) or not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite S734 head proposal/attention")
    return inds,frame,pitch,heads,weights


def per_row_combinations(heads,weights):
    """[N,32] distinct subset predictions; H0 always included.

    This is a normalized subset of the FROZEN full-context attention;
    it is NOT a new neural training per combination.
    """
    mask=masks()
    mass=weights@mask.T
    if np.any(mass<=0):raise AssertionError("H0 fallback missing attention")
    predicted=((weights*heads)@mask.T)/mass
    # Max independent evidence is separately reported, never substituted
    # as an untrained global action probability.
    possible=np.max(np.where(mask[None,:,:].astype(bool),
                             heads[:,None,:],-1.),axis=2)
    # Each row can select its own H0-containing combination by an acoustic-
    # only context score. Penalize large subsets, avoid label/oracle K.
    additional=mask.sum(1)-1
    utility=predicted+.015*mass-.018*additional[None,:]
    winning=utility.argmax(1)
    return predicted,possible,winning


def assemble_modified(post,frame,pitch,heads,pooled,possible,winner,ctx,arm):
    """Candidate must pass same fixed causal physical gate in all masks."""
    n=len(frame)
    if n==0:return post.copy(),dict(candidates=0,boosted=0,unique_masks=0)
    original=post[frame,pitch,0]
    local=ctx[frame,pitch]
    acoustic=((local[:,4]>=MIN_NOVELTY) |
              (local[:,8]>=MIN_RISE) | (local[:,9]>=MIN_RISE))
    # Require a reasonably sustained, non-releasing candidate; other
    # heads may support overlapping notes without true K as hard mask.
    is_physical=(acoustic & (local[:,1]>=MIN_SUSTAIN) &
                 (local[:,2]<MAX_RELEASE) & (original<.75))
    if arm==DYNAMIC:
        selected=pooled[np.arange(n),winner]
        eligible_gain=possible[np.arange(n),winner]
        selected_masks=winner
    else:
        idx=int(arm.split("_")[1])
        selected=pooled[:,idx]
        eligible_gain=possible[:,idx]
        selected_masks=np.full(n,idx,np.int32)
    # ADD-only: regression-prone S734 suppression is deliberately not copied.
    potential=(is_physical & (eligible_gain>=MIN_POST) &
               (selected>=original+MIN_GAIN) &
               (eligible_gain>=original+MIN_GAIN))
    amended=post.copy()
    if potential.any():
        amended[frame[potential],pitch[potential],0]=np.maximum(
            amended[frame[potential],pitch[potential],0],.78)
    if not np.array_equal(amended[:,:,1:],post[:,:,1:]):
        raise AssertionError("S735 changed Sustain/Release")
    return amended,dict(candidates=int(n),boosted=int(potential.sum()),
                        unique_masks=int(len(set(selected_masks[potential]))),
                        masks_used={str(k):int(v) for k,v in
                            Counter(selected_masks[potential]).items()})


def event_delta(old,new):
    def normalized(events,kind):
        return Counter((int(e["pitch"]),int(e["available_sample"]))
                for e in events if e["type"]==kind)
    a,b=normalized(old,"note_on"),normalized(new,"note_on")
    return dict(added_note_on=int(sum((b-a).values())),
                removed_note_on=int(sum((a-b).values())))


def evaluation(specs,phase,archive,root,heads,selection,offset,network,
               policy,release,learner,arms):
    scores={arm:[] for arm in [BASE,*arms]}
    counts={arm:Counter() for arm in arms}
    paired={arm:Counter() for arm in arms}
    records=[]
    for i,spec in enumerate(specs):
        item,post,k,original,candidates,ctx,eligible=baseline_input(
            spec,archive,root,heads,selection,offset,network,policy,release)
        _,frame,pitch,p,w=audit_selector_rows(learner,candidates,ctx,
                                                eligible,k,post)
        combos,potential,winner=per_row_combinations(p,w)
        outputs={BASE:original}
        attempts={}
        for arm in arms:
            amended,stats=assemble_modified(post,frame,pitch,p,
                                      combos,potential,winner,ctx,arm)
            attempts[arm]=stats
            outputs[arm]=emit(item,amended,k,policy,offset,release)
        # Ground truth opened ONLY after all 32 causal decisions are final.
        truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
        measured={name:metric(events,truth,item["ends"])
                  for name,events in outputs.items()}
        base=measured[BASE]
        baseline_correct_offsets=set(base["correct_offset_truth_ids"])
        baseline_long=set(base["long_onset_truth_ids"])
        track=dict(name=spec["name"],audio_sha256=item["sha256"],
                   reference=dict(tp=int(base["tp"]),fp=int(base["fp"]),
                                  fn=int(base["fn"])),
                   arms={})
        for arm in [BASE,*arms]:
            scores[arm].append(measured[arm])
            if arm==BASE:continue
            deltas=event_delta(original,outputs[arm])
            stat=attempts[arm]
            counts[arm].update({
                "boosted":stat["boosted"],
                "added_note_on":deltas["added_note_on"],
                "removed_note_on":deltas["removed_note_on"]})
            paired[arm].update({
                "lost_correct_offsets":len(baseline_correct_offsets-
                              set(measured[arm]["correct_offset_truth_ids"])),
                "lost_long":len(baseline_long-
                              set(measured[arm]["long_onset_truth_ids"]))})
            track["arms"][arm]=dict(
                boosted=stat["boosted"],**deltas,
                tp=int(measured[arm]["tp"]),fp=int(measured[arm]["fp"]),
                fn=int(measured[arm]["fn"]))
        records.append(track)
        print("S735",phase,i+1,len(specs),spec["name"],
              "changed_arms",sum(track["arms"][a]["added_note_on"]>0
                                for a in arms),flush=True)
    aggregate_scores={arm:aggregate(rows) for arm,rows in scores.items()}
    return dict(scores={
        arm:{m:float(a[m]) for m in SCORES}
        for arm,a in aggregate_scores.items()},
        events={arm:{m:int(a[m]) for m in ("tp","fp","fn")}
            for arm,a in aggregate_scores.items()},
        changes={arm:dict(c) for arm,c in counts.items()},
        losses={arm:dict(c) for arm,c in paired.items()},
        records=records)


def qualify(report,arm):
    base=report["scores"][BASE];new=report["scores"][arm]
    count=report["events"];guard=report["losses"][arm]
    return (report["changes"][arm].get("added_note_on",0)>0 and
            count[arm]["tp"]>count[BASE]["tp"] and
            all(new[m]>=base[m]-1e-12 for m in SCORES)
            and guard.get("lost_correct_offsets",0)==0
            and guard.get("lost_long",0)==0)


def run(args):
    if args.out.exists():raise FileExistsError("S735 out immutable")
    args.out.mkdir(parents=True)
    frozen=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=frozen["checkpoint_sha256"]:raise AssertionError("Wrong S708 SHA")
    saved=torch.load(args.s734_checkpoint,map_location="cpu",weights_only=False)
    if saved["base_sha256"]!=sha or saved["target"]!="onset":
        raise AssertionError("S734 must be trained on EXACT S708 base")
    learner=OneTargetClassConditionalSelector().eval()
    learner.load_state_dict(saved["model"],strict=True)
    policy=GuidedPolicy(**frozen["frozen_candidate"]["choice"]["policy"])
    release=frozen["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("Original frozen S707 release mismatch")
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S735_ALL_32_CAUSAL_H0_ANCHORED_ONSET_COMBINATIONS",
       promoted=False,LIVE_changed=False,TEST_opened=False,
       base_sha256=sha,s734_sha256=digest(args.s734_checkpoint.read_bytes()),
       masks={f"mask_{i:02d}":["H0"]+[
          "H%d"%j for j in range(1,6) if i&(1<<(j-1))]
               for i in COMBOS},
       calibration_only="CAL32",training="NONE; frozen S734 and S708",
       no_oracle_per_row_selection=True,complete=False,
       fixed_gate=dict(min_gain=MIN_GAIN,min_post=MIN_POST,
              min_novelty=MIN_NOVELTY,min_rise=MIN_RISE,
              min_sustain=MIN_SUSTAIN,max_release=MAX_RELEASE))
    def checkpoint_report():
        (args.out/"report.json").write_text(
            json.dumps(report,indent=2,sort_keys=True)+"\n")
    checkpoint_report()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for kind,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                          ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            archive=protocol["guitar_archives"][kind]
            download(archive["url"],root/"data"/name,archive["size"],
                     archive["sha256"])
        train=[s for s in protocol["sources"] if
                s["corpus"]=="GuitarSet" and s["split"]=="train"]
        dev=[s for s in protocol["sources"] if
                s["corpus"]=="GuitarSet" and s["split"]=="dev"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav,\
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as xml:
            expanded=expanded_train_specs(train,wav,xml)
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            held=[s for s in expanded if s["composition"] in HELD_COMPS]
            if (len(cal),len(held),len(dev))!=(32,32,12):
                raise AssertionError("Unexpected reserved GuitarSet splits")
            calreport=evaluation(cal,"CAL",wav,root,heads,selection,offset,
                                network,policy,release,learner,NAMES)
            for key in SCORES:
                original=frozen["frozen_candidate"]["CAL"][key]
                reproduced=calreport["scores"][BASE][key]
                if abs(original-reproduced)>1e-8:
                    raise AssertionError("Original S708 CAL mismatch "+key)
            eligible=[arm for arm in NAMES if qualify(calreport,arm)]
            winner=max(eligible,key=lambda x:(
                calreport["scores"][x]["onset_f1"],
                calreport["scores"][x]["full_note_f1"]),
                default=BASE)
            report["CAL"]={**calreport,"eligible":eligible,
                 "selected":winner}
            checkpoint_report()
            print("S735 CAL FROZEN",json.dumps(dict(
                winner=winner,eligible=eligible,
                original=calreport["scores"][BASE],
                selected=calreport["scores"][winner],
                changed={a:calreport["changes"].get(a,{})
                  for a in ["mask_00","mask_31",DYNAMIC]})),flush=True)
            for phase,specs in (("HELD",held),("DEV",dev)):
                choices=[] if winner==BASE else [winner]
                row=evaluation(specs,phase,wav,root,heads,selection,
                     offset,network,policy,release,learner,choices)
                for met in SCORES:
                    if abs(row["scores"][BASE][met]-
                           frozen[phase]["candidate"][met])>1e-8:
                        raise AssertionError("Original S708 "+phase+
                                             " mismatch "+met)
                report[phase]=row
                checkpoint_report()
    report["complete"]=True
    report["eligible_for_live"]=False
    checkpoint_report()
    print("S735 FINAL",json.dumps(dict(
        selected=report["CAL"]["selected"],
        eligible=len(report["CAL"]["eligible"]),
        cal_baseline=report["CAL"]["scores"][BASE],
        cal_selected=report["CAL"]["scores"][report["CAL"]["selected"]],
        masks_tested=32,dynamic_per_row=True,promoted=False)),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--s734-checkpoint",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(ap.parse_args())
