"""S735: Exhaustive 32-subset + row-specific Onset selection diagnostic.

Research only, no promotion. Each subset includes H0. Frozen signed
S708 and trained S734 H5 are unchanged, and inference reads no labels.
The dynamic selector chooses a different subset per (frame,pitch) from
the pretrained S734 attention weights, not truth. CAL is diagnostic.
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

from causal_note.s733_one_target_selector import OneTargetClassConditionalSelector
from scripts.s733_train_onset_only import (
    baseline_input, head_schema, REFERENCE, H, collected_examples, raw_risk_stats)
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s706_train_sharp_temporal_loops import (
    FIT_COMPS, CAL_COMPS, HELD_COMPS, load_model)
from scripts.s705_train_temporal_loops import setup
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s731_auditory_scene_calibration import emit
from scripts.s704_train_joint_loops import targets
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.strict_live import SR

DYNAMIC="dynamic_learned_top2_individual"
ALL=[1+2*j for j in range(32)]
assert len(ALL)==32 and len(set(ALL))==32 and ALL[-1]==63


def load_s734(path,expected_sha):
    # Trusted run artifact, selected explicitly; never accept arbitrary pickle.
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved["base_sha256"]!=expected_sha or saved["target"]!="onset":
        raise AssertionError("S734 checkpoint provenance mismatch")
    model=OneTargetClassConditionalSelector()
    model.load_state_dict(saved["model"],strict=True)
    model.eval()
    return model


def k_head_and_dynamic(model,candidates,ctx,eligible,kprobs,audit):
    """One frozen full-head evaluation; H5 and attention 6D per candidate."""
    t,p=eligible.shape
    augmented=np.zeros((t,p,6),np.float32)
    augmented[:,:,:5]=candidates
    weights=np.zeros_like(augmented)
    # Fallback always H0, even on rows below the candidate gate.
    weights[:,:,0]=1.
    ids=np.flatnonzero(eligible.ravel())
    with torch.inference_mode():
        for start in range(0,len(ids),512):
            rows=ids[start:start+512]
            fi,pi=np.divmod(rows,p)
            count=len(rows)
            typ,mask,kmask=head_schema(count)
            args=(
                torch.from_numpy(candidates[fi,pi].copy()),
                torch.from_numpy(np.broadcast_to(audit,(count,H,21)).copy()),
                torch.from_numpy(typ),torch.from_numpy(mask),
                torch.from_numpy(kmask),
                torch.from_numpy(ctx[fi,pi].copy()),
                torch.from_numpy(kprobs[fi].copy()),
                torch.from_numpy(candidates[fi,pi,0].copy()))
            predicted=model(*args)
            augmented[fi,pi,5]=predicted["k_linked_onset"].numpy()
            weights[fi,pi]=predicted["selection_weights"].numpy()
    # Dynamic mask: baseline H0 plus 2 highest weighted optional specialists,
    # one combination for EACH pitch and decision frame, not one song.
    # All decisions use learned S734 weights only (no CAL truth).
    order=np.argsort(-weights[:,:,1:],axis=-1,kind="stable")
    chosen=np.zeros((t,p),np.uint8)
    chosen[:]=1
    for j in range(2):
        idx=order[:,:,j]+1
        weight=np.take_along_axis(weights,idx[:,:,None],axis=-1)[:,:,0]
        # Only trust a specialist with measurable learned attention.
        picked=eligible & (weight>=.10)
        chosen|=np.where(picked,np.left_shift(
            np.ones_like(idx,dtype=np.uint8),idx.astype(np.uint8)),0).astype(np.uint8)
    return augmented,chosen


def mask_posterior(base,heads,eligible,mask):
    """OR/max diagnostic; never touches Sustain, Release or Exact-K."""
    selected=[i for i in range(6) if mask&(1<<i)]
    revised=base.copy()
    offered=np.max(heads[:,:,selected],axis=-1)
    revised[:,:,0]=np.where(eligible,np.maximum(base[:,:,0],offered),base[:,:,0])
    if not np.array_equal(revised[:,:,1:],base[:,:,1:]):
        raise AssertionError("Onset changed non-Onset channels")
    return revised


def dynamic_posterior(base,heads,eligible,masks):
    revised=base.copy()
    chosen=np.zeros(base.shape[:2],np.float32)
    for i in range(6):
        chosen=np.maximum(chosen,np.where(
            (masks&(1<<i))!=0,heads[:,:,i],0.))
    revised[:,:,0]=np.where(eligible,np.maximum(base[:,:,0],chosen),base[:,:,0])
    if not np.array_equal(revised[:,:,1:],base[:,:,1:]):
        raise AssertionError("Modified Sustain or Release")
    return revised


def scores_only(r):
    return {k:float(r[k]) for k in SCORES}


def inspect(args):
    if args.out.exists():
        raise FileExistsError("Audit output must be new")
    args.out.mkdir(parents=True)
    raw=json.loads((args.s708_dir/"report.json").read_text())
    sha=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if sha!=raw["checkpoint_sha256"]:
        raise AssertionError("Signed S708 was modified")
    s734report=json.loads(args.s734_report.read_text())
    if s734report["base_sha256"]!=sha or not s734report["complete"]:
        raise AssertionError("Wrong S734 source or incomplete FIT run")
    network=load_model(args.s708_dir/"s708_selected.pt")[0]
    policy=GuidedPolicy(**raw["frozen_candidate"]["choice"]["policy"])
    release=raw["frozen_candidate"]["choice"]["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S708 release variant not frozen")
    model=load_s734(args.s734_checkpoint,sha)
    report=dict(schema="S735_32_H0_BASED_ONSET_SUBSETS_AND_PER_PITCH_DYNAMIC",
       signed_sha256=sha,source_run=39137653526,
       checkpoint_source="S734 trained FIT80 H5 with frozen Exact-K",
       audited_profiles_recomputed_from_FIT80_only=True,
       evaluation="32 OR/max subsets (not retrained) + 1 frozen dynamic attention policy",
       dynamic_policy="H0 plus up to two heads with largest S734 attention weights >= 0.10",
       exhaustive_subsets=32, frozen=True, promoted=False,
       LIVE_changed=False,TEST_opened=False,labels_only_after_emissions=True,
       whole_track_best_subset_not_deployable_without_independent_validation=True,
       completed=False,groups={},tracks=[],mask_examples={})
    if args.phase not in ("CAL","HELD"):
        raise ValueError("CAL or HELD only")
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        proto,native_heads,native_selection,offset=setup(root,args.offset_checkpoint)
        for key,archive_name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                                 ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            asset=proto["guitar_archives"][key]
            download(asset["url"],root/"data"/archive_name,asset["size"],asset["sha256"])
        originals=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with (zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as aud,
              zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann):
            expanded=expanded_train_specs(originals,aud,ann)
            # Recreate the FIT-only audit features seen in S734 training.
            # Zero-filled risk features would change the input distribution.
            fit_specs=[s for s in expanded if s["composition"] in FIT_COMPS]
            if len(fit_specs)!=80:
                raise AssertionError("Missing FIT80 compositions")
            fit_stats=np.zeros((H,7,3),np.float64)
            fit_hist=np.zeros(7,np.float64)
            for fi,source in enumerate(fit_specs):
                item0,post0,k0,stream0,cand0,ctx0,eligible0=baseline_input(
                    source,aud,root,native_heads,native_selection,offset,
                    network,policy,release)
                real=real_notes(labels(source,root)["notes"],item0["samples"]/SR)
                binary,ktruth=targets(item0["decision_ends"],real)
                sample=collected_examples(cand0,ctx0,eligible0,binary[:,:,0],
                    ktruth,k0,source["composition"],seed=733+fi)
                fit_stats+=raw_risk_stats(sample)
                fit_hist+=np.bincount(sample["true_k"],minlength=7)
                if (fi+1)%20==0:
                    print("S735 FIT-only risk replay",fi+1,flush=True)
            learned_audit=(fit_stats/(fit_hist[None,:,None]+16)).reshape(
                H,21).astype(np.float32)
            required=CAL_COMPS if args.phase=="CAL" else HELD_COMPS
            specs=[s for s in expanded if s["composition"] in required]
            if len(specs)!=32:
                raise AssertionError("Unexpected GuitarSet split")
            events_per_mask={str(mask):[] for mask in ALL}
            events_per_mask[DYNAMIC]=[]
            events_per_mask[REFERENCE]=[]
            subset_total_rescues=Counter()
            histogram=Counter()
            for j,spec in enumerate(specs):
                item,post,k,baseline,candidates,ctx,eligible=baseline_input(
                    spec,aud,root,native_heads,native_selection,offset,network,policy,release)
                # FIT80 summaries only: no labels from this track enter decisions.
                # Exact S734 FIT80 risk profile; CAL labels are still unopened.
                augmented,dynamic_mask=k_head_and_dynamic(
                    model,candidates,ctx,eligible,k,learned_audit)
                histogram.update(str(int(i)) for i in dynamic_mask[eligible].ravel())
                streams={REFERENCE:baseline}
                streams[DYNAMIC]=emit(item,dynamic_posterior(
                    post,augmented,eligible,dynamic_mask),k,policy,offset,release)
                for m in ALL:
                    name=str(m)
                    if m==1:
                        streams[name]=baseline
                    else:
                        streams[name]=emit(item,mask_posterior(
                            post,augmented,eligible,m),k,policy,offset,release)
                # Annotations opened only AFTER computing all candidate streams.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                track={name:metric(stream,truth,item["ends"])
                       for name,stream in streams.items()}
                for name,score in track.items():
                    events_per_mask[name].append(score)
                base_hits=set(track[REFERENCE]["matched_truth_ids"])
                ceiling=set().union(*(set(track[str(m)]["matched_truth_ids"]) for m in ALL))
                newly=ceiling-base_hits
                changes={
                    name:dict(recovered_truth=len(
                        set(value["matched_truth_ids"])-base_hits),
                        lost_truth=len(base_hits-set(value["matched_truth_ids"])),
                        tp=int(value["tp"]),fp=int(value["fp"]),fn=int(value["fn"]))
                    for name,value in track.items() if name!=REFERENCE}
                best_local=sorted(changes,key=lambda n:(
                    changes[n]["recovered_truth"]-changes[n]["lost_truth"],
                    -changes[n]["fp"]),reverse=True)[:3]
                subset_total_rescues.update({str(len(newly)):1})
                report["tracks"].append(dict(
                    name=spec["name"],texture=("comp" if spec["name"].endswith("_comp") else "solo"),
                    base_onset=scores_only(track[REFERENCE])["onset_f1"],
                    oracle_union_any_fixed_mask_rescued=len(newly),
                    best_exploratory_masks=[dict(mask=x,**changes[x]) for x in best_local],
                    dynamic=changes[DYNAMIC]))
                print("S735 SUBSETS",args.phase,j+1,len(specs),spec["name"],
                      "unique_any_mask_rescues",len(newly),
                      "dynamic",changes[DYNAMIC],flush=True)
            for category in ("comp","solo","all"):
                ix=[i for i,x in enumerate(report["tracks"]) if
                    category=="all" or x["texture"]==category]
                total={name:aggregate([rows[i] for i in ix])
                       for name,rows in events_per_mask.items()}
                baseline=total[REFERENCE]
                if category=="all":
                    original=raw[args.phase]["candidate"]
                    for m in SCORES:
                        if abs(baseline[m]-original[m])>1.e-8:
                            raise AssertionError("S708 signed replay mismatch "+m)
                ranked=sorted((str(m) for m in ALL),key=lambda name:(
                    total[name]["onset_f1"]-baseline["onset_f1"],
                    min(total[name][m]-baseline[m] for m in SCORES)),reverse=True)
                report["groups"][category]=dict(
                    n_tracks=len(ix),reference=scores_only(baseline),
                    dynamic=scores_only(total[DYNAMIC]),
                    dynamic_delta={m:float(total[DYNAMIC][m]-baseline[m]) for m in SCORES},
                    top_masks=[dict(mask=int(name),
                        heads=[h for h in range(6) if int(name)&(1<<h)],
                        scores=scores_only(total[name]),
                        delta={metric_name:float(total[name][metric_name]-baseline[metric_name])
                               for metric_name in SCORES},
                        onset_totals={name_:int(total[name][name_])
                                      for name_ in ("tp","fp","fn")})
                               for name in ranked[:10]],
                    valid_noregression_masks=[int(name) for name in ranked
                        if all(total[name][metric_name]>=baseline[metric_name]-1e-12
                               for metric_name in SCORES)])
            report["mask_histogram_dynamic"]=dict(histogram)
    report["completed"]=True
    args.out.joinpath("report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S735_FINAL",json.dumps({
        "phase":args.phase,"top_masks":report["groups"]["all"]["top_masks"][:5],
        "dynamic":report["groups"]["all"]["dynamic"],
        "reference":report["groups"]["all"]["reference"],
        "safe_masks":report["groups"]["all"]["valid_noregression_masks"],
        "promoted":False}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s708-dir",required=True,type=Path)
    parser.add_argument("--s734-checkpoint",required=True,type=Path)
    parser.add_argument("--s734-report",required=True,type=Path)
    parser.add_argument("--offset-checkpoint",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--phase",default="CAL")
    with threadpool_limits(limits=2):
        inspect(parser.parse_args())
