"""S820 read-only SS1/Funk2/Rock3 K-transition and spectral-proxy forensics.

Frozen SHA-pinned S815 and S817 heads, fresh original mono GuitarSet
audio from SHA-pinned archive. This computes all GT-K to K_old/K_new
transitions and acoustic *descriptors*, never new training or policy
calibration. The sole counterfactual policy max_high@0.70 was fixed by
the preceding S819 research audit and is RETROSPECTIVE, NOT DEPLOYABLE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import joblib
import numpy as np

from causal_note.s817_highpoly_temporal import proposals, select
from causal_note.s814_activek_teacher import active_score
from scripts.s813_prepare_native256 import digest_file, save_json
from scripts.s814_train_activek256 import totals

SCHEMA="S820_TRUE_K_TRANSITION_SPECTRAL_PROXY_FIT80_RESEARCH_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
THRESHOLD=.70
EPS=1e-8

def sharp_proxies(X, changed):
    """Explicit DSP coordinates; not calibrated physical harmonic amplitudes.

    'sharp' consists of 3 resolution groups of 32 features at channel 126.
    The project's existing specialist-region head uses offsets 0,8,16
    for fundamental-like bins. Here we compare those with +1 offsets and
    pitchwise rank statistics. These are *descriptors*, not proof of f0.
    """
    x=np.asarray(X,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(49,222) or changed.shape!=(len(x),):
        raise ValueError("original fresh PCM256 ABI and aligned changed mask required")
    index=np.flatnonzero(changed)
    if not len(index):
        return {"n":0,"fundamental_proxy_mean":None,
                "harmonic_2_proxy_mean":None,"fund_minus_h2_mean":None,
                "fund_pitch_top4_vs_top1":None,
                "n_sharp_features_per_pitch":96}
    # Sparse slices avoid allocating entire original acoustic tensor.
    fund=x[index][:,:,[126,134,142]].mean(axis=2)
    h2=x[index][:,:,[127,135,143]].mean(axis=2)
    ranks=np.sort(fund,axis=1)
    return dict(n=len(index),
        fundamental_proxy_mean=float(np.mean(fund)),
        harmonic_2_proxy_mean=float(np.mean(h2)),
        fund_minus_h2_mean=float(np.mean(fund-h2)),
        fund_pitch_top4_vs_top1=float(np.mean(ranks[:,-4:].mean(axis=1)-ranks[:,-1])),
        n_sharp_features_per_pitch=96)

def acc_bucket(acc,key,val=1):
    acc[key]=acc.get(key,0)+int(val)

def record_counts(truth,base,alt,conf,high,X):
    """GT per-K transitions on ALL frames, and GT-conditioned changed bins.

    Gives 7x7x7 counts of true K, S815 K, S817 what-if K. K0 retained
    diagnostically but excluded from Active Exact-K denominator.
    """
    t=np.asarray(truth,np.int16)
    b=np.asarray(base,np.int16)
    n=np.asarray(alt,np.int16)
    if not(t.shape==b.shape==n.shape==conf.shape==high.shape):
        raise AssertionError("unaligned original framewise K labels/decisions")
    if not all(np.isin(x,np.arange(7)).all() for x in (t,b,n)):
        raise AssertionError("K outside 0-6")
    if np.any(n<b):
        raise AssertionError("max_high can never subtract notes")
    mask=n!=b
    gt=np.bincount(t,minlength=7)
    triple=np.zeros((7,7,7),dtype=np.int64)
    np.add.at(triple,(t,b,n),1)
    if int(triple.sum())!=len(t):
        raise AssertionError("lost 256-hop frame")
    # Change gains/losses are about Exact K, not number of pitches.
    fixed=mask&(b!=t)&(n==t)
    harmed=mask&(b==t)&(n!=t)
    still_under=mask&(n<t)
    overshot=mask&(n>t)
    became_over=mask&(b<=t)&(n>t)
    was_over=mask&(b>t)
    changed_and_active=mask&(t>0)
    counter=dict(
        changed_all=int(mask.sum()),
        changed_active=int(changed_and_active.sum()),
        became_correct=int(fixed.sum()),
        became_incorrect=int(harmed.sum()),
        changed_still_undercount=int(still_under.sum()),
        changed_overshoots=int(overshot.sum()),
        newly_overcounted=int(became_over.sum()),
        already_overcounted=int(was_over.sum()),
        gate_at_70_all=int(np.count_nonzero(conf>=THRESHOLD)),
        high_prediction_mismatch_count=int(np.count_nonzero(high!=b)),
        low_k_to_high_k=int(np.count_nonzero(mask&(b<4)&(n>=4))),
        original_k6_true=int(gt[6]),
        affected_k6_true=int(np.count_nonzero(mask&(t==6))),
        )
    if counter["became_correct"]-counter["became_incorrect"] != (
        int(np.count_nonzero(n==t))-int(np.count_nonzero(b==t))):
        raise AssertionError("ground truth correctness delta fails algebraic check")
    # Split counts by true K and by exact base->new integer transition.
    by_true={}
    for k in range(7):
        kt=t==k
        by_true[str(k)]=dict(total=int(gt[k]),
            changed=int(np.count_nonzero(mask&kt)),
            corrected=int(np.count_nonzero(fixed&kt)),
            broken=int(np.count_nonzero(harmed&kt)),
            old_correct=int(np.count_nonzero((b==t)&kt)),
            new_correct=int(np.count_nonzero((n==t)&kt)),
            old_over=int(np.count_nonzero((b>t)&kt)),
            new_over=int(np.count_nonzero((n>t)&kt)),
            old_under=int(np.count_nonzero((b<t)&kt)),
            new_under=int(np.count_nonzero((n<t)&kt)))
    xbands={
        "changed_all":sharp_proxies(X,mask),
        "newly_overcounted":sharp_proxies(X,became_over),
        "became_correct":sharp_proxies(X,fixed),
        "became_incorrect":sharp_proxies(X,harmed),
        "unchanged":sharp_proxies(X,~mask),
        "changed_true_k_1_3":sharp_proxies(X,mask&(t>=1)&(t<=3)),
        "changed_true_k_4_6":sharp_proxies(X,mask&(t>=4)),
    }
    return triple,counter,by_true,xbands

def nested_add(a,b):
    if a is None:return np.asarray(b,dtype=np.int64)
    return a+np.asarray(b,dtype=np.int64)

def true_k_analysis(matrix):
    out={}
    for k in range(7):
        a=np.asarray(matrix[k],dtype=np.int64)
        n=int(a.sum())
        b=np.sum(a,axis=1)
        c=np.sum(a,axis=0)
        out[str(k)]=dict(
            n=n,old_correct=int(b[k]),new_correct=int(c[k]),
            correct_delta=int(c[k]-b[k]),
            old_over=int(b[k+1:].sum()),new_over=int(c[k+1:].sum()),
            old_under=int(b[:k].sum()),new_under=int(c[:k].sum()),
            changed=int(np.count_nonzero(a[a!=0])) if False else int(
                n-int(np.trace(a))),
            changed_transitions=[
              {"from_K":int(old),"to_K":int(new),"frames":int(a[old,new])}
               for old in range(7) for new in range(7)
               if old!=new and a[old,new]>0
            ])
    return out

def aggregate_proxies(recs,category):
    fields=("fundamental_proxy_mean","harmonic_2_proxy_mean","fund_minus_h2_mean",
            "fund_pitch_top4_vs_top1")
    counts=[r["spectral_proxies"][category]["n"] for r in recs]
    n=sum(counts)
    return dict(n=n,**{
        field:(float(sum(r["spectral_proxies"][category]["n"]*
           r["spectral_proxies"][category][field]
           for r in recs if r["spectral_proxies"][category]["n"])/n)
           if n else None)
        for field in fields})

def run(cache,s817_dir,s815_dir,s819_report,out):
    if out.exists():raise FileExistsError("immutable analysis output already exists")
    fresh=json.loads((cache/"report.json").read_text())
    p817=json.loads((s817_dir/"report.json").read_text())
    p815=json.loads((s815_dir/"report.json").read_text())
    prior=json.loads(s819_report.read_text())
    if (not fresh.get("complete") or len(fresh.get("tracks",[]))!=80
        or fresh.get("schema")!="S813_FIT80_FRESH_UNSUPERVISED_DSP_MIDI256_V1"
        or not p817.get("complete") or not p815.get("complete")
        or not prior.get("complete")
        or p817["schema"]!="S817_REAL_FIT80_FIVE_PRIVATE_ACTIVEK_HEADS_MIDI256_V1"
        or p815["schema"]!="S815_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1"
        or prior["schema"]!="S819_S817_FROZEN_HIGH_POLYPHONY_GATE_FORENSIC_FIT80_V1"):
        raise ValueError("need real complete S813,S815,S817,S819 evidence")
    if (prior["source_817_report_sha256"]!=digest_file(s817_dir/"report.json")
        or prior["source_815_report_sha256"]!=digest_file(s815_dir/"report.json")):
        raise AssertionError("S819 earlier signed comparison mismatch")
    o817=s817_dir.parent/"fresh256"/"report.json"
    if digest_file(o817)!=p817["new_cache_sha256"]:
        raise AssertionError("S817 original PCM extraction source not authenticated")
    old=json.loads(o817.read_text())
    for key in ("source_manifest_sha256","audio_archive_sha256",
                "feature_hop_samples","decision_hop_samples",
                "metric_hop_samples","total_frames","feature_dimension"):
        if fresh[key]!=old[key]:
            raise AssertionError("new data not identical to frozen S817 PCM: "+key)
    identity=lambda rows:sorted((int(x["index"]),x["name"],x["composition"],
                    int(x["frames"]),x["audio_member_sha256"],x["original_json_sha256"])
                    for x in rows)
    if identity(fresh["tracks"])!=identity(old["tracks"]):
        raise AssertionError("signed WAV or annotation mismatch")
    out.mkdir(parents=True)
    result=dict(schema=SCHEMA,complete=False,
       sha_s819_prior=digest_file(s819_report),
       source_s819_run=39553290558,source_s817_run=39543749810,
       source_s815_run=39536863317,frames_per_hop=256,
       acoustic_frontend_features_per_pitch=222,
       policy=dict(mode="max_high",threshold=THRESHOLD,retrospective_only=True),
       FIT80_historically_explored=True,independent_external_validation=False,
       source_labels_used_in_inference=False,
       source_labels_used_for_selection=False,
       new_model_training=False,lookahead_samples_added=0,hold_samples_added=0,
       other_axes_modified=False,LIVE_modified=False,promotion=False,
       by_composition={},by_recording=[],whole_transition_7x7x7=None)
    save_json(out/"report.json",result)
    sum_all=None
    original_s815=[];counter_s820=[]
    previous_tracks={int(x["index"]):x for x in prior["tracks"]}
    previous_s817={int(x["index"]):x for x in p817["tracks"]}
    for comp in COMPS:
        m817=p817["folds"][comp]
        m815=p815["folds"][comp]
        f817=s817_dir/m817["checkpoint"]
        f815=s815_dir/m815["checkpoint"]
        if digest_file(f817)!=m817["checkpoint_sha256"] or digest_file(f815)!=m815["checkpoint_sha256"]:
            raise AssertionError("frozen real teacher checkpoint hash failed")
        high=joblib.load(f817)
        base=joblib.load(f815)
        for model in (high,base):
            if comp in model.metadata["train_compositions"]+model.metadata["calibration_compositions"]:
                raise AssertionError("frozen head trained on evaluated composition")
        composition=[]
        matrix=None
        for tr in sorted((x for x in fresh["tracks"] if x["composition"]==comp),
                         key=lambda x:x["index"]):
            k=int(tr["index"])
            if (tr["name"]!=previous_tracks[k]["name"]
                or tr["name"]!=previous_s817[k]["name"]):
                raise AssertionError("track names no longer identical")
            for file in tr["files"].values():
                if digest_file(cache/file["file"])!=file["sha256"]:
                    raise AssertionError("acoustic or truth file altered")
            X=np.load(cache/tr["files"]["features"]["file"],mmap_mode="r")
            truth=np.load(cache/tr["files"]["count_target"]["file"],mmap_mode="r")
            pred=base.predict(X)
            if active_score(pred,truth)!=previous_tracks[k]["original_s815"]:
                raise AssertionError("S815 original GT score not reproduced")
            s817=high.predict(X,pred)
            if active_score(s817,truth)!=previous_s817[k]["new"]:
                raise AssertionError("S817 original GT score not reproduced")
            conf,hyp=proposals(X,high.gate,high.high)
            alt=select(pred,conf,hyp,"max_high",THRESHOLD)
            actual=active_score(alt,truth)
            source=previous_tracks[k]["by_policy"]["max_high@0.70"]
            if any(actual[key]!=source[key] for key in
                   ("correct","frames","overcount","undercount")):
                raise AssertionError("S819 real counterfactual score not reproduced")
            mat,counters,by_true,proxies=record_counts(truth,pred,alt,conf,hyp,X)
            matrix=nested_add(matrix,mat)
            sum_all=nested_add(sum_all,mat)
            original_s815.append(active_score(pred,truth))
            counter_s820.append(actual)
            piece=dict(index=k,name=tr["name"],composition=comp,
                 active_k=actual,baseline=original_s815[-1],
                 matrix_7x7x7=mat.tolist(),counters=counters,
                 by_true_k=by_true,spectral_proxies=proxies)
            composition.append(piece)
            result["by_recording"].append(piece)
        if len(composition)!=16:raise AssertionError("16 songs for each composition required")
        oldcomp=prior["folds"][comp]
        predicted=totals([x["active_k"] for x in composition])
        if predicted!=oldcomp["actual_scores_by_fixed_policy"]["max_high@0.70"]:
            raise AssertionError("S819 composition result mismatch")
        result["by_composition"][comp]=dict(
            n_tracks=16,score=predicted,
            baseline=totals([x["baseline"] for x in composition]),
            transitions=matrix.tolist(),
            by_true_k=true_k_analysis(matrix),
            changed_frames=sum(x["counters"]["changed_all"] for x in composition),
            corrected_frames=sum(x["counters"]["became_correct"] for x in composition),
            harmed_frames=sum(x["counters"]["became_incorrect"] for x in composition),
            by_spectral_proxy={cat:aggregate_proxies(composition,cat) for cat in
                ("changed_all","newly_overcounted","became_correct",
                 "became_incorrect","unchanged","changed_true_k_1_3",
                 "changed_true_k_4_6")})
        print("S820_COMP",comp,"changed",result["by_composition"][comp]["changed_frames"],
             "corrected",result["by_composition"][comp]["corrected_frames"],
             "harmed",result["by_composition"][comp]["harmed_frames"],
             "activeK",predicted["accuracy"],flush=True)
        save_json(out/"report.json",result)
    baseline=totals(original_s815)
    counter=totals(counter_s820)
    if (baseline!=prior["original_baseline_s815"]
        or counter!=prior["outer_policy_totals"]["max_high@0.70"]["active_k"]
        or counter["correct"]!=141263 or baseline["correct"]!=141946
        or sum(int(z["counters"]["changed_all"]) for z in result["by_recording"])!=4867):
        raise AssertionError("S819 scientific ground truth differs")
    result.update(complete=True,baseline_activeK=baseline,counterfactual_activeK=counter,
        whole_transition_7x7x7=sum_all.tolist(),
        whole_by_true_k=true_k_analysis(sum_all),
        net_correct_delta=int(counter["correct"]-baseline["correct"]),
        total_changed_frames=4867,
        all_spectral_proxies={cat:aggregate_proxies(result["by_recording"],cat)
             for cat in ("changed_all","newly_overcounted","became_correct",
                         "became_incorrect","unchanged","changed_true_k_1_3",
                         "changed_true_k_4_6")})
    if result["net_correct_delta"]!=-683:
        raise AssertionError("S819 documented negative result absent")
    if sum(x["n"] for x in result["whole_by_true_k"].values())!=306536:
        raise AssertionError("true K 0..6 total not 306536")
    save_json(out/"report.json",result)
    print("S820_COMPLETE",json.dumps({"SS1":result["by_composition"]["SS1"]["by_true_k"],
         "total":result["net_correct_delta"],"proxies":result["by_composition"]["SS1"]["by_spectral_proxy"]},
         sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for a in ("cache","s817-dir","s815-dir","s819-report","out"):
        p.add_argument("--"+a,type=Path,required=True)
    a=p.parse_args()
    run(a.cache,a.s817_dir,a.s815_dir,a.s819_report,a.out)
