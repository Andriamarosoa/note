"""S803: exploratory current-frame vetoes, actual GuitarSet FIT80 MIDI256 replay.

Frozen heads, frozen decoder and S708 native output. Post-hoc truth is used
solely for measurement. Thresholds are PREDECLARED, never optimized on FIT.
This analysis does not change production LIVE or authorize promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from causal_note.s795_activek_experts import HEADS, decode
from causal_note.s797_fragment_router import CUT
from scripts.s704_joint_metrics import SCORES, metric, aggregate
from scripts.s795_train_activek_midi256 import COUNT_KEYS
from scripts.s800_oof_trace import SCHEMA as TRACE_SCHEMA, verify_snapshot
from scripts.s801_fit_fragment_ablations import route
from scripts.s802_retrigger_forensics import SCHEMA as S802_SCHEMA

SCHEMA="S803_FIT80_CAUSAL_VETO_SIX_METRIC_V1"
POLICIES=("rise_attack","parent_conflict","agreement_attack")
BASE={"rise_attack":"adaptive","parent_conflict":"consensus",
      "agreement_attack":"adaptive"}
FEATURE_SIGNS={"on":1,"sustain":1,"off":-1,"on_rise":1,
               "sustain_rise":1,"harmonic_parent":-1,
               "general_p":1,"bass_p":1,"polyphonic_p":1,
               "transient_p":1,"harmonic_p":1}

def gating(X,P,policy):
    """Only information available at the current (frame,pitch) decision."""
    if policy not in POLICIES: raise ValueError("unregistered trial")
    routed,masks,meta=route(X,P,BASE[policy])
    if policy=="rise_attack":
        accept=(X[:,:,0]>=.20)&(X[:,:,5]>=.12)
    elif policy=="parent_conflict":
        accept=(X[:,:,10]<=.02)|(X[:,:,5]>=.20)
    else:
        accept=(P["general"]>=.96)&(P["transient"]>=.93)&(X[:,:,5]>=.10)
    blocked=int(np.count_nonzero((masks!=0)&~accept))
    routed={h:np.where(accept,routed[h],0).astype(np.float32)
            for h in HEADS}
    return routed,blocked

def auc_diagnostics(forensic):
    """Retrospective discrimination: never feeds the MIDI decision."""
    if forensic.get("schema")!=S802_SCHEMA or not forensic.get("complete"):
        raise AssertionError("Authentic S802 signed observations required")
    expected={"consensus":(136,74,62),"poly_gap":(90,44,46),
              "adaptive":(118,67,51)}
    by_policy={}
    for policy,(n,tp,fp) in expected.items():
        rows=forensic["observations"][policy]
        y=np.asarray([int(z["synthetic_matched_truth"]) for z in rows])
        if len(y)!=n or int(y.sum())!=tp or n-tp!=fp:
            raise AssertionError("S802 authentic TP/FP counts drift: "+policy)
        policy_auc={}
        for feature,sign in FEATURE_SIGNS.items():
            scores=np.asarray([float(z[feature])*sign for z in rows])
            if not np.isfinite(scores).all():raise AssertionError("nonfinite evidence")
            auc=float(roc_auc_score(y,scores))
            by_comp={}
            for c in sorted({z["composition"] for z in rows}):
                ix=[i for i,z in enumerate(rows) if z["composition"]==c]
                yy=y[ix]
                by_comp[c]=dict(n=len(ix),tp=int(yy.sum()),fp=len(ix)-int(yy.sum()),
                                auc=(round(float(roc_auc_score(yy,scores[ix])),6)
                                     if len(np.unique(yy))==2 else None))
            # Stratified cluster bootstrap over whole tracks (not per-note).
            groups=defaultdict(list)
            for i,z in enumerate(rows):groups[(z["composition"],z["track"])].append(i)
            keys=sorted(groups)
            rng=np.random.default_rng(803)
            boot=[]
            if len(keys)>1:
                for _ in range(192):
                    drawn=rng.integers(0,len(keys),size=len(keys))
                    ids=[i for k in drawn for i in groups[keys[int(k)]]]
                    if ids and len(np.unique(y[ids]))==2:
                        boot.append(float(roc_auc_score(y[ids],scores[ids])))
            ci=([round(float(v),6) for v in np.quantile(boot,[.025,.975])]
                if len(boot)>=40 else None)
            policy_auc[feature]=dict(auc=round(auc,6),track_bootstrap_95=ci,
                                     valid_bootstrap=len(boot),by_composition=by_comp)
        by_policy[policy]=dict(n=n,tp=tp,fp=fp,features=policy_auc)
    return by_policy

def evaluate(manifest_path,traces,s801_path,s802_path):
    m=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    a=json.loads(Path(s801_path).read_text(encoding="utf-8"))
    f=json.loads(Path(s802_path).read_text(encoding="utf-8"))
    if m.get("schema")!=TRACE_SCHEMA or not m.get("complete") or not m.get("FIT_source_parity_S795"):
        raise AssertionError("S800 signed FIT80 manifest required")
    if len(m.get("fit_files",[]))!=80 or len({z["index"] for z in m["fit_files"]})!=80:
        raise AssertionError("not 80 distinct real FIT tracks")
    if a.get("schema")!="S801_FIT80_CAUSAL_DYNAMIC_ABLATION_V1" or not a.get("completed"):
        raise AssertionError("signed S801 run absent")
    if a.get("provisional_fit_candidate")!="identity":
        raise AssertionError("S801 identity gate unexpectedly changed")
    if f.get("s801_policy_six_scores_and_counts_parity") is not True:
        raise AssertionError("S802 six-metric parity absent")
    for x in (m,a,f):
        if any(x.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):
            raise AssertionError("reserved validation set opened")
    aucs=auc_diagnostics(f)
    per={p:[] for p in POLICIES}; native=[]; blocked={p:0 for p in POLICIES}
    for entry in sorted(m["fit_files"],key=lambda q:q["index"]):
        info=verify_snapshot(entry,traces)
        with np.load(Path(traces)/f"fit_{entry['index']:03d}.npz",
                     allow_pickle=False) as z:
            X=z["features"].astype(np.float32)
            d=z["decision_ends"]
            ends=z["metric_ends"]
            P={h:z["head_"+h].astype(np.float32) for h in HEADS}
        source=metric(info["source_events"],info["fit_truth"],ends)
        if any(int(source[k])!=int(info["source_counts"][k]) for k in COUNT_KEYS):
            raise AssertionError("S800 track source count mismatch")
        native.append((info["composition"],source))
        for policy in POLICIES:
            routed,nblocked=gating(X,P,policy)
            out,delta=decode(info["source_events"],d,X,routed,31,CUT,
                             int(info["samples"]))
            if any(delta.get(k,0)!=0 for k in
                   ("max_added_native_delay_samples","extra_lookahead_samples",
                    "extra_hold_samples","extra_confirmation_samples",
                    "extra_buffer_samples")):
                raise AssertionError("extra MIDI timing/hold introduced")
            sc=metric(out,info["fit_truth"],ends)
            blocked[policy]+=nblocked
            per[policy].append((info["composition"],sc,delta))
        print("S803_REAL_FIT_TRACK",entry["index"],info["name"],flush=True)
    source=aggregate([row for _,row in native])
    if any(int(source[k])!=int(m["S795_archived_fit_counts"][k]) or
           int(source[k])!=int(a["signed_source_counts"][k]) for k in COUNT_KEYS):
        raise AssertionError("signed S800/S801 source parity drift")
    if any(abs(float(source[k])-float(a["baseline_fit"][k]))>1e-12 for k in SCORES):
        raise AssertionError("S801 baseline six-ratio parity drift")
    comps=sorted({c for c,_ in native})
    if len(comps)!=5:raise AssertionError("all 5 composition folds required")
    results={}
    for policy,rows in per.items():
        agg=aggregate([sc for _,sc,_ in rows])
        deltas={k:float(agg[k]-source[k]) for k in SCORES}
        bycomp={}
        for c in comps:
            b=aggregate([sc for co,sc in native if co==c])
            now=aggregate([sc for co,sc,_ in rows if co==c])
            bycomp[c]={k:float(now[k]-b[k]) for k in SCORES}
        no_regression=(all(x>=-1e-12 for x in deltas.values()) and
                       all(all(x>=-1e-12 for x in v.values()) for v in bycomp.values()))
        improvement=any(x>1e-12 for x in deltas.values())
        results[policy]=dict(metrics={k:float(agg[k]) for k in SCORES},
            counts={k:int(agg[k]) for k in COUNT_KEYS},delta=deltas,
            by_composition_delta=bycomp,frames_blocked=int(blocked[policy]),
            synthetic_onsets=sum(int(z["new_onsets"]) for _,_,z in rows),
            six_metric_no_regression_fit=no_regression,
            fit_exploratory_eligible=bool(no_regression and improvement))
    return dict(schema=SCHEMA,complete=True,dataset="GuitarSet 80 real FIT OOF MIDI256",
        source_runs={"S800":39352880035,"S801":39354733640,"S802":39359182007},
        source_counts={k:int(source[k]) for k in COUNT_KEYS},
        baseline_fit={k:float(source[k]) for k in SCORES},
        tracks=80,compositions=comps,policies=list(POLICIES),
        retrospection_only_auc=aucs,policy_results=results,
        threshold_search_performed=False,feature_oracle_in_decoder=False,
        heads_dynamic_per_frame_pitch=True,independent_frozen_heads=5,
        native_decoder_unchanged=True,production_LIVE_modified=False,
        synthetic_future_lookahead_samples=0,extra_hold_samples=0,
        CAL_opened=False,HELD_opened=False,TEST_opened=False,
        physical_live_latency_measured=False,official_reference="S708",
        official_promotion=False,
        fit_provisional_eligible=[p for p in POLICIES
                              if results[p]["fit_exploratory_eligible"]])

def main():
    p=argparse.ArgumentParser()
    for key in ("manifest","traces","s801","s802","out"):
        p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable S803 output exists")
    result=evaluate(a.manifest,a.traces,a.s801,a.s802)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                     encoding="utf-8")
    print("S803_REAL_FIT80",json.dumps(dict(candidates=result["fit_provisional_eligible"],
       scores=result["policy_results"],source=result["baseline_fit"])),flush=True)

if __name__=="__main__":main()
