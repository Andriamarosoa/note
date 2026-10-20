"""S801 real FIT80 ablations on frozen S800 OOF probability snapshots.

NO training. NO CAL/HELD/TEST. NO delayed source or synthetic NoteOff.
The original S795 decoder and S797 per-(frame,pitch) head router are reused,
and each policy is scored on full emitted MIDI256 with all six S704 metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from causal_note.s795_activek_experts import HEADS,FEATURES,PITCHES,decode
from causal_note.s797_fragment_router import route_probabilities,CUT
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s795_train_activek_midi256 import COUNT_KEYS
from scripts.s800_oof_trace import SCHEMA as TRACE_SCHEMA,verify_snapshot

POLICIES=(
    "identity",
    "consensus",
    "harmonic_veto",
    "poly_gap",
    "adaptive",
    "onset_native_guard",
    "offset_birth_guard",
    "harmonic_parent_guard",
)
NEW_POLICY_BASE={
    "onset_native_guard":"adaptive",
    "offset_birth_guard":"adaptive",
    "harmonic_parent_guard":"consensus",
}
# Predeclared fixed ablations. No candidate threshold search, no CAL tuning.
def route(X,P,policy):
    if policy not in POLICIES:
        raise ValueError("undeclared S801 ablation")
    X=np.asarray(X,dtype=np.float32)
    if X.ndim!=3 or X.shape[1:]!=(PITCHES,FEATURES):
        raise ValueError("invalid signed causal feature ABI")
    base=NEW_POLICY_BASE.get(policy,policy)
    routed,masks,meta=route_probabilities(X,P,base)
    if policy in NEW_POLICY_BASE:
        if policy=="onset_native_guard":
            # A supplemental attack must be supported by current native
            # onset and positive onset rise (no future evidence).
            accept=(X[:,:,0]>=.20)&(X[:,:,5]>=.065)
        elif policy=="offset_birth_guard":
            # H4: prevent short-lived extras at creation, do not delay
            # a synthetic NoteOff or change the S795 release policy.
            accept=(X[:,:,1]>=.55)&(X[:,:,2]<=.35)
        else:
            # H2: harmonic parent may be strong while true played octave
            # still has independent current native onset evidence.
            accept=(X[:,:,10]<=.30)|(X[:,:,5]>=.20)|(X[:,:,0]>=.35)
        masks=np.where(accept,masks,0).astype(np.uint16)
        routed={h:np.where(accept,routed[h],0).astype(np.float32)
                for h in HEADS}
        meta=dict(meta,ablation=policy,blocked_by_extra_guard=int(
            np.count_nonzero(~accept & (np.asarray(masks)==0))),no_future_input=True)
    return routed,masks,meta

def source_signature(events):
    return [(e["type"],int(e["pitch"]),int(e["available_sample"]),
             int(e["decision_sample"])) for e in events]

def evaluated_row(manifest,trace_dir,entry,policy):
    info=verify_snapshot(entry,trace_dir)
    with np.load(Path(trace_dir)/f"fit_{entry['index']:03d}.npz",
                 allow_pickle=False) as z:
        X=z["features"].astype(np.float32)
        d=z["decision_ends"]
        ends=z["metric_ends"]
        P={h:z["head_"+h].astype(np.float32) for h in HEADS}
    truth=info["fit_truth"]
    originals=info["source_events"]
    source=metric(originals,truth,ends)
    if any(int(source[k])!=int(info["source_counts"][k]) for k in COUNT_KEYS):
        raise AssertionError("S801 original source score differs from S800 trace")
    probabilities,masks,diagnostics=route(X,P,policy)
    if policy=="identity":
        out,changes=decode(originals,d,X,probabilities,0,0.,int(info["samples"]))
    else:
        out,changes=decode(originals,d,X,probabilities,31,CUT,int(info["samples"]))
    score=metric(out,truth,ends)
    if changes["max_added_native_delay_samples"] or changes["extra_lookahead_samples"] \
            or changes["extra_hold_samples"] or changes["extra_confirmation_samples"]:
        raise AssertionError("S801 synthetic or native event timing was held")
    if policy=="identity" and source_signature(out)!=source_signature(originals):
        raise AssertionError("S801 source identity semantic stream changed")
    by_reason=Counter(e.get("reason","") for e in out
                     if e.get("reason","").startswith("s795_"))
    return dict(index=entry["index"],name=info["name"],
        composition=info["composition"],texture=info["texture"],
        policy=policy,metrics={k:float(score[k]) for k in SCORES},
        metric_counts={k:int(score[k]) for k in COUNT_KEYS},
        raw=score,fragment_masks=diagnostics["fragment_masks"],
        route_allowed=int(diagnostics["allowed"]),
        new_onsets=int(changes["new_onsets"]),
        new_offsets=int(changes["new_offsets"]),
        added_release_reasons=dict(by_reason),
        extra_latency_samples=int(changes["max_added_native_delay_samples"]))

def safe_all_six(baseline,candidate):
    return all(candidate[k]>=baseline[k]-1e-12 for k in SCORES)

def analyze(manifest_path,trace_dir):
    m=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not m.get("complete") or m.get("schema")!=TRACE_SCHEMA:
        raise AssertionError("S800 FIT80 complete signed manifest required")
    if not m.get("FIT_source_parity_S795") or len(m.get("fit_files",[]))!=80:
        raise AssertionError("S800 S795 FIT signed parity required")
    if any(m.get(key) for key in (
        "CAL_opened","HELD_opened","TEST_opened",
        "cal_used_for_training","cal_used_for_selection",
        "production_LIVE_modified","promoted")):
        raise AssertionError("S800 forbidden evaluation split or LIVE mutation")
    by_policy={policy:[] for policy in POLICIES}
    for entry in m["fit_files"]:
        for policy in POLICIES:
            row=evaluated_row(m,trace_dir,entry,policy)
            by_policy[policy].append(row)
        print("S801_FIT_EVALUATED",entry["index"],entry["name"],
              "policies",len(POLICIES),flush=True)
    baseline=aggregate([z["raw"] for z in by_policy["identity"]])
    for k in COUNT_KEYS:
        if int(baseline[k])!=int(m["S795_archived_fit_counts"][k]):
            raise AssertionError("S801 FIT identity count drift: "+k)
    for k in SCORES:
        if abs(baseline[k]-m["FIT_source_scores"][k])>1e-12:
            raise AssertionError("S801 FIT identity metric drift: "+k)
    trials={}
    for policy in POLICIES:
        rows=by_policy[policy]
        stats=aggregate([r["raw"] for r in rows])
        comps={}
        for composition in m["folds_done"]:
            part=[r["raw"] for r in rows if r["composition"]==composition]
            bpart=[r["raw"] for r in by_policy["identity"]
                   if r["composition"]==composition]
            curr,source=aggregate(part),aggregate(bpart)
            comps[composition]=dict(scores={k:curr[k] for k in SCORES},
                                    deltas={k:curr[k]-source[k] for k in SCORES})
        pass_six=safe_all_six(baseline,stats)
        # Must improve at least one target with no metric regression
        # at aggregate or composition level; CAL is still sealed.
        improvement=any(stats[k]>baseline[k]+1e-12 for k in SCORES)
        comp_safe=all(all(z>=-1e-12 for z in x["deltas"].values())
                      for x in comps.values())
        trials[policy]=dict(metrics={k:stats[k] for k in SCORES},
            delta={k:stats[k]-baseline[k] for k in SCORES},
            counts={k:int(stats[k]) for k in COUNT_KEYS},
            six_metric_nonregression_fit=pass_six,
            per_composition_nonregression_fit=comp_safe,
            eligible_fit_only=(pass_six and comp_safe and improvement),
            aggregate_new_onsets=sum(r["new_onsets"] for r in rows),
            aggregate_new_offsets=sum(r["new_offsets"] for r in rows),
            added_offset_reason_totals=dict(sum((Counter(r["added_release_reasons"])
                for r in rows),Counter())),
            distinct_per_fragment_masks=sorted(set(
                label for r in rows for label in r["fragment_masks"])),
            by_composition=comps)
    accepted=[k for k in POLICIES if trials[k]["eligible_fit_only"]]
    chosen=(max(accepted,key=lambda k:(min(trials[k]["delta"].values()),
                                      sum(trials[k]["delta"].values())))
            if accepted else "identity")
    return dict(schema="S801_FIT80_CAUSAL_DYNAMIC_ABLATION_V1",
       completed=True,scientific_evidence="GuitarSet FIT80 real OOF MIDI256",
       source="S800 signed FIT80 fragments",CAL_opened=False,
       HELD_opened=False,TEST_opened=False,
       models_retrained=False,heads_per_fragment_variable=True,
       production_LIVE_modified=False,latency_physical_measured=False,
       used_future_frames=False,extra_hold_samples=0,
       promoted=False,
       baseline_fit={k:baseline[k] for k in SCORES},
       signed_source_counts={k:int(baseline[k]) for k in COUNT_KEYS},
       policy_results=trials,provisional_fit_candidate=chosen,
       eligible_nonidentity_candidates=accepted,
       cal_policy_selection_performed=False,
       cal_evaluation_performed=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("S801 result immutable")
    result=analyze(a.manifest,a.traces)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S801_REAL_FIT80_RESULT",json.dumps({
        "candidate":result["provisional_fit_candidate"],
        "admissible":result["eligible_nonidentity_candidates"],
        "identity":result["baseline_fit"],"promoted":False
    }),flush=True)

if __name__=="__main__":main()
