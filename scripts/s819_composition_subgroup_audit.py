"""S819 independent composition / recording subgroup audit of two immutable CI reports.

Input report.json from S819 run 39553290558 + S817 run 39543749810;
never trains, selects, changes source MIDI, or opens CAL32/HELD/TEST.
Original research FIT80 split is historically explored, so this is a
descriptive audit only, not new independent validation.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from collections import defaultdict
from pathlib import Path

COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
POLICY="max_high@0.70"
def totals(rows,key):
    return sum(int(t[key]) for t in rows)
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(gate,prior,*,prior_sha):
    if gate.get("schema")!="S819_S817_FROZEN_HIGH_POLYPHONY_GATE_FORENSIC_FIT80_V1" or not gate.get("complete"):
        raise ValueError("signed completed S819 real-audio gating experiment needed")
    if prior.get("schema")!="S817_REAL_FIT80_FIVE_PRIVATE_ACTIVEK_HEADS_MIDI256_V1" or not prior.get("complete"):
        raise ValueError("completed five-fold source S817 needed")
    if gate["source_817_report_sha256"]!=prior_sha:
        raise AssertionError("S817 report original SHA changed")
    if len(gate["tracks"])!=len(prior["tracks"])!=80:
        raise AssertionError("not exactly 80 signed recordings")
    old={(r["index"],r["name"],r["composition"]):r for r in prior["tracks"]}
    if len(old)!=80:
        raise AssertionError("duplicate S817 source")
    if len(gate["folds"])!=5 or set(gate["folds"])!=set(COMPS):
        raise AssertionError("missing composition")
    out=dict(schema="S819_COMPOSITION_SUBGROUP_REAL_FORensics_V1",
         s819_original_run=39553290558,s817_original_run=39543749810,
         prior_s817_sha256=prior_sha,analyzed_policy=POLICY,
         new_training=False,independent_held_out_validation=False,
         selecting_on_outer_scores_permitted=False,production_live_changed=False,
         promotion=False,compositions={},recording_groups={},styles={},
         GT_K_base_rates={})
    total_delta=0;total_modified=0
    for comp in COMPS:
        rows=[r for r in gate["tracks"] if r["composition"]==comp]
        if len(rows)!=16:
            raise AssertionError("not 16 held-out composition tracks")
        delta=sum(r["by_policy"][POLICY]["correct"]-r["original_s815"]["correct"] for r in rows)
        over=sum(r["by_policy"][POLICY]["overcount"]-r["original_s815"]["overcount"] for r in rows)
        under=sum(r["by_policy"][POLICY]["undercount"]-r["original_s815"]["undercount"] for r in rows)
        modified=sum(r["by_policy"][POLICY]["changed_frames"] for r in rows)
        total_delta+=delta;total_modified+=modified
        if delta!=-over-under:
            raise AssertionError("correct/under/over count conservation failed")
        fold=gate["folds"][comp]
        if fold["potential_changed_frames_by_policy"][POLICY]!=modified:
            raise AssertionError("changed decision totals disagree")
        if fold["actual_scores_by_fixed_policy"][POLICY]["correct"] - fold["actual_scores_by_fixed_policy"]["base@0.50"]["correct"]!=delta:
            raise AssertionError("composition counterfactual score mismatch")
        by_k={}
        denominator=totals([r["original_s815"] for r in rows],"frames")
        for k in range(1,7):
            kk=[old[(r["index"],r["name"],r["composition"])]["by_true_k"][str(k)] for r in rows]
            n=totals(kk,"frames");correct=totals(kk,"correct")
            by_k[str(k)]=dict(frames=n,accurate=correct,accuracy=correct/max(n,1))
        if sum(v["frames"] for v in by_k.values())!=denominator:
            raise AssertionError("S817 by-K GT ground-truth denominator differs")
        out["GT_K_base_rates"][comp]=by_k
        out["compositions"][comp]=dict(n_tracks=16,active_frames=denominator,
              changed_audio_decisions=modified,delta_correct=delta,
              delta_overcount=over,delta_undercount=under,
              actual_gt_K_ge4=sum(by_k[str(k)]["frames"] for k in (4,5,6)),
              actual_gt_K_ge4_fraction=sum(by_k[str(k)]["frames"] for k in (4,5,6))/denominator)
        groups=defaultdict(lambda:dict(n_tracks=0,delta_correct=0,
            changed_audio_decisions=0,delta_overcount=0,delta_undercount=0))
        styles=defaultdict(lambda:dict(n_tracks=0,delta_correct=0,
            changed_audio_decisions=0,delta_overcount=0,delta_undercount=0))
        for r in rows:
            m=re.fullmatch(r"\d\d_(.+)_(comp|solo)",r["name"])
            if not m:
                raise AssertionError("unexpected recording name; cannot group")
            group,style=m.groups()
            b=r["original_s815"];v=r["by_policy"][POLICY]
            for key in (group,style):
                target=groups[key] if key==group else styles[key]
                target["n_tracks"]+=1
                target["delta_correct"]+=v["correct"]-b["correct"]
                target["changed_audio_decisions"]+=v["changed_frames"]
                target["delta_overcount"]+=v["overcount"]-b["overcount"]
                target["delta_undercount"]+=v["undercount"]-b["undercount"]
        if sum(v["delta_correct"] for v in groups.values())!=delta or sum(v["delta_correct"] for v in styles.values())!=delta:
            raise AssertionError("group/style delta does not reconcile")
        out["recording_groups"][comp]=dict(sorted(groups.items()))
        out["styles"][comp]=dict(sorted(styles.items()))
    if total_delta!=-683 or total_modified!=4867:
        raise AssertionError("original S819 known score drift")
    if out["compositions"]["SS1"]["delta_correct"]!=-558:
        raise AssertionError("SS1 original failure changed")
    out["all_compositions"]=dict(delta_correct=total_delta,
        changed_audio_decisions=total_modified)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s819-report",required=True,type=Path)
    p.add_argument("--s817-report",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    out=summarize(read(a.s819_report),read(a.s817_report),
                  prior_sha=sha(a.s817_report))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print("S819_DEEP_COMPOSITION_AUDIT",out["all_compositions"],flush=True)
    for k,v in out["compositions"].items():
        print("S819_COMP",k,v,"GROUPS",out["recording_groups"][k],
              "STYLES",out["styles"][k],flush=True)
if __name__=="__main__":
    main()
