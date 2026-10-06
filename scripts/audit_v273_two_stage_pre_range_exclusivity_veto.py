"""Two-stage strict OOF veto: pre+range first, exclusivity-range second.

Stage 1 is the completed strict guard from run 37532627926:
  pre_high AND amp_range_low.

Stage 2 is predeclared from run 37532931017 as the strongest stable annotation-free
residual signature after stage 1:
  selected-F0 shared-attack-fraction range HIGH.

For each held-out fold, both stage thresholds are medians learned from the other
internal folds only. Stage 2 is fit only on stage-1 survivors in the fit folds.
No numeric threshold search. Fold 3 excluded. No automatic promotion.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_oof_source_onset_contrast_guard import build_oof,account
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum,f0_pool,template,fit_best
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.audit_v273_exclusive_harmonic_support import attack_spectrum,support_against_set
from scripts.v273_residual_audit import FOLDS,require

EPS=1e-12

def style(member):
    m=re.match(r"^\d{2}_([A-Za-z]+)\d+-",Path(member).name);require(m is not None,"bad style")
    return m.group(1)

def features(samples,start):
    freq,x=transition_spectrum(samples,start)
    pool=f0_pool(freq,x)
    if len(pool)<3:return None
    D=np.column_stack([template(freq,f) for f in pool])
    trip,_,_,_=fit_best(D,x,3)
    _,combo,amps=trip
    f0=np.asarray([pool[i] for i in combo],float); amps=np.asarray(amps,float)
    if len(f0)!=3 or len(amps)!=3:return None
    rf,powers,_=raw_powers(samples,start)
    nov=[component_metrics(rf,powers,float(q)) for q in f0]
    attack=attack_spectrum(powers)
    shared=[]
    for i,q in enumerate(f0):
        peers=[float(f0[j]) for j in range(3) if j!=i]
        shared.append(float(support_against_set(rf,attack,float(q),peers)["shared_attack_fraction"]))
    return {
      "pre":float(np.median([q["pre_norm"] for q in nov])),
      "amp_range":float(np.ptp(amps)),
      "exc_shared_range":float(np.ptp(np.asarray(shared,float))),
    }

def median(vals):
    require(len(vals)>=20,"small median fit")
    return float(np.median(np.asarray(vals,float)))

def summarize(rows,mask):
    y=np.asarray([r["y"] for r in rows],int);m=np.asarray(mask,bool)
    corr=int(np.sum(m&(y==2)));reg=int(np.sum(m&(y==3)));other=int(np.sum(m&~np.isin(y,(2,3))))
    return {"applied":int(m.sum()),"corrections":corr,"regressions":reg,"other_k_actions":other,"net":corr-reg}

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"overwrite")
    action_inputs(a.exports);rows_map=build_oof(a.exports)
    wanted={r["member"] for r in rows_map.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for rid,r in rows_map.items():
        r["style"]=style(r["member"])
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows_map[rid]["features"]=features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows_map.values())}),flush=True)
    rows=[r for r in rows_map.values() if r["source_apply"]]
    require(len(rows)==427,"action drift")
    require(all(r.get("features") is not None for r in rows),"feature missing")

    s1=np.zeros(len(rows),bool);s2=np.zeros(len(rows),bool)
    thresholds={}
    for f in FOLDS:
        fit_idx=[i for i,r in enumerate(rows) if r["fold"]!=f and r["y"] in (2,3)]
        val_idx=[i for i,r in enumerate(rows) if r["fold"]==f]
        pre_thr=median([rows[i]["features"]["pre"] for i in fit_idx])
        amp_thr=median([rows[i]["features"]["amp_range"] for i in fit_idx])
        fit_survive=[i for i in fit_idx if not (rows[i]["features"]["pre"]>=pre_thr and rows[i]["features"]["amp_range"]<=amp_thr)]
        exc_thr=median([rows[i]["features"]["exc_shared_range"] for i in fit_survive])
        thresholds[str(f)]={"pre_high":pre_thr,"amp_range_low":amp_thr,"exc_shared_range_high":exc_thr,
                            "stage2_fit_survivors":len(fit_survive)}
        for i in val_idx:
            q=rows[i]["features"]
            s1[i]=q["pre"]>=pre_thr and q["amp_range"]<=amp_thr
            if not s1[i]:
                s2[i]=q["exc_shared_range"]>=exc_thr

    keep1=~s1; keep2=~(s1|s2)
    source=summarize(rows,np.ones(len(rows),bool))
    after1=summarize(rows,keep1); after2=summarize(rows,keep2)
    block1=summarize(rows,s1);block2=summarize(rows,s2)
    require((block1["corrections"],block1["regressions"])==(24,48),f"stage1 drift {block1}")

    per=[]
    for f in FOLDS:
        m=np.asarray([r["fold"]==f for r in rows],bool)
        q1=summarize(rows,s1&m);q2=summarize(rows,s2&m);final=summarize(rows,keep2&m)
        per.append({"fold":int(f),"stage1_block":q1,"stage2_block":q2,"final":final,
                    "stage2_gain":q2["regressions"]-q2["corrections"]})

    style_rows=[]
    for st in sorted({r["style"] for r in rows}):
        m=np.asarray([r["style"]==st for r in rows],bool)
        q2=summarize(rows,s2&m);final=summarize(rows,keep2&m)
        style_rows.append({"style":st,"stage2_block":q2,"final":final,
                           "stage2_gain":q2["regressions"]-q2["corrections"]})

    report={"status":"completed","experiment":"v273_two_stage_pre_range_exclusivity_veto",
            "source":source,"stage1_block":block1,"after_stage1":after1,
            "stage2_block":block2,"after_stage2":after2,
            "stage2_incremental_gain":block2["regressions"]-block2["corrections"],
            "total_gain_vs_source":after2["net"]-source["net"],
            "thresholds":thresholds,"per_fold":per,"by_style":style_rows,
            "strict_stage2_pass":{
              "incremental_gain_positive":block2["regressions"]>block2["corrections"],
              "all_folds_incremental_nonnegative":all(q["stage2_gain"]>=0 for q in per),
              "all_styles_incremental_nonnegative":all(q["stage2_gain"]>=0 for q in style_rows),
            },
            "outer_fold_3_used":False,"prediction_changes":False,"posthoc_internal_reuse":True,"automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Two-stage pre+range → exclusivity-range veto","",
           f"Source net: **{source['net']:+d}**.",
           f"After stage 1: **{after1['corrections']} corr / {after1['regressions']} reg**, net **{after1['net']:+d}**.",
           f"Stage 2 blocks **{block2['regressions']} regressions / {block2['corrections']} corrections**, incremental gain **{report['stage2_incremental_gain']:+d}**.",
           f"After stage 2: **{after2['corrections']} corr / {after2['regressions']} reg**, net **{after2['net']:+d}**.","",
           "## Stage 2 by fold","",
           "| fold | blocked K3 | blocked K2 | incremental gain | final net |","|---:|---:|---:|---:|---:|"]
    for q in per:
        b=q["stage2_block"];lines.append(f"| {q['fold']} | {b['regressions']} | {b['corrections']} | {q['stage2_gain']:+d} | {q['final']['net']:+d} |")
    lines+=["","## Stage 2 by style","","| style | blocked K3 | blocked K2 | gain | final net |","|---|---:|---:|---:|---:|"]
    for q in style_rows:
        b=q["stage2_block"];lines.append(f"| {q['style']} | {b['regressions']} | {b['corrections']} | {q['stage2_gain']:+d} | {q['final']['net']:+d} |")
    lines+=["",f"Strict stage-2 pass: **{report['strict_stage2_pass']}**.","","Internal/post-hoc only; no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
