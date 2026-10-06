"""Stage-1 domain-shift audit: internal OOF K2/K3 vs exposed outer fold 3.

Questions:
- Does pre_high keep the same K3-vs-K2 direction on outer fold 3?
- Does amp_range_low keep the same direction?
- Which component makes the pre_high + amp_range_low intersection flip from
  +24 internal gain to -6 outer gain?

Thresholds are frozen as medians over the 233 internal K2/K3 cases, exactly as in
the exploratory outer evaluation. No tuning. No prediction changes.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_stage2_exclusivity_failure import features
from scripts.v273_window_experiment import load_bundle
from scripts.v273_residual_audit import require

GROUPS=("K2_corrected","K3_regressed")

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def style(member):
    m=re.match(r"^\d{2}_([A-Za-z]+)\d+-",Path(member).name)
    return m.group(1) if m else "UNKNOWN"

def collect_audio_features(dataset,items):
    wanted={m for _,m,_ in items}
    tracks={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for key,m,start in items:
        by[m].append((key,start))
    out={}
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for key,start in by[member]:
            q=features(samples,int(start)); require(q is not None,"feature missing")
            out[key]=q
        print(json.dumps({"recording":member,"done":len(out)}),flush=True)
    return out

def med(x): return float(np.median(np.asarray(x,float)))

def auc_oriented(y,x,direction):
    x=np.asarray(x,float);y=np.asarray(y,int)
    s=x if direction=="high" else -x
    return float(roc_auc_score(y,s))

def signal_stats(rows,key,direction,threshold):
    k2=[r for r in rows if r["y"]==2]
    k3=[r for r in rows if r["y"]==3]
    def hit(r):
        v=r["f"][key]
        return v>=threshold if direction=="high" else v<=threshold
    h2=sum(hit(r) for r in k2); h3=sum(hit(r) for r in k3)
    yy=np.asarray([1 if r["y"]==3 else 0 for r in rows],int)
    xx=np.asarray([r["f"][key] for r in rows],float)
    return {
      "threshold":float(threshold),"direction":direction,
      "K2_n":len(k2),"K3_n":len(k3),
      "K2_median":med([r["f"][key] for r in k2]),
      "K3_median":med([r["f"][key] for r in k3]),
      "K2_hit":int(h2),"K3_hit":int(h3),
      "K2_hit_rate":h2/max(1,len(k2)),"K3_hit_rate":h3/max(1,len(k3)),
      "veto_gain":int(h3-h2),
      "auc_oriented":auc_oriented(yy,xx,direction),
    }

def combo_stats(rows,pre_thr,amp_thr):
    out={"K2":0,"K3":0,"other":0,"n":0}
    cells={a:{b:{"K2":0,"K3":0} for b in ("amp_high","amp_low")} for a in ("pre_low","pre_high")}
    by_style={}
    for r in rows:
        pre_high=r["f"]["pre"]>=pre_thr
        amp_low=r["f"]["amp_range"]<=amp_thr
        pa="pre_high" if pre_high else "pre_low"; aa="amp_low" if amp_low else "amp_high"
        if r["y"] in (2,3):
            cells[pa][aa]["K3" if r["y"]==3 else "K2"]+=1
        if pre_high and amp_low:
            out["n"]+=1
            if r["y"]==2: out["K2"]+=1
            elif r["y"]==3: out["K3"]+=1
            else: out["other"]+=1
            st=r.get("style","UNKNOWN")
            q=by_style.setdefault(st,{"K2":0,"K3":0,"other":0})
            if r["y"]==2:q["K2"]+=1
            elif r["y"]==3:q["K3"]+=1
            else:q["other"]+=1
    out["gain"]=out["K3"]-out["K2"]
    for st,q in by_style.items(): q["gain"]=q["K3"]-q["K2"]
    return out,cells,by_style

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-cases","outer-predictions","bundle","config","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"overwrite")

    cases=[json.loads(s) for s in a.internal_cases.read_text().splitlines() if s.strip()]
    cases=[r for r in cases if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in cases)==108,"internal K2 drift")
    require(sum(r["group"]=="K3_regressed" for r in cases)==125,"internal K3 drift")

    internal_items=[(i,str(r["recording_id"]),int(r["start_sample"])) for i,r in enumerate(cases)]
    int_feat=collect_audio_features(a.dataset,internal_items)
    internal=[]
    for i,r in enumerate(cases):
        internal.append({"id":i,"y":2 if r["group"]=="K2_corrected" else 3,
                         "fold":int(r["fold"]),"member":str(r["recording_id"]),
                         "style":style(str(r["recording_id"])),"f":int_feat[i]})

    pre_thr=med([r["f"]["pre"] for r in internal])
    amp_thr=med([r["f"]["amp_range"] for r in internal])

    cache,_,_=load_bundle(a.bundle,a.config)
    outer=load_npz(a.outer_predictions)
    gids=np.asarray(outer["global_index"],int)
    source=np.asarray(outer["source_action_mask"],bool)
    y=np.asarray(outer["k"],int)
    require(len(gids)==len(source)==len(y),"outer length drift")
    source_ids=gids[source]; source_y=y[source]
    items=[(int(gid),str(cache["members"][gid]),int(cache["cluster_start_samples"][gid])) for gid in source_ids]
    out_feat=collect_audio_features(a.dataset,items)
    outer_rows=[]
    for gid,yy in zip(source_ids,source_y):
        mem=str(cache["members"][gid])
        outer_rows.append({"id":int(gid),"y":int(yy),"member":mem,"style":style(mem),"f":out_feat[int(gid)]})
    require(sum(r["y"]==2 for r in outer_rows)==54,"outer K2 drift")
    require(sum(r["y"]==3 for r in outer_rows)==40,"outer K3 drift")

    internal_stats={
      "pre":signal_stats(internal,"pre","high",pre_thr),
      "amp_range":signal_stats(internal,"amp_range","low",amp_thr),
    }
    outer_k23=[r for r in outer_rows if r["y"] in (2,3)]
    outer_stats={
      "pre":signal_stats(outer_k23,"pre","high",pre_thr),
      "amp_range":signal_stats(outer_k23,"amp_range","low",amp_thr),
    }
    int_combo,int_cells,int_style=combo_stats(internal,pre_thr,amp_thr)
    out_combo,out_cells,out_style=combo_stats(outer_rows,pre_thr,amp_thr)

    rep={
      "status":"completed","experiment":"v273_stage1_domain_shift",
      "thresholds":{"pre_high":pre_thr,"amp_range_low":amp_thr},
      "internal":{"signals":internal_stats,"intersection":int_combo,"cells":int_cells,"by_style":int_style},
      "outer_fold3":{"signals":outer_stats,"intersection":out_combo,"cells":out_cells,"by_style":out_style},
      "domain_shift":{
        "pre_auc_delta_outer_minus_internal":outer_stats["pre"]["auc_oriented"]-internal_stats["pre"]["auc_oriented"],
        "amp_auc_delta_outer_minus_internal":outer_stats["amp_range"]["auc_oriented"]-internal_stats["amp_range"]["auc_oriented"],
        "pre_gain_delta":outer_stats["pre"]["veto_gain"]-internal_stats["pre"]["veto_gain"],
        "amp_gain_delta":outer_stats["amp_range"]["veto_gain"]-internal_stats["amp_range"]["veto_gain"],
        "intersection_gain_delta":out_combo["gain"]-int_combo["gain"],
      },
      "outer_fold_3_fresh_independent":False,"prediction_changes":False,"threshold_tuning":False
    }
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Stage-1 domain-shift audit","",
           "Frozen internal-median thresholds; no outer tuning.","",
           "| signal | internal AUC | outer AUC | internal veto gain | outer veto gain |",
           "|---|---:|---:|---:|---:|"]
    for name in ("pre","amp_range"):
        i=internal_stats[name];o=outer_stats[name]
        lines.append(f"| {name} | {i['auc_oriented']:.3f} | {o['auc_oriented']:.3f} | {i['veto_gain']:+d} | {o['veto_gain']:+d} |")
    lines+=["",
            f"Intersection internal: **{int_combo['K3']} K3 / {int_combo['K2']} K2 = {int_combo['gain']:+d}**.",
            f"Intersection outer fold3: **{out_combo['K3']} K3 / {out_combo['K2']} K2 = {out_combo['gain']:+d}**.","",
            "## Outer intersection by style","",
            "| style | K3 blocked | K2 blocked | gain |","|---|---:|---:|---:|"]
    for st,q in sorted(out_style.items()):
        lines.append(f"| {st} | {q['K3']} | {q['K2']} | {q['gain']:+d} |")
    lines+=["","Fold 3 is historically exposed; diagnostic only."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__": main()
