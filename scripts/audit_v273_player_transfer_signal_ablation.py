"""Player-transfer ablation for the three stable veto signals.

No numeric threshold search. For every held-out player and held-out fold, thresholds
are medians from other players and other folds, using the K2/K3 source-action
training population. We compare the three fixed 2-of-3 pairs and the original trio.

Also reports player-03 per-signal hit rates and class-conditional medians to explain
the only negative player transfer observed in run 37532036886.

Post-hoc internal audit only. Fold 3 excluded. No prediction changes.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_oof_source_onset_contrast_guard import build_oof
from scripts.audit_v273_log_frequency_guard import action_inputs
from scripts.audit_v273_full_oof_stable_signature_veto import extract_features,hit
from scripts.v273_residual_audit import FOLDS,require

SIGNALS={
 "pre_high":("nov_pre_norm_median","high"),
 "amp_range_low":("amp_range","low"),
 "amp_cv_low":("amp_cv","low"),
}
PATTERNS={
 "pre_plus_range":("pre_high","amp_range_low"),
 "pre_plus_cv":("pre_high","amp_cv_low"),
 "range_plus_cv":("amp_range_low","amp_cv_low"),
 "trio":("pre_high","amp_range_low","amp_cv_low"),
}

def player(member):
    m=re.match(r"^(\d{2})_",Path(member).name); require(m is not None,"bad member"); return m.group(1)

def thresholds(rows):
    out={}
    for alias,(feature,direction) in SIGNALS.items():
        vals=[r["features"][feature] for r in rows if r.get("features") is not None and r["y"] in (2,3)]
        require(len(vals)>=20,f"small fit {alias}: {len(vals)}")
        out[alias]={"feature":feature,"direction":direction,"threshold":float(np.median(vals))}
    return out

def account(rows,veto):
    k3=k2=other=0
    for r,v in zip(rows,veto):
        if not v:continue
        if r["y"]==3:k3+=1
        elif r["y"]==2:k2+=1
        else:other+=1
    return {"blocked_reg":k3,"blocked_corr":k2,"blocked_other":other,
            "gain":k3-k2,"precision":None if k3+k2==0 else k3/(k3+k2)}

def main():
    p=argparse.ArgumentParser()
    for n in ("exports","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"overwrite")
    action_inputs(a.exports); rows_map=build_oof(a.exports)
    wanted={r["member"] for r in rows_map.values()}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for rid,r in rows_map.items():
        r["player"]=player(r["member"])
        if r["source_apply"]:by[r["member"]].append((rid,r["start"]))
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid,start in by[member]:rows_map[rid]["features"]=extract_features(samples,start)
        print(json.dumps({"recording":member,"done":sum(r.get("features") is not None for r in rows_map.values())}),flush=True)

    rows=[r for r in rows_map.values() if r["source_apply"]]
    require(sum(r["y"]==2 for r in rows)==108 and sum(r["y"]==3 for r in rows)==125,"k23 drift")

    players=sorted({r["player"] for r in rows})
    thresholds_by_player_fold={}
    for pl in players:
        thresholds_by_player_fold[pl]={}
        for f in FOLDS:
            fit=[r for r in rows if r["player"]!=pl and r["fold"]!=f and r.get("features") is not None and r["y"] in (2,3)]
            thresholds_by_player_fold[pl][str(f)]=thresholds(fit)

    pattern_reports={}
    for pname,pat in PATTERNS.items():
        per=[];all_eval=[];all_veto=[]
        for pl in players:
            held=[r for r in rows if r["player"]==pl]
            vv=[r.get("features") is not None and hit(r["features"],thresholds_by_player_fold[pl][str(r["fold"])],pat) for r in held]
            q=account(held,vv);q["player"]=pl;per.append(q)
            all_eval.extend(held);all_veto.extend(vv)
        total=account(all_eval,all_veto)
        pattern_reports[pname]={"pattern":list(pat),"per_player":per,"total":total,
                                "min_player_gain":min(q["gain"] for q in per),
                                "positive_players":sum(q["gain"]>0 for q in per),
                                "nonnegative_players":sum(q["gain"]>=0 for q in per),
                                "all_players_nonnegative":all(q["gain"]>=0 for q in per)}

    # Explain player 03 signal-by-signal under the exact leave-player+fold thresholds.
    p03=[r for r in rows if r["player"]=="03" and r["y"] in (2,3)]
    sig={}
    for alias,(feature,direction) in SIGNALS.items():
        hit_k2=hit_k3=n2=n3=0
        vals03={2:[],3:[]}; vals_other={2:[],3:[]}
        for r in rows:
            if r["y"] not in (2,3) or r.get("features") is None:continue
            (vals03 if r["player"]=="03" else vals_other)[r["y"]].append(r["features"][feature])
        for r in p03:
            spec=thresholds_by_player_fold["03"][str(r["fold"])][alias]
            h=hit(r["features"],{alias:spec},(alias,))
            if r["y"]==2:n2+=1;hit_k2+=int(h)
            else:n3+=1;hit_k3+=int(h)
        sig[alias]={
          "player03_K3_hit_rate":None if n3==0 else hit_k3/n3,
          "player03_K2_hit_rate":None if n2==0 else hit_k2/n2,
          "player03_K3_n":n3,"player03_K2_n":n2,
          "median_player03_K3":float(np.median(vals03[3])),"median_other_K3":float(np.median(vals_other[3])),
          "median_player03_K2":float(np.median(vals03[2])),"median_other_K2":float(np.median(vals_other[2])),
        }

    report={"status":"completed","experiment":"v273_player_transfer_signal_ablation",
            "outer_fold_3_used":False,"prediction_changes":False,"posthoc_internal_reuse":True,
            "threshold_rule":"held-out player + held-out fold excluded; median only; fixed signal directions",
            "patterns":pattern_reports,"player03_signal_diagnostic":sig}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Player-transfer stable-signal ablation","",
           "Leave-one-player-out + leave-fold-out median calibration. No numeric threshold search.","",
           "| pattern | blocked K3 | blocked K2 | gain | min player gain | nonnegative players |",
           "|---|---:|---:|---:|---:|---:|"]
    for name,q in pattern_reports.items():
        t=q["total"];lines.append(f"| {name} | {t['blocked_reg']} | {t['blocked_corr']} | {t['gain']:+d} | {q['min_player_gain']:+d} | {q['nonnegative_players']}/5 |")
    lines+=["","## Per-player gains","","| pattern | p00 | p01 | p02 | p03 | p04 |","|---|---:|---:|---:|---:|---:|"]
    for name,q in pattern_reports.items():
        d={z["player"]:z["gain"] for z in q["per_player"]}
        lines.append(f"| {name} | {d['00']:+d} | {d['01']:+d} | {d['02']:+d} | {d['03']:+d} | {d['04']:+d} |")
    lines+=["","## Player 03 signal diagnostic","",
            "| signal | K3 hit rate | K2 hit rate | p03 K3 median | other K3 median | p03 K2 median | other K2 median |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for alias,q in sig.items():
        lines.append(f"| {alias} | {q['player03_K3_hit_rate']:.3f} | {q['player03_K2_hit_rate']:.3f} | {q['median_player03_K3']:.6g} | {q['median_other_K3']:.6g} | {q['median_player03_K2']:.6g} | {q['median_other_K2']:.6g} |")
    lines+=["","Post-hoc internal audit only; no promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
