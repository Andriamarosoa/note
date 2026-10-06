"""Descriptive audit of common signatures in residual corrections vs regressions.

Primary populations are frozen residual-guard actions:
- K2_corrected: true K2, source predicted 3 then correction to 2 was beneficial (108)
- K3_regressed: true K3, source predicted 3 then correction to 2 was harmful (125)

This audit does NOT tune a threshold or modify Exact-K. It ranks descriptive
differences and requires direction consistency across folds 0/1/2/4.

Important control: annotation-centered pick features are summarized by per-note
mean/median, not weakest-of-K, to avoid the mechanical min-of-3 vs min-of-2
order-statistic bias.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_annotation_pick_burst import window, note_burst
from scripts.audit_v273_attack_novelty import raw_powers, component_metrics
from scripts.audit_v273_exclusive_harmonic_support import attack_spectrum, support_against_set
from scripts.audit_v273_selected_f0_attack_exclusive import row_features as selected_f0_features
from scripts.v273_residual_audit import FOLDS, require

GROUPS=("K2_corrected","K3_regressed")
PICK_METRICS=("rms_ratio","diff_rms_ratio","max_rms_ratio","max_diff_ratio","hf_delta","centroid_delta","flatness_delta","post_crest","post_zcr")
NOV_METRICS=("pre_norm","post1_norm","onset_contrast_raw","onset_contrast_norm","log_ratio_raw","late_retention_raw","positive_retained_fraction")
EX_METRICS=("unique_attack_fraction","shared_attack_fraction","attack_energy_total")
EPS=1e-12

def add_num(out, domain, name, value):
    try:
        v=float(value)
    except (TypeError,ValueError):
        return
    if np.isfinite(v):
        out[f"{domain}.{name}"]=v

def agg(vals, prefix, out):
    if not vals:
        return
    keys=vals[0].keys()
    for k in keys:
        a=np.asarray([float(v[k]) for v in vals if k in v and np.isfinite(float(v[k]))],float)
        if not len(a):
            continue
        add_num(out,prefix,k+"_mean",a.mean())
        add_num(out,prefix,k+"_median",np.median(a))

def build_features(row, samples):
    f={}
    # Frozen source/corrector scalars.
    for k in ("global_score","probability_K2_weighted","best_pair_residual_ratio",
              "best_triplet_residual_ratio","median_triplet_f0","candidate_span_ms"):
        if k in row: add_num(f,"source",k,row[k])
    if "best_pair_residual_ratio" in row and "best_triplet_residual_ratio" in row:
        add_num(f,"source","pair_minus_triplet",
                float(row["best_pair_residual_ratio"])-float(row["best_triplet_residual_ratio"]))

    # Already-audited timing / spectral / ownership measurements.
    for k,v in row.get("measurements",{}).items():
        add_num(f,"residual_measure",k,v)

    # Decomposition geometry.
    d=row.get("decomposition",{})
    amps=np.asarray(d.get("triplet_amplitudes",[]),float)
    f0=np.asarray(d.get("triplet_f0",[]),float)
    if len(amps):
        for n,v in (("amp_min",amps.min()),("amp_median",np.median(amps)),("amp_max",amps.max()),
                    ("amp_range",np.ptp(amps)),("amp_cv",amps.std()/(abs(amps.mean())+EPS))):
            add_num(f,"decomposition",n,v)
        s=np.sort(amps)[::-1]
        if len(s)>=3:
            add_num(f,"decomposition","amp3_over_amp2",s[2]/(s[1]+EPS))
            add_num(f,"decomposition","amp3_over_amp1",s[2]/(s[0]+EPS))
    if len(f0):
        sf=np.sort(f0)
        add_num(f,"decomposition","f0_min",sf[0]); add_num(f,"decomposition","f0_median",np.median(sf))
        add_num(f,"decomposition","f0_max",sf[-1]); add_num(f,"decomposition","f0_span_oct",math.log2((sf[-1]+EPS)/(sf[0]+EPS)))

    comps=d.get("components",[])
    if comps:
        for metric in ("nearest_owned_cents","nearest_owned_harmonic_2_to_6_cents"):
            a=np.asarray([float(c[metric]) for c in comps if c.get(metric) is not None],float)
            if len(a):
                add_num(f,"decomposition",metric+"_min",a.min())
                add_num(f,"decomposition",metric+"_median",np.median(a))
        add_num(f,"decomposition","components_within_55c",
                sum(float(c.get("nearest_owned_cents",1e9))<=55 for c in comps))

    # True-note geometry (diagnostic annotations).
    notes=row["owned_notes"]
    midis=np.asarray([float(n["midi"]) for n in notes],float)
    onsets=np.asarray([int(n["onset_sample"]) for n in notes],int)
    add_num(f,"true_geometry","midi_mean",midis.mean())
    add_num(f,"true_geometry","midi_median",np.median(midis))
    add_num(f,"true_geometry","midi_span",np.ptp(midis))
    if len(onsets)>1:
        dif=np.diff(np.sort(onsets))*1000/44100.0
        add_num(f,"true_geometry","onset_gap_mean_ms",dif.mean())
        add_num(f,"true_geometry","onset_gap_min_ms",dif.min())
        add_num(f,"true_geometry","onset_gap_max_ms",dif.max())

    # Annotation-centered local pick: per-note mean/median only.
    bursts=[note_burst(window(samples,int(n["onset_sample"]))) for n in notes]
    agg([{k:b[k] for k in PICK_METRICS} for b in bursts],"true_pick",f)

    # Three-window spectral attack novelty.
    freq,powers,_=raw_powers(samples,int(row["start_sample"]))
    expected_f0=np.asarray([float(n["frequency_hz"]) for n in notes],float)
    nov=[component_metrics(freq,powers,q) for q in expected_f0]
    agg([{k:v[k] for k in NOV_METRICS} for v in nov],"true_novelty",f)

    # Expected-source exclusivity among true note F0s.
    attack=attack_spectrum(powers)
    ex=[]
    for i,q in enumerate(expected_f0):
        peers=[float(expected_f0[j]) for j in range(len(expected_f0)) if j!=i]
        ex.append(support_against_set(freq,attack,float(q),peers))
    ex_clean=[]
    for v in ex:
        z={k:(math.log1p(max(float(v[k]),0.0)) if k=="attack_energy_total" else float(v[k])) for k in EX_METRICS}
        ex_clean.append(z)
    agg(ex_clean,"true_exclusive",f)

    # Inference-safe selected residual-F0 features.
    for k,v in selected_f0_features(row,freq,powers).items():
        add_num(f,"selected_f0",k,v)
    return f

def qstats(x):
    a=np.asarray(x,float)
    return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}

def feature_report(rows,name):
    vals=np.asarray([r["features"].get(name,np.nan) for r in rows],float)
    y=np.asarray([r["group"]=="K3_regressed" for r in rows],int)
    folds=np.asarray([r["fold"] for r in rows],int)
    ok=np.isfinite(vals)
    if ok.sum()<20 or len(np.unique(y[ok]))<2:
        return None
    auc=float(roc_auc_score(y[ok],vals[ok]))
    direction=1 if auc>=.5 else -1
    fold_rows=[]
    for fold in FOLDS:
        m=ok&(folds==fold)
        if m.sum()<4 or len(np.unique(y[m]))<2:
            return None
        a=float(roc_auc_score(y[m],vals[m]))
        fold_rows.append({"fold":int(fold),"auc_raw":a,"auc_oriented":a if direction==1 else 1-a,
                          "k2_median":float(np.median(vals[m&(y==0)])),
                          "k3_median":float(np.median(vals[m&(y==1)]))})
    oriented=[q["auc_oriented"] for q in fold_rows]
    k2=vals[ok&(y==0)]; k3=vals[ok&(y==1)]
    same=sum((q["auc_raw"]>=.5)==(direction==1) for q in fold_rows)
    denom=np.subtract(*np.percentile(vals[ok],[75,25]))+EPS
    return {
      "feature":name,"domain":name.split(".",1)[0],
      "direction":"higher_in_regressions" if direction==1 else "lower_in_regressions",
      "auc_raw_global":auc,"mean_oriented_auc":float(np.mean(oriented)),"min_oriented_auc":float(np.min(oriented)),
      "same_direction_folds":int(same),"stable_4_of_4":bool(same==4),
      "median_delta_k3_minus_k2":float(np.median(k3)-np.median(k2)),
      "median_delta_over_iqr":float((np.median(k3)-np.median(k2))/denom),
      "K2_corrected":qstats(k2),"K3_regressed":qstats(k3),"per_fold":fold_rows
    }

def main():
    ap=argparse.ArgumentParser()
    for n in ("cases","dataset","output"): ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"refusing overwrite")
    rows0=[json.loads(x) for x in a.cases.read_text().splitlines()]
    rows0=[r for r in rows0 if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in rows0)==108,"K2 drift")
    require(sum(r["group"]=="K3_regressed" for r in rows0)==125,"K3 drift")
    require(all(r["fold"] in FOLDS and r["fold"]!=3 for r in rows0),"fold leak")
    wanted={r["recording_id"] for r in rows0}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in rows0: by[r["recording_id"]].append(r)
    rows=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for r in sorted(by[member],key=lambda z:z["row_id"]):
            rows.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],
                         "recording_id":member,"features":build_features(r,samples)})
        print(json.dumps({"recording":member,"done":len(rows)}),flush=True)
    common=sorted(set.intersection(*(set(r["features"]) for r in rows)))
    reports=[q for n in common if (q:=feature_report(rows,n)) is not None]
    reports.sort(key=lambda q:(q["stable_4_of_4"],q["min_oriented_auc"],q["mean_oriented_auc"]),reverse=True)
    stable=[q for q in reports if q["stable_4_of_4"] and q["min_oriented_auc"]>=.55]
    suggestive=[q for q in reports if q["same_direction_folds"]>=3 and q["mean_oriented_auc"]>=.58]
    domain_best={}
    for q in reports:
        domain_best.setdefault(q["domain"],[])
        if len(domain_best[q["domain"]])<5: domain_best[q["domain"]].append(q)
    rep={"status":"completed","experiment":"v273_correction_regression_common_signatures",
         "populations":{"K2_corrected":108,"K3_regressed":125},"folds":list(FOLDS),"outer_fold_3_used":False,
         "prediction_changes":False,"threshold_tuning":False,
         "order_statistic_control":"true-note pick features use mean/median, never weakest-of-K",
         "features_evaluated":len(reports),"stable_signatures":stable,"suggestive_signatures":suggestive,
         "domain_best":domain_best,"all_features":reports}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    (a.output/"rows.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    lines=["# Corrections vs regressions — common signature audit","",
           "Population: **108 K2 corrections** vs **125 K3 regressions**. Fold 3 excluded. No threshold is tuned.","",
           "Stable signature criterion: same direction on all 4 folds and minimum oriented fold AUC >= 0.55.","",
           f"Stable signatures found: **{len(stable)}** / {len(reports)} numeric features.","",
           "| feature | domain | direction in regressions | K2 median | K3 median | mean AUC | min fold AUC |",
           "|---|---|---|---:|---:|---:|---:|"]
    for q in stable[:25]:
        lines.append(f"| {q['feature']} | {q['domain']} | {q['direction']} | {q['K2_corrected']['median']:.6g} | {q['K3_regressed']['median']:.6g} | {q['mean_oriented_auc']:.3f} | {q['min_oriented_auc']:.3f} |")
    lines+=["","## Best signature by domain","",
            "| domain | feature | direction | mean AUC | min fold AUC | folds same direction |",
            "|---|---|---|---:|---:|---:|"]
    for d,qq in sorted(domain_best.items()):
        q=qq[0]; lines.append(f"| {d} | {q['feature']} | {q['direction']} | {q['mean_oriented_auc']:.3f} | {q['min_oriented_auc']:.3f} | {q['same_direction_folds']}/4 |")
    lines+=["","This is descriptive diagnosis only. Annotation-derived domains are not inference inputs."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__": main()
