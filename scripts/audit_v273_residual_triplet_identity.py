"""Audit identity between archived residual-case triplet F0s and recomputed extract_one triplets.

This resolves whether the +6 K2/K3 onset-contrast guard and the -2 full-action
guard used exactly the same underlying F0 hypotheses.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum,extract_one
from scripts.audit_v273_attack_novelty import raw_powers,component_metrics
from scripts.v273_residual_audit import FOLDS,require

GROUPS=("K2_corrected","K3_regressed")
def cents(a,b): return abs(1200*math.log2(float(a)/float(b)))

def contrast(samples,start,f0s):
    freq,powers,_=raw_powers(samples,start)
    return float(np.median([component_metrics(freq,powers,float(f))["onset_contrast_norm"] for f in f0s]))

def main():
    p=argparse.ArgumentParser()
    for n in ("cases","dataset","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"refusing overwrite")
    rows=[json.loads(x) for x in a.cases.read_text().splitlines()]
    rows=[r for r in rows if r["group"] in GROUPS]
    require(len(rows)==233,"cohort drift");require(all(r["fold"] in FOLDS for r in rows),"fold leak")
    wanted={r["recording_id"] for r in rows};tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted};require(set(tracks)==wanted,"audio")
    by=defaultdict(list)
    for r in rows:by[r["recording_id"]].append(r)
    out=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member);s=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            archived=np.sort(np.asarray(r["decomposition"]["triplet_f0"],float))
            ft,x=transition_spectrum(s,int(r["start_sample"]));z=extract_one(ft,x)
            require(z is not None,"recompute failed")
            recomputed=np.sort(np.asarray([z["min_triplet_f0"],z["median_triplet_f0"],z["max_triplet_f0"]],float))
            diffs=np.asarray([cents(a,b) for a,b in zip(archived,recomputed)])
            ca=contrast(s,int(r["start_sample"]),archived);cr=contrast(s,int(r["start_sample"]),recomputed)
            out.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],
                        "archived_f0":archived.tolist(),"recomputed_f0":recomputed.tolist(),
                        "cents":diffs.tolist(),"max_cents":float(diffs.max()),
                        "archived_contrast":ca,"recomputed_contrast":cr,"contrast_abs_diff":abs(ca-cr)})
        print(json.dumps({"recording":member,"done":len(out)}),flush=True)
    maxc=np.asarray([r["max_cents"] for r in out]);cd=np.asarray([r["contrast_abs_diff"] for r in out])
    report={"status":"completed","experiment":"v273_residual_triplet_identity","rows":len(out),"outer_fold_3_used":False,
            "exact_triplet_rows":int(np.sum(maxc<1e-9)),
            "within_1_cent":int(np.sum(maxc<=1)),"within_10_cents":int(np.sum(maxc<=10)),
            "within_50_cents":int(np.sum(maxc<=50)),"different_gt50_cents":int(np.sum(maxc>50)),
            "max_cents":{"median":float(np.median(maxc)),"q75":float(np.quantile(maxc,.75)),"max":float(maxc.max())},
            "contrast_abs_diff":{"median":float(np.median(cd)),"q75":float(np.quantile(cd,.75)),"max":float(cd.max())},
            "by_group":{}}
    for g in GROUPS:
        rr=[r for r in out if r["group"]==g];m=np.asarray([r["max_cents"] for r in rr]);d=np.asarray([r["contrast_abs_diff"] for r in rr])
        report["by_group"][g]={"rows":len(rr),"exact":int(np.sum(m<1e-9)),"gt50c":int(np.sum(m>50)),
                               "median_max_cents":float(np.median(m)),"median_contrast_abs_diff":float(np.median(d))}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Residual triplet F0 identity audit","",
           f"Rows: **{len(out)}**; exact triplets: **{report['exact_triplet_rows']}**; >50 cents different: **{report['different_gt50_cents']}**.",
           f"Median max-F0 mismatch: **{report['max_cents']['median']:.3f} cents**.",
           f"Median onset-contrast feature absolute drift: **{report['contrast_abs_diff']['median']:.6f}**.","",
           "| group | rows | exact | >50c | median max cents | median feature drift |","|---|---:|---:|---:|---:|---:|"]
    for g,q in report["by_group"].items():lines.append(f"| {g} | {q['rows']} | {q['exact']} | {q['gt50c']} | {q['median_max_cents']:.3f} | {q['median_contrast_abs_diff']:.6f} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
