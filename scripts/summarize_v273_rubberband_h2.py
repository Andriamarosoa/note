"""Summarize fixed Rubber Band H2 across internal folds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.v273_residual_audit import FOLDS,require,write_json

LIBROSA={0:{"corrections":14,"regressions":9,"net":5},
         1:{"net":6},2:{"net":-4},4:{"net":3}}
LIBROSA_TOTAL={"corrections":26,"regressions":16,"net":10}

def main():
    p=argparse.ArgumentParser();p.add_argument("--input-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    rows=[]
    for q in a.input_root.rglob("report.json"):
        try:r=json.loads(q.read_text())
        except Exception:continue
        if r.get("experiment")=="v273_rubberband_h2_fold": rows.append(r)
    require({int(r["validation_fold"]) for r in rows}==set(FOLDS),"missing folds")
    rows=sorted(rows,key=lambda r:int(r["validation_fold"]))
    total={k:int(sum(int(r[k]) for r in rows)) for k in ("actions","corrections","regressions","net")}
    report={"status":"completed","experiment":"v273_rubberband_h2_summary","folds":rows,
            "rubberband_total":total,"librosa_reference_total":LIBROSA_TOTAL,
            "net_delta_vs_librosa":int(total["net"]-10),"outer_fold_3_used":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Rubber Band H2 — internal comparison","",
      "| fold | Rubber corr | Rubber reg | Rubber net | Librosa H2 net |",
      "|---:|---:|---:|---:|---:|"]
    for r in rows:
        f=int(r["validation_fold"]); lines.append(
          f"| {f} | {r['corrections']} | {r['regressions']} | {r['net']:+d} | {LIBROSA[f]['net']:+d} |")
    lines+=["",f"Rubber Band total: **{total['corrections']} / {total['regressions']} = {total['net']:+d}**.",
            "Librosa H2 reference: **26 / 16 = +10**.",
            f"Net delta vs Librosa: **{total['net']-10:+d}**.","","Fold 3 excluded."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__": main()
