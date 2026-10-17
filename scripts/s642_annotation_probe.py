"""Inventory real GuitarSet TRAIN isolated E2=40 and E3=52 candidate onsets."""
from pathlib import Path
import tempfile,json,zipfile
from scripts.s637_guitarset_three_view_groundtruth import ROOT
from scripts.s613_prepare_strict_live import download,labels
from scripts.s613_train_strict_live import real_notes

def eligible(truth,pitch, min_d=.25, onset_window=.12):
    rows=[]
    for n in truth:
        if n["pitch"]!=pitch or n["duration"]<min_d or n["time"]<.75 or n["time"]>22.75:continue
        start=n["time"]
        if any(o is not n and o["pitch"]!=pitch
               and o["time"]<start+onset_window
               and o["time"]+o["duration"]>start-.05 for o in truth):
            continue
        rows.append(round(start,4))
    return rows

def main():
    p=json.loads((ROOT/"protocol.json").read_text())
    tracks=[s for s in p["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        item=p["guitar_archives"]["GuitarSet_annotations"]
        download(item["url"],root/"data/GuitarSet_annotations.zip",item["size"],item["sha256"])
        out=[]
        for s in tracks:
            truth=real_notes(labels(s,root)["notes"],24.)
            a={str(m):eligible(truth,m) for m in range(40,77)}
            counts={str(m):sum(n["pitch"]==m for n in truth) for m in range(40,77)}
            out.append(dict(name=s["name"],composition=s["composition"],E2=a["40"],E3=a["52"],**a,count=counts))
        print("S642 ISOLATION",json.dumps(dict(n=len(out),
            total_E2=sum(r["count"]["40"] for r in out),
            total_E3=sum(r["count"]["52"] for r in out),
            eligible_E2=sum(len(r["E2"]) for r in out),
            eligible_E3=sum(len(r["E3"]) for r in out),
            tracks_E2=[dict(name=r["name"],n=len(r["E2"])) for r in out if r["E2"]],
            tracks_E3=[dict(name=r["name"],n=len(r["E3"])) for r in out if r["E3"]],
            pairs=sorted([dict(pitch=m,low=sum(len(r[str(m)]) for r in out),high=sum(len(r[str(m+12)]) for r in out),distinct_low=sum(bool(r[str(m)]) for r in out),distinct_high=sum(bool(r[str(m+12)]) for r in out)) for m in range(40,65)],key=lambda r:-min(r["low"],r["high"]))[:25])),flush=True)

if __name__=="__main__":main()
