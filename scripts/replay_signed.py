"""Signed 80-track H0 per-axis replay, no PCM256 modernization claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from historical.scorer import SR,onset_score,offset_score
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(manifest,traces,out,axis):
    if out.exists():raise FileExistsError("immutable reference record")
    source=json.loads(manifest.read_text())
    if source.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not source.get("complete") or len(source.get("fit_files",[]))!=80:
        raise ValueError("signed FIT80 required")
    if any(source.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):raise ValueError("sealed outside split")
    out.mkdir(parents=True);tracks=[]
    for comp in ("BN1","Funk2","Jazz1","Rock3","SS1"):
        specs=sorted([x for x in source["fit_files"] if x["composition"]==comp],key=lambda x:x["index"])
        if len(specs)!=16:raise AssertionError("missing signed tracks")
        for spec in specs:
            p=traces/f"fit_{spec['index']:03d}.json"
            if digest(p)!=spec["json_sha256"]:raise ValueError("unsigned source track")
            meta=json.loads(p.read_text())
            if meta["name"]!=spec["name"] or meta["composition"]!=comp:raise AssertionError("wrong song")
            events=meta["source_events"];notes=meta["fit_truth"]
            ends=np.arange(256,meta["samples"]+1,256,dtype=np.int64)
            for e in events:
                if e["available_sample"]<e["decision_sample"] or abs(e["time"]-e["available_sample"]/SR)>1e-10:
                    raise AssertionError("signed source noncausal")
            counters=(onset_score(events,notes) if axis=="onset" else offset_score(events,notes,ends))
            baseline=meta["source_counts"]
            for k,v in counters.items():
                if v!=baseline[k]:raise AssertionError(f"event-level score mismatch {k}: {v}!={baseline[k]}")
            tracks.append(dict(name=meta["name"],composition=comp,counts=counters,frames=len(ends)))
            (out/"report.json").write_text(json.dumps(dict(complete=False,tracks=tracks),indent=2))
            print("H0_"+axis.upper()+"_TRACK",spec["index"],comp,flush=True)
    totals={k:sum(t["counts"][k] for t in tracks) for k in tracks[0]["counts"]}
    totals["frames"]=sum(t["frames"] for t in tracks)
    expected=({"tp":8528,"fp":3489,"fn":2739,"frames":306536} if axis=="onset"
              else {"offset_eligible":8438,"offset_correct":5897,"frames":306536})
    if any(totals[k]!=v for k,v in expected.items()):
        raise AssertionError(f"S708 source H0 not reproduced {totals} != {expected}")
    totals["score"]=(2*totals["tp"]/(2*totals["tp"]+totals["fp"]+totals["fn"]) if axis=="onset"
       else totals["offset_correct"]/totals["offset_eligible"])
    (out/"report.json").write_text(json.dumps(dict(complete=True,axis=axis,totals=totals,
      tracks=tracks,model_retrained=False,original_historical_model_hop=1024,
      source_events_from_signed_h0=True,native_pcm256_inference_proven=False,
      no_other_axes_present=True,promoted=False),indent=2))
    print("H0_"+axis.upper()+"_FIT80_FINAL",json.dumps(totals),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--axis",choices=("onset","offset"),required=True)
    for x in ("manifest","traces","out"):p.add_argument("--"+x,type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.traces,a.out,a.axis)
