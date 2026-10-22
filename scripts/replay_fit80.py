"""True signed FIT80 80-track S834 frame_pitch-only replay."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from frame_pitch.acoustics import SR,HOP,load_native_heads
from frame_pitch.temporal import load_frozen_source
from frame_pitch.runtime import Stream
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(decisions,gt,ends,tail=()):
    target=np.zeros((len(ends),49),bool)
    t=ends/SR
    for n in gt:
        i=int(n["pitch"])-40
        target[(t>=n["time"])&(t<n["time"]+n["duration"]),i]=True
    guess=np.zeros_like(target)
    for t,d in enumerate(decisions):
        for p in d.pitches:guess[t,p-40]=True
    return dict(tp=int((guess&target).sum()),
        fp=int((guess&~target).sum()),fn=int((~guess&target).sum()))

def run(audio_zip,traces,manifest,checkpoint,source_report,native_model,out):
    if out.exists():raise FileExistsError("do not overwrite replay evidence")
    m=json.loads(manifest.read_text())
    if m.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not m.get("complete") or len(m.get("fit_files",[]))!=80:raise ValueError("unsigned FIT80")
    if any(m.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):raise ValueError("prohibited split opened")
    blend=float(json.loads(source_report.read_text())["frozen_candidate"]["choice"]["policy"]["blend"])
    native=load_native_heads(native_model)
    model,sha=load_frozen_source(checkpoint,source_report)
    out.mkdir(parents=True)
    records=[]
    with zipfile.ZipFile(audio_zip) as z:
        for comp in ("BN1","Funk2","Jazz1","Rock3","SS1"):
            files=sorted([a for a in m["fit_files"] if a["composition"]==comp],key=lambda a:a["index"])
            if len(files)!=16:raise AssertionError("missing original signed files")
            for item in files:
                mp=traces/f"fit_{item['index']:03d}.json"
                if digest(mp)!=item["json_sha256"]:raise ValueError("signed track metadata changed")
                meta=json.loads(mp.read_text())
                if meta["name"]!=item["name"]:raise ValueError("wrong track")
                sr,pcm=wavfile.read(io.BytesIO(z.read(meta["name"]+"_mic.wav")))
                if sr!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:raise ValueError("bad original WAV")
                x=pcm[:meta["samples"]].astype(np.float32)/32768
                ends=np.arange(HOP,len(x)+1,HOP)
                s=Stream(native,model,blend);decisions=[]
                for pos in range(0,len(x),HOP):
                    obs=s.push(x[pos:pos+HOP],start_sample=pos)
                    if len(obs)!=(1 if pos+HOP<=len(x) else 0):raise AssertionError("256 boundary drift")
                    decisions.extend(obs)
                if len(decisions)!=len(ends) or s.calls!=len(ends):raise AssertionError("not every 256")
                tail=s.finish(meta["samples"])
                counts=score(decisions,meta["fit_truth"],ends,tail)
                records.append(dict(index=item["index"],name=item["name"],composition=comp,frames=len(ends),counts=counts))
                (out/"report.json").write_text(json.dumps(dict(complete=False,records=records),indent=2))
                print("FRAME_PITCH_256_TRACK",item["index"],comp,flush=True)
    total={k:sum(x["counts"][k] for x in records) for k in records[0]["counts"]}
    total["frames"]=sum(x["frames"] for x in records)
    expected={'tp':467540,'fp':71501,'fn':93528,'frames':306536}
    denom=2*total["tp"]+total["fp"]+total["fn"]
    total["frame_pitch_f1"]=2*total["tp"]/denom
    reference_denom=2*expected["tp"]+expected["fp"]+expected["fn"]
    validation_passed=(total["frames"]==expected["frames"] and
                       total["tp"]*reference_denom>=expected["tp"]*denom)
    if not validation_passed:
        (out/"report.json").write_text(json.dumps(dict(complete=False,records=records,totals=total,validation_passed=False),indent=2))
        raise AssertionError(f"S834 frame_pitch F1 regression: {total['frame_pitch_f1']:.8f} < {2*expected['tp']/reference_denom:.8f}")
    report=dict(complete=True,axis="frame_pitch",tracks=records,totals=total,
       model_sha256=sha,source_training_hop=1024,decision_hop=256,
       public_midi_modified=False,promoted=False,validation_passed=validation_passed)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("FRAME_PITCH_256_FINAL",json.dumps(total),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest","checkpoint","source-report","native-model","out"):
        p.add_argument("--"+name,required=True,type=Path)
    x=p.parse_args()
    run(x.audio_zip,x.traces,x.manifest,x.checkpoint,x.source_report,x.native_model,x.out)
