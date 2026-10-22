"""True signed FIT80 80-track S834 full_note-only replay."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
from full_note.acoustics import SR,HOP,load_native_heads
from full_note.temporal import load_frozen_source
from full_note.runtime import Stream
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(decisions,gt,ends,tail=()):
    events=[e for d in decisions for e in d.intents]+list(tail)
    predicted=[e for e in events if e.kind=="note_on"]
    pairs=[]
    for p in range(40,89):
        pi=[i for i,e in enumerate(predicted) if e.pitch==p]
        ti=[j for j,n in enumerate(gt) if n["pitch"]==p]
        if not pi or not ti:continue
        delta=np.abs(np.asarray([predicted[i].sample/SR for i in pi])[:,None]
                     -np.asarray([gt[j]["time"] for j in ti])[None,:])
        ii,jj=linear_sum_assignment(np.where(delta<=.080,delta,1e6))
        pairs.extend((pi[i],ti[j]) for i,j in zip(ii,jj) if delta[i,j]<=.080)
    intervals=[];held={}
    for e in events:
        if e.kind=="note_on":
            if e.pitch in held:raise AssertionError("duplicate active note")
            held[e.pitch]=len(intervals);intervals.append(dict(on=e))
        elif e.pitch in held:
            intervals[held.pop(e.pitch)]["off"]=e
    eligible=correct=0
    for i,j in pairs:
        n=gt[j];stop=n["time"]+n["duration"]
        if stop>=ends[-1]/SR:continue
        eligible+=1
        off=intervals[i].get("off")
        if off and off.reason not in ("stream_end","shutdown","stream_error") and abs(off.sample/SR-stop)<=.050:correct+=1
    return dict(tp=len(pairs),fp=len(predicted)-len(pairs),
        fn=len(gt)-len(pairs),offset_eligible=eligible,offset_correct=correct)

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
                    if len(obs)!=(1 if pos+HOP<=len(x) else 0):raise AssertionError("1024 boundary drift")
                    decisions.extend(obs)
                if len(decisions)!=len(ends) or s.calls!=len(ends):raise AssertionError("not every 1024")
                tail=s.finish(meta["samples"])
                counts=score(decisions,meta["fit_truth"],ends,tail)
                records.append(dict(index=item["index"],name=item["name"],composition=comp,frames=len(ends),counts=counts))
                (out/"report.json").write_text(json.dumps(dict(complete=False,records=records),indent=2))
                print("FULL_NOTE_1024_TRACK",item["index"],comp,flush=True)
    total={k:sum(x["counts"][k] for x in records) for k in records[0]["counts"]}
    total["frames"]=sum(x["frames"] for x in records)
    original_pcm256={'tp':8224,'fp':3640,'fn':3043,'offset_correct':4093,'offset_eligible':8135,'frames':306536}
    denom=2*total["tp"]+total["fp"]+total["fn"]
    total["noteon_f1"]=2*total["tp"]/denom if denom else 0.0
    total["full_note_f1"]=2*total["offset_correct"]/denom if denom else 0.0
    if len(records)!=80:raise AssertionError("incomplete 80-track PCM1024 replay")
    report=dict(complete=True,axis="full_note",tracks=records,totals=total,
       model_sha256=sha,source_training_hop=1024,decision_hop=HOP,
       historical_pcm256_reference=original_pcm256,reference_comparable=False,
       validation_passed=None,public_midi_modified=False,promoted=False)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("FULL_NOTE_1024_FINAL",json.dumps(total),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest","checkpoint","source-report","native-model","out"):
        p.add_argument("--"+name,required=True,type=Path)
    x=p.parse_args()
    run(x.audio_zip,x.traces,x.manifest,x.checkpoint,x.source_report,x.native_model,x.out)
