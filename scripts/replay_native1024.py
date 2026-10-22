"""True causal PCM1024 offset-only FIT80 evaluation; no production MIDI edits."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
from offset1024.acoustics import SR,HOP,load_native_heads
from offset1024.temporal import load_frozen_source
from offset1024.runtime import Stream
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def match_onsets(predicted,gt,tolerance=.080):
    pairs=[]
    for pitch in range(40,89):
        pi=[i for i,e in enumerate(predicted) if e.pitch==pitch]
        gi=[i for i,n in enumerate(gt) if int(n["pitch"])==pitch]
        if not pi or not gi:continue
        delta=np.abs(np.asarray([predicted[i].sample/SR for i in pi])[:,None]-
                     np.asarray([gt[j]["time"] for j in gi])[None,:])
        ii,jj=linear_sum_assignment(np.where(delta<=tolerance,delta,1e6))
        pairs.extend((pi[i],gi[j]) for i,j in zip(ii,jj) if delta[i,j]<=tolerance)
    return pairs

def score(decisions,gt,ends,tail=()):
    # The independent Offset head keeps PRIVATE diagnostic onsets solely to
    # define S704-style matched-onset eligibility, never exposed as NoteOn MIDI.
    predicted=[e for d in decisions for e in d.onsets]
    pairs=match_onsets(predicted,gt)
    intervals=[]
    held={}
    for d in decisions:
        for e in d.intents:
            if e.kind=="note_off" and e.pitch in held:
                intervals[held.pop(e.pitch)]["off"]=e
        for e in d.onsets:
            if e.pitch in held:raise AssertionError("duplicate internal onset")
            held[e.pitch]=len(intervals)
            intervals.append(dict(on=e))
    for e in tail:
        if e.pitch in held:intervals[held.pop(e.pitch)]["off"]=e
    eligible=correct=0
    for i,j in pairs:
        note=gt[j]
        stop=float(note["time"])+float(note["duration"])
        if stop>=ends[-1]/SR:continue
        eligible+=1
        off=intervals[i].get("off")
        if off and off.reason not in ("stream_end","shutdown","stream_error") and abs(off.sample/SR-stop)<=.050:
            correct+=1
    return dict(offset_eligible=eligible,offset_correct=correct,
                internal_onset_tp=len(pairs),
                internal_onset_fp=len(predicted)-len(pairs),
                internal_onset_fn=len(gt)-len(pairs))

def run(audio_zip,traces,manifest,checkpoint,source_report,native_model,out):
    if out.exists():raise FileExistsError("do not overwrite immutable research evidence")
    m=json.loads(manifest.read_text())
    if m.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not m.get("complete") or len(m.get("fit_files",[]))!=80:
        raise ValueError("unsigned FIT80")
    if any(m.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):
        raise ValueError("forbidden evaluation split opened")
    blend=float(json.loads(source_report.read_text())["frozen_candidate"]["choice"]["policy"]["blend"])
    native=load_native_heads(native_model)
    model,sha=load_frozen_source(checkpoint,source_report)
    out.mkdir(parents=True)
    records=[]
    with zipfile.ZipFile(audio_zip) as z:
        for comp in ("BN1","Funk2","Jazz1","Rock3","SS1"):
            files=sorted([a for a in m["fit_files"] if a["composition"]==comp],key=lambda a:a["index"])
            if len(files)!=16:raise AssertionError("missing signed GuitarSet tracks")
            for item in files:
                mp=traces/f"fit_{item['index']:03d}.json"
                if digest(mp)!=item["json_sha256"]:raise ValueError("signed track metadata changed")
                meta=json.loads(mp.read_text())
                if meta["name"]!=item["name"]:raise ValueError("wrong signed track")
                sr,pcm=wavfile.read(io.BytesIO(z.read(meta["name"]+"_mic.wav")))
                if sr!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:raise ValueError("unsigned original WAV")
                x=pcm[:meta["samples"]].astype(np.float32)/32768
                ends=np.arange(HOP,len(x)+1,HOP)
                stream=Stream(native,model,blend)
                decisions=[]
                for pos in range(0,len(x),HOP):
                    obs=stream.push(x[pos:pos+HOP],start_sample=pos)
                    if len(obs)!=(1 if pos+HOP<=len(x) else 0):
                        raise AssertionError("not one decision per 1024 PCM samples")
                    decisions.extend(obs)
                if len(decisions)!=len(ends) or stream.calls!=len(ends):
                    raise AssertionError("not one fresh acoustic+temporal model call every 1024")
                if any(d.sample!=int(end) for d,end in zip(decisions,ends)):
                    raise AssertionError("causal time-grid drift")
                tail=stream.finish(meta["samples"])
                # Ground truth is accessed ONLY after the complete PCM-only inference pass.
                counts=score(decisions,meta["fit_truth"],ends,tail)
                records.append(dict(index=item["index"],name=meta["name"],
                   composition=comp,frames=len(ends),counts=counts))
                (out/"report.json").write_text(json.dumps(dict(complete=False,records=records),indent=2))
                print("OFFSET_NATIVE1024_TRACK",item["index"],comp,flush=True)
    total={k:sum(rec["counts"][k] for rec in records) for k in records[0]["counts"]}
    total["frames"]=sum(rec["frames"] for rec in records)
    if len(records)!=80:raise AssertionError("incomplete signed FIT80 1024-hop evaluation")
    total["conditional_offset_accuracy"]=(total["offset_correct"]/total["offset_eligible"]
                                          if total["offset_eligible"] else 0.)
    reference_f1=5897/8438
    # Do not claim improvement by making fewer difficult note offsets eligible.
    validation_passed=(total["offset_eligible"]>=8438 and
                       total["conditional_offset_accuracy"]>=reference_f1)

    report=dict(complete=True,axis="offset",tracks=records,totals=total,
       reference_metric=reference_f1,validation_passed=validation_passed,
       model_sha256=sha,source_training_hop=1024,decision_hop=HOP,
       native_pcm1024_inference_proven=True,hardware_latency_measured=False,
       gt_inference_leakage=False,fit_source_explored=True,
       public_midi_modified=False,promoted=False)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("OFFSET_NATIVE1024_FIT80_FINAL",json.dumps(total),flush=True)
    if not validation_passed:
        raise AssertionError("offset native PCM1024 below preserved S708/H0 reference; research candidate only")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest","checkpoint","source-report","native-model","out"):
        p.add_argument("--"+name,type=Path,required=True)
    x=p.parse_args()
    run(x.audio_zip,x.traces,x.manifest,x.checkpoint,x.source_report,x.native_model,x.out)
