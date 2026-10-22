"""True causal PCM256 onset-only FIT80 evaluation; no production MIDI edits."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
from onset256.acoustics import SR,HOP
from onset256.model import load
from onset256.runtime import Stream
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

def score(decisions,gt,ends,tail=(),*,axis='intents'):
    predicted=[e for d in decisions for e in getattr(d,axis) if e.kind=="note_on"]
    pairs=match_onsets(predicted,gt)
    tp=len(pairs)
    return dict(tp=tp,fp=len(predicted)-tp,fn=len(gt)-tp)

def run(audio_zip,traces,manifest,native_model,s701_model,out):
    if out.exists():raise FileExistsError("do not overwrite immutable research evidence")
    m=json.loads(manifest.read_text())
    if m.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not m.get("complete") or len(m.get("fit_files",[]))!=80:
        raise ValueError("unsigned FIT80")
    if any(m.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):
        raise ValueError("forbidden evaluation split opened")
    native,specialist=load(native_model,s701_model)
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
                stream=Stream(native,specialist)
                decisions=[]
                for pos in range(0,len(x),HOP):
                    obs=stream.push(x[pos:pos+HOP],start_sample=pos)
                    if len(obs)!=(1 if pos+HOP<=len(x) else 0):
                        raise AssertionError("not one decision per 256 PCM samples")
                    decisions.extend(obs)
                if len(decisions)!=len(ends) or stream.calls!=len(ends) or stream.s701_calls!=len(ends):
                    raise AssertionError("not one fresh acoustic+temporal model call every 256")
                if any(d.sample!=int(end) for d,end in zip(decisions,ends)):
                    raise AssertionError("causal time-grid drift")
                tail=stream.finish(meta["samples"])
                # Ground truth is accessed ONLY after the complete PCM-only inference pass.
                counts=score(decisions,meta["fit_truth"],ends,tail)
                shadow=score(decisions,meta["fit_truth"],ends,tail,axis="candidate_intents")
                records.append(dict(index=item["index"],name=meta["name"],
                   composition=comp,frames=len(ends),counts=counts,shadow_counts=shadow))
                (out/"report.json").write_text(json.dumps(dict(complete=False,records=records),indent=2))
                print("ONSET_NATIVE256_TRACK",item["index"],comp,flush=True)
    total={k:sum(rec["counts"][k] for rec in records) for k in records[0]["counts"]}
    shadow={k:sum(rec["shadow_counts"][k] for rec in records) for k in records[0]["shadow_counts"]}
    total["frames"]=sum(rec["frames"] for rec in records)
    if total["frames"]!=306536 or len(records)!=80:
        raise AssertionError("wrong signed 80-track PCM256 decision grid")
    for result in (total,shadow):
        den=2*result["tp"]+result["fp"]+result["fn"]
        result["onset_f1"]=2*result["tp"]/den if den else 0.
    reference=dict(tp=8528,fp=3489,fn=2739,f1=2*8528/(2*8528+3489+2739))
    # S701 selected the native branch before FIT80 evaluation; do not
    # select the highest-scoring alternative AFTER opening FIT80 labels.
    passed=total["onset_f1"]>=reference["f1"]
    report=dict(complete=True,axis="onset",
      tracks=records,totals=total,s701_shadow_totals=shadow,
      historical_H0_reference=reference,validation_passed=passed,
      s701_preselected_policy="native",shadow_promoted=False,
      feature_hop_samples=256,decision_hop_samples=256,metric_hop_samples=256,
      s613_training_hop_samples=256,s701_training_hop_samples=256,
      original_s613_sha256="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f",
      private_s701_sha256="cc72cf81f477c6b734e253df5fb04d1789ab707dc160f06bdc5cf9ffdff2fb0f",
      no_s708_temporal_checkpoint=True,native_pcm256_inference_proven=True,
      hardware_latency_measured=False,fit_source_explored=True,
      data_external_generalization=False,gt_inference_leakage=False,
      public_midi_modified=False,promoted=False)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("ONSET_S613_S701_TRUE256_FIT80_FINAL",json.dumps({
      "s613_native":total,"s701_shadow":shadow,"reference":reference,
      "validation_passed":passed}),flush=True)
    if not passed:
        raise AssertionError("Onset trained256 native below preserved historical H0 threshold")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest","native-model","s701-model","out"):
        p.add_argument("--"+name,type=Path,required=True)
    x=p.parse_args()
    run(x.audio_zip,x.traces,x.manifest,x.native_model,x.s701_model,x.out)
