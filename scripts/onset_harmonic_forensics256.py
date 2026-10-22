"""Event-level harmonic forensics on the ORIGINAL 80 GuitarSet PCM recordings.

No policy tuning or weight training. For each 256-sample Onset emitted by the
genuine S613/S701 model, store the frozen CURRENT frame's fundamental/parent
spectral context. Only AFTER a full track's causal inference, attach matching
labels and compare TP/FP distributions. Report is descriptive (FIT80 has
historical train overlap), never a promotion or independent validation.
"""
import argparse, hashlib, io, json, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from onset256.acoustics import HOP, SR
from onset256.model import load
from onset256.runtime import Stream
from scripts.replay_native256 import match_onsets

COMPOSITIONS = ("BN1","Funk2","Jazz1","Rock3","SS1")
EXPECTED_NATIVE = (8099,4600,3168)
EXPECTED_SHADOW = (8113,4603,3154)
HARMONICS = (12,19,24,28)
SCHEMA = "ONSET_S613_S701_CAUSAL_HARMONIC_EVENTS_PCM256_V1"

def file_sha(path):
    d=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()

def register(pitch):
    return ("E2_B2" if pitch < 48 else
            "C3_B3" if pitch < 60 else "C4_E6")

def texture(name):
    if name.endswith("_comp"):return "comp"
    if name.endswith("_solo"):return "solo"
    raise ValueError("unknown GuitarSet texture")

def score(tp,fp,fn):
    d=2*tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/d if d else 0.)

def annotate(decisions, truth, policy, track, composition, writer):
    intents_field, evidence_field = (
        ("intents","evidence") if policy=="native" else
        ("candidate_intents","candidate_evidence"))
    pairs=[]
    for decision in decisions:
        intents=getattr(decision,intents_field)
        context=getattr(decision,evidence_field)
        if len(intents)!=len(context):
            raise AssertionError("event evidence lost or duplicate event")
        for event, features in zip(intents,context):
            if int(event.pitch)!=features["pitch"] or int(event.sample)!=features["sample"]:
                raise AssertionError("evidence timestamp/pitch out of sync")
            pairs.append((event,features))
    pred=[e for e,_ in pairs]
    matched=dict(match_onsets(pred,truth))
    if len(matched)!=len(set(matched.values())):
        raise AssertionError("duplicate target assignment")
    current=dict(tp=len(matched),fp=len(pred)-len(matched),fn=len(truth)-len(matched))
    seen_gt=set(matched.values())
    tex=texture(track)
    rows=[]
    for i,(e,features) in enumerate(pairs):
        label="TP" if i in matched else "FP"
        t=e.sample/SR
        # Labels and harmonic-neighbor truth are computed only POST-inference.
        nearby_harmonic_truth=any(
            0 <= e.pitch-int(n["pitch"]) in HARMONICS and
            abs(t-float(n["time"])) <= .080
            for n in truth)
        evidence=dict(features)
        evidence.update(schema=SCHEMA,track=track,composition=composition,
                        texture=tex,policy=policy,register=register(e.pitch),
                        label=label,
                        matched_truth_index=(matched.get(i)),
                        harmonic_lower_pitch_in_truth_80ms=nearby_harmonic_truth)
        writer.write(json.dumps(evidence,sort_keys=True)+"\n")
        rows.append(evidence)
    for i,n in enumerate(truth):
        if i not in seen_gt:
            writer.write(json.dumps(dict(schema=SCHEMA,track=track,
                 composition=composition,texture=tex,policy=policy,
                 pitch=int(n["pitch"]),truth_time=float(n["time"]),
                 register=register(int(n["pitch"])),label="FN"),sort_keys=True)+"\n")
    return current,rows

def aggregate(records):
    collected=defaultdict(lambda:defaultdict(int))
    for r in records:
        key=(r["policy"],r["texture"],r["register"],
             "parent_stronger" if r["harmonic_parent_stronger"] else "parent_not_stronger")
        d=collected[key]
        d[r["label"]]+=1
        if r["harmonic_lower_pitch_in_truth_80ms"]:
            d["harmonic_truth_neighbor_80ms"]+=1
        if r["context"]["strongest_lower_fundamental"]>r["context"]["own_fundamental"]:
            d["spectral_parent_stronger"]+=1
    return [dict(policy=p,texture=t,register=g,harmonic_group=h,
                 tp=d["TP"],fp=d["FP"],n=d["TP"]+d["FP"],
                 harmonic_truth_neighbor_80ms=d["harmonic_truth_neighbor_80ms"])
            for (p,t,g,h),d in sorted(collected.items())]

def run(audio_zip,traces,manifest,native_model,s701_model,out):
    if out.exists():
        raise FileExistsError("scientific audit evidence is immutable")
    m=json.loads(Path(manifest).read_text())
    if m.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not m.get("complete"):
        raise ValueError("unsigned FIT80 manifest")
    if len(m.get("fit_files",[]))!=80:
        raise ValueError("signed FIT80 must contain precisely 80 recordings")
    if any(m.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):
        raise ValueError("other evaluation sets forbidden")
    native,s701=load(native_model,s701_model)
    if HOP!=256:
        raise AssertionError("must use trained native 256-sample grid")
    out.mkdir(parents=True)
    rows=[]
    event_evidence=[]
    with zipfile.ZipFile(audio_zip) as z, (out/"events.jsonl").open("w") as writer:
        for comp in COMPOSITIONS:
            items=sorted([v for v in m["fit_files"] if v["composition"]==comp],
                        key=lambda v:v["index"])
            if len(items)!=16:raise AssertionError("wrong GuitarSet composition count")
            for v in items:
                md=Path(traces)/f"fit_{v['index']:03d}.json"
                if file_sha(md)!=v["json_sha256"]:
                    raise ValueError("S800 signed trace SHA mismatch")
                meta=json.loads(md.read_text())
                if v["name"]!=meta["name"]:raise ValueError("wrong signed audio identity")
                rate,pcm=wavfile.read(io.BytesIO(z.read(meta["name"]+"_mic.wav")))
                if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                    raise ValueError("not original signed PCM waveform")
                audio=pcm[:meta["samples"]].astype(np.float32)/32768
                ends=np.arange(HOP,len(audio)+1,HOP)
                stream=Stream(native,s701,audit_events=True)
                decisions=[]
                for start in range(0,len(audio),HOP):
                    obs=stream.push(audio[start:start+HOP],start_sample=start)
                    expected=int(start+HOP<=len(audio))
                    if len(obs)!=expected:
                        raise AssertionError("fresh trained Onset missing PCM256 call")
                    decisions.extend(obs)
                if (len(decisions)!=len(ends) or stream.calls!=len(ends)
                    or stream.s701_calls!=len(ends)):
                    raise AssertionError("not exactly 1 S613 + S701 call per 256 samples")
                if any(d.sample!=int(end) for d,end in zip(decisions,ends)):
                    raise AssertionError("noncontiguous or backdated Onset")
                stream.finish(meta["samples"])
                # Never consult GT until the full track's audio-only inference.
                truth=meta["fit_truth"]
                counts,base=annotate(decisions,truth,"native",meta["name"],comp,writer)
                shadow,changed=annotate(decisions,truth,"s701_shadow",meta["name"],comp,writer)
                event_evidence.extend(base)
                event_evidence.extend(changed)
                rows.append(dict(index=v["index"],name=meta["name"],
                                 composition=comp,texture=texture(meta["name"]),
                                 frames=len(ends),native=counts,shadow=shadow))
                if len(rows)%8==0:
                    print("ONSET_HARMONIC_AUDIT_PROGRESS",len(rows),flush=True)
    def totals(policy):
        c={k:sum(r[policy][k] for r in rows) for k in ("tp","fp","fn")}
        return score(**c)
    native_score,shadow_score=totals("native"),totals("shadow")
    if (native_score["tp"],native_score["fp"],native_score["fn"])!=EXPECTED_NATIVE:
        raise AssertionError("S613 PCM256 matched counts changed unexpectedly")
    if (shadow_score["tp"],shadow_score["fp"],shadow_score["fn"])!=EXPECTED_SHADOW:
        raise AssertionError("S701 PCM256 matched counts changed unexpectedly")
    if sum(r["frames"] for r in rows)!=306536:
        raise AssertionError("signed FIT80 frame count mismatch")
    by_texture={}
    for t in ("comp","solo"):
        records=[r for r in rows if r["texture"]==t]
        by_texture[t]=dict(tracks=len(records),
            native=score(**{k:sum(v["native"][k] for v in records) for k in ("tp","fp","fn")}),
            shadow=score(**{k:sum(v["shadow"][k] for v in records) for k in ("tp","fp","fn")}))
    if any(v["tracks"]!=40 for v in by_texture.values()):
        raise AssertionError("wrong solo/comp split")
    grouped=aggregate(event_evidence)
    report=dict(schema=SCHEMA,complete=True,tracks=len(rows),
        frames=sum(r["frames"] for r in rows),native=native_score,
        s701_shadow=shadow_score,by_texture=by_texture,
        per_event_harmonic_groups=grouped,per_track=rows,
        checkpoints=dict(s613_sha256=file_sha(native_model),s701_sha256=file_sha(s701_model)),
        historical_H0_onset_f1=2*8528/(2*8528+3489+2739),
        training_hop_samples=256,decision_hop_samples=256,
        false_positive_harmonic_attribution_proven=False,
        labels_read_only_after_track_inference=True,
        train_test_overlap_known=True,
        independent_outer_validation=False,promoted=False,
        public_midi_modified=False,other_axes_modified=False,
        hardware_latency_measured=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_HARMONIC_256_FORENSIC_FINAL",
          json.dumps({k:report[k] for k in ("tracks","frames","native","s701_shadow","by_texture","per_event_harmonic_groups")}),
          flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for flag in ("audio-zip","traces","manifest","native-model","s701-model","out"):
        p.add_argument("--"+flag,type=Path,required=True)
    args=p.parse_args()
    run(args.audio_zip,args.traces,args.manifest,args.native_model,args.s701_model,args.out)
if __name__=="__main__":
    main()
