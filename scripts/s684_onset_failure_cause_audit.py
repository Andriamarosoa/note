"""S684 measurement: WHY native S630/S675 causal NOTE Onset misses/false attacks.

TRUE GuitarSet MIC only, no model updates or audio future in inference. This
script is diagnostic: it inspects ground truth ±80ms to categorize failures.
It does not backdate outputs, train, or alter the deployed Live branch.
Baseline NoteOff, original pitch predictions and native K are unmodified.
"""
from __future__ import annotations
import argparse,json,tempfile,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    ROOT,MODEL,raw_guitar_audio,native_posteriors
)


def register(pitch):
    return "bass_E2_B2" if pitch<=47 else "mid_C3_B3" if pitch<=59 else "treble_C4_E6"


def classify(item, selection):
    """Failure categories are first-match distinct; all use REAL annotations
    for audit only, never as selection inputs."""
    truth=item["truth"]
    events=item["events"]
    ends=item["ends"]
    post=item["p"]
    _,paired=event_score(events,truth,tolerance=.080)
    on=[e for e in events if e["type"]=="note_on"]
    pset=set(i for i,_ in paired)
    tset=set(j for _,j in paired)
    fn=Counter(); fp=Counter(); register_by=defaultdict(Counter)
    detailed=[]
    for j,note in enumerate(truth):
        if j in tset: continue
        pitch=int(note["pitch"]);nidx=pitch-40;t=float(note["time"])
        lo=max(0,int(np.searchsorted(ends,(t-.012)*SR,"left")))
        hi=min(len(ends),int(np.searchsorted(ends,(t+.08)*SR,"right")))
        if hi<=lo:continue
        onset=post[lo:hi,nidx,0]
        sustain=post[lo:hi,nidx,1]
        high=(onset>=selection.onset)&(sustain>=selection.active_on)
        any_onset=bool(np.any(onset>=selection.onset))
        any_sustain=bool(np.any(sustain>=selection.active_on))
        near_other=[e for e in on if abs(e["time"]-t)<=.08 and
                    abs(int(e["pitch"])-pitch) in (12,19,24)]
        near_same=[e for e in on if e["pitch"]==pitch and
                   abs(e["time"]-t)<.25]
        if not any_onset:
            category="onset_posterior_below_threshold"
        elif not any_sustain:
            category="sustain_posterior_below_threshold"
        elif not high.any():
            category="onset_and_sustain_not_simultaneous"
        elif near_same:
            category="same_pitch_attack_outside_80ms"
        else:
            category="selector_K_refractory_rearm_or_confirm"
        fn[category]+=1
        reg=register(pitch)
        register_by[reg][category]+=1
        if near_other:
            fn["near_octave_or_harmonic_noteon"]+=1
        detailed.append(dict(pitch=pitch,ground_time=t,
            category=category,peak_onset=round(float(onset.max()),4),
            peak_sustain=round(float(sustain.max()),4),
            octave_neighbor=bool(near_other)))
    for i,event in enumerate(on):
        if i in pset:continue
        pitch=int(event["pitch"]);t=event["time"]
        # A native NoteOn is 'harmonic adjacent' if its expected harmonic
        # fundamental is already present in a distinct labelled note.
        near_harmonic=any(
            abs(int(n["pitch"])-pitch) in (12,19,24)
            and n["time"]-.06<=t<=n["time"]+n["duration"]+.08
            for n in truth)
        same_early_or_late=any(
            n["pitch"]==pitch and abs(n["time"]-t)<.25 for n in truth)
        if near_harmonic:label="harmonic_competitor_near_gt"
        elif same_early_or_late:label="true_pitch_wrong_attack_time"
        else:label="isolated_unmatched_attack"
        fp[label]+=1
        register_by[register(pitch)]["FP_"+label]+=1
    baseline,_=event_score(events,truth,tolerance=.080)
    assert sum(v for k,v in fn.items() if k!="near_octave_or_harmonic_noteon")==baseline["FN"]
    assert sum(fp.values())==baseline["FP"]
    return dict(baseline={k:v for k,v in baseline.items() if k!="latency_ms"},
                true_count=len(truth),FN_types=dict(fn),FP_types=dict(fp),
                by_register={k:dict(v) for k,v in register_by.items()},
                failures=detailed)


def run(out):
    if out.exists():raise FileExistsError("Immutable report path exists")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    sources=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and
             s["split"] in ("train","dev")]
    assert len(sources)==48
    traincomp=sorted({s["composition"] for s in sources if s["split"]=="train"})
    assert len(traincomp)==9
    held=set(traincomp[5:])
    chosen=[s for s in sources if s["split"]=="dev" or s["composition"] in held]
    assert len(chosen)==28
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    result=dict(schema="S684_AUDIT_ONLY",native_original_sha256=frozen["model_sha256"],
        no_changes_to_heads_K_pitch_onset_offset=True,
        no_inference_uses_true_labels=True,
        training_held_compositions=sorted(held),
        TEST_unopened=True,DEV_previously_exposed=True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=proto["guitar_archives"][key]
            download(src["url"],root/"data"/path,src["size"],src["sha256"])
        rows=[]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for i,spec in enumerate(chosen,1):
                audio,_=raw_guitar_audio(spec,archive)
                true=real_notes(labels(spec,root)["notes"],len(audio)/SR)
                ends,feat=feature_track(audio,block_size=4096)
                p,k=native_posteriors(feat,heads)
                events=decode(ends,p,k,config,len(audio))
                row=classify(dict(truth=true,events=events,ends=ends,p=p),config)
                rows.append(dict(split=spec["split"],name=spec["name"],
                                 composition=spec["composition"],**row))
                print("S684 AUDIT",i,"/",len(chosen),spec["name"],
                      row["baseline"],flush=True)
    for split in ("train","dev"):
        subset=[r for r in rows if r["split"]==split]
        counts=Counter();errors=Counter();pitch=defaultdict(Counter)
        metrics=Counter()
        for row in subset:
            metrics.update({x:row["baseline"][x] for x in ("TP","FP","FN")})
            counts.update(row["FN_types"])
            errors.update(row["FP_types"])
            for k,v in row["by_register"].items():pitch[k].update(v)
        f1=2*metrics["TP"]/max(1,2*metrics["TP"]+metrics["FP"]+metrics["FN"])
        result[split]=dict(recordings=len(subset),metrics=dict(metrics),
                           F1=f1,FN_types=dict(counts),FP_types=dict(errors),
                           by_register={k:dict(v) for k,v in pitch.items()})
        print("S684",split.upper(),"CONCLUSION",json.dumps(result[split]),flush=True)
    result["by_recording"]=rows
    result["proposal"]="Use TRAIN-held FN and FP decomposition to choose one targeted strictly-causal veto or rescue. No automatic promotion."
    out.mkdir(parents=True)
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S684 FINAL",json.dumps({s:result[s] for s in ("train","dev")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.out)
