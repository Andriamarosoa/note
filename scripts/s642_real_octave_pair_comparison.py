"""S642 — paired REAL GuitarSet guitar recordings: one note vs octave duo.

E2/E3 can't be cleanly measured from S613 GuitarSet TRAIN: 0 isolated E2.
Use closest exact-octave physical analogue with isolated TRAIN sources:
E3 (MIDI52) alone versus the SAME E3 plus independently played E4 (MIDI64).

REAL GuitarSet individual-note snippets are mixed at PCM level only;
those mixes are *controlled constructed audio*, not existing guitar
performance recordings. Never report this as held-out natural audio F1.
Original NOTE model and MIDI LIVE untouched; TEST/DEV not accessed.

Measure physical spectral patterns and actual NOTE NoteOn decisions for:
- E3 single
- E4 single
- E3 + E4 concurrent (both onset at .65sec)
- E3, then E4 added 120ms later (independent onset)
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from causal_note.strict_live import SR,HOP,Selection,CausalSelection,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,raw_guitar_audio
from scripts.s613_prepare_strict_live import labels,download,digest
from scripts.s613_train_strict_live import real_notes,event_score
from scripts.s640_learned_overlap_rescue import iter_acoustic
from scripts.s642_annotation_probe import eligible

LOW=52
HIGH=64
BASE_T=.65
DELAY=.12
DURATION=1.65
EXAMPLES=8
# Both the primary note and the secondary sampled source are
# real GuitarSet audio and labelled real MIDI note pitch.
CASES=("E3_alone","E4_alone","E3_plus_E4_simultaneous","E3_then_E4_120ms")
POLICIES=("native_S614","bass_080_ungated")
EVAL_TOL=.080


def normalize_loudness(sound,*,rms_target=.035):
    """Match target steady onset loudness across conditions.

    Clamp prevents clip artefacts. No standardization on annotations.
    """
    s=np.asarray(sound,np.float32).copy()
    section=s[round((BASE_T+.045)*SR):round((BASE_T+.205)*SR)]
    rms=float(np.sqrt(np.mean(section*section)))
    if rms<1e-5:raise ValueError("Guitar clip has no audible attack")
    s*=rms_target/rms
    peak=float(np.max(np.abs(s)))
    if peak>.9:s*=.9/peak
    edge=round(.012*SR)
    s[:edge]*=np.linspace(0.,1.,edge,dtype=np.float32)
    s[-edge:]*=np.linspace(1.,0.,edge,dtype=np.float32)
    return s


def source_clip(wave,onset):
    """1.65s real source, unaltered timing; target attacks .65sec."""
    start=int(round((float(onset)-BASE_T)*SR))
    n=int(round(DURATION*SR))
    if start<0 or start+n>len(wave):raise ValueError("Insufficient physical note audio")
    return wave[start:start+n].copy()


def clip_has_other_true_note(truth,pitch,onset):
    # No other pitch active at onset or beginning of the analysis.
    return any(n["pitch"]!=pitch and n["time"]<onset+.12
               and n["time"]+n["duration"]>onset-.05 for n in truth)


def select_real_examples(protocol,root,archive):
    low=[];high=[];all_counts=Counter()
    for s in protocol["sources"]:
        if s["corpus"]!="GuitarSet" or s["split"]!="train":continue
        truth=real_notes(labels(s,root)["notes"],24.)
        chosen={pitch:eligible(truth,pitch) for pitch in (LOW,HIGH)}
        all_counts[LOW]+=len(chosen[LOW]);all_counts[HIGH]+=len(chosen[HIGH])
        if not (chosen[LOW] or chosen[HIGH]):continue
        wave,sha=raw_guitar_audio(s,archive)
        for pitch,collection in ((LOW,low),(HIGH,high)):
            if not chosen[pitch]:continue
            onset=float(chosen[pitch][0])
            if clip_has_other_true_note(truth,pitch,onset):
                raise AssertionError("GuitarSet annotation not physically isolated")
            x=normalize_loudness(source_clip(wave,onset))
            collection.append(dict(name=s["name"],composition=s["composition"],
                                   onset=onset,sha256=sha,pitch=pitch,wave=x))
    low.sort(key=lambda c:(c["composition"],c["name"]))
    high.sort(key=lambda c:(c["composition"],c["name"]))
    if len(low)<5 or len(high)<5:
        raise ValueError(f"Not enough real isolated E3/E4 WAV candidates: {len(low)}, {len(high)}")
    # Prefer different compositions for unbiased cross-source pairing.
    selected=[]
    for n,a in enumerate(low):
        b=next((v for v in high if v["composition"]!=a["composition"]
                and v["name"] not in {p[1]["name"] for p in selected}),None)
        if b is None:break
        selected.append((a,b))
        if len(selected)==EXAMPLES:break
    if len(selected)<5:
        raise ValueError("Need >=5 independent octave pairs without same source")
    return selected,dict(isolated_guitarset_train_E3=all_counts[LOW],
                         isolated_guitarset_train_E4=all_counts[HIGH],
                         selected_pairs=len(selected))


def conditions(low,high):
    """Same original low E3 waveform in BOTH lone and overlapped cases.

    Added E4 is actual isolated guitar at the same physical onset, or
    120 ms later. Mix RMS matched to sole note control, not merely louder.
    """
    a=low["wave"];b=high["wave"]
    mix=a+b
    shift=round(DELAY*SR)
    later=a.copy()
    later[shift:]+=b[:-shift]
    sounds={
        "E3_alone":a.copy(),
        "E4_alone":b.copy(),
        "E3_plus_E4_simultaneous":normalize_loudness(mix),
        "E3_then_E4_120ms":normalize_loudness(later),
    }
    truth={
        "E3_alone":[dict(time=BASE_T,duration=.35,pitch=LOW)],
        "E4_alone":[dict(time=BASE_T,duration=.35,pitch=HIGH)],
        "E3_plus_E4_simultaneous":[dict(time=BASE_T,duration=.35,pitch=LOW),
                                  dict(time=BASE_T,duration=.35,pitch=HIGH)],
        "E3_then_E4_120ms":[dict(time=BASE_T,duration=.35,pitch=LOW),
                            dict(time=BASE_T+DELAY,duration=.35,pitch=HIGH)],
    }
    return sounds,truth


def _spectral_fingerprint(x):
    """S613 physical log harmonics and two-lag pitch onset at current time."""
    return dict(
        low_E3_f0=np.mean([x[LOW-40,8*j] for j in range(3)]).item(),
        high_E4_f0=np.mean([x[HIGH-40,8*j] for j in range(3)]).item(),
        high_E4_over_low_E3_f0=(np.mean([x[HIGH-40,8*j] for j in range(3)])
                               -np.mean([x[LOW-40,8*j] for j in range(3)])).item(),
        high_E4_onset_novelty=np.mean([x[HIGH-40,57+j] for j in range(3)]).item(),
        low_E3_onset_novelty=np.mean([x[LOW-40,57+j] for j in range(3)]).item(),
        high_E4_vs_low_E3_onset_novelty=(np.mean([x[HIGH-40,57+j]
                                      -x[LOW-40,57+j] for j in range(3)])).item(),
        high_E4_harmonic3=np.mean([x[HIGH-40,8*j+2] for j in range(3)]).item(),
        high_E4_native_onset=float('nan'), # overwritten by head
    )


def eval_pcm(sound,truth,heads,config):
    states={name:CausalSelection(config) for name in POLICIES}
    events={name:[] for name in POLICIES}
    sampled=[]
    for end,x,p,k,bass,valid in iter_acoustic(sound,heads,batch_frames=64):
        time=end/SR
        for case,state in states.items():
            probs=p
            if case=="bass_080_ungated" and valid:
                probs=p.copy()
                mask=(np.arange(49)<24)&(~state.active)&state.armed&(p[:,0]<config.onset)&(bass[:,0]>=.8)&(bass[:,1]>=.5)
                probs[mask,0]=config.onset
                probs[mask,1]=np.maximum(probs[mask,1],bass[mask,1])
            events[case].extend(state.step(probs,k,int(end),int(end)))
        if BASE_T+.025 <= time<=BASE_T+.195 or BASE_T+DELAY+.020<=time<=BASE_T+DELAY+.12:
            fp=_spectral_fingerprint(x)
            fp["high_E4_native_onset"]=float(p[HIGH-40,0])
            fp["low_E3_native_onset"]=float(p[LOW-40,0])
            fp["high_E4_native_sustain"]=float(p[HIGH-40,1])
            fp["low_E3_native_sustain"]=float(p[LOW-40,1])
            fp["high_E4_bass_remap_onset"]=float(bass[HIGH-40,0])
            fp["low_E3_bass_remap_onset"]=float(bass[LOW-40,0])
            fp["time"]=float(time)
            sampled.append(fp)
    per={}
    for case,state in states.items():
        events[case].extend(state.panic(len(sound)))
        score,pairs=event_score(events[case],truth,tolerance=EVAL_TOL)
        ons=[e for e in events[case] if e["type"]=="note_on"]
        per[case]=dict(
            TP=score["TP"],FP=score["FP"],FN=score["FN"],
            matched_pitch={str(n):sum(1 for _,j in pairs if truth[j]["pitch"]==n)
                           for n in (LOW,HIGH)},
            predicted_E3=sum(e["pitch"]==LOW for e in ons),
            predicted_E4=sum(e["pitch"]==HIGH for e in ons),
            predicted_other=sum(e["pitch"] not in (LOW,HIGH) for e in ons),
            events=[dict(pitch=e["pitch"],time=round(e["time"],3)) for e in ons],
        )
    def fingerprint(lo,hi):
        rows=[r for r in sampled if lo<=r["time"]<=hi]
        if not rows:return {}
        return {k:round(float(np.mean([r[k] for r in rows])),5)
                for k in rows[0] if k!="time"}
    return dict(predictions=per,
                physical_onset_E3=fingerprint(BASE_T+.04,BASE_T+.12),
                physical_post_delay=fingerprint(BASE_T+DELAY+.025,BASE_T+DELAY+.10))


def run(out):
    if out.exists():raise FileExistsError("Immutable S642 results")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    p=json.loads((ROOT/"protocol.json").read_text())
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    records=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=p["guitar_archives"][key]
            download(item["url"],root/"data"/path,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            pairs,coverage=select_real_examples(p,root,archive)
            print("S642 SELECT",json.dumps(coverage),flush=True)
            for i,(low,high) in enumerate(pairs,1):
                waves,truth=conditions(low,high)
                cases={}
                for name in CASES:
                    cases[name]=eval_pcm(waves[name],truth[name],heads,config)
                records.append(dict(low_source=low["name"],high_source=high["name"],
                                    low_sha=low["sha256"],high_sha=high["sha256"],cases=cases))
                print("S642 PAIR",i,"/",len(pairs),
                      low["name"],high["name"],
                      "S614 native pitch E3/E4",{
                          c:cases[c]["predictions"]["native_S614"]["matched_pitch"]
                          for c in CASES},flush=True)
    agg={}
    for cond in CASES:
        policies={}
        for policy in POLICIES:
            vals=[row["cases"][cond]["predictions"][policy] for row in records]
            summed={k:sum(v[k] for v in vals)
                    for k in ("TP","FP","FN","predicted_E3","predicted_E4","predicted_other")}
            summed["E3_correct"]=sum(v["matched_pitch"][str(LOW)] for v in vals)
            summed["E4_correct"]=sum(v["matched_pitch"][str(HIGH)] for v in vals)
            summed["expected_noteon"]=len(records)*(2 if cond.startswith("E3_plus") or cond.startswith("E3_then") else 1)
            policies[policy]=summed
        physical={}
        for region in ("physical_onset_E3","physical_post_delay"):
            for field in ("high_E4_f0","low_E3_f0","high_E4_over_low_E3_f0",
                          "high_E4_onset_novelty","low_E3_onset_novelty",
                          "high_E4_vs_low_E3_onset_novelty",
                          "high_E4_native_onset","low_E3_native_onset"):
                v=[row["cases"][cond][region][field] for row in records
                   if field in row["cases"][cond][region]]
                if v:
                    physical[f"{region}.{field}"]=float(np.mean(v))
        agg[cond]=dict(policies=policies,physical=physical)
        print("S642 CONDITION",cond,json.dumps(dict(policies=policies,
            physical_just_main={k:round(v,4) for k,v in physical.items()
                                if k.endswith(("high_E4_f0","high_E4_onset_novelty",
                                               "high_E4_native_onset"))})),flush=True)
    report=dict(
        status="S642_REAL_GUITARSET_ANNOTATED_ISOLATED_OCTAVE_PAIR_MIX_TEST",
        example_E2_E3_not_cleanly_available=True,
        actual_pair="E3 MIDI52 164.8Hz vs E3+E4 MIDI64 329.6Hz",
        instrumental_source="original real GuitarSet guitar microphone recordings, TRAIN only",
        experimental_stimuli="digitally mixed isolated real guitar snippets (not naturally performed simultaneous guitar chords)",
        music_truth_of_stimuli_known_from_constructed_mixes=True,
        run_is_diagnostic_not_official_F1=True,
        TOLERANCE_MS=EVAL_TOL*1000,
        native_S614_frozen=True, native_K_unchanged=True,
        test_and_dev_not_used=True,
        no_future_audio=True,
        model_sha=frozen["model_sha256"],
        source_inventory=coverage, pairs=len(records),
        cases=agg,per_pair=records)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S642 FINAL",json.dumps({k:v["policies"] for k,v in agg.items()}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(a.out)
