"""S652: causal Tone-control-like LOWPASS multiview hypothesis vs original NOTE.

This is a digital FIRST-ORDER lowpass approximation, NOT a measurement
of the Stratocaster passive Tone circuit (pickup impedance, capacitor,
potentiometer, resonances, guitar cable and player matter). A passive
Tone control does not pitch-shift and this experiment adds NO delay.

Compare causal PCM views: original, mild lowpass fc=2800Hz, dark
lowpass fc=1100Hz. Each passes through the exact frozen NOTE S614
acoustic frontend+pitch head. The original K posterior is preserved
for ALL output selection policies to prevent changes to Exact-K.

The stream is filtered using scipy sosfilt (FORWARD ONLY), then run
through native causal features. The three sample clocks match at each
256-sample hop; no future audio/no zero-phase filtfilt/no time-shifting.

Nine TRAIN composition groups: first 5 (sorted) choose one optional
selection policy by F1; remaining four are an independently held
TRAIN-check with full composition isolation; only afterwards examine
12 GuitarSet DEV (historically exposed). Never read TEST sources.
No modification to NOTE original main/LIVE code.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy.signal import butter,sosfilt

from causal_note.strict_live import HOP,SR,Selection,CausalSelection,feature_track,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,raw_guitar_audio,GUITARSET_PINNED_CONTROL
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups

FILTERS={"tone_mild_2800":2800.,"tone_dark_1100":1100.}
POLICIES=("S614_original","tone_mild_2800","tone_dark_1100",
          "avg_original_mild","avg_three",
          "vote_2of3","original_AND_mild","original_AND_dark",
          "original_OR_mild_070","original_OR_mild_085",
          "original_OR_dark_070","original_OR_dark_085",
          "original_OR_both_confirmed")
# Fixed choices BEFORE looking at GuitarSet DEV or the internal TRAIN holdout.
REQUIRE_MIN_CALIBRATION_F1_GAIN=.004
REQUIRE_MIN_HELD_GAIN=.002
REQUIRE_POSITIVE_HELD_GROUPS=3


def causal_lowpass(audio,cutoff_hz):
    """One-pole Butterworth LOWPASS, forward causal, no lookahead."""
    if cutoff_hz not in FILTERS.values():
        raise ValueError("Only pre-registered Tone proxy settings allowed")
    x=np.asarray(audio,np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid mono PCM")
    sos=butter(1,cutoff_hz,fs=SR,btype="lowpass",output="sos")
    y=sosfilt(sos,x).astype(np.float32)
    if len(y)!=len(x) or not np.isfinite(y).all():
        raise AssertionError("Causal lowpass sample alignment")
    return y


def predict_causal_view(audio,heads):
    """Same native NOTE frontend and head; memory bounded per model batch."""
    ends,features=feature_track(audio,block_size=4096)
    if features.shape[1:]!=(49,126):
        raise AssertionError("Native NOTE 49x126 acoustic ABI")
    chunks=[heads.predict(features[i:i+128]) for i in range(0,len(features),128)]
    p=np.concatenate([x for x,_ in chunks]).astype(np.float32)
    k=np.concatenate([x for _,x in chunks]).astype(np.float32)
    assert p.shape==(len(ends),49,3) and k.shape==(len(ends),7)
    return ends,p,k


def probabilities(policy,orig,mild,dark,cutoff):
    """All transforms are physical-hop-local, never annotation-driven.

    Native K and NoteOn state are reused unmodified. Morphological
    rescue requires a HIGH-confidence filtered note onset, not just
    a spectral bin to appear.
    """
    if policy=="S614_original":return orig
    if policy=="tone_mild_2800":return mild
    if policy=="tone_dark_1100":return dark
    if policy=="avg_original_mild":return .5*(orig+mild)
    if policy=="avg_three":return (orig+mild+dark)/3.
    if policy=="vote_2of3":
        strong=[(v[:,0]>=cutoff.onset)&(v[:,1]>=cutoff.active_on)
                for v in (orig,mild,dark)]
        yes=(strong[0].astype(int)+strong[1].astype(int)+strong[2].astype(int))>=2
        out=orig.copy()
        out[:,0]=np.where(yes,np.maximum.reduce(
            [orig[:,0],mild[:,0],dark[:,0]]),0.)
        out[:,1]=np.where(yes,np.maximum.reduce(
            [orig[:,1],mild[:,1],dark[:,1]]),0.)
        return out
    if policy in ("original_AND_mild","original_AND_dark"):
        f=mild if policy.endswith("mild") else dark
        out=orig.copy()
        agree=(orig[:,0]>=cutoff.onset)&(orig[:,1]>=cutoff.active_on)&(
            f[:,0]>=cutoff.onset)&(f[:,1]>=cutoff.active_on)
        out[:,0]=np.where(agree,orig[:,0],0.)
        return out
    if policy.startswith("original_OR_"):
        out=orig.copy()
        if policy=="original_OR_both_confirmed":
            extra=(mild[:,0]>=.85)&(dark[:,0]>=.85)&(
                mild[:,1]>=cutoff.active_on)&(dark[:,1]>=cutoff.active_on)
            strongest=np.maximum(mild[:,0],dark[:,0])
            sustain=np.maximum(mild[:,1],dark[:,1])
        else:
            view=mild if "_mild_" in policy else dark
            floor=.70 if policy.endswith("_070") else .85
            extra=(view[:,0]>=floor)&(view[:,1]>=cutoff.active_on)
            strongest=view[:,0];sustain=view[:,1]
        missing=extra & (orig[:,0]<cutoff.onset)
        out[missing,0]=np.maximum(out[missing,0],strongest[missing])
        out[missing,1]=np.maximum(out[missing,1],sustain[missing])
        return out
    raise ValueError(f"Unknown policy {policy}")


def exact_k_at_annotated_onsets(ends,preds,truth):
    """Count-head diagnostic; no K information from labels at inference."""
    by_pitch=Counter()
    correct=Counter()
    for note in truth:
        t=float(note["time"])+.030
        if t>=ends[-1]/SR:continue
        target=min(6,len({int(other["pitch"]) for other in truth
                 if other["time"]<=t and other["time"]+other["duration"]>t}))
        if not target:continue
        idx=int(np.searchsorted(ends,round(t*SR),side="left"))
        idx=min(idx,len(ends)-1)
        by_pitch[str(target)]+=1
        if int(np.argmax(preds[idx]))==target:correct[str(target)]+=1
    return dict(total=sum(by_pitch.values()),correct=sum(correct.values()),
                by_true_k=dict(by_pitch),correct_by_true_k=dict(correct))


def measure_one(spec,root,archive,heads,cfg):
    audio,sha=raw_guitar_audio(spec,archive)
    gt=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,native,native_k=predict_causal_view(audio,heads)
    tone_predictions={}
    tone_counts={}
    for name,fc in FILTERS.items():
        tone=causal_lowpass(audio,fc)
        filtered_ends,p,k=predict_causal_view(tone,heads)
        if not np.array_equal(filtered_ends,ends):
            raise AssertionError("Tone view changed NoteOn/Exact-K time grid")
        tone_predictions[name]=p
        tone_counts[name]=k
    decoders={name:CausalSelection(cfg) for name in POLICIES}
    emitted={name:[] for name in POLICIES}
    for z,end in enumerate(ends):
        orig=native[z];mild=tone_predictions["tone_mild_2800"][z]
        dark=tone_predictions["tone_dark_1100"][z]
        for name in POLICIES:
            p=probabilities(name,orig,mild,dark,cfg)
            emitted[name].extend(decoders[name].step(p,native_k[z],int(end),int(end)))
    groups=overlap_groups(gt)
    m={}
    native_seen=None
    for name,decoder in decoders.items():
        emitted[name].extend(decoder.panic(len(audio)))
        measure,pairs=event_score(emitted[name],gt,tolerance=.080)
        correct={j for _,j in pairs}
        if name=="S614_original":native_seen=correct
        pitch_count=Counter(gt[j]["pitch"] for j in correct)
        m[name]=dict(score=measure,
                     onset_overlap_TP=len(correct&groups["onset_overlap"]),
                     onset_K2_TP=len(correct&groups["onset_K2"]),
                     onset_K3plus_TP=len(correct&groups["onset_K3plus"]),
                     pitch_E2_correct=int(pitch_count.get(40,0)),
                     pitch_E3_correct=int(pitch_count.get(52,0)),
                     matched=correct)
    for name in POLICIES:
        seen=m[name].pop("matched")
        m[name]["new_vs_S614"]=len(seen-native_seen)
        m[name]["lost_vs_S614"]=len(native_seen-seen)
    kstats={"S614_native":exact_k_at_annotated_onsets(ends,native_k,gt)}
    for name,k in tone_counts.items():
        kstats[name+"_diagnostic_only"]=exact_k_at_annotated_onsets(ends,k,gt)
    return dict(name=spec["name"],composition=spec["composition"],sha256=sha,
                original_truth_onsets=len(gt),
                annotated_E2=sum(int(n["pitch"]==40) for n in gt),
                annotated_E3=sum(int(n["pitch"]==52) for n in gt),
                results=m,count_head_diagnostics=kstats,
                no_extra_future_samples=True)


def aggregate(rows):
    out={}
    for name in POLICIES:
        parts=[r["results"][name] for r in rows]
        score=summary([p["score"] for p in parts])
        out[name]=dict(score=score,
                       overlap_TP=sum(p["onset_overlap_TP"] for p in parts),
                       K2_TP=sum(p["onset_K2_TP"] for p in parts),
                       K3plus_TP=sum(p["onset_K3plus_TP"] for p in parts),
                       E2_TP=sum(p["pitch_E2_correct"] for p in parts),
                       E3_TP=sum(p["pitch_E3_correct"] for p in parts),
                       new_vs_native=sum(p["new_vs_S614"] for p in parts),
                       lost_vs_native=sum(p["lost_vs_S614"] for p in parts))
    return out


def choose_on_calibration(calib,held):
    """Fixed policies; train-only two-stage composition-level selection."""
    calibrated=aggregate(calib)
    baseline=calibrated["S614_original"]["score"]
    # Reject any added FP when the corresponding TP gain fails to
    # raise true NoteOn F1. Prefer the *smallest* extra FP on ties.
    winner=max(POLICIES,key=lambda x:(
        calibrated[x]["score"]["F1"],-calibrated[x]["score"]["FP"],
        x=="S614_original"))
    best=calibrated[winner]["score"]
    if best["F1"]<baseline["F1"]+REQUIRE_MIN_CALIBRATION_F1_GAIN:
        winner="S614_original"
    held_scores=aggregate(held)
    base_h=held_scores["S614_original"]["score"]
    trial=held_scores[winner]["score"]
    per_comp={}
    for comp in sorted({r["composition"] for r in held}):
        records=[r for r in held if r["composition"]==comp]
        data=aggregate(records)
        per_comp[comp]=dict(
            F1_native=data["S614_original"]["score"]["F1"],
            F1_selected=data[winner]["score"]["F1"],
            delta_TP=data[winner]["score"]["TP"]-
                     data["S614_original"]["score"]["TP"],
            delta_FP=data[winner]["score"]["FP"]-
                     data["S614_original"]["score"]["FP"])
    positive=sum(v["F1_selected"]>v["F1_native"]+1e-8 for v in per_comp.values())
    confirmed=(winner!="S614_original" and
               trial["F1"]>=base_h["F1"]+REQUIRE_MIN_HELD_GAIN and
               positive>=REQUIRE_POSITIVE_HELD_GROUPS)
    final=winner if confirmed else "S614_original"
    return final,dict(
        selected_by_calibration=winner,
        selected_for_DEV=final,
        calibration_native=baseline,
        calibration_selected=best,
        held_native=base_h,held_selected=trial,
        held_positive_compositions=positive,
        held_per_composition=per_comp,
        fixed_min_calibration_delta=REQUIRE_MIN_CALIBRATION_F1_GAIN,
        fixed_min_held_delta=REQUIRE_MIN_HELD_GAIN)


def sum_k(rows):
    groups={}
    for name in ("S614_native","tone_mild_2800_diagnostic_only",
                 "tone_dark_1100_diagnostic_only"):
        count=Counter();correct=Counter()
        for row in rows:
            diag=row["count_head_diagnostics"][name]
            count.update(diag["by_true_k"])
            correct.update(diag["correct_by_true_k"])
        groups[name]=dict(total=sum(count.values()),correct=sum(correct.values()),
                          accuracy=sum(correct.values())/max(1,sum(count.values())),
                          per_true_K={k:dict(n=int(v),correct=int(correct[k])) for k,v in sorted(count.items())})
    return groups


def run(out):
    if out.exists():raise FileExistsError("S652 evidence path immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    training=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in proto["sources"] if s["split"]=="test"]
    assert len(training)==36 and len(dev)==12 and len({x["composition"] for x in training})==9
    assert not {(s["corpus"],s["composition"]) for s in training}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    all_comp=sorted({s["composition"] for s in training})
    calibration_comps=set(all_comp[:5])
    internal_hold_comps=set(all_comp[5:])
    assert calibration_comps.isdisjoint(internal_hold_comps)
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            archive=proto["guitar_archives"][key]
            download(archive["url"],root/"data"/filename,archive["size"],archive["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            calib=[];held=[]
            for idx,spec in enumerate(training,1):
                r=measure_one(spec,root,arc,heads,cfg)
                (calib if r["composition"] in calibration_comps else held).append(r)
                print("S652 REAL TRAIN",idx,"/36",r["name"],
                      "native",r["results"]["S614_original"]["score"]["TP"],
                      "mild",r["results"]["tone_mild_2800"]["score"]["TP"],
                      flush=True)
            selected,calibration=choose_on_calibration(calib,held)
            print("S652 TRAIN CHOICE",json.dumps(calibration),flush=True)
            measured=[]
            for idx,spec in enumerate(dev,1):
                r=measure_one(spec,root,arc,heads,cfg)
                measured.append(r)
                print("S652 REAL DEV",idx,"/12",r["name"],
                      "originalFP",r["results"]["S614_original"]["score"]["FP"],
                      "filteredFP",r["results"]["tone_dark_1100"]["score"]["FP"],
                      flush=True)
    train_all=aggregate(calib+held)
    dev_result=aggregate(measured)
    original=dev_result["S614_original"]["score"]
    assert all(original[k]==v for k,v in GUITARSET_PINNED_CONTROL.items()),original
    for name,row in dev_result.items():
        sc=row["score"]
        row["delta_F1_points_vs_S614"]=round(100*(sc["F1"]-original["F1"]),6)
        row["delta_TP_vs_S614"]=int(sc["TP"]-original["TP"])
        row["delta_FP_vs_S614"]=int(sc["FP"]-original["FP"])
        print("S652 SCORE",name,
              "TP",sc["TP"],"FP",sc["FP"],"F1",round(sc["F1"],7),
              "E2",row["E2_TP"],"E3",row["E3_TP"],
              "overlap",row["overlap_TP"],"K3",row["K3plus_TP"],flush=True)
    report=dict(
        status="S652_CAUSAL_TONE_FILTER_GUITARSET_MULTIVIEW_NO_PITCH_SHIFT",
        digital_proxy_not_measured_stratocaster_tone=True,
        digital_filters="first_order_Butterworth_sosfilt_forward_only",
        cutoff_hz=FILTERS,original_model_SHA256=frozen["model_sha256"],
        NOTE_S614_and_ExactK_unchanged=True,
        output_selection_K_uses_original_for_all_policies=True,
        no_added_pitch_transposition=True,no_delay_or_future_audio=True,
        added_physical_filter_group_delay_not_measured=True,
        no_official_microphone_LIVE_integration=True,
        guitarset_TRAIN=36,guitarset_DEV=12,
        TEST_never_opened=True,DEV_historically_exposed=True,
        held_out_TRAIN_compositions=list(sorted(internal_hold_comps)),
        calibration_TRAIN_compositions=list(sorted(calibration_comps)),
        train_choice=calibration,chosen_policy_before_DEV=selected,
        calibration_train_aggregates=aggregate(calib),
        held_train_aggregates=aggregate(held),
        complete_train_aggregates=train_all,
        actual_DEV_F1_noteon=dev_result,
        E2_E3_truth_COUNTS={
            "E2":sum(r["annotated_E2"] for r in measured),
            "E3":sum(r["annotated_E3"] for r in measured)},
        DEV_exactK_original_and_filter_view_diagnostics=sum_k(measured),
        DEV_per_recording=measured,
        S641_historically_observed_DEV_F1=0.6339930882814955,
        runtime_seconds=round(time.perf_counter()-started,2),
        automatic_promotion=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S652 FINAL",json.dumps({
        "chosen_train":selected,
        "calibration_choice":calibration["selected_by_calibration"],
        "score":{n:dict(F1=r["score"]["F1"],TP=r["score"]["TP"],
                        FP=r["score"]["FP"],E2=r["E2_TP"],
                        overlap=r["overlap_TP"]) for n,r in dev_result.items()},
        "K_accuracy_diagnostic":{n:r["accuracy"] for n,r in
                                report["DEV_exactK_original_and_filter_view_diagnostics"].items()}
    }),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
