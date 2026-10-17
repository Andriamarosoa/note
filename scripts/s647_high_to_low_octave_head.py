"""S647 -- NOTE high attack candidates for independent low octave rescue.

Goal: rescue LOW fundamentals masked by simultaneous high-octave notes in ORIGINAL NOTE
causal NoteOn selection in a SHADOW decoder, retaining checkpoint and native K.
In contrast to S645, a candidate is born only when S614 itself emits a
LOW NoteOn. The head sees original PCM through its real emission time.

Experimental delays 0/40/80/190ms are rounded UP to 256-sample hops.
Each delay trains its own head on *native NOTE-generated candidates*
from the 36 TRAIN GuitarSet tracks. Composition-level five-fold CV
chooses the abstention threshold BEFORE touching the 12 DEV labels.

Output scores actual, non-backdated NoteOn events from a separate
CausalSelection with unchanged K and unchanged native pitch predictions,
apart from admissible octave proposal injections. Do NOT modify the
official microphone runtime or claim CPU throughput at microphone speed.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR,HOP,Selection,CausalSelection,feature_track,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,GUITARSET_PINNED_CONTROL,raw_guitar_audio
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s644_natural_octave_transfer import truth_case
from scripts.s643_octave_duet_head import NFFT

SEED=646
DELAYS_MS=(0,40,80,190)
THRESHOLDS=(.50,.65,.75,.85,.90,.95,.98,1.01) # 1.01 = abstain
FUTURE_LOOKAHEAD=0
# 256-sample NOTE decision hops. No hidden retroactive events.
def steps_for_delay(delay_ms):
    if delay_ms not in DELAYS_MS:
        raise ValueError("Unknown precommitted S646 delay")
    return int(np.ceil(delay_ms*SR/(1000*HOP)))


def right_edge_signature(wave,pitch,end_sample):
    """Causal spectrum; window ends at measured sample; no future PCM."""
    if not 40<=pitch<=64 or not 0<=end_sample<=len(wave):
        raise ValueError("S646 pitch or sound endpoint out of bounds")
    low_hz=440.*2.**((pitch-69)/12.)
    vals=[pitch/127.]
    for n in NFFT:
        fragment=wave[max(0,end_sample-n):end_sample]
        if len(fragment)<n:
            fragment=np.pad(fragment,(n-len(fragment),0))
        taper=np.hanning(n)
        spectrum=(2./taper.sum())*np.abs(np.fft.rfft(fragment*taper))
        freqbins=np.clip(np.rint(np.arange(1,9)*low_hz*n/SR).astype(int),
                         1,n//2-2)
        h=np.log1p(700*np.array([np.max(spectrum[b-1:b+2]) for b in freqbins]))
        vals.extend(h.tolist())
        vals.extend((float(h[1]-h[0]),float(h[3]-h[1]),
                     float(h[5]-h[2]),float(h[1]-h[2]),
                     float(h[3]-h[4]),float(h[7]-h[5]),
                     float(h[1]+h[3]-h[0]-h[2]),
                     float(h[1]+h[3]-h[2]-h[4]),
                     float(h[1]-max(h[0],h[2])),
                     float(h[3]-max(h[2],h[4]))))
        b=int(freqbins[1])
        neighbor=min(len(spectrum)-2,int(round(b*2**(1/12.))))
        vals.append(float(np.log1p(700*spectrum[b])-
                          np.log1p(700*max(spectrum[neighbor],spectrum[max(1,b-3)]))))
    result=np.asarray(vals,np.float32)
    if result.shape!=(58,) or not np.isfinite(result).all():
        raise ValueError("S646 causal 58D spectral signature")
    return result


def sample_times(low_sample,delay_ms):
    late=low_sample+steps_for_delay(delay_ms)*HOP
    if delay_ms==0:
        earlier=max(0,low_sample-steps_for_delay(40)*HOP)
    elif delay_ms==40:
        earlier=low_sample
    elif delay_ms==80:
        earlier=low_sample+steps_for_delay(40)*HOP
    else:
        earlier=low_sample+int(round(100*SR/(1000*HOP)))*HOP
    assert earlier<=late
    return earlier,late


def octave_features(wave,pitch,low_sample,delay_ms):
    early,end=sample_times(low_sample,delay_ms)
    if end>len(wave):
        return None
    a=right_edge_signature(wave,pitch,early)
    b=right_edge_signature(wave,pitch,end)
    row=np.concatenate((a,b,b[1:]-a[1:]))
    if row.shape!=(173,) or not np.isfinite(row).all():
        raise AssertionError("S646 exact 173D acoustic feature ABI")
    return row.astype(np.float32)


def native_notes_from_audio(wave,heads,config):
    """Original NOTE history and frozen heads, original causal selector only."""
    ends,x=feature_track(wave,block_size=4096)
    chunks=[heads.predict(x[start:start+128]) for start in range(0,len(x),128)]
    p=np.concatenate([a for a,_ in chunks])
    k=np.concatenate([b for _,b in chunks])
    decoder=CausalSelection(config)
    events=[]
    for end,probs,counts in zip(ends,p,k):
        events.extend(decoder.step(probs,counts,int(end),int(end)))
    events.extend(decoder.panic(len(wave)))
    return ends,p,k,events


def candidate_labels(events,truth):
    """TRAIN-only annotations: native HIGH pitch -> independent lower octave.

    This is the inverse of S646 and targets the validated S642 failure:
    a real HIGH note was heard but the real simultaneous LOW pitch was
    masked. Exclude an upper NOTE onset whose lower real note has a
    different attack time, rather than treating delayed lower notes as
    a false independent low.
    """
    _,pairs=event_score(events,truth)
    matched={i:j for i,j in pairs}
    emitted=[e for e in events if e["type"]=="note_on"]
    results=[]
    for i,e in enumerate(emitted):
        high_pitch=int(e["pitch"])
        if not 52<=high_pitch<=76:
            continue
        j=matched.get(i)
        if j is None:
            label=0
            source="unmatched_native_high"
        else:
            true_high=truth[j]
            t=float(true_high["time"])
            low_pitch=high_pitch-12
            if t<.75 or t>22.8:
                continue
            lower=[n for n in truth
                   if int(n["pitch"])==low_pitch
                   and float(n["time"])+float(n["duration"])>=t-.05
                   and float(n["time"])<=t+.20]
            simultaneous=[n for n in lower
                          if abs(float(n["time"])-t)<=.030
                          and float(n["duration"])>=.080]
            if simultaneous:
                label=1
            elif lower:
                continue
            else:
                label=0
            source="matched_native_high"
        results.append(dict(pitch=high_pitch-12,
                            onset_sample=int(e["decision_sample"]),
                            label=label,origin=source))
    return results


def training_source(spec,root,archive,heads,config):
    sound,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(sound)/SR)
    ends,p,k,events=native_notes_from_audio(sound,heads,config)
    items=candidate_labels(events,truth)
    X={d:[] for d in DELAYS_MS}
    y={d:[] for d in DELAYS_MS}
    for item in items:
        for delay in DELAYS_MS:
            f=octave_features(sound,item["pitch"],item["onset_sample"],delay)
            if f is None:continue
            X[delay].append(f)
            y[delay].append(item["label"])
    return dict(name=spec["name"],composition=spec["composition"],source_sha256=sha,
                generated_native_noteons=sum(e["type"]=="note_on" for e in events),
                generated_high_candidates=len(items),
                true_high_to_low_candidates=sum(x["label"] for x in items),
                unmatched_high_candidates=sum(x["origin"]=="unmatched_native_high" for x in items),
                matrices={str(d):np.asarray(X[d],np.float32).reshape(-1,173)
                          for d in DELAYS_MS},
                targets={str(d):np.asarray(y[d],np.int8) for d in DELAYS_MS})


def head_factory():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=.20,class_weight="balanced",
                                            max_iter=1400,random_state=SEED))


def choose_thresholds_train_only(rows):
    fit={};diagnostics={}
    for delay in DELAYS_MS:
        X=np.concatenate([r["matrices"][str(delay)] for r in rows])
        y=np.concatenate([r["targets"][str(delay)] for r in rows])
        groups=np.concatenate([np.full(len(r["targets"][str(delay)]),
                                       r["composition"],dtype=object)
                               for r in rows])
        if len(set(groups))!=9 or len(y)<100 or int(y.sum())<8:
            raise ValueError(f"Insufficient NOTE-candidate native TRAIN {delay}: {len(y)} / {y.sum()}")
        OOF=np.zeros(len(y),np.float64)
        folds=[]
        for train_i,held_i in GroupKFold(n_splits=5).split(X,y,groups):
            if set(groups[train_i])&set(groups[held_i]):
                raise AssertionError("Train composition overlap")
            if len(np.unique(y[train_i]))!=2:
                raise ValueError("Native NOTE low candidate only one class")
            model=head_factory().fit(X[train_i],y[train_i])
            OOF[held_i]=model.predict_proba(X[held_i])[:,1]
            folds.append(dict(train=sorted(set(groups[train_i])),
                              held=sorted(set(groups[held_i])),
                              pos_train=int(y[train_i].sum()),
                              pos_held=int(y[held_i].sum())))
        trial=[]
        for threshold in THRESHOLDS:
            select=OOF>=threshold
            TP=int(np.sum(y[select]))
            FP=int(np.sum(select))-TP
            FN=int(y.sum())-TP
            utility=TP-FP/2
            precision=TP/max(1,TP+FP)
            trial.append(dict(threshold=threshold,TP=TP,FP=FP,FN=FN,
                              utility=utility,precision=precision))
        winner=max(trial,key=lambda r:(r["utility"],-r["FP"],r["threshold"]))
        fitted=head_factory().fit(X,y)
        fit[delay]=dict(head=fitted,threshold=float(winner["threshold"]))
        diagnostics[str(delay)]=dict(total=len(y),positives=int(y.sum()),
                                     negatives=int(len(y)-y.sum()),
                                     OOF=trial,selected=winner,folds=folds)
        print("S647 OOF TRAIN",delay,"n",len(y),"positives",int(y.sum()),
              "chosen",winner,flush=True)
    return fit,diagnostics


def replay_with_gate(sound,ends,p,k,heads,config,models):
    """Run baseline NOTE and 4 augmented CausalSelections in chronological order.

    NOTE high proposals come ONLY from baseline native decoder at the
    sample that they are actually emitted. Pending decisions mature at
    the requested future sample, never backdated.
    """
    ctrl=CausalSelection(config)
    augmented={d:CausalSelection(config) for d in DELAYS_MS}
    events={"S614_original":[]}
    for delay in DELAYS_MS:
        events[f"head_{delay}ms"]=[]
    pending={d:defaultdict(list) for d in DELAYS_MS}
    admission=Counter()
    attempted=Counter()
    expired=Counter()
    for end,probs,counts in zip(ends,p,k):
        end=int(end)
        control_emitted=ctrl.step(probs,counts,end,end)
        events["S614_original"].extend(control_emitted)
        for e in control_emitted:
            if e["type"]=="note_on" and 40<=int(e["pitch"])<=64:
                pitch=int(e["pitch"])
                for delay in DELAYS_MS:
                    intended=end+steps_for_delay(delay)*HOP
                    if intended<=int(ends[-1]):
                        pending[delay][intended].append((pitch,end))
                    else:
                        expired[delay]+=1
        for delay,decoder in augmented.items():
            modified=probs.copy()
            ready=pending[delay].pop(end,[])
            for pitch,start in ready:
                attempted[delay]+=1
                ix=pitch-40
                if (ix>=49 or decoder.active[ix] or not decoder.armed[ix]
                        or probs[ix,0]>=config.onset):
                    continue
                features=octave_features(sound,pitch,start,delay)
                if features is None:
                    raise AssertionError("Scheduled octave evidence unavailable")
                score=float(models[delay]["head"].predict_proba(features[None,:])[0,1])
                if score<models[delay]["threshold"]:
                    continue
                # ONLY inject into experimental NOTE selection. Original
                # NOTE K/posterior and per-pitch state machine are preserved.
                modified[ix,0]=config.onset
                modified[ix,1]=max(float(probs[ix,1]),.55)
                admission[delay]+=1
            key=f"head_{delay}ms"
            events[key].extend(decoder.step(modified,counts,end,end))
    for key,decoder in [("S614_original",ctrl),*[
        (f"head_{d}ms",decoder) for d,decoder in augmented.items()
    ]]:
        events[key].extend(decoder.panic(len(sound)))
    return events,dict(attempted=dict(attempted),admitted=dict(admission),
                       expired_at_end=dict(expired))


def evaluate_source(spec,root,archive,heads,config,models):
    sound,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(sound)/SR)
    ends,p,k,native=native_notes_from_audio(sound,heads,config)
    events,admissions=replay_with_gate(sound,ends,p,k,heads,config,models)
    assert [(e["pitch"],e["time"],e["type"]) for e in native]==[
        (e["pitch"],e["time"],e["type"]) for e in events["S614_original"]]
    gt_groups=overlap_groups(truth)
    metrics={}
    baseline_matches=None
    for name,emitted in events.items():
        sc,pairs=event_score(emitted,truth)
        matched={j for _,j in pairs}
        if name=="S614_original":
            baseline_matches=matched
        metrics[name]=dict(score=sc,
                           matched_with_overlap=len(matched & gt_groups["onset_overlap"]),
                           matched_K3plus=len(matched & gt_groups["onset_K3plus"]),
                           correct_truth=matched)
    for name in metrics:
        matched=metrics[name].pop("correct_truth")
        metrics[name]["new_vs_baseline"]=len(matched-baseline_matches)
        metrics[name]["lost_vs_baseline"]=len(baseline_matches-matched)
    return dict(name=spec["name"],composition=spec["composition"],
                sha256=sha,original_notes=len(truth),metrics=metrics,
                admission=admissions)


def run(out):
    if out.exists():raise FileExistsError("S647 evidence immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12 and len({s["composition"] for s in train})==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selection=Selection(**heads.metadata["selection"])
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            z=protocol["guitar_archives"][key]
            download(z["url"],root/"data"/path,z["size"],z["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as audio:
            observations=[]
            for idx,s in enumerate(train,1):
                r=training_source(s,root,audio,heads,selection)
                observations.append(r)
                print("S647 REAL TRAIN",idx,"/36",r["name"],
                      "NOTE candidate",r["generated_high_candidates"],
                      "true octave",r["true_high_to_low_candidates"],flush=True)
            models,validation=choose_thresholds_train_only(observations)
            reports=[]
            for idx,s in enumerate(dev,1):
                r=evaluate_source(s,root,audio,heads,selection,models)
                reports.append(r)
                print("S647 REAL DEV",idx,"/12",r["name"],
                      "originalTP",r["metrics"]["S614_original"]["score"]["TP"],
                      "fast0",r["metrics"]["head_0ms"]["score"]["TP"],
                      "fast0FP",r["metrics"]["head_0ms"]["score"]["FP"],
                      flush=True)
    scores={}
    for name in reports[0]["metrics"]:
        subset=[r["metrics"][name] for r in reports]
        sc=summary([v["score"] for v in subset])
        scores[name]=dict(score=sc,
                          TP_poly=sum(v["matched_with_overlap"] for v in subset),
                          TP_K3plus=sum(v["matched_K3plus"] for v in subset),
                          new=sum(v["new_vs_baseline"] for v in subset),
                          lost=sum(v["lost_vs_baseline"] for v in subset))
    baseline=scores["S614_original"]["score"]
    assert all(baseline[k]==v for k,v in GUITARSET_PINNED_CONTROL.items()),baseline
    for key,v in scores.items():
        sc=v["score"]
        v["delta_TP"]=int(sc["TP"]-baseline["TP"])
        v["delta_FP"]=int(sc["FP"]-baseline["FP"])
        v["delta_F1_points"]=round(100*(sc["F1"]-baseline["F1"]),6)
        print("S647 SCORES",key,"TP",sc["TP"],"FP",sc["FP"],
              "F1",sc["F1"],"deltaPts",v["delta_F1_points"],
              "poly",v["TP_poly"],"K3",v["TP_K3plus"],flush=True)
    report=dict(
        status="S647_HIGH_NOTE_GENERATED_LOWER_FUNDAMENTAL_CAUSAL_0_40_80_190MS",
        model_SHA256=frozen["model_sha256"],
        original_NOTE_K_posterior_unchanged=True,
        native_NOTE_pitch_probs_unchanged=True,
        native_NOTE_attack_selection_retained=True,
        shadow_supplemental_note_on_not_backdated=True,
        no_TEST_access=True,DEV_previously_exposed=True,
        training_only_GuitarSet_TRAIN=True,
        training_recordings=36,dev_recordings=12,
        training_labels_used_only_for_supervised_fit=True,
        annotation_used_only_for_final_scoring_on_DEV=True,
        candidates_created_only_from_NATIVE_HIGH_NOTE_note_on=True,\n        octave_rescue_direction="detected_high_note_to_missing_low_octave",\n        physical_motivation="S642 E3 was hidden while E4 survived synchronous octave overlap",
        delays_ms_requested=list(DELAYS_MS),
        delays_ms_actual={str(d):round(1000*steps_for_delay(d)*HOP/SR,4)
                          for d in DELAYS_MS},
        latency_is_NOT_cpu_throughput=True,
        CV_training_only=validation,
        DEV_scores=scores,
        DEV_by_source=reports,
        S641_previous_best_dev_F1=0.6339930882814955,
        seconds_wall=round(time.perf_counter()-start,2),
        promoted=False,
    )
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S647 FINAL",json.dumps({
        "duration_s":report["seconds_wall"],
        "training_OOF":{k:v["selected"] for k,v in validation.items()},
        "results":{name:dict(TP=row["score"]["TP"],FP=row["score"]["FP"],
                             F1=row["score"]["F1"],
                             net_F1_points=row["delta_F1_points"],
                             new=row["new"],lost=row["lost"],
                             poly=row["TP_poly"])
                   for name,row in scores.items()}}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
