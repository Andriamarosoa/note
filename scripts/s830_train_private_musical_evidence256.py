"""S830: 80 GuitarSet FIT tracks; independent musical evidence at native256.

All model inputs come exclusively from private PCM spectroscopy. S800 note
annotations are available for supervised TRAIN targets and scoring; archived
MIDI is a read-only scoring reference. No other axis's model predictions,
checkpoints, features or state are accepted at runtime.
"""
from __future__ import annotations
import argparse,io,json,hashlib,time,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from causal_note.s829_exactk_s809_native256 import (
    S829AcousticFrontEnd, fit_exact, score_exact, summarize_one, SR,HOP
)
from causal_note.s830_private_musical_evidence256 import (
    S830PCMAcoustics,train_evidence,choose_selector,
    counts_from_evidence,selected,SCHEMA,DOMAINS
)
from scripts.s829_train_exactk_s809_native256 import (
    digest_file,save_json,targets_at_ends,original_count,signature,totals,
    active_score
)

SOURCE_SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")


def extract_audio(pcm):
    """Offline replay of exactly the SAME private causal 256 streaming frontend."""
    stream=S830PCMAcoustics()
    pitch=[];agg=[];raw=[];ends=[]
    for start in range(0,len(pcm),HOP):
        results=stream.push(pcm[start:start+HOP],start_sample=start)
        if len(results)!=1:
            raise AssertionError("no dropped/duplicated PCM256 boundary")
        end,features,k=results[0]
        ends.append(end);pitch.append(features.astype(np.float16))
        agg.append(summarize_one(features,k));raw.append(k)
    return np.asarray(ends,np.int64),np.stack(agg),np.asarray(raw,np.uint8),np.stack(pitch)


def run(audio_zip,traces,manifest_path,out,tracks_per_composition):
    if out.exists():raise FileExistsError("immutable experiment: output exists")
    if not 1<=tracks_per_composition<=16:
        raise ValueError("tracks_per_composition must be 1..16")
    source=json.loads(manifest_path.read_text())
    if (source.get("schema")!=SOURCE_SCHEMA or not source.get("complete")
        or len(source.get("fit_files",[]))!=80 or
        any(source.get(x) for x in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise ValueError("require untouched complete S800 FIT80 manifest")
    selections=[(comp,e) for comp in COMPS
       for e in sorted((r for r in source["fit_files"]
                        if r["composition"]==comp),key=lambda r:r["index"])
       [:tracks_per_composition]]
    if len(selections)!=5*tracks_per_composition:
        raise ValueError("incorrect GuitarSet composition tracks")
    out.mkdir(parents=True)
    report=dict(schema=SCHEMA,series="S830",complete=False,promoted=False,
                protocol="PCM256 own FFT -> 2 K+3 acoustic musical heads -> private CAL selector, 5 OOF compositions",
                source_manifest_sha256=digest_file(manifest_path),
                feature_hop_samples=HOP,decision_hop_samples=HOP,
                independent_acoustics=True,private_trained_heads=5,
                any_other_axis_as_model_input=False,
                shared_checkpoint_loaded=False,
                source_audio_sha256=None,fit_source_explored=True,
                external_generalization_verified=False,
                hardware_latency_verified=False,
                CAL_opened=False,HELD_opened=False,TEST_opened=False,
                public_midi_modified=False,other_five_axes_modified=False,
                tracks=[],folds={},runtime_causal_replay_only=True)
    cache={}
    t0=time.perf_counter()
    with zipfile.ZipFile(audio_zip) as z:
        for comp,spec in selections:
            idx=int(spec["index"]);path=traces/f"fit_{idx:03d}.json"
            if digest_file(path)!=spec["json_sha256"]:
                raise ValueError("source metadata SHA mismatch")
            meta=json.loads(path.read_text())
            if meta["name"]!=spec["name"] or meta["composition"]!=comp:
                raise ValueError("source composition mismatch")
            blob=z.read(meta["name"]+"_mic.wav")
            rate,pcm=wavfile.read(io.BytesIO(blob))
            if rate!=SR or pcm.ndim!=1 or pcm.dtype!=np.int16:
                raise ValueError("original mono PCM44.1kHz/int16 mandatory")
            audio=pcm[:meta["samples"]].astype(np.float32)/32768.
            ends,X,raw,pitch=extract_audio(audio)
            if not np.array_equal(ends,np.arange(HOP,len(audio)+1,HOP)):
                raise AssertionError("one fresh acoustic frame every PCM256")
            # Labels and archived MIDI opened ONLY after private acoustic
            # inference features have been frozen from raw PCM.
            y,truth=targets_at_ends(meta["fit_truth"],ends)
            source_events=meta["source_events"]
            before=signature(source_events)
            original=original_count(source_events,ends)
            checksum=active_score(original,truth)
            historical=meta["source_counts"]
            if (checksum["correct"]!=historical["active_k_correct"] or
                checksum["frames"]!=historical["active_frames"] or
                signature(source_events)!=before):
                raise AssertionError("original MIDI score/events changed")
            cache[idx]=dict(index=idx,name=meta["name"],composition=comp,
                 X=X,baseline=raw,pitch_features=pitch,pitch_target=y,
                 k=truth,source=score_exact(original,truth),
                 acoustic_sha256=hashlib.sha256(blob).hexdigest())
            print("S830_AUDIO_PCM256",idx,comp,len(ends),flush=True)
    if len(cache)!=5*tracks_per_composition:raise AssertionError("track count drift")
    if tracks_per_composition==16:
        assert sum(len(x["k"]) for x in cache.values())==306536
    results=[];results829=[];sources=[]
    for i,outer in enumerate(COMPS):
        inner=COMPS[(i+1)%5]
        train=[r for r in cache.values() if r["composition"] not in (outer,inner)]
        cal=[r for r in cache.values() if r["composition"]==inner]
        test=[r for r in cache.values() if r["composition"]==outer]
        if (len(train),len(cal),len(test))!=(3*tracks_per_composition,
                                               tracks_per_composition,
                                               tracks_per_composition):
            raise AssertionError("TRAIN inner-CAL OUTER data split wrong")
        start=time.perf_counter()
        baseline=fit_exact(train,cal,seed=829+i)
        evidence=train_evidence(train,seed=830+i)
        if (outer in baseline.metadata["train_compositions"] or
            outer in evidence.meta["training_compositions"] or
            inner in evidence.meta["training_compositions"]):
            raise AssertionError("outer label leaked into model")
        caldata=[]
        for r in cal:
            base=baseline.predict(r["X"],r["baseline"])
            post=evidence.posterior(r["pitch_features"])
            confidence=np.mean(np.maximum(post[:,:,1],1-post[:,:,1]),axis=1)
            caldata.append(dict(base=base,evidence=post,confidence=confidence,k=r["k"]))
        selector,selector_scores=choose_selector(caldata)
        report["folds"][outer]=dict(
             train_compositions=sorted({x["composition"] for x in train}),
             calibration=inner,test_composition=outer,
             mode=selector[0],threshold=selector[1],shift=selector[2],
             selected_CAL_stats=selector_scores[selector],
             independent_three_musical_heads=True,
             independent_legacy_two_K_heads=True,
             seconds_training_and_CAL=time.perf_counter()-start)
        for r in test:
            base=baseline.predict(r["X"],r["baseline"])
            post=evidence.posterior(r["pitch_features"])
            direct,state=counts_from_evidence(post,selector[1])
            confidence=np.mean(np.maximum(post[:,:,1],1-post[:,:,1]),axis=1)
            pred=selected(base,direct,state,confidence,selector[0],selector[2])
            sc=score_exact(pred,r["k"]);sb=score_exact(base,r["k"])
            per_k={str(k):dict(
                 frames=int(np.count_nonzero(r["k"]==k)),
                 correct=int(np.count_nonzero((pred==r["k"])&(r["k"]==k))),
                 undercount=int(np.count_nonzero((pred<r["k"])&(r["k"]==k))),
                 overcount=int(np.count_nonzero((pred>r["k"])&(r["k"]==k))))
                 for k in range(7)}
            rec=dict(index=r["index"],name=r["name"],
                 composition=outer,S830=sc,S829_reproduced=sb,
                 H0=r["source"],by_true_K=per_k,
                 selector=dict(mode=selector[0],threshold=selector[1],shift=selector[2]),
                 source_pcm_sha256=r["acoustic_sha256"],
                 no_other_axis_called=True)
            results.append(sc);results829.append(sb);sources.append(r["source"])
            report["tracks"].append(rec)
        save_json(out/"report.json",report)
        print("S830_OUTER_FOLD",outer,selector,totals([
          z["S830"] for z in report["tracks"] if z["composition"]==outer
        ]),flush=True)
    report["S830_exactk"]=totals(results)
    report["S829_reproduced_exactk"]=totals(results829)
    report["H0_original_exactk"]=totals(sources)
    report["elapsed_seconds"]=time.perf_counter()-t0
    report["complete"]=True
    save_json(out/"report.json",report)
    print("S830_TRUE_PCM256_FINAL",json.dumps({
      "S830":report["S830_exactk"],
      "S829":report["S829_reproduced_exactk"],
      "H0":report["H0_original_exactk"],
      "folds":report["folds"],
      "n_tracks":len(report["tracks"]),
      "no_other_axis_changed":True},sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest-path","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--tracks-per-composition",type=int,default=1)
    args=p.parse_args()
    run(args.audio_zip,args.traces,args.manifest_path,
        args.out,args.tracks_per_composition)
