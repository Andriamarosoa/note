"""S829 independent Exact-K K0..K6 from S809 isolation on S828 native PCM256, five composition-disjoint folds.

This is a FIT research evaluation, not independent external generalization.
The 2 neural-free gradient boosting count heads are trained exclusively on independent
PCM-derived DSP, with private CAL selector. Nothing from any other task is loaded.
"""
from __future__ import annotations
import argparse,io,json,zipfile,hashlib,time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from causal_note.s829_exactk_s809_native256 import SR,HOP
from causal_note.s829_exactk_s809_native256 import S829Stream,fit_exact,score_exact,SCHEMA

SOURCE_SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")


# These reference/scoring helpers are PRIVATE to this Exact-K experiment.
# They receive labels or historical MIDI only AFTER all PCM-only prediction
# rows are frozen, and never enter S829Stream/FittedExactK.
def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def save_json(path,obj):
    Path(path).write_text(
        json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")


def targets_at_ends(truth,ends):
    pitch=np.zeros((len(ends),49),bool)
    count=np.zeros(len(ends),np.int16)
    times=np.asarray(ends)/SR
    for note in truth:
        k=int(note["pitch"])-40
        if not 0<=k<49:
            raise ValueError("FIT truth outside MIDI40..88")
        mask=(times>=float(note["time"])) & (
            times<float(note["time"])+float(note["duration"]))
        pitch[mask,k]=True
        count+=mask
    return pitch,np.minimum(count,6).astype(np.uint8)


def signature(events):
    return [(e["type"],int(e["pitch"]),int(e["available_sample"]),
             int(e["decision_sample"]),float(e["time"]),e.get("reason",""))
            for e in events]


def original_count(events,ends):
    """Original signed MIDI occupancy for reference scoring ONLY."""
    occupancy=np.zeros((len(ends),49),np.uint8)
    held={}
    for e in events:
        k=int(e["pitch"])-40
        if not 0<=k<49:raise ValueError("historical MIDI pitch outside scope")
        boundary=int(np.searchsorted(ends,int(e["available_sample"])))
        if e["type"]=="note_on":
            if k in held:raise AssertionError("duplicate NoteOn")
            held[k]=boundary
        elif e["type"]=="note_off":
            if k not in held:raise AssertionError("orphan NoteOff")
            occupancy[held.pop(k):boundary,k]=1
        else:raise ValueError("invalid MIDI type")
    for k,beg in held.items():
        occupancy[beg:,k]=1
    return np.minimum(occupancy.sum(axis=1),6).astype(np.uint8)


def active_score(pred,truth):
    """Historical H0 checksum uses GT-active frames, never a training input."""
    p,y=np.asarray(pred),np.asarray(truth)
    if p.ndim!=1 or p.shape!=y.shape:raise ValueError("misaligned K")
    mask=y>0
    return dict(correct=int(np.count_nonzero((p==y)&mask)),
        frames=int(mask.sum()),undercount=int(np.count_nonzero((p<y)&mask)),
        overcount=int(np.count_nonzero((p>y)&mask)))


def totals(rows):
    out={k:sum(int(r[k]) for r in rows) for k in (
        "correct","frames","undercount","overcount")}
    if out["correct"]+out["undercount"]+out["overcount"]!=out["frames"]:
        raise AssertionError("invalid K denominator")
    out["accuracy"]=out["correct"]/max(1,out["frames"])
    return out


def run(audio_zip,traces,manifest_path,out,tracks_per_composition):
    if out.exists():raise FileExistsError("S829 evidence is immutable")
    if not 1<=tracks_per_composition<=16:raise ValueError("1..16 tracks/composition only")
    source=json.loads(manifest_path.read_text())
    if (source.get("schema")!=SOURCE_SCHEMA or not source.get("complete")
        or len(source.get("fit_files",[]))!=80 or
        any(source.get(f) for f in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise ValueError("S829 authentic FIT80-only S800 provenance required")
    out.mkdir(parents=True)
    chosen=[(comp,e) for comp in COMPS for e in
        sorted((r for r in source["fit_files"] if r["composition"]==comp),
               key=lambda r:r["index"])[:tracks_per_composition]]
    if len(chosen)!=5*tracks_per_composition:
        raise ValueError("wrong GuitarSet source composition split")
    report=dict(schema=SCHEMA,series="S829",complete=False,promoted=False,
        protocol="Own PCM->DSP->two trained K heads->private CAL selector, 48/16/16 FIT80",
        independent_neural_checkpoints_sha256=[],
        shared_neural_encoder_used=False,
        source_manifest_sha256=digest_file(manifest_path),
        private_acoustic_frontend=True,private_head_count=2,
        only_exact_k_trained=True,other_five_axes_modified=False,
        public_midi_modified=False,fit_source_explored=True,
        external_generalization_verified=False,hardware_latency_verified=False,
        feature_hop_samples=256,decision_hop_samples=256,
        annotation_inferred_at_runtime=False,CAL_opened=False,
        HELD_opened=False,TEST_opened=False,
        source_audio_sha256=None,tracks=[],folds={})
    cache={}
    start=time.perf_counter()
    with zipfile.ZipFile(audio_zip) as archive:
        for comp,spec in chosen:
            meta_path=traces/f"fit_{spec['index']:03d}.json"
            if digest_file(meta_path)!=spec["json_sha256"]:
                raise ValueError("GT/source metadata SHA drift")
            meta=json.loads(meta_path.read_text())
            if meta["name"]!=spec["name"] or meta["composition"]!=comp:
                raise ValueError("mismatched source metadata")
            filename=meta["name"]+"_mic.wav"
            blob=archive.read(filename)
            rate,pcm=wavfile.read(io.BytesIO(blob))
            if rate!=SR or pcm.ndim!=1 or pcm.dtype!=np.int16:
                raise ValueError("S829 must receive signed original mono PCM44.1kHz")
            audio=pcm[:meta["samples"]].astype(np.float32)/32768.
            ends=np.arange(HOP,len(audio)+1,HOP,dtype=np.int64)
            teacher=S829Stream()
            decisions=[]
            for pos in range(0,len(audio),HOP):
                decisions.extend(teacher.push(audio[pos:pos+HOP],start_sample=pos))
            features,baseline=teacher.collected()
            if (len(features)!=len(ends) or
                [d.decision_sample for d in decisions]!=ends.tolist()):
                raise AssertionError("independent raw PCM256 K replay drift")
            # GT and historical output opened only after all source-PCM
            # inference features and private S708 predictions have been fixed.
            _,truth=targets_at_ends(meta["fit_truth"],ends)
            source_midi=meta["source_events"]
            protected_sig=signature(source_midi)
            original=original_count(source_midi,ends)
            old=score_exact(original,truth)
            signed_active=active_score(original,truth)
            sig=meta["source_counts"]
            if (signed_active["correct"]!=sig["active_k_correct"] or
                signed_active["frames"]!=sig["active_frames"] or
                signature(source_midi)!=protected_sig):
                raise AssertionError("original H0 five-axis MIDI not protected")
            native_score=score_exact(baseline,truth)
            record=dict(index=spec["index"],name=meta["name"],
                composition=comp,texture=meta["texture"],X=features,
                baseline=baseline,k=truth,source=old,native=native_score)
            cache[spec["index"]]=record
            print("S829_INDEPENDENT_PCM256",spec["index"],comp,len(ends),
                  round(native_score["correct"]/max(1,native_score["frames"]),6),
                  flush=True)
    if len(cache)!=5*tracks_per_composition:
        raise AssertionError("source recording count changed")
    expected_H0=totals([r["source"] for r in cache.values()])
    if tracks_per_composition==16 and expected_H0["frames"]!=306536:
        raise AssertionError("full FIT80 source H0 not bit-exact")
    predicted=[]
    for i,outer in enumerate(COMPS):
        inner=COMPS[(i+1)%5]
        tr=[r for r in cache.values() if r["composition"] not in (outer,inner)]
        cal=[r for r in cache.values() if r["composition"]==inner]
        te=[r for r in cache.values() if r["composition"]==outer]
        if tuple(map(len,(tr,cal,te)))!=(3,1,1) and \
           tuple(map(len,(tr,cal,te)))!=(48,16,16):
            if not(len(tr)==3*tracks_per_composition and
                   len(cal)==len(te)==tracks_per_composition):
                raise AssertionError("invalid disjoint TRAIN/CAL/TEST fold sizes")
        now=time.perf_counter()
        fitted=fit_exact(tr,cal,seed=829+i)
        if (outer in fitted.metadata["train_compositions"] or
            outer in fitted.metadata["calibration"] or
            inner in fitted.metadata["train_compositions"]):
            raise AssertionError("composition leakage train/cal/test")
        report["folds"][outer]=dict(
            train_compositions=fitted.metadata["train_compositions"],
            calibration=fitted.metadata["calibration"],
            test_composition=outer,mode=fitted.mode,shift=fitted.shift,
            samples_trained=sum(len(x["k"]) for x in tr),
            training_seconds=time.perf_counter()-now,
            private_model_head_only=True)
        for r in te:
            out_pred=fitted.predict(r["X"],r["baseline"])
            sc=score_exact(out_pred,r["k"])
            by_k={str(k):dict(correct=int(np.sum((out_pred==r["k"])&(r["k"]==k))),
                frames=int(np.sum(r["k"]==k)),
                undercount=int(np.sum((out_pred<r["k"])&(r["k"]==k))),
                overcount=int(np.sum((out_pred>r["k"])&(r["k"]==k))))
                for k in range(7)}
            predicted.append(sc)
            report["tracks"].append(dict(index=r["index"],name=r["name"],
                composition=outer,selected_mode=fitted.mode,
                selected_shift=fitted.shift,S829=sc,
                S829_raw_acoustic_k=r["native"],H0_original=r["source"],
                by_true_K=by_k,other_axes_h0_untouched=True))
        save_json(out/"report.json",report)
        print("S829_CAL_AND_OUTER",outer,fitted.mode,fitted.shift,
              totals([r["S829"] for r in report["tracks"]
                     if r["composition"]==outer]),flush=True)
    report["S829_exactk"]=totals(predicted)
    report["S829_raw_acoustic_k_exactk"]=totals(
        [r["S829_raw_acoustic_k"] for r in report["tracks"]])
    report["H0_original_exactk"]=totals(
        [r["H0_original"] for r in report["tracks"]])
    report["total_elapsed_seconds"]=time.perf_counter()-start
    report["complete"]=True
    save_json(out/"report.json",report)
    print("S829_TWO_PRIVATE_HEADS_NATIVE256_EXACTK_FINAL",json.dumps({
        "n_tracks":len(report["tracks"]),
        "S829":report["S829_exactk"],
        "raw_acoustic_k":report["S829_raw_acoustic_k_exactk"],
        "H0":report["H0_original_exactk"],
        "only_exact_k_changed":True,
        "no_other_axis_changed":True,"public_midi_not_modified":True
        },sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for a in ("audio-zip","traces","manifest-path","out"):
        p.add_argument("--"+a,type=Path,required=True)
    p.add_argument("--tracks-per-composition",type=int,default=1)
    x=p.parse_args()
    run(x.audio_zip,x.traces,x.manifest_path,x.out,x.tracks_per_composition)
