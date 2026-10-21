"""S827: real signed GuitarSet verification of private S708 Active-K at 256.

The S826 source-wide MIDI model is used as an INDEPENDENT oracle of
equivalence only, never as inference input. Signed original S800 reference
MIDI/GT are opened exclusively AFTER the private PCM predictions are frozen.
All five other axes remain the original unmodified H0; their output data
never pass through this candidate.
"""
from __future__ import annotations

import argparse,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from causal_note.strict_live import SR,HOP,load_heads
from causal_note.s712_s708_live import load_s708_checkpoint
from causal_note.s827_s708_activek_only import S827S708ActiveKOnly,SCHEMA
from causal_note.s814_activek_teacher import active_score
from causal_note.s812_historical_axis_h0 import signature
from scripts.s815_train_activek256 import original_count,totals
from scripts.s813_prepare_native256 import digest_file,save_json,targets_at_ends

COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
SOURCE_SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"

def run(audio_zip,traces,manifest_path,source_checkpoint,source_report,
        native_model,s826_events,out,tracks_per_composition):
    if out.exists():raise FileExistsError("S827 immutable results; refusing overwrite")
    if not 1<=tracks_per_composition<=16:
        raise ValueError("S827 strict FIT80 composition-only pilot 1..16 recordings")
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get("schema")!=SOURCE_SCHEMA or not manifest.get("complete")
        or len(manifest.get("fit_files",[]))!=80
        or manifest.get("CAL_opened") or manifest.get("HELD_opened")
        or manifest.get("TEST_opened")):
        raise AssertionError("original authentic S800 FIT80 signed source required")
    model,policy,sha=load_s708_checkpoint(source_checkpoint,source_report)
    heads=load_heads(native_model)
    chosen=[(comp,entry) for comp in COMPS for entry in
            sorted((e for e in manifest["fit_files"] if e["composition"]==comp),
                   key=lambda e:e["index"])[:tracks_per_composition]]
    assert len(chosen)==5*tracks_per_composition
    out.mkdir(parents=True)
    report=dict(schema="S827_ORIGINAL_S708_PRIVATE_ACTIVEK_REAL_MIDI256_V1",
        model_sha256=sha,original_training_decision_hop_samples=1024,
        inference_feature_hop_samples=256,inference_decision_hop_samples=256,
        private_output_axis="active_exact_k",other_axes_frozen=True,
        production_midi_modified=False,source_S708_changed=False,
        S826_shared_pipeline_merged=False,added_future_samples=0,
        added_hold_samples=0,trained_at_native256=False,promoted=False,
        GT_not_provided_to_inference=True,
        source_manifest_sha256=digest_file(manifest_path),
        source_native_sha256=digest_file(native_model),completed=False,
        rows=[])
    with zipfile.ZipFile(audio_zip) as archive:
        for comp,spec in chosen:
            stem=f"fit_{spec['index']:03d}"
            meta_path=traces/(stem+".json")
            if digest_file(meta_path)!=spec["json_sha256"]:
                raise AssertionError("signed original metadata changed")
            meta=json.loads(meta_path.read_text())
            if meta["name"]!=spec["name"] or meta["composition"]!=comp:
                raise AssertionError("GuitarSet source name/composition changed")
            data=archive.read(meta["name"]+"_mic.wav")
            rate,pcm=wavfile.read(io.BytesIO(data))
            if rate!=SR or pcm.ndim!=1 or pcm.dtype!=np.int16:
                raise AssertionError("strict pinned mono integer 44.1kHz GuitarSet required")
            audio=(pcm[:meta["samples"]].astype(np.float32)/32768.)
            ends=np.arange(HOP,len(audio)+1,HOP,dtype=np.int64)
            private=S827S708ActiveKOnly(heads,model,policy)
            decisions=[]
            for sample in range(0,len(audio),HOP):
                decisions.extend(private.push(audio[sample:sample+HOP],start_sample=sample))
            private.finish()
            guess=np.array([d.count for d in decisions],dtype=np.uint8)
            if len(guess)!=len(ends) or private.model_calls!=len(ends):
                raise AssertionError("not every 256 is independently inferred")
            if np.any(np.asarray([d.decision_sample for d in decisions])!=ends):
                raise AssertionError("non-native decision timestamp")
            # Prediction is COMPLETED before ANY annotation or historical S826
            # output is opened. S826 is only a read-only equivalence comparator.
            s826_file=s826_events/(stem+"_strict256.json")
            s826_original=json.loads(s826_file.read_text())
            source_signature=signature(meta["source_events"])
            target_s826=original_count(s826_original,ends)
            if not np.array_equal(guess,target_s826):
                ix=np.flatnonzero(guess!=target_s826)
                raise AssertionError("private S708 ActiveK diverges from S826 full-S708 clock on "+stem+
                    "; mismatched_count="+str(len(ix))+"; first="+str(ix[:10].tolist()))
            _,truth=targets_at_ends(meta["fit_truth"],ends)
            original=original_count(meta["source_events"],ends)
            reference=active_score(original,truth)
            if reference["frames"]!=meta["source_counts"]["active_frames"]:
                raise AssertionError("official GT active denominator changed")
            if reference["correct"]!=meta["source_counts"]["active_k_correct"]:
                raise AssertionError("original signed H0 MIDI score drift")
            score=active_score(guess,truth)
            if signature(meta["source_events"])!=source_signature:
                raise AssertionError("signed other five axes modified")
            report["rows"].append(dict(index=spec["index"],name=meta["name"],
                composition=comp,frames=len(ends),activek=score,
                H0_activek=reference,exact_S826_activek_parity=True,
                real_neural_inference_calls=private.model_calls,
                other_axes_source_signature_preserved=True))
            save_json(out/"report.json",report)
            print("S827_PRIVATE_256_REAL",stem,comp,
                "activeK",round(score["correct"]/max(1,score["frames"]),6),
                "h0",round(reference["correct"]/max(1,reference["frames"]),6),
                "neurals",private.model_calls,flush=True)
    report["total_private_activek"]=totals([r["activek"] for r in report["rows"]])
    report["total_H0_activek"]=totals([r["H0_activek"] for r in report["rows"]])
    if len(report["rows"])==80 and (
        report["total_H0_activek"]["correct"]!=142213 or
        report["total_H0_activek"]["frames"]!=249411):
        raise AssertionError("signed FIT80 score is not identical")
    report["completed"]=True
    save_json(out/"report.json",report)
    print("S827_PRIVATE_ACTIVEK_VALIDATED",json.dumps({
        "ntracks":len(report["rows"]),
        "activek":report["total_private_activek"],
        "H0":report["total_H0_activek"],
        "source_S708_unchanged":True,
        "five_other_axes_unchanged":True,
        "equivalent_real_S826_256_predictions":True
    },sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for a in ("audio-zip","traces","manifest-path","source-checkpoint",
              "source-report","native-model","s826-events","out"):
        ap.add_argument("--"+a,type=Path,required=True)
    ap.add_argument("--tracks-per-composition",type=int,default=1)
    x=ap.parse_args()
    run(x.audio_zip,x.traces,x.manifest_path,x.source_checkpoint,
        x.source_report,x.native_model,x.s826_events,x.out,
        x.tracks_per_composition)
