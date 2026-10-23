"""S834: original S809 six independent HEAD/SELECTOR/DECODER pipelines
ported to TRUE S708 audio PCM256. Signed 80 GuitarSet tracks, six real
standalone shadow metrics plus counterfactual 6x5 non-interference tests.

Never trains/reads GT during PCM inference, never alters H0 public MIDI.
No relabeling 1024 scores as 256 scores; every new neural posterior is
computed from current/past PCM only and S708 original frozen weights.
"""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from causal_note.strict_live import SR,HOP,load_heads
from causal_note.s712_s708_live import load_s708_checkpoint
from causal_note.s834_native256_decoders import AXES,DECISION_SAMPLES
from causal_note.s834_s809_six_axis_pcm256 import (
    SixIndependentAxes,default_pipelines,AxisSelector
)
from causal_note.s834_s809_native256_audio import (
    S834S809Native256,SCHEMA
)
from scripts.s813_prepare_native256 import digest_file,save_json
from scripts.s809_fit80_independence_replay import (
    shadow_diagnostics
)
from scripts.s800_oof_trace import verify_snapshot
from scripts.s704_joint_metrics import metric,aggregate,SCORES
from scripts.s795_train_activek_midi256 import COUNT_KEYS

SOURCE_SCHEMA="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
COMPS=("BN1","Funk2","Jazz1","Rock3","SS1")
AUDIT_PREFIX_FRAMES=64

def summarize_scores(summed):
    def safe(a,b):return float(a/max(1,b))
    return {
       "exact_k":safe(summed["exact_k"]["correct"],summed["exact_k"]["frames"]),
       "active_exact_k":safe(summed["active_exact_k"]["correct"],summed["active_exact_k"]["active_frames"]),
       "onset_f1":safe(2*summed["onset"]["tp"],2*summed["onset"]["tp"]+summed["onset"]["fp"]+summed["onset"]["fn"]),
       "offset_f1":safe(2*summed["offset"]["tp"],2*summed["offset"]["tp"]+summed["offset"]["fp"]+summed["offset"]["fn"]),
       "frame_pitch_f1":safe(2*summed["frame_pitch"]["tp"],2*summed["frame_pitch"]["tp"]+summed["frame_pitch"]["fp"]+summed["frame_pitch"]["fn"]),
       "full_note_f1":safe(2*summed["full_note"]["tp"],2*summed["full_note"]["tp"]+summed["full_note"]["fp"]+summed["full_note"]["fn"])
    }

def run(audio_zip,traces,manifest_path,source_checkpoint,source_report,
        native_model,out,tracks_per_composition,*,sharp_engine="reference"):
    if sharp_engine not in ("reference","incremental"):
        raise ValueError("S836 sharp mode must be reference or incremental")
    if out.exists():raise FileExistsError("S834 immutable evidence cannot be replaced")
    if not 1<=tracks_per_composition<=16:
        raise ValueError("tracks per composition 1..16")
    source=json.loads(manifest_path.read_text())
    if (source.get("schema")!=SOURCE_SCHEMA or not source.get("complete") or
        len(source.get("fit_files",[]))!=80 or
        any(source.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise ValueError("S834 only authentic untouched signed S800 FIT80")
    nn,policy,checkpoint_sha=load_s708_checkpoint(source_checkpoint,source_report)
    native=load_heads(native_model)
    chosen=[(comp,r) for comp in COMPS for r in
        sorted((e for e in source["fit_files"] if e["composition"]==comp),
               key=lambda x:x["index"])[:tracks_per_composition]]
    if len(chosen)!=5*tracks_per_composition:raise AssertionError("wrong track split")
    out.mkdir(parents=True)
    report=dict(schema=SCHEMA,series="S834",complete=False,
        s836_sharp_engine=sharp_engine,
        branch_native_audio=True,private_s708_256=True,
        original_teacher_training_hop=1024,
        decision_hop_samples=256,acoustic_feature_hop_samples=256,
        source_s708_model_sha256=checkpoint_sha,
        native_heads_sha256=digest_file(native_model),
        manifest_sha256=digest_file(manifest_path),
        source_signed_midi_unchanged=True,
        original_five_axes_untouched=True,
        any_other_axis_output_regression_under_single_axis_mutation=False,
        all_new_axes_shadow_only=True,
        official_midi_modified=False,promoted=False,
        data_fit80_previously_explored=True,
        external_generalization_verified=False,
        p99_hardware_iRig_verified=False,
        no_lookahead_samples=0,added_hold_samples=0,
        CAL_opened=False,HELD_opened=False,TEST_opened=False,
        sources=5,tracks=[],source_scores={},axis_metrics={},
        counterfactual_pairs_passed=0,counterfactual_pairs_tested=0,
        mutated_decisions={},mutated_selector_cells={},
        longest_256_track=0)
    totals=defaultdict(Counter)
    changes=Counter();cells=Counter()
    source_counts=[]
    start=time.perf_counter()
    with zipfile.ZipFile(audio_zip) as archive:
        for comp,spec in chosen:
            idx=spec["index"]
            meta_path=traces/f"fit_{idx:03d}.json"
            if digest_file(meta_path)!=spec["json_sha256"]:
                raise ValueError("S834 source JSON signature drift")
            meta=json.loads(meta_path.read_text())
            if (meta["name"]!=spec["name"] or
                meta["composition"]!=comp):raise ValueError("signed metadata mismatch")
            rate,pcm=wavfile.read(io.BytesIO(archive.read(meta["name"]+"_mic.wav")))
            if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                raise ValueError("must use original mono 44.1kHz int16 guitar audio")
            audio=pcm[:meta["samples"]].astype(np.float32)/32768
            ends=np.arange(HOP,len(audio)+1,HOP,dtype=np.int64)
            shadow=S834S809Native256(native,nn,policy,sharp_engine=sharp_engine)
            variants={}
            for axis in AXES:
                pipelines=default_pipelines()
                pipelines[axis].selector=AxisSelector(axis,"causal_contrast",threshold=.12)
                variants[axis]=SixIndependentAxes(pipelines)
            track_decisions=[]
            # Audits run AFTER timed push(): no scientific score or CPU timing change.
            abi_288_hash=hashlib.sha256()
            sharp_96_hash=hashlib.sha256()
            axis_decision_hash={axis:hashlib.sha256() for axis in AXES}
            for i,pos in enumerate(range(0,len(audio),HOP)):
                results=shadow.push(audio[pos:pos+HOP],start_sample=pos)
                if pos+HOP<=len(audio):
                    if len(results)!=1:raise AssertionError("one S834 decision each received 256")
                    end,decision,selections,snapshot=results[0]
                    if end!=pos+HOP or end!=int(ends[i]):
                        raise AssertionError("wrong decision time")
                    track_decisions.append(decision)
                    last_abi=shadow.s836_last_input_288d
                    if last_abi.shape!=(49,288):
                        raise AssertionError("S836 strict current 288D ABI")
                    abi_288_hash.update(last_abi.tobytes())
                    sharp_96_hash.update(last_abi[:,192:].tobytes())
                    for axis in AXES:
                        axis_decision_hash[axis].update(
                            repr((end,decision[axis],selections[axis])).encode('utf-8')+b'\n')
                    if i<AUDIT_PREFIX_FRAMES:
                        for axis,counterfactual in variants.items():
                            outdec,outsel=counterfactual.step(snapshot)
                            changes[axis]+=int(outdec[axis]!=decision[axis])
                            cells[axis]+=int(outsel[axis].selected_alternative_pitches)
                            for other in AXES:
                                if other==axis:continue
                                report["counterfactual_pairs_tested"]+=1
                                if (outdec[other]!=decision[other] or
                                    outsel[other]!=selections[other]):
                                    raise AssertionError(
                                      f"S834 {axis}->{other} interference on "
                                      f"{meta['name']} at PCM sample {end}")
                                report["counterfactual_pairs_passed"]+=1
                elif results:
                    raise AssertionError("partial block generated future decision")
            if len(track_decisions)!=len(ends) or shadow.model_calls!=len(ends):
                raise AssertionError("one fresh frozen S708 NN call per complete PCM256")
            # Labels and archived MIDI are accessed ONLY after full PCM replay.
            signed=meta["source_events"]
            signed_sha=hashlib.sha256(json.dumps(signed,sort_keys=True).encode()).hexdigest()
            h0=metric(signed,meta["fit_truth"],ends)
            expected=meta["source_counts"]
            for key in COUNT_KEYS:
                if int(h0[key])!=int(expected[key]):
                    raise AssertionError(f"protected H0 {key} changed in {meta['name']}")
            if (hashlib.sha256(json.dumps(signed,sort_keys=True).encode()).hexdigest()
                !=signed_sha or SixIndependentAxes.signed_midi_identity(signed)!=signed):
                raise AssertionError("original public MIDI is altered")
            final=shadow.six.pipelines["full_note"].decoder.finish(int(meta["samples"]))
            single=shadow_diagnostics(track_decisions,final,meta["fit_truth"],ends)
            for axis,counts in single.items():totals[axis].update(counts)
            source_counts.append(h0)
            report["tracks"].append(dict(index=idx,name=meta["name"],
                composition=comp,frames=len(ends),real_pcm_sha256=hashlib.sha256(
                    pcm.tobytes()).hexdigest(),
                six_axis_standalone=single,source_counts={k:int(h0[k]) for k in COUNT_KEYS},
                causal_s708_model_calls=shadow.model_calls,
                protected_h0=True,other_five_axes_identical_all_counterfactuals=True,
                s836_input_288_sha256=abi_288_hash.hexdigest(),
                s836_spectral_96_sha256=sharp_96_hash.hexdigest(),
                s836_axis_decision_sha256={axis:d.hexdigest() for axis,d in axis_decision_hash.items()},
                runtime_cpu_diagnostics=shadow.diagnostics()))
            report["longest_256_track"]=max(report["longest_256_track"],len(ends))
            print("S834_REAL_PCM256_S809_PORT",idx,comp,len(ends),flush=True)
            save_json(out/"report.json",report)
    report["source_counts"]={k:int(v) for k,v in aggregate(source_counts).items() if k in COUNT_KEYS}
    report["source_scores"]={k:float(aggregate(source_counts)[k]) for k in SCORES}
    report["axis_metrics"]={x:dict(z) for x,z in totals.items()}
    report["six_axis_shadow_scores"]=summarize_scores(totals)
    report["mutated_decisions"]=dict(changes)
    report["mutated_selector_cells"]=dict(cells)
    report["frames"]=sum(x["frames"] for x in report["tracks"])
    report["complete"]=True
    report["elapsed_seconds"]=time.perf_counter()-start
    assert report["counterfactual_pairs_passed"]==report["counterfactual_pairs_tested"]
    if tracks_per_composition==16:
        if report["frames"]!=306536:raise AssertionError("FIT80 clock mismatch")
        if report["counterfactual_pairs_tested"]!=80*64*6*5:
            raise AssertionError("80 x 64 x six directional five-axis mutation checks required")
    save_json(out/"report.json",report)
    print("S834_SIX_NATIVE_PCM256_FINAL",json.dumps(dict(
        tracks=len(report["tracks"]),frames=report["frames"],
        axis_scores=report["six_axis_shadow_scores"],
        source_scores=report["source_scores"],
        protected_pairs=report["counterfactual_pairs_passed"],
        zero_cross_axis_regression=True),sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("audio-zip","traces","manifest-path","source-checkpoint",
              "source-report","native-model","out"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--tracks-per-composition",type=int,default=1)
    p.add_argument("--sharp-engine",choices=("reference","incremental"),
                   default="reference",help="S836 opt-in CPU cache; old path is default")
    z=p.parse_args()
    run(z.audio_zip,z.traces,z.manifest_path,z.source_checkpoint,
        z.source_report,z.native_model,z.out,z.tracks_per_composition,
        sharp_engine=z.sharp_engine)
