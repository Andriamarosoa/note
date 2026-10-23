"""SCIENTIFIC TRUE real PCM256 replay, independent 69D/70D/74D FP heads.

Same frozen S613+S701+ErrorPrediction69+FNRecovery heads for all variants.
Three separately trained neural FP heads predict BEFORE Onset selection on
EVERY fresh 256-sample input. Their private selectors only contest emitted
original risk Notes. Neither Main re-arm nor fourth FNRecovery is changed.
Only AFTER complete WAV audio inference is ground truth ever consulted.

The preregistered issue #142 is decided by genuine 80-WAV F1 and FP,
NEVER by an event proxy or held-out composition parameter tuning.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.io import wavfile
from onset256.acoustics import HOP,SR
from onset256.model import load
from onset256.error_evidence69 import load_hard_model
from onset256.fn_recovery import load_fn_recovery
from onset256.fp_discriminator import load_fp_discriminator,VARIANTS
from onset256.fp_expert_gate import load_gate
from onset256.runtime import Stream
from scripts.replay_native256 import score,match_onsets
from scripts.train_onset_fp_discriminator256 import (
    COMPS,BASE_4,BASE_3,f1,SCHEMA)

REAL_S613=dict(tp=8099,fp=4600,fn=3168)
MIN_F1_OLD_FOUR=2*8160/(2*8160+4605+3107)
THRESHOLD="preregistered GitHub issue #142: temporal74 FP < both controls AND real F1 >= prior four-head AND >= control70"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for c in iter(lambda:fd.read(2*1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def stats(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),
                f1=f1(tp,fp,fn))

def run(audio_zip,traces,manifest,native_model,s701_model,old_risk_models,
        old_fn_models,fp_models,gate_models,out):
    out=Path(out)
    if out.exists():raise FileExistsError("scientific replay must be immutable")
    m=json.loads(Path(manifest).read_text())
    if (m.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
        or not m.get("complete") or len(m.get("fit_files",()))!=80
        or any(m.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise ValueError("unsigned / invalid heldout real 80 WAV manifest")
    original,s701=load(native_model,s701_model)
    third_models={};fourth_models={};trained={};lineage={};gates={}
    for comp in COMPS:
        old_risk=Path(old_risk_models)/"checkpoints"
        rm=old_risk/f"onset_error_outer_{comp}.json"
        meta=json.loads(rm.read_text())
        if (meta["outer_composition"]!=comp or
            comp in meta["train_compositions"] or
            comp==meta["cal_composition"]):
            raise ValueError("original MLP69 heldout split mismatch")
        third_models[comp]=load_hard_model(
            old_risk/f"onset_error_outer_{comp}.joblib",rm,
            expected_sha256=meta["checkpoint_sha256"])
        old_fn=Path(old_fn_models)/"checkpoints"
        fm=old_fn/f"fn_recovery_outer_{comp}.json"
        fmeta=json.loads(fm.read_text())
        if (fmeta["outer_composition"]!=comp or
            comp in fmeta["train_compositions"] or
            comp==fmeta["cal_composition"]):
            raise ValueError("frozen FNRecovery checkpoint split drift")
        fourth_models[comp]=load_fn_recovery(
            old_fn/f"fn_recovery_outer_{comp}.joblib",fm,
            expected_sha256=fmeta["checkpoint_sha256"])
        trained[comp]={}
        lineage[comp]={}
        gate_root=Path(gate_models)/"checkpoints"
        gate_metadata=gate_root/f"fp_gate_outer_{comp}.json"
        gate_meta=json.loads(gate_metadata.read_text())
        if (gate_meta["outer_composition"]!=comp or
            comp in gate_meta["original_three_train_compositions"] or
            gate_meta["original_cal_composition"]==comp or
            not set(gate_meta["gate_train_tracks"]).isdisjoint(
                gate_meta["gate_validation_tracks"])):
            raise ValueError("gate violated additional eight-track CAL split")
        gates[comp]=load_gate(
            gate_root/f"fp_gate_outer_{comp}.joblib",
            gate_metadata,gate_meta["checkpoint_sha256"])
        for name in VARIANTS:
            root=Path(fp_models)/"checkpoints"
            pm=root/f"fp_{name}_outer_{comp}.json"
            pmeta=json.loads(pm.read_text())
            if (pmeta["outer_composition"]!=comp or
                comp in pmeta["train_compositions"] or
                comp==pmeta["cal_composition"] or
                len(pmeta["train_compositions"])!=3 or
                pmeta["variant"]!=name):
                raise ValueError("FP specialist trained on the measured OUTER composition")
            trained[comp][name]=load_fp_discriminator(
                root/f"fp_{name}_outer_{comp}.joblib",
                pm,expected_sha256=pmeta["checkpoint_sha256"])
            lineage[comp][name]=dict(sha256=pmeta["checkpoint_sha256"],
                                     calibrated_cut=pmeta["calibrated_cut"],
                                     train=pmeta["train_compositions"],
                                     cal=pmeta["cal_composition"])
    out.mkdir(parents=True)
    records=[]
    with zipfile.ZipFile(audio_zip) as z:
        for comp in COMPS:
            files=sorted((it for it in m["fit_files"]
                          if it["composition"]==comp),key=lambda it:it["index"])
            if len(files)!=16:raise ValueError("composition missing 16 signed WAVs")
            for it in files:
                ann=Path(traces)/f"fit_{it['index']:03d}.json"
                if digest(ann)!=it["json_sha256"]:
                    raise ValueError("original GT SHA changed")
                meta=json.loads(ann.read_text())
                if meta["name"]!=it["name"]:
                    raise ValueError("wrong 80-track identity")
                rate,pcm=wavfile.read(io.BytesIO(z.read(meta["name"]+"_mic.wav")))
                if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                    raise ValueError("original WAV is not mono 44100 int16 PCM")
                wav=pcm[:meta["samples"]].astype(np.float32)/32768.
                n=len(wav)//HOP
                risk=third_models[comp]
                recovery=fourth_models[comp]
                control=trained[comp]
                before={name:mdl.calls for name,mdl in control.items()}
                gate=gates[comp]
                previous_gate_calls=gate.calls
                original_risk=risk.calls;original_recovery=recovery.calls
                stream=Stream(original,s701,risk_model=risk,
                    recovery_model=recovery,fp_models=control,
                    gate_model=gate)
                decisions=[]
                for start in range(0,len(wav),HOP):
                    d=stream.push(wav[start:start+HOP],start_sample=start)
                    if len(d)!=int(start+HOP<=len(wav)):
                        raise AssertionError("not one fresh 256 PCM Onset output")
                    decisions.extend(d)
                if (len(decisions)!=n or stream.calls!=n or
                    stream.risk_calls!=n or
                    stream.fn_recovery_calls!=n or stream.fp_calls!=n or
                    stream.fp_gate_calls!=n or
                    stream.onset_heads.fp_gate.calls!=n or
                    gate.calls-previous_gate_calls!=n or
                    stream.onset_heads.main.calls!=n or
                    stream.onset_heads.rescue.calls!=n or
                    stream.onset_heads.error.calls!=n or
                    stream.onset_heads.recovery.calls!=n or
                    risk.calls-original_risk!=n or
                    recovery.calls-original_recovery!=n or
                    any(control[name].calls-before[name]!=n for name in VARIANTS) or
                    any(stream.onset_heads.fp_heads[name].calls!=n
                        for name in VARIANTS) or
                    any(d.sample!=(i+1)*HOP for i,d in enumerate(decisions))):
                    raise AssertionError("not all five independent heads causal each 256")
                stream.finish(meta["samples"])
                # Annotation may ONLY enter evaluation after complete PCM inference.
                truth=meta["fit_truth"]
                ends=range(256,256*(n+1),256)
                raw=score(decisions,truth,ends)
                old3=score(decisions,truth,ends,axis="predecision_intents")
                old4=score(decisions,truth,ends,axis="fn_unified_intents")
                candidates=[e for d in decisions for e in d.predecision_intents]
                four=[e for d in decisions for e in d.fn_unified_intents]
                if len(four)<len(candidates):
                    raise AssertionError("original fourth head note recovery disappeared")
                by_variant={}
                for name in VARIANTS:
                    old_recover=[e for d in decisions
                                 for e in d.fn_recovery_intents]
                    filtered=[e for d in decisions
                              for e in d.fp_discriminator_intents[name]]
                    combined=[e for d in decisions
                              for e in d.fp_discriminator_unified[name]]
                    if (len(combined)!=len(filtered)+len(old_recover)
                        or not {(e.sample,e.pitch) for e in filtered}.issubset(
                            {(e.sample,e.pitch) for e in candidates})):
                        raise AssertionError("FP specialist invented a NoteOn or modified FNRecovery")
                    matched=match_onsets(combined,truth)
                    by_variant[name]=dict(
                        tp=len(matched),
                        fp=len(combined)-len(matched),
                        fn=len(truth)-len(matched))
                gated=[e for d in decisions for e in d.fp_gated_unified]
                risk_keys={(e.pitch,e.sample) for e in candidates}
                base_keys={(e.pitch,e.sample) for e in old_recover}
                if (not {(e.pitch,e.sample) for e in gated}.issubset(
                        risk_keys|base_keys) or
                    not base_keys.issubset({(e.pitch,e.sample) for e in gated})):
                    raise AssertionError("FP gate invented a new attack or removed FNRecovery")
                gate_matches=match_onsets(gated,truth)
                gate_result=dict(tp=len(gate_matches),
                    fp=len(gated)-len(gate_matches),
                    fn=len(truth)-len(gate_matches))
                records.append(dict(index=it["index"],track=it["name"],
                                    composition=comp,frames=n,
                                    native=raw,old3=old3,old4=old4,
                                    variants=by_variant,
                                    gated=gate_result))
                if len(records)%8==0:
                    print("ONSET_FP_ABLATION_TRUE_PCM256_PROGRESS",len(records),flush=True)
    if len(records)!=80 or sum(r["frames"] for r in records)!=306536:
        raise AssertionError("not the correct 80 WAV/306536 256 hop science")
    def agg(key,variant=None):
        scores=[r[key] if variant is None else r[key][variant] for r in records]
        return stats(*(sum(s[k] for s in scores) for k in ("tp","fp","fn")))
    base=agg("native");third=agg("old3");fourth=agg("old4")
    if (any(base[k]!=v for k,v in REAL_S613.items())
        or any(third[k]!=v for k,v in BASE_3.items())
        or any(fourth[k]!=v for k,v in BASE_4.items())):
        raise AssertionError("independent FP heads altered one of four frozen references")
    totals={name:agg("variants",name) for name in VARIANTS}
    gated_total=agg("gated")
    temporal=totals["temporal74"];control69=totals["control69"]
    if any(control69[k]!=v for k,v in dict(tp=8127,fp=4519,fn=3140).items()):
        raise AssertionError("Gate run altered heldout CONTROL 69D scientific reference")
    control70=totals["control70"]
    fp_pass=(temporal["fp"]<control69["fp"]
             and temporal["fp"]<control70["fp"])
    f1_pass=(temporal["f1"]>=MIN_F1_OLD_FOUR
             and temporal["f1"]>=control70["f1"])
    pass_all=(gated_total["fp"]<control69["fp"] and
              gated_total["f1"]>control69["f1"]+1e-12)
    bycomposition={}
    for comp in COMPS:
        subset=[r for r in records if r["composition"]==comp]
        def total_for(field,variant=None):
            rows=[r[field] if variant is None else r[field][variant]
                  for r in subset]
            return stats(*(sum(t[k] for t in rows) for k in ("tp","fp","fn")))
        bycomposition[comp]=dict(
            original_four=total_for("old4"),
            control69=total_for("variants","control69"),
            control70=total_for("variants","control70"),
            temporal74=total_for("variants","temporal74"),
            learned_gate=total_for("gated"),
            gate_enabled=gates[comp].enabled,
            training=lineage[comp])
    regressions={
        name:[dict(track=row["track"],comp=row["composition"],
                   old_four_f1=f1(**row["old4"]),
                   new_f1=f1(**row["variants"][name]),
                   old_four_fp=row["old4"]["fp"],
                   new_fp=row["variants"][name]["fp"])
              for row in records
              if f1(**row["variants"][name])<f1(**row["old4"])-1e-12]
        for name in VARIANTS}
    gate_regressions=[
        dict(track=r["track"],composition=r["composition"],
             source69=r["variants"]["control69"],
             learned_gate=r["gated"],
             delta_fp=r["gated"]["fp"]-r["variants"]["control69"]["fp"],
             delta_tp=r["gated"]["tp"]-r["variants"]["control69"]["tp"])
        for r in records
        if f1(**r["gated"])<f1(**r["variants"]["control69"])-1e-12]
    report=dict(schema="ONSET_FP_EXPERT_GATE_CAL_SPLIT_REAL_PCM256_V1",complete=True,
        tracks=80,frames=306536,
        native_pcm_samples=256,model_training_hop_samples=256,
        truly_replayed_original_audio=True,
        neural_models_trained=15,thresholds_frozen_before_outer=True,
        five_composition_heldout_new_discriminators=True,
        frozen_teacher_training_overlap=True,
        old_main_s613=base,old_three_head=third,old_four_head=fourth,
        fifth_head_three_ablations=totals,
        learned_gate_real=gated_total,
        control69_reference_f1=control69["f1"],
        control69_reference_fp=control69["fp"],
        gate_enabled_folds=sum(m.enabled for m in gates.values()),
        gate_per_track_regressions=gate_regressions,
        H_FP_EXPERT_GATE_02_preregistered_issue=143,
        hypothesis_gate_beats_control69_f1=gated_total["f1"]>control69["f1"],
        hypothesis_gate_reduces_fp=gated_total["fp"]<control69["fp"],
        bycomposition=bycomposition,per_track=records,
        regressed_tracks=regressions,
        original_five_other_axes_unchanged=True,
        production_midi_untouched=True,
        no_1024_trained_weights=True,
        no_future_audio_no_hold=True,
        scientific_gate_FP_vs_both_controls=fp_pass,
        scientific_gate_F1_vs_prior_four_and_70D=f1_pass,
        scientific_hypothesis_142_supported=pass_all,
        preregistered_rule=THRESHOLD,promoted=False,
        full_pipeline_external_generalization=False,
        measured_system_e2e_latency_p95=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_FP_GATE_CAL_SPLIT_TRUE_WAV256_FINAL",
          json.dumps({k:report[k] for k in ("old_four_head",
             "fifth_head_three_ablations","learned_gate_real",
             "gate_enabled_folds",
             "hypothesis_gate_beats_control69_f1",
             "hypothesis_gate_reduces_fp")}),flush=True)
    if not pass_all:
        raise AssertionError(
            "H_FP_EXPERT_GATE_02 refuted or incomplete: native256 learned gate did not simultaneously exceed F1 control69 and reduce FP; DO NOT promote")
    return report

def main():
    p=argparse.ArgumentParser()
    for x in ("audio-zip","traces","manifest","native-model","s701-model",
              "old-risk-models","old-fn-models","fp-models","gate-models","out"):
        p.add_argument("--"+x,required=True,type=Path)
    a=p.parse_args()
    return run(a.audio_zip,a.traces,a.manifest,a.native_model,
        a.s701_model,a.old_risk_models,a.old_fn_models,a.fp_models,a.gate_models,a.out)
if __name__=="__main__":main()
