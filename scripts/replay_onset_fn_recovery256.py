"""True causal real-WAV replay of independent fourth FNRecovery Onset head.

Models: frozen S613/S701 (native 256), frozen ErrorPrediction MLP69,
separately trained FNRecovery MLP69 (15 models trained, CAL selects five).
Every head predicts before each Onset selector on each 256-sample hop.
No future, no GT during inference, no official MIDI / other axis touched.
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
from onset256.runtime import Stream
from scripts.replay_native256 import score,match_onsets
from scripts.train_onset_predecision256 import COMPOSITIONS

EXPECTED_MAIN=dict(tp=8099,fp=4600,fn=3168)
EXPECTED_MLP69=dict(tp=8054,fp=4411,fn=3213)
REFERENCE_FOUR_HEAD_F1=0.6790945406125166
REFERENCE_FOUR_HEAD_FP=4605
REFERENCE_FOUR_HEAD_TP=8160
# Prospective issue #141: add unique TRAIN negatives while preserving ~1:3
# learning class ratio; reject promotion unless real audio improves all
# predeclared primary criteria, irrespective of offline candidate proxy.
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()
def stats(tp,fp,fn):
    den=2*tp+fp+fn
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),
                f1=2*tp/den if den else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.)
def run(audio_zip,traces,manifest,native_model,s701_model,risk_models,
        recovery_models,out):
    if out.exists():raise FileExistsError("never overwrite scientific experiment")
    manifest_obj=json.loads(Path(manifest).read_text())
    if (manifest_obj.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
        or not manifest_obj.get("complete") or
        len(manifest_obj.get("fit_files",()))!=80):
        raise ValueError("signed 80-original GuitarSet corpus required")
    native,s701=load(native_model,s701_model)
    risk_heads={};recover_heads={};foldinfo={}
    for comp in COMPOSITIONS:
        old=Path(risk_models)/"checkpoints"
        pa=old/f"onset_error_outer_{comp}.joblib"
        mt=old/f"onset_error_outer_{comp}.json"
        mi=json.loads(mt.read_text())
        if comp!=mi["outer_composition"] or comp in mi["train_compositions"]:
            raise ValueError("third risk head has outer composition leakage")
        risk_heads[comp]=load_hard_model(pa,mt,expected_sha256=mi["checkpoint_sha256"])
        root=Path(recovery_models)/"checkpoints"
        pn=root/f"fn_recovery_outer_{comp}.joblib"
        mn=root/f"fn_recovery_outer_{comp}.json"
        inf=json.loads(mn.read_text())
        if (comp!=inf["outer_composition"] or comp in inf["train_compositions"]
            or comp==inf["cal_composition"] or
            len(inf["train_compositions"])!=3):
            raise ValueError("fourth FN head not composition-disjoint")
        recover_heads[comp]=load_fn_recovery(
            pn,mn,expected_sha256=inf["checkpoint_sha256"])
        foldinfo[comp]=dict(policy=inf["candidate_policy"],
                            cut=inf["calibrated_cut"],
                            enabled=inf["enabled"],checkpoint_sha256=inf["checkpoint_sha256"])
    out.mkdir(parents=True)
    rows=[]
    bycomp=Counter()
    with zipfile.ZipFile(audio_zip) as z:
        for comp in COMPOSITIONS:
            files=sorted((v for v in manifest_obj["fit_files"]
                          if v["composition"]==comp),key=lambda x:x["index"])
            if len(files)!=16:raise ValueError("16 original tracks per outer composition")
            for item in files:
                mp=Path(traces)/f"fit_{item['index']:03d}.json"
                if sha256(mp)!=item["json_sha256"]:
                    raise ValueError("signed original truth changed")
                meta=json.loads(mp.read_text())
                if meta["name"]!=item["name"]:
                    raise ValueError("WAV-truth identity mismatch")
                rate,pcm=wavfile.read(io.BytesIO(z.read(meta["name"]+"_mic.wav")))
                if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                    raise ValueError("original PCM WAV not signed correctly")
                audio=pcm[:meta["samples"]].astype(np.float32)/32768
                ends=np.arange(HOP,len(audio)+1,HOP)
                third=risk_heads[comp]
                fourth=recover_heads[comp]
                old_calls=(third.calls,fourth.calls)
                stream=Stream(native,s701,risk_model=third,recovery_model=fourth)
                decisions=[]
                for start in range(0,len(audio),HOP):
                    seq=stream.push(audio[start:start+HOP],start_sample=start)
                    if len(seq)!=int(start+HOP<=len(audio)):
                        raise AssertionError("FNRecovery missed causal PCM256 boundary")
                    decisions.extend(seq)
                n=len(ends)
                if (len(decisions)!=n or stream.calls!=n
                    or stream.s701_calls!=n or stream.risk_calls!=n
                    or stream.fn_recovery_calls!=n
                    or stream.onset_heads.main.calls!=n
                    or stream.onset_heads.rescue.calls!=n
                    or stream.onset_heads.error.calls!=n
                    or stream.onset_heads.recovery.calls!=n
                    or third.calls-old_calls[0]!=n
                    or fourth.calls-old_calls[1]!=n
                    or stream.fn_recovery_arbiter.calls!=n):
                    raise AssertionError("four independent learned Onset head calls missing")
                if any(d.sample!=int(end) for d,end in zip(decisions,ends)):
                    raise AssertionError("new fourth head delayed Onset decisions")
                stream.finish(meta["samples"])
                # Only after ENTIRE PCM256 audio-only forward pass access GT.
                truth=meta["fit_truth"]
                main=score(decisions,truth,ends)
                third_sc=score(decisions,truth,ends,axis="predecision_intents")
                combined=score(decisions,truth,ends,axis="fn_unified_intents")
                source=[e for d in decisions for e in d.intents]
                recovered=[e for d in decisions for e in d.fn_recovery_intents]
                risk_stream=[e for d in decisions for e in d.predecision_intents]
                combined_stream=[e for d in decisions for e in d.fn_unified_intents]
                if (len(combined_stream)!=len(risk_stream)+len(recovered)
                    or any(e.sample%256 for e in recovered)):
                    raise AssertionError("FN head corrupted existing NoteOn path")
                if any((e.pitch,e.sample) in {(x.pitch,x.sample) for x in source}
                       for e in recovered):
                    raise AssertionError("FN head duplicated main attacks")
                matches=match_onsets(combined_stream,truth)
                recovered_match=sum(i>=len(risk_stream) for i,_ in matches)
                # Note: model reports stream flattened BY-FRAME not all main then
                # extra, so match by the event's independent reason instead.
                recovered_match=sum(
                    combined_stream[i].reason=="fn_recovery_trained_neural256"
                    for i,_ in matches)
                third_matches=match_onsets(risk_stream,truth)
                t=dict(index=item["index"],name=item["name"],composition=comp,
                       frames=n,main=main,third=third_sc,combined=combined,
                       recovered_proposals=len(recovered),
                       recovered_matched_final=int(recovered_match),
                       tp_delta=int(combined["tp"]-third_sc["tp"]),
                       fp_delta=int(combined["fp"]-third_sc["fp"]),
                       fn_delta=int(combined["fn"]-third_sc["fn"]),
                       original_main_events=len(source),
                       frozen_third_events=len(risk_stream))
                rows.append(t)
                bycomp[(comp,"recovered")]+=len(recovered)
                if len(rows)%8==0:
                    print("ONSET_FN_RECOVERY_FOUR_HEAD_CAUSAL_PROGRESS",
                          len(rows),flush=True)
    def aggregate(field):
        d={k:sum(int(row[field][k]) for row in rows)
           for k in ("tp","fp","fn")}
        return stats(**d)
    main=aggregate("main")
    third=aggregate("third")
    combo=aggregate("combined")
    if any(main[k]!=v for k,v in EXPECTED_MAIN.items()):
        raise AssertionError("fourth head altered S613 main reference")
    if any(third[k]!=v for k,v in EXPECTED_MLP69.items()):
        raise AssertionError("fourth head altered ErrorPrediction 69D correction")
    if len(rows)!=80 or sum(row["frames"] for row in rows)!=306536:
        raise AssertionError("four-head Onset did not replay 80 original WAVs")
    by_comp={}
    for comp in COMPOSITIONS:
        subset=[x for x in rows if x["composition"]==comp]
        by_comp[comp]=dict(**foldinfo[comp],
              baseline=stats(*[sum(r["third"][k] for r in subset)
                               for k in ("tp","fp","fn")]),
              combined=stats(*[sum(r["combined"][k] for r in subset)
                               for k in ("tp","fp","fn")]),
              recovered_proposals=sum(x["recovered_proposals"] for x in subset))
    win=(combo["f1"]>REFERENCE_FOUR_HEAD_F1+1e-12 and
         combo["fp"]<REFERENCE_FOUR_HEAD_FP and
         combo["tp"]>third["tp"])
    regressions=[dict(track=row["name"],composition=row["composition"],
                      delta_tp=row["tp_delta"],delta_fp=row["fp_delta"],
                      delta_fn=row["fn_delta"])
                 for row in rows if (row["tp_delta"]<0 or row["fp_delta"]>0)]
    result=dict(schema="ONSET_FOUR_INDEPENDENT_HEADS_FN_RECOVERY_PCM256_REAL_WAV_V1",
        complete=True,tracks=80,pcm256_frames=306536,
        four_heads=("main_attack","rescue_attack","error_prediction","fn_recovery"),
        first_three_head_reference_unchanged=True,
        predecision_all_heads_each_256=True,
        source_trained_hop=256,recovery_trained_hop=256,
        main=main,frozen_three_head_MLP69=third,four_head_unified=combo,
        candidate_added_total=sum(x["recovered_proposals"] for x in rows),
        recovered_gt_matches_in_final=sum(x["recovered_matched_final"] for x in rows),
        folds=by_comp,per_track=rows,regressions=regressions,
        oracle_candidate_counts_are_not_F1=True,
        historical_train_overlap_s613_s701=True,
        independently_validated_on_unseen_corpus=False,
        strict_improvement_vs_three_head=combo["f1"]>third["f1"],
        preregistered_negative_coverage_ablation_issue=141,
        previous_four_head_f1=REFERENCE_FOUR_HEAD_F1,
        previous_four_head_tp=REFERENCE_FOUR_HEAD_TP,
        previous_four_head_fp=REFERENCE_FOUR_HEAD_FP,
        new_fp_from_recovery=int(combo["fp"]-third["fp"]),
        new_tp_from_recovery=int(combo["tp"]-third["tp"]),
        scientific_gate_f1_beats_previous_four_head=combo["f1"]>REFERENCE_FOUR_HEAD_F1,
        scientific_gate_fp_decreases_vs_previous_four_head=combo["fp"]<REFERENCE_FOUR_HEAD_FP,
        scientific_gate_tp_above_three_head=combo["tp"]>third["tp"],
        strict_preregistered_improvement=win,
        external_latency_p95_unmeasured=True,
        production_midi_unchanged=True,other_five_axes_unchanged=True,
        promoted=False)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("ONSET_FN_RECOVERY_FOUR_HEAD_TRUE_256_WAV_FINAL",
          json.dumps({k:result[k] for k in (
              "main","frozen_three_head_MLP69","four_head_unified",
              "candidate_added_total","recovered_gt_matches_in_final",
              "strict_improvement_vs_three_head","strict_preregistered_improvement",
              "scientific_gate_fp_decreases_vs_previous_four_head",
              "new_fp_from_recovery","new_tp_from_recovery","folds")}),flush=True)
    if not win:
        raise AssertionError("H_FN_NEGCOVERAGE_01 predeclared real-WAV F1/FP gate failed: audit hypothesis refuted or incomplete; DO NOT promote")
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest","native-model","s701-model",
                 "risk-models","fn-models","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    run(a.audio_zip,a.traces,a.manifest,a.native_model,a.s701_model,
        a.risk_models,a.fn_models,a.out)
if __name__=="__main__":main()
