"""Hard-negative learned error HEAD live replay on original signed GuitarSet PCM256.

Loads one trained risk MLP per composition (three TRAIN, one CAL, one OUTER).
For every 256 samples, a neural P(correct) is produced for 49 pitches BEFORE
the private Onset selector is permitted to emit notes. The original frozen
S613 and S701 outputs remain bit-for-bit independent controls. GT is read
only after each complete PCM track. No production MIDI or other axes touched.
"""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from onset256.acoustics import HOP,SR
from onset256.model import load
from onset256.error_evidence69 import load_hard_model,SCHEMA
from onset256.runtime import Stream
from scripts.replay_native256 import score,match_onsets

COMPOSITIONS=("BN1","Funk2","Jazz1","Rock3","SS1")
EXPECTED_NATIVE=dict(tp=8099,fp=4600,fn=3168)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(tp,fp,fn):
    den=2*tp+fp+fn
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),
                f1=2*tp/den if den else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.)

def run(audio_zip,traces,manifest,native_model,s701_model,trained_models,out):
    if out.exists():
        raise FileExistsError("never overwrite independent OOF experiment")
    signed=json.loads(manifest.read_text())
    if signed.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1" or not signed.get("complete"):
        raise ValueError("unsigned FIT80")
    if len(signed.get("fit_files",[]))!=80:raise AssertionError("not 80 tracks")
    native,s701=load(native_model,s701_model)
    experts={}
    for comp in COMPOSITIONS:
        path=trained_models/"checkpoints"/f"onset_error_outer_{comp}.joblib"
        meta_path=trained_models/"checkpoints"/f"onset_error_outer_{comp}.json"
        meta=json.loads(meta_path.read_text())
        if meta.get("outer_composition")!=comp or meta.get("schema")!=SCHEMA:
            raise ValueError("outer-fold weight leakage or wrong model")
        if comp in set(meta.get("train_compositions",())) or comp==meta.get("cal_composition"):
            raise ValueError("outer composition present during risk head training/calibration")
        experts[comp]=load_hard_model(path,meta_path,expected_sha256=meta["checkpoint_sha256"])
    out.mkdir(parents=True)
    rows=[]
    with zipfile.ZipFile(audio_zip) as archive:
        for comp in COMPOSITIONS:
            items=sorted((a for a in signed["fit_files"] if a["composition"]==comp),
                         key=lambda a:a["index"])
            if len(items)!=16:raise AssertionError("16 tracks per composition")
            for item in items:
                filepath=traces/f"fit_{item['index']:03d}.json"
                if digest(filepath)!=item["json_sha256"]:
                    raise ValueError("signed metadata changed")
                meta=json.loads(filepath.read_text())
                if meta["name"]!=item["name"]:
                    raise ValueError("track identity mismatch")
                rate,pcm=wavfile.read(io.BytesIO(archive.read(meta["name"]+"_mic.wav")))
                if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                    raise ValueError("wrong original PCM")
                x=pcm[:meta["samples"]].astype(np.float32)/32768
                ends=np.arange(HOP,len(x)+1,HOP)
                model=experts[comp]
                count_start=model.calls
                stream=Stream(native,s701,risk_model=model)
                decisions=[]
                for start in range(0,len(x),HOP):
                    obs=stream.push(x[start:start+HOP],start_sample=start)
                    if len(obs)!=int(start+HOP<=len(x)):
                        raise AssertionError("missing fresh 256-sample decision")
                    decisions.extend(obs)
                if (len(decisions)!=len(ends) or stream.calls!=len(ends)
                    or stream.risk_calls!=len(ends) or stream.s701_calls!=len(ends)
                    or model.calls-count_start!=len(ends)
                    or stream.onset_heads.calls!=len(ends)
                    or stream.onset_heads.main.calls!=len(ends)
                    or stream.onset_heads.rescue.calls!=len(ends)
                    or stream.onset_heads.error.calls!=len(ends)):
                    raise AssertionError("independent neural Onset heads missed 256 decisions")
                if any(v.sample!=int(t) for v,t in zip(decisions,ends)):
                    raise AssertionError("not strict causal PCM256")
                stream.finish(meta["samples"])
                # Never allow the GT to affect the model's selected event stream.
                baseline=score(decisions,meta["fit_truth"],ends)
                risk=score(decisions,meta["fit_truth"],ends,axis="predecision_intents")
                # The error-prediction HEAD has spoken before decisions;
                # only now, post-inference, identify which doubts were correct.
                main_events=[e for d in decisions for e in d.intents]
                main_doubts=[a for d in decisions for a in d.head_doubts]
                if len(main_events)!=len(main_doubts):
                    raise AssertionError("every native NoteOn needs its risk-head opinion")
                for e,a in zip(main_events,main_doubts):
                    if (e.pitch!=a.pitch or e.sample!=a.sample or
                        a.origin!="learned_error_prediction_head"):
                        raise AssertionError("error head opinion not attached to main Onset")
                true_matches={i for i,_ in match_onsets(main_events,meta["fit_truth"])}
                challenged_tp=sum(i in true_matches and a.contested
                                  for i,a in enumerate(main_doubts))
                challenged_fp=sum(i not in true_matches and a.contested
                                  for i,a in enumerate(main_doubts))
                doubt=dict(challenged_tp=int(challenged_tp),
                           challenged_fp=int(challenged_fp),
                           retained_tp=int(baseline["tp"]-challenged_tp),
                           retained_fp=int(baseline["fp"]-challenged_fp),
                           main_proposals=len(main_doubts),
                           head_name="error_prediction")
                if (doubt["challenged_tp"]+doubt["retained_tp"]!=baseline["tp"]
                    or doubt["challenged_fp"]+doubt["retained_fp"]!=baseline["fp"]):
                    raise AssertionError("risk doubt lost genuine Onset candidates")
                rows.append(dict(index=item["index"],name=meta["name"],
                                 composition=comp,frames=len(ends),
                                 native=baseline,neural_predecision=risk,
                                 learned_head_doubt=doubt,
                                 calibrated_cut=float(model.cut)))
                if len(rows)%8==0:
                    print("ONSET_PREDECISION256_REPLAY_PROGRESS",len(rows),flush=True)
    def totals(key):
        vals={k:sum(int(row[key][k]) for row in rows) for k in ("tp","fp","fn")}
        return metrics(**vals)
    baseline=totals("native")
    neural=totals("neural_predecision")
    if any(baseline[k]!=v for k,v in EXPECTED_NATIVE.items()):
        raise AssertionError("original S613 frozen decisions mutated by risk head")
    if len(rows)!=80 or sum(row["frames"] for row in rows)!=306536:
        raise AssertionError("bad FIT80 native hop256 denominator")
    doubt_totals={key:sum(r["learned_head_doubt"][key] for r in rows)
                  for key in ("challenged_tp","challenged_fp",
                              "retained_tp","retained_fp","main_proposals")}
    doubt_totals["precision_for_fp"]=(
        doubt_totals["challenged_fp"] /
        (doubt_totals["challenged_fp"]+doubt_totals["challenged_tp"])
        if (doubt_totals["challenged_fp"]+doubt_totals["challenged_tp"])
        else 0.)
    changes=[dict(name=r["name"],composition=r["composition"],
                  delta_f1=metrics(**r["neural_predecision"])["f1"]-
                           metrics(**r["native"])["f1"],
                  delta_tp=r["neural_predecision"]["tp"]-r["native"]["tp"],
                  delta_fp=r["neural_predecision"]["fp"]-r["native"]["fp"])
             for r in rows if r["native"]!=r["neural_predecision"]]
    winner=neural["f1"]>=baseline["f1"]
    report=dict(schema="ONSET_HARDNEGATIVE_HARMONIC_RESTRIKE_MLP69_REAL_CAUSAL_PCM256_V1",
        complete=True,tracks=80,frames=306536,source_inference_hop=256,
        model_training_hop=256,risk_model_invoked_before_selector=True,
        source_no_s708_temporal_weight=True,
        native=baseline,neural_predecision=neural,
        hardnegative_head_features=69,
        hardnegative_head_trained_natively_256=True,
        specialist_hypotheses_tested=("balanced","harmonic","restrike"),
        decisions_made_after_frozen_three_head_stack=True,
        embedded_heads=("main_attack","rescue_attack","error_prediction"),
        risk_is_onset_architecture_head=True,
        learned_head_doubt_totals=doubt_totals,
        paired_changes=changes,changed_tracks=len(changes),
        folds={c:dict(cut=experts[c].cut,
                      expert=json.loads((trained_models/"checkpoints"/
                          f"onset_error_outer_{c}.json").read_text())["selected_expert"])
               for c in COMPOSITIONS},
        historical_H0_f1=2*8528/(2*8528+3489+2739),
        oof_risk_head_compositions=True,
        frozen_teacher_fit80_train_overlap=True,
        independent_generalization_of_entire_pipeline=False,
        scientific_non_regression_vs_S613=winner,
        scientific_non_regression_vs_S708=neural["f1"]>=2*8528/(2*8528+3489+2739),
        public_midi_modified=False,other_axes_modified=False,
        promoted=False,results=rows)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_HARDNEGATIVE_MLP69_TRUE_CAUSAL_FIT80_FINAL",
          json.dumps({k:report[k] for k in ("native","neural_predecision",
              "embedded_heads","learned_head_doubt_totals","changed_tracks",
              "folds","scientific_non_regression_vs_S613",
              "scientific_non_regression_vs_S708")}),flush=True)
    # The run can be scientifically red; retaining report is mandatory.
    if not winner:
        raise AssertionError("trained 69D error HEAD regressed S613 Onset; never promote")
    return report

def main():
    p=argparse.ArgumentParser()
    for flag in ("audio-zip","traces","manifest","native-model","s701-model",
                 "trained-models","out"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    run(a.audio_zip,a.traces,a.manifest,a.native_model,a.s701_model,
        a.trained_models,a.out)
if __name__=="__main__":
    main()
