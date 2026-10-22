"""Strictly causal REAL neural PRE-DECISION Onset replay on signed GuitarSet PCM256.

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
from onset256.predecision_risk import load_frozen,SCHEMA
from onset256.runtime import Stream
from scripts.replay_native256 import score

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
        path=trained_models/"checkpoints"/f"onset_risk_outer_{comp}.joblib"
        meta_path=trained_models/"checkpoints"/f"onset_risk_outer_{comp}.json"
        meta=json.loads(meta_path.read_text())
        if meta.get("outer_composition")!=comp or meta.get("schema")!=SCHEMA:
            raise ValueError("outer-fold weight leakage or wrong model")
        if comp in set(meta.get("train_compositions",())) or comp==meta.get("cal_composition"):
            raise ValueError("outer composition present during risk head training/calibration")
        experts[comp]=load_frozen(path,meta_path,expected_sha256=meta["checkpoint_sha256"])
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
                    or model.calls-count_start!=len(ends)):
                    raise AssertionError("neural risk did not run BEFORE EVERY Onset decision")
                if any(v.sample!=int(t) for v,t in zip(decisions,ends)):
                    raise AssertionError("not strict causal PCM256")
                stream.finish(meta["samples"])
                # Never allow the GT to affect the model's selected event stream.
                baseline=score(decisions,meta["fit_truth"],ends)
                risk=score(decisions,meta["fit_truth"],ends,axis="predecision_intents")
                rows.append(dict(index=item["index"],name=meta["name"],
                                 composition=comp,frames=len(ends),
                                 native=baseline,neural_predecision=risk,
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
    changes=[dict(name=r["name"],composition=r["composition"],
                  delta_f1=metrics(**r["neural_predecision"])["f1"]-
                           metrics(**r["native"])["f1"],
                  delta_tp=r["neural_predecision"]["tp"]-r["native"]["tp"],
                  delta_fp=r["neural_predecision"]["fp"]-r["native"]["fp"])
             for r in rows if r["native"]!=r["neural_predecision"]]
    winner=neural["f1"]>=baseline["f1"]
    report=dict(schema="ONSET_NEURAL_PREDECISION_PCM256_CAUSAL_OOF_REAL_REPLAY_V1",
        complete=True,tracks=80,frames=306536,source_inference_hop=256,
        model_training_hop=256,risk_model_invoked_before_selector=True,
        source_no_s708_temporal_weight=True,
        native=baseline,neural_predecision=neural,
        paired_changes=changes,changed_tracks=len(changes),
        folds={c:dict(cut=experts[c].cut) for c in COMPOSITIONS},
        historical_H0_f1=2*8528/(2*8528+3489+2739),
        oof_risk_head_compositions=True,
        frozen_teacher_fit80_train_overlap=True,
        independent_generalization_of_entire_pipeline=False,
        scientific_non_regression_vs_S613=winner,
        scientific_non_regression_vs_S708=neural["f1"]>=2*8528/(2*8528+3489+2739),
        public_midi_modified=False,other_axes_modified=False,
        promoted=False,results=rows)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_PREDECISION_MLP256_CAUSAL_FIT80_FINAL",
          json.dumps({k:report[k] for k in ("native","neural_predecision",
              "changed_tracks","scientific_non_regression_vs_S613",
              "scientific_non_regression_vs_S708")}),flush=True)
    # The run can be scientifically red; retaining report is mandatory.
    if not winner:
        raise AssertionError("neural predecision Onset regressed S613: never promote")
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
