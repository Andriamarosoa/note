"""S335: recover 527D original S5 features and PR16 geometry on S332/S334 native rows.

Label-free source, no checkpoints trained or revised. Inputs must be exact
canonical new Dataset1 event IDs. Replay actual original historical
S4-owned 527 feature constructor and original PR16 temporal ownership.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s226_original_S71_S72_frozen_on_GOAT import geometry46
from scripts.v273_s4_owned527 import build_owned527

AUDIO={"row_id","member","start_samples","incumbent_freeze_K","summary58",
       "sequence42x49","flow245","morphology490","physics294"}
BASELINE={"row_id","member","start_samples","end_samples","incumbent_K",
          "incumbent_probabilities"}
def need(v,why):
    if not v:raise ValueError(why)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==fields,"unexpected label or source field: "+str(path))
        return {k:z[k].copy() for k in z.files}
def run(s334,s332,out):
    need(not out.exists(),"S335 native candidate feature bank must be immutable")
    a=read(s334,AUDIO)
    b=read(s332,BASELINE)
    n=len(a["row_id"])
    need(n==477 and all(np.array_equal(a[k],b[k]) for k in
        ("row_id","member","start_samples"))
        and np.array_equal(a["incumbent_freeze_K"],b["incumbent_K"]),
        "S335 must align with all S332 original native V88 events")
    need(b["end_samples"].shape==(n,) and
         np.all(b["end_samples"]>=b["start_samples"]) and
         np.all(b["end_samples"]-b["start_samples"]<=1764),
         "native V88 <=40ms boundaries not preserved")
    geom=geometry46({k:b[k] for k in ("member","start_samples","end_samples")})
    need(geom.shape==(n,46) and np.isfinite(geom).all() and
         np.isin(geom[:,:42],[0,1]).all(),
         "historical source PR16 ownership mask differs")
    x=build_owned527(a["summary58"],geom,a["sequence42x49"],b["incumbent_K"])
    need(x.shape==(n,527) and np.isfinite(x).all(),
         "S5 527D feature inputs from true external native audio malformed")
    out.mkdir(parents=True)
    path=out/"S335_IDMT_dataset1_true_original_S5_527D_input.npz"
    np.savez_compressed(path,row_id=b["row_id"],member=b["member"],
      start_samples=b["start_samples"],end_samples=b["end_samples"],
      incumbent_freeze_K=b["incumbent_K"],
      original_PR16_geometry46=geom,original_S5_input_owned527=x)
    rep=dict(status="S335_TRUE_EXTERNAL_IDMT_DATASET1_S5_527D_INPUT_NO_LABELS",
       native_events=n,WAV_count=12,
       true_original_S332_baseline_SHA256=sha(s332),
       true_original_S334_S97_features_SHA256=sha(s334),
       original_feature_width=527,
       original_46D_PR16_ownership=True,
       original_trained_S5_model_checkpoints_applied=False,
       no_XML_or_MIDI_contents_read=True,
       source_models_retrained=False,
       new_S275_full_prediction_not_yet_frozen=True,
       output_SHA256=sha(path),independent_ExactK_measured=False,
       S80_promoted=False)
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("s334","s332","out"):p.add_argument("--"+k,required=True,type=Path)
    a=p.parse_args();run(a.s334,a.s332,a.out)
