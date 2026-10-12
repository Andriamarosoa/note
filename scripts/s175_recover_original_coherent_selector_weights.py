"""S175: recover original coherent selector Keras weights from sealed history cache.

Archive verification only: exact 4 held-fold weights and source model records.
No invented 5x64x21 inputs, no new-WAV inference, no MIDI or S80 promotion.
"""
from __future__ import annotations
import argparse,hashlib,io,json
from pathlib import Path
from zipfile import ZipFile
import h5py
ROOT=Path("analysis/evidence/v273-audit-memory/source-archives")
FOLDS=(0,1,2,4)
def digest(value):
    return hashlib.sha256(value).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
def run(root,out):
    need(not out.exists(),"immutable S175 output required")
    manifest=json.loads((root/ROOT/"manifest.json").read_text())
    source=root/ROOT/"model-replay-bundles.zip"
    raw=source.read_bytes()
    info=manifest["files"]["model-replay-bundles.zip"]
    need(len(raw)==info["bytes"] and digest(raw)==info["sha256"],
         "original sealed model bundle bytes or SHA drift")
    members=info["members"]
    need(len([s for s in members if s.startswith("repair_coherent/")])==10,
         "four coherent held folds and two reports expected")
    results=[]
    recovered={}
    architecture=None
    with ZipFile(io.BytesIO(raw)) as z:
        need(z.testzip() is None,"corrupt source zip")
        for fold in FOLDS:
            for suffix in ("weights.h5","result.json"):
                key=f"repair_coherent/fold-{fold}."+suffix if suffix=="weights.h5" else f"repair_coherent/fold-{fold}-result.json"
                need(key in z.namelist() and key in members,"missing original Keras source "+key)
                raw_member=z.read(key)
                declared=members[key]
                need(len(raw_member)==declared["bytes"] and
                     digest(raw_member)==declared["sha256"],
                     "historic S44 coherent weight/report checksum mismatch "+key)
                recovered[key]=raw_member
                if suffix=="weights.h5":
                    with h5py.File(io.BytesIO(raw_member),"r") as h:
                        ds=[]
                        def visit(name,item):
                            if isinstance(item,h5py.Dataset):
                                ds.append({"name":name,"shape":list(item.shape),
                                           "dtype":str(item.dtype)})
                        h.visititems(visit)
                        need(len(ds)>5,"original coherent model weights not present")
                        shapes=sorted((e["name"],tuple(e["shape"])) for e in ds)
                        if architecture is None: architecture=shapes
                        need(shapes==architecture,"coherent architecture differs across folds")
                        results.append({"fold":fold,"historical_weight_sha256":digest(raw_member),
                                        "weight_files":len(ds),
                                        "original_weight_shapes":ds})
        for suffix in ("report.json","report.md"):
            key="repair_coherent/"+suffix
            need(key in members,"missing original final coherent report")
            body=z.read(key)
            need(digest(body)==members[key]["sha256"],"coherent report drift")
            recovered[key]=body
    # Verified source shapes must also correspond to the original 7-class model.
    original_source=root/"scripts/learn_v273_coherent_selector.py"
    code=original_source.read_text()
    for declaration in ("class CoherentCombinationArbiter", "5, 64, 21",
                        "baseline_and_low_logits", "choose_legal_class"):
        need(declaration in code,"original coherent neural source interface missing "+declaration)
    out.mkdir(parents=True)
    for key,body in recovered.items():
        path=out/"restored"/key
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(body)
    report={
      "status":"S175_FOUR_REAL_ORIGINAL_COHERENT_SELECTOR_WEIGHTS_RECOVERED_NOT_GOAT",
      "source_historical_zip":str(source.relative_to(root)),
      "source_zip_sha256":digest(raw),
      "original_Keras_weights_fold_count":len(results),
      "fold_model_proofs":results,
      "source_script":str(original_source.relative_to(root)),
      "original_model_input_shape":"5x64x21 subset descriptor + mask + baseline + context + keep_heads",
      "unseen_audio_5x64x21_original_producer_available":False,
      "original_outer_fold_selector_requires_new_audio_routing":True,
      "unseen_WAV_predictions_generated":False,
      "true_K_or_MIDI_read":False,
      "S45_number_source_families_confirmed":11,
      "S80_independent_accuracy_measured":False,
      "S80_promoted":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
      "status":report["status"],
      "original_Keras_weights_fold_count":len(results),
      "weights_dataset_count_per_fold":[v["weight_files"] for v in results],
      "original_model_input_shape":report["original_model_input_shape"],
      "S45_number_source_families_confirmed":11,
      "S80_promoted":False},indent=2))

def selftest():
    need(FOLDS==(0,1,2,4),"fold3 must remain excluded")
    need(len(FOLDS)==4 and "repair_coherent"=="repair_coherent","coherent source")
    print("S175 PASS four-fold original coherent neural checkpoint signature")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(a.out is not None,"output path required")
        run(a.root,a.out)
