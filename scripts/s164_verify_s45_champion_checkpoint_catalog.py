"""S164: verify EXACT S44 champion IDs and 304 original S45 Ridge checkpoints.

Proof covers immutable source family names, variants, SHA-256 prediction
vectors and *trained checkpoint provenance*. It DOES NOT demonstrate that
the original 16 source models can make predictions on a NEW WAV.
No GOAT MIDI/label access; original S80 promotion forbidden.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib

def need(ok,msg):
    if not ok:raise ValueError(msg)

def audit(catalog,graph,s98,s45,out):
    need(not out.exists(),"S164 evidence immutable")
    c=json.loads(catalog.read_text())
    g=json.loads(graph.read_text())
    p=json.loads(s98.read_text())
    need(p["families_found"]==16 and
         p["status"]=="S98_S45_16_ORIGINAL_PROPOSAL_PROVENANCE_MAPPED",
         "unpinned S98 original champion evidence")
    expected={x["family"]:x for x in c["original_families"]}
    need(len(expected)==16 and len(c["original_families"])==16,
         "sixteen original families lost")
    need({f["id"] for f in g["families"]}==set(expected),
         "authoritative dependency graph includes invented/aliased family name")
    need(set(p["original_families"])==set(expected),
         "original S98 source keys inconsistent")
    records=[]
    for family in sorted(expected):
        name=expected[family]
        data=p["original_families"][family]
        need(name["original_variant"]==data["original_variant"] and
             name["original_vector_sha256"]==data["original_vector_sha256"],
             "original frozen family champion SHA mismatch for "+family)
        files=sorted((s45/"models").glob("*__"+family+".joblib"))
        need(len(files)==19,"S45 missing fitted original checkpoints for "+family)
        source_digests=set()
        for path in files:
            model=joblib.load(path) # fixed original GitHub run, trusted model
            need(model.get("family")==family and
                 name["original_vector_sha256"]==model.get("source_original_sha256"),
                 "S45 trained model source SHA drift: "+path.name)
            need(model["model"].n_features_in_==156 and
                 model["scaler"].n_features_in_==156,
                 "S45 checkpoint numeric trained input schema")
            source_digests.add(model["source_original_sha256"])
            records.append({"family":family,"trained_piece":model["piece"],
                  "fold":int(model["fold"]),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        need(len(source_digests)==1,"S45 trained source cohort inconsistent")
    need(len(records)==304,"all 304 historic trained checkpoints must be present")
    out.parent.mkdir(parents=True,exist_ok=True)
    report={"status":"S164_EXACT_S44_S45_ORIGINAL_16_SOURCE_AND_304_TRAINED_CHECKPOINT_PROVENANCE",
       "all_sixteen_original_family_ids_verified":True,
       "all_original_vector_sha256_against_S98_verified":True,
       "all_304_trained_ridge_family_model_source_sha256_verified":True,
       "historical_trained_checkpoint_count":len(records),
       "original_families":sorted(expected),
       "checkpoints":records,
       "original_S45_source_model_inference_on_new_audio_complete":False,
       "original_S80_full_inference_scored_independently":False,
       "promoted":False}
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
       "status","all_sixteen_original_family_ids_verified",
       "historical_trained_checkpoint_count",
       "original_S45_source_model_inference_on_new_audio_complete","promoted")},indent=2))
def selftest():
    ids=("S11_votes_time_audio","S12_audio_logistic","S41_neural_fusion")
    need(len(set(ids))==3 and not set(ids).intersection(
         {"S11_votes","S12_logistic","S41_fusion"}),"aliased source keys")
    print("S164 PASS: original sixteen-family strict alias guard")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("catalog","graph","s98","s45","out"):p.add_argument("--"+n,type=Path)
    p.add_argument("--self-test",action="store_true")
    x=p.parse_args()
    if x.self_test:selftest()
    else:
        need(all(getattr(x,n) is not None for n in (
         "catalog","graph","s98","s45","out")),"all original artifacts required")
        audit(x.catalog,x.graph,x.s98,x.s45,x.out)
