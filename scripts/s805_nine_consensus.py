"""S805 fail-closed research 9/9; no musical promotion permitted."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,subprocess,re
from scripts.s804_nvidia_deep_review import ROLES,SCHEMA,quality_check
from scripts.s805_three_model_review import SOURCES
def checked(six,docs,r,expected_sha):
    if six.get("schema")!="NOTE_SIX_ROLE_CONSENSUS_V1" or (
        six.get("passed") is not True or six.get("audits_received")!=6 or
        six.get("issues") or six.get("official_promotion_eligible") is not False):
        raise AssertionError("six independent local scientific audits required")
    if set(six.get("auditors",{}))!={
        "data","acoustics","architecture","training","metrics","live"}:
        raise AssertionError("incomplete local roles")
    if set(docs)!=set(ROLES):raise AssertionError("3 different NVIDIA roles missing")
    for role,model in ROLES.items():
        x=docs[role]
        if x.get("schema")!=SCHEMA or x.get("model")!=model or (
            x.get("role")!=role or x.get("series")!="S805" or
            x.get("provider")!="NVIDIA NIM" or x.get("review_received") is not True):
            raise AssertionError("stale/foreign external model result")
        if x.get("source_sha")!=expected_sha or x.get("evidence_digest")!=r:
            raise AssertionError("cross-run evidence mismatch")
        text=x.get("review_text","")
        if not isinstance(text,str) or hashlib.sha256(text.encode()).hexdigest()!=x.get("response_sha256"):
            raise AssertionError("model response hash invalid")
        quality_check(text,SOURCES)
        if x.get("scientific_promotion") is not False:
            raise AssertionError("adviser cannot grant promotion")
    return dict(schema="S805_NINE_OF_NINE_INDEPENDENT_EVIDENCE_GATE_V1",
                passed=True,official_reference="S708",promotion=False,
                local_six=6,nvidia_three=3,models=ROLES,
                source_sha=expected_sha,evidence_sha256=r,
                note="Audit verdict is NOT metric improvement or physical CPU latency proof")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--six",required=True,type=pathlib.Path)
    p.add_argument("--reviews-dir",required=True,type=pathlib.Path)
    p.add_argument("--report",required=True,type=pathlib.Path)
    p.add_argument("--out",required=True,type=pathlib.Path)
    a=p.parse_args()
    docs={role:json.loads((a.reviews_dir/("review_"+role+".json")).read_text())
          for role in ROLES}
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    r=hashlib.sha256(a.report.read_bytes()).hexdigest()
    result=checked(json.loads(a.six.read_text()),docs,r,source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("S805_NINE_OF_NINE_PASS promotion=false")
if __name__=="__main__":main()
