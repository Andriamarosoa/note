"""S804 final 9/9 gate: six real local audits + three independent NVIDIA NIM."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,re
from scripts.s804_nvidia_deep_review import ROLES,SCHEMA
SIX_SCHEMA="NOTE_SIX_ROLE_CONSENSUS_V1"
def checked(six,reviews,bundle):
    if six.get("schema")!=SIX_SCHEMA or six.get("passed") is not True or (
        six.get("audits_received")!=6 or six.get("issues") or
        six.get("official_promotion_eligible") is not False):
        raise AssertionError("six local audits failed or incomplete")
    if set(six.get("auditors",{}))!={
        "data","acoustics","architecture","training","metrics","live"}:
        raise AssertionError("six distinct local auditors missing")
    if len(reviews)!=3 or set(reviews)!=set(ROLES):
        raise AssertionError("three independent model reviews required")
    for role,expected in ROLES.items():
        doc=reviews[role]
        if doc.get("schema")!=SCHEMA or doc.get("role")!=role or (
            doc.get("model")!=expected) or doc.get("series")!="S804" or (
            doc.get("provider")!="NVIDIA NIM") or doc.get("review_received") is not True:
            raise AssertionError("wrong/stale/fabricated reviewer "+role)
        if doc.get("evidence_digest")!=bundle["evidence_digest"] or (
            doc.get("source_sha")!=bundle["commit"]):
            raise AssertionError("cross-run or cross-source review "+role)
        text=doc.get("review_text")
        if not isinstance(text,str) or len(text.strip())<160:
            raise AssertionError("empty reviewer "+role)
        if hashlib.sha256(text.encode()).hexdigest()!=doc.get("response_sha256"):
            raise AssertionError("review transcript SHA drift "+role)
        if not re.fullmatch("[0-9a-f]{64}",doc.get("context_sha256","")):
            raise AssertionError("no outbound code/data context hash "+role)
        if doc.get("scientific_promotion") is not False:
            raise AssertionError("advisor illegally approved promotion")
    if len({doc["model"] for doc in reviews.values()})!=3:
        raise AssertionError("not three distinct models")
    return {"schema":"NOTE_S804_NINE_INDEPENDENT_AUDIT_CONSENSUS_V1",
            "passed":True,"local_roles":6,"nvidia_reviewers":3,
            "models":ROLES,"reference":"S708","promotion":False,
            "evidence_digest":bundle["evidence_digest"],
            "source_sha":bundle["commit"],
            "scientific_scores_improved":False,
            "warning":"Audit completion is not certification of model accuracy/CPU/live hardware"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--six",required=True,type=pathlib.Path)
    p.add_argument("--reviews-dir",required=True,type=pathlib.Path)
    p.add_argument("--bundle",required=True,type=pathlib.Path)
    p.add_argument("--out",required=True,type=pathlib.Path)
    a=p.parse_args()
    bundle=json.loads(a.bundle.read_text())
    reviews={}
    for role in ROLES:
        file=a.reviews_dir/("review_"+role+".json")
        if not file.exists():raise FileNotFoundError(str(file))
        reviews[role]=json.loads(file.read_text())
    final=checked(json.loads(a.six.read_text()),reviews,bundle)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(final,indent=2)+"\n")
    print("NOTE_NINE_OF_NINE_PASS S804",json.dumps(final["models"]),
          "promotion=false",flush=True)
if __name__=="__main__":main()
