"""S804 no-network tests: three distinct advisers, 9/9 fail-closed, integrity."""
import ast,hashlib,json,unittest
from pathlib import Path
from scripts.s804_nvidia_deep_review import ROLES,SCHEMA,context,payload,canonical_digest,quality_check
from scripts.s804_nine_consensus import checked
class DeepAudits(unittest.TestCase):
    def bundle(self):
        b={"schema":"S804_FIT80_CODE_AND_DATA_DEEP_AUDIT_BUNDLE_V1",
           "commit":"a"*40,"reference":"S708","repo":"Andriamarosoa/note",
           "review_limitations":{"audio":False},
           "code_audit":{"all_python_files_AST_parsed":True,"python_file_count":200,
              "deep_code_excerpts":[{"path":"src/module.py","sha256":"b"*64,
                "total_lines":2,"excerpt":"1: def validate():\n2:     return True\n"}]},
           "dataset_audit":{"count_tracks":80,"count_frames":306536,
               "policy_results":{},"per_track_audit":[]}}
        b["evidence_digest"]=hashlib.sha256(json.dumps(
            b,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        return b
    def test_legacy_v273_syntax_regression_is_fixed(self):
        path=Path("scripts/audit_v273_stage2_nested_calibration.py")
        ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

    def test_shallow_external_review_fails(self):
        with self.assertRaises(ValueError):
            quality_check("Lorem ipsum",["scripts/a.py","scripts/b.py"])
        with self.assertRaises(ValueError):
            quality_check("A"*1300+" scripts/a.py",["scripts/a.py","scripts/b.py"])
        txt="scripts/a.py and scripts/b.py: "+("Source-backed audit. "*65)
        self.assertEqual(len(quality_check(txt,["scripts/a.py","scripts/b.py"])),2)
    def test_three_distinct_models_and_code_in_context(self):
        b=self.bundle();canonical_digest(b)
        self.assertEqual(len(set(ROLES.values())),3)
        for role in ROLES:
            c=context(role,b)
            self.assertIn("src/module.py",c)
            self.assertIn("count_tracks",c)
            self.assertEqual(payload(role,c)["model"],ROLES[role])
        bad=dict(b);bad["commit"]="c"*40
        with self.assertRaises(AssertionError):canonical_digest(bad)
    def test_nine_gate_requires_all_models_and_evidence(self):
        b=self.bundle()
        six={"schema":"NOTE_SIX_ROLE_CONSENSUS_V1","passed":True,
              "audits_received":6,"issues":[],"official_promotion_eligible":False,
              "auditors":{k:{"passed":True} for k in
                 ("data","acoustics","architecture","training","metrics","live")}}
        reviews={}
        for role,model in ROLES.items():
            txt="A"*200
            reviews[role]={"schema":SCHEMA,"role":role,"series":"S804",
                "model":model,"provider":"NVIDIA NIM",
                "review_received":True,"evidence_digest":b["evidence_digest"],
                "source_sha":b["commit"],"response_sha256":
                    hashlib.sha256(txt.encode()).hexdigest(),
                "context_sha256":"f"*64,"review_text":txt,
                "scientific_promotion":False}
        self.assertTrue(checked(six,reviews,b)["passed"])
        for role in ROLES:
            q=dict(reviews);q.pop(role)
            with self.assertRaises(AssertionError):checked(six,q,b)
        wrong=dict(reviews)
        wrong["data"]=dict(wrong["data"],model="z-ai/glm-5.3-flash")
        with self.assertRaises(AssertionError):checked(six,wrong,b)
        with self.assertRaises(AssertionError):
            checked(dict(six,passed=False),reviews,b)
if __name__=="__main__":unittest.main()
