"""S804 local six-role deep evidence auditors; fail closed and data-backed."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROLES=("data","acoustics","architecture","training","metrics","live")
SCORES=("onset_f1","exact_k","active_exact_k","offset_accuracy",
        "full_note_f1","frame_pitch_f1")
def audit(role,r):
    def need(test,why):
        if not test:raise AssertionError(why)
    need(r.get("schema")=="S804_FIT80_CODE_AND_DATA_DEEP_AUDIT_BUNDLE_V1","bundle missing")
    src=r["code_audit"];d=r["dataset_audit"]
    need(src["python_file_count"]>=100 and src["all_python_files_AST_parsed"] and
         not src["syntax_errors"],"full Python AST coverage invalid")
    need(d["count_tracks"]==80 and len(d["per_track_audit"])==80 and
         d["count_frames"]==306536,"real OOF evidence missing")
    need(len(d["compositions"])==5 and d["SHA256_snapshots_verified"],
         "GuitarSet signed partitions absent")
    need(d["source_signed_counts"]["k_correct"]==189124,"source drift")
    need(not d["CAL_HELD_TEST_opened"] and r["reference"]=="S708" and
         not r["production_LIVE_modified"] and not r["science_promoted"],
         "reserved split or production regression")
    if role=="data":
        need(len({z["track_index"] for z in d["per_track_audit"]})==80,
             "duplicated track index")
        need(sum(z["tracks"] for z in d["compositions"].values())==80,
             "composition missing records")
        return {"signed80":True,"source_sha":r["commit"],
                "files_AST":src["python_file_count"]}
    if role=="acoustics":
        need(set(d["retrospective_s802_collisions"])==
            {"consensus","poly_gap","adaptive"},"source onset matching missing")
        need(d["retrospective_s802_collisions"]["consensus"]["total"]==136,
             "signed TP/FP evidence changed")
        need(len(d["pitch_histogram_native_onsets"])>20,"pitch diversity missing")
        return {"pitch_and_collision_breakdown_present":True}
    if role=="architecture":
        need(all(z["excerpt_lines"]>10 for z in src["deep_code_excerpts"]),
             "insufficient code evidence")
        need(len(src["deep_code_excerpts"])>=10,"critical code not inspected")
        return {"all_python_ast_scanned":True,
                "hot_path_excerpts":len(src["deep_code_excerpts"])}
    if role=="training":
        need(not d["CAL_HELD_TEST_opened"] and len(d["compositions"])==5,
             "potential heldout contamination")
        return {"OOF_only":True,"no_reserved_split":True}
    if role=="metrics":
        need(set(d["native_fit_six_scores"])==set(SCORES),
             "six source metrics absent")
        for policy,row in d["policy_results"].items():
            need(set(row["real_MIDI_six_metrics"])==set(SCORES) and
                 len(row["composition_deltas"])==5,"partial metric audit")
            safe=all(float(v)>=-1e-12 for v in row["delta_vs_FIT_native"].values()) and all(
                all(float(v)>=-1e-12 for v in comp.values())
                for comp in row["composition_deltas"].values())
            need(safe==row["no_regression"],"non-regression label mismatches scores")
        return {"six_ratios_and_five_compositions_checked":True}
    if role=="live":
        need(not r["production_LIVE_modified"] and
             len(src["deep_code_excerpts"])>=10,"live source or code absent")
        return {"production_untouched":True,
                "physical_latency_validation":"not measured in S804"}
    raise ValueError("invalid audit role")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=ROLES,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    out=dict(schema="NOTE_SIX_ROLE_AUDIT_V1",role=a.role,passed=False,
             experimental_promotion_allowed=False,findings={},errors=[])
    try:
        out["findings"]=audit(a.role,json.loads(a.bundle.read_text()))
        out["passed"]=True
    except Exception as e:
        out["errors"]=[type(e).__name__+": "+str(e)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("S804_LOCAL_AUDIT",a.role,"PASS" if out["passed"] else "FAIL",flush=True)
    if not out["passed"]:sys.exit(1)
if __name__=="__main__":main()
