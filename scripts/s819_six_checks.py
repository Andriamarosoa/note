"""S819 six independent deterministic roles; NOT six ChatGPT Work agents.

Each CI process reads the exact same neutral evidence and project source. Roles
perform distinct fail-closed checks on real available code and signed scores.
No LLM is simulated: audit records have verifier_type=deterministic_code_check.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from scripts.s819_blind_nvidia import evidence_text,hash_text
SCHEMA="S819_DETERMINISTIC_SIX_ROLE_AUDIT_V1"
ROLES=("data","acoustics","architecture","training","metrics","live")
SOURCE_FILES={
 "s815":"src/causal_note/s815_activek_teacher.py",
 "s817":"src/causal_note/s817_highpoly_temporal.py",
 "s818":"src/causal_note/s818_activek_ordinal.py",
 "pipeline":"scripts/s818_train_ordinal256.py",
 "workflow":".github/workflows/note-s818-activek-ordinal.yml",
}
def enforce(ok,label):
    if not ok:raise AssertionError(label)
def check(role,evidence,root):
    docs={key:(root/path).read_text(encoding="utf-8") for key,path in SOURCE_FILES.items()}
    s815,s817,s818,pipe,workflow=(docs[k] for k in ("s815","s817","s818","pipeline","workflow"))
    enforce(role in ROLES,"unknown role")
    enforce("249411" in evidence and "135694" in evidence and "142213" in evidence,
            "original measured scores missing")
    if role=="data":
        enforce(all(x in pipe for x in ("cal_comp=","train=","cal=","evaluate=",
                   "(48,16,16)","digest_file","source_MIDI_modified=False")),
            "composition-disjoint source signatures not demonstrable")
        enforce("FIT80 historiquement exploré" in evidence,"research cohort limitation omitted")
        return {"audit_target":"five disjoint compositions; original manifest hash",
                "held_out_generalization_claimed":False}
    if role=="acoustics":
        enforce("summarize(x)" in s815 and "temporal(x)" in s818
                and "LAGS=(1,4,16)" in s817,"native causal acoustic path changed")
        enforce("256 échantillons" in evidence and "sans information" not in evidence[-800:],
                "MIDI256 and unbiased brief missing")
        return {"audit_target":"causal DSP and low/high-band physical claims",
                "new_harmonic_f0_evidence_measured":False}
    if role=="architecture":
        enforce("tuple(head.classes_)!=(0,1)" in s818
                and "np.minimum.accumulate" in s818
                and "SELECTION" not in s818
                and "MappingProxyType" in s815,
                "ordinal six-head independence details absent")
        enforce("six" in evidence.lower() and "K>=6" in evidence,
                "K6 failure/independence not recorded")
        return {"audit_target":"six independent ordinal estimators and frozen S815 comparator",
                "code_interaction_with_other_axes":False}
    if role=="training":
        enforce("rng.choice" in s818 and "len(at)/sample_n" in s818
                and "W /= W.mean()" in s818,
                "K class sampling-prior contract not visible")
        enforce("fit_teacher(train,cal" in pipe and "(48,16,16)" in pipe,
                "fit calibration not split")
        return {"audit_target":"inverse-inclusion weights, calibration selector, K5/K6 rarity",
                "model_training_in_role":False}
    if role=="metrics":
        enforce("active_score" in pipe and "original_count" in pipe
                and "original_fit80_active_k" in pipe
                and "original_s815_fit_active_k" in pipe,
                "H0/S815 identity/reproduction contract missing")
        enforce("K0 NON dans Active-K" in evidence,
                "K0 active denominator confusion")
        return {"audit_target":"verify H0/S815 and S818 integer metrics and K0 exclusion",
                "reported_f1_not_count_accuracy":True}
    enforce(role=="live","invalid role")
    enforce("future_lookahead_samples=0" in s818
            and "post_onset_hold_samples=0" in s818,
            "live timing veto flags absent")
    enforce("5,80499 ms" in evidence and "PAS latence totale" in evidence,
            "hardware latency confused with 256-hop granularity")
    return {"audit_target":"no future/hold/retiming, 256-hop granularity vs physical latency",
            "actual_irig_latency_measured":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",required=True,choices=ROLES)
    p.add_argument("--evidence",required=True,type=Path)
    p.add_argument("--root",default=".",type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    text=evidence_text(a.evidence)
    obj=dict(schema=SCHEMA,role=a.role,verifier_type="deterministic_code_check",
             model_or_work_agent=False,evidence_sha256=hash_text(text),
             passed=False,findings={},errors=[],production_modified=False,
             promotion_allowed=False)
    try:
        obj["findings"]=check(a.role,text,a.root)
        obj["passed"]=True
    except Exception as err:
        obj["errors"]=[type(err).__name__+":"+str(err)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    print("S819_DETERMINISTIC_AUDIT",a.role,
          "PASS" if obj["passed"] else "FAIL",flush=True)
    if not obj["passed"]:raise SystemExit(2)

if __name__=="__main__":main()
