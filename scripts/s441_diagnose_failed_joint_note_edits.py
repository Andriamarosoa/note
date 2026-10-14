"""S441: diagnostic only; measure per-note changes of rejected S439 policies."""
import argparse,collections,json
from pathlib import Path

def overlap(a,b):
    return sum((collections.Counter(a)&collections.Counter(b)).values())

def run(source,changes,out):
    prior=json.loads(source.read_text())
    assert prior["status"]=="S440_PAIRED_REAL_MIC_12_GUITAR_JOINT_PITCH_UNPROMOTED_AUDIT"
    rows=[json.loads(x) for x in changes.read_text().splitlines()]
    result={}
    for v in ("conservative","exploratory"):
        subset=[r for r in rows if r["variant"]==v]
        counts=collections.Counter()
        byK=collections.defaultdict(collections.Counter)
        for r in subset:
            assert collections.Counter(r["baseline"])!=collections.Counter(r["new"])
            assert len(r["baseline"])==len(r["new"])==r["original_predicted_K"]
            delta=overlap(r["new"],r["ground_truth"])-overlap(r["baseline"],r["ground_truth"])
            key="more_true_notes" if delta>0 else "fewer_true_notes" if delta<0 else "same_true_notes"
            counts[key]+=1
            counts["note_tp_net"]+=delta
            counts["changed_sets"]+=1
            byK[str(r["true_K"])][key]+=1
            byK[str(r["true_K"])]["note_tp_net"]+=delta
        parent=prior["score"][v]["paired_vs_frozen_S437"]
        assert counts["changed_sets"]==sum(z["changes"] for z in parent["by_true_K"].values())
        result[v]={"changes":counts["changed_sets"],"note_true_positive_net":counts["note_tp_net"],
                   "more_true_notes":counts["more_true_notes"],"fewer_true_notes":counts["fewer_true_notes"],
                   "same_true_notes":counts["same_true_notes"],"chord_corrections":parent["corrections"],
                   "chord_regressions":parent["regressions"],
                   "by_true_K":{k:dict(z) for k,z in byK.items()}}
    out.mkdir(parents=True,exist_ok=False)
    (out/"report.json").write_text(json.dumps({"status":"S441_FORENSIC_NO_NEW_MODEL",
        "no_training_or_promotion":True,"original_S80_unchanged":True,
        "variants":result},indent=2)+"\n")
    print("S441 DIAGNOSTIC",json.dumps(result))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("source","changes","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:
        assert overlap(["E3","G3"],["E3","B3"])==1
        assert overlap(["G3","E3"],["E3","G3"])==2
        print("S441 SELFTEST PASS")
    else:run(a.source,a.changes,a.out)
