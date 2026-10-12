"""S326: native source-only 964-row S80 versus S274-twofold S80 paired audit.

Frozen original S302 freeze_local_combo, S297 original-S291 and strict
S297 S274-S292 on identical original V88 event IDs.  Never reads
XML/MIDI/true_K; this is prediction-change census, NOT Exact-K accuracy.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
PRED={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
      'original_PR16_policy22_K','K1_by_fold_threshold_action',
      'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
NATIVE={'row_id','member','start_samples','end_samples','incumbent_K',
        'incumbent_probabilities'}
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,fields):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==fields,"no accuracy labels or unknown fields allowed")
        return {k:z[k].copy() for k in z.files}
def run(s302,old,new,out):
    need(not out.exists(),"S326 never overwrite frozen prospective predictions")
    b=load(s302,NATIVE)
    a=load(old,PRED)
    c=load(new,PRED)
    ids=b['row_id'];n=len(ids)
    need(n==964 and len(set(map(str,ids)))==n and
         np.array_equal(ids,a['row_id']) and np.array_equal(ids,c['row_id']) and
         np.array_equal(b['incumbent_K'],a['incumbent_freeze_K']) and
         np.array_equal(b['incumbent_K'],c['incumbent_freeze_K']),
         "S80 candidate must score EXACT same native V88 event IDs and official freeze")
    before=a['s80_K'].astype(np.int8)
    after=c['s80_K'].astype(np.int8)
    ref=b['incumbent_K'].astype(np.int8)
    need(np.isin(before,range(7)).all() and np.isin(after,range(7)).all()
         and np.isin(ref,range(7)).all(),"all K0..6 only")
    changed=np.flatnonzero(before!=after)
    raw=[]
    for item in changed:
        raw.append(dict(row_id=str(ids[item]),WAV_name_sha256=str(b['member'][item]),
           V88_native_start_samples=int(b['start_samples'][item]),
           original_S80_K=int(before[item]),S274_twofold_S80_K=int(after[item]),
           original_official_freeze_K=int(ref[item])))
    group=[]
    for name in sorted(set(b['member'].astype(str))):
        mask=b['member'].astype(str)==name
        group.append(dict(WAV_filename_sha256=name,
            events=int(mask.sum()),
            S80_original_changed_from_official=int(np.count_nonzero(before[mask]!=ref[mask])),
            S80_twofold_changed_from_official=int(np.count_nonzero(after[mask]!=ref[mask])),
            S80_pairwise_change_count=int(np.count_nonzero(before[mask]!=after[mask]))))
    report=dict(status="S326_TWO_COMPLETE_S80_S274_CANDIDATES_PRELABEL_IDENTICAL_964_V88_EVENTS",
        native_events=964,WAV_count=len(group),
        original_baseline_freeze_SHA256=sha(s302),
        original_S80_SHA256=sha(old),
        S274_twofold_S80_SHA256=sha(new),
        original_official_freeze_histogram_K0to6=np.bincount(ref,minlength=7).tolist(),
        original_S80_histogram_K0to6=np.bincount(before,minlength=7).tolist(),
        S274_twofold_S80_histogram_K0to6=np.bincount(after,minlength=7).tolist(),
        original_S80_changes_vs_freeze=int(np.count_nonzero(before!=ref)),
        strict_S274_S80_changes_vs_freeze=int(np.count_nonzero(after!=ref)),
        S274_vs_original_S80_changed_events=int(len(changed)),
        per_recording_change_census=group,
        original_vs_S274_S80_changed_native_ids=raw,
        all_results_on_exact_same_original_V88_native_events=True,
        original_models_all_frozen=True,
        truth_XML_or_MIDI_read=False,
        performance_ExactK_not_scored=True,
        composition_disjointness_NOT_verified=True,
        official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'prelabel_comparison.json').write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in (
        'original_vs_S274_S80_changed_native_ids','per_recording_change_census')},indent=2),
        flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("s302","old","new","out"):
        p.add_argument("--"+name,type=Path,required=True)
    q=p.parse_args();run(q.s302,q.old,q.new,q.out)
