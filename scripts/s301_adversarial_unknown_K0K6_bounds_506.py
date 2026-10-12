"""S301: adversarial worst-case full-506 delta bounds from S299 SHA-sealed labels.

Does NOT assign unmatched XML V88 events to K0 or to any other K;
instead enumerate all possible truth values K0..K6 for each changed
event whose XML truth is unknown. If the worst-case relative gain is
positive, this is a descriptive finite-population robustness statement,
NOT a verified global Exact-K score or a song-disjoint promotion test.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

FIELDS={"row_id","member","start_samples","XML_true_K_or_minus1",
        "matched_offset_ms","original_official_freeze_K","original_S80_K",
        "S274_twofold_full_S80_K"}
def need(c,msg):
    if not c:raise ValueError(msg)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def calc(true,official,proposed):
    need(len(true)==len(official)==len(proposed) and len(true)==506,
         "exact 506 original frozen native rows required")
    need(np.isin(true,range(-1,7)).all() and
         np.isin(official,range(7)).all() and np.isin(proposed,range(7)).all(),
         "unknown labels must be exactly -1; real labels K0..6")
    known=true>=0
    changed=official!=proposed
    changed_known=known&changed
    unknown=~known
    unknown_changed=changed&unknown
    oldok=(official[known]==true[known])
    newok=(proposed[known]==true[known])
    known_net=int(newok.sum()-oldok.sum())
    known_fix=int(np.count_nonzero(~oldok&newok))
    known_reg=int(np.count_nonzero(oldok&~newok))
    need(known_net==known_fix-known_reg,"paired known truth exact-K ledger inconsistent")
    # Each changed unknown row can contribute -1, 0 or +1 in exact count
    # under unknown K in [0,6]; each unchanged unknown contributes 0.
    minnet=known_net-int(unknown_changed.sum())
    maxnet=known_net+int(unknown_changed.sum())
    # Exhaustively check every possible K0..6 for each changed unknown
    # (2 unknown in current S299) before using bounds.
    from itertools import product
    ix=np.flatnonzero(unknown_changed)
    need(len(ix)<=8,"avoid combinatorial explosion; use symbolic bounds")
    observed=[]
    for ys in product(range(7),repeat=len(ix)):
        old=np.asarray([official[i] for i in ix])
        new=np.asarray([proposed[i] for i in ix])
        yy=np.asarray(ys)
        observed.append(known_net+int(np.count_nonzero(new==yy)-np.count_nonzero(old==yy)))
    need(min(observed)==minnet and max(observed)==maxnet,
         "independent exhaustive symbolic bounds do not agree")
    return dict(population=506,truth_known=int(known.sum()),
      truth_unknown=int(unknown.sum()),
      known_changed=int(changed_known.sum()),
      unknown_changed=int(unknown_changed.sum()),
      known_fixes=known_fix,known_regressions=known_reg,
      known_both_wrong_changed=int(np.count_nonzero(
         changed_known& (official!=true)&(proposed!=true))),
      paired_net_known=known_net,
      worst_case_full_population_net=minnet,
      best_case_full_population_net=maxnet,
      worst_case_full_population_percentage_points=100*minnet/506,
      best_case_full_population_percentage_points=100*maxnet/506,
      adversarial_unknown_truth_assignments_enumerated=7**len(ix),
      definite_full_population_sign_positive=bool(minnet>0),
      true_full_population_ExactK_percentage_measured=False)
def selftest():
    y=np.full(506,-1,np.int16)
    old=np.zeros(506,np.int8);new=old.copy()
    # 4 fixes, 1 loss, 4 changed but both wrong in known set
    y[:4]=2;new[:4]=2
    y[4]=0;new[4]=1
    y[5:9]=3;new[5:9]=2
    y[9:47]=1;old[9:47]=1;new[9:47]=1
    new[47:49]=2
    z=calc(y,old,new)
    need(z["known_fixes"]==4 and z["known_regressions"]==1 and
         z["known_both_wrong_changed"]==4 and
         z["unknown_changed"]==2 and
         z["worst_case_full_population_net"]==1 and
         z["best_case_full_population_net"]==5,
         "S301 symbolic worst/best unknown K0-K6 test")
    print("S301 test passed: unknown K0..K6 explicitly adversarial")
def run(s299,pred,out):
    need(not out.exists(),"S301 immutable audit: no source overwrite")
    meta=json.loads(s299.read_text())
    need(meta["status"]=="S299_FROZEN_ORIGINAL_VS_STRICT_S80_IDMT_DATASET3_XML_MATCHED_NOTE_ONLY"
         and meta["event_population"]==506 and
         meta["matched_true_K1to6_events"]==47 and
         meta["unmatched_or_unknown_label_events"]==459 and
         not meta["full_Global_ExactK_K0to6_measured"] and
         not meta["S80_promoted"],"S299 postfrozen provenance changed")
    with np.load(pred,allow_pickle=False) as z:
        need(set(z.files)==FIELDS,"S299 checked truth/prediction archive schema drift")
        a={k:z[k].copy() for k in z.files}
    y=a["XML_true_K_or_minus1"]
    official=a["original_official_freeze_K"]
    original=a["original_S80_K"]
    strict=a["S274_twofold_full_S80_K"]
    need(np.array_equal(original,strict),"previous S298 candidates must be identical")
    result=calc(y,official,original)
    need(result["truth_known"]==47 and result["truth_unknown"]==459
         and result["known_fixes"]==4 and result["known_regressions"]==1
         and result["unknown_changed"]==2 and
         result["known_both_wrong_changed"]==4,
         "S299 original 47 matched note truth ledger mismatch")
    report=dict(
        status="S301_ADVERSARIAL_UNKNOWN_K_506_DELTA_BOUNDS_NO_SCORE",
        original_S299_report_sha256=digest(s299),
        original_S299_XML_pair_archive_sha256=digest(pred),
        fixed_comparison="S80_RECONSTRUCTED_vs_freeze_local_combo",
        audited_result=result,
        truth_xml_opened_during_this_run=False,
        new_audio_models_evaluated_during_this_run=False,
        exact_K0_to_K6_true_ground_truth_full_population=False,
        four_composition_groups_unverified_train_disjoint=True,
        statistically_significant_generalization_claim=False,
        official_S80_promotion_allowed=False,
        caution="Bound only on 506 preselected original V88 events conditional on trusting 47 matched XML labels, not an Exact-K rate, nor generalization or historical S80 identity.")
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for name in ("s299","pred","out"):ap.add_argument("--"+name,type=Path)
    args=ap.parse_args()
    if args.self_test:selftest()
    else:
        need(all(getattr(args,k) is not None for k in ("s299","pred","out")),"sources required")
        selftest()
        run(args.s299,args.pred,args.out)
