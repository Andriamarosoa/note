"""S465 — read-only source onset-retention audit before any K corrections.

Uses immutable ORIGINAL S464 WAV energies and original S453 matched true
notes. Tests predeclared *diagnostic* post/pre RMS attack cuts and proposed
source-K safety veto: retain original S80 K>0 regardless of transient.
NO ORIGINAL EVENT / K / NOTE MODEL OR DECODER IS MODIFIED.
No thresholds are to be promoted from these retrospectively exposed XMLs.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
CUTS_DB=(-4.,0.,2.,4.,6.,8.,12.)
def need(x,desc):
    if not x:raise ValueError(desc)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(kept,truth,far,k,poly):
    # "far" is NOT ground truth false onset; call candidate drop, not precision.
    return dict(native_506_retained=int(kept.sum()),
      original_47_XML_anchors_retained=int(np.sum(kept&truth)),
      original_47_XML_anchors_lost=int(np.sum(~kept&truth)),
      original_21_poly_XML_anchors_retained=int(np.sum(kept&poly)),
      original_21_poly_XML_anchors_lost=int(np.sum(~kept&poly)),
      original_418_far_candidates_retained=int(np.sum(kept&far)),
      original_418_far_candidates_dropped=int(np.sum(~kept&far)),
      original_417_distant_predicted_K0_dropped=int(np.sum(~kept&far&(k==0))),
      matched_source_wrong_K0_protected=int(np.sum(kept&truth&(k==0))),
      matched_source_wrong_K0_lost=int(np.sum(~kept&truth&(k==0))))

def run(s464,s464_report,s453,s453_report,out):
    need(not out.exists(),"S465 original gate-audit SHA-frozen report exists")
    w=json.loads(s464_report.read_text())
    f=json.loads(s453_report.read_text())
    need(w["status"]=="S464_ORIGINAL_IDMT_V88_EXCESS_K0_EVENT_WAVEFORM_RMS_FORENSIC"
         and w["output_acoustic_ledger_SHA256"]==sha(s464)
         and w["original_native_events"]==506
         and w["distant_original_S80_K0"]==417
         and w["original_WAV_hashes_verified_against_S277"]
         and f["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and f["annotated_rows_SHA256"]==sha(s453)
         and f["matched_true_poly_events"]==21,
         "S465 source original S464 waveform or S453 XML labels changed")
    native=[json.loads(line) for line in s464.read_text().splitlines()]
    annotations=[json.loads(line) for line in s453.read_text().splitlines()]
    need(len(native)==506 and len(annotations)==47,"S465 original source cardinality")
    ix={r["row_id"]:i for i,r in enumerate(native)}
    need(len(ix)==506 and len({r["source_row_id"] for r in annotations})==47,
         "S465 duplicate source event identities")
    source_K=np.asarray([int(a["original_S80_predicted_K"]) for a in native],np.int8)
    original_matched=np.asarray([bool(a["greedy_S451_matched"]) for a in native],bool)
    far=np.asarray([not bool(a["near_XML_onset_80ms"]) for a in native],bool)
    strength=np.asarray([float(a["post_vs_pre_attack_rms_db"]) for a in native])
    poly=np.zeros(506,bool)
    valid=np.zeros(506,bool)
    for row in annotations:
        ii=ix[row["source_row_id"]]
        need(native[ii]["greedy_S451_matched"]
             and source_K[ii]==int(row["original_S80_K"]),
             "S465 original matched labels or original K drift")
        valid[ii]=bool(row["within_supported_notes"])
        if row["within_supported_notes"] and int(row["actual_K"])>=2:
            poly[ii]=True
    need(int(original_matched.sum())==47
         and int(valid.sum())==45 and int(poly.sum())==21
         and int(far.sum())==418
         and int((far&(source_K==0)).sum())==417
         and int((original_matched&(source_K==0)).sum())>=7
         and np.isfinite(strength).all(),
         "S465 must reproduce full source K and aligned XML positives")
    policies=[]
    for t in CUTS_DB:
        transient=strength>=t
        # Source-K lock: preserves every event from which original S80
        # actually emits at least one note. Potential risk remains for
        # the S80 erroneously K0-assigned TRUE notes; measure that risk.
        withK=(source_K>0)|transient
        policies.append(dict(
            threshold_db=float(t),diagnostic_only_not_selected_for_deployment=True,
            pure_waveform_attack_rise=metrics(transient,original_matched,
                                              far,source_K,poly),
            with_unchanged_original_nonzero_K_safety=metrics(
                withK,original_matched,far,source_K,poly),
            source_predicted_K0_kept=int(np.sum(withK&(source_K==0))),
            source_predicted_K0_dropped=int(np.sum(~withK&(source_K==0)))))
    out.mkdir(parents=True)
    resulting=dict(status="S465_S464_IDMT_ATTACK_VS_SUSTAIN_RETENTION_ABLATION_NO_PROMOTION",
        native_original_events=506,
        original_XML_note_attack_events=47,
        original_XML_supported_true_poly=21,
        original_external_far_candidates=418,
        original_far_predicted_K0=417,
        XML_previously_exposed_originally=True,
        false_positives_among_unmatched_not_proven=True,
        source_S80_K_preserved=True,
        source_S80_K0_was_never_a_ground_truth_label=True,
        no_filter_or_new_S80_predictions_executed=True,
        diagnostic_thresholds_declared_before_performance_report=True,
        threshold_search_NOT_promotion_eligible=True,
        no_train_refit_or_external_threshold_selection=True,
        no_promotion=True,
        per_fixed_RMS_threshold=policies,
        SHA256=dict(S464_WAV_acoustic=sha(s464),
                    S453_47_original_XML_matched=sha(s453)))
    (out/"report.json").write_text(json.dumps(resulting,indent=2)+"\n")
    for row in policies:
        print("S465 COUNTERFACTUAL",json.dumps(dict(
            threshold_db=row["threshold_db"],
            pure=row["pure_waveform_attack_rise"],
            sourceKprotected=row["with_unchanged_original_nonzero_K_safety"])),flush=True)

def selftest():
    native=np.array([True,False,True,False])
    far=np.array([False,True,False,True])
    k=np.array([2,0,0,0])
    poly=np.array([True,False,True,False])
    m=metrics(np.array([True,False,False,False]),native,far,k,poly)
    need(m["original_47_XML_anchors_lost"]==1 and
         m["original_21_poly_XML_anchors_lost"]==1 and
         m["original_418_far_candidates_dropped"]==2,
         "S465 counterfactual recall cannot trade for candidate reduction silently")
    print("S465 SELFTEST PASS K0 true note guard and no invented true-K0 on unmatched")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for arg in ("s464","s464-report","s453","s453-report","out"):
        p.add_argument("--"+arg,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for key in ("s464","s464_report","s453","s453_report","out"):
            need(getattr(a,key)is not None,"S465 missing --"+key)
        run(a.s464,a.s464_report,a.s453,a.s453_report,a.out)
