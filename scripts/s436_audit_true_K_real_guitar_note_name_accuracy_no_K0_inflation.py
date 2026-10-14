"""S436: distribution-aware real guitar post-audit, avoiding K0 inflation.

Operates ONLY on the immutable already-scored S435 JSON report. No WAV,
labels, training, or teacher content is read. Distinguishes complete chord
rate conditional on true K2+ from overall event accuracy (K0 dominates).
Tiny diagnostic sample: do NOT generalize to independent real guitar data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def summarize(file,out):
    data=json.loads(file.read_text())
    assert data["status"]=="S435_FROZEN_REAL_GUITARSET_MIC_WAV_NOTE_IDENTITY_DIAGNOSTIC"
    assert data["sample_source_events"]==30
    rows=data["by_true_K"]
    count=lambda cls:sum(v.get("events",0) for k,v in rows.items()
                         if cls(int(k.removeprefix("true_K"))))
    correct=lambda cls:sum(v.get("chords_correct",0) for k,v in rows.items()
                         if cls(int(k.removeprefix("true_K"))))
    all_n=count(lambda k: True); all_ok=correct(lambda k: True)
    assert all_n==30 and all_ok==data["sample_correct_chords"]
    strata={}
    for label,fn in [
      ("K0",lambda k:k==0),("K1",lambda k:k==1),
      ("poly_K2_to_K6",lambda k:k>=2),
      ("nonzero_K1_to_K6",lambda k:k>=1)]:
        n=count(fn);ok=correct(fn)
        strata[label]=dict(n=n,exact=ok,accuracy=(ok/n if n else None))
    report=dict(status="S436_REAL_GUITAR_POSTHOC_PER_TRUE_K_STRATUM_AUDIT",
      no_model_fit_or_parameter_selection=True,
      inputs= "Frozen S435 report only",
      total_events=30,complete_note_sets_correct=all_ok,
      complete_note_set_acc=all_ok/all_n,
      true_K_subgroups=strata,
      S80_original_K_unchanged=True,
      genuine_real_guitar_MIC_recording=True,
      guitarset_historically_exposed_to_upstream_S80=True,
      original_microphone_vs_pickup_modality_difference=True,
      cannot_estimate_unseen_generalization=True,
      source_report_status=data["status"],
      benchmark_promoted=False)
    assert strata["K0"]["n"]+strata["K1"]["n"]+strata["poly_K2_to_K6"]["n"]==30
    assert strata["K0"]["exact"]+strata["K1"]["exact"]+strata["poly_K2_to_K6"]["exact"]==all_ok
    assert strata["poly_K2_to_K6"]["n"]>0, "real polyphonic audio score absent"
    assert not out.exists(),"Never overwrite a previous audit"
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S436 BY TRUE K",json.dumps(strata),flush=True)
    print("S436 PASS real guitar K0 inflated event-level accuracy separated from polyphony",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--s435-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    summarize(a.s435_report,a.out)
