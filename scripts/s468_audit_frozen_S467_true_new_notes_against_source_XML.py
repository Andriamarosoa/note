"""S468 score S467 SHA-frozen GuitarSet-only birth detector on annotated IDMT.

Identity audit on 47 known XML-matched IDMT attacks (45 MIDI-supported, 21
polyphonic) and 418 distant acoustic candidates (UNLABELED not proven false).
All decisions/calibration were frozen before labels are inspected here.
Diagnostic attack-score separability is NOT a new blind benchmark.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
def need(c,message):
    if not c:raise ValueError(message)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frac(v):
    a=np.asarray(v,float)
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
        q10=float(np.quantile(a,.1)) if len(a) else None,
        q90=float(np.quantile(a,.9)) if len(a) else None)

def run(frozen,report,matched,matched_report,near,near_report,out):
    need(not out.exists(),"S468 original postfreeze truth audit already exists")
    h=json.loads(report.read_text())
    r=json.loads(matched_report.read_text())
    q=json.loads(near_report.read_text())
    need(h["status"]=="S467_FROZEN_SOURCE_K_BLIND_BIRTH_GATE_IDMT_NO_XML"
         and h["source_SHA"]["frozen_predictions"]==sha(frozen)
         and h["IDMT_XML_never_read_for_inference_or_thresholds"]
         and h["original_true_S80_K_changed"]==0
         and r["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and r["annotated_rows_SHA256"]==sha(matched)
         and q["status"]=="S462_IDMT_EXTERNAL_ORIGINAL_ONSET_XML_MATCHING_FORENSIC"
         and q["all_IDMT_sources_pinned_SHA256"]["nearest_source_ledger"]==sha(near),
         "S468 genuine pretrained no-label event model or original XML ledger changed")
    with np.load(frozen,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        k=z["original_S80_K"].astype(np.int8)
        p=z["individual_fold_birth_probabilities"].astype(np.float32)
        cuts=z["original_trainonly_thresholds"].astype(float)
        keep=z["gated_retain_suggestion"].astype(bool)
        members=z["member"].astype(str)
    need(len(ids)==506 and len(set(ids))==506
         and p.shape==(4,506)
         and len(cuts)==4
         and np.array_equal(keep[k>0],np.ones(int(np.sum(k>0)),bool)),
         "S468 source original 506 S80 events or four fold model probabilities mutated")
    rows={v:i for i,v in enumerate(ids)}
    truth=[json.loads(x) for x in matched.read_text().splitlines()]
    nearrows=[json.loads(x) for x in near.read_text().splitlines()]
    need(len(truth)==47 and len(nearrows)==506 and
         set(rows)=={r["row_id"] for r in nearrows},
         "S468 original 47 XML attacks or 506 candidate nearness unavailable")
    first=np.zeros(506,bool)
    poly=np.zeros(506,bool)
    inrange=np.zeros(506,bool)
    perclass=collections.defaultdict(list)
    for x in truth:
        ix=rows[x["source_row_id"]]
        need(int(k[ix])==x["original_S80_K"],
             "S468 frozen original S80 K differs from matched XML ledger")
        first[ix]=True
        if x["within_supported_notes"]:
            inrange[ix]=True
            perclass[str(x["actual_K"])].append(int(ix))
            if int(x["actual_K"])>=2:
                poly[ix]=True
    far=np.zeros(506,bool)
    original_match=np.zeros(506,bool)
    for x in nearrows:
        i=rows[x["row_id"]]
        far[i]=(x["nearest_XML_group_ms"] is None or
                x["nearest_XML_group_ms"]>80)
        original_match[i]=bool(x["original_S451_matched"])
    need(np.array_equal(first,original_match)
         and int(np.sum(first))==47
         and int(np.sum(inrange))==45
         and int(np.sum(poly))==21
         and int(np.sum(far))==418
         and int(np.sum(far&(k==0)))==417,
         "S468 original annotation population was artificially changed")
    soft=np.mean(p,axis=0)
    diagnostics=dict(
        real_XML_47_attack_probabilities=frac(soft[first]),
        real_XML_21_poly_attack_probabilities=frac(soft[poly]),
        far_418_unmatched_probabilities=frac(soft[far]),
        far_417_predictedK0_probabilities=frac(soft[far&(k==0)]),
        matched_sourceK0_probabilities=frac(soft[first&(k==0)]),
        probability_AUC_matched_attacks_vs_unmatched_far_PROXY_NOT_TRUE_AUC=(
          float(roc_auc_score(np.concatenate((np.ones(int(np.sum(first))),
              np.zeros(int(np.sum(far))))),
              np.concatenate((soft[first],soft[far]))))),
        individual_model_mean_attack_scores=[
          dict(fold=int(i),matched_mean=float(np.mean(p[i,first])),
               far_unmatched_mean=float(np.mean(p[i,far])))
          for i in range(4)],
        cuts_selected_from_GuitarSet_train_validation=[
          float(x) if np.isfinite(x) else None for x in cuts])
    metrics=dict(
        suggested_event_retained=int(np.sum(keep)),
        suggested_event_drop=int(np.sum(~keep)),
        original_47_matched_attacks_retained=int(np.sum(keep&first)),
        original_47_matched_attacks_lost=int(np.sum(~keep&first)),
        original_45_supported_retained=int(np.sum(keep&inrange)),
        original_21_true_poly_retained=int(np.sum(keep&poly)),
        original_21_true_poly_lost=int(np.sum(~keep&poly)),
        original_418_distant_candidates_dropped=int(np.sum(~keep&far)),
        original_417_far_predK0_dropped=int(np.sum(~keep&far&(k==0))),
        original_47_matched_wrongK0_lost=int(np.sum(~keep&first&(k==0))),
        original_47_matched_wrongK0_retained=int(np.sum(keep&first&(k==0))),
        per_true_K={key:dict(n=len(ix),retained=int(np.sum(keep[ix])),
                   lost=int(np.sum(~keep[ix])))
            for key,ix in sorted(perclass.items())},
        per_recording={
          name:dict(n_native=int(np.sum(members==name)),
             true_annotated_retained=int(np.sum(keep&first&(members==name))),
             true_annotated_lost=int(np.sum(~keep&first&(members==name))),
             far_unmatched_candidates_dropped=int(np.sum(~keep&far&(members==name))))
          for name in sorted(set(members))}
    )
    # Safe-to-experiment criterion is an explicit predeclared external guard.
    # This does not greenlight official promotion because sample is exposed.
    passes_retention=(
         metrics["original_47_matched_attacks_lost"]==0
         and metrics["original_21_true_poly_lost"]==0
         and metrics["original_47_matched_wrongK0_lost"]==0)
    changed=int(metrics["suggested_event_drop"])
    verdict=("NO_SOURCE_WAV_ONSET_LOSS_BUT_NOT_BLIND_NEEDS_NEW_WAV"
              if passes_retention and changed else
             "ABSTENTION_NO_GAIN" if not changed else
             "REJECT_DROPS_TRUE_ANNOTATED_ONSETS")
    result=dict(status="S468_EXTERNAL_SOURCE_K_PRESERVING_BIRTH_GATE_XML_PAIRED_DIAGNOSTIC",
        original_source_events=506,
        original_matched_XML_attack_events=47,
        original_matched_XML_pitch_supported=45,
        original_true_poly_XML_events=21,
        unmatched_far_candidates_not_groundtruth_false=True,
        original_S80_K_and_note_decoder_unchanged=True,
        no_models_retrained_or_external_thresholds_adjusted=True,
        previously_exposed_IDMT_XML_no_new_blind_validation=True,
        unknown_unmatched_K0_truth_not_assumed=True,
        frozen_model_thresholds_not_optimized_on_IDMT=True,
        experimental_retention_not_applied=True,
        not_an_official_new_ExactK_score=True,
        no_promotion=True,
        preserves_all_original_47_real_XML_attacks=passes_retention,
        verdict=verdict,
        paired_recall_vs_extra_candidate_counts=metrics,
        model_score_diagnostics=diagnostics,
        evidence_sha256=dict(S467_predictions=sha(frozen),
                             source_47_XML=sha(matched),
                             original_506_nearest=sha(near)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S468 GENUINE IDMT BIRTH VERDICT",json.dumps(dict(
         verdict=verdict,score=metrics,
         matched_mean=diagnostics["real_XML_47_attack_probabilities"]["median"],
         far_mean=diagnostics["far_417_predictedK0_probabilities"]["median"],
         proxy_auc=diagnostics["probability_AUC_matched_attacks_vs_unmatched_far_PROXY_NOT_TRUE_AUC"])),
         flush=True)

def selftest():
    a=frac([.1,.2,.3])
    need(a["n"]==3 and abs(a["median"]-.2)<1e-10,
         "S468 known frozen probability summary wrong")
    print("S468 SELFTEST PASS XML-only positive onset accounting")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for x in ("frozen","report","matched","matched-report",
              "near","near-report","out"):
        p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for x in ("frozen","report","matched","matched_report",
                  "near","near_report","out"):
            need(getattr(a,x)is not None,"S468 missing --"+x)
        run(a.frozen,a.report,a.matched,a.matched_report,
            a.near,a.near_report,a.out)
