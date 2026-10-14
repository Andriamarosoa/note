"""S300 read-only independent audit of postfrozen S299 external note-only ledger.

Print full per K and composition-level uncertainty, with strict warnings:
four composition groups and no labelled K0 prohibit official promotion.
Never changes model rules, predictions, weights or labels.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
def need(ok,msg):
    if not ok:raise ValueError(msg)
def audit(path,out):
    need(not out.exists(),"S300 immutable read-only report")
    source=Path(path).read_bytes()
    r=json.loads(source)
    need(r['status']=='S299_FROZEN_ORIGINAL_VS_STRICT_S80_IDMT_DATASET3_XML_MATCHED_NOTE_ONLY'
         and r['event_population']==506 and r['selected_WAVs']==5
         and r['matched_true_K1to6_events']==47
         and r['matched_true_poly_K2to6_events']==23
         and r['unmatched_or_unknown_label_events']==459
         and r['dataset3_unique_named_composition_families']==4
         and r['full_Global_ExactK_K0to6_measured'] is False
         and not r['S80_promoted'],
         "immutable original S299 postfrozen report changed or claims independent full metric")
    pair=r['original_paired_comparisons']
    expect=['S80_original_vs_official_freeze',
            'S80_S274_twofold_vs_official_freeze',
            'S80_S274_twofold_vs_original_S80']
    need(set(pair)==set(expect),"wrong compared original S80 and stricter candidate identities")
    for label in expect:
        data=pair[label]
        print("### "+label,flush=True)
        print(json.dumps({
            "K1to6":data['matched_K1to6'],
            "K2to6":data['matched_true_poly_K2to6'],
            "per_K":data['matched_per_K'],
            "cluster_bootstrap_K1to6":data['bootstrap_K1to6'],
            "cluster_bootstrap_K2to6":data['bootstrap_poly_K2to6'],
            "change_total":data['changed_events_total'],
            "change_XML_matched":data['changed_events_XML_matched'],
            "change_XML_unmatched":data['changed_events_unmatched'],
        },indent=2),flush=True)
    original=pair[expect[0]]
    stricter=pair[expect[1]]
    between=pair[expect[2]]
    need(original['matched_K1to6']==stricter['matched_K1to6']
         and original['matched_true_poly_K2to6']==stricter['matched_true_poly_K2to6']
         and between['changed_events_total']==0,
         "S274 strict and original S80 unexpectedly diverged on fully frozen new audio")
    gate=r['promotion_gates']
    need(not gate['certified_composition_disjoint_from_all_training']
         and not gate['independent_exactk_global_K0to6_available']
         and not gate['complete_negative_K0_annotations']
         and not gate['official_S80_promotion_allowed'],
         "unsupported positive promotion claim")
    report=dict(status="S300_READ_ONLY_EXTERNAL_S299_CAUSAL_CLASS_AUDIT",
        original_S299_report_SHA256=hashlib.sha256(source).hexdigest(),
        matched_annotated_K1to6=47,matched_true_poly_K2to6=23,
        unknown_native_events=459,
        original_full_S80_changed_vs_S274=0,
        actual_improvement_poly_correct_events=(
           original['matched_true_poly_K2to6']['candidate_correct']-
           original['matched_true_poly_K2to6']['baseline_correct']),
        original_vs_freeze_per_class=original['matched_per_K'],
        matched_K1to6_cluster_bootstrap=original['bootstrap_K1to6'],
        matched_K2to6_cluster_bootstrap=original['bootstrap_poly_K2to6'],
        musical_compositions_in_cohort=4,
        global_K0to6_ExactK_proven=False,
        composition_disjointness_proven=False,
        no_fitted_model_changed=True,
        promotion_allowed=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+"\n")
    print("S300 independent read-only audit finished; historical reference stays freeze_local_combo",flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s299','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();audit(args.s299,args.out)
