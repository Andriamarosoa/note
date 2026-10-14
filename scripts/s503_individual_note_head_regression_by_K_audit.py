"""S503: audit every real S502 specialist selection by true K and action.

Only READ external annotated source note sets AFTER real S502 source
predictions are SHA frozen. These same five guitar & three piano
recordings were explored historically; this is DIAGNOSTIC, NOT OOF
training, independent blind validation, or eligible head promotion.

For each 45 guitar S80 and 322 piano V88 original annotated attack,
pair exactly one canonical onset within 80ms. Compare each eligible
head action against independent BP source note-set reference, tracking:
actions, head note-set fixes, regressions, wrong->wrong, exact K
fixes/regressions and every true K0..K6 separately.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s499_score_canonical_musical_onsets_against_true_MIDI_XML import pairing
from scripts.s502_note_birth_head_selection_adapters import HEADS,need
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(true,base,prop,oldK):
    y=set(true);b=set(base);p=set(prop)
    changed=p!=b
    first=b==y;later=p==y
    Kb=len(b);Kp=len(p);Kt=len(y)
    return dict(action=int(changed),fix=int(changed and not first and later),
       regression=int(changed and first and not later),
       wrong_to_wrong=int(changed and not first and not later),
       proposed_full_note_correct=int(later),
       base_full_note_correct=int(first),
       K_fix=int(Kb!=Kt and Kp==Kt),
       K_regression=int(Kb==Kt and Kp!=Kt),
       proposed_K_exact=int(Kp==Kt),
       base_K_exact=int(Kb==Kt),
       true_original_exactK=int(int(oldK)==Kt))
def run(heads,headsreport,truth_g,truth_g_report,truth_p,truth_p_report,canonical,canonical_report,out):
    need(not out.exists(),"S503 immutable diagnostic already written")
    registered=json.loads(headsreport.read_text())
    source=json.loads(canonical_report.read_text())
    sg=json.loads(truth_g_report.read_text());sp=json.loads(truth_p_report.read_text())
    need(registered["status"]=="S502_CLASS_CONDITIONAL_NOTE_HEADS_SELECTIONS_ARCHITECTURE_PRELABEL"
         and registered["no_NN_training_or_external_ground_truth_used"]
         and registered["specialist_OOF_crossfit_not_yet_available"]
         and registered["number_of_registered_heads"]==13
         and source["status"]=="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
         and sg["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and sg["matched_rows_SHA256"]==sha(truth_g)
         and sp["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and sp["SHA256"]["paired_label_comparison"]==sha(truth_p),
         "S503 external truth must never feed head availability/selection")
    results={}
    for corpus,truthfile,sourceid,clock,truefield,origK,expected,correct,poly in (
      ("guitar_S80",truth_g,"source_wav","original_XML_anchor",
       "true_active","original_S80_K",45,25,21),
      ("piano_V88",truth_p,None,"onset_at_excerpt_s",
       "true_new_onset_MIDI_pitches","original_V88_K",322,42,73)):
        headpath=heads/f"S502_{corpus}_HEAD_SELECTIONS_PRELABEL.jsonl"
        canonicalpath=canonical/f"S498_{corpus}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        need(registered["by_source"][corpus]["ledger_SHA256"]==sha(headpath)
             and source["frozen_group_sha256"][corpus]==sha(canonicalpath),
             "S503 original S502 K conditional note specialists changed after label audit")
        ledger=[json.loads(t) for t in headpath.read_text().splitlines()]
        events=[json.loads(t) for t in canonicalpath.read_text().splitlines()]
        truth=[json.loads(t) for t in truthfile.read_text().splitlines()]
        need(len(truth)==expected and len(ledger)==len(events),
             "S503 source true note attack cohort or original canonical event counts changed")
        groups={z["canonical_event_id"]:z for z in ledger}
        bysource=collections.defaultdict(list)
        for z in events:bysource[z["source_recording_sha"]].append(z)
        real_cases=[]
        matches=0
        for recording in sorted({(v[sourceid] if sourceid else v["row_id"].split(":")[0])
                                 for v in truth}):
            ix=[i for i,v in enumerate(truth) if
                (v[sourceid] if sourceid else v["row_id"].split(":")[0])==recording]
            candidate=bysource.get(recording,[])
            assign=pairing([float(truth[i][clock]) for i in ix],
                           [float(e["event_time_seconds"]) for e in candidate])
            for i,j,lag in assign:
                real_cases.append((truth[ix[i]],groups[candidate[j]["canonical_event_id"]]))
                matches+=1
        # Unmatched TRUE positive attack contributes a no-note detection
        # for all heads; do not assume all unpaired BP groups are FALSE.
        known={x["row_id"] for x,_ in real_cases}
        for x in truth:
            if x["row_id"] not in known:real_cases.append((x,None))
        need(len(real_cases)==expected and
             sum(int(len(row[truefield])==int(row[origK])) for row in truth)==correct
             and sum(int(len(row[truefield])>=2) for row in truth)==poly,
             "S503 historical original K true attack denominator mismatch")
        head_stats={}
        baseline_index=1 if corpus=="guitar_S80" else 0
        for j,h in enumerate(HEADS):
            c=collections.Counter()
            by_trueK={str(k):collections.Counter() for k in range(7)}
            by_sourceK={str(k):collections.Counter() for k in range(7)}
            for row,hgroup in real_cases:
                target=tuple(sorted(set(int(x) for x in row[truefield])))
                orig=int(row[origK])
                if hgroup is not None:
                    baseline=tuple(hgroup["head_selections"][baseline_index]["note_pitches"])
                    offered=hgroup["head_selections"][j]
                    eligible=bool(offered["structural_available"])
                    selected=tuple(offered["note_pitches"]) if eligible else baseline
                else:
                    baseline=()
                    selected=()
                    eligible=False
                got=evaluate(target,baseline,selected,orig)
                got["eligible"]=int(eligible)
                got["target_K"]=len(target)
                got["source_K"]=orig
                c.update({k:v for k,v in got.items() if k not in ("target_K","source_K")})
                by_trueK[str(len(target))].update({k:v for k,v in got.items() if
                    k not in ("target_K","source_K")})
                by_sourceK[str(orig)].update({k:v for k,v in got.items() if
                    k not in ("target_K","source_K")})
            d=dict(c);d["n"]=expected
            d["net_note_sets"]=d.get("fix",0)-d.get("regression",0)
            d["net_ExactK"]=d.get("K_fix",0)-d.get("K_regression",0)
            head_stats[h.name]=dict(action=h.action,kind=h.kind,
                source_context_only_not_OOF=True,
                summary=d,
                by_true_K={k:dict(v) for k,v in by_trueK.items()},
                by_original_predicted_K={k:dict(v) for k,v in by_sourceK.items()})
        results[corpus]=dict(
           verified_real_note_events=expected,matched_canonical_music_events=matches,
           true_poly_events=poly,original_source_Kcorrect=correct,
           by_head=head_stats)
        print("S503 INDIVIDUAL HEAD AUDIT",json.dumps(dict(
          source=corpus,
          head_summaries={k:v["summary"] for k,v in head_stats.items()})),flush=True)
    out.mkdir(parents=True)
    report=dict(status="S503_HISTORICAL_NOTE_HEAD_INDIVIDUAL_K_CLASS_ACTION_AUDIT",
        includes_13_head_fixes_and_regressions_by_trueK0toK6=True,
        gate_input_never_uses_true_K_to_choose_head=True,
        S80_guitar_vs_V88_piano_are_different_sources=True,
        source_external_reused_for_diagnostic_only_NOT_OOF=True,
        per_head_audit_MUST_NOT_be_fed_to_same_test_rows_for_selector_training=True,
        trained_neural_selector_not_yet_available=True,
        actual_model_promotions=0,no_promotion=True,
        by_source=results,
        source_SHA256=dict(S502_prelabel=sha(headsreport),
            original_canonical=sha(canonical_report),
            guitar_XML=sha(truth_g),piano_MIDI=sha(truth_p)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def selftest():
    f=evaluate({52,55},{52,64},{52,55},2)
    r=evaluate({52,55},{52,55},{52,64},2)
    need(f["fix"]==1 and r["regression"]==1 and
         f["K_fix"]==0 and r["K_regression"]==0,
         "S503 note identity audit must catch regressions even with K unchanged")
    print("S503 SELFTEST PASS each selection can fix pitches without changing Exact-K")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--self-test",action="store_true")
    for k in ("heads","headsreport","truth-g","truth-g-report","truth-p",
              "truth-p-report","canonical","canonical-report","out"):
        a.add_argument("--"+k,type=Path)
    x=a.parse_args()
    if x.self_test:selftest()
    else:
        for k in ("heads","headsreport","truth_g","truth_g_report","truth_p",
                  "truth_p_report","canonical","canonical_report","out"):
            need(getattr(x,k)is not None,"S503 missing --"+k)
        run(x.heads,x.headsreport,x.truth_g,x.truth_g_report,x.truth_p,
            x.truth_p_report,x.canonical,x.canonical_report,x.out)
