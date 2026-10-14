"""S490 true paired corrections and regressions before cross-source note model routing.

Join SHA-verified original real piano and guitar XML/MIDI matched-onset
rows with held frozen independent Basic Pitch note-name suggestions.
For guitar original baseline is S80 source original (25/45 K, 7/21 poly);
for piano baseline is V88 source original (42/322 K, 7/73 poly).
DO NOT mix these stages, train or select a router on exposed test truth.
Compute corrections and regressions of a hypothetical unconditional switch
and protected-K0-only shadow selector as read-only risk decomposition.
"""
from __future__ import annotations
import argparse,collections,json,hashlib
from pathlib import Path
def need(c,msg):
    if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pair(rows):
    stats=collections.Counter()
    per_original_K=collections.defaultdict(collections.Counter)
    per_true_K=collections.defaultdict(collections.Counter)
    per_source=collections.defaultdict(collections.Counter)
    for row in rows:
        k=int(row["true"]);old=int(row["source"]);new=int(row["bp"])
        poly=k>=2
        good_old=k==old
        good_new=k==new
        for bag in (stats,per_original_K[str(old)],per_true_K[str(k)],
                    per_source[row["source_wav"]]):
            bag["n"]+=1
            bag["correct_original"]+=int(good_old)
            bag["correct_basicpitch"]+=int(good_new)
            bag["corrected"]+=int(not good_old and good_new)
            bag["regressed"]+=int(good_old and not good_new)
            bag["original_under"]+=int(old<k)
            bag["new_under"]+=int(new<k)
            bag["original_over"]+=int(old>k)
            bag["new_over"]+=int(new>k)
            if poly:
                bag["true_poly"]+=1
                bag["poly_corrected"]+=int(not good_old and good_new)
                bag["poly_regressed"]+=int(good_old and not good_new)
                bag["poly_original_correct"]+=int(good_old)
                bag["poly_basicpitch_correct"]+=int(good_new)
            if old==0 and k>0:
                bag["true_K0_collapsed"]+=1
                bag["real_K0_recovered"]+=int(new==k)
    def unroll(z):
        out=dict(z)
        out["net"]=out.get("corrected",0)-out.get("regressed",0)
        out["poly_net"]=out.get("poly_corrected",0)-out.get("poly_regressed",0)
        return out
    return dict(all=unroll(stats),
        per_original_predicted_K={k:unroll(v) for k,v in per_original_K.items()},
        per_true_K={k:unroll(v) for k,v in per_true_K.items()},
        per_recording={k:unroll(v) for k,v in per_source.items()})
def run(idmt,idmtreport,piano,pianoreport,piano_truth,piano_truth_report,out):
    need(not out.exists(),"S490 original paired no-promotion audit immutable")
    a=json.loads(idmtreport.read_text())
    b=json.loads(pianoreport.read_text())
    c=json.loads(piano_truth_report.read_text())
    need(a["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
         and a["matched_rows_SHA256"]==sha(idmt)
         and a["exact_original_matched_annotation_rows"]==45
         and a["original_S80_exactK_matching_active_30ms"]==25
         and b["status"]=="S486_PAIRED_PRETRAINED_BASIC_PITCH_PIANO_322_MIDI_NEW_ONSETS"
         and b["SHA256"]["S486_paired_80ms"]==sha(piano)
         and b["original_V88_same322_ExactK_correct"]==42
         and c["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and c["SHA256"]["paired_label_comparison"]==sha(piano_truth),
         "S490 original three pretrained/frozen audio truth source ledgers differ")
    g=[json.loads(line) for line in idmt.read_text().splitlines()]
    p=[json.loads(line) for line in piano.read_text().splitlines()]
    pr=[json.loads(line) for line in piano_truth.read_text().splitlines()]
    byid={q["row_id"]:q for q in pr}
    need(len(g)==45 and len(p)==len(pr)==322 and
         len(byid)==322
         and set(byid)=={q["row_id"] for q in p},
         "S490 exact original genuine note-on source populations must be identical")
    guitar=[dict(true=len(x["true_active"]),source=int(x["original_S80_K"]),
       bp=len(x["basicpitch_active_30ms"]),source_wav=x["source_wav"])
            for x in g]
    piano_rows=[dict(true=len(q["truth_MIDI"]),
      source=int(byid[q["row_id"]]["original_V88_K"]),
      bp=len(q["proposed_MIDI"]),
      source_wav=q["source_sha"])
       for q in p]
    results=dict(IDMT_true_guitar_45_S80_historical=pair(guitar),
         MAESTRO_true_piano_322_V88_historical=pair(piano_rows))
    need(results["IDMT_true_guitar_45_S80_historical"]["all"]["n"]==45
         and results["IDMT_true_guitar_45_S80_historical"]["all"]["correct_original"]==25
         and results["IDMT_true_guitar_45_S80_historical"]["all"]["correct_basicpitch"]==30
         and results["IDMT_true_guitar_45_S80_historical"]["all"]["true_poly"]==21
         and results["IDMT_true_guitar_45_S80_historical"]["all"]["poly_original_correct"]==7
         and results["IDMT_true_guitar_45_S80_historical"]["all"]["poly_basicpitch_correct"]==11
         and results["MAESTRO_true_piano_322_V88_historical"]["all"]["n"]==322
         and results["MAESTRO_true_piano_322_V88_historical"]["all"]["correct_original"]==42
         and results["MAESTRO_true_piano_322_V88_historical"]["all"]["correct_basicpitch"]==151
         and results["MAESTRO_true_piano_322_V88_historical"]["all"]["true_poly"]==73
         and results["MAESTRO_true_piano_322_V88_historical"]["all"]["poly_original_correct"]==7
         and results["MAESTRO_true_piano_322_V88_historical"]["all"]["poly_basicpitch_correct"]==21,
         "S490 paired correction correction/regression accounting invalidates original S486 S488 evidence")
    out.mkdir(parents=True)
    result=dict(status="S490_PAIRED_REAL_GUITAR_AND_PIANO_BASIC_PITCH_CORRECTION_REGRESSION",
        real_original_source_pieces_guitar=5,
        real_original_source_pieces_piano=3,
        never_merge_different_full_S80_guitar_and_source_V88_piano=True,
        truth_protocol_guitar="active_sounding_30ms",
        truth_protocol_piano="new_MIDI_note_on_20ms",
        no_source_training_or_new_thresholds=True,
        no_safety_router_deployed=True,
        no_ExactK_or_Live_or_original_named_notes_changed=True,
        source_IDMT_or_piano_note_truth_previously_seen_historically=True,
        no_promotion=True,
        paired_cross_source=results,
        SHA256=dict(guitar_true=sha(idmt),piano_true=sha(piano),
                    piano_original_V88_K=sha(piano_truth)))
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S490 SOURCE-SPECIFIC COUNTERFACTUAL",json.dumps(dict(
       genuine_guitar_vs_S80=results["IDMT_true_guitar_45_S80_historical"]["all"],
       genuine_piano_vs_V88=results["MAESTRO_true_piano_322_V88_historical"]["all"],
       note="Reject deployment if regressions present; no target leakage")),flush=True)
def selftest():
    r=pair([
      dict(true=2,source=2,bp=1,source_wav="guitar"),
      dict(true=3,source=2,bp=3,source_wav="guitar"),
      dict(true=2,source=2,bp=2,source_wav="guitar")])
    need(r["all"]["corrected"]==1 and r["all"]["regressed"]==1
         and r["all"]["poly_net"]==0,
         "S490 never report only net improvements without individual regressions")
    print("S490 SELFTEST PASS corrections vs regressions paired K")
if __name__=="__main__":
    q=argparse.ArgumentParser()
    q.add_argument("--self-test",action="store_true")
    for name in ("idmt","idmtreport","piano","pianoreport",
                 "piano-truth","piano-truth-report","out"):
        q.add_argument("--"+name,type=Path)
    a=q.parse_args()
    if a.self_test:selftest()
    else:
        for name in ("idmt","idmtreport","piano","pianoreport",
                     "piano_truth","piano_truth_report","out"):
            need(getattr(a,name)is not None,"S490 missing "+name)
        run(a.idmt,a.idmtreport,a.piano,a.pianoreport,
            a.piano_truth,a.piano_truth_report,a.out)
