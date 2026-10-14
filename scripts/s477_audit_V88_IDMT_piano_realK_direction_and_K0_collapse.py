"""S477 independent error-direction audit: original V88 piano vs original V88 IDMT.

MAESTRO third corpus uses *new notes* per onset MIDI groups (S476).
IDMT original S453 uses all sounding notes +30ms after grouped guitar XML
onset; these K definitions are not identical, so this is exploratory only.
Compute original V88 confusion by TRUE pitch count and source: K0-collapse
versus overcount or undercount, and guide future K0 rescue research.
Read SHA-pinned outcomes only. No acoustic learning or promotion.
"""
from __future__ import annotations
import argparse,json,collections,hashlib
from pathlib import Path
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def confusion(rows):
    m=[[0]*7 for _ in range(7)]
    bysong=collections.defaultdict(list)
    for row in rows:
        k=int(row["truth_K"]);p=int(row["pred_K"])
        if not (0<=k<=6 and 0<=p<=6):continue
        m[k][p]+=1
        bysong[row["song"]].append((k,p))
    def stat(group):
        s=list(group)
        return dict(n=len(s),correct=sum(k==p for k,p in s),
            predicted_zero=sum(p==0 for k,p in s),
            under=sum(p<k for k,p in s),
            over=sum(p>k for k,p in s),
            by_predicted_K_hist=[sum(p==z for k,p in s) for z in range(7)],
            by_true_K_hist=[sum(k==z for k,p in s) for z in range(7)])
    allpairs=[(k,p) for row in rows
        for k,p in [(int(row["truth_K"]),int(row["pred_K"]))] if 1<=k<=6]
    poly=[pair for pair in allpairs if pair[0]>=2]
    return dict(all_known=stat(allpairs),true_poly=stat(poly),
        trueK_predK_confusion=m,
        per_song={name:stat(vals) for name,vals in sorted(bysong.items())})
def run(piano,piano_report,idmt,idmt_report,out):
    need(not out.exists(),"S477 source V88 original confusion audit already exists")
    m=json.loads(piano_report.read_text())
    i=json.loads(idmt_report.read_text())
    need(m["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
         and m["SHA256"]["paired_label_comparison"]==sha(piano)
         and m["matched_onsets"]==322
         and i["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and i["annotated_rows_SHA256"]==sha(idmt)
         and i["valid_pitch_supported_events"]==45,
         "S477 original piano and IDMT precommitted source results incompatible")
    newer=[json.loads(x) for x in piano.read_text().splitlines()]
    older=[json.loads(x) for x in idmt.read_text().splitlines()]
    need(len(newer)==322 and len(older)==47,
         "S477 original matched MIDI / XML events missing")
    main=[]
    for r in newer:
        if not r["both_MIDI40_88"]:continue
        k=int(r["true_new_onset_K"])
        if not (1<=k<=6):continue
        main.append(dict(truth_K=k,pred_K=int(r["original_V88_K"]),
                         song=r["row_id"].split(":")[0]))
    old=[]
    for r in older:
        if not r["within_supported_notes"]:continue
        old.append(dict(truth_K=int(r["actual_K"]),
                 pred_K=int(r["each_original_stage_K"]["V88_freeze"]),
                 song=r["member_SHA"]))
    a=confusion(main);b=confusion(old)
    need(a["all_known"]["n"]==322
         and a["all_known"]["correct"]==42
         and a["true_poly"]["n"]==73
         and a["true_poly"]["correct"]==7
         and b["all_known"]["n"]==45
         and b["true_poly"]["n"]==21
         and b["true_poly"]["correct"]==4,
         "S477 original source sample selection or learned count predictions differ")
    out.mkdir(parents=True)
    report=dict(status="S477_ORIGINAL_V88_DIRECTIONAL_SOURCE_ERRORS_PIANO_VS_IDMT",
       V88_original_both_sources=True,
       note_MAESTRO_trueK_new_MIDI_onset=True,
       note_IDMT_trueK_sounding_at_30ms_after_XML_onset=True,
       K_label_definitions_not_identical_and_cannot_merge_score=True,
       MAESTRO_first_three_piano_excerpt_length_seconds=12,
       IDMT_first_five_guitar_excerpt_length_seconds=10,
       MAESTRO_V88_third_piano=a,IDMT_original_V88_guitar=b,
       no_new_model_or_threshold_selection=True,
       S80_official_NOT_TESTED_on_MAESTRO=True,
       no_promotion=True,
       SHA256=dict(MAESTRO_322=sha(piano),IDMT_47=sha(idmt)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S477 SOURCE COMPARISON",json.dumps(dict(
        MAESTRO_PIANO_NEWK=dict(all=a["all_known"],poly=a["true_poly"]),
        IDMT_GUITAR_ACTIVEK=dict(all=b["all_known"],poly=b["true_poly"]),
        warning="V88 only and different MIDI/XML K truth definitions")),flush=True)
def selftest():
    x=confusion([dict(truth_K=2,pred_K=0,song="piano"),
        dict(truth_K=2,pred_K=3,song="piano"),dict(truth_K=2,pred_K=2,song="piano")])
    need(x["true_poly"]["n"]==3 and x["true_poly"]["under"]==1
         and x["true_poly"]["over"]==1 and x["true_poly"]["correct"]==1,
         "S477 K0-collapse versus K overcount source confusion incorrect")
    print("S477 SELFTEST PASS classify under/over from true K without labels on unmatched")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for a in ("piano","piano-report","idmt","idmt-report","out"):
        p.add_argument("--"+a,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for name in ("piano","piano_report","idmt","idmt_report","out"):
            need(getattr(a,name)is not None,"S477 missing "+name)
        run(a.piano,a.piano_report,a.idmt,a.idmt_report,a.out)
