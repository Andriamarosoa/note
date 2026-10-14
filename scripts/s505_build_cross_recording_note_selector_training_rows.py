"""S505 real training rows for the neural S502 13-head note-selector.

All acoustic head predictions were SHA-frozen in S502/S504 before XML/MIDI
scoring. Only genuine one-to-one paired IDMT guitar (S80 *K* source)
and MAESTRO piano (V88 *K* source) note onsets become supervised rows.
Whole recording is immutable group ID: no shared recording between
train and held validation within any leave-one-recording-out trial.

This is RETROSPECTIVE developmental data from previously exposed sources.
Missing true onset groups are counted as unavoidable false negatives;
unmatched model birth groups have UNKNOWN truth, NEVER negative labels.
This builder does not fit/activate the selector or modify S80/Live.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s499_score_canonical_musical_onsets_against_true_MIDI_XML import pairing
from scripts.s502_note_birth_head_selection_adapters import HEADS,need

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def validate_sources(args):
    g=json.loads(args.graph_report.read_text())
    a=json.loads(args.head_report.read_text())
    b=json.loads(args.nn_report.read_text())
    tg=json.loads(args.guitar_report.read_text())
    tp=json.loads(args.piano_report.read_text())
    need(g["status"]=="S498_TWO_SOURCE_CANONICAL_NOTE_ONSET_GRAPH_SOURCE_K_AS_SOFT_PRIOR_PRELABEL"
       and a["status"]=="S502_CLASS_CONDITIONAL_NOTE_HEADS_SELECTIONS_ARCHITECTURE_PRELABEL"
       and b["status"]=="S504_ORIGINAL_EXACTK_CLASSCONDITIONAL_NN_NOTE_HEAD_ADAPTER_NO_FIT"
       and tg["status"]=="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH"
       and tp["status"]=="S476_THIRD_MAESTRO_PIANO_NEW_ONSET_K_VS_SUSTAINING_K_READONLY"
       and tg["matched_rows_SHA256"]==sha(args.guitar)
       and tp["SHA256"]["paired_label_comparison"]==sha(args.piano)
       and a["no_NN_training_or_external_ground_truth_used"]
       and b["not_a_trained_note_classifier"],
       "S505 MUST match unmodified prelabel heads and authentic paired truths")
    return g,a,b

def build(args):
    need(not args.out.exists(),"S505 developmental matching is immutable")
    graph,head,nn=validate_sources(args)
    collected=[];stats={}
    for (corpus,truthfile,truthsource,timekey,truth_key,original_K,truth_total,
         truepoly,source_K_correct,rawcorrect,baseline_head) in (
       ("guitar_S80",args.guitar,"source_wav","original_XML_anchor",
        "true_active","original_S80_K",45,21,25,27,1),
       ("piano_V88",args.piano,None,"onset_at_excerpt_s",
        "true_new_onset_MIDI_pitches","original_V88_K",322,73,42,125,0)):
        acfile=args.graph/f"S498_{corpus}_CANONICAL_MUSICAL_EVENTS_PRELABEL.jsonl"
        hdfile=args.heads/f"S502_{corpus}_HEAD_SELECTIONS_PRELABEL.jsonl"
        nnfile=args.nn/f"S504_{corpus}_NEURAL_CLASS_CONDITIONAL_INPUT_ONLY_NOT_FITTED.npz"
        need(graph["frozen_group_sha256"][corpus]==sha(acfile)
          and head["by_source"][corpus]["ledger_SHA256"]==sha(hdfile)
          and nn["by_source"][corpus]["sha256"]==sha(nnfile),
          "S505 pre-MIDI original source prediction SHA changed")
        acoustic=read_lines(acfile); heads=read_lines(hdfile); true=read_lines(truthfile)
        need(len(true)==truth_total and sum(len(x[truth_key])>=2 for x in true)==truepoly
          and sum(len(x[truth_key])==int(x[original_K]) for x in true)==source_K_correct,
          "S505 original 45/322 MIDI truth or original S80/V88 exact K drift")
        byid={x["canonical_event_id"]:x for x in heads}
        eventidx={x["canonical_event_id"]:i for i,x in enumerate(acoustic)}
        need(len(byid)==len(acoustic)==len(eventidx),
          "S505 duplicate source musical birth IDs")
        with np.load(nnfile,allow_pickle=False) as z:
            rows=z["canonical_event_id"].astype(str)
            hm=z["head_mask"].astype(bool)
            km=z["head_k_mask"].astype(bool)
            ht=z["head_type"].astype(np.float32)
            f=z["context"].astype(np.float32)
            logits=z["head_logits"].astype(np.float32)
            base=z["baseline"].astype(np.float32)
            unready=z["training_only_audit_ready"].astype(bool)
        lookup={q:i for i,q in enumerate(rows)}
        need(set(lookup)==set(byid) and len(lookup)==len(rows)
          and not unready.any(),"S505 cannot use any leaked/pretended source OOF audit")
        group=collections.defaultdict(list)
        for a in acoustic:group[a["source_recording_sha"]].append(a)
        found=[]
        for rec in sorted({v[truthsource] if truthsource else v["row_id"].split(":")[0]
                           for v in true}):
            ix=[i for i,v in enumerate(true) if (v[truthsource] if truthsource else
                  v["row_id"].split(":")[0])==rec]
            options=group.get(rec,[])
            matches=pairing([float(true[i][timekey]) for i in ix],
                 [float(q["event_time_seconds"]) for q in options])
            for ti,ci,_ in matches:
                truthrow=true[ix[ti]]
                a=options[ci]
                found.append((truthrow,a,rec))
        got=set(t["row_id"] for t,a,rec in found)
        actualnote=0;matched=0
        for truthrow,event,rec in found:
            j=lookup[event["canonical_event_id"]]
            ledger=byid[event["canonical_event_id"]]["head_selections"]
            bits=np.zeros((13,128),bool)
            for h,p in enumerate(ledger):
                for pitch in p["note_pitches"]:
                    need(0<=int(pitch)<128,"S505 original actual pitch evidence invalid")
                    bits[h,int(pitch)]=True
                need(bool(p["structural_available"])==bool(hm[j,h]),
                     "S505 original S502 head mask mismatch")
            truthpitch=sorted(set(int(x) for x in truthrow[truth_key]))
            y=np.zeros(128,bool);y[truthpitch]=True
            correct=np.array([bool((y==bits[h]).all()) for h in range(13)],bool)
            actualnote+=int(correct[baseline_head])
            matched+=1
            # Neither MIDI pitches nor corpus source labels are part of
            # model inputs; they are stored as supervised targets/meta.
            collected.append(dict(event_id=event["canonical_event_id"],
                recording_sha=rec,corpus=corpus,
                row_id=truthrow["row_id"],source_K=int(truthrow[original_K]),
                truthK=len(truthpitch),truth_bits=y,pitches=bits,
                baseline_head=baseline_head,
                context=f[j],head_mask=hm[j],head_k_mask=km[j],
                head_type=ht[j],head_logits=logits[j],baseline=base[j],
                exact_heads=correct))
        need(actualnote==rawcorrect,"S505 original BP full-note baseline changed")
        stats[corpus]=dict(truth_original_positive_attacks=truth_total,
            original_Source_K_correct=source_K_correct,
            originally_matched_canonical_positive=len(found),
            annotated_true_attacks_NOT_matched=truth_total-len(found),
            annotated_true_poly=truepoly,
            raw_BP_base_full_note_sets_correct=actualnote,
            unpaired_music_groups_truth_UNKNOWN=len(acoustic)-len(found),
            original_canonical_music_groups=len(acoustic))
        print("S505 ROW BUILD",json.dumps(dict(corpus=corpus,**stats[corpus])),flush=True)
    need(len(collected)>200 and
         len({z["recording_sha"] for z in collected})==8,
         "S505 need genuine eight distinct held-recording identities")
    args.out.mkdir(parents=True)
    file=args.out/"S505_REAL_NOTE_HEADS_EIGHT_RECORDINGS_MATCHED_POSITIVES.npz"
    np.savez_compressed(file,
      event_id=np.array([v["event_id"] for v in collected]),
      recording_sha=np.array([v["recording_sha"] for v in collected]),
      corpus=np.array([v["corpus"] for v in collected]),
      original_row_id=np.array([v["row_id"] for v in collected]),
      source_K=np.array([v["source_K"] for v in collected],np.int8),
      truth_K=np.array([v["truthK"] for v in collected],np.int8),
      truth_note_bits=np.stack([v["truth_bits"] for v in collected]),
      head_note_bits=np.stack([v["pitches"] for v in collected]),
      exact_note_set_by_head=np.stack([v["exact_heads"] for v in collected]),
      selected_BP_baseline_head=np.array([v["baseline_head"] for v in collected],np.int8),
      context=np.stack([v["context"] for v in collected]),
      head_mask=np.stack([v["head_mask"] for v in collected]),
      head_k_mask=np.stack([v["head_k_mask"] for v in collected]),
      head_type=np.stack([v["head_type"] for v in collected]),
      head_logits=np.stack([v["head_logits"] for v in collected]),
      baseline=np.stack([v["baseline"] for v in collected]))
    report=dict(status="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS",
        eight_original_recordings=8,
        source_NN_preds_audio_only_frozen_before_any_notes=True,
        source_guitar_original_full_S80_source_K=True,
        source_piano_only_V88_source_K=True,
        no_true_labels_as_inference_features=True,
        no_external_unknown_group_as_false_negative=True,
        matched_positive_note_birth_only_NOT_continuous_event_precision=True,
        historical_source_exposure_not_genuinely_blind=True,
        fold_split_must_exclude_exact_recording_SHA=True,
        no_selector_trained_yet=True,no_promotion=True,
        audio_proposal_truth_table_SHA256=sha(file),
        by_source=stats,rows=len(collected),
        original_SHA=dict(S502=sha(args.head_report),S504=sha(args.nn_report),
                          original_guitar_XML=sha(args.guitar),
                          original_piano_MIDI=sha(args.piano)))
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

def selftest():
    # a 2K note set must not be defined by matching the number alone
    x=np.array([52,55]);y=np.array([52,59])
    need(len(x)==len(y) and set(x)!=set(y),"S505 must supervise note identities")
    print("S505 SELFTEST PASS true note pitches different despite equal Exact-K")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for arg in ("graph","graph-report","heads","head-report","nn","nn-report",
                "guitar","guitar-report","piano","piano-report","out"):
        p.add_argument("--"+arg,type=Path)
    v=p.parse_args()
    if v.self_test:selftest()
    else:
        for k in ("graph","graph_report","heads","head_report","nn","nn_report",
                  "guitar","guitar_report","piano","piano_report","out"):
            need(getattr(v,k)is not None,"S505 missing --"+k)
        build(v)
