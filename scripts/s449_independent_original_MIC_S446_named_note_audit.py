"""S449: paired retrospective GuitarSet MIC note scoring, NO retraining.

S448 outputs must be SHA sealed BEFORE this script reads original JAMS.
Same 174 S437 events and 12 compositions. A useful microphone domain
transfer diagnostic, not truly blind/unseen music: S438 labels were scored
in prior research and S80 saw GuitarSet historically.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s440_pairwise_chord_corrections_S439_vs_S437_real_MIC import evaluate
from scripts.s402_k_input_first_supervised_midi_pitch_network import (
    composition_key,need,
)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import read_note_events
from scripts.yourmt3_exactk_common import assign_onsets

def run(s437_names,s437_report,s448_names,s448_report,
        original_events,s80,truth,annotations,out):
    need(not out.exists(),"S449 cannot overwrite earlier audit")
    source=json.loads(s437_report.read_text())
    experiment=json.loads(s448_report.read_text())
    need(source["status"]=="S437_FROZEN_MULTI_GUITAR_REAL_MIC_WAV_S80_SOURCE_NOTE_NAMES"
         and experiment["status"]=="S448_S446_CROSSFIT_REAL_GUITAR_MIC_WAV_NOTE_NAMES_FROZEN"
         and experiment["no_truth_or_original_JAMS_loaded"]
         and experiment["no_training_or_retuning"]
         and experiment["source_S80_original_K_unchanged"]
         and experiment["original_frozen_S437_SHA256"]==sha(s437_names)
         and experiment["output_SHA256"]==sha(s448_names)
         and source["output_SHA256"]==sha(s437_names),
         "S449 must score only frozen S437/S448 predictions")
    base=[json.loads(row) for row in s437_names.read_text().splitlines()]
    cand=[json.loads(row) for row in s448_names.read_text().splitlines()]
    need(len(base)==len(cand)==174,"S449 real source row count altered")
    for row,a in zip(base,cand):
        need(all(row[col]==a[col] for col in (
            "original_row_id","member","start_sample",
            "end_sample","original_S80_K"))
             and len(row["notes"])==len(a["notes"])==row["original_S80_K"],
             "S449 unauthorized original S80 K or MIC event changed")
    original=collections.defaultdict(list)
    with np.load(original_events,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        member=z["member"].astype(str)
        start=z["start_samples"].astype(np.int64)
        end=z["end_samples"].astype(np.int64)
    with np.load(s80,allow_pickle=False) as z:
        need(np.array_equal(z["row_id"].astype(str),ids),
             "S449 original S80 K row IDs invalid")
        predicted_K=z["s80_K"]
    with np.load(truth,allow_pickle=False) as z:
        need(np.array_equal(z["row_id"].astype(str),ids),
             "S449 true-K source drift")
        true_K=z["true_K"]
    need(len(ids)==59309 and int(predicted_K.sum())==33475,
         "S449 original S80 source inventory altered")
    row_ix={r:i for i,r in enumerate(ids)}
    members={row["member"] for row in base}
    need(len(members)==12 and len(set(composition_key(x) for x in members))==12,
         "S449 not same 12 real MIC test compositions")
    with zipfile.ZipFile(annotations) as z:
        names=collections.defaultdict(list)
        for path in z.namelist():
            if path.endswith(".jams"):
                names[PurePosixPath(path).name].append(path)
        for m in sorted(members):
            matches=names[PurePosixPath(m).name]
            need(len(matches)==1,"S449 ambiguous JAMS musical notes")
            ii=np.flatnonzero(member==m)
            ii=ii[np.argsort(start[ii],kind="stable")]
            notes=read_note_events(z.read(matches[0]))
            _,mapping=assign_onsets(
                np.asarray([x[0] for x in notes],np.int64),start[ii],end[ii])
            for (_,pitch,string),pos in zip(notes,mapping):
                if pos>=0:
                    original[int(ii[int(pos)])].append(int(pitch))
            need(all(len(original[j])==int(true_K[j]) for j in ii),
                 "S449 exact original song JAMS note count verification failed")
    pairs={}
    for row in base:
        rid=row["original_row_id"]
        need(rid in row_ix,"S449 frozen MIC event ID missing")
        i=row_ix[rid]
        need(row["member"]==member[i]
             and row["start_sample"]==int(start[i])
             and row["end_sample"]==int(end[i])
             and row["original_S80_K"]==int(predicted_K[i]),
             "S449 row identity does not match protected S80 source")
        pairs[rid]=original[i]
    scored,changes=evaluate(base,{"S448":cand},pairs)
    need(scored["original_S437"]["all_events"]==174
         and scored["original_S437"]["whole_chords_correct"]==78
         and scored["original_S437"]["poly_K2plus_events"]==90
         and scored["original_S437"]["poly_K2plus_whole_chords_correct"]==27,
         "S449 original S438 12-MIC guitar accuracy not reproduced")
    change=scored["S448"]["paired_vs_frozen_S437"]
    need(change["corrections"]-change["regressions"]
         ==scored["S448"]["whole_chords_correct"]-78,
         "S449 paired correction and regression numbers inconsistent")
    out.mkdir(parents=True)
    rep=dict(status="S449_S446_REAL_GUITAR_MIC_PAIRED_RETROSPECTIVE_AUDIT",
        score=scored,
        original_S80_K_unchanged=True,exactK_regressions=0,
        no_model_refit_or_threshold_tuning=True,
        prediction_outputs_S448_frozen_before_opening_own_annotations=True,
        S438_labels_known_from_prior_work_not_a_true_blind_holdout=True,
        S80_GuitarSet_historically_exposed=True,
        source_MIC_domain_not_pickup_mix=True,
        independent_new_instrument_generalization_not_demonstrated=True,
        active_references_untouched=True,
        promotion=False,
        SHA256=dict(S437=sha(s437_names),S448=sha(s448_names),
                    S446=experiment["pretrained_S446_report_SHA256"],
                    original_annotation=sha(annotations)))
    (out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    (out/"paired_changes.jsonl").write_text(
        "".join(json.dumps(change)+"\n" for change in changes))
    print("S449 REAL MIC PAIRED",json.dumps(
        {k:dict(all_correct=v["whole_chords_correct"],
                poly_correct=v["poly_K2plus_whole_chords_correct"],
                poly_events=v["poly_K2plus_events"],note_F1=v["per_note_F1"],
                paired=v.get("paired_vs_frozen_S437"))
         for k,v in scored.items()}),flush=True)
    print("S449 PASS independently re-parsed original GuitarSet MIC notes; zero S80 K changes",flush=True)

def selftest():
    base=[dict(original_row_id="a",notes=["E3","G3"]),
          dict(original_row_id="b",notes=["E3","B3"])]
    future=[dict(original_row_id="a",notes=["E3","B3"]),
            dict(original_row_id="b",notes=["E3","B3"])]
    truth={"a":[52,59],"b":[52,55]}
    result,_=evaluate(base,{"S448":future},truth)
    v=result["S448"]["paired_vs_frozen_S437"]
    assert v["corrections"]==1 and v["regressions"]==0
    print("S449 SELFTEST PASS per-event K locked note correction/regression")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    for name in ("s437-names","s437-report","s448-names","s448-report",
                 "original-events","s80","truth","annotations","out"):
        ap.add_argument("--"+name,type=Path)
    x=ap.parse_args()
    if x.self_test:selftest()
    else:
        for name in ("s437_names","s437_report","s448_names","s448_report",
                     "original_events","s80","truth","annotations","out"):
            need(getattr(x,name)is not None,"S449 missing --"+name)
        run(x.s437_names,x.s437_report,x.s448_names,x.s448_report,
            x.original_events,x.s80,x.truth,x.annotations,x.out)
