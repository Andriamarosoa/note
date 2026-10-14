"""S433: real WAV + ORIGINAL S80 count NPZ -> automatically musical note names.

Eliminates manually typed K-per-event JSONL from S432 by JOINING by the
original immutable native event ROW ID, even if K and event timing archives
use different row ordering. S80 must be inferred upstream by its actual
historical 16 S45 / 193 S46 / S79 / S80 chain; this adapter DOES NOT run or
replace that trained stack. Requires one WAV and a trusted original source
event archive aligned with genuine S80 predictions. No MIDI file.

Usage:
 python -m scripts.s433_join_original_S80_native_events_to_WAV_note_names \
  --wav song.wav --member exact-member-name-in-events \
  --s80-npz original_S80_prediction.npz \
  --native-events original_V88_events.npz --out model/new_notes \
  --s402-models model/S402 --s416-models model/S416 \
  --s428-models model/S428 --s419-report model/S419/report.json \
  --s428-report model/S428/report.json

Truth/label/MIDI annotations are forbidden as source-input keys.
No synthetic fallback K, no S80 estimates from harmonic peak counts.
K0 = no *new* note, not zero sounding/active notes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.s402_k_input_first_supervised_midi_pitch_network import need
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
from scripts.s432_WAV_original_harmonic_input_to_note_names_with_S80_K import (
    run as wav_to_names,SR,
)

FORBIDDEN={"true_K","label","labels","ground_truth","GT","target","midi_truth",
           "truth","correct_K","reference_notes","annotation"}
ALLOWED_COUNT={"row_id","s80_K"}
ALLOWED_EVENT={"row_id","member","start_samples","end_samples"}

def read_fields(npz,needed,forbidden,kind):
    with np.load(npz,allow_pickle=False) as z:
        fields=set(z.files)
        need(needed<=fields,kind+" missing original source ABI "+str(needed-fields))
        need(not (fields&forbidden),kind+" must never include truth or labels")
        d={name:z[name].copy() for name in needed}
    return d

def join_predictions(s80_npz,event_npz,member,out):
    """Source adapter: no change to K, original timing, identity or order."""
    need(not out.exists(),"S433 cannot overwrite source adapter results")
    k=read_fields(s80_npz,ALLOWED_COUNT,FORBIDDEN,"S80")
    e=read_fields(event_npz,ALLOWED_EVENT,FORBIDDEN,"native events")
    ids=np.asarray(k["row_id"]).astype(str)
    vals=k["s80_K"]
    need(vals.ndim==1 and len(ids)==len(vals)
         and len(ids)==len(set(ids)) and np.issubdtype(vals.dtype,np.integer)
         and np.isin(vals,range(7)).all(),
         "S433 invalid original S80 K0..K6 or duplicate row IDs")
    keys=e["row_id"].astype(str);members=e["member"].astype(str)
    start=e["start_samples"];end=e["end_samples"]
    need(len(keys)==len(set(keys))
         and all(len(a)==len(keys) for a in (members,start,end))
         and np.issubdtype(start.dtype,np.integer)
         and np.issubdtype(end.dtype,np.integer)
         and np.all(start>=0) and np.all(end>=start)
         and np.all((end-start)<=1764),
         "S433 native event source timestamps invalid")
    need(isinstance(member,str) and member.strip(),
         "S433 require exact original native source member name")
    choose=np.flatnonzero(members==member)
    need(len(choose)>0,"S433 requested WAV member absent from event source")
    order=choose[np.argsort(start[choose],kind="stable")]
    need(len(set(keys[order]))==len(order) and
         np.all(np.diff(start[order])>0),
         "S433 repeated events or nonmonotonic native onset time")
    ref={name:int(i) for i,name in enumerate(ids)}
    need(all(key in ref for key in keys[order]),
         "S433 never fabricate original K when source row ID missing")
    # Historical archive could contain many other songs. Join ONLY named WAV.
    ordered_count=np.asarray([int(vals[ref[r]]) for r in keys[order]],np.int8)
    need(np.isin(ordered_count,range(7)).all(),"S433 invalid source K")
    source_rows=[]
    for row,k0,st,ed in zip(keys[order],ordered_count,start[order],end[order]):
        source_rows.append(dict(
            row_id=str(row),start_sample=int(st),
            end_sample=int(ed),s80_K=int(k0)))
    # Persist original predicted K *before* passing to note classifier.
    out.mkdir(parents=True)
    decision_file=out/"original_S80_K_native_join.jsonl"
    with decision_file.open("w",encoding="utf8") as f:
        for item in source_rows:
            f.write(json.dumps(item,sort_keys=True)+"\n")
    # Independently reparse the actual intermediary and check every original
    # source cardinality, including K0 rows (no forced K1 on candidates).
    check=[json.loads(s) for s in decision_file.read_text().splitlines()]
    need(len(check)==len(order)
         and all(row["row_id"]==keys[order][i]
                 and row["s80_K"]==int(ordered_count[i])
                 and row["start_sample"]==int(start[order][i])
                 for i,row in enumerate(check)),
         "S433 actual original S80 event selection was altered")
    report=dict(status="S433_ORIGINAL_S80_NATIVE_WAV_EVENT_ID_JOIN",
        originally_predicted_S80_rows=len(ids),
        original_selected_native_events=len(order),
        originally_predicted_K_sum=int(ordered_count.sum()),
        original_zero_new_attack_events=int((ordered_count==0).sum()),
        original_row_ids_not_replaced=True,
        exact_original_s80_K_per_selected_source_row=True,
        raw_WAV_not_used_to_guess_K=True,
        musical_notes_not_in_native_source=True,
        original_S80_models_not_retrained_or_altered=True,
        original_S80_inference_must_be_run_externally=True,
        selected_native_member=str(member),
        original_input_sha256=dict(s80=sha(s80_npz),events=sha(event_npz)),
        original_join_jsonl_sha256=sha(decision_file))
    (out/"join_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return decision_file,report

def run(wav,member,s80_npz,native_events,s402,s416,s428,r419,r428,out):
    need(not out.exists(),"S433 can only produce a new immutable output")
    decisions,j=join_predictions(
        s80_npz,native_events,member,out/"joined_S80_events")
    wav_to_names(wav,decisions,s402,s416,s428,r419,r428,out/"note_names")
    file=out/"note_names/recognized_note_names_from_WAV.jsonl"
    notes=[json.loads(line) for line in file.read_text().splitlines()]
    original=[json.loads(line) for line in decisions.read_text().splitlines()]
    need(len(notes)==len(original) and all(
       x["source_row_id"]==y["row_id"] and
       x["K"]==y["s80_K"] and len(x["notes"])==y["s80_K"]
       and x["start_sample"]==y["start_sample"]
       for x,y in zip(notes,original)),
       "S433 original S80 cardinality not preserved in actual output")
    report=dict(status="S433_ACTUAL_WAV_TO_E3_NOTE_NAMES_USING_AUTOMATIC_ORIGINAL_S80_NPZ_JOIN",
        output_sound_note_name_file=str(file),
        source_real_WAV_sha256=sha(wav),
        original_S80_prediction_sha256=sha(s80_npz),
        sound_source_event_sha256=sha(native_events),
        original_S80_count_unchanged_for_each_event=True,
        number_audio_events=len(notes),
        number_new_notes=int(sum(x["K"] for x in notes)),
        K0_no_new_notes_confirmed=True,
        original_S80_historical_training_and_producers_unchanged=True,
        input_includes_separately_computed_real_S80_K=True,
        model_itself_does_not_infer_S80_K_from_raw_WAV=True,
        no_true_notes_or_annotations_loaded=True,
        S430_four_fold_ensemble_not_independently_scored=True,
        live_and_offset_unchanged=True,
        no_external_independent_validation_claim=True,
        official_promotion=False)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S433 NOTES",json.dumps(notes,ensure_ascii=False),flush=True)
    print("S433 PASS: join real S80 source events to WAV, detect note names, retain K0..K6",flush=True)

def selftest():
    import tempfile
    from scipy.io import wavfile
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        n=np.array(["foreign_3","song:7","foreign_1","song:14","song:21"])
        k=np.array([5,2,3,0,1],np.int8)
        np.savez_compressed(root/"s80.npz",row_id=n,s80_K=k)
        # Different order: missing/extra false detections never silently
        # borrow a nearby K or sort by K.
        np.savez_compressed(root/"events.npz",
            row_id=np.array(["song:21","foreign_1","song:14","song:7"]),
            member=np.array(["song","other","song","song"]),
            start_samples=np.array([63000,43000,42000,21000],np.int64),
            end_samples=np.array([63256,43256,42256,21256],np.int64))
        p,r=join_predictions(root/"s80.npz",root/"events.npz",
                             "song",root/"join")
        parsed=[json.loads(x) for x in p.read_text().splitlines()]
        need([x["s80_K"] for x in parsed]==[2,0,1]
           and [x["row_id"] for x in parsed]==["song:7","song:14","song:21"],
           "S433 original unrelated K or native event order changed")
        need(r["original_zero_new_attack_events"]==1,
             "S433 K0 was not preserved")
        with np.load(root/"events.npz") as z:
            data={v:z[v] for v in z.files}
        data["row_id"][0]="unknown_source_row"
        np.savez_compressed(root/"bad.npz",**data)
        try:
            join_predictions(root/"s80.npz",root/"bad.npz",
                             "song",root/"bad_out")
        except ValueError:
            pass
        else:
            raise AssertionError("S433 missing K source row silently accepted")
        try:
            with np.load(root/"s80.npz") as z:
                x={v:z[v] for v in z.files}
            x["true_K"]=k
            np.savez_compressed(root/"oracle.npz",**x)
            join_predictions(root/"oracle.npz",root/"events.npz",
                             "song",root/"bad_oracle")
        except ValueError:
            pass
        else:
            raise AssertionError("S433 ground-truth leakage not rejected")
    print("S433 SELFTEST PASS: join shuffled genuine K2/K0/K1, reject missing IDs and truth")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    for option in ("wav","member","s80-npz","native-events",
                   "s402-models","s416-models","s428-models",
                   "s419-report","s428-report","out"):
        parser.add_argument("--"+option,type=str if option=="member" else Path)
    x=parser.parse_args()
    if x.self_test:selftest()
    else:
        for key in ("wav","member","s80_npz","native_events","s402_models",
                    "s416_models","s428_models","s419_report","s428_report","out"):
            need(getattr(x,key)is not None,"S433 missing --"+key)
        run(x.wav,x.member,x.s80_npz,x.native_events,
            x.s402_models,x.s416_models,x.s428_models,
            x.s419_report,x.s428_report,x.out)
