"""S438 — independent scoring of frozen S437 real guitar MIC note names.

No models trained, no thresholds selected, no edits to predictions. The
original S437 recordings and predicted K-only sample selection were
FROZEN before this stage reads labels. Uses all native source groups
of each composition to assign JAMS original note attacks reliably.
Previous historical S80/GuitarSet exposure and MIC vs pickup mix remain
major caveats. Report per true K0/K1/K2/K3/K4+ and *nonzero attacks*.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s400_evaluate_S399_MIDI_note_pitches_GuitarSet import read_note_events
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s429_audit_actual_E3_G3_B3_note_names_59309_S80_Klocked import categorize_confusion
from scripts.s402_k_input_first_supervised_midi_pitch_network import need,composition_key
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import note_name,sha

N=59309
def subset(name):
    if name=="K0":return lambda k:k==0
    if name=="K1":return lambda k:k==1
    if name=="K2":return lambda k:k==2
    if name=="K3":return lambda k:k==3
    if name=="K4plus":return lambda k:k>=4
    if name=="poly_K2plus":return lambda k:k>=2
    if name=="notes_K1plus":return lambda k:k>=1
    return lambda k:True

def summarize(rows,name):
    items=[x for x in rows if subset(name)(x["true_K"])]
    n=len(items)
    yes=sum(x["pitch_set_exact"] for x in items)
    cK=sum(x["source_K_correct"] for x in items)
    tp=sum(x["correct_pitch_instances"] for x in items)
    truth=sum(len(x["true_note_MIDI"]) for x in items)
    pred=sum(len(x["predicted_note_MIDI"]) for x in items)
    f1=(2*tp/(truth+pred)) if truth+pred else 1.
    return dict(events=n,full_chords_correct=int(yes),
        exact_chord_rate=(yes/n if n else None),
        original_S80_K_correct=int(cK),
        original_S80_K_accuracy=(cK/n if n else None),
        matched_MIDI_note_instances=int(tp),
        total_true_notes=int(truth),
        total_predicted_notes=int(pred),
        note_F1=float(f1))

def audit(names,source_report,source,source_s80,groundtruth,annotations,out):
    need(not out.exists(),"S438 immutable note identity report exists")
    parent=json.loads(source_report.read_text())
    need(parent["status"]=="S437_FROZEN_MULTI_GUITAR_REAL_MIC_WAV_S80_SOURCE_NOTE_NAMES"
         and parent["number_distinct_original_compositions"]==12
         and parent["true_K_labels_and_JAMS_NEVER_loaded_for_sampling_or_inference"],
         "S438 input predictions not sealed independently of JAMS")
    rows=[json.loads(x) for x in names.read_text(encoding="utf8").splitlines()]
    need(len(rows)==parent["sampled_number_audio_events"]
         and len(rows)>=100
         and len(set(x["original_row_id"] for x in rows))==len(rows),
         "S438 original frozen guitar events missing or reordered")
    with np.load(source,allow_pickle=False) as f:
        need({"row_id","member","start_samples","end_samples"}<=set(f.files),
             "S438 source native 59309 group archive missing")
        ids=f["row_id"].astype(str)
        member=f["member"].astype(str)
        start=f["start_samples"].astype(np.int64)
        end=f["end_samples"].astype(np.int64)
    with np.load(source_s80,allow_pickle=False) as f:
        need({"row_id","s80_K"}<=set(f.files)
             and np.array_equal(f["row_id"].astype(str),ids),
             "S438 actual original 59309 S80 source drift")
        original_k=f["s80_K"].astype(np.int8)
    with np.load(groundtruth,allow_pickle=False) as f:
        need({"row_id","true_K"}<=set(f.files)
             and np.array_equal(f["row_id"].astype(str),ids),
             "S438 original true K audit source drift")
        truth_k=f["true_K"].astype(np.int8)
    need(len(ids)==N and len(set(ids))==N and int(original_k.sum())==33475,
         "S438 protected original S80 inventory changed")
    byid={key:i for i,key in enumerate(ids)}
    selected_members=set(x["member"] for x in rows)
    need(len(selected_members)==12 and
         len(set(composition_key(m) for m in selected_members))==12,
         "S438 song-composition count drift")
    target=collections.defaultdict(list)
    byfilename={}
    with zipfile.ZipFile(annotations) as z:
        for p in z.namelist():
            if p.endswith(".jams"):
                byfilename.setdefault(PurePosixPath(p).name,[]).append(p)
        for m in sorted(selected_members):
            loc=byfilename.get(PurePosixPath(m).name,[])
            need(len(loc)==1,"S438 ambiguous JAMS member")
            ii=np.flatnonzero(member==m)
            ii=ii[np.argsort(start[ii],kind="stable")]
            notes=read_note_events(z.read(loc[0]))
            _,mapping=assign_onsets(
                np.asarray([x[0] for x in notes],np.int64),start[ii],end[ii])
            for (_,pitch,string),pos in zip(notes,mapping):
                if pos>=0:target[int(ii[int(pos)])].append(int(pitch))
            need(all(len(target[j])==int(truth_k[j]) for j in ii),
                 "S438 JAMS assignment mismatches original GuitarSet true K")
    note_to_midi={note_name(k):k for k in range(40,89)}
    result=[];confusions=collections.Counter()
    for obj in rows:
        key=obj["original_row_id"]
        need(key in byid,"S438 unknown frozen ID")
        j=byid[key]
        need(obj["member"]==member[j]
             and int(obj["start_sample"])==int(start[j])
             and int(obj["end_sample"])==int(end[j])
             and int(obj["original_S80_K"])==int(original_k[j])
             and len(obj["notes"])==int(original_k[j]),
             "S438 output changed source K, native onset or identity")
        need(all(x in note_to_midi for x in obj["notes"]),
             "S438 note name outside original MIDI-range ABI")
        pred=[note_to_midi[x] for x in obj["notes"]]
        truth=target[j]
        y=collections.Counter(truth);p=collections.Counter(pred)
        correct_set=(y==p)
        common=int(sum((y&p).values()))
        rec=dict(row_id=key,member=obj["member"],
            original_S80_K=int(original_k[j]),
            true_K=int(truth_k[j]),
            source_K_correct=bool(original_k[j]==truth_k[j]),
            pitch_set_exact=bool(correct_set),
            correct_pitch_instances=common,
            true_note_MIDI=truth,
            predicted_note_MIDI=pred,
            true_note_names=[note_name(x) for x in truth],
            predicted_note_names=obj["notes"])
        result.append(rec)
        if not correct_set:
            confusions.update(categorize_confusion(truth,pred))
    acc={name:summarize(result,name) for name in
         ("all","K0","K1","K2","K3","K4plus","poly_K2plus","notes_K1plus")}
    songs={name:summarize([r for r in result if r["member"]==name],"all")
           for name in sorted(selected_members)}
    need(acc["all"]["events"]==len(rows) and
         acc["K0"]["events"]+acc["notes_K1plus"]["events"]==len(rows)
         and acc["notes_K1plus"]["events"]>=30,
         "S438 K-stratified native event totals inconsistent")
    output=dict(status="S438_FROZEN_REAL_GUITAR_MIC_12PIECE_BALANCED_NOTE_IDENTITY_AUDIT",
        source_S437_predicted_K_based_preregistered_sampling=True,
        original_GuitarSet_annotations_opened_only_AFTER_note_output_frozen=True,
        authentic_real_recorded_guitar_WAV_12_distinct_compositions=True,
        complete_original_member_JAMS_true_K_verified=True,
        original_all_59309_S80_source_K_unchanged=True,
        exactK_regressions=0,
        original_MIC_vs_pickup_mix_domain_shift=True,
        historical_S80_GuitarSet_train_overlap=True,
        score_is_a_selected_sample_not_random_population_estimate=True,
        not_independent_unseen_music=True,
        no_training_or_calibration_on_test_labels=True,
        no_YourMT_teacher_present=True,
        no_Offset_or_Live_changes=True,
        original_active_references_untouched=True,
        by_true_K_stratum=acc,
        by_original_song=songs,
        pitch_confusion_types=dict(confusions),
        examples_of_wrong_note_names=[rec for rec in result if not rec["pitch_set_exact"]][:16],
        official_promotion=False,
        source_SHA256=dict(frozen_notes=sha(names),origin_report=sha(source_report),
            original_S80=sha(source_s80),native_events=sha(source),
            labels=sha(annotations)))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(output,indent=2)+"\n")
    print("S438 ACCURACY BY TRUE K",json.dumps(acc),flush=True)
    print("S438 CONFUSIONS",json.dumps(dict(confusions)),flush=True)
    print("S438 PASS real 12-guitar source-K aligned note-name truth AFTER freeze",flush=True)

def selftest():
    a=[dict(true_K=0,pitch_set_exact=True,source_K_correct=True,
             correct_pitch_instances=0,true_note_MIDI=[],predicted_note_MIDI=[]),
       dict(true_K=2,pitch_set_exact=False,source_K_correct=True,
             correct_pitch_instances=1,true_note_MIDI=[52,55],predicted_note_MIDI=[52,59])]
    assert summarize(a,"K0")["full_chords_correct"]==1
    assert summarize(a,"poly_K2plus")["full_chords_correct"]==0
    assert summarize(a,"poly_K2plus")["note_F1"]==.5
    print("S438 SELFTEST PASS: K0 and poly correctly isolated")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for n in ("names","s437-report","source","s80",
              "truth","annotations","out"):
        p.add_argument("--"+n,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for n in ("names","s437_report","source","s80",
                  "truth","annotations","out"):
            need(getattr(a,n)is not None,"S438 missing --"+n)
        audit(a.names,a.s437_report,a.source,
              a.s80,a.truth,a.annotations,a.out)
