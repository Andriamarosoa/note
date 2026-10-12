"""S303: audit why 459 original V88 events lack matched XML onsets.

The XMLs for the five Dataset3 WAVs were ALREADY first opened by S299,
after all original/S274 S80 predictions had been frozen. S303 is a
read-only diagnostic on that exposed cohort; cannot requalify it as a
fresh validation. Note intervals at each event time do NOT justify
labelling gaps as K0 without transcription completeness assurance.

No fitting, no candidate decision changes, no global Exact-K score.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import (
    parse_original_pitch_events,nearest_annotated_k)

SR=44100
SOUND=.030
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def hist(k):
    x={}
    for i in range(7):
        x["K"+str(i)]=int(np.count_nonzero(k==i))
    x["higher_than_K6"]=int(np.count_nonzero(k>6))
    return x
def run(s276,s299report,s299pred,out):
    need(not out.exists(),"S303 must not overwrite the original exposed-label audit")
    sel=json.loads(s276.read_text())
    report=json.loads(s299report.read_text())
    need(sel["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
        and sel["XML_MIDI_annotation_contents_read"] is False,
        "S276 prospective XML-unread selection required")
    need(report["status"]=="S299_FROZEN_ORIGINAL_VS_STRICT_S80_IDMT_DATASET3_XML_MATCHED_NOTE_ONLY"
         and report["event_population"]==506 and report["selected_WAVs"]==5
         and report["matched_true_K1to6_events"]==47
         and report["unmatched_or_unknown_label_events"]==459
         and report["initial_S276_selection_SHA256"]==sha_file(s276)
         and not report["full_Global_ExactK_K0to6_measured"],
         "S299 immutable post-frozen source required")
    with np.load(s299pred,allow_pickle=False) as z:
        required={"row_id","member","start_samples","XML_true_K_or_minus1",
            "matched_offset_ms","original_official_freeze_K","original_S80_K",
            "S274_twofold_full_S80_K"}
        need(set(z.files)==required,"S303 truth/original prediction source mismatch")
        fields={k:z[k].copy() for k in z.files}
    ids=fields["row_id"]
    members=fields["member"].astype(str)
    starts=fields["start_samples"]
    onset_truth=fields["XML_true_K_or_minus1"]
    need(len(ids)==506 and len(set(map(str,ids)))==506 and
         (onset_truth>=1).sum()==47 and (onset_truth==-1).sum()==459,
         "S303 506-row IDMT original V88 evidence altered")
    missing=onset_truth==-1
    selected=sel["candidate_groups"]["dataset3_first"]
    original_proofs={x["wav_filename_sha256"]:x for x in
                     report["WAV_level_annotation_proofs"]}
    need(len(selected)==len(original_proofs)==5,"all five original XML proofs required")
    with urllib.request.urlopen(urllib.request.Request(
         API,headers={"Accept":"application/json",
          "User-Agent":"S303-already-exposed-XML-diagnostic-not-promotion"}),timeout=60) as response:
        info=audit(json.load(response))
    stream=HttpRangeZIP(info["source_url"],info["size_bytes"])
    timed_k=np.full(506,-1,np.int16)
    per_wav=[]
    with zipfile.ZipFile(stream,"r") as archive:
        need(sha("\n".join(sorted(archive.namelist())).encode())==
             sel["zip_central_names_sha256"],
             "published WAV/XML ZIP directory changed")
        for entry in selected:
            wav=entry["wav"]
            identity=entry["wav_filename_sha256"]
            xmlname=entry["annotation_path_name_only"]
            rows=np.flatnonzero(members==identity)
            need(len(rows)>0 and identity in original_proofs,"missing S299 original selected WAV")
            payload=archive.read(xmlname)
            need(sha(payload)==original_proofs[identity]["XML_SHA256"],
                 "S303 XML differs from S299 first post-freeze annotation opening")
            notes=parse_original_pitch_events(payload)
            onset_check,_,_=nearest_annotated_k(notes,
                starts[rows]/SR,.080)
            # Recheck each original S299 onset association in its stored
            # per-WAV native-order, WITHOUT introducing a new scoring rule.
            need(np.array_equal(onset_check,onset_truth[rows]),
                 "original S299 matched onset labels cannot be reproduced")
            onset=np.array([note[0] for note in notes],dtype=np.float64)
            end=np.array([note[1] for note in notes],dtype=np.float64)
            pitch=np.array([note[2] for note in notes],dtype=np.int16)
            for i in rows:
                t=float(starts[i])/SR+SOUND
                active=(onset<=t)&(end>t)
                timed_k[i]=len(set(map(int,pitch[active])))
            unmatched_local=rows[missing[rows]]
            vals=timed_k[unmatched_local]
            per_wav.append(dict(
                wav_filename_sha256=identity,
                wav_name=Path(wav).name,
                original_native_events=len(rows),
                original_S299_matched_K1to6=int(np.count_nonzero(onset_truth[rows]>=1)),
                unmatched_native_events=len(unmatched_local),
                unmatched_interval_active_K_histogram=hist(vals),
                source_note_event_count=len(notes),
                last_note_offset_sec=(float(end.max()) if len(end) else None),
                first_note_onset_sec=(float(onset.min()) if len(onset) else None),
                XML_SHA256=sha(payload)))
    need(np.all(timed_k>=0) and
         np.array_equal(fields["original_S80_K"],fields["S274_twofold_full_S80_K"]),
         "original XML interval diagnostic must cover all rows and keep S80 source")
    unknown=timed_k[missing]
    known=timed_k[~missing]
    reference=fields["original_official_freeze_K"]
    new=fields["original_S80_K"]
    rpt=dict(
        status="S303_ALREADY_EXPOSED_XML_INTERVAL_ACTIVITY_AUDIT_NO_K0_LABEL_CERTIFICATION",
        rows=506,XML_matched_note_onset_events=47,
        XML_unmatched_onset_events=459,
        XML_unmatched_events_active_note_interval_counts=hist(unknown),
        all_events_XML_interval_activity_counts=hist(timed_k),
        matched_event_onset_label_vs_continuous_activity_differ=int(
           np.count_nonzero(known!=onset_truth[~missing])),
        unmatched_interval_active_pitches_positive=int(np.count_nonzero(unknown>0)),
        unmatched_interval_no_active_XML_note=int(np.count_nonzero(unknown==0)),
        baseline_predicts_K0_at_unmatched_but_XML_interval_active=int(
            np.count_nonzero(missing&(reference==0)&(timed_k>0))),
        S80_predicts_K0_at_unmatched_but_XML_interval_active=int(
            np.count_nonzero(missing&(new==0)&(timed_k>0))),
        original_S299_XML_and_predictions_sha256=sha_file(s299pred),
        original_S299_report_SHA256=sha_file(s299report),
        original_S276_prior_XML_unread_selection_sha256=sha_file(s276),
        per_wav=per_wav,
        XML_was_already_opened_in_earlier_postfreeze_S299=True,
        original_prediction_K_modified=False,models_retrained=False,
        unmatched_K0_certification=False,
        XML_transcription_completeness_independently_verified=False,
        audio_verified_silence_where_XML_note_absent=False,
        fixed_506_population_global_K0toK6_ExactK_MEASURED=False,
        independent_composition_generalization_verified=False,
        official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(rpt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in rpt.items() if k!="per_wav"},indent=2),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for name in ("s276","s299report","s299pred","out"):
        a.add_argument("--"+name,type=Path,required=True)
    z=a.parse_args();run(z.s276,z.s299report,z.s299pred,z.out)
