"""S488 independent postfreeze evaluation Spotify Basic Pitch on actual IDMT guitar.

Same 47 true XML matched attack events, 45 MIDI-supported, 21 true poly,
same S453 true source and S297 S80 K. Compare TWO distinct label protocols:
(a) new note starts in 20ms onset group vs pretrained emitted note onsets
(b) all notes sounding +30ms vs pretrained sustained active notes with
fixed probe at +30ms AND +130ms.
The second is comparable to original S80 K truth definition (not the
same decoder latency), not a new S80 score or deployment.
Source original S487 Basic Pitch predictions fully frozen and SHA verified
before XML is opened. Unknown 459 native events are not true K0.
"""
from __future__ import annotations
import argparse,collections,json,hashlib,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s464_validate_IDMT_farK0_audio_rms_against_recording import ReliableReadOnlyRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s471_independent_MAESTRO_MIDI_postfrozen_real_V88_K_audit import match_times
SR=44100
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):return sha(Path(p).read_bytes())
def group_true(events):
    groups=[]
    for st,en,p in events:
        if not groups or float(st)-groups[-1]["anchor"]>.020:
            groups.append(dict(anchor=float(st),new=[]))
        groups[-1]["new"].append(int(p))
    return groups
def group_pred(notes):
    groups=[]
    for note in sorted(notes,key=lambda a:(a["predicted_onset_s"],a["pitch_MIDI"])):
        t=float(note["predicted_onset_s"])
        if not groups or t-groups[-1]["anchor"]>.020:
            groups.append(dict(anchor=t,notes=set()))
        groups[-1]["notes"].add(int(note["pitch_MIDI"]))
    return groups
def report(truth,estimate,poly):
    good=np.asarray([a==b for a,b in zip(truth,estimate)],bool)
    trueK=np.asarray([len(a) for a in truth],int)
    predK=np.asarray([len(a) for a in estimate],int)
    target=np.asarray(poly,bool)
    need(len(good)==len(target) and np.all(target==(trueK>=2)),
         "S488 poly annotation mask must be from genuine true MIDI pitches")
    tp=sum(len(a&b) for a,b in zip(truth,estimate))
    tr=sum(map(len,truth));pr=sum(map(len,estimate))
    return dict(events=len(good),poly_events=int(target.sum()),
        whole_chord_note_set_exact=int(good.sum()),
        poly_note_set_exact=int(np.sum(good&target)),
        whole_chord_K_exact=int(np.sum(trueK==predK)),
        poly_K_exact=int(np.sum((trueK==predK)&target)),
        note_precision=float(tp/pr) if pr else None,
        note_recall=float(tp/tr) if tr else None,
        K_under=int(np.sum(predK<trueK)),K_over=int(np.sum(predK>trueK)),
        predicted_no_notes=int(np.sum(predK==0)))
def run(frozen,frozenreport,s276,s277,s453,s453report,s297,out):
    need(not out.exists(),"S488 postfreeze genuine IDMT XML note scoring already exists")
    r=json.loads(frozenreport.read_text())
    cohort=json.loads(s276.read_text())
    control=json.loads(s453report.read_text())
    need(r["status"]=="S487_FIVE_GUITAR_IDMT_REAL_WAV_PRETRAINED_BASIC_PITCH_FROZEN_NOXML"
         and r["all_output_note_events_sha256"]==file_sha(frozen)
         and r["truly_original_WAV_SHA_verified"]
         and r["original_XML_NOT_READ_before_note_freeze"]
         and r["model_no_IDMT_truth_training_or_postprocessing_tuning"]
         and cohort["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and control["status"]=="S453_ORIGINAL_S5_S8_S18_S80_EXTERNAL_IDMT_WRONGK_STAGE_AUDIT"
         and control["annotated_rows_SHA256"]==file_sha(s453),
         "S488 genuine original IDMT pretrained note event freeze must precede any annotations")
    with np.load(s277,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        members=z["member"].astype(str)
        start=z["start_samples"].astype(np.int64)
    with np.load(s297,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str)),
             "S488 original S80 source row identity must remain exactly S277")
        baseline=z["s80_K"].astype(int)
    source=[json.loads(x) for x in s453.read_text().splitlines()]
    notes=[json.loads(x) for x in frozen.read_text().splitlines()]
    bymember=collections.defaultdict(list)
    for item in notes:bymember[item["WAV_filename_SHA256"]].append(item)
    index={q:i for i,q in enumerate(ids)}
    selected={v["wav_filename_sha256"]:v for v in
              cohort["candidate_groups"]["dataset3_first"]}
    need(len(source)==47 and len(index)==506
         and set(selected)==set(members)==set(bymember),
         "S488 original true native IDMT matched 47 attacks or 5 WAV drift")
    req=urllib.request.Request(API,headers={
        "Accept":"application/json","User-Agent":"S488-postfrozen-Notes-real-IDMT-XML"})
    with urllib.request.urlopen(req,timeout=60) as response:
        loc=audit(json.load(response))
    zreader=ReliableReadOnlyRangeZIP(loc["source_url"],loc["size_bytes"])
    rows=[];proof=[]
    with zipfile.ZipFile(zreader) as z:
        need(sha("\n".join(sorted(z.namelist())).encode())==
             cohort["zip_central_names_sha256"],
             "S488 original IDMT annotation source changed after freeze")
        for ident,entry in sorted(selected.items()):
            xml=entry["annotation_path_name_only"]
            content=z.read(xml)
            events=parse_original_pitch_events(content)
            groups=group_true(events)
            pred=group_pred(bymember[ident])
            trueanchors=[];truthrows=[]
            memberix=np.flatnonzero(members==ident)
            memberix=memberix[np.argsort(start[memberix],kind="stable")]
            used=set()
            for idx in memberix:
                t=float(start[idx]/SR)
                options=sorted((abs(q["anchor"]-t),j)
                                for j,q in enumerate(groups)
                                if j not in used and abs(q["anchor"]-t)<=.080)
                if not options:continue
                _,j=options[0];used.add(j)
                g=groups[j]
                active={int(p) for st,en,p in events
                          if st<=g["anchor"]+.030 and en>g["anchor"]+.030}
                if not (1<=len(active)<=6):continue
                truthrows.append(dict(row_id=str(ids[idx]),member=ident,
                    true_anchor=g["anchor"],true_active=active,
                    true_new=set(g["new"]),original_S80_K=int(baseline[idx])))
                trueanchors.append(g["anchor"])
            pairs=match_times(np.asarray(trueanchors,float),
                              np.asarray([g["anchor"] for g in pred],float),.080)
            paired={i:j for i,j,d in pairs}
            for i,q in enumerate(truthrows):
                if not all(40<=p<=88 for p in q["true_active"]):
                    continue
                proposed_onset=(set(pred[paired[i]]["notes"]) if i in paired else set())
                anchor=q["true_anchor"]
                active030={int(a["pitch_MIDI"]) for a in bymember[ident]
                      if a["predicted_onset_s"]<=anchor+.030<
                        a["predicted_offset_s"]}
                active130={int(a["pitch_MIDI"]) for a in bymember[ident]
                      if a["predicted_onset_s"]<=anchor+.130<
                        a["predicted_offset_s"]}
                rows.append(dict(row_id=q["row_id"],source_wav=ident,
                     true_new=q["true_new"],true_active=q["true_active"],
                     original_S80_K=q["original_S80_K"],
                     original_XML_anchor=anchor,
                     basicpitch_new_80ms=proposed_onset,
                     basicpitch_active_30ms=active030,
                     basicpitch_active_130ms=active130))
            proof.append(dict(source_SHA=ident,XML_SHA256=sha(content),
                       native_V88_rows=len(memberix),
                       original_matched_XML=len(truthrows),
                       BasicPitch_detected_onsets=len(pred),
                       matched_original_XML_to_BasicPitch_groups=len(pairs)))
    actual={x["source_row_id"]:x for x in source
            if x["within_supported_notes"]}
    need(len(rows)==45 and len(actual)==45
         and set(actual)=={a["row_id"] for a in rows}
         and sum(len(x["true_active"])>=2 for x in rows)==21
         and sum(x["original_S80_K"]==len(x["true_active"]) for x in rows)==25,
         "S488 paired EXACT original 45 supported guitar attacks/S80 25-of45 provenance mismatch")
    for q in rows:
        need(len(q["true_active"])==int(actual[q["row_id"]]["actual_K"]),
             "S488 genuine IDMT XML sounding group vs S453 true note K changed")
    result={}
    for key,truekey,predkey in [
       ("newnotes_80ms","true_new","basicpitch_new_80ms"),
       ("sounding_active_30ms","true_active","basicpitch_active_30ms"),
       ("sounding_active_130ms","true_active","basicpitch_active_130ms")]:
        truth=[q[truekey] for q in rows]
        guess=[q[predkey] for q in rows]
        result[key]=report(truth,guess,[len(a)>=2 for a in truth])
    out.mkdir(parents=True)
    details=out/"S488_45_TRUE_GUITAR_SOURCE_NOTESETS_PAIRED.jsonl"
    details.write_text("".join(json.dumps({
       k:sorted(v) if isinstance(v,set) else v for k,v in obj.items()})+"\n"
       for obj in rows))
    reportjson=dict(status="S488_PAIRED_IDMT_45_TRUE_GUITAR_ATTACKS_MULTISOURCE_BASIC_PITCH",
       true_IDMT_WAV_count=5,exact_original_matched_annotation_rows=45,
       original_matched_true_poly=21,
       original_S80_exactK_matching_active_30ms=25,
       original_S80_true_poly_exactK_matching_active=7,
       BasicPitch_untrained_on_IDMT_this_trial=True,
       source_note_predictions_frozen_before_xml=True,
       label_new_onsets_and_active_sounding_NOT_the_same=True,
       missing_459_native_candidates_not_true_K0=True,
       no_model_threshold_tuning_on_IDMT=True,
       no_S80_live_or_pitches_modified=True,
       source_original_annotation_prior_exposure=True,
       no_promotion=True,
       score_protocols=result,per_source=proof,
       frozen_note_SHA256=file_sha(frozen),
       matched_rows_SHA256=file_sha(details))
    (out/"report.json").write_text(json.dumps(reportjson,indent=2)+"\n")
    print("S488 ORIGINAL IDMT GUITAR NOTES",json.dumps(dict(
       S80_active_baseline=25,
       scores=result)),flush=True)
def selftest():
    x=report([{52,55}],[{52,55}],[True])
    need(x["whole_chord_K_exact"]==1 and x["whole_chord_note_set_exact"]==1
         and x["poly_note_set_exact"]==1,
         "S488 note E3 G3 whole source correctness broken")
    print("S488 SELFTEST PASS grouped true guitar notes and matched BasicPitch distinct true protocols")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("frozen","frozenreport","s276","s277","s453","s453report","s297","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("frozen","frozenreport","s276","s277","s453","s453report","s297","out"):
            need(getattr(a,k)is not None,"S488 missing "+k)
        run(a.frozen,a.frozenreport,a.s276,a.s277,a.s453,a.s453report,a.s297,a.out)
