"""S451: audit SHA-frozen S450 external note NAMES on IDMT Dataset3 XML.

The prospective S450 musical outputs are frozen and SHA-verified FIRST.
Only matched true note-attack onsets are scored: no artificial K0 labels.
The onset grouping is unchanged S190/S299: 20ms group, max 80ms to native
event, sounding pitches at +30ms. This is retrospective external-source
transfer, NOT a newly blinded dataset: IDMT dataset3 XML was opened in S299.
No model fitting or threshold calibration allowed.
"""
from __future__ import annotations
import argparse,collections,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import (
    parse_original_pitch_events,nearest_annotated_k)
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import note_name,sha
from scripts.s402_k_input_first_supervised_midi_pitch_network import need

SR=44100
TOL=.080
WINDOW=.020
SOUND=.030
MAX_MIDI=88
MIN_MIDI=40

def annotated_sets(notes,acoustic_time):
    """Reproduce S299's used-once 20ms onset group assignment precisely."""
    groups=[]
    for st,en,p in notes:
        if not groups or st-groups[-1][0]>WINDOW:
            groups.append((st,[]))
        groups[-1][1].append((st,en,p))
    out=[None]*len(acoustic_time)
    used=set()
    for i,t in enumerate(acoustic_time):
        possible=sorted((abs(on-t),j) for j,(on,_) in enumerate(groups)
                        if j not in used and abs(on-t)<=TOL)
        if not possible:continue
        _,j=possible[0]
        used.add(j)
        anchor=groups[j][0]
        sounding=sorted({int(p) for st,en,p in notes
                         if st<=anchor+SOUND and en>anchor+SOUND})
        if not (1<=len(sounding)<=6):
            continue
        out[i]=sounding
    expected,_,_=nearest_annotated_k(notes,np.asarray(acoustic_time),TOL)
    need(all((len(x) if x is not None else -1)==int(y) for x,y in zip(out,expected)),
         "S451 per-note groups diverge from precommitted S299 K-only labels")
    return out

def evaluate(rows,truth,variant):
    subsets={"all_matched":lambda k:True,"true_poly_K2plus":lambda k:k>=2}
    for k in range(1,7):subsets["true_K"+str(k)]=lambda v,kk=k:v==kk
    measured={}
    for key,rule in subsets.items():
        indices=[i for i,g in enumerate(truth) if g is not None and rule(len(g))
                 and all(MIN_MIDI<=p<=MAX_MIDI for p in g)]
        old_hits=new_hits=old_tp=new_tp=true_notes=old_notes=new_notes=0
        fixes=regressions=0;exactK_hits=0;changed=0
        bysong=collections.defaultdict(list)
        for i in indices:
            baseline=collections.Counter(int(p) for p in rows[i]["S430"])
            challenger=collections.Counter(int(p) for p in rows[i][variant])
            actual=collections.Counter(truth[i])
            was=baseline==actual;now=challenger==actual
            old_hits+=was;new_hits+=now
            fixes+=now and not was;regressions+=was and not now
            old_tp+=sum((baseline&actual).values())
            new_tp+=sum((challenger&actual).values())
            true_notes+=sum(actual.values())
            old_notes+=sum(baseline.values())
            new_notes+=sum(challenger.values())
            changed+=int(baseline!=challenger)
            exactK_hits+=int(rows[i]["s80_K"]==len(truth[i]))
            bysong[rows[i]["member"]].append(int(now)-int(was))
        measured[key]=dict(n=len(indices),original_S80_K_correct=exactK_hits,
            baseline_exact_notes=int(old_hits),candidate_exact_notes=int(new_hits),
            corrections=int(fixes),regressions=int(regressions),
            net=int(new_hits-old_hits),changed_note_sets=int(changed),
            baseline_note_TP=int(old_tp),candidate_note_TP=int(new_tp),
            note_TP_delta=int(new_tp-old_tp),
            baseline_micro_note_F1=(2*old_tp/(true_notes+old_notes)
                    if true_notes+old_notes else None),
            candidate_micro_note_F1=(2*new_tp/(true_notes+new_notes)
                    if true_notes+new_notes else None),
            per_WAV_net={x:int(sum(v)) for x,v in sorted(bysong.items())})
    return measured

def run(s450,s450_report,s276,output):
    need(not output.exists(),"S451 immutable IDMT note identity audit exists")
    info=json.loads(s450_report.read_text())
    selected=json.loads(s276.read_text())
    need(info["status"]=="S450_ORIGINAL_S80_EXTERNAL_IDMT_506_NOTE_NAMES_FROZEN"
         and info["original_events"]==506 and info["recordings"]==5
         and info["no_XML_MIDI_truth_opened"]
         and info["original_S80_predictions_unchanged"]
         and info["exactK_regressions"]==0
         and info["checksums"]["prediction_npz"]==sha(s450),
         "S451 prospective S450 original model outputs not SHA-frozen")
    need(selected["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and selected["counts_WAV"]["dataset3"]==5
         and selected["XML_MIDI_annotation_contents_read"] is False,
         "S451 original five IDMT recording manifest invalid")
    with np.load(s450,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","member","start_samples",
                    "end_samples","s80_K","S430_pitch_slots","S450_pitch_slots"},
             "S451 frozen candidate contains unexpected true labels")
        arr={k:z[k].copy() for k in z.files}
    ids=arr["row_id"].astype(str);member=arr["member"].astype(str)
    start=arr["start_samples"].astype(np.int64);K=arr["s80_K"].astype(int)
    old=arr["S430_pitch_slots"];candidate=arr["S450_pitch_slots"]
    need(len(ids)==506 and len(set(ids))==len(ids)
         and np.array_equal((old>=0).sum(1),K)
         and np.array_equal((candidate>=0).sum(1),K),
         "S451 original S80 K from actual IDMT native rows changed")
    manifest=selected["candidate_groups"]["dataset3_first"]
    path={obj["wav_filename_sha256"]:obj for obj in manifest}
    need(len(path)==5 and set(member)==set(path),"S451 original external WAV group drift")
    # Only now open XML after verifying frozen source and manifest.
    with urllib.request.urlopen(urllib.request.Request(
         API,headers={"Accept":"application/json","User-Agent":"S451-S450-frozen-note-names"}),
         timeout=60) as response:
        meta=audit(json.load(response))
    reader=HttpRangeZIP(meta["source_url"],meta["size_bytes"])
    truth=[None]*len(ids);proof=[]
    with zipfile.ZipFile(reader) as z:
        need(sha_bytes("\n".join(sorted(z.namelist())).encode())==
             selected["zip_central_names_sha256"],
             "S451 original IDMT ZIP source identity not same as S276")
        for key,obj in sorted(path.items()):
            xml=obj["annotation_path_name_only"]
            need(xml in z.namelist() and xml.lower().endswith(".xml"),
                 "S451 exact predeclared note-level XML missing")
            content=z.read(xml)
            events=parse_original_pitch_events(content)
            ix=np.flatnonzero(member==key)
            ii=ix[np.argsort(start[ix],kind="stable")]
            notes=annotated_sets(events,start[ii]/SR)
            for pos,v in zip(ii,notes):truth[int(pos)]=v
            proof.append(dict(name_sha=key,XML_sha256=sha_bytes(content),
                total_XML_events=len(events),native_events=len(ii),
                onset_matched_notes=int(sum(x is not None for x in notes))))
    rows=[dict(member=str(m),s80_K=int(k),
         S430=[int(p) for p in a if p>=0],S450=[int(p) for p in b if p>=0])
          for m,k,a,b in zip(member,K,old,candidate)]
    matched=sum(t is not None for t in truth)
    eligible=sum(t is not None and all(MIN_MIDI<=p<=MAX_MIDI for p in t)
                 for t in truth)
    unsupported=sum(t is not None and not all(MIN_MIDI<=p<=MAX_MIDI for p in t)
                    for t in truth)
    metric=evaluate(rows,truth,"S450")
    need(matched>0 and eligible>0 and metric["all_matched"]["n"]==eligible
         and matched+sum(t is None for t in truth)==506,
         "S451 impossible IDMT true-note event population or matching")
    output.mkdir(parents=True)
    rep=dict(status="S451_ORIGINAL_S80_EXTERNAL_IDMT_TRUE_NOTE_IDENTITY_PAIRED",
        original_S80_K_predictions_unchanged=True,exactK_regressions=0,
        n_external_native_V88_events=506,
        total_original_IDMT_WAV=5,
        matched_annotation_onsets=matched,
        pitches_in_supported_MIDI40to88=eligible,
        pitches_outside_supported_range_skipped=unsupported,
        unknown_or_unmatched_events_excluded=506-matched,
        complete_negative_K0_labels_not_available=True,
        original_IDMT_S299_XML_labels_previously_exposed=True,
        not_new_blind_independent_composition_validation=True,
        same_audio_IDMT_distinct_GuitarSet_training_source=True,
        no_refit_no_threshold_tuning_on_IDMT=True,
        matched_onset_rule_S299_20ms_group_80ms_tolerance_30ms_sounding=True,
        no_external_other_instrument_claim=True,
        no_active_reference_change=True,experimental_no_promotion=True,
        scored=metric,proof=proof,
        checksums=dict(frozen_candidate_sha=sha(s450),original_S276_manifest_sha=sha(s276)))
    (output/"report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("S451 NOTE IDENTITY",json.dumps(dict(
        matched=matched,eligible=eligible,unknown=506-matched,
        poly=metric["true_poly_K2plus"],by_true_K={
            k:v for k,v in metric.items() if k.startswith("true_K")})),flush=True)

def sha_bytes(x):
    import hashlib
    return hashlib.sha256(x).hexdigest()

def selftest():
    xml=b'<doc><event><onsetSec>1.000</onsetSec><offsetSec>1.5</offsetSec><pitch>52</pitch></event><event><onsetSec>1.008</onsetSec><offsetSec>1.5</offsetSec><pitch>55</pitch></event><event><onsetSec>2.000</onsetSec><offsetSec>2.5</offsetSec><pitch>59</pitch></event></doc>'
    x=annotated_sets(parse_original_pitch_events(xml),[1.02,2.01,3.01])
    need(x==[[52,55],[59],None],"S451 original XML notes/onset matching ABI drift")
    rows=[dict(S430=[52,59],S450=[52,55],s80_K=2,member="guitar"),
          dict(S430=[59],S450=[59],s80_K=1,member="guitar")]
    d=evaluate(rows,[[52,55],[59]],"S450")
    need(d["all_matched"]["corrections"]==1
         and d["all_matched"]["regressions"]==0,
         "S451 paired complete-note correction broken")
    print("S451 SELFTEST PASS actual note identities, unmatched excluded")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for flag in ("s450","s450-report","s276","out"):
        p.add_argument("--"+flag,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for name in ("s450","s450_report","s276","out"):
            need(getattr(a,name)is not None,"S451 missing --"+name)
        run(a.s450,a.s450_report,a.s276,a.out)
