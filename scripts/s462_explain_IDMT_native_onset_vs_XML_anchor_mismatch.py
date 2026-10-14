"""S462: forensic onset-candidate audit of IDMT domain transfer and XML matching.

Source: genuine original S277 V88 506 native events and immutable S297
original S80; 5 already exposed IDMT Dataset3 XML annotations.
Report *independent* one-to-one maximum-cardinality onset correspondence,
precommitted S451 greedy matches, timing shifts, duplicate/extra candidates,
per-WAV annotation coverage. This is DIAGNOSTIC, NOT a new scored benchmark.
Never classify unmatched candidate as true K0 or tune S80 thresholds.
"""
from __future__ import annotations
import argparse,hashlib,json,collections,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s451_score_frozen_S450_IDMT_XML_true_notes import annotated_sets

SR=44100
CLIP=10.0
WINDOW=.020
SOUND=.030
CUTS=(.02,.04,.08,.12,.20,.35)
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def onset_groups(events):
    """Exact S451 20ms grouping; retain full onset group schedule as reference."""
    groups=[]
    for on,off,p in events:
        if not groups or float(on)-groups[-1][0]>WINDOW:
            groups.append((float(on),[]))
        groups[-1][1].append((float(on),float(off),int(p)))
    return groups

def associate_best(native_times,anchors,tol):
    """Maximum cardinality AND minimum total lag, not greedy opportunism.
    Diagnostic-only across native candidates and annotation groups,
    no K/probabilities and no model weights.
    """
    a=np.asarray(native_times,float)
    b=np.asarray(anchors,float)
    n=len(a);m=len(b)
    if not n or not m:return []
    distances=np.abs(a[:,None]-b[None,:])
    # Cost <1 for admissible pair, exactly 1 for dummy (unmatched native),
    # >1 for forbidden pair. All 1-to-1 assignments maximally cover onsets.
    real=np.where(distances<=tol,distances/(tol+1e-9)*.25,1e5)
    matrix=np.concatenate((real,np.ones((n,n))),axis=1)
    rows,cols=linear_sum_assignment(matrix)
    pairs=[(int(i),int(j),float(a[i]-b[j]))
           for i,j in zip(rows,cols) if j<m and distances[i,j]<=tol]
    need(len({i for i,_,_ in pairs})==len(pairs) and
         len({j for _,j,_ in pairs})==len(pairs),
         "S462 optimum one-to-one onset matching is not a valid matching")
    return sorted(pairs)

def descriptive(t,p):
    if not len(t):return dict(n=0)
    d=np.asarray(t,float)-np.asarray(p,float)
    return dict(n=int(len(d)),signed_median_ms=float(np.median(d)*1000),
       signed_mean_ms=float(np.mean(d)*1000),
       abs_median_ms=float(np.median(np.abs(d))*1000),
       abs_p90_ms=float(np.percentile(np.abs(d),90)*1000),
       negative_lag_n=int(np.sum(d<0)),positive_lag_n=int(np.sum(d>0)))

def run(s276,s277,s277_report,s297,s451_report,out):
    need(not out.exists(),"S462 must never overwrite retrospective forensic result")
    manifest=json.loads(s276.read_text())
    old=json.loads(s451_report.read_text())
    source_report=json.loads(s277_report.read_text())
    need(manifest["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and manifest["counts_WAV"]["dataset3"]==5
         and not manifest["XML_MIDI_annotation_contents_read"]
         and old["status"]=="S451_ORIGINAL_S80_EXTERNAL_IDMT_TRUE_NOTE_IDENTITY_PAIRED"
         and old["matched_annotation_onsets"]==47
         and old["pitches_in_supported_MIDI40to88"]==45
         and old["unknown_or_unmatched_events_excluded"]==459
         and old["no_refit_no_threshold_tuning_on_IDMT"]
         and source_report["sealed_original_predictions_SHA256"]==sha(s277)
         and source_report["original_native_event_count"]==506,
         "S462 original source archives or frozen annotation protocol changed")
    with np.load(s277,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        member=z["member"].astype(str)
        starts=z["start_samples"].astype(np.int64)
        K0=z["incumbent_K"].astype(np.int8)
    with np.load(s297,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str)),"S462 original S80 row alignment")
        original=z["s80_K"].astype(np.int8)
    need(len(ids)==506 and len(set(ids))==506
         and np.array_equal(ids,np.asarray([m+":"+str(int(s)) for m,s in zip(member,starts)]))
         and np.isin(original,range(7)).all(),
         "S462 original V88/S80 event counts or source identities invalid")
    group=manifest["candidate_groups"]["dataset3_first"]
    byname={o["wav_filename_sha256"]:o for o in group}
    need(len(byname)==5 and set(byname)==set(member),
         "S462 preregistered five recording identities drifted")
    meta_request=urllib.request.Request(API,headers={
        "Accept":"application/json","User-Agent":"S462-IDMT-annotation-onset-forensic-no-training"})
    with urllib.request.urlopen(meta_request,timeout=60) as response:
        meta=audit(json.load(response))
    reader=HttpRangeZIP(meta["source_url"],meta["size_bytes"])
    reports=[]
    all_rows=[]
    sums=collections.Counter()
    all_lags=[]
    hist_original=np.zeros(7,dtype=np.int64)
    hist_matched=np.zeros(7,dtype=np.int64)
    with zipfile.ZipFile(reader) as z:
        need(sha_bytes("\n".join(sorted(z.namelist())).encode())==
             manifest["zip_central_names_sha256"],
             "S462 original Zenodo XML annotation ZIP source changed")
        for wid,rec in sorted(byname.items()):
            xmlname=rec["annotation_path_name_only"]
            need(xmlname in z.namelist() and xmlname.endswith(".xml"),
                 "S462 original corresponding XML missing")
            xml=z.read(xmlname)
            events=parse_original_pitch_events(xml)
            groups=onset_groups(events)
            groupsclip=[g for g in groups if 0<=g[0]<CLIP]
            onsets=np.asarray([g[0] for g in groupsclip],float)
            indices=np.flatnonzero(member==wid)
            indices=indices[np.argsort(starts[indices],kind="stable")]
            t=starts[indices]/SR
            need(np.all((0<=t)&(t<CLIP+.1)),"S462 native original onset beyond S277 ten-second excerpt")
            # Frozen published XML rule (native events sorted by source time).
            original_match=annotated_sets(events,t)
            original_index=np.asarray([j for j,x in enumerate(original_match) if x is not None],int)
            valid=np.asarray([j for j,x in enumerate(original_match)
                              if x is not None and all(40<=p<=88 for p in x)],int)
            ktrue=np.asarray([len(original_match[j]) for j in valid],int)
            original_hits=int(np.sum(original[indices[valid]]==ktrue))
            matched_k0=int(np.sum(original[indices[valid]]==0))
            n0=int(np.sum(original[indices]==0))
            hist_original+=np.bincount(original[indices],minlength=7)
            hist_matched+=np.bincount(original[indices[valid]],minlength=7)
            comparison={}
            for cut in CUTS:
                best=associate_best(t,onsets,cut)
                comparison[str(round(cut*1000))]=dict(
                    unique_matched=int(len(best)),
                    XML_onsets_recalled_pct=(100*len(best)/len(onsets) if len(onsets) else None),
                    native_candidates_supported_pct=100*len(best)/len(t),
                    nearest_signed_lag=descriptive(
                        [t[i] for i,j,d in best],
                        [onsets[j] for i,j,d in best]))
            pair80=associate_best(t,onsets,.08)
            near=np.abs(t[:,None]-onsets[None,:]) if len(onsets) else np.full((len(t),0),np.inf)
            nearest=np.min(near,axis=1) if len(onsets) else np.full(len(t),np.inf)
            nearest_group=np.argmin(near,axis=1) if len(onsets) else np.full(len(t),-1)
            observed = set(original_index.tolist())
            original_greedy_near=[min((abs(t[j]-anchor),anchor)
                                     for anchor in onsets) if len(onsets) else (float("inf"),float("nan"))
                                  for j in original_index]
            lags= [float(t[j]-anchor) for j,(_,anchor) in zip(original_index,original_greedy_near)]
            all_lags.extend(lags)
            # The above is nearest *all* XML group anchors, not necessarily
            # S451's greedy assigned group; report it as nearest, not signed match.
            within80=(nearest<=.08)
            near80_K0=int(np.sum((original[indices]==0)&within80))
            far80_K0=int(np.sum((original[indices]==0)&~within80))
            spacing=np.diff(t)
            row=dict(recording_filename_SHA256=wid,
                source_XML_SHA256=sha_bytes(xml),
                acoustic_native_events=int(len(t)),
                mean_native_events_per_second=float(len(t)/CLIP),
                native_predicted_original_S80_K0=n0,
                native_original_S80_K_hist=np.bincount(original[indices],minlength=7).tolist(),
                XML_original_note_events=int(len(events)),
                XML_20ms_group_anchors_all=int(len(groups)),
                XML_20ms_group_anchors_first_10seconds=int(len(groupsclip)),
                S451_exact_used_once_80ms_matched_events=int(len(original_index)),
                S451_eligible_MIDI40_88=int(len(valid)),
                S451_eligible_K_correct=original_hits,
                S451_eligible_predicted_K0=matched_k0,
                independent_optimal_one_to_one_onsets_by_ms=comparison,
                native_candidates_near_any_XML_anchor_80ms=int(within80.sum()),
                native_candidates_far_from_any_XML_anchor_80ms=int((~within80).sum()),
                native_K0_near_any_XML_anchor_80ms=near80_K0,
                native_K0_far_from_any_XML_anchor_80ms=far80_K0,
                median_spacing_between_native_onsets_ms=(float(np.median(spacing)*1000)
                    if len(spacing) else None),
                native_adjacent_spacings_20ms=int(np.sum(spacing<=.02)),
                native_adjacent_spacings_80ms=int(np.sum(spacing<=.08)),
                native_adjacent_spacings_200ms=int(np.sum(spacing<=.2)),
                nearest_annotation_lag_on_greedy_matched_rows=descriptive(
                    [t[j] for j in original_index],
                    [anchor for _,anchor in original_greedy_near]),
                diagnostic_best_matching_changes_SCORING=False)
            reports.append(row)
            sums.update(native=len(t),xml_anchors=len(groupsclip),
                s451_matched=len(original_index),s451_valid=len(valid),
                s451_correct=original_hits,s451_predK0=matched_k0,
                near80=int(within80.sum()),far80=int((~within80).sum()),
                near80K0=near80_K0,far80K0=far80_K0,
                native_spacing20=int(np.sum(spacing<=.02)))
            for j,rowid in enumerate(indices):
                all_rows.append(dict(row_id=str(ids[rowid]),source_wav_SHA256=wid,
                    native_time_seconds=float(t[j]),
                    original_S80_K=int(original[rowid]),
                    nearest_XML_group_ms=(float(nearest[j]*1000)
                        if np.isfinite(nearest[j]) else None),
                    original_S451_matched=bool(j in observed),
                    original_S451_matched_true_K=(
                        len(original_match[j]) if original_match[j] is not None else None)))
    need(sums["native"]==506 and sums["s451_matched"]==47
         and sums["s451_valid"]==45 and sums["s451_correct"]==25
         and len(reports)==5 and int(hist_original[0])==456,
         "S462 must reproduce all immutable S451/S297 population counts")
    out.mkdir(parents=True)
    ledger=out/"S462_original_onset_candidate_nearest_XML_506.jsonl"
    ledger.write_text("".join(json.dumps(x)+"\n" for x in all_rows))
    result=dict(status="S462_IDMT_EXTERNAL_ORIGINAL_ONSET_XML_MATCHING_FORENSIC",
        question="Is IDMT transfer loss partly event detection or annotation matching?",
        recordings=5,original_native_event_count=506,
        original_s451_80ms_matched_count=47,
        original_s451_inrange_count=45,
        original_s451_inrange_S80_K_correct=25,
        original_predicted_S80_K0_count=456,
        original_XML_onset_group_count_first_10seconds=int(sums["xml_anchors"]),
        original_S451_matched_fraction_native=47/506,
        source_K0_held_untouched=True,
        unpaired_native_events_NOT_construed_as_true_K0=True,
        no_audio_model_fitting_or_predictor_replay=True,
        XML_previously_exposed_in_S299=True,
        sensitivity_checks_are_not_benchmark_selection=True,
        all_IDMT_sources_pinned_SHA256=dict(manifest=sha(s276),
            native_V88=sha(s277),original_S80=sha(s297),
            baseline_S451=sha(s451_report),nearest_source_ledger=sha(ledger)),
        native_K0_histogram_over_entire_source=hist_original.tolist(),
        native_matched_true_label_predK_histogram=hist_matched.tolist(),
        total_per_recording_aggregates=dict(sums),
        per_recording=reports,
        no_promotion=True)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("S462 ONSET_FORENSIC",json.dumps(dict(
        original_native=506,original_XML_groups=sums["xml_anchors"],
        original_S451_matched=47,
        best_matching_80ms=sum(r["independent_optimal_one_to_one_onsets_by_ms"]["80"]["unique_matched"]
                              for r in reports),
        matched_K_correct=25,
        native_K0=456,
        near80=sums["near80"],far80=sums["far80"],
        native_K0_near80=sums["near80K0"],native_K0_far80=sums["far80K0"],
        eachWAV=[dict(events=r["acoustic_native_events"],
            XML_groups=r["XML_20ms_group_anchors_first_10seconds"],
            S451_matched=r["S451_exact_used_once_80ms_matched_events"],
            optimal_80ms=r["independent_optimal_one_to_one_onsets_by_ms"]["80"]["unique_matched"],
            native_K0=r["native_predicted_original_S80_K0"]) for r in reports])),flush=True)

def selftest():
    a=np.array([.02,.03,.5])
    b=np.array([0,.04,.55])
    m=associate_best(a,b,.06)
    need(len(m)==3 and len({j for i,j,l in m})==3,"S462 optimal matching unique")
    # Greedy candidate-ordered can miss an onset: optimize cardinality.
    x=associate_best([.05,.085],[.02,.06],.04)
    need(len(x)==2,"S462 avoid greedily swallowing more flexible XML group")
    gs=onset_groups([(1.,1.2,50),(1.007,1.3,55),(1.1,1.4,59)])
    need(len(gs)==2,"S462 reproduce S451 20ms grouping")
    print("S462 SELFTEST PASS XML onset grouping and max one-to-one matching")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s276","s277","s277-report","s297","s451-report","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s276","s277","s277_report","s297","s451_report","out"):
            need(getattr(a,k)is not None,"S462 missing --"+k)
        run(a.s276,a.s277,a.s277_report,a.s297,a.s451_report,a.out)
