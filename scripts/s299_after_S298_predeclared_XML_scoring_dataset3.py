"""S299 predeclared first post-freeze IDMT Dataset3 true-note scoring.

Only AFTER independent native 506 rows and all three predictions are
SHA-sealed by S298 may we read exactly five paired IDMT Dataset3 XMLs.

Locked alignment from S190/S254: 20ms onset grouping, 80ms max event
association, +30ms sounding evaluation. Unmatched events NEVER K0.
Pathétique mono/poly are ONE composition. No tuning, no fitting, and
NO independent full Exact-K global score nor S80 promotion.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from scripts.s190_score_frozen_dataset2_XML_exactk import (
    parse_original_pitch_events,nearest_annotated_k)
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP

SR=44100
MATCH_SEC=.080
CLUSTER_SEC=.020
SOUND_SEC=.030
BOOT=4096
SEED=299
S298_STATUS="S298_TWO_COMPLETE_S80_S274_CANDIDATES_PRELABEL_IDENTICAL_506_V88_EVENTS"
NATIVE={"row_id","member","start_samples","end_samples","incumbent_K","incumbent_probabilities"}
PRED={"row_id","incumbent_freeze_K","s72_K","s79_K","s80_K",
      "original_PR16_policy22_K","K1_by_fold_threshold_action",
      "K1_by_fold_threshold_scores","action_K1","action_K0","action_K2"}
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):
    return hashlib.sha256(b).hexdigest()
def sha_file(p):
    return sha(Path(p).read_bytes())
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,"S299 only three frozen prediction sources")
        return {k:z[k].copy() for k in z.files}
def ledger(y,old,new,mask):
    ii=np.flatnonzero(mask)
    if not len(ii):
        return dict(n=0,baseline_correct=0,candidate_correct=0,
            baseline_exactk=None,candidate_exactk=None,delta=None,
            corrections=0,regressions=0,changed_both_wrong=0)
    t=y[ii];b=old[ii];c=new[ii]
    correctb=b==t;correctc=c==t
    fixes=int(np.count_nonzero(~correctb&correctc))
    lost=int(np.count_nonzero(correctb&~correctc))
    need(int(correctc.sum()-correctb.sum())==fixes-lost,
         "paired exact-K corrections/regressions balance drift")
    return dict(n=len(ii),baseline_correct=int(correctb.sum()),
        candidate_correct=int(correctc.sum()),
        baseline_exactk=float(correctb.mean()),candidate_exactk=float(correctc.mean()),
        delta=float(correctc.mean()-correctb.mean()),
        corrections=fixes,regressions=lost,
        changed_both_wrong=int(np.count_nonzero((b!=c)&~correctb&~correctc)))
def composition(wav):
    stem=Path(wav).stem.lower()
    if stem in ('pathetique_mono','pathetique_poly'):return 'pathetique'
    return stem
def bootstrap(y,old,new,composition_by_row,mask):
    eligible=np.flatnonzero(mask)
    if not len(eligible):return None
    groups=sorted(set(composition_by_row[eligible]))
    if len(groups)<2:return dict(compositions=len(groups),ci95=None,
        statistical_promotion_allowed=False)
    parts=[eligible[composition_by_row[eligible]==group] for group in groups]
    rng=np.random.default_rng(SEED)
    draws=[]
    for _ in range(BOOT):
        pick=rng.integers(0,len(parts),size=len(parts))
        idx=np.concatenate([parts[j] for j in pick])
        draws.append(float(np.mean((new[idx]==y[idx]).astype(float)-(old[idx]==y[idx]).astype(float))))
    ci=np.quantile(draws,[.025,.975]).tolist()
    return dict(compositions=len(groups),bootstrap_replicates=BOOT,seed=SEED,
       exploratory_composition_cluster_CI95=list(map(float,ci)),
       descriptive_only_not_promotion_evidence=True,
       statistical_promotion_allowed=False)

def selftest():
    y=np.array([2,3,1,2],np.int16);a=np.array([3,3,2,2])
    b=np.array([2,2,1,2])
    q=ledger(y,a,b,np.ones(4,bool))
    need(q['corrections']==2 and q['regressions']==1
         and q['candidate_correct']-q['baseline_correct']==1,
         "paired original note-only exactK scoring regression test failed")
    from scripts.s190_score_frozen_dataset2_XML_exactk import (
        parse_original_pitch_events,nearest_annotated_k)
    xml=b'<doc><event><onsetSec>1.0</onsetSec><offsetSec>1.4</offsetSec><pitch>60</pitch></event><event><onsetSec>1.008</onsetSec><offsetSec>1.4</offsetSec><pitch>64</pitch></event></doc>'
    notes=parse_original_pitch_events(xml)
    k,_,_=nearest_annotated_k(notes,np.array([1.01,3.]),MATCH_SEC)
    need(k.tolist()==[2,-1],"unmatched annotation must never be K0")
    need(composition('pathetique_mono.wav')==composition('pathetique_poly.wav'),
         "Pathétique recordings must form the same statistical group")
    print("S299 prelabel alignment/paired scoring test PASS")

def run(s276,s298,s277,oldfile,newfile,out):
    need(not out.exists(),"S299 XML verdict immutable")
    selftest()
    manifest=json.loads(s276.read_text())
    pre=json.loads(s298.read_text())
    need(pre['status']==S298_STATUS
         and pre['native_events']==506
         and pre['all_results_on_exact_same_original_V88_native_events']
         and pre['performance_ExactK_not_scored']
         and not pre['truth_XML_or_MIDI_read']
         and pre['original_baseline_freeze_SHA256']==sha_file(s277)
         and pre['original_S80_SHA256']==sha_file(oldfile)
         and pre['S274_twofold_S80_SHA256']==sha_file(newfile),
         "S298 full S80 postfreeze immutable hashes not verified before labels")
    need(not manifest['WAV_file_contents_read']
         and not manifest['XML_MIDI_annotation_contents_read']
         and manifest['counts_WAV']['dataset3']==5,
         "original prospective S276 dataset3 selection changed")
    selected=manifest['candidate_groups']['dataset3_first']
    need(len(selected)==5 and len(set(i['wav'] for i in selected))==5
         and all(i['annotation_pair_status'] in
                 ('exact_expected_subdirectory','unique_matching_stem') for i in selected),
         "all 5 exact prospective WAV/XML filename pairs required")
    baseline=load(s277,NATIVE)
    a=load(oldfile,PRED)
    b=load(newfile,PRED)
    ids=baseline['row_id'].astype(str);n=len(ids)
    member=baseline['member'].astype(str)
    starts=baseline['start_samples'].astype(np.int64)
    need(n==506 and np.array_equal(ids,a['row_id'])
         and np.array_equal(ids,b['row_id'])
         and np.array_equal(baseline['incumbent_K'],a['incumbent_freeze_K'])
         and np.array_equal(baseline['incumbent_K'],b['incumbent_freeze_K']),
         "original V88 native S80 source rows diverged before annotation")
    names_sha={i['wav_filename_sha256']:i for i in selected}
    need(set(member).issubset(set(names_sha)) and len(set(member))==5,
         "different recordings/labels from prospective five-WAV cohort")
    # Only now, after ALL source model, IDs, SHA, selection and selftests
    # have passed, access exactly five original XML members in pinned ZIP.
    with urllib.request.urlopen(urllib.request.Request(
       API,headers={'Accept':'application/json','User-Agent':'S299-locked-post-S298-note-only'}),timeout=60) as response:
        source=audit(json.load(response))
    stream=HttpRangeZIP(source['source_url'],source['size_bytes'])
    truth=np.full(n,-1,np.int16)
    offset=np.full(n,np.nan,np.float32)
    comp=np.empty(n,dtype='<U64')
    proofs=[]
    with zipfile.ZipFile(stream) as z:
        need(sha('\n'.join(sorted(z.namelist())).encode())==
             manifest['zip_central_names_sha256'],
             "published IDMT ZIP central directory identity not equal prelabel S276")
        existing=set(z.namelist())
        for item in selected:
            wav=item['wav'];wavsha=item['wav_filename_sha256']
            xml=item['annotation_path_name_only']
            need(xml in existing and xml.lower().endswith('.xml')
                 and '/dataset3/' in ('/'+xml.lower()),
                 'missing prospective paired dataset3 original XML file')
            idx=np.flatnonzero(member==wavsha)
            need(len(idx)>0,'one pretrained source WAV absent from native events')
            record=z.read(xml)  # FIRST AND ONLY OPEN of annotation content.
            notes=parse_original_pitch_events(record)
            ordered=idx[np.argsort(starts[idx],kind='stable')]
            assigned,deltas,group_count=nearest_annotated_k(notes,
                     starts[ordered]/SR,MATCH_SEC)
            truth[ordered]=assigned;offset[ordered]=deltas
            comp[idx]=composition(wav)
            proofs.append(dict(wav_filename_sha256=wavsha,
                composition=composition(wav),
                XML_SHA256=sha(record),
                true_note_event_count=len(notes),
                annotation_onset_groups=group_count,
                frozen_V88_events=len(idx),
                matched_K1to6_events=int(np.count_nonzero(
                    (assigned>=1)&(assigned<=6))),
                unmatched_or_unsupported_events=int(np.count_nonzero(
                    (assigned<1)|(assigned>6)))))
    valid=(truth>=1)&(truth<=6)
    poly=valid&(truth>=2)
    ref=baseline['incumbent_K']
    original=a['s80_K']
    stricter=b['s80_K']
    pairs={'S80_original_vs_official_freeze':(ref,original),
           'S80_S274_twofold_vs_official_freeze':(ref,stricter),
           'S80_S274_twofold_vs_original_S80':(original,stricter)}
    result={}
    for label,(old,new) in pairs.items():
        result[label]=dict(
          matched_K1to6=ledger(truth,old,new,valid),
          matched_true_poly_K2to6=ledger(truth,old,new,poly),
          matched_per_K={f"K{k}":ledger(truth,old,new,valid&(truth==k))
                         for k in range(1,7)},
          bootstrap_K1to6=bootstrap(truth,old,new,comp,valid),
          bootstrap_poly_K2to6=bootstrap(truth,old,new,comp,poly),
          changed_events_total=int(np.count_nonzero(old!=new)),
          changed_events_XML_matched=int(np.count_nonzero((old!=new)&valid)),
          changed_events_unmatched=int(np.count_nonzero((old!=new)&~valid)))
    eligibility=dict(
        exact_native_original_reference_and_two_complete_S80_506_frozen=True,
        training_weights_unchanged=True,
        original_XML_annotations_unread_before_prelabel_freeze=True,
        certified_composition_disjoint_from_all_training=False,
        independent_exactk_global_K0to6_available=False,
        complete_negative_K0_annotations=False,
        distinct_compositions_in_cohort=len(set(comp)),
        independent_promotion_evidence_sufficient=False,
        official_S80_promotion_allowed=False)
    report=dict(status="S299_FROZEN_ORIGINAL_VS_STRICT_S80_IDMT_DATASET3_XML_MATCHED_NOTE_ONLY",
       event_population=n,
       selected_WAVs=5,
       dataset3_unique_named_composition_families=len(set(comp)),
       prospective_source_prelabel_S298_SHA256=sha_file(s298),
       initial_S276_selection_SHA256=sha_file(s276),
       note_onset_group_span_seconds=CLUSTER_SEC,
       event_onset_match_max_seconds=MATCH_SEC,
       sounding_pitch_offset_seconds=SOUND_SEC,
       unmatched_event_K0_assignment_forbidden=True,
       matched_true_K1to6_events=int(valid.sum()),
       matched_true_poly_K2to6_events=int(poly.sum()),
       unmatched_or_unknown_label_events=int((~valid).sum()),
       original_paired_comparisons=result,
       WAV_level_annotation_proofs=proofs,
       promotion_gates=eligibility,
       full_Global_ExactK_K0to6_measured=False,
       S80_promoted=False)
    out.mkdir(parents=True)
    np.savez_compressed(out/'S299_XML_note_matched_only_506.npz',
       row_id=ids,member=member,start_samples=starts,
       XML_true_K_or_minus1=truth,
       matched_offset_ms=offset*1000,
       original_official_freeze_K=ref,
       original_S80_K=original,
       S274_twofold_full_S80_K=stricter)
    (out/'report.json').write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items()
        if k not in ('WAV_level_annotation_proofs','original_paired_comparisons')},
        indent=2),flush=True)
    for k,v in result.items():
        print(json.dumps({k:dict(matched_K1to6=v['matched_K1to6'],
             matched_true_poly_K2to6=v['matched_true_poly_K2to6'],
             changed_events_total=v['changed_events_total'])}),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    for key in ('s276','s298','s277','old','new','out'):
        ap.add_argument('--'+key,type=Path)
    z=ap.parse_args()
    if z.self_test:selftest()
    else:
        need(all(getattr(z,k) is not None for k in
            ('s276','s298','s277','old','new','out')),'all S298 SHA sources required')
        run(z.s276,z.s298,z.s277,z.old,z.new,z.out)
