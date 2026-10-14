"""S304: exposed-IDMT XML evaluation with historical attack-count targets.

Training target in V9/V27: min(6, number of annotation onsets
assigned within 20ms to each original 40ms V88 candidate group).
Use original yourmt3_exactk_common.assign_onsets. Active/sustained
pitches are NOT the target. Five Dataset3 XML were already opened in
S299. This is retrospective, cannot promote S80 or prove K0 completeness.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import assign_onsets,CLUSTER_SAMPLES
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
SR=44100
NATIVE={'row_id','member','start_samples','end_samples','incumbent_K','incumbent_probabilities'}
EXPOSED={'row_id','member','start_samples','XML_true_K_or_minus1','matched_offset_ms',
         'original_official_freeze_K','original_S80_K','S274_twofold_full_S80_K'}
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(path):return sha(Path(path).read_bytes())
def read(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,'source frozen fields incorrect')
        return {k:z[k].copy() for k in z.files}
def ledger(y,ref,candidate,mask):
    idx=np.flatnonzero(mask)
    if len(idx)==0:
        return dict(events=0,baseline_correct=0,S80_correct=0,
                    baseline_ExactK=None,S80_ExactK=None,fixes=0,regressions=0)
    a=ref[idx]==y[idx]
    b=candidate[idx]==y[idx]
    fixes=int(np.count_nonzero(~a&b))
    regressions=int(np.count_nonzero(a&~b))
    need(fixes-regressions==int(b.sum()-a.sum()),'paired ledger drift')
    return dict(events=int(len(idx)),baseline_correct=int(a.sum()),
                S80_correct=int(b.sum()),baseline_ExactK=float(a.mean()),
                S80_ExactK=float(b.mean()),fixes=fixes,regressions=regressions)
def selftest():
    starts=np.array([1000,4000],np.int64)
    ends=np.array([1300,4200],np.int64)
    onsets=np.array([1010,1070,3999,4205,9000],np.int64)
    k,assignment=assign_onsets(onsets,starts,ends)
    need(k.tolist()==[2,2] and assignment.tolist()==[0,0,1,1,-1],
         'historical assigned attack target source mismatch')
    k,_=assign_onsets([1000],np.array([2000],np.int64),np.array([2200],np.int64))
    need(k.tolist()==[0],'sounding held note is not new attack')
    print('S304 original historical onset matching selftest PASS',flush=True)
def run(s276,s299report,s299pred,s277,s277report,out):
    need(not out.exists(),'immutable audit')
    selection=json.loads(s276.read_text())
    exposed_report=json.loads(s299report.read_text())
    parent=json.loads(s277report.read_text())
    need(selection['status']=='S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED'
         and exposed_report['status']=='S299_FROZEN_ORIGINAL_VS_STRICT_S80_IDMT_DATASET3_XML_MATCHED_NOTE_ONLY'
         and exposed_report['initial_S276_selection_SHA256']==digest(s276)
         and parent['status']=='S277_TRUE_OFFICIAL_FREEZE_ON_DATASET3_FIVE_WAV_UNLABELLED'
         and parent['sealed_original_predictions_SHA256']==digest(s277)
         and not parent['XML_MIDI_annotated_contents_read'],
         'source baseline/XML provenance changed')
    orig=read(s277,NATIVE)
    x=read(s299pred,EXPOSED)
    member=orig['member'].astype(str)
    starts=orig['start_samples'];ends=orig['end_samples']
    ids=orig['row_id'];official=orig['incumbent_K'];new=x['original_S80_K']
    need(len(ids)==506 and np.array_equal(ids,x['row_id'])
         and np.array_equal(member,x['member'])
         and np.array_equal(starts,x['start_samples'])
         and np.array_equal(official,x['original_official_freeze_K'])
         and np.array_equal(new,x['S274_twofold_full_S80_K']),
         'three SHA-locked native model prediction sources differ')
    selected=selection['candidate_groups']['dataset3_first']
    known_proofs={p['wav_filename_sha256']:p
                  for p in exposed_report['WAV_level_annotation_proofs']}
    with urllib.request.urlopen(urllib.request.Request(API,headers={
          'Accept':'application/json','User-Agent':'S304-exposed-historical-K0-onset-audit'}),
          timeout=60) as response:
        source=audit(json.load(response))
    stream=HttpRangeZIP(source['source_url'],source['size_bytes'])
    y=np.full(506,-1,np.int16)
    per_wav=[]
    with zipfile.ZipFile(stream,'r') as z:
        need(sha('\n'.join(sorted(z.namelist())).encode())==
             selection['zip_central_names_sha256'],'IDMT source ZIP central directory drift')
        for item in selected:
            h=item['wav_filename_sha256']
            rows=np.flatnonzero(member==h)
            need(len(rows)>0 and h in known_proofs,'missing original WAV or XML proof')
            ordered=rows[np.argsort(starts[rows],kind='stable')]
            first=starts[ordered].astype(np.int64)
            last=ends[ordered].astype(np.int64)
            need(np.all(first[1:]>last[:-1]) and
                 np.all(last>=first) and np.all((last-first)<=CLUSTER_SAMPLES),
                 'original V88 40ms groups unexpectedly overlap')
            xml=z.read(item['annotation_path_name_only'])
            need(sha(xml)==known_proofs[h]['XML_SHA256'],
                 'original already-opened S299 note annotation SHA changed')
            notes=parse_original_pitch_events(xml)
            onset=[int(round(entry[0]*SR)) for entry in notes]
            count,assigned=assign_onsets(onset,first,last)
            y[ordered]=np.minimum(6,count)
            per_wav.append(dict(WAV_filename_sha256=h,
               wav_file=Path(item['wav']).name,
               original_V88_event_count=len(ordered),
               XML_note_event_count=len(notes),
               XML_onsets_assigned=int(np.count_nonzero(assigned>=0)),
               XML_onsets_unassigned=int(np.count_nonzero(assigned<0)),
               groups_K0=int(np.count_nonzero(count==0)),
               groups_K1=int(np.count_nonzero(count==1)),
               groups_poly_K2plus=int(np.count_nonzero(count>=2))))
    need(np.isin(y,range(7)).all(),'not all 506 original V88 groups labelled by onset-rule')
    poly=y>=2
    full=ledger(y,official,new,np.ones(506,bool))
    poly_stats=ledger(y,official,new,poly)
    by_k={f'K{k}':ledger(y,official,new,y==k) for k in range(7)}
    report=dict(status='S304_EXPOSED_XML_TRAINING_ALIGNED_ONSET_K0K6_DIAGNOSTIC',
       cohort_events=506,recordings=5,
       target_definition='Number of new annotated note attack events assigned at <=20ms to each original frozen V88 <=40ms candidate group, capped at K6',
       matching_implementation='scripts.yourmt3_exactk_common.assign_onsets',
       original_S277_baseline_archive_SHA256=digest(s277),
       original_S299_label_archive_SHA256=digest(s299pred),
       original_s276_selection_SHA256=digest(s276),
       XML_derived_attacks_K0to6_histogram=np.bincount(y,minlength=7).tolist(),
       XML_derived_attacks_K0_count=int(np.count_nonzero(y==0)),
       XML_derived_attacks_poly_K2plus_count=int(poly.sum()),
       if_XML_transcriptions_complete_global_K0to6=full,
       if_XML_transcriptions_complete_true_poly_K2plus=poly_stats,
       if_XML_transcriptions_complete_by_K=by_k,
       per_WAV_onset_coverage=per_wav,
       source_original_K_semantics_verified=True,
       XML_all_note_transcription_completeness_independently_proven=False,
       true_negative_K0_exhaustiveness_independently_proven=False,
       corpus_already_exposed_to_S299=True,
       song_composition_disjointness_not_proven=True,
       weights_modified=False,model_fitted=False,
       official_K0toK6_exactk_measured=False,
       official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_WAV_onset_coverage','if_XML_transcriptions_complete_by_K')},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for key in ('s276','s299report','s299pred','s277','s277report','out'):
        p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,k) is not None for k in (
            's276','s299report','s299pred','s277','s277report','out')),
            'original frozen evidence required')
        selftest()
        run(a.s276,a.s299report,a.s299pred,a.s277,a.s277report,a.out)
