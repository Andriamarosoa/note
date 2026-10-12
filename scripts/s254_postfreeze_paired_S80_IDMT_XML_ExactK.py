"""S254 post-frozen paired S80 versus OFFICIAL freeze_local_combo on untouched IDMT.

Labels XML opened ONLY after permanent S252 SHA and S253 independently
replayed exact 1034 predictions. Same original S190 note-level onset
matching (20ms group / 80ms event association, sounding pitches after
30ms, one annotation anchor per source event), fixed in a prelabel commit.
No retraining, no adaptation after opening XML. K0 is explicitly NOT
labelled by an unmatched onset: global K0..6 unmeasurable by this rule.
"""
from __future__ import annotations
import argparse,hashlib,json,re,urllib.request,zipfile
from collections import defaultdict
from pathlib import Path,PurePosixPath
import numpy as np
from scripts.s190_score_frozen_dataset2_XML_exactk import (
    parse_original_pitch_events,nearest_annotated_k)
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s231_IDMT_rank33to48_seal_audio_only import select_wavs
LOCKED_S80_SHA='5689b5098a6462acc343d4ce11e8da536f0757848908fdf1300187727edc0106'
LOCKED_BASELINE_SHA='6f18a476a865e0791e17c7e826ca630042b206be463f2cc5b500108bf8c626da'
MATCH_SEC=.080;CLUSTER_SEC=.020;SOUND_OFFSET_SEC=.030;BOOT=4096;SEED=254
def need(v,msg):
    if not v:raise ValueError(msg)
def hashbuf(b):return hashlib.sha256(b).hexdigest()
def open_exact(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(fields),'immutable source fields mismatch '+str(p)+': '+str(z.files))
        return {k:z[k].copy() for k in z.files}
def measure(y,old,new):
    need(len(y)==len(old)==len(new) and
         np.isin(y,np.arange(1,7)).all() and
         np.isin(old,range(7)).all() and np.isin(new,range(7)).all(),
         'paired ordinal K0..6 / matched nonzero XML note label population')
    if not len(y):
        return dict(n=0,reference_correct=0,S80_correct=0,
                    reference_exactK=None,S80_exactK=None,delta=None,
                    fixes=0,regressions=0,changed_neither=0,unchanged=0)
    oldok=old==y;newok=new==y
    changed=old!=new
    fixes=int(np.count_nonzero(~oldok&newok))
    regressions=int(np.count_nonzero(oldok&~newok))
    neither=int(np.count_nonzero(changed&(~oldok)&(~newok)))
    ref=int(oldok.sum());candidate=int(newok.sum())
    need(candidate-ref==fixes-regressions,'paired correction ledger identity broken')
    return dict(n=len(y),reference_correct=ref,S80_correct=candidate,
        reference_exactK=float(ref/len(y)),S80_exactK=float(candidate/len(y)),
        delta=float((candidate-ref)/len(y)),fixes=fixes,
        regressions=regressions,changed_neither=neither,
        unchanged=int(np.count_nonzero(~changed)))
def bootstrap(y,old,new,members,mask):
    active=np.flatnonzero(mask)
    if not len(active):return None
    song_index=np.asarray(members).astype(str)
    tracks=sorted(set(song_index[active]))
    if len(tracks)<2:return dict(unique_WAV_count=len(tracks),
          ci95=None,positive_lower_bound=False)
    grouped=[active[song_index[active]==song] for song in tracks]
    rng=np.random.default_rng(SEED)
    delta=[]
    for _ in range(BOOT):
        pick=rng.integers(0,len(grouped),size=len(grouped))
        rows=np.concatenate([grouped[k] for k in pick])
        delta.append(float(np.mean((new[rows]==y[rows]).astype(float)-
                                   (old[rows]==y[rows]).astype(float))))
    lo,hi=np.quantile(delta,[.025,.975])
    return dict(unique_WAV_count=len(tracks),bootstrap_replicates=BOOT,
        seed=SEED,ci95=[float(lo),float(hi)],
        positive_lower_bound=bool(lo>0))
def selftest():
    y=np.array([2,3,1]);old=np.array([3,3,2]);new=np.array([2,2,1])
    z=measure(y,old,new)
    need(z['fixes']==2 and z['regressions']==1 and
         z['delta']==1/3,'paired correction/regression accounting')
    ex=b'<doc><event><onsetSec>1.0</onsetSec><offsetSec>1.4</offsetSec><pitch>60</pitch></event><event><onsetSec>1.008</onsetSec><offsetSec>1.4</offsetSec><pitch>64</pitch></event></doc>'
    notes=parse_original_pitch_events(ex)
    out,_,total=nearest_annotated_k(notes,np.array([1.01,3.]))
    need(total==1 and out.tolist()==[2,-1],
         'original S190 80ms XML pitch-note association must be unchanged')
    print('S254 PASS original frozen S190 XML parser, unmatched not K0, paired ledger and bootstrap prelabel contract')
def run(S252,S233,S253,S231,out):
    need(not out.exists(),'never overwrite the first post-frozen XML verdict')
    selftest()
    # CRITICAL: everything below until the explicit ZIP annotation read
    # is source-only verification against earlier CI artifact hashes.
    need(hashbuf(S252.read_bytes())==LOCKED_S80_SHA,
         'S252 true original-model K results diverged from PRE-XML immutable SHA!')
    need(hashbuf(S233.read_bytes())==LOCKED_BASELINE_SHA,
         'original official freeze_local_combo baseline diverged from PRE-XML SHA!')
    audit_report=json.loads(S253.read_text())
    need(audit_report['event_count']==1034 and
         all(audit_report['independent_formula_checks'].values()) and
         not audit_report['truth_or_MIDI_read'] and
         not audit_report['official_S80_promoted'],
         'independent S253 prelabel S80 arithmetic replay has not succeeded')
    inputs=json.loads(S231.read_text())
    need(inputs['chosen_song_count']==16 and
         inputs['this_cohort_SHA256_filename_rank_start']==33 and
         inputs['this_cohort_SHA256_filename_rank_end']==48 and
         not inputs['XML_note_annotation_content_read'] and
         not inputs['MIDI_content_read'],
         'source WAV selection not predeclared ranks33-48')
    s80=open_exact(S252,{'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
         'original_PR16_policy22_K','K1_by_fold_threshold_action',
         'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'})
    baseline=open_exact(S233,{'row_id','member','start_samples','incumbent_K',
         'incumbent_probabilities','nearest_S232_V84_onset_distance_samples'})
    ids=s80['row_id'].astype(str);member=baseline['member'].astype(str)
    start=baseline['start_samples'].astype(np.int64)
    old=baseline['incumbent_K'].astype(np.int8)
    new=s80['s80_K'].astype(np.int8)
    n=len(ids)
    need(n==1034 and np.array_equal(ids,baseline['row_id']) and
         np.array_equal(old,s80['incumbent_freeze_K']) and
         len(set(ids))==1034 and
         all(i==m+':'+str(int(t)) for i,m,t in zip(ids,member,start)),
         'S252 S233 original V88 native matched-event reference disagreement')
    # Lock the event matching protocol and source hashes in report BEFORE XML.
    pre=dict(S252_SHA256=LOCKED_S80_SHA,S233_SHA256=LOCKED_BASELINE_SHA,
        sealed_original_IDMT_WAV_ranks='SHA256 filename 33-48',
        annotation_onset_matching_max_s=MATCH_SEC,
        XML_group_onset_span_s=CLUSTER_SEC,
        note_count_sounding_offset_s=SOUND_OFFSET_SEC,
        true_K0_unmatched_event_assignment_prohibited=True,
        S253_independent_replay_confirmed=True,
        prior_32_WAV_XML_read=True,
        selected_16_new_WAV_XML_content_not_yet_read=True)
    with urllib.request.urlopen(urllib.request.Request(API,
          headers={'Accept':'application/json','User-Agent':'S254-only-after-S252-S253-precommit'}),
          timeout=50) as resp:
        info=audit(json.load(resp))
    z=HttpRangeZIP(info['source_url'],info['size_bytes'])
    truth=np.full(n,-1,np.int16)
    offsets=np.full(n,np.nan,np.float32)
    proofs=[]
    selected=inputs['chosen_track_provenance']
    need(len(selected)==16,'expected sixteen original WAV SHA256 ID')
    names=[i['original_archive_filename'] for i in selected]
    sha_names=[hashbuf(i.encode()) for i in names]
    need([i['WAV_name_SHA256'] for i in selected]==sha_names,
         'original filenames SHAs from S231 changed before XML')
    with zipfile.ZipFile(z,'r') as source:
        allwav=sorted((s for s in source.namelist() if '/dataset2/' in ('/'+s.lower())
            and s.lower().endswith('.wav')),key=lambda x:hashbuf(x.encode()))
        need(allwav[32:48]==names and not set(names).intersection(allwav[:32]),
             'selected WAV tracks leak the 32 old previously labelled XML songs')
        original_xml_names=set(source.namelist())
        # AFTER all prelabel SHA checks above, read ONLY the 16 paired XML.
        for entry in selected:
            wav=entry['original_archive_filename']
            wavsha=entry['WAV_name_SHA256']
            pos=np.flatnonzero(member==wavsha)
            need(len(pos)>0,'original frozen 1034 native event index lacks WAV '+wavsha)
            location=wav.replace('/dataset2/audio/','/dataset2/annotation/').rsplit('.',1)[0]+'.xml'
            need(location in original_xml_names,
                 'paired pitch-level XML absent for original WAV '+wav)
            xml=source.read(location)
            ann=parse_original_pitch_events(xml)
            order=pos[np.argsort(start[pos],kind='stable')]
            labels,delta,ngroups=nearest_annotated_k(
                ann,start[order]/44100.,MATCH_SEC)
            truth[order]=labels;offsets[order]=delta
            proofs.append(dict(wav_name_SHA256=wavsha,
                XML_SHA256=hashbuf(xml),XML_parsed_note_events=len(ann),
                XML_onset_groups=ngroups,
                original_official_native_event_count=len(pos),
                matched_events=int(np.count_nonzero(labels>=0)),
                unmatched_events=int(np.count_nonzero(labels<0)),
                matching_sample_timing_accuracy_ms_max=(
                  float(np.nanmax(delta)*1e3) if np.any(labels>=0) else None)))
            print(json.dumps(proofs[-1]),flush=True)
    # No candidate tuning/changes or model fitting after loading XML.
    valid=(truth>=1)&(truth<=6)
    poly=valid&(truth>=2)
    global_data=measure(truth[valid],old[valid],new[valid])
    poly_data=measure(truth[poly],old[poly],new[poly])
    perK={f'K{k}':measure(truth[valid&(truth==k)],
                  old[valid&(truth==k)],new[valid&(truth==k)]) for k in range(1,7)}
    overall_delta=bootstrap(truth,old,new,member,valid)
    poly_delta=bootstrap(truth,old,new,member,poly)
    changed=(old!=new)
    changed_matched=int(np.count_nonzero(changed&valid))
    changes_unmatched=int(np.count_nonzero(changed&(~valid)))
    # Official global K0..6 is not identifiable here. Likewise
    # SHA-distinct WAV does not prove artist/tune disjointness.
    eligibility=dict(
       complete_original_incumbent_and_S80_1034_K_predictions_sealed=True,
       independent_S253_S80_exact_replay_verified=True,
       complete_labelled_K0_through_K6=False,
       WAV_file_disjoint_from_previous_32=True,
       music_composition_song_disjoint_verified=False,
       global_matched_K1to6_CI_positive=bool(overall_delta and
                                  overall_delta['positive_lower_bound']),
       poly_matched_K2plus_CI_positive=bool(poly_delta and
                                  poly_delta['positive_lower_bound']),
       exactK_global_K0to6_promotion_metric_measured=False,
       official_S80_promotion_eligible=False)
    out.mkdir(parents=True)
    np.savez_compressed(out/'S254_frozen_IDMT_paired_XML_matching_only.npz',
        row_id=ids,member=member,start_samples=start,
        XML_true_K_or_minus1=truth,matched_offset_ms=offsets*1000,
        frozen_official_freeze_K=old,frozen_full_S80_K=new)
    result=dict(status='S254_POSTFROZEN_PAIRED_ORIGINAL_S80_VS_FREEZE_IDMT_XML',
       source_prediction_SHA256=pre,original_native_events=n,
       selected_WAV_count=len(selected),
       original_total_XML_onset_groups=sum(p['XML_onset_groups'] for p in proofs),
       XML_matched_events=int(valid.sum()),
       XML_unmatched_events=int((~valid).sum()),
       native_matching_coverage=float(valid.mean()),
       original_S80_change_event_count=int(changed.sum()),
       original_S80_changed_and_XML_matched=changed_matched,
       original_S80_changed_and_XML_unmatched=changes_unmatched,
       metric_global_matched_onset_K1_K6_only=global_data,
       metric_matched_true_poly_K2_K6_only=poly_data,
       metric_true_K1to6_by_class=perK,
       song_resampling_matched_K1to6=overall_delta,
       song_resampling_matched_poly=poly_delta,
       original_XML_per_WAV_proofs=proofs,
       promotion_gates=eligibility,
       limitation='No validated labelled K0 silent-event population; no proof of tune/composition disjointness. Report only matched onset K1–K6, never full global K0–K6 ExactK.',
       historical_original_S80_piece_route_equivalence_proven=False,
       model_fits_after_XML_opened=False,
       XML_opened_only_after_SHA_S252_S233_and_S253_verified=True,
       independent_ExactK_full_K0_K6_proven=False,
       S80_promoted=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
        if k not in ('original_XML_per_WAV_proofs','metric_true_K1to6_by_class')},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for k in ('s252','s233','s253','s231','out'):
        p.add_argument('--'+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,k) is not None for k in
                ('s252','s233','s253','s231','out')),
             'all four independently sealed data source checks required before labels')
        run(a.s252,a.s233,a.s253,a.s231,a.out)
