"""S327: first postfreeze HISTORICAL ATTACK EXACT-K for 16 NEW Dataset2 WAVs.

Run ONLY after original pretrained S302 reference + both full S80 outputs
on all 964 original native V88 events were SHA sealed in S326.
Uses original S305 preregistered GuitarSet/V90 native 20ms NOTE-ATTACK
assignment: K=min(6, number of original XML note onsets assigned to
nearest V88 <=40ms original cluster). NOT sustained pitch count; NOT
S299's unrelated 80ms/+30ms note-matching target.

No model fit / recalibration / retrospective selection. XML per-WAV
completeness is not independently certified; rank filename disjointness
is not proof of composition-disjointness. Score conditional and do NOT
promote official freeze_local_combo on this run.
"""
from __future__ import annotations
import argparse,hashlib,json,re,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import assign_onsets,CLUSTER_SAMPLES
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
SR=44100
SEED=327
BOOT=4096
PRED={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
      'original_PR16_policy22_K','K1_by_fold_threshold_action',
      'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
NATIVE={'row_id','member','start_samples','end_samples','incumbent_K','incumbent_probabilities'}
S326_STATUS='S326_TWO_COMPLETE_S80_S274_CANDIDATES_PRELABEL_IDENTICAL_964_V88_EVENTS'
S305_STATUS='S305_PRELABEL_ORIGINAL_TRAINED_20MS_ATTACK_K0K6_POLICY_SEALED'
def need(v,why):
    if not v:raise ValueError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,'Only originally frozen source fields allowed')
        return {k:z[k].copy() for k in z.files}
def ledger(y,old,new,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):
        return dict(n=0,reference_correct=0,candidate_correct=0,
            reference_exactK=None,candidate_exactK=None,delta=None,
            fixes=0,regressions=0,changed_both_wrong=0)
    truth=y[ix];b=old[ix];p=new[ix]
    b_ok=(b==truth);p_ok=(p==truth)
    fix=int((~b_ok&p_ok).sum())
    regression=int((b_ok&~p_ok).sum())
    need(int(p_ok.sum()-b_ok.sum())==fix-regression,'paired correct ledger invalid')
    return dict(n=int(len(ix)),reference_correct=int(b_ok.sum()),
       candidate_correct=int(p_ok.sum()),
       reference_exactK=float(b_ok.mean()),
       candidate_exactK=float(p_ok.mean()),
       delta=float(p_ok.mean()-b_ok.mean()),
       fixes=fix,regressions=regression,
       changed_both_wrong=int(((b!=p)&~b_ok&~p_ok).sum()))
def bootstrap(y,old,new,member_groups,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return None
    groups=sorted(set(member_groups[ix]))
    if len(groups)<2:
        return dict(composition_groups=len(groups),CI95=None,
                    confidently_positive=False)
    parts=[ix[member_groups[ix]==g] for g in groups]
    rng=np.random.default_rng(SEED)
    deltas=[]
    for _ in range(BOOT):
        chosen=rng.integers(0,len(parts),size=len(parts))
        rows=np.concatenate([parts[k] for k in chosen])
        deltas.append(float(np.mean(
            (new[rows]==y[rows]).astype(float)-
            (old[rows]==y[rows]).astype(float))))
    low,high=np.quantile(deltas,[.025,.975])
    return dict(composition_groups=len(groups),replicates=BOOT,
       seed=SEED,CI95=[float(low),float(high)],
       confidently_positive=bool(low>0),
       grouping_name_based_not_proven_song_disjoint=True)
def selftest():
    k,index=assign_onsets(np.array([1010,1070,3999,4205,9000]),
        np.array([1000,4000]),np.array([1300,4200]))
    need(k.tolist()==[2,2] and index.tolist()==[0,0,1,1,-1],
         'original preregistered historical K0-K6 attack target drift')
    arr=np.array([0,2,3,1]);b=np.array([0,3,3,2]);a=np.array([0,2,2,1])
    stats=ledger(arr,b,a,np.ones(4,bool))
    need(stats['fixes']==2 and stats['regressions']==1,
         'paired corrected/regressed ledger')
    print('S327 historical attack target and paired ledger self-test PASS')
def run(selection,protocol,freeze,freeze_report,pair,oldp,newp,out):
    need(not out.exists(),'never overwrite originally scored prospective XML')
    original_selection=json.loads(selection.read_text())
    prereg=json.loads(protocol.read_text())
    source=json.loads(freeze_report.read_text())
    before=json.loads(pair.read_text())
    need(prereg['status']==S305_STATUS
         and before['status']==S326_STATUS
         and source['status']=='S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED'
         and prereg['source_first_original_S276_WAV_manifest_sha256']==sha_file(selection)
         and prereg['source_original_S302_official_predict_sha256']==sha_file(freeze)
         and prereg['source_S302_report_sha256']==sha_file(freeze_report)
         and source['sealed_original_predictions_SHA256']==sha_file(freeze)
         and before['original_baseline_freeze_SHA256']==sha_file(freeze)
         and before['original_S80_SHA256']==sha_file(oldp)
         and before['S274_twofold_S80_SHA256']==sha_file(newp)
         and prereg['all_XML_contents_untouched']
         and before['truth_XML_or_MIDI_read']==False
         and before['all_results_on_exact_same_original_V88_native_events']
         and not before['official_S80_promotion'],
         'Pre-XML original source/score hashes and provenance MUST all match')
    a=load(freeze,NATIVE);b=load(oldp,PRED);c=load(newp,PRED)
    ids=a['row_id'];member=a['member'].astype(str)
    starts=a['start_samples'];ends=a['end_samples']
    n=len(ids);official=a['incumbent_K'].astype(np.int8)
    original=b['s80_K'].astype(np.int8);strict=c['s80_K'].astype(np.int8)
    need(n==964 and len(set(map(str,ids)))==n and
         np.array_equal(ids,b['row_id']) and np.array_equal(ids,c['row_id'])
         and np.array_equal(b['incumbent_freeze_K'],official)
         and np.array_equal(c['incumbent_freeze_K'],official),
         'Actual original 964 native V88 event IDs no longer match')
    selected=prereg['per_WAV_preregistered']
    need(len(selected)==16 and len(set(x['original_SHA256_filename'] for x in selected))==16
         and set(member)==set(x['original_SHA256_filename'] for x in selected),
         'New dataset2 16 WAVs differ from prelabel cohort')
    known={x['original_SHA256_filename']:x for x in selected}
    need(all(x['original_SHA256_filename'] not in
             original_selection['dataset2_previous_48_wav_names_sha256'] for x in selected),
         'first 48 exposed WAV source used by mistake')
    # FIRST XML BYTE READ AFTER ALL THREE S80/REFERENCE IMMUTABLE HASHES verified.
    with urllib.request.urlopen(urllib.request.Request(API,headers={
         'Accept':'application/json','User-Agent':'S327-ATTACK-K0-K6-after-all-three-predictions-frozen'}),timeout=60) as resp:
        published=audit(json.load(resp))
    stream=HttpRangeZIP(published['source_url'],published['size_bytes'])
    y=np.full(n,-1,np.int16)
    family=np.empty(n,dtype='<U96')
    seen=np.zeros(n,bool)
    tracks=[]
    with zipfile.ZipFile(stream,'r') as archive:
        need(sha('\n'.join(sorted(archive.namelist())).encode())==
             original_selection['zip_central_names_sha256'],
             'Published original source directory differs from prospective 16 WAV precommit')
        for proof in selected:
            key=proof['original_SHA256_filename']
            rows=np.flatnonzero(member==key)
            ordered=rows[np.argsort(starts[rows],kind='stable')]
            left=starts[ordered].astype(np.int64)
            right=ends[ordered].astype(np.int64)
            need(len(ordered)>0 and np.all(left[1:]>right[:-1])
                 and np.all(right>=left) and
                 np.all((right-left)<=CLUSTER_SAMPLES),
                 'invalid original native V88 group boundaries')
            xml=archive.read(proof['XML_name_precommitted'])
            notes=parse_original_pitch_events(xml)
            onsets=[int(round(x[0]*SR)) for x in notes]
            count,assignment=assign_onsets(onsets,left,right)
            y[ordered]=np.minimum(count,6)
            family[ordered]=proof['composition_family_name_only']
            seen[ordered]=True
            tracks.append(dict(wav_filename_SHA256=key,
              original_WAV_name=proof['WAV'],
              composition_group_filename_guess=proof['composition_family_name_only'],
              XML_sha256=sha(xml),
              XML_note_onsets=int(len(onsets)),
              source_native_events=len(ordered),
              XML_onsets_matched_to_V88=int(np.count_nonzero(assignment>=0)),
              XML_onsets_unmatched_to_V88=int(np.count_nonzero(assignment<0)),
              native_K0_groups_if_XML_complete=int(np.count_nonzero(count==0)),
              native_poly_K2plus_groups_if_XML_complete=int(np.count_nonzero(count>=2))))
    need(seen.all() and np.isin(y,range(7)).all(),
         'XMLs do not cover all sixteen originally sealed WAV native groups')
    pairs={
        'S80_reconstructed_vs_official_freeze':(official,original),
        'S80_S274_twofold_vs_official_freeze':(official,strict),
        'S80_S274_twofold_vs_S80_reconstructed':(original,strict)
    }
    poly=(y>=2)
    comparison={}
    for label,(base,candidate) in pairs.items():
        comparison[label]=dict(
            native_all_K0to6_if_XML_complete=ledger(y,base,candidate,np.ones(n,bool)),
            true_new_attack_poly_K2to6_if_XML_complete=ledger(y,base,candidate,poly),
            by_true_K_if_XML_complete={f'K{i}':ledger(y,base,candidate,y==i)
                                       for i in range(7)},
            name_group_bootstrap_all=bootstrap(y,base,candidate,family,
                                               np.ones(n,bool)),
            name_group_bootstrap_poly=bootstrap(y,base,candidate,family,poly),
            changed_total=int(np.count_nonzero(base!=candidate)))
    result=dict(
        status='S327_POSTFREEZE_PROSPECTIVE_IDMT_DATASET2_HISTORICAL_ATTACK_K0K6_CONDITIONAL',
        selected_WAV=16,original_V88_native_events=n,
        S305_original_prelabel_20ms_new_attack_protocol_SHA256=sha_file(protocol),
        S326_original_complete_S80s_PREXML_SHA256=sha_file(pair),
        first_XML_read_after_source_checks=True,
        historical_train_target_20ms_attack_cardinality_not_sustained=True,
        true_attack_K0to6_if_XML_complete=np.bincount(y,minlength=7).tolist(),
        true_attack_poly_K2to6_if_XML_complete=int(poly.sum()),
        annotation_onsets_total=int(sum(t['XML_note_onsets'] for t in tracks)),
        annotation_onsets_not_assigned_to_native=int(sum(t['XML_onsets_unmatched_to_V88'] for t in tracks)),
        paired_comparisons=comparison,
        per_WAV_annotation_proofs=tracks,
        XML_transcription_exhaustive_independently_certified=False,
        filename_grouped_licks_are_composition_disjoint_from_train_proven=False,
        candidate_model_tuned_after_XML=False,
        source_original_weights_retrained=False,
        independent_historical_S80_exact_route_identity_proven=False,
        official_global_ExactK_certified=False,
        official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'S327_original_964_attackK0to6_XML_annotations_postfrozen.npz',
       row_id=ids,member=member,start_samples=starts,
       true_attack_K_if_XML_complete=y,
       official_freeze_K=official,
       reconstructed_S80_K=original,
       S274_twofold_S80_K=strict)
    print(json.dumps({k:v for k,v in result.items() if k not in
         ('paired_comparisons','per_WAV_annotation_proofs')},indent=2),flush=True)
    for label,values in comparison.items():
        print(json.dumps({label:dict(global_964=values['native_all_K0to6_if_XML_complete'],
            poly=values['true_new_attack_poly_K2to6_if_XML_complete'],
            global_CI=values['name_group_bootstrap_all'],
            poly_CI=values['name_group_bootstrap_poly'])},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for x in ('selection','protocol','freeze','freeze_report','pair','old','new','out'):
        p.add_argument('--'+x,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,x) is not None for x in
             ('selection','protocol','freeze','freeze_report','pair','old','new','out')),
             'All prelabel source/score SHA files required before XML')
        selftest()
        run(a.selection,a.protocol,a.freeze,a.freeze_report,a.pair,a.old,a.new,a.out)
