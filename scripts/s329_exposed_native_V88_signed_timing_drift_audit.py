"""S329: signed temporal offset forensic on already exposed 16 IDMT XML WAVs.

Only a descriptive diagnostic, NOT score tuning, not reassigning K labels.
For every annotated onset in first 10s use ORIGINAL frozen S302 V88
group interval; find nearest interval and signed time offset.
Positive = candidate occurs after annotated attack; negative = candidate
occurs before annotation; 0 = inside interval. Keep S305 original
20ms assignment and S327 postfrozen truth fully unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
SR=44100
def need(v,m):
    if not v:raise ValueError(m)
def sha(buf):return hashlib.sha256(buf).hexdigest()
def digest(p):return sha(Path(p).read_bytes())
def summarize_signed(ms):
    if not len(ms):return dict(n=0)
    positive=ms>0;negative=ms<0
    return dict(n=len(ms),
         predicted_candidate_after_XML_onset=int(np.count_nonzero(positive)),
         predicted_candidate_before_XML_onset=int(np.count_nonzero(negative)),
         inside_native_interval=int(np.count_nonzero(ms==0)),
         signed_median_ms=float(np.median(ms)),
         signed_q10_q25_q75_q90_ms=np.quantile(ms,[.1,.25,.75,.9]).tolist(),
         absolute_median_ms=float(np.median(np.abs(ms))),
         absolute_q10_q90_ms=np.quantile(np.abs(ms),[.1,.9]).tolist())
def run(s305,s302,s327,s327data,out):
    need(not out.exists(),'S329 original scoring stays unmodified')
    pre=json.loads(s305.read_text())
    score=json.loads(s327.read_text())
    need(pre['status']=='S305_PRELABEL_ORIGINAL_TRAINED_20MS_ATTACK_K0K6_POLICY_SEALED'
         and score['status']=='S327_POSTFREEZE_PROSPECTIVE_IDMT_DATASET2_HISTORICAL_ATTACK_K0K6_CONDITIONAL'
         and score['S305_original_prelabel_20ms_new_attack_protocol_SHA256']==digest(s305)
         and score['original_V88_native_events']==964,
         'source already-opened independent score provenance missing')
    with np.load(s302,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','member','start_samples','end_samples','incumbent_K','incumbent_probabilities'},
             'only original baseline source expected')
        a={k:z[k].copy() for k in z.files}
    with np.load(s327data,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','member','start_samples',
            'true_attack_K_if_XML_complete','official_freeze_K',
            'reconstructed_S80_K','S274_twofold_S80_K'},
             'post-frozen scored source fields drift')
        b={k:z[k].copy() for k in z.files}
    need(np.array_equal(a['row_id'],b['row_id']) and
         np.array_equal(a['incumbent_K'],b['official_freeze_K']),
         'S329 original native 964 event identities changed')
    proofs={p['WAV_filename_SHA256']:p for p in score['per_WAV_annotation_proofs']}
    chosen=pre['per_WAV_preregistered']
    need(len(chosen)==len(proofs)==16,'all exact 16 selected original IDMT WAVs required')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
        'Accept':'application/json','User-Agent':'S329-original-V88-signed-onset-timing-exposed-readonly'}),timeout=60) as res:
        source=audit(json.load(res))
    stream=HttpRangeZIP(source['source_url'],source['size_bytes'])
    allmiss=[];allassigned=[];notes_within=0;late=[];early=[];matched_note_count=0
    per=[]
    with zipfile.ZipFile(stream,'r') as z:
        for p in chosen:
            h=p['original_SHA256_filename']
            ix=np.flatnonzero(a['member'].astype(str)==h)
            order=ix[np.argsort(a['start_samples'][ix],kind='stable')]
            starts=a['start_samples'][order]
            ends=a['end_samples'][order]
            payload=z.read(p['XML_name_precommitted'])
            need(sha(payload)==proofs[h]['XML_sha256'],
                 'already exposed XML annotation SHA differs from first S327')
            notes=parse_original_pitch_events(payload)
            onsets=np.array([int(round(x[0]*SR)) for x in notes],np.int64)
            count,assignment=assign_onsets(onsets,starts,ends)
            need(np.array_equal(np.minimum(count,6),b['true_attack_K_if_XML_complete'][order]),
                 'S329 cannot alter historically scored truth')
            inside=onsets<441000
            signed=[]
            misses=[]
            matched=[]
            for idx,sample in enumerate(onsets):
                if not inside[idx]:
                    continue
                d=np.maximum(np.maximum(starts-sample,sample-ends),0)
                j=int(np.argmin(d))
                if sample<starts[j]:signed_ms=(starts[j]-sample)*1000/SR
                elif sample>ends[j]:signed_ms=(ends[j]-sample)*1000/SR
                else:signed_ms=0.
                signed.append(float(signed_ms))
                if assignment[idx]>=0:matched.append(float(signed_ms))
                else:misses.append(float(signed_ms))
            notes_within+=int(inside.sum())
            allmiss.extend(misses);allassigned.extend(matched)
            per.append(dict(WAV=p['WAV'],composition_group=p['composition_family_name_only'],
                new_note_attacks_within_first10s=int(inside.sum()),
                matched_attacks=len(matched),unmatched_attacks=len(misses),
                unmatched_signed_timing=summarize_signed(np.asarray(misses)),
                matched_signed_timing=summarize_signed(np.asarray(matched))))
    misses=np.asarray(allmiss)
    match=np.asarray(allassigned)
    need(notes_within==250 and len(misses)==203 and len(match)==47,
         'S327 and S328 original 964 native onset timing counts mismatch')
    result=dict(status='S329_S327_EXPOSED_IDMT_NATIVE_V88_SIGNED_TIMING_DIAGNOSTIC',
       original_964_events=True,original_16_WAVs=True,
       first10s_note_attacks=250,
       correct_original_20ms_assignment=47,
       missed_original_20ms_assignment=203,
       missed_signed_relative_nearest_original_V88_interval=summarize_signed(misses),
       matched_signed_relative_nearest_original_V88_interval=summarize_signed(match),
       per_WAV=per,
       original_S305_prelabel_protocol_SHA256=digest(s305),
       original_S327_postfrozen_score_SHA256=digest(s327),
       original_S327_postfrozen_prediction_SHA256=digest(s327data),
       original_predictions_changed=False,truth_target_labels_changed=False,
       candidate_timing_shift_tuned=False,
       radius_changed=False,
       future_data_training_composition_disjoint_verified=False,
       official_global_ExactK_claim=False,
       official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_WAV'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s305','s302','s327','s327data','out'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();run(a.s305,a.s302,a.s327,a.s327data,a.out)
