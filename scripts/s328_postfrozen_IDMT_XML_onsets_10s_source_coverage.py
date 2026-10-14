"""S328: forensic audit of unassigned IDMT Dataset2 XML note attacks in S327.

On SAME already post-frozen / already-opened 16 XMLs. Split XML attacks
outside first 10s truncated source audio versus inside analyzed excerpt;
for inside, assess genuine V88 native onset proposal coverage under
PRE-REGISTERED historical 20ms assignment. No thresholds tuned, no new
labels used in S80, no new model selection, no official promotion.
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
TEN=441000
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(buf):return hashlib.sha256(buf).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def run(s305,s302,s302report,s327,s327data,out):
    need(not out.exists(),"S328 diagnostic immutable on already-exposed XML")
    pre=json.loads(s305.read_text())
    baseline=json.loads(s302report.read_text())
    scored=json.loads(s327.read_text())
    need(pre['status']=='S305_PRELABEL_ORIGINAL_TRAINED_20MS_ATTACK_K0K6_POLICY_SEALED'
         and scored['status']=='S327_POSTFREEZE_PROSPECTIVE_IDMT_DATASET2_HISTORICAL_ATTACK_K0K6_CONDITIONAL'
         and scored['S305_original_prelabel_20ms_new_attack_protocol_SHA256']==sha_file(s305)
         and baseline['status']=='S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED'
         and baseline['sealed_original_predictions_SHA256']==sha_file(s302)
         and scored['original_V88_native_events']==964,
         "S328 original prospective source and known scored XML source not proven")
    with np.load(s302,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','member','start_samples','end_samples',
                            'incumbent_K','incumbent_probabilities'},
             'baseline source cannot contain label fields')
        a={k:z[k].copy() for k in z.files}
    with np.load(s327data,allow_pickle=False) as z:
        need(set(z.files)=={'row_id','member','start_samples',
             'true_attack_K_if_XML_complete','official_freeze_K',
             'reconstructed_S80_K','S274_twofold_S80_K'},
             'original S327 scored immutable fields not found')
        b={k:z[k].copy() for k in z.files}
    need(np.array_equal(a['row_id'],b['row_id'])
         and np.array_equal(a['member'],b['member'])
         and np.array_equal(a['start_samples'],b['start_samples'])
         and np.array_equal(a['incumbent_K'],b['official_freeze_K']),
         'S328 original new model cohort differs from first frozen S327')
    proof={x['original_SHA256_filename']:x for x in pre['per_WAV_preregistered']}
    seen_xml={x['wav_filename_SHA256']:x for x in scored['per_WAV_annotation_proofs']}
    need(len(proof)==len(seen_xml)==16 and set(proof)==set(seen_xml),
         'unknown or missing already annotated recording')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
        'Accept':'application/json',
        'User-Agent':'S328-postfrozen-exposed-XML-timing-no-training'}),timeout=60) as rsp:
        source=audit(json.load(rsp))
    stream=HttpRangeZIP(source['source_url'],source['size_bytes'])
    rows=[]
    with zipfile.ZipFile(stream,'r') as z:
        for h,p in proof.items():
            pos=np.flatnonzero(a['member'].astype(str)==h)
            ordered=pos[np.argsort(a['start_samples'][pos],kind='stable')]
            starts=a['start_samples'][ordered]
            ends=a['end_samples'][ordered]
            xml=z.read(p['XML_name_precommitted'])
            need(sha(xml)==seen_xml[h]['XML_sha256'],
                 'S328 XML content differs from first post-frozen S327')
            ann=parse_original_pitch_events(xml)
            onset=np.asarray([int(round(occ[0]*SR)) for occ in ann],np.int64)
            count,assigned=assign_onsets(onset,starts,ends)
            need(np.array_equal(np.minimum(count,6),b['true_attack_K_if_XML_complete'][ordered]),
                 'original S327 historical K score not bit-exact replayed')
            eligible=onset<TEN
            within=onset[eligible]
            matched=assigned>=0
            before=int(np.count_nonzero(onset<0))
            in_clip=int(eligible.sum())
            outside=int((~eligible).sum())
            candidate_missing=int(np.count_nonzero(eligible&~matched))
            near=[]
            for sample in within[~matched[eligible]]:
                d=np.maximum(np.maximum(starts-sample,sample-ends),0)
                near.append(int(np.min(d)))
            nearest=np.asarray(near,np.int64)
            rows.append(dict(WAV=p['WAV'],WAV_filename_SHA256=h,
              composition_group_filename_only=p['composition_family_name_only'],
              xml_onsets_total=len(onset),xml_onsets_beyond_first_10s=outside,
              xml_onsets_within_first_10s=in_clip,
              XML_within_first_10s_assigned_original_20ms=int(np.count_nonzero(eligible&matched)),
              XML_within_first_10s_missed_original_20ms=candidate_missing,
              missed_within40ms=int(np.count_nonzero(nearest<=1764)),
              missed_within80ms=int(np.count_nonzero(nearest<=3528)),
              missed_over80ms=int(np.count_nonzero(nearest>3528)),
              earliest_XML_onset_sec=float(onset.min()/SR) if len(onset) else None,
              latest_XML_onset_sec=float(onset.max()/SR) if len(onset) else None,
              V88_native_events=int(len(ordered)),
              nearest_unmatched_median_ms=float(np.median(nearest)*1000/SR) if len(nearest) else None))
    measures=['xml_onsets_total','xml_onsets_beyond_first_10s','xml_onsets_within_first_10s',
       'XML_within_first_10s_assigned_original_20ms',
       'XML_within_first_10s_missed_original_20ms',
       'missed_within40ms','missed_within80ms','missed_over80ms']
    total={name:int(sum(item[name] for item in rows)) for name in measures}
    need(total['xml_onsets_total']==scored['annotation_onsets_total']==300
         and total['XML_within_first_10s_missed_original_20ms']+total['xml_onsets_beyond_first_10s']==
            scored['annotation_onsets_not_assigned_to_native']==253
         and total['XML_within_first_10s_assigned_original_20ms']==47,
         "historical native V88 candidate recall and S327 XML ledger do not match")
    result=dict(status='S328_POSTFROZEN_16_WAV_XML_NOTE_ATTACK_COVERAGE_DIAG',
       original_native_V88_events=964,
       source_original_S327_report_SHA256=sha_file(s327),
       source_original_S327_predictions_SHA256=sha_file(s327data),
       S327_historical_onset_matched_labels_bit_exact_verified=True,
       first_10s_audio_vs_full_XML_coverage=total,
       within_audio_XML_attack_assignment_recall20ms=(
           total['XML_within_first_10s_assigned_original_20ms']/
           total['xml_onsets_within_first_10s']
           if total['xml_onsets_within_first_10s'] else None),
       per_wav=rows,
       XML_was_previously_opened_by_S327=True,
       historical_20ms_assignment_unchanged=True,
       alternative_match_radius_tuning_performed=False,
       global_ExactK_retrained_or_changed=False,
       onset_XML_transcription_complete_independently_verified=False,
       recording_composition_disjointness_verified=False,
       official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_wav'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s305','s302','s302report','s327','s327data','out'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();run(a.s305,a.s302,a.s302report,a.s327,a.s327data,a.out)
