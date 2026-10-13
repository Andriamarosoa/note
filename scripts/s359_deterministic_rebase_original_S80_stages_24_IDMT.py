"""S359: deterministic source-only rebase of 20 original S80 replay stages to NEW 24 WAV IDs.

No trained model can be changed; only replace old event-count invariants and
original S332/S276 WAV metadata input contracts to new SHA-locked S358/S357.
For each derived module record original source SHA + derived source SHA.
Downstream GitHub jobs can ONLY execute these generated modules after S358
has committed 24 real original V88 event IDs. No label/XML files read.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

NAMES=(
 's334_original_dataset1_S97_477.py',
 's335_original_S5_527D_dataset1_477.py',
 's336_four_S5_dataset1_477.py',
 's337_original_S18_133_dataset1_477.py',
 's338_S18_consensus_dataset1_477.py',
 's339_S97_bridge_dataset1_477.py',
 's340_S29_19_dataset1_477.py',
 's341_S35_19_dataset1_477.py',
 's342_original_five_S45_dataset1_477.py',
 's343_recurrent_S45_dataset1_477.py',
 's344_original_fundamental58_dataset1_477.py',
 's345_full_S45_S46_dataset1_477.py',
 's346_S46_consensus_dataset1_477.py',
 's347_S274_twofold_dataset1_477.py',
 's348_PR16_36_dataset1_477.py',
 's349_S69_fourfold_dataset1_477.py',
 's350_S70_fourfold_dataset1_477.py',
 's351_S71_S72_dataset1_477.py',
 's352_full_S80_dataset1_477.py',
 's353_freeze_paired_S80_dataset1_477.py')
OLD_MODEL_SOURCE='scripts.s332_original_V88_dataset1_12_prospective_chords'
NEW_MODEL_SOURCE='scripts.s358_original_freeze_V88_24_IDMT_chords_PREXML'
OLD_STATUS='S332_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET1_16_WAV_UNLABELLED'
NEW_STATUS='S358_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET1_24_WAV_UNLABELLED'
def need(v,why):
    if not v:raise ValueError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rewrite(text,n):
    need(OLD_MODEL_SOURCE not in text or text.count(OLD_MODEL_SOURCE)<=1,
         "unexpected original model import count")
    s=text.replace(OLD_MODEL_SOURCE,NEW_MODEL_SOURCE)
    s=s.replace(OLD_STATUS,NEW_STATUS)
    s=s.replace('dataset1_filename_prioritized_second','next_twentfour')
    s=s.replace('seal["candidate_groups"]["next_twentfour"]','seal["next_twentfour"]')
    s=s.replace("seal['candidate_groups']['next_twentfour']","seal['next_twentfour']")
    s=s.replace('seal["zip_central_names_sha256"]','seal["central_directory_sha256"]')
    s=s.replace("seal['zip_central_names_sha256']","seal['central_directory_sha256']")
    s=s.replace("len(selected)==12","len(selected)==24")
    s=s.replace("evidence['count_WAV']==12","evidence['count_WAV']==24")
    s=s.replace('ref_report["count_WAV"]==12','ref_report["count_WAV"]==24')
    s=s.replace('WAV_count=12','WAV_count=24')
    s=s.replace('["WAV_count"]==12','["WAV_count"]==24')
    s=s.replace('477',str(n))
    # Find other accidental changes: report source family count, checkpoint #12, etc.
    need(NEW_MODEL_SOURCE not in s or OLD_MODEL_SOURCE not in s,
         'new and old direct trained V88 source simultaneously imported')
    need(OLD_STATUS not in s,"old 12-file official pretrained source status used on 24 files")
    need('dataset1_filename_prioritized_second' not in s,"stale exposed cohort")
    return s
def prepare(root,n,out):
    need(out!=root and not out.exists(),'never overwrite original source scripts')
    need(n>=24 and n<=100000,'unexpected original frozen V88 native event count')
    reports=[]
    out.mkdir(parents=True)
    for original in NAMES:
        source=(root/original).read_bytes()
        before=source.decode()
        after=rewrite(before,n)
        compile(after,str(out/original),'exec')
        (out/original).write_text(after)
        reports.append(dict(stage=original,source_sha256=sha(source),
            derived_sha256=sha(after.encode()),original_length=len(source),
            derived_length=len(after.encode())))
    r=dict(status='S359_DETERMINISTIC_PREXML_REBASE_20_ORIGINAL_S80_SOURCE_STAGES',
       original_count_477=477,new_native_event_count=n,new_original_WAV_count=24,
       parent_S357_filename_only=24,parent_S358_original_V88_required=True,
       mappings=dict(reference_model_import=f'{OLD_MODEL_SOURCE}->{NEW_MODEL_SOURCE}',
          old_status=OLD_STATUS,new_status=NEW_STATUS,
          previous_chords_key='dataset1_filename_prioritized_second',
          prospective_chords_key='next_twentfour',
          timeline_changes='ONLY event count and original input/source manifest; no changed network parameters'),
       derived_stage_source_proofs=reports,
       trained_model_architecture_changed=False,
       annotation_XML_opened=False,
       model_fit_or_checkpoint_tuning=False,
       S80_promoted=False)
    (out/'generation_report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='derived_stage_source_proofs'},indent=2))
def run(s358,report,root,out):
    import numpy as np
    info=json.loads(report.read_text())
    need(info['status']==NEW_STATUS and info['count_WAV']==24 and
         info['sealed_original_predictions_SHA256']==sha(s358.read_bytes()) and
         not info['XML_MIDI_annotated_contents_read'],
         'S358 original official SHA-unlabelled 24 WAV source required')
    with np.load(s358,allow_pickle=False) as z:
        n=len(z['row_id'])
        need(n==info['original_native_event_count']
             and len(set(map(str,z['member'])))==24,
             'S358 genuine original V88 24-WAV source did not match')
    prepare(root,n,out)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--test-n',type=int)
    for x in ('s358','report','source-root','out'):
        p.add_argument('--'+x,type=Path)
    q=p.parse_args()
    need(q.out is not None,'--out required')
    if q.test_n:
        prepare(q.source_root,q.test_n,q.out)
    else:
        need(q.s358 and q.report and q.source_root,
             'S358 source and provenance required')
        run(q.s358,q.report,q.source_root,q.out)
