"""S385: ORIGINAL pretrained Dataset2 S97, 527D S5, four S5, 133 S18, S217 source ABI for 32 NEW WAVs.

Auditable text-only changes of original S306→S310 ORIGINAL working
checkpoint source modules previously validated on 964 true native
Dataset2 rows. Only source external manifest, original WAV count 16→32,
native row count 964→new actual N and original V88 source importer
are changed. No checkpoint retrain, no XML/MIDI reads, no score.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
MODULES=[
 's306_original_IDMT_dataset2_964_S97_native_views.py',
 's307_original_S5_527D_IDMT_dataset2_964.py',
 's308_original_4S5_trained_fold_964_dataset2.py',
 's309_original_S8_S18_133_heads_dataset2_964.py',
 's310_S217_fold_consensus_dataset2_964.py']
def sha(x):return hashlib.sha256(x).hexdigest()
def need(x,msg):
    if not x:raise ValueError(msg)
def adapt(original,N):
    source=original.replace('scripts.s302_official_V88_dataset2_rank49plus_16_WAV_source_only',
                            'scripts.s384_original_V88_source_32_new_IDMT_dataset2_unseen_WAV')
    source=source.replace('S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED',
                         'S384_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_32_WAV_UNLABELLED')
    source=source.replace('S302','S384').replace('s302','s384')
    source=source.replace('S276','S383').replace('s276','s383')
    source=source.replace('seal["candidate_groups"]["dataset2_unexposed_XML_paired_reserve"]',
                          'seal["untouched_new_32"]')
    source=source.replace('seal["zip_central_names_sha256"]',
                          'seal["original_ZIP_central_names_sha256"]')
    source=source.replace('len(selected)==16','len(selected)==32')
    source=source.replace('WAV_count=16','WAV_count=32')
    source=source.replace('964',str(N))
    need('S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16' not in source
         and 'dataset2_unexposed_XML_paired_reserve' not in source
         and 'scripts.s302_official_' not in source,
         'old exposed Dataset2 original source still referenced')
    return source
def run(original,official,src_root,out):
    need(not out.exists(),'S385 immutable original model ABI proof')
    info=json.loads(original.read_text())
    need(info['status']=='S384_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_32_WAV_UNLABELLED'
         and info['count_WAV']==32
         and info['sealed_original_predictions_SHA256']==sha(official.read_bytes())
         and not info['XML_MIDI_annotated_contents_read'],
         'S385 authentic SHA sealed official original new32 baseline required')
    with np.load(official,allow_pickle=False) as z:
        n=len(z['row_id'])
        need(n==info['original_native_event_count'] and
             len(set(map(str,z['member'])))==32,
             'S385 32 source WAVs are not all original V88 native rows')
    out.mkdir(parents=True)
    proofs=[]
    for name in MODULES:
        raw=(src_root/name).read_bytes()
        code=adapt(raw.decode(),n)
        compile(code,name,'exec')
        (out/name).write_text(code)
        proofs.append(dict(original=name,original_source_sha256=sha(raw),
                           generated_original_checkpoint_ABI_sha256=sha(code.encode())))
    report=dict(status='S385_AUTHENTIC_5_GUITARSET_HELDFOLD_S5_S18_SOURCE_MODULES_FOR_NEW32_DATASET2',
      N=n,source_new_WAVs=32,
      original_source_sha256=sha(official.read_bytes()),
      model_weight_changes=False,
      original_training_dataset_changed=False,
      first_dataset2_new32_XML_MIDI_read=False,
      all_original_5_modules_compiled=True,
      code_source_hashes=proofs,
      official_promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='code_source_hashes'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('report','official','source','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.report,a.official,a.source,a.out)
