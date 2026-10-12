"""S283: immutable true 506-row IDMT audio adapter for unchanged S209/S210
original source heads. Never read XML/MIDI, don't create inferred labels.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.s97_external_s45_original_features import geometry
def req(v,msg):
    if not v:raise ValueError(msg)
def read(p,expected):
    with np.load(p,allow_pickle=False) as z:
        req(set(z.files)==set(expected),'original native source mismatch '+str(p))
        return {k:z[k].copy() for k in z.files}
def run(s281,s278,s277,out):
    req(not out.exists(),'immutable original IDMT audio bridge')
    src=read(s281,{'row_id','freeze_K','fold_ids','global_K','poly_K','S8_K',
         'S9_K','S13_K','S14_K','S15_K','S16_K','S17_K','S18_K'})
    audio=read(s278,{'row_id','member','start_samples','incumbent_freeze_K',
         'summary58','sequence42x49','flow245','morphology490','physics294'})
    original=read(s277,{'row_id','member','start_samples','end_samples','incumbent_K',
         'incumbent_probabilities'})
    n=len(src['row_id'])
    req(n==506 and np.array_equal(src['row_id'],audio['row_id']) and
        np.array_equal(src['row_id'],original['row_id']) and
        np.array_equal(src['freeze_K'],original['incumbent_K']) and
        np.array_equal(src['freeze_K'],audio['incumbent_freeze_K']),
        'original 506 native IDMT exact V88 source identities')
    req(np.array_equal(original['start_samples'],audio['start_samples']) and
        np.array_equal(original['member'],audio['member']) and
        np.array_equal(src['fold_ids'],[0,1,2,4]),'source native fold identity')
    geom=geometry(original['member'],original['start_samples'],original['end_samples'])
    req(geom.shape==(n,3) and np.isfinite(geom).all(),'original geometry3 drift')
    out.mkdir(parents=True)
    archive=out/'S283_original_IDMT_dataset3_native_S97_audio.npz'
    np.savez_compressed(archive,row_id=src['row_id'],summary58=audio['summary58'],
         flow245=audio['flow245'],geometry3=geom,
         sequence42x49=audio['sequence42x49'],morphology490=audio['morphology490'],
         physics294=audio['physics294'],member=original['member'],
         start_samples=original['start_samples'],end_samples=original['end_samples'])
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    result=dict(status='S283_EXTERNAL_ORIGINAL_S97_NATIVE_506_ONLY',
        original_canonical_reference_events=n,
        original_S18_fourfold_input_run=38319911630,
        output_SHA256=digest,labels_MIDI_XML_read=False,
        models_refitted=False,original_S29_models_executed=False,
        independent_ExactK_scored=False,official_S80_promoted=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s281','s278','s277','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.s281,a.s278,a.s277,a.out)
