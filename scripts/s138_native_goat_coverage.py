"""S138: independent forensic coverage of GOAT's first 6s native V84 groups.
Uses only already-sealed S137 predictions and pinned GOAT MIDI. No new
model, selection, or hyperparameter tuning; S137 accuracy unchanged.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.s137_external_s135_smoke import digest,nearest,onsets,MIDI_HASH,SR
def require(c,m):
    if not c:raise ValueError(m)
def run(sealed,manifest,midi,out):
    require(not out.exists(),'immutable forensic coverage output')
    m=json.loads(manifest.read_text())
    require(m['status']=='SEALED_S135_NEW_STUDENT_VS_OFFICIAL_FREEZE' and
            digest(sealed)==m['prediction_sha256'],'prelabel prediction seal changed')
    with np.load(sealed,allow_pickle=False) as z:
        require(set(z.files)=={'row_id','member','starts','ends','incumbent_K','candidate_K'},
                'fixed native group contract')
        members=z['member'].astype(str)
        start=z['starts'].astype(np.int64)
        end=z['ends'].astype(np.int64)
    require(len(start)==57,'locked 57 sample')
    report={'status':'S138_FROZEN_NATIVE_GROUP_COVERAGE_ONLY',
            'native_groups_total':57,'sample_rate':SR,
            'group_window_seconds':6.,
            'original_S80_tested':False,'models_changed':False,
            'MIDI_labels_opened_only_after_S137_seal':True,
            'promotion':False,'members':{}}
    cap=6*SR
    for n in (1,2):
        name=f'example{n}_di.wav'
        p=midi/f'example{n}.mid'
        require(digest(p)==MIDI_HASH[p.name],'MIDI pinned hash mismatch')
        times=onsets(p)
        idx=np.flatnonzero(members==name)
        inside=times[times<cap]
        matched=0
        dists=[]
        missing_dist=[]
        for k in inside:
            # Distance to nearest interval in sample units, whether or not
            # within 20ms. This is a diagnostic, not an alternative metric.
            d=np.maximum.reduce((start[idx]-k,k-end[idx],
                                 np.zeros(len(idx),np.int64)))
            nearest_gap=int(d.min())
            dists.append(nearest_gap)
            if nearest_gap<=882: matched+=1
            else: missing_dist.append(nearest_gap)
        report['members'][name]={
          'native_audio_groups':int(len(idx)),
          'note_on_full_midi':int(len(times)),
          'note_on_first_6s':int(len(inside)),
          'note_on_first_6s_covered_20ms':matched,
          'note_on_first_6s_missed_20ms':len(inside)-matched,
          'onset_coverage_20ms':float(matched/len(inside)) if len(inside) else None,
          'first_6s_nearest_group_gap_ms_median':
             float(np.median(dists)/44.1) if dists else None,
          'first_6s_missed_nearest_gap_ms_median':
             float(np.median(missing_dist)/44.1) if missing_dist else None,
          'native_groups_first_6s':bool(np.all(start[idx]<cap))}
    group_tot=sum(k['native_audio_groups'] for k in report['members'].values())
    labels=sum(k['note_on_first_6s'] for k in report['members'].values())
    matched=sum(k['note_on_first_6s_covered_20ms'] for k in report['members'].values())
    require(group_tot==57,'group loss')
    report['total_note_on_first_6s']=labels
    report['total_note_on_first_6s_covered_20ms']=matched
    report['total_note_on_first_6s_missed_20ms']=labels-matched
    report['coverage_fraction']=matched/labels if labels else None
    report['diagnosis']='onset detection coverage diagnostic, not Exact-K model improvement'
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
def selftest():
    idx=np.array([0,1],dtype=int)
    s=np.array([10000,20000],np.int64)
    e=np.array([10200,20200],np.int64)
    assert nearest(10200,idx,s,e)==0
    assert nearest(11082,idx,s,e)==0
    assert nearest(11083,idx,s,e) is None
    print('S138 PASS: independent first-6s onset coverage fixed metric')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('sealed','manifest','midi','out'):
        a.add_argument('--'+k,type=Path)
    a.add_argument('--self-test',action='store_true')
    p=a.parse_args()
    if p.self_test:selftest()
    else:
        require(all((p.sealed,p.manifest,p.midi,p.out)),'inputs missing')
        run(p.sealed,p.manifest,p.midi,p.out)
