"""S316: real original harmonic-vs-fundamental 58D on 964 original V88 Dataset2 events.

Only uses S276 frozen filename list, S302 SHA-pinned audio provenance,
S306 exact acoustic summary and original historic fundamental bank.
No label/XML/MIDI access; no training; stereo->mono follows S302 exactly.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s302_official_V88_dataset2_rank49plus_16_WAV_source_only import decode_external_wav
from scripts.extract_v273_harmonic_trajectory import (
    _frames,features_from_spectrogram,FUNDAMENTAL_BANK)
def need(v,m):
    if not v:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def load(path,keys):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==keys,"S316 source-only archive unexpected field")
        return {k:z[k].copy() for k in z.files}
def run(s306,s302,s302report,s276,out):
    need(not out.exists(),"immutable S316 source features")
    acoustic=load(s306,{'row_id','member','start_samples','incumbent_freeze_K',
        'summary58','sequence42x49','flow245','morphology490','physics294'})
    ref=load(s302,{'row_id','member','start_samples','end_samples',
        'incumbent_K','incumbent_probabilities'})
    seal=json.loads(s276.read_text())
    evidence=json.loads(s302report.read_text())
    ids=acoustic['row_id'];n=len(ids)
    need(n==964 and all(np.array_equal(ref[k],acoustic[k]) for k in
         ('row_id','member','start_samples'))
         and np.array_equal(ref['incumbent_K'],acoustic['incumbent_freeze_K']),
         "all 964 original S302/S306 native event identities must match")
    need(evidence['count_WAV']==16 and
         evidence['XML_MIDI_annotated_contents_read']==False and
         sha(s302.read_bytes())==evidence['sealed_original_predictions_SHA256'],
         "S302 original data/model source was modified")
    selected=seal['candidate_groups']['dataset2_unexposed_XML_paired_reserve']
    frozen={p['WAV_filename_SHA256']:p for p in evidence['original_audio_fingerprints']}
    need(len(selected)==16 and set(frozen)=={
        f['wav_filename_sha256'] for f in selected},'source WAV set changed')
    with urllib.request.urlopen(urllib.request.Request(API,headers={
        'Accept':'application/json','User-Agent':'S316-original-fundamental-dataset2-no-XML'}),
        timeout=60) as response:
        source=audit(json.load(response))
    archive=HttpRangeZIP(source['source_url'],source['size_bytes'])
    fundamental=np.zeros((n,58),np.float64)
    harmonic=np.zeros((n,58),np.float64)
    done=np.zeros(n,bool)
    names=None
    with zipfile.ZipFile(archive,'r') as z:
        need(sha('\n'.join(sorted(z.namelist())).encode())==
             seal['zip_central_names_sha256'],
             "pinned original ZIP directory identity drift")
        for obj in selected:
            wavname=obj['wav'];identifier=obj['wav_filename_sha256']
            idx=np.flatnonzero(ref['member'].astype(str)==identifier)
            payload=z.read(wavname)
            need(sha(payload)==frozen[identifier]['original_WAV_SHA256'],
                 'WAV recording original SHA changed since predictions sealed')
            samples,meta=decode_external_wav(payload)
            need(meta==frozen[identifier]['waveform_preparation'],
                 'native WAV resample/channel normalization changed')
            for i in idx:
                start=int(ref['start_samples'][i])
                need(0<=start<min(len(samples),441000),"event sample outside first10s")
                spec,time=_frames(samples,start)
                normal=features_from_spectrogram(spec,time)
                rare=features_from_spectrogram(spec,time,bank=FUNDAMENTAL_BANK)
                sortednames=sorted(normal)
                need(len(sortednames)==58 and sortednames==sorted(rare),
                     'historic fundamental and harmonic feature names incompatible')
                if names is None:names=sortednames
                need(names==sortednames,'original feature order drift')
                harmonic[i]=[normal[k] for k in sortednames]
                fundamental[i]=[rare[k] for k in sortednames]
                done[i]=True
            print(json.dumps(dict(wav_name_SHA256=identifier,
                original_native_events=int(len(idx)))),flush=True)
    need(done.all() and np.isfinite(harmonic).all()
         and np.isfinite(fundamental).all(),'S316 native S1-S4 inputs incomplete')
    deviation=np.max(np.abs(harmonic-acoustic['summary58'])/
                     (1+np.abs(acoustic['summary58'])))
    need(deviation<1e-3,
         "S316 primary 58D differs from already frozen original S306 S97")
    out.mkdir(parents=True)
    path=out/'S316_fundamental58_dataset2_original_964.npz'
    np.savez_compressed(path,row_id=ids,features=harmonic,
        fundamental=fundamental,feature_names=np.asarray(names))
    parent=out/'S316_original_S5_label_free_reference.npz'
    np.savez_compressed(parent,row_id=ids,freeze_K=ref['incumbent_K'])
    report=dict(status='S316_ORIGINAL_FUNDAMENTAL58_DATASET2_964_AUDIO_ONLY',
      original_reference_events=n,original_feature_count=58,
      original_fundamental_count=58,
      original_primary_max_scaled_deviation_from_S306=float(deviation),
      original_fundamental_audio_SHA256=sha(path.read_bytes()),
      original_S5_reference_ID_SHA256=sha(parent.read_bytes()),
      WAV_count=16,annotation_XML_MIDI_contents_read=False,
      no_new_training=True,source_original_training_models_untouched=True,
      independent_ExactK_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s306','s302','s302report','s276','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.s306,a.s302,a.s302report,a.s276,a.out)
