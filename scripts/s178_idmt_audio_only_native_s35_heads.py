"""S178: acquire pinned external IDMT dataset, extract WAV-only features, run genuine S35 audio experts.

This is an audio-blind-to-annotations test. Never open MIDI, XML, JSON/CSV
annotations; no calibration/fit on IDMT. No raw audio redistributed in artifacts.
H5 and E12 require historical vote32, so deliberately NOT called.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,urllib.request,wave,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.signal import find_peaks
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s177_preregister_external_polyphony_source import (
    RECORD,API,FILENAME,ORIGINAL_MD5,audit as record_audit)

FOLDS=(0,1,2,4)
HEADS=('H1_spectral','H2_lifecycle','H3_harmonics',
       'H4_fundamentals','H6_log_energy_flow','H7_attack_morphology',
       'H10_energy_source_coupling')
def check(ok,why):
    if not ok:raise ValueError(why)
def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):
            h.update(data)
    return h.hexdigest()
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()
def download(url,path,expected_size):
    check(not path.exists(),'refuse overwrite of external source archive')
    req=urllib.request.Request(url,headers={'User-Agent':'S80-public-audio-unseen-research/1.0'})
    size=0
    with urllib.request.urlopen(req,timeout=180) as r,path.open('wb') as f:
        check(r.status==200,'incomplete external source download')
        for chunk in iter(lambda:r.read(4*1024*1024),b''):
            size+=len(chunk)
            check(size<=expected_size,'external download exceeded published size')
            f.write(chunk)
    check(size==expected_size,'archive download size mismatch')
def decode_wav(raw):
    with wave.open(io.BytesIO(raw),'rb') as w:
        check(w.getframerate()==44100 and w.getnchannels()==1,
              'IDMT source mono 44.1k native waveform required')
        bits=w.getsampwidth();count=w.getnframes()
        check(bits in (2,3) and 44100<=count<=44100*120,
              'unexpected waveform encoding or duration')
        x=w.readframes(count)
    if bits==2:
        a=np.frombuffer(x,dtype='<i2').astype(np.float32)/32768.
    else:
        packed=np.frombuffer(x,dtype=np.uint8).reshape(-1,3)
        v=(packed[:,0].astype(np.int32)|
           (packed[:,1].astype(np.int32)<<8)|
           (packed[:,2].astype(np.int32)<<16))
        v=(v^0x800000)-0x800000
        a=v.astype(np.float32)/8388608.
    check(len(a)==count and np.isfinite(a).all(),'nonfinite WAV')
    return a
def acoustic_onsets(signal,max_events):
    """Fixed acoustic-only positive RMS flux; not an Exact-K estimator."""
    hop=441;frame=1764
    n=(len(signal)-frame)//hop+1
    check(n>0,'short WAV')
    xs=np.lib.stride_tricks.sliding_window_view(signal,frame)[::hop][:n]
    rms=np.sqrt(np.mean(np.square(xs),axis=1)+1e-12)
    smooth=np.convolve(rms,np.ones(3)/3,mode='same')
    upward=np.maximum(np.diff(smooth,prepend=smooth[0]),0.)
    floor=max(float(np.quantile(upward,.85)),1e-5)
    peaks,_=find_peaks(upward,distance=18,prominence=floor*.45)
    idx=peaks[upward[peaks]>=floor]
    samples=((idx*hop)+frame//2).astype(np.int64)
    valid=(samples>=.13*44100)&(samples<len(signal)-.22*44100)
    samples=samples[valid]
    if len(samples)>max_events:
        # Choose by time grid, not by predicted K or source metadata.
        samples=samples[np.linspace(0,len(samples)-1,max_events).astype(int)]
    return np.unique(samples)
def group_wav_names(z,max_tracks):
    wav=[n for n in z.namelist() if n.lower().endswith('.wav')
         and '/dataset4/' in ('/'+n.lower()) and not n.startswith('__MACOSX')
         and not Path(n).name.startswith('._')]
    check(len(wav)>=max_tracks,'no polyphonic dataset4 guitar recordings in genuine archive')
    # Immutable selection by filename digest, independent of labels and model outputs.
    wav.sort(key=lambda n:sha_bytes(n.encode()))
    return wav[:max_tracks],len(wav)
def views(summary,flow,morph,physics):
    # Only views independent of missing vote32 are usable.
    return {'H1_spectral':summary,
            'H2_lifecycle':np.column_stack([summary,morph[:,:196]]),
            'H3_harmonics':flow,
            'H4_fundamentals':np.column_stack([flow[:,:49],flow[:,49:98],flow[:,147:196]]),
            'H6_log_energy_flow':flow[:,196:245],
            'H7_attack_morphology':morph,
            'H10_energy_source_coupling':physics}
def predict_original_s35(original_model_dir,feature_views):
    import joblib
    result={}
    proofs=[]
    for name in HEADS:
        preds=[]
        for fold in FOLDS:
            path=original_model_dir/f'models/fold{fold}__outer__{name}.joblib'
            check(path.is_file(),'genuine original S35 trained acoustic head missing: '+str(path))
            model=joblib.load(path)  # trusted original GitHub pretrained artifact ONLY
            check(model['name']==name and model['outer_fold']==fold and model['inner_fold'] is None,
                  'S35 outer original source provenance drift')
            scaler=model['scaler'];est=model['model']
            x=np.asarray(feature_views[name],np.float32)
            check(x.shape[1]==est.n_features_in_==scaler.n_features_in_,
                  'wrong original S35 numeric head dimension '+name)
            values=est.predict_proba(np.clip(scaler.transform(x),-5,5))
            classes=np.asarray(est.classes_,int)
            support=(list(range(2,7)) if name in ('H1_spectral','H2_lifecycle',
                       'H3_harmonics','H4_fundamentals') else list(range(7)))
            # Correct original historical K2..K6 source heads.
            if name in ('H1_spectral','H2_lifecycle','H3_harmonics','H4_fundamentals'):
                check(classes.tolist()==list(range(2,7)),
                      'source historical specialist support drift')
            else:
                check(classes.tolist()==list(range(7)),'source all-class support drift')
            p=np.zeros((len(x),7),np.float32);p[:,classes]=values
            check(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-4),
                  'invalid original class probability')
            preds.append(p)
            proofs.append({'head':name,'fold':fold,'sha256':sha_bytes(path.read_bytes())})
        result[name]=np.mean(preds,axis=0).astype(np.float32)
    return result,proofs
def run(zipfile_path,models,out,max_tracks=10,max_events=10):
    check(not out.exists(),'immutable S178, no overwrite')
    check(zipfile_path.stat().st_size>1_000_000_000,'archive unexpectedly small')
    check(md5(zipfile_path)==ORIGINAL_MD5,'downloaded original IDMT source checksum changed')
    summaries=[];seqs=[];file_ids=[];samples_at=[];manifest=[]
    with zipfile.ZipFile(zipfile_path) as z:
        # Never use ZipFile.testzip(): it would read annotation contents.
        # The complete original archive MD5 is checked before opening WAV members.
        selected,eligible=group_wav_names(z,max_tracks)
        for member in selected:
            check(not member.startswith('/') and '..' not in Path(member).parts,
                  'zip path traversal')
            payload=z.read(member)
            signal=decode_wav(payload)
            starts=acoustic_onsets(signal,max_events)
            for start in starts:
                spec,time=_frames(signal,int(start))
                desc,activ=features_from_spectrogram(spec,time,return_states=True)
                check(len(desc)==58 and activ.shape==(42,49),'old S97 audio shape mismatch')
                summaries.append([desc[k] for k in sorted(desc)])
                seqs.append(np.log1p(activ))
                file_ids.append(member)
                samples_at.append(int(start))
            manifest.append({'wav_sha256':sha_bytes(payload),'filename':member,
               'duration_seconds':round(len(signal)/44100,3),'events_selected':len(starts)})
            print(json.dumps({'track_sha256':sha_bytes(member.encode()),
                  'events':len(starts)}),flush=True)
    check(len(summaries)>=max_tracks,'not enough audio-only acoustic onsets')
    summary=np.asarray(summaries,np.float32)
    seq=np.asarray(seqs,np.float32)
    flow=flow_profile(seq)
    morph,physics=audio_temporal_views(seq)
    check(summary.shape==(len(seq),58) and flow.shape==(len(seq),245) and
          morph.shape==(len(seq),490) and physics.shape==(len(seq),294),
          'original 58/245/490/294D native descriptors invalid')
    probability,checkpoints=predict_original_s35(models,views(summary,flow,morph,physics))
    ids=np.asarray([sha_bytes((n+':'+str(t)).encode()) for n,t in zip(file_ids,samples_at)])
    out.mkdir(parents=True)
    np.savez_compressed(out/'S178_IDMT_audio_only_S35_source_heads.npz',
       row_id=ids,summary58=summary,flow245=flow,morphology490=morph,
       physics294=physics,sequence42x49=seq,
       source_head_ids=np.asarray(HEADS),
       source_probabilities=np.stack([probability[h] for h in HEADS],axis=1),
       source_predicted_K=np.column_stack([probability[h].argmax(1) for h in HEADS]).astype(np.int8))
    counts={h:np.bincount(probability[h].argmax(1),minlength=7).tolist() for h in HEADS}
    report={'status':'S178_INDEPENDENT_PUBLIC_POLYPHONIC_AUDIO_SOURCE_AND_REAL_S35_HEADS',
      'original_corpus_md5':ORIGINAL_MD5,
      'source_wav_in_dataset4_count':eligible,
      'selected_wav_count':len(manifest),
      'acoustic_only_events':len(summary),
      'instrument_audio_manifest':manifest,
      'trained_original_S35_checkpoint_sha256':checkpoints,
      'original_pretrained_acoustic_expert_count':len(checkpoints),
      'class_prediction_histograms_K0toK6':counts,
      'original_audio_features_58_245_490_294':True,
      'training_on_new_dataset':False,'XML_MIDI_or_truth_read':False,
      'released_original_WAV_bytes':False,
      'original_255_group_proposals_generated':False,
      'full_S45_16_sources_available':False,
      'baseline_freeze_original_predictions_generated':False,
      'independent_ExactK_accuracy_measured':False,
      'promoted_S80':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in (
       'status','selected_wav_count','acoustic_only_events',
       'original_pretrained_acoustic_expert_count',
       'class_prediction_histograms_K0toK6','independent_ExactK_accuracy_measured')},
       indent=2),flush=True)
def selftest():
    rng=np.random.default_rng(177)
    t=np.linspace(0,2,88200,endpoint=False)
    sig=(.05*np.sin(2*np.pi*220*t)).astype(np.float32)
    for s in (15000,30000,60000):
        sig[s:s+1500]+=.4*np.hanning(1500).astype(np.float32)
    positions=acoustic_onsets(sig,5)
    check(len(positions)>0 and len(positions)<=5,'audio-only events not found')
    z=views(np.zeros((2,58)),np.zeros((2,245)),np.zeros((2,490)),np.zeros((2,294)))
    check(set(z)==set(HEADS),'no invented pretrained S35 heads')
    print('S178 PASS real audio-onset selector; seven true historical source-view dimensions; no annotation')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--self-test',action='store_true')
    a.add_argument('--archive',type=Path)
    a.add_argument('--models',type=Path)
    a.add_argument('--out',type=Path)
    a.add_argument('--max-tracks',type=int,default=10)
    a.add_argument('--max-events',type=int,default=10)
    args=a.parse_args()
    if args.self_test:selftest()
    else:
        check(args.archive and args.models and args.out,'all paths required')
        run(args.archive,args.models,args.out,args.max_tracks,args.max_events)
