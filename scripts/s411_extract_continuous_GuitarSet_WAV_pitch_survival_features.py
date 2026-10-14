"""S411: full-continuous-WAV feature source for future pitch-conditioned offsets.

GuitarSet mono-microphone official WAV archive:
  https://zenodo.org/records/3371780 (audio_mono-mic.zip)
  MD5 275966d6610ac34999b58426beb119c3
Note: S258 acoustic event bank used a different original pickup-mix source.
Do not treat this source as identical to S258 or as a measured offset gain.

Inputs: raw WAV ZIP + S408 frozen per-note prediction NPZ. Does NOT read
JAMS labels, trained S80/S402/S407/S408 weights or any GT duration.
Outputs per-piece continuous 10ms time-grid acoustic harmonic trajectories,
a diagnostic matched-to-predicted-note feature archive, source hashes.
No selection using note annotations; no baseline changes or promotion.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,wave,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from scipy.signal import stft
from scipy.ndimage import maximum_filter1d
from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
SR=44100
HOP=441
NFFT=4096
MIDI_MIN=40
MIDI_MAX=88
OFFSETS_MS=(100,200,350,500,750,1000,1400,1800,2500,3500)
PARTIALS=6
MAX_NOTE=33475
def req(b,s):
    if not b:raise ValueError(s)
def decode_pcm(raw):
    with wave.open(io.BytesIO(raw),'rb') as wav:
        req(wav.getframerate()==SR,'GuitarSet WAV sample rate must be 44100Hz')
        chans=wav.getnchannels();width=wav.getsampwidth()
        req(chans in (1,2) and width in (2,4) and wav.getcomptype()=='NONE',
            'Unexpected WAV PCM channel/width/compression')
        buf=wav.readframes(wav.getnframes())
    dtype='<i2' if width==2 else '<i4'
    data=np.frombuffer(buf,dtype=dtype).reshape(-1,chans).astype(np.float32)/(2**(8*width-1))
    return np.mean(data,axis=1)
def continuous_pitch_strength(audio):
    """Continuous frames, including K0 and stretches without attack groups."""
    freq,sec,complex_stft=stft(audio,fs=SR,window='hann',
        nperseg=NFFT,noverlap=NFFT-HOP,
        nfft=NFFT,boundary='zeros',padded=False)
    mag=np.abs(complex_stft).astype(np.float32)
    # ±1 bin note evidence; deterministic, label-blind harmonic filter bank.
    local=maximum_filter1d(mag,size=3,axis=0,mode='nearest')
    frames=local.shape[1]
    s=np.zeros((frames,MIDI_MAX-MIDI_MIN+1),np.float32)
    for index,m in enumerate(range(MIDI_MIN,MIDI_MAX+1)):
        f0=440.*2**((m-69)/12)
        for harmonic in range(1,PARTIALS+1):
            hz=f0*harmonic
            if hz>=SR/2:break
            b=int(np.rint(hz*NFFT/SR))
            s[:,index]+=local[b]/float(np.sqrt(harmonic))
    s=np.log1p(1000*s)
    sum_energy=np.log1p(1000*mag.sum(axis=0))
    # Note evidence first, derivative explicitly separates decays from attacks.
    delta=np.diff(s,axis=0,prepend=s[:1])
    rise=np.maximum(delta,0)
    fall=np.maximum(-delta,0)
    frame=np.rint(sec*SR).astype(np.int64)
    req(s.shape==(len(frame),49) and
        np.all(np.isfinite(s)) and len(frame)>100,
        'Full WAV continuous harmonic pitch bank invalid')
    return frame,s,rise,fall,sum_energy

def features_for_notes(p,track,frame,strength,rise,fall,energy):
    keys=np.flatnonzero(p['track_member'].astype(str)==track)
    onset=p['S407_onset_samples'][keys].astype(np.int64)
    midi=p['S402_midi_pitch'][keys].astype(int)
    pidx=midi-MIDI_MIN
    req(np.isin(pidx,range(49)).all(),'Predicted MIDI outside source bank')
    out=np.zeros((len(keys),9+5*len(OFFSETS_MS)),np.float32)
    for n,(on,index) in enumerate(zip(onset,pidx)):
        t=np.searchsorted(frame,int(on),side='left')
        t=min(max(0,int(t)),len(frame)-1)
        row=[float(midi[n])/88,float(p['S80_K'][keys[n]])/6,
             float(strength[t,index]),float(rise[t,index]),
             float(fall[t,index]),float(energy[t]),
             float(np.log1p((p['S408_offset_samples'][keys[n]]-on)/SR*1000)),
             float(on/SR),float(t/len(frame))]
        for h in OFFSETS_MS:
            target=int(on+SR*h/1000)
            i=min(max(0,np.searchsorted(frame,target)),len(frame)-1)
            exists=target<=int(frame[-1])
            if exists:
                row.extend([1.,float(strength[i,index]),float(rise[i,index]),
                     float(fall[i,index]),float(energy[i])])
            else:row.extend([0.,0.,0.,0.,0.])
        out[n]=np.array(row,np.float32)
    req(np.isfinite(out).all(),'S411 pitch features nonfinite')
    return keys,out
def main(audio_zip,notes,out,max_tracks=None):
    req(not out.exists(),'S411 immutable output cannot overwrite')
    with np.load(notes,allow_pickle=False) as z:
        p={k:z[k].copy() for k in z.files}
    need={'track_member','S402_midi_pitch','S407_onset_samples',
          'S408_offset_samples','S80_K'}
    req(need.issubset(p),'S411 frozen S408 prediction keys unavailable')
    n=len(p['S80_K'])
    req(n==MAX_NOTE and len(set(p['track_member'].astype(str)))==190,
        'S411 predictions original track and 33475 note count drift')
    out.mkdir(parents=True)
    needed=np.unique(p['track_member'].astype(str))
    master=np.zeros((n,9+len(OFFSETS_MS)*5),np.float32)
    assigned=np.zeros(n,bool)
    stats=[]
    with zipfile.ZipFile(audio_zip,'r') as z:
        names=collections.defaultdict(list)
        for name in z.namelist():
            if not name.endswith('.wav'):continue
            # Strict naming: <original JAMS base>_mic.wav.
            b=PurePosixPath(name).name
            if b.endswith('_mic.wav'):names[b[:-8]+'.jams'].append(name)
        req(len(names)>=190,'S411 official mic ZIP track names not recognized')
        for t,track in enumerate(needed):
            options=names.get(PurePosixPath(track).name,[])
            req(len(options)==1,'Missing/ambiguous original mic WAV for '+track)
            raw=z.read(options[0])
            wav=decode_pcm(raw)
            frame,mag,rise,fall,energy=continuous_pitch_strength(wav)
            ids,X=features_for_notes(p,track,frame,mag,rise,fall,energy)
            master[ids]=X
            assigned[ids]=True
            stats.append(dict(track=track,frames=int(len(frame)),
                duration_seconds=round(float(len(wav)/SR),2),
                predicted_notes=int(len(ids)),K0_frames_retained=True,
                source_audio_member=options[0]))
            if (t+1)%20==0:print('S411 PROGRESS',t+1,'of',len(needed),flush=True)
            if max_tracks is not None and t+1>=max_tracks:break
    if max_tracks is None:req(assigned.all(),'S411 full source track feature coverage incomplete')
    req(np.isfinite(master).all(),'S411 all generated source features must be finite')
    filename=out/'S411_continuous_mic_pitch_features_for_S408_notes.npz'
    np.savez_compressed(filename,source_row_id=p['source_row_id'],
          source_note_index=np.arange(n,dtype=np.int32),
          notes_midi=p['S402_midi_pitch'],S80_K=p['S80_K'],
          note_features=master,valid_note_mask=assigned)
    obj=dict(status='S411_FULL_CONTINUOUS_MIC_WAV_FEATURE_SOURCE_ONLY',
      input_source_is_mono_MIC_not_original_S258_pickup_mix=True,
      warning='Different microphone domain from S258 pickup mix! Must train/evaluate separately.',
      native_audio_recovered=bool(max_tracks is None),
      seen_tracks=len(stats),expected_tracks=190,
      original_note_predictions=n,notes_with_WAV_context=int(assigned.sum()),
      temporal_hop_ms=10,FFT_bins=NFFT,pitch_min=40,pitch_max=88,
      future_horizon_ms=OFFSETS_MS,
      full_continuous_audio_evidence_even_without_new_events=True,
      original_S80_S402_S407_S408_preserved=True,
      no_JAMS_annotations_read=True,
      offsets_not_trained_nor_scored=True,
      predictions_source_SHA256=sha(notes),
      source_zip_SHA256=sha(audio_zip),
      output_npz_SHA256=sha(filename),tracks=stats)
    (out/'report.json').write_text(json.dumps(obj,indent=2)+'\n')
    print('S411 SOURCE',json.dumps({k:v for k,v in obj.items() if k!='tracks'}),flush=True)
    print('S411 PASS source-only continuous 10ms WAV trajectories including K0 and >1s sustain',flush=True)
def test():
    import struct
    s=(np.random.default_rng(411).normal(size=SR)*.01).astype(np.float32)
    q=(np.clip(s,-1,1)*32767).astype('<i2').tobytes()
    h=io.BytesIO()
    with wave.open(h,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR);w.writeframes(q)
    actual=decode_pcm(h.getvalue())
    frame,mag,rise,fall,energy=continuous_pitch_strength(actual)
    req(mag.shape[1]==49 and len(frame)>90 and
        np.isfinite(rise).all() and np.isfinite(fall).all(),
        'S411 WAV full-grid features failed')
    print('S411 SELFTEST PASS: WAV PCM -> full 10ms harmonic MIDI 40..88 frame bank')
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--self-test',action='store_true')
    for n in ('audio-zip','notes','out'):a.add_argument('--'+n,type=Path)
    a.add_argument('--max-tracks',type=int)
    x=a.parse_args()
    if x.self_test:test()
    else:
        for n in ('audio_zip','notes','out'):
            req(getattr(x,n)is not None,'Missing --'+n)
        main(x.audio_zip,x.notes,x.out,x.max_tracks)
