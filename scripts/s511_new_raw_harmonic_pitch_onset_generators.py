"""S511: independent RAW WAV harmonic/note-birth candidate generator.

Ideas audited from github.com/Andriamarosoa/midi:
- independent multi-label frame/fundamental/onset observations, not softmax
- harmonic partial evidence is *soft* (real octaves may exist)
- per-pitch onset vs sustained energy, temporal hysteresis.
From YourMT3+ paper: combine time-frequency evidence and different expert
proposals, but DO NOT call this code pretrained YourMT3+.

This is a deterministic, genuine new acoustic PITCH GENERATOR, NOT a
trained classifier. No MIDI/XML labels read, no original S80 or Live
edited. Raw IDMT guitar WAVs independently verified against S277 and
raw MAESTRO piano clips independently verified against S470.
Outputs offline diagnostic note-at-onset candidates at two FIXED top-K
sizes (6 and 12), never selected by future true-label scores.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import stft,find_peaks
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s464_validate_IDMT_farK0_audio_rms_against_recording import ReliableReadOnlyRangeZIP
from scipy.io import wavfile
from scipy.signal import resample_poly
import io,time,urllib.parse
from math import gcd
def decode_external_wav(raw):
    """Independent conversion matching original S277 documented audio ABI."""
    frequency,signal=wavfile.read(io.BytesIO(raw))
    channels=signal.shape[1] if signal.ndim==2 else 1
    need(channels in (1,2) and 20000<=int(frequency)<=96000,
         "S511 source 44.1k WAV compatibility invalid")
    source_dtype=str(signal.dtype)
    if signal.dtype==np.int16:signal=signal.astype(np.float32)/32768.
    elif signal.dtype==np.int32:signal=signal.astype(np.float64)/2147483648.
    elif signal.dtype==np.uint8:signal=(signal.astype(np.float32)-128.)/128.
    elif np.issubdtype(signal.dtype,np.floating):signal=signal.astype(np.float64)
    else:raise ValueError("S511 unsupported original WAV sample")
    if channels>1:signal=signal.mean(axis=1)
    if frequency!=SR:
        div=gcd(int(frequency),SR)
        signal=resample_poly(signal,SR//div,int(frequency)//div)
    signal=np.clip(np.asarray(signal,np.float32),-1.,1.)
    need(len(signal)>2*SR and np.isfinite(signal).all(),
         "S511 source WAV not finite or too short")
    return signal,dict(original_sample_rate_hz=int(frequency),
                       original_channels=int(channels),
                       original_dtype=source_dtype)
def fetch_track(base,name):
    """Use exact independent MAESTRO WAV mirror as in original S470."""
    url=base+urllib.parse.quote(name,safe="/")+"?download=true"
    req=urllib.request.Request(url,headers={
         "Accept-Encoding":"identity","User-Agent":"S511-raw-piano-only-noMIDI"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=240) as response:
                need(response.status==200 and
                     "text/html" not in response.headers.get("Content-Type",""),
                     "S511 mirror returned HTML")
                blob=response.read(140_000_001)
                need(100_000<len(blob)<=140_000_000 and blob[:4]==b"RIFF",
                     "S511 unsafe size or incorrect WAV bytes")
                return blob,url
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))


SR=44100
FFT=8192
HOP=441
MIDIS=np.arange(40,89,dtype=int)
MIDI_PITCH_HZ=440.*(2.**((MIDIS-69.)/12.))
CONFIG=dict(fft=FFT,hop=HOP,notes_range=[40,88],
            topk=[6,12],harmonics=8,attack_local_top_proposals=True,
            separate_frame_note_and_onset_evidence=True,
            independent_pitch_score_threshold_for_rank_only=.0,
            max_physical_onsets_per_12s=180)
def need(c,m):
    if not c:raise ValueError(m)
def sha(x):
    return hashlib.sha256(x).hexdigest()
def filesha(p):return sha(Path(p).read_bytes())
def note_name(p):
    chars=("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
    return chars[int(p)%12]+str(int(p)//12-1)
def per_pitch_harmonic_scores(wave):
    wave=np.asarray(wave,dtype=np.float32)
    need(wave.ndim==1 and len(wave)>=SR and np.isfinite(wave).all(),
         "S511 genuine 44.1k mono recording and finite samples required")
    _,t,spec=stft(wave,fs=SR,window="hann",nperseg=FFT,
                  noverlap=FFT-HOP,nfft=FFT,padded=False,boundary=None)
    amp=np.abs(spec).astype(np.float32)
    # Local contrast normalization: avoid assuming one amplitude scale
    # works for all guitar microphones and piano.
    frequency_hz=np.arange(amp.shape[0],dtype=float)*SR/FFT
    outputs=[]
    for midi,fundamental in zip(MIDIS,MIDI_PITCH_HZ):
        harmonics=[];weights=[]
        for h in range(1,9):
            freq=float(fundamental)*h
            if freq>SR/2-30:continue
            centre=int(np.rint(freq*FFT/SR))
            radius=2 if h<=3 else 3
            lo=max(1,centre-radius);hi=min(amp.shape[0],centre+radius+1)
            power=np.max(amp[lo:hi],axis=0)
            harmonics.append(power)
            weights.append(1./(h**.66))
        parts=np.stack(harmonics,axis=0)
        w=np.asarray(weights,np.float32)
        # A subharmonic is not proven from one isolated higher harmonic:
        # favour coherent sets but keep independent real octave candidates.
        weighted=np.sum(np.log1p(parts*600.)*w[:,None],axis=0)/w.sum()
        fundamental_power=np.log1p(parts[0]*600.)
        harmonically_supported=.75*weighted+.25*fundamental_power
        outputs.append(harmonically_supported)
    pitch=np.asarray(outputs,np.float32)
    # Normalize across source time, not by true pitch annotation:
    median=np.median(pitch,axis=1,keepdims=True)
    p95=np.quantile(pitch,.95,axis=1,keepdims=True)
    scale=np.maximum(.2,p95-median)
    relative=np.clip((pitch-median)/scale,0,2.5)
    # Attack novelty: compare latest 30ms to history 70-120ms earlier.
    before=np.pad(relative,((0,0),(12,0)),mode="edge")[:,:-12]
    rise=np.maximum(0,relative-before)
    # Source-independent physical onset from positive spectral change.
    logarithmic=np.log1p(amp*800.)
    derivative=np.maximum(0,np.diff(logarithmic,axis=1,prepend=logarithmic[:,:1]))
    flux=np.mean(derivative,axis=0)
    robust=float(np.median(flux))
    mad=float(np.median(np.abs(flux-robust)))
    z=(flux-robust)/max(.005,1.4826*mad)
    # Positive physical attack novelty is a separate head, not proof K.
    positive=np.maximum(z,0)
    return np.asarray(t),pitch,relative,rise,positive
def infer(wave,ident,source):
    t,raw,rel,rise,physical=per_pitch_harmonic_scores(wave)
    candidate_peaks,_=find_peaks(physical,distance=7,prominence=.6,height=.80)
    # A constant or silent waveform yields no phantom physical attacks.
    physical_peaks=candidate_peaks[physical[candidate_peaks]>=.80]
    if len(physical_peaks)>CONFIG["max_physical_onsets_per_12s"]:
        top=sorted(physical_peaks,key=lambda i:(-float(physical[i]),int(i)))
        physical_peaks=np.asarray(sorted(top[:CONFIG["max_physical_onsets_per_12s"]]),int)
    events=[]
    for frame in physical_peaks:
        tt=float(t[frame])
        if tt>=len(wave)/SR:continue
        lo=max(0,frame-3);hi=min(len(t),frame+5)
        # independent fundamental/temporal experts; score has NO MIDI labels
        score=.43*np.max(rel[:,lo:hi],axis=1)
        score+=.42*np.max(rise[:,lo:hi],axis=1)
        score+=.15*np.clip(physical[frame]/5.,0,2.)
        # Higher fundamental attack novelty is rewarded, but we DO NOT
        # eliminate true octaves when harmonic relation exists.
        rank=np.argsort(-score,kind="stable")
        for choice in (6,12):
            selected=rank[:choice]
            events.append(dict(WAV_filename_SHA256=ident,source=source,
                 independent_physical_onset_s=tt,
                 independent_physical_strength=float(physical[frame]),
                 policy=f"top{choice}",
                 candidate_MIDI=[int(MIDIS[i]) for i in selected],
                 candidate_note_names=[note_name(MIDIS[i]) for i in selected],
                 candidate_scores=[round(float(score[i]),5) for i in selected],
                 full_candidate_count=49,
                 physical_event_id=f"{ident}:{frame}",
                 class_truth_never_read=True))
    return events
def parse_sources(s276,s277,s277report,s469,s470,enable_piano):
    manifest=json.loads(s276.read_text());original=json.loads(s277report.read_text())
    need(manifest["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and original["status"]=="S277_TRUE_OFFICIAL_FREEZE_ON_DATASET3_FIVE_WAV_UNLABELLED"
         and original["sealed_original_predictions_SHA256"]==filesha(s277)
         and manifest["counts_WAV"]["dataset3"]==5,
         "S511 original IDMT source must match S276/S277 prelabel SHA")
    chosen=manifest["candidate_groups"]["dataset3_first"]
    source_proofs={x["WAV_filename_SHA256"]:x for x in original["original_audio_fingerprints"]}
    req=urllib.request.Request(API,headers={"Accept":"application/json",
                                          "User-Agent":"S511-note-generator-MIDI-blind"})
    with urllib.request.urlopen(req,timeout=60) as response:
        server=audit(json.load(response))
    reader=ReliableReadOnlyRangeZIP(server["source_url"],server["size_bytes"])
    sources=[]
    with zipfile.ZipFile(reader,"r") as z:
        need(sha("\n".join(sorted(z.namelist())).encode())==
             manifest["zip_central_names_sha256"],
             "S511 historical original five-WAV ZIP directory changed")
        for item in chosen:
            ident=item["wav_filename_sha256"]
            actual=z.read(item["wav"])
            need(ident in source_proofs and
                 sha(actual)==source_proofs[ident]["original_WAV_SHA256"],
                 "S511 genuine audio bytes differ from source V88 evidence")
            wave,fmt=decode_external_wav(actual)
            sources.append((ident,"guitar_S80",wave[:10*SR],
                           source_proofs[ident]["original_WAV_SHA256"]))
    if enable_piano:
        m=json.loads(s469.read_text());baseline=json.loads(s470.read_text())
        need(m["status"]=="S469_PREREGISTERED_MAESTRO_V3_THIRD_SOURCE_FILENAME_ONLY"
             and baseline["status"]==
                 "S470_MAESTRO_V3_REAL_AUDIO_ORIGINAL_PRETRAINED_V88_PLUS_S466_PREMIDI"
             and baseline["preregistered_S469_SHA"]==filesha(s469)
             and len(m["selected_sources"])==3,
             "S511 original three piano recording manifest changed")
        proof={x["WAV_filename_SHA256"]:x
               for x in baseline["exact_original_audio_source_fingerprints"]}
        for item in m["selected_sources"]:
            ident=item["filename_SHA256"]
            wav,_=fetch_track(m["audio_mirror_directory"],item["audio_filename"])
            need(ident in proof and sha(wav)==proof[ident]["original_WAV_SHA256"],
                 "S511 original held piano whole WAV bytes changed")
            wave,fmt=decode_external_wav(wav)
            st=int(float(item["audio_excerpt_start_seconds"])*SR)
            width=int(float(item["audio_excerpt_duration_seconds"])*SR)
            need(width==12*SR and st+width<=len(wave),
                 "S511 original MAESTRO piano excerpt timing drifted")
            sources.append((ident,"piano_V88",wave[st:st+width],
                            proof[ident]["original_WAV_SHA256"]))
    return sources
def run(a):
    need(not a.out.exists(),"S511 no new independent raw-pitch freeze overwrite")
    sources=parse_sources(a.s276,a.s277,a.s277_report,a.s469,a.s470,a.include_piano)
    a.out.mkdir(parents=True)
    entries=[];counts=[]
    for ident,kind,wave,whole_source_SHA in sources:
        predicted=infer(wave,ident,kind)
        entries.extend(predicted)
        counts.append(dict(recording=ident,source=kind,
             original_complete_wav_SHA256=whole_source_SHA,
             seconds=float(len(wave)/SR),
             unique_physical_onsets=sum(x["policy"]=="top6" for x in predicted),
             candidate_pitches_at_each_onset_6_or_12=True))
        print("S511 REAL RAW SOUND TO NOTES",json.dumps(counts[-1]),flush=True)
    file=a.out/"S511_RAW_HARMONIC_FUNDAMENTAL_ONSET_TOP6_TOP12_PRETRUTH.jsonl"
    file.write_text("".join(json.dumps(x)+"\n" for x in entries))
    status=dict(status="S511_NEW_INDEPENDENT_RAW_HARMONIC_PITCH_ONSET_CANDIDATES_PREMIDI",
       original_WAVs=8 if a.include_piano else 5,
       WAV_byte_SHA_source_verified=True,
       not_YourMT3_pretrained_model=True,
       source_midi_v2_2_framed_multi_label_harmonic_onset_ideas=True,
       independent_acoustic_pitch_generator_not_a_trained_classifier=True,
       no_XML_or_MIDI_ground_truth_read_in_training_or_inference=True,
       offline_noncausal_centered_harmonic_window_not_live=True,
       fixed_two_top6_top12_policies_before_ground_truth=True,
       no_original_source_S80_or_V88_changed=True,
       no_promotion=True,
       params=CONFIG,per_recording=counts,
       number_of_acoustic_pitch_onset_events=sum(x["policy"]=="top6" for x in entries),
       candidates_sha256=filesha(file))
    (a.out/"report.json").write_text(json.dumps(status,indent=2)+"\n")
def selftest():
    # Genuine synthetic TEST ONLY validates internal DSP; not a test score.
    t=np.arange(2*SR)/SR
    wave=np.zeros(2*SR,np.float32)
    start=int(.3*SR)
    envelope=np.exp(-3*np.arange(SR)/SR)
    tone=(np.sin(2*np.pi*164.81*np.arange(SR)/SR)
         +.60*np.sin(2*np.pi*329.62*np.arange(SR)/SR))
    wave[start:start+SR]=(.2*envelope*tone).astype(np.float32)
    q=infer(wave,"test","synthetic")
    need(len(q)>0 and all("candidate_MIDI" in x for x in q)
         and all(len(x["candidate_MIDI"]) in (6,12) for x in q),
         "S511 must detect synthetic E3 physical onset and generate MIDI candidate pitches")
    print("S511 SELFTEST PASS DSP raw-wave pitch/birth detector; synthetic not benchmark")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    p.add_argument("--include-piano",action="store_true")
    for k in ("s276","s277","s277-report","s469","s470","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s276","s277","s277_report","out"):
            need(getattr(a,k)is not None,"S511 missing --"+k)
        if a.include_piano:
            need(a.s469 and a.s470,"S511 explicit original S469/S470 piano required")
        run(a)
