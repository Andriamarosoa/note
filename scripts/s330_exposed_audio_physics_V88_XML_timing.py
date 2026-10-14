"""S330: audio-physics timing of original V88 vs XML, strictly diagnostic.

On the original 16 *already exposed* IDMT Dataset2 recordings, compare
XML onsets with waveform-local novelty estimates derived from raw WAV
only (RMS slope and positive log spectral flux), and original S302
candidate group locations. NEVER move onset labels or original V88
events, fit offsets, score S80, or declare a new benchmark.

Each XML onset is a time locator for a diagnostic search in [-25,+90]ms.
The result is NOT an independent validation of corrected K.
"""
from __future__ import annotations
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP
from scripts.s190_score_frozen_dataset2_XML_exactk import parse_original_pitch_events
from scripts.s302_official_V88_dataset2_rank49plus_16_WAV_source_only import decode_external_wav
SR=44100
HOP=256
FFT=1024
WIN=256
MIN_DT_MS=-25.
MAX_DT_MS=90.
ISOLATION_MS=115.
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(path):return sha(Path(path).read_bytes())
def median_or_none(values):
    return float(np.median(values)) if len(values) else None
def ci(v):
    return list(map(float,np.quantile(v,[.1,.25,.5,.75,.9]))) if len(v) else None
def novelty(audio):
    """Two physically independent, fixed sample-indexed onset energy proxies."""
    a=np.asarray(audio,np.float64)
    need(len(a)>10*FFT and np.all(np.isfinite(a)), 'invalid waveform')
    # Zero-phase high pass filters broad hum; avoids adding causal time lag.
    sos=butter(3,80,fs=SR,btype="highpass",output="sos")
    hp=sosfiltfilt(sos,a)
    energy=np.square(hp)
    rms=np.sqrt(np.convolve(energy,np.ones(WIN)/WIN,mode="same"))
    r=np.arange(HOP,len(rms)-HOP,HOP,dtype=np.int64)
    rms_jump=np.maximum(0.,rms[r]-rms[r-HOP])
    # 1024 Hann spectral frames; timestamps are true *centers*,
    # no implicit FFT-time shift and NO learned model prediction here.
    starts=np.arange(0,len(hp)-FFT+1,HOP,dtype=np.int64)
    frame=np.lib.stride_tricks.sliding_window_view(hp,FFT)[starts]
    mag=np.abs(np.fft.rfft(frame*np.hanning(FFT)[None,:],axis=1))
    f=np.fft.rfftfreq(FFT,1/SR)
    keep=(f>=90)&(f<=7800)
    log=np.log1p(mag[:,keep]*15)
    d=np.maximum(0,np.diff(log,axis=0)).sum(axis=1)
    spectral_t=(starts[1:]+FFT//2).astype(np.int64)
    return r,rms_jump,spectral_t,d
def pick(times,strength,onset):
    lo=onset+int(round(MIN_DT_MS*SR/1000))
    hi=onset+int(round(MAX_DT_MS*SR/1000))
    choices=np.flatnonzero((times>=lo)&(times<=hi))
    if len(choices)==0:return None
    best=choices[int(np.argmax(strength[choices]))]
    top=float(strength[best])
    # Contrast vs local novelty distribution, not other files' labels.
    background=np.median(strength[choices])
    confident=(top>1e-8 and top>=2.5*(background+1e-12))
    return dict(sample=int(times[best]),offset_ms=float((times[best]-onset)*1000/SR),
                strength=top,contrast=float(top/(background+1e-12)),
                confident=bool(confident))
def selftest():
    n=SR
    x=np.zeros(n,np.float64)
    onset=int(.4*SR)
    x[onset:onset+1500]=np.sin(2*np.pi*550*np.arange(1500)/SR)*np.exp(-np.arange(1500)/400)
    a,b,c,d=novelty(x)
    one=pick(a,b,onset);two=pick(c,d,onset)
    need(one and two and abs(one["offset_ms"])<16 and abs(two["offset_ms"])<23,
         "physics novelty has wrong sample-time orientation on known synthetic onset")
    z=np.zeros(n,np.float64)
    u,v,w,q=novelty(z)
    need(not pick(u,v,onset)["confident"] and
         not pick(w,q,onset)["confident"],
         "silence must not pass physical novelty confidence")
    print("S330 synthetic known-sample onset timing tests PASS")
def run(s305,s302,s302report,s327,s327pred,out):
    need(not out.exists(),"immutable exposed-audio diagnostic")
    pre=json.loads(s305.read_text());parent=json.loads(s302report.read_text())
    scored=json.loads(s327.read_text())
    need(pre["status"]=="S305_PRELABEL_ORIGINAL_TRAINED_20MS_ATTACK_K0K6_POLICY_SEALED"
         and parent["status"]=="S302_TRUE_OFFICIAL_FREEZE_ON_UNEXPOSED_DATASET2_16_WAV_UNLABELLED"
         and scored["status"]=="S327_POSTFREEZE_PROSPECTIVE_IDMT_DATASET2_HISTORICAL_ATTACK_K0K6_CONDITIONAL"
         and scored["S305_original_prelabel_20ms_new_attack_protocol_SHA256"]==digest(s305)
         and parent["sealed_original_predictions_SHA256"]==digest(s302),
         "unverified original SHA frozen S327 16 tracks")
    with np.load(s302,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","member","start_samples","end_samples",
                             "incumbent_K","incumbent_probabilities"},
             "true native V88 source only")
        original={k:z[k].copy() for k in z.files}
    with np.load(s327pred,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","member","start_samples",
             "true_attack_K_if_XML_complete","official_freeze_K",
             "reconstructed_S80_K","S274_twofold_S80_K"},
             "original postfrozen fields changed")
        post={k:z[k].copy() for k in z.files}
    need(len(original["row_id"])==964 and
         np.array_equal(original["row_id"],post["row_id"]) and
         np.array_equal(original["incumbent_K"],post["official_freeze_K"]),
         "S330 cannot analyze different reference events")
    manifest={x["original_SHA256_filename"]:x for x in pre["per_WAV_preregistered"]}
    fingerprints={x["WAV_filename_SHA256"]:x for x in parent["original_audio_fingerprints"]}
    annotations={x["wav_filename_SHA256"]:x for x in scored["per_WAV_annotation_proofs"]}
    need(len(manifest)==len(fingerprints)==len(annotations)==16 and
         set(manifest)==set(fingerprints)==set(annotations),
         "wrong original 16 S327 WAV/XML source cohort")
    with urllib.request.urlopen(urllib.request.Request(API,headers={
         "Accept":"application/json","User-Agent":"S330-exposed-audio-physics-diagnostic-no-K-tuning"}),timeout=60) as res:
        source=audit(json.load(res))
    archive=HttpRangeZIP(source["source_url"],source["size_bytes"])
    rows=[]
    with zipfile.ZipFile(archive,"r") as z:
        for h,sourceinfo in manifest.items():
            # Original pre-registration stores a canonical WAV basename;
            # match by SHA filename to the EXACT S276 raw ZIP WAV member.
            # As a fail-closed lookup, unique basename only if it
            # matches the original SHA of the entire archive path.
            wavname=None
            for name in z.namelist():
                if name.endswith("/"+sourceinfo["WAV"]) and sha(name.encode())==h:
                    wavname=name;break
            need(wavname is not None,"original S276 full WAV path with matching SHA missing")
            raw=z.read(wavname)
            need(sha(raw)==fingerprints[h]["original_WAV_SHA256"],
                 "original recorded audio changed")
            signal,meta=decode_external_wav(raw)
            need(meta==fingerprints[h]["waveform_preparation"],
                 "original deterministic downmix/resample provenance drift")
            xml=z.read(sourceinfo["XML_name_precommitted"])
            need(sha(xml)==annotations[h]["XML_sha256"],
                 "already first opened XML SHA changed")
            anns=parse_original_pitch_events(xml)
            onsets=np.array([int(round(t[0]*SR)) for t in anns],np.int64)
            idx=np.flatnonzero(original["member"].astype(str)==h)
            pos=idx[np.argsort(original["start_samples"][idx],kind="stable")]
            left=original["start_samples"][pos];right=original["end_samples"][pos]
            r,rs,f,fs=novelty(signal[:min(len(signal),SR*10)])
            for k,onset in enumerate(onsets):
                if not (onset>=0 and onset<SR*10):continue
                nearest=np.maximum(np.maximum(left-onset,onset-right),0)
                j=int(np.argmin(nearest))
                if onset<left[j]:candidate_signed=float((left[j]-onset)*1000/SR)
                elif onset>right[j]:candidate_signed=float((right[j]-onset)*1000/SR)
                else:candidate_signed=0.
                near=onsets[(onsets>=0)&(onsets<SR*10)]
                isolated=bool(np.count_nonzero(np.abs(near-onset)<round(ISOLATION_MS*SR/1000))==1)
                rms=pick(r,rs,onset);flux=pick(f,fs,onset)
                if rms is None or flux is None:continue
                rows.append(dict(wav_SHA=h,composition_group=sourceinfo["composition_family_name_only"],
                   isolated=isolated,XML_onset_sample=int(onset),
                   original_nearest_V88_minus_XML_ms=candidate_signed,
                   RMS_energy_rise_minus_XML_ms=rms["offset_ms"],
                   spectral_flux_minus_XML_ms=flux["offset_ms"],
                   RMS_contrast=rms["contrast"],spectral_flux_contrast=flux["contrast"],
                   RMS_confident=rms["confident"],spectral_flux_confident=flux["confident"],
                   original_V88_minus_spectral_flux_ms=candidate_signed-flux["offset_ms"],
                   original_V88_minus_RMS_ms=candidate_signed-rms["offset_ms"]))
    need(len(rows)==250,"all original 250 in-excerpt XML note events needed")
    subgroups={"all":rows,
       "isolated":list(x for x in rows if x["isolated"]),
       "isolated_both_strong":list(x for x in rows if x["isolated"] and x["RMS_confident"] and x["spectral_flux_confident"]),
       "missed20ms":list(x for x in rows if abs(x["original_nearest_V88_minus_XML_ms"])>20)}
    views={}
    for key,subset in subgroups.items():
        views[key]=dict(n=len(subset),
          V88_minus_XML_median_ms=median_or_none([x["original_nearest_V88_minus_XML_ms"] for x in subset]),
          RMS_minus_XML_median_ms=median_or_none([x["RMS_energy_rise_minus_XML_ms"] for x in subset]),
          spectral_flux_minus_XML_median_ms=median_or_none([x["spectral_flux_minus_XML_ms"] for x in subset]),
          V88_minus_spectral_flux_median_ms=median_or_none([x["original_V88_minus_spectral_flux_ms"] for x in subset]),
          V88_minus_RMS_median_ms=median_or_none([x["original_V88_minus_RMS_ms"] for x in subset]),
          V88_minus_spectral_flux_q10_q25_q50_q75_q90_ms=ci([x["original_V88_minus_spectral_flux_ms"] for x in subset]),
          spectral_flux_minus_XML_q10_q25_q50_q75_q90_ms=ci([x["spectral_flux_minus_XML_ms"] for x in subset]),
          independent_RMS_and_flux_peak_close_within20ms=int(sum(abs(x["RMS_energy_rise_minus_XML_ms"]-x["spectral_flux_minus_XML_ms"])<=20 for x in subset)),
          V88_arrives_after_flux20ms=int(sum(x["original_V88_minus_spectral_flux_ms"]>20 for x in subset)))
    report=dict(status="S330_EXPOSED_XML_WAV_PHYSICAL_ATTACK_RMS_SPECTRAL_FLUX_LATENCY_DIAGNOSTIC",
         original_WAVs=16,original_V88_events=964,original_inclip_annotated_onsets=250,
         source_S302_frozen_official_sha256=digest(s302),
         source_S327_postfrozen_annotation_sha256=digest(s327pred),
         estimates=views,
         analysis_windows_fixed_a_priori=dict(min_ms=MIN_DT_MS,max_ms=MAX_DT_MS,
                  window_rms_samples=WIN,spectral_FFT_samples=FFT,hop_samples=HOP),
         these_are_XML_local_diagnostic_peaks_not_independent_true_audio_onsets=True,
         no_model_or_candidate_time_shift_applied=True,
         no_prediction_or_truth_reassignment=True,
         XML_previously_opened=True,
         no_new_model_trained=True,
         new_song_composition_disjointness_proven=False,
         official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"per_event_data.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s305","s302","s302report","s327","s327pred","out"):
        p.add_argument("--"+k,type=Path)
    z=p.parse_args()
    if z.self_test:selftest()
    else:
        need(all(getattr(z,k) is not None for k in ("s305","s302","s302report","s327","s327pred","out")),
             "all original source hashes mandatory")
        selftest();run(z.s305,z.s302,z.s302report,z.s327,z.s327pred,z.out)
