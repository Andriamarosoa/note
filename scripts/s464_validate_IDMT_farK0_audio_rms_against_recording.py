"""S464: original IDMT acoustic candidate energy vs annotated onset proximity.

Does NOT use annotation-driven thresholds for inference or any retraining.
Retrieves the identical 5 recorded audio files from the preregistered
Zenodo ZIP, verifies exact original WAV SHA, decodes identically to S277,
and quantifies source V88/S80 candidate waveform RMS/transient evidence.
Combines with immutable S462 XML-nearness only AFTER audio event features
are computed. K0 on unmatched events is PREDICTED K0, never true K0.
"""
from __future__ import annotations
import argparse,hashlib,io,json,collections,urllib.request,zipfile
from pathlib import Path
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from scripts.s177_preregister_external_polyphony_source import API,audit
from scripts.s183_idmt_selective_http_range_s35 import HttpRangeZIP

SR=44100
CLIP=10.
def need(x,msg):
    if not x:raise ValueError(msg)
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()
def sha(p):
    return sha_bytes(Path(p).read_bytes())
def pcm_same_S277(raw):
    rate,x=wavfile.read(io.BytesIO(raw))
    channels=x.shape[1] if x.ndim==2 else 1
    if x.dtype==np.int16:x=x.astype(np.float32)/32768.
    elif x.dtype==np.int32:x=x.astype(np.float64)/2147483648.
    elif x.dtype==np.uint8:x=(x.astype(np.float32)-128.)/128.
    elif np.issubdtype(x.dtype,np.floating):x=x.astype(np.float64)
    else:raise ValueError("S464 unsupported original PCM")
    if channels>1:x=x.mean(axis=1)
    if rate!=SR:
        fac=gcd(int(rate),SR)
        x=resample_poly(x,SR//fac,int(rate)//fac)
    x=np.clip(np.asarray(x,np.float32),-1,1)
    need(len(x)>=2*SR and np.isfinite(x).all(),
         "S464 decoded waveform source invalid")
    return x[:10*SR]

def rms(audio,start,end):
    lo=max(0,min(len(audio),int(start)))
    hi=max(lo,min(len(audio),int(end)))
    if hi-lo<20:return 0.
    s=audio[lo:hi].astype(np.float64)
    return float(np.sqrt(np.mean(s*s)))

def event_energies(audio,onset):
    x=int(onset)
    before=rms(audio,x-int(.065*SR),x-int(.020*SR))
    after=rms(audio,x,x+int(.045*SR))
    sustained=rms(audio,x+int(.065*SR),x+int(.120*SR))
    return (before,after,sustained,
            float(20*np.log10((after+1e-7)/(before+1e-7))))

def summary(rows):
    if not rows:return dict(n=0)
    amp=np.asarray([r["rms_relative_to_sameWAV_matched_onsets_db"]
                     for r in rows],float)
    attack=np.asarray([r["post_vs_pre_attack_rms_db"] for r in rows],float)
    tail=np.asarray([r["sustain_vs_attack_rms_db"] for r in rows],float)
    return dict(n=len(rows),
        median_rms_db_below_matched_onsets=float(np.median(amp)),
        p25_rms_relative_db=float(np.percentile(amp,25)),
        p75_rms_relative_db=float(np.percentile(amp,75)),
        below_minus12dB_from_known_matched_attack_n=int(np.sum(amp< -12)),
        below_minus20dB_from_known_matched_attack_n=int(np.sum(amp< -20)),
        above_minus3dB_from_known_matched_attack_n=int(np.sum(amp>=-3)),
        transient_rms_rise_over_6dB_n=int(np.sum(attack>=6)),
        median_transient_rms_rise_db=float(np.median(attack)),
        median_sustain_vs_attack_db=float(np.median(tail)))

def run(s276,s277,s277_report,s462_ledger,s462_report,s297,out):
    need(not out.exists(),"S464 immutable signal-level audit exists")
    original=json.loads(s277_report.read_text())
    match=json.loads(s462_report.read_text())
    cohort=json.loads(s276.read_text())
    need(original["status"]=="S277_TRUE_OFFICIAL_FREEZE_ON_DATASET3_FIVE_WAV_UNLABELLED"
         and original["sealed_original_predictions_SHA256"]==sha(s277)
         and match["status"]=="S462_IDMT_EXTERNAL_ORIGINAL_ONSET_XML_MATCHING_FORENSIC"
         and match["all_IDMT_sources_pinned_SHA256"]["nearest_source_ledger"]==sha(s462_ledger)
         and match["original_native_event_count"]==506
         and cohort["status"]=="S276_IDMT_CANDIDATE_COHORT_CENTRAL_DIRECTORY_ONLY_NOT_SCORED"
         and cohort["counts_WAV"]["dataset3"]==5,
         "S464 requires actual frozen original S277/V88 and S462 source")
    with np.load(s277,allow_pickle=False) as z:
        ids=z["row_id"].astype(str)
        members=z["member"].astype(str)
        starts=z["start_samples"].astype(np.int64)
    with np.load(s297,allow_pickle=False) as z:
        need(np.array_equal(ids,z["row_id"].astype(str)),
             "S464 original pretrained IDMT row indices drifted")
        pred=z["s80_K"].astype(np.int8)
    known=[json.loads(l) for l in s462_ledger.read_text().splitlines()]
    byid={r["row_id"]:r for r in known}
    need(len(byid)==506 and set(ids)==set(byid)
         and all(byid[q]["original_S80_K"]==int(k) for q,k in zip(ids,pred)),
         "S464 original K or matched event identity differs from S462")
    bysource={a["wav_filename_sha256"]:a for a in
              cohort["candidate_groups"]["dataset3_first"]}
    proofs={o["WAV_filename_SHA256"]:o for o in
            original["original_audio_fingerprints"]}
    need(len(bysource)==len(proofs)==5 and
         set(proofs)==set(bysource)==set(members),
         "S464 original waveform hashes or prereg manifest differ")
    req=urllib.request.Request(API,headers={
        "Accept":"application/json","User-Agent":"S464-IDMT-audio-rms-candidates-original-WAV"})
    with urllib.request.urlopen(req,timeout=60) as response:
        info=audit(json.load(response))
    remote=HttpRangeZIP(info["source_url"],info["size_bytes"])
    auditrows=[];bywav=[]
    with zipfile.ZipFile(remote) as z:
        need(sha_bytes("\n".join(sorted(z.namelist())).encode())==
             cohort["zip_central_names_sha256"],"S464 original ZIP source changed")
        for key,obj in sorted(bysource.items()):
            path=obj["wav"]
            raw=z.read(path)
            need(sha_bytes(raw)==proofs[key]["original_WAV_SHA256"],
                 "S464 real original WAV bytes changed after S277 freeze")
            wav=pcm_same_S277(raw)
            indices=np.flatnonzero(members==key)
            details=[]
            reference_rms=[]
            for i in indices:
                before,after,tail,attack=event_energies(wav,starts[i])
                t=byid[str(ids[i])]
                desc=dict(row_id=str(ids[i]),recording_sha=key,
                    original_S80_predicted_K=int(pred[i]),
                    greedy_S451_matched=bool(t["original_S451_matched"]),
                    nearest_XML_onset_ms=t["nearest_XML_group_ms"],
                    near_XML_onset_80ms=(t["nearest_XML_group_ms"] is not None and
                        t["nearest_XML_group_ms"]<=80.),
                    before_rms=float(before),after_rms=float(after),
                    later_rms=float(tail),
                    post_vs_pre_attack_rms_db=float(attack),
                    sustain_vs_attack_rms_db=float(
                        20*np.log10((tail+1e-7)/(after+1e-7))))
                if desc["greedy_S451_matched"]:reference_rms.append(after)
                details.append(desc)
            need(len(reference_rms)>0,
                 "S464 every WAV must have original matched XML attacks")
            base=float(np.median(reference_rms))
            need(base>0,"S464 no real acoustic reference energy for matched notes")
            for row in details:
                row["rms_relative_to_sameWAV_matched_onsets_db"]=float(
                    20*np.log10((row["after_rms"]+1e-7)/(base+1e-7)))
            bywav.append(dict(recording_sha=key,
                expected_native_events=len(indices),
                matched_S451=sum(d["greedy_S451_matched"] for d in details),
                reference_median_matched_attacks_rms=base,
                near80=summary([q for q in details if q["near_XML_onset_80ms"]]),
                far80=summary([q for q in details if not q["near_XML_onset_80ms"]]),
                far80_predK0=summary([q for q in details if
                    not q["near_XML_onset_80ms"] and
                    q["original_S80_predicted_K"]==0])))
            auditrows.extend(details)
            print("S464 ORIGINAL_WAV",json.dumps(bywav[-1]),flush=True)
    need(len(auditrows)==506 and sum(q["greedy_S451_matched"] for q in auditrows)==47,
         "S464 original XML matched set differs from S462")
    near=[q for q in auditrows if q["near_XML_onset_80ms"]]
    far=[q for q in auditrows if not q["near_XML_onset_80ms"]]
    far0=[q for q in far if q["original_S80_predicted_K"]==0]
    near0=[q for q in near if q["original_S80_predicted_K"]==0]
    need(len(near)==88 and len(far)==418 and
         len(far0)==417 and len(near0)==39 and
         len(near)+len(far)==506,
         "S464 must reuse verified original S462 XML-onset candidate proximity")
    res=dict(status="S464_ORIGINAL_IDMT_V88_EXCESS_K0_EVENT_WAVEFORM_RMS_FORENSIC",
        original_IDMT_real_WAV=5,original_native_events=506,
        distant_80ms_from_any_XML_onset=418,
        distant_original_S80_K0=417,
        original_matched_XML_onsets=47,
        original_S80_K_locked_and_unmodified=True,
        original_WAV_hashes_verified_against_S277=True,
        raw_WAV_excerpt_first_10s_mono_resampled_as_S277=True,
        no_new_training_no_retuning_or_event_filtering=True,
        K0_only_predictions_NOT_true_silence_labels=True,
        source_diagnostic_RMS_not_a_causal_attribution=True,
        XML_notes_already_exposed_originally=True,
        no_promotion=True,
        acoustic_groups=dict(
            native_near_XML_80ms=summary(near),
            native_far_XML_80ms=summary(far),
            far80_original_predicted_K0=summary(far0),
            near80_original_predicted_K0=summary(near0),
            matched_true_XML_attacks=summary(
                [q for q in auditrows if q["greedy_S451_matched"]])),
        per_WAV=bywav,
        SHA256=dict(S276=sha(s276),S277=sha(s277),
            S462_ledger=sha(s462_ledger),S297=sha(s297)))
    out.mkdir(parents=True)
    path=out/"S464_original_realWAV_event_acoustic_RMS_506.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in auditrows))
    res["output_acoustic_ledger_SHA256"]=sha(path)
    (out/"report.json").write_text(json.dumps(res,indent=2)+"\n")
    print("S464 SIGNAL_CLASSIFICATION",json.dumps(res["acoustic_groups"]),flush=True)

def selftest():
    audio=np.zeros(int(SR*.5),np.float32)
    audio[int(.1*SR):int(.14*SR)]=.5
    x=event_energies(audio,int(.1*SR))
    need(x[1]>.1 and x[3]>60,"S464 simple onset RMS diagnostic wrong")
    r=summary([dict(rms_relative_to_sameWAV_matched_onsets_db=-18.,
        post_vs_pre_attack_rms_db=7.,sustain_vs_attack_rms_db=-4.)])
    need(r["below_minus12dB_from_known_matched_attack_n"]==1
         and r["transient_rms_rise_over_6dB_n"]==1,
         "S464 thresholds purely descriptive selftest")
    print("S464 SELFTEST PASS waveform onset energy and conservative descriptive bins")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("s276","s277","s277-report","s462-ledger","s462-report","s297","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("s276","s277","s277_report","s462_ledger",
                  "s462_report","s297","out"):
            need(getattr(a,k)is not None,"S464 missing --"+k)
        run(a.s276,a.s277,a.s277_report,a.s462_ledger,
            a.s462_report,a.s297,a.out)
