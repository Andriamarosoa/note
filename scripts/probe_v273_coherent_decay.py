"""Development-only audit of phase-preserving prediction, never a K decoder.

A real damped sinusoid is the real part of a complex exponential. A finite
sum obeys a linear recurrence. Fit that recurrence from raw PAST PCM, then
forecast the next 40 ms. No known frequencies, amplitudes, labels, or future
samples enter the predictor. Compare against the existing log-power feature.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.train_v100_spectral_string_slots import (
    SAMPLE_RATE,PRE_SAMPLES,POST_SAMPLES,_spectral_map_from_segment)
from scripts.v273_decay_native_inputs import native_decay_features
from scripts.rebuild_v273_sources import digest,write_json

LONG_HISTORY=8820  # 200 ms of past; never additional lookahead
RCOND=1e-7
CONFIGS={
    'wave_short':dict(history=PRE_SAMPLES,order=16,lag=32),
    'wave_long_same_recurrence':dict(history=LONG_HISTORY,order=16,lag=32),
    'wave_long_wider_recurrence':dict(history=LONG_HISTORY,order=32,lag=64),
}


def forecast(past,length,*,order,lag):
    """Only the supplied past is fitted; the returned future is autonomous."""
    past=np.asarray(past,dtype=np.float64)
    if past.ndim!=1 or not np.isfinite(past).all() or length<1 or order<1 or lag<1:
        raise ValueError('invalid history or recurrence')
    lags=lag*np.arange(1,order+1)
    if len(past)<=lags[-1]+order:raise ValueError('insufficient past')
    rows=np.arange(lags[-1],len(past))
    matrix=past[rows[:,None]-lags]
    coefficients,_,rank,_=np.linalg.lstsq(matrix,past[rows],rcond=RCOND)
    wave=np.r_[past,np.zeros(length)]
    for sample in range(len(past),len(wave)):
        wave[sample]=wave[sample-lags]@coefficients
    if not np.isfinite(wave).all():raise FloatingPointError('nonfinite forecast')
    return wave[len(past):],dict(rank=int(rank),coefficient_l2=float(np.linalg.norm(coefficients)))


def evidence(wave):
    x=np.asarray(wave,dtype=np.float32)
    if x.shape!=(LONG_HISTORY+POST_SAMPLES,) or not np.isfinite(x).all():
        raise ValueError('wrong probe support')
    crop=x[LONG_HISTORY-PRE_SAMPLES:]
    spectral=_spectral_map_from_segment(crop).astype(np.float16)
    features,audit=native_decay_features(spectral[None])
    scores={'legacy_log_power':float(features.max())}
    diagnostics={'legacy_reliable_bands':int(audit['reliable'].sum())}
    # Identical normalization for all three waveform methods.
    scale=max(float(np.mean(x[LONG_HISTORY-PRE_SAMPLES:LONG_HISTORY].astype(float)**2)),1e-12)
    for name,cfg in CONFIGS.items():
        pred,info=forecast(x[LONG_HISTORY-cfg['history']:LONG_HISTORY],POST_SAMPLES,
                           order=cfg['order'],lag=cfg['lag'])
        scores[name]=float(np.sqrt(np.mean((x[LONG_HISTORY:].astype(float)-pred)**2)/scale))
        diagnostics[name]=info
    return scores,diagnostics


def make_wave(frequencies,phase,*,noise_db,event,seed,envelope='steady'):
    # Time origin matches the old 3072-sample probe, enabling exact comparison
    # on its final crop while providing earlier history to the new controls.
    t=(np.arange(LONG_HISTORY+POST_SAMPLES)-LONG_HISTORY+PRE_SAMPLES)/SAMPLE_RATE
    modes=[]
    for i,f in enumerate(frequencies):
        angle=2*np.pi*f*t+(phase if i else .3)
        amp=.2/(i+1)
        if envelope=='decaying':amp=amp*np.exp(-(3+i)*t)
        elif envelope=='vibrato':angle+=.25*np.sin(2*np.pi*5*t+phase)
        elif envelope=='glide':angle+=2*np.pi*40*t*t
        modes.append(amp*np.sin(angle))
    base=np.sum(modes,axis=0)
    wave=base.copy();relative=(np.arange(len(t))-LONG_HISTORY)/SAMPLE_RATE
    elapsed=np.maximum(relative-.005,0)
    attack=np.where(relative>=.005,(1-np.exp(-elapsed/.002))*np.exp(-elapsed/.15),0.)
    if event in ('new_weak','new_strong','same_pitch','two_notes'):
        fnew=[frequencies[0]] if event=='same_pitch' else ([164.8138,329.6276] if event=='two_notes' else [329.6276])
        strength=.04 if event=='new_weak' else .14
        for f in fnew:
            wave+=strength*attack*sum(np.sin(2*np.pi*f*h*elapsed+.7)/h for h in range(1,5))
    elif event=='noise_burst':
        wave+=.18*attack*np.random.default_rng(seed+10101).normal(size=len(t))
    elif event!='none':raise ValueError('unknown event')
    if noise_db is not None:
        rms=np.sqrt(np.mean(base[LONG_HISTORY-PRE_SAMPLES:LONG_HISTORY]**2))
        wave+=rms*10**(-noise_db/20)*np.random.default_rng(seed).normal(size=len(t))
    return wave.astype(np.float32)


def auc(positive,negative):
    a=np.asarray(positive)[:,None];b=np.asarray(negative)[None,:]
    return float(np.mean((a>b)+.5*(a==b)))


def summarize(records):
    result={}
    for db in ('clean','40','20'):
        selected=[r for r in records if r['noise_db']==db]
        negatives=[r for r in selected if r['event']=='none' and r['envelope'] in ('steady','decaying')]
        positives=[r for r in selected if r['event'] in ('new_weak','new_strong','same_pitch','two_notes')]
        stress=[r for r in selected if r['event']=='noise_burst' or r['envelope'] in ('vibrato','glide')]
        methods={}
        for method in records[0]['scores']:
            a=[r['scores'][method] for r in positives];b=[r['scores'][method] for r in negatives]
            s=[r['scores'][method] for r in stress]
            methods[method]=dict(development_auc=auc(a,b),
                development_auc_including_stress=auc(a,b+s),held_sound_median=float(np.median(b)),
                held_sound_max=float(np.max(b)),attack_min=float(np.min(a)),attack_median=float(np.median(a)),
                stress_no_new_note_median=float(np.median(s)),stress_no_new_note_max=float(np.max(s)),
                zero_scores_on_attacks=int(np.sum(np.asarray(a)==0)))
        result[db]=dict(held_sounds=len(negatives),attacks=len(positives),stress_no_new_note=len(stress),methods=methods)
    return result


def probe():
    families=[('single_220',[220]),('pair_220_233',[220,233.08188]),
              ('pair_110_116',[110,116.54094]),('harmonics_110',[110*h for h in range(1,9)]),
              ('pair_173_184',[173.2,184.7]),('inharmonic_137',[137*h*np.sqrt(1+.0002*h*h) for h in range(1,9)])]
    phases=np.linspace(0,2*np.pi,16,endpoint=False)
    records=[]
    for db in (None,40,20):
        for family,freqs in families:
            for i,phase in enumerate(phases):
                seed=91531+100*i+len(records)
                cases=[('steady',event) for event in ('none','new_weak','new_strong','same_pitch','two_notes')]
                cases += [('decaying','none')]
                if i%4==0:cases += [('vibrato','none'),('glide','none'),('steady','noise_burst')]
                for envelope,event in cases:
                    wave=make_wave(freqs,float(phase),noise_db=db,event=event,seed=seed,envelope=envelope)
                    scores,diagnostics=evidence(wave)
                    records.append(dict(family=family,phase_index=i,phase=float(phase),
                        frequencies_hz=freqs,noise_db='clean' if db is None else str(db),
                        envelope=envelope,event=event,seed=seed,
                        new_notes=2 if event=='two_notes' else int(event not in ('none','noise_burst')),
                        scores=scores,diagnostics=diagnostics))
            print(json.dumps(dict(noise_db=db,family=family,completed_cases=len(records))),flush=True)
    result=dict(status='completed',scope='synthetic development probe; no GuitarSet or K-network inference',
        parameters=CONFIGS,rcond=RCOND,sample_rate=SAMPLE_RATE,forecast_samples=POST_SAMPLES,
        uses_future_to_fit=False,uses_known_frequencies_to_predict=False,
        spectral_phase_available_to_legacy=False,raw_pcm_available_to_wave_predictors=True,
        full_v273_score=False,training_launched=False,automatic_promotion=False,
        numpy=np.__version__,script_sha256=digest(__file__),
        protocol='analysis/v273-coherent-decay-protocol.md',
        protocol_sha256=digest('analysis/v273-coherent-decay-protocol.md'),
        scores_have_different_units=True,summary=summarize(records),cases=records,
        limitations=[
            'Exploratory benchmark: configurations were developed on related synthetic oscillators; no independent validation.',
            'Waveform methods receive phase information absent from the old spectral features.',
            'The long wider model changes history, recurrence order, and lag; intermediate controls isolate only history.',
            'Scores measure innovation, never number of distinct notes or ownership of those notes.',
            'Ongoing pitch changes and nonmusical transients can also be unpredictable.',
            'No amplitude/phase estimator supplied from known generator parameters; fitting only past PCM.',
            'No runtime latency measurement; equal future sample support does not establish real-time performance.'])
    return result


def save_result(path,result):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    records=result.pop('cases')
    csv_path=path.with_name(path.stem+'-cases.csv')
    fields=['family','phase_index','phase','noise_db','envelope','event','seed','new_notes',
            'legacy_reliable_bands',*records[0]['scores']]
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in records:
            writer.writerow({**{k:row[k] for k in fields[:8]},
                'legacy_reliable_bands':row['diagnostics']['legacy_reliable_bands'],**row['scores']})
    result.update(case_count=len(records),cases_file=csv_path.name,cases_sha256=digest(csv_path),
        families={r['family']:r['frequencies_hz'] for r in records})
    write_json(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();save_result(args.output,probe())
