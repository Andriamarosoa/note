"""S851 external A/B/C: same original guitar and standard +12 as S848.

Check S850 BCE vs RANK trained with physical GuitarSet micro/JAMS on
INDIVIDUAL 256-hop streaming decisions. Compare to unchanged S848 standard
and earlier S847 one-note failure. Only CAL (guitarist 04) selects network
and silence threshold. External A/B cannot select weights or thresholds.

Audio output remains PLAYBACK of actual recorded GuitarSet note clips:
not the actual pitch-shifted original performance, not a synthesized tone.
No DSP frequency shifter in the IA; standard is offline only and unchanged.
"""
from __future__ import annotations

import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile

from onset256.s850_causal_polyphonic_transposer256 import (
  ABI,HOP,SR,LOW,NeuralPolyphonicTranspose256,
  CausalPolyphonicTransposeStream256)
from scripts.s847_train_real_recorded_transposer256 import read_recorded_bank
from scripts.s848_ia_vs_standard_pitchshift256 import (
  spec_profile,envelope_similarity,log_mel_distance)

def wav44(path):
    sr,x=wavfile.read(path)
    if sr!=SR or x.dtype!=np.int16 or x.ndim!=1:
        raise ValueError("S851 must compare exact S848 source and standard PCM16")
    return x.astype(np.float32)/32768.

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()

def play_shadow(model,original,bank,gate_threshold,*,reset_hops=0):
    """A real bank of authentic notes, *never* generated pitches/synthesis."""
    stream=CausalPolyphonicTransposeStream256(model,shift=12)
    notes=[]
    buffers=[]
    cursor=0
    previous=None
    active=[]
    for sample in range(0,len(original),HOP):
        if reset_hops and (sample//HOP)%reset_hops==0:
            stream.state=None  # reset past state, NOT delay or future audio
        x=np.zeros(HOP,np.float32)
        duration=min(HOP,len(original)-sample)
        x[:duration]=original[sample:sample+duration]
        predicted=stream.push(x,start_sample=sample)
        active.append(predicted.real_note_presence_probability)
        if predicted.real_note_presence_probability<gate_threshold:
            note=None
        else:
            note=max(bank,key=lambda p:float(predicted.midi_probabilities[p-LOW]))
        notes.append(note)
        if note!=previous:
            cursor=0
            previous=note
        if note is None:
            wave=np.zeros(HOP,np.float32)
        else:
            original_sample=bank[note]
            wave=original_sample[cursor:cursor+HOP]
            cursor=(cursor+HOP)%len(original_sample)
        buffers.append(wave[:duration].copy())
    return np.concatenate(buffers),notes,np.asarray(active)

def safe_corr(x,y):
    try:
        return envelope_similarity(x,y)
    except ValueError:return None

def result_for(model,original,standard,bank,gate_cut,*,reset_hops=0):
    audio,notes,gate=play_shadow(model,original,bank,gate_cut,reset_hops=reset_hops)
    histogram=Counter(str(p) for p in notes if p is not None)
    fraction=max(histogram.values())/len(notes) if histogram else 0
    non_silent=int(np.count_nonzero(audio))
    correlation=safe_corr(audio,standard)
    distance=log_mel_distance(audio,standard) if non_silent else None
    reasons=[]
    if fraction>.95:reasons.append("neural notes still stuck on one pitch >95% of frames")
    if correlation is None or correlation<.5:
        reasons.append("neural playback does not preserve original musical dynamics vs standard")
    if not non_silent:reasons.append("AI has no physical audio output")
    if distance is not None and distance>=18.3773:
        reasons.append("audio differs from standard at least as much as rejected S847")
    record=dict(
        confidence_gate_from_CAL_only=gate_cut,
        recurrent_state_reset_diagnostic_every_hops=reset_hops,
        output_all_256_audio_causal_no_midi=True,
        output_is_real_recorded_note_playback_not_faithful_input_transposition=True,
        source_future_pcm_used=0,source_added_hold=0,
        chosen_notes=dict(Counter(str(p) if p is not None else 'SILENCE' for p in notes)),
        largest_single_note_fraction=fraction,
        unique_physical_notes_selected=len(histogram),
        played_hops=int(sum(p is not None for p in notes)),
        complete_hops=len(notes),gate_mean=float(gate.mean()),
        audio_non_silent_samples=non_silent,
        envelope_correlation_to_standard=correlation,
        absolute_logmel_distance_to_standard_db=distance,
        predicted_audio_spectrum=spec_profile(audio),
        musical_promotion_passed=not bool(reasons),
        musical_rejection_reasons=reasons)
    return audio,record

def evaluate(source_wav,standard_wav,trained_results,model_dir,real_bank_dir,out):
    out=Path(out)
    if out.exists():raise FileExistsError("immutable S851 audit")
    input_wav=wav44(source_wav)
    standard=wav44(standard_wav)
    if len(input_wav)!=len(standard):
        raise ValueError("standard must have exactly same real original duration")
    training=json.loads(Path(trained_results).read_text())
    if (training.get('schema')!='S850_TRUE_GUITAR_CAUSAL_POLY_NOTE_TRAIN_RANK_AUDIT' or
        training.get('selected_variant_on_CAL') not in ('BCE','RANK')):
        raise ValueError("not trained S850 CAL-only model selection")
    bank,bankinfo=read_recorded_bank(real_bank_dir)
    torch.set_num_threads(1)
    report=dict(schema="S851_REAL_SOURCE_SAME_STANDARD_AI_PCM256_AUDIT",
        source_wav=str(source_wav),standard_wav=str(standard_wav),
        input_sha256=sha256(source_wav),standard_sha256=sha256(standard_wav),
        original_audio_measured=spec_profile(input_wav),
        standard_audio_measured=spec_profile(standard),
        reference_STANDARD_OFFLINE_NO_DELAY_CLAIM=True,
        S847_original_MIDI73_stuck_603_of_603=True,
        original_standard_vs_input_envelope_correlation=0.9038408365131547,
        S847_audio_vs_standard_envelope_correlation=0.11642527469600572,
        original_Onset_80_F1_unchanged=.6925843310420176,
        selected_variant_from_CAL_only=training['selected_variant_on_CAL'],
        development_WAV_cannot_override_CAL_choice=True,
        official_six_NOTE_axes_and_MIDI_unchanged=True,
        recorded_bank_training_source=bankinfo["source_guitarist"],
        variants={})
    out.mkdir(parents=True)
    for variant in ('BCE','RANK'):
        checkpoint=Path(model_dir)/('s850_'+variant.lower()+'_real_model.pt')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['abi']!=ABI or payload['trained_real_PCM_hops']!=256 or not payload['no_promotion']:
            raise ValueError("invalid shadow model")
        model=NeuralPolyphonicTranspose256()
        model.load_state_dict(payload["model_state"],strict=True)
        threshold=training["variant_results"][variant]["ranks"]["12"]["activity_threshold_CAL"]
        if threshold is None:threshold=1.0
        report['variants'][variant]={}
        for reset in (0,32):
            label='continuous' if reset==0 else 'reset32'
            audio,metrics=result_for(model,input_wav,standard,bank,threshold,
                                     reset_hops=reset)
            output=out/("s850_"+variant.lower()+"_"+label+"_real_recording_plus12.wav")
            wavfile.write(output,SR,
                          np.int16(np.clip(audio,-1,1-1/32768)*32768))
            report['variants'][variant][label]=metrics
    preferred=training['selected_variant_on_CAL']
    report['CAL_preferred_quality']=report['variants'][preferred]['continuous']
    report["selected_model_promoted"]=False
    report['all_model_outputs_are_real_clip_playback_not_true_guitar_transpose']=True
    (out/"resultat.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S851_TRUE_STANDARD_COMPARISON",json.dumps({
       "preferred_CAL":preferred,
       "BCE":report['variants']['BCE'],
       "RANK":report['variants']['RANK']},sort_keys=True),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--standard',type=Path,required=True)
    p.add_argument('--trained-results',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--real-bank-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    evaluate(a.source,a.standard,a.trained_results,a.model_dir,a.real_bank_dir,a.out)
