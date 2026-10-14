"""S333 preregistered audio-physical onset alignment for FUTURE, untouched XML.

The original GuitarSet training target requires new note attacks to
overlap candidate groups within 20ms, but IDMT XML nominal onsets can
precede measured acoustic transients. A model-neutral alternative
reference maps annotations to reproducible PCM acoustic peaks, NEVER to
S80/ref logits. This can only be an EXPLORATORY EVALUATION on later
untouched audio, and does not modify historical benchmark semantics.

This module itself reads no dataset and no annotations. The policy is
sealed by GitHub SHA before future XMLs are opened.
"""
from __future__ import annotations
import hashlib,json,argparse
from pathlib import Path
import numpy as np
from scripts.s330_exposed_audio_physics_V88_XML_timing import novelty,pick,SR
from scripts.yourmt3_exactk_common import assign_onsets
ABS_MAX_PEAK_DISAGREEMENT_MS=20.0
PHYSICS_WINDOW_MS=(-25.0,+90.0)
CONFIDENCE_MIN_CONTRAST=2.5
ORIGINAL_ASSIGNMENT_RADIUS_MS=20.0
def need(x,msg):
    if not x:raise ValueError(msg)
def audio_anchors(signal,xml_onsets_sec):
    """Return immutable acoustic rise-peak anchors; unknowns remain UNKNOWN."""
    r,rms,f,flux=novelty(signal)
    anchors=np.full(len(xml_onsets_sec),-1,np.int64)
    evidence=[]
    for k,t in enumerate(xml_onsets_sec):
        orig=int(round(float(t)*SR))
        q=pick(r,rms,orig)
        p=pick(f,flux,orig)
        accepted=bool(q and p and q['confident'] and p['confident']
           and abs(q['offset_ms']-p['offset_ms'])<=ABS_MAX_PEAK_DISAGREEMENT_MS)
        # The center of two independent peaks is an AUDIO transient-peak
        # reference, NOT a claim about first string displacement onset.
        if accepted:
            anchors[k]=int(round((q['sample']+p['sample'])/2))
        evidence.append(dict(xml_onset_sample=orig,
            RMS_peak_ms=(q['offset_ms'] if q else None),
            spectral_peak_ms=(p['offset_ms'] if p else None),
            both_confident=bool(accepted),
            acoustic_peak_anchor_sample=int(anchors[k]) if accepted else None))
    return anchors,evidence
def label_audio_anchored_attacks(audio,xml_onsets_sec,group_starts,group_ends):
    anchors,proof=audio_anchors(audio,xml_onsets_sec)
    confident=anchors>=0
    needs_unknown=int(np.count_nonzero(~confident))
    if confident.any():
        counts,assigned=assign_onsets(anchors[confident],group_starts,group_ends)
    else:
        counts=np.zeros(len(group_starts),np.int32)
        assigned=np.zeros(0,np.int64)
    # Fail closed: if annotated notes lack reliable acoustic anchor, even
    # adjacent apparent-zero groups cannot receive certified K0 labels.
    return counts,dict(original_XML_attacks=len(anchors),
        acoustic_peak_anchored=int(confident.sum()),
        unanchored_unknown=needs_unknown,
        anchored_nearest_V88_count=int((assigned>=0).sum()),
        anchored_V88_unmatched=int((assigned<0).sum()),
        unverified_negative_K0=True,
        all_original_XML_notes_exhaustively_verified=False),proof
def selftest():
    samples=SR*2
    x=np.zeros(samples,dtype=np.float64)
    onsets=[.3,.8,1.3]
    for t in onsets:
        n=int(round(t*SR))
        tt=np.arange(2300,dtype=np.float64)/SR
        x[n:n+2300]+=.22*np.sin(2*np.pi*440*tt)*np.exp(-55*tt)
    anchors,proof=audio_anchors(x,onsets)
    need(np.all(anchors>=0),'synthetic physically known plucks must produce anchors')
    errors=(anchors-np.rint(np.asarray(onsets)*SR))*1000/SR
    need(np.max(np.abs(errors))<20,
         'physical transient localization failed known synthetic attack timing')
    counts,stats,details=label_audio_anchored_attacks(x,onsets,anchors,anchors)
    need(counts.tolist()==[1,1,1] and
         stats['unanchored_unknown']==0 and stats['anchored_V88_unmatched']==0,
         'known real acoustic attacks must map one-to-one under original 20ms matcher')
    silence=np.zeros(SR,dtype=np.float64)
    missing,_=audio_anchors(silence,[.5])
    need(missing.tolist()==[-1],'no audio transient => do not fake true K0')
    print(json.dumps(dict(status="S333_PHYSICALLY_KNOWN_AUDIO_UNIT_TEST_PASS",
       known_attack_localization_ms=list(map(float,errors)),
       synthetic_negative_label_never_assigned=True)),flush=True)
def run(out):
    need(not out.exists(),'immutable prospective alignment precommit')
    selftest()
    result=dict(status='S333_PREXML_PHYSICAL_ATTACK_K_LABEL_POLICY_VERSION1',
      original_legacy_GuitarSet_XML_20ms_benchmark_kept_separate=True,
      audio_only_signal_processing='highpass80+RMS256 + FFT1024/hop256 log flux',
      acoustic_transient_anchor='mean of sample positions of independent RMS/spectral flux peaks',
      input_XML_window_ms=list(PHYSICS_WINDOW_MS),
      max_RMS_spectral_peak_disagreement_ms=ABS_MAX_PEAK_DISAGREEMENT_MS,
      peak_strength_contrast_threshold=CONFIDENCE_MIN_CONTRAST,
      group_label_K_definition='new physically anchored XML note attacks assigned within original 20ms V88 group interval',
      if_no_physical_audio_peak='ABSTAIN_UNKNOWN_NOT_K0',
      unanchored_XML_notes_must_not_be_K0=True,
      no_K0_negatives_claimed_without_independent_exhaustiveness=True,
      source_model_predictions_never_used_to_select_audio_anchors=True,
      experimental_not_historical_GuitarSet_K_target=True,
      test_data_XML_already_opened=False,
      no_training_or_threshold_fit_on_candidate_evaluation=True,
      no_S80_official_promotion=True)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    z=p.parse_args();run(z.out)
