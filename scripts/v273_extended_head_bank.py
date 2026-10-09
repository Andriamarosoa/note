"""Audited adapter contract for historical V27.3 heads in the dynamic scheduler.

This registry is an EXECUTABLE routing interface, not proof that every
historical hypothesis already has a validated standalone predictor.

H9 (YourMT3+ event representation) is deliberately not a registered head.
Actual H8 pitch-shift needs aligned AUDIO-TRANSFORMED predictions; do not
substitute a shuffled frequency vector or fictional pitch shift.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

K_COUNT=7
AB_ACTIONS=('A','BA','STOP')
FEATURE_HEADS=(
 'H1_spectral','H2_lifecycle','H3_harmonics','H4_fundamentals',
 'H5_sources','H6_log_energy_flow','H7_attack_morphology',
 'H8_pitch_shift','H10_energy_source_coupling',
)
CORRECTORS=(('C23',2,3),('C32',3,2),('C34',3,4),('C43',4,3))
KEEP_HEADS=('F_keep2','F_keep3','F_keep4','F_keep_any')
ADDITIONAL=('E12_second_note',)
ALL_HEADS=FEATURE_HEADS+tuple(v[0] for v in CORRECTORS)+KEEP_HEADS+ADDITIONAL
assert len(ALL_HEADS)==18 and 'H9' not in ''.join(ALL_HEADS)

@dataclass(frozen=True)
class Head:
    key: str
    kind: str
    evidence: str
    eligible_k: tuple[int,...]
    oof_required: bool=True

REGISTRY=(
 Head('H1_spectral','predictor','58 summary audio',tuple(range(7))),
 Head('H2_lifecycle','predictor','58 summary+42-frame morphology',tuple(range(7))),
 Head('H3_harmonics','predictor','245 log harmonic activation moments',tuple(range(7))),
 Head('H4_fundamentals','predictor','49 spectral candidate bins pre+on+birth',tuple(range(7))),
 Head('H5_sources','predictor','candidate votes+source timeflow',tuple(range(7))),
 Head('H6_log_energy_flow','predictor','positive 49-bin temporal flux',tuple(range(7))),
 Head('H7_attack_morphology','predictor','42-frame attack/decay/width metrics',tuple(range(7))),
 Head('H8_pitch_shift','predictor','externally validated transformed-audio predictions',tuple(range(7))),
 Head('H10_energy_source_coupling','predictor','damping and energy-transport proxies, NOT Navier-Stokes solution',tuple(range(7))),
 *(Head(name,'correction',f'transition K{a}->K{b}',(a,))
   for name,a,b in CORRECTORS),
 *(Head(name,'keep',f'preserve correct source K{name[-1]}', (int(name[-1]),))
   for name in KEEP_HEADS[:3]),
 Head('F_keep_any','keep','preserve any S18 source K',tuple(range(7))),
 Head('E12_second_note','predictor','other-piece-trained original K1/K2 acoustic expert',(1,2))
)
assert tuple(h.key for h in REGISTRY)==ALL_HEADS

def inspect_registry():
    assert len(set(ALL_HEADS))==18
    assert all(v in tuple(range(7)) for h in REGISTRY for v in h.eligible_k)
    assert not any('yourmt' in (h.key+h.evidence).lower() for h in REGISTRY)
    assert not any(h.key=='H9' for h in REGISTRY)
    return {h.key:dict(kind=h.kind, evidence=h.evidence,
            eligible_source_K=list(h.eligible_k),oof_required=h.oof_required)
            for h in REGISTRY}

def waveform_pitch_source(raw,identities):
    """Require evidence from actual transformed WAV, not label-derived proxies.

    The external provenance is part of the contract. No misaligned partial
    B_low 488-case pitch experiment may masquerade as complete native OOF.
    """
    if raw is None:
        return None
    n=len(identities)
    ids=np.asarray(raw['global_index'])
    values=np.asarray(raw['class_probs'],np.float32)
    if not np.array_equal(ids,np.asarray(identities)):
        raise ValueError('H8: transformed waveform outputs not aligned with native event IDs')
    if values.shape!=(n,7) or not np.isfinite(values).all():
        raise ValueError('H8: expected finite (N,7) K0-K6 probabilities')
    if np.max(np.abs(values.sum(1)-1))>1e-4 or (values<0).any():
        raise ValueError('H8: transformed WAV scores must be normalized')
    origin=str(np.asarray(raw['source_method']).item())
    if origin not in ('librosa.effects.pitch_shift', 'pyrubberband.pitch_shift'):
        raise ValueError('H8: no certified waveform pitch-shift method')
    if 'fit_fold_exclusions' not in raw:
        raise ValueError('H8: missing OOF/fold-exclusion proof')
    trained=np.asarray(raw['fit_fold_exclusions'])
    if trained.shape!=(n,) or not np.all(trained):
        raise ValueError('H8: held-fold labels might have leaked')
    return values

def audio_temporal_views(sequence):
    """Genuine 42-frame ×49 harmonic trajectory; no true K, no model labels."""
    z=np.asarray(sequence,np.float32)
    if z.ndim!=3 or z.shape[1:]!=(42,49) or not np.isfinite(z).all():
        raise ValueError('42×49 original trajectory required')
    amp=np.expm1(z)
    if (amp<0).any():
        raise ValueError('negative physical amplitudes')
    eps=1e-5
    # Real early/late bins and frame-to-frame derivatives (not feature
    # reshuffling). Negative and positive energy transitions are separate.
    early=amp[:,:13]
    attack=amp[:,13:24]
    late=amp[:,24:]
    flux=np.diff(amp,axis=1)
    positive=np.maximum(flux,0.)
    negative=np.maximum(-flux,0.)
    peak=np.max(amp,axis=1)
    peakpos=np.argmax(amp,axis=1)/41.
    centered=(amp-amp.mean(axis=1,keepdims=True))
    dt0=(amp[:,1:]-amp[:,:-1])
    attack_width=np.sum(amp>peak[:,None,:]*.5,axis=1)/42.
    stats=[
        np.log1p(early.mean(axis=1)),
        np.log1p(attack.mean(axis=1)),
        np.log1p(late.mean(axis=1)),
        np.log1p(np.maximum(attack.mean(axis=1)-early.mean(axis=1),0)),
        np.log1p(positive.sum(axis=1)),
        np.log1p(negative.sum(axis=1)),
        peakpos,
        attack_width,
        np.log1p(np.sqrt((centered**2).mean(axis=1))),
        np.log1p(np.maximum(dt0.max(axis=1),0)),
    ]
    # 10 metrics × 49 pitch-harmonic candidates.
    morphology=np.concatenate(stats,axis=1).astype(np.float32)
    # Conservation/dissipation inspired descriptors — physical proxies,
    # not a numerical Navier–Stokes solver.
    gain=positive.sum(axis=1)
    loss=negative.sum(axis=1)
    physics=np.concatenate([
       np.log1p(gain),np.log1p(loss),
       np.log1p(np.abs(gain-loss)),
       gain/(gain+loss+eps),
       np.log1p(late.mean(axis=1)),
       np.log1p(early.mean(axis=1)),
    ],axis=1).astype(np.float32)
    if morphology.shape!=(len(z),490) or physics.shape!=(len(z),294):
        raise ValueError('temporal evidence dimensions drifted')
    return morphology,physics

def bank_feature_views(x,morphology,physics):
    """Identifiable head views; H8 cannot fabricate missing transformed audio."""
    x=np.asarray(x,np.float32)
    n=len(x)
    if x.shape!=(n,335) or not np.isfinite(x).all():
        raise ValueError('source vote/audio/flow feature tensor must be Nx335')
    morphology=np.asarray(morphology,np.float32)
    physics=np.asarray(physics,np.float32)
    if morphology.shape!=(n,490) or physics.shape!=(n,294):
        raise ValueError('unaligned time-domain spectral morphology')
    # Position of the first 49 bins for pre/on/birth from 5×49 source flow.
    pre=x[:,90:139]; onset=x[:,139:188];birth=x[:,237:286]
    return {
      'H1_spectral':x[:,32:90],
      'H2_lifecycle':np.column_stack([x[:,32:90],morphology[:,:196]]),
      'H3_harmonics':x[:,90:335],
      'H4_fundamentals':np.column_stack([pre,onset,birth]),
      'H5_sources':np.column_stack([x[:,:32],x[:,90:335]]),
      'H6_log_energy_flow':x[:,286:335],
      'H7_attack_morphology':morphology,
      'H10_energy_source_coupling':physics,
      'E12_second_note':x,
    }

def action_candidates(parent,expert_probs,head_available=None):
    """Head proposals and *structural* per-event masks; never use true K.

    Probabilities for 9 acoustic experts must be genuine prefit OOF model
    outputs. Cxx are action adapters, F_keep protect KEEP decisions.
    """
    parent=np.asarray(parent,np.int64)
    n=len(parent)
    if not np.isin(parent,np.arange(7)).all():raise ValueError('bad parent K')
    allowed=set(FEATURE_HEADS).union({'E12_second_note'})
    if not set(expert_probs).issubset(allowed):
        raise ValueError('unknown specialist (including forbidden H9)')
    proposal=np.tile(parent[:,None],(1,len(ALL_HEADS))).astype(np.int8)
    confidence=np.zeros((n,len(ALL_HEADS)),np.float32)
    mask=np.zeros((n,len(ALL_HEADS)),bool)
    names={key:i for i,key in enumerate(ALL_HEADS)}
    for key in FEATURE_HEADS:
        if key not in expert_probs:continue
        p=np.asarray(expert_probs[key],np.float32)
        if p.shape!=(n,7) or not np.isfinite(p).all():
            raise ValueError('bad OOF predictor output for '+key)
        j=names[key];v=p.argmax(1)
        proposal[:,j]=v
        confidence[:,j]=p[np.arange(n),v]
        mask[:,j]=True
    # Correctors use actual acoustic class posteriors; no fake new logits.
    # They can suggest only their destination K when current K is source.
    for key,source,target in CORRECTORS:
        j=names[key]
        evidence=[]
        for h in ('H1_spectral','H3_harmonics'):
            if h in expert_probs:evidence.append(expert_probs[h][:,target])
        if not evidence:continue
        strength=np.mean(evidence,axis=0)
        proposal[:,j]=target
        confidence[:,j]=strength
        mask[:,j]=(parent==source)
    for k in (2,3,4):
        j=names[f'F_keep{k}']
        proposal[:,j]=parent
        confidence[:,j]=1.
        mask[:,j]=(parent==k)
    j=names['F_keep_any']
    confidence[:,j]=1.
    mask[:,j]=True
    if 'E12_second_note' in expert_probs:
        q=np.asarray(expert_probs['E12_second_note'],np.float32)
        proposal[:,names['E12_second_note']]=np.where(q[:,1]>=q[:,2],1,2)
        confidence[:,names['E12_second_note']]=np.maximum(q[:,1],q[:,2])
        mask[:,names['E12_second_note']]=np.isin(parent,[1,2])
    if head_available is not None:
        for key,value in head_available.items():
            if key not in names:raise ValueError('unknown availability name')
            mask[:,names[key]] &= np.asarray(value,bool)
    return proposal,confidence,mask

def selftest():
    n=8
    z=np.zeros((n,42,49),np.float32)
    z[:,13:17,3]=1
    morph,physical=audio_temporal_views(z)
    x=np.zeros((n,335),np.float32)
    feat=bank_feature_views(x,morph,physical)
    assert len(feat)==9 and len(REGISTRY)==18
    assert feat['H7_attack_morphology'].shape==(n,490)
    assert feat['H10_energy_source_coupling'].shape==(n,294)
    assert feat['H6_log_energy_flow'].shape==(n,49)
    assert 'H8_pitch_shift' not in feat
    parent=np.array([0,1,2,3,4,5,6,2])
    good=np.tile(np.eye(7)[parent][:,None,:],(1,1,1)).reshape(n,7)
    p,c,m=action_candidates(parent,{'H1_spectral':good,
                                   'H3_harmonics':good})
    assert p.shape==c.shape==m.shape==(n,18)
    assert not m[:,ALL_HEADS.index('H8_pitch_shift')].any()
    assert not m[0,ALL_HEADS.index('C23')]
    assert m[2,ALL_HEADS.index('C23')]
    assert m[3,ALL_HEADS.index('C32')]
    assert m[4,ALL_HEADS.index('C43')]
    assert m[:,ALL_HEADS.index('F_keep_any')].all()
    try:action_candidates(parent,{'H9':good})
    except ValueError:pass
    else:raise AssertionError('H9 explicitly excluded but reached routing')
    try:waveform_pitch_source({'global_index':np.arange(n),
        'class_probs':good,'source_method':np.array('fake')},np.arange(n))
    except ValueError:pass
    else:raise AssertionError('synthetic non-waveform H8 incorrectly accepted')
    print('PASS: 18 interfaces, H9 completely excluded, H8 safely unavailable, structural K masks, true 42-frame morphology and dissipation proxies')

if __name__=='__main__':
    selftest()
