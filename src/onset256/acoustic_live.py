"""Direct acoustic head integrated into the private Onset streaming output."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np

from .acoustic_attack import acoustic_matrix, AttackPolicy, DirectAttackSelector, DIM, SCHEMA
from .compiled_inference import accelerated, SharedEvidenceRisk
from .error_evidence69 import load_hard_model
from .event_confidence import load_event_head
from .fn_recovery import load_fn_recovery
from .live_event_stream import LowLatencyOnsetStream, OnsetDecision
from .model import load
from .runtime import Intent


class AcousticRecordingRisk(SharedEvidenceRisk):
    def predict_correct_probability(self,feat,p,k,rescue,history,state):
        q=super().predict_correct_probability(feat,p,k,rescue,history,state)
        self.acoustic_evidence=acoustic_matrix(feat,self.evidence)
        return q


def current_reference(fold,base_bundle,recovery_bundle,native_path):
    """Load the exact 68.370987% reference with its original hash contracts."""
    base,new=Path(base_bundle),Path(recovery_bundle)
    for root in (base,new):
        manifest=json.loads((root/'manifest.json').read_text())
        for filename,expected in manifest['files_sha256'].items():
            if hashlib.sha256((root/filename).read_bytes()).hexdigest()!=expected:
                raise ValueError('reference bundle SHA mismatch')
    native,s701=load(native_path,base/'s701.joblib')
    def signed(root,stem,loader):
        path=root/f'{stem}_{fold}.joblib'
        meta=json.loads(path.with_suffix('.json').read_text())
        if (meta['outer_composition']!=fold or fold in meta['train_compositions']
                or meta['cal_composition']==fold):
            raise ValueError('reference fold mismatch')
        return loader(path,path.with_suffix('.json'),expected_sha256=meta['checkpoint_sha256'])
    risk=signed(base,'risk',load_hard_model)
    recovery=signed(new,'recovery',load_fn_recovery)
    event=load_event_head(base/f'selected_{fold}.joblib',base/f'selected_{fold}.json',outer=fold)
    return LowLatencyOnsetStream(native,s701,risk,recovery,event,
                                 evidence_risk_class=AcousticRecordingRisk)


class AcousticOnsetStream:
    hop_samples=256
    added_hold_samples=0
    future_audio_samples=0
    def __init__(self,reference,model,policy):
        if model is not None and getattr(model,'n_features_in_',None)!=DIM:
            raise ValueError('direct acoustic model requires 195 features')
        if policy.enabled and model is None:
            raise ValueError('enabled direct head requires learned weights')
        self.reference=reference
        self.model=accelerated(model) if model is not None else None
        self.selector=DirectAttackSelector(policy)
        self.last_reference=None
        self.calls=0

    def push(self,pcm,*,start_sample):
        base=self.reference.push(pcm,start_sample=start_sample)
        self.last_reference=base
        q=(np.asarray(self.model.predict_proba(self.reference.risk.acoustic_evidence),
                      np.float32)[:,1] if self.model is not None else np.zeros(49,np.float32))
        pitches=self.selector.step(q,[e.pitch for e in base.intents],end_sample=base.sample)
        self.calls+=1
        if not self.selector.policy.enabled:return base
        return OnsetDecision(base.sample,tuple(Intent('note_on',pitch,base.sample,
            float(q[pitch-40]),'direct_acoustic_195d_pcm256') for pitch in pitches))

    def finish(self):
        self.reference.finish()
        return ()


def load_acoustic_stream(directory,*,fold,base_bundle,recovery_bundle,native_path):
    directory=Path(directory)
    meta=json.loads((directory/'frozen_model_and_policy.json').read_text())
    if (meta['schema']!=SCHEMA or meta['dim']!=DIM or meta['trained_hop_samples']!=256
            or meta['outer_composition']!=fold or fold in meta['train_compositions']
            or meta['cal_composition']==fold or not meta['frozen_before_outer']
            or meta['added_hold_samples']!=0 or meta['future_audio_samples']!=0):
        raise ValueError('direct head lineage/causality contract mismatch')
    selected=meta['selected'];policy=AttackPolicy(**selected['policy'])
    model=None
    if selected['name']!='reference':
        path=directory/f"acoustic_w{selected['width']}.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['checkpoint_sha256']:
            raise ValueError('direct head SHA mismatch')
        model=joblib.load(path)
    reference=current_reference(fold,base_bundle,recovery_bundle,native_path)
    return AcousticOnsetStream(reference,model,policy)
