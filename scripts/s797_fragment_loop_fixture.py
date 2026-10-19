"""Deterministic S797 synthetic fixture; never use this for real score claims."""
import numpy as np
from causal_note.strict_live import SR
from causal_note.s795_activek_experts import HEADS,FEATURES,decode
from causal_note.s797_fragment_router import route_probabilities

def example():
    d=np.array([1024,2048,3072],np.int64)
    X=np.zeros((3,49,FEATURES),np.float32)
    P={name:np.zeros((3,49),np.float32) for name in HEADS}
    X[:2,0,12]=1
    X[:2,:,15]=1/6
    X[:2,:,13]=3/6
    fixtures={42:("bass",.14,.10),55:("polyphonic",.15,.10),
              60:("transient",.29,.10),52:("harmonic",.07,.72)}
    for pitch,(specialist,rise,parent) in fixtures.items():
        ix=pitch-40
        X[0,ix,0]=.8;X[0,ix,1]=.75;X[0,ix,2]=.03
        X[0,ix,5]=rise;X[0,ix,6]=.15;X[0,ix,10]=parent
        P["general"][0,ix]=.98
        P[specialist][0,ix]=.99
    original=[
        dict(type="note_on",pitch=40,available_sample=1024,
             decision_sample=1024,time=1024/SR,reason="native",velocity=96),
        dict(type="note_off",pitch=40,available_sample=3072,
             decision_sample=3072,time=3072/SR,reason="native",velocity=0)]
    return d,X,P,original

def trial(mode):
    d,X,P,original=example()
    routed,masks,meta=route_probabilities(X,P,mode)
    events,changes=decode(original,d,X,routed,31,.93,3072)
    attacks={e["pitch"] for e in events
             if e["reason"]=="s795_multihead_attack"}
    return attacks,masks,meta,events,changes,original
