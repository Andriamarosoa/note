"""Original signed S708/H0 event preservation. Historical grid 1024.

ONLY original MIDI events from signed FIT80 metadata: not a new PCM256 model.
No inference from other axis predictions, no transcribed MIDI manipulation.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
SR=44100
def signed_onsets(events,notes,tolerance=.080):
    predict=[e for e in events if e["type"]=="note_on"]
    pairs=[]
    for pitch in range(40,89):
        pi=[i for i,e in enumerate(predict) if e["pitch"]==pitch]
        ti=[j for j,n in enumerate(notes) if n["pitch"]==pitch]
        if not pi or not ti:continue
        delta=np.abs(np.asarray([predict[i]["time"] for i in pi])[:,None]-
                     np.asarray([notes[j]["time"] for j in ti])[None,:])
        ii,jj=linear_sum_assignment(np.where(delta<=tolerance,delta,1e6))
        pairs.extend((pi[i],ti[j]) for i,j in zip(ii,jj) if delta[i,j]<=tolerance)
    return predict,pairs

def onset_score(events,notes):
    predicted,pairs=signed_onsets(events,notes)
    tp=len(pairs)
    return dict(tp=tp,fp=len(predicted)-tp,fn=len(notes)-tp)

