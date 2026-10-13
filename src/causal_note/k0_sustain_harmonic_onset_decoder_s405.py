"""S405 — Sustain/harmonic presence is NOT an onset: S80 K0-safe MIDI event decoder.

S80 predicts exactly the count of NEW note attacks at each native event,
not the count of ringing/active notes. K0 means "no NEW NoteOn", even when
the waveform, frame evidence, and many harmonics remain strong.

This is a structural inference contract test, NOT an S404 model training
or quality claim. The onset and frame heads were not trained yet.
No thresholds in this module were calibrated against note annotations.

References: Andriamarosoa/midi polyphonic.decoder global per-pitch state;
the S404 architecture uses separate frame/onset/harmonic evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from src.causal_note.k_first_multihead_notes_s404 import (
    MAX_K,MIDI_MIN,NUM_MIDI,HARMONICS,select_exact_k_notes)


@dataclass(frozen=True)
class MidiEvent:
    action:str   # "note_on" or "note_off"
    midi:int
    event_index:int
    reason:str


class KOnsetStateDecoder:
    """One chronological stream. A NoteOn is driven by onset logits and S80 K,
    NEVER by frame/harmonic activation.

    Frames may prolong a PREVIOUSLY started note; sustained frames cannot
    independently manufacture a note when K0. A second onset of an already
    active same pitch retriggers it (off+on at the same event).
    """
    def __init__(self,off_frame_probability:float=0.35,
                 release_events:int=3):
        if not 0<off_frame_probability<1:
            raise ValueError("Release frame probability must be in (0,1)")
        if release_events<1 or isinstance(release_events,bool):
            raise ValueError("Release event count must be positive")
        self.off_threshold=float(off_frame_probability)
        self.release_events=int(release_events)
        self.active=np.zeros(NUM_MIDI,dtype=bool)
        self.inactive_votes=np.zeros(NUM_MIDI,dtype=np.int32)
        self.last_event_index=-1

    def reset(self)->None:
        """Must be called between pieces/streams to avoid sustain leakage."""
        self.active[:]=False
        self.inactive_votes[:]=0
        self.last_event_index=-1

    def step_from_s404(self,*,event_index:int,predicted_k:int,
                       model_output:dict[str,Tensor],
                       batch_index:int=0)->dict:
        """Directly consume S404 output heads; NO frame+onset summation.

        This is inference wiring, not training: S404 must already have
        separately trained onset/frame heads before qualitative performance
        on real audio can be claimed.
        """
        required=("onset_logits","frame_logits","harmonic_amplitude_logits")
        if not isinstance(model_output,dict) or not all(k in model_output for k in required):
            raise ValueError("Missing S404 onset/frame/harmonic output heads")
        if not isinstance(batch_index,int) or batch_index<0:
            raise ValueError("Invalid S404 batch index")
        onset=model_output["onset_logits"]
        frame=model_output["frame_logits"]
        harmonic=model_output["harmonic_amplitude_logits"]
        if onset.ndim!=2 or frame.shape!=onset.shape or            harmonic.ndim!=3 or onset.shape[1]!=NUM_MIDI or            harmonic.shape[1:]!=(NUM_MIDI,HARMONICS) or            not batch_index<onset.shape[0]:
            raise ValueError("S404 returned mismatched batch/head outputs")
        return self.step(
            event_index=event_index,predicted_k=predicted_k,
            onset_logits=onset[batch_index],
            frame_logits=frame[batch_index],
            harmonic_amplitude_logits=harmonic[batch_index]
        )

    @staticmethod
    def _vector(logits:Tensor,width:int,name:str)->Tensor:
        if not isinstance(logits,torch.Tensor) or logits.ndim!=1            or logits.shape[0]!=width or not bool(torch.isfinite(logits).all()):
            raise ValueError(f"Expected finite {width}-element {name} logits")
        return logits.detach().to(device="cpu",dtype=torch.float32)

    def step(self,*,event_index:int,predicted_k:int,
             onset_logits:Tensor,frame_logits:Tensor,
             harmonic_amplitude_logits:Tensor)->dict:
        if not isinstance(event_index,int) or event_index<=self.last_event_index:
            raise ValueError("Event indices must increase strictly within stream")
        if isinstance(predicted_k,bool) or not isinstance(predicted_k,int)            or not 0<=predicted_k<=MAX_K:
            raise ValueError("S80 K must be an integer 0..6")
        onset=self._vector(onset_logits,NUM_MIDI,"onset")
        frame=self._vector(frame_logits,NUM_MIDI,"sustain frame")
        harmonics=harmonic_amplitude_logits
        if not isinstance(harmonics,torch.Tensor) or            harmonics.shape!=(NUM_MIDI,HARMONICS) or            not bool(torch.isfinite(harmonics).all()):
            raise ValueError("Expected finite 49 x 20 harmonic evidence")
        # Event ranking derives ONLY from NEW-ONSET head and K.
        # Harmonics and strong sustained notes must NOT create new pitches.
        chosen=select_exact_k_notes(
            onset.unsqueeze(0),torch.tensor([predicted_k],dtype=torch.long)
        )[0]
        new_note_indices=(chosen[chosen>=0]-MIDI_MIN).to(dtype=torch.long).tolist()
        previous=self.active.copy()
        notes=[]
        midi_events=[]
        probabilities=torch.sigmoid(frame).numpy()

        # Existing notes remain audible despite K0 until measured frame
        # evidence falls below release threshold for multiple native events.
        for idx in np.flatnonzero(self.active):
            if float(probabilities[idx])<self.off_threshold:
                self.inactive_votes[idx]+=1
            else:
                self.inactive_votes[idx]=0
            if self.inactive_votes[idx]>=self.release_events:
                self.active[idx]=False
                self.inactive_votes[idx]=0
                midi_events.append(MidiEvent(
                    "note_off",int(MIDI_MIN+idx),event_index,"sustain_release"))

        for idx in new_note_indices:
            pitch=MIDI_MIN+idx
            if self.active[idx]:
                midi_events.append(MidiEvent(
                    "note_off",int(pitch),event_index,"new_attack_retrigger"))
            self.active[idx]=True
            self.inactive_votes[idx]=0
            midi_events.append(MidiEvent(
                "note_on",int(pitch),event_index,"S80_new_attack"))
            notes.append(int(pitch))

        self.last_event_index=event_index
        if len(notes)!=predicted_k:
            raise AssertionError("Onset note count must equal S80 K")
        if predicted_k==0 and any(e.action=="note_on" for e in midi_events):
            raise AssertionError("K0 must never emit a fresh note_on")

        # Diagnostics available even when K0: energy and harmonics can exist
        # independently from newly created note attacks.
        harmonic_presence=torch.sigmoid(harmonics.detach().cpu()).numpy()
        ringing_midi=(np.flatnonzero(self.active)+MIDI_MIN).tolist()
        return dict(
            K_new_attacks=int(predicted_k),
            new_note_midi=notes,
            active_note_midi=[int(v) for v in ringing_midi],
            events=midi_events,
            frame_probabilities=probabilities,
            harmonic_amplitude_probabilities=harmonic_presence,
            incoming_active_notes=int(np.count_nonzero(previous)),
            # K0 does NOT force either frame or harmonic predictions to zero.
            harmonic_presence_nonzero=bool(np.any(harmonic_presence>0.5)),
        )


def _logits(high:list[int],*,hi:float=8.,low:float=-8.)->Tensor:
    a=torch.full((NUM_MIDI,),float(low))
    for pitch in high:a[pitch-MIDI_MIN]=float(hi)
    return a


def _harmonics(high_pitch:int)->Tensor:
    a=torch.full((NUM_MIDI,HARMONICS),-7.0)
    a[high_pitch-MIDI_MIN,:10]=7.0
    return a


def self_test():
    torch.set_num_threads(2)
    # Tests use synthetic logits to demonstrate architecture invariants.
    # Passing them says NOTHING about note-detection accuracy on GuitarSet.
    d=KOnsetStateDecoder(off_frame_probability=.35,release_events=2)
    c3,e3,c4=48,52,60
    harmonic=_harmonics(c3)
    # First physical strike produces C3. C4 is loud as a harmonic and
    # must not be introduced by the separate frame/harmonic stream.
    one=d.step(event_index=0,predicted_k=1,
       onset_logits=_logits([c3]),
       frame_logits=_logits([c3,c4]),
       harmonic_amplitude_logits=harmonic)
    assert one["new_note_midi"]==[c3]
    assert one["active_note_midi"]==[c3]
    assert one["harmonic_presence_nonzero"]
    # K0: still loud C3 and its C4 octave, even artificially high onset
    # evidence. S80 has forbidden a *new event*, NOT resonance.
    zero=d.step(event_index=1,predicted_k=0,
       onset_logits=_logits([c4],hi=14.),
       frame_logits=_logits([c3,c4],hi=14.),
       harmonic_amplitude_logits=harmonic)
    assert zero["new_note_midi"]==[]
    assert zero["active_note_midi"]==[c3]
    assert zero["harmonic_presence_nonzero"]
    assert not any(e.action=="note_on" for e in zero["events"])
    assert float(zero["frame_probabilities"][c4-MIDI_MIN])>.99
    # C3 is still ringing, but a truly new E3 is attacked: K1 and E3.
    two=d.step(event_index=2,predicted_k=1,
       onset_logits=_logits([e3],hi=13.),
       frame_logits=_logits([c3,e3,c4]),
       harmonic_amplitude_logits=harmonic)
    assert two["new_note_midi"]==[e3]
    assert two["active_note_midi"]==[c3,e3]
    # Even silence releases NOTES OFF but never creates notes with K0.
    decay=d.step(event_index=3,predicted_k=0,
       onset_logits=_logits([]),
       frame_logits=_logits([e3]),
       harmonic_amplitude_logits=harmonic)
    assert decay["active_note_midi"]==[c3,e3]
    decay=d.step(event_index=4,predicted_k=0,
       onset_logits=_logits([]),
       frame_logits=_logits([e3]),
       harmonic_amplitude_logits=harmonic)
    assert decay["active_note_midi"]==[e3]
    assert any(e.action=="note_off" and e.midi==c3 for e in decay["events"])
    assert not any(e.action=="note_on" for e in decay["events"])
    # Retriggering an active pitch is still a NEW attack event.
    repeat=d.step(event_index=5,predicted_k=1,
       onset_logits=_logits([e3]),
       frame_logits=_logits([e3]),
       harmonic_amplitude_logits=harmonic)
    assert repeat["new_note_midi"]==[e3]
    assert [e.action for e in repeat["events"]]==["note_off","note_on"]
    assert repeat["active_note_midi"]==[e3]
    # Multiple simultaneous new attacks obey cardinality despite
    # additional ringing harmonic evidence.
    more=d.step(event_index=6,predicted_k=2,
       onset_logits=_logits([48,55]),
       frame_logits=_logits([c3,e3,c4,55]),
       harmonic_amplitude_logits=harmonic)
    assert set(more["new_note_midi"])=={48,55}
    assert len([e for e in more["events"] if e.action=="note_on"])==2
    d.reset()
    fresh=d.step(event_index=0,predicted_k=0,
       onset_logits=_logits([c3]),
       frame_logits=_logits([c3,c4]),
       harmonic_amplitude_logits=harmonic)
    assert not fresh["active_note_midi"] and fresh["new_note_midi"]==[]
    # End-to-end model-output wiring, without reading/using labels.
    from src.causal_note.k_first_multihead_notes_s404 import KFirstMultiHeadNoteNet
    torch.manual_seed(405)
    net=KFirstMultiHeadNoteNet(dimension=32,heads=4,layers=1).eval()
    with torch.no_grad():
        generated=net(torch.rand(1,42,NUM_MIDI)*.8,
                      torch.tensor([0],dtype=torch.long))
    connected=KOnsetStateDecoder().step_from_s404(
        event_index=0,predicted_k=0,model_output=generated,batch_index=0)
    assert connected["new_note_midi"]==[]
    assert not any(e.action=="note_on" for e in connected["events"])
    assert connected["frame_probabilities"].shape==(NUM_MIDI,)
    assert connected["harmonic_amplitude_probabilities"].shape==(NUM_MIDI,HARMONICS)
    print("S405 PASS: K0 does not silence resonances or emit NoteOn; "
          "K1 with old sustain, K2 multiple attacks, releases, retriggers,"
          " harmonics cannot create notes and each piece resets its state")


if __name__=="__main__":
    self_test()
