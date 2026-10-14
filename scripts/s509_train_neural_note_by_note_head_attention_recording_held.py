"""S509: real neural PER-PITCH selection across 13 musical evidence heads.

The previous S506 whole-head attention cannot combine E3 from head A with
G3 from head B. Train a shared trainable head-attention PER pitch on each
candidate MIDI frequency observed by at least one available S505 head.

For every one of 8 recording-held folds, the held recording's MIDI labels
are absent from training, thresholds, and inputs. No pitches outside the
acoustically observed union can be predicted. This is RETROSPECTIVE:
these source recordings already appeared in earlier studies and upstream
BasicPitch historical training exposure is not independently known.

Two FIXED policies, not tuned on held labels:
  ungated p(note) >= 0.50 (diagnostic)
  guarded: add note only if p>=0.85; remove BP note only if p<=0.15;
           otherwise KEEP existing note state; avoid empty outputs.
No trained S80/Live production change and no false event score from
unpaired, unknown native candidate onsets.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def need(c,m):
    if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def features(d):
    """Return per-observed-MIDI-pitch rows with model-only observable features."""
    proposals=d["head_note_bits"].astype(bool)
    source=d["corpus"].astype(str)
    mask=d["head_mask"].astype(bool)
    types=d["head_type"].astype(np.float32)
    ctx=d["context"].astype(np.float32)
    baseline=d["selected_BP_baseline_head"].astype(int)
    true=d["truth_note_bits"].astype(bool)
    X=[];C=[];indices=[];labels=[];origins=[];pitches=[]
    for i in range(len(proposals)):
        allowed=proposals[i]&mask[i,:,None]
        acoustic=np.flatnonzero(allowed.any(axis=0))
        bp=proposals[i,baseline[i]]
        need(len(acoustic)>0,"S509 observed canonical musical event has no acoustic note")
        for pitch in acoustic:
            vote=allowed[:,pitch].astype(np.float32)
            # S505 copies S504's globally observable 16D context [N,16].
            # Repeat the SAME measured acoustic context across specialists;
            # head type, eligibility and per-pitch votes distinguish them.
            acoustic_context=np.repeat(ctx[i][None,:],13,axis=0)
            H=np.concatenate([acoustic_context,types[i],
                              mask[i,:,None].astype(np.float32),
                              vote[:,None]],axis=1)
            lo=acoustic[acoustic<pitch]
            is_harmonic=any(int(pitch)-int(lower) in (12,19,24) for lower in lo)
            # Recorded provenance of source (guitar/piano) is observable,
            # not the true note-set. The two datasets have distinct label
            # conventions for active vs newly born pitches.
            gl=np.concatenate([ctx[i],
              np.array([pitch/127.,float(pitch%12)/11.,
                float(source[i]=="guitar_S80"),float(bp[pitch]),
                len(acoustic)/12.,float(is_harmonic)],np.float32)])
            need(H.shape==(13,22) and gl.shape==(22,),
                 "S509 original 13 head audio feature contract incorrect")
            X.append(H);C.append(gl)
            indices.append(i);pitches.append(int(pitch));labels.append(bool(true[i,pitch]))
            origins.append(bool(bp[pitch]))
    x=np.asarray(X,np.float32);globalctx=np.asarray(C,np.float32)
    labels=np.asarray(labels,np.float32)
    need(np.isfinite(x).all() and np.isfinite(globalctx).all(),
         "S509 missing external pretrained head-audio feature values")
    return dict(head=x,global_context=globalctx,
                event_index=np.asarray(indices,int),
                pitch=np.asarray(pitches,int),
                truth=labels,baseline=np.asarray(origins,bool))

def compile_model():
    import tensorflow as tf
    class PerPitchHeadAttention(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.proj=tf.keras.layers.Dense(32,activation="gelu")
            self.att=tf.keras.layers.MultiHeadAttention(num_heads=2,key_dim=16)
            self.norm=tf.keras.layers.LayerNormalization()
            self.gate=tf.keras.layers.Dense(1)
            self.globalproj=tf.keras.layers.Dense(24,activation="gelu")
            self.mix=tf.keras.layers.Dense(32,activation="gelu")
            self.logits=tf.keras.layers.Dense(1)
        def call(self,data,training=False):
            z=tf.convert_to_tensor(data["head"],tf.float32)
            context=tf.convert_to_tensor(data["global_context"],tf.float32)
            mask=z[:,:,20]>.5
            encoded=self.proj(z)
            attmask=tf.logical_and(mask[:,:,None],mask[:,None,:])
            x=self.norm(encoded+self.att(encoded,encoded,
                   attention_mask=attmask,training=training))
            gate=tf.squeeze(self.gate(x),axis=-1)
            gate=tf.where(mask,gate,tf.constant(-1.e9,tf.float32))
            w=tf.nn.softmax(gate,axis=1)
            state=tf.reduce_sum(x*w[:,:,None],axis=1)
            global_state=self.globalproj(context)
            return tf.squeeze(self.logits(self.mix(tf.concat(
                [state,global_state],axis=-1))),axis=-1)
    return PerPitchHeadAttention()

def reassemble(d,data,index,probabilities,guarded):
    N=len(d["truth_note_bits"])
    baseline=d["selected_BP_baseline_head"].astype(int)
    current=d["head_note_bits"][np.arange(N),baseline].copy().astype(bool)
    if not guarded:current[:]=False
    for q,prob in zip(index,probabilities):
        ev=int(data["event_index"][q]);p=int(data["pitch"][q])
        if guarded:
            if current[ev,p] and prob<=.15:current[ev,p]=False
            elif not current[ev,p] and prob>=.85:current[ev,p]=True
        elif prob>=.5:current[ev,p]=True
    # Avoid emitting a spurious K0 for an observed canonical event:
    # uncertain all-zero decisions abstain to the original BP note set.
    empty=~current.any(axis=1)
    old=d["head_note_bits"][np.arange(N),baseline]
    current[empty]=old[empty]
    return current

def performance(truth,base,new,sourceK):
    o=np.all(base==truth,axis=1)
    p=np.all(new==truth,axis=1)
    change=np.any(base!=new,axis=1)
    K=truth.sum(1);poly=K>=2
    baseK=base.sum(1);nextK=new.sum(1)
    def metric(mask):
        return dict(n=int(mask.sum()),
          baseline_full_chord_correct=int(np.sum(mask&o)),
          predicted_full_chord_correct=int(np.sum(mask&p)),
          corrected=int(np.sum(mask&~o&p&change)),
          regressed=int(np.sum(mask&o&~p&change)),
          wrong_to_wrong=int(np.sum(mask&~o&~p&change)),
          changes=int(np.sum(mask&change)),
          baseline_ExactK=int(np.sum(mask&(baseK==K))),
          new_ExactK=int(np.sum(mask&(nextK==K))),
          original_source_ExactK=int(np.sum(mask&(sourceK==K))),
          correctly_recovered_pitches=int(np.sum((new&truth)[mask])),
          original_true_pitches=int(np.sum(truth[mask])),
          emitted_pitches=int(np.sum(new[mask])))
    return dict(overall=metric(np.ones(len(K),bool)),
                poly=metric(poly),
                per_trueK={str(k):metric(K==k) for k in range(7)})

def run(args):
    r=json.loads(args.report.read_text())
    need(not args.out.exists() and
       r["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
       and r["audio_proposal_truth_table_SHA256"]==sha(args.data)
       and r["eight_original_recordings"]==8
       and r["historical_source_exposure_not_genuinely_blind"],
       "S509 original audio/pitch training source SHA and exposed source provenance mismatch")
    with np.load(args.data,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    events=len(d["truth_note_bits"])
    recordings=d["recording_sha"].astype(str)
    src=d["corpus"].astype(str)
    true=d["truth_note_bits"].astype(bool)
    old=d["head_note_bits"][np.arange(events),
           d["selected_BP_baseline_head"].astype(int)].astype(bool)
    note=features(d)
    need(len(np.unique(recordings))==8 and
         events==r["rows"] and len(note["truth"])>=events,
         "S509 8 recordings and candidate note set coverage changed")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    prob=np.zeros(len(note["truth"]),np.float32)
    report_folds=[]
    args.out.mkdir(parents=True)
    for fold,heldgroup in enumerate(sorted(set(recordings))):
        train_events=np.flatnonzero(recordings!=heldgroup)
        test_events=np.flatnonzero(recordings==heldgroup)
        tr=np.flatnonzero(np.isin(note["event_index"],train_events))
        te=np.flatnonzero(np.isin(note["event_index"],test_events))
        need(len(train_events) and len(test_events) and len(tr) and len(te) and
             not set(recordings[train_events])&set(recordings[test_events]),
             "S509 one recording is in both model training and target evaluation")
        model_input_train={key:note[key][tr] for key in ("head","global_context")}
        model_input_test={key:note[key][te] for key in ("head","global_context")}
        Y=tf.constant(note["truth"][tr],tf.float32)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(509+fold)
        model=compile_model()
        opt=tf.keras.optimizers.Adam(learning_rate=.002,clipnorm=1.)
        npos=int(Y.numpy().sum());nneg=len(Y)-npos
        # Class balancing is chosen using TRAINING labels only.
        positive_weight=float(min(2.,max(1.,nneg/max(npos,1))))
        for epoch in range(24):
            with tf.GradientTape() as tape:
                logits=model(model_input_train,training=True)
                ce=tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=logits)
                weights=tf.where(Y>.5,positive_weight,1.)
                loss=tf.reduce_mean(ce*weights)
            grad=tape.gradient(loss,model.trainable_variables)
            opt.apply_gradients(zip(grad,model.trainable_variables))
        checkpoint=args.out/f"S509_fold{fold}_REAL_PITCH_ATTENTION.weights.h5"
        model.save_weights(checkpoint)
        out=model(model_input_test,training=False).numpy()
        prob[te]=tf.math.sigmoid(out).numpy()
        report_folds.append(dict(fold=fold,held_recording_sha=heldgroup,
            held_events=len(test_events),training_events=len(train_events),
            held_pitch_candidates=len(te),training_pitch_candidates=len(tr),
            train_positive_note_candidates=npos,
            train_negative_note_candidates=nneg,
            training_only_class_weight=positive_weight,
            fixed_epochs=24,
            trained_checkpoint_sha256=sha(checkpoint),
            evaluation_recording_excluded_from_training=True))
        print("S509 NEURAL NOTE FOLD",json.dumps(report_folds[-1]),flush=True)
    need(np.all(np.isfinite(prob)),"S509 per-pitch head selector missing real held predictions")
    selectors={}
    for guarded,name in ((False,"per_pitch_diagnostic"),(True,"per_pitch_guarded")):
        reconstructed=reassemble(d,note,np.arange(len(prob)),prob,guarded)
        source_summaries={}
        for corpus in ("guitar_S80","piano_V88"):
            ix=np.flatnonzero(src==corpus)
            m=performance(true[ix],old[ix],reconstructed[ix],d["source_K"][ix].astype(int))
            ref=r["by_source"][corpus]
            need(m["overall"]["baseline_full_chord_correct"]==
                 ref["raw_BP_base_full_note_sets_correct"],
                 "S509 original BP baseline note set changed")
            m["missing_true_attacks_without_predicted_note_group"]=(
                ref["annotated_true_attacks_NOT_matched"])
            m["all_true_annotated_attacks"]=ref["truth_original_positive_attacks"]
            m["cannot_score_unknown_native_audio_as_false_notes"]=True
            source_summaries[corpus]=m
        selectors[name]=source_summaries
        print("S509 NEURAL PER PITCH VERDICT",json.dumps(dict(
            policy=name,guitar=source_summaries["guitar_S80"]["overall"],
            piano=source_summaries["piano_V88"]["overall"],
            guitar_poly=source_summaries["guitar_S80"]["poly"],
            piano_poly=source_summaries["piano_V88"]["poly"])),flush=True)
    candidates=args.out/"S509_HELD_RECORDING_NOTE_PITCH_PROBABILITIES.jsonl"
    candidates.write_text("".join(json.dumps(dict(
         canonical_event_id=str(d["event_id"][int(note["event_index"][j])]),
         original_recording_SHA=str(recordings[int(note["event_index"][j])]),
         real_acoustic_candidate_MIDI_pitch=int(note["pitch"][j]),
         learned_probability=float(prob[j]),
         ground_truth_not_used_to_choose_this_probability=True))
         +"\n" for j in range(len(prob))))
    rep=dict(status="S509_REAL_13_HEAD_ATTENTION_NOTE_BY_NOTE_EIGHT_RECORDING_OOF",
       truly_trained_per_pitch_neural_attention=True,
       no_true_note_K_or_correctness_in_held_model_inputs=True,
       eight_holdout_recording_models_fitted=True,
       original_K_new_and_K_active_distinguished_by_source=True,
       one_audio_observed_note_at_most_once_per_musical_group=True,
       no_pitch_outside_existing_acoustic_head_union=True,
       fixed_policies_thresholds_not_calibrated_on_held_truth=True,
       exposed_historical_source_not_independent_blind=True,
       no_verified_unpaired_native_negatives_for_false_alarms=True,
       original_S80_V88_live_offset_onset_untouched=True,no_promotion=True,
       hyperparameters=dict(epochs=24,lr=.002,head_attention_heads=2,
                            diagnostic_probability=.5,guarded_add=.85,
                            guarded_remove=.15),
       folds=report_folds,policies=selectors,
       source_S505_sha256=sha(args.data),
       probability_table_sha256=sha(candidates))
    (args.out/"report.json").write_text(json.dumps(rep,indent=2)+"\n")

def selftest():
    # Two heads can each contribute distinct real notes absent from the
    # first baseline. A single whole-head selection would be insufficient.
    d=dict(head_note_bits=np.zeros((1,2,128),bool),
           selected_BP_baseline_head=np.array([0]),
           truth_note_bits=np.zeros((1,128),bool))
    d["head_note_bits"][0,0,52]=True
    d["head_note_bits"][0,1,55]=True
    rows=dict(event_index=np.array([0,0]),pitch=np.array([52,55]))
    z=reassemble(d,rows,np.array([0,1]),np.array([.92,.94]),True)
    need(set(np.flatnonzero(z[0]))=={52,55},
         "S509 two specialist heads must combine E3 and G3 without hallucinated pitch")
    print("S509 SELFTEST PASS different heads can select E3 and G3 independently")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for k in ("data","report","out"):p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for k in ("data","report","out"):need(getattr(a,k)is not None,"S509 missing --"+k)
        run(a)
