"""S506: actual neural 13-head note selector with leave-recording-out OOF.

Uses original TensorFlow ClassConditionalAuditSelector and its separate
KEEP / K0-K6 logits, attention by head/K and fix/regression risk outputs.
Each held recording is excluded from model fit AND source audit profiles.
Training rows originate only from SHA-frozen S505 genuine acoustic heads
paired to existing IDMT/MAESTRO note truth; no unpaired negative invented.

CORPUS CAVEAT: these 8 recordings were exposed in prior investigations.
Thus this trial is exploratory GROUP-HELD selector evaluation, NOT a
new prospective blind test. No full-stream false positives measurable.
No historical S80/V88/live/onset/offset/notes are modified or promoted.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train_only_audit(truthK,y,note,bheads,mask,subset,alpha=16.):
    """[13,21] from OTHER recordings: by TRUE-K fix,regression,coverage."""
    h=note.shape[1]; out=np.zeros((h,7,3),np.float32)
    subset=np.asarray(subset,int)
    for k in range(7):
        tr=subset[truthK[subset]==k]
        if not len(tr):continue
        ref=note[tr,bheads[tr]]
        correct=np.all(ref==y[tr],axis=1)
        denom=float(len(tr))+alpha
        for j in range(h):
            offered=note[tr,j]
            changed=np.any(offered!=ref,axis=1)&mask[tr,j]
            good=np.all(offered==y[tr],axis=1)
            out[j,k]=[np.sum(changed&~correct&good)/denom,
                      np.sum(changed&correct&~good)/denom,
                      np.sum(changed)/denom]
    return out.reshape(h,21)

def inputs(d,idx,audits):
    return dict(head_logits=d["head_logits"][idx].astype(np.float32),
        audit=audits,head_type=d["head_type"][idx].astype(np.float32),
        head_mask=d["head_mask"][idx].astype(bool),
        head_k_mask=d["head_k_mask"][idx].astype(bool),
        context=d["context"][idx].astype(np.float32),
        baseline=d["baseline"][idx].astype(np.float32))

def targets(truth,heads,baseline_id,eligible):
    n=len(truth)
    base=heads[np.arange(n),baseline_id]
    baseline_correct=np.all(base==truth,axis=-1)
    diff=np.any(heads!=base[:,None,:],axis=-1)&eligible
    head_correct=np.all(heads==truth[:,None,:],axis=-1)
    fix=diff&head_correct&(~baseline_correct[:,None])
    reg=diff&(~head_correct)&baseline_correct[:,None]
    action=np.full(n,7,np.int32)
    usable=fix.any(axis=1)
    action[usable]=truth.sum(axis=1)[usable].astype(np.int32)
    return action,np.stack([fix,reg],axis=-1).astype(np.float32)

def select(outputs,headsets,base_index,available,Kmask,guarded):
    from scipy.special import softmax
    action=softmax(outputs["action_logits"],axis=1)
    risk=np.asarray(outputs["head_risk"],float)
    kselect=np.asarray(outputs["class_selection_weights"],float)
    decision=np.full(len(action),-1,int)
    for i in range(len(action)):
        k=int(np.argmax(action[i]))
        if k==7:continue
        origin=headsets[i,int(base_index[i])]
        choices=[]
        for j in range(headsets.shape[1]):
            if j>=8:continue # non-pitch actions are not an actual note-set edit
            if (not available[i,j] or not Kmask[i,j,k]
                or np.array_equal(origin,headsets[i,j])
                or int(headsets[i,j].sum())!=k):
                continue
            correction=float(risk[i,j,0]); regression=float(risk[i,j,1])
            score=correction-regression+.25*float(kselect[i,j,k])
            choices.append((score,j,correction,regression))
        if not choices:continue
        score,j,correction,regression=max(choices,key=lambda v:(v[0],-v[1]))
        if guarded and not (float(action[i,k])>=.55
              and correction>=.60 and correction-regression>=.30):
            continue
        decision[i]=j
    return decision

def summarize(y,headsets,baseid,decisions,sourceK):
    origin=headsets[np.arange(len(y)),baseid]
    changed=np.stack([headsets[i,j] if j>=0 else origin[i]
                      for i,j in enumerate(decisions)])
    good0=np.all(origin==y,axis=1);good=np.all(changed==y,axis=1)
    ktrue=y.sum(axis=1)
    k0=origin.sum(axis=1);k1=changed.sum(axis=1)
    diff=np.any(origin!=changed,axis=1);poly=ktrue>=2
    def measures(m):
        return dict(n=int(m.sum()),
          BP_base_correct=int(np.sum(m&good0)),proposed_correct=int(np.sum(m&good)),
          fixes=int(np.sum(m&(~good0)&good&diff)),
          regressions=int(np.sum(m&good0&(~good)&diff)),
          wrong_to_wrong=int(np.sum(m&(~good0)&(~good)&diff)),
          edits=int(np.sum(m&diff)),
          source_original_ExactK_correct=int(np.sum(m&(sourceK==ktrue))),
          BP_original_K_correct=int(np.sum(m&(k0==ktrue))),
          candidate_K_correct=int(np.sum(m&(k1==ktrue))))
    return dict(all=measures(np.ones(len(y),bool)),poly=measures(poly),
        trueK={str(k):measures(ktrue==k) for k in range(7)})

def run(args):
    report=json.loads(args.report.read_text())
    need(not args.out.exists() and
         report["status"]=="S505_RETROSPECTIVE_REAL_EIGHT_RECORDINGS_HEAD_PITCH_SET_TRAIN_ROWS"
         and report["audio_proposal_truth_table_SHA256"]==sha(args.data)
         and report["historical_source_exposure_not_genuinely_blind"]
         and report["eight_original_recordings"]==8,
         "S506 requires verified immutable real eight-piece acoustic-note labels")
    with np.load(args.data,allow_pickle=False) as z:
        d={key:z[key].copy() for key in z.files}
    groups=d["recording_sha"].astype(str)
    source=d["corpus"].astype(str)
    y=d["truth_note_bits"].astype(bool)
    headsets=d["head_note_bits"].astype(bool)
    baseid=d["selected_BP_baseline_head"].astype(int)
    eligible=d["head_mask"].astype(bool)
    k_mask=d["head_k_mask"].astype(bool)
    sourceK=d["source_K"].astype(int)
    # S504 defaults to BP new K for both instruments; guitar scoring is
    # ACTIVE note K. Adapt observable baseline to the source-specific
    # existing audio head (never true MIDI notes).
    audio_baseK=headsets[np.arange(len(y)),baseid].sum(axis=1)
    d["baseline"]=np.eye(7,dtype=np.float32)[np.clip(audio_baseK,0,6)]
    truthK=y.sum(axis=1).astype(int)
    N=len(y)
    need(N==report["rows"] and len(set(groups))==8 and
         headsets.shape==(N,13,128) and k_mask.shape==(N,13,7)
         and np.array_equal(truthK,d["truth_K"]) and
         np.all((truthK>=1)&(truthK<=6)),
         "S506 note identity/K source integrity failure")
    # Ground truth enters ONLY training targets and audit profiles. It
    # never enters held-row model features or calibration thresholds.
    train_actions,risk_target=targets(y,headsets,baseid,eligible)
    import tensorflow as tf
    from scripts.learn_v273_neural_head_selector import (
         ClassConditionalAuditSelector,train_loss)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    outidx=np.full((N,2),-1,np.int16)
    observations=[]
    folds=[]
    for fold,held_recording in enumerate(sorted(set(groups))):
        te=np.flatnonzero(groups==held_recording)
        tr=np.flatnonzero(groups!=held_recording)
        need(len(te) and len(tr) and not set(groups[tr])&set(groups[te]),
             "S506 whole recording split leaked into train")
        # Every training row gets head audit on OTHER training recordings.
        cache={s:train_only_audit(truthK,y,headsets,baseid,eligible,
                      tr[groups[tr]!=s]) for s in set(groups[tr])}
        audit_train=np.stack([cache[groups[i]] for i in tr])
        audit_test=np.repeat(train_only_audit(truthK,y,headsets,baseid,eligible,
                                 tr)[None,:,:],len(te),axis=0)
        Xtr=inputs(d,tr,audit_train)
        Xte=inputs(d,te,audit_test)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(506+fold)
        model=ClassConditionalAuditSelector(hidden=24,context_dim=16)
        optimizer=tf.keras.optimizers.Adam(learning_rate=.0015,clipnorm=1.)
        losses=[]
        # Fixed budget predeclared and no early stopping using HELD song.
        for epoch in range(20):
            with tf.GradientTape() as tape:
                outputs=model(Xtr,training=True)
                loss=train_loss(outputs,train_actions[tr],risk_target[tr],
                                eligible[tr],risk_weight=.40)
                ce=tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.convert_to_tensor(train_actions[tr],tf.int32),
                    logits=outputs["action_logits"])
                rare=tf.where(train_actions[tr]==7,1.,2.)
                loss=loss+.25*tf.reduce_mean(ce*rare)
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            losses.append(float(loss.numpy()))
            need(np.isfinite(losses[-1]),"S506 network diverged")
        out={key:value.numpy() for key,value in model(Xte,training=False).items()}
        outidx[te,0]=select(out,headsets[te],baseid[te],eligible[te],k_mask[te],False)
        outidx[te,1]=select(out,headsets[te],baseid[te],eligible[te],k_mask[te],True)
        need(np.all((outidx[te]>=-1)&(outidx[te]<13)),"S506 invalid head action")
        folds.append(dict(fold=fold,held_recording_sha=held_recording,
            corpus=str(source[te[0]]),held_positive_event_count=len(te),
            train_positive_event_count=len(tr),train_recording_count=len(set(groups[tr])),
            training_only_audits_exclude_training_row_source=True,
            held_audits_only_include_other_recordings=True,
            held_recording_absent_from_training=True,
            fixed_epochs=20,last_loss=losses[-1],
            training_action_not_KEEP=int(np.sum(train_actions[tr]!=7)),
            exploratory_only=True))
        observations.extend(dict(canonical_event_id=str(d["event_id"][i]),
            source_recording_sha=held_recording,corpus=str(source[i]),
            selected_neural_head_ungated=int(outidx[i,0]),
            selected_neural_head_guarded=int(outidx[i,1]),
            old_baseline_head=int(baseid[i]),
            historical_true_pitches_available_only_after_inference=True)
            for i in te)
        print("S506 TRAINED HELD RECORDING",json.dumps(folds[-1]),flush=True)
    need(np.all(outidx>=-1),"S506 some original true source recording not evaluated")
    scores={}
    for corpus in ("guitar_S80","piano_V88"):
        idx=np.flatnonzero(source==corpus)
        src=report["by_source"][corpus]
        summary={name:summarize(y[idx],headsets[idx],baseid[idx],outidx[idx,i],sourceK[idx])
                  for i,name in enumerate(("ungated_diagnostic","fixed_guarded"))}
        for policy in summary.values():
            need(policy["all"]["BP_base_correct"]==
                  src["raw_BP_base_full_note_sets_correct"],
                 "S506 original source guitar/piano note-set baseline drift")
            policy["truth_positive_attacks_with_no_canonical_candidate"]=(
                 src["annotated_true_attacks_NOT_matched"])
            policy["all_annotated_positive_attacks_denominator"]=(
                 src["truth_original_positive_attacks"])
            policy["unknown_unpaired_predicted_births_not_counted_as_false"]=True
        scores[corpus]=summary
        print("S506 REAL OOF SCORES",json.dumps(dict(corpus=corpus,policies=summary)),flush=True)
    args.out.mkdir(parents=True)
    predictions=args.out/"S506_RECORDING_HELD_NEURAL_NOTE_SELECTOR_PREDICTIONS.jsonl"
    predictions.write_text("".join(json.dumps(o)+"\n" for o in observations))
    result=dict(status="S506_TRUE_NEURAL_13_HEADS_RECORDING_HELD_EXPLORATORY_NO_PROMOTION",
        actual_neural_model="ClassConditionalAuditSelector",
        eight_recordings_held_out_in_outer_selector_training=True,
        training_audits_crossfit_excluding_corresponding_training_recording=True,
        held_audits_derived_only_from_training_recordings=True,
        held_true_notes_never_used_in_training_or_thresholds=True,
        existing_pretrained_acoustic_heads_may_have_unknown_prior_exposure=True,
        historical_IDMT_MAESTRO_examples_previously_explored_NOT_independent_blind=True,
        no_per_class_or_source_performance_promotion=True,
        false_positive_on_continuous_stream_NOT_evaluable_without_true_negative_events=True,
        source_guitar_S80_K_active_vs_piano_V88_K_new_distinct=True,
        original_S80_V88_live_offsets_unchanged=True,no_promotion=True,
        hyperparams_predeclared=dict(epochs=20,learning_rate=.0015,hidden=24,
           risk_weight=.40,action_prob_guard=.55,fix_prob_guard=.60,risk_margin_guard=.30),
        songheld_folds=folds,by_source=scores,source_S505_sha256=sha(args.data),
        predictions_sha256=sha(predictions))
    (args.out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
def selftest():
    y=np.zeros((2,128),bool);y[0,52]=True;y[1,55]=True
    proposals=np.stack([np.stack([y[0],y[0]]),np.stack([y[0],y[1]])])
    action,risk=targets(y,proposals,np.array([0,0]),np.ones((2,2),bool))
    need(action.tolist()==[7,1] and risk[1,1,0]==1,
         "S506 training targets must supervise KEEP and actual note correction")
    print("S506 SELFTEST PASS trained note-set labels and correction-risk classification")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true")
    for field in ("data","report","out"):p.add_argument("--"+field,type=Path)
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        for field in ("data","report","out"):need(getattr(a,field)is not None,"S506 missing "+field)
        run(a)
