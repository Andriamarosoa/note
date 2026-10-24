"""S850 repeatable TRAIN-only learned octave: BCE vs hard-negative ranking.

All training guitar PCM originates from two signed GuitarSet mic/JAMS
archives, not DSP-pitch shifted nor synthesized. Source and shifted *labels*
are derived offline from original annotated notes. No standard DSP teacher
inside the neural model. TRAIN guitarist 00..03, CAL 04, OUTER 05.

Evaluate two ranking strengths on exactly the same real data; choose candidate
using CAL only; OUTER cannot select weights, thresholds or architecture.
The F1 values are note activity per block, NOT NOTE's onset F1.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch

from onset256.s850_causal_polyphonic_transposer256 import (
    ABI,HOP,LOW,PITCHES,NeuralPolyphonicTranspose256,
    train_only_weights,real_note_loss,shifted_truth_only_offline)

SEED=850
VARIANTS={"BCE":0.,"RANK":.75}


def read_split(root,name):
    with np.load(Path(root)/(name+".npz"),allow_pickle=False) as z:
        x=z["pcm"].copy()
        gt=z["truth"].copy()
    if (x.ndim!=3 or x.shape[-1]!=HOP or x.dtype!=np.float32 or
        gt.shape!=(*x.shape[:2],PITCHES) or gt.dtype!=np.float32 or
        not np.isfinite(x).all() or not np.isfinite(gt).all() or
        not np.isin(gt,[0,1]).all()):
        raise ValueError("signed real GuitarSet PCM256/GT only")
    return dict(pcm=x,truth=gt)


def probabilities(model,data,shift,batch_size=24):
    model.eval()
    outs=[]
    with torch.inference_mode():
        for i in range(0,len(data["pcm"]),batch_size):
            x=torch.from_numpy(data["pcm"][i:i+batch_size])
            shifts=torch.full((len(x),),shift,dtype=torch.long)
            values,_=model(x,shifts)
            prob=torch.sigmoid(values["target_logits"]).cpu().numpy()
            presence=torch.sigmoid(values["target_activity_logits"]).cpu().numpy()[...,0]
            outs.append((prob,presence))
    return (np.concatenate([a for a,b in outs]),
            np.concatenate([b for a,b in outs]))


def select_threshold(prob,truth):
    if not np.any(truth):return None
    p=np.asarray(prob,np.float64).ravel()
    y=np.asarray(truth,bool).ravel()
    # Dedicated offline CAL-only selection, no OUTER thresholds.
    quantiles=np.quantile(p,np.linspace(0,1,101))
    candidates=np.unique(np.concatenate((quantiles,np.linspace(0,1,101),
                                          p[y],np.array([1.]))))
    if len(candidates)>3000:
        candidates=np.unique(np.quantile(candidates,np.linspace(0,1,250)))
    best=None
    for cut in candidates:
        pred=p>=cut
        tp=np.count_nonzero(pred&y)
        fp=np.count_nonzero(pred&~y)
        fn=np.count_nonzero(~pred&y)
        f1=(2*tp)/max(1,2*tp+fp+fn)
        rank=(f1,tp,-fp)
        if best is None or rank>best[0]:best=(rank,float(cut))
    return best[1]


def metrics(truth,prob,threshold):
    gt=np.asarray(truth,bool)
    if threshold is None:
        return dict(status="NO_POSITIVE_CAL",f1=None,tp=None,fp=None,fn=None)
    if not np.any(gt):
        return dict(status="NO_POSITIVE_EVAL",f1=None,tp=0,
                    fp=int(np.count_nonzero(prob>=threshold)),fn=0)
    p=prob>=threshold
    tp=int(np.count_nonzero(p & gt));fp=int(np.count_nonzero(p&~gt))
    fn=int(np.count_nonzero(~p & gt))
    return dict(status="MEASURED_FRAME_GT_NOT_ONSET",
                tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn),
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))


def shifted_labels(data,shift):
    return shifted_truth_only_offline(torch.from_numpy(data["truth"]),
       torch.full((len(data["truth"]),),shift,dtype=torch.long)).numpy()


def eval_cal_outer(model,cal,outer):
    result=dict()
    for shift in (-12,0,12):
        cp,cgate=probabilities(model,cal,shift)
        op,ogate=probabilities(model,outer,shift)
        cy=shifted_labels(cal,shift)>.5
        oy=shifted_labels(outer,shift)>.5
        pitches=(40,52) if shift==-12 else ((52,64) if shift==12 else (40,52))
        targets=dict()
        for pitch in pitches:
            idx=pitch-LOW
            cut=select_threshold(cp[...,idx],cy[...,idx])
            report=metrics(oy[...,idx],op[...,idx],cut)
            cal_report=metrics(cy[...,idx],cp[...,idx],cut)
            targets[str(pitch)]=dict(
                threshold_chosen_on_CAL_only=cut,
                cal_positive=int(np.sum(cy[...,idx])),
                outer_positive=int(np.sum(oy[...,idx])),
                CAL=cal_report,OUTER=report)
        # A true chord may activate several labels in a block. Never
        # suppress it by using argmax for the scientific accuracy score.
        valid=cy.sum(-1)>0
        if valid.any():
            correct=(np.take_along_axis(cy,cp.argmax(axis=-1)[...,None],axis=-1)[...,0])
            top1_cal=float(correct[valid].mean())
        else:top1_cal=None
        valid_outer=oy.sum(-1)>0
        if valid_outer.any():
            correct_outer=np.take_along_axis(oy,op.argmax(axis=-1)[...,None],axis=-1)[...,0]
            top1_outer=float(correct_outer[valid_outer].mean())
        else:top1_outer=None
        # Gate uses true existence, not volume or arbitrary transposed note.
        gate_cut=select_threshold(cgate,cy.any(-1))
        gate_outer=metrics(oy.any(-1),ogate,gate_cut)
        result[str(shift)]=dict(note_targets=targets,
                                CAL_top1_contains_GT_note=top1_cal,
                                OUTER_top1_contains_GT_note=top1_outer,
                                activity_threshold_CAL=gate_cut,
                                activity_gate_OUTER=gate_outer)
    return result


def choose_on_cal_only(stats):
    # All three shift targets equally represented in the selection score.
    chosen=[]
    for shift,pitch in ((-12,40),(0,40),(0,52),(12,52),(12,64)):
        entry=stats[str(shift)]["note_targets"][str(pitch)]["CAL"]
        if entry["f1"] is not None:chosen.append(entry["f1"])
    # A model scoring from predicting all notes will have low F1.
    return float(np.mean(chosen)) if chosen else 0.


def train(data_root,out,*,epochs=4,seed=SEED):
    out=Path(out)
    if out.exists():raise FileExistsError("immutable S850 evaluation")
    torch.set_num_threads(1)
    tr=read_split(data_root,"train")
    cal=read_split(data_root,"cal")
    outer=read_split(data_root,"outer")
    weights,counts=train_only_weights(torch.from_numpy(tr["truth"]))
    results=dict(schema="S850_TRUE_GUITAR_CAUSAL_POLY_NOTE_TRAIN_RANK_AUDIT",
                 sample_rate=44100,exact_new_samples_per_hop=HOP,
                 train_hops_per_sequence=tr["pcm"].shape[1],
                 no_future_in_network=True,added_latency_samples=0,
                 source_real_guitar_PCM_only=True,no_synth_audio_training=True,
                 truth_based_label_shift_only_OFFLINE=True,
                 training_players=["00","01","02","03"],cal_players=["04"],
                 outer_players=["05"],
                 official_midi_and_all_six_heads_unchanged=True,
                 original_Onset_reference_F1=.6925843310420176,
                 F1_note_activity_not_Onset=True,
                 classes_in_train=int((counts>0).sum()),
                 E2_train_hops=int(counts[0]),
                 E3_train_hops=int(counts[12]),
                 train_positive_weights_E2=float(weights[0]),
                 train_positive_weights_E3=float(weights[12]),
                 variant_results={},selected_variant_on_CAL=None,
                 promoted=False)
    out.mkdir(parents=True)
    for index,(label,ranking) in enumerate(VARIANTS.items()):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        model=NeuralPolyphonicTranspose256()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        epochs_done=[]
        best_model=None
        best_val=float('inf');best_epoch=0
        for epoch in range(epochs):
            model.train()
            losses=[]
            for batch in np.array_split(
                rng.permutation(len(tr["pcm"])),
                max(1,math.ceil(len(tr["pcm"])/24))):
                x=torch.from_numpy(tr["pcm"][batch])
                y=torch.from_numpy(tr["truth"][batch])
                shifts=torch.from_numpy(rng.choice([-12,0,12],len(batch)).astype(np.int64))
                loss,_=real_note_loss(model,x,y,shifts,weights,ranking=ranking)
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                opt.step()
                losses.append(float(loss.detach()))
            model.eval()
            validate=[]
            with torch.inference_mode():
                for batch in np.array_split(
                    np.arange(len(cal["pcm"])),max(1,math.ceil(len(cal["pcm"])/24))):
                    x=torch.from_numpy(cal["pcm"][batch])
                    y=torch.from_numpy(cal["truth"][batch])
                    shift=torch.full((len(batch),),12,dtype=torch.int64)
                    vl,_=real_note_loss(model,x,y,shift,weights,ranking=ranking)
                    validate.append(float(vl))
            mean_val=float(np.mean(validate))
            epochs_done.append(dict(epoch=epoch+1,
                                   train_loss=float(np.mean(losses)),
                                   CAL_loss=mean_val))
            print("S850_REAL_TRAIN_VARIANT",label,epoch+1,
                  "train",epochs_done[-1]["train_loss"],
                  "CAL",mean_val,flush=True)
            if mean_val<best_val:
                best_val=mean_val;best_epoch=epoch+1
                best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_model)
        model.eval()
        stats=eval_cal_outer(model,cal,outer)
        cal_f1=choose_on_cal_only(stats)
        torch.save(dict(abi=ABI,model_state=best_model,
                CAL_selected_epoch=best_epoch,rank_strength=ranking,
                trained_real_PCM_hops=HOP,no_promotion=True),
                   out/("s850_"+label.lower()+"_real_model.pt"))
        results["variant_results"][label]=dict(
             CAL_selection_score=cal_f1,CAL_checkpoint_epoch=best_epoch,
             epochs=epochs_done,ranks=stats)
        print("S850_VARIANT_TRUE_GUITAR_METRICS",label,
              json.dumps(dict(CAL_score=cal_f1,
                        shifted_plus12=stats["12"],
                        shifted_minus12=stats["-12"]),sort_keys=True),flush=True)
    results["selected_variant_on_CAL"]=max(
        results["variant_results"],
        key=lambda label:results["variant_results"][label]["CAL_selection_score"])
    results["winner_selected_before_OUTER_comparison"]=True
    (out/"resultat.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    print("S850_FINAL_CAL_SELECTED_VARIANT",
          results["selected_variant_on_CAL"],flush=True)
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=4)
    a=p.parse_args()
    train(a.data,a.out,epochs=a.epochs)
