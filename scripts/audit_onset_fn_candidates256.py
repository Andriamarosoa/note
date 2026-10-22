"""Falsifiable, fully causal Onset FALSE-NEGATIVE forensic audit on real PCM256.

Audit first, correction later. The 80 signed GuitarSet tracks are replayed
through the genuine SHA-pinned native256 S613/S701 weights. Ground truth is
not consulted until a WHOLE track has finished audio-only inference.

A separate low-threshold Onset candidate-proposal selector (NO trained head,
NO production route) runs on the same PCM256 frame without modifying S613.
It tests whether a future independently trained FN-rescue head would even
have observable candidates. An oracle-coverage count is not a measured gain.

This audit does not train, threshold-optimize or promote any new model.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from onset256.acoustics import HOP,SR
from onset256.selector import Selection,CausalSelection
from onset256.model import load
from onset256.runtime import Stream
from onset256.error_evidence69 import capture_selector_state,hard_matrix
from scripts.replay_native256 import match_onsets
from scripts.onset_harmonic_forensics256 import texture,register

SCHEMA="ONSET_CAUSAL_FN_FALSIFIABLE_AUDIT_PCM256_V1"
COMPOSITIONS=("BN1","Funk2","Jazz1","Rock3","SS1")
EXPECTED=dict(tp=8099,fp=4600,fn=3168)
# FIXED BEFORE INSPECTING THE SIGNED FN LABELS. No search on evaluation.
PROPOSAL_CONFIGS={
    "sensitive":dict(onset=.06,active_on=.10,onset_reset=.025),
    "moderate":dict(onset=.12,active_on=.20,onset_reset=.05),
    "cautious":dict(onset=.20,active_on=.30,onset_reset=.08),
}
HYPOTHESIS_LOW_ONSET_FRACTION=.50
SCORE_WINDOW_SECONDS=.080

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

class CapturingMain:
    """Pure read-only wrapper, never changes pretrained S613 decisions."""
    def __init__(self,original):
        self.original=original
        self.metadata=original.metadata
        self.calls=0
        self.p=None
        self.k=None

    def predict(self,features):
        p,k=self.original.predict(features)
        self.p=np.asarray(p,np.float32).copy()
        self.k=np.asarray(k,np.float32).copy()
        self.calls+=1
        return p,k

class CapturingRescue:
    """Pure read-only wrapper, preserves original S701 trained 39D ABI."""
    def __init__(self,original):
        self.original=original
        self.calls=0
        self.q=None
        self.history=None

    def predict_proba(self,context):
        probability=np.asarray(self.original.predict_proba(context),np.float32)
        self.q=probability[:,1].copy()
        self.history=np.asarray(context,np.float32).copy()
        self.calls+=1
        return probability

class FNExplorer(Stream):
    """Observe frozen 256 Onset + independently generated proposal candidates.

    No label or future signal enters _step, including the permissive selector.
    """

    def __init__(self,native,s701):
        self.main_view=CapturingMain(native)
        self.rescue_view=CapturingRescue(s701)
        super().__init__(self.main_view,self.rescue_view)
        base_config=self.base.config
        self.proposals={
            name:CausalSelection(Selection(
                onset=spec["onset"],active_on=spec["active_on"],
                onset_reset=spec["onset_reset"],
                active_off=base_config.active_off,
                release_frames=base_config.release_frames,
                refractory_frames=base_config.refractory_frames,
                max_notes=base_config.max_notes,
                count_confidence=base_config.count_confidence,
                confirmation_frames=1,
                restrike_onset=base_config.restrike_onset
            ),hop=HOP)
            for name,spec in PROPOSAL_CONFIGS.items()
        }
        self.proposed={name:[] for name in PROPOSAL_CONFIGS}
        self.proposal_evidence={name:[] for name in PROPOSAL_CONFIGS}
        self.frame_onset=[]
        self.frame_sustain=[]
        self.frame_rescue=[]
        self.frame_armed=[]
        self.frame_active=[]
        self.frame_refractory=[]
        self.frame_joint=[]
        self.frame_sample=[]
        self.overview_frames=0

    def _step(self,end,features):
        state=capture_selector_state(self.base,end)
        decision=super()._step(end,features)
        p=self.main_view.p
        k=self.main_view.k
        q=self.rescue_view.q
        h=self.rescue_view.history
        if self.main_view.calls!=self.calls or self.rescue_view.calls!=self.calls:
            raise AssertionError("S613/S701 wrapped observer changed 256 cadence")
        self.frame_onset.append(p[:,0].copy())
        self.frame_sustain.append(p[:,1].copy())
        self.frame_rescue.append(q.copy())
        self.frame_armed.append(state[:,1].copy())
        self.frame_active.append(state[:,0].copy())
        age=np.asarray(state[:,5],np.float32)
        self.frame_refractory.append(age.copy())
        self.frame_joint.append(((p[:,0]>=self.base.config.onset)&
                                (p[:,1]>=self.base.config.active_on)).copy())
        self.frame_sample.append(end)
        by_policy={}
        for name,selector in self.proposals.items():
            proposed=selector.step(p,k,end,end)
            new=[e for e in proposed if e["type"]=="note_on"]
            if new:
                by_policy[name]=new
                self.proposed[name].extend(new)
        if by_policy:
            # Save EXACT future FN head training/inference ABI from this very
            # decision boundary, never ground-truth or future audio.
            vectors=hard_matrix(features,p,k,q,h,state)
            for name,events in by_policy.items():
                self.proposal_evidence[name].extend(
                    vectors[int(e["pitch"])-40].copy() for e in events)
            self.overview_frames+=1
        return decision

    def close_proposals(self):
        for selector in self.proposals.values():
            selector.panic(self.received)
        self.finish(self.received)

def categorize(truth,base_decisions,explorer):
    """Post-inference only: inspect every ground-truth FN at ±80ms."""
    predicted=[a for d in base_decisions for a in d.intents]
    native_pairs=match_onsets(predicted,truth)
    if len(native_pairs)!=len(set(j for _,j in native_pairs)):
        raise AssertionError("native GT matching repeated an annotation")
    matched_gt={j for _,j in native_pairs}
    base=dict(tp=len(native_pairs),fp=len(predicted)-len(native_pairs),
              fn=len(truth)-len(native_pairs))
    names={name:[] for name in PROPOSAL_CONFIGS}
    for name,events in explorer.proposed.items():
        names[name]=[type("Proposal",(),{"pitch":int(e["pitch"]),
                        "sample":int(e["decision_sample"])})()
                    for e in events]
    gt_unmatched=[j for j in range(len(truth)) if j not in matched_gt]
    fn_truth=[truth[j] for j in gt_unmatched]
    captured={}
    captured_gt={}
    for name,candidates in names.items():
        matches=match_onsets(candidates,fn_truth)
        captured[name]=len(matches)
        captured_gt[name]={gt_unmatched[j] for _,j in matches}
        if len(matches)>len(fn_truth):
            raise AssertionError("candidate coverage over-count")
    ends=np.asarray(explorer.frame_sample,np.int64)
    onset=np.stack(explorer.frame_onset).astype(np.float32)
    sustain=np.stack(explorer.frame_sustain).astype(np.float32)
    q=np.stack(explorer.frame_rescue).astype(np.float32)
    armed=np.stack(explorer.frame_armed).astype(np.float32)
    active=np.stack(explorer.frame_active).astype(np.float32)
    refractory=np.stack(explorer.frame_refractory).astype(np.float32)
    joint=np.stack(explorer.frame_joint)
    if (ends.shape!=(explorer.calls,) or onset.shape!=(explorer.calls,49)
        or sustain.shape!=onset.shape or q.shape!=onset.shape):
        raise AssertionError("full predecision FN evidence not covered")
    threshold=float(explorer.base.config.onset)
    sustain_min=float(explorer.base.config.active_on)
    category_counts=Counter()
    by_texture=Counter()
    by_register=Counter()
    by_pitch=Counter()
    rows=[]
    for gi in gt_unmatched:
        note=truth[gi]
        pitch=int(note["pitch"])
        if not 40<=pitch<=88:
            raise ValueError("GT annotation outside verified MIDI range")
        time=float(note["time"])
        j=pitch-40
        low=np.searchsorted(ends,(time-SCORE_WINDOW_SECONDS)*SR,side="left")
        high=np.searchsorted(ends,(time+SCORE_WINDOW_SECONDS)*SR,side="right")
        if high<=low:
            category="no_frame_within_80ms"
            maximum=mean_q=peak_sustain=0.
            any_onset=any_joint=False
            was_armed=was_refractory=False
        else:
            on=onset[low:high,j]
            sus=sustain[low:high,j]
            resc=q[low:high,j]
            arm=armed[low:high,j]
            ref=refractory[low:high,j]
            maximum=float(on.max())
            peak_sustain=float(sus.max())
            mean_q=float(resc.max())
            any_onset=bool(np.any(on>=threshold))
            any_joint=bool(np.any(joint[low:high,j]))
            was_armed=bool(np.any(arm>0.5))
            was_refractory=bool(np.any(ref>=.999))
            if maximum<PROPOSAL_CONFIGS["moderate"]["onset"]:
                category="acoustic_onset_below_0_12"
            elif not any_onset:
                category="main_onset_below_calibrated_threshold"
            elif not np.any(sus>=sustain_min):
                category="sustain_below_main_threshold"
            elif not any_joint:
                category="onset_and_sustain_not_simultaneous"
            elif not was_armed:
                category="selector_not_armed"
            elif not was_refractory:
                category="selector_refractory"
            else:
                category="selector_budget_timing_or_assignment"
        category_counts[category]+=1
        by_register[register(pitch)]+=1
        by_pitch[pitch]+=1
        row=dict(index=gi,pitch=pitch,truth_time=time,
                 cause_evidence_category=category,
                 onset_max_in_gt80ms=maximum,
                 sustain_max_in_gt80ms=peak_sustain,
                 s701_quality_max_in_gt80ms=mean_q,
                 original_onset_threshold=threshold,
                 original_sustain_threshold=sustain_min,
                 is_armed_some_frame=was_armed,
                 refractory_elapsed_some_frame=was_refractory,
                 simultaneously_passes_main_gates=any_joint,
                 candidate_oracle_recoverable={
                     name:gi in captured_gt[name] for name in PROPOSAL_CONFIGS})
        rows.append(row)
    if sum(category_counts.values())!=base["fn"]:
        raise AssertionError("FN categorization neither exhaustive nor disjoint")
    # This fraction directly tests the pre-registered 'head weakness' claim.
    low_cats=("acoustic_onset_below_0_12",
              "main_onset_below_calibrated_threshold")
    low=sum(category_counts[c] for c in low_cats)
    prognosis=dict(
        target_fn=int(base["fn"]),
        below_main_onset_threshold=int(low),
        below_main_fraction=float(low/base["fn"]) if base["fn"] else 0.,
        threshold_model_limitation_supported=bool(
            base["fn"] and low/base["fn"]>=HYPOTHESIS_LOW_ONSET_FRACTION),
        causal_diagnosis_proven=False,
        category_counts=dict(category_counts),
        candidate_oracle_captured=captured,
        candidate_oracle_unreached={k:int(base["fn"]-v) for k,v in captured.items()},
        distinct_FN_covered_any_tier=sum(any(row["candidate_oracle_recoverable"].values())
                                         for row in rows),
        register_counts=dict(by_register),
        pitch_counts={str(k):int(v) for k,v in sorted(by_pitch.items())},
        cause_labels_are_descriptive_not_causal_proof=True)
    return base,prognosis,rows

def run(audio_zip,traces,manifest,native_model,s701_model,out):
    if out.exists():
        raise FileExistsError("FN audit artifacts cannot be overwritten")
    signature=json.loads(Path(manifest).read_text())
    if (signature.get("schema")!="S800_FIT80_OOF_PER_FRAGMENT_MIDI256_V1"
        or not signature.get("complete")
        or len(signature.get("fit_files",()))!=80
        or any(signature.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened"))):
        raise ValueError("signed FIT80 corpus does not match declared protocol")
    native,s701=load(native_model,s701_model)
    out.mkdir(parents=True)
    reports=[]
    totals=Counter()
    categorization=Counter()
    oracle=Counter()
    with (zipfile.ZipFile(audio_zip) as audio,
          (out/"false_negatives.jsonl").open("w") as fns,
          (out/"candidate_proposals.jsonl").open("w") as candidates,
          (out/"main_events.jsonl").open("w") as mains):
        for comp in COMPOSITIONS:
            items=sorted((v for v in signature["fit_files"]
                          if v["composition"]==comp),key=lambda v:v["index"])
            if len(items)!=16:
                raise ValueError("missing signed composition")
            for item in items:
                path=Path(traces)/f"fit_{item['index']:03d}.json"
                if sha256(path)!=item["json_sha256"]:
                    raise ValueError("signed trace SHA changed")
                metadata=json.loads(path.read_text())
                if metadata["name"]!=item["name"]:
                    raise ValueError("track identity mismatch")
                rate,pcm=wavfile.read(io.BytesIO(
                    audio.read(metadata["name"]+"_mic.wav")))
                if rate!=SR or pcm.dtype!=np.int16 or pcm.ndim!=1:
                    raise ValueError("audio must be ORIGINAL signed mono PCM16")
                waveform=pcm[:metadata["samples"]].astype(np.float32)/32768.
                explorer=FNExplorer(native,s701)
                decisions=[]
                for start in range(0,len(waveform),HOP):
                    observed=explorer.push(
                        waveform[start:start+HOP],start_sample=start)
                    if len(observed)!=(1 if start+HOP<=len(waveform) else 0):
                        raise AssertionError("not exactly one PCM256 Onset decision")
                    decisions.extend(observed)
                explorer.close_proposals()
                if (len(decisions)!=len(waveform)//HOP
                    or explorer.calls!=len(decisions)
                    or explorer.main_view.calls!=explorer.calls
                    or explorer.rescue_view.calls!=explorer.calls):
                    raise AssertionError("S613/S701 causal frame-count drift")
                if any(d.sample!=(i+1)*HOP for i,d in enumerate(decisions)):
                    raise AssertionError("Onset 256 decision time drift")
                baseline,classification,fn_rows=categorize(
                    metadata["fit_truth"],decisions,explorer)
                for k in ("tp","fp","fn"):totals[k]+=baseline[k]
                categorization.update(classification["category_counts"])
                oracle.update(classification["candidate_oracle_captured"])
                for row in fn_rows:
                    row.update(schema=SCHEMA,track=item["name"],
                               composition=comp,texture=texture(item["name"]))
                    fns.write(json.dumps(row,sort_keys=True)+"\n")
                for d in decisions:
                    for event in d.intents:
                        mains.write(json.dumps(dict(schema=SCHEMA,
                            track=item["name"],composition=comp,
                            pitch=int(event.pitch),sample=int(event.sample)),
                            sort_keys=True)+"\n")
                # Candidate vectors are generated from all audio-only frames
                # BEFORE any truth matching; only now attach GT-FN labels.
                unmatched_ids=set(classification["index"] for classification in fn_rows)
                fn_truth=[metadata["fit_truth"][gi] for gi in sorted(unmatched_ids)]
                idx_to_truth=sorted(unmatched_ids)
                for name,events in explorer.proposed.items():
                    predictions=[type("Candidate",(),dict(
                        pitch=int(e["pitch"]),sample=int(e["decision_sample"])))()
                        for e in events]
                    pairs=match_onsets(predictions,fn_truth)
                    matched={i:idx_to_truth[j] for i,j in pairs}
                    evidence=explorer.proposal_evidence[name]
                    if len(evidence)!=len(events):
                        raise AssertionError("rescue training evidence not event-aligned")
                    for i,e in enumerate(events):
                        candidates.write(json.dumps(dict(
                            hard_predecision_vector=evidence[i].astype(float).tolist(),
                            schema=SCHEMA,track=item["name"],composition=comp,
                            texture=texture(item["name"]),policy=name,
                            sample=int(e["decision_sample"]),
                            pitch=int(e["pitch"]),
                            label="recoverable_fn" if i in matched else "not_fn",
                            GT_fn_index=matched.get(i)),
                            sort_keys=True)+"\n")
                reports.append(dict(index=item["index"],name=item["name"],
                                    texture=texture(item["name"]),
                                    comp=comp,frames=len(decisions),
                                    native=baseline,fn=classification))
                if len(reports)%8==0:
                    print("ONSET_FN_CAUSAL_AUDIT_256_PROGRESS",
                          len(reports),flush=True)
    if (len(reports)!=80 or sum(r["frames"] for r in reports)!=306536
        or any(totals[k]!=EXPECTED[k] for k in EXPECTED)):
        raise AssertionError("native 256 S613 baseline / original WAV drift")
    fraction=(categorization["acoustic_onset_below_0_12"]+
              categorization["main_onset_below_calibrated_threshold"])/totals["fn"]
    report=dict(schema=SCHEMA,complete=True,axis="Onset",tracks=80,
                pcm256_frames=306536,training_hop_samples=256,
                inference_hop_samples=256,real_original_audio=True,
                signed_sha256_original_s613=sha256(native_model),
                signed_sha256_original_s701=sha256(s701_model),
                native=dict(totals),initial_fn=totals["fn"],
                low_onset_fraction=float(fraction),
                hypothesis_low_onset_gt_50pct=
                    bool(fraction>=HYPOTHESIS_LOW_ONSET_FRACTION),
                hypothesis_pre_registered_threshold=HYPOTHESIS_LOW_ONSET_FRACTION,
                fn_failure_classifications=dict(categorization),
                candidate_oracle_captured=dict(oracle),
                candidate_thresholds=PROPOSAL_CONFIGS,
                audit_is_descriptive_not_causal_confirmation=True,
                oracle_not_a_trained_model=True,
                not_a_real_live_f1_improvement=True,
                labels_only_after_complete_pcm_inference=True,
                fit80_teacher_train_overlap=True,
                full_independent_external_test=False,
                other_axes_unchanged=True,official_midi_unchanged=True,
                not_promoted=True,per_track=reports)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ONSET_FN_CAUSAL_256_SIGNED_80_FINAL",
          json.dumps({k:report[k] for k in ("tracks","pcm256_frames",
                "native","low_onset_fraction","hypothesis_low_onset_gt_50pct",
                "fn_failure_classifications","candidate_oracle_captured",
                "audit_is_descriptive_not_causal_confirmation")}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    for name in ("audio-zip","traces","manifest","native-model","s701-model","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.audio_zip,a.traces,a.manifest,a.native_model,a.s701_model,a.out)
if __name__=="__main__":main()
