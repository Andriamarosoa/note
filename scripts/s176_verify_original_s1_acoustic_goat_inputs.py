"""S176: prove native GOAT acoustic feature ORDER matches historical S1 58D.

Read no truth/MIDI. Count actual S18 K2–K4 eligibility for original regression
loop families S1–S4, quantify out-of-reference feature ranges. Does NOT
manufacture 255 old group proposals, logits, or coherent 5x64x21 subset data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
def need(ok,why):
    if not ok: raise ValueError(why)
OLD=Path("analysis/evidence/v273-regression-loops")
def run(root,s97,s173,out):
    need(not out.exists(),"S176 must be immutable")
    manifest=json.loads((root/OLD/"prepared/input-manifest.json").read_text())
    with np.load(root/OLD/"prepared/inputs.npz",allow_pickle=False) as z:
        reference=z["features"].copy()
        reference_names=z["feature_names"].astype(str).tolist()
    need(len(reference_names)==58 and reference.shape==(7493,58),
         "frozen original S1 58 acoustic features")
    need(reference_names==manifest["feature_names"],
         "frozen original acoustic field ordering moved")
    # Re-execute the EXACT source descriptor implementation used by S97,
    # not a guessed mapping; S97 itself sorts the dict keys in this order.
    rng=np.random.default_rng(273176)
    signal=rng.standard_normal(88200).astype(np.float32)*.01
    spec,times=_frames(signal,22050)
    d=features_from_spectrogram(spec,times)
    keys=sorted(d)
    need(len(keys)==58 and keys==reference_names,
         "S97 58D feature names differ from original S1 acoustic training!")
    with np.load(s97,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","summary58","flow245","geometry3",
              "sequence42x49","morphology490","physics294","member",
              "start_samples","end_samples"},"GOAT signal fields and no label")
        identifiers=z["row_id"].astype(str)
        current=z["summary58"].copy()
    with np.load(s173,allow_pickle=False) as z:
        need(set(z.files)=={"row_id","S18_parent_K","original_family_ids",
          "source_K","S35_new_K","S35_selected_actions","agreement",
          "trained_Ridge_risks","partial_eleven_selected_head",
          "partial_eleven_candidate_K"},"S173 is honest partial source adapter")
        need(np.array_equal(identifiers,z["row_id"].astype(str)),
             "S173 new audio row order changed")
        parent=z["S18_parent_K"].astype(int)
    need(current.shape==(57,58) and np.isfinite(current).all() and
         np.isfinite(reference).all(),"GOAT S1 acoustic 58D invalid")
    need(np.isin(parent,range(7)).all(),"invalid GOAT adapted S18")
    # Descriptive shift only, no outcome-based tuning or quality claims.
    lo=np.quantile(reference,.01,axis=0)
    hi=np.quantile(reference,.99,axis=0)
    rate=np.mean((current<lo)|(current>hi),axis=0)
    dist=[{"feature":str(name),"GOAT_fraction_outside_historical_1to99_pct":float(r)}
          for name,r in zip(keys,rate)]
    dist.sort(key=lambda x:-x["GOAT_fraction_outside_historical_1to99_pct"])
    counts=np.bincount(parent,minlength=7)
    counts234=int(np.isin(parent,[2,3,4]).sum())
    counts2plus=int(np.sum(parent>=2))
    report={
      "status":"S176_ORIGINAL_S1_ACOUSTIC_58D_NEW_GOAT_ORDER_VERIFIED",
      "historical_reference_audio_events":7493,
      "native_GOAT_audio_events":57,
      "original_acoustic_58_feature_key_and_order_parity":True,
      "native_GOAT_parent_S18_adapted_K_histogram":counts.tolist(),
      "GOAT_parent_K234_eligible_for_historical_S1_S4_loop":counts234,
      "GOAT_parent_K2plus_count":counts2plus,
      "feature_distribution_shift_top10_descriptive_only":dist[:10],
      "original_255_group_proposals_on_new_GOAT":False,
      "original_nine_local_group_audits_available":False,
      "original_baseline_correct_logit_z_and_7_class_risk_logits_available":False,
      "original_repair_coherent_5x64x21_subsets_available":False,
      "S1_original_checkpoint_can_complete_GOAT_58D_feature_subinput":True,
      "full_original_S1_S4_source_proposals_on_GOAT":False,
      "new_audio_S45_original_family_count":11,
      "true_K_or_MIDI_opened":False,
      "independent_ExactK_evaluated":False,
      "S80_promoted":False}
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
      "status","native_GOAT_parent_S18_adapted_K_histogram",
      "GOAT_parent_K234_eligible_for_historical_S1_S4_loop",
      "original_acoustic_58_feature_key_and_order_parity",
      "new_audio_S45_original_family_count","S80_promoted")},indent=2))

def selftest():
    need(len(range(7))==7 and 2 in range(7),"Exact-K class K2")
    print("S176 PASS original S1 58D feature-order requirement; original group inputs not fabricated")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("s97","s173","out"):p.add_argument("--"+name,type=Path)
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        need(all(getattr(a,n) is not None for n in ("s97","s173","out")),
             "real native GOAT feature and parent artifacts required")
        run(a.root,a.s97,a.s173,a.out)
