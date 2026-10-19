"""S749 independent supplemental-note Offset head, FIT80 -> CAL32.

Three fixed causal S707 release policies are evaluated on identical
S748 H1..H5 acoustic NoteOn proposals. The original S708 MIDI stream
is frozen; FIT-only onset admission models, and CAL truth is read only
after every full MIDI trial has been rendered. No inference oracle.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import product

from causal_note.s707_causal_release import prepare_hazards, render_release
from causal_note.s746_instance_guard import (
    InstanceGate, extra_pairs, onset_features)
from causal_note.s747_independent_heads import (
    IsolatedBirthPolicy, isolated_onsets)
from scripts.s746_instance_safe_fit_cal_loops import run,parse_args

BIRTH_POLICY=IsolatedBirthPolicy().validate()

# Strict FIT/CAL experiment grid declared before the CAL labels are read.
# Gate and persistence change ONLY supplemental note termination.
# Native source S708 uses its own unchanged (gate=.8,persistence=1,grace=24).
OFFSETS={
    "sensitive":dict(gate=.45,persistence=1,grace=24),
    "standard":dict(gate=.80,persistence=1,grace=24),
    "persistent":dict(gate=.60,persistence=2,grace=24),
}
TRIALS={
    f"{family}_risk{int(risk*100)}_{offset_name}":
        (family,InstanceGate(risk=risk,k_tail=.20),offset_name)
    for family,risk,offset_name in product(
        ("depth3","depth5"),(.40,.55),tuple(OFFSETS))
}
assert len(TRIALS)==12


def all_release_variants(track,policy,offset):
    item=track["item"]
    raw,diagnostics=isolated_onsets(
        item["decision_ends"],track["post"],track["k"],
        track["physical_heads"],track["context"],
        track["baseline"],BIRTH_POLICY)
    # Hazard predictions use only signed causal features.
    # The common plan is evaluated by three independent supplemental
    # release specialists; S708 is NOT routed through this object.
    hazard=prepare_hazards(item,raw,offset)
    variants={}
    for name,config in OFFSETS.items():
        completed,skipped=render_release(
            hazard,item["ends"],item["samples"],**config)
        pairs=extra_pairs(completed)
        features,indices=onset_features(
            pairs,item["decision_ends"],track["birth_input"],
            track["physical_heads"],track["k"])
        variants[name]=(pairs,features,indices,dict(
            masks=diagnostics["mask_histogram"],
            proposed=diagnostics["proposed"],
            source_skipped=diagnostics["rejected_source"],
            cross_head_dedup=diagnostics["rejected_cross_head_restrike"],
            native_tail_reject=diagnostics["rejected_native_tail"],
            per_head=diagnostics["per_head"],
            offset_specialist=dict(config),
            s707_budget_skipped=int(skipped)))
    # Source metadata and all six S708 control outputs remain unchanged.
    return variants


def source_fit_proposals(track,policy,offset):
    # Match S748 as a paired FIT-only training control.
    return all_release_variants(track,policy,offset)["standard"]


def main():
    run(parse_args(),
        proposal_builder=source_fit_proposals,
        schema="S749_INDEPENDENT_SUPPLEMENTAL_OFFSET_FIT80_CAL32",
        proposal_metadata=dict(
            birth=asdict(BIRTH_POLICY),
            release_policies=OFFSETS,
            strategy="native_source_immutable_three_s707_release_specialists",
            admission_model="FIT80 complete-event HGB, causal onset features"),
        choices_override=TRIALS,
        cal_variant_builder=all_release_variants)


if __name__=="__main__":
    main()
