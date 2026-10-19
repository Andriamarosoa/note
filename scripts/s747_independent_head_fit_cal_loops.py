"""S747 launch 12 FIT/CAL loops with isolated native NOTE acoustic heads.

Reuse the S746 frozen-baseline / 6-metric evaluator, changing ONLY the
candidate proposal component. The official S708, S707, LIVE and S80 are
not modified. No future labels can enter inference.
"""
from __future__ import annotations

from dataclasses import asdict
from causal_note.s707_causal_release import prepare_hazards,render_release
from causal_note.s746_instance_guard import extra_pairs,onset_features
from causal_note.s747_independent_heads import (
    IsolatedBirthPolicy, isolated_onsets)
from scripts.s746_instance_safe_fit_cal_loops import run,parse_args

CONFIG=IsolatedBirthPolicy().validate()


def independent_proposals(track,policy,offset):
    item=track["item"]
    raw,diag=isolated_onsets(
        item["decision_ends"],track["post"],track["k"],
        track["physical_heads"],track["context"],
        track["baseline"],CONFIG)
    # S707 is used solely to terminate supplemental proposals. Source
    # intervals remain entirely outside this call and unchanged.
    finished,skipped=render_release(
        prepare_hazards(item,raw,offset),
        item["ends"],item["samples"],
        gate=.8,persistence=1,grace=24)
    pairs=extra_pairs(finished)
    features,indices=onset_features(
        pairs,item["decision_ends"],track["birth_input"],
        track["physical_heads"],track["k"])
    return pairs,features,indices,dict(
        masks=diag["mask_histogram"],
        proposed=diag["proposed"],
        source_skipped=diag["rejected_source"],
        s707_budget_skipped=int(skipped),
        per_head=diag["per_head"])


def main():
    run(parse_args(),proposal_builder=independent_proposals,
        schema="S747_INDEPENDENT_HEAD_STATES_FIT80_CAL32",
        proposal_metadata=asdict(CONFIG))


if __name__=="__main__":
    main()
