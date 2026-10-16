"""S626: fail/success counting by exact note, not generic interval or K."""
import unittest

from scripts.s626_audit_missed_combinations import (
    pair_condition,group_condition,accumulate,serialize,init_bucket,
)
from collections import defaultdict


SPEC=dict(id="f",name="fixture",corpus="GuitarSet",composition="independent_1")


def note(pitch,t,duration):
    return dict(pitch=pitch,time=t,duration=duration)


class ConditionFailures(unittest.TestCase):
    def test_e2_e3_both_detected_exact_pair_success(self):
        labels=[note(40,.1,.7),note(52,.12,.6)]
        p=serialize(pair_condition(labels,{0,1},SPEC))
        self.assertEqual(len(p),1)
        self.assertEqual(p[0]["notes"],"E2+E3")
        self.assertEqual(p[0]["examples"],1)
        self.assertEqual(p[0]["failed_at_least_one"],0)
        self.assertEqual(p[0]["complete"],1)
        self.assertEqual(p[0]["distinct_compositions"],1)

    def test_missing_only_e2_fails_pair_and_group_without_double_count(self):
        labels=[note(40,.1,.7),note(52,.12,.6)]
        p=serialize(pair_condition(labels,{1},SPEC))[0]
        g=serialize(group_condition(labels,{1},SPEC))[0]
        self.assertEqual(p["failed_at_least_one"],1)
        self.assertEqual(p["missed_note_instances"],1)
        self.assertEqual(g["missed_note_instances"],1)
        self.assertEqual(g["examples"],1)

    def test_k3_three_pairs_but_one_group_and_one_missed_note(self):
        labels=[note(p,.1,.7) for p in (40,52,55)]
        pair=serialize(pair_condition(labels,{0,2},SPEC))
        group=serialize(group_condition(labels,{0,2},SPEC))
        self.assertEqual(len(pair),3)
        self.assertEqual(sum(p["failed_at_least_one"] for p in pair),2)
        self.assertEqual(len(group),1)
        self.assertEqual(group[0]["failed_at_least_one"],1)
        self.assertEqual(group[0]["missed_note_instances"],1)
        self.assertEqual(group[0]["notes"],"E2+E3+G3")

    def test_late_e3_not_simultaneous_onset_but_is_coactive(self):
        labels=[note(40,.1,.7),note(52,.3,.5)]
        self.assertEqual(len(pair_condition(labels,set(),SPEC)),1)
        self.assertEqual(len(group_condition(labels,set(),SPEC)),0)

    def test_nearby_no_temporal_overlap_not_a_pair(self):
        labels=[note(40,.1,.09),note(52,.2,.4)]
        self.assertEqual(len(pair_condition(labels,set(),SPEC)),0)

    def test_group_samples_protect_source_diversity_in_aggregation(self):
        src={**SPEC,"id":"g","composition":"independent_2"}
        labels=[note(40,.1,.5),note(52,.1,.5)]
        combined=defaultdict(init_bucket)
        accumulate(combined,pair_condition(labels,{1},SPEC))
        accumulate(combined,pair_condition(labels,set(),src))
        row=serialize(combined)[0]
        self.assertEqual(row["examples"],2)
        self.assertEqual(row["failed_at_least_one"],2)
        self.assertEqual(row["distinct_compositions"],2)


if __name__=="__main__":
    unittest.main()
