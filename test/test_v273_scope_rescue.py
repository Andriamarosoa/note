"""Tests for the new scope-rescue boundary and train-only acceptance."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.extract_v273_harmonic_global import baseline_selection
from scripts.evaluate_v273_scope_rescue import (
    SOURCES, choose_threshold, probabilities, proposals, read_scope,
)


class TestScopeRescue(unittest.TestCase):
    def test_default_baseline_scope_unchanged(self):
        b = np.arange(7)
        np.testing.assert_array_equal(baseline_selection(b, (2,3,4)), [2,3,4])
        np.testing.assert_array_equal(baseline_selection(b, (0,1,5)), [0,1,5])
        self.assertEqual(SOURCES, (0,1,5))

    def test_reject_invalid_or_duplicate_selection(self):
        for bad in ((), (0,0), (-1,), (7,)):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    baseline_selection(np.arange(7), bad)

    def test_best_threshold_prioritizes_no_regression(self):
        y = np.array([1,0,1,0])
        base = np.zeros(4, dtype=int)
        offer = np.ones(4, dtype=int)
        margin = np.array([.9,.8,.7,.6])
        chosen = choose_threshold(y,base,offer,margin)
        self.assertEqual(chosen["threshold"], .9)
        self.assertEqual(chosen["net"], 1)
        self.assertEqual(chosen["actions"], 1)

    def test_abstains_when_every_change_is_harmful(self):
        y = np.array([0,0,0])
        base = np.zeros(3, dtype=int)
        offer = np.array([1,2,1])
        margin = np.array([.7,.6,.5])
        selected = choose_threshold(y, base, offer, margin)
        self.assertEqual(selected["actions"], 0)
        self.assertTrue(np.isinf(selected["threshold"]))

    def test_unchanged_action_is_not_counted(self):
        chosen = choose_threshold(
            np.array([0,1,1]), np.array([0,0,0]),
            np.array([0,1,1]), np.array([0,.4,.3]))
        self.assertEqual(chosen["actions"], 2)

    def test_missing_class_probability_is_zero(self):
        x = np.array([[0.0],[.1],[2.0],[2.1]])
        out = probabilities(x, np.array([0,0,1,1]), x)
        self.assertEqual(out.shape, (4,7))
        np.testing.assert_allclose(out[:,2:], 0)
        np.testing.assert_allclose(out.sum(1),1)
        p,d = proposals(out,np.array([0,0,1,1]))
        self.assertTrue(np.isfinite(d).all())
        self.assertTrue(np.isin(p,[0,1]).all())

    def test_feature_coverage_rejects_unselected_rows(self):
        folds = np.array([0,1,2,4],int)
        ids = np.array([100,101,102,103],int)
        b = np.array([0,1,5,2],int)
        y = np.array([1,1,4,2],int)
        names = np.array(["00_A","01_B","02_C","04_D"])
        times = np.arange(4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for fold in folds:
                folder = root/str(fold)
                folder.mkdir()
                idx = int(np.flatnonzero(folds==fold)[0])
                records = []
                if b[idx] in SOURCES:
                    records.append(dict(global_index=int(ids[idx]),fold=int(fold),
                                        recording_id=str(names[idx]),start_sample=int(times[idx]),
                                        baseline_k=int(b[idx]),true_k=int(y[idx]),
                                        features={"acoustic":float(idx)}))
                (folder/"rows.jsonl").write_text("".join(json.dumps(v)+"\n" for v in records))
                (folder/"report.json").write_text(json.dumps(
                    dict(experiment="v273_harmonic_global_candidate_extract",
                         candidate_baselines=[0,1,5],primary_only=True,
                         fold=int(fold),eligible_rows=len(records))))
            pos, rows, features = read_scope(root,y,b,ids,folds,names,times)
            np.testing.assert_array_equal(pos,[0,1,2])
            self.assertEqual(features,["acoustic"])
            extra=root/"3"/"rows.jsonl"
            with extra.open("a") as f:
                f.write(json.dumps(dict(global_index=103,fold=4,recording_id="04_D",
                                        start_sample=3,baseline_k=2,true_k=2,
                                        features={"acoustic":3}))+"\n")
            with self.assertRaises(Exception):
                read_scope(root,y,b,ids,folds,names,times)


if __name__ == "__main__":
    unittest.main()
