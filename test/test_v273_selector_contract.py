"""Tests of observable contracts and complete audit producer separation."""
import unittest

import numpy as np

from scripts.v273_selector_contract import (
    MASKS, CORRECTORS, ProducerPool, assemble_outer, context_columns,
    direct_options, fallback_metadata,
)


class SelectorContractTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(271)
        folds = np.repeat([0, 1, 2, 4], 42)
        truth = np.tile(np.arange(7), 24)
        base = np.tile([2, 3, 4], 56)
        x = {k: rng.normal(size=(168, 3)) for k in
             ("spectral", "lifecycle", "harmonic_full", "fundamentals", "sources")}
        raw = rng.normal(size=(168, 4))
        return x, truth, base, folds, np.arange(168), raw

    def test_context_includes_all_declared_families(self):
        names = [f"{family}__{i:02d}" for family, count in
                 (("birth", 11), ("damping", 11), ("persistence", 7), ("spectral", 14))
                 for i in range(count)]
        self.assertEqual(len(context_columns(names)), 43)
        self.assertEqual(sum(n.startswith("spectral__") for n in context_columns(names)), 14)
        with self.assertRaises(ValueError):
            context_columns([n for n in names if not n.startswith("spectral__")])

    def test_exact_membership_and_legal_subsets(self):
        rng = np.random.default_rng(41)
        p = rng.dirichlet(np.ones(5), size=(3, 6)).astype(np.float32)
        base = np.array([2, 3, 4])
        options, valid, identity = direct_options(p, base)
        for i, source in enumerate(base):
            for j, target in enumerate(range(2, 7)):
                expected = 0 if source == target else (64 if (source, target) in CORRECTORS else 32)
                self.assertEqual(int(valid[i, j].sum()), expected)
                np.testing.assert_array_equal(options[i, j, valid[i, j], 5],
                                               identity[i, j, valid[i, j], 6])
        self.assertEqual(options[0, 1, 0, 5], 0)
        self.assertEqual(options[0, 1, 32, 5], 1)
        np.testing.assert_array_equal(identity[..., 0], valid)

    def test_no_fictional_h0_confidence(self):
        x, y, b, folds, ids, _ = self.fixture()
        pool = ProducerPool(x, y, b, folds, ids)
        at = np.flatnonzero(folds == 1)
        p = pool.predict({2}, at, {0, 1})
        np.testing.assert_array_equal(p[:, 0], np.eye(5)[b[at]-2])
        metadata = fallback_metadata(p, b[at])
        np.testing.assert_array_equal(metadata[:, 0], 1.)
        np.testing.assert_array_equal(metadata[:, 10:13], np.eye(3)[b[at]-2])

    def test_producer_rejects_forbidden_and_in_sample_ids(self):
        x, y, b, folds, ids, _ = self.fixture()
        pool = ProducerPool(x, y, b, folds, ids)
        with self.assertRaises(ValueError):
            pool.predict({0, 1}, np.flatnonzero(folds == 2), {0, 2})
        with self.assertRaises(ValueError):
            pool.predict({1}, np.flatnonzero(folds == 1), {0})

    def test_receiver_labels_cannot_return_through_audit_producers(self):
        x, y, b, folds, ids, raw = self.fixture()
        pool = ProducerPool(x, y, b, folds, ids)
        train, held, tr, val, manifest = assemble_outer(pool, raw, 0)
        changed = y.copy()
        changed[folds == 1] = (changed[folds == 1]+1) % 7
        pool2 = ProducerPool(x, changed, b, folds, ids)
        train2, held2, tr2, val2, _ = assemble_outer(pool2, raw, 0)
        receive = folds[tr] == 1
        np.testing.assert_array_equal(train["subset_features"][receive],
                                      train2["subset_features"][receive])
        self.assertFalse(np.array_equal(held["subset_features"], held2["subset_features"]))
        for item in manifest["inner_audits"]:
            excluded = {0, item["receiver_fold"]}
            self.assertFalse(set(item["reference_folds"]) & excluded)
            for producer in item["reference_expert_producers"]:
                self.assertFalse(set(producer["train_folds"]) & excluded)

    def test_outer_labels_never_change_inference_inputs(self):
        x, y, b, folds, ids, raw = self.fixture()
        train, held, _, _, _ = assemble_outer(ProducerPool(x, y, b, folds, ids), raw, 0)
        changed = y.copy(); changed[folds == 0] = 6-changed[folds == 0]
        train2, held2, _, _, _ = assemble_outer(ProducerPool(x, changed, b, folds, ids), raw, 0)
        for name in train:
            np.testing.assert_array_equal(train[name], train2[name])
            np.testing.assert_array_equal(held[name], held2[name])


if __name__ == "__main__":
    unittest.main()
