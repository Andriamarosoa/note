"""Fit a deployable correction proposal on fourteen strictly excluded fold sets.

Inputs are acoustic context and the frozen action, never a receiver's truth,
observed error status, outcome audit, or predictions trained on that receiver.
Seven-class CE uses both successes and errors. Unsupported K0/K1 probability
is conservatively assigned to the baseline action when proposing K2..K6.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.v273_selector_contract import FOLDS, require, id_digest, matrices
from scripts.v273_group127_contract import seven_candidates
from scripts.prepare_v273_group127_producers import cache_key
from scripts.audit_v273_selector_design import aligned_positions, load_feature_rows
from scripts.yourmt3_exactk_common import digest

SEED = 27403


def project_actions(probability, baseline):
    """Inference-only projection: no true label or observed correction needed."""
    p = np.asarray(probability, np.float32); b = np.asarray(baseline, int)
    require(p.shape == (len(b), 7) and np.isin(b, [2, 3, 4]).all(), 'corrector input shape')
    require(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-6), 'corrector distribution')
    q = p[:, 2:].copy()
    q[np.arange(len(b)), b-2] += p[:, :2].sum(1)
    require(np.allclose(q.sum(1), 1, atol=1e-6), 'action projection normalization')
    return q


def fit_corrector(raw, truth, baseline, epochs=30):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    scaler = StandardScaler().fit(raw)
    features = np.concatenate([np.clip(scaler.transform(raw), -6, 6),
                               np.eye(7)[baseline]], 1).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(64, activation='gelu'),
        tf.keras.layers.Dense(32, activation='gelu'),
        tf.keras.layers.Dense(7),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(.002),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    dataset = tf.data.Dataset.from_tensor_slices((features, np.asarray(truth, np.int32)))
    dataset = dataset.shuffle(len(truth), seed=SEED, reshuffle_each_iteration=True).batch(192)
    history = model.fit(dataset, epochs=epochs, verbose=0).history['loss']
    require(np.isfinite(history).all(), 'nonfinite corrector training')
    return model, scaler, [dict(epoch=e+1, loss=float(history[e]))
                           for e in sorted(set((0, epochs//2, epochs-1)))]


def infer_corrector(model, scaler, raw, baseline):
    import tensorflow as tf
    x = np.concatenate([np.clip(scaler.transform(raw), -6, 6),
                        np.eye(7)[baseline]], 1).astype(np.float32)
    return np.concatenate([tf.nn.softmax(model(x[i:i+192], training=False)).numpy()
                           for i in range(0, len(x), 192)])


class FrozenCorrector:
    def __init__(self, truth, baseline, folds, ids, directory):
        self.y, self.b, self.fold, self.ids = map(np.asarray, (truth, baseline, folds, ids))
        self.manifest = json.loads((directory/'manifest.json').read_text())
        self.cache_sha256 = digest(directory/'correctors.npz')
        require(self.cache_sha256 == self.manifest['cache_sha256'], 'corrector cache digest')
        with np.load(directory/'correctors.npz', allow_pickle=False) as z:
            self.cache = {k: z[k] for k in z.files}
        for key, actual in [('truth', self.y), ('baseline', self.b), ('fold', self.fold), ('eligible_global_index', self.ids)]:
            require(np.array_equal(self.cache[key], actual), 'corrector cache alignment '+key)
        self.manifests = {tuple(p['train_folds']): p for p in self.manifest['producers']}
        require(len(self.manifests) == 14, 'corrector producer inventory')
        for key, record in self.manifests.items():
            fit = np.isin(self.fold, key)
            require(np.array_equal(record['train_global_ids'], self.ids[fit]), 'corrector fit IDs')
            require(record['train_id_sha256'] == id_digest(self.ids[fit]), 'corrector fit digest')
            values = self.cache[cache_key(key)]
            require(values.shape == (len(ids), 7) and np.isnan(values[fit]).all()
                    and np.isfinite(values[~fit]).all(), 'corrector cache must exclude all fit rows')

    def probabilities(self, train_folds, positions, forbidden_folds):
        key = tuple(sorted(map(int, train_folds))); positions = np.asarray(positions, int)
        require(key in self.manifests and not set(key) & set(forbidden_folds), 'forbidden corrector fit fold')
        require(not np.isin(self.fold[positions], key).any(), 'in-sample corrector use')
        p = self.cache[cache_key(key)][positions]
        require(np.isfinite(p).all(), 'missing OOF corrector')
        return p.copy()


class CataloguePool:
    def __init__(self, experts, corrector=None):
        self.experts, self.corrector = experts, corrector
        self.y, self.b, self.fold, self.ids = experts.y, experts.b, experts.fold, experts.ids
        self.manifests = experts.manifests

    def predict(self, train_folds, positions, forbidden_folds):
        votes = seven_candidates(self.experts.predict(train_folds, positions, forbidden_folds))
        if self.corrector is not None:
            q = self.corrector.probabilities(train_folds, positions, forbidden_folds)
            votes = np.concatenate([votes, project_actions(q, self.b[positions])[:, None]], 1)
        return votes

    def crossfit(self, positions, excluded):
        positions = np.asarray(positions, int)
        available = set(map(int, np.unique(self.fold[positions])))
        require(not available & set(excluded) and len(available) >= 2, 'catalogue crossfit split')
        out = np.empty((len(positions), 8 if self.corrector else 7, 5), np.float32)
        records = []
        for receiver in sorted(available):
            local = self.fold[positions] == receiver
            key = tuple(sorted(available-{receiver}))
            out[local] = self.predict(key, positions[local], set(excluded) | {receiver})
            record = dict(excluded_folds=sorted(set(excluded)), reference_prediction_fold=receiver,
                          train_folds=list(key), train_id_sha256=self.manifests[key]['train_id_sha256'],
                          forbidden_train_overlap=0)
            if self.corrector:
                record['corrector_train_id_sha256'] = self.corrector.manifests[key]['train_id_sha256']
            records.append(record)
        return out, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing cache overwrite')
    require(digest(args.reference) == '707fc1681e2b0b1ef01805710186c4119ad697a0b52ed3330698e48f34c2f77b', 'native reference digest')
    with np.load(args.reference, allow_pickle=False) as z:
        ids, y, b, f, eligible = (z[k] for k in ['global_index','true_K','frozen_baseline_K','fold','eligible_global_index'])
    pos = aligned_positions(ids, eligible)
    rows, _ = load_feature_rows(args.features, ids, y, b, f, pos)
    _, raw, names = matrices(rows)
    y, b, f = y[pos], b[pos], f[pos]
    require(raw.shape == (7493,43) and set(f) == set(FOLDS), 'corrector cohort')
    args.output.mkdir(parents=True)
    data = dict(eligible_global_index=eligible, truth=y, baseline=b, fold=f)
    producers = []
    import tensorflow as tf
    for size in (1,2,3):
        for key in itertools.combinations(FOLDS, size):
            fit = np.flatnonzero(np.isin(f, key)); held = np.flatnonzero(~np.isin(f, key))
            require(not set(eligible[fit]) & set(eligible[held]), 'corrector train/test overlap')
            model, scaler, history = fit_corrector(raw[fit], y[fit], b[fit])
            require(model.count_params() == 5575, 'corrector architecture')
            prob = np.full((len(y),7), np.nan, np.float32)
            prob[held] = infer_corrector(model, scaler, raw[held], b[held])
            project_actions(prob[held], b[held])
            data[cache_key(key)] = prob
            weight_file = args.output/(cache_key(key)+'.weights.h5')
            model.save_weights(str(weight_file))
            record = dict(train_folds=list(key), train_rows=len(fit), train_global_ids=eligible[fit].tolist(),
                train_id_sha256=id_digest(eligible[fit]), true_k_counts=np.bincount(y[fit], minlength=7).tolist(),
                scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                weights=weight_file.name, weights_sha256=digest(weight_file), training=history, parameters=5575)
            producers.append(record)
            print(json.dumps(dict(stage='corrector_fitted', train_folds=key, rows=len(fit), history=history)), flush=True)
            del model
            tf.keras.backend.clear_session(); gc.collect()
    np.savez_compressed(args.output/'correctors.npz', **data)
    manifest = dict(producers=producers, cache_sha256=digest(args.output/'correctors.npz'),
        source_reference_sha256=digest(args.reference), context_names=names,
        architecture=[50,64,32,7], loss='unweighted seven-class categorical CE on all fit rows',
        inference_inputs=['43 acoustic observables', 'frozen K'], epochs=30, seed=SEED, learning_rate=.002,
        batch_size=192, fit_rows_excluded_from_cache=True, action_projection='K0/K1 probability routed to frozen K')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    FrozenCorrector(y, b, f, eligible, args.output)
    print(json.dumps(dict(status='completed', cache_sha256=manifest['cache_sha256'])), flush=True)


if __name__ == '__main__':
    main()
