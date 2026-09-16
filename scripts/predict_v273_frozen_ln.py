"""Load the selected count bundle and predict from native inputs, without labels.

The callable accepts the four original count-network inputs and the shared
V104 anchor counts. The CLI replays only the recorded outer fold 3 to verify
the exported weights; annotation labels enter only subsequent reporting.
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.audit_v273_native_paired_results import load_npz, read_json, require, verify_inventory
from scripts.audit_v273_fold3_ablation import metrics
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v102_source_time_assignment as v102
from scripts import train_v273_native_paired as paired
from scripts import v273_decay_native_inputs as native
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_native_protocol import verify_native_cache


def predict_native_counts(inputs, anchor, bundle_dir):
    """Return count decisions/probabilities using a verified selected bundle.

    No labels, test scores, or threshold-selection operations are accepted.
    A selected projection already includes its internally chosen strength.
    """
    import tensorflow as tf
    require(tf.__version__ == '2.15.1', 'TensorFlow 2.15.1 required')
    bundle = Path(bundle_dir)
    verify_inventory(bundle)
    report, selection = read_json(bundle/'report.json'), read_json(bundle/'selection.json')
    require(digest(bundle/'selection.json') == report['selection_sha256'], 'bundle selection changed')
    chosen = selection['selected']
    require(chosen == report['selected_before_outer_evaluation'], 'bundle decision mismatch')
    calibration = read_json(bundle/'calibration.json')
    required = {'candidate_set', 'candidate_mask', 'cluster_stats', 'spectral_map'}
    require(set(inputs) == required, 'expected the four native inputs, without labels')
    anchor = np.asarray(anchor, dtype=np.int32)
    require(anchor.ndim == 1 and len(anchor) > 0 and np.all((anchor >= 0) & (anchor <= 6)), 'invalid anchor')
    require(all(len(x) == len(anchor) and np.isfinite(x).all() for x in inputs.values()), 'invalid native inputs')
    features = (np.zeros((len(anchor), native.FEATURE_DIM), np.float32) if chosen == 'control'
                else paired.evidence_table(inputs['spectral_map'])[0])
    probabilities, weights = {}, {}
    for component in native.COMPONENTS:
        path = bundle/'final'/component/'model.weights.h5'
        model, _ = native.build_count_model(component, 0)
        model.load_weights(path)
        model.trainable = False
        if chosen == 'control':
            require(not np.any(model.get_layer('native_decay_projection').get_weights()[0]),
                    'selected control contains a nonzero ln projection')
        probability = np.asarray(model.predict({**inputs, 'decay_evidence': features},
                                               batch_size=128, verbose=0), dtype=np.float32)
        require(np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, atol=1e-5), 'invalid model output')
        probabilities[component] = probability
        weights[component] = digest(path)
        print({'loaded_and_predicted': component, 'selected': chosen, 'rows': len(anchor)}, flush=True)
        del model
        tf.keras.backend.clear_session()
    prediction = paired.decode(anchor, probabilities, calibration)
    return prediction, probabilities, {'selected': chosen, 'weights_sha256': weights,
                                       'calibration_sha256': digest(bundle/'calibration.json')}


def verify_export(args):
    require(not args.output_dir.exists(), 'output already exists')
    verified = verify_native_cache(args.cache_dir, args.config)
    bundle = Path(args.bundle_dir)
    verify_inventory(bundle)
    require(read_json(bundle/'protocol.json')['outer_fold'] == 3, 'only outer fold 3 is in scope')
    saved = load_npz(bundle/'predictions.npz')
    cache = v100._load_spectral_caches(args.cache_dir)
    indices = saved['global_index'].astype(np.int64)
    require(np.array_equal(cache['members'][indices].astype(str), saved['member']), 'native row ordering differs')
    prediction, probabilities, loaded = predict_native_counts(
        v102._inputs(cache, indices), saved['shared_anchor'], bundle)
    require(np.array_equal(prediction, saved['selected']), 'exported decisions do not reproduce the selected model')
    for component, p in probabilities.items():
        require(np.allclose(p, saved[component], atol=2e-5, rtol=1e-5), 'exported probability mismatch: '+component)
    # Ground truth is used only after all model outputs have been computed.
    k = np.minimum(cache['exact'][indices], 6).astype(np.int32)
    require(np.array_equal(k, saved['k']), 'ground truth ordering differs')
    score = metrics(k, prediction)
    expected = read_json(bundle/'report.json')['metrics']['selected_frozen_model']
    require(score == expected, 'exported score mismatch')
    result = {'outer_folds_verified': [3], 'runtime_contains_label_input': False,
              'weights_reloaded': True, 'saved_decisions_exactly_reproduced': True,
              'rows': len(prediction), 'metrics': score, 'loaded': loaded,
              'source_config_sha256': digest(args.config), 'native_input_provenance': verified,
              'maximum_probability_difference': {c: float(np.max(np.abs(p-saved[c]))) for c, p in probabilities.items()}}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'runtime-verification.json', result)
    np.savez_compressed(args.output_dir/'runtime-predictions.npz', global_index=indices, prediction=prediction,
                        **probabilities)
    print({'export_verification': result}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('analysis/v273-native-paired-config.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    verify_export(p.parse_args())
