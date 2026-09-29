"""Prove the local-input/global-target mismatch without training a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import train_v90_structured_cluster_cardinality as v90
from scripts import train_v87_causal_candidate_memory as v87
from scripts import train_v100_spectral_string_slots as v100
from scripts.evaluate_v90_cardinality_realization import _clusters_window
from scripts.evaluate_v90_cluster_oracles import _assign_refs
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.spectral_window import COVERED
from scripts.v273_window_experiment import batch_inputs, array_hash
from scripts.v273_ownership_context import owners_at_samples, owned_sample_mask, required_proposal_watermark


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counterexample():
    """Same four native input tensors, different K after changing a future proposal.

    This is a constructive counterexample at the frozen-proposal interface.
    It is not a claim that duplicate waveform/model-input rows were discovered
    in GuitarSet, nor an end-to-end run of the upstream proposal networks.
    """
    start, onset = 10000, 12600
    local = [start, start + 1764]
    rng = np.random.default_rng(29001)
    embeddings = rng.uniform(0, 1, size=(3, v87.V86_EMBEDDING_DIM)).astype(np.float32)
    hidden87 = rng.uniform(0, 1, size=(3, 48)).astype(np.float32)
    probabilities = np.full((3, v87.V86_CLASS_DIM), 1 / v87.V86_CLASS_DIM, np.float32)
    out = {'local_cardinality': np.tile([.1, .5, .3, .1], (3, 1)),
           'cluster_router': np.full((3, 1), .4, np.float32),
           'isolated_birth': np.full((3, 1), .5, np.float32),
           'cluster_birth': np.full((3, 1), .5, np.float32),
           'fused_birth': np.full((3, 1), .5, np.float32)}
    scores = np.full(3, .5, np.float32)
    # The same observable acoustic segment in both proposal configurations.
    t = np.arange(COVERED.segment_samples) - COVERED.pre_samples
    segment = (.001 * np.sin(2*np.pi*110*t/44100) +
               .1 * (t >= onset-start) * np.sin(2*np.pi*220*t/44100)).astype(np.float32)
    spectral = v100._spectral_map_from_segment(segment, window=COVERED)
    configurations, native, histories = [], [], []
    for neighbor in (start + 3300, start + 3500):
        records = [dict(member='synthetic', arrangement='comp', sample=s,
                        score=.5, count=1, sources=[], class_id=0) for s in local + [neighbor]]
        clusters = _clusters_window(records, 40.)
        assert clusters[0]['indices'] == (0, 1) and len(clusters) == 2
        x88 = v90._feature_matrix(records, embeddings, probabilities, hidden87, probabilities)
        features, _ = v90._extended_candidate_features(x88, out)
        assigned, _ = _assign_refs(clusters, records, {'synthetic': [onset]})
        seq, mask, stats, target, exact, _ = v90._cluster_arrays(
            clusters, assigned, records, features, out, scores)
        inputs = batch_inputs(dict(sequence=seq, mask=mask, stats=stats,
                                   spectral=np.stack([spectral, spectral])), np.asarray([0]), 31)
        history, _ = v87._sequence_arrays(records, embeddings, probabilities)
        native.append(inputs); histories.append(history[:2])
        groups = [np.asarray(local), np.asarray([neighbor])]
        watermark = required_proposal_watermark(groups[0])
        owner = int(owners_at_samples(groups, np.asarray([onset]))[0])
        owned = bool(owned_sample_mask(groups, 0, np.asarray([onset]), complete_through=watermark)[0])
        assert int(target[0]) == int(exact[0]) == int(owner == 0) == int(owned)
        configurations.append(dict(neighbor_relative_sample=neighbor-start,
            current_candidate_distance=abs(onset-local[-1]),
            neighbor_candidate_distance=abs(neighbor-onset),
            target_k=int(target[0]), owner_group=owner, new_owned_input=owned))
        try:
            owned_sample_mask(groups, 0, np.asarray([onset]), complete_through=start+COVERED.post_samples)
        except ValueError:
            pass
        else:
            raise AssertionError('the current observation deadline must not certify complete ownership')
    for key in native[0]:
        np.testing.assert_array_equal(native[0][key], native[1][key])
    np.testing.assert_array_equal(histories[0], histories[1])
    assert configurations[0]['target_k'] != configurations[1]['target_k']
    return dict(proof_scope='frozen-proposal input contract; not a real-data collision search',
        current_candidates_relative_samples=[0, 1764], onset_relative_sample=2600,
        current_audio_end_relative_sample=COVERED.post_samples,
        sufficient_proposal_watermark_relative_sample=required_proposal_watermark(np.asarray([0,1764])),
        configurations=configurations, native_inputs_identical=True,
        current_past_candidate_history_identical=True,
        per_input_sha256={key:array_hash(value) for key,value in native[0].items()},
        proposed_geometry_distinguishes_targets=True, incomplete_context_rejected=True)


def frozen_evidence(root, provenance_path):
    """Re-use the pinned fold-3 audit; no new fold or model inference."""
    validate_original_inventory(root)
    provenance = json.loads(provenance_path.read_text())
    for name, digest in provenance['file_inventory'].items():
        assert sha(root/name) == digest, name
    traces = json.loads((root/'overcount-traces.json').read_text())
    with np.load(root/'rows.npz', allow_pickle=False) as z:
        rows = {key:z[key] for key in z.files}
    over = (rows['k'] < 4) & (rows['predicted'] > rows['k'])
    assert len(rows['k']) == 15279 and int(over.sum()) == len(traces) == 874
    ids = {int(value):i for i,value in enumerate(rows['global_index'])}
    future, past, contested = set(), set(), set()
    local_equals_prediction = 0
    for trace in traces:
        gid = trace['global_index']; i = ids[gid]
        assert over[i] and trace['member'] == rows['member'][i]
        assert trace['true_k'] == rows['k'][i] and trace['predicted_k'] == rows['predicted'][i]
        foreign = [e for e in trace['events'] if e['distance_to_current_candidate'] <= 882 and not e['own']]
        assert len(foreign) == trace['contested_births']
        assert trace['local_eligible_births'] == trace['true_k'] + len(foreign)
        if foreign: contested.add(gid)
        if any(e['owner_track_row'] > trace['track_row'] for e in foreign): future.add(gid)
        if any(0 <= e['owner_track_row'] < trace['track_row'] for e in foreign): past.add(gid)
        local_equals_prediction += trace['local_eligible_births'] == trace['predicted_k']
    assert contested == future | past
    example = next(r for r in traces if r['global_index'] == 17020)
    return dict(source_run=provenance['audit_run'], source_training_run=provenance['training_run'],
        outer_fold=3, inference_performed=False, error_rows=len(traces),
        contested_error_rows=len(contested), future_group_error_rows=len(future),
        previous_group_error_rows=len(past), both_directions_rows=len(future & past),
        error_rows_without_contested_birth=len(traces)-len(contested),
        local_eligible_count_equals_prediction_rows=int(local_equals_prediction),
        example=example, archive_file_inventory=provenance['file_inventory'],
        warning='Observed association is not an attributable error fraction or a measured gain.')


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(status='verified', counterexample=counterexample(),
        frozen_evidence=frozen_evidence(args.evidence, args.provenance),
        source_files_sha256={path:sha(path) for path in (
            'scripts/evaluate_v90_cluster_oracles.py',
            'scripts/train_v90_structured_cluster_cardinality.py',
            'scripts/train_v87_causal_candidate_memory.py',
            'scripts/train_v250_count_only.py', 'scripts/v273_window_experiment.py',
            'scripts/v273_ownership_context.py', 'scripts/audit_v273_design_contract.py')},
        no_training=True, no_output_corrector=True, model_integration_performed=False,
        no_model_promotion=True, measured_exact_k_gain=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result['frozen_evidence'][k] for k in (
        'error_rows','contested_error_rows','future_group_error_rows','previous_group_error_rows',
        'both_directions_rows','error_rows_without_contested_birth')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--provenance', type=Path, default=Path('analysis/v273-acoustic-residual-sources.json'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
