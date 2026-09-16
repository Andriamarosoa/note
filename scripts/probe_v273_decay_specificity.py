"""Probe decay-feature specificity using stationary, onset-free sinusoids.

This exercises the actual spectral preprocessing and float16 cache format.
It does not run the count network or claim that every GuitarSet regression
has this acoustic cause. Frequencies, amplitudes and phase offsets are constant
through time; the segment is a crop of indefinitely sustained oscillators.
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.train_v100_spectral_string_slots import (
    _spectral_map_from_segment, SEGMENT_SAMPLES, SAMPLE_RATE,
)
from scripts.v273_decay_native_inputs import native_decay_features
from scripts.rebuild_v273_sources import write_json


def probe():
    time = np.arange(SEGMENT_SAMPLES, dtype=np.float64)/SAMPLE_RATE
    signals = [('single_220', [220]), ('steady_220_233', [220, 233.08188]),
               ('steady_110_116', [110, 116.54094]),
               ('single_110_harmonics', [110*i for i in range(1, 9)])]
    results = []
    for name, frequencies in signals:
        phases = [0.] if len(frequencies) == 1 else np.linspace(0, 2*np.pi, 64, endpoint=False)
        crops = []
        for phase in phases:
            wave = sum((.2/(i+1))*np.sin(2*np.pi*f*time + (phase if i else .3))
                       for i, f in enumerate(frequencies))
            crops.append(_spectral_map_from_segment(wave.astype(np.float32)).astype(np.float16))
        features, audit = native_decay_features(np.stack(crops))
        maximum = features.max(axis=1)
        example = int(np.argmax(maximum))
        results.append({'signal': name, 'frequencies_hz': frequencies,
                        'amplitudes': [.2/(i+1) for i in range(len(frequencies))],
                        'new_onsets_in_each_crop': 0, 'phase_settings': len(phases),
                        'max_feature_above_0_1': int(np.sum(maximum > .1)),
                        'max_feature_above_0_5': int(np.sum(maximum > .5)),
                        'max_feature': float(maximum.max()),
                        'example_phase_index': example,
                        'example_relative_phase_radians': float(phases[example]),
                        'reliable_bands_in_example': int(audit['reliable'][example].sum())})
    return {'method': 'actual V100 preprocessing then float16 then native_decay_features',
            'no_new_attack_by_construction': True,
            'count_network_inference_performed': False,
            'cutoffs_are_descriptive_not_model_decision_thresholds': True,
            'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    write_json(args.output, result)
    print(result)
