import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import rebuild_v273_sources as v
from scripts import summarize_v273_rebuilt_cache as summary


class RecoverySourceTests(unittest.TestCase):
    def test_historical_programs_accept_every_stage_command(self):
        for stage in v.STAGES:
            command, outputs = v.stage_spec(stage, Path('/data/GuitarSet'), Path('/output'))
            module = importlib.import_module('scripts.' + Path(command[2]).stem)
            parsed = module.create_argument_parser().parse_args(command[3:])
            self.assertEqual(parsed.dataset_dir, Path('/data/GuitarSet'))
            self.assertTrue(outputs)
            if stage in ('v86', 'v87', 'v88'):
                self.assertEqual(parsed.train_members, 30)
                self.assertEqual(parsed.epochs, 20)
            if stage == 'v84':
                self.assertEqual(parsed.epochs, 1)
                self.assertEqual(parsed.source_model.name, 'stream.epoch-03.keras')

    def test_altered_completed_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'model.bin'
            model.write_bytes(b'original checkpoint')
            state = {'source_kind': v.SOURCE_KIND, 'stages': {'v81': {
                'status': 'completed', 'outputs': [{'path': 'model.bin', 'sha256': v.digest(model)}],
            }}}
            v.write_json(root / 'rebuild-state.json', state)
            v.validate_sources(root, ('v81',))
            model.write_bytes(b'different checkpoint')
            with self.assertRaises(RuntimeError):
                v.validate_sources(root, ('v81',))

    def test_incomplete_source_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            v.write_json(root / 'rebuild-state.json', {'source_kind': v.SOURCE_KIND, 'stages': {}})
            with self.assertRaises(RuntimeError):
                v.validate_sources(root)

    def test_distilled_v28_metadata_cannot_be_reported_as_native_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(8):
                np.savez(root / f'v100-spectral-shard-{i:02d}.npz', exact=np.array([2]),
                         sequence=np.zeros((1,48,3)), schema_version=np.array([1]))
            with self.assertRaisesRegex(RuntimeError, 'incomplete native cache'):
                summary.summarize(root, root/'unused.npz', root/'out')

    def test_matching_metadata_never_claims_the_missing_historical_inputs_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chunks = {key: [] for key in summary.SHARED}
            tracks = []
            for i in range(8):
                members = np.array([f'track-{i}-{j}' for j in range(30)])
                data = {'sequence': np.zeros((30,48,2)), 'mask': np.ones((30,48)),
                        'exact': np.full(30,2), 'members': members,
                        'top_samples': np.zeros((30,48)), 'slot_targets': np.zeros((30,6))}
                np.savez(root / f'v100-spectral-shard-{i:02d}.npz', **data,
                         spectral=np.zeros((30,23,64,3)), stats=np.zeros((30,8)),
                         target=np.full(30,2), track_members=members, schema_version=np.array([1]))
                for key in chunks:
                    chunks[key].append(data[key])
                tracks.extend(members)
            old = {key: np.concatenate(values) for key, values in chunks.items()}
            np.savez(root/'old.npz', **old, track_members=np.array(tracks))
            result = summary.summarize(root, root/'old.npz', root/'out')
            self.assertEqual(result['rows'], 240)
            self.assertTrue(result['shared_metadata_values_identical'])
            self.assertFalse(result['historical_v273_reproduced'])
            self.assertFalse(result['native_decay_comparison_started'])


if __name__ == '__main__':
    unittest.main()
