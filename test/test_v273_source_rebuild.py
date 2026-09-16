import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import rebuild_v273_sources as v
from scripts import summarize_v273_rebuilt_cache as summary
from scripts import restore_v273_original_backup as backup
from scripts import restore_v273_rebuild_checkpoint as resume


class RecoverySourceTests(unittest.TestCase):
    def test_preserved_backup_verifies_both_archive_and_file_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root/'checkpoint'
            model.write_bytes(b'checkpoint')
            archive = root/'backup.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.write(model, 'weights/checkpoint')
                z.writestr('file-sha256.json', json.dumps({'weights/checkpoint': backup.sha256(model)}))
            backup.extract_verified(archive, root/'good', backup.sha256(archive))
            self.assertEqual((root/'good/weights/checkpoint').read_bytes(), b'checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'archive checksum mismatch'):
                backup.extract_verified(archive, root/'bad', '0'*64)
            self.assertFalse((root/'bad').exists())
            bad_manifest = root/'bad-manifest.zip'
            with zipfile.ZipFile(bad_manifest, 'w') as z:
                z.write(model, 'weights/checkpoint')
                z.writestr('file-sha256.json', json.dumps({'weights/checkpoint': '0'*64}))
            with self.assertRaisesRegex(RuntimeError, 'source file changed'):
                backup.extract_verified(bad_manifest, root/'bad-file', backup.sha256(bad_manifest))
            self.assertFalse((root/'bad-file').exists())

    def test_backup_cannot_extract_outside_its_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root/'traversal.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside', b'unsafe')
                z.writestr('file-sha256.json', '{}')
            with self.assertRaisesRegex(RuntimeError, 'unsafe backup path'):
                backup.extract_verified(archive, root/'output', backup.sha256(archive))
            self.assertFalse((root/'output').exists())

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
            if stage == 'audit':
                # The FP target can be met early. The stop floor must still
                # satisfy all three downstream programs' source-track contract.
                for downstream in ('v86', 'v87', 'v88'):
                    cmd, _ = v.stage_spec(downstream, Path('/data/GuitarSet'), Path('/output'))
                    module = importlib.import_module('scripts.' + Path(cmd[2]).stem)
                    need = module.create_argument_parser().parse_args(cmd[3:]).train_members
                    self.assertGreaterEqual(parsed.min_tracks, need)

    def checkpoint_archive(self, root):
        weights = root/'checkpoint.keras'
        weights.write_bytes(b'completed epoch three')
        producer = 'a'*40
        state = {'source_kind': v.SOURCE_KIND, 'source_code_sha': producer,
                 'dataset_md5': v.DATA_MD5, 'stages': {
                     'v81': {'status': 'completed', 'outputs': [
                         {'path': 'v81/stream.epoch-03.keras', 'sha256': v.digest(weights)}]},
                     'audit': {'status': 'failed'}}}
        archive = root/'checkpoint.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.write(weights, 'v81/stream.epoch-03.keras')
            z.writestr('rebuild-state.json', json.dumps(state))
            z.writestr('audit/report.json', json.dumps({'scope': {'members': list(range(24))}}))
        source = {'run_id': 123, 'producer_commit': producer,
                  'checkpoint_sha256': v.digest(weights), 'sha256': v.digest(archive)}
        return archive, source

    def test_resume_keeps_weights_and_failed_audit_then_runs_fixed_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, source = self.checkpoint_archive(root)
            output = root/'resumed'
            resume.restore_archive(archive, output, source)
            state = v.validate_sources(output, ('v81',))
            self.assertEqual(set(state['stages']), {'v81'})
            self.assertEqual(state['stages']['v81']['reused_from_run_id'], 123)
            self.assertEqual(v.digest(output/'v81/stream.epoch-03.keras'), source['checkpoint_sha256'])
            self.assertTrue((output/'history/run-123/audit/report.json').exists())
            self.assertEqual(json.loads((output/'history/run-123/rebuild-state.json').read_text())
                             ['stages']['audit']['status'], 'failed')
            self.assertFalse((output/'audit').exists())
            with self.assertRaises(FileExistsError):
                resume.restore_archive(archive, output, source)

            def fake_audit(command, **kwargs):
                self.assertTrue(command[2].endswith('audit_v81_train_fp_harmonics.py'))
                module = importlib.import_module('scripts.audit_v81_train_fp_harmonics')
                args = module.create_argument_parser().parse_args(command[3:])
                # Simulate reaching 4,000 FPs before the minimum track count.
                v.write_json(args.output, {'scope': {'members': list(range(args.min_tracks))}})

            args = type('Args', (), {'stage': 'audit', 'dataset_dir': root/'dataset',
                                    'output_dir': output})()
            with mock.patch.object(v, 'verify_dataset'), mock.patch.object(v.subprocess, 'run', fake_audit):
                v.run_stage(args)
            state = v.validate_sources(output, ('v81', 'audit'))
            self.assertEqual(state['stages']['audit']['status'], 'completed')
            self.assertEqual(v.digest(output/'v81/stream.epoch-03.keras'), source['checkpoint_sha256'])

    def test_resume_rejects_wrong_producer_or_checkpoint_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, source = self.checkpoint_archive(root)
            for key in ('producer_commit', 'checkpoint_sha256', 'sha256'):
                with self.subTest(key=key):
                    with self.assertRaises(RuntimeError):
                        resume.restore_archive(archive, root/key, {**source, key: '0'*64})
                    self.assertFalse((root/key).exists())

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
