"""Regression checks for two independently reviewed provenance bypasses."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from scripts.s813_prepare_native256 import digest_file, save_json
from scripts.s813_verify_evidence import verify


EVIDENCE = Path('series/S813/evidence')


@unittest.skipUnless((EVIDENCE/'bundle.json').exists(), 'Real experiment evidence required')
class EvidenceProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = self.root/'evidence'
        shutil.copytree(EVIDENCE, self.evidence)
        # An equivalent tiny manifest is sufficient for these structural checks;
        # production verification separately fetches the actual signed manifest.
        import zipfile
        with zipfile.ZipFile(self.evidence/'source_metadata.zip') as z:
            source_rows = []
            report = json.loads((self.evidence/'report.json').read_text())
            for row in report['tracks']:
                name = f"fit_{row['index']:03d}.json"
                raw = z.read(name)
                import hashlib
                source_rows.append(dict(index=row['index'], name=row['name'],
                    composition=row['composition'], frames=row['frames'],
                    json_sha256=hashlib.sha256(raw).hexdigest()))
        self.manifest = self.root/'manifest.json'
        save_json(self.manifest, dict(fit_files=source_rows))
        extracted = json.loads((self.evidence/'extraction.json').read_text())
        extracted['source_manifest_sha256'] = digest_file(self.manifest)
        self.replace('extraction.json', extracted)

    def replace(self, name, content):
        save_json(self.evidence/name, content)
        bundle = json.loads((self.evidence/'bundle.json').read_text())
        bundle['files'][name] = digest_file(self.evidence/name)
        save_json(self.evidence/'bundle.json', bundle)

    def run_verifier(self):
        return verify(SimpleNamespace(evidence=self.evidence, manifest=self.manifest,
                                      out=self.root/'verification.json'))

    def test_empty_model_set_cannot_claim_training_provenance(self):
        report = json.loads((self.evidence/'report.json').read_text())
        report['folds'] = {}
        self.replace('report.json', report)
        with self.assertRaisesRegex(ValueError, 'five outer composition models'):
            self.run_verifier()

    def test_valid_counts_cannot_hide_a_wrong_track_identity(self):
        report = json.loads((self.evidence/'report.json').read_text())
        report['tracks'][0]['name'] = 'wrong_track_identity'
        self.replace('report.json', report)
        with self.assertRaisesRegex(ValueError, 'track identity'):
            self.run_verifier()


if __name__ == '__main__':
    unittest.main()
