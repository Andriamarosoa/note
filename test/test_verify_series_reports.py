"""Regression tests: every S788+ requires two truthful, consistent reports."""
from __future__ import annotations
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.verify_series_reports import from_branch, validate_series, ROLES

SOURCE = Path(__file__).resolve().parents[1] / "series" / "S788"


class MandatorySeriesReportsTests(unittest.TestCase):
    def fixture(self):
        ctx=TemporaryDirectory()
        self.addCleanup(ctx.cleanup)
        root=Path(ctx.name)
        (root/"series").mkdir()
        shutil.copytree(SOURCE,root/"series"/"S788")
        # Tests use a synthetic DRAFT fixture even after the committed S788
        # series itself has moved to a verified, completed state.
        for name in ("resume.md", "resultat.md"):
            path = root / "series" / "S788" / name
            txt = path.read_text(encoding="utf-8")
            txt = txt.replace("statut: termine", "statut: en_cours", 1)
            txt = txt.replace("run_id: 39272602006", "run_id: en_attente", 1)
            txt = txt.replace(
                "commit_sha: 6dc47ac5056458a86f5ecab9a981bc7ed8603bba",
                "commit_sha: en_attente", 1)
            path.write_text(txt, encoding="utf-8")
        return root

    def replace(self,root,name,a,b):
        path=root/"series"/"S788"/name
        txt=path.read_text(encoding="utf-8")
        self.assertIn(a,txt)
        path.write_text(txt.replace(a,b),encoding="utf-8")

    def test_new_series_valid_draft(self):
        root=self.fixture()
        info=validate_series(root,"S788")
        self.assertEqual(info["status"],"en_cours")
        self.assertEqual(info["promotion"],"false")

    def test_branch_identity_includes_series(self):
        self.assertEqual(
            from_branch("codex/note-s788-mandatory-series-reports"),"S788")
        self.assertEqual(from_branch("codex/note-s1024-fragment-heads"),"S1024")
        with self.assertRaises(ValueError):
            from_branch("codex/note-s787-six-auditor")

    def test_missing_resume_or_resultat_fails(self):
        for missing in ("resume.md","resultat.md"):
            root=self.fixture()
            (root/"series"/"S788"/missing).unlink()
            with self.subTest(missing=missing),self.assertRaisesRegex(
                ValueError,"missing mandatory file"):
                validate_series(root,"S788")

    def test_conflicting_status_or_reference_fails(self):
        root=self.fixture()
        self.replace(root,"resultat.md","statut: en_cours","statut: termine")
        with self.assertRaises(ValueError):validate_series(root,"S788")
        root=self.fixture()
        self.replace(root,"resultat.md","reference: S708","reference: S777")
        with self.assertRaises(ValueError):validate_series(root,"S788")

    def test_release_mode_refuses_fake_success(self):
        root=self.fixture()
        with self.assertRaisesRegex(ValueError,"cannot release"):
            validate_series(root,"S788","release")
        for name in ("resume.md","resultat.md"):
            self.replace(root,name,"statut: en_cours","statut: termine")
        with self.assertRaisesRegex(ValueError,"needs a GitHub run ID"):
            validate_series(root,"S788","release")

    def test_missing_required_heading_fails(self):
        root=self.fixture()
        self.replace(root,"resultat.md","## Métriques","## Métriques absent")
        with self.assertRaisesRegex(ValueError,"missing section"):
            validate_series(root,"S788")

    def make_completed_experience(self,root):
        for name in ("resume.md","resultat.md"):
            self.replace(root,name,"statut: en_cours","statut: termine")
            self.replace(root,name,"type: infrastructure","type: experience")
            self.replace(root,name,"run_id: en_attente","run_id: 12345678901")
            self.replace(root,name,"commit_sha: en_attente","commit_sha: "+("a"*40))
        path=root/"series"/"S788"/"resultat.md"
        path.write_text(path.read_text(encoding="utf-8")+
            "\nRun GitHub vérifié : https://github.com/Andriamarosoa/note/actions/runs/12345678901\n",
            encoding="utf-8")
        return path

    def test_completed_experience_requires_all_six_audit_results(self):
        root=self.fixture()
        report=self.make_completed_experience(root)
        with self.assertRaisesRegex(ValueError,"audit table row"):
            validate_series(root,"S788","release")
        with report.open("a",encoding="utf-8") as out:
            out.write("\n| Rôle | Statut | Preuve |\n| --- | --- | --- |\n")
            for role in ROLES:
                out.write(f"| {role} | PASS | github-run-12345678901 |\n")
        result=validate_series(root,"S788","release")
        self.assertEqual(result["status"],"termine")

    def test_promotion_refused_if_an_audit_fails(self):
        root=self.fixture()
        report=self.make_completed_experience(root)
        for name in ("resume.md","resultat.md"):
            self.replace(root,name,"promotion: false","promotion: true")
        with report.open("a",encoding="utf-8") as out:
            out.write("\n| Rôle | Statut | Preuve |\n| --- | --- | --- |\n")
            for role in ROLES:
                out.write(f"| {role} | {'FAIL' if role=='LIVE' else 'PASS'} | run-12345678901 |\n")
        with self.assertRaisesRegex(ValueError,"promotion blocked by LIVE"):
            validate_series(root,"S788","release")


if __name__ == "__main__":
    unittest.main()
