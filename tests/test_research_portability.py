import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from research_portability import MANIFEST, VERSION, export_state, restore_state


class ResearchPortabilityTests(unittest.TestCase):
    def prepare(self, base):
        source = base / "input" / "research.pdf"
        source.parent.mkdir()
        source.write_bytes(b"same research source")
        root = base / "draft_history/parse-api-v1/document"
        run = root / "research-evidence-v3/run"
        run.mkdir(parents=True)
        evidence = {"source": {"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                    "run_dir": str(run), "source_archive_path": str(run / "source-archive.json"),
                    "items": [{"statement": "Keep 12.5ms", "support": "unsupported"}],
                    "completion": {"ready_for_drafting": False},
                    "evidence_recovery_history": [str(run / "pending-recovery/old/recovery-report.json")]}
        (run / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        (run / "source-archive.json").write_text(json.dumps({"source": evidence["source"], "elements": []}), encoding="utf-8")
        (run / "human-fact-reviews.json").write_text('{"events": []}', encoding="utf-8")
        return root, evidence

    def test_roundtrip_relocates_paths_and_preserves_facts_gate_and_human_records(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary) / "old"
            base.mkdir()
            root, evidence = self.prepare(base)
            archive = base / "state.zip"
            result = export_state(base, root, archive)
            self.assertEqual(result["files"], 3)
            target = Path(temporary) / "new"
            (target / "input").mkdir(parents=True)
            (target / "input/research.pdf").write_bytes((base / "input/research.pdf").read_bytes())
            restored = restore_state(archive, target)
            actual = json.loads(Path(restored["latest_evidence"]).read_text(encoding="utf-8"))
            self.assertEqual(actual["items"], evidence["items"])
            self.assertFalse(actual["completion"]["ready_for_drafting"])
            self.assertTrue(Path(actual["source_archive_path"]).is_file())
            self.assertTrue((Path(actual["run_dir"]) / "human-fact-reviews.json").is_file())
            self.assertTrue(actual["evidence_recovery_history"][0].startswith(str(target)))
            with self.assertRaises(FileExistsError):
                restore_state(archive, target)

    def test_wrong_pdf_is_rejected_before_writing(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, _ = self.prepare(base)
            archive = base / "state.zip"
            export_state(base, root, archive)
            (base / "input/research.pdf").write_bytes(b"different")
            with self.assertRaises(ValueError):
                restore_state(archive, base)

    def test_traversal_and_corrupt_files_are_rejected_before_writing(self):
        for name, corrupt in (("draft_history/../../outside.json", False), ("draft_history/new.json", True)):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                base = Path(temporary)
                _, evidence = self.prepare(base)
                data = b'{}'
                manifest = {"version": VERSION, "original_workspace": str(base),
                            "source_pdf": "input/research.pdf", "source": evidence["source"],
                            "files": [{"path": name, "size": len(data),
                                       "sha256": "bad" if corrupt else hashlib.sha256(data).hexdigest()}]}
                archive = base / "bad.zip"
                with ZipFile(archive, "w") as handle:
                    handle.writestr(MANIFEST, json.dumps(manifest))
                    handle.writestr(name, data)
                with self.assertRaises(ValueError):
                    restore_state(archive, base)
                self.assertFalse((base / "draft_history/new.json").exists())


if __name__ == "__main__":
    unittest.main()
