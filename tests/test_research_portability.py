import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from research_portability import MANIFEST, VERSION, export_state, restore_state, repair_parse_caches, rebase
from document_parse import DEFAULT_MODEL, parse_document


class ResearchPortabilityTests(unittest.TestCase):
    def add_parse(self, base, root):
        root.mkdir(parents=True, exist_ok=True)
        native = {"elements": [{"id": 1, "page": 1, "category": "paragraph", "content": {"text": "원문 그대로"}}],
                  "original_metadata": str(base / "native-source")}
        data = (json.dumps(native, ensure_ascii=False, indent=2) + "\r\n").encode("utf-8")
        (root / "parse-response.json").write_bytes(data)
        metadata = {"source_sha256": hashlib.sha256((base / "input/research.pdf").read_bytes()).hexdigest(),
            "request_settings": {"model": DEFAULT_MODEL, "mode": "standard", "ocr": "auto",
                                 "output_formats": '["text", "html", "markdown"]', "coordinates": "true"},
            "response_sha256": hashlib.sha256(data).hexdigest()}
        (root / "parse-manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        return data, native

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

    def test_restored_native_responses_remain_byte_exact_and_parse_reuses_cache_without_api(self):
        with TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            old.mkdir()
            root, _ = self.prepare(old)
            original, native = self.add_parse(old, root)
            (root / "answer.native.json").write_bytes(original)
            archive = old / "state.zip"
            export_state(old, root, archive)
            (new / "input").mkdir(parents=True)
            (new / "input/research.pdf").write_bytes((old / "input/research.pdf").read_bytes())
            restore_state(archive, new)
            restored_root = new / root.relative_to(old)
            self.assertEqual((restored_root / "parse-response.json").read_bytes(), original)
            self.assertEqual((restored_root / "answer.native.json").read_bytes(), original)
            self.assertEqual((restored_root / "research-evidence-v3/run/human-fact-reviews.json").read_bytes(),
                             (root / "research-evidence-v3/run/human-fact-reviews.json").read_bytes())
            with patch("document_parse.requests.post", side_effect=AssertionError("external API called")) as post:
                actual = parse_document(new / "input/research.pdf", restored_root)
            self.assertEqual(actual, native)
            post.assert_not_called()

    def test_proven_old_restore_formatting_is_backed_up_and_repaired_without_changing_manifest(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, _ = self.prepare(base)
            original, _ = self.add_parse(base, root)
            archive = base / "state.zip"
            export_state(base, root, archive)
            raw = root / "parse-response.json"
            broken = json.dumps(rebase(json.loads(original), str(base), base), ensure_ascii=False, indent=2).encode("utf-8")
            raw.write_bytes(broken)
            manifest_before = (root / "parse-manifest.json").read_bytes()
            result = repair_parse_caches(archive, base)
            self.assertEqual(result["repaired_files"], 1)
            self.assertEqual(raw.read_bytes(), original)
            self.assertEqual((root / "parse-manifest.json").read_bytes(), manifest_before)
            self.assertEqual((Path(result["backup_dir"]) / raw.relative_to(base)).read_bytes(), broken)
            self.assertEqual(repair_parse_caches(archive, base)["repaired_files"], 0)

    def test_unrecognized_changes_are_rejected_before_any_cache_is_replaced(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, _ = self.prepare(base)
            original, _ = self.add_parse(base, root)
            self.add_parse(base, root / "second")
            archive = base / "state.zip"
            export_state(base, root, archive)
            broken = json.dumps(json.loads(original), ensure_ascii=False, indent=2).encode("utf-8")
            (root / "parse-response.json").write_bytes(broken)
            (root / "second/parse-response.json").write_text('{"user_edit": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "수정하지 않습니다"):
                repair_parse_caches(archive, base)
            self.assertEqual((root / "parse-response.json").read_bytes(), broken)
            self.assertEqual((root / "second/parse-response.json").read_text(encoding="utf-8"), '{"user_edit": true}')

    def test_empty_parse_response_cache_keeps_its_hash_for_full_page_fallback(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, _ = self.prepare(base)
            original = b'{\n  "elements": []\n}\n'
            failed = root / "parse-failed-response.json"
            failed.write_bytes(original)
            (root / "parse-failed-manifest.json").write_text(json.dumps({
                "response_sha256": hashlib.sha256(original).hexdigest()}), encoding="utf-8")
            http_error = root / "http-error/parse-failed-response.json"
            http_error.parent.mkdir()
            http_error.write_bytes(b'{"error": "previous HTTP failure"}\n')
            archive = base / "state.zip"
            export_state(base, root, archive)
            failed.write_bytes(json.dumps(json.loads(original), ensure_ascii=False, indent=2).encode("utf-8"))
            result = repair_parse_caches(archive, base)
            self.assertEqual(result["repaired_files"], 1)
            self.assertEqual(failed.read_bytes(), original)
            self.assertEqual(http_error.read_bytes(), b'{"error": "previous HTTP failure"}\n')

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
