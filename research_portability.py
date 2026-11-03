"""Export and restore research artifacts without model calls or extra dependencies."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile


BASE = Path(__file__).resolve().parent
MANIFEST = "research-portability-manifest.json"
VERSION = "research-portability-v1"


def local_path(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export_state(base, document_root, output, include_pilot=False):
    base, document_root, output = Path(base).resolve(), Path(document_root).resolve(), Path(output).resolve()
    document_root.relative_to(base / "draft_history" / "parse-api-v1")
    if output.exists():
        raise FileExistsError(f"이미 있는 내보내기 파일을 덮어쓰지 않습니다: {output}")
    candidates = list(local_path(document_root).glob("research-evidence-v3/*/evidence.json"))
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    evidence = json.loads(latest.read_text(encoding="utf-8"))
    pdf = base / "input" / evidence["source"]["file"]
    if digest(pdf.read_bytes()) != evidence["source"]["sha256"]:
        raise ValueError("연구 원본 PDF의 해시가 근거와 다릅니다.")
    roots = [document_root]
    if include_pilot:
        roots.extend(base / name for name in (
            "draft_history/runtime/recovery-v2-pilot", "draft_history/runtime/run_recovery_v2_pilot.py"))
    files = []
    for root in roots:
        files.extend([local_path(root)] if root.is_file() else
                     (path for path in local_path(root).rglob("*") if path.is_file()))
    # Strip the extended Windows prefix before computing portable relative names.
    def relative(path):
        return Path(str(path).removeprefix("\\\\?\\")).relative_to(base).as_posix()
    files.sort(key=relative)
    manifest = {"version": VERSION, "original_workspace": str(base),
                "latest_evidence": relative(latest), "source_pdf": pdf.relative_to(base).as_posix(),
                "source": evidence["source"], "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                if path.suffix.lower() not in (".json", ".md", ".png", ".pdf", ".py", ".txt"):
                    raise ValueError(f"검토되지 않은 파일 형식: {relative(path)}")
                before = path.stat()
                data = path.read_bytes()
                after = path.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    raise RuntimeError(f"내보내는 동안 파일이 변경됐습니다: {relative(path)}")
                name = relative(path)
                manifest["files"].append({"path": name, "size": len(data), "sha256": digest(data)})
                archive.writestr(name, data)
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        # A completed run must remain stable for the entire export, not just each read.
        for record in manifest["files"]:
            if digest(local_path(base / record["path"]).read_bytes()) != record["sha256"]:
                raise RuntimeError(f"내보내는 동안 파일이 변경됐습니다: {record['path']}")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"files": len(files), "archive_bytes": output.stat().st_size, "sha256": digest(output.read_bytes())}


def rebase(value, old_base, new_base):
    if isinstance(value, dict):
        return {key: rebase(item, old_base, new_base) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item, old_base, new_base) for item in value]
    if isinstance(value, str):
        normalized = value.removeprefix("\\\\?\\").replace("\\", "/")
        prefix = old_base.replace("\\", "/").rstrip("/")
        if normalized == prefix or normalized.startswith(prefix + "/"):
            tail = normalized[len(prefix):].lstrip("/")
            return str(Path(new_base) / tail)
    return value


def relocated_bytes(name, data, old_base, new_base):
    """Native responses are immutable; only rewrite derived JSON that contains relocated paths."""
    path = PurePosixPath(name)
    if path.suffix != ".json" or path.name in ("parse-response.json", "parse-failed-response.json") or path.name.endswith(".native.json"):
        return data
    original = json.loads(data)
    value = rebase(original, old_base, new_base)
    if value == original:
        return data
    return (json.dumps(value, ensure_ascii=False, indent=2) + ("\n" if data.endswith(b"\n") else "")).encode("utf-8")


def restore_state(archive_path, base):
    base = Path(base).resolve()
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("version") != VERSION:
            raise ValueError("지원하지 않는 연구 근거 묶음입니다.")
        pdf = base / manifest["source_pdf"]
        if digest(pdf.read_bytes()) != manifest["source"]["sha256"]:
            raise ValueError("먼저 동일한 원본 PDF가 있는 저장소를 준비해주세요.")
        names = [record["path"] for record in manifest["files"]]
        if len(set(names)) != len(names) or sorted(archive.namelist()) != sorted([MANIFEST, *names]):
            raise ValueError("묶음의 파일 목록이 manifest와 다릅니다.")
        # Validate every target and checksum before writing any artifact.
        for record in manifest["files"]:
            name = PurePosixPath(record["path"])
            if name.is_absolute() or ".." in name.parts or "\\" in str(name) or ":" in str(name) or name.parts[0] != "draft_history":
                raise ValueError("복원 대상은 저장소의 draft_history 내부여야 합니다.")
            target = local_path(base / str(name))
            target.relative_to(local_path(base / "draft_history"))
            if target.exists():
                raise FileExistsError(f"기존 연구 기록을 덮어쓰지 않습니다: {target}")
            data = archive.read(record["path"])
            if len(data) != record["size"] or digest(data) != record["sha256"]:
                raise ValueError(f"파일 검증 실패: {name}")
        for record in manifest["files"]:
            target = local_path(base / record["path"])
            data = relocated_bytes(record["path"], archive.read(record["path"]), manifest["original_workspace"], base)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
    return {"restored_files": len(manifest["files"]), "latest_evidence": str(base / manifest["latest_evidence"])}


def repair_parse_caches(archive_path, base):
    """Undo only the proven JSON reserialization bug, preserving checksums and local edits."""
    base = Path(base).resolve()
    repairs = []
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("version") != VERSION:
            raise ValueError("지원하지 않는 연구 근거 묶음입니다.")
        if digest((base / manifest["source_pdf"]).read_bytes()) != manifest["source"]["sha256"]:
            raise ValueError("연구 원본 PDF가 이관 묶음과 다릅니다.")
        for record in manifest["files"]:
            name = PurePosixPath(record["path"])
            if name.name not in ("parse-response.json", "parse-failed-response.json"):
                continue
            if name.is_absolute() or ".." in name.parts or "\\" in str(name) or ":" in str(name) or name.parts[0] != "draft_history":
                raise ValueError("복구 대상은 저장소의 draft_history 내부여야 합니다.")
            original = archive.read(record["path"])
            if len(original) != record["size"] or digest(original) != record["sha256"]:
                raise ValueError(f"이관 원본 검증 실패: {name}")
            path = local_path(base / str(name))
            path.relative_to(local_path(base / "draft_history"))
            current = path.read_bytes()
            metadata_path = path.with_name(name.name.replace("-response.json", "-manifest.json"))
            if name.name == "parse-failed-response.json" and not metadata_path.is_file():
                # HTTP error logs have no cache manifest and are not used as empty-image cache entries.
                continue
            metadata = json.loads(metadata_path.read_bytes())
            if digest(current) == metadata.get("response_sha256"):
                continue
            old_bug_output = json.dumps(rebase(json.loads(original), manifest["original_workspace"], base),
                                        ensure_ascii=False, indent=2).encode("utf-8")
            if digest(original) != metadata.get("response_sha256") or current != old_bug_output:
                raise ValueError(f"복원 형식 변경으로 확인할 수 없는 파일은 수정하지 않습니다: {name}")
            repairs.append((str(name), path, current, original))
    # Finish all checks and preserve every current file before replacing any cache.
    if not repairs:
        return {"repaired_files": 0, "backup_dir": None}
    backup = local_path(base / "draft_history/cache-repairs" /
                        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
    for name, path, current, _ in repairs:
        if path.read_bytes() != current:
            raise RuntimeError("검사 중 캐시가 바뀌어 복구를 중단했습니다.")
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open("xb") as handle:
            handle.write(current)
    for _, path, current, original in repairs:
        if path.read_bytes() != current:
            raise RuntimeError("복구 중 캐시가 바뀌었습니다. 보존한 백업을 확인해주세요.")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(original)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    report = {"repaired_files": len(repairs), "backup_dir": str(backup),
              "files": [{"path": name, "previous_sha256": digest(current), "restored_sha256": digest(original)}
                        for name, _, current, original in repairs]}
    (backup / "repair-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--document-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-pilot", action="store_true")
    restore = commands.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--base-dir", type=Path, default=BASE)
    repair = commands.add_parser("repair-parse-cache")
    repair.add_argument("archive", type=Path)
    repair.add_argument("--base-dir", type=Path, default=BASE)
    args = parser.parse_args()
    if args.command == "export":
        result = export_state(BASE, args.document_root, args.output, args.include_pilot)
    elif args.command == "restore":
        result = restore_state(args.archive, args.base_dir)
    else:
        result = repair_parse_caches(args.archive, args.base_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
