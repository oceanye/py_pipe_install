from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipe_twin.cli import _write_json, main
from pipe_twin.pipeline import (
    AssetIntegrityError,
    analyze_manifest,
    atomic_write_text_bundle,
    load_json_snapshot,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "test_model" / "manifest.json"
MODEL_PATH = ROOT / "test_model" / "管道群.3mf"
VIDEO_PATH = ROOT / "test_model" / "管道群.mkv"


class ManifestSnapshotTests(unittest.TestCase):
    def test_snapshot_parses_and_hashes_one_identical_byte_sequence(self) -> None:
        raw = '{"name":"管道群","value":3}\n'.encode("utf-8")
        with mock.patch.object(Path, "read_bytes", autospec=True, return_value=raw) as reader:
            payload, digest = load_json_snapshot("ignored-by-mock.json")

        self.assertEqual(payload, {"name": "管道群", "value": 3})
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(reader.call_count, 1)

    def test_asset_hash_mismatch_fails_before_video_replay(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["model"]["path"] = str(MODEL_PATH.resolve())
        manifest["video"]["path"] = str(VIDEO_PATH.resolve())
        manifest["video"]["sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as temporary:
            manifest_copy = Path(temporary) / "manifest.json"
            manifest_copy.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(AssetIntegrityError):
                analyze_manifest(manifest_copy)


class OutputPathSafetyTests(unittest.TestCase):
    def test_pipeline_rejects_each_output_colliding_with_any_bound_input(self) -> None:
        inputs = {
            "manifest": MANIFEST_PATH,
            "model": MODEL_PATH,
            "video": VIDEO_PATH,
        }
        hashes_before = {name: sha256_file(path) for name, path in inputs.items()}

        for output_name in ("observations_path", "report_output_path"):
            for input_name, input_path in inputs.items():
                with self.subTest(output=output_name, input=input_name):
                    with self.assertRaises(ValueError):
                        analyze_manifest(MANIFEST_PATH, **{output_name: input_path})
                    self.assertEqual(sha256_file(input_path), hashes_before[input_name])

    def test_pipeline_rejects_report_and_observations_alias_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "same-output.json"

            with self.assertRaises(ValueError):
                analyze_manifest(
                    MANIFEST_PATH,
                    observations_path=output,
                    report_output_path=output,
                )

            self.assertFalse(output.exists())

    def test_inspect_cli_rejects_model_as_its_own_output_without_mutation(self) -> None:
        digest_before = sha256_file(MODEL_PATH)

        with self.assertRaises(ValueError):
            main(["inspect-model", str(MODEL_PATH), "--output", str(MODEL_PATH)])

        self.assertEqual(sha256_file(MODEL_PATH), digest_before)

    def test_analyze_cli_preflights_report_collision_without_mutating_manifest(self) -> None:
        digest_before = sha256_file(MANIFEST_PATH)

        with self.assertRaises(ValueError):
            main(
                [
                    "analyze",
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--output",
                    str(MANIFEST_PATH),
                ]
            )

        self.assertEqual(sha256_file(MANIFEST_PATH), digest_before)

    def test_json_writer_atomically_creates_valid_utf8_output(self) -> None:
        payload = {"name": "管道群", "passed": True}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"

            resolved = _write_json(output, payload)

            self.assertEqual(resolved, output.resolve())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertTrue(output.read_bytes().endswith(b"\n"))
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
                "Atomic writer left a temporary file behind",
            )

    def test_output_bundle_restores_both_previous_files_if_second_promote_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            observations = root / "observations.jsonl"
            report.write_text("old-report", encoding="utf-8")
            observations.write_text("old-observations", encoding="utf-8")
            real_replace = os.replace

            def fail_second_promote(source: str | Path, destination: str | Path) -> None:
                source_path = Path(source)
                if Path(destination) == observations and source_path.suffix == ".tmp":
                    real_replace(source, destination)
                    raise OSError("injected second-output failure")
                real_replace(source, destination)

            with mock.patch(
                "pipe_twin.pipeline.os.replace", side_effect=fail_second_promote
            ):
                with self.assertRaises(OSError):
                    atomic_write_text_bundle(
                        {
                            report: "new-report",
                            observations: "new-observations",
                        }
                    )

            self.assertEqual(report.read_text(encoding="utf-8"), "old-report")
            self.assertEqual(
                observations.read_text(encoding="utf-8"), "old-observations"
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])
            self.assertEqual(list(root.glob(".*.backup")), [])

    def test_output_bundle_cleanup_interrupt_cannot_roll_back_committed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            observations = root / "observations.jsonl"
            report.write_text("old-report", encoding="utf-8")
            observations.write_text("old-observations", encoding="utf-8")
            real_unlink = Path.unlink
            backup_unlink_calls = 0

            def interrupt_first_committed_backup(
                path: Path, missing_ok: bool = False
            ) -> None:
                nonlocal backup_unlink_calls
                if path.suffix == ".backup":
                    backup_unlink_calls += 1
                    if backup_unlink_calls == 3:
                        raise KeyboardInterrupt("injected backup-cleanup interruption")
                real_unlink(path, missing_ok=missing_ok)

            with mock.patch.object(
                Path, "unlink", autospec=True, side_effect=interrupt_first_committed_backup
            ):
                with self.assertRaises(KeyboardInterrupt):
                    atomic_write_text_bundle(
                        {
                            report: "new-report",
                            observations: "new-observations",
                        }
                    )

            self.assertEqual(report.read_text(encoding="utf-8"), "new-report")
            self.assertEqual(
                observations.read_text(encoding="utf-8"), "new-observations"
            )


if __name__ == "__main__":
    unittest.main()
