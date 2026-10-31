"""Safe, atomic artifact I/O shared by analysis and capture workflows.

This module owns filesystem mechanics only.  Domain pipelines keep their
validation and report semantics while importing these primitives through a
small, stable boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_paths_distinct(**named_paths: str | Path | None) -> None:
    """Reject aliases between any supplied input and output paths."""

    entries = [(name, Path(value)) for name, value in named_paths.items() if value is not None]
    for index, (left_name, left_path) in enumerate(entries):
        left_resolved = left_path.resolve(strict=False)
        left_key = os.path.normcase(str(left_resolved))
        for right_name, right_path in entries[index + 1 :]:
            right_resolved = right_path.resolve(strict=False)
            same_path = left_key == os.path.normcase(str(right_resolved))
            if not same_path and left_path.exists() and right_path.exists():
                try:
                    same_path = os.path.samefile(left_path, right_path)
                except OSError:
                    same_path = False
            if same_path:
                raise ValueError(
                    f"Path collision: {left_name} and {right_name} refer to {left_resolved}"
                )


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Atomically replace a text output after its complete content is durable."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return target.resolve()


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Atomically replace a binary evidence file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return target.resolve()


def atomic_write_text_bundle(outputs: dict[str | Path, str]) -> dict[Path, Path]:
    """Stage related text outputs and roll them back together on commit failure."""

    entries = [(Path(path), content) for path, content in outputs.items()]
    if not entries:
        return {}

    resolved_targets = [target.resolve(strict=False) for target, _ in entries]
    normalized = [os.path.normcase(str(target)) for target in resolved_targets]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Output bundle paths must be distinct")
    for index, left in enumerate(resolved_targets):
        for right in resolved_targets[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("Output bundle paths must not contain one another")
    for target, _ in entries:
        if target.is_symlink():
            raise ValueError(f"Output bundle target must not be a symbolic link: {target}")
        if target.exists() and not target.is_file():
            raise IsADirectoryError(target)

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for target, content in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                staged[target] = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

        for target, _ in entries:
            if not target.exists():
                continue
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.",
                suffix=".backup",
                dir=target.parent,
                delete=False,
            ) as stream:
                backup = Path(stream.name)
            backup.unlink()
            backups[target] = backup
            os.replace(target, backup)

        for target, _ in entries:
            promoted.append(target)
            os.replace(staged[target], target)
    except BaseException as error:
        rollback_errors: list[BaseException] = []
        for target in reversed(promoted):
            try:
                target.unlink(missing_ok=True)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        for target, backup in reversed(list(backups.items())):
            if not backup.exists():
                continue
            try:
                os.replace(backup, target)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "Output bundle commit failed and rollback was incomplete"
            ) from error
        raise
    finally:
        for temporary_path in staged.values():
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Reaching this point is the bundle commit point. Backup cleanup is
    # deliberately outside the rollback region: an interruption here must
    # never remove already-promoted outputs or partially restore old ones.
    for backup in backups.values():
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass

    return {target: target.resolve() for target, _ in entries}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_snapshot(path: str | Path) -> tuple[dict[str, Any], str]:
    """Parse JSON and hash the exact same immutable byte snapshot."""

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The JSON root must be an object")
    return payload, digest


__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_text_bundle",
    "ensure_paths_distinct",
    "load_json_snapshot",
    "sha256_file",
]
