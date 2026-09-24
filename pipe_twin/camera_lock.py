"""OS camera leases shared by the GUI and capture worker.

Run the GUI and agent under the same account. Unrelated applications do not
participate in these advisory locks. The OS releases locks after a kill.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO


class CameraBusyError(ValueError):
    """Another pipe-twin process owns a requested camera."""


class CameraLease:
    def __init__(self, indices: list[int]):
        if not indices or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("camera indices must be non-negative integers")
        self.indices = sorted(set(indices))
        self._streams: list[BinaryIO] = []

    def acquire(self) -> None:
        if self._streams:
            raise RuntimeError("camera lease is already acquired")
        default_base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "PipeTwin" / "camera_locks"
        base = Path(os.environ.get("PIPE_TWIN_CAMERA_LOCK_DIR") or default_base)
        base.mkdir(parents=True, exist_ok=True)
        try:
            for index in self.indices:
                path = base / f"camera-{index}.lock"
                path.touch(exist_ok=True)
                stream = path.open("r+b")
                try:
                    # Never delete a lock file: replacement breaks the shared lock.
                    if stream.seek(0, os.SEEK_END) == 0:
                        stream.write(b"0")
                        stream.flush()
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    stream.close()
                    raise CameraBusyError(f"CAMERA_BUSY: camera index {index}") from error
                except BaseException:
                    stream.close()
                    raise
                self._streams.append(stream)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        while self._streams:
            self._streams.pop().close()

    def __enter__(self) -> "CameraLease":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
