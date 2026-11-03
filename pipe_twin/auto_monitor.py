"""Serial single-pair monitoring, independent of Tk and workbench preferences.

The camera reader continuously drains the device while computation runs. Each
cycle waits for a read that began *after* its capture request; cached preview
frames and previously captured files are never reused. A local analyzer that
does not accept cancel_event is cancelled at progress/return checkpoints.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import queue
import tempfile
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis_transport import RemoteAnalysisCancelled, RemoteAnalysisClient
from .calibration_wizard import rectifier_for_calibration
from .capture_contracts import field_calibration_problem
from .elevation_dataset import _calibration_ok, create_elevation_dataset
from .remote_capture import _encode_png, _pair_image_health
from .stereo_analyzer import _parse_timestamp, analyze_stereo_capture
from .stereo_camera import LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL, StereoCameraSession
from .workbench_profile import validate_rectification_recipe


class _Stopped(Exception):
    pass


class SingleFrameMonitor:
    """Nonblocking start/stop; ``active`` stays true until both workers exit.

    ``dataset_arguments`` uses create_elevation_dataset keyword names. Captured
    image paths/timestamps/provenance, previous_manifest and temporal_frames
    are always replaced per cycle. ``camera_options`` uses StereoCameraSession
    keyword names; an optional right_frame_transform must match the recipe.
    Factories/analyzers run exclusively on workers. ``clock`` is an injectable
    monotonic clock for deterministic scheduling tests.

    Events are JSON-compatible dicts with kind/cycle_id/at. State events have
    state; progress/error have message; capture/result have manifest_path,
    captured_at and report_path; result additionally has report. The bounded
    queue retains the latest 128 events if a consumer stops draining it.
    """
    def __init__(self, dataset_arguments: dict, camera_options: dict, *,
                 interval_seconds: float = 60, mode: str = "local", server_url: str | None = None,
                 session_factory: Callable | None = None, analyzer: Callable | None = None,
                 warmup_seconds: float = 8.0, clock: Callable[[], float] = time.monotonic) -> None:
        for name, value in (("interval_seconds", interval_seconds), ("warmup_seconds", warmup_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if interval_seconds <= 0 or warmup_seconds < 0:
            raise ValueError("interval_seconds must be positive; warmup_seconds must be non-negative")
        if mode not in {"local", "client"}:
            raise ValueError("自动监控只支持本地或前端模式，服务器模式不可采集")
        if mode == "client" and not server_url:
            raise ValueError("前端监控需要计算服务器地址")
        if not isinstance(dataset_arguments, dict) or not isinstance(camera_options, dict):
            raise ValueError("dataset_arguments and camera_options must be dictionaries")
        self._arguments = copy.deepcopy(dataset_arguments)
        self._camera_options = copy.deepcopy(camera_options)
        self.interval_seconds = float(interval_seconds)
        self.warmup_seconds = float(warmup_seconds)
        self.mode, self.server_url = mode, server_url
        self._session_factory = session_factory or StereoCameraSession
        self._analyzer = analyzer
        self._clock = clock
        self.events: queue.Queue[dict] = queue.Queue(maxsize=128)
        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._active = False
        self._next_due: float | None = None
        self._cycle_id = 0
        self._worker: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._latest: tuple[int, float, Any] | None = None
        self._reader_error: Exception | None = None
        self._opened_at: float | None = None
        self._last_capture_times: tuple | None = None

    @property
    def active(self) -> bool:
        with self._condition:
            return self._active

    @property
    def next_due(self) -> float | None:
        with self._condition:
            return self._next_due

    def _emit(self, kind: str, **values: Any) -> None:
        event = {"kind": kind, "cycle_id": self._cycle_id, "at": self._clock(), **values}
        with self._condition:
            # Linearize cancellation and publication: no capture, progress or
            # success may be queued after stop() has acquired this same lock.
            if kind in {"capture", "progress", "result"} or (
                    kind == "state" and values.get("state") != "STOPPING"):
                self._check()
            try:
                self.events.put_nowait(event)
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass
                self.events.put_nowait(event)

    def _state(self, state: str, **values: Any) -> None:
        self._emit("state", state=state, next_due=self.next_due, **values)

    def start(self) -> SingleFrameMonitor:
        with self._condition:
            if self._active:
                raise RuntimeError("监控尚未结束，请等待停止完成后再启动")
            # Previous terminal events must not stop a newly restarted GUI run.
            while not self.events.empty():
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    break
            self._stop.clear()
            self._active = True
            self._next_due = None
            self._cycle_id = 0
            self._latest = None
            self._reader_error = None
            self._opened_at = None
            self._last_capture_times = None
            self._reader = None
            self._worker = threading.Thread(target=self._run, name="single-frame-monitor", daemon=True)
            self._worker.start()
        return self

    def stop(self) -> None:
        """Request cancellation without waiting on a camera driver or analysis."""
        with self._condition:
            if not self._active or self._stop.is_set():
                return
            self._stop.set()
            self._next_due = None
            self._state("STOPPING")
            self._condition.notify_all()

    def _check(self) -> None:
        if self._reader_error is not None:
            raise self._reader_error
        if self._stop.is_set():
            raise _Stopped()

    def _wait_until(self, deadline: float | None) -> None:
        with self._condition:
            self._check()
            if deadline is None:
                raise _Stopped()
            while self._clock() < deadline:
                self._check()
                self._condition.wait(timeout=min(.05, max(.001, deadline - self._clock())))
            self._check()

    def _prepare(self) -> None:
        arguments = self._arguments
        problem = field_calibration_problem(arguments["calibration"])
        if problem:
            raise ValueError(problem)
        self._calibration = _calibration_ok(arguments["calibration"])
        options = self._camera_options
        if options.get("layout") not in {LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL}:
            raise ValueError("自动监控仅支持同一 UVC 帧并排双目；独立相机请使用人工同步确认抓拍")
        if (options.get("eye_width"), options.get("eye_height")) != (
                self._calibration.left.width, self._calibration.left.height):
            raise ValueError("相机输出尺寸与当前标定不一致")
        recipe = arguments.get("rectification_recipe")
        if recipe is not None:
            recipe = validate_rectification_recipe(recipe, calibration=arguments["calibration"])
            arguments["rectification_recipe"] = recipe
        transform = options.pop("right_frame_transform", None)
        if transform is not None and transform != (recipe or {}).get("right_frame_transform", "none"):
            raise ValueError("相机右目方向与标定矫正配方不一致")
        self._rectifier = rectifier_for_calibration(arguments["calibration"], {"rectification_recipe": recipe})
        self._model_path = Path(arguments["model_path"]).resolve()
        self._model_hash = hashlib.sha256(self._model_path.read_bytes()).hexdigest()
        self._output_root = Path(arguments["output_root"]).resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._client = RemoteAnalysisClient(self.server_url) if self.mode == "client" else None
        if self._client is not None:
            self._client.health()
        self._check()

    def _read_camera(self) -> None:
        session = None
        try:
            self._check()
            session = self._session_factory(**self._camera_options)
            session.open()
            with self._condition:
                self._opened_at = self._clock()
                self._condition.notify_all()
            sequence = 0
            while not self._stop.is_set():
                started = self._clock()
                pair = session.read_pair()
                if self._stop.is_set():
                    break
                # Drivers/test doubles may reuse backing arrays. Freeze the
                # latest pair before another read can replace those pixels.
                pair = replace(pair, left=pair.left.copy(), right=pair.right.copy(),
                               provenance=copy.deepcopy(pair.provenance))
                sequence += 1
                with self._condition:
                    self._latest = (sequence, started, pair)
                    self._condition.notify_all()
                # Real devices block in read_pair; this also bounds fast fakes.
                self._stop.wait(.001)
        except Exception as error:
            if not self._stop.is_set():
                with self._condition:
                    self._reader_error = error
                    self._condition.notify_all()
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as error:
                    self._emit("error", stage="camera_close", message=str(error))
            with self._condition:
                self._condition.notify_all()

    def _capture(self, after: float) -> Any:
        deadline = self._clock() + 10.0
        with self._condition:
            sequence = self._latest[0] if self._latest else 0
            while True:
                self._check()
                latest = self._latest
                if latest is not None and latest[0] > sequence and latest[1] >= after:
                    pair = latest[2]
                    break
                if self._clock() >= deadline:
                    raise TimeoutError("等待本轮新相机帧超时；未使用缓存图像")
                self._condition.wait(timeout=.05)
        times = tuple(_parse_timestamp(getattr(pair, f"{role}_captured_at"), f"{role}_captured_at")
                      for role in ("left", "right"))
        if self._last_capture_times is not None and any(a <= b for a, b in zip(times, self._last_capture_times)):
            raise ValueError("相机返回旧时间戳；未复用上次图像")
        sync = float(pair.sync_delta_ms)
        actual_sync = abs((times[1] - times[0]).total_seconds()) * 1000
        if not math.isfinite(sync) or sync < 0 or max(sync, actual_sync) > self._calibration.max_sync_delta_ms + 1e-6:
            raise ValueError("左右抓拍不同步，超过标定允许的时间差")
        for role in ("left", "right"):
            frame = getattr(pair, role)
            camera = getattr(self._calibration, role)
            if frame.dtype != np.uint8 or frame.shape != (camera.height, camera.width, 3):
                raise ValueError(f"{role} 相机图像尺寸或格式与标定不一致")
        health = _pair_image_health(pair)
        if not health["pair_usable"]:
            raise ValueError("相机图像不可用：" + ", ".join(health["reason_codes"]))
        if self._rectifier is not None:
            pair = replace(pair, left=self._rectifier.rectify("left", pair.left),
                           right=self._rectifier.rectify("right", pair.right))
        health = _pair_image_health(pair)
        if not health["pair_usable"]:
            raise ValueError("矫正后图像不可用：" + ", ".join(health["reason_codes"]))
        self._last_capture_times = times
        return pair

    def _save_capture(self, pair: Any) -> Path:
        self._check()
        if hashlib.sha256(self._model_path.read_bytes()).hexdigest() != self._model_hash:
            raise ValueError("监控期间模型文件发生变化，请停止并重新确认配置")
        arguments = copy.deepcopy(self._arguments)
        # Manual OBS identifiers belong to the old photograph. Never attach
        # those assignments to newly captured pixel hashes automatically.
        registration = dict(arguments.get("registration_settings") or {})
        registration["anchors"] = {}
        arguments["registration_settings"] = registration
        for key in ("left_path", "right_path", "left_time", "right_time", "timestamp_sources",
                    "camera_capture_provenance", "pair_confirmed", "previous_manifest", "temporal_frames"):
            arguments.pop(key, None)
        with tempfile.TemporaryDirectory(prefix="m", dir=self._output_root) as staging:
            paths = {role: Path(staging) / f"{role}.png" for role in ("left", "right")}
            for role, path in paths.items():
                path.write_bytes(_encode_png(getattr(pair, role)))
            self._check()
            manifest = create_elevation_dataset(**arguments, left_path=paths["left"], right_path=paths["right"],
                left_time=pair.left_captured_at, right_time=pair.right_captured_at,
                timestamp_sources={role: pair.timestamp_source for role in paths},
                camera_capture_provenance=copy.deepcopy(pair.provenance), pair_confirmed=True,
                previous_manifest=None, temporal_frames=[])
        payload = json.loads(manifest.read_bytes())
        if payload["model"]["sha256"] != self._model_hash:
            raise ValueError("监控期间模型文件发生变化，当前现场未进入分析")
        return manifest

    def _analyze(self, manifest: Path) -> dict:
        self._check()
        options = {"report_output_path": manifest.parent / "report.json",
                   "evidence_dir": manifest.parent / "evidence", "progress": self._progress}
        if self._client is not None:
            return self._client.analyze(manifest, **options, cancel_event=self._stop)
        analyzer = self._analyzer or analyze_stereo_capture
        parameters = inspect.signature(analyzer).parameters
        if "cancel_event" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            options["cancel_event"] = self._stop
        report = analyzer(manifest, **options)
        self._check()
        if not isinstance(report, dict):
            raise ValueError("分析器未返回有效报告")
        return report

    def _progress(self, message: Any) -> None:
        self._check()
        self._emit("progress", message=str(message))

    def _postpone_missed_capture(self) -> None:
        """Give an overdue result a full display interval; skip missed slots."""
        with self._condition:
            now = self._clock()
            if not self._stop.is_set() and self._next_due is not None and now >= self._next_due:
                self._next_due = now + self.interval_seconds

    def _run(self) -> None:
        stage = "configuration"
        try:
            self._state("OPENING")
            self._prepare()
            stage = "camera"
            self._reader = threading.Thread(target=self._read_camera, name="monitor-camera-reader", daemon=True)
            self._reader.start()
            with self._condition:
                while self._opened_at is None:
                    self._check()
                    self._condition.wait(timeout=.05)
                self._next_due = self._opened_at + self.warmup_seconds
            self._state("WARMING_UP")
            self._wait_until(self.next_due)
            while not self._stop.is_set():
                self._cycle_id += 1
                started = self._clock()
                with self._condition:
                    self._next_due = started + self.interval_seconds
                manifest = None
                try:
                    stage = "capture"
                    self._state("CAPTURING")
                    pair = self._capture(started)
                    stage = "dataset"
                    manifest = self._save_capture(pair)
                    self._check()
                    details = {"manifest_path": str(manifest), "report_path": str(manifest.parent / "report.json"),
                               "captured_at": pair.left_captured_at, "capture_started": started}
                    self._emit("capture", **details)
                    stage = "analysis"
                    self._state("ANALYZING")
                    report = self._analyze(manifest)
                    self._check()
                    self._postpone_missed_capture()
                    self._emit("result", report=report,
                               finished_at=datetime.now(timezone.utc).isoformat(),
                               next_due=self.next_due, **details)
                except (_Stopped, RemoteAnalysisCancelled):
                    raise
                except Exception as error:
                    self._postpone_missed_capture()
                    self._emit("error", stage=stage, message=f"{type(error).__name__}: {error}",
                               manifest_path=str(manifest) if manifest else None)
                    if self._reader_error is not None:
                        break
                self._check()
                self._state("WAITING")
                self._wait_until(self.next_due)
        except (_Stopped, RemoteAnalysisCancelled):
            pass
        except Exception as error:
            self._emit("error", stage=stage, message=f"{type(error).__name__}: {error}")
        finally:
            self._stop.set()
            with self._condition:
                self._next_due = None
                self._condition.notify_all()
            # Never declare idle while a camera driver still owns the device.
            if self._reader is not None:
                self._reader.join()
            with self._condition:
                self._active = False
                self._emit("stopped")
                self._condition.notify_all()


__all__ = ["SingleFrameMonitor"]
