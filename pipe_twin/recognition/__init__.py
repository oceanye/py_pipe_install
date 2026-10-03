"""Pluggable local pipe-recognition backends.

The elevation workflow owns capture, stereo depth, model registration, and
state history.  This package owns only the conversion from one rectified
stereo pair to a bounded list of local pipe observations.  Keeping that
boundary small means a new detector can be registered without changing the
manifest writer, GUI, or installation-state logic.

Every backend is a callable with this signature::

    backend(left, right, depth, calibration, pipe_specs) -> surface

``surface`` must be a mapping containing ``observations`` (a list) and may
contain ``audit`` and ``point_cloud``.  The existing cylinder, parallel-strip,
and depth-only implementations are registered below as built-ins.  Projects
can add a backend at startup with :func:`register_recognizer` and select its
name in ``registration.local_observation_mode``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class RecognitionBackendError(ValueError):
    """Raised when a recognizer is unknown or violates the surface contract."""


Recognizer = Callable[[Any, Any, Any, Any, list[dict]], Mapping[str, Any]]


_BACKENDS: dict[str, Recognizer] = {}


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("识别算法名称必须是非空文本")
    normalized = name.strip()
    if any(ch.isspace() for ch in normalized) or len(normalized) > 64:
        raise ValueError("识别算法名称不能包含空白且长度不能超过64")
    return normalized


def register_recognizer(name: str, backend: Recognizer, *, replace: bool = False) -> None:
    """Register a local observation backend.

    Registration is process-local by design.  A backend is not silently
    imported from a manifest, so a field package cannot execute arbitrary
    code merely by naming an algorithm.  ``replace=True`` is useful for
    controlled experiments that replace one of the built-ins.
    """
    normalized = _check_name(name)
    if not callable(backend):
        raise TypeError("识别算法后端必须是可调用对象")
    if normalized in _BACKENDS and not replace:
        raise ValueError(f"识别算法已注册：{normalized}")
    _BACKENDS[normalized] = backend


def unregister_recognizer(name: str) -> None:
    """Remove a process-local backend, primarily for tests and experiments."""
    normalized = _check_name(name)
    if normalized not in _BACKENDS:
        raise KeyError(normalized)
    del _BACKENDS[normalized]


def is_recognizer_registered(name: str) -> bool:
    return isinstance(name, str) and name.strip() in _BACKENDS


def available_recognizers() -> tuple[str, ...]:
    """Return deterministic names accepted by the current process."""
    return tuple(sorted(_BACKENDS))


def _invoke(name: str, left: Any, right: Any, depth: Any, calibration: Any,
            pipe_specs: list[dict]) -> dict[str, Any]:
    backend = _BACKENDS.get(name)
    if backend is None:
        names = "、".join(available_recognizers()) or "无"
        raise RecognitionBackendError(f"未注册的管道识别算法：{name}（当前可用：{names}）")
    result = backend(left, right, depth, calibration, pipe_specs)
    if not isinstance(result, Mapping):
        raise RecognitionBackendError(f"识别算法 {name} 未返回对象")
    observations = result.get("observations")
    if not isinstance(observations, list):
        raise RecognitionBackendError(f"识别算法 {name} 的结果缺少 observations 列表")
    normalized = dict(result)
    audit = normalized.get("audit")
    normalized["audit"] = dict(audit) if isinstance(audit, Mapping) else {}
    normalized["audit"].setdefault("recognizer_backend", name)
    return normalized


def recognize_local_pipes(mode: str, left: Any, right: Any, depth: Any,
                          calibration: Any, pipe_specs: list[dict]) -> dict[str, Any]:
    """Run a registered backend and add a stable audit record.

    ``mode`` is deliberately the same value persisted by the elevation
    manifest.  ``auto`` records both the requested backend and the backend
    actually used after its bounded fallback decision.
    """
    name = _check_name(mode)
    result = _invoke(name, left, right, depth, calibration, pipe_specs)
    audit = result["audit"]
    audit["recognizer_backend"] = name
    audit.setdefault("recognizer_backend_used", name)
    result["audit"] = audit
    return result


def _cylinder(left: Any, right: Any, depth: Any, calibration: Any,
              pipe_specs: list[dict]) -> Mapping[str, Any]:
    from ..local_surface import extract_local_pipes

    return extract_local_pipes(left, right, depth, calibration, pipe_specs)


def _geometry_only(left: Any, right: Any, depth: Any, calibration: Any,
                   pipe_specs: list[dict]) -> Mapping[str, Any]:
    from ..local_surface import extract_local_pipes

    return extract_local_pipes(
        left, right, depth, calibration, pipe_specs,
        config={"geometry_only": True},
    )


def _parallel_strip(left: Any, right: Any, depth: Any, calibration: Any,
                    pipe_specs: list[dict]) -> Mapping[str, Any]:
    from ..parallel_local import extract_parallel_local_pipes

    return extract_parallel_local_pipes(left, right, depth, calibration, pipe_specs)


def _auto(left: Any, right: Any, depth: Any, calibration: Any,
          pipe_specs: list[dict]) -> Mapping[str, Any]:
    cylinder = _invoke("cylinder", left, right, depth, calibration, pipe_specs)
    if cylinder["observations"] and not cylinder.get("audit", {}).get("truncated"):
        cylinder["audit"].setdefault("recognizer_backend_used", "cylinder")
        return cylinder

    # A bounded strip fallback is useful when a foreground board hides the
    # curved section needed by the strict cylinder fitter.  Keep both audits
    # so a future backend can be evaluated against the same evidence.
    strip = _invoke("parallel_strip", left, right, depth, calibration, pipe_specs)
    audit = dict(strip.get("audit", {}))
    audit["fallback_from"] = "CYLINDER_SURFACE"
    audit["cylinder_surface_audit"] = cylinder.get("audit", {})
    audit["recognizer_backend_used"] = "parallel_strip"
    strip["audit"] = audit
    return strip


register_recognizer("auto", _auto)
register_recognizer("cylinder", _cylinder)
register_recognizer("geometry_only", _geometry_only)
register_recognizer("parallel_strip", _parallel_strip)


__all__ = [
    "RecognitionBackendError",
    "Recognizer",
    "available_recognizers",
    "is_recognizer_registered",
    "recognize_local_pipes",
    "register_recognizer",
    "unregister_recognizer",
]
