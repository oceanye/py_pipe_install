"""OpenCV color-plus-projected-diameter detector for the M0 fixture.

This module is deliberately labelled as a constrained baseline.  It does not
turn an uncalibrated screen measurement into a physical millimetre claim, and
it never infers installation state from visibility alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .photo_capture import analysis_config, validate_still_capture_manifest


class UnsupportedCapabilityError(ValueError):
    """Raised when a manifest asks this M0 implementation to claim more capability."""


def _load_manifest(manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


def validate_m0_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed when a sidecar exceeds the implemented M0 capability."""

    scope = manifest.get("scope")
    if not isinstance(scope, Mapping):
        raise UnsupportedCapabilityError("Manifest scope must be an object")
    if scope.get("layer_model") != "single":
        raise UnsupportedCapabilityError("M0 supports only layer_model='single'")
    if scope.get("metric_calibrated") is not False:
        raise UnsupportedCapabilityError("M0 is uncalibrated and requires metric_calibrated=false")
    for key in ("supports_stereo", "supports_occlusion_reasoning", "supports_3dgs_training"):
        if scope.get(key) is not False:
            raise UnsupportedCapabilityError(f"M0 requires {key}=false")
    required_features = {"appearance_color", "projected_diameter_bucket"}
    features = scope.get("identity_features")
    if not isinstance(features, list) or set(features) != required_features:
        raise UnsupportedCapabilityError(
            "M0 identity_features must be appearance_color and projected_diameter_bucket"
        )

    model = manifest.get("model")
    if not isinstance(model, Mapping) or model.get("unit") != "millimeter":
        raise UnsupportedCapabilityError("M0 requires a millimeter model")
    pipes = model.get("pipes")
    if not isinstance(pipes, list) or not pipes:
        raise UnsupportedCapabilityError("M0 requires at least one pipe definition")
    layer_id = scope.get("layer_id")
    if not isinstance(layer_id, str) or not layer_id:
        raise UnsupportedCapabilityError("M0 requires one explicit scope.layer_id")
    if any(not isinstance(pipe, Mapping) or pipe.get("layer_id") != layer_id for pipe in pipes):
        raise UnsupportedCapabilityError("Every M0 pipe must use the single declared layer_id")

    has_video_key = "video" in manifest
    has_capture_key = "capture" in manifest
    if has_video_key == has_capture_key:
        raise UnsupportedCapabilityError(
            "M0 requires exactly one input source: legacy video or still capture"
        )
    source_key = "capture" if has_capture_key else "video"
    if not isinstance(manifest.get(source_key), Mapping):
        raise UnsupportedCapabilityError(f"Manifest {source_key} must be an object")
    if has_capture_key:
        try:
            validate_still_capture_manifest(manifest)
        except ValueError as error:
            raise UnsupportedCapabilityError(str(error)) from error


def _opencv_lab_to_cie(value: np.ndarray) -> np.ndarray:
    """Convert OpenCV's uint8 Lab encoding to conventional CIE L*a*b*."""

    encoded = value.astype(float)
    return np.asarray([encoded[0] * 100.0 / 255.0, encoded[1] - 128.0, encoded[2] - 128.0])


def _hex_to_lab(color: str) -> np.ndarray:
    value = color.removeprefix("#")[:6]
    if len(value) != 6:
        raise ValueError(f"Invalid sRGB color: {color!r}")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    pixel = np.asarray([[[blue, green, red]]], dtype=np.uint8)
    encoded = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    return _opencv_lab_to_cie(encoded)


def _mask_for_rules(hsv: np.ndarray, rules: list[list[int]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rule in rules:
        if len(rule) != 6:
            raise ValueError(f"HSV rule must contain six integers, got {rule!r}")
        lower = np.asarray([rule[0], rule[2], rule[4]], dtype=np.uint8)
        upper = np.asarray([rule[1], rule[3], rule[5]], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)


def _robust_component_geometry(
    labels: np.ndarray,
    label: int,
    stats: np.ndarray,
    roi_offset: tuple[int, int],
) -> dict[str, Any]:
    """Measure a long component in its PCA frame using central cross-sections."""

    rows, columns = np.nonzero(labels == label)
    points = np.column_stack((columns, rows)).astype(np.float64)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[0] < 0:
        axis = -axis
    normal = np.asarray([-axis[1], axis[0]])

    axial = centered @ axis
    normal_distance = centered @ normal
    axial_low, axial_high = np.percentile(axial, (0.5, 99.5))
    long_side = float(axial_high - axial_low + 1.0)
    central = (axial >= axial_low + 0.15 * (axial_high - axial_low)) & (
        axial <= axial_high - 0.15 * (axial_high - axial_low)
    )

    axial_bins = np.floor(axial[central] - axial_low).astype(np.int32)
    normal_values = normal_distance[central]
    bin_count = max(int(np.ceil(long_side)) + 2, 3)
    minimum = np.full(bin_count, np.inf)
    maximum = np.full(bin_count, -np.inf)
    counts = np.zeros(bin_count, dtype=np.int32)
    valid_bins = (axial_bins >= 0) & (axial_bins < bin_count)
    axial_bins = axial_bins[valid_bins]
    normal_values = normal_values[valid_bins]
    np.minimum.at(minimum, axial_bins, normal_values)
    np.maximum.at(maximum, axial_bins, normal_values)
    np.add.at(counts, axial_bins, 1)
    widths = maximum[counts >= 2] - minimum[counts >= 2] + 1.0
    if len(widths) == 0:
        widths = np.asarray([float(stats[label, cv2.CC_STAT_HEIGHT])])
    diameter_px = float(np.median(widths))
    width_mad_px = float(np.median(np.abs(widths - diameter_px)))
    aspect_ratio = long_side / max(diameter_px, 1e-9)

    x = int(stats[label, cv2.CC_STAT_LEFT]) + roi_offset[0]
    y = int(stats[label, cv2.CC_STAT_TOP]) + roi_offset[1]
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    axis_start = mean + axis * axial_low + np.asarray(roi_offset)
    axis_end = mean + axis * axial_high + np.asarray(roi_offset)
    area = int(stats[label, cv2.CC_STAT_AREA])

    return {
        "component_label": label,
        "area_px": area,
        "bbox_xywh": [x, y, width, height],
        "centerline_px": [axis_start.tolist(), axis_end.tolist()],
        "orientation_degrees": float(np.degrees(np.arctan2(axis[1], axis[0]))),
        "long_side_px": long_side,
        "diameter_px": diameter_px,
        "diameter_mad_px": width_mad_px,
        "aspect_ratio": float(aspect_ratio),
        "fill_ratio": float(min(1.0, area / max(long_side * diameter_px, 1.0))),
        "_rows": rows,
        "_columns": columns,
    }


class ColorDiameterDetector:
    """Detect the current single-layer fixture using color and size consistency."""

    def __init__(self, manifest: Mapping[str, Any] | str | Path):
        self.manifest = _load_manifest(manifest)
        validate_m0_manifest(self.manifest)
        self.analysis_config = analysis_config(self.manifest)
        self.scope = self.manifest.get("scope", {})
        self.pipes = list(self.manifest.get("model", {}).get("pipes", []))
        if not self.pipes:
            raise ValueError("Manifest contains no model pipes")
        self.target_labs = {
            pipe["pipe_id"]: _hex_to_lab(pipe["nominal_color_srgb"]) for pipe in self.pipes
        }

    def _roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        height, width = frame.shape[:2]
        configured = self.analysis_config.get("analysis_roi_xyxy", [0, 0, width, height])
        x1, y1, x2, y2 = (int(value) for value in configured)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Configured analysis ROI is empty after clipping to the frame")
        return frame[y1:y2, x1:x2], (x1, y1)

    def _default_observation(self, pipe: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "pipe_id": pipe["pipe_id"],
            "layer_id": pipe.get("layer_id", self.scope.get("layer_id", "L0")),
            "appearance_color": pipe["appearance_color"],
            "visibility": "NOT_OBSERVED",
            "visible_state": "NOT_OBSERVED",
            "installation_state": "UNKNOWN",
            "decision": "UNKNOWN",
            "reason": "no_qualified_component",
            "nominal_diameter_mm": float(pipe["nominal_diameter_mm"]),
            "diameter_source": "model_prior",
            "metric_calibrated": False,
            "projected_diameter_estimate_mm": None,
            "projected_diameter_method": "uncalibrated_aspect_ratio",
            "scale_source": "nominal_length_prior",
            "diameter_px": None,
            "candidates": [],
        }

    def detect(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Return per-pipe visibility observations for one BGR frame."""

        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 NumPy array")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must use uint8 pixels")

        roi, offset = self._roi(frame_bgr)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        observations = {pipe["pipe_id"]: self._default_observation(pipe) for pipe in self.pipes}
        unmatched_components: list[dict[str, Any]] = []

        rules_by_color = self.analysis_config.get("color_rules", {})
        minimum_area = int(self.analysis_config.get("minimum_component_area_px", 100))
        minimum_long_side = float(self.analysis_config.get("minimum_long_side_px", 20.0))
        minimum_aspect = float(self.analysis_config.get("minimum_aspect_ratio", 3.0))
        delta_tolerance = float(self.analysis_config.get("delta_e76_tolerance", 35.0))
        absolute_tolerance = float(
            self.analysis_config.get("diameter_absolute_tolerance_mm", 5.0)
        )
        relative_tolerance = float(
            self.analysis_config.get("diameter_relative_tolerance", 0.2)
        )

        for appearance_color, rules in rules_by_color.items():
            mask = _mask_for_rules(hsv, rules)
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            components: list[dict[str, Any]] = []
            for label in range(1, component_count):
                if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area:
                    continue
                component = _robust_component_geometry(labels, label, stats, offset)
                if component["long_side_px"] < minimum_long_side:
                    continue
                if component["aspect_ratio"] < minimum_aspect:
                    continue

                rows = component.pop("_rows")
                columns = component.pop("_columns")
                mean_bgr = roi[rows, columns].mean(axis=0)
                mean_lab = cv2.cvtColor(
                    np.clip(np.rint(mean_bgr), 0, 255).astype(np.uint8).reshape(1, 1, 3),
                    cv2.COLOR_BGR2LAB,
                )[0, 0].astype(float)
                component["mean_bgr"] = mean_bgr.tolist()
                component["mean_lab_opencv"] = mean_lab.tolist()
                component["mean_lab_cie"] = _opencv_lab_to_cie(mean_lab).tolist()
                components.append(component)

            eligible_pipes = [
                pipe for pipe in self.pipes if pipe.get("appearance_color") == appearance_color
            ]
            pairings: list[tuple[float, dict[str, Any], Mapping[str, Any], dict[str, Any]]] = []
            for component in components:
                for pipe in eligible_pipes:
                    nominal_length = float(pipe["nominal_length_mm"])
                    estimate = nominal_length * component["diameter_px"] / component["long_side_px"]
                    nominal_diameter = float(pipe["nominal_diameter_mm"])
                    residual = abs(estimate - nominal_diameter)
                    diameter_tolerance = max(absolute_tolerance, relative_tolerance * nominal_diameter)
                    color_delta = float(
                        np.linalg.norm(
                            np.asarray(component["mean_lab_cie"]) - self.target_labs[pipe["pipe_id"]]
                        )
                    )
                    score = max(
                        0.0,
                        1.0
                        - 0.5 * color_delta / max(delta_tolerance, 1e-9)
                        - 0.5 * residual / max(diameter_tolerance, 1e-9),
                    )
                    candidate = {
                        "pipe_id": pipe["pipe_id"],
                        "delta_e76": color_delta,
                        "projected_diameter_estimate_mm": estimate,
                        "diameter_residual_mm": residual,
                        "diameter_tolerance_mm": diameter_tolerance,
                        "score": score,
                        "passes_color_gate": color_delta <= delta_tolerance,
                        "passes_diameter_gate": residual <= diameter_tolerance,
                    }
                    pairings.append((score, component, pipe, candidate))

            used_components: set[int] = set()
            used_pipes: set[str] = set()
            for _, component, pipe, candidate in sorted(pairings, key=lambda item: item[0], reverse=True):
                component_key = id(component)
                pipe_id = str(pipe["pipe_id"])
                observations[pipe_id]["candidates"].append(candidate)
                if component_key in used_components or pipe_id in used_pipes:
                    continue
                if not candidate["passes_color_gate"] or not candidate["passes_diameter_gate"]:
                    continue

                used_components.add(component_key)
                used_pipes.add(pipe_id)
                observations[pipe_id].update(
                    {
                        "visibility": "VISIBLE",
                        "visible_state": "VISIBLE",
                        "installation_state": "UNKNOWN",
                        "decision": "MATCHED",
                        "reason": "color_and_projected_diameter_match",
                        "projected_diameter_estimate_mm": candidate[
                            "projected_diameter_estimate_mm"
                        ],
                        "diameter_px": component["diameter_px"],
                        "diameter_mad_px": component["diameter_mad_px"],
                        "bbox_xywh": component["bbox_xywh"],
                        "centerline_px": component["centerline_px"],
                        "long_side_px": component["long_side_px"],
                        "aspect_ratio": component["aspect_ratio"],
                        "orientation_degrees": component["orientation_degrees"],
                        "mean_bgr": component["mean_bgr"],
                        "mean_lab_opencv": component["mean_lab_opencv"],
                        "mean_lab_cie": component["mean_lab_cie"],
                        "delta_e76": candidate["delta_e76"],
                        "match_score": candidate["score"],
                    }
                )

            for component in components:
                if id(component) not in used_components:
                    unmatched_components.append(
                        {
                            "appearance_color": appearance_color,
                            **component,
                            "reason": "no_unique_color_diameter_match",
                        }
                    )

        visible_pipe_ids = [
            pipe["pipe_id"]
            for pipe in self.pipes
            if observations[pipe["pipe_id"]]["visibility"] == "VISIBLE"
        ]
        return {
            "capability_level": "M0_COLOR_DIAMETER_SINGLE_LAYER",
            "layer_model": self.scope.get("layer_model", "single"),
            "metric_calibrated": False,
            "visible_pipe_ids": visible_pipe_ids,
            "installation_state": "UNKNOWN",
            "observations": observations,
            "unmatched_components": unmatched_components,
        }
