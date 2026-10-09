"""Small ASCII DXF side-elevation reader and pipe-layout catalogue builder.

The reader intentionally handles the 2-D entities useful for a pipe elevation
(LINE, CIRCLE, ARC and lightweight polylines).  Layer ACI colours are exposed
as sRGB values and are used as the default display colour in the GUI.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


class DxfError(ValueError):
    pass


@dataclass(frozen=True)
class DxfEntity:
    kind: str
    layer: str
    color: str
    entity_id: str = ""
    points: tuple[tuple[float, float], ...] = ()
    radius: float | None = None
    start_angle: float = 0.0
    end_angle: float = 360.0
    elevation: float = 0.0
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class DxfElevation:
    source_path: Path
    source_sha256: str
    source_unit: str
    unit_scale_to_mm: float
    layers: dict[str, str]
    entities: tuple[DxfEntity, ...]


_ACI = {
    1: "#FF0000", 2: "#FFFF00", 3: "#00FF00", 4: "#00FFFF",
    5: "#0000FF", 6: "#FF00FF", 7: "#FFFFFF", 8: "#808080", 9: "#C0C0C0",
}

_UNIT_SCALES = {"millimeter": 1.0, "centimeter": 10.0, "meter": 1000.0, "inch": 25.4, "foot": 304.8}
DXF_PREVIEW_HALF_LENGTH_MM = 1000.0


def _aci_color(index: int) -> str:
    index = abs(int(index))
    if index == 0:
        return "#FFFFFF"
    if index in _ACI:
        return _ACI[index]
    # AutoCAD's extended palette is deterministic; this gives a useful colour
    # for uncommon indices without pretending to reproduce every display theme.
    hue = (index % 256) / 256.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return f"#{round(red*255):02X}{round(green*255):02X}{round(blue*255):02X}"


def _true_color(value: object) -> str | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= number <= 0xFFFFFF:
        return f"#{(number >> 16) & 255:02X}{(number >> 8) & 255:02X}{number & 255:02X}"
    return None


def _entity_color(record: list[tuple[int, str]], layer_color: str) -> str:
    values = {code: value for code, value in record}
    true = _true_color(values.get(420))
    if true:
        return true
    try:
        if 62 in values and int(values[62]) not in (0, 256):
            return _aci_color(int(values[62]))
    except ValueError:
        pass
    return layer_color


def _pairs(raw: bytes) -> list[tuple[int, str]]:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    if len(lines) % 2:
        lines = lines[:-1]
    result: list[tuple[int, str]] = []
    for index in range(0, len(lines), 2):
        try:
            code = int(lines[index])
        except ValueError:
            continue
        result.append((code, lines[index + 1]))
    return result


def _float(values: dict[int, str], code: int, default: float = 0.0) -> float:
    try:
        return float(values.get(code, default))
    except (TypeError, ValueError):
        return default


def _units(pairs: list[tuple[int, str]]) -> tuple[str, float]:
    values: dict[int, str] = {}
    for index, (code, value) in enumerate(pairs):
        if code == 9 and value.upper() == "$INSUNITS":
            values = {item_code: item_value for item_code, item_value in pairs[index + 1 : index + 5]}
            break
    try:
        code = int(values.get(70, "0"))
    except ValueError:
        code = 0
    table = {1: ("inch", 25.4), 2: ("foot", 304.8), 4: ("millimeter", 1.0), 5: ("centimeter", 10.0), 6: ("meter", 1000.0)}
    return table.get(code, ("unitless", 1.0))


def read_dxf_elevation(path: str | Path) -> DxfElevation:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    if len(raw) > 128 * 1024 * 1024:
        raise DxfError("DXF file is larger than 128 MiB")
    # Prefer ezdxf for binary files (and as a fallback for richer ASCII files),
    # while retaining the dependency-free reader for minimal fixtures.
    if b"\x00" in raw:
        try:
            return _read_with_ezdxf(source, raw)
        except Exception as error:
            raise DxfError(f"Unable to read binary DXF: {source}") from error
    pairs = _pairs(raw)
    source_unit, unit_scale = _units(pairs)
    layers: dict[str, str] = {"0": "#FFFFFF"}
    # TABLES/LAYER records carry the default colour for entities on that layer.
    for i, (code, value) in enumerate(pairs):
        if code == 0 and value.upper() == "LAYER":
            record: dict[int, str] = {}
            for rcode, rvalue in pairs[i + 1 :]:
                if rcode == 0:
                    break
                record[rcode] = rvalue
            name = record.get(2, "0")
            try:
                true_color = int(record.get(420, ""))
            except (TypeError, ValueError):
                true_color = None
            if true_color is not None:
                layers[name] = f"#{(true_color >> 16) & 255:02X}{(true_color >> 8) & 255:02X}{true_color & 255:02X}"
                continue
            try:
                aci = int(record.get(62, "7"))
            except ValueError:
                aci = 7
            layers[name] = _aci_color(aci)
    entities: list[DxfEntity] = []
    for i, (code, value) in enumerate(pairs):
        if code != 0 or value.upper() not in {"LINE", "CIRCLE", "ARC", "LWPOLYLINE"}:
            continue
        kind = value.upper()
        record: list[tuple[int, str]] = []
        for rcode, rvalue in pairs[i + 1 :]:
            if rcode == 0:
                break
            record.append((rcode, rvalue))
        layer = next((v for c, v in record if c == 8), "0")
        color = _entity_color(record, layers.get(layer, "#FFFFFF"))
        values = {c: v for c, v in record}
        if kind == "LINE":
            entities.append(DxfEntity(kind="LINE", layer=layer, color=color, points=((_float(values, 10), _float(values, 20)), (_float(values, 11), _float(values, 21)))))
        elif kind in {"CIRCLE", "ARC"}:
            radius = _float(values, 40)
            if radius <= 0:
                continue
            entities.append(DxfEntity(kind=kind, layer=layer, color=color, points=((_float(values, 10), _float(values, 20)),), radius=radius, start_angle=_float(values, 50), end_angle=_float(values, 51, 360.0) if kind == "ARC" else 360.0,
                                      elevation=_float(values, 30), normal=(_float(values, 210), _float(values, 220), _float(values, 230, 1.0))))
        else:
            points = []
            current_x: float | None = None
            for rcode, rvalue in record:
                if rcode == 10:
                    if current_x is not None:
                        points.append((current_x, current_y))
                    current_x = float(rvalue)
                    current_y = 0.0
                elif rcode == 20 and current_x is not None:
                    current_y = float(rvalue)
            if current_x is not None:
                points.append((current_x, current_y))
            if len(points) >= 2:
                entities.append(DxfEntity(kind="LWPOLYLINE", layer=layer, color=color, points=tuple(points)))
    if not entities:
        raise DxfError("DXF contains no supported 2-D side-elevation entities")
    if unit_scale != 1.0:
        entities = [_scale_entity(entity, unit_scale) for entity in entities]
    entities = [_number_entity(entity, index) for index, entity in enumerate(entities, 1)]
    return DxfElevation(source, hashlib.sha256(raw).hexdigest(), source_unit, unit_scale, layers, tuple(entities))


def _read_with_ezdxf(source: Path, raw: bytes) -> DxfElevation:
    """Read a DXF through ezdxf when available (binary DXF or rich entities)."""
    try:
        import ezdxf  # type: ignore
    except ImportError as error:
        raise DxfError("binary DXF import requires the 'ezdxf' package") from error
    document = ezdxf.readfile(str(source))
    layers: dict[str, str] = {}
    for item in document.layers:
        name = str(item.dxf.name)
        true_color = getattr(item.dxf, "true_color", None)
        if true_color is not None:
            value = int(true_color)
            layers[name] = f"#{(value >> 16) & 255:02X}{(value >> 8) & 255:02X}{value & 255:02X}"
        else:
            layers[name] = _aci_color(int(getattr(item.dxf, "color", 7) or 7))
    entities: list[DxfEntity] = []
    for entity in document.modelspace():
        kind = entity.dxftype().upper()
        if kind == "LINE":
            points = ((float(entity.dxf.start.x), float(entity.dxf.start.y)), (float(entity.dxf.end.x), float(entity.dxf.end.y)))
            radius = None
        elif kind in {"CIRCLE", "ARC"}:
            center = entity.dxf.center
            points = ((float(center.x), float(center.y)),)
            radius = float(entity.dxf.radius)
        elif kind == "LWPOLYLINE":
            points = tuple((float(point[0]), float(point[1])) for point in entity.get_points("xy"))
            radius = None
        else:
            continue
        if len(points) < 2 and radius is None:
            continue
        layer = str(getattr(entity.dxf, "layer", "0"))
        true_color = _true_color(getattr(entity.dxf, "true_color", None))
        try:
            aci = int(getattr(entity.dxf, "color", 256) or 256)
        except (TypeError, ValueError):
            aci = 256
        entity_color = true_color or (_aci_color(aci) if aci not in (0, 256) else layers.get(layer, "#FFFFFF"))
        entities.append(DxfEntity(kind=kind, layer=layer, color=entity_color, points=points, radius=radius, start_angle=float(getattr(entity.dxf, "start_angle", 0.0)), end_angle=float(getattr(entity.dxf, "end_angle", 360.0)),
                                  elevation=float(entity.dxf.center.z) if kind in {"CIRCLE", "ARC"} else 0.0,
                                  normal=tuple(entity.dxf.extrusion) if kind in {"CIRCLE", "ARC"} else (0.0, 0.0, 1.0)))
    if not entities:
        raise DxfError("DXF contains no supported 2-D side-elevation entities")
    unit_code = int(getattr(document.header, "__getitem__", lambda key: 0)("$INSUNITS") or 0)
    source_unit, unit_scale = {1: ("inch", 25.4), 2: ("foot", 304.8), 4: ("millimeter", 1.0), 5: ("centimeter", 10.0), 6: ("meter", 1000.0)}.get(unit_code, ("unitless", 1.0))
    if unit_scale != 1.0:
        entities = [_scale_entity(entity, unit_scale) for entity in entities]
    entities = [_number_entity(entity, index) for index, entity in enumerate(entities, 1)]
    return DxfElevation(source, hashlib.sha256(raw).hexdigest(), source_unit, unit_scale, layers, tuple(entities))


def _scale_entity(entity: DxfEntity, scale: float) -> DxfEntity:
    return DxfEntity(entity.kind, entity.layer, entity.color, entity.entity_id, tuple((x * scale, y * scale) for x, y in entity.points), None if entity.radius is None else entity.radius * scale, entity.start_angle, entity.end_angle, entity.elevation * scale, entity.normal)


def _number_entity(entity: DxfEntity, index: int) -> DxfEntity:
    return DxfEntity(entity.kind, entity.layer, entity.color, entity.entity_id or f"P{index:03d}", entity.points, entity.radius, entity.start_angle, entity.end_angle, entity.elevation, entity.normal)


def arc_points(entity: DxfEntity, segments: int = 48) -> tuple[tuple[float, float], ...]:
    if entity.radius is None or not entity.points:
        return entity.points
    start, end = entity.start_angle, entity.end_angle
    while end < start:
        end += 360.0
    count = max(2, min(256, round(abs(end - start) / 360.0 * segments)))
    cx, cy = entity.points[0]
    return tuple((cx + entity.radius * math.cos(math.radians(start + (end - start) * i / (count - 1))), cy + entity.radius * math.sin(math.radians(start + (end - start) * i / (count - 1)))) for i in range(count))


def _unit_axis(value: Sequence[float] | None) -> np.ndarray:
    """Validate the common pipe axis used to lift a 2-D DXF layout into 3-D."""
    axis = np.asarray((0.0, 0.0, 1.0) if value is None else value, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise DxfError("DXF 管长方向必须是有限的三维向量")
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 1.0e-9:
        raise DxfError("DXF 管长方向不能为零向量")
    axis /= length
    if not np.allclose(np.abs(axis), [0.0, 0.0, 1.0], atol=1e-9, rtol=0):
        raise DxfError("DXF 圆形截面的管轴固定垂直于 XY 图面（±Z）；俯视角度由双目配准估计，无需改成 X/Y")
    return axis


def dxf_source_unit(document: DxfElevation, unitless_unit: str | None = None) -> str:
    """Use the drawing's declared units; require an explicit fallback otherwise."""
    unit = document.source_unit if document.source_unit != "unitless" else unitless_unit
    if unit not in _UNIT_SCALES:
        raise DxfError("DXF 未声明有效单位，请在基础立面评估中选择原始单位后导入")
    return str(unit)


def catalog_from_dxf(path: str | Path, *, axis_world: Sequence[float] | None = None,
                     half_length_mm: float = DXF_PREVIEW_HALF_LENGTH_MM,
                     unitless_unit: str | None = None,
                     diameter_colors: Mapping[str, str] | None = None) -> tuple[list[dict], list[DxfEntity]]:
    """Create an automatic-elevation pipe catalogue from DXF pipe circles.

    A side-elevation DXF contains cross-section/layout information, rather than
    a full 3-D solid.  Each full ``CIRCLE`` therefore becomes a finite display
    centreline centred at its DXF XY position and aligned with ``axis_world``.
    The depth matcher evaluates an *infinite* cylinder, so ``half_length_mm``
    only controls the catalogue preview and the registration gauge.  Non-circle
    entities (for example the rectangular drawing border) are returned as
    ``skipped`` and are never treated as pipes.
    """
    document = read_dxf_elevation(path)
    axis = _unit_axis(axis_world)
    unit = dxf_source_unit(document, unitless_unit)
    scale = _UNIT_SCALES[unit] if document.source_unit == "unitless" else 1.0
    if type(half_length_mm) not in (int, float) or not math.isfinite(float(half_length_mm)) or float(half_length_mm) <= 0:
        raise DxfError("DXF 管道显示半长必须为正数")
    half = float(half_length_mm)
    overrides = {f"{float(key):.6f}": value for key, value in (diameter_colors or {}).items()}
    pipes: list[dict] = []
    skipped: list[DxfEntity] = []
    for entity in document.entities:
        if entity.kind != "CIRCLE" or entity.radius is None or not entity.points:
            skipped.append(entity)
            continue
        if (not math.isfinite(entity.elevation) or abs(entity.elevation) > 1e-6
                or not np.allclose(entity.normal, [0.0, 0.0, 1.0], atol=1e-9, rtol=0)):
            raise DxfError(f"{entity.entity_id} 不在受支持的 XY 截面中，请先将 DXF 圆形截面展开到 XY 图面")
        cx, cy = (value * scale for value in entity.points[0])
        if not all(math.isfinite(value) for value in (cx, cy, entity.radius)) or entity.radius <= 0:
            raise DxfError(f"{entity.entity_id} 圆心或半径无效")
        center = np.asarray([cx, cy, 0.0], dtype=float)
        line = np.stack((center - axis * half, center + axis * half))
        diameter = float(2.0 * entity.radius * scale)
        key = f"{diameter:.6f}"
        color = str(overrides.get(key, entity.color)).upper()
        if len(color) != 7 or color[0] != "#":
            raise DxfError(f"DXF 管径 {key} 的颜色必须是 #RRGGBB")
        try:
            int(color[1:], 16)
        except ValueError as error:
            raise DxfError(f"DXF 管径 {key} 的颜色必须是 #RRGGBB") from error
        pipes.append({
            "instance_id": len(pipes) + 1,
            "pipe_id": entity.entity_id,
            "cad_object_id": entity.entity_id,
            "layer_id": entity.layer,
            "color_class": entity.layer,
            "color_srgb": color,
            "color_source": "dxf",
            "nominal_diameter_mm": diameter,
            "centerline_world_mm": line.tolist(),
            "source_label": f"DXF {entity.entity_id} {entity.kind}",
        })
    if not pipes:
        raise DxfError("DXF 中没有可作为管道目录的完整 CIRCLE")
    return pipes, skipped


__all__ = ["DXF_PREVIEW_HALF_LENGTH_MM", "DxfError", "DxfEntity", "DxfElevation", "arc_points", "catalog_from_dxf", "dxf_source_unit", "read_dxf_elevation"]
