"""Small, dependency-free ASCII DXF side-elevation reader.

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
from typing import Iterable


class DxfError(ValueError):
    pass


@dataclass(frozen=True)
class DxfEntity:
    kind: str
    layer: str
    color: str
    points: tuple[tuple[float, float], ...] = ()
    radius: float | None = None
    start_angle: float = 0.0
    end_angle: float = 360.0


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
        color = layers.get(layer, "#FFFFFF")
        values = {c: v for c, v in record}
        if kind == "LINE":
            entities.append(DxfEntity(kind="LINE", layer=layer, color=color, points=((_float(values, 10), _float(values, 20)), (_float(values, 11), _float(values, 21)))))
        elif kind in {"CIRCLE", "ARC"}:
            radius = _float(values, 40)
            if radius <= 0:
                continue
            entities.append(DxfEntity(kind=kind, layer=layer, color=color, points=((_float(values, 10), _float(values, 20)),), radius=radius, start_angle=_float(values, 50), end_angle=_float(values, 51, 360.0) if kind == "ARC" else 360.0))
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
        entities.append(DxfEntity(kind=kind, layer=layer, color=layers.get(layer, "#FFFFFF"), points=points, radius=radius, start_angle=float(getattr(entity.dxf, "start_angle", 0.0)), end_angle=float(getattr(entity.dxf, "end_angle", 360.0))))
    if not entities:
        raise DxfError("DXF contains no supported 2-D side-elevation entities")
    unit_code = int(getattr(document.header, "__getitem__", lambda key: 0)("$INSUNITS") or 0)
    source_unit, unit_scale = {1: ("inch", 25.4), 2: ("foot", 304.8), 4: ("millimeter", 1.0), 5: ("centimeter", 10.0), 6: ("meter", 1000.0)}.get(unit_code, ("unitless", 1.0))
    if unit_scale != 1.0:
        entities = [_scale_entity(entity, unit_scale) for entity in entities]
    return DxfElevation(source, hashlib.sha256(raw).hexdigest(), source_unit, unit_scale, layers, tuple(entities))


def _scale_entity(entity: DxfEntity, scale: float) -> DxfEntity:
    return DxfEntity(entity.kind, entity.layer, entity.color, tuple((x * scale, y * scale) for x, y in entity.points), None if entity.radius is None else entity.radius * scale, entity.start_angle, entity.end_angle)


def arc_points(entity: DxfEntity, segments: int = 48) -> tuple[tuple[float, float], ...]:
    if entity.radius is None or not entity.points:
        return entity.points
    start, end = entity.start_angle, entity.end_angle
    while end < start:
        end += 360.0
    count = max(2, min(256, round(abs(end - start) / 360.0 * segments)))
    cx, cy = entity.points[0]
    return tuple((cx + entity.radius * math.cos(math.radians(start + (end - start) * i / (count - 1))), cy + entity.radius * math.sin(math.radians(start + (end - start) * i / (count - 1)))) for i in range(count))


__all__ = ["DxfError", "DxfEntity", "DxfElevation", "arc_points", "read_dxf_elevation"]
