"""Portable appearance hints for qualitative installation assistance only."""
from collections.abc import Mapping
import math
import re

SUPPORTED_COLOR_CLASSES = ('RED', 'BLUE', 'WHITE', 'GREEN', 'YELLOW', 'UNKNOWN')
ASSIST_TOLERANCE_DEFAULTS = {
    'measured_diameter_tolerance_ratio': .10,
    'relative_distance_tolerance_ratio': .10,
}


def assist_tolerances(payload=None):
    """Recognition tolerances, independent of metric acceptance thresholds."""
    options = payload or {}
    result = {}
    for key, default in ASSIST_TOLERANCE_DEFAULTS.items():
        value = options.get(key, default)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= .5:
            raise ValueError('辅助识别比例容差必须大于 0 且不超过 0.5')
        result[key] = float(value)
    return result


def normalize_installation_assist_settings(payload=None):
    if payload is None:
        return {}
    if not isinstance(payload, Mapping) or set(payload)-{'appearance_diameter_hints', 'hint_source', *ASSIST_TOLERANCE_DEFAULTS}:
        raise ValueError('辅助识别配置包含未知字段')
    values = payload.get('appearance_diameter_hints', {})
    if not isinstance(values, Mapping) or len(values) > 32:
        raise ValueError('颜色与采购管径关系必须为映射')
    hints = {}
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch('[A-Za-z_]{1,32}', key):
            raise ValueError('辅助识别颜色名称无效')
        key = key.upper()
        if key not in SUPPORTED_COLOR_CLASSES:
            raise ValueError('辅助识别暂不支持此颜色名称')
        if key in hints:
            raise ValueError('辅助识别颜色名称重复')
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('采购管径提示必须是有限正数；它不是实测管径')
        hints[key] = float(value)
    source = payload.get('hint_source', 'USER_CONFIGURED')
    if not isinstance(source, str) or not source.strip() or len(source) > 256:
        raise ValueError('辅助识别提示来源无效')
    result = {'appearance_diameter_hints': hints, 'hint_source': source} if hints else {}
    tolerances = assist_tolerances(payload)
    result.update({key: value for key, value in tolerances.items() if key in payload})
    return result
