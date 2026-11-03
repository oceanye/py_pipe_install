"""Changes in displayed auxiliary pipe hints, never installation acceptance."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .assist_display import assist_data, hypotheses, hypothesis_assignments


def hinted_pipe_ids(report, allowed_ids):
    data = assist_data(report)
    choices = hypotheses(report)
    if data.get('status') not in ('LIKELY', 'AMBIGUOUS') or not choices or choices[0].get('near_best') is False:
        return None
    observations = {row['observation_id'] for row in data.get('observations', [])
                    if row.get('status') in ('LIKELY', 'AMBIGUOUS')}
    current = {pipe for observation, pipe in hypothesis_assignments(report).items()
               if observation in observations and pipe in allowed_ids}
    # No reliable hint is not proof that all previously hinted pipes vanished.
    return current or None


class MonitorHistory:
    def __init__(self, pipe_specs):
        self.session_id = uuid4().hex
        self.allowed_ids = {pipe['pipe_id'] for pipe in pipe_specs}
        self.previous = None
        self.last_cycle = -1

    def record(self, cycle_id, *, report=None, captured_at=None, evaluated_at=None,
               manifest_path=None, error=None):
        if cycle_id <= self.last_cycle:
            return None
        current = None if error is not None else hinted_pipe_ids(report, self.allowed_ids)
        added, removed = set(), set()
        if error is not None:
            kind = 'ERROR'
        elif current is None:
            kind = 'UNRESOLVED'
        elif self.previous is None:
            kind = 'BASELINE'
        else:
            added, removed = current - self.previous, self.previous - current
            kind = 'CHANGED' if added or removed else 'UNCHANGED'
        if current is not None:
            self.previous = current
        self.last_cycle = cycle_id
        return {'schema_version': 'pipe-monitor-history-v1', 'session_id': self.session_id,
                'cycle_id': cycle_id, 'kind': kind, 'advisory_only': True,
                'captured_at': captured_at, 'evaluated_at': evaluated_at,
                'recorded_at': datetime.now(timezone.utc).isoformat(),
                'manifest_path': str(manifest_path) if manifest_path else None,
                'model_pipe_ids': sorted(self.allowed_ids),
                'pipe_ids': sorted(current) if current is not None else None,
                'added': sorted(added), 'removed': sorted(removed),
                'error': str(error)[:500] if error is not None else None}


def history_rows(entry):
    """One readable table row per changed pipe, retaining every cycle in JSONL."""
    if entry['kind'] == 'BASELINE':
        return [('基线', pipe) for pipe in entry['pipe_ids']]
    if entry['kind'] == 'CHANGED':
        return [('新增提示', pipe) for pipe in entry['added']] + [('减少提示', pipe) for pipe in entry['removed']]
    return [({'UNCHANGED': '无变化', 'UNRESOLVED': '证据不足', 'ERROR': '本轮失败'}[entry['kind']], '—')]
