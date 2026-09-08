from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipe_twin.log_viewer import read_structured_log, structured_log_files


class LogViewerTests(unittest.TestCase):
    def test_rotated_logs_are_read_oldest_first_and_can_be_filtered(self):
        with tempfile.TemporaryDirectory() as temp:
            current = Path(temp) / "pipe_twin.log.jsonl"
            current.with_name(current.name + ".2").write_text(
                json.dumps({"logger": "pipe_twin.calibration_wizard", "event": "old"}) + "\n",
                encoding="utf-8",
            )
            current.with_name(current.name + ".1").write_text(
                json.dumps({"logger": "pipe_twin.gui", "event": "middle"}) + "\n",
                encoding="utf-8",
            )
            current.write_text(
                json.dumps({"logger": "pipe_twin.calibration_wizard.solve", "event": "new"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                [path.name for path in structured_log_files(current)],
                [current.name + ".2", current.name + ".1", current.name],
            )
            payload, count = read_structured_log(
                current, logger_prefix="pipe_twin.calibration_wizard"
            )
            self.assertEqual(count, 2)
            self.assertLess(payload.index('"old"'), payload.index('"new"'))
            self.assertNotIn('"middle"', payload)


if __name__ == "__main__":
    unittest.main()
