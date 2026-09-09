"""In-app viewer for the complete local structured log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def structured_log_files(path: str | Path) -> list[Path]:
    """Return rotated logs from oldest to newest, ending with ``path``."""
    current = Path(path)
    backups: list[tuple[int, Path]] = []
    for candidate in current.parent.glob(current.name + ".*"):
        try:
            number = int(candidate.name.removeprefix(current.name + "."))
        except ValueError:
            continue
        if candidate.is_file():
            backups.append((number, candidate))
    return [item[1] for item in sorted(backups, reverse=True)] + (
        [current] if current.is_file() else []
    )


def read_structured_log(
    path: str | Path, *, logger_prefix: str | tuple[str, ...] | None = None
) -> tuple[str, int]:
    """Read complete JSONL logs, optionally retaining one logger family."""
    selected: list[str] = []
    for source in structured_log_files(path):
        for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
            if logger_prefix:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                prefixes = (
                    (logger_prefix,) if isinstance(logger_prefix, str) else logger_prefix
                )
                if not str(record.get("logger", "")).startswith(prefixes):
                    continue
            selected.append(raw)
    text = "\n".join(selected)
    return (text + "\n" if text else "", len(selected))


class LogViewerDialog:
    """Show the current and rotated logs without leaving the GUI."""

    def __init__(
        self,
        app: Any,
        *,
        parent: Any | None = None,
        logger_prefix: str | tuple[str, ...] = (
            "pipe_twin.calibration_wizard",
            "pipe_twin.qr_registration",
        ),
    ) -> None:
        from .logging_config import configure_logging

        self.app = app
        self.path = configure_logging()
        self.logger_prefix = logger_prefix
        tk, ttk = app.tk, app.ttk
        self.window = tk.Toplevel(parent or app.root)
        self.window.title("完整运行日志")
        self.window.geometry("1120x720")
        self.window.minsize(800, 520)
        self.window.transient(parent or app.root)
        self.only_calibration = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="")

        toolbar = ttk.Frame(self.window, padding=(10, 10, 10, 5))
        toolbar.pack(fill="x")
        ttk.Checkbutton(
            toolbar,
            text="仅显示相机标定日志",
            variable=self.only_calibration,
            command=self.reload,
        ).pack(side="left")
        ttk.Button(toolbar, text="刷新", command=self.reload).pack(side="left", padx=6)
        ttk.Button(toolbar, text="复制日志路径", command=self.copy_path).pack(
            side="left", padx=6
        )
        ttk.Label(toolbar, textvariable=self.status, foreground="#355371").pack(
            side="right"
        )

        body = ttk.Frame(self.window, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.text = tk.Text(body, wrap="none", font=("Consolas", 9), undo=False)
        vertical = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.reload()

    def reload(self) -> None:
        prefix = self.logger_prefix if self.only_calibration.get() else None
        try:
            payload, count = read_structured_log(self.path, logger_prefix=prefix)
        except OSError as error:
            payload, count = f"日志读取失败：{error}\n", 0
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", payload or "当前没有符合筛选条件的日志。\n")
        self.text.configure(state="disabled")
        self.text.see("end")
        self.status.set(f"{count} 条 · {self.path}")

    def copy_path(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(str(self.path))
        self.status.set(f"已复制：{self.path}")


__all__ = ["LogViewerDialog", "read_structured_log", "structured_log_files"]
