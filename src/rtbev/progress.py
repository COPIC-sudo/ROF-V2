from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressReporter:
    def __init__(self, work_dir: str | Path, task_name: str, total: int | None = None, enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.task_name = str(task_name)
        self.total = int(total) if total is not None else None
        self.enabled = bool(enabled)
        self.started_at = _utc_now()
        self.started_perf = time.perf_counter()
        self.stage = "initialized"
        self.step: int | None = None
        self.last_message = ""
        self.status = "running"
        self.out_dir = self.work_dir / "results" / "_progress" / self.task_name
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._write_status()

    def update(
        self,
        stage: str,
        step: int | None = None,
        total: int | None = None,
        message: str = "",
    ) -> None:
        if total is not None:
            self.total = int(total)
        self.stage = str(stage)
        if step is not None:
            self.step = int(step)
        if message:
            self.last_message = str(message)
        self.status = "running"
        self._write_status()
        if message:
            self.event(message)

    def event(self, message: str, level: str = "info") -> None:
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "time": _utc_now(),
                "level": str(level),
                "stage": self.stage,
                "message": str(message),
            },
            ensure_ascii=True,
        )
        with (self.out_dir / "events.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def complete(self, message: str = "") -> None:
        if message:
            self.last_message = str(message)
            self.event(message, level="info")
        self.status = "completed"
        self.stage = "completed"
        if self.total is not None:
            self.step = self.total
        self._write_status()

    def fail(self, message: str = "") -> None:
        if message:
            self.last_message = str(message)
            self.event(message, level="error")
        self.status = "failed"
        self._write_status()

    def _status_payload(self) -> dict[str, Any]:
        elapsed_s = max(time.perf_counter() - self.started_perf, 0.0)
        percent = None
        eta_s = None
        if self.step is not None and self.total:
            percent = min(max(float(self.step) / max(float(self.total), 1.0) * 100.0, 0.0), 100.0)
            if self.status == "running" and self.step > 0 and self.step < self.total:
                rate = elapsed_s / max(float(self.step), 1.0)
                eta_s = max(rate * (float(self.total) - float(self.step)), 0.0)
        return {
            "task": self.task_name,
            "stage": self.stage,
            "step": self.step,
            "total": self.total,
            "percent": percent,
            "elapsed_s": elapsed_s,
            "eta_s": eta_s,
            "last_message": self.last_message,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "status": self.status,
        }

    def _write_status(self) -> None:
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = self._status_payload()
        status_path = self.out_dir / "status.json"
        tmp_path = self.out_dir / "status.json.tmp"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, status_path)
