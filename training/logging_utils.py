"""CSV/JSONL metric logging always works; TensorBoard is a best-effort extra
layered on top so training never crashes over an environment/display issue
in the SummaryWriter.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, TextIO

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TrainingLogger:
    def __init__(self, log_dir: Path, run_name: str, use_tensorboard: bool = True) -> None:
        self.run_dir = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._writer: SummaryWriter | None = None
        if use_tensorboard:
            try:
                self._writer = SummaryWriter(log_dir=str(self.run_dir))
            except Exception:
                logger.warning(
                    "Failed to initialize TensorBoard SummaryWriter; continuing "
                    "with CSV/JSONL logging only.",
                    exc_info=True,
                )
                self._writer = None

        self._csv_path = self.run_dir / "metrics.csv"
        self._jsonl_path = self.run_dir / "metrics.jsonl"

        csv_exists = self._csv_path.exists() and self._csv_path.stat().st_size > 0
        self._csv_file: TextIO = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not csv_exists:
            self._csv_writer.writerow(["step", "tag", "value"])
            self._csv_file.flush()

        self._jsonl_file: TextIO = open(self._jsonl_path, "a")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

        self._csv_writer.writerow([step, tag, value])
        self._csv_file.flush()

        self._jsonl_file.write(json.dumps({"step": step, "tag": tag, "value": value}) + "\n")
        self._jsonl_file.flush()

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def close(self) -> None:
        self._csv_file.flush()
        self._csv_file.close()
        self._jsonl_file.flush()
        self._jsonl_file.close()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def read_metrics_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "tag": row["tag"],
                    "value": float(row["value"]),
                }
            )
    return rows
