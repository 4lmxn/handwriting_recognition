import csv
import json
from pathlib import Path

from training.logging_utils import TrainingLogger, read_metrics_csv


def test_construction_creates_run_directory(tmp_path: Path):
    TrainingLogger(tmp_path, "run1", use_tensorboard=False)
    assert (tmp_path / "run1").is_dir()


def test_log_scalar_appends_to_csv_and_jsonl(tmp_path: Path):
    logger = TrainingLogger(tmp_path, "run1", use_tensorboard=False)
    logger.log_scalar("loss", 0.5, 1)
    logger.log_scalar("loss", 0.25, 2)
    logger.close()

    run_dir = tmp_path / "run1"
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "tag", "value"]
    assert rows[1] == ["1", "loss", "0.5"]
    assert rows[2] == ["2", "loss", "0.25"]

    with open(run_dir / "metrics.jsonl") as f:
        lines = [json.loads(line) for line in f]
    assert lines == [
        {"step": 1, "tag": "loss", "value": 0.5},
        {"step": 2, "tag": "loss", "value": 0.25},
    ]

    parsed = read_metrics_csv(run_dir / "metrics.csv")
    assert parsed == [
        {"step": 1, "tag": "loss", "value": 0.5},
        {"step": 2, "tag": "loss", "value": 0.25},
    ]


def test_log_scalars_logs_multiple_metrics_at_same_step(tmp_path: Path):
    logger = TrainingLogger(tmp_path, "run1", use_tensorboard=False)
    logger.log_scalars({"loss": 0.5, "accuracy": 0.9}, 3)
    logger.close()

    parsed = read_metrics_csv(tmp_path / "run1" / "metrics.csv")
    assert parsed == [
        {"step": 3, "tag": "loss", "value": 0.5},
        {"step": 3, "tag": "accuracy", "value": 0.9},
    ]


def test_context_manager_closes_cleanly_and_flushes(tmp_path: Path):
    with TrainingLogger(tmp_path, "run1", use_tensorboard=False) as logger:
        logger.log_scalar("loss", 1.0, 0)
        logger.log_scalar("loss", 0.8, 1)

    parsed = read_metrics_csv(tmp_path / "run1" / "metrics.csv")
    assert parsed == [
        {"step": 0, "tag": "loss", "value": 1.0},
        {"step": 1, "tag": "loss", "value": 0.8},
    ]

    with open(tmp_path / "run1" / "metrics.jsonl") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2


def test_csv_header_appears_exactly_once(tmp_path: Path):
    logger = TrainingLogger(tmp_path, "run1", use_tensorboard=False)
    for step in range(5):
        logger.log_scalar("loss", float(step), step)
    logger.close()

    with open(tmp_path / "run1" / "metrics.csv") as f:
        content = f.read()
    assert content.count("step,tag,value") == 1


def test_use_tensorboard_false_produces_correct_output(tmp_path: Path):
    logger = TrainingLogger(tmp_path, "run_no_tb", use_tensorboard=False)
    assert logger._writer is None
    logger.log_scalar("cer", 0.12, 10)
    logger.close()

    parsed = read_metrics_csv(tmp_path / "run_no_tb" / "metrics.csv")
    assert parsed == [{"step": 10, "tag": "cer", "value": 0.12}]

    with open(tmp_path / "run_no_tb" / "metrics.jsonl") as f:
        lines = [json.loads(line) for line in f]
    assert lines == [{"step": 10, "tag": "cer", "value": 0.12}]


def test_reopening_run_directory_appends_without_duplicate_header(tmp_path: Path):
    logger1 = TrainingLogger(tmp_path, "run1", use_tensorboard=False)
    logger1.log_scalar("loss", 1.0, 0)
    logger1.close()

    logger2 = TrainingLogger(tmp_path, "run1", use_tensorboard=False)
    logger2.log_scalar("loss", 0.5, 1)
    logger2.close()

    with open(tmp_path / "run1" / "metrics.csv") as f:
        content = f.read()
    assert content.count("step,tag,value") == 1

    parsed = read_metrics_csv(tmp_path / "run1" / "metrics.csv")
    assert parsed == [
        {"step": 0, "tag": "loss", "value": 1.0},
        {"step": 1, "tag": "loss", "value": 0.5},
    ]
