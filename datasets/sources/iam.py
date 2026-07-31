"""IAM Handwriting Database — requires manual download (free account required,
see datasets/registry.py). Expects `lines.tgz` and `ascii.tgz` already
extracted under raw_dir/iam/ as:
    raw_dir/iam/lines/<form-group>/<form-id>/<line-id>.png
    raw_dir/iam/ascii/lines.txt

lines.txt format (one line per sample, IAM's own spec):
    <line-id> <ok|err> <graylevel> <n-components> <x> <y> <w> <h> <transcription>
where <transcription> has words joined by '|' instead of spaces.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from datasets.manifest import DatasetSample
from datasets.sources.base import DatasetSource


def _image_relative_path(line_id: str) -> Path:
    # e.g. "a01-000u-00" -> lines/a01/a01-000u/a01-000u-00.png
    parts = line_id.split("-")
    form_group, form_id = parts[0], f"{parts[0]}-{parts[1]}"
    return Path("lines") / form_group / form_id / f"{line_id}.png"


def parse_lines_txt(path: Path, include_err: bool = False) -> list[tuple[str, str]]:
    """Returns (line_id, transcript) pairs, skipping comments and (by default)
    samples IAM itself flagged as failed segmentation ('err')."""
    entries = []
    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.split(" ", 8)
            if len(fields) < 9:
                continue
            line_id, status = fields[0], fields[1]
            if status != "ok" and not include_err:
                continue
            transcript = fields[8].replace("|", " ")
            entries.append((line_id, transcript))
    return entries


class IamDatasetSource(DatasetSource):
    """All samples are currently assigned to the "train" split. IAM publishes
    an official writer-independent train/val/test split as separate list
    files; wiring that up is a documented follow-up (needs verification
    against an actual downloaded copy) rather than something to guess at here.
    """

    name = "iam"

    def __init__(self, include_err: bool = False) -> None:
        self._include_err = include_err

    def prepare(self, raw_dir: Path, processed_dir: Path) -> list[DatasetSample]:
        iam_raw = raw_dir / self.name
        lines_txt = iam_raw / "ascii" / "lines.txt"
        if not lines_txt.exists():
            raise FileNotFoundError(
                f"{lines_txt} not found. See datasets/registry.py for IAM acquisition "
                "instructions — this dataset requires a manual, registered download."
            )

        entries = parse_lines_txt(lines_txt, include_err=self._include_err)

        samples = []
        for line_id, transcript in entries:
            source_image_path = iam_raw / _image_relative_path(line_id)
            if not source_image_path.exists():
                continue  # image missing from the user's download; skip rather than fail the batch

            relative_path = Path(self.name) / "train" / "line" / f"{line_id}.png"
            full_path = processed_dir / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(full_path), image)

            samples.append(
                DatasetSample(
                    image_path=str(relative_path),
                    transcript=transcript,
                    source=self.name,
                    split="train",
                    label_type="line",
                    writer_id=None,
                )
            )
        return samples
