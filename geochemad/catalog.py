from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_PATTERN = re.compile(r"area(?P<area>\d+)_(?P<source>[a-z]+)_(?P<target>[a-z]+)\.csv$")


@dataclass
class DatasetRecord:
    subset_id: str
    area: str
    source: str
    target: str
    sample_path: Path
    site_path: Path
    sample_rows: int
    site_rows: int
    sample_columns: list[str]
    site_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sample_path"] = str(self.sample_path)
        data["site_path"] = str(self.site_path)
        return data


def _read_csv_shape(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, header


def scan_datasets(data_dir: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for sample_path in sorted(data_dir.glob("*.csv")):
        if sample_path.name.endswith("_site.csv"):
            continue
        match = DATASET_PATTERN.match(sample_path.name)
        if not match:
            continue
        site_path = sample_path.with_name(sample_path.stem + "_site.csv")
        if not site_path.exists():
            continue
        sample_rows, sample_columns = _read_csv_shape(sample_path)
        site_rows, site_columns = _read_csv_shape(site_path)
        area = match.group("area")
        source = match.group("source")
        target = match.group("target")
        subset_id = f"area{area}_{source}_{target}"
        records.append(
            DatasetRecord(
                subset_id=subset_id,
                area=area,
                source=source,
                target=target.upper(),
                sample_path=sample_path,
                site_path=site_path,
                sample_rows=sample_rows,
                site_rows=site_rows,
                sample_columns=sample_columns,
                site_columns=site_columns,
            )
        )
    return records
