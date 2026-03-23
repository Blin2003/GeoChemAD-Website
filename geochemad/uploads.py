from __future__ import annotations

import csv
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import UploadFile

from .catalog import DatasetRecord


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "uploaded_dataset"


def _read_shape(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, header


def save_uploaded_dataset(
    upload_root: Path,
    dataset_name: str,
    target_element: str,
    sample_file: UploadFile,
    site_file: UploadFile,
) -> DatasetRecord:
    dataset_id = slugify(dataset_name)
    target = target_element.upper()
    dataset_dir = upload_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sample_path = dataset_dir / "samples.csv"
    site_path = dataset_dir / "sites.csv"
    with sample_path.open("wb") as handle:
        shutil.copyfileobj(sample_file.file, handle)
    with site_path.open("wb") as handle:
        shutil.copyfileobj(site_file.file, handle)
    sample_rows, sample_columns = _read_shape(sample_path)
    site_rows, site_columns = _read_shape(site_path)
    return DatasetRecord(
        subset_id=dataset_id,
        area="upload",
        source="uploaded",
        target=target,
        sample_path=sample_path,
        site_path=site_path,
        sample_rows=sample_rows,
        site_rows=site_rows,
        sample_columns=sample_columns,
        site_columns=site_columns,
    )


def load_uploaded_datasets(upload_root: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for dataset_dir in sorted(upload_root.iterdir() if upload_root.exists() else []):
        if not dataset_dir.is_dir():
            continue
        sample_path = dataset_dir / "samples.csv"
        site_path = dataset_dir / "sites.csv"
        meta_path = dataset_dir / "meta.json"
        if not sample_path.exists() or not site_path.exists():
            continue
        sample_rows, sample_columns = _read_shape(sample_path)
        site_rows, site_columns = _read_shape(site_path)
        target = "UNKNOWN"
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            target = str(meta.get("target_element", "UNKNOWN")).upper()
        records.append(
            DatasetRecord(
                subset_id=dataset_dir.name,
                area="upload",
                source="uploaded",
                target=target,
                sample_path=sample_path,
                site_path=site_path,
                sample_rows=sample_rows,
                site_rows=site_rows,
                sample_columns=sample_columns,
                site_columns=site_columns,
            )
        )
    return records
