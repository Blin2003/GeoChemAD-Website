from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEMO_REGIONS = {
    "pilbara_hamersley": {"west": 117.5, "east": 119.8, "south": -23.8, "north": -21.2},
    "eastern_goldfields": {"west": 120.5, "east": 122.5, "south": -31.8, "north": -29.2},
    "murchison_yalgoo": {"west": 116.5, "east": 118.6, "south": -28.8, "north": -26.8},
}

ASSAY_FILES = {
    "sediment": "gswa_all_sediment.csv",
    "rockchips": "gswa_all_rockchips.csv",
    "drillhole": "gswa_all_drillhole_maxgrade.csv",
    "shallowdrill": "gswa_all_shallowdrill.csv",
    "soil": "gswa_all_surfsoilgeochem.csv",
}


def inside_regions(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for region in DEMO_REGIONS.values():
        mask |= (
            df["X"].between(region["west"], region["east"])
            & df["Y"].between(region["south"], region["north"])
        )
    return mask


def filter_csv(source: Path, dest: Path, chunksize: int) -> dict[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    total = 0
    kept = 0
    wrote_header = False
    for chunk in pd.read_csv(source, chunksize=chunksize):
        total += len(chunk)
        filtered = chunk[inside_regions(chunk)]
        kept += len(filtered)
        filtered.to_csv(dest, mode="a", header=not wrote_header, index=False)
        wrote_header = True
    return {"sourceRows": total, "keptRows": kept}


def copy_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def crop_coordinate_csv_tree(source: Path, dest: Path, chunksize: int) -> dict[str, dict[str, int] | str]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int] | str] = {}
    for src in sorted(source.rglob("*")):
        rel = src.relative_to(source)
        out = dest / rel
        if src.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix.lower() != ".csv":
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            stats[str(rel)] = "copied"
            continue
        columns = pd.read_csv(src, nrows=0).columns
        if {"X", "Y"}.issubset(columns):
            stats[str(rel)] = filter_csv(src, out, chunksize)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            stats[str(rel)] = "copied"
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small GeoChemAD demo data directory from teacher datasets.")
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "gad_reasoning_full_20260610",
        help="Path to gad_reasoning_full_20260610.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "demo",
        help="Output data root used by GEOCHEMAD_DATA_ROOT.",
    )
    parser.add_argument(
        "--sources",
        default="sediment",
        help="Comma-separated assay sources to crop, or 'all'. Current frontend needs sediment.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    teacher_data = args.teacher_root / "datasets"
    output = args.output.resolve()
    selected_sources = list(ASSAY_FILES) if args.sources == "all" else [s.strip() for s in args.sources.split(",") if s.strip()]

    stats: dict[str, object] = {
        "teacherRoot": str(args.teacher_root.resolve()),
        "output": str(output),
        "regions": DEMO_REGIONS,
        "sources": {},
        "notes": [
            "This demo data root contains complete rows inside selected regions, not synthetic data.",
            "Raster/vector layers are not cropped here because this local environment lacks GIS dependencies.",
            "Use the full teacher datasets for full geophysics/structure scoring.",
        ],
    }

    geochem_src = teacher_data / "geochemical" / "states"
    geochem_dest = output / "geochemical" / "states"
    for name in selected_sources:
        if name not in ASSAY_FILES:
            raise ValueError(f"Unknown source {name!r}; choose from {', '.join(ASSAY_FILES)}")
        src = geochem_src / "state_geochemical" / ASSAY_FILES[name]
        dest = geochem_dest / "state_geochemical" / ASSAY_FILES[name]
        print(f"Cropping {src.name} -> {dest}")
        stats["sources"][name] = filter_csv(src, dest, args.chunksize)

    stats["siteTables"] = {}
    for dirname in ("sites", "sites_enriched"):
        src = geochem_src / dirname
        dest = geochem_dest / dirname
        print(f"Cropping coordinate tables {src} -> {dest}")
        stats["siteTables"][dirname] = crop_coordinate_csv_tree(src, dest, args.chunksize)

    (output / "selected_layers").mkdir(parents=True, exist_ok=True)
    (output / "selected_layers" / "README.txt").write_text(
        "Demo data does not copy the multi-GB raster/vector layers. "
        "Set GEOCHEMAD_DATA_ROOT to the full teacher datasets directory for those layers.\n",
        encoding="utf-8",
    )
    (output / "demo_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nUse with:\nGEOCHEMAD_DATA_ROOT={output} python3 scripts/serve.py")


if __name__ == "__main__":
    main()
