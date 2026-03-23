from geochemad.catalog import scan_datasets
from geochemad.settings import ProjectPaths


def test_scan_datasets_finds_all_subsets():
    paths = ProjectPaths()
    datasets = scan_datasets(paths.data_dir)
    assert len(datasets) == 8
    assert any(item.subset_id == "area1_sediment_au" for item in datasets)
