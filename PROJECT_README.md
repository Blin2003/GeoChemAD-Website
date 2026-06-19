# GeoChemAD

GeoChemAD is the frontend and HTTP presentation adapter for the supervisor-provided
`gad_reasoning_full_20260610` project.

## Architecture Boundary

The supervisor project owns:

- target configurations
- GSWA source loading
- enriched known-site data
- background generation
- expert construction and fitting
- `ProspectivityModel` and `NodeScore`
- geological narrative generation

GeoChemAD owns only:

- target selection
- map, coordinate, place, and region interactions
- API parameter/result serialization
- visual presentation of scores, signals, experts, and narratives

There is no independent GeoChemAD scoring model, CSV training workflow, IDW fallback,
or synthetic prospectivity demo.

## Targets

The target list is read from the supervisor comprehensive evaluation configuration:

`Cu`, `Au`, `Ni`, `W`, `Sn`, `Co`, `Ta`, `Mn`, and `REE`.

Spatial AUC values shown in the interface come from the supervisor evaluation report.

## Local Execution Profile

The local adapter runs the supervisor `geochem` layer with the real GSWA
stream-sediment source. The full supervisor catalog also contains multi-gigabyte
geophysics, structure, and additional assay layers intended for the research/HPC
environment. GeoChemAD does not emulate those missing layers.

Scores are returned only where the active supervisor experts have sufficient local
evidence.

## Run

```bash
python3 -m pip install --user -r requirements.txt
python3 scripts/serve.py
```

Open:

```text
http://127.0.0.1:8000
```

The adapter expects the supervisor project code at:

```text
../gad_reasoning_full_20260610
```

Set `GAD_REASONING_ROOT` to override that location.

By default, data is read from:

```text
$GAD_REASONING_ROOT/datasets
```

Set `GEOCHEMAD_DATA_ROOT` to use another data directory, for example a small
demo subset created from the teacher data.

## Data Profiles

Full local data:

```bash
GAD_REASONING_ROOT=/path/to/gad_reasoning_full_20260610 \
python3 scripts/serve.py
```

Demo data subset:

```bash
python3 scripts/prepare_demo_data.py --sources sediment
GEOCHEMAD_DATA_ROOT=/path/to/GeoChemAD/data/demo python3 scripts/serve.py
```

The demo data script copies complete rows inside selected WA regions from the
teacher datasets. It does not synthesize values and does not ingest user-provided datasets.
Large raster/vector layers remain in the full teacher data profile unless a GIS
tiling/cropping step is added.

## Project Layout

- `geochemad/research_backend.py`: thin adapter around supervisor APIs
- `geochemad/api.py`: research session and UI support endpoints
- `web/home.html`: supervisor target selector
- `scripts/prepare_demo_data.py`: create a small local demo data root from teacher datasets
- `web/analysis.html`: map and result workspace
- `scripts/serve.py`: local server entrypoint
