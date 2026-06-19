# GeoChemAD

GeoChemAD is the frontend and HTTP presentation adapter for the supervisor-provided
`gad_reasoning_full_20260610` backend. The supervisor backend code is vendored in
this repository under `supervisor_backend/gad_reasoning_full_20260610`; large data
files are intentionally not vendored.

## Architecture Boundary

The vendored supervisor backend owns:

- target configurations
- GSWA source loading
- enriched known-site data
- background generation
- expert construction and fitting
- `ProspectivityModel` and `NodeScore`
- geological narrative generation

GeoChemAD owns:

- target selection
- map, coordinate, place, and region interactions
- API parameter/result serialization
- visual presentation of scores, signals, experts, and narratives

There is no independent GeoChemAD scoring model, CSV training workflow, IDW fallback,
or synthetic prospectivity demo. Scoring uses the vendored supervisor backend.

## Targets

The target list is read from the supervisor comprehensive evaluation configuration:

`Cu`, `Au`, `Ni`, `W`, `Sn`, `Co`, `Ta`, `Mn`, and `REE`.

Spatial AUC values shown in the interface come from the supervisor evaluation report.

## Local Execution Profile

The local adapter can run either a fast demo subset or a full-data profile. The
full-data profile registers the supervisor geochemistry, geophysics, and
structure/geology sources when the external data root contains those files.

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

The supervisor backend code is included at:

```text
supervisor_backend/gad_reasoning_full_20260610
```

Set `GAD_REASONING_ROOT` only if you intentionally want to test another backend
code checkout.

By default, full data is read from:

```text
../gad_reasoning_full_20260610/datasets
```

Set `GEOCHEMAD_FULL_DATA_ROOT` to point full-data mode at another datasets
directory. Set `GEOCHEMAD_DATA_ROOT` only for a custom single data root.

## Data Profiles

Full local data:

```bash
GEOCHEMAD_FULL_DATA_ROOT=/path/to/gad_reasoning_full_20260610/datasets \
python3 scripts/serve.py
```

Demo data subset:

```bash
python3 scripts/prepare_demo_data.py --sources sediment
python3 scripts/serve.py
```

The demo data script copies complete rows inside selected WA regions from the
teacher datasets. It does not synthesize values and does not ingest user-provided datasets.
Large raster/vector layers remain in the full teacher data profile unless a GIS
tiling/cropping step is added.

## Project Layout

- `geochemad/research_backend.py`: thin adapter around supervisor APIs
- `geochemad/api.py`: research session and UI support endpoints
- `supervisor_backend/gad_reasoning_full_20260610`: vendored supervisor backend code
- `web/home.html`: supervisor target selector
- `scripts/prepare_demo_data.py`: create a small local demo data root from teacher datasets
- `web/analysis.html`: map and result workspace
- `scripts/serve.py`: local server entrypoint
