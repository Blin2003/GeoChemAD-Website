from __future__ import annotations

import os
import sys
import threading
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


WA_BBOX = (114.5, 128.5, -33.5, -15.0)

TARGETS: dict[str, dict[str, Any]] = {
    "Cu": {"label": "Copper", "spatial_auc": 0.5758333333333333},
    "Au": {"label": "Gold", "spatial_auc": 0.7303333333333334},
    "Ni": {"label": "Nickel", "spatial_auc": 0.917},
    "W": {"label": "Tungsten", "spatial_auc": 0.7408333333333333},
    "Sn": {"label": "Tin", "spatial_auc": 0.6643333333333333},
    "Co": {"label": "Cobalt", "spatial_auc": 0.8045652173913043},
    "Ta": {"label": "Tantalum", "spatial_auc": 0.782},
    "Mn": {"label": "Manganese", "spatial_auc": 0.2565},
    "REE": {"label": "Rare earth elements", "spatial_auc": 0.5773333333333334},
}

SCORE_TIERS = (
    (0.80, "Strong", "HIGH"),
    (0.65, "Moderate", "MODERATE"),
    (0.52, "Weak", "MODERATE"),
    (0.00, "Background", "LOW"),
)


def research_root() -> Path:
    configured = os.environ.get("GAD_REASONING_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "gad_reasoning_full_20260610"


def data_root() -> Path:
    configured = os.environ.get("GEOCHEMAD_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return research_root() / "datasets"


def data_profile() -> dict[str, str]:
    root = data_root()
    if os.environ.get("GEOCHEMAD_DATA_ROOT"):
        name = "Demo subset" if root.name == "demo" else "Custom data root"
        mode = "custom"
    else:
        name = "Full teacher datasets"
        mode = "teacher-full"
    return {"name": name, "mode": mode, "path": str(root)}


def demo_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "demo"


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_teacher_symbols() -> dict[str, Any]:
    root = research_root()
    if not root.exists():
        raise RuntimeError(f"Teacher backend was not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from core.catalog import DataCatalog, SourceSpec
    from domains.geochem.narrative import _expert_source, describe
    from domains.geochem.prospectivity_model import ProspectivityModel, TargetConfig
    from domains.geochem.samples import GeochemSample
    from scripts.eval_comprehensive import METAL_CONFIGS

    return {
        "DataCatalog": DataCatalog,
        "SourceSpec": SourceSpec,
        "ProspectivityModel": ProspectivityModel,
        "TargetConfig": TargetConfig,
        "GeochemSample": GeochemSample,
        "describe": describe,
        "expert_source": _expert_source,
        "METAL_CONFIGS": METAL_CONFIGS,
    }


def _target_key(value: str | None) -> str:
    key = str(value or "Cu").strip()
    if key not in TARGETS:
        raise ValueError(f"The supervisor model supports: {', '.join(TARGETS)}")
    return key


def _inside_wa(lon: float, lat: float) -> bool:
    west, east, south, north = WA_BBOX
    return west <= lon <= east and south <= lat <= north


def _distance_km(lon: float, lat: float, points: np.ndarray) -> np.ndarray:
    lon_scale = 111.0 * np.cos(np.radians(abs(lat)))
    dx = (points[:, 0] - lon) * lon_scale
    dy = (points[:, 1] - lat) * 111.0
    return np.sqrt(dx * dx + dy * dy)


def _score_tier(score: float) -> tuple[str, str]:
    for threshold, tier, level in SCORE_TIERS:
        if score >= threshold:
            return tier, level
    return "Background", "LOW"


@dataclass
class ResearchSession:
    session_id: str
    target: str
    profile_key: str = "default"


class ResearchModelService:
    """Adapter around the supervisor-provided ProspectivityModel.

    One model is kept active at a time because the teacher backend uses a
    process-global DataSourceRegistry. Target switching therefore rebuilds the
    active model, keeping memory bounded and scoring behavior deterministic.
    """

    def __init__(
        self,
        n_background: int = 60,
        preview_limit: int = 2500,
        data_root_override: Path | None = None,
        profile_name: str | None = None,
        profile_mode: str | None = None,
    ) -> None:
        self._n_background = n_background
        self._preview_limit = preview_limit
        self._data_root_override = data_root_override
        self._profile_name = profile_name
        self._profile_mode = profile_mode
        self._lock = threading.RLock()
        self._symbols: dict[str, Any] | None = None
        self._model: Any = None
        self._target: str | None = None
        self._source: Any = None
        self._enabled_layers: frozenset[str] = frozenset({"geochem"})
        self._sites = pd.DataFrame()
        self._site_tree: cKDTree | None = None

    def _data_root(self) -> Path:
        if self._data_root_override is not None:
            return self._data_root_override.expanduser().resolve()
        return data_root()

    def _data_profile(self) -> dict[str, str]:
        root = self._data_root()
        if self._profile_name:
            name = self._profile_name
            mode = self._profile_mode or "custom"
        elif self._data_root_override is not None:
            name = "Demo subset" if root.name == "demo" else "Custom data root"
            mode = self._profile_mode or "custom"
        else:
            return data_profile()
        return {"name": name, "mode": mode, "path": str(root)}

    def coverage_regions(self) -> list[dict[str, Any]]:
        manifest = self._data_root() / "demo_manifest.json"
        if manifest.exists():
            try:
                regions = json.loads(manifest.read_text(encoding="utf-8")).get("regions", {})
            except (json.JSONDecodeError, OSError):
                regions = {}
            return [
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "bounds": bounds,
                    "type": "demo-data-region",
                }
                for key, bounds in regions.items()
            ]
        west, east, south, north = WA_BBOX
        return [
            {
                "key": "wa_model_extent",
                "label": "WA model extent",
                "bounds": {"west": west, "east": east, "south": south, "north": north},
                "type": "model-extent",
            }
        ]

    def target_options(self) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "label": config["label"],
                "spatialAuc": config["spatial_auc"],
                "validation": "spatial holdout",
            }
            for key, config in TARGETS.items()
        ]

    def layer_inventory(self) -> dict[str, Any]:
        root = research_root()
        datasets = self._data_root()
        selected = datasets / "selected_layers"
        geochem = datasets / "geochemical" / "states"

        def item(name: str, kind: str, path: Path, role: str, *, active: bool = False, note: str = "") -> dict[str, Any]:
            return {
                "name": name,
                "kind": kind,
                "path": _display_path(path, root),
                "exists": path.exists(),
                "active": active,
                "role": role,
                "note": note,
            }

        return {
            "source": "gad_reasoning_full_20260610",
            "dataRoot": str(datasets),
            "dataProfile": self._data_profile(),
            "coverageRegions": self.coverage_regions(),
            "scoringMode": "geochem-only frontend adapter",
            "groups": [
                {
                    "key": "geochem",
                    "label": "Geochemistry",
                    "active": True,
                    "items": [
                        item(
                            "GSWA stream sediment samples",
                            "csv",
                            geochem / "state_geochemical" / "gswa_all_sediment.csv",
                            "Active scoring input used by the supervisor geochemical experts.",
                            active=True,
                        ),
                        item(
                            "Target mineral sites",
                            "csv",
                            geochem / "sites_enriched",
                            "Known mine/deposit labels used by the supervisor target configuration.",
                            active=True,
                        ),
                    ],
                },
                {
                    "key": "geophys",
                    "label": "Magnetics and gravity",
                    "active": False,
                    "items": [
                        item(
                            "Magnetic 1VD raster",
                            "raster",
                            selected / "rasters" / "magnetics" / "WA_80m_Mag_Merge_1VD_v1_2020.ers",
                            "Teacher geophysics source channel for magnetic-edge evidence.",
                            note="Available in the teacher package; not activated by the current lightweight frontend adapter.",
                        ),
                        item(
                            "Bouguer gravity raster",
                            "raster",
                            selected / "rasters" / "gravity" / "WA_400m_Grav_Merge_v1_2020.ers",
                            "Teacher geophysics source channel for density-contrast evidence.",
                            note="Available in the teacher package; needs tiling before browser map overlay.",
                        ),
                        item(
                            "Radiometrics K/Th/U rasters",
                            "raster",
                            selected / "rasters" / "radiometrics",
                            "Teacher geophysics source channels for alteration and surface radiometric context.",
                            note="Available in the teacher package; not part of the current score.",
                        ),
                    ],
                },
                {
                    "key": "structure",
                    "label": "Structure and geology",
                    "active": False,
                    "items": [
                        item(
                            "Fault and structural lines",
                            "vector",
                            selected / "vectors" / "structures" / "500k_interpstrucl20.shp",
                            "Teacher structure source for fault proximity and density evidence.",
                            note="Available in the teacher package; vector preview requires a GeoJSON/tile conversion step.",
                        ),
                        item(
                            "Magnetic and gravity worms",
                            "vector",
                            selected / "vectors" / "geophysics_worms",
                            "Teacher structure source for geophysical-gradient line proximity.",
                            note="Available in the teacher package; not activated by the current score.",
                        ),
                        item(
                            "Geology and Cenozoic cover",
                            "vector",
                            selected / "vectors" / "geology_context",
                            "Teacher geology source for rock type, age, craton membership, and cover masking.",
                            note="Available in the teacher package; not activated by the current score.",
                        ),
                    ],
                },
            ],
        }

    def create_session(self, target: str | None = None) -> ResearchSession:
        key = _target_key(target)
        self.ensure_model(key)
        return ResearchSession(session_id=uuid4().hex, target=key)

    def ensure_model(self, target: str) -> Any:
        key = _target_key(target)
        with self._lock:
            if self._model is not None and self._target == key:
                return self._model

            symbols = self._symbols or _load_teacher_symbols()
            self._symbols = symbols
            datasets = self._data_root()
            config = symbols["METAL_CONFIGS"][key]

            catalog = symbols["DataCatalog"]()
            geochem_dir = datasets / "geochemical/states/state_geochemical"
            raster_dir = datasets / "selected_layers/rasters"
            vector_dir = datasets / "selected_layers/vectors"
            assay_specs = [
                ("sediment", "sediment", "gswa_all_sediment.csv", "STREA"),
                ("rockchips", "rockchip", "gswa_all_rockchips.csv", "ROCKC"),
                ("drillhole", "drillhole", "gswa_all_drillhole_maxgrade.csv", "DRILL"),
                ("shallowdrill", "shallowdrill", "gswa_all_shallowdrill.csv", "SHALL"),
                ("soil", "soil", "gswa_all_surfsoilgeochem.csv", "SOIL"),
            ]
            assay_registered = []
            for name, subtype, filename, sample_filter in assay_specs:
                path = geochem_dir / filename
                if path.exists():
                    catalog.register(
                        symbols["SourceSpec"](
                            name=name,
                            source_type="assay_spatial",
                            modality="geochemistry",
                            subtype=subtype,
                            path=str(path),
                            loader_kwargs={"sample_type_filter": sample_filter},
                        )
                    )
                    assay_registered.append(name)
            raster_registered = []
            for rkey, subdir, filename, resolution in [
                ("mag", "magnetics", "WA_80m_Mag_Merge_1VD_v1_2020.ers", 0.08),
                ("grav", "gravity", "WA_400m_Grav_Merge_v1_2020.ers", 0.40),
                ("K", "radiometrics", "WA_80m_K_Merge_v1_2018.ers", 0.08),
                ("Th", "radiometrics", "WA_80m_Th_Merge_v1_2018.ers", 0.08),
                ("U", "radiometrics", "WA_80m_U_Merge_v1_2018.ers", 0.08),
                ("LuHf", "geochronology", "WA_LuHf_TDM2_masked.bil", 0.10),
                ("SmNd", "geochronology", "WA_SmNd_TDM2_masked.bil", 0.10),
            ]:
                path = raster_dir / subdir / filename
                if path.exists():
                    catalog.register(
                        symbols["SourceSpec"](
                            name=f"raster_{rkey}",
                            source_type="raster",
                            modality="geophysics",
                            subtype=rkey,
                            path=str(path),
                            resolution_km=resolution,
                            loader_kwargs={"raster_key": rkey},
                        )
                    )
                    raster_registered.append(rkey)
            vector_registered = []
            for layer_key, subdir, filename in [
                ("fault", "structures", "500k_interpstrucl20.shp"),
                ("worm_mag", "geophysics_worms", "worm_mag.shp"),
                ("worm_grav", "geophysics_worms", "worm_grav.shp"),
                ("geology", "geology_context", "GeologyMERGED.shp"),
                ("cenozoic", "geology_context", "500k_cenozoicp20.shp"),
            ]:
                path = vector_dir / subdir / filename
                if path.exists():
                    catalog.register(
                        symbols["SourceSpec"](
                            name=f"vector_{layer_key}",
                            source_type="vector",
                            modality="geology",
                            subtype=layer_key,
                            path=str(path),
                            loader_kwargs={"layer_key": layer_key},
                        )
                    )
                    vector_registered.append(layer_key)
            enabled_layers = set()
            if assay_registered:
                enabled_layers.add("geochem")
            if raster_registered:
                enabled_layers.add("geophys")
            if {"fault", "worm_mag", "worm_grav", "geology", "cenozoic"}.issubset(vector_registered):
                enabled_layers.add("structure")
            target_config = symbols["TargetConfig"](
                target=key,
                pathfinders=config["pathfinders"],
                ratio_features=config["ratio_features"],
                confidence_map=config["confidence_map"],
                sites_csv=str(datasets / f"geochemical/states/sites_enriched/enriched_{key}_sites.csv"),
                layers=frozenset(enabled_layers),
            )
            model = symbols["ProspectivityModel"](
                catalog=catalog,
                config=target_config,
                n_bg=self._n_background,
                bbox=WA_BBOX,
                seed=42,
            )
            model.plan()
            model.setup()

            self._model = model
            self._target = key
            self._enabled_layers = frozenset(enabled_layers)
            self._source = model._registry.get("sediment")
            self._load_sites(key)
            return model

    def _load_sites(self, target: str) -> None:
        path = self._data_root() / f"geochemical/states/sites_enriched/enriched_{target}_sites.csv"
        sites = pd.read_csv(path)
        target_col = f"{target}_SITES"
        if target_col in sites:
            sites = sites[sites[target_col].astype(str).str.contains("Mine|Deposit", regex=True, na=False)]
        sites = sites.dropna(subset=["X", "Y"]).copy()
        if self._source is not None and hasattr(self._source, "_assay_xy") and len(sites):
            assay_xy = self._source._assay_xy
            if len(assay_xy):
                site_xy = sites[["X", "Y"]].to_numpy(dtype=float)
                site_km = np.column_stack([site_xy[:, 0] * 100.0, site_xy[:, 1] * 111.0])
                sample_km = np.column_stack([assay_xy[:, 0] * 100.0, assay_xy[:, 1] * 111.0])
                distances, _ = cKDTree(sample_km).query(site_km, k=1)
                sites = sites[distances <= 30.0].copy()
        name_col = next((column for column in ("SHORT_NAME", "SITE_TITLE", "SITE_CODE") if column in sites), None)
        sites["_name"] = sites[name_col].fillna("Site").astype(str) if name_col else "Site"
        self._sites = sites.reset_index(drop=True)
        coords = self._sites[["X", "Y"]].to_numpy(dtype=float)
        km_coords = np.column_stack([coords[:, 0] * 100.0, coords[:, 1] * 111.0])
        self._site_tree = cKDTree(km_coords) if len(km_coords) else None

    def session_payload(self, session: ResearchSession, target: str | None = None) -> dict[str, Any]:
        key = _target_key(target or session.target)
        self.ensure_model(key)
        session.target = key
        config = TARGETS[key]

        xy = self._source._assay_xy
        count = len(xy)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(count, size=min(self._preview_limit, count), replace=False)
        sample_preview = [
            {"lon": float(xy[index, 0]), "lat": float(xy[index, 1]), "label": "survey"}
            for index in sample_idx
        ]
        site_points = [
            {
                "lon": float(row.X),
                "lat": float(row.Y),
                "name": str(row["_name"]),
                "type": str(row.get(f"{key}_SITES", "Mine/Deposit")),
                "commodity": key,
            }
            for _, row in self._sites.head(2500).iterrows()
        ]

        return {
            "sessionId": session.session_id,
            "sessionKind": "research",
            "target": key,
            "targetLabel": config["label"],
            "targetOptions": self.target_options(),
            "sampleCount": int(count),
            "siteCount": int(len(self._sites)),
            "targetSiteCount": int(len(self._sites)),
            "samplePreview": sample_preview,
            "sitePoints": site_points,
            "coverageRegions": self.coverage_regions(),
            "availableElements": [],
            "knowledgeEntry": {
                "text_description": (
                    f"Supervisor-provided GAD reasoning model for {config['label']}. "
                    "GeoChemAD only renders the model output; target configuration, "
                    "background generation, expert fitting, scoring, and narrative come from the supervisor project."
                ),
                "top_element_combination": {
                    "combination": " / ".join(self._symbols["METAL_CONFIGS"][key]["ratio_features"]),
                    "co_anomaly_pct": None,
                },
            },
            "ranking": [],
            "combos": [],
            "evaluation": {"auc": config["spatial_auc"], "type": "spatial_holdout"},
            "modelMetadata": {
                "source": "gad_reasoning_full_20260610",
                "dataRoot": str(self._data_root()),
                "dataProfile": self._data_profile(),
                "engine": "ProspectivityModel",
                "layers": ["geochem", "GSWA stream sediment"],
                "experts": self._model.active_experts(),
                "activeSources": self._model.active_sources(),
                "spatialAuc": config["spatial_auc"],
                "validation": "spatial holdout",
                "ownership": "Supervisor backend",
                "layerStatus": [
                    {
                        "name": "Geochemistry",
                        "active": True,
                        "role": "Scoring evidence from GSWA stream sediment",
                    },
                    {
                        "name": "Magnetics and gravity",
                        "active": bool({"magnetic_geophysics", "gravity_geophysics", "radiometric_geophysics", "geochron_geophysics"} & set(self._model.active_experts())),
                        "role": "Teacher geophysics rasters registered when this profile has raster data and GIS dependencies",
                    },
                    {
                        "name": "Structure and geology",
                        "active": bool({"fault_structure", "worm_structure", "geology_structure"} & set(self._model.active_experts())),
                        "role": "Teacher fault, worm, and geology vectors registered when this profile has vector data and GIS dependencies",
                    },
                ],
                "geophysicsFeatures": ["mag", "mag_grad", "grav", "grav_grad"],
                "geophysicsOverlay": False,
                "coverage": {"west": WA_BBOX[0], "east": WA_BBOX[1], "south": WA_BBOX[2], "north": WA_BBOX[3]},
            },
            "layerInventory": self.layer_inventory(),
            "modelWarning": (
                "Scores are available only where the supervisor model has local GSWA sediment evidence inside "
                "Western Australia. Magnetics and gravity are supervisor evidence layers but are not active in "
                "this local geochem-only run. Spatial AUC is evaluation evidence, not a probability guarantee."
            ),
            "nearCount": 0,
            "bufferCount": 0,
            "backgroundCount": 0,
            "nearDepositCoverage": int(len(self._sites)),
            "referenceExample": None,
            "manualTemplate": {},
        }

    def _nearest_site(self, lon: float, lat: float) -> dict[str, Any] | None:
        if self._site_tree is None or self._sites.empty:
            return None
        query = np.array([lon * 100.0, lat * 111.0])
        distance, index = self._site_tree.query(query, k=1)
        row = self._sites.iloc[int(index)]
        return {
            "name": str(row["_name"]),
            "type": str(row.get(f"{self._target}_SITES", "Mine/Deposit")),
            "commodity": str(self._target),
            "distanceKm": float(distance),
        }

    def _evidence_layers(self, signals: list[dict[str, Any]], expert_breakdown: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active_sources = set(self._model.active_sources())
        active_experts = set(self._model.active_experts())
        group_defs = [
            {
                "key": "geochem",
                "label": "Geochemistry",
                "sources": {"sediment", "rockchips", "drillhole", "shallowdrill", "soil"},
                "expertTokens": ("enrichment", "pathfinder", "correlation"),
            },
            {
                "key": "geophys",
                "label": "Magnetics / Gravity / Radiometrics",
                "sources": {"geophysics", "magnetic", "gravity", "radiometric", "geochron"},
                "expertTokens": ("magnetic", "gravity", "radiometric", "geochron"),
            },
            {
                "key": "structure",
                "label": "Structure / Geology",
                "sources": {"structure", "fault", "worm", "geology"},
                "expertTokens": ("fault", "worm", "geology"),
            },
        ]
        layers = []
        for group in group_defs:
            group_signals = [
                signal for signal in signals
                if signal.get("source") in group["sources"]
                or any(token in str(signal.get("feature", "")).lower() for token in group["expertTokens"])
            ]
            group_experts = [
                expert for expert in expert_breakdown
                if any(token in str(expert.get("name", "")).lower() for token in group["expertTokens"])
            ]
            active = bool(active_sources & group["sources"] or active_experts & {expert["name"] for expert in group_experts})
            contribution = sum(abs(signal["zScore"] * signal["weight"]) for signal in group_signals)
            status = "active" if active else "not_used"
            layers.append(
                {
                    "key": group["key"],
                    "label": group["label"],
                    "available": True,
                    "active": active,
                    "status": status,
                    "signalCount": len(group_signals),
                    "expertCount": len(group_experts),
                    "contribution": float(contribution),
                    "topSignals": group_signals[:4],
                    "note": (
                        "Used in this score via the supervisor NodeScore."
                        if active
                        else "Available in the teacher package, but not part of this current frontend scoring run."
                    ),
                }
            )
        return layers

    def _feature_weight(self, node_score: Any, feature: str, source: str | None = None) -> float:
        if ":" in feature:
            inferred_source, bare_feature = feature.split(":", 1)
        else:
            inferred_source, bare_feature = source, feature
        best = 0.0

        def collect(node: Any) -> None:
            nonlocal best
            weight = float(
                node.weights.get(
                    feature,
                    node.weights.get(bare_feature, node.weights.get(inferred_source or "", 0.0)),
                )
            )
            if weight > best:
                best = weight
            for child in node.children.values():
                collect(child)

        collect(node_score)
        return best

    def score_point(self, session: ResearchSession, lon: float, lat: float, target: str | None = None) -> dict[str, Any]:
        key = _target_key(target or session.target)
        if not _inside_wa(lon, lat):
            raise ValueError("This research model only covers Western Australia (114.5-128.5E, 33.5-15S).")

        with self._lock:
            model = self.ensure_model(key)
            session.target = key
            sample = self._symbols["GeochemSample"](site_code=f"web_{uuid4().hex}", x=lon, y=lat)
            node_score = model.score_batch([sample])[0]
            if node_score is None:
                raise ValueError(
                    "The supervisor model has insufficient local GSWA sediment coverage at this location. "
                    "Choose a nearby blue survey area or draw a wider region."
                )
            return self._result_payload(key, lon, lat, node_score, mode="research_point")

    def _result_payload(
        self,
        target: str,
        lon: float,
        lat: float,
        node_score: Any,
        *,
        mode: str,
    ) -> dict[str, Any]:
        narrative = self._symbols["describe"](node_score, target=target, x=lon, y=lat)
        data = narrative.to_dict()
        level = {"Strong": "HIGH", "Moderate": "MODERATE", "Weak": "MODERATE"}.get(data["tier"], "LOW")
        signals = [
            {
                "feature": signal["feature"],
                "label": signal["feature"],
                "zScore": float(signal["z_score"]),
                "weight": self._feature_weight(node_score, signal["feature"], signal["source"]),
                "source": signal["feature"].split(":", 1)[0] if ":" in signal["feature"] else signal["source"],
                "direction": "supporting" if signal["z_score"] > 0 else "opposing",
            }
            for signal in data["top_signals"]
        ]
        expert_breakdown = [
            {
                "name": expert["name"],
                "score": float(expert["score"]),
                "weight": float(expert["tree_weight"]),
            }
            for expert in data["expert_contribs"]
        ]
        interpretation = narrative.to_text().splitlines()[-1]
        details = {
            signal["feature"]: {
                "value": signal["zScore"],
                "threshold": 0.0,
                "contrib": signal["zScore"] * signal["weight"],
                "anomalous": signal["zScore"] > 0,
                "source": signal["source"],
            }
            for signal in signals
        }
        nearest = self._nearest_site(lon, lat)
        zone = None
        if nearest:
            zone = "near_deposit" if nearest["distanceKm"] <= 5 else "background" if nearest["distanceKm"] > 50 else "buffer"

        return {
            "mode": mode,
            "researchModel": True,
            "target": target,
            "targetLabel": TARGETS[target]["label"],
            "point": {"lat": lat, "lon": lon},
            "nearestSampleKm": None,
            "nearestSite": nearest,
            "zone": zone,
            "zoneLabel": "Research grid point" if not zone else zone.replace("_", " ").title(),
            "interpolatedValues": {},
            "featureSignals": signals,
            "evidenceLayers": self._evidence_layers(signals, expert_breakdown),
            "anomaly": {"score": float(node_score.score), "details": details},
            "prospectivity": {
                "rows": [
                    {
                        "element": signal["feature"],
                        "sampleValue": signal["zScore"],
                        "threshold": 0.0,
                        "anomalous": signal["zScore"] > 0,
                        "weight": signal["weight"],
                    }
                    for signal in signals
                ],
                "scorePct": float(node_score.score * 100.0),
                "level": level,
            },
            "evidence": "; ".join(
                f"{signal['label']} z={signal['zScore']:+.2f} ({signal['source']})" for signal in signals[:5]
            ),
            "interpretation": interpretation,
            "explanation": {
                "tier": data["tier"],
                "summary": (
                    f"{TARGETS[target]['label']} score {node_score.score:.3f} ({data['tier']}). "
                    f"{data['tier_reason']} {interpretation}"
                ),
                "topSignals": signals,
                "expertBreakdown": expert_breakdown,
                "audit": {
                    "target": target,
                    "activeSources": self._model.active_sources(),
                    "activeExperts": self._model.active_experts(),
                    "method": "Supervisor ProspectivityModel / NodeScore",
                },
            },
            "modelMetadata": {
                "source": "gad_reasoning_full_20260610",
                "spatialAuc": TARGETS[target]["spatial_auc"],
                "validation": "spatial holdout",
            },
            "sampleRecord": None,
            "inputDistanceKm": None,
        }

    def _regional_feature_signals(self, scored: list[tuple[tuple[float, float], Any]]) -> list[dict[str, Any]]:
        total_points = len(scored)
        grouped: dict[str, list[dict[str, Any]]] = {}

        for _, node_score in scored:
            point_signals: dict[str, dict[str, Any]] = {}

            def collect(node: Any) -> None:
                for feature, z_score in node.feature_z.items():
                    if ":" in feature:
                        source, bare_feature = feature.split(":", 1)
                    else:
                        source = self._symbols["expert_source"](node.name)
                        bare_feature = feature
                    weight = float(
                        node.weights.get(
                            feature,
                            node.weights.get(bare_feature, node.weights.get(source, 0.0)),
                        )
                    )
                    candidate = {
                        "feature": feature,
                        "zScore": float(z_score),
                        "weight": weight,
                        "source": source,
                    }
                    current = point_signals.get(feature)
                    if current is None or abs(candidate["zScore"] * weight) > abs(current["zScore"] * current["weight"]):
                        point_signals[feature] = candidate
                for child in node.children.values():
                    collect(child)

            collect(node_score)
            for feature, signal in point_signals.items():
                grouped.setdefault(feature, []).append(signal)

        aggregated: list[dict[str, Any]] = []
        for feature, rows in grouped.items():
            z_scores = np.array([row["zScore"] for row in rows], dtype=float)
            weights = np.array([row["weight"] for row in rows], dtype=float)
            source_counts: dict[str, int] = {}
            for row in rows:
                source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            source = max(source_counts, key=source_counts.get)
            support_count = int(np.count_nonzero(z_scores > 0))
            oppose_count = int(np.count_nonzero(z_scores < 0))
            median_z = float(np.median(z_scores))
            median_weight = float(np.median(weights))
            coverage_pct = len(rows) / total_points * 100.0
            support_pct = support_count / total_points * 100.0
            oppose_pct = oppose_count / total_points * 100.0
            aggregated.append(
                {
                    "feature": feature,
                    "label": feature,
                    "zScore": median_z,
                    "zMin": float(z_scores.min()),
                    "zMax": float(z_scores.max()),
                    "weight": median_weight,
                    "source": source,
                    "direction": "supporting" if median_z > 0 else "opposing",
                    "supportCount": support_count,
                    "opposeCount": oppose_count,
                    "observedCount": len(rows),
                    "supportPct": support_pct,
                    "opposePct": oppose_pct,
                    "coveragePct": coverage_pct,
                    "_impact": abs(median_z * median_weight) * coverage_pct / 100.0,
                }
            )

        aggregated.sort(key=lambda signal: signal["_impact"], reverse=True)
        for signal in aggregated:
            signal.pop("_impact", None)
        return aggregated[:10]

    @staticmethod
    def _regional_expert_breakdown(scored: list[tuple[tuple[float, float], Any]]) -> list[dict[str, Any]]:
        total_points = len(scored)
        grouped: dict[str, list[tuple[float, float]]] = {}
        for _, node_score in scored:
            for name, child in node_score.children.items():
                grouped.setdefault(name, []).append(
                    (float(child.score), float(node_score.weights.get(name, 0.0)))
                )

        experts = []
        for name, rows in grouped.items():
            scores = np.array([row[0] for row in rows], dtype=float)
            weights = np.array([row[1] for row in rows], dtype=float)
            experts.append(
                {
                    "name": name,
                    "score": float(np.median(scores)),
                    "scoreMin": float(scores.min()),
                    "scoreMax": float(scores.max()),
                    "weight": float(np.median(weights)),
                    "observedCount": len(rows),
                    "coveragePct": len(rows) / total_points * 100.0,
                }
            )
        experts.sort(key=lambda expert: expert["weight"] * abs(expert["score"] - 0.5), reverse=True)
        return experts

    def score_region(self, session: ResearchSession, payload: dict[str, Any], target: str | None = None) -> dict[str, Any]:
        key = _target_key(target or session.target)
        geometry_type = str(payload.get("type", "polygon"))
        west, east, south, north = self._geometry_bounds(payload)
        if not _inside_wa(west, south) or not _inside_wa(east, north):
            raise ValueError("The selected region must be fully inside the Western Australia model extent.")

        xs = np.linspace(west, east, 8)
        ys = np.linspace(south, north, 8)
        points = [(float(x), float(y)) for y in ys for x in xs if self._geometry_contains(payload, float(x), float(y))]
        if not points:
            raise ValueError("The selected region does not contain any scoring grid points.")

        with self._lock:
            model = self.ensure_model(key)
            session.target = key
            samples = [
                self._symbols["GeochemSample"](site_code=f"region_{index}_{uuid4().hex}", x=lon, y=lat)
                for index, (lon, lat) in enumerate(points)
            ]
            node_scores = model.score_batch(samples)
            scored = [
                (point, node_score)
                for point, node_score in zip(points, node_scores)
                if node_score is not None
            ]
            if not scored:
                raise ValueError(
                    "The selected region has insufficient local GSWA sediment coverage. "
                    "Choose an area containing blue survey samples."
                )

            scores = np.array([node_score.score for _, node_score in scored], dtype=float)
            median_score = float(np.median(scores))
            representative_index = int(np.argmin(np.abs(scores - median_score)))
            (rep_lon, rep_lat), representative = scored[representative_index]
            result = self._result_payload(key, rep_lon, rep_lat, representative, mode="research_region")
            centroid_lon = float(np.mean([point[0] for point, _ in scored]))
            centroid_lat = float(np.mean([point[1] for point, _ in scored]))
            tier, level = _score_tier(median_score)
            tier_counts = {tier_name: 0 for _, tier_name, _ in SCORE_TIERS}
            for score in scores:
                score_tier, _ = _score_tier(float(score))
                tier_counts[score_tier] += 1
            tier_distribution = [
                {
                    "tier": tier_name,
                    "count": tier_counts[tier_name],
                    "pct": tier_counts[tier_name] / len(scored) * 100.0,
                }
                for _, tier_name, _ in SCORE_TIERS
            ]
            signals = self._regional_feature_signals(scored)
            expert_breakdown = self._regional_expert_breakdown(scored)
            coverage_pct = len(scored) / len(points) * 100.0
            dominant_tiers = ", ".join(
                f"{item['tier']} {item['pct']:.0f}%"
                for item in tier_distribution
                if item["count"]
            )
            leading_signals = ", ".join(
                f"{signal['label']} ({signal['supportPct']:.0f}% support)"
                for signal in signals[:3]
            )
            interpretation = (
                f"{TARGETS[key]['label']} regional aggregate uses {len(scored)} of {len(points)} candidate grid points "
                f"({coverage_pct:.0f}% model coverage). Tier distribution: {dominant_tiers}."
            )
            if leading_signals:
                interpretation += f" Dominant regional signals: {leading_signals}."
            details = {
                signal["feature"]: {
                    "value": signal["zScore"],
                    "threshold": 0.0,
                    "contrib": signal["zScore"] * signal["weight"],
                    "anomalous": signal["zScore"] > 0.1 and signal["supportPct"] >= 50.0,
                    "suppressed": signal["zScore"] < -0.1 and signal["opposePct"] >= 50.0,
                    "source": signal["source"],
                    "supportPct": signal["supportPct"],
                    "coveragePct": signal["coveragePct"],
                }
                for signal in signals
            }
            result["mode"] = "region"
            result["prospectivity"]["scorePct"] = median_score * 100.0
            result["prospectivity"]["level"] = level
            result["prospectivity"]["rows"] = [
                {
                    "element": signal["feature"],
                    "sampleValue": signal["zScore"],
                    "threshold": 0.0,
                    "anomalous": signal["zScore"] > 0.1 and signal["supportPct"] >= 50.0,
                    "weight": signal["weight"],
                    "supportPct": signal["supportPct"],
                    "coveragePct": signal["coveragePct"],
                }
                for signal in signals
            ]
            result["anomaly"]["score"] = median_score
            result["anomaly"]["details"] = details
            result["point"] = {"lat": centroid_lat, "lon": centroid_lon}
            result["featureSignals"] = signals
            result["evidenceLayers"] = self._evidence_layers(signals, expert_breakdown)
            result["sampleCount"] = len(scored)
            result["region"] = {
                "type": geometry_type,
                "sampleCount": len(scored),
                "gridPointCount": len(scored),
                "candidateGridCount": len(points),
                "validGridCount": len(scored),
                "coveragePct": coverage_pct,
                "centroid": {"lon": centroid_lon, "lat": centroid_lat},
                "bounds": {"west": west, "east": east, "south": south, "north": north},
                "scoreMin": float(scores.min() * 100.0),
                "scoreMax": float(scores.max() * 100.0),
                "scoreMedian": float(median_score * 100.0),
                "representativePoint": {
                    "lon": rep_lon,
                    "lat": rep_lat,
                    "scorePct": float(representative.score * 100.0),
                    "purpose": "Audit reference only; not used for regional evidence.",
                },
                "tierDistribution": tier_distribution,
            }
            result["topSamples"] = [
                {"lon": lon, "lat": lat, "scorePct": float(node_score.score * 100.0)}
                for (lon, lat), node_score in sorted(scored, key=lambda item: item[1].score, reverse=True)[:8]
            ]
            result["explanation"]["summary"] = (
                f"{TARGETS[key]['label']} regional median score is {median_score:.3f} across "
                f"{len(scored)} valid model grid points ({coverage_pct:.0f}% coverage). "
                f"Scores range from {scores.min():.3f} to {scores.max():.3f}; evidence is aggregated across the region."
            )
            result["explanation"]["tier"] = tier
            result["explanation"]["topSignals"] = signals
            result["explanation"]["expertBreakdown"] = expert_breakdown
            result["explanation"]["audit"]["method"] = (
                "Supervisor ProspectivityModel per grid point; regional median score and aggregated feature/expert evidence"
            )
            result["evidence"] = "; ".join(
                f"{signal['label']} median z={signal['zScore']:+.2f}, "
                f"{signal['supportPct']:.0f}% support ({signal['source']})"
                for signal in signals[:5]
            )
            result["interpretation"] = interpretation
            result["nearestSite"] = None
            result["zone"] = None
            result["zoneLabel"] = "Selected research region"
            return result

    @staticmethod
    def _geometry_bounds(payload: dict[str, Any]) -> tuple[float, float, float, float]:
        if payload.get("type") == "circle":
            center = payload["center"]
            radius = float(payload.get("radiusKm", 0))
            lat_delta = radius / 111.0
            lon_delta = radius / (111.0 * np.cos(np.radians(abs(float(center["lat"])))))
            return (
                float(center["lon"]) - lon_delta,
                float(center["lon"]) + lon_delta,
                float(center["lat"]) - lat_delta,
                float(center["lat"]) + lat_delta,
            )
        points = payload.get("points") or []
        if len(points) < 3:
            raise ValueError("Region polygon needs at least three points.")
        lons = [float(point["lon"]) for point in points]
        lats = [float(point["lat"]) for point in points]
        return min(lons), max(lons), min(lats), max(lats)

    @staticmethod
    def _geometry_contains(payload: dict[str, Any], lon: float, lat: float) -> bool:
        geometry_type = payload.get("type")
        if geometry_type == "circle":
            center = payload["center"]
            point = np.array([[lon, lat]], dtype=float)
            return bool(_distance_km(float(center["lon"]), float(center["lat"]), point)[0] <= float(payload["radiusKm"]))
        points = payload.get("points") or []
        if geometry_type == "bbox":
            lons = [float(point["lon"]) for point in points]
            lats = [float(point["lat"]) for point in points]
            return min(lons) <= lon <= max(lons) and min(lats) <= lat <= max(lats)
        inside = False
        j = len(points) - 1
        for i, point in enumerate(points):
            xi, yi = float(point["lon"]), float(point["lat"])
            xj, yj = float(points[j]["lon"]), float(points[j]["lat"])
            if ((yi > lat) != (yj > lat)) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
            j = i
        return inside
