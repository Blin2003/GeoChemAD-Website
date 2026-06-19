from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .research_backend import ResearchModelService, data_profile, demo_data_root, research_root
from .settings import ProjectPaths


LOCAL_AU_PLACES: list[dict[str, Any]] = [
    {"name": "Claremont, Western Australia, Australia", "lat": -31.9813, "lon": 115.7799, "bbox": [-32.005, -31.957, 115.750, 115.807], "type": "suburb"},
    {"name": "Nedlands, Western Australia, Australia", "lat": -31.9802, "lon": 115.8072, "bbox": [-32.011, -31.955, 115.780, 115.836], "type": "suburb"},
    {"name": "Crawley, Western Australia, Australia", "lat": -31.9845, "lon": 115.8171, "bbox": [-31.997, -31.970, 115.803, 115.830], "type": "suburb"},
    {"name": "Perth, Western Australia, Australia", "lat": -31.9523, "lon": 115.8613, "bbox": [-32.08, -31.86, 115.74, 116.02], "type": "city"},
    {"name": "Kalgoorlie, Western Australia, Australia", "lat": -30.7489, "lon": 121.4658, "bbox": [-30.86, -30.65, 121.34, 121.58], "type": "city"},
    {"name": "Geraldton, Western Australia, Australia", "lat": -28.7774, "lon": 114.6149, "bbox": [-28.88, -28.68, 114.52, 114.72], "type": "city"},
    {"name": "Port Hedland, Western Australia, Australia", "lat": -20.3107, "lon": 118.6011, "bbox": [-20.43, -20.20, 118.48, 118.72], "type": "town"},
]


def create_app(paths: ProjectPaths | None = None) -> FastAPI:
    paths = paths or ProjectPaths()
    paths.ensure()

    app = FastAPI(title="GeoChemAD Supervisor Model UI", version="3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    full_data_root = research_root() / "datasets"
    demo_root = demo_data_root()
    app.state.research_sessions = {}
    app.state.research_services = {
        "default": ResearchModelService(),
        "demo": ResearchModelService(
            data_root_override=demo_root,
            profile_name="Demo subset",
            profile_mode="demo",
        ),
        "full": ResearchModelService(
            data_root_override=full_data_root,
            profile_name="Full teacher datasets",
            profile_mode="teacher-full",
        ),
    }

    def profile_options() -> list[dict[str, Any]]:
        return [
            {
                "key": "demo",
                "label": "Demo subset",
                "description": "Fast local subset cut from the teacher data.",
                "path": str(demo_root),
                "available": demo_root.exists(),
            },
            {
                "key": "full",
                "label": "Full teacher data",
                "description": "Complete local gad_reasoning_full_20260610 datasets.",
                "path": str(full_data_root),
                "available": full_data_root.exists(),
            },
        ]

    def research_service(profile: str | None = None) -> ResearchModelService:
        key = profile or "default"
        if key not in app.state.research_services:
            raise HTTPException(status_code=400, detail=f"Unknown data profile: {key}")
        service = app.state.research_services[key]
        root = Path(service._data_profile()["path"])
        if not root.exists():
            raise HTTPException(status_code=400, detail=f"Data profile {key!r} is not available at {root}")
        return service

    def session_service(session: Any) -> ResearchModelService:
        return research_service(getattr(session, "profile_key", "default"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "gad_reasoning_full_20260610",
            "role": "frontend-adapter",
            "dataProfile": data_profile(),
            "profiles": profile_options(),
        }

    @app.get("/api/research/targets")
    def research_targets(profile: str = "demo") -> dict[str, Any]:
        service = research_service(profile)
        return {
            "targets": service.target_options(),
            "coverage": {"west": 114.5, "east": 128.5, "south": -33.5, "north": -15.0},
            "source": "gad_reasoning_full_20260610",
            "dataProfile": service._data_profile(),
            "profiles": profile_options(),
            "coverageRegions": service.coverage_regions(),
            "layers": ["geochem", "geophys_available", "structure_available"],
        }

    @app.get("/api/research/layers")
    def research_layers(profile: str = "demo") -> dict[str, Any]:
        return research_service(profile).layer_inventory()

    @app.post("/api/research/start")
    def research_start(target: str = "Cu", profile: str = "demo") -> dict[str, Any]:
        try:
            service = research_service(profile)
            session = service.create_session(target)
            session.profile_key = profile
            app.state.research_sessions[session.session_id] = session
            return service.session_payload(session)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/research/{session_id}")
    def research_session(session_id: str, target: Optional[str] = None) -> dict[str, Any]:
        session = app.state.research_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Research session not found.")
        try:
            return session_service(session).session_payload(session, target)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/research/{session_id}/point")
    def research_point(session_id: str, lon: float, lat: float, target: Optional[str] = None) -> dict[str, Any]:
        session = app.state.research_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Research session not found.")
        try:
            return session_service(session).score_point(session, lon=lon, lat=lat, target=target)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/research/{session_id}/region")
    async def research_region(
        session_id: str,
        payload: dict[str, Any],
        target: Optional[str] = None,
    ) -> dict[str, Any]:
        session = app.state.research_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Research session not found.")
        try:
            return session_service(session).score_region(session, payload=payload, target=target)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/places/search")
    def place_search(q: str) -> list[dict[str, Any]]:
        query = q.strip()
        if len(query) < 2:
            return []
        local = [place for place in LOCAL_AU_PLACES if query.lower() in place["name"].lower()]
        params = urlencode(
            {
                "q": f"{query}, Australia",
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 8,
                "countrycodes": "au",
            }
        )
        request = Request(
            f"https://nominatim.openstreetmap.org/search?{params}",
            headers={"User-Agent": "GeoChemAD supervisor-model frontend"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return local[:8]
        remote = [
            {
                "name": item.get("display_name", ""),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "bbox": [float(value) for value in item.get("boundingbox", [])],
                "type": item.get("type", ""),
            }
            for item in data
            if item.get("lat") and item.get("lon")
        ]
        seen = {place["name"] for place in local}
        return (local + [place for place in remote if place["name"] not in seen])[:8]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            paths.web_dir / "home.html",
            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
        )

    @app.get("/web/index.html")
    @app.get("/web/run.html")
    @app.get("/web/point-analysis.html")
    def old_page() -> RedirectResponse:
        return RedirectResponse(url="/", status_code=307)

    app.mount("/web", StaticFiles(directory=paths.web_dir), name="web")
    return app
