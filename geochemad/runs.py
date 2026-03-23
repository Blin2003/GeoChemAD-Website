from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .catalog import DatasetRecord
from .training import ExperimentConfig, save_run, train_experiment


@dataclass
class RunState:
    run_id: str
    status: str
    created_at: float
    dataset_id: str
    model_name: str
    metrics: dict[str, Any] | None = None
    error: str | None = None
    can_cancel: bool = True


class RunRegistry:
    def __init__(self, run_root: Path, dataset_index: dict[str, DatasetRecord]):
        self.run_root = run_root
        self.dataset_index = dataset_index
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._states: dict[str, RunState] = {}
        self._cancel_flags: dict[str, threading.Event] = {}

    def list_runs(self) -> list[RunState]:
        with self._lock:
            return sorted(self._states.values(), key=lambda item: item.created_at, reverse=True)

    def get_run(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._states.get(run_id)

    def _persist_state(self, state: RunState) -> None:
        path = self.run_root / state.run_id / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    def start(self, config: ExperimentConfig) -> RunState:
        if config.dataset_id not in self.dataset_index:
            raise KeyError(f"Unknown dataset: {config.dataset_id}")
        run_id = uuid.uuid4().hex[:10]
        state = RunState(
            run_id=run_id,
            status="queued",
            created_at=time.time(),
            dataset_id=config.dataset_id,
            model_name=config.model_name,
        )
        with self._lock:
            self._states[run_id] = state
            self._cancel_flags[run_id] = threading.Event()
        self._persist_state(state)
        thread = threading.Thread(target=self._worker, args=(run_id, config), daemon=True)
        thread.start()
        return state

    def start_with_record(self, record: DatasetRecord, config: ExperimentConfig) -> RunState:
        self.dataset_index[record.subset_id] = record
        return self.start(config)

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            state = self._states[run_id]
            for key, value in changes.items():
                setattr(state, key, value)
        self._persist_state(self._states[run_id])

    def _worker(self, run_id: str, config: ExperimentConfig) -> None:
        self._update(run_id, status="running")
        try:
            record = self.dataset_index[config.dataset_id]
            stop_flag = self._cancel_flags[run_id]
            _, metrics, payload = train_experiment(record, config, should_stop=stop_flag.is_set)
            run_dir = self.run_root / run_id
            save_run(run_dir, config, metrics, payload)
            self._update(run_id, status="completed", metrics=metrics)
        except Exception as exc:
            status = "cancelled" if "cancelled by user" in str(exc).lower() else "failed"
            self._update(run_id, status=status, error=str(exc))

    def cancel(self, run_id: str) -> RunState | None:
        with self._lock:
            state = self._states.get(run_id)
            flag = self._cancel_flags.get(run_id)
            if state is None or flag is None:
                return None
            if state.status in {"completed", "failed", "cancelled"}:
                return state
            flag.set()
            state.status = "cancelling"
        self._persist_state(self._states[run_id])
        return self._states[run_id]
