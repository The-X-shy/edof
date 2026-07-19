"""Small local trace and artifact registry used by the standalone runner."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def make_deterministic_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{stable_hash(parts)[:16]}"


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetaTrace(StrictModel):
    trace_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    branch_id: Optional[str] = None
    step_id: Optional[str] = None
    actor: str
    phase: str
    task: str = Field(min_length=1)
    skill_id: Optional[str] = None
    skill_version: Optional[str] = None
    tool: Optional[str] = None
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_action: Optional[str] = None
    status: str
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    parents: list[str] = Field(default_factory=list)
    content_hash: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(StrictModel):
    artifact_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    uri: str = Field(min_length=1)
    mime: Optional[str] = None
    content_hash: str = Field(min_length=64, max_length=64)
    producer: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SQLiteStore:
    """Two-table SQLite store for traces and registered output files."""

    _tables = ("meta_traces", "artifacts")

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            for table in self._tables:
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ("
                    "id TEXT PRIMARY KEY, workspace_id TEXT, run_id TEXT, "
                    "payload_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT)"
                )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_run ON {table}(run_id)"
                )

    @staticmethod
    def _payload(payload: dict[str, Any] | BaseModel) -> str:
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def get(self, table: str, row_id: str) -> dict[str, Any] | None:
        self.init_db()
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE id = ?", (row_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def insert_once(
        self,
        table: str,
        row_id: str,
        payload: dict[str, Any] | BaseModel,
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        self.init_db()
        now = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    f"INSERT INTO {table} "
                    "(id, workspace_id, run_id, payload_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (row_id, workspace_id, run_id, self._payload(payload), now, now),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def upsert(
        self,
        table: str,
        row_id: str,
        payload: dict[str, Any] | BaseModel,
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.init_db()
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                f"INSERT INTO {table} (id, workspace_id, run_id, payload_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET workspace_id=excluded.workspace_id, "
                "run_id=excluded.run_id, payload_json=excluded.payload_json, updated_at=excluded.updated_at",
                (row_id, workspace_id, run_id, self._payload(payload), now, now),
            )


class MetaTraceWriter:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        self.store.init_db()

    def write_trace(self, trace: MetaTrace) -> MetaTrace:
        payload = trace.model_dump(mode="json")
        payload["content_hash"] = stable_hash({key: value for key, value in payload.items() if key != "content_hash"})
        candidate = MetaTrace(**payload)
        existing_payload = self.store.get("meta_traces", candidate.trace_id)
        if existing_payload:
            existing = MetaTrace(**existing_payload)
            if existing.model_dump(mode="json") != candidate.model_dump(mode="json"):
                raise ValueError(f"trace conflict: {candidate.trace_id}")
            return existing
        self.store.insert_once(
            "meta_traces", candidate.trace_id, candidate,
            workspace_id=candidate.workspace_id, run_id=candidate.run_id,
        )
        return candidate


def make_artifact_id(
    workspace_id: str,
    run_id: str | None,
    trace_id: str | None,
    content_hash: str,
    producer: str | None,
    name: str | None,
) -> str:
    return make_deterministic_id("artifact", workspace_id, run_id, trace_id, content_hash, producer, name)


class FileArtifactStore:
    def __init__(self, root: str | Path, store: SQLiteStore) -> None:
        self.root = Path(root)
        self.store = store
        self.root.mkdir(parents=True, exist_ok=True)
        self.store.init_db()

    def register_file(
        self,
        path: str | Path,
        workspace_id: str,
        run_id: str | None,
        trace_id: str | None,
        producer: str | None,
        metadata: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
    ) -> ArtifactRef:
        source = Path(path)
        content_hash = compute_file_sha256(source)
        artifact_id = make_artifact_id(workspace_id, run_id, trace_id, content_hash, producer, source.name)
        destination = self.root / workspace_id / (run_id or "unscoped") / f"{artifact_id}{source.suffix or '.bin'}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or compute_file_sha256(destination) != content_hash:
            shutil.copy2(source, destination)
        reference = ArtifactRef(
            artifact_id=artifact_id,
            workspace_id=workspace_id,
            run_id=run_id,
            trace_id=trace_id,
            uri=destination.relative_to(self.root.parent).as_posix(),
            mime=mimetypes.guess_type(source.name)[0],
            content_hash=content_hash,
            producer=producer,
            metadata={**(metadata or {}), "filename": source.name},
            metrics=metrics or {},
        )
        self.store.upsert("artifacts", reference.artifact_id, reference, workspace_id, run_id)
        return reference
