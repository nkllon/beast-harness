"""SQLite durable session and run ledger for beast-harness."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from beast_harness.types import (
    DriverRunResult,
    ExecutionPosture,
    HarnessError,
    PostureViolationError,
    RunState,
    SessionRecord,
    VerifiedRepoRef,
    ensure_directory,
)


def _compute_audit_digest(audit_events: list[dict[str, Any]], state: RunState) -> str | None:
    if not audit_events and not state.is_terminal:
        return None
    canonical = json.dumps(audit_events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class SessionLedger:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        ensure_directory(self.db_path.parent)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_sessions (
                    session_id TEXT PRIMARY KEY,
                    endpoint_id TEXT NOT NULL,
                    driver_id TEXT NOT NULL,
                    native_agent_id TEXT NOT NULL,
                    latest_run_id TEXT NOT NULL,
                    repo_url TEXT,
                    commit_sha TEXT,
                    base_sha TEXT,
                    worktree_path TEXT,
                    posture TEXT NOT NULL,
                    independent_review INTEGER NOT NULL,
                    facets_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    structured_output_json TEXT,
                    audit_events_json TEXT NOT NULL,
                    audit_digest TEXT,
                    error_detail TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> SessionRecord:
        structured_raw = row["structured_output_json"]
        return SessionRecord(
            session_id=row["session_id"],
            endpoint_id=row["endpoint_id"],
            driver_id=row["driver_id"],
            native_agent_id=row["native_agent_id"],
            latest_run_id=row["latest_run_id"],
            repo_url=row["repo_url"],
            commit_sha=row["commit_sha"],
            base_sha=row["base_sha"],
            worktree_path=row["worktree_path"],
            posture=ExecutionPosture(row["posture"]),
            independent_review=bool(row["independent_review"]),
            facets=json.loads(row["facets_json"]),
            status=RunState(row["status"]),
            answer=row["answer"],
            structured_output=json.loads(structured_raw) if structured_raw is not None else None,
            audit_events=json.loads(row["audit_events_json"]),
            audit_digest=row["audit_digest"],
            error_detail=row["error_detail"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def create_session(
        self,
        *,
        session_id: str,
        endpoint_id: str,
        driver_id: str,
        repo_ref: VerifiedRepoRef | None,
        posture: ExecutionPosture,
        independent_review: bool,
        facets: dict[str, Any],
        initial_run: DriverRunResult,
    ) -> SessionRecord:
        now = datetime.now(UTC)
        audit_digest = _compute_audit_digest(initial_run.audit_events, initial_run.state)
        record = SessionRecord(
            session_id=session_id,
            endpoint_id=endpoint_id,
            driver_id=driver_id,
            native_agent_id=initial_run.native_agent_id,
            latest_run_id=initial_run.run_id,
            repo_url=repo_ref.repo_url if repo_ref else None,
            commit_sha=repo_ref.commit_sha if repo_ref else None,
            base_sha=repo_ref.base_sha if repo_ref else None,
            worktree_path=repo_ref.local_worktree_path if repo_ref else None,
            posture=posture,
            independent_review=independent_review,
            facets=dict(facets),
            status=initial_run.state,
            answer=initial_run.answer,
            structured_output=initial_run.structured_output,
            audit_events=list(initial_run.audit_events),
            audit_digest=audit_digest,
            error_detail=initial_run.error_detail,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO harness_sessions (
                    session_id, endpoint_id, driver_id, native_agent_id, latest_run_id,
                    repo_url, commit_sha, base_sha, worktree_path, posture,
                    independent_review, facets_json, status, answer,
                    structured_output_json, audit_events_json, audit_digest,
                    error_detail, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.endpoint_id,
                    record.driver_id,
                    record.native_agent_id,
                    record.latest_run_id,
                    record.repo_url,
                    record.commit_sha,
                    record.base_sha,
                    record.worktree_path,
                    record.posture.value,
                    1 if record.independent_review else 0,
                    json.dumps(record.facets, sort_keys=True),
                    record.status.value,
                    record.answer,
                    json.dumps(record.structured_output, sort_keys=True)
                    if record.structured_output is not None
                    else None,
                    json.dumps(record.audit_events, sort_keys=True),
                    record.audit_digest,
                    record.error_detail,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM harness_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise HarnessError(f"Unknown session_id: {session_id!r}")
        return self._row_to_record(row)

    def update_from_driver_result(
        self,
        session_id: str,
        result: DriverRunResult,
        *,
        clear_worktree: bool = False,
    ) -> SessionRecord:
        existing = self.get_session(session_id)
        merged_events = (
            list(result.audit_events) if result.audit_events else list(existing.audit_events)
        )
        answer = result.answer if result.answer is not None else existing.answer
        structured = (
            result.structured_output
            if result.structured_output is not None
            else existing.structured_output
        )
        worktree_path = None if clear_worktree else existing.worktree_path
        audit_digest = _compute_audit_digest(merged_events, result.state) or existing.audit_digest
        now = datetime.now(UTC)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE harness_sessions
                SET native_agent_id = ?,
                    latest_run_id = ?,
                    worktree_path = ?,
                    status = ?,
                    answer = ?,
                    structured_output_json = ?,
                    audit_events_json = ?,
                    audit_digest = ?,
                    error_detail = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    result.native_agent_id or existing.native_agent_id,
                    result.run_id or existing.latest_run_id,
                    worktree_path,
                    result.state.value,
                    answer,
                    json.dumps(structured, sort_keys=True) if structured is not None else None,
                    json.dumps(merged_events, sort_keys=True),
                    audit_digest,
                    result.error_detail,
                    now.isoformat(),
                    session_id,
                ),
            )
            conn.commit()
        return self.get_session(session_id)

    def assert_followup_allowed(self, session_id: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session.independent_review:
            raise PostureViolationError(
                f"Session {session_id!r} was created with independent_review=True; "
                "follow-up messages are forbidden to preserve review independence."
            )
        return session

    def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM harness_sessions ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]
