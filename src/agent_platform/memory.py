"""SQLite-backed conversation memory (the orchestrator's MemorySaver).

Stores, per session:

* the live student profile (merged turn by turn),
* the message transcript,
* generated artifacts (plans, documents, advisor reports).

The store is intentionally small and dependency-free (stdlib ``sqlite3``) so
it can be unit-tested and embedded in the FastAPI process.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import settings
from .schemas import ChatMessage, StudentProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS artifacts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_artifacts_session ON artifacts(session_id, kind);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = str(path or settings.sessions_db_path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- sessions / profile ----------------------------------------------
    def load_profile(self, session_id: str) -> StudentProfile:
        with self._lock:
            row = self._conn.execute(
                "SELECT profile_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return StudentProfile(student_id=session_id)
        try:
            return StudentProfile.model_validate_json(row["profile_json"])
        except Exception:  # pragma: no cover - corrupted row safety
            return StudentProfile(student_id=session_id)

    def save_profile(self, session_id: str, profile: StudentProfile) -> None:
        payload = profile.model_dump_json()
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, profile_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, payload, now, now),
            )
            self._conn.commit()

    # -- transcript ------------------------------------------------------
    def append_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, _now()),
            )
            self._conn.commit()

    def history(self, session_id: str, limit: int = 12) -> list[ChatMessage]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, max(1, limit)),
            ).fetchall()
        return [
            ChatMessage(role=row["role"], content=row["content"], created_at=row["created_at"])
            for row in reversed(rows)
        ]

    def message_count(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    # -- artifacts -------------------------------------------------------
    def save_artifact(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts (session_id, kind, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    kind,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    _now(),
                ),
            )
            self._conn.commit()

    def artifacts(
        self, session_id: str, kind: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        query = "SELECT kind, payload_json, created_at FROM artifacts WHERE session_id = ?"
        params: list[Any] = [session_id]
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:  # pragma: no cover - corrupted row safety
                continue
            result.append(
                {"kind": row["kind"], "created_at": row["created_at"], "payload": payload}
            )
        return result

    # -- reset -----------------------------------------------------------
    def reset(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM artifacts WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()

    # -- retention / observability ---------------------------------------
    def purge_older_than(self, days: int) -> dict[str, int]:
        """Delete sessions untouched for ``days`` days and their rows.

        Returns the number of removed rows per table.  ``days <= 0`` is a
        no-op so a misconfigured value can never wipe live data.
        """
        removed = {"sessions": 0, "messages": 0, "artifacts": 0}
        if days <= 0:
            return removed
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        with self._lock:
            stale = [
                row["session_id"]
                for row in self._conn.execute(
                    "SELECT session_id FROM sessions WHERE updated_at < ?", (cutoff,)
                ).fetchall()
            ]
            if stale:
                placeholders = ",".join("?" * len(stale))
                removed["messages"] = self._conn.execute(
                    f"DELETE FROM messages WHERE session_id IN ({placeholders})", stale
                ).rowcount
                removed["artifacts"] = self._conn.execute(
                    f"DELETE FROM artifacts WHERE session_id IN ({placeholders})", stale
                ).rowcount
                removed["sessions"] = self._conn.execute(
                    f"DELETE FROM sessions WHERE session_id IN ({placeholders})", stale
                ).rowcount
            # Rows that never got a session record cannot be aged out by
            # updated_at; fall back to their own creation time.
            removed["messages"] += self._conn.execute(
                "DELETE FROM messages WHERE created_at < ? AND session_id NOT IN "
                "(SELECT session_id FROM sessions)",
                (cutoff,),
            ).rowcount
            removed["artifacts"] += self._conn.execute(
                "DELETE FROM artifacts WHERE created_at < ? AND session_id NOT IN "
                "(SELECT session_id FROM sessions)",
                (cutoff,),
            ).rowcount
            self._conn.commit()
        return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM sessions) AS sessions, "
                "(SELECT COUNT(*) FROM messages) AS messages, "
                "(SELECT COUNT(*) FROM artifacts) AS artifacts"
            ).fetchone()
        result = {
            "sessions": int(row["sessions"]) if row else 0,
            "messages": int(row["messages"]) if row else 0,
            "artifacts": int(row["artifacts"]) if row else 0,
        }
        if self.path != ":memory:":
            try:
                result["bytes"] = Path(self.path).stat().st_size
            except OSError:  # pragma: no cover - file removed under us
                result["bytes"] = 0
        return result
