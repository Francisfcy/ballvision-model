from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from .events import build_box_score
from .score_validation import validate_score_changes
from .schemas import Event, EventUpdate, ReviewSession, ScoreObservation, ScoreObservationCreate


class ReviewStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.environ.get("BALLVISION_DB_PATH", "data/ballvision.sqlite3"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES review_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS score_observations (
                    observation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    observation_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES review_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS review_audit (
                    audit_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    event_id TEXT,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create(self, events: list[Event]) -> ReviewSession:
        session_id = str(uuid4())
        with self._connect() as connection:
            connection.execute("INSERT INTO review_sessions(session_id, created_at) VALUES (?, datetime('now'))", (session_id,))
            for event in events:
                connection.execute(
                    "INSERT INTO review_events(event_id, session_id, event_json) VALUES (?, ?, ?)",
                    (event.event_id, session_id, json.dumps(event.model_dump(), ensure_ascii=False)),
                )
        return ReviewSession(session_id=session_id, events=events)

    def get(self, session_id: str) -> ReviewSession:
        with self._connect() as connection:
            rows = connection.execute("SELECT event_json FROM review_events WHERE session_id = ? ORDER BY json_extract(event_json, '$.timestamp_s')", (session_id,)).fetchall()
            exists = connection.execute("SELECT 1 FROM review_sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not exists:
            raise KeyError(session_id)
        return ReviewSession(session_id=session_id, events=[Event.model_validate(json.loads(row["event_json"])) for row in rows])

    def update_event(self, session_id: str, event_id: str, update: EventUpdate) -> ReviewSession:
        session = self.get(session_id)
        for event in session.events:
            if event.event_id == event_id:
                updated = event.model_copy(update=update.model_dump(exclude_none=True))
                with self._connect() as connection:
                    connection.execute("UPDATE review_events SET event_json = ? WHERE event_id = ? AND session_id = ?", (json.dumps(updated.model_dump(), ensure_ascii=False), event_id, session_id))
                    connection.execute(
                        "INSERT INTO review_audit(audit_id, session_id, event_id, action, payload_json, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                        (str(uuid4()), session_id, event_id, "update_event", json.dumps(update.model_dump(exclude_none=True), ensure_ascii=False)),
                    )
                return self.get(session_id)
        raise KeyError(event_id)

    def add_score_observation(self, session_id: str, payload: ScoreObservationCreate) -> ScoreObservation:
        self.get(session_id)
        observation = ScoreObservation(observation_id=str(uuid4()), **payload.model_dump())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO score_observations(observation_id, session_id, observation_json) VALUES (?, ?, ?)",
                (observation.observation_id, session_id, json.dumps(observation.model_dump(), ensure_ascii=False)),
            )
        return observation

    def get_score_observations(self, session_id: str) -> list[ScoreObservation]:
        self.get(session_id)
        with self._connect() as connection:
            rows = connection.execute("SELECT observation_json FROM score_observations WHERE session_id = ? ORDER BY json_extract(observation_json, '$.timestamp_s')", (session_id,)).fetchall()
        return [ScoreObservation.model_validate(json.loads(row["observation_json"])) for row in rows]

    def recalculate(self, session_id: str) -> dict[str, object]:
        session = self.get(session_id)
        observations = self.get_score_observations(session_id)
        return {
            "session_id": session_id,
            "events": session.events,
            "box_score": build_box_score(session.events),
            "score_validation": validate_score_changes(session.events, observations),
        }
