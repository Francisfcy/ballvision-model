from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PlayerProfile(BaseModel):
    player_id: str
    name: str
    team_id: str
    jersey_number: str | None = None


class Detection(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    track_id: str
    object_type: Literal["player", "ball", "rim"]
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0, le=1)
    team_hint: str | None = None
    appearance_embedding: list[float] | None = None
    jersey_number: str | None = None


class TrackObservation(BaseModel):
    track_id: str
    timestamp_s: float = Field(ge=0)
    team_hint: str | None = None
    appearance_embedding: list[float] | None = None
    jersey_number: str | None = None


class IdentityCandidate(BaseModel):
    player_id: str
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class TrackIdentity(BaseModel):
    track_id: str
    candidates: list[IdentityCandidate]
    status: Literal["unresolved", "candidate", "confirmed"] = "unresolved"
    confirmed_player_id: str | None = None


class Event(BaseModel):
    event_id: str
    event_type: Literal["shot_attempt", "shot_made", "shot_missed", "rebound", "shot_candidate", "rebound_candidate"]
    timestamp_s: float = Field(ge=0)
    track_id: str | None = None
    player_id: str | None = None
    team_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    confirmed: bool = False
    outcome: Literal["unknown", "made", "missed"] = "unknown"
    review_status: Literal["pending", "confirmed", "rejected"] = "pending"
    evidence_start_s: float | None = None
    evidence_end_s: float | None = None
    evidence: dict[str, str | int | float | None] = Field(default_factory=dict)


class AnalysisJob(BaseModel):
    job_id: str
    video_name: str
    status: Literal["queued", "running", "review_required", "completed", "failed"]
    created_at: datetime
    message: str | None = None


class EventUpdate(BaseModel):
    event_type: Literal["shot_attempt", "shot_made", "shot_missed", "rebound", "shot_candidate", "rebound_candidate"] | None = None
    player_id: str | None = None
    team_id: str | None = None
    outcome: Literal["unknown", "made", "missed"] | None = None
    review_status: Literal["pending", "confirmed", "rejected"] | None = None
    confirmed: bool | None = None


class ReviewSessionCreate(BaseModel):
    events: list[Event] = Field(default_factory=list)


class ReviewSession(BaseModel):
    session_id: str
    events: list[Event]


class ScoreObservation(BaseModel):
    observation_id: str
    timestamp_s: float = Field(ge=0)
    team_scores: dict[str, int] = Field(default_factory=dict)
    source: Literal["manual", "ocr", "imported"] = "manual"


class ScoreObservationCreate(BaseModel):
    timestamp_s: float = Field(ge=0)
    team_scores: dict[str, int] = Field(default_factory=dict)
    source: Literal["manual", "ocr", "imported"] = "manual"
