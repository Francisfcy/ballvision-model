from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI

from .review_store import ReviewStore
from .schemas import AnalysisJob, EventUpdate, ReviewSession, ReviewSessionCreate, ScoreObservation, ScoreObservationCreate, TrackBinding, TrackBindingCreate

app = FastAPI(title="BallVision Model API", version="0.1.0")
review_store = ReviewStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ballvision-model"}


@app.post("/analysis/jobs", response_model=AnalysisJob, status_code=202)
def create_analysis_job(video_name: str) -> AnalysisJob:
    """Create the online job contract used by the future upload service."""
    return AnalysisJob(
        job_id=str(uuid4()),
        video_name=video_name,
        status="queued",
        created_at=datetime.now(timezone.utc),
        message="Upload the video to object storage, then dispatch the detector worker.",
    )


@app.post("/review/sessions", response_model=ReviewSession)
def create_review_session(payload: ReviewSessionCreate) -> ReviewSession:
    return review_store.create(payload.events)


@app.get("/review/sessions/{session_id}", response_model=ReviewSession)
def get_review_session(session_id: str) -> ReviewSession:
    return review_store.get(session_id)


@app.patch("/review/sessions/{session_id}/events/{event_id}", response_model=ReviewSession)
def update_review_event(session_id: str, event_id: str, payload: EventUpdate) -> ReviewSession:
    return review_store.update_event(session_id, event_id, payload)


@app.post("/review/sessions/{session_id}/recalculate")
def recalculate_review_session(session_id: str) -> dict[str, object]:
    return review_store.recalculate(session_id)


@app.post("/review/sessions/{session_id}/score-observations", response_model=ScoreObservation)
def add_score_observation(session_id: str, payload: ScoreObservationCreate) -> ScoreObservation:
    return review_store.add_score_observation(session_id, payload)


@app.post("/review/sessions/{session_id}/validate-score")
def validate_score(session_id: str) -> dict[str, object]:
    return review_store.recalculate(session_id)["score_validation"]


@app.post("/review/sessions/{session_id}/bindings", response_model=TrackBinding)
def bind_track(session_id: str, payload: TrackBindingCreate) -> TrackBinding:
    return review_store.bind_track(session_id, payload)


@app.get("/review/sessions/{session_id}/bindings", response_model=list[TrackBinding])
def get_bindings(session_id: str) -> list[TrackBinding]:
    return review_store.get_bindings(session_id)
