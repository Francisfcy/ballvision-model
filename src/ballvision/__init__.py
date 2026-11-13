"""Baseline components for basketball video analysis."""

from .events import build_box_score
from .identity import IdentityResolver
from .pipeline import BaselinePipeline
from .schemas import Detection, Event, PlayerProfile, TrackObservation

__all__ = [
    "Detection",
    "Event",
    "BaselinePipeline",
    "IdentityResolver",
    "PlayerProfile",
    "TrackObservation",
    "build_box_score",
]
