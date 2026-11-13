from __future__ import annotations

from dataclasses import dataclass

from .events import build_box_score
from .identity import IdentityResolver
from .schemas import Event, PlayerProfile, TrackIdentity, TrackObservation


@dataclass
class BaselineAnalysis:
    identities: list[TrackIdentity]
    events: list[Event]

    @property
    def box_score(self) -> dict[str, dict[str, int]]:
        return build_box_score(self.events)


class BaselinePipeline:
    """Model-agnostic orchestration layer.

    Detector, tracker and action models can be replaced independently. This first
    version consumes their structured outputs so identity and statistics can be
    validated before expensive model training begins.
    """

    def __init__(self, roster: list[PlayerProfile]) -> None:
        self.identity_resolver = IdentityResolver(roster)

    def analyze(
        self,
        observations: list[TrackObservation],
        events: list[Event],
        *,
        manual_bindings: dict[str, str] | None = None,
    ) -> BaselineAnalysis:
        identities = self.identity_resolver.resolve(observations, manual_bindings=manual_bindings)
        identity_map = {
            item.track_id: item.confirmed_player_id
            for item in identities
            if item.confirmed_player_id
        }
        linked_events = [event.model_copy(update={"player_id": identity_map.get(event.track_id, event.player_id)}) for event in events]
        return BaselineAnalysis(identities=identities, events=linked_events)
