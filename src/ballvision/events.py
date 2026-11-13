from __future__ import annotations

from collections import defaultdict

from .schemas import Event


def build_box_score(events: list[Event]) -> dict[str, dict[str, int]]:
    """Aggregate confirmed events into a small, explainable box score."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"points": 0, "shot_attempts": 0, "made_shots": 0, "missed_shots": 0, "rebounds": 0})
    for event in events:
        if not event.confirmed or not event.player_id:
            continue
        player = stats[event.player_id]
        if event.event_type == "shot_attempt":
            player["shot_attempts"] += 1
        elif event.event_type == "shot_made":
            player["shot_attempts"] += 1
            player["made_shots"] += 1
            player["points"] += 2
        elif event.event_type == "shot_missed":
            player["missed_shots"] += 1
        elif event.event_type == "rebound":
            player["rebounds"] += 1
    return dict(stats)
