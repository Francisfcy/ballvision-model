from __future__ import annotations

from collections import defaultdict

from .schemas import Event, ScoreObservation


def validate_score_changes(events: list[Event], observations: list[ScoreObservation]) -> dict[str, object]:
    ordered = sorted(observations, key=lambda item: item.timestamp_s)
    intervals: list[dict[str, object]] = []
    for previous, current in zip(ordered, ordered[1:]):
        observed_delta = {
            team: current.team_scores.get(team, 0) - previous.team_scores.get(team, 0)
            for team in set(previous.team_scores) | set(current.team_scores)
        }
        confirmed_delta: defaultdict[str, int] = defaultdict(int)
        pending_events = 0
        for event in events:
            if not (previous.timestamp_s < event.timestamp_s <= current.timestamp_s):
                continue
            if event.event_type in {"shot_attempt", "shot_candidate", "rebound_candidate"} and event.review_status == "pending":
                pending_events += 1
            if event.confirmed and event.review_status == "confirmed" and event.event_type == "shot_made" and event.team_id:
                confirmed_delta[event.team_id] += int(event.evidence.get("points", 2) or 2)
        confirmed_delta_dict = dict(confirmed_delta)
        if observed_delta == confirmed_delta_dict:
            status = "matched"
        elif pending_events:
            status = "pending_review"
        else:
            status = "mismatch"
        intervals.append({
            "start_s": previous.timestamp_s,
            "end_s": current.timestamp_s,
            "observed_delta": observed_delta,
            "confirmed_delta": confirmed_delta_dict,
            "pending_events": pending_events,
            "status": status,
        })
    return {
        "status": "matched" if intervals and all(item["status"] == "matched" for item in intervals) else "needs_review",
        "intervals": intervals,
    }
