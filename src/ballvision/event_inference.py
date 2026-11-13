from __future__ import annotations

from collections import defaultdict
from math import hypot
from uuid import uuid4

from .schemas import Event


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def infer_candidate_events(
    records: list[dict],
    *,
    rim_bbox: list[float] | None = None,
    rim_bboxes: list[list[float]] | None = None,
    rim_margin: float = 42.0,
    player_radius: float = 125.0,
) -> list[Event]:
    """Find reviewable shot and rebound candidates from dense track records.

    The CPU baseline deliberately returns reviewable candidates. Made/missed
    outcomes remain unknown until scoreboard evidence or a human confirms them.
    """
    all_rim_bboxes = rim_bboxes or ([rim_bbox] if rim_bbox else [])
    if not all_rim_bboxes:
        raise ValueError("At least one rim bounding box is required")
    rim_centers = [_center(box) for box in all_rim_bboxes]
    ball_points: dict[int, list[tuple[float, float, float, dict]]] = defaultdict(list)
    players_by_time: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        timestamp = float(record.get("timestamp_s", 0.0))
        for ball in record.get("balls", []):
            track_id = int(ball["track_id"])
            ball_points[track_id].append((timestamp, *_center(ball["bbox"]), ball))
        for player in record.get("people", []):
            players_by_time[timestamp].append(_center(player["bbox"]))

    candidates: list[Event] = []
    for track_id, points in ball_points.items():
        points.sort(key=lambda item: item[0])
        for index, (timestamp, x, y, ball) in enumerate(points):
            nearest_rim = min(rim_centers, key=lambda center: _distance((x, y), center))
            if _distance((x, y), nearest_rim) > rim_margin:
                continue
            previous = points[max(0, index - 3):index]
            current_distance = _distance((x, y), nearest_rim)
            previous_distance = sum(_distance((item[1], item[2]), nearest_rim) for item in previous) / len(previous) if previous else current_distance + rim_margin
            if previous and current_distance >= previous_distance:
                continue
            start_time = previous[0][0] if previous else max(0.0, timestamp - 0.5)
            end_time = points[min(len(points) - 1, index + 3)][0]
            candidates.append(
                Event(
                    event_id=str(uuid4()),
                    event_type="shot_attempt",
                    timestamp_s=timestamp,
                    track_id=str(track_id),
                    confidence=min(0.88, 0.35 + (previous_distance - current_distance) / max(rim_margin * 4, 1)),
                    outcome="unknown",
                    review_status="pending",
                    evidence_start_s=start_time,
                    evidence_end_s=end_time,
                    evidence={"ball_track_id": track_id, "rim_distance": round(current_distance, 2)},
                )
            )
            future_points = points[index + 1:index + 9]
            for future_timestamp, future_x, future_y, _ in future_points:
                if _distance((future_x, future_y), nearest_rim) <= rim_margin:
                    continue
                nearby_players = players_by_time.get(future_timestamp, [])
                if nearby_players and min(_distance((future_x, future_y), player) for player in nearby_players) <= player_radius:
                    candidates.append(
                        Event(
                            event_id=str(uuid4()),
                            event_type="rebound_candidate",
                            timestamp_s=future_timestamp,
                            track_id=str(track_id),
                            confidence=0.48,
                            outcome="unknown",
                            review_status="pending",
                            evidence_start_s=timestamp,
                            evidence_end_s=future_timestamp,
                            evidence={"ball_track_id": track_id, "source_shot_event": candidates[-1].event_id},
                        )
                    )
                    break
            break

    candidates.sort(key=lambda event: event.timestamp_s)
    deduplicated: list[Event] = []
    for event in candidates:
        if any(abs(event.timestamp_s - previous.timestamp_s) < 1.5 and event.event_type == previous.event_type for previous in deduplicated):
            continue
        deduplicated.append(event)
    return deduplicated
