from __future__ import annotations

import json

from ballvision import BaselinePipeline, Event, PlayerProfile, TrackObservation


def main() -> None:
    roster = [
        PlayerProfile(player_id="p1", name="林嘉豪", team_id="red"),
        PlayerProfile(player_id="p2", name="周子谦", team_id="red"),
        PlayerProfile(player_id="p3", name="陈宇航", team_id="blue"),
    ]
    pipeline = BaselinePipeline(roster)
    result = pipeline.analyze(
        [
            TrackObservation(track_id="track-1", timestamp_s=1.0, team_hint="red"),
            TrackObservation(track_id="track-2", timestamp_s=1.0, team_hint="blue"),
        ],
        [
            Event(event_id="e1", event_type="shot_made", timestamp_s=12.4, track_id="track-1", confidence=0.88, confirmed=True),
            Event(event_id="e2", event_type="rebound", timestamp_s=13.1, track_id="track-2", confidence=0.72, confirmed=True),
        ],
        manual_bindings={"track-1": "p1", "track-2": "p3"},
    )
    print(json.dumps({"identities": [item.model_dump() for item in result.identities], "box_score": result.box_score}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
