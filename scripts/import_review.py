from __future__ import annotations

import argparse
import json
from pathlib import Path

from ballvision.review_store import ReviewStore
from ballvision.schemas import Event, ScoreObservationCreate


parser = argparse.ArgumentParser(description="Persist review events and scoreboard observations in SQLite")
parser.add_argument("events_json", type=Path)
parser.add_argument("--score", nargs="+", action="append", default=[], metavar="VALUE")
parser.add_argument("--score-json", type=Path)
args = parser.parse_args()

payload = json.loads(args.events_json.read_text(encoding="utf-8"))
store = ReviewStore()
session = store.create([Event.model_validate(event) for event in payload.get("events", [])])
for score_args in args.score:
    timestamp_s = float(score_args[0])
    team_scores = {}
    for item in score_args[1:]:
        team, score = item.split("=", 1)
        team_scores[team] = int(score)
    store.add_score_observation(session.session_id, ScoreObservationCreate(timestamp_s=timestamp_s, team_scores=team_scores, source="manual"))
if args.score_json:
    score_payload = json.loads(args.score_json.read_text(encoding="utf-8"))
    for observation in score_payload.get("observations", []):
        store.add_score_observation(session.session_id, ScoreObservationCreate(**observation))
if not args.score and not args.score_json:
    raise SystemExit("Provide --score or --score-json")
print(json.dumps({"session_id": session.session_id, "database": str(store.db_path), "result": store.recalculate(session.session_id)}, ensure_ascii=False, indent=2, default=lambda value: value.model_dump() if hasattr(value, "model_dump") else str(value)))
