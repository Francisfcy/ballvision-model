from __future__ import annotations

import argparse
import json
from pathlib import Path

from ballvision.event_inference import infer_candidate_events


parser = argparse.ArgumentParser(description="Infer reviewable shot and rebound candidates from dense tracks")
parser.add_argument("tracks_json", type=Path)
parser.add_argument("--rim", nargs=4, type=float, action="append", required=True, metavar=("X1", "Y1", "X2", "Y2"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

payload = json.loads(args.tracks_json.read_text(encoding="utf-8"))
events = infer_candidate_events(payload.get("records_every_5_frames", []), rim_bboxes=args.rim)
result = {
    "video": payload.get("video"),
    "model": payload.get("model"),
    "rim_bboxes": args.rim,
    "events": [event.model_dump() for event in events],
}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"events": len(events), "output": str(args.output)}, ensure_ascii=False, indent=2))
