from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


VIDEO = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_EVERY = int(os.environ.get("BALLVISION_SAMPLE_EVERY", "10"))
OUTPUT_SUFFIX = os.environ.get("BALLVISION_OUTPUT_SUFFIX", "")

model = YOLO("yolo11n.pt")
track_counts: Counter[int] = Counter()
track_first: dict[int, float] = {}
track_last: dict[int, float] = {}
ball_frames = 0
person_frames = 0
court_person_detections = 0
court_track_ids: set[int] = set()
frame_count = 0
detections = []

for result in model.track(
    source=str(VIDEO),
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0, 32],
    conf=0.25,
    iou=0.5,
    imgsz=640,
    vid_stride=SAMPLE_EVERY,
    verbose=False,
):
    frame_count += 1
    timestamp = float(result.speed.get("preprocess", 0.0) * 0)  # keep schema stable before FPS metadata is added
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        continue
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
    classes = boxes.cls.int().cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    coords = boxes.xyxy.cpu().tolist()
    frame_records = []
    for track_id, cls_id, confidence, xyxy in zip(ids, classes, confs, coords):
        if cls_id == 0:
            person_frames += 1
        if cls_id == 32:
            ball_frames += 1
        x1, y1, x2, y2 = xyxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if cls_id == 0 and 110 <= center_x <= 940 and 80 <= center_y <= 365 and (y2 - y1) >= 30:
            court_person_detections += 1
            if track_id >= 0:
                court_track_ids.add(track_id)
        if track_id >= 0:
            track_counts[track_id] += 1
            track_first.setdefault(track_id, frame_count)
            track_last[track_id] = frame_count
        frame_records.append({"track_id": track_id, "class_id": cls_id, "confidence": round(float(confidence), 4), "bbox": [round(float(value), 1) for value in xyxy]})
    if len(detections) < 300:
        detections.append({"sample_index": frame_count, "objects": frame_records})

summary = {
    "video": str(VIDEO),
    "model": "yolo11n.pt",
    "tracker": "bytetrack.yaml",
    "sample_stride": SAMPLE_EVERY,
    "sampled_frames": frame_count,
    "frames_with_person_detections": person_frames,
    "frames_with_ball_detections": ball_frames,
    "court_person_detections": court_person_detections,
    "court_unique_track_ids": len(court_track_ids),
    "unique_track_ids": len(track_counts),
    "track_lengths": {
        str(track_id): {
            "sample_count": count,
            "first_sample": track_first[track_id],
            "last_sample": track_last[track_id],
        }
        for track_id, count in track_counts.most_common()
    },
    "sample_detections": detections,
}
output = OUT_DIR / f"{VIDEO.stem}-yolo-baseline{OUTPUT_SUFFIX}.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["model", "tracker", "sampled_frames", "frames_with_person_detections", "frames_with_ball_detections", "unique_track_ids"]} | {"output": str(output)}, ensure_ascii=False, indent=2))
