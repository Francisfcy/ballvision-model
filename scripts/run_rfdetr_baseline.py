from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image
from rfdetr import RFDETRNano


VIDEO = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_EVERY = 60
BATCH_SIZE = 1
MAX_SAMPLES = int(os.environ.get("BALLVISION_MAX_SAMPLES", "0"))
START_SAMPLE = int(os.environ.get("BALLVISION_START_SAMPLE", "0"))
OUTPUT_SUFFIX = os.environ.get("BALLVISION_OUTPUT_SUFFIX", "")


def iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def update_tracks(tracks: dict[int, dict], detections: list[dict], next_id: int, frame_index: int, threshold: float) -> int:
    unmatched = set(range(len(detections)))
    for track_id, track in list(tracks.items()):
        if frame_index - track["last_frame"] > 3:
            del tracks[track_id]
            continue
        best_index = None
        best_score = threshold
        for index in unmatched:
            score = iou(track["bbox"], detections[index]["bbox"])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None:
            detection = detections[best_index]
            track["bbox"] = detection["bbox"]
            track["last_frame"] = frame_index
            track["hits"] += 1
            detection["track_id"] = track_id
            unmatched.remove(best_index)
    for index in unmatched:
        tracks[next_id] = {"bbox": detections[index]["bbox"], "last_frame": frame_index, "hits": 1}
        detections[index]["track_id"] = next_id
        next_id += 1
    return next_id


model = RFDETRNano()
predictor = model.inference(compile=False, batch_size=BATCH_SIZE)
capture = cv2.VideoCapture(str(VIDEO))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
sample_index = 0
batch_images: list[Image.Image] = []
batch_meta: list[tuple[int, float]] = []
person_tracks: dict[int, dict] = {}
ball_tracks: dict[int, dict] = {}
next_person_id = 1
next_ball_id = 1
person_detection_count = 0
ball_detection_count = 0
ball_frames = 0
records: list[dict] = []


def process_batch() -> None:
    global next_person_id, next_ball_id, person_detection_count, ball_detection_count, ball_frames
    if not batch_images:
        return
    predictions = predictor.predict(batch_images, threshold=0.15, batch=1, shape=(384, 384), include_source_image=False)
    for prediction, (frame_index, timestamp_s) in zip(predictions, batch_meta):
        people: list[dict] = []
        balls: list[dict] = []
        for bbox, confidence, class_id in zip(prediction.xyxy.tolist(), prediction.confidence.tolist(), prediction.class_id.tolist()):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if class_id == 1 and 110 <= center_x <= 940 and 80 <= center_y <= 365 and (y2 - y1) >= 30:
                people.append({"bbox": [round(value, 1) for value in bbox], "confidence": round(float(confidence), 4)})
            elif class_id == 37 and 90 <= center_x <= 950 and 50 <= center_y <= 400:
                balls.append({"bbox": [round(value, 1) for value in bbox], "confidence": round(float(confidence), 4)})
        person_detection_count += len(people)
        ball_detection_count += len(balls)
        if balls:
            ball_frames += 1
        next_person_id = update_tracks(person_tracks, people, next_person_id, frame_index, 0.05)
        next_ball_id = update_tracks(ball_tracks, balls, next_ball_id, frame_index, 0.01)
        records.append({"frame_index": frame_index, "timestamp_s": round(timestamp_s, 3), "people": people, "balls": balls})
    batch_images.clear()
    batch_meta.clear()


while True:
    ok, frame = capture.read()
    if not ok:
        break
    if sample_index % SAMPLE_EVERY == 0 and sample_index // SAMPLE_EVERY >= START_SAMPLE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_images.append(Image.fromarray(rgb))
        batch_meta.append((sample_index, sample_index / fps))
        if len(batch_images) >= BATCH_SIZE:
            process_batch()
        if MAX_SAMPLES and len(records) >= MAX_SAMPLES:
            break
    sample_index += 1
process_batch()
capture.release()

track_lengths = Counter()
for record in records:
    for item in record["people"]:
        if "track_id" in item:
            track_lengths[item["track_id"]] += 1
    for item in record["balls"]:
        if "track_id" in item:
            track_lengths[f"ball-{item['track_id']}"] += 1

summary = {
    "video": str(VIDEO),
    "model": "RF-DETR Nano COCO baseline",
    "sample_every_frames": SAMPLE_EVERY,
    "start_sample": START_SAMPLE,
    "sampled_frames": len(records),
    "person_detection_count": person_detection_count,
    "ball_detection_count": ball_detection_count,
    "frames_with_ball": ball_frames,
    "person_track_count": len([key for key in track_lengths if not str(key).startswith("ball-")]),
    "ball_track_count": len([key for key in track_lengths if str(key).startswith("ball-")]),
    "track_lengths": dict(track_lengths),
    "records": records,
}
output = OUT_DIR / f"{VIDEO.stem}-rfdetr-baseline{OUTPUT_SUFFIX}.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: summary[key] for key in ["model", "sampled_frames", "person_detection_count", "ball_detection_count", "frames_with_ball", "person_track_count", "ball_track_count"]} | {"output": str(output)}, ensure_ascii=False, indent=2))
