from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


VIDEO = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_EVERY = int(os.environ.get("BALLVISION_SAMPLE_EVERY", "60"))
START_SAMPLE = int(os.environ.get("BALLVISION_START_SAMPLE", "0"))
MAX_SAMPLES = int(os.environ.get("BALLVISION_MAX_SAMPLES", "0"))
OUTPUT_SUFFIX = os.environ.get("BALLVISION_OUTPUT_SUFFIX", "")
WEIGHTS = Path(os.environ.get("BALLVISION_HF_WEIGHTS", "weights/hf-basketball"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


processor = AutoImageProcessor.from_pretrained(WEIGHTS)
model = AutoModelForObjectDetection.from_pretrained(WEIGHTS)
model.to(DEVICE)
model.eval()
capture = cv2.VideoCapture(str(VIDEO))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
frame_index = 0
records: list[dict] = []
person_tracks: dict[int, dict] = {}
ball_tracks: dict[int, dict] = {}
next_person_id = 1
next_ball_id = 1
person_detection_count = 0
ball_detection_count = 0
frames_with_ball = 0

while True:
    ok, frame = capture.read()
    if not ok:
        break
    if frame_index % SAMPLE_EVERY == 0 and frame_index // SAMPLE_EVERY >= START_SAMPLE:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = {key: value.to(DEVICE) for key, value in processor(images=image, return_tensors="pt").items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs,
            threshold=0.15,
            target_sizes=torch.tensor([[image.height, image.width]]),
        )[0]
        people: list[dict] = []
        balls: list[dict] = []
        for bbox, confidence, label in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if label == 1 and confidence >= 0.35 and 110 <= center_x <= 940 and 80 <= center_y <= 365 and (y2 - y1) >= 30:
                people.append({"bbox": [round(value, 1) for value in bbox], "confidence": round(float(confidence), 4)})
            elif label == 0 and confidence >= 0.15 and 90 <= center_x <= 950 and 50 <= center_y <= 400:
                balls.append({"bbox": [round(value, 1) for value in bbox], "confidence": round(float(confidence), 4)})
        person_detection_count += len(people)
        ball_detection_count += len(balls)
        if balls:
            frames_with_ball += 1
        sample_number = frame_index // SAMPLE_EVERY
        next_person_id = update_tracks(person_tracks, people, next_person_id, sample_number, 0.05)
        next_ball_id = update_tracks(ball_tracks, balls, next_ball_id, sample_number, 0.01)
        records.append({"frame_index": frame_index, "timestamp_s": round(frame_index / fps, 3), "people": people, "balls": balls})
        if MAX_SAMPLES and len(records) >= MAX_SAMPLES:
            break
    frame_index += 1
capture.release()

person_track_lengths = Counter()
ball_track_lengths = Counter()
for record in records:
    for item in record["people"]:
        if "track_id" in item:
            person_track_lengths[item["track_id"]] += 1
    for item in record["balls"]:
        if "track_id" in item:
            ball_track_lengths[item["track_id"]] += 1

summary = {
    "video": str(VIDEO),
    "model": "koppolusameer/rfdetr-basketball-player-ball-referee-detection",
    "device": str(DEVICE),
    "sample_every_frames": SAMPLE_EVERY,
    "start_sample": START_SAMPLE,
    "sampled_frames": len(records),
    "person_detection_count": person_detection_count,
    "ball_detection_count": ball_detection_count,
    "frames_with_ball": frames_with_ball,
    "person_track_count": len(person_track_lengths),
    "ball_track_count": len(ball_track_lengths),
    "person_track_lengths": dict(person_track_lengths),
    "ball_track_lengths": dict(ball_track_lengths),
    "records": records,
}
output = OUT_DIR / f"{VIDEO.stem}-hf-basketball{OUTPUT_SUFFIX}.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: summary[key] for key in ["model", "sampled_frames", "person_detection_count", "ball_detection_count", "frames_with_ball", "person_track_count", "ball_track_count"]} | {"output": str(output)}, ensure_ascii=False, indent=2))
