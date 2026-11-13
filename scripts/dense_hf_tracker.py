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
WEIGHTS = Path(os.environ.get("BALLVISION_HF_WEIGHTS", "weights/hf-basketball"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECT_EVERY = int(os.environ.get("BALLVISION_DETECT_EVERY", "10"))
MAX_FRAMES = int(os.environ.get("BALLVISION_MAX_FRAMES", "0"))
START_FRAME = int(os.environ.get("BALLVISION_START_FRAME", "0"))
OUTPUT_SUFFIX = os.environ.get("BALLVISION_OUTPUT_SUFFIX", "")
WRITE_VIDEO = os.environ.get("BALLVISION_WRITE_VIDEO", "1") == "1"


def shift_box(box: list[float], dx: float, dy: float) -> list[float]:
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union else 0.0


def center_distance(left: list[float], right: list[float]) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    return ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5


def propagate_tracks(tracks: dict[int, dict], previous_gray, current_gray, frame_shape) -> None:
    if previous_gray is None:
        return
    height, width = frame_shape[:2]
    for track in tracks.values():
        box = track["bbox"]
        points = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]
        old_points = __import__("numpy").float32(points)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, old_points, None, winSize=(21, 21), maxLevel=3)
        if status is not None and int(status.reshape(-1)[0]) == 1:
            old_point = old_points.reshape(-1, 2)[0]
            new_point = new_points.reshape(-1, 2)[0]
            dx = float(new_point[0] - old_point[0])
            dy = float(new_point[1] - old_point[1])
            track["bbox"] = shift_box(box, dx, dy)
        track["bbox"] = [
            max(0.0, min(width, track["bbox"][0])),
            max(0.0, min(height, track["bbox"][1])),
            max(0.0, min(width, track["bbox"][2])),
            max(0.0, min(height, track["bbox"][3])),
        ]
        track["age"] += 1


def associate(tracks: dict[int, dict], detections: list[dict], next_id: int, frame_index: int, kind: str) -> int:
    available = set(tracks)
    for detection in detections:
        best_id = None
        best_score = 0.05
        for track_id in available:
            track = tracks[track_id]
            overlap = iou(track["bbox"], detection["bbox"])
            height = max(track["bbox"][3] - track["bbox"][1], 1.0)
            distance_score = max(0.0, 1.0 - center_distance(track["bbox"], detection["bbox"]) / (height * (2.5 if kind == "person" else 4.0)))
            score = max(overlap, distance_score * 0.4)
            if score > best_score:
                best_score = score
                best_id = track_id
        if best_id is None:
            best_id = next_id
            next_id += 1
            tracks[best_id] = {"bbox": detection["bbox"], "age": 0, "hits": 0, "last_detection": frame_index}
        else:
            available.remove(best_id)
            tracks[best_id]["bbox"] = detection["bbox"]
            tracks[best_id]["age"] = 0
            tracks[best_id]["last_detection"] = frame_index
        tracks[best_id]["hits"] += 1
        detection["track_id"] = best_id
    for track_id in list(tracks):
        if tracks[track_id]["age"] > DETECT_EVERY * 3:
            del tracks[track_id]
    return next_id


processor = AutoImageProcessor.from_pretrained(WEIGHTS)
model = AutoModelForObjectDetection.from_pretrained(WEIGHTS)
model.to(DEVICE)
model.eval()
capture = cv2.VideoCapture(str(VIDEO))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = None
if WRITE_VIDEO:
    output_video = OUT_DIR / f"{VIDEO.stem}-dense{OUTPUT_SUFFIX}.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

person_tracks: dict[int, dict] = {}
ball_tracks: dict[int, dict] = {}
next_person_id = 1
next_ball_id = 1
previous_gray = None
frame_index = 0
records: list[dict] = []
detected_persons = 0
detected_balls = 0

while True:
    ok, frame = capture.read()
    if not ok:
        break
    if frame_index < START_FRAME:
        frame_index += 1
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    propagate_tracks(person_tracks, previous_gray, gray, frame.shape)
    propagate_tracks(ball_tracks, previous_gray, gray, frame.shape)

    detections_people: list[dict] = []
    detections_balls: list[dict] = []
    if frame_index % DETECT_EVERY == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = {key: value.to(DEVICE) for key, value in processor(images=image, return_tensors="pt").items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outputs = model(**inputs)
        result = processor.post_process_object_detection(outputs, threshold=0.15, target_sizes=torch.tensor([[height, width]]))[0]
        for bbox, confidence, label in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            bbox = [max(0.0, min(width, x1)), max(0.0, min(height, y1)), max(0.0, min(width, x2)), max(0.0, min(height, y2))]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if label == 1 and confidence >= 0.35 and 110 <= center_x <= width - 20 and 60 <= center_y <= height - 60 and (y2 - y1) >= 30:
                detections_people.append({"bbox": [x1, y1, x2, y2], "confidence": float(confidence)})
            elif label == 0 and confidence >= 0.15 and 70 <= center_x <= width - 10 and 40 <= center_y <= height - 35:
                detections_balls.append({"bbox": [x1, y1, x2, y2], "confidence": float(confidence)})
        detected_persons += len(detections_people)
        detected_balls += len(detections_balls)
        next_person_id = associate(person_tracks, detections_people, next_person_id, frame_index, "person")
        next_ball_id = associate(ball_tracks, detections_balls, next_ball_id, frame_index, "ball")

    if frame_index % 5 == 0:
        records.append({
            "frame_index": frame_index,
            "timestamp_s": round(frame_index / fps, 3),
            "people": [{"track_id": track_id, "bbox": [round(value, 1) for value in track["bbox"]], "age": track["age"], "hits": track["hits"]} for track_id, track in person_tracks.items()],
            "balls": [{"track_id": track_id, "bbox": [round(value, 1) for value in track["bbox"]], "age": track["age"], "hits": track["hits"]} for track_id, track in ball_tracks.items()],
        })
    if writer is not None:
        for track_id, track in person_tracks.items():
            x1, y1, x2, y2 = [int(value) for value in track["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 30), 1)
            cv2.putText(frame, f"P{track_id}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 30), 1)
        for track_id, track in ball_tracks.items():
            x1, y1, x2, y2 = [int(value) for value in track["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(frame, f"B{track_id}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)
        writer.write(frame)
    previous_gray = gray
    frame_index += 1
    if MAX_FRAMES and frame_index - START_FRAME >= MAX_FRAMES:
        break

capture.release()
if writer is not None:
    writer.release()

person_lengths = Counter()
ball_lengths = Counter()
for record in records:
    for item in record["people"]:
        person_lengths[item["track_id"]] += 1
    for item in record["balls"]:
        ball_lengths[item["track_id"]] += 1

summary = {
    "video": str(VIDEO),
    "model": "koppolusameer/rfdetr-basketball-player-ball-referee-detection",
    "device": str(DEVICE),
    "detect_every_frames": DETECT_EVERY,
    "processed_frames": frame_index - START_FRAME,
    "detector_person_boxes": detected_persons,
    "detector_ball_boxes": detected_balls,
    "person_track_count": len(person_lengths),
    "ball_track_count": len(ball_lengths),
    "person_track_lengths": dict(person_lengths),
    "ball_track_lengths": dict(ball_lengths),
    "records_every_5_frames": records,
}
output = OUT_DIR / f"{VIDEO.stem}-dense{OUTPUT_SUFFIX}.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: summary[key] for key in ["processed_frames", "detector_person_boxes", "detector_ball_boxes", "person_track_count", "ball_track_count"]} | {"output": str(output)}, ensure_ascii=False, indent=2))
