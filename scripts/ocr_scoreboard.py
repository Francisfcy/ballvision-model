from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR


parser = argparse.ArgumentParser(description="Extract scoreboard observations from a basketball broadcast")
parser.add_argument("video", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sample-every", type=int, default=30)
parser.add_argument("--roi", nargs=4, type=int, default=[280, 385, 730, 447], metavar="N")
parser.add_argument("--teams", nargs=2, default=["GS", "HOU"])
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--max-frames", type=int, default=0)
args = parser.parse_args()

ocr = RapidOCR()
capture = cv2.VideoCapture(str(args.video))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
frame_index = 0
last_scores: dict[str, int] | None = None
observations: list[dict] = []
x1, y1, x2, y2 = args.roi


def parse_tokens(result: list) -> list[dict]:
    tokens = []
    for item in result or []:
        points, text, confidence = item
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        normalized = re.sub(r"[^A-Z0-9:.-]", "", str(text).upper())
        tokens.append({"text": normalized, "x": center_x, "y": center_y, "confidence": float(confidence)})
    return sorted(tokens, key=lambda token: token["x"])


while True:
    ok, frame = capture.read()
    if not ok:
        break
    if frame_index < args.start_frame:
        frame_index += 1
        continue
    if frame_index % args.sample_every == 0:
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result, _ = ocr(gray)
        tokens = parse_tokens(result)
        scores: dict[str, int] = {}
        for team in args.teams:
            team_tokens = [token for token in tokens if token["text"] == team]
            if not team_tokens:
                continue
            team_token = team_tokens[0]
            candidates = []
            for token in tokens:
                if token["x"] <= team_token["x"] or abs(token["y"] - team_token["y"]) > 90:
                    continue
                if not re.fullmatch(r"\d{1,3}", token["text"]):
                    continue
                value = int(token["text"])
                if value <= 199:
                    candidates.append((token["x"] - team_token["x"], value, token))
            if candidates:
                _, value, _ = min(candidates, key=lambda item: item[0])
                scores[team] = value
        if len(scores) == 2 and scores != last_scores:
            observations.append({
                "timestamp_s": round(frame_index / fps, 3),
                "team_scores": scores,
                "source": "ocr",
                "tokens": tokens,
            })
            last_scores = scores
    frame_index += 1
    if args.max_frames and frame_index - args.start_frame >= args.max_frames:
        break
capture.release()

args.output.write_text(json.dumps({"video": str(args.video), "roi": args.roi, "observations": observations}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"observations": len(observations), "output": str(args.output)}, ensure_ascii=False, indent=2))
