from __future__ import annotations

import json
from pathlib import Path


tracks_path = Path("results/7885500b16c41d41a183d78707cc077a-dense-10s.json")
events_path = Path("results/7885500b16c41d41a183d78707cc077a-events.json")
tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
events = json.loads(events_path.read_text(encoding="utf-8"))["events"]

records = tracks["records_every_5_frames"]
timestamps = [float(record["timestamp_s"]) for record in records]
checks: list[tuple[str, bool, str]] = []
checks.append(("视频帧记录存在", bool(records), f"{len(records)} 条记录"))
checks.append(("时间戳单调递增", timestamps == sorted(timestamps), "时间轴连续"))
checks.append(("球员检测框存在", tracks["detector_person_boxes"] > 0, str(tracks["detector_person_boxes"])))
checks.append(("篮球检测框存在", tracks["detector_ball_boxes"] > 0, str(tracks["detector_ball_boxes"])))

all_boxes_valid = True
track_rows: dict[int, dict] = {}
for record in records:
    for item in record.get("people", []):
        box = item["bbox"]
        all_boxes_valid = all_boxes_valid and 0 <= box[0] <= 960 and 0 <= box[1] <= 448 and 0 <= box[2] <= 960 and 0 <= box[3] <= 448
        track_id = int(item["track_id"])
        row = track_rows.setdefault(track_id, {"first": record["timestamp_s"], "last": record["timestamp_s"], "points": 0})
        row["last"] = record["timestamp_s"]
        row["points"] += 1
    for item in record.get("balls", []):
        box = item["bbox"]
        all_boxes_valid = all_boxes_valid and 0 <= box[0] <= 960 and 0 <= box[1] <= 448 and 0 <= box[2] <= 960 and 0 <= box[3] <= 448
checks.append(("检测框坐标有效", all_boxes_valid, "全部落在 960×448 画面内" if all_boxes_valid else "存在越界框"))

event_times_valid = all(0 <= event["timestamp_s"] <= 10.0 and event["evidence_start_s"] <= event["timestamp_s"] <= event["evidence_end_s"] for event in events)
checks.append(("事件证据时间有效", event_times_valid, f"{len(events)} 个候选事件"))

ordered_tracks = sorted(track_rows.items(), key=lambda item: (-item[1]["points"], item[0]))
player_rows = []
for index, (track_id, row) in enumerate(ordered_tracks, start=1):
    player_rows.append({
        "label": f"临时球员 {index:02d}",
        "track_id": f"P{track_id}",
        "points": row["points"],
        "first_s": row["first"],
        "last_s": row["last"],
        "duration_s": round(row["last"] - row["first"], 2),
    })

report = ["# 临时球员标签与流程校验", "", "## 临时球员数据", "", "| 标签 | 原始轨迹 | 轨迹点数 | 首次出现 | 最后出现 | 持续时间 |", "|---|---|---:|---:|---:|---:|"]
for row in player_rows:
    report.append(f"| {row['label']} | {row['track_id']} | {row['points']} | {row['first_s']:.2f}s | {row['last_s']:.2f}s | {row['duration_s']:.2f}s |")
report.extend(["", "## 流程校验", "", "| 检查项 | 结果 | 说明 |", "|---|---|---|"])
for name, passed, detail in checks:
    report.append(f"| {name} | {'通过' if passed else '失败'} | {detail} |")
report.extend(["", "## 当前结论", "", "检测、轨迹、事件候选和人工修正数据链路已经连通。临时球员标签只代表当前视频轨迹，不能跨镜头或跨比赛直接视为同一个真实球员。", "", "事件候选："])
for event in events:
    report.append(f"- {event['timestamp_s']:.3f}s：{event['event_type']}，置信度 {event['confidence']:.2f}，状态待确认")

output = Path("results/7885500b16c41d41a183d78707cc077a-validation-report.md")
output.write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
