# BallVision Model
# BallVision Model Baseline

这是第一版云端视频识别服务的工程基线。它先把模型输出、身份绑定、事件流和统计计算连通起来，身份识别不依赖球衣号码。

## 当前能力

- 选手名单中的号码字段是可选的
- 轨迹身份支持人工绑定
- 轨迹身份接口预留队伍标签和外观嵌入
- 号码识别只作为额外证据
- 事件流可以生成可解释的基础统计
- FastAPI 提供健康检查和分析任务契约

## 运行烟雾测试

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_demo.py
```

## 启动 API

```powershell
python -m uvicorn ballvision.api:app --app-dir src --reload
```

当前 API 还没有接入 GPU 检测器。下一步接入顺序是：

1. 视频读取和抽帧
2. 球员、篮球、篮筐检测器
3. ByteTrack 基线
4. 轨迹级外观特征
5. 球场关键点和单应性
6. 投篮事件识别

## 在 RTX 5070 上启用 CUDA

当前运行环境检测到的是 CPU 版 PyTorch，所以 `torch.cuda.is_available()` 为 `False`。在 RTX 5070 主机上，先安装 NVIDIA 驱动，再在项目环境安装带 CUDA 的 PyTorch。PyTorch 官方安装页建议根据 Windows、Pip、Python 和 CUDA 版本选择对应命令。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`run_hf_basketball_detector.py` 和 `dense_hf_tracker.py` 会自动选择 CUDA，并在 CUDA 上使用半精度推理。

## 人工修正和比分校验

人工修正保存在 SQLite。默认数据库是 `data/ballvision.sqlite3`，可以通过 `BALLVISION_DB_PATH` 指定其他路径。

```powershell
$env:PYTHONPATH = "src"
$env:BALLVISION_DB_PATH = "data\video-demo.sqlite3"
python scripts/import_review.py results\events.json --score 0 GS=7 HOU=12 --score 10 GS=7 HOU=17
```

接口包括事件更新、比分观测和比分校验：

```text
PATCH /review/sessions/{session_id}/events/{event_id}
POST  /review/sessions/{session_id}/score-observations
POST  /review/sessions/{session_id}/validate-score
POST  /review/sessions/{session_id}/recalculate
```

## 篮球专用检测器

当前已经加入公开的篮球微调 RF-DETR 权重，存放在 `weights/hf-basketball/`。它输出 `ball`、`player` 和 `referee` 三类结果。完整视频的抽样结果保存在 `results/7885500b16c41d41a183d78707cc077a-hf-basketball.json`。

运行专用检测器：

```powershell
$env:BALLVISION_SAMPLE_EVERY = "120"
python scripts/run_hf_basketball_detector.py path\to\game.mp4 results
```

球衣号码没有参与身份绑定。当前身份和轨迹仍需要连续帧检测、队伍外观特征和 ReID。

## 连续帧轨迹基线

连续帧版本每 10 帧运行一次篮球检测器，在检测之间使用光流传播球员和篮球框。它会输出每 5 帧一条轨迹记录，并可生成标注视频。

```powershell
$env:BALLVISION_DETECT_EVERY = "10"
$env:BALLVISION_WRITE_VIDEO = "1"
python scripts/dense_hf_tracker.py path\to\game.mp4 results
```

真实模型接入后，检测器和跟踪器只需要写入 `Detection` 和 `TrackObservation`，身份和统计层可以继续复用。
