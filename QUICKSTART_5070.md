# RTX 5070 快速启动

在 5070 主机上执行：

```powershell
git clone https://github.com/Francisfcy/ballvision-model.git
cd ballvision-model

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/download_hf_basketball.py
```

看到 `True` 和 `NVIDIA GeForce RTX 5070` 后，可以运行连续跟踪：

```powershell
$env:BALLVISION_DETECT_EVERY = "5"
$env:BALLVISION_WRITE_VIDEO = "1"
python scripts/dense_hf_tracker.py path\to\game.mp4 results
```

生成事件候选时，需要提供两个篮筐区域：

```powershell
$env:PYTHONPATH = "src"
python scripts/infer_events.py results\game-dense.json `
  --rim 120 70 190 145 `
  --rim 800 70 890 145 `
  --output results\game-events.json
```

模型权重和比赛视频不放进 Git。`download_hf_basketball.py` 会重新下载篮球微调 RF-DETR 权重，结果文件会在本机生成。
