from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="koppolusameer/rfdetr-basketball-player-ball-referee-detection",
    local_dir="weights/hf-basketball",
    allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt"],
)
print("downloaded", flush=True)
