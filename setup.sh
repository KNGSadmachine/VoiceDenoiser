#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== VoiceDenoiser Setup ==="

if [ ! -d venv ]; then
    echo "[1/4] Creating venv..."
    python3 -m venv venv
else
    echo "[1/4] venv already exists. Skipping."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[2/4] NVIDIA GPU detected. Installing CUDA build of PyTorch..."
    venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "[2/4] No NVIDIA GPU detected. Installing CPU build of PyTorch..."
    venv/bin/python -m pip install torch torchaudio
fi

echo "[3/4] Installing dependencies..."
venv/bin/python -m pip install -r requirements.txt

echo "[4/4] Downloading Resemble Enhance model (approx. 700MB)..."
venv/bin/python -m resemble_enhance.enhancer.download

echo
echo "Setup complete. Run ./run.sh to launch."
