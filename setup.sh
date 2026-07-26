#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
TORCH_VERSION="2.5.1"

# DeepFilterLib currently publishes macOS/Apple Silicon wheels for Python
# 3.10 and 3.11 only.  Letting `python3` pick a newer interpreter makes pip
# fall back to a Rust build, which is unnecessary and commonly fails on macOS.
find_supported_python() {
    local candidate
    for candidate in "${PYTHON_BIN:-}" python3.11 python3.10 python3; do
        [ -n "$candidate" ] || continue
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if "$candidate" -c \
            'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)' \
            >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "Python 3.10 or 3.11 is required (DeepFilterLib has no wheel for Python 3.12+)." >&2
    echo "Install one of them, then rerun ./setup.sh. You can also set PYTHON_BIN explicitly." >&2
    return 1
}

PYTHON_BIN="$(find_supported_python)"

echo "=== VoiceDenoiser Setup ==="

if [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)' \
    >/dev/null 2>&1; then
    echo "[1/4] venv already exists with a compatible Python. Skipping."
elif [ -d "$VENV_DIR" ]; then
    echo "[1/4] Existing venv uses an incompatible Python. Recreating with $PYTHON_BIN..."
    rm -rf -- "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[1/4] Creating venv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[2/4] NVIDIA GPU detected. Installing CUDA build of PyTorch..."
    "$VENV_PYTHON" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu121
else
    echo "[2/4] No NVIDIA GPU detected. Installing CPU build of PyTorch..."
    "$VENV_PYTHON" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"
fi

echo "[3/4] Installing dependencies..."
"$VENV_PYTHON" -m pip install -r requirements.txt

echo "[4/4] Downloading Resemble Enhance model (approx. 700MB)..."
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    echo "Skipping model download (SKIP_MODEL_DOWNLOAD=1)."
elif "$VENV_PYTHON" -m resemble_enhance.enhancer.download; then
    :
else
    echo "Warning: model download failed or was interrupted."
    echo "Dependencies are ready; rerun ./setup.sh later to resume the model download."
fi

echo
echo "Setup complete. Run ./run.sh to launch."
