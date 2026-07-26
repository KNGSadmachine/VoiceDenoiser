#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "venv not found. Run ./setup.sh first."
    exit 1
fi

echo "Starting VoiceDenoiser..."
echo "GUI: http://127.0.0.1:7860"
echo "If the browser does not open automatically, copy the URL above into your browser."
echo "Press Ctrl+C to stop."

exec venv/bin/python -u app.py
