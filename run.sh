#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "venv not found. Run ./setup.sh first."
    exit 1
fi

venv/bin/python app.py
