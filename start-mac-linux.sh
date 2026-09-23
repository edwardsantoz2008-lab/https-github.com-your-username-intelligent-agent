#!/usr/bin/env bash
set -e
python3 -m venv .venv 2>/dev/null || true
.venv/bin/python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
.venv/bin/python app.py
