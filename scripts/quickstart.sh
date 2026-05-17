#!/usr/bin/env bash
set -euo pipefail

# Move to project root so all relative paths stay consistent.
cd "$(dirname "$0")/.."

# 1. Train and save all artifacts.
python scripts/train_models.py

# 2. Start the API from inside imdb_analysis because the API imports use
#    `from src...` and expect that package root to be on the Python path.
cd imdb_analysis
uvicorn api.main:app --reload
