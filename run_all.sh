#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="${1:-data/backup-20260223102103.tar.bz2}"

python -m src.pipeline --snapshot "$SNAPSHOT"
jupyter lab
