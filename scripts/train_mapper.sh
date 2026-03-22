#!/usr/bin/env bash
set -euo pipefail

uv run train_mapper/main.py --config train_mapper/config.yaml
