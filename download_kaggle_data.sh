#!/usr/bin/env bash
set -euo pipefail

if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  read -rsp "KAGGLE_API_TOKEN not set. Enter your Kaggle API token: " KAGGLE_API_TOKEN
  echo
  export KAGGLE_API_TOKEN
fi

if [ -z "$KAGGLE_API_TOKEN" ]; then
  echo "Error: no token provided, use [export KAGGLE_API_TOKEN=<your_new_token>] before running this script"
  exit 1
fi

python3 -m pip install kaggle --break-system-packages --quiet
export PATH="$HOME/.local/bin:$PATH"

mkdir -p data

kaggle datasets download \
  -d gpiosenka/butterfly-images40-species \
  -p data \
  --unzip

echo "Done. Dataset extracted to ./data"