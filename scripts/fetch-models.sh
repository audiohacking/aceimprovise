#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
ACE="$ROOT/third_party/acestep.cpp"
mkdir -p "$MODELS"
if [[ -x "$ACE/models.sh" ]]; then
  (cd "$ACE" && ./models.sh)
  echo "Models downloaded into $ACE/models/"
  echo "Symlink or copy to $MODELS when inference is wired."
else
  echo "Run from aceimprovise root after submodule init."
  exit 1
fi
