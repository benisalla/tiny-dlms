#!/usr/bin/env bash
# Publish every trained model as a GitHub Release, so anyone can download them without training.
# Git keeps the code; the checkpoints (~500 MB) go into the release as files. Each .pt holds the weights,
# the config and the tokenizer, so it is all you need to use a model. The training curves (.metrics.jsonl) go too.
#
#   bash scripts/publish_models.sh               # release "models-v1" from checkpoints/
#   TAG=models-v2 bash scripts/publish_models.sh # a new release after training again
#
# Running it again on an existing release replaces the files that changed (--clobber).
# Needs the GitHub CLI (gh), logged in with write access to the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-models-v1}
DIR=${DIR:-checkpoints}
REPO=${REPO:-benisalla/tiny-dlms}

shopt -s nullglob
files=("$DIR"/*.pt "$DIR"/*.pt.metrics.jsonl)
[ ${#files[@]} -gt 0 ] || { echo "no checkpoints in $DIR"; exit 1; }

# A list of every model with its size and sha256, so a download can be checked.
manifest=$(mktemp -d)/MODELS.txt
( cd "$DIR" && for f in *.pt; do
    printf '%s  %s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$(du -h "$f" | cut -f1)" "$f"
  done ) > "$manifest"

notes="Every trained Rhoa and Mēlon checkpoint. Each .pt holds the weights, the config and the tokenizer.

Download them all:  python scripts/download_models.py
Or one:             python scripts/download_models.py rhoa_web2.pt

Best Rhoa: rhoa_web2.pt (100% bAbI, 72.6% CLINC150, 84.4% Banking77). Best Mēlon: melon.pt (car), melon_maze.pt (maze).
MODELS.txt lists every file with its sha256."

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
    echo "release $TAG exists: uploading ${#files[@]} files (replacing old ones)"
else
    echo "creating release $TAG"
    gh release create "$TAG" -R "$REPO" --title "Trained models ($TAG)" --notes "$notes"
fi
gh release upload "$TAG" -R "$REPO" --clobber "$manifest" "${files[@]}"
echo "done: https://github.com/$REPO/releases/tag/$TAG"
