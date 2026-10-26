#!/usr/bin/env bash
# Download RU–UZ parallel corpora from OPUS (Moses format) into data/raw/<corpus>/.
#
# Usage: scripts/download_data.sh [corpus ...]
#   Default corpora: wikimedia TED2020 QED Tatoeba tldr-pages
#   Others available for ru-uz: Tanzil (religious), GNOME, KDE4 (software UI).
# Env: RAW_DIR (default data/raw)
set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw}"
OPUS_API="https://opus.nlpl.eu/opusapi/"
CORPORA=("$@")
if [ ${#CORPORA[@]} -eq 0 ]; then
  CORPORA=(wikimedia TED2020 QED Tatoeba tldr-pages)
fi

mkdir -p "$RAW_DIR"

# Resolve the latest Moses zip URL for a corpus via the OPUS API.
resolve_url() {
  curl -sfL "${OPUS_API}?corpus=$1&source=ru&target=uz&preprocessing=moses&version=latest" \
    | python3 -c 'import sys, json; c = json.load(sys.stdin)["corpora"]; print(c[0]["url"] if c else "")'
}

for corpus in "${CORPORA[@]}"; do
  dest="$RAW_DIR/$corpus"
  if ls "$dest"/*.ru >/dev/null 2>&1; then
    echo "[skip] $corpus already in $dest"
    continue
  fi
  url="$(resolve_url "$corpus")"
  if [ -z "$url" ]; then
    echo "[warn] $corpus: no ru-uz Moses release on OPUS" >&2
    continue
  fi
  echo "[get ] $corpus <- $url"
  mkdir -p "$dest"
  tmp="$(mktemp -t opus.XXXXXX)"
  curl -fsSL --retry 3 -o "$tmp" "$url"
  unzip -oq "$tmp" -d "$dest"
  rm -f "$tmp"
  wc -l "$dest"/*.ru
done

echo "Done. Raw data in $RAW_DIR"
