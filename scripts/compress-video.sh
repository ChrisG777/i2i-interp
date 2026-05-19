#!/usr/bin/env bash
# Compress an animation MP4 (e.g. PowerPoint export) for web use.
# Default workflow: replace sources/<name>.mp4 with a new uncompressed export,
# then run this script with no args to refresh every static/*.mp4 from its
# matching sources/*.mp4. Pass explicit paths to compress a single file.
#
# Usage:
#   scripts/compress-video.sh                  # refresh all sources/*.mp4 -> static/
#   scripts/compress-video.sh <input> [output] # compress one file
#
# This script does NOT produce Twitter/Tweetfully-compatible MP4s. Every
# automated permutation tried (silent AAC, mp42 brand, BT.709 VUI tags, x264
# colorprim) still hit "greyed out" rejections in the upload UI. Compress
# Twitter copies manually in HandBrake / QuickTime / your tool of choice and
# drop them in twitter/ (gitignored) for upload.
#
# Tuning knobs (override via env):
#   CRF=30      H.264 quality (lower = bigger/sharper; 28-32 is a good band)
#   FPS=30      output frame rate cap
#   PRESET=slow x264 preset (slow/medium/fast)

set -euo pipefail

CRF="${CRF:-30}"
FPS="${FPS:-30}"
PRESET="${PRESET:-slow}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

compress_one() {
  local input="$1"
  local output="$2"
  local tmp
  tmp="$(mktemp -t compress-video).mp4"
  trap 'rm -f "$tmp"' RETURN

  echo "  $input -> $output (crf=$CRF fps=$FPS preset=$PRESET)"
  ffmpeg -y -hide_banner -loglevel error -stats \
    -i "$input" \
    -c:v libx264 -crf "$CRF" -preset "$PRESET" -pix_fmt yuv420p \
    -vf "fps=$FPS" \
    -an \
    -movflags +faststart \
    "$tmp"

  mkdir -p "$(dirname "$output")"
  mv "$tmp" "$output"

  local in_size out_size
  in_size=$(stat -f%z "$input")
  out_size=$(stat -f%z "$output")
  awk -v i="$in_size" -v o="$out_size" 'BEGIN {
    fmt = "%.1f %s";
    split("B KiB MiB GiB", u);
    pct = o / i * 100;
    iv = i; for (k = 1; k <= 4 && iv >= 1024; k++) iv /= 1024;
    in_h = sprintf(fmt, iv, u[k]);
    ov = o; for (k = 1; k <= 4 && ov >= 1024; k++) ov /= 1024;
    out_h = sprintf(fmt, ov, u[k]);
    printf "  size: %s -> %s (%.1f%%)\n", in_h, out_h, pct;
  }'
}

if [[ $# -eq 0 ]]; then
  shopt -s nullglob
  sources=(sources/*.mp4)
  if [[ ${#sources[@]} -eq 0 ]]; then
    echo "No sources/*.mp4 to compress." >&2
    exit 1
  fi
  for src in "${sources[@]}"; do
    name="$(basename "$src")"
    compress_one "$src" "static/$name"
  done
elif [[ $# -le 2 ]]; then
  input="$1"
  output="${2:-static/$(basename "$input")}"
  compress_one "$input" "$output"
else
  echo "Usage: $0 [<input> [<output>]]" >&2
  exit 2
fi
