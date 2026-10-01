#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 2 ]] || { echo "Usage: $0 baseline.mp4 candidate.mp4" >&2; exit 2; }
baseline=$(ffmpeg -nostdin -v error -i "$1" -map 0:v:0 -an -f framemd5 -)
candidate=$(ffmpeg -nostdin -v error -i "$2" -map 0:v:0 -an -f framemd5 -)
if [[ "$baseline" != "$candidate" ]]; then
    echo "FAIL: decoded video frames or timing differ" >&2
    exit 1
fi
echo "PASS: decoded video frames and timing are identical"
