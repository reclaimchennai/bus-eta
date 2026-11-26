#!/usr/bin/env bash
# Download the data archives listed in data/MANIFEST.json from this repo's GitHub Release, verify each sha256 and
# unpack into data/.
#
#   scripts/fetch_data.sh                 everything (~10 GB download)
#   scripts/fetch_data.sh raw-mtc audit   only these assets (names as in MANIFEST.json)
#
# Needs curl, python3, sha256sum or shasum. Set REPO=owner/name to fetch from a fork.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=${REPO:-reclaimchennai/bus-eta}
cd "$ROOT"
TAG=$(python3 -c 'import json; print(json.load(open("data/MANIFEST.json"))["tag"])')
BASE="https://github.com/$REPO/releases/download/$TAG"
DL=${DL:-$ROOT/data/.download}; mkdir -p "$DL"
sum() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
python3 - "$@" <<'EOF' > "$DL/plan.txt"
import json, sys
m = json.load(open("data/MANIFEST.json")); want = set(sys.argv[1:])
for a in m["assets"]:
    if want and a["name"] not in want: continue
    print(a["name"], a["into"], " ".join(f'{f["asset"]}:{f["sha256"]}' for f in a["files"]))
EOF
while read -r name into files; do
  if [ -f "data/$into/.fetched" ]; then echo "$name: already unpacked"; continue; fi
  parts=()
  for fs in $files; do
    f=${fs%%:*}; want=${fs##*:}
    [ -f "$DL/$f" ] && [ "$(sum "$DL/$f")" = "$want" ] || curl -fL --retry 5 -C - -o "$DL/$f" "$BASE/$f"
    got=$(sum "$DL/$f"); [ "$got" = "$want" ] || { echo "sha256 mismatch for $f"; exit 1; }
    parts+=("$DL/$f")
  done
  mkdir -p "data/$into"
  cat "${parts[@]}" | tar -xz -C "data/$into"
  touch "data/$into/.fetched"; echo "$name -> data/$into"
done < "$DL/plan.txt"
echo "done. Downloads kept in $DL (delete to free space)."
