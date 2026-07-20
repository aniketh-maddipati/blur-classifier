#!/usr/bin/env bash
set -euo pipefail
trap 'echo ""; echo "!! FAILED at line $LINENO -- nothing further was run. Fix the issue above and re-run."; exit 1' ERR

# ---- EDIT THESE TWO BEFORE RUNNING ----
CARD_NAME="Untitled"
IMPORT_LABEL="card2"
# ----------------------------------------

DATE_TAG="$(date +%Y-%m-%d)"
SOURCE="/Volumes/${CARD_NAME}/DCIM/100MSDCF"
DEST="/Volumes/T7/photo-archive/raw-import-${DATE_TAG}-${IMPORT_LABEL}"
CANDIDATES_OUT="./dataset_candidates_${IMPORT_LABEL}"

echo "== Step 1: Verify source exists =="
if [ ! -d "$SOURCE" ]; then
  echo "ERROR: $SOURCE not found. Currently mounted volumes:"
  ls /Volumes
  exit 1
fi
echo "OK: $SOURCE found."

echo "== Step 2: Copy (excluding macOS sidecar files) =="
mkdir -p "$DEST"
rsync -av --progress --exclude='._*' "$SOURCE/" "$DEST/"

echo "== Step 3: Verify copy is byte-for-byte complete =="
VERIFY_OUTPUT=$(rsync -rc --dry-run --out-format='%n' --exclude='._*' "$SOURCE/" "$DEST/")
if [ -z "$VERIFY_OUTPUT" ]; then
  echo "OK: source and destination match byte-for-byte."
else
  echo "ERROR: differences found -- do not proceed, re-run the copy first:"
  echo "$VERIFY_OUTPUT"
  exit 1
fi

echo "== Step 4: Count real files (sidecars excluded) =="
ARW_COUNT=$(find "$DEST" -maxdepth 1 -iname '*.ARW' ! -name '._*' | wc -l | tr -d ' ')
JPG_COUNT=$(find "$DEST" -maxdepth 1 -iname '*.JPG' ! -name '._*' | wc -l | tr -d ' ')
echo "ARW files: $ARW_COUNT"
echo "JPG files: $JPG_COUNT"

if [ "$ARW_COUNT" -eq 0 ]; then
  echo "ERROR: no ARW files found at all -- something is wrong with the source path."
  exit 1
fi

echo "== Step 5: Check RAW+JPEG pairing -- report, never assume =="
UNPAIRED=0
for arw in "$DEST"/*.ARW; do
  [ -e "$arw" ] || continue
  base="${arw%.ARW}"
  [ -f "${base}.JPG" ] || UNPAIRED=$((UNPAIRED + 1))
done
echo "ARW files with NO matching JPG: $UNPAIRED out of $ARW_COUNT"

if [ "$UNPAIRED" -eq "$ARW_COUNT" ]; then
  echo ""
  echo "*** RAW-ONLY BATCH -- no paired JPEGs exist for any file. ***"
  echo "*** Anything selected from here needs a manual neutral RAW export"
  echo "*** before it can be used with classify.py."
elif [ "$UNPAIRED" -gt 0 ]; then
  echo ""
  echo "*** PARTIAL pairing -- $UNPAIRED files are not paired. Check those individually. ***"
else
  echo "OK: every ARW has a matching JPG."
fi

echo "== Step 6: Run EXIF triage on this batch =="
python triage_dataset.py "$DEST" "$CANDIDATES_OUT"

echo ""
echo "== Done =="
echo "Candidates written to: $CANDIDATES_OUT"
echo "No files were moved into dataset/train or dataset/holdout."
echo "No class labels were assigned. Both remain manual."
