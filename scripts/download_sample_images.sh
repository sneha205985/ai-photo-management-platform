#!/usr/bin/env bash
# Download copyright-free sample images for the AI Photo Management Platform.
# Sources: Unsplash, Pexels, Wikimedia Commons (free / public-domain licenses).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHOTOS_DIR="${SCRIPT_DIR}/../data/photos"
mkdir -p "${PHOTOS_DIR}"

download() {
  local url="$1"
  local outfile="$2"
  echo "Downloading ${outfile}..."
  curl -fsSL --retry 3 --retry-delay 2 -L -o "${PHOTOS_DIR}/${outfile}" "${url}"
}

echo "=== People (3) — Unsplash ==="
download "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80" "person_portrait_man_smiling.jpg"
download "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80" "person_portrait_woman_outdoors.jpg"
download "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=800&q=80" "person_portrait_woman_casual.jpg"

echo "=== Travel (3) — Pexels + Unsplash ==="
download "https://images.pexels.com/photos/457878/pexels-photo-457878.jpeg?auto=compress&cs=tinysrgb&w=800" "travel_beach_palm_trees.jpg"
download "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800&q=80" "travel_road_trip_car.jpg"
download "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=800&q=80" "travel_map_camera_passport.jpg"

echo "=== Pets (3) — Pexels ==="
download "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=800" "pet_dog_puppy_grass.jpg"
download "https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg?auto=compress&cs=tinysrgb&w=800" "pet_cat_tabby_window.jpg"
download "https://images.pexels.com/photos/1462637/pexels-photo-1462637.jpeg?auto=compress&cs=tinysrgb&w=800" "pet_dog_golden_portrait.jpg"

echo "=== Receipts / Documents (2) — Wikimedia Commons + Unsplash ==="
download "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg" "receipt_swiss_restaurant.jpg"
download "https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=800&q=80" "document_office_paperwork.jpg"

echo "=== Identical duplicates (2) ==="
cp "${PHOTOS_DIR}/travel_beach_palm_trees.jpg" "${PHOTOS_DIR}/travel_beach_palm_trees_copy.jpg"
cp "${PHOTOS_DIR}/pet_cat_tabby_window.jpg" "${PHOTOS_DIR}/pet_cat_tabby_window_duplicate.jpg"

echo "=== Near-duplicates (2) — resize / crop variants ==="
sips -Z 720 "${PHOTOS_DIR}/travel_road_trip_car.jpg" --out "${PHOTOS_DIR}/travel_road_trip_car_resized.jpg" >/dev/null
sips -c 600 600 "${PHOTOS_DIR}/person_portrait_man_smiling.jpg" --out "${PHOTOS_DIR}/person_portrait_man_smiling_cropped.jpg" >/dev/null

echo ""
echo "Done. Sample images saved to: ${PHOTOS_DIR}"
echo ""
ls -lh "${PHOTOS_DIR}"
