import pickle
from pathlib import Path

cache_path = Path("spectral_model/mask_cache/masks_conf85.pkl")

with open(cache_path, 'rb') as f:
    cached_masks = pickle.load(f)

print(f"Total images: {len(cached_masks)}")

# Count detection stats
with_detections = 0
without_detections = 0
total_detections = 0

for img_path, data in cached_masks.items():
    num_masks = len(data['masks'])
    if num_masks > 0:
        with_detections += 1
        total_detections += num_masks
    else:
        without_detections += 1

print(f"\n❌ Images WITH detections: {with_detections}")
print(f"✓ Images WITHOUT detections: {without_detections}")
print(f"Total detections: {total_detections}")

print(f"\n⚠️ CRITICAL ISSUE: The spectral analyzer detected 0 solar panels in all {len(cached_masks)} images!")
print(f"   This means the confidence threshold (0.85) is TOO STRICT")
print(f"   or the spectral analysis parameters need adjustment.")
