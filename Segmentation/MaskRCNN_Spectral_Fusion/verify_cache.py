import pickle
import numpy as np
from pathlib import Path

cache_path = Path("spectral_model/mask_cache/masks_conf85.pkl")

print("=" * 80)
print("VERIFYING CACHED MASKS")
print("=" * 80)

# Load cached masks
with open(cache_path, 'rb') as f:
    cached_masks = pickle.load(f)

print(f"\n📦 Cache file: {cache_path}")
print(f"   Size: {cache_path.stat().st_size / (1024*1024):.1f} MB")
print(f"\n📊 Total images cached: {len(cached_masks)}")

# Analyze cached data
has_masks = 0
no_masks = 0
total_masks = 0
mask_sizes = []
confidence_scores = []

for img_path, data in cached_masks.items():
    masks = data['masks']
    boxes = data['boxes']
    confidences = data['confidences']
    
    if len(masks) > 0:
        has_masks += 1
        total_masks += len(masks)
        mask_sizes.extend([mask.sum() for mask in masks])  # Count pixels per mask
        confidence_scores.extend(confidences)
    else:
        no_masks += 1

print(f"\n🎯 Detection statistics:")
print(f"   Images WITH solar panels detected: {has_masks}")
print(f"   Images WITHOUT detections: {no_masks}")
print(f"   Total masks generated: {total_masks}")
print(f"   Average masks per positive image: {total_masks/has_masks:.2f}")

if mask_sizes:
    print(f"\n📏 Mask statistics:")
    print(f"   Average mask area: {np.mean(mask_sizes):.0f} pixels")
    print(f"   Min mask area: {np.min(mask_sizes):.0f} pixels")
    print(f"   Max mask area: {np.max(mask_sizes):.0f} pixels")

if confidence_scores:
    print(f"\n⭐ Confidence statistics:")
    print(f"   Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"   Min confidence: {np.min(confidence_scores):.3f}")
    print(f"   Max confidence: {np.max(confidence_scores):.3f}")
    print(f"   Confidence threshold used: 0.85")

# Sample a few entries
print(f"\n🔍 Sample entries (first 5 with detections):")
count = 0
for img_path, data in cached_masks.items():
    if len(data['masks']) > 0 and count < 5:
        count += 1
        print(f"\n   {Path(img_path).name}:")
        print(f"      Masks: {len(data['masks'])}")
        print(f"      Confidences: {data['confidences']}")
        print(f"      Box shapes: {[box.shape for box in data['boxes']]}")
        
print(f"\n✅ Cache validation complete!")
print(f"   Ready for training with {len(cached_masks)} pre-computed images")
print("=" * 80)
