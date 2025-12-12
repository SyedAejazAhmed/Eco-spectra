import pickle
from pathlib import Path

cache_path = Path("spectral_model/mask_cache/masks_conf85.pkl")

print(f"Loading cache from: {cache_path}")
print(f"File size: {cache_path.stat().st_size:,} bytes ({cache_path.stat().st_size / 1024:.1f} KB)")

with open(cache_path, 'rb') as f:
    cached_masks = pickle.load(f)

print(f"\nCache type: {type(cached_masks)}")
print(f"Number of entries: {len(cached_masks)}")

# Check first few entries
print("\nFirst 10 entries:")
for i, (key, value) in enumerate(list(cached_masks.items())[:10]):
    print(f"{i+1}. Key: {Path(key).name if isinstance(key, str) else key}")
    print(f"   Value type: {type(value)}")
    if isinstance(value, dict):
        print(f"   Keys: {value.keys()}")
        print(f"   Masks: {len(value.get('masks', []))}")
        print(f"   Boxes: {len(value.get('boxes', []))}")
        print(f"   Confidences: {value.get('confidences', [])}")
