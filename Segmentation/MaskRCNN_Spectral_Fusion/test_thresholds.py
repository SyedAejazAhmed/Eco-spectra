import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from spectral_analyzer import UltimateSpectralAnalyzer

# Test on a few solar images
data_dir = Path(r"D:\Projects\Solar Detection\Data Analytics\Google_MapStaticAPI\images")
test_images = [
    "1.0_1.0.png",
    "10.0_1.0.png", 
    "100.0_1.0.png",
    "500.0_1.0.png",
    "1000.0_1.0.png"
]

analyzer = UltimateSpectralAnalyzer()

print("=" * 80)
print("TESTING DIFFERENT CONFIDENCE THRESHOLDS")
print("=" * 80)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]

for img_name in test_images:
    img_path = data_dir / img_name
    if not img_path.exists():
        continue
        
    print(f"\n📷 {img_name}")
    img = cv2.imread(str(img_path))
    
    for threshold in thresholds:
        masks, boxes, confidences, _ = analyzer.segment_from_spectral(img, confidence_threshold=threshold)
        print(f"   threshold={threshold:.2f}: {len(masks)} detections", end="")
        if len(masks) > 0:
            print(f" (conf: {confidences})")
        else:
            print()

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("Use a threshold where you get detections on known solar images.")
print("Start with 0.5-0.6 and adjust based on validation results.")
print("=" * 80)
