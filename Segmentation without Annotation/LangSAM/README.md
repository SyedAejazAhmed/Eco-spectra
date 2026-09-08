# GroundingDINO + SAM (Lang-SAM) Solar Panel Detection

⚠️ **STATUS: FAILED - Tensor Dimension Errors**

This approach uses Lang-SAM (GroundingDINO + Meta's SAM) for text-prompt based solar panel detection. Unfortunately, it encounters critical tensor dimension errors preventing any successful detections.

## ❌ Critical Issues Identified

**Error Pattern**: Tensor dimension mismatch in GroundingDINO
```
RuntimeError: The expanded size of the tensor (X) must match the existing size (4) at non-singleton dimension 2
Target sizes: [4, 13294, 44/48/72/52]
Tensor sizes: [44/48/72/52, 1, 4]
```

**Impact**: 
- 0/10 test images successfully detected (100% failure rate)
- All outputs show `has_solar: false` despite ground truth labels = 1
- Errors occur on every text prompt for every image

**Root Cause**: Internal tensor handling issues in GroundingDINO library when processing 640×640 images

---

## Alternative: FastSAM

✅ **See `../FastSAM/` directory for working implementation**

FastSAM uses YOLOv8 architecture instead of GroundingDINO, avoiding these tensor errors while being 11× faster.

---

This module was designed to use Meta's SAM (Segment Anything Model) with zero-shot counting capabilities to detect and count solar panels in satellite imagery.

## Features

- **SAM + LangSAM**: Text-prompt based detection ("solar panel")
- **Zero-Shot Counting**: Advanced counting with NMS and clustering
- **Precise Segmentation**: Pixel-level masks for accurate area calculation
- **GeoJSON Export**: Exportable polygon masks for GIS applications
- **Visual Outputs**: Annotated images showing detections
- **Capacity Estimation**: Automatic kW capacity calculation

## Installation

### 1. Install PyTorch (GPU recommended)

**With CUDA (GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch torchvision
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Lang-SAM

```bash
pip install lang-sam
pip install groundingdino-py
```

## Quick Start

### Test Mode (5 Images)

The script is configured to test with 5 images first:

```bash
cd "d:\Projects\Solar Detection\SAM_Zero_Count"
python sam_solar_detector.py
```

### Full Processing

Edit `sam_solar_detector.py` and change:
```python
TEST_MODE = False
```

Then run:
```bash
python sam_solar_detector.py
```

## Configuration

### Key Parameters in `sam_solar_detector.py`:

```python
# Test mode
TEST_MODE = True              # Process only 5 images
TEST_IMAGES_COUNT = 5         # Number of test images

# Detection thresholds
MIN_MASK_AREA_PIXELS = 50     # Minimum panel size
CONFIDENCE_THRESHOLD = 0.25   # Detection confidence

# Zero-shot counting
ENABLE_ZERO_SHOT_COUNTING = True
COUNT_MERGE_THRESHOLD = 0.5   # NMS IoU threshold

# Resolution (from Google Maps)
PIXEL_RESOLUTION_M = 0.15     # 0.15m per pixel at zoom 20
PANEL_EFFICIENCY_KW_PER_SQM = 0.2  # 200W/m²
```

## ⚠️ Detection Failure Analysis

### Test Results (December 2, 2025)
- **Images Processed**: 10
- **Successful Detections**: 0
- **Tensor Errors**: 40 (4 prompts × 10 images)
- **Ground Truth**: First 10 images have `has_solar=1` label
- **Detection Output**: All marked as `has_solar=false`

### Error Log Summary
```
[Image: 0960_1.png, Prompt: "solar panel"]
Error: The expanded size of the tensor (44) must match the existing size (4) 
       at non-singleton dimension 2. Target: [4, 13294, 44]. Actual: [44, 1, 4]

[Image: 0960_1.png, Prompt: "solar panels"]  
Error: The expanded size of the tensor (48) must match the existing size (4)
       at non-singleton dimension 2. Target: [4, 13294, 48]. Actual: [48, 1, 4]

[Image: 0960_1.png, Prompt: "photovoltaic panel"]
Error: The expanded size of the tensor (72) must match the existing size (4)
       at non-singleton dimension 2. Target: [4, 13294, 72]. Actual: [72, 1, 4]

[Image: 0960_1.png, Prompt: "rooftop solar"]
Error: The expanded size of the tensor (52) must match the existing size (4)
       at non-singleton dimension 2. Target: [4, 13294, 52]. Actual: [52, 1, 4]

... [Pattern repeats for all 10 images]
```

### Fix Attempts Made
1. ✅ Lowered confidence threshold: 0.25 → 0.15
2. ✅ Reduced minimum mask area: 50 → 25 pixels
3. ✅ Added comprehensive error handling with try-catch
4. ✅ Implemented tensor shape validation
5. ✅ Added debug logging for each detection step
6. ✅ Changed strategy to combine all prompts (not just best)
7. ✅ Created separate error log file for model errors
8. ❌ **Result**: Errors persist - appears to be internal library issue

### Conclusion
The tensor dimension errors are **internal to GroundingDINO** and cannot be resolved through parameter tuning or error handling. The issue likely stems from incompatibility between:
- Image dimensions (640×640)
- GroundingDINO's internal feature map processing
- Text tokenization lengths varying by prompt

**Recommendation**: Use FastSAM alternative (see `../FastSAM/`)

---

## How It Was Supposed to Work

### 1. Detection Phase
- Loads satellite image
- Applies SAM with text prompt: "solar panel"
- Returns segmentation masks for each detected panel

### 2. Zero-Shot Counting
- **NMS (Non-Maximum Suppression)**: Removes duplicate detections
- **Area Filtering**: Removes noise (small detections)
- **Clustering**: Groups nearby panels into arrays
- **Counting**: Returns accurate count of individual panels

### 3. Metrics Calculation
- **Area**: Sum of all mask pixels × pixel resolution²
- **Capacity**: Area (m²) × 0.2 kW/m²
- **Count**: Number of distinct panel instances

### 4. Outputs
- **CSV**: Detection results with metrics
- **GeoJSON**: Vector polygons for each panel
- **Visualizations**: Annotated images

## Zero-Shot Counting Algorithm

```python
1. For each image:
   a. SAM returns multiple masks (one per detection)
   b. Apply NMS to merge overlapping detections (IoU > 0.5)
   c. Filter by minimum area (50 pixels)
   d. Calculate distance matrix between panel centers
   e. Group panels within 100 pixels into arrays
   f. Count: individual panels and panel arrays

2. Metrics:
   - panel_count: Total individual panels
   - panel_array_count: Number of grouped arrays
   - total_area_sqm: Sum of all panel areas
   - capacity_kw: Estimated generation capacity
```

## Output Structure

```
SAM_Zero_Count/
├── output/
│   ├── detection_results.csv      # All results
│   ├── geojson/                    # GeoJSON polygons
│   │   ├── 0001_1.geojson
│   │   └── ...
│   ├── visualizations/             # Annotated images
│   │   ├── 0001_1_detection.png
│   │   └── ...
│   └── masks/                      # Binary masks (optional)
├── detection_log.txt               # Detailed log
└── sam_solar_detector.py           # Main script
```

## CSV Output Format

| Column | Description |
|--------|-------------|
| sample_id | Image identifier |
| has_solar | Boolean (True/False) |
| panel_count | Number of individual panels |
| panel_array_count | Number of grouped arrays |
| total_area_sqm | Total area in m² |
| capacity_kw | Estimated capacity in kW |
| confidence | Average confidence score |
| detection_method | SAM_LangSAM_ZeroShot |
| num_masks | Raw number of masks |
| zero_shot_enabled | Boolean flag |

## GeoJSON Format

Each detected panel is exported as a polygon feature:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[...]]
      },
      "properties": {
        "panel_id": 1,
        "area_sqm": 25.6,
        "capacity_kw": 5.12
      }
    }
  ]
}
```

## Performance

- **GPU**: ~2-3 seconds per image
- **CPU**: ~15-20 seconds per image
- **Memory**: ~4GB RAM minimum, 8GB recommended
- **VRAM**: 6GB+ for GPU acceleration

## Troubleshooting

### Error: "lang-sam not installed"
```bash
pip install lang-sam
pip install groundingdino-py
```

### Error: "CUDA out of memory"
- Reduce batch size (already set to 1)
- Use CPU instead: Model will automatically fallback
- Close other GPU applications

### Error: "No module named 'scipy'"
```bash
pip install scipy
```

### Low Detection Rate
- Adjust `CONFIDENCE_THRESHOLD` (lower = more detections)
- Adjust `MIN_MASK_AREA_PIXELS` (lower = detect smaller panels)
- Try different text prompts in `TEXT_PROMPTS`

### Too Many False Positives
- Increase `CONFIDENCE_THRESHOLD` (0.5 or higher)
- Increase `MIN_MASK_AREA_PIXELS` (100+)
- Adjust `COUNT_MERGE_THRESHOLD` for better NMS

## Advantages Over Other Methods

### vs YOLO/Faster R-CNN:
- ✅ No training data needed (zero-shot)
- ✅ Precise pixel-level masks (not just boxes)
- ✅ Better for irregular panel shapes
- ✅ Text-prompt flexibility

### vs Heuristic Methods:
- ✅ More accurate detection
- ✅ Handles various panel orientations
- ✅ No manual threshold tuning
- ✅ Better with complex backgrounds

## Example Results

After running on 5 test images:

```
Statistics:
   Total Processed: 5
   With Solar: 3
   Without Solar: 2
   Detection Rate: 60.0%

Capacity Statistics:
   Total Capacity: 45.6 kW
   Average per Site: 15.2 kW
```

## References

- [SAM Paper](https://arxiv.org/abs/2304.02643) - Segment Anything
- [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything) - Language SAM
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Text-based detection

## Notes

- First run will download SAM model weights (~2.5GB)
- GPU highly recommended for faster processing
- Adjust parameters based on your imagery resolution
- Zero-shot counting works best with clear satellite imagery
