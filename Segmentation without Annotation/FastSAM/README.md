# FastSAM Solar Panel Detection

Fast Segment Anything Model implementation for solar panel detection from satellite imagery.

## 🎯 Overview

FastSAM is a CNN-based Segment Anything Model that achieves comparable performance to Meta's SAM at 50× higher speed. This implementation uses FastSAM with text-guided prompts to detect and segment solar panels in satellite images.

## 🚀 Key Features

- **Speed**: ~40ms per image (vs 456ms for original SAM)
- **Efficiency**: 68M parameters (vs 615M for SAM-H)
- **Memory**: 2.6GB RAM usage (vs 7GB for SAM-H)
- **Architecture**: YOLOv8-based segmentation
- **Text-guided**: Uses CLIP for natural language prompts

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory
- 8GB+ RAM

### Dependencies
Install all dependencies:
```bash
pip install -r requirements_fastsam.txt
```

Or install individually:
```bash
pip install torch torchvision
pip install ultralytics fastsam
pip install opencv-python pillow pandas numpy scipy
pip install git+https://github.com/openai/CLIP.git
```

### Model Weights
The FastSAM-x checkpoint will be automatically downloaded on first run:
- **Model**: FastSAM-x.pt (YOLOv8x-based)
- **Size**: ~140MB
- **Source**: Ultralytics Hub

Alternatively, manually download:
```bash
# Create weights directory
mkdir weights
cd weights

# Download FastSAM-x
wget https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing
```

## 🔧 Configuration

Edit `fastsam_solar_detector.py` to adjust parameters:

```python
# Detection parameters
CONFIDENCE_THRESHOLD = 0.3      # Lower = more detections (range: 0.0-1.0)
IOU_THRESHOLD = 0.7            # NMS overlap threshold
MIN_MASK_AREA_PIXELS = 25      # Minimum panel size in pixels

# Mode settings
DEBUG_MODE = True               # Detailed logging
TEST_MODE = True                # Process only TEST_IMAGES_COUNT images
TEST_IMAGES_COUNT = 10         # Number of test images

# Text prompts for detection
TEXT_PROMPTS = [
    "solar panel",
    "solar panels", 
    "photovoltaic panel",
    "rooftop solar",
    "PV panel",
    "solar array"
]
```

## 🏃 Usage

### Basic Detection
Process test images (first 10):
```bash
python fastsam_solar_detector.py
```

### Full Dataset
Edit script to disable test mode:
```python
TEST_MODE = False  # Process all images
```

Then run:
```bash
python fastsam_solar_detector.py
```

### Expected Runtime
- **Test Mode** (10 images): ~5-10 seconds
- **Full Dataset** (3000 images): ~2-4 minutes

## 📊 Output Structure

```
FastSAM/
├── output/
│   ├── detection_results.csv          # Aggregated results
│   ├── json_records/                  # Individual JSON per image
│   │   ├── 0960_1.json
│   │   ├── 0988_1.json
│   │   └── ...
│   ├── visualizations/                # Detection overlays
│   │   ├── 0960_1_detection.png
│   │   └── ...
│   ├── masks/                         # Binary masks
│   │   └── ...
│   └── geojson/                       # Polygon features
│       └── ...
├── detection_log_fastsam.txt          # Processing log
└── FastSAM_model_logs.txt             # Model-specific errors
```

## 📄 Output Format

### JSON Record Structure
Each image generates a comprehensive JSON:

```json
{
  "sample_id": "0960",
  "lat": 21.198934,
  "lon": 72.797171,
  "has_solar": true,
  "has_solar_label": 1,
  "confidence": 0.75,
  "panel_count_Est": 42,
  "panel_array_count": 3,
  "pv_area_sqm_est": 118.5,
  "capacity_kw_est": 23.7,
  "qc_status": "high_confidence",
  "qc_notes": "high confidence: 0.75, 42 panels detected, moderate panel sizes",
  "bbox_or_mask": {
    "type": "mask",
    "data": [[...]]
  },
  "image_metadata": {
    "filename": "0960_1.png",
    "width": 640,
    "height": 640,
    "pixel_resolution_m": 0.15,
    "detection_timestamp": "2025-12-02T16:45:00"
  },
  "model": "FastSAM",
  "detection_method": "text_prompt_segmentation"
}
```

### CSV Results
Aggregated results in `detection_results.csv`:

| Field | Description |
|-------|-------------|
| sample_id | Sample identifier from filename |
| lat | Latitude from CSV dataset |
| lon | Longitude from CSV dataset |
| has_solar | Detection result (true/false) |
| has_solar_label | Ground truth label (1/0) |
| confidence | Maximum confidence score |
| panel_count_Est | Estimated number of panels |
| panel_array_count | Number of panel clusters |
| pv_area_sqm_est | Total PV area (m²) |
| capacity_kw_est | Estimated capacity (kW) |
| qc_status | Quality control status |
| qc_notes | Quality control notes |

## 🔍 Quality Control Statuses

| Status | Description |
|--------|-------------|
| `no_detection` | No solar panels detected |
| `low_confidence` | Confidence < 0.3 |
| `medium_confidence` | Confidence 0.3-0.5 |
| `high_confidence` | Confidence > 0.5 |

## 🎯 Detection Algorithm

### Pipeline Overview
1. **Load Image**: Read PNG image (640×640)
2. **Everything Mode**: Run FastSAM segmentation
3. **Text Filtering**: Apply text prompts via CLIP
4. **Area Filtering**: Remove masks < MIN_MASK_AREA_PIXELS
5. **NMS**: Remove duplicate detections (IoU > 0.5)
6. **Clustering**: Group nearby panels into arrays
7. **Metrics**: Calculate area, capacity, QC status
8. **Output**: Generate JSON, CSV, visualizations

### Zero-Shot Counting
- **NMS**: Removes overlapping detections (IoU threshold = 0.5)
- **Clustering**: Groups panels within 100px proximity
- **Arrays**: Counts distinct panel clusters

### Area Calculation
```python
pv_area_sqm = mask_pixels × (0.15m/pixel)²
capacity_kw = pv_area_sqm × 0.2 kW/m²
```

## 🐛 Troubleshooting

### Issue: Model Download Fails
**Solution**:
```bash
# Manually download FastSAM-x.pt
mkdir weights
cd weights
# Download from: https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints
```

### Issue: CUDA Out of Memory
**Solutions**:
1. Reduce batch size (edit script: `imgsz=512`)
2. Use CPU mode (edit script: `device='cpu'`)
3. Process fewer images (increase TEST_MODE)

### Issue: No Detections
**Solutions**:
1. Lower confidence threshold: `CONFIDENCE_THRESHOLD = 0.2`
2. Reduce min area: `MIN_MASK_AREA_PIXELS = 10`
3. Add more prompts: `TEXT_PROMPTS += ["solar", "panels"]`
4. Check image quality and solar visibility

### Issue: ImportError for FastSAM
**Solution**:
```bash
pip uninstall fastsam -y
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git
```

### Issue: CLIP Installation Error
**Solution**:
```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## 📈 Performance Metrics

### Speed Comparison
| Model | Speed (ms/image) | Speedup |
|-------|------------------|---------|
| SAM-H | 456 | 1× |
| SAM-B | 230 | 2× |
| FastSAM | 40 | **11.4×** |

### Memory Comparison
| Model | GPU Memory | Parameters |
|-------|------------|------------|
| SAM-H | 7060MB | 615M |
| SAM-B | 4670MB | 136M |
| FastSAM | 2608MB | **68M** |

### Accuracy (COCO)
| Model | AR@1000 | AP@IoU50 |
|-------|---------|----------|
| SAM-H | 45.6 | 15.5 |
| FastSAM | 47.3 | 15.7 |

## 🔬 Technical Details

### Architecture
- **Base**: YOLOv8x segmentation model
- **Training**: 2% of SA-1B dataset (~100k images)
- **Encoder**: YOLOv8 CNN (68M params)
- **Decoder**: FastSAM prompt-guided decoder
- **Text Guidance**: CLIP for text-to-visual matching

### Advantages over GroundingDINO_SAM
1. **No Tensor Errors**: More stable architecture
2. **Faster Processing**: CNN-based vs Transformer-based
3. **Lower Memory**: 3× less GPU memory
4. **Simpler Pipeline**: Single model vs two-stage
5. **Better Maintained**: Active community support

## 📚 References

### Papers
- [FastSAM: Fast Segment Anything](https://arxiv.org/abs/2306.12156)
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)

### Code
- [FastSAM GitHub](https://github.com/CASIA-IVA-Lab/FastSAM)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CLIP](https://github.com/openai/CLIP)

### Demos
- [HuggingFace Demo](https://huggingface.co/spaces/An-619/FastSAM)
- [Replicate Demo](https://replicate.com/casia-iva-lab/fastsam)
- [Colab Demo](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9)

## 📝 Notes

### Comparison with Lang-SAM
- **Lang-SAM Status**: ❌ Failed (tensor dimension errors)
- **FastSAM Status**: 🔄 Ready for testing
- **Key Difference**: FastSAM uses YOLOv8 instead of GroundingDINO
- **Expected Outcome**: Should avoid tensor errors seen in Lang-SAM

### Next Steps
1. Run detection on test images
2. Compare results with ground truth labels
3. Calculate accuracy metrics (precision, recall, F1)
4. Document results in main README
5. If successful, process full 3000-image dataset

## 🤝 Contributing

When modifying this implementation:
1. Test on small batch first (TEST_MODE=True)
2. Document parameter changes
3. Report detection accuracy
4. Update this README with findings

## 📄 License

Apache 2.0 (inherited from FastSAM)

---

**Created**: December 2, 2025  
**Author**: Solar Detection Project Team  
**Model**: FastSAM-x (YOLOv8-based)
