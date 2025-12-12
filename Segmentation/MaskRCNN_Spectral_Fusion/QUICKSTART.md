# QUICK START GUIDE
## Ultimate Spectral Fusion Solar Panel Detector

### 🚀 Fast Track (Automated)

Run the complete pipeline in one command:

```bash
cd "d:\Projects\Solar Detection\Segmentation\MaskRCNN_Spectral_Fusion"
python run_complete_pipeline.py
```

This will automatically:
1. ✅ Train the model (30 epochs, ~2-3 hours)
2. ✅ Run inference on all images
3. ✅ Generate evaluation report

---

### 📋 Step-by-Step (Manual)

#### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or install PyTorch separately with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy pandas matplotlib scikit-learn scikit-image scipy
```

#### Step 1: Pre-compute Masks (ONE-TIME)

```bash
python precompute_masks.py
```

**IMPORTANT**: This must be run ONCE before training!

**What happens:**
- Runs 6-channel spectral analysis on all 2,500 images
- Generates ultra-high-confidence masks (0.85+ threshold)
- Caches results to disk (~50-100 MB)
- **This takes 10-15 minutes but saves 6 days of training time!**

**Expected output:**
```
Computing masks: 100%|████████████| 2497/2497 [12:34<00:00, 3.31it/s]
✅ Masks cached to: spectral_model/mask_cache/masks_conf85.pkl
Cache size: 67.3 MB

Training samples with high-conf detections: 1834
Validation samples with high-conf detections: 456
```

**Time:** ~10-15 minutes (one-time only!)

#### Step 2: Train the Model

```bash
python train_spectral_fusion.py
```

**What happens:**
- Loads 3,000 images from Google Maps Static API
- Generates ultra-high-confidence pseudo-masks (0.85+ threshold)
- Trains Enhanced Mask R-CNN for 30 epochs
- Saves best model based on validation loss
- Logs everything to `spectral_model/logs/`

**Expected output:**
```
Epoch 1 [10/600] Loss: 0.8234 | loss_classifier: 0.2145 | ...
...
Epoch 30 Summary:
  Average Loss: 0.3456
  🏆 New best model! Loss: 0.3456

✅ Training completed!
   Best model: Epoch 25, Loss: 0.3321
```

**Time:** ~2-3 hours on GPU (CUDA), ~8-12 hours on CPU (now that masks are cached!)

#### Step 3: Run Inference

```bash
python inference_spectral_fusion.py
```

**What happens:**
- Loads trained model from `spectral_model/weights/final_model.pth`
- Processes all 3,000 images
- Generates masks, counts panels, computes metrics
- Performs quality analysis
- Saves JSON, CSV, visualizations, GeoJSON

**Expected output:**
```
[10/3000] Detected: 7 | Accuracy: 92.0%
[20/3000] Detected: 14 | Accuracy: 93.5%
...
INFERENCE SUMMARY
  Total samples processed: 3000
  Accuracy: 95.2%
  
✅ Inference completed! Results saved to: spectral_output/
```

**Time:** ~30-60 minutes

#### Step 4: Generate Evaluation Report

```bash
python evaluate_model.py
```

**What happens:**
- Computes metrics (accuracy, precision, recall, F1)
- Generates confusion matrix
- Analyzes confidence distributions
- Compares with baseline (if available)
- Performs error analysis

**Expected output:**
```
PERFORMANCE METRICS
  accuracy: 0.9520
  precision: 0.9345
  recall: 0.9187
  f1_score: 0.9265
  
✅ Evaluation completed! All outputs saved to: evaluation/
```

**Time:** ~1-2 minutes

---

### 📂 Output Structure

After running the complete pipeline:

```
MaskRCNN_Spectral_Fusion/
│
├── spectral_model/              ← TRAINING OUTPUTS
│   ├── weights/
│   │   ├── best_model_epoch_25.pth
│   │   └── final_model.pth     ← USE THIS FOR INFERENCE
│   └── logs/
│       ├── training_log.txt
│       ├── training_history.csv
│       └── training_curves.png  ← CHECK THIS FIRST
│
├── spectral_output/             ← INFERENCE OUTPUTS
│   ├── detection_results.csv    ← MAIN RESULTS
│   ├── json_records/            ← Individual JSON files
│   ├── visualizations/          ← Visual overlays
│   └── geojson/                 ← Polygon exports
│
└── evaluation/                  ← EVALUATION OUTPUTS
    ├── metrics.json             ← KEY METRICS
    ├── confusion_matrix.png
    ├── confidence_distribution.png
    ├── error_analysis.txt
    └── comparison_plot.png      ← If baseline available
```

---

### 🎯 What to Check First

1. **Training Curves** (`spectral_model/logs/training_curves.png`)
   - Loss should decrease steadily
   - Validation detections should increase
   - LR should decay properly

2. **Metrics** (`evaluation/metrics.json`)
   - Target: Accuracy ≥ 95%
   - Precision & Recall ≥ 90%
   - False Positive Rate < 5%

3. **Sample Visualizations** (`spectral_output/visualizations/`)
   - Check if masks align with actual panels
   - Verify confidence scores make sense
   - Look for false positives/negatives

4. **Detection Results CSV** (`spectral_output/detection_results.csv`)
   - Open in Excel or pandas
   - Filter by confidence, panel_count, etc.
   - Spot-check predictions vs labels

---

### 🔧 Common Adjustments

#### Lower Accuracy (<90%)

**Try:**
```python
# In train_spectral_fusion.py CONFIG:
'confidence_threshold': 0.80,  # Lower from 0.85
'num_epochs': 40,              # More training
```

#### Too Many False Positives

**Try:**
```python
# In inference_spectral_fusion.py CONFIG:
'confidence_threshold': 0.75,  # Raise from 0.70
'min_panel_area': 150,         # Raise from 100
```

#### Out of Memory (OOM)

**Try:**
```python
# In train_spectral_fusion.py CONFIG:
'batch_size': 2,               # Lower from 4
'num_workers': 0,              # Single-threaded
```

#### Training Too Slow

**Option 1: Use fewer samples**
```python
# Edit spectral_dataset.py create_dataloaders()
# Add after loading CSV:
df = df.sample(n=500, random_state=42)  # Use 500 samples
```

**Option 2: Reduce epochs**
```python
# In train_spectral_fusion.py CONFIG:
'num_epochs': 15,  # Half the epochs
```

---

### 📊 Expected Results

| Metric | Target | Typical Range |
|--------|--------|---------------|
| **Accuracy** | ≥ 95% | 92-97% |
| **Precision** | ≥ 90% | 88-95% |
| **Recall** | ≥ 90% | 86-94% |
| **F1-Score** | ≥ 90% | 87-94% |
| **False Positive Rate** | < 5% | 3-8% |
| **False Negative Rate** | < 10% | 6-14% |

---

### 🐛 Troubleshooting

#### "Model file not found"
```bash
# You need to train first
python train_spectral_fusion.py
```

#### "CSV file not found"
```
Check path: d:\Projects\Solar Detection\Data Analytics\EI_train_data(Sheet1).csv
```

#### "Images not found"
```
Check path: d:\Projects\Solar Detection\Data Analytics\Google_MapStaticAPI\images\
Should contain *.png files (0001.png, 0001_1.png, etc.)
```

#### "CUDA out of memory"
```python
# Lower batch size in CONFIG
'batch_size': 2  # or even 1
```

#### "No detections produced"
```python
# Lower confidence threshold
'confidence_threshold': 0.60  # in inference_spectral_fusion.py
```

---

### 📈 Monitoring Training

Watch the log file in real-time:

```bash
# Windows PowerShell
Get-Content "spectral_model\logs\training_log.txt" -Wait

# Or open in text editor and refresh periodically
notepad "spectral_model\logs\training_log.txt"
```

Check GPU usage:

```bash
# If CUDA available
nvidia-smi

# Should show ~80-95% GPU utilization during training
```

---

### 🎓 Understanding the Output

**Sample JSON Record Explained:**

```json
{
  "sample_id": "0001",
  "has_solar_detected": true,          ← Model prediction
  "has_solar_label": 1,                ← Ground truth label
  "confidence": 0.923,                 ← Max detection confidence
  "panel_count": 12,                   ← Individual panels detected
  "array_count": 2,                    ← Panel arrays (clusters)
  "total_area_m2": 45.67,             ← Total area covered
  "capacity_kw": 9.134,               ← Estimated capacity
  "qc_status": "VERIFIABLE",          ← Image quality assessment
  "prediction_correct": true,          ← Correct prediction?
  "reason_codes": [                    ← Why detected?
    "high_confidence_features",
    "spectral_signature_match",
    "module_grid"
  ],
  "detection_reasoning": "...",        ← Human-readable explanation
  "image_quality": {                   ← Quality metrics
    "resolution_score": 1.0,
    "clarity_score": 0.823,
    "occlusion_percent": 8.2
  },
  "spectral_analysis": {               ← Spectral validation
    "spectral_detections": 12,
    "spectral_confidence_avg": 0.887
  }
}
```

---

### ✅ Success Checklist

After running the pipeline, verify:

- [ ] Training completed all 30 epochs
- [ ] `final_model.pth` exists in `spectral_model/weights/`
- [ ] Training curves show decreasing loss
- [ ] Inference processed all 3,000 images
- [ ] `detection_results.csv` has 3,000 rows
- [ ] Accuracy in `metrics.json` ≥ 90%
- [ ] Visualizations look reasonable
- [ ] No major errors in logs

---

### 🚀 Next Steps

Once you have good results:

1. **Fine-tune hyperparameters** for your specific use case
2. **Adjust confidence thresholds** based on precision/recall trade-off
3. **Export GeoJSON** for GIS integration
4. **Deploy model** for production inference
5. **Compare with baseline** MaskRCNN_Solar results

---

### 📞 Support

If issues persist:
1. Check `spectral_model/logs/training_log.txt` for errors
2. Review `evaluation/error_analysis.txt` for failure patterns
3. Inspect sample visualizations for quality issues
4. Verify image paths and CSV format match expected structure

---

**Version**: 1.0  
**Last Updated**: December 9, 2025  
**Status**: Ready for Training ✅
