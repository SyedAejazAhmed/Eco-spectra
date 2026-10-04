# Annotated Solar Panel Segmentation

This directory contains supervised semantic-segmentation experiments for detecting solar panels in satellite imagery. Seven baseline architectures were trained and evaluated using the same train, validation, and test split IDs. A proposed UNET++ model was then developed by adding spectral shadow analysis during post-training and evaluated on train, validation, and test splits. Each model directory contains its training notebook, saved weights, predictions, plots, and metric CSV files.

## Models

| Model | Description |
|---|---|
| **UNET** | Encoder-decoder architecture with skip connections that preserve fine-grained panel boundaries. |
| **Segnet** | Encoder-decoder network that reuses pooling indices to recover spatial structure efficiently. |
| **DeepLabV3** | Atrous convolutions and spatial-pyramid pooling capture context at multiple receptive-field sizes. |
| **Attention UNET** | U-Net with attention gates that suppress irrelevant background features and focus on panel regions. |
| **UNET++** | Nested U-Net with dense skip pathways that reduce the semantic gap between encoder and decoder features. |
| **Proposed UNET++ + Spectral Analysis** | UNET++ post-trained with shadow-aware spectral preprocessing, a shadow-mask input channel, synthetic shadow augmentation, shadow-weighted BCE + Dice loss, and extended evaluation metrics. |
| **Proposed UNET ++ with Solar Shadow Filters** | UNET++ trained with solar shadow filtering and evaluated with pixel-wise and connected-component instance metrics. |
| **DFANET** | Lightweight feature-aggregation architecture designed to combine low-level detail with deeper semantic features. |
| **SegFormer** | Transformer-based encoder with a lightweight decoder for multi-scale segmentation features. |

## Evaluation Metrics

The values below are class-averaged segmentation metrics, reported without the `pixel_` prefix used in the CSV files:

- **Accuracy**: proportion of correctly classified pixels.
- **Precision**: proportion of predicted foreground pixels that are correct.
- **Recall**: proportion of annotated foreground pixels that are detected.
- **F1 Score**: harmonic mean of precision and recall.
- **Dice**: overlap score between prediction and annotation.
- **IoU**: intersection over union between prediction and annotation.
- **Dice Loss**: $1 - \text{Dice}$; lower values are better.

The summary CSV stores a model loss produced by the training objective. The tables below show the requested Dice Loss calculated consistently as `1 - Dice` from the reported evaluation Dice value. **Bold values are the best value in each column for that split.** Higher is better for every metric except Dice Loss.

## Training Metrics

| Model | Accuracy | Precision | Recall | F1 Score | Dice | IoU | Dice Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNET | 0.9873 | 0.9278 | 0.9440 | 0.9357 | 0.9357 | 0.8847 | 0.0643 |
| Segnet | 0.9803 | 0.8986 | 0.8982 | 0.8984 | 0.8984 | 0.8281 | 0.1016 |
| DeepLabV3 | 0.9782 | 0.8709 | 0.9246 | 0.8956 | 0.8956 | 0.8239 | 0.1044 |
| Attention UNET | 0.9816 | 0.9065 | 0.9030 | 0.9047 | 0.9047 | 0.8373 | 0.0953 |
| **UNET++** | **0.9910** | **0.9519** | **0.9559** | **0.9539** | **0.9539** | **0.9148** | **0.0461** |
| DFANET | 0.9488 | 0.4744 | 0.5000 | 0.4869 | 0.4869 | 0.4744 | 0.5131 |
| SegFormer | 0.9709 | 0.8554 | 0.8385 | 0.8467 | 0.8467 | 0.7594 | 0.1533 |
| UNET++ + post training Spectral Analysis | 0.9794 | 0.8138 | 0.7941 | 0.8038 | 0.7970 | 0.6720 | 0.2030 |
| Proposed UNET ++ with Solar Shadow Filters | 0.9854 | 0.9197 | 0.9327 | 0.9261 | 0.9261 | 0.8695 | 0.0739 |

## Validation Metrics

| Model | Accuracy | Precision | Recall | F1 Score | Dice | IoU | Dice Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNET | 0.9752 | 0.8740 | **0.8852** | 0.8795 | 0.8795 | 0.8015 | 0.1205 |
| Segnet | 0.9730 | 0.8710 | 0.8586 | 0.8647 | 0.8647 | 0.7819 | 0.1353 |
| DeepLabV3 | 0.9638 | 0.8193 | 0.8300 | 0.8245 | 0.8245 | 0.7321 | 0.1755 |
| Attention UNET | 0.9764 | 0.8863 | 0.8782 | 0.8822 | 0.8822 | 0.8052 | 0.1178 |
| **UNET++** | **0.9768** | **0.8908** | 0.8771 | **0.8838** | **0.8838** | **0.8075** | **0.1162** |
| DFANET | 0.9464 | 0.4732 | 0.5000 | 0.4862 | 0.4862 | 0.4732 | 0.5138 |
| SegFormer | 0.9667 | 0.8402 | 0.8237 | 0.8317 | 0.8317 | 0.7409 | 0.1683 |
| UNET++ + post training Spectral Analysis | **0.9775** | 0.7962 | 0.7654 | 0.7805 | 0.7690 | 0.6400 | 0.2310 |
| Proposed UNET ++ with Solar Shadow Filters | 0.9742 | 0.8702 | 0.8788 | 0.8744 | 0.8744 | 0.7946 | 0.1256 |

## Test Metrics

| Model | Accuracy | Precision | Recall | F1 Score | Dice | IoU | Dice Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNET | 0.9751 | 0.8707 | **0.8789** | 0.8747 | 0.8747 | 0.7951 | 0.1253 |
| Segnet | 0.9740 | 0.8748 | 0.8541 | 0.8641 | 0.8641 | 0.7813 | 0.1359 |
| DeepLabV3 | 0.9651 | 0.8217 | 0.8251 | 0.8234 | 0.8234 | 0.7311 | 0.1766 |
| **Attention UNET** | **0.9768** | 0.8876 | 0.8717 | **0.8795** | **0.8795** | **0.8017** | **0.1205** |
| **UNET++** | 0.9766 | **0.8881** | 0.8684 | 0.8780 | 0.8780 | 0.7997 | 0.1220 |
| DFANET | 0.9482 | 0.4741 | 0.5000 | 0.4867 | 0.4867 | 0.4741 | 0.5133 |
| SegFormer | 0.9681 | 0.8439 | 0.8210 | 0.8320 | 0.8320 | 0.7415 | 0.1680 |
| UNET++ + post training Spectral Analysis | **0.9800** | 0.8051 | 0.7866 | 0.7958 | 0.7883 | 0.6608 | 0.2117 |
| Proposed UNET ++ with Solar Shadow Filters | 0.9752 | 0.8745 | 0.8722 | 0.8734 | 0.8734 | 0.7934 | 0.2066 |

### Best-Metric Summary

The proposed UNET++ + Spectral Analysis model achieves the best **Accuracy** among all compared models on both the validation split (0.9775) and the test split (0.9800). The original UNET++ remains strongest on the other validation metrics, while Attention UNET and the original UNET++ lead several test metrics. No proposed-model value is the best training metric in the current comparison. The **Proposed UNET ++ with Solar Shadow Filters** run is included as a separate experiment, but its supplied pixel metrics do not exceed the existing best values in any split, so no values from that row are bolded.

## Proposed Model: UNET++ With Spectral Analysis

The proposed model starts from the original UNET++ checkpoint and performs shadow-aware post-training. It is kept separate from the baseline UNET++ results above so the effect of the additional spectral processing can be evaluated independently.

### Proposed Model Pipeline

1. Load the original `unet_best.pth` UNET++ checkpoint.
2. Detect shadow regions from the RGB satellite image using the spectral analysis pipeline.
3. Apply shadow correction to the RGB input and append the detected shadow mask as a fourth input channel.
4. Use random horizontal and vertical flips, 90-degree rotations, and synthetic panel-shadow augmentation during post-training.
5. Optimize with shadow-weighted BCE plus Dice loss using AdamW, differential encoder/decoder learning rates, gradient accumulation, AMP, OneCycle learning-rate scheduling, and early stopping on validation IoU.
6. Evaluate the best checkpoint with pixel metrics, connected-component instance metrics, mAP@0.5, and mAP@[0.5:0.95].

The post-training run used the fixed split saved in `outputs/spectral_analysis/split.json`. The reported loss is the combined shadow-weighted BCE + Dice objective; it should not be interpreted as a percentage error. Dice Loss in the tables above is calculated as `1 - pixel_dice` for consistency with the baseline model tables. For the proposed model, `pixel_dice` is the mean of per-image Dice values, while `pixel_f1` is aggregated across all pixels in the split, so those values are not expected to be identical.

### Proposed Model Results

| Split | Loss | Accuracy | Precision | Recall | F1 Score | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.2977 | 0.9794 | 0.8138 | 0.7941 | 0.8038 | 0.7970 | 0.6720 |
| Validation | 0.3376 | 0.9775 | 0.7962 | 0.7654 | 0.7805 | 0.7690 | 0.6400 |
| Test | 0.3056 | 0.9800 | 0.8051 | 0.7866 | 0.7958 | 0.7883 | 0.6608 |

### Proposed Model Instance Results

Instances are connected components of the predicted and ground-truth semantic masks. Adjacent panels can merge into one component, so these counts are not equivalent to object-detector instance counts.

| Split | Instance Precision | Instance Recall | Instance F1 | mAP@0.5 | mAP@[0.5:0.95] | Ground-Truth Instances | Predicted Instances |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.7213 | 0.7977 | 0.7576 | 0.7044 | 0.3747 | 24,052 | 26,599 |
| Validation | 0.6869 | 0.7763 | 0.7289 | 0.6807 | 0.3533 | 4,751 | 5,369 |
| Test | 0.7136 | 0.7974 | 0.7531 | 0.7031 | 0.3723 | 2,971 | 3,320 |

The proposed model has a small train-to-validation gap: validation loss is 0.3376 versus a training loss of 0.2977, while test loss is 0.3056. This indicates that the post-training run generalizes reasonably across the fixed splits. Its aggregate pixel scores are below the original UNET++ baseline reported above; the instance and mAP results provide additional shadow-aware diagnostics rather than a directly comparable baseline ranking. The proposed model is therefore best treated as a shadow-aware research variant and a foundation for further tuning, rather than as a replacement for the current validation-leading baseline without additional experiments.

### Proposed Model Artifacts

The post-training notebook is [UNET++/spectral_model.ipynb](UNET++/spectral_model.ipynb). Its outputs are stored under `UNET++/outputs/spectral_analysis/`:

| Artifact | Purpose |
|---|---|
| `metrics/spectral_metrics_v2.csv` | Pixel and instance metrics for train, validation, and test splits. |
| `metrics/baseline_metrics_v2.csv` | Original UNET++ baseline metrics on the same evaluation splits. |
| `metrics/test_comparison_v2.csv` | Test-set baseline versus proposed-model deltas. |
| `metrics/shadow_stratified_v2.csv` | Shadow-over-panel recall and shadow-over-roof false-positive rate. |
| `metrics/training_history_v2.csv` | Per-epoch training loss, validation loss, overlap metrics, and learning rates. |
| `models/unet_best_spectral_v2.pth` | Best proposed-model checkpoint selected by validation IoU. |
| `visualizations/` | Training curves and qualitative shadow-heavy test comparisons. |

## Training Wall Time

The total wall time is the elapsed time reported by each notebook for the complete training loop. Models are listed in the order in which their training runs were recorded.

| Training Run Order | Model | Total Wall Time | Wall Time (min) |
|---:|---|---:|---:|
| 1 | UNET | 1,350.1 s | 22.5 |
| 2 | Segnet | 1,098.5 s | 18.3 |
| 3 | Attention UNET | 1,838.2 s | 30.6 |
| 4 | DeepLabV3 | 588.4 s | 9.8 |
| 5 | UNET++ | 4,461.6 s | 74.4 |
| 6 | DFANET | 515.4 s | 8.6 |
| 7 | SegFormer | 415.5 s | 6.9 |

## Model Introduction Timeline

The table below lists the models in order of their original publication or introduction year. Models introduced in the same year are ordered by the commonly cited publication sequence.

| Introduction Order | Model | Introduction Year |
|---:|---|---:|
| 1 | UNET | 2015 |
| 2 | Segnet | 2015 |
| 3 | DeepLabV3 | 2017 |
| 4 | Attention UNET | 2018 |
| 5 | UNET++ | 2018 |
| 6 | DFANET | 2019 |
| 7 | SegFormer | 2021 |

## Model Recommendation

**The original UNET++ remains the recommended baseline for deployment or further tuning based on the current validation results.** It achieves the strongest validation Accuracy, Precision, F1 Score, Dice, IoU, and Dice Loss, and it also produces the strongest training results across the original baseline comparison.

The **Proposed UNET++ + Spectral Analysis** model is a shadow-aware research variant. It adds spectral preprocessing and post-training diagnostics, including instance metrics and mAP, but its current pixel scores are lower than the original UNET++ baseline. It should be retained as the proposed method for shadow robustness experiments and improved through additional tuning before deployment.

Attention UNET achieves the strongest test Accuracy, F1 Score, Dice, IoU, and Dice Loss among the original baselines, while UNET++ achieves the strongest test Precision. The proposed model provides an additional shadow-focused evaluation path rather than replacing these baseline rankings.

## Output Structure

Each model follows this structure:

```text
<model>/
├── model.ipynb
└── outputs/
	├── metrics/
	│   ├── metrics_summary_train_val_test.csv
	│   ├── metrics_train_val_per_epoch.csv
	│   └── split_image_ids.json
	├── models/
	├── plots/
	└── predictions/
```

The proposed UNET++ spectral post-training run additionally uses:

```text
UNET++/outputs/spectral_analysis/
├── metrics/
│   ├── spectral_metrics_v2.csv
│   ├── baseline_metrics_v2.csv
│   ├── test_comparison_v2.csv
│   ├── shadow_stratified_v2.csv
│   └── training_history_v2.csv
├── models/
│   └── unet_best_spectral_v2.pth
└── visualizations/
```

The main comparison source is `outputs/metrics/metrics_summary_train_val_test.csv`. Per-epoch learning curves and training/validation history are stored in `metrics_train_val_per_epoch.csv`.
