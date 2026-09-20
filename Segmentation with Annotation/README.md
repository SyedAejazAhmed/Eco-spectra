# Annotated Solar Panel Segmentation

This directory contains supervised semantic-segmentation experiments for detecting solar panels in satellite imagery. Seven architectures were trained and evaluated using the same train, validation, and test split IDs. Each model directory contains its training notebook, saved weights, predictions, plots, and metric CSV files.

## Models

| Model | Description |
|---|---|
| **UNET** | Encoder-decoder architecture with skip connections that preserve fine-grained panel boundaries. |
| **Segnet** | Encoder-decoder network that reuses pooling indices to recover spatial structure efficiently. |
| **DeepLabV3** | Atrous convolutions and spatial-pyramid pooling capture context at multiple receptive-field sizes. |
| **Attention UNET** | U-Net with attention gates that suppress irrelevant background features and focus on panel regions. |
| **UNET++** | Nested U-Net with dense skip pathways that reduce the semantic gap between encoder and decoder features. |
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

## Best Model

**UNET++ is the recommended model for deployment or further tuning.** It achieves the strongest validation Accuracy, Precision, F1 Score, Dice, IoU, and Dice Loss, which makes it the best choice based on held-out validation performance. It also produces the strongest training results across every requested metric.

Attention UNET achieves the strongest test Accuracy, F1 Score, Dice, IoU, and Dice Loss, while UNET++ achieves the strongest test Precision. The small difference between these two models on the test split should be considered when selecting a final checkpoint.

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

The main comparison source is `outputs/metrics/metrics_summary_train_val_test.csv`. Per-epoch learning curves and training/validation history are stored in `metrics_train_val_per_epoch.csv`.
