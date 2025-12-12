# ESRI World Imagery Downloader

Automated satellite image downloader using ESRI's free World Imagery service for solar panel detection dataset.

## Overview

Downloads high-resolution satellite imagery (256x256 tiles) based on latitude/longitude coordinates from a CSV dataset. Includes validation, logging, and resume capability.

## Features

- **Free API**: Uses ESRI World Imagery tile service (no API key required)
- **Image Validation**: Verifies downloaded images for corruption
- **Smart Resume**: Skips existing valid images automatically
- **Progress Tracking**: Real-time logging with detailed statistics
- **Error Handling**: Continues on failure with comprehensive error reporting

## Requirements

```
pandas
requests
pillow
```

## Usage

```bash
python app.py
```

## Configuration

- **Input**: `../EI_train_data(Sheet1).csv` (sampleid, latitude, longitude, has_solar)
- **Output**: `images/` folder with numbered PNG files (e.g., `0001.png`, `0002.png`)
- **Zoom Level**: 19 (high detail, ~1m/pixel)
- **Log File**: `log.txt`

## Dataset

Successfully downloaded **3,000 satellite images** for solar classification training data.

## Output Format

- Format: PNG (256x256 pixels)
- Naming: `{sampleid}.png`
- Average size: ~15-25KB per image
