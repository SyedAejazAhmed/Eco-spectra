# Mapbox Satellite Image Downloader

Automated satellite image downloader using Mapbox Static Images API for solar panel detection dataset.

## Overview

Downloads high-resolution satellite imagery (600x600@2x) based on latitude/longitude coordinates from a CSV dataset. Uses Mapbox's satellite-v9 style for pure satellite imagery.

## Features

- **Retina Quality**: 600x600@2x resolution for high-definition imagery
- **API Token Protection**: Secure `.env` configuration
- **Smart Resume**: Skips existing images automatically
- **Test Mode**: Download subset to verify quality first
- **Rate Limiting**: Built-in delays to prevent API throttling
- **Progress Tracking**: Real-time ETA and statistics in `log.txt`
- **Coordinate Order**: Handles Mapbox format (longitude, latitude)

## Requirements

```bash
pip install pandas requests python-dotenv
```

## Setup

1. Create `.env` file with your Mapbox token:
   ```
   mapbox_token=YOUR_MAPBOX_TOKEN_HERE
   ```

2. Ensure CSV file exists at `../EI_train_data(Sheet1).csv`

## Usage

### Test Mode (Recommended First)
Edit `app.py` and set:
```python
TEST_MODE = True
```
Then run:
```bash
python test.py
```
Downloads only 5 images to verify quality.

### Full Download
Edit `app.py` and set:
```python
TEST_MODE = False
```
Then run:
```bash
python app.py
```
Downloads all 3,000 images.

## Configuration

- **Input**: `../EI_train_data(Sheet1).csv` (sampleid, latitude, longitude, has_solar)
- **Output**: `images/` folder with files named `{sampleid}_{has_solar}.png`
- **Style**: `mapbox/satellite-v9` (pure satellite imagery)
- **Zoom Level**: 19.5 (optimal for rooftop detail)
- **Image Size**: 600x600@2x (Retina quality, 1200x1200 pixels)
- **Delay**: 0.1s between requests
- **Log File**: `log.txt`

## Important Note

**Mapbox uses LONGITUDE, LATITUDE order** (opposite of Google Maps). The script handles this automatically.

## Performance

- **Speed**: ~0.5-0.6 images/second
- **Total Time**: ~90 minutes for 3,000 images
- **Success Rate**: >99%
- **File Size**: ~50-150KB per image

## Dataset

Successfully downloaded **3,000 satellite images** for solar classification training data.

## Output Format

- Format: PNG
- Dimensions: 1200×1200 pixels (600x600@2x)
- Naming: `{sampleid}_{has_solar}.png` (e.g., `0001_1.png`, `2501_0.png`)
- Average size: ~100KB per image
