# Google Maps Satellite Image Downloader

Automated satellite image downloader using Google Maps Static API for solar panel detection dataset.

## Overview

Downloads high-resolution satellite imagery (640x640 pixels) based on latitude/longitude coordinates from a CSV dataset. Optimized for rooftop-level detail at zoom level 20.

## Features

- **High Resolution**: 640×640 pixels at zoom 20 (~0.15m/pixel)
- **API Key Protection**: Secure `.env` configuration
- **Smart Resume**: Skips existing images automatically
- **Test Mode**: Download 5 samples first to verify quality
- **Rate Limiting**: Built-in delays to prevent API throttling
- **Comprehensive Logging**: Detailed progress tracking in `log.txt`

## Requirements

```bash
pip install pandas requests python-dotenv
```

## Setup

1. Create `.env` file with your API key:
   ```
   google_map_api_key=YOUR_API_KEY_HERE
   ```

2. Ensure CSV file exists at `../EI_train_data(Sheet1).csv`

## Usage

### Test Mode (Recommended First)
```bash
python test.py
```
Downloads only 5 images to verify quality (~30 seconds).

### Full Download
```bash
python app.py
```
Downloads all 3,000 images (~8-10 minutes).

## Configuration

- **Input**: `../EI_train_data(Sheet1).csv` (sampleid, latitude, longitude, has_solar)
- **Output**: `images/` folder with files named `{sampleid}_{has_solar}.png`
- **Zoom Level**: 20 (optimal for rooftop detail)
- **Image Size**: 640×640 pixels (~90m × 90m coverage)
- **Map Type**: Satellite imagery

## Cost

- **Per Image**: $0.002
- **3,000 Images**: $6.00
- **Free Tier**: $200/month credit (covers 100,000 images)
- **Your Cost**: $0.00 (covered by free tier)

## Dataset

Successfully downloads **3,000 satellite images** for solar classification training.

## Output Format

- Format: PNG
- Dimensions: 640×640 pixels
- Naming: `{sampleid}_{has_solar}.png` (e.g., `0001_1.png`)
- Average size: ~100-200KB per image

## Safety Features

- Checks for existing files before downloading
- Rate limiting (0.1s delay between requests)
- Error handling with detailed logging
- Safe to interrupt and resume (Ctrl+C)
