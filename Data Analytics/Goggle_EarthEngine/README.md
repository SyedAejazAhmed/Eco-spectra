# Google Earth Engine Sentinel-2 Downloader

Downloads high-resolution Sentinel-2 satellite imagery using Google Earth Engine for solar panel detection.

## Features

- **Sentinel-2 RGB Imagery**: B4, B3, B2 bands (natural color)
- **Cloud-Free Selection**: Automatically selects least cloudy image from 2024
- **Service Account Auth**: Non-interactive authentication for automation
- **Comprehensive Logging**: Detailed logs in `log.txt`
- **Smart Retry**: Skips already downloaded images
- **Free Tier**: Completely free for research/education use

## Setup

### 1. Install Dependencies

```bash
pip install earthengine-api pandas requests python-dotenv
```

### 2. Get Service Account Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Earth Engine API
4. Go to IAM & Admin → Service Accounts
5. Create Service Account with Earth Engine access
6. Create JSON key and download as `service-account-key.json`
7. Place the file in this directory

### 3. Register Service Account with Earth Engine

```bash
earthengine authenticate --service-account
```

## Usage

### Test First (5 images)

```bash
python test.py
```

Check `gee_images/` folder and `log.txt` to verify image quality.

### Full Download (3,000 images)

```bash
python app.py
```

## Configuration

Edit `app.py` or `test.py`:

```python
DATE_RANGE = ['2024-01-01', '2024-12-31']  # Date range for imagery
BUFFER_DISTANCE = 1000  # meters (2km x 2km coverage)
IMAGE_DIMENSIONS = 512  # 512x512 pixels
BANDS = ['B4', 'B3', 'B2']  # RGB bands
VIS_PARAMS = {'min': 0, 'max': 3000}  # Reflectance normalization
```

## Output

- **Filename Format**: `{sampleid}_S2_{has_solar}.png`
- **Example**: `0001_S2_1.png` (sample 0001, has solar panels)
- **Directory**: `gee_images/`
- **Resolution**: 512x512 pixels
- **Coverage**: 2km x 2km per image

## Sentinel-2 Advantages

✅ **Free**: No cost for research/education  
✅ **High Resolution**: 10m per pixel (RGB bands)  
✅ **Recent Data**: Images from 2024  
✅ **Cloud Filtering**: Automatic selection of clearest images  
✅ **Global Coverage**: Worldwide availability  

## Troubleshooting

### "No images found"
- Location may not have Sentinel-2 coverage in 2024
- Try adjusting DATE_RANGE to include more years

### "Region too large"
- Reduce BUFFER_DISTANCE (currently 1000m)
- Reduce IMAGE_DIMENSIONS

### Authentication Failed
- Verify `service-account-key.json` exists
- Check service account has Earth Engine access
- Re-run `earthengine authenticate --service-account`

## Quota

🆓 **Completely FREE** for non-commercial research and education use. No quota limits!

## Files

- `test.py` - Test script (first 5 images)
- `app.py` - Full download (all 3,000 images)
- `service-account-key.json` - Service account credentials (NOT in git)
- `log.txt` - Download logs
- `gee_images/` - Output directory
