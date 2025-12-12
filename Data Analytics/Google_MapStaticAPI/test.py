import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv

# ============================================
# CONFIGURATION
# ============================================

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("google_map_api_key")

if not API_KEY:
    raise ValueError("API key not found! Please set google_map_api_key in .env file")

# Dataset path (relative to Google folder)
CSV_PATH = r"..\EI_train_data(Sheet1).csv"

# Output directory for images
OUTPUT_DIR = "images"

# Image parameters - CRITICAL for rooftop resolution
ZOOM_LEVEL = 20        # Non-negotiable for rooftops (~0.15m/pixel in India)
IMAGE_SIZE = "640x640" # Standard max free tier (~90m x 90m coverage)

# Safe mode: Set to True to download only 5 test images first
TEST_MODE = True  # Set to False to download all 3000 images

DELAY_BETWEEN_REQUESTS = 0.1

# MAIN EXECUTION

def download_satellite_images():
    """
    Downloads satellite images from Google Maps Static API for each location in the CSV.
    Implements idempotency, rate limiting, and error handling.
    """
    
    # Verify CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ['sampleid', 'latitude', 'longitude', 'has_solar']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            return
    
    # Apply test mode limit
    total_records = len(df)
    if TEST_MODE:
        df = df.head(5)
        print(f"\nTEST MODE ACTIVE - Processing only {len(df)} images")
        print(f"   Set TEST_MODE = False to process all {total_records} records\n")
    else:
        print(f"\n🚀 FULL MODE - Processing all {len(df)} images\n")
    
    # Statistics
    downloaded = 0
    skipped = 0
    errors = 0
    
    # Main download loop
    print(f"{'='*60}")
    print(f"Starting image download...")
    print(f"{'='*60}\n")
    
    for index, row in df.iterrows():
        sid = row['sampleid']
        lat = row['latitude']
        lon = row['longitude']
        label = row['has_solar']
        
        # Construct filename: 0001_1.png (ID_Label.png)
        filename = f"{sid}_{label}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if already exists (idempotency)
        if os.path.exists(filepath):
            skipped += 1
            if index % 10 == 0:
                print(f"⏭️  [{index+1}/{len(df)}] Skipped {sid} (already exists)")
            continue
        
        # Construct Google Maps Static API URL
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}"
            f"&zoom={ZOOM_LEVEL}"
            f"&size={IMAGE_SIZE}"
            f"&maptype=satellite"
            f"&key={API_KEY}"
        )
        
        try:
            # Make API request
            response = requests.get(url, stream=True, timeout=30)
            
            if response.status_code == 200:
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
                
                # Progress update every 10 images
                if (index + 1) % 10 == 0:
                    print(f"[{index+1}/{len(df)}] Downloaded {downloaded} images | "
                          f"Skipped: {skipped} | Errors: {errors}")
            else:
                errors += 1
                print(f"Error on ID {sid}: HTTP {response.status_code}")
                
        except Exception as e:
            errors += 1
            print(f"Failed {sid}: {e}")
        
        # Rate limiting - be polite to the API
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Final summary
    print(f"DOWNLOAD COMPLETE")
    print(f"Statistics:")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")
    
    if TEST_MODE:
        print(f"\nTEST MODE WAS ACTIVE")
        print(f"   1. Check the {downloaded} images in the '{OUTPUT_DIR}' folder")
        print(f"   2. Verify rooftop visibility and image quality")
        print(f"   3. If satisfied, set TEST_MODE = False and re-run")
        print(f"   4. Remaining images to download: {total_records - len(df)}")


if __name__ == "__main__":
    try:
        download_satellite_images()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
