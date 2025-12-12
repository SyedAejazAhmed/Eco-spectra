import pandas as pd
import requests
import os
import time
from datetime import datetime
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

# Log file path
LOG_FILE = "log.txt"

# Image parameters - CRITICAL for rooftop resolution
ZOOM_LEVEL = 20        # Non-negotiable for rooftops (~0.15m/pixel in India)
IMAGE_SIZE = "640x640" # Standard max free tier (~90m x 90m coverage)

# Safe mode: Set to False to download ALL images
TEST_MODE = False  # Set to True to download only 5 test images first

DELAY_BETWEEN_REQUESTS = 0.1

# ============================================
# LOGGING FUNCTIONS
# ============================================

def log_message(message, log_file=LOG_FILE):
    """Log message to both console and file."""
    print(message)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# ============================================
# MAIN EXECUTION
# ============================================

def download_satellite_images():
    """
    Downloads satellite images from Google Maps Static API for each location in the CSV.
    Implements idempotency, rate limiting, and error handling.
    """
    
    # Initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Google Maps Satellite Image Download Log\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    # Verify CSV exists
    if not os.path.exists(CSV_PATH):
        log_message(f"Error: CSV file not found at {CSV_PATH}")
        log_message(f"   Current directory: {os.getcwd()}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    log_message(f"Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    log_message(f"Loaded {len(df)} records")
    log_message(f"   Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ['sampleid', 'latitude', 'longitude', 'has_solar']
    for col in required_cols:
        if col not in df.columns:
            log_message(f"Error: Required column '{col}' not found in CSV")
            return
    
    # Apply test mode limit
    total_records = len(df)
    if TEST_MODE:
        df = df.head(5)
        log_message(f"\n⚠️  TEST MODE ACTIVE - Processing only {len(df)} images")
        log_message(f"   Set TEST_MODE = False to process all {total_records} records\n")
    else:
        log_message(f"\n🚀 FULL MODE - Processing all {len(df):,} images\n")
    
    # Statistics
    downloaded = 0
    skipped = 0
    errors = 0
    error_details = []
    
    # Main download loop
    log_message(f"{'='*60}")
    log_message(f"Starting image download...")
    log_message(f"{'='*60}\n")
    
    start_time = time.time()
    
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
            if (index + 1) % 100 == 0:
                log_message(f"⏭️  [{index+1:>4}/{len(df)}] Progress: {downloaded:>4} downloaded | {skipped:>4} skipped | {errors:>3} errors")
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
                
                # Progress update every 100 images
                if (index + 1) % 100 == 0:
                    log_message(f"✅ [{index+1:>4}/{len(df)}] Downloaded {downloaded:>4} images | "
                          f"Skipped: {skipped:>4} | Errors: {errors:>3}")
            else:
                errors += 1
                error_msg = f"Error on ID {sid}: HTTP {response.status_code}"
                log_message(error_msg)
                error_details.append(f"{sid} - HTTP {response.status_code}")
                
        except Exception as e:
            errors += 1
            error_msg = f"Failed {sid}: {str(e)}"
            log_message(error_msg)
            error_details.append(f"{sid} - {str(e)}")
        
        # Rate limiting - be polite to the API
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final summary
    log_message(f"\n{'='*60}")
    log_message(f"✅ DOWNLOAD COMPLETE")
    log_message(f"{'='*60}")
    log_message(f"⏱️  Total Time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    log_message(f"📊 Statistics:")
    log_message(f"   ✅ Downloaded: {downloaded:,} images")
    log_message(f"   ⏭️  Skipped: {skipped:,} images")
    log_message(f"   ❌ Errors: {errors:,} images")
    log_message(f"   📈 Success Rate: {(downloaded/(downloaded+errors)*100) if (downloaded+errors) > 0 else 0:.1f}%")
    log_message(f"   📁 Saved to: {os.path.abspath(OUTPUT_DIR)}")
    log_message(f"{'='*60}")
    
    # Log error details if any
    if error_details:
        log_message(f"\n❌ Error Details:")
        for error in error_details:
            log_message(f"   - {error}")
    
    if TEST_MODE:
        log_message(f"\nTEST MODE WAS ACTIVE")
        log_message(f"   1. Check the {downloaded} images in the '{OUTPUT_DIR}' folder")
        log_message(f"   2. Verify rooftop visibility and image quality")
        log_message(f"   3. If satisfied, set TEST_MODE = False and re-run")
        log_message(f"   4. Remaining images to download: {total_records - len(df)}")
    
    # Log session end
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        log_message(f"\n{'='*60}")
        log_message(f"🛰️  GOOGLE MAPS SATELLITE IMAGE DOWNLOADER")
        log_message(f"{'='*60}")
        log_message(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"🎯 Project: Solar Classification Dataset")
        log_message(f"{'='*60}\n")
        
        download_satellite_images()
        
        log_message(f"\n{'='*60}")
        log_message(f"✅ PROGRAM COMPLETED SUCCESSFULLY")
        log_message(f"📄 Log file saved to: {os.path.abspath(LOG_FILE)}")
        log_message(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        log_message(f"\n\n⚠️  Download interrupted by user (Ctrl+C)")
        log_message(f"   Re-run the script to resume from where you left off.")
    except Exception as e:
        log_message(f"\n{'='*60}")
        log_message(f"❌ FATAL ERROR")
        log_message(f"{'='*60}")
        log_message(f"Error: {e}")
        log_message(f"{'='*60}\n")
