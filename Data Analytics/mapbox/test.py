import pandas as pd
import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# Mapbox Configuration
MAPBOX_TOKEN = os.getenv('mapbox_token')
STYLE_ID = "mapbox/satellite-v9"  # Pure satellite imagery
ZOOM_LEVEL = 19.5
IMAGE_SIZE = "600x600"
RESOLUTION = "@2x"  # High-definition Retina quality

# File paths
CSV_PATH = r"..\EI_train_data(Sheet1).csv"
OUTPUT_DIR = "images"

# Test mode - download only first 5 images
TEST_MODE = True

# ============================================
# LOGGING FUNCTIONS
# ============================================

LOG_FILE = "log.txt"

def log_message(message):
    """Log message to both console and file."""
    print(message)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# ============================================
# MAPBOX DOWNLOAD FUNCTIONS
# ============================================

def construct_mapbox_url(longitude, latitude, zoom, size, resolution, token):
    """
    Construct Mapbox Static Image URL.
    CRITICAL: Mapbox uses LONGITUDE,LATITUDE order (opposite of Google Maps)
    """
    base_url = f"https://api.mapbox.com/styles/v1/{STYLE_ID}/static"
    
    # Format: {lon},{lat},{zoom},{bearing},{pitch}/{width}x{height}{@2x}
    # bearing=0 (north), pitch=0 (top-down view)
    location = f"{longitude},{latitude},{zoom},0,0"
    dimensions = f"{size}{resolution}"
    
    url = f"{base_url}/{location}/{dimensions}?access_token={token}"
    return url

def download_mapbox_image(sample_id, latitude, longitude, has_solar):
    """Download a single satellite image from Mapbox."""
    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_{has_solar}.png")
    
    # Skip if already exists
    if os.path.exists(output_path):
        return "skipped", "Already exists"
    
    # Construct URL - CRITICAL: longitude FIRST, then latitude
    url = construct_mapbox_url(longitude, latitude, ZOOM_LEVEL, IMAGE_SIZE, RESOLUTION, MAPBOX_TOKEN)
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Save image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_path)
            return "success", f"Downloaded ({file_size:,} bytes)"
        else:
            # Mapbox returns JSON error messages
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}"
            return "error", error_msg
            
    except requests.exceptions.Timeout:
        return "error", "Request timeout"
    except Exception as e:
        return "error", str(e)[:80]

# ============================================
# MAIN DOWNLOAD FUNCTION
# ============================================

def download_satellite_images():
    """Main function to download Mapbox satellite images."""
    
    # Initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Mapbox Satellite Image Download Log\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    log_message(f"\n{'='*60}")
    log_message(f"🗺️  MAPBOX SATELLITE IMAGE DOWNLOADER")
    log_message(f"{'='*60}")
    log_message(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"🎯 Project: Solar Classification Dataset")
    log_message(f"🔍 Testing with first 5 images")
    log_message(f"{'='*60}\n")
    
    # Verify CSV exists
    if not os.path.exists(CSV_PATH):
        log_message(f"❌ Error: CSV file not found at {CSV_PATH}")
        return
    
    # Read CSV
    df = pd.read_csv(CSV_PATH, dtype={'sampleid': str})
    log_message(f"📂 Loaded {len(df):,} total records from CSV")
    
    # Test mode: only first 5
    if TEST_MODE:
        df = df.head(5)
        log_message(f"🧪 TEST MODE: Processing first {len(df)} images")
    
    log_message(f"📋 Columns: {list(df.columns)}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_message(f"📁 Output directory: {os.path.abspath(OUTPUT_DIR)}\n")
    
    log_message(f"{'='*60}")
    log_message(f"Mapbox Configuration:")
    log_message(f"   Style: {STYLE_ID}")
    log_message(f"   Zoom: {ZOOM_LEVEL}")
    log_message(f"   Size: {IMAGE_SIZE}{RESOLUTION}")
    log_message(f"   Coordinate Order: LONGITUDE, LATITUDE (Mapbox format)")
    log_message(f"{'='*60}\n")
    
    # Statistics
    downloaded = 0
    skipped = 0
    errors = 0
    error_details = []
    
    start_time = time.time()
    
    # Download images
    for index, row in df.iterrows():
        sample_id = row['sampleid']
        latitude = row['latitude']
        longitude = row['longitude']
        has_solar = row['has_solar']
        
        log_message(f"[{index+1}/{len(df)}] Processing {sample_id} (Lon: {longitude}, Lat: {latitude})...")
        
        status, message = download_mapbox_image(sample_id, latitude, longitude, has_solar)
        
        if status == "success":
            downloaded += 1
            log_message(f"   ✅ {message}")
        elif status == "skipped":
            skipped += 1
            log_message(f"   ⏭️  {message}")
        else:  # error
            errors += 1
            error_details.append(f"{sample_id} - {message}")
            log_message(f"   ❌ Error: {message}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final summary
    log_message(f"\n{'='*60}")
    log_message(f"✅ TEST COMPLETE")
    log_message(f"{'='*60}")
    log_message(f"⏱️  Total Time: {total_time:.2f} seconds")
    log_message(f"📊 Statistics:")
    log_message(f"   ✅ Downloaded: {downloaded} images")
    log_message(f"   ⏭️  Skipped: {skipped} images")
    log_message(f"   ❌ Errors: {errors} images")
    if downloaded + errors > 0:
        log_message(f"   📈 Success Rate: {(downloaded/(downloaded+errors)*100):.1f}%")
    log_message(f"   📁 Saved to: {os.path.abspath(OUTPUT_DIR)}")
    log_message(f"{'='*60}")
    
    # Log error details if any
    if error_details:
        log_message(f"\n❌ Error Details:")
        for error in error_details:
            log_message(f"   - {error}")
    
    log_message(f"\n📄 Log file: {os.path.abspath(LOG_FILE)}")
    log_message(f"\n✅ If images look good, set TEST_MODE=False in app.py for full download")
    log_message(f"\n{'='*60}\n")

if __name__ == "__main__":
    try:
        download_satellite_images()
    except KeyboardInterrupt:
        log_message(f"\n\n⚠️  Download interrupted by user (Ctrl+C)")
    except Exception as e:
        log_message(f"\n❌ FATAL ERROR: {e}")
        raise
