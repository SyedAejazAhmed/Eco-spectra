import pandas as pd
import requests
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import ee

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# Earth Engine Configuration
SERVICE_ACCOUNT_KEY_FILE = "service-account-key.json"
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DATE_RANGE = ['2024-01-01', '2024-12-31']
BANDS = ['B4', 'B3', 'B2']  # RGB bands for Sentinel-2
VIS_PARAMS = {'min': 0, 'max': 3000}
BUFFER_DISTANCE = 1000  # meters (creates 2km x 2km box)
IMAGE_DIMENSIONS = 640  # 640x640 pixels (standard size)
DELAY_BETWEEN_REQUESTS = 0.5  # seconds

# File paths
CSV_PATH = r"..\EI_train_data(Sheet1).csv"
OUTPUT_DIR = "images"

# Full mode - download all images
TEST_MODE = False

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
# EARTH ENGINE AUTHENTICATION
# ============================================

def authenticate_earth_engine():
    """Authenticate with Earth Engine using Service Account."""
    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY_FILE):
            log_message(f"❌ Error: Service account key file not found: {SERVICE_ACCOUNT_KEY_FILE}")
            return False
        
        # Load service account credentials
        with open(SERVICE_ACCOUNT_KEY_FILE) as f:
            key_data = json.load(f)
        
        service_account = key_data['client_email']
        credentials = ee.ServiceAccountCredentials(service_account, SERVICE_ACCOUNT_KEY_FILE)
        
        # Initialize Earth Engine
        ee.Initialize(credentials)
        
        log_message(f"✅ Earth Engine authenticated successfully")
        log_message(f"   Service Account: {service_account}")
        return True
        
    except Exception as e:
        log_message(f"❌ Earth Engine authentication failed: {str(e)}")
        return False

# ============================================
# EARTH ENGINE DOWNLOAD FUNCTIONS
# ============================================

def get_sentinel2_image(longitude, latitude, sample_id):
    """
    Get the least cloudy Sentinel-2 image for a location.
    Returns the thumb URL or None if failed.
    """
    try:
        # Create point geometry
        point = ee.Geometry.Point([longitude, latitude])
        
        # Create buffer region (1km radius = 2km x 2km box)
        region = point.buffer(BUFFER_DISTANCE).bounds()
        
        # Filter Sentinel-2 collection
        collection = (ee.ImageCollection(SENTINEL2_COLLECTION)
                     .filterBounds(point)
                     .filterDate(DATE_RANGE[0], DATE_RANGE[1])
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        # Get the least cloudy image
        image = collection.first()
        
        # Check if image exists
        if image is None:
            return None, "No images found for this location"
        
        # Select RGB bands
        rgb_image = image.select(BANDS)
        
        # Get thumbnail URL
        thumb_url = rgb_image.getThumbURL({
            'dimensions': f'{IMAGE_DIMENSIONS}x{IMAGE_DIMENSIONS}',
            'region': region,
            'format': 'png',
            'min': VIS_PARAMS['min'],
            'max': VIS_PARAMS['max']
        })
        
        return thumb_url, "URL generated"
        
    except ee.EEException as e:
        error_msg = str(e)
        if "No valid" in error_msg or "No image" in error_msg:
            return None, "No valid image found"
        elif "too large" in error_msg.lower():
            return None, "Region too large"
        else:
            return None, f"EE Error: {error_msg[:80]}"
    except Exception as e:
        return None, f"Exception: {str(e)[:80]}"

def download_earth_engine_image(sample_id, latitude, longitude, has_solar):
    """Download a single satellite image from Earth Engine."""
    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_S2_{has_solar}.png")
    
    # Skip if already exists
    if os.path.exists(output_path):
        return "skipped", "Already exists", 0
    
    # Get Sentinel-2 image URL
    thumb_url, message = get_sentinel2_image(longitude, latitude, sample_id)
    
    if thumb_url is None:
        return "error", message, 0
    
    try:
        # Download the image
        response = requests.get(thumb_url, timeout=60)
        
        if response.status_code == 200:
            # Save image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_path)
            return "success", f"Downloaded ({file_size:,} bytes)", file_size
        else:
            return "error", f"HTTP {response.status_code}", 0
            
    except requests.exceptions.Timeout:
        return "error", "Download timeout", 0
    except Exception as e:
        return "error", f"Download error: {str(e)[:80]}", 0

# ============================================
# MAIN DOWNLOAD FUNCTION
# ============================================

def download_satellite_images():
    """Main function to download Earth Engine satellite images."""
    
    # Initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Google Earth Engine Sentinel-2 Download Log\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    log_message(f"\n{'='*60}")
    log_message(f"🛰️  GOOGLE EARTH ENGINE SENTINEL-2 DOWNLOADER")
    log_message(f"{'='*60}")
    log_message(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"🎯 Project: Solar Classification Dataset")
    log_message(f"{'='*60}\n")
    
    # Authenticate with Earth Engine
    log_message(f"🔐 Authenticating with Google Earth Engine...")
    if not authenticate_earth_engine():
        log_message(f"❌ Cannot proceed without authentication. Exiting.")
        return
    log_message("")
    
    # Verify CSV exists
    if not os.path.exists(CSV_PATH):
        log_message(f"❌ Error: CSV file not found at {CSV_PATH}")
        return
    
    # Read CSV
    df = pd.read_csv(CSV_PATH, dtype={'sampleid': str})
    log_message(f"📂 Loaded {len(df):,} records from CSV")
    log_message(f"📋 Columns: {list(df.columns)}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_message(f"📁 Output directory: {os.path.abspath(OUTPUT_DIR)}\n")
    
    log_message(f"{'='*60}")
    log_message(f"Earth Engine Configuration:")
    log_message(f"   Collection: {SENTINEL2_COLLECTION}")
    log_message(f"   Date Range: {DATE_RANGE[0]} to {DATE_RANGE[1]}")
    log_message(f"   Bands: {', '.join(BANDS)} (RGB)")
    log_message(f"   Visualization: min={VIS_PARAMS['min']}, max={VIS_PARAMS['max']}")
    log_message(f"   Buffer: {BUFFER_DISTANCE}m (2km x 2km coverage)")
    log_message(f"   Dimensions: {IMAGE_DIMENSIONS}x{IMAGE_DIMENSIONS} pixels")
    log_message(f"   Cloud Filter: Least cloudy image selected")
    log_message(f"   Delay: {DELAY_BETWEEN_REQUESTS}s between requests")
    log_message(f"{'='*60}\n")
    
    # Statistics
    downloaded = 0
    skipped = 0
    errors = 0
    total_size = 0
    error_details = []
    
    start_time = time.time()
    
    # Download images
    for index, row in df.iterrows():
        sample_id = row['sampleid']
        latitude = row['latitude']
        longitude = row['longitude']
        has_solar = row['has_solar']
        
        status, message, file_size = download_earth_engine_image(sample_id, latitude, longitude, has_solar)
        
        if status == "success":
            downloaded += 1
            total_size += file_size
        elif status == "skipped":
            skipped += 1
        else:  # error
            errors += 1
            error_details.append(f"{sample_id} - {message}")
            log_message(f"❌ [{index+1}/{len(df)}] Failed {sample_id}: {message}")
        
        # Progress update every 100 images
        if (index + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (index + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(df) - index - 1) / rate if rate > 0 else 0
            log_message(f"📊 [{index+1:>4}/{len(df)}] Downloaded: {downloaded:>4} | Skipped: {skipped:>4} | Errors: {errors:>3} | Rate: {rate:.1f} img/s | ETA: {remaining/60:.1f}m")
        
        # Rate limiting
        if status == "success":
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
    if downloaded + errors > 0:
        log_message(f"   📈 Success Rate: {(downloaded/(downloaded+errors)*100):.1f}%")
    if total_size > 0:
        log_message(f"   💾 Total Size: {total_size/1024/1024:.2f} MB")
    log_message(f"   ⚡ Average Rate: {(downloaded+skipped)/total_time:.2f} images/second")
    log_message(f"   📁 Saved to: {os.path.abspath(OUTPUT_DIR)}")
    log_message(f"{'='*60}")
    
    # Log error details if any
    if error_details:
        log_message(f"\n❌ Error Details:")
        for error in error_details[:20]:  # Show first 20 errors
            log_message(f"   - {error}")
        if len(error_details) > 20:
            log_message(f"   ... and {len(error_details) - 20} more errors")
    
    log_message(f"\n💡 Earth Engine Usage:")
    log_message(f"   🆓 FREE for research and education")
    log_message(f"   📊 Images Downloaded: {downloaded:,}")
    log_message(f"   ✅ No quota limits for non-commercial use")
    
    log_message(f"\n📄 Log file: {os.path.abspath(LOG_FILE)}")
    log_message(f"\n{'='*60}\n")

if __name__ == "__main__":
    try:
        download_satellite_images()
    except KeyboardInterrupt:
        log_message(f"\n\n⚠️  Download interrupted by user (Ctrl+C)")
        log_message(f"   Re-run to resume from where you left off.")
    except Exception as e:
        log_message(f"\n❌ FATAL ERROR: {e}")
        raise
