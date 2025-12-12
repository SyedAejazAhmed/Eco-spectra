import pandas as pd
import requests
import math
import os
import time
from datetime import datetime
from PIL import Image

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

def verify_image(filepath):
    """Verify if downloaded image is valid."""
    try:
        if not os.path.exists(filepath):
            return False, "File does not exist"
        
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return False, "File is empty (0 bytes)"
        
        # Try to open image to verify it's valid
        with Image.open(filepath) as img:
            width, height = img.size
            if width == 0 or height == 0:
                return False, f"Invalid dimensions ({width}x{height})"
            return True, f"Valid image ({width}x{height}, {file_size} bytes)"
    except Exception as e:
        return False, f"Image validation error: {str(e)[:50]}"

# ============================================
# CONVERSION AND DOWNLOAD FUNCTIONS
# ============================================

# --- Convert lat/lon to tile numbers for Esri ---
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

# --- Fetch free satellite tile from Esri World Imagery ---
def fetch_esri_satellite(lat, lon, zoom, output_path, sample_id):
    x, y = deg2num(lat, lon, zoom)

    url = (
        f"https://services.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )

    try:
        resp = requests.get(url, timeout=30)

        if resp.status_code != 200:
            error_msg = f"HTTP {resp.status_code} for {sample_id} ({lat},{lon})"
            log_message(f"❌ Error: {error_msg}")
            return False, error_msg

        # Save image
        with open(output_path, "wb") as f:
            f.write(resp.content)

        # Verify image
        is_valid, validation_msg = verify_image(output_path)
        
        if is_valid:
            log_message(f"✅ {sample_id}: {validation_msg}")
            return True, validation_msg
        else:
            log_message(f"⚠️  {sample_id}: {validation_msg}")
            return False, validation_msg
            
    except Exception as e:
        error_msg = f"Exception for {sample_id}: {str(e)[:80]}"
        log_message(f"❌ Error: {error_msg}")
        return False, error_msg


# ============================================
# MAIN: Load dataset + download
# ============================================

def download_all_images(out_dir="images", zoom=19):
    # Initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"ESRI World Imagery Download Log\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    log_message(f"\n{'='*60}")
    log_message(f"🛰️  ESRI WORLD IMAGERY DOWNLOADER")
    log_message(f"{'='*60}")
    log_message(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"🎯 Project: Solar Classification Dataset")
    log_message(f"{'='*60}\n")
    
    # Load your file - path relative to ESRI_Data folder
    input_path = r"..\EI_train_data(Sheet1).csv"
    
    if not os.path.exists(input_path):
        log_message(f"❌ Error: CSV file not found at {input_path}")
        log_message(f"   Current directory: {os.getcwd()}")
        return
    
    # Read CSV with sampleid as string
    df = pd.read_csv(input_path, dtype={'sampleid': str})
    log_message(f"📂 Loaded {len(df):,} records from {input_path}")
    log_message(f"📋 Columns: {list(df.columns)}\n")

    # Create output folder
    os.makedirs(out_dir, exist_ok=True)
    log_message(f"📁 Output directory: {os.path.abspath(out_dir)}\n")
    
    log_message(f"{'='*60}")
    log_message(f"Starting image download...")
    log_message(f"🔍 Zoom Level: {zoom}")
    log_message(f"{'='*60}\n")
    
    # Statistics
    downloaded = 0
    skipped = 0
    errors = 0
    invalid = 0
    error_details = []
    
    start_time = time.time()

    for index, row in df.iterrows():
        sample_id = row["sampleid"]
        lat = row["latitude"]
        lon = row["longitude"]

        out_path = os.path.join(out_dir, f"{sample_id}.png")
        
        # Skip if already exists
        if os.path.exists(out_path):
            # Verify existing image
            is_valid, msg = verify_image(out_path)
            if is_valid:
                skipped += 1
                if (index + 1) % 100 == 0:
                    log_message(f"⏭️  [{index+1:>4}/{len(df)}] Progress: {downloaded:>4} downloaded | {skipped:>4} skipped | {errors:>3} errors")
                continue
            else:
                log_message(f"⚠️  {sample_id}: Existing file invalid, re-downloading...")
                invalid += 1

        # Download image
        success, msg = fetch_esri_satellite(lat, lon, zoom, out_path, sample_id)
        
        if success:
            downloaded += 1
        else:
            errors += 1
            error_details.append(f"{sample_id} - {msg}")
        
        # Progress update every 100 images
        if (index + 1) % 100 == 0:
            log_message(f"📊 [{index+1:>4}/{len(df)}] Downloaded: {downloaded:>4} | Skipped: {skipped:>4} | Errors: {errors:>3}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final summary
    log_message(f"\n{'='*60}")
    log_message(f"✅ DOWNLOAD COMPLETE")
    log_message(f"{'='*60}")
    log_message(f"⏱️  Total Time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    log_message(f"📊 Statistics:")
    log_message(f"   ✅ Downloaded: {downloaded:,} images")
    log_message(f"   ⏭️  Skipped: {skipped:,} images (already existed)")
    log_message(f"   ⚠️  Invalid files replaced: {invalid:,}")
    log_message(f"   ❌ Errors: {errors:,} images")
    log_message(f"   📈 Success Rate: {(downloaded/(downloaded+errors)*100) if (downloaded+errors) > 0 else 0:.1f}%")
    log_message(f"   📁 Saved to: {os.path.abspath(out_dir)}")
    log_message(f"{'='*60}")
    
    # Log error details if any
    if error_details:
        log_message(f"\n❌ Error Details:")
        for error in error_details[:20]:  # Show first 20 errors
            log_message(f"   - {error}")
        if len(error_details) > 20:
            log_message(f"   ... and {len(error_details) - 20} more errors")
    
    # Log session end
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
    
    log_message(f"\n📄 Log file saved to: {os.path.abspath(LOG_FILE)}")
    log_message(f"\n{'='*60}")
    log_message(f"✅ PROGRAM COMPLETED SUCCESSFULLY")
    log_message(f"{'='*60}\n")

if __name__ == "__main__":
    try:
        download_all_images()
    except KeyboardInterrupt:
        log_message(f"\n\n⚠️  Download interrupted by user (Ctrl+C)")
        log_message(f"   Re-run the script to resume from where you left off.")
    except Exception as e:
        log_message(f"\n{'='*60}")
        log_message(f"❌ FATAL ERROR")
        log_message(f"{'='*60}")
        log_message(f"Error: {e}")
        log_message(f"{'='*60}\n")
        raise