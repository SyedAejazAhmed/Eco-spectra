"""
Retry Failed Downloads - Reads log.txt and re-downloads failed images
"""
import pandas as pd
import requests
import os
import re
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

# Paths
CSV_PATH = r"..\EI_train_data(Sheet1).csv"
OUTPUT_DIR = "images"
LOG_FILE = "log.txt"
RETRY_LOG_FILE = "retry_log.txt"

# Image parameters
ZOOM_LEVEL = 20
IMAGE_SIZE = "640x640"
DELAY_BETWEEN_REQUESTS = 0.1

# ============================================
# LOGGING FUNCTIONS
# ============================================

def log_message(message, log_file=RETRY_LOG_FILE):
    """Log message to both console and file."""
    print(message)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

# ============================================
# EXTRACT FAILED IDS FROM LOG
# ============================================

def extract_failed_ids(log_file):
    """Extract failed sample IDs from log file."""
    failed_ids = []
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return failed_ids
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for error details section
    error_section_match = re.search(r'❌ Error Details:(.*?)(?:=====|$)', content, re.DOTALL)
    
    if error_section_match:
        error_section = error_section_match.group(1)
        # Extract sample IDs from error lines
        # Pattern: "- 960.0 - " or "Failed 960.0:"
        patterns = [
            r'- (\d+\.?\d*) -',  # From error details section
            r'Failed (\d+\.?\d*):'  # From inline error messages
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, error_section)
            for match in matches:
                sample_id = str(match).replace('.0', '').zfill(4)  # Convert to 4-digit format
                if sample_id not in failed_ids:
                    failed_ids.append(sample_id)
    
    # Also check inline error messages throughout the log
    inline_errors = re.findall(r'Failed (\d+\.?\d*):', content)
    for match in inline_errors:
        sample_id = str(match).replace('.0', '').zfill(4)
        if sample_id not in failed_ids:
            failed_ids.append(sample_id)
    
    return sorted(failed_ids)

# ============================================
# RETRY DOWNLOAD
# ============================================

def retry_failed_downloads():
    """Retry downloading failed images from log file."""
    
    # Initialize retry log
    with open(RETRY_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Retry Failed Downloads Log\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    log_message(f"\n{'='*60}")
    log_message(f"🔄 RETRY FAILED DOWNLOADS")
    log_message(f"{'='*60}\n")
    
    # Extract failed IDs from log
    log_message(f"📄 Reading error log: {LOG_FILE}")
    failed_ids = extract_failed_ids(LOG_FILE)
    
    if not failed_ids:
        log_message(f"✅ No failed downloads found in log file!")
        log_message(f"   All images downloaded successfully.")
        return
    
    log_message(f"❌ Found {len(failed_ids)} failed downloads:")
    for fid in failed_ids:
        log_message(f"   - Sample ID: {fid}")
    
    # Load CSV to get coordinates
    if not os.path.exists(CSV_PATH):
        log_message(f"\n❌ Error: CSV file not found at {CSV_PATH}")
        return
    
    log_message(f"\n📂 Loading dataset: {CSV_PATH}")
    # Read CSV with sampleid as string to preserve leading zeros
    df = pd.read_csv(CSV_PATH, dtype={'sampleid': str})
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Statistics
    downloaded = 0
    already_exists = 0
    errors = 0
    error_details = []
    
    log_message(f"\n{'='*60}")
    log_message(f"Starting retry download...")
    log_message(f"{'='*60}\n")
    
    start_time = time.time()
    
    for idx, sample_id in enumerate(failed_ids, 1):
        # Find the row in CSV
        row = df[df['sampleid'] == sample_id]
        
        if row.empty:
            log_message(f"⚠️  [{idx}/{len(failed_ids)}] Sample ID {sample_id} not found in CSV")
            errors += 1
            error_details.append(f"{sample_id} - Not found in CSV")
            continue
        
        row = row.iloc[0]
        lat = row['latitude']
        lon = row['longitude']
        label = row['has_solar']
        
        # Construct filename
        filename = f"{sample_id}_{label}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Check if already exists
        if os.path.exists(filepath):
            log_message(f"⏭️  [{idx}/{len(failed_ids)}] {sample_id} already exists, skipping")
            already_exists += 1
            continue
        
        # Construct URL
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}"
            f"&zoom={ZOOM_LEVEL}"
            f"&size={IMAGE_SIZE}"
            f"&maptype=satellite"
            f"&key={API_KEY}"
        )
        
        try:
            # Make API request with retry
            response = requests.get(url, stream=True, timeout=60)  # Increased timeout
            
            if response.status_code == 200:
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
                log_message(f"✅ [{idx}/{len(failed_ids)}] Downloaded {sample_id}")
            else:
                errors += 1
                error_msg = f"{sample_id} - HTTP {response.status_code}"
                log_message(f"❌ [{idx}/{len(failed_ids)}] {error_msg}")
                error_details.append(error_msg)
                
        except Exception as e:
            errors += 1
            error_msg = f"{sample_id} - {str(e)[:80]}"
            log_message(f"❌ [{idx}/{len(failed_ids)}] Failed {error_msg}")
            error_details.append(error_msg)
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final summary
    log_message(f"\n{'='*60}")
    log_message(f"✅ RETRY COMPLETE")
    log_message(f"{'='*60}")
    log_message(f"⏱️  Total Time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    log_message(f"📊 Statistics:")
    log_message(f"   🔄 Retry Attempted: {len(failed_ids)} images")
    log_message(f"   ✅ Successfully Downloaded: {downloaded} images")
    log_message(f"   ⏭️  Already Existed: {already_exists} images")
    log_message(f"   ❌ Still Failed: {errors} images")
    log_message(f"   📈 Retry Success Rate: {(downloaded/(downloaded+errors)*100) if (downloaded+errors) > 0 else 0:.1f}%")
    log_message(f"   📁 Saved to: {os.path.abspath(OUTPUT_DIR)}")
    log_message(f"{'='*60}")
    
    # Log remaining error details if any
    if error_details:
        log_message(f"\n❌ Still Failed - Error Details:")
        for error in error_details:
            log_message(f"   - {error}")
    else:
        log_message(f"\n🎉 All failed images successfully recovered!")
    
    # Log session end
    with open(RETRY_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
    
    log_message(f"\n📄 Retry log saved to: {os.path.abspath(RETRY_LOG_FILE)}")


if __name__ == "__main__":
    try:
        retry_failed_downloads()
        
        print(f"\n{'='*60}")
        print(f"✅ PROGRAM COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Retry interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ FATAL ERROR")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        raise
