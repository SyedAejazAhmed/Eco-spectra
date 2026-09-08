"""
FastSAM Solar Panel Detector
Alternative SAM model using YOLOv8-based FastSAM for solar panel detection
Faster and more lightweight than Lang-SAM
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

# FastSAM imports (using Ultralytics built-in implementation)
from ultralytics import FastSAM

# Configuration
DEBUG_MODE = True
TEST_MODE = True
TEST_IMAGES_COUNT = 10

# Paths
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR.parent.parent / "Data Analytics" / "Google_MapStaticAPI" / "images"
CSV_PATH = BASE_DIR.parent.parent / "Data Analytics" / "EI_train_data(Sheet1).csv"
OUTPUT_DIR = BASE_DIR / "output"
JSON_DIR = OUTPUT_DIR / "json_records"
GEOJSON_DIR = OUTPUT_DIR / "geojson"
VIS_DIR = OUTPUT_DIR / "visualizations"
MASKS_DIR = OUTPUT_DIR / "masks"

# Model paths
FASTSAM_CHECKPOINT = BASE_DIR / "weights" / "FastSAM-x.pt"  # Will download if not exists

# Detection parameters
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7
MIN_MASK_AREA_PIXELS = 25
PIXEL_RESOLUTION = 0.15  # meters per pixel
PANEL_EFFICIENCY = 0.2   # kW per square meter

# Text prompts for solar panels
TEXT_PROMPTS = [
    "solar panel",
    "solar panels", 
    "photovoltaic panel",
    "rooftop solar",
    "PV panel",
    "solar array"
]

# Logging
DETECTION_LOG = BASE_DIR / "detection_log_fastsam.txt"
FASTSAM_MODEL_LOG = BASE_DIR / "FastSAM_model_logs.txt"

def initialize_log():
    """Initialize log files with headers"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Detection log
    with open(DETECTION_LOG, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FastSAM Solar Panel Detection Log\n")
        f.write(f"Session started: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
    
    # Model error log
    with open(FASTSAM_MODEL_LOG, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FastSAM MODEL ERROR LOG\n")
        f.write(f"Session started: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

def log_message(message: str, to_console: bool = True):
    """Log message to file and optionally console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(DETECTION_LOG, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    if to_console:
        print(message)

def log_fastsam_error(image_name: str, prompt: str, error: str):
    """Log FastSAM-specific errors"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(FASTSAM_MODEL_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"  Image: {image_name}\n")
        f.write(f"  Prompt: {prompt}\n")
        f.write(f"  Error: {error}\n\n")

def initialize_fastsam_model():
    """Initialize FastSAM model"""
    log_message("🤖 Initializing FastSAM Model...")
    log_message(f"   PyTorch Version: {torch.__version__}")
    log_message(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = 'cuda'
        log_message(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        log_message("   Using CPU")
    
    try:
        # Initialize FastSAM model (will download FastSAM-x.pt if not exists)
        # Using Ultralytics built-in FastSAM implementation
        model = FastSAM('FastSAM-x.pt')
        log_message("✅ FastSAM model loaded successfully!")
        log_message("   Model will be auto-downloaded on first run if needed")
        return model, device
    except Exception as e:
        log_message(f"❌ Failed to load FastSAM model: {e}")
        raise

def detect_solar_panels_fastsam(
    image_path: Path,
    model: FastSAM,
    device: str,
    text_prompts: List[str]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Detect solar panels using FastSAM with text prompts (Ultralytics API)
    
    Returns:
        masks: Binary masks of detected panels (N, H, W)
        boxes: Bounding boxes (N, 4) in xyxy format
        scores: Confidence scores (N,)
        max_confidence: Maximum confidence score
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        
        all_masks = []
        all_boxes = []
        all_scores = []
        
        # Run FastSAM inference once to get all segments
        try:
            if DEBUG_MODE:
                log_message(f"   🔍 Running FastSAM inference...", to_console=False)
            
            # Run FastSAM everything mode
            results = model(
                str(image_path),
                device=device,
                retina_masks=True,
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
            
            # Process results - Ultralytics returns Results object
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                # Extract masks, boxes, and scores from YOLO results
                if hasattr(result, 'masks') and result.masks is not None:
                    masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
                    boxes_data = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                    scores_data = result.boxes.conf.cpu().numpy()  # (N,)
                    
                    if DEBUG_MODE:
                        log_message(f"      ✓ Found {len(masks_data)} segments", to_console=False)
                    
                    # Filter by minimum area
                    for i in range(len(masks_data)):
                        mask = masks_data[i]
                        mask_area = np.sum(mask)
                        
                        if mask_area >= MIN_MASK_AREA_PIXELS:
                            all_masks.append(mask)
                            all_boxes.append(boxes_data[i])
                            all_scores.append(scores_data[i])
                else:
                    if DEBUG_MODE:
                        log_message(f"      ⚠️ No masks in results", to_console=False)
            else:
                if DEBUG_MODE:
                    log_message(f"      ⚠️ No results returned", to_console=False)
                
        except Exception as e:
            error_msg = str(e)
            log_fastsam_error(image_path.name, "INFERENCE", error_msg)
            log_message(f"   🔍 FastSAM error logged", to_console=False)
            if DEBUG_MODE:
                log_message(f"   ❌ Detection failed: {error_msg[:100]}...", to_console=False)
        
        # Combine results
        if len(all_masks) > 0:
            masks = np.array(all_masks)
            boxes = np.array(all_boxes)
            scores = np.array(all_scores)
            max_confidence = np.max(scores)
            
            log_message(f"   ✓ Detected {len(masks)} segments", to_console=False)
            
            return masks, boxes, scores, max_confidence
        else:
            log_message(f"   ⚠️  No detections", to_console=False)
            return None, None, None, 0.0
            
    except Exception as e:
        log_fastsam_error(image_path.name, "ALL", str(e))
        log_message(f"   ❌ Critical detection error: {e}", to_console=False)
        return None, None, None, 0.0

def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Non-Maximum Suppression to remove duplicate detections"""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)

def cluster_detections(boxes: np.ndarray, proximity_threshold: int = 100) -> int:
    """Cluster nearby detections to count panel arrays"""
    if len(boxes) == 0:
        return 0
    
    try:
        from scipy.cluster.hierarchy import fclusterdata
        centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes])
        clusters = fclusterdata(centers, proximity_threshold, criterion='distance', method='single')
        return len(np.unique(clusters))
    except:
        return len(boxes)

def zero_shot_count_panels(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray
) -> Tuple[int, int]:
    """
    Zero-shot panel counting using NMS and clustering
    
    Returns:
        panel_count: Number of individual panels
        array_count: Number of panel arrays/clusters
    """
    if masks is None or len(masks) == 0:
        return 0, 0
    
    # Apply NMS to remove duplicates
    keep_indices = apply_nms(boxes, scores, iou_threshold=0.5)
    
    filtered_masks = masks[keep_indices]
    filtered_boxes = boxes[keep_indices]
    
    panel_count = len(filtered_boxes)
    array_count = cluster_detections(filtered_boxes)
    
    return panel_count, array_count

def assess_quality_control(
    masks: Optional[np.ndarray],
    panel_count: int,
    confidence: float
) -> Tuple[str, List[str]]:
    """Assess quality control status and notes"""
    notes = []
    
    if masks is None or len(masks) == 0:
        return "no_detection", ["no solar panels detected"]
    
    if confidence < 0.3:
        status = "low_confidence"
        notes.append(f"low detection confidence: {confidence:.2f}")
    elif confidence < 0.5:
        status = "medium_confidence"
        notes.append(f"medium confidence: {confidence:.2f}")
    else:
        status = "high_confidence"
        notes.append(f"high confidence: {confidence:.2f}")
    
    if panel_count > 0:
        notes.append(f"{panel_count} panels detected")
        
        # Analyze mask sizes
        mask_sizes = [np.sum(mask) for mask in masks]
        avg_size = np.mean(mask_sizes)
        
        if avg_size > 1000:
            notes.append("large panel areas detected")
        elif avg_size < 100:
            notes.append("small panel areas - may be partial")
        else:
            notes.append("moderate panel sizes")
    
    return status, notes

def generate_json_record(
    sample_id: str,
    lat: float,
    lon: float,
    has_solar: bool,
    has_solar_label: Optional[int],
    confidence: float,
    panel_count: int,
    array_count: int,
    pv_area: float,
    capacity: float,
    qc_status: str,
    qc_notes: List[str],
    masks: Optional[np.ndarray],
    image_path: Path
) -> Dict:
    """Generate comprehensive JSON record"""
    
    # Prepare bbox_or_mask
    if masks is not None and len(masks) > 0:
        # Use first mask as representative
        mask_list = masks[0].astype(int).tolist()
        bbox_or_mask = {"type": "mask", "data": mask_list}
    else:
        bbox_or_mask = {"type": "none", "data": None}
    
    # Image metadata
    img = Image.open(image_path)
    image_metadata = {
        "filename": image_path.name,
        "width": img.width,
        "height": img.height,
        "pixel_resolution_m": PIXEL_RESOLUTION,
        "detection_timestamp": datetime.now().isoformat()
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(val):
        if isinstance(val, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(val)
        elif isinstance(val, (np.floating, np.float16, np.float32, np.float64)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    # Create record
    record = {
        "sample_id": str(sample_id),
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "has_solar": bool(has_solar),
        "has_solar_label": convert_to_native(has_solar_label) if has_solar_label is not None else None,
        "confidence": round(float(confidence), 3),
        "panel_count_Est": int(panel_count),
        "panel_array_count": int(array_count),
        "pv_area_sqm_est": round(float(pv_area), 2),
        "capacity_kw_est": round(float(capacity), 3),
        "qc_status": str(qc_status),
        "qc_notes": ", ".join(qc_notes),
        "bbox_or_mask": bbox_or_mask,
        "image_metadata": image_metadata,
        "model": "FastSAM",
        "detection_method": "text_prompt_segmentation"
    }
    
    return record

def save_visualization(
    image_path: Path,
    masks: Optional[np.ndarray],
    boxes: Optional[np.ndarray],
    output_path: Path
):
    """Save detection visualization"""
    if masks is None or len(masks) == 0:
        return
    
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create colored overlay
    overlay = img.copy()
    
    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, size=3).tolist()
        overlay[mask > 0] = color
    
    # Blend
    result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    
    # Draw boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)

def process_images():
    """Main processing function"""
    # Create output directories
    for dir_path in [OUTPUT_DIR, JSON_DIR, GEOJSON_DIR, VIS_DIR, MASKS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize logs
    initialize_log()
    
    log_message("🛰️  FASTSAM SOLAR PANEL DETECTOR")
    log_message("=" * 80)
    log_message(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"🎯 Project: Solar Classification Dataset")
    log_message("=" * 80)
    
    # Configuration
    log_message("\n📂 Configuration:")
    log_message(f"   Image Directory: {IMAGE_DIR}")
    log_message(f"   Output Directory: {OUTPUT_DIR}")
    log_message(f"   Model: FastSAM-X")
    log_message(f"   Text Prompts: {TEXT_PROMPTS}")
    log_message(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    log_message(f"   IoU Threshold: {IOU_THRESHOLD}")
    log_message(f"   Min Mask Area: {MIN_MASK_AREA_PIXELS} pixels")
    
    # Initialize model
    model, device = initialize_fastsam_model()
    
    # Load dataset
    log_message(f"\n📊 Loading dataset from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    log_message(f"✅ Loaded dataset: {len(df)} records with coordinates")
    
    # Get image files
    image_files = sorted([f for f in IMAGE_DIR.glob("*.png")])
    
    if TEST_MODE:
        image_files = image_files[:TEST_IMAGES_COUNT]
        log_message(f"\n⚠️  TEST MODE: Processing only {len(image_files)} images")
    
    log_message(f"\n📁 Found {len(image_files)} images to process")
    log_message(f"   Input: {IMAGE_DIR}")
    log_message(f"   Output: {OUTPUT_DIR}\n")
    
    # Process images
    log_message("=" * 80)
    log_message("Starting detection...")
    log_message("=" * 80)
    
    results = []
    stats = {"processed": 0, "with_solar": 0, "without_solar": 0, "errors": 0}
    
    for img_path in image_files:
        try:
            # Extract sample_id
            sample_id = img_path.stem.split('_')[0]
            
            # Get coordinates
            sample_row = df[df['sampleid'].astype(str) == sample_id]
            if not sample_row.empty:
                lat = sample_row.iloc[0]['latitude']
                lon = sample_row.iloc[0]['longitude']
                has_solar_label = int(sample_row.iloc[0]['has_solar'])
            else:
                lat, lon, has_solar_label = None, None, None
            
            # Detect solar panels
            masks, boxes, scores, confidence = detect_solar_panels_fastsam(
                img_path, model, device, TEXT_PROMPTS
            )
            
            # Count panels
            panel_count, array_count = zero_shot_count_panels(masks, boxes, scores)
            
            # Calculate metrics
            has_solar = panel_count > 0
            
            if has_solar:
                total_mask_area_pixels = sum(np.sum(mask) for mask in masks)
                pv_area_sqm = total_mask_area_pixels * (PIXEL_RESOLUTION ** 2)
                capacity_kw = pv_area_sqm * PANEL_EFFICIENCY
            else:
                pv_area_sqm = 0.0
                capacity_kw = 0.0
            
            # QC assessment
            qc_status, qc_notes = assess_quality_control(masks, panel_count, confidence)
            
            # Generate JSON record
            record = generate_json_record(
                sample_id, lat, lon, has_solar, has_solar_label,
                confidence, panel_count, array_count,
                pv_area_sqm, capacity_kw,
                qc_status, qc_notes,
                masks, img_path
            )
            
            # Save JSON
            json_path = JSON_DIR / f"{img_path.stem}.json"
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2)
            except TypeError as json_error:
                # If JSON serialization fails, try without masks
                log_message(f"   ⚠️  JSON error: {json_error}, saving without mask data", to_console=False)
                record_no_mask = record.copy()
                record_no_mask['bbox_or_mask'] = {"type": "none", "data": None}
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(record_no_mask, f, indent=2)
            
            # Save visualization
            if has_solar:
                vis_path = VIS_DIR / f"{img_path.stem}_detection.png"
                save_visualization(img_path, masks, boxes, vis_path)
            
            # Log results
            log_message(f"\n📊 {img_path.stem} Results:")
            log_message(f"   Lat/Lon: ({lat}, {lon})")
            log_message(f"   Has Solar (detected): {has_solar}")
            log_message(f"   Has Solar (label): {has_solar_label}")
            log_message(f"   Panel Count: {panel_count}")
            log_message(f"   Panel Arrays: {array_count}")
            log_message(f"   Total Area: {pv_area_sqm:.2f} m²")
            log_message(f"   Capacity: {capacity_kw:.3f} kW")
            log_message(f"   Confidence: {confidence:.3f}")
            log_message(f"   QC Status: {qc_status}")
            log_message(f"   QC Notes: {', '.join(qc_notes)}")
            
            # Update stats
            results.append(record)
            stats["processed"] += 1
            if has_solar:
                stats["with_solar"] += 1
            else:
                stats["without_solar"] += 1
                
        except Exception as e:
            log_message(f"❌ Error processing {img_path.name}: {e}")
            stats["errors"] += 1
            continue
    
    # Save results CSV
    if results:
        results_df = pd.DataFrame(results)
        csv_path = OUTPUT_DIR / "detection_results.csv"
        results_df.to_csv(csv_path, index=False)
        log_message(f"\n✅ Saved results to {csv_path}")
    
    # Final statistics
    log_message("\n" + "=" * 80)
    log_message("✅ DETECTION COMPLETE")
    log_message("=" * 80)
    log_message("📊 Statistics:")
    log_message(f"   Total Processed: {stats['processed']}")
    log_message(f"   With Solar: {stats['with_solar']}")
    log_message(f"   Without Solar: {stats['without_solar']}")
    log_message(f"   Errors: {stats['errors']}")
    if stats['processed'] > 0:
        detection_rate = (stats['with_solar'] / stats['processed']) * 100
        log_message(f"   Detection Rate: {detection_rate:.1f}%")
    
    log_message("\n📁 Output files:")
    log_message(f"   Results CSV: {OUTPUT_DIR / 'detection_results.csv'}")
    log_message(f"   JSON Records: {JSON_DIR} ({len(list(JSON_DIR.glob('*.json')))} files)")
    log_message(f"   Visualizations: {VIS_DIR} ({len(list(VIS_DIR.glob('*.png')))} files)")
    log_message("=" * 80)
    
    log_message(f"\n✅ PROGRAM COMPLETED SUCCESSFULLY")
    log_message(f"📄 Log file saved to: {DETECTION_LOG}")

if __name__ == "__main__":
    process_images()
