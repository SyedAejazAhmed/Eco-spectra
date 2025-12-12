"""
SAM (Segment Anything Model) Solar Panel Detector
Uses Meta's SAM with text prompts to detect and segment solar panels in satellite imagery.
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent  # Go up to Solar Detection folder
sys.path.append(str(ROOT_DIR))

try:
    from lang_sam import LangSAM
    LANG_SAM_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: lang-sam not installed. Install with: pip install lang-sam")
    LANG_SAM_AVAILABLE = False

try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scipy not installed for advanced counting. Install with: pip install scipy")
    SCIPY_AVAILABLE = False

import torch

# ============================================
# CONFIGURATION
# ============================================

# Image source (relative to project root)
IMAGE_DIR = ROOT_DIR / "Data Analytics" / "Google_MapStaticAPI" / "images"

# CSV dataset with coordinates
CSV_DATASET = ROOT_DIR / "Data Analytics" / "EI_train_data(Sheet1).csv"

# Output directories
OUTPUT_DIR = Path(__file__).parent / "output"
MASKS_DIR = OUTPUT_DIR / "masks"
GEOJSON_DIR = OUTPUT_DIR / "geojson"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json_records"

# Log files
LOG_FILE = Path(__file__).parent / "detection_log.txt"
SAM_MODEL_LOG = Path(__file__).parent / "SAM_model_logs.txt"  # Separate SAM model error log

# Image parameters (from Google Maps Static API)
# 640x640 pixels at zoom 20 (~0.15m/pixel)
PIXEL_RESOLUTION_M = 0.15  # meters per pixel at zoom 20
PIXEL_AREA_SQM = PIXEL_RESOLUTION_M ** 2  # m² per pixel

# Solar panel estimation parameters
PANEL_EFFICIENCY_KW_PER_SQM = 0.2  # 200W/m² (standard solar panel efficiency)

# SAM text prompts
TEXT_PROMPTS = [
    "solar panel",
    "solar panels",
    "photovoltaic panel",
    "rooftop solar",
]

# Processing configuration
TEST_MODE = True  # Set to True to process only 5 images for testing
TEST_IMAGES_COUNT = 5  # Number of images to process in test mode
BATCH_SIZE = 1  # Process images one at a time for stability
MIN_MASK_AREA_PIXELS = 25  # Minimum panel area to consider (lowered for better detection)
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence score for detections (lowered)
DEBUG_MODE = True  # Enable detailed error logging

# Zero-shot counting configuration
ENABLE_ZERO_SHOT_COUNTING = True  # Enable zero-shot object counting
COUNTING_GRID_SIZE = 4  # Divide image into grid for better counting
COUNT_MERGE_THRESHOLD = 0.5  # IoU threshold for merging overlapping detections

# ============================================
# LOGGING FUNCTIONS
# ============================================

def log_message(message):
    """Log message to both console and file."""
    print(message)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def initialize_log():
    """Initialize log file with header."""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"SAM Solar Panel Detection Log\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    # Initialize SAM model log
    with open(SAM_MODEL_LOG, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"SAM MODEL ERROR LOG\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

def log_sam_error(message, image_path=None, prompt=None):
    """Log SAM model-specific errors to separate file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(SAM_MODEL_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}]\n")
        if image_path:
            f.write(f"  Image: {Path(image_path).name}\n")
        if prompt:
            f.write(f"  Prompt: {prompt}\n")
        f.write(f"  Error: {message}\n\n")

# ============================================
# SAM MODEL INITIALIZATION
# ============================================

def initialize_sam_model():
    """Initialize Lang-SAM model with text prompt capability."""
    log_message("\n🤖 Initializing SAM Model...")
    log_message(f"   PyTorch Version: {torch.__version__}")
    log_message(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log_message(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        log_message("   Using CPU (this will be slower)")
        device = "cpu"
    
    if not LANG_SAM_AVAILABLE:
        log_message("❌ Error: lang-sam package not installed!")
        log_message("   Install with: pip install lang-sam")
        return None
    
    try:
        # Initialize Lang-SAM model
        model = LangSAM()
        log_message("✅ Lang-SAM model loaded successfully!")
        return model
    except Exception as e:
        log_message(f"❌ Error loading SAM model: {str(e)}")
        return None

# ============================================
# DETECTION FUNCTIONS
# ============================================

def detect_solar_panels(model, image_path, text_prompt="solar panel"):
    """
    Detect solar panels in image using SAM with text prompt.
    Includes fallback mechanisms and better error handling.
    
    Returns:
        masks: List of binary masks (numpy arrays)
        boxes: List of bounding boxes
        scores: List of confidence scores
    """
    try:
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        
        # Try main detection
        try:
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        except RuntimeError as e:
            # Log tensor dimension errors
            error_msg = str(e)
            if "expanded size" in error_msg or "dimension" in error_msg:
                log_sam_error(f"Tensor dimension error: {error_msg[:200]}", image_path, text_prompt)
                if DEBUG_MODE:
                    log_message(f"   🔍 SAM tensor error logged")
            raise
        
        # Handle empty results
        if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
            if DEBUG_MODE:
                log_message(f"   ⚠️  No detections for prompt: '{text_prompt}'")
            return np.array([]), np.array([]), np.array([])
        
        # Convert to numpy if needed
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()
        
        # Ensure proper shape
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, :, :]
        if len(boxes.shape) == 1:
            boxes = boxes[np.newaxis, :]
        if len(logits.shape) == 0:
            logits = np.array([logits])
        
        # Filter by confidence threshold
        if len(logits) > 0:
            valid_indices = logits >= CONFIDENCE_THRESHOLD
            if np.any(valid_indices):
                masks = masks[valid_indices]
                boxes = boxes[valid_indices]
                logits = logits[valid_indices]
                if DEBUG_MODE:
                    log_message(f"   ✓ Found {np.sum(valid_indices)} valid detections (>{CONFIDENCE_THRESHOLD})")
            else:
                if DEBUG_MODE:
                    log_message(f"   ⚠️  All detections below threshold (max: {np.max(logits):.3f})")
                return np.array([]), np.array([]), np.array([])
        
        return masks, boxes, logits
        
    except Exception as e:
        error_msg = str(e)[:200]
        log_sam_error(f"Detection error: {error_msg}", image_path, text_prompt)
        if DEBUG_MODE:
            log_message(f"   ❌ Detection failed: {error_msg[:80]}")
        return np.array([]), np.array([]), np.array([])

# ============================================
# ZERO-SHOT COUNTING FUNCTIONS
# ============================================

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: Array of bounding boxes [N, 4] (x1, y1, x2, y2)
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Keep the box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])
        
        # Keep only boxes with IoU below threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return keep_indices

def zero_shot_count_panels(masks, boxes, scores):
    """
    Zero-shot counting of solar panels using advanced techniques.
    
    This function:
    1. Applies NMS to remove duplicate detections
    2. Filters small/noise detections
    3. Counts distinct panel instances
    4. Groups nearby panels into arrays
    
    Returns:
        dict with counting results
    """
    if len(masks) == 0:
        return {
            'panel_count': 0,
            'panel_array_count': 0,
            'individual_panels': [],
            'panel_arrays': []
        }
    
    # Apply NMS to remove duplicates
    keep_indices = non_maximum_suppression(boxes, scores, COUNT_MERGE_THRESHOLD)
    
    filtered_masks = masks[keep_indices]
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    
    # Filter by area
    individual_panels = []
    for i, mask in enumerate(filtered_masks):
        area_pixels = np.sum(mask > 0)
        if area_pixels >= MIN_MASK_AREA_PIXELS:
            individual_panels.append({
                'mask_idx': i,
                'area_pixels': int(area_pixels),
                'box': filtered_boxes[i].tolist(),
                'score': float(filtered_scores[i])
            })
    
    # Group nearby panels into arrays
    panel_arrays = []
    if SCIPY_AVAILABLE and len(individual_panels) > 1:
        # Calculate centers of panels
        centers = []
        for panel in individual_panels:
            box = panel['box']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Calculate pairwise distances
        distances = cdist(centers, centers, metric='euclidean')
        
        # Group panels that are close together (within 100 pixels)
        proximity_threshold = 100
        visited = set()
        
        for i in range(len(individual_panels)):
            if i in visited:
                continue
            
            # Find all panels close to this one
            close_panels = np.where(distances[i] < proximity_threshold)[0]
            
            # Create a panel array
            array_panels = [individual_panels[j] for j in close_panels]
            panel_arrays.append({
                'panels': array_panels,
                'panel_count': len(array_panels),
                'total_area_pixels': sum(p['area_pixels'] for p in array_panels)
            })
            
            # Mark as visited
            visited.update(close_panels)
    else:
        # If scipy not available, treat each panel as separate array
        for panel in individual_panels:
            panel_arrays.append({
                'panels': [panel],
                'panel_count': 1,
                'total_area_pixels': panel['area_pixels']
            })
    
    return {
        'panel_count': len(individual_panels),
        'panel_array_count': len(panel_arrays),
        'individual_panels': individual_panels,
        'panel_arrays': panel_arrays
    }

# ============================================
# AREA AND CAPACITY CALCULATION
# ============================================

def calculate_panel_metrics(masks, boxes=None, scores=None):
    """
    Calculate area and capacity from segmentation masks.
    Includes zero-shot counting if enabled.
    
    Args:
        masks: List of binary masks
        boxes: Bounding boxes (optional, for zero-shot counting)
        scores: Confidence scores (optional, for zero-shot counting)
    
    Returns:
        dict with metrics
    """
    if len(masks) == 0:
        return {
            'panel_count': 0,
            'panel_array_count': 0,
            'total_area_pixels': 0,
            'total_area_sqm': 0.0,
            'capacity_kw': 0.0,
            'has_solar': False,
            'confidence': 0.0,
            'zero_shot_count': None
        }
    
    # Zero-shot counting (if enabled and data available)
    zero_shot_result = None
    if ENABLE_ZERO_SHOT_COUNTING and boxes is not None and scores is not None:
        zero_shot_result = zero_shot_count_panels(masks, boxes, scores)
        
        # Use zero-shot count as primary count
        panel_count = zero_shot_result['panel_count']
        panel_array_count = zero_shot_result['panel_array_count']
        
        # Calculate total area from zero-shot results
        total_area_pixels = sum(p['area_pixels'] for p in zero_shot_result['individual_panels'])
    else:
        # Fallback to simple counting
        areas_pixels = []
        for mask in masks:
            mask_binary = (mask > 0).astype(np.uint8)
            area_pixels = np.sum(mask_binary)
            if area_pixels >= MIN_MASK_AREA_PIXELS:
                areas_pixels.append(area_pixels)
        
        panel_count = len(areas_pixels)
        panel_array_count = panel_count  # Assume each panel is separate
        total_area_pixels = sum(areas_pixels)
    
    # Calculate metrics
    total_area_sqm = total_area_pixels * PIXEL_AREA_SQM
    capacity_kw = total_area_sqm * PANEL_EFFICIENCY_KW_PER_SQM
    avg_confidence = float(np.mean(scores)) if scores is not None and len(scores) > 0 else 0.8
    
    return {
        'panel_count': panel_count,
        'panel_array_count': panel_array_count,
        'total_area_pixels': int(total_area_pixels),
        'total_area_sqm': round(total_area_sqm, 2),
        'capacity_kw': round(capacity_kw, 3),
        'has_solar': panel_count > 0,
        'confidence': round(avg_confidence, 3),
        'zero_shot_count': zero_shot_result
    }

# ============================================
# GEOJSON EXPORT
# ============================================

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates."""
    # Find contours
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify polygon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to coordinate list
    coordinates = approx_polygon.reshape(-1, 2).tolist()
    
    # Close polygon if needed
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    
    return coordinates

def export_geojson(sample_id, masks, metrics, output_path):
    """Export detection masks as GeoJSON polygons."""
    try:
        features = []
        
        for i, mask in enumerate(masks):
            # Get area for this mask
            mask_area_pixels = np.sum(mask > 0)
            
            # Skip if too small
            if mask_area_pixels < MIN_MASK_AREA_PIXELS:
                continue
            
            # Convert mask to polygon
            coordinates = mask_to_polygon(mask)
            if coordinates is None:
                continue
            
            # Calculate metrics for this panel
            area_sqm = mask_area_pixels * PIXEL_AREA_SQM
            capacity_kw = area_sqm * PANEL_EFFICIENCY_KW_PER_SQM
            
            # Create feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                },
                "properties": {
                    "sample_id": sample_id,
                    "panel_id": i + 1,
                    "area_pixels": int(mask_area_pixels),
                    "area_sqm": round(area_sqm, 2),
                    "capacity_kw": round(capacity_kw, 3),
                    "pixel_resolution_m": PIXEL_RESOLUTION_M
                }
            }
            features.append(feature)
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "sample_id": sample_id,
                "total_panels": metrics['panel_count'],
                "total_area_sqm": metrics['total_area_sqm'],
                "total_capacity_kw": metrics['capacity_kw'],
                "detection_method": "SAM_LangSAM"
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return True
        
    except Exception as e:
        log_message(f"   ⚠️  GeoJSON export error: {str(e)[:100]}")
        return False

# ============================================
# QC (QUALITY CONTROL) ASSESSMENT
# ============================================

def assess_quality_control(masks, boxes, scores, image_path):
    """
    Assess quality control metrics for detection.
    
    Returns:
        dict with QC status and notes
    """
    qc_notes = []
    
    # Check if any panels detected
    if len(masks) == 0:
        return {
            'qc_status': 'no_detection',
            'qc_notes': ['no solar panels detected']
        }
    
    # Assess confidence levels
    avg_confidence = 0.0
    if scores is not None and len(scores) > 0:
        avg_confidence = float(np.mean(scores))
        max_confidence = float(np.max(scores))
        
        if avg_confidence >= 0.8:
            qc_notes.append('high confidence detections')
        elif avg_confidence >= 0.5:
            qc_notes.append('moderate confidence detections')
        else:
            qc_notes.append('low confidence detections')
    
    # Check for consistent panel sizes (indicates grid pattern)
    areas = [np.sum(mask > 0) for mask in masks]
    if len(areas) > 1:
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        if area_mean > 0 and (area_std / area_mean) < 0.3:
            qc_notes.append('consistent panel sizes')
            qc_notes.append('distinct module grid')
    
    # Check mask quality
    for mask in masks:
        # Check if mask is compact (not fragmented)
        mask_binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            qc_notes.append('clear roof view')
            break
    
    # Check for rectangular shapes (typical for solar panels)
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            if 1.0 <= aspect_ratio <= 3.0:
                qc_notes.append('rectangular panel shapes')
                break
    
    # Determine overall QC status
    if avg_confidence >= 0.7 and len(qc_notes) >= 2:
        qc_status = 'verifiable'
    elif avg_confidence >= 0.5:
        qc_status = 'review_recommended'
    else:
        qc_status = 'low_confidence'
    
    # Deduplicate notes
    qc_notes = list(dict.fromkeys(qc_notes))
    
    return {
        'qc_status': qc_status,
        'qc_notes': qc_notes
    }

# ============================================
# JSON RECORD GENERATION
# ============================================

def generate_json_record(sample_id, lat, lon, has_solar_label, masks, boxes, scores, metrics, qc_result, image_source='Google_Maps_Static_API'):
    """
    Generate comprehensive JSON record for a detection result.
    
    Returns:
        dict in the required JSON format
    """
    # Generate bbox string (top detection if available)
    bbox_str = ''
    if boxes is not None and len(boxes) > 0:
        top_box = boxes[0]
        bbox_str = f"{int(top_box[0])},{int(top_box[1])},{int(top_box[2])},{int(top_box[3])}"
    
    # Generate mask info (count and total pixels)
    mask_info = ''
    if masks is not None and len(masks) > 0:
        total_mask_pixels = sum(np.sum(mask > 0) for mask in masks)
        mask_info = f"{len(masks)} masks, {int(total_mask_pixels)} pixels"
    
    # Image metadata
    image_metadata = {
        'source': image_source,
        'capture_date': 'unknown',  # Google Static API doesn't provide date
        'zoom_level': 20,
        'image_size': '640x640',
        'pixel_resolution_m': PIXEL_RESOLUTION_M
    }
    
    # Build the complete record
    record = {
        'sample_id': int(sample_id) if str(sample_id).isdigit() else sample_id,
        'lat': round(float(lat), 6) if lat else None,
        'lon': round(float(lon), 6) if lon else None,
        'has_solar': metrics['has_solar'],
        'has_solar_label': bool(has_solar_label) if has_solar_label is not None else None,
        'confidence': metrics['confidence'],
        'panel_count_est': metrics['panel_count'],
        'panel_array_count': metrics.get('panel_array_count', 0),
        'pv_area_sqm_est': metrics['total_area_sqm'],
        'capacity_kw_est': metrics['capacity_kw'],
        'qc_status': qc_result['qc_status'],
        'qc_notes': qc_result['qc_notes'],
        'bbox_or_mask': f"bbox: {bbox_str}" if bbox_str else mask_info,
        'image_metadata': image_metadata,
        'detection_method': 'SAM_LangSAM_ZeroShot',
        'zero_shot_enabled': ENABLE_ZERO_SHOT_COUNTING
    }
    
    return record

def save_json_record(record, output_path):
    """Save individual JSON record to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_message(f"   ⚠️  JSON save error: {str(e)[:100]}")
        return False

# ============================================
# VISUALIZATION
# ============================================

def visualize_detection(image_path, masks, boxes, metrics, output_path):
    """Create visualization of detected solar panels."""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = image.copy()
        
        # Draw masks
        for i, mask in enumerate(masks):
            if np.sum(mask) < MIN_MASK_AREA_PIXELS:
                continue
            
            # Create colored mask
            mask_binary = (mask > 0).astype(np.uint8)
            color = np.array([0, 255, 0], dtype=np.uint8)  # Green
            colored_mask = np.zeros_like(image)
            colored_mask[mask_binary > 0] = color
            
            # Blend with image
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add text overlay
        text_lines = [
            f"Panels: {metrics['panel_count']}",
            f"Arrays: {metrics.get('panel_array_count', 0)}",
            f"Area: {metrics['total_area_sqm']:.1f} m²",
            f"Capacity: {metrics['capacity_kw']:.2f} kW",
            f"Conf: {metrics.get('confidence', 0):.2f}"
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(overlay, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 30
        
        # Save visualization
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), overlay_bgr)
        
        return True
        
    except Exception as e:
        log_message(f"   ⚠️  Visualization error: {str(e)[:100]}")
        return False

# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def process_images(model):
    """Process all images in the input directory."""
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    MASKS_DIR.mkdir(exist_ok=True)
    GEOJSON_DIR.mkdir(exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    JSON_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load CSV dataset with coordinates
    dataset_df = None
    if CSV_DATASET.exists():
        dataset_df = pd.read_csv(CSV_DATASET, dtype={'sampleid': str})
        log_message(f"\n📊 Loaded dataset: {len(dataset_df)} records with coordinates")
    else:
        log_message(f"\n⚠️  Warning: CSV dataset not found at {CSV_DATASET}")
        log_message(f"   Proceeding without coordinate data")
    
    # Get list of images
    image_files = sorted(IMAGE_DIR.glob("*.png"))
    
    if TEST_MODE:
        image_files = image_files[:10]
        log_message(f"\n⚠️  TEST MODE: Processing only {len(image_files)} images")
    
    log_message(f"\n📁 Found {len(image_files)} images to process")
    log_message(f"   Input: {IMAGE_DIR}")
    log_message(f"   Output: {OUTPUT_DIR}\n")
    
    # Results storage
    results = []
    
    # Statistics
    total_processed = 0
    total_with_solar = 0
    total_errors = 0
    
    # Process each image
    log_message("="*80)
    log_message("Starting detection...")
    log_message("="*80)
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            sample_id = image_file.stem
            
            # Try different text prompts
            all_masks_list = []
            all_boxes_list = []
            all_scores_list = []
            
            successful_prompts = 0
            for prompt in TEXT_PROMPTS:
                masks, boxes, scores = detect_solar_panels(model, image_file, prompt)
                if len(masks) > 0:
                    successful_prompts += 1
                    all_masks_list.append(masks)
                    all_boxes_list.append(boxes)
                    all_scores_list.append(scores)
                    if DEBUG_MODE and TEST_MODE:
                        log_message(f"   ✓ Prompt '{prompt}': {len(masks)} detections")
            
            # Combine all detections
            if len(all_masks_list) > 0:
                best_masks = np.concatenate(all_masks_list, axis=0)
                best_boxes = np.concatenate(all_boxes_list, axis=0)
                best_scores = np.concatenate(all_scores_list, axis=0)
                if DEBUG_MODE and TEST_MODE:
                    log_message(f"   📊 Combined: {len(best_masks)} total detections from {successful_prompts} prompts")
            else:
                best_masks = np.array([])
                best_boxes = np.array([])
                best_scores = np.array([])
                if DEBUG_MODE and TEST_MODE:
                    log_message(f"   ⚠️  No detections from any prompt")
            
            # Calculate metrics (with zero-shot counting)
            metrics = calculate_panel_metrics(best_masks, best_boxes, best_scores)
            
            # Get coordinates from dataset
            lat, lon, has_solar_label = None, None, None
            if dataset_df is not None:
                # Extract sample ID (remove file extension and has_solar suffix)
                sample_id_clean = sample_id.split('_')[0]
                row = dataset_df[dataset_df['sampleid'] == sample_id_clean]
                if not row.empty:
                    lat = row.iloc[0]['latitude']
                    lon = row.iloc[0]['longitude']
                    has_solar_label = row.iloc[0].get('has_solar', None)
            
            # Assess quality control
            qc_result = assess_quality_control(best_masks, best_boxes, best_scores, image_file)
            
            # Generate comprehensive JSON record
            json_record = generate_json_record(
                sample_id, lat, lon, has_solar_label,
                best_masks, best_boxes, best_scores,
                metrics, qc_result
            )
            
            # Save JSON record
            json_path = JSON_OUTPUT_DIR / f"{sample_id}.json"
            save_json_record(json_record, json_path)
            
            # Export GeoJSON
            if metrics['has_solar']:
                geojson_path = GEOJSON_DIR / f"{sample_id}.geojson"
                export_geojson(sample_id, best_masks, metrics, geojson_path)
                
                # Create visualization
                viz_path = VISUALIZATIONS_DIR / f"{sample_id}_detection.png"
                visualize_detection(image_file, best_masks, best_boxes, metrics, viz_path)
            
            # Store results (using JSON record as base)
            results.append(json_record)
            
            # Log detailed results for test mode
            if TEST_MODE:
                log_message(f"\n📊 {sample_id} Results:")
                log_message(f"   Lat/Lon: ({lat}, {lon})")
                log_message(f"   Has Solar (detected): {metrics['has_solar']}")
                log_message(f"   Has Solar (label): {has_solar_label}")
                log_message(f"   Panel Count: {metrics['panel_count']}")
                log_message(f"   Panel Arrays: {metrics['panel_array_count']}")
                log_message(f"   Total Area: {metrics['total_area_sqm']:.2f} m²")
                log_message(f"   Capacity: {metrics['capacity_kw']:.3f} kW")
                log_message(f"   Confidence: {metrics['confidence']:.3f}")
                log_message(f"   QC Status: {qc_result['qc_status']}")
                log_message(f"   QC Notes: {', '.join(qc_result['qc_notes'])}")
            
            # Update statistics
            total_processed += 1
            if metrics['has_solar']:
                total_with_solar += 1
            
            # Log progress
            if total_processed % 100 == 0:
                log_message(f"✅ [{total_processed}/{len(image_files)}] "
                          f"Detected: {total_with_solar} | "
                          f"Errors: {total_errors}")
        
        except Exception as e:
            log_message(f"❌ Error processing {image_file.name}: {str(e)[:100]}")
            total_errors += 1
            continue
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / "detection_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    # Print summary
    log_message("\n" + "="*80)
    log_message("✅ DETECTION COMPLETE")
    log_message("="*80)
    log_message(f"📊 Statistics:")
    log_message(f"   Total Processed: {total_processed:,}")
    log_message(f"   With Solar: {total_with_solar:,}")
    log_message(f"   Without Solar: {total_processed - total_with_solar:,}")
    log_message(f"   Errors: {total_errors:,}")
    log_message(f"   Detection Rate: {(total_with_solar/total_processed*100):.1f}%")
    
    log_message(f"\n📁 Output files:")
    log_message(f"   Results CSV: {results_csv}")
    log_message(f"   JSON Records: {JSON_OUTPUT_DIR} ({total_processed} files)")
    log_message(f"   GeoJSON files: {GEOJSON_DIR} ({total_with_solar} files)")
    log_message(f"   Visualizations: {VISUALIZATIONS_DIR} ({total_with_solar} files)")
    
    # Calculate aggregate statistics
    if total_with_solar > 0:
        total_capacity = results_df[results_df['has_solar']]['capacity_kw'].sum()
        avg_capacity = results_df[results_df['has_solar']]['capacity_kw'].mean()
        
        log_message(f"\n⚡ Capacity Statistics:")
        log_message(f"   Total Capacity: {total_capacity:.2f} kW")
        log_message(f"   Average per Site: {avg_capacity:.2f} kW")
    
    log_message("="*80)
    
    return results_df

# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """Main execution function."""
    initialize_log()
    
    log_message("🛰️  SAM SOLAR PANEL DETECTOR")
    log_message("="*80)
    log_message(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"🎯 Project: Solar Classification Dataset")
    log_message("="*80)
    
    log_message(f"\n📂 Configuration:")
    log_message(f"   Image Directory: {IMAGE_DIR}")
    log_message(f"   Output Directory: {OUTPUT_DIR}")
    log_message(f"   Pixel Resolution: {PIXEL_RESOLUTION_M}m/pixel")
    log_message(f"   Panel Efficiency: {PANEL_EFFICIENCY_KW_PER_SQM} kW/m²")
    log_message(f"   Text Prompts: {TEXT_PROMPTS}")
    log_message(f"   Min Mask Area: {MIN_MASK_AREA_PIXELS} pixels")
    log_message(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    log_message(f"   Zero-Shot Counting: {'ENABLED' if ENABLE_ZERO_SHOT_COUNTING else 'DISABLED'}")
    if ENABLE_ZERO_SHOT_COUNTING:
        log_message(f"   Count Merge Threshold: {COUNT_MERGE_THRESHOLD}")
        log_message(f"   Scipy Available: {SCIPY_AVAILABLE}")
    # Check if image directory exists
    if not IMAGE_DIR.exists():
        log_message(f"❌ Error: Image directory not found: {IMAGE_DIR}")
        log_message(f"   Expected path: {IMAGE_DIR.absolute()}")
        return
    
    log_message(f"\n📂 Configuration:")
    log_message(f"   Image Directory: {IMAGE_DIR}")
    log_message(f"   Output Directory: {OUTPUT_DIR}")
    log_message(f"   Pixel Resolution: {PIXEL_RESOLUTION_M}m/pixel")
    log_message(f"   Panel Efficiency: {PANEL_EFFICIENCY_KW_PER_SQM} kW/m²")
    log_message(f"   Text Prompts: {TEXT_PROMPTS}")
    log_message(f"   Min Mask Area: {MIN_MASK_AREA_PIXELS} pixels")
    log_message(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    # Initialize SAM model
    model = initialize_sam_model()
    if model is None:
        log_message("\n❌ Failed to initialize SAM model. Exiting.")
        return
    
    # Process images
    results = process_images(model)
    
    log_message("\n✅ PROGRAM COMPLETED SUCCESSFULLY")
    log_message(f"📄 Log file saved to: {LOG_FILE.absolute()}")

if __name__ == "__main__":
    main()
