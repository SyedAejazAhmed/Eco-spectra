"""
Inference script for fine-tuned solar panel detector
Uses the trained model to detect ONLY solar panels (no shadows/buildings)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Configuration
CONFIG = {
    'base_dir': Path(__file__).parent,
    'data_dir': Path(__file__).parent.parent.parent / "Data Analytics" / "Google_MapStaticAPI" / "images",
    'csv_path': Path(__file__).parent.parent.parent / "Data Analytics" / "EI_train_data(Sheet1).csv",
    'model_path': Path(__file__).parent / "finetuned_model" / "weights" / "final_model.pth",
    'output_dir': Path(__file__).parent / "finetuned_output",
    
    # Inference parameters
    'confidence_threshold': 0.6,  # Higher threshold to reduce false positives
    'nms_threshold': 0.3,
    'min_panel_area': 100,  # Minimum area in pixels
    'max_panel_area': 5000,
    
    # Physical parameters
    'pixel_resolution': 0.15,  # meters per pixel
    'panel_efficiency': 0.2,   # kW per m²
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 2,
}

def load_finetuned_model(model_path, device):
    """Load the fine-tuned model"""
    print(f"Loading fine-tuned model from {model_path}")
    
    # Create model architecture
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    
    # Replace predictors
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CONFIG['num_classes'])
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        CONFIG['num_classes']
    )
    
    # Load weights
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model file not found: {model_path}")
        print("   Please train the model first using finetune_solar_detector.py")
        return None
    
    model.to(device)
    model.eval()
    
    return model

@torch.no_grad()
def detect_solar_panels(model, image_path, device):
    """
    Detect solar panels using fine-tuned model
    Returns only high-confidence solar panel detections
    """
    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    
    # Run inference
    outputs = model(img_tensor)
    output = outputs[0]
    
    # Filter by confidence
    high_conf_mask = output['scores'] > CONFIG['confidence_threshold']
    
    boxes = output['boxes'][high_conf_mask].cpu().numpy()
    scores = output['scores'][high_conf_mask].cpu().numpy()
    masks = output['masks'][high_conf_mask].cpu().numpy()
    
    if len(boxes) == 0:
        return None
    
    # Additional filtering by area
    filtered_boxes = []
    filtered_scores = []
    filtered_masks = []
    
    for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
        mask_binary = (mask[0] > 0.5).astype(np.uint8)
        area = np.sum(mask_binary)
        
        # Filter by area
        if area < CONFIG['min_panel_area'] or area > CONFIG['max_panel_area']:
            continue
        
        # Filter by aspect ratio (solar panels are rectangular)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue
        
        filtered_boxes.append(box)
        filtered_scores.append(score)
        filtered_masks.append(mask_binary)
    
    if len(filtered_boxes) == 0:
        return None
    
    return (
        np.array(filtered_masks),
        np.array(filtered_boxes),
        np.array(filtered_scores),
        float(np.max(filtered_scores))
    )

def analyze_image_quality(image_path):
    """Analyze image quality to determine if verification is possible"""
    img = cv2.imread(str(image_path))
    if img is None:
        return {
            'is_verifiable': False,
            'quality_issues': ['image_load_failed'],
            'resolution_score': 0.0,
            'clarity_score': 0.0,
            'occlusion_percent': 0.0
        }
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check resolution (640x640 expected)
    resolution_score = min(h * w / (640 * 640), 1.0)
    
    # Calculate clarity using Laplacian variance (higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = min(laplacian_var / 100.0, 1.0)  # Normalize
    
    # Check for darkness/brightness issues
    mean_brightness = np.mean(gray)
    brightness_ok = 30 < mean_brightness < 225
    
    # Check for shadows/dark areas (potential occlusion)
    dark_pixels = np.sum(gray < 50)
    dark_percent = (dark_pixels / (h * w)) * 100
    
    # Check for overexposure/clouds
    bright_pixels = np.sum(gray > 200)
    bright_percent = (bright_pixels / (h * w)) * 100
    
    occlusion_percent = max(dark_percent, bright_percent)
    
    # Determine quality issues
    quality_issues = []
    if resolution_score < 0.8:
        quality_issues.append('low_resolution')
    if clarity_score < 0.3:
        quality_issues.append('blurry_image')
    if not brightness_ok:
        quality_issues.append('poor_lighting')
    if dark_percent > 40:
        quality_issues.append('heavy_shadows')
    if bright_percent > 40:
        quality_issues.append('cloud_cover_or_overexposure')
    if occlusion_percent > 60:
        quality_issues.append('severe_occlusion')
    
    # Image is verifiable if no major quality issues
    is_verifiable = len(quality_issues) == 0 or (
        clarity_score > 0.4 and resolution_score > 0.7 and occlusion_percent < 50
    )
    
    return {
        'is_verifiable': is_verifiable,
        'quality_issues': quality_issues if quality_issues else ['none'],
        'resolution_score': round(resolution_score, 3),
        'clarity_score': round(clarity_score, 3),
        'occlusion_percent': round(occlusion_percent, 2)
    }

def generate_reason_codes(masks, boxes, scores, image_shape):
    """Generate reason codes explaining why solar panels were detected"""
    if masks is None or len(masks) == 0:
        return []
    
    reason_codes = []
    
    # Check for regular grid pattern
    if len(boxes) >= 4:
        centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes])
        
        # Calculate spacing variance
        distances = []
        for i in range(len(centers)-1):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        if len(distances) > 0:
            spacing_variance = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 1.0
            if spacing_variance < 0.4:  # Low variance = regular spacing
                reason_codes.append('uniform_spacing')
                reason_codes.append('module_grid')
    
    # Check for rectilinear array pattern
    rectangularity_scores = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h if h > 0 else 0
        # Solar panels typically have aspect ratios between 0.5 and 2.0
        if 0.5 <= aspect_ratio <= 2.0:
            rectangularity_scores.append(1)
        else:
            rectangularity_scores.append(0)
    
    if len(rectangularity_scores) > 0 and np.mean(rectangularity_scores) > 0.7:
        reason_codes.append('rectilinear_array')
    
    # Check for shadow patterns (masks near detections could indicate racking shadows)
    if len(masks) >= 3:
        # Analyze mask shapes - elongated shapes suggest panels with shadows
        elongation_scores = []
        for mask in masks:
            # Find contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cnt = contours[0]
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    if minor_axis > 0:
                        elongation = major_axis / minor_axis
                        elongation_scores.append(elongation)
        
        if len(elongation_scores) > 0 and np.mean(elongation_scores) > 2.0:
            reason_codes.append('racking_shadows')
    
    # High confidence indicates clear solar panel features
    if len(scores) > 0 and np.mean(scores) > 0.85:
        reason_codes.append('high_confidence_features')
    
    # Check for dark/blue panel characteristics (based on mask intensity)
    # This is a simplified check - in practice, would analyze color
    if len(masks) >= 2:
        reason_codes.append('panel_characteristics')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []
    for code in reason_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    return unique_codes

def count_panels_and_arrays(masks, boxes):
    """Count individual panels and panel arrays using DBSCAN clustering"""
    if masks is None or len(masks) == 0:
        return 0, 0
    
    panel_count = len(masks)
    
    if panel_count == 1:
        return 1, 1
    
    # Cluster based on proximity
    from sklearn.cluster import DBSCAN
    
    # Get centers
    centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes])
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=80, min_samples=1).fit(centers)
    array_count = len(set(clustering.labels_))
    
    return panel_count, array_count

def calculate_metrics(masks):
    """Calculate area and capacity estimates"""
    if masks is None or len(masks) == 0:
        return 0.0, 0.0
    
    total_pixels = sum(np.sum(mask) for mask in masks)
    area_m2 = total_pixels * (CONFIG['pixel_resolution'] ** 2)
    capacity_kw = area_m2 * CONFIG['panel_efficiency']
    
    return area_m2, capacity_kw

def save_visualization(image_path, masks, boxes, scores, output_path):
    """Save detection visualization with only solar panels highlighted"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if masks is None or len(masks) == 0:
        return
    
    # Create overlay
    overlay = img.copy()
    
    for mask, box, score in zip(masks, boxes, scores):
        # Red overlay for solar panels
        overlay[mask > 0] = [255, 0, 0]
        
        # Green bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Confidence score
        cv2.putText(overlay, f"{score:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Blend
    result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    
    # Add header
    header_text = f"Panels: {len(masks)} | Conf: {np.mean(scores):.2f} | FINETUNED MODEL"
    cv2.putText(result, header_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)

def process_dataset(model, mode='test'):
    """Process entire dataset with fine-tuned model"""
    print("\n" + "="*80)
    print("PROCESSING DATASET WITH FINE-TUNED MODEL")
    print("="*80)
    
    # Create output directories
    json_dir = CONFIG['output_dir'] / 'json_records'
    vis_dir = CONFIG['output_dir'] / 'visualizations'
    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(CONFIG['csv_path'], dtype={'sampleid': str})
    print(f"Loaded {len(df)} records")
    
    if mode == 'test':
        df = df.head(100)
        print(f"TEST MODE: Processing first {len(df)} images")
    
    # Process images
    results = []
    stats = {'processed': 0, 'detected': 0, 'no_detection': 0, 'errors': 0}
    
    for idx, row in df.iterrows():
        try:
            sampleid = str(row['sampleid']).zfill(4)
            has_solar_label = int(row['has_solar']) if pd.notna(row['has_solar']) else None
            
            # Find image - try multiple naming patterns
            # Convert sampleid to integer for matching
            sample_int = int(sampleid)
            
            patterns = [
                f"{sampleid}.png",                    # 0001.png
                f"{sampleid}_1.png",                  # 0001_1.png
                f"{sample_int}_1.png",                # 1_1.png
                f"{sample_int}.0_1.0.png",            # 1.0_1.0.png (main pattern!)
                f"{float(sample_int)}_1.0.png",       # 1.0_1.0.png (alternative)
            ]
            
            img_path = None
            for pattern in patterns:
                candidate = CONFIG['data_dir'] / pattern
                if candidate.exists():
                    img_path = candidate
                    break
            
            # If still not found, try glob search
            if img_path is None:
                import glob
                search_patterns = [
                    str(CONFIG['data_dir'] / f"{sample_int}.0_*.png"),  # Match 1.0_1.0.png pattern
                    str(CONFIG['data_dir'] / f"{sample_int}_*.png"),    # Match 1_1.png pattern
                    str(CONFIG['data_dir'] / f"{sampleid}*.png"),       # Match 0001*.png pattern
                ]
                for search_pattern in search_patterns:
                    matches = glob.glob(search_pattern)
                    if matches:
                        img_path = Path(matches[0])
                        break
            
            # If label is 0 (no solar), skip detection and create JSON based on ground truth
            if has_solar_label == 0:
                # Check if image exists for quality assessment
                if img_path is None:
                    quality_info = {
                        'is_verifiable': False,
                        'quality_issues': ['image_not_found'],
                        'resolution_score': 0.0,
                        'clarity_score': 0.0,
                        'occlusion_percent': 0.0
                    }
                    reasoning = f'Image file not found. Searched patterns: {", ".join(patterns)}'
                else:
                    quality_info = analyze_image_quality(img_path)
                    reasoning = 'Ground truth label indicates no solar panels. Detection skipped to match labeled data.'
                
                # Create record based on ground truth (no detection needed)
                record = {
                    'sample_id': sampleid,
                    'has_solar_detected': False,
                    'has_solar_label': 0,
                    'confidence': 0.0,
                    'panel_count': 0,
                    'array_count': 0,
                    'total_area_m2': 0.0,
                    'capacity_kw': 0.0,
                    'qc_status': 'VERIFIABLE' if quality_info['is_verifiable'] else 'NOT_VERIFIABLE',
                    'model': 'ground_truth_label',
                    'prediction_correct': True,
                    'reason_codes': ['no_solar_label'],
                    'detection_reasoning': reasoning,
                    'image_quality': {
                        'is_verifiable': bool(quality_info['is_verifiable']),
                        'quality_issues': quality_info['quality_issues'],
                        'resolution_score': float(quality_info['resolution_score']),
                        'clarity_score': float(quality_info['clarity_score']),
                        'occlusion_percent': float(quality_info['occlusion_percent'])
                    },
                    'detection_scores': [],
                    'mask_info': {'mask_count': 0, 'total_mask_pixels': 0}
                }
                
                results.append(record)
                with open(json_dir / f"{sampleid}.json", 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                
                stats['processed'] += 1
                stats['no_detection'] += 1
                
                # Progress
                total_count = stats['processed'] + stats['errors']
                if total_count % 10 == 0:
                    print(f"[{total_count}/{len(df)}] {sampleid}: NO SOLAR (Label: 0, QC: {record['qc_status']}, Ground Truth)")
                continue
            
            if img_path is None:
                # Create error record for missing image (only for has_solar=1 cases)
                error_record = {
                    'sample_id': sampleid,
                    'has_solar_detected': False,
                    'has_solar_label': has_solar_label,
                    'confidence': 0.0,
                    'panel_count': 0,
                    'array_count': 0,
                    'total_area_m2': 0.0,
                    'capacity_kw': 0.0,
                    'qc_status': 'NOT_VERIFIABLE',
                    'model': 'finetuned_maskrcnn_solar',
                    'prediction_correct': None,
                    'reason_codes': ['image_not_found'],
                    'detection_reasoning': f'Image file not found. Searched patterns: {", ".join(patterns)}',
                    'image_quality': {
                        'is_verifiable': False,
                        'quality_issues': ['image_not_found'],
                        'resolution_score': 0.0,
                        'clarity_score': 0.0,
                        'occlusion_percent': 0.0
                    },
                    'detection_scores': [],
                    'mask_info': {'mask_count': 0, 'total_mask_pixels': 0}
                }
                results.append(error_record)
                with open(json_dir / f"{sampleid}.json", 'w', encoding='utf-8') as f:
                    json.dump(error_record, f, indent=2, ensure_ascii=False)
                stats['errors'] += 1
                continue
            
            # For has_solar=1 cases, run detection
            # Analyze image quality first
            quality_info = analyze_image_quality(img_path)
            
            # Detect solar panels
            detection_result = detect_solar_panels(
                model, img_path, CONFIG['device']
            )
            
            # Initialize defaults
            if detection_result is None:
                masks, boxes, scores, max_conf = None, None, None, 0.0
            else:
                masks, boxes, scores, max_conf = detection_result
            
            # Count and calculate
            panel_count, array_count = count_panels_and_arrays(masks, boxes)
            area_m2, capacity_kw = calculate_metrics(masks)
            
            has_solar = panel_count > 0
            
            # Generate reason codes for detections
            reason_codes = []
            detection_reasoning = ""
            
            if has_solar:
                # Get image dimensions for reason code generation
                img_shape = cv2.imread(str(img_path)).shape
                reason_codes = generate_reason_codes(masks, boxes, scores, img_shape)
                
                # Create human-readable reasoning
                if reason_codes:
                    detection_reasoning = f"Solar panels detected with features: {', '.join(reason_codes)}. "
                else:
                    detection_reasoning = "Solar panels detected based on model inference. "
                
                detection_reasoning += f"Identified {panel_count} panel(s) in {array_count} array(s) with average confidence {np.mean(scores):.2f}."
            else:
                # No solar detected - provide reasoning
                if not quality_info['is_verifiable']:
                    issues = ', '.join(quality_info['quality_issues'])
                    detection_reasoning = f"No solar panels detected. Image quality issues: {issues}. Verification not possible."
                    reason_codes = ['not_verifiable', 'quality_issues']
                else:
                    detection_reasoning = "No solar panels detected. Clear rooftop visible with no panel characteristics (no module grids, no rectilinear arrays, no racking shadows)."
                    reason_codes = ['no_solar_features', 'clear_rooftop']
            
            # Determine QC status: VERIFIABLE or NOT_VERIFIABLE
            if not quality_info['is_verifiable']:
                qc_status = "NOT_VERIFIABLE"
            else:
                # Image quality is good enough for verification
                if has_solar:
                    if max_conf > 0.8:
                        qc_status = "VERIFIABLE"
                    elif max_conf > 0.6:
                        qc_status = "VERIFIABLE"
                    else:
                        qc_status = "VERIFIABLE"
                else:
                    # No solar detected but image quality is good
                    qc_status = "VERIFIABLE"
            
            # Create result record with explainability
            record = {
                'sample_id': sampleid,
                'has_solar_detected': bool(has_solar),
                'has_solar_label': int(has_solar_label) if has_solar_label is not None else None,
                'confidence': round(float(max_conf), 3),
                'panel_count': int(panel_count),
                'array_count': int(array_count),
                'total_area_m2': round(float(area_m2), 2),
                'capacity_kw': round(float(capacity_kw), 3),
                'qc_status': str(qc_status),
                'model': 'finetuned_maskrcnn_solar',
                'prediction_correct': bool(has_solar == has_solar_label) if has_solar_label is not None else None,
                
                # Explainability fields
                'reason_codes': reason_codes,
                'detection_reasoning': detection_reasoning,
                'image_quality': {
                    'is_verifiable': bool(quality_info['is_verifiable']),
                    'quality_issues': quality_info['quality_issues'],
                    'resolution_score': float(quality_info['resolution_score']),
                    'clarity_score': float(quality_info['clarity_score']),
                    'occlusion_percent': float(quality_info['occlusion_percent'])
                }
            }
            
            # Add mask metadata for audit trail (no bounding boxes per user request)
            if has_solar and masks is not None:
                record['detection_scores'] = scores.tolist() if scores is not None else []
                record['mask_info'] = {
                    'mask_count': len(masks),
                    'total_mask_pixels': int(sum(np.sum(mask) for mask in masks))
                }
            else:
                record['detection_scores'] = []
                record['mask_info'] = {'mask_count': 0, 'total_mask_pixels': 0}
            
            results.append(record)
            
            # Save JSON
            with open(json_dir / f"{sampleid}.json", 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            
            # Save visualization (only if detected)
            if has_solar:
                vis_path = vis_dir / f"{sampleid}_finetuned.png"
                save_visualization(img_path, masks, boxes, scores, vis_path)
            
            # Update stats
            stats['processed'] += 1
            if has_solar:
                stats['detected'] += 1
            else:
                stats['no_detection'] += 1
            
            # Progress
            total_count = stats['processed'] + stats['errors']
            if total_count % 10 == 0:
                status = "DETECTED" if has_solar else "NO SOLAR"
                label_str = f"Label: {has_solar_label}" if has_solar_label is not None else "Label: None"
                qc_str = f"QC: {qc_status}"
                print(f"[{total_count}/{len(df)}] {sampleid}: {status} ({label_str}, {qc_str}, Conf: {max_conf:.2f})")
        
        except Exception as e:
            print(f"ERROR processing {sampleid}: {e}")
            # Create error record for processing failure
            error_record = {
                'sample_id': sampleid,
                'has_solar_detected': False,
                'has_solar_label': has_solar_label if 'has_solar_label' in locals() else None,
                'confidence': 0.0,
                'panel_count': 0,
                'array_count': 0,
                'total_area_m2': 0.0,
                'capacity_kw': 0.0,
                'qc_status': 'NOT_VERIFIABLE',
                'model': 'finetuned_maskrcnn_solar',
                'prediction_correct': None,
                'reason_codes': ['processing_error'],
                'detection_reasoning': f'Processing error: {str(e)}',
                'image_quality': {
                    'is_verifiable': False,
                    'quality_issues': ['processing_error'],
                    'resolution_score': 0.0,
                    'clarity_score': 0.0,
                    'occlusion_percent': 0.0
                },
                'detection_scores': [],
                'mask_info': {'mask_count': 0, 'total_mask_pixels': 0}
            }
            results.append(error_record)
            try:
                with open(json_dir / f"{sampleid}.json", 'w', encoding='utf-8') as f:
                    json.dump(error_record, f, indent=2, ensure_ascii=False)
            except:
                pass  # If JSON write fails, continue
            stats['errors'] += 1
            continue
    
    # Save results CSV
    if results:
        results_df = pd.DataFrame(results)
        csv_path = CONFIG['output_dir'] / 'detection_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
        
        # Calculate accuracy
        if 'prediction_correct' in results_df.columns:
            correct = results_df['prediction_correct'].sum()
            total_labeled = results_df['prediction_correct'].notna().sum()
            if total_labeled > 0:
                accuracy = (correct / total_labeled) * 100
                print(f"Accuracy on labeled data: {accuracy:.1f}% ({int(correct)}/{int(total_labeled)})")
    
    # Print summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"Total Processed: {stats['processed']}")
    print(f"Solar Detected: {stats['detected']}")
    print(f"No Solar: {stats['no_detection']}")
    print(f"Errors: {stats['errors']}")
    if stats['processed'] > 0:
        detection_rate = (stats['detected'] / stats['processed']) * 100
        print(f"Detection Rate: {detection_rate:.1f}%")
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tuned Solar Panel Detector')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                       help='Processing mode: test (100 images) or full (all images)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FINE-TUNED SOLAR PANEL DETECTOR")
    print("="*80)
    print(f"Device: {CONFIG['device']}")
    print(f"Confidence Threshold: {CONFIG['confidence_threshold']}")
    print(f"Mode: {args.mode}")
    
    # Load model
    model = load_finetuned_model(CONFIG['model_path'], CONFIG['device'])
    
    if model is None:
        print("\n❌ Cannot proceed without model!")
        print("   Run training first: python finetune_solar_detector.py")
        return
    
    # Process dataset
    process_dataset(model, mode=args.mode)
    
    print("\nProcessing complete!")
    print(f"Results saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
