"""
ULTIMATE SPECTRAL FUSION INFERENCE PIPELINE
===========================================

Complete inference pipeline for solar panel detection with:
- Enhanced Mask R-CNN predictions
- Spectral quality analysis
- Comprehensive explainability features
- Multiple output formats (JSON, CSV, visualizations, GeoJSON)

Author: Solar Detection Team
Date: December 9, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from spectral_analyzer import UltimateSpectralAnalyzer
from spectral_maskrcnn import EnhancedSolarMaskRCNN


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Paths
    'base_dir': Path(__file__).parent,
    'data_dir': Path(__file__).parent.parent.parent / "Data Analytics" / "Google_MapStaticAPI" / "images",
    'csv_path': Path(__file__).parent.parent.parent / "Data Analytics" / "EI_train_data(Sheet1).csv",
    'model_path': Path(__file__).parent / "spectral_model" / "weights" / "final_model.pth",
    'output_dir': Path(__file__).parent / "spectral_output",
    
    # Inference parameters
    'confidence_threshold': 0.70,
    'nms_threshold': 0.4,
    'min_panel_area': 100,
    'max_panel_area': 5000,
    'min_aspect_ratio': 0.3,
    'max_aspect_ratio': 3.5,
    
    # Physical parameters
    'pixel_resolution': 0.15,  # meters per pixel (zoom level 20)
    'panel_efficiency': 0.2,   # kW per m²
    
    # Quality thresholds
    'min_resolution_score': 0.5,
    'min_clarity_score': 0.3,
    'max_occlusion_percent': 40,
    
    # Spectral analysis
    'use_spectral_analysis': True,
    'spectral_confidence_threshold': 0.65,
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 2,
}


# ============================================================================
# QUALITY ANALYSIS
# ============================================================================

class ImageQualityAnalyzer:
    """Analyze image quality for solar panel detection."""
    
    @staticmethod
    def analyze_quality(img_array: np.ndarray) -> Dict:
        """
        Comprehensive image quality analysis.
        
        Returns:
            Dictionary with quality metrics and issues
        """
        h, w = img_array.shape[:2]
        
        # Resolution score
        resolution_score = min((h * w) / (640 * 640), 1.0)
        
        # Clarity score (sharpness using Laplacian variance)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity_score = min(laplacian_var / 500, 1.0)
        
        # Brightness analysis
        mean_brightness = np.mean(gray)
        brightness_ok = 30 <= mean_brightness <= 225
        
        # Detect issues
        issues = []
        
        if resolution_score < 0.5:
            issues.append('low_resolution')
        
        if clarity_score < 0.3:
            issues.append('blurry_image')
        
        if not brightness_ok:
            if mean_brightness < 30:
                issues.append('too_dark')
            else:
                issues.append('too_bright')
        
        # Shadow detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        dark_pixels = (v < 70).sum()
        dark_percent = (dark_pixels / (h * w)) * 100
        
        if dark_percent > 30:
            issues.append('heavy_shadows')
        
        # Bright spots (clouds, reflections)
        bright_pixels = (v > 240).sum()
        bright_percent = (bright_pixels / (h * w)) * 100
        
        if bright_percent > 20:
            issues.append('cloud_cover')
        
        # Occlusion detection
        occlusion_percent = max(dark_percent, bright_percent)
        
        if occlusion_percent > 40:
            issues.append('severe_occlusion')
        
        if not issues:
            issues.append('none')
        
        # Determine if verifiable
        is_verifiable = (
            resolution_score >= CONFIG['min_resolution_score'] and
            clarity_score >= CONFIG['min_clarity_score'] and
            occlusion_percent <= CONFIG['max_occlusion_percent']
        )
        
        return {
            'is_verifiable': is_verifiable,
            'quality_issues': issues,
            'resolution_score': round(resolution_score, 3),
            'clarity_score': round(clarity_score, 3),
            'mean_brightness': round(mean_brightness, 1),
            'occlusion_percent': round(occlusion_percent, 1),
        }


# ============================================================================
# DETECTION AND ANALYSIS
# ============================================================================

class SolarPanelDetector:
    """Complete solar panel detection system."""
    
    def __init__(self, model_path: str, config: Dict):
        """Initialize detector."""
        self.config = config
        self.device = config['device']
        
        # Load model
        print(f"\n🔧 Loading model from: {model_path}")
        self.model_wrapper = EnhancedSolarMaskRCNN(
            num_classes=config['num_classes'],
            device=str(self.device),
            pretrained=False
        )
        self.model_wrapper.load_model_only(model_path)
        self.model_wrapper.eval_mode()
        
        # Initialize spectral analyzer
        if config['use_spectral_analysis']:
            self.spectral_analyzer = UltimateSpectralAnalyzer()
        else:
            self.spectral_analyzer = None
        
        # Quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer()
        
        print("✅ Detector initialized successfully!")
    
    def detect(self, img_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, float, Dict]:
        """
        Run detection on image.
        
        Returns:
            masks, boxes, scores, max_confidence, spectral_info
        """
        # Convert to PIL and tensor
        img_pil = Image.fromarray(img_array)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        # Run model inference
        with torch.no_grad():
            prediction = self.model_wrapper([img_tensor.to(self.device)])[0]
        
        # Extract predictions
        masks = prediction['masks'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        # Filter by confidence
        conf_mask = scores >= self.config['confidence_threshold']
        masks = masks[conf_mask]
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        
        # Additional filtering
        filtered_masks = []
        filtered_boxes = []
        filtered_scores = []
        
        for mask, box, score in zip(masks, boxes, scores):
            # Convert mask
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            area = mask_binary.sum()
            
            # Area filter
            if area < self.config['min_panel_area'] or area > self.config['max_panel_area']:
                continue
            
            # Aspect ratio filter
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            if height > 0:
                aspect = width / height
                if aspect < self.config['min_aspect_ratio'] or aspect > self.config['max_aspect_ratio']:
                    continue
            
            filtered_masks.append(mask_binary)
            filtered_boxes.append(box)
            filtered_scores.append(score)
        
        # Convert to arrays
        if len(filtered_masks) > 0:
            masks = np.array(filtered_masks)
            boxes = np.array(filtered_boxes)
            scores = np.array(filtered_scores)
            max_confidence = float(scores.max())
        else:
            masks = np.array([])
            boxes = np.array([])
            scores = np.array([])
            max_confidence = 0.0
        
        # Spectral analysis
        spectral_info = {}
        if self.spectral_analyzer and len(masks) > 0:
            spectral_masks, spectral_boxes, spectral_confs, spectral_score_map = \
                self.spectral_analyzer.segment_from_spectral(
                    img_array, 
                    confidence_threshold=self.config['spectral_confidence_threshold']
                )
            
            spectral_info = {
                'spectral_detections': len(spectral_masks),
                'spectral_confidence_avg': float(np.mean(spectral_confs)) if spectral_confs else 0.0,
                'spectral_confidence_max': float(np.max(spectral_confs)) if spectral_confs else 0.0,
            }
        
        return masks, boxes, scores, max_confidence, spectral_info
    
    def count_panels_and_arrays(self, boxes: np.ndarray) -> Tuple[int, int]:
        """
        Count individual panels and arrays using DBSCAN clustering.
        
        Returns:
            panel_count, array_count
        """
        if len(boxes) == 0:
            return 0, 0
        
        panel_count = len(boxes)
        
        # Compute centroids
        centroids = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ])
        
        # Cluster using DBSCAN
        if len(centroids) > 1:
            clustering = DBSCAN(eps=80, min_samples=1).fit(centroids)
            array_count = len(set(clustering.labels_))
        else:
            array_count = 1
        
        return panel_count, array_count
    
    def compute_metrics(self, masks: np.ndarray) -> Dict:
        """Compute physical metrics."""
        if len(masks) == 0:
            return {
                'total_area_m2': 0.0,
                'capacity_kw': 0.0,
            }
        
        # Total pixels
        total_pixels = sum(mask.sum() for mask in masks)
        
        # Convert to m²
        pixel_area_m2 = self.config['pixel_resolution'] ** 2
        total_area_m2 = total_pixels * pixel_area_m2
        
        # Estimate capacity
        capacity_kw = total_area_m2 * self.config['panel_efficiency']
        
        return {
            'total_area_m2': round(total_area_m2, 2),
            'capacity_kw': round(capacity_kw, 3),
        }
    
    def analyze_explainability(self, masks: np.ndarray, boxes: np.ndarray, 
                               scores: np.ndarray, spectral_info: Dict) -> Tuple[List[str], str]:
        """
        Generate explainability features and reasoning.
        
        Returns:
            reason_codes, detection_reasoning
        """
        reason_codes = []
        reasoning_parts = []
        
        if len(masks) == 0:
            return [], "No solar panels detected in the image."
        
        # High confidence
        if scores.max() >= 0.85:
            reason_codes.append('high_confidence_features')
            reasoning_parts.append(f"High confidence detections (max: {scores.max():.2f})")
        
        # Spectral validation
        if spectral_info:
            if spectral_info.get('spectral_detections', 0) > 0:
                reason_codes.append('spectral_signature_match')
                reasoning_parts.append(f"Spectral analysis confirms {spectral_info['spectral_detections']} panels")
        
        # Multiple detections
        if len(masks) >= 4:
            reason_codes.append('module_grid')
            reasoning_parts.append(f"Multiple panels detected ({len(masks)} total)")
        
        # Geometric regularity
        if len(boxes) >= 3:
            # Check for alignment
            centroids_y = [(b[1] + b[3]) / 2 for b in boxes]
            y_std = np.std(centroids_y)
            
            if y_std < 50:
                reason_codes.append('uniform_spacing')
                reasoning_parts.append("Panels arranged in regular pattern")
        
        # Rectangular shapes
        rect_count = 0
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect = width / height if height > 0 else 0
            
            if 0.4 <= aspect <= 2.5:
                rect_count += 1
        
        if rect_count >= len(boxes) * 0.7:
            reason_codes.append('rectilinear_array')
            reasoning_parts.append("Rectangular panel shapes detected")
        
        # Panel characteristics
        if scores.mean() >= 0.75:
            reason_codes.append('consistent_detection_quality')
            reasoning_parts.append(f"Consistent detection quality (avg: {scores.mean():.2f})")
        
        if not reason_codes:
            reason_codes = ['detection_made']
            reasoning_parts = ['Solar panels detected based on model predictions']
        
        detection_reasoning = ". ".join(reasoning_parts) + "."
        
        return reason_codes, detection_reasoning


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_visualization(img_array: np.ndarray, masks: np.ndarray, boxes: np.ndarray,
                      scores: np.ndarray, output_path: Path, sample_id: str):
    """Save visualization with masks and boxes."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display image
    ax.imshow(img_array)
    
    # Draw masks and boxes
    for mask, box, score in zip(masks, boxes, scores):
        # Draw mask overlay
        colored_mask = np.zeros_like(img_array)
        colored_mask[mask > 0] = [0, 255, 0]
        
        # Blend with image
        alpha = 0.4
        img_overlay = img_array.copy()
        img_overlay[mask > 0] = (
            img_array[mask > 0] * (1 - alpha) +
            colored_mask[mask > 0] * alpha
        ).astype(np.uint8)
        
        ax.imshow(img_overlay)
        
        # Draw box
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime',
                                 facecolor='none')
        ax.add_patch(rect)
        
        # Add confidence label
        ax.text(x1, y1 - 5, f'{score:.2f}',
               color='lime', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title(f'Sample {sample_id}: {len(masks)} Panels Detected',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_geojson(masks: np.ndarray, sample_id: str, 
                    latitude: float, longitude: float) -> Dict:
    """
    Generate GeoJSON format for detected panels.
    
    Note: This is a simplified version. For accurate georeferencing,
    you would need to consider image bounds and pixel-to-coordinate transform.
    """
    features = []
    
    for i, mask in enumerate(masks):
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to coordinates (simplified - just pixel coords for now)
            coordinates = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            coordinates.append(coordinates[0])  # Close polygon
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coordinates]
                },
                'properties': {
                    'panel_id': i + 1,
                    'sample_id': sample_id,
                    'area_pixels': int(mask.sum()),
                }
            }
            
            features.append(feature)
    
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    return geojson


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def run_inference(config: Dict = None):
    """
    Main inference pipeline.
    
    Args:
        config: Optional configuration dictionary
    """
    # Use provided config or default
    cfg = CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Create output directories
    output_dir = cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'json_records').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'geojson').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("ULTIMATE SPECTRAL FUSION INFERENCE PIPELINE")
    print("="*80)
    
    # Check model exists
    if not cfg['model_path'].exists():
        print(f"\n❌ ERROR: Model not found: {cfg['model_path']}")
        print("   Please train the model first using train_spectral_fusion.py")
        return
    
    # Load CSV
    print(f"\n📄 Loading CSV: {cfg['csv_path']}")
    df = pd.read_csv(cfg['csv_path'])
    print(f"   Total samples: {len(df)}")
    
    # Initialize detector
    detector = SolarPanelDetector(str(cfg['model_path']), cfg)
    
    # Process all images
    results = []
    data_dir = cfg['data_dir']
    
    for idx, row in df.iterrows():
        sampleid = str(row['sampleid']).zfill(4)
        sample_int = int(row['sampleid'])
        has_solar_label = int(row['has_solar'])
        
        # Find image
        patterns = [
            f"{sampleid}.png",
            f"{sampleid}_1.png",
            f"{sample_int}.0_1.0.png",
        ]
        
        img_path = None
        for pattern in patterns:
            candidate = data_dir / pattern
            if candidate.exists():
                img_path = candidate
                break
        
        if not img_path:
            print(f"   ⚠️  Image not found for sample {sampleid}")
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Quality analysis
        quality_info = detector.quality_analyzer.analyze_quality(img)
        
        # Run detection
        masks, boxes, scores, max_confidence, spectral_info = detector.detect(img)
        
        # Count panels and arrays
        panel_count, array_count = detector.count_panels_and_arrays(boxes)
        
        # Compute metrics
        metrics = detector.compute_metrics(masks)
        
        # Explainability
        reason_codes, reasoning = detector.analyze_explainability(
            masks, boxes, scores, spectral_info
        )
        
        # Determine detection status
        has_solar_detected = len(masks) > 0
        prediction_correct = (has_solar_detected == bool(has_solar_label))
        
        # QC status
        qc_status = "VERIFIABLE" if quality_info['is_verifiable'] else "NOT_VERIFIABLE"
        
        # Create result record
        result = {
            'sample_id': sampleid,
            'has_solar_detected': has_solar_detected,
            'has_solar_label': has_solar_label,
            'confidence': round(max_confidence, 3),
            'panel_count': panel_count,
            'array_count': array_count,
            'total_area_m2': metrics['total_area_m2'],
            'capacity_kw': metrics['capacity_kw'],
            'qc_status': qc_status,
            'model': 'spectral_fusion_maskrcnn',
            'prediction_correct': prediction_correct,
            'reason_codes': reason_codes,
            'detection_reasoning': reasoning,
            'image_quality': quality_info,
            'detection_scores': [round(float(s), 3) for s in scores] if len(scores) > 0 else [],
            'mask_info': {
                'mask_count': len(masks),
                'total_mask_pixels': int(sum(m.sum() for m in masks)) if len(masks) > 0 else 0
            }
        }
        
        # Add spectral info if available
        if spectral_info:
            result['spectral_analysis'] = spectral_info
        
        results.append(result)
        
        # Save JSON record
        json_path = output_dir / 'json_records' / f'{sampleid}.json'
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save visualization (only if detections)
        if has_solar_detected:
            vis_path = output_dir / 'visualizations' / f'{sampleid}_spectral_fusion.png'
            save_visualization(img, masks, boxes, scores, vis_path, sampleid)
            
            # Save GeoJSON
            geojson = generate_geojson(masks, sampleid, 
                                      row.get('latitude', 0), 
                                      row.get('longitude', 0))
            geojson_path = output_dir / 'geojson' / f'{sampleid}.geojson'
            with open(geojson_path, 'w') as f:
                json.dump(geojson, f, indent=2)
        
        # Progress
        if (idx + 1) % 10 == 0:
            detected = sum(1 for r in results if r['has_solar_detected'])
            accuracy = sum(1 for r in results if r['prediction_correct']) / len(results) * 100
            print(f"   [{idx+1}/{len(df)}] Detected: {detected} | Accuracy: {accuracy:.1f}%")
    
    # Save CSV results
    print(f"\n💾 Saving results...")
    results_df = pd.DataFrame(results)
    csv_path = output_dir / 'detection_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"   ✓ CSV saved: {csv_path}")
    
    # Print summary
    print(f"\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Total samples processed: {len(results)}")
    print(f"Samples with detections: {sum(1 for r in results if r['has_solar_detected'])}")
    print(f"Samples with labels: {sum(r['has_solar_label'] for r in results)}")
    print(f"Correct predictions: {sum(1 for r in results if r['prediction_correct'])}")
    print(f"Accuracy: {sum(1 for r in results if r['prediction_correct']) / len(results) * 100:.2f}%")
    print(f"Verifiable images: {sum(1 for r in results if r['qc_status'] == 'VERIFIABLE')}")
    
    print(f"\n✅ Inference completed! Results saved to: {output_dir}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              ULTIMATE SPECTRAL FUSION INFERENCE                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        run_inference()
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
