"""
Data Analysis/spectral_analysis.py

Non-ML spectral fusion pipeline for automatic solar panel detection
in RGB aerial/satellite imagery, with shadow detection & removal.

No deep learning / trained models are used anywhere in this module.
All methods are classical computer vision / spectral analysis techniques:

    1. Multi-colour-space analysis   (HSV, Lab, YCrCb)
    2. RGB-only spectral signature   (no NIR band available -> proxy indices)
    3. Otsu / adaptive threshold segmentation
    4. K-Means colour clustering     (unsupervised, non-DL)
    5. Edge + colour combination     (Canny + contour geometry)
    6. Shadow detection (YCbCr ratio + Tsai HSI method) and
       linear shadow compensation (illumination correction)
    7. Weighted confidence fusion of methods 1-5
    8. COCO-style annotation loading + IoU / Precision / Recall / F1 evaluation

Imported from:  Segmentation with Annotation/spectral_analysis.ipynb
"""

import os
import json
import cv2
import numpy as np
from dataclasses import dataclass


# ============================================================================
# 0. CONFIG
# ============================================================================

@dataclass
class SpectralConfig:
    # Ground sample distance (m/pixel) - matches convention already used
    # elsewhere in this project (Google Maps Static API @ zoom used).
    gsd_m_per_px: float = 0.15
    capacity_kw_per_m2: float = 0.2

    # HSV thresholds for solar-panel colour range (blue-grey, low-mid V)
    # cv2 HSV: H in [0,179], S,V in [0,255]
    hsv_lower: tuple = (95, 15, 15)
    hsv_upper: tuple = (135, 160, 130)

    # Lab thresholds (cv2 Lab: L,a,b all in [0,255], neutral a=b=128)
    lab_L_range: tuple = (25, 150)
    lab_a_range: tuple = (110, 145)
    lab_b_range: tuple = (100, 140)

    # YCrCb thresholds
    ycrcb_Y_range: tuple = (30, 190)
    ycrcb_Cr_range: tuple = (110, 150)
    ycrcb_Cb_range: tuple = (110, 150)

    # K-means
    kmeans_k: int = 6
    kmeans_attempts: int = 5

    # Edge + colour
    canny_low: int = 50
    canny_high: int = 150

    # Geometric filters (pixels, tuned for ~640x640 @ ~0.15 m/px images)
    min_blob_area_px: int = 40
    max_blob_area_px: int = 20000
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 4.0

    # Fusion weights (should sum to 1.0)
    w_color: float = 0.25
    w_spectral: float = 0.20
    w_threshold: float = 0.15
    w_kmeans: float = 0.20
    w_edge: float = 0.20
    fusion_threshold: float = 0.55

    # Shadow handling
    shadow_y_k: float = 1.0          # std-dev multiplier for the shadow luma cut
    shadow_downweight: float = 0.35  # confidence multiplier inside shadow


# ============================================================================
# 1. IO HELPERS
# ============================================================================

def load_image(path):
    """Load an image from disk as an RGB uint8 array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_coco_annotations(json_path):
    """
    Load a COCO-format instances json.

    Returns:
        images_by_id[id]      -> image dict {file_name, height, width, id}
        anns_by_image_id[id]  -> list of annotation dicts (with 'segmentation')
        categories[id]        -> category name
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image_id = {}
    for ann in coco["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return images_by_id, anns_by_image_id, categories


def polygon_annotation_to_mask(ann, height, width):
    """Convert one COCO annotation (polygon OR RLE) into a binary uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    seg = ann.get("segmentation")
    if seg is None:
        return mask

    if isinstance(seg, list):  # polygon(s)
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    elif isinstance(seg, dict):  # RLE
        try:
            from pycocotools import mask as maskUtils
        except ImportError as e:
            raise ImportError(
                "RLE segmentation found but pycocotools is not installed. "
                "Run: pip install pycocotools"
            ) from e
        rle = seg
        if isinstance(rle["counts"], list):
            rle = maskUtils.frPyObjects(rle, height, width)
        m = maskUtils.decode(rle)
        mask = (m > 0).astype(np.uint8)
    return mask


def ground_truth_mask_for_image(image_id, anns_by_image_id, height, width,
                                 category_ids=None):
    """Union every annotated instance for an image into one binary GT mask."""
    gt = np.zeros((height, width), dtype=np.uint8)
    for ann in anns_by_image_id.get(image_id, []):
        if category_ids and ann["category_id"] not in category_ids:
            continue
        gt |= polygon_annotation_to_mask(ann, height, width)
    return gt


def load_mask_folder_annotation(mask_path):
    """
    Fallback loader: a single-channel PNG mask (0=background, 255=panel).
    Use this instead of load_coco_annotations() if your annotations were
    exported as raster masks rather than a COCO json.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return (m > 127).astype(np.uint8)


# ============================================================================
# 2. COLOR SPACE CONVERSIONS
# ============================================================================

def to_hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def to_ycrcb(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)


def as_uint8_rgb(rgb):
    """Convert RGB input from uint8 or [0, 1] float format to uint8."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if np.nanmax(rgb) <= 1.0 else 1.0
        rgb = rgb * scale
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================================
# 3. SHADOW DETECTION
# ============================================================================

def detect_shadow_ycbcr(rgb, k=1.0):
    """
    Luma/chroma shadow rule: shadows are darker (low Y) AND relatively
    blue (high Cb) compared to the image mean - a standard non-ML cue.
    Returns a binary mask (1 = shadow).
    """
    rgb = as_uint8_rgb(rgb)
    ycrcb = to_ycrcb(rgb).astype(np.float32)
    Y, Cb = ycrcb[..., 0], ycrcb[..., 2]

    y_mean, y_std = Y.mean(), Y.std()
    cb_mean = Cb.mean()

    dark = Y < (y_mean - k * y_std)
    blueish = Cb > cb_mean

    return np.logical_and(dark, blueish).astype(np.uint8)


def detect_shadow_hsi_tsai(rgb):
    """
    Tsai (2006) shadow index: n = (H + 1) / (I + 1)
    Shadow pixels get a high ratio because hue stays roughly constant
    while intensity drops. Threshold chosen with Otsu on n.
    """
    rgb = as_uint8_rgb(rgb)
    hsv = to_hsv(rgb).astype(np.float32)
    H = hsv[..., 0]
    I = rgb.astype(np.float32).mean(axis=2)  # intensity proxy

    n = (H + 1.0) / (I + 1.0)
    n_norm = cv2.normalize(n, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, shadow = cv2.threshold(n_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return shadow.astype(np.uint8)


def detect_shadow_mask(rgb, k=1.0, min_area_px=25):
    """
    Combine YCbCr and Tsai HSI shadow cues (intersection -> higher
    precision, avoids flagging naturally dark panels as shadow), then
    clean with morphology and drop tiny specks.
    """
    rgb = as_uint8_rgb(rgb)
    s1 = detect_shadow_ycbcr(rgb, k=k)
    s2 = detect_shadow_hsi_tsai(rgb)
    shadow = np.logical_and(s1, s2).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, kernel)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(shadow, connectivity=8)
    clean = np.zeros_like(shadow)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            clean[labels == i] = 1
    return clean


def extract_shadow_regions(rgb, shadow_mask, min_area_px=50):
    """Extract connected shadow regions and their image statistics.

    The returned region dictionaries are serialisable except for
    ``region_mask``, which is retained for pixel-accurate crops.
    """
    rgb = as_uint8_rgb(rgb)
    shadow_mask = np.asarray(shadow_mask).astype(bool)
    if shadow_mask.shape != rgb.shape[:2]:
        raise ValueError("shadow_mask must match the image height and width")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    regions = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        region_mask = labels == label
        pixels = rgb[region_mask].astype(np.float32)
        gray_pixels = gray[region_mask]
        contours, _ = cv2.findContours(
            region_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = max((cv2.arcLength(c, True) for c in contours), default=0.0)
        contour_area = max((cv2.contourArea(c) for c in contours), default=0.0)

        regions.append({
            "region_id": label,
            "area": area,
            "perimeter": float(perimeter),
            "centroid_y": float(centroids[label][1]),
            "centroid_x": float(centroids[label][0]),
            "bbox_min_row": y,
            "bbox_min_col": x,
            "bbox_max_row": y + height,
            "bbox_max_col": x + width,
            "width": width,
            "height": height,
            "solidity": float(area / contour_area) if contour_area else 0.0,
            "extent": float(area / (width * height)) if width and height else 0.0,
            "aspect_ratio": float(height / width) if width else 0.0,
            "mean_brightness": float(gray_pixels.mean()),
            "std_brightness": float(gray_pixels.std()),
            "min_brightness": float(gray_pixels.min()),
            "mean_red": float(pixels[:, 0].mean()),
            "mean_green": float(pixels[:, 1].mean()),
            "mean_blue": float(pixels[:, 2].mean()),
            "region_mask": region_mask,
        })

    regions.sort(key=lambda region: region["area"], reverse=True)
    return regions, labels


# ============================================================================
# 4. SHADOW REMOVAL (ILLUMINATION CORRECTION)
# ============================================================================

def remove_shadow_linear(rgb, shadow_mask, border_width=4):
    """
    Classic border-ratio shadow compensation (non-ML): for each channel,
    compare the mean intensity of a thin 'lit' border surrounding each
    shadow blob to the mean intensity inside the shadow, then scale the
    shadow pixels up by that ratio (clipped so it can only brighten).
    """
    if shadow_mask.sum() == 0:
        return rgb.copy()

    corrected = rgb.astype(np.float32).copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1)
    )
    dilated = cv2.dilate(shadow_mask, kernel, iterations=1)
    border = np.logical_and(dilated == 1, shadow_mask == 0)

    if border.sum() < 10:
        border = shadow_mask == 0  # fallback: whole non-shadow image

    for c in range(3):
        shadow_mean = rgb[..., c][shadow_mask == 1].mean()
        lit_mean = rgb[..., c][border].mean()
        if shadow_mean < 1e-3:
            continue
        ratio = np.clip(lit_mean / shadow_mean, 1.0, 4.0)
        channel = corrected[..., c]
        channel[shadow_mask == 1] *= ratio
        corrected[..., c] = channel

    return np.clip(corrected, 0, 255).astype(np.uint8)


def remove_shadow_homomorphic(rgb, sigma=15):
    """
    Alternative global illumination normalisation: divide the Lab L
    channel by a heavily-blurred illumination estimate, keep a,b as-is.
    Useful when shadows are large/diffuse rather than crisp blobs.
    """
    lab = to_lab(rgb).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    illum = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma)
    illum = np.clip(illum, 1, 255)
    reflectance = np.clip((L / illum) * illum.mean(), 0, 255)

    lab_corrected = np.stack([reflectance, a, b], axis=-1).astype(np.uint8)
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)


# ============================================================================
# 5. METHOD 1 - COLOR SPACE ANALYSIS
# ============================================================================

def method_color_space(rgb, cfg: SpectralConfig):
    hsv = to_hsv(rgb)
    lab = to_lab(rgb)
    ycrcb = to_ycrcb(rgb)

    hsv_mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))

    lab_mask = (np.logical_and.reduce([
        lab[..., 0] >= cfg.lab_L_range[0], lab[..., 0] <= cfg.lab_L_range[1],
        lab[..., 1] >= cfg.lab_a_range[0], lab[..., 1] <= cfg.lab_a_range[1],
        lab[..., 2] >= cfg.lab_b_range[0], lab[..., 2] <= cfg.lab_b_range[1],
    ]).astype(np.uint8) * 255)

    ycrcb_mask = (np.logical_and.reduce([
        ycrcb[..., 0] >= cfg.ycrcb_Y_range[0], ycrcb[..., 0] <= cfg.ycrcb_Y_range[1],
        ycrcb[..., 1] >= cfg.ycrcb_Cr_range[0], ycrcb[..., 1] <= cfg.ycrcb_Cr_range[1],
        ycrcb[..., 2] >= cfg.ycrcb_Cb_range[0], ycrcb[..., 2] <= cfg.ycrcb_Cb_range[1],
    ]).astype(np.uint8) * 255)

    votes = ((hsv_mask > 0).astype(np.float32) +
              (lab_mask > 0).astype(np.float32) +
              (ycrcb_mask > 0).astype(np.float32))
    return (votes / 3.0).astype(np.float32)


# ============================================================================
# 6. METHOD 2 - SPECTRAL SIGNATURE DETECTION (RGB-only proxy)
# ============================================================================

def spectral_reflectance_indices(rgb):
    """
    No NIR band is available from Google Maps Static / ESRI RGB tiles,
    so this builds two RGB-only proxy indices that stand in for a
    spectral signature:

      NBI - Normalised Blueness Index: anti-reflective glass/EVA on
            panels pushes them blue-grey relative to tiled/concrete roofs.
                NBI = (B - (R+G)/2) / (B + (R+G)/2 + eps)

      DVI - Darkness / visible-dip Index: panels reflect noticeably
            less broadband visible light than most roofing material.
                DVI = 1 - (R+G+B) / (3*255)
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    eps = 1e-6

    nbi = (b - (r + g) / 2.0) / (b + (r + g) / 2.0 + eps)
    dvi = 1.0 - (r + g + b) / (3.0 * 255.0)
    return nbi, dvi


def method_spectral_signature(rgb, nbi_thresh=0.03, dvi_thresh=0.35):
    nbi, dvi = spectral_reflectance_indices(rgb)
    nbi_score = np.clip((nbi - nbi_thresh) / (0.25 - nbi_thresh), 0, 1)
    dvi_score = np.clip((dvi - dvi_thresh) / (0.75 - dvi_thresh), 0, 1)
    return (0.5 * nbi_score + 0.5 * dvi_score).astype(np.float32)


# ============================================================================
# 7. METHOD 3 - THRESHOLD-BASED SEGMENTATION
# ============================================================================

def method_threshold(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

    votes = (otsu > 0).astype(np.float32) + (adaptive > 0).astype(np.float32)
    return (votes / 2.0).astype(np.float32)


# ============================================================================
# 8. METHOD 4 - K-MEANS CLUSTERING
# ============================================================================

def method_kmeans(rgb, cfg: SpectralConfig):
    h, w = rgb.shape[:2]
    hsv = to_hsv(rgb)
    lab = to_lab(rgb)

    features = np.concatenate([
        rgb.reshape(-1, 3).astype(np.float32),
        hsv.reshape(-1, 3).astype(np.float32),
        lab.reshape(-1, 3).astype(np.float32),
    ], axis=1)
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, _ = cv2.kmeans(
        features, cfg.kmeans_k, None, criteria,
        cfg.kmeans_attempts, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(h, w)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cluster_scores = []
    for k in range(cfg.kmeans_k):
        m = labels == k
        if m.sum() == 0:
            cluster_scores.append(1e9)
            continue
        mean_intensity = gray[m].mean()
        mean_blue_bias = (rgb[..., 2][m].astype(np.float32) -
                           rgb[..., 0][m].astype(np.float32)).mean()
        # lower intensity + higher blue bias => more panel-like => lower score
        cluster_scores.append(mean_intensity - mean_blue_bias)

    panel_cluster = int(np.argmin(cluster_scores))
    return (labels == panel_cluster).astype(np.float32)


# ============================================================================
# 9. METHOD 5 - EDGE + COLOR COMBINATION
# ============================================================================

def method_edge_color(rgb, cfg: SpectralConfig, color_confidence=None):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    confidence = np.zeros(gray.shape, dtype=np.float32)
    if color_confidence is None:
        color_confidence = method_color_space(rgb, cfg)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_blob_area_px or area > cfg.max_blob_area_px:
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue

        aspect = max(rw, rh) / max(min(rw, rh), 1e-3)
        if not (cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio):
            continue

        rect_area = rw * rh
        rectangularity = area / max(rect_area, 1e-3)
        if rectangularity < 0.6:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
        region_color_conf = color_confidence[mask == 1].mean() if mask.sum() else 0.0
        score = 0.5 * rectangularity + 0.5 * region_color_conf
        confidence[mask == 1] = np.maximum(confidence[mask == 1], score)

    return confidence


# ============================================================================
# 10. FUSION
# ============================================================================

def fuse_confidence_maps(maps: dict, cfg: SpectralConfig, shadow_mask=None):
    """
    maps: {'color', 'spectral', 'threshold', 'kmeans', 'edge'} -> float32 [0,1] arrays
    """
    total = (
        cfg.w_color * maps["color"] +
        cfg.w_spectral * maps["spectral"] +
        cfg.w_threshold * maps["threshold"] +
        cfg.w_kmeans * maps["kmeans"] +
        cfg.w_edge * maps["edge"]
    )

    if shadow_mask is not None:
        # Down-weight (not delete) confidence inside detected shadow -
        # evidence there is uncertain, but geometry/edge cues can still win.
        total = np.where(shadow_mask == 1, total * cfg.shadow_downweight, total)

    return np.clip(total, 0, 1).astype(np.float32)


def confidence_to_mask(confidence, cfg: SpectralConfig):
    binary = (confidence >= cfg.fusion_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if cfg.min_blob_area_px <= area <= cfg.max_blob_area_px:
            clean[labels == i] = 1
    return clean


# ============================================================================
# 11. FULL PIPELINE
# ============================================================================

def run_pipeline(rgb, cfg: SpectralConfig = None, remove_shadows=True):
    """
    Run the complete non-ML spectral fusion pipeline on one RGB image.
    Returns every intermediate result for step-by-step notebook use
    as well as fast batch scoring.
    """
    cfg = cfg or SpectralConfig()
    rgb = as_uint8_rgb(rgb)

    shadow_mask = detect_shadow_mask(rgb, k=cfg.shadow_y_k)
    shadow_regions, shadow_labels = extract_shadow_regions(
        rgb, shadow_mask, min_area_px=25
    )

    if remove_shadows and shadow_mask.sum() > 0:
        working_rgb = remove_shadow_linear(rgb, shadow_mask)
    else:
        working_rgb = rgb.copy()

    color_conf = method_color_space(working_rgb, cfg)
    spectral_conf = method_spectral_signature(working_rgb)
    threshold_conf = method_threshold(working_rgb)
    kmeans_conf = method_kmeans(working_rgb, cfg)
    edge_conf = method_edge_color(working_rgb, cfg, color_confidence=color_conf)

    maps = {
        "color": color_conf,
        "spectral": spectral_conf,
        "threshold": threshold_conf,
        "kmeans": kmeans_conf,
        "edge": edge_conf,
    }

    fused_confidence = fuse_confidence_maps(maps, cfg, shadow_mask=shadow_mask)
    predicted_mask = confidence_to_mask(fused_confidence, cfg)

    return {
        "shadow_mask": shadow_mask,
        "shadow_regions": shadow_regions,
        "shadow_labels": shadow_labels,
        "shadow_corrected_rgb": working_rgb,
        "method_maps": maps,
        "fused_confidence": fused_confidence,
        "predicted_mask": predicted_mask,
    }


# ============================================================================
# 12. EVALUATION
# ============================================================================

def compute_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(inter) / float(union)


def compute_precision_recall_f1(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ============================================================================
# 13. VISUALISATION HELPERS
# ============================================================================

def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.45):
    colored = np.zeros_like(rgb)
    colored[mask == 1] = color
    return cv2.addWeighted(rgb, 1 - alpha, colored, alpha, 0)


def area_and_capacity(mask, cfg: SpectralConfig):
    px_count = int(mask.sum())
    area_m2 = px_count * (cfg.gsd_m_per_px ** 2)
    capacity_kw = area_m2 * cfg.capacity_kw_per_m2
    return area_m2, capacity_kw


# ============================================================================
# 14. SHADOW-REMOVAL ABLATION (measure the effect, don't assume it)
# ============================================================================

def compare_with_without_shadow_removal(rgb, gt_mask, cfg: SpectralConfig = None):
    """
    Runs the full pipeline twice on the SAME image - once with shadow
    removal enabled, once with it disabled - and scores both against the
    ground-truth mask. Use this across your annotated set before assuming
    shadow removal is a net positive on your data; it usually helps but
    can occasionally hurt (over-brightened dark roofs -> false positives).
    """
    cfg = cfg or SpectralConfig()

    out_with = run_pipeline(rgb, cfg, remove_shadows=True)
    out_without = run_pipeline(rgb, cfg, remove_shadows=False)

    m_with = compute_precision_recall_f1(out_with["predicted_mask"], gt_mask)
    m_with["iou"] = compute_iou(out_with["predicted_mask"], gt_mask)

    m_without = compute_precision_recall_f1(out_without["predicted_mask"], gt_mask)
    m_without["iou"] = compute_iou(out_without["predicted_mask"], gt_mask)

    return {"with_shadow_removal": m_with, "without_shadow_removal": m_without}


# ============================================================================
# 15. FEATURE STACK FOR DOWNSTREAM DEEP MODELS (e.g. UNet++)
# ============================================================================

def build_feature_stack(rgb, cfg: SpectralConfig = None, remove_shadows=True):
    """
    Builds a multi-channel feature stack for use as CNN input (e.g. a
    pretrained-encoder UNet++). Bundles raw colour + classical colour-space
    channels + spectral indices + the shadow mask, so a downstream network
    doesn't have to re-derive shadow-invariance or colour cues from a
    small dataset on its own.

    Channels (all float32, normalised to roughly [0,1]):
        0-2   R, G, B                (shadow-corrected if remove_shadows=True)
        3-5   H, S, V                (HSV of the corrected image)
        6-8   L, a, b                (Lab of the corrected image)
        9     NBI  (normalised blueness index, rescaled from [-1,1])
        10    DVI  (darkness index)
        11    shadow mask            (0 = lit, 1 = detected shadow)

    Returns:
        feature_stack : np.ndarray [H, W, 12] float32
        shadow_mask   : np.ndarray [H, W]     uint8 (0/1)
        corrected_rgb : np.ndarray [H, W, 3]  uint8 (shadow-corrected RGB,
                                                       for visualisation)
    """
    cfg = cfg or SpectralConfig()
    rgb = as_uint8_rgb(rgb)

    shadow_mask = detect_shadow_mask(rgb, k=cfg.shadow_y_k)
    if remove_shadows and shadow_mask.sum() > 0:
        corrected_rgb = remove_shadow_linear(rgb, shadow_mask)
    else:
        corrected_rgb = rgb.copy()

    hsv = to_hsv(corrected_rgb).astype(np.float32) / 255.0
    lab = to_lab(corrected_rgb).astype(np.float32) / 255.0
    nbi, dvi = spectral_reflectance_indices(corrected_rgb)
    nbi_norm = np.clip((nbi + 1.0) / 2.0, 0, 1)  # NBI is in [-1,1] -> [0,1]
    dvi_norm = np.clip(dvi, 0, 1)
    rgb_norm = corrected_rgb.astype(np.float32) / 255.0

    feature_stack = np.dstack([
        rgb_norm,                          # 0-2
        hsv,                               # 3-5
        lab,                               # 6-8
        nbi_norm,                          # 9
        dvi_norm,                          # 10
        shadow_mask.astype(np.float32),    # 11
    ]).astype(np.float32)

    return feature_stack, shadow_mask, corrected_rgb