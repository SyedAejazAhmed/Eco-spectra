"""
Shadow Detection for Solar Panel Analysis
Detects and visualizes shadow pixels in multispectral/RGB imagery
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


class ShadowDetector:
    """
    Shadow detection using visible brightness and NIR thresholds.
    
    Shadow detection is critical for solar panel analysis because:
    1. Shadows reduce reflectance across all bands
    2. Shadow pixels distort spectral signatures
    3. Must exclude shadows from spectral averaging
    """
    
    def __init__(self, 
                 vis_threshold: float = 0.10,
                 nir_threshold: float = 0.12,
                 vis_low_threshold: float = 0.05):
        """
        Parameters:
        -----------
        vis_threshold : float
            Minimum visible brightness to be considered sunlit (0-1 scale)
        nir_threshold : float
            Minimum NIR brightness to be considered sunlit (0-1 scale)
        vis_low_threshold : float
            Very dark pixels threshold (likely deep shadows)
        """
        self.vis_threshold = vis_threshold
        self.nir_threshold = nir_threshold
        self.vis_low_threshold = vis_low_threshold
    
    def compute_visible_brightness(self, image: np.ndarray, 
                                   red_idx: int = 0, 
                                   green_idx: int = 1, 
                                   blue_idx: int = 2) -> np.ndarray:
        """
        Compute luminance using standard photometric weights.
        
        Formula: V = 0.2126*R + 0.7152*G + 0.0722*B
        (ITU-R BT.709 standard)
        
        Returns:
        --------
        np.ndarray : Visible brightness map (H x W)
        """
        r = image[:, :, red_idx]
        g = image[:, :, green_idx]
        b = image[:, :, blue_idx]
        
        # Photometric weights (human eye sensitivity)
        vis_brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        return vis_brightness
    
    def detect_shadows_rgb(self, image: np.ndarray) -> Dict:
        """
        Shadow detection using RGB only (no NIR).
        
        Uses multiple heuristics:
        1. Low brightness (V < threshold)
        2. RGB ratio analysis (shadows shift blue)
        3. Saturation analysis (shadows are less saturated)
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image (H x W x 3) normalized to [0, 1]
        
        Returns:
        --------
        dict : Shadow detection results
        """
        height, width = image.shape[:2]
        
        # Method 1: Visible brightness threshold
        vis_brightness = self.compute_visible_brightness(image)
        shadow_brightness = vis_brightness < self.vis_threshold
        
        # Method 2: RGB ratio analysis
        # Shadows often have higher blue ratio
        r = image[:, :, 0] + 1e-8
        g = image[:, :, 1] + 1e-8
        b = image[:, :, 2] + 1e-8
        
        # Shadow pixels: B/(R+G+B) is relatively high
        rgb_sum = r + g + b
        blue_ratio = b / rgb_sum
        shadow_blue = blue_ratio > 0.36  # Tune based on scene
        
        # Method 3: Low saturation (shadows are desaturated)
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] / 255.0
        shadow_desat = saturation < 0.15
        
        # Combine methods (AND for higher confidence)
        shadow_mask = shadow_brightness & (shadow_blue | shadow_desat)
        
        # Deep shadows (very dark)
        deep_shadow_mask = vis_brightness < self.vis_low_threshold
        
        # Sunlit mask (inverse)
        sunlit_mask = ~shadow_mask
        
        shadow_count = np.sum(shadow_mask)
        shadow_percent = (shadow_count / (height * width)) * 100
        
        return {
            'shadow_mask': shadow_mask,
            'deep_shadow_mask': deep_shadow_mask,
            'sunlit_mask': sunlit_mask,
            'vis_brightness': vis_brightness,
            'shadow_count': int(shadow_count),
            'shadow_percent': float(shadow_percent),
            'sunlit_count': int(np.sum(sunlit_mask)),
            'deep_shadow_count': int(np.sum(deep_shadow_mask))
        }
    
    def detect_shadows_multispectral(self, image: np.ndarray, nir_idx: int = 3) -> Dict:
        """
        Shadow detection using RGB + NIR (more robust).
        
        NIR band is particularly useful because:
        - Vegetation reflects high NIR even in partial shade
        - Shadows have low NIR reflectance
        - NIR + VIS provides better discrimination
        
        Parameters:
        -----------
        image : np.ndarray
            Multispectral image (H x W x Bands) with NIR
        nir_idx : int
            Index of NIR band
        
        Returns:
        --------
        dict : Shadow detection results
        """
        height, width = image.shape[:2]
        
        # Visible brightness
        vis_brightness = self.compute_visible_brightness(image)
        
        # NIR brightness
        nir = image[:, :, nir_idx]
        
        # Shadow rule: BOTH visible AND NIR must be low
        shadow_vis = vis_brightness < self.vis_threshold
        shadow_nir = nir < self.nir_threshold
        shadow_mask = shadow_vis & shadow_nir
        
        # Deep shadows (very dark in both)
        deep_shadow_mask = (vis_brightness < self.vis_low_threshold) & \
                          (nir < self.vis_low_threshold)
        
        # Sunlit mask
        sunlit_mask = (vis_brightness >= self.vis_threshold) & \
                     (nir >= self.nir_threshold)
        
        # Partial shadow (one threshold met but not both)
        partial_shadow = (shadow_vis | shadow_nir) & ~shadow_mask
        
        shadow_count = np.sum(shadow_mask)
        shadow_percent = (shadow_count / (height * width)) * 100
        
        return {
            'shadow_mask': shadow_mask,
            'deep_shadow_mask': deep_shadow_mask,
            'sunlit_mask': sunlit_mask,
            'partial_shadow_mask': partial_shadow,
            'vis_brightness': vis_brightness,
            'nir_brightness': nir,
            'shadow_count': int(shadow_count),
            'shadow_percent': float(shadow_percent),
            'sunlit_count': int(np.sum(sunlit_mask)),
            'partial_shadow_count': int(np.sum(partial_shadow)),
            'deep_shadow_count': int(np.sum(deep_shadow_mask))
        }
    
    def visualize_results(self, image: np.ndarray, results: Dict, 
                         save_path: str = None, show_plot: bool = True):
        """
        Visualize shadow detection results.
        
        Parameters:
        -----------
        image : np.ndarray
            Original image (RGB)
        results : dict
            Shadow detection results
        save_path : str, optional
            Path to save visualization
        show_plot : bool
            Whether to display the plot
        """
        has_nir = 'nir_brightness' in results
        
        if has_nir:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        fig.suptitle('Shadow Detection Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        ax = axes[0, 0] if has_nir else axes[0, 0]
        ax.imshow(image)
        ax.set_title('Original Image', fontweight='bold')
        ax.axis('off')
        
        # Visible brightness
        ax = axes[0, 1] if has_nir else axes[0, 1]
        im1 = ax.imshow(results['vis_brightness'], cmap='gray')
        ax.set_title(f'Visible Brightness\n(threshold={self.vis_threshold})', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046)
        
        if has_nir:
            # NIR brightness
            ax = axes[0, 2]
            im2 = ax.imshow(results['nir_brightness'], cmap='gray')
            ax.set_title(f'NIR Brightness\n(threshold={self.nir_threshold})', fontweight='bold')
            ax.axis('off')
            plt.colorbar(im2, ax=ax, fraction=0.046)
        
        # Shadow mask overlay
        ax = axes[1, 0] if has_nir else axes[1, 0]
        overlay = image.copy()
        # Red overlay for shadows
        shadow_overlay = np.zeros_like(image)
        shadow_overlay[results['shadow_mask']] = [1, 0, 0]  # Red
        overlay = overlay * 0.6 + shadow_overlay * 0.4
        ax.imshow(overlay)
        shadow_pct = results['shadow_percent']
        ax.set_title(f'Shadow Mask\n{results["shadow_count"]:,} pixels ({shadow_pct:.1f}%)', 
                    fontweight='bold', color='darkred')
        ax.axis('off')
        
        # Sunlit mask overlay
        ax = axes[1, 1] if has_nir else axes[1, 1]
        overlay = image.copy()
        # Green overlay for sunlit
        sunlit_overlay = np.zeros_like(image)
        sunlit_overlay[results['sunlit_mask']] = [0, 1, 0]  # Green
        overlay = overlay * 0.6 + sunlit_overlay * 0.4
        ax.imshow(overlay)
        sunlit_pct = (results['sunlit_count'] / (image.shape[0] * image.shape[1])) * 100
        ax.set_title(f'Sunlit Mask\n{results["sunlit_count"]:,} pixels ({sunlit_pct:.1f}%)', 
                    fontweight='bold', color='darkgreen')
        ax.axis('off')
        
        if has_nir:
            # Combined visualization
            ax = axes[1, 2]
            viz = np.zeros((image.shape[0], image.shape[1], 3))
            viz[results['sunlit_mask']] = [0, 1, 0]  # Green = sunlit
            viz[results['partial_shadow_mask']] = [1, 1, 0]  # Yellow = partial
            viz[results['shadow_mask']] = [1, 0, 0]  # Red = shadow
            viz[results['deep_shadow_mask']] = [0.5, 0, 0]  # Dark red = deep shadow
            ax.imshow(viz)
            ax.set_title('Combined Analysis\nGreen=Sunlit | Yellow=Partial | Red=Shadow', 
                        fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def load_image(image_path: str) -> np.ndarray:
    """Load and normalize image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def demo_shadow_detection():
    """Demonstrate shadow detection on sample images."""
    
    print("\n" + "█" * 70)
    print("SHADOW DETECTION FOR SOLAR PANEL ANALYSIS")
    print("█" * 70 + "\n")
    
    # Load first image from dataset
    image_dir = Path("Solar_Images_Sample")
    if not image_dir.exists():
        print(f"Error: {image_dir} not found!")
        return
    
    image_files = sorted(list(image_dir.glob("*.png")))
    if not image_files:
        print("No images found!")
        return
    
    # Process first 3 images
    detector = ShadowDetector(
        vis_threshold=0.15,    # Tune these thresholds
        nir_threshold=0.12,
        vis_low_threshold=0.08
    )
    
    for idx, img_path in enumerate(image_files[:3], 1):
        print(f"\n{'='*70}")
        print(f"Processing Image {idx}: {img_path.name}")
        print(f"{'='*70}\n")
        
        # Load image
        image = load_image(img_path)
        print(f"✓ Loaded: {image.shape}")
        
        # Create synthetic NIR band for demonstration
        nir_band = np.mean(image, axis=2) * 0.8
        multispectral = np.dstack([image, nir_band[:, :, np.newaxis]])
        
        # Detect shadows (RGB only)
        print("\n[1] RGB-based Shadow Detection:")
        results_rgb = detector.detect_shadows_rgb(image)
        print(f"  • Shadow pixels: {results_rgb['shadow_count']:,} ({results_rgb['shadow_percent']:.2f}%)")
        print(f"  • Sunlit pixels: {results_rgb['sunlit_count']:,}")
        print(f"  • Deep shadow pixels: {results_rgb['deep_shadow_count']:,}")
        
        # Detect shadows (RGB + NIR)
        print("\n[2] Multispectral (RGB+NIR) Shadow Detection:")
        results_nir = detector.detect_shadows_multispectral(multispectral, nir_idx=3)
        print(f"  • Shadow pixels: {results_nir['shadow_count']:,} ({results_nir['shadow_percent']:.2f}%)")
        print(f"  • Sunlit pixels: {results_nir['sunlit_count']:,}")
        print(f"  • Partial shadow pixels: {results_nir['partial_shadow_count']:,}")
        print(f"  • Deep shadow pixels: {results_nir['deep_shadow_count']:,}")
        
        # Visualize
        save_path = f"shadow_detection_{img_path.stem}.png"
        detector.visualize_results(
            image, 
            results_nir, 
            save_path=save_path,
            show_plot=(idx == 1)  # Show first image only
        )
        
        print(f"\n✓ Saved visualization: {save_path}")
    
    print("\n" + "█" * 70)
    print("SHADOW DETECTION SUMMARY")
    print("█" * 70)
    print("\n📊 Key Metrics:")
    print("  • Shadow pixels: Low brightness in BOTH visible AND NIR")
    print("  • Sunlit pixels: High brightness in BOTH visible AND NIR")
    print("  • Partial shadow: Mixed illumination conditions")
    print("\n🔧 Tuning Recommendations:")
    print("  • vis_threshold: 0.10-0.20 (lower = more sensitive)")
    print("  • nir_threshold: 0.12-0.18 (depends on scene)")
    print("  • For panels: exclude shadow pixels from spectral analysis")
    print("\n💡 Why Shadow Detection Matters:")
    print("  1. Shadows reduce reflectance across all bands")
    print("  2. Shadow pixels distort spectral signatures")
    print("  3. Must exclude shadows before computing panel spectra")
    print("  4. Affects power prediction accuracy")
    print("\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    demo_shadow_detection()
