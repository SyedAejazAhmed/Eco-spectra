"""
Shadow Region Extraction and Analysis
Extracts individual shadow regions with statistics and exports them separately
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure, morphology
import pandas as pd
from typing import List, Dict, Tuple


class ShadowRegionExtractor:
    """
    Extract and analyze individual shadow regions from images.
    Provides per-region statistics, boundaries, and visualization.
    """
    
    def __init__(self,
                 vis_threshold: float = 0.15,
                 nir_threshold: float = 0.12,
                 min_shadow_area: int = 50):
        """
        Parameters:
        -----------
        vis_threshold : float
            Visible brightness threshold for shadow detection
        nir_threshold : float
            NIR brightness threshold for shadow detection
        min_shadow_area : int
            Minimum area (pixels) for valid shadow region
        """
        self.vis_threshold = vis_threshold
        self.nir_threshold = nir_threshold
        self.min_shadow_area = min_shadow_area
    
    def detect_shadow_mask(self, image: np.ndarray, nir_band: np.ndarray = None) -> np.ndarray:
        """
        Create binary shadow mask.
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image (H x W x 3) normalized to [0, 1]
        nir_band : np.ndarray, optional
            NIR band (H x W) normalized to [0, 1]
        
        Returns:
        --------
        np.ndarray : Binary shadow mask (H x W)
        """
        # Compute visible brightness (ITU-R BT.709 weights)
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        vis_brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Shadow detection
        shadow_vis = vis_brightness < self.vis_threshold
        
        if nir_band is not None:
            # Use NIR for more robust detection
            shadow_nir = nir_band < self.nir_threshold
            shadow_mask = shadow_vis & shadow_nir
        else:
            # RGB-only: add blue ratio heuristic
            blue_ratio = b / (r + g + b + 1e-8)
            shadow_blue = blue_ratio > 0.36
            shadow_mask = shadow_vis & shadow_blue
        
        return shadow_mask
    
    def extract_shadow_regions(self, image: np.ndarray, 
                               shadow_mask: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Extract individual shadow regions with connected component analysis.
        
        Parameters:
        -----------
        image : np.ndarray
            Original RGB image (H x W x 3)
        shadow_mask : np.ndarray
            Binary shadow mask (H x W)
        
        Returns:
        --------
        tuple : (regions_list, labeled_mask)
            - regions_list: List of shadow region dictionaries
            - labeled_mask: Integer mask with region labels
        """
        # Clean up mask with morphology
        shadow_mask_clean = morphology.binary_opening(shadow_mask, morphology.disk(2))
        shadow_mask_clean = morphology.binary_closing(shadow_mask_clean, morphology.disk(3))
        
        # Remove small regions
        shadow_mask_clean = morphology.remove_small_objects(
            shadow_mask_clean, 
            min_size=self.min_shadow_area
        )
        
        # Label connected components
        labeled_mask = measure.label(shadow_mask_clean, connectivity=2)
        regions_props = measure.regionprops(labeled_mask, intensity_image=image[:, :, 0])
        
        shadow_regions = []
        
        for region in regions_props:
            # Extract region properties
            minr, minc, maxr, maxc = region.bbox
            area = region.area
            centroid = region.centroid
            perimeter = region.perimeter
            
            # Compute shape metrics
            solidity = region.solidity
            extent = region.extent
            aspect_ratio = (maxr - minr) / (maxc - minc + 1e-8)
            
            # Extract region mask
            region_mask = (labeled_mask == region.label)
            
            # Compute brightness statistics within shadow region
            region_pixels = image[region_mask]
            mean_brightness = np.mean(region_pixels)
            std_brightness = np.std(region_pixels)
            min_brightness = np.min(region_pixels)
            
            # Compute RGB statistics
            mean_rgb = np.mean(region_pixels, axis=0)
            
            shadow_regions.append({
                'region_id': region.label,
                'area': int(area),
                'perimeter': float(perimeter),
                'centroid_y': float(centroid[0]),
                'centroid_x': float(centroid[1]),
                'bbox_min_row': int(minr),
                'bbox_min_col': int(minc),
                'bbox_max_row': int(maxr),
                'bbox_max_col': int(maxc),
                'width': int(maxc - minc),
                'height': int(maxr - minr),
                'solidity': float(solidity),
                'extent': float(extent),
                'aspect_ratio': float(aspect_ratio),
                'mean_brightness': float(mean_brightness),
                'std_brightness': float(std_brightness),
                'min_brightness': float(min_brightness),
                'mean_red': float(mean_rgb[0]),
                'mean_green': float(mean_rgb[1]),
                'mean_blue': float(mean_rgb[2]),
                'region_mask': region_mask
            })
        
        return shadow_regions, labeled_mask
    
    def save_shadow_regions_csv(self, shadow_regions: List[Dict], output_path: str):
        """
        Save shadow region statistics to CSV.
        
        Parameters:
        -----------
        shadow_regions : list
            List of shadow region dictionaries
        output_path : str
            Path to save CSV file
        """
        # Remove mask from data for CSV export
        csv_data = []
        for region in shadow_regions:
            data = {k: v for k, v in region.items() if k != 'region_mask'}
            csv_data.append(data)
        
        df = pd.DataFrame(csv_data)
        df = df.sort_values('area', ascending=False)
        df.to_csv(output_path, index=False)
        print(f"✓ Shadow regions CSV saved: {output_path}")
        
        return df
    
    def extract_individual_shadow_images(self, image: np.ndarray, 
                                         shadow_regions: List[Dict],
                                         output_dir: str = "shadow_regions"):
        """
        Extract and save each shadow region as individual image file.
        
        Parameters:
        -----------
        image : np.ndarray
            Original RGB image
        shadow_regions : list
            List of shadow region dictionaries
        output_dir : str
            Directory to save individual shadow images
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📁 Extracting individual shadow regions to: {output_path}")
        
        for region in shadow_regions:
            region_id = region['region_id']
            
            # Extract bounding box crop
            minr = region['bbox_min_row']
            minc = region['bbox_min_col']
            maxr = region['bbox_max_row']
            maxc = region['bbox_max_col']
            
            # Crop image
            crop = image[minr:maxr, minc:maxc].copy()
            
            # Apply mask within crop
            region_mask = region['region_mask'][minr:maxr, minc:maxc]
            
            # Create masked version (shadow region only, rest transparent)
            masked_crop = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.float32)
            masked_crop[:, :, :3] = crop
            masked_crop[:, :, 3] = region_mask.astype(np.float32)  # Alpha channel
            
            # Save both versions
            # 1. Full crop with bounding box
            crop_path = output_path / f"shadow_region_{region_id:03d}_crop.png"
            cv2.imwrite(str(crop_path), (crop * 255).astype(np.uint8)[:, :, ::-1])
            
            # 2. Masked version (shadow only)
            masked_path = output_path / f"shadow_region_{region_id:03d}_masked.png"
            cv2.imwrite(str(masked_path), (masked_crop * 255).astype(np.uint8)[:, :, [2,1,0,3]])
            
            print(f"  Region {region_id:3d}: {region['area']:6d} px | Saved to {crop_path.name}")
        
        print(f"\n✓ Extracted {len(shadow_regions)} shadow regions")
    
    def visualize_shadow_regions(self, image: np.ndarray, 
                                 shadow_regions: List[Dict],
                                 labeled_mask: np.ndarray,
                                 save_path: str = None,
                                 show_plot: bool = True):
        """
        Visualize all detected shadow regions with annotations.
        
        Parameters:
        -----------
        image : np.ndarray
            Original RGB image
        shadow_regions : list
            List of shadow region dictionaries
        labeled_mask : np.ndarray
            Labeled shadow mask
        save_path : str, optional
            Path to save visualization
        show_plot : bool
            Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Shadow Region Analysis - {len(shadow_regions)} Regions Detected', 
                    fontsize=16, fontweight='bold')
        
        # 1. Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # 2. Shadow mask overlay
        overlay = image.copy()
        shadow_overlay = np.zeros_like(image)
        shadow_overlay[labeled_mask > 0] = [1, 0, 0]  # Red for shadows
        overlay = overlay * 0.6 + shadow_overlay * 0.4
        axes[0, 1].imshow(overlay)
        total_shadow_px = np.sum(labeled_mask > 0)
        total_px = image.shape[0] * image.shape[1]
        shadow_pct = (total_shadow_px / total_px) * 100
        axes[0, 1].set_title(f'Shadow Mask Overlay\n{total_shadow_px:,} pixels ({shadow_pct:.2f}%)', 
                            fontweight='bold', fontsize=12, color='darkred')
        axes[0, 1].axis('off')
        
        # 3. Labeled regions (color-coded)
        from matplotlib import cm
        import matplotlib.patches as mpatches
        
        axes[1, 0].imshow(image)
        
        # Color map for regions
        n_regions = len(shadow_regions)
        colors = cm.rainbow(np.linspace(0, 1, n_regions))
        
        for idx, region in enumerate(shadow_regions[:50]):  # Limit to 50 for visibility
            minr = region['bbox_min_row']
            minc = region['bbox_min_col']
            maxr = region['bbox_max_row']
            maxc = region['bbox_max_col']
            
            color = colors[idx]
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor=color, linewidth=2)
            axes[1, 0].add_patch(rect)
            
            # Label with region ID and area
            label_text = f"{region['region_id']}"
            axes[1, 0].text(minc, minr - 5, label_text,
                          color='white', fontsize=8, fontweight='bold',
                          bbox=dict(facecolor=color, alpha=0.8, pad=2))
        
        axes[1, 0].set_title(f'Labeled Shadow Regions (Top 50 shown)', 
                            fontweight='bold', fontsize=12)
        axes[1, 0].axis('off')
        
        # 4. Statistics plot
        axes[1, 1].axis('off')
        
        # Sort regions by area
        sorted_regions = sorted(shadow_regions, key=lambda x: x['area'], reverse=True)
        
        # Create statistics text
        stats_text = "SHADOW REGION STATISTICS\n"
        stats_text += "=" * 50 + "\n\n"
        stats_text += f"Total Regions:     {len(shadow_regions)}\n"
        stats_text += f"Total Shadow Area: {total_shadow_px:,} px ({shadow_pct:.2f}%)\n\n"
        
        stats_text += "Top 10 Largest Shadow Regions:\n"
        stats_text += "-" * 50 + "\n"
        stats_text += f"{'ID':<5} {'Area':<10} {'Size (WxH)':<15} {'Brightness':<12}\n"
        stats_text += "-" * 50 + "\n"
        
        for region in sorted_regions[:10]:
            rid = region['region_id']
            area = region['area']
            size = f"{region['width']}x{region['height']}"
            brightness = region['mean_brightness']
            stats_text += f"{rid:<5} {area:<10} {size:<15} {brightness:<12.3f}\n"
        
        stats_text += "\n" + "=" * 50 + "\n"
        stats_text += f"\nArea Distribution:\n"
        areas = [r['area'] for r in shadow_regions]
        stats_text += f"  Mean:   {np.mean(areas):.0f} px\n"
        stats_text += f"  Median: {np.median(areas):.0f} px\n"
        stats_text += f"  Min:    {np.min(areas):.0f} px\n"
        stats_text += f"  Max:    {np.max(areas):.0f} px\n"
        
        axes[1, 1].text(0.1, 0.95, stats_text, 
                       transform=axes[1, 1].transAxes,
                       fontsize=10, fontfamily='monospace',
                       verticalalignment='top')
        axes[1, 1].set_title('Statistics', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
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


def demo_shadow_region_extraction():
    """Demonstrate shadow region extraction."""
    
    print("\n" + "█" * 70)
    print("SHADOW REGION EXTRACTION AND ANALYSIS")
    print("█" * 70 + "\n")
    
    # Load image
    image_dir = Path(r"D:\Projects\Solar Detection\Data Analytics\Google_MapStaticAPI\images")
    if not image_dir.exists():
        print(f"Error: {image_dir} not found!")
        return
    
    image_files = sorted(list(image_dir.glob("*.png")))
    if not image_files:
        print("No images found!")
        return
    
    img_path = image_files[0]  # Process first image
    print(f"Processing: {img_path.name}\n")
    
    # Load image
    image = load_image(img_path)
    height, width = image.shape[:2]
    print(f"✓ Image loaded: {width}x{height} pixels\n")
    
    # Create synthetic NIR band
    nir_band = np.mean(image, axis=2) * 0.8
    
    # Initialize extractor
    extractor = ShadowRegionExtractor(
        vis_threshold=0.15,
        nir_threshold=0.12,
        min_shadow_area=50
    )
    
    # Step 1: Detect shadow mask
    print("[1] Detecting shadow mask...")
    shadow_mask = extractor.detect_shadow_mask(image, nir_band)
    shadow_count = np.sum(shadow_mask)
    shadow_pct = (shadow_count / (height * width)) * 100
    print(f"✓ Shadow pixels detected: {shadow_count:,} ({shadow_pct:.2f}%)\n")
    
    # Step 2: Extract individual shadow regions
    print("[2] Extracting individual shadow regions...")
    shadow_regions, labeled_mask = extractor.extract_shadow_regions(image, shadow_mask)
    print(f"✓ Shadow regions found: {len(shadow_regions)}\n")
    
    # Step 3: Save statistics to CSV
    print("[3] Saving shadow region statistics...")
    csv_path = f"shadow_regions_{img_path.stem}.csv"
    df = extractor.save_shadow_regions_csv(shadow_regions, csv_path)
    print()
    
    # Step 4: Extract individual shadow images
    print("[4] Extracting individual shadow region images...")
    extractor.extract_individual_shadow_images(
        image, 
        shadow_regions,
        output_dir=f"shadow_regions_{img_path.stem}"
    )
    print()
    
    # Step 5: Visualize results
    print("[5] Creating visualization...")
    viz_path = f"shadow_regions_visualization_{img_path.stem}.png"
    extractor.visualize_shadow_regions(
        image,
        shadow_regions,
        labeled_mask,
        save_path=viz_path,
        show_plot=True
    )
    
    # Summary
    print("\n" + "█" * 70)
    print("SHADOW REGION EXTRACTION COMPLETE")
    print("█" * 70)
    print(f"\n📊 Summary:")
    print(f"  • Total shadow regions: {len(shadow_regions)}")
    print(f"  • Total shadow area: {shadow_count:,} pixels ({shadow_pct:.2f}%)")
    print(f"  • Largest region: {max(r['area'] for r in shadow_regions):,} pixels")
    print(f"  • Smallest region: {min(r['area'] for r in shadow_regions):,} pixels")
    print(f"  • Average region size: {np.mean([r['area'] for r in shadow_regions]):.0f} pixels")
    
    print(f"\n📁 Output Files:")
    print(f"  • CSV statistics: {csv_path}")
    print(f"  • Visualization: {viz_path}")
    print(f"  • Individual regions: shadow_regions_{img_path.stem}/")
    
    print("\n💡 Use Cases:")
    print("  • Exclude shadow regions from solar panel detection")
    print("  • Analyze shadow patterns for site assessment")
    print("  • Track shadow movement across time series")
    print("  • Estimate shading impact on panel performance")
    
    print("\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    demo_shadow_region_extraction()
