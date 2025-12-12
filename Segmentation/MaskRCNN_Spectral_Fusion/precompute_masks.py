"""
PRE-COMPUTE MASKS - RUN THIS ONCE BEFORE TRAINING
=================================================

This script pre-computes all spectral masks and caches them.
Run this ONCE, then training will be 100x faster!

Estimated time: 10-15 minutes for 2,500 images
Training time after caching: 2-3 hours instead of 6 days!

Author: Solar Detection Team
Date: December 9, 2025
"""

import sys
from pathlib import Path
from fast_spectral_dataset import create_fast_dataloaders

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              PRE-COMPUTE SPECTRAL MASKS (ONE-TIME SETUP)                     ║
║                                                                              ║
║  This will compute all masks ONCE and cache them for fast training           ║
║  Time: ~10-15 minutes | Benefit: Training 100x faster!                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Configuration
csv_path = Path(__file__).parent.parent.parent / "Data Analytics" / "EI_train_data(Sheet1).csv"
data_dir = Path(__file__).parent.parent.parent / "Data Analytics" / "Google_MapStaticAPI" / "images"
cache_dir = Path(__file__).parent / "spectral_model" / "mask_cache"

# Check if data exists
if not csv_path.exists():
    print(f"❌ ERROR: CSV file not found: {csv_path}")
    sys.exit(1)

if not data_dir.exists():
    print(f"❌ ERROR: Data directory not found: {data_dir}")
    sys.exit(1)

print("✓ Data files found")
print(f"  CSV: {csv_path}")
print(f"  Images: {data_dir}")
print(f"  Cache: {cache_dir}")

# Check if cache already exists
cache_file = cache_dir / 'masks_conf85.pkl'
if cache_file.exists():
    print(f"\n⚠️  Cache already exists: {cache_file}")
    response = input("Regenerate masks? This will take 10-15 minutes. (y/n): ").strip().lower()
    if response != 'y':
        print("Using existing cache. Run training with: python train_spectral_fusion.py")
        sys.exit(0)
    force_recompute = True
else:
    print("\n🔧 No cache found - will compute masks for the first time")
    force_recompute = True

print("\n" + "="*80)
print("STARTING MASK PRE-COMPUTATION")
print("="*80)
print("\nThis will take ~10-15 minutes but only needs to be done ONCE.")
print("After this, training will complete in 2-3 hours instead of 6 days!")
print("\n" + "="*80)

try:
    # Create dataloaders (this triggers mask pre-computation)
    train_loader, val_loader = create_fast_dataloaders(
        csv_path=str(csv_path),
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        batch_size=4,
        val_split=0.2,
        random_seed=42,
        confidence_threshold=0.85,
        num_workers=0,
        force_recompute=force_recompute
    )
    
    print("\n" + "="*80)
    print("✅ MASK PRE-COMPUTATION COMPLETED!")
    print("="*80)
    print(f"\nMasks cached to: {cache_file}")
    print(f"Cache size: {cache_file.stat().st_size / (1024*1024):.1f} MB")
    
    print("\n📊 Dataset Statistics:")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    
    # Count samples with detections
    train_with_masks = sum(1 for entry in train_loader.dataset.cached_masks.values() 
                          if entry['has_detections'])
    val_with_masks = sum(1 for entry in val_loader.dataset.cached_masks.values() 
                        if entry['has_detections'])
    
    print(f"   Training samples with high-conf detections: {train_with_masks}")
    print(f"   Validation samples with high-conf detections: {val_with_masks}")
    
    print("\n🚀 Next step: Run training with:")
    print("   python train_spectral_fusion.py")
    print("\n   Training will now be FAST! (~2-3 hours)")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Pre-computation interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
