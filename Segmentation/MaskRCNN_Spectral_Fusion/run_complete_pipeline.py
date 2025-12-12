"""
QUICK START - TRAIN AND EVALUATE
=================================

This script runs the complete pipeline:
1. Train the model
2. Run inference
3. Generate evaluation report

Author: Solar Detection Team
Date: December 9, 2025
"""

import sys
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              ULTIMATE SPECTRAL FUSION - QUICK START                          ║
║                                                                              ║
║  This will train the model, run inference, and generate evaluation report   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Check if data exists
csv_path = Path(__file__).parent.parent.parent / "Data Analytics" / "EI_train_data(Sheet1).csv"
data_dir = Path(__file__).parent.parent.parent / "Data Analytics" / "Google_MapStaticAPI" / "images"

if not csv_path.exists():
    print(f"❌ ERROR: CSV file not found: {csv_path}")
    sys.exit(1)

if not data_dir.exists():
    print(f"❌ ERROR: Data directory not found: {data_dir}")
    sys.exit(1)

print("✓ Data files found")
print(f"  CSV: {csv_path}")
print(f"  Images: {data_dir}")

# Ask for confirmation
print("\n" + "="*80)
print("PIPELINE STEPS:")
print("="*80)
print("1. Train Enhanced Mask R-CNN with 6-channel spectral analysis (30 epochs)")
print("2. Run inference on all images")
print("3. Generate evaluation report with metrics and visualizations")
print("\nEstimated time: 2-4 hours (depending on GPU)")
print("="*80)

response = input("\nProceed? (y/n): ").strip().lower()

if response != 'y':
    print("Cancelled by user.")
    sys.exit(0)

# Step 1: Training
print("\n" + "="*80)
print("STEP 1: TRAINING")
print("="*80)

try:
    from train_spectral_fusion import train_spectral_fusion_model
    
    print("\n🚀 Starting training...")
    model, best_loss, best_epoch = train_spectral_fusion_model()
    print(f"\n✅ Training completed!")
    print(f"   Best model: Epoch {best_epoch}, Loss: {best_loss:.4f}")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Inference
print("\n" + "="*80)
print("STEP 2: INFERENCE")
print("="*80)

try:
    from inference_spectral_fusion import run_inference
    
    print("\n🔍 Starting inference...")
    run_inference()
    print("\n✅ Inference completed!")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Inference interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ ERROR during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Evaluation
print("\n" + "="*80)
print("STEP 3: EVALUATION")
print("="*80)

try:
    from evaluate_model import generate_evaluation_report
    
    results_csv = Path(__file__).parent / "spectral_output" / "detection_results.csv"
    baseline_csv = Path(__file__).parent.parent / "MaskRCNN_Solar" / "finetuned_output" / "detection_results.csv"
    output_dir = Path(__file__).parent / "evaluation"
    
    baseline = str(baseline_csv) if baseline_csv.exists() else None
    
    print("\n📊 Generating evaluation report...")
    generate_evaluation_report(
        results_csv=str(results_csv),
        output_dir=str(output_dir),
        baseline_csv=baseline
    )
    print("\n✅ Evaluation completed!")
    
except Exception as e:
    print(f"\n❌ ERROR during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nOutputs:")
print(f"  Training logs & weights: spectral_model/")
print(f"  Inference results: spectral_output/")
print(f"  Evaluation report: evaluation/")
print("\nNext steps:")
print("  1. Review training curves: spectral_model/logs/training_curves.png")
print("  2. Check metrics: evaluation/metrics.json")
print("  3. Inspect visualizations: spectral_output/visualizations/")
print("\n✅ All done!")
