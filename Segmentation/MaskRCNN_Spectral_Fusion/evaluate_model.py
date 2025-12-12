"""
EVALUATION AND COMPARISON UTILITIES
===================================

Tools for evaluating model performance and comparing with baseline methods.

Author: Solar Detection Team
Date: December 9, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def load_results(csv_path: str) -> pd.DataFrame:
    """Load detection results from CSV."""
    return pd.read_csv(csv_path)


def compute_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        df: DataFrame with columns [has_solar_detected, has_solar_label, ...]
        
    Returns:
        Dictionary with metrics
    """
    y_true = df['has_solar_label'].values
    y_pred = df['has_solar_detected'].astype(int).values
    
    # Basic metrics
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Detection rates
    total_samples = len(df)
    detected_count = y_pred.sum()
    labeled_count = y_true.sum()
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False negative rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'total_samples': total_samples,
        'detected_count': int(detected_count),
        'labeled_count': int(labeled_count),
    }


def compare_models(spectral_csv: str, baseline_csv: str = None) -> pd.DataFrame:
    """
    Compare Spectral Fusion with baseline MaskRCNN.
    
    Args:
        spectral_csv: Path to Spectral Fusion results
        baseline_csv: Path to baseline MaskRCNN results (optional)
        
    Returns:
        Comparison DataFrame
    """
    results = []
    
    # Spectral Fusion
    df_spectral = load_results(spectral_csv)
    metrics_spectral = compute_metrics(df_spectral)
    metrics_spectral['model'] = 'Spectral Fusion'
    results.append(metrics_spectral)
    
    # Baseline (if provided)
    if baseline_csv and Path(baseline_csv).exists():
        df_baseline = load_results(baseline_csv)
        metrics_baseline = compute_metrics(df_baseline)
        metrics_baseline['model'] = 'Baseline MaskRCNN'
        results.append(metrics_baseline)
    
    return pd.DataFrame(results)


def plot_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Plot model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        values = comparison_df[metric].values
        models = comparison_df['model'].values
        
        bars = ax.bar(models, values, color=['#2ecc71', '#3498db'])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved: {output_path}")


def plot_confusion_matrix(df: pd.DataFrame, output_path: str, model_name: str):
    """Plot confusion matrix."""
    y_true = df['has_solar_label'].values
    y_pred = df['has_solar_detected'].astype(int).values
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Solar', 'Solar'],
               yticklabels=['No Solar', 'Solar'],
               cbar_kws={'label': 'Count'},
               ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved: {output_path}")


def plot_confidence_distribution(df: pd.DataFrame, output_path: str):
    """Plot confidence score distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # All detections
    confidences = df[df['confidence'] > 0]['confidence'].values
    
    axes[0].hist(confidences, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Confidence Distribution (All Detections)', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axvline(x=0.70, color='red', linestyle='--', linewidth=2, label='Threshold (0.70)')
    axes[0].legend()
    
    # True positives vs False positives
    df_detected = df[df['confidence'] > 0].copy()
    tp_confs = df_detected[df_detected['prediction_correct'] & df_detected['has_solar_label']]['confidence']
    fp_confs = df_detected[df_detected['has_solar_detected'] & ~df_detected['has_solar_label']]['confidence']
    
    axes[1].hist([tp_confs, fp_confs], bins=15, color=['#2ecc71', '#e74c3c'],
                alpha=0.7, label=['True Positives', 'False Positives'], edgecolor='black')
    axes[1].set_xlabel('Confidence Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Confidence: True Positives vs False Positives', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confidence distribution saved: {output_path}")


def analyze_errors(df: pd.DataFrame, output_path: str):
    """Analyze error cases."""
    # False positives
    fp = df[(df['has_solar_detected']) & (df['has_solar_label'] == 0)].copy()
    
    # False negatives
    fn = df[(~df['has_solar_detected']) & (df['has_solar_label'] == 1)].copy()
    
    # Generate report
    report = []
    report.append("="*80)
    report.append("ERROR ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    report.append(f"Total False Positives: {len(fp)}")
    if len(fp) > 0:
        report.append(f"  Average confidence: {fp['confidence'].mean():.3f}")
        report.append(f"  Average panel count: {fp['panel_count'].mean():.1f}")
        report.append(f"  Quality issues:")
        for issue in fp['image_quality'].apply(lambda x: eval(x)['quality_issues'] if isinstance(x, str) else x.get('quality_issues', [])):
            for iss in issue:
                report.append(f"    - {iss}")
    report.append("")
    
    report.append(f"Total False Negatives: {len(fn)}")
    if len(fn) > 0:
        report.append(f"  Quality issues:")
        for issue in fn['image_quality'].apply(lambda x: eval(x)['quality_issues'] if isinstance(x, str) else x.get('quality_issues', [])):
            for iss in issue:
                report.append(f"    - {iss}")
    report.append("")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Error analysis saved: {output_path}")
    
    return fp, fn


def generate_evaluation_report(results_csv: str, output_dir: str, 
                               baseline_csv: str = None):
    """
    Generate complete evaluation report with all visualizations.
    
    Args:
        results_csv: Path to Spectral Fusion results
        output_dir: Directory to save evaluation outputs
        baseline_csv: Optional baseline results for comparison
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING EVALUATION REPORT")
    print("="*80)
    
    # Load results
    print("\n📊 Computing metrics...")
    df = load_results(results_csv)
    metrics = compute_metrics(df)
    
    # Print metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved: {metrics_path}")
    
    # Comparison (if baseline provided)
    if baseline_csv:
        print("\n📊 Comparing with baseline...")
        comparison_df = compare_models(results_csv, baseline_csv)
        print("\n" + comparison_df.to_string(index=False))
        
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ Comparison saved: {comparison_path}")
        
        plot_comparison(comparison_df, str(output_dir / 'comparison_plot.png'))
    
    # Confusion matrix
    print("\n📊 Generating confusion matrix...")
    plot_confusion_matrix(df, str(output_dir / 'confusion_matrix.png'), 
                         'Spectral Fusion')
    
    # Confidence distribution
    print("\n📊 Analyzing confidence distribution...")
    plot_confidence_distribution(df, str(output_dir / 'confidence_distribution.png'))
    
    # Error analysis
    print("\n📊 Analyzing errors...")
    fp, fn = analyze_errors(df, str(output_dir / 'error_analysis.txt'))
    
    print("\n✅ Evaluation report completed!")
    print(f"   All outputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    # Default paths
    results_csv = Path(__file__).parent / "spectral_output" / "detection_results.csv"
    baseline_csv = Path(__file__).parent.parent / "MaskRCNN_Solar" / "finetuned_output" / "detection_results.csv"
    output_dir = Path(__file__).parent / "evaluation"
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              SPECTRAL FUSION EVALUATION                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if not results_csv.exists():
        print(f"\n❌ ERROR: Results file not found: {results_csv}")
        print("   Please run inference first using inference_spectral_fusion.py")
        sys.exit(1)
    
    # Check if baseline exists
    baseline = str(baseline_csv) if baseline_csv.exists() else None
    if baseline:
        print(f"✓ Baseline results found - will include comparison")
    else:
        print(f"⚠️  No baseline results found - skipping comparison")
    
    try:
        generate_evaluation_report(
            results_csv=str(results_csv),
            output_dir=str(output_dir),
            baseline_csv=baseline
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
