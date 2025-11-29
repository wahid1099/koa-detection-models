# ==========================================================
# CLEANLAB LABEL REFINEMENT
# Detect and correct noisy labels in training data
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

from aeelr_config import CFG


# ==========================================================
# LABEL ISSUE DETECTION
# ==========================================================

def detect_label_issues(model, data_generator, df, verbose=True):
    """
    Use CleanLab to detect label issues
    
    Args:
        model: Trained model
        data_generator: Data generator for predictions
        df: DataFrame with original labels
        verbose: Print progress
    
    Returns:
        Dictionary with label issues information
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        print("⚠ CleanLab not installed. Install with: pip install cleanlab")
        return None
    
    if verbose:
        print("\n" + "="*70)
        print("CLEANLAB LABEL REFINEMENT")
        print("="*70)
    
    # Get predictions
    if verbose:
        print("\n[1/4] Getting model predictions...")
    
    pred_probs = model.predict(data_generator, verbose=1 if verbose else 0)
    
    # Handle hierarchical outputs
    if isinstance(pred_probs, list):
        pred_probs = pred_probs[-1]  # Use KL grade output
    
    # Get true labels
    true_labels = data_generator.classes
    
    if verbose:
        print(f"  Predictions shape: {pred_probs.shape}")
        print(f"  True labels shape: {true_labels.shape}")
    
    # Find label issues
    if verbose:
        print("\n[2/4] Detecting label issues...")
    
    label_issues_mask = find_label_issues(
        labels=true_labels,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    # Get issue indices
    issue_indices = np.where(label_issues_mask)[0]
    n_issues = len(issue_indices)
    
    if verbose:
        print(f"  Found {n_issues} potential label issues ({n_issues/len(true_labels)*100:.2f}%)")
    
    # Compute confidence scores
    confidences = np.max(pred_probs, axis=1)
    predicted_labels = np.argmax(pred_probs, axis=1)
    
    # Categorize issues
    if verbose:
        print("\n[3/4] Categorizing issues...")
    
    # Sort by confidence (ascending)
    sorted_indices = issue_indices[np.argsort(confidences[issue_indices])]
    
    # Top issues to relabel
    n_relabel = int(len(true_labels) * CFG.CLEANLAB_RELABEL_TOP_PERCENT / 100)
    relabel_indices = sorted_indices[:n_relabel]
    
    # Next issues to down-weight
    n_downweight = int(len(true_labels) * CFG.CLEANLAB_DOWNWEIGHT_PERCENT / 100)
    downweight_indices = sorted_indices[n_relabel:n_relabel + n_downweight]
    
    if verbose:
        print(f"  Relabel: {len(relabel_indices)} samples ({len(relabel_indices)/len(true_labels)*100:.2f}%)")
        print(f"  Down-weight: {len(downweight_indices)} samples ({len(downweight_indices)/len(true_labels)*100:.2f}%)")
    
    # Create results dictionary
    results = {
        'issue_mask': label_issues_mask,
        'issue_indices': issue_indices,
        'relabel_indices': relabel_indices,
        'downweight_indices': downweight_indices,
        'pred_probs': pred_probs,
        'confidences': confidences,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels
    }
    
    if verbose:
        print("\n[4/4] Label issue detection complete!")
    
    return results


# ==========================================================
# LABEL REFINEMENT
# ==========================================================

def refine_labels(df, issue_results, verbose=True):
    """
    Relabel noisy samples based on CleanLab results
    
    Args:
        df: Original DataFrame
        issue_results: Results from detect_label_issues
        verbose: Print progress
    
    Returns:
        Refined DataFrame
    """
    if issue_results is None:
        return df
    
    df_refined = df.copy()
    
    relabel_indices = issue_results['relabel_indices']
    predicted_labels = issue_results['predicted_labels']
    
    if verbose:
        print(f"\nRelabeling {len(relabel_indices)} samples...")
    
    # Relabel
    for idx in relabel_indices:
        old_label = df_refined.iloc[idx]['labels']
        new_label_idx = predicted_labels[idx]
        new_label = CFG.CLASS_NAMES[new_label_idx]
        
        df_refined.at[df_refined.index[idx], 'labels'] = new_label
        
        if verbose and len(relabel_indices) <= 20:
            print(f"  Sample {idx}: {old_label} → {new_label}")
    
    if verbose:
        print(f"✅ Relabeled {len(relabel_indices)} samples")
    
    return df_refined


def apply_sample_weights(df, issue_results, verbose=True):
    """
    Create sample weights for down-weighting hard examples
    
    Args:
        df: DataFrame
        issue_results: Results from detect_label_issues
        verbose: Print progress
    
    Returns:
        Array of sample weights
    """
    if issue_results is None:
        return np.ones(len(df))
    
    weights = np.ones(len(df))
    
    downweight_indices = issue_results['downweight_indices']
    
    # Down-weight hard examples
    weights[downweight_indices] = 0.5
    
    if verbose:
        print(f"\nDown-weighted {len(downweight_indices)} hard examples (weight=0.5)")
    
    return weights


# ==========================================================
# VISUALIZATION AND REPORTING
# ==========================================================

def generate_label_report(df, issue_results, save_dir=None):
    """
    Generate comprehensive label issue report
    
    Args:
        df: Original DataFrame
        issue_results: Results from detect_label_issues
        save_dir: Directory to save report
    
    Returns:
        Report dictionary
    """
    if issue_results is None:
        return None
    
    if save_dir is None:
        save_dir = CFG.RESULTS_DIR
    
    print("\n" + "="*70)
    print("LABEL ISSUE REPORT")
    print("="*70)
    
    true_labels = issue_results['true_labels']
    predicted_labels = issue_results['predicted_labels']
    issue_indices = issue_results['issue_indices']
    relabel_indices = issue_results['relabel_indices']
    downweight_indices = issue_results['downweight_indices']
    confidences = issue_results['confidences']
    
    # Overall statistics
    print(f"\nTotal samples: {len(df)}")
    print(f"Label issues found: {len(issue_indices)} ({len(issue_indices)/len(df)*100:.2f}%)")
    print(f"Samples to relabel: {len(relabel_indices)} ({len(relabel_indices)/len(df)*100:.2f}%)")
    print(f"Samples to down-weight: {len(downweight_indices)} ({len(downweight_indices)/len(df)*100:.2f}%)")
    
    # Per-class statistics
    print("\nPer-class label issues:")
    for i, class_name in enumerate(CFG.CLASS_NAMES):
        class_mask = (true_labels == i)
        class_issues = np.sum(issue_results['issue_mask'][class_mask])
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            print(f"  {class_name}: {class_issues}/{class_total} ({class_issues/class_total*100:.2f}%)")
    
    # Confusion among issues
    print("\nConfusion matrix for label issues:")
    issue_true = true_labels[issue_indices]
    issue_pred = predicted_labels[issue_indices]
    
    confusion = np.zeros((CFG.NUM_CLASSES, CFG.NUM_CLASSES), dtype=int)
    for t, p in zip(issue_true, issue_pred):
        confusion[t, p] += 1
    
    print("\n" + pd.DataFrame(
        confusion,
        index=CFG.CLASS_NAMES,
        columns=CFG.CLASS_NAMES
    ).to_string())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confidence distribution
    axes[0, 0].hist(confidences, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.3f}')
    axes[0, 0].set_xlabel('Confidence', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Issues by class
    issue_counts = [np.sum(issue_results['issue_mask'][true_labels == i]) 
                    for i in range(CFG.NUM_CLASSES)]
    axes[0, 1].bar(CFG.CLASS_NAMES, issue_counts, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Class', fontsize=12)
    axes[0, 1].set_ylabel('Number of Issues', fontsize=12)
    axes[0, 1].set_title('Label Issues by Class', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # 3. Confusion heatmap
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Reds', 
                xticklabels=CFG.CLASS_NAMES, yticklabels=CFG.CLASS_NAMES,
                ax=axes[1, 0], cbar_kws={'label': 'Count'})
    axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 0].set_ylabel('True Label', fontsize=12)
    axes[1, 0].set_title('Confusion Matrix (Issues Only)', fontsize=14, fontweight='bold')
    
    # 4. Confidence vs correctness
    correct_mask = (true_labels == predicted_labels)
    axes[1, 1].scatter(confidences[correct_mask], np.ones(np.sum(correct_mask)), 
                       alpha=0.3, label='Correct', color='green', s=10)
    axes[1, 1].scatter(confidences[~correct_mask], np.zeros(np.sum(~correct_mask)),
                       alpha=0.3, label='Incorrect', color='red', s=10)
    axes[1, 1].set_xlabel('Confidence', fontsize=12)
    axes[1, 1].set_ylabel('Correctness', fontsize=12)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Incorrect', 'Correct'])
    axes[1, 1].set_title('Confidence vs Correctness', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"{save_dir}/cleanlab_report.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved report to {save_path}")
    plt.show()
    
    # Create report dict
    report = {
        'total_samples': len(df),
        'n_issues': len(issue_indices),
        'n_relabel': len(relabel_indices),
        'n_downweight': len(downweight_indices),
        'mean_confidence': np.mean(confidences),
        'confusion_matrix': confusion,
        'per_class_issues': issue_counts
    }
    
    return report


# ==========================================================
# EXAMPLE VISUALIZATION
# ==========================================================

def visualize_label_issues(df, issue_results, n_examples=10, save_dir=None):
    """
    Visualize examples of label issues
    
    Args:
        df: DataFrame with filepaths
        issue_results: Results from detect_label_issues
        n_examples: Number of examples to show
        save_dir: Directory to save visualization
    """
    import cv2
    
    if issue_results is None:
        return
    
    if save_dir is None:
        save_dir = CFG.FIGURES_DIR
    
    relabel_indices = issue_results['relabel_indices'][:n_examples]
    true_labels = issue_results['true_labels']
    predicted_labels = issue_results['predicted_labels']
    confidences = issue_results['confidences']
    
    n_cols = 5
    n_rows = (len(relabel_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, idx in enumerate(relabel_indices):
        img_path = df.iloc[idx]['filepaths']
        true_label = CFG.CLASS_NAMES[true_labels[idx]]
        pred_label = CFG.CLASS_NAMES[predicted_labels[idx]]
        conf = confidences[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {conf:.2f}',
                         fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(relabel_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    save_path = f"{save_dir}/label_issue_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved examples to {save_path}")
    plt.show()


if __name__ == "__main__":
    print("CleanLab Label Refinement Module")
    print("="*70)
    print("\nThis module provides:")
    print("  1. detect_label_issues() - Find noisy labels")
    print("  2. refine_labels() - Relabel top issues")
    print("  3. apply_sample_weights() - Down-weight hard examples")
    print("  4. generate_label_report() - Comprehensive report")
    print("  5. visualize_label_issues() - Visual examples")
    print("\nRequires: pip install cleanlab")
