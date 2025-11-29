# ==========================================================
# CALIBRATION MODULE
# Temperature scaling and calibration metrics
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import tensorflow as tf

from aeelr_config import CFG


# ==========================================================
# TEMPERATURE SCALING
# ==========================================================

class TemperatureScaling:
    """
    Temperature Scaling for model calibration
    
    Learns a single temperature parameter T to scale logits:
    calibrated_probs = softmax(logits / T)
    """
    
    def __init__(self):
        self.temperature = CFG.TEMPERATURE_INIT
    
    def fit(self, logits, labels, max_iter=None, verbose=True):
        """
        Find optimal temperature using validation set
        
        Args:
            logits: Model logits [N, num_classes]
            labels: True labels [N]
            max_iter: Maximum optimization iterations
            verbose: Print progress
        
        Returns:
            self
        """
        if max_iter is None:
            max_iter = CFG.TEMPERATURE_MAX_ITER
        
        if verbose:
            print("\n" + "="*70)
            print("TEMPERATURE SCALING CALIBRATION")
            print("="*70)
        
        def nll_loss(temp):
            """Negative log-likelihood loss"""
            scaled_logits = logits / temp[0]
            probs = tf.nn.softmax(scaled_logits).numpy()
            
            # Avoid log(0)
            probs = np.clip(probs, 1e-12, 1.0)
            
            # NLL
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            return nll
        
        # Optimize
        result = minimize(
            nll_loss,
            x0=[self.temperature],
            bounds=[(0.1, 10.0)],
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )
        
        self.temperature = result.x[0]
        
        if verbose:
            print(f"\nOptimal temperature: {self.temperature:.4f}")
            print(f"Optimization success: {result.success}")
            print(f"Final NLL: {result.fun:.4f}")
        
        return self
    
    def predict(self, logits):
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits [N, num_classes]
        
        Returns:
            Calibrated probabilities [N, num_classes]
        """
        scaled_logits = logits / self.temperature
        calibrated_probs = tf.nn.softmax(scaled_logits).numpy()
        return calibrated_probs
    
    def save(self, filepath):
        """Save temperature to file"""
        np.save(filepath, self.temperature)
        print(f"✅ Saved temperature to {filepath}")
    
    def load(self, filepath):
        """Load temperature from file"""
        self.temperature = np.load(filepath)
        print(f"✅ Loaded temperature: {self.temperature:.4f}")
        return self


# ==========================================================
# CALIBRATION METRICS
# ==========================================================

def calculate_ece_mce(y_true, y_pred_probs, n_bins=None):
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    
    Args:
        y_true: True labels [N]
        y_pred_probs: Predicted probabilities [N, num_classes]
        n_bins: Number of bins for calibration
    
    Returns:
        (ece, mce) tuple
    """
    if n_bins is None:
        n_bins = CFG.ECE_BINS
    
    # Get confidences and predictions
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_confidence = np.mean(confidences[mask])
            bin_accuracy = np.mean(accuracies[mask])
            bin_size = np.sum(mask) / len(y_true)
            
            calibration_error = np.abs(bin_confidence - bin_accuracy)
            
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)
    
    return ece, mce


def calculate_class_wise_ece(y_true, y_pred_probs, n_bins=None):
    """
    Calculate ECE for each class
    
    Args:
        y_true: True labels [N]
        y_pred_probs: Predicted probabilities [N, num_classes]
        n_bins: Number of bins
    
    Returns:
        Dictionary of class-wise ECE
    """
    if n_bins is None:
        n_bins = CFG.ECE_BINS
    
    class_ece = {}
    
    for class_idx in range(CFG.NUM_CLASSES):
        class_mask = (y_true == class_idx)
        if np.sum(class_mask) > 0:
            class_probs = y_pred_probs[class_mask]
            class_labels = y_true[class_mask]
            
            ece, _ = calculate_ece_mce(class_labels, class_probs, n_bins)
            class_ece[CFG.CLASS_NAMES[class_idx]] = ece
    
    return class_ece


# ==========================================================
# RELIABILITY DIAGRAMS
# ==========================================================

def plot_reliability_diagram(y_true, y_pred_probs, n_bins=None, 
                             title='Reliability Diagram', save_path=None):
    """
    Plot reliability diagram for calibration visualization
    
    Args:
        y_true: True labels [N]
        y_pred_probs: Predicted probabilities [N, num_classes]
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
    """
    if n_bins is None:
        n_bins = CFG.ECE_BINS
    
    # Get confidences and predictions
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_confidences.append(np.mean(confidences[mask]))
            bin_accuracies.append(np.mean(accuracies[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_confidences.append((bins[i] + bins[i+1]) / 2)
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    # Calculate ECE and MCE
    ece, mce = calculate_ece_mce(y_true, y_pred_probs, n_bins)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reliability diagram
    ax1.bar(range(n_bins), bin_accuracies, width=0.8, alpha=0.7,
           label='Accuracy', color='steelblue', edgecolor='black')
    ax1.plot(range(n_bins), bin_confidences, 'go-',
            label='Confidence', markersize=8, linewidth=2)
    ax1.plot([0, n_bins-1], [bin_accuracies[0], bin_accuracies[-1]], 'r--',
            label='Perfect Calibration', linewidth=2)
    
    ax1.set_xlabel('Confidence Bin', fontsize=12)
    ax1.set_ylabel('Accuracy / Confidence', fontsize=12)
    ax1.set_title(f'{title}\nECE: {ece:.4f}, MCE: {mce:.4f}',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(range(n_bins))
    ax1.set_xticklabels([f'{bins[i]:.1f}' for i in range(n_bins)], rotation=45)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sample distribution
    ax2.bar(range(n_bins), bin_counts, width=0.8, alpha=0.7,
           color='coral', edgecolor='black')
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f'{bins[i]:.1f}' for i in range(n_bins)], rotation=45)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved reliability diagram to {save_path}")
    
    plt.show()
    
    return ece, mce


def plot_class_wise_reliability(y_true, y_pred_probs, n_bins=None, save_path=None):
    """
    Plot reliability diagrams for each class
    
    Args:
        y_true: True labels [N]
        y_pred_probs: Predicted probabilities [N, num_classes]
        n_bins: Number of bins
        save_path: Path to save figure
    """
    if n_bins is None:
        n_bins = CFG.ECE_BINS
    
    n_classes = CFG.NUM_CLASSES
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        # Filter for this class
        class_mask = (y_true == class_idx)
        if np.sum(class_mask) == 0:
            ax.text(0.5, 0.5, f'No samples\nfor {CFG.CLASS_NAMES[class_idx]}',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        class_probs = y_pred_probs[class_mask]
        class_labels = y_true[class_mask]
        
        # Get confidences
        confidences = np.max(class_probs, axis=1)
        predictions = np.argmax(class_probs, axis=1)
        accuracies = (predictions == class_labels).astype(float)
        
        # Bin
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_confidences = []
        bin_accuracies = []
        
        for i in range(n_bins):
            mask = (bin_indices == i)
            if np.sum(mask) > 0:
                bin_confidences.append(np.mean(confidences[mask]))
                bin_accuracies.append(np.mean(accuracies[mask]))
            else:
                bin_confidences.append((bins[i] + bins[i+1]) / 2)
                bin_accuracies.append(0)
        
        # Calculate ECE
        ece, _ = calculate_ece_mce(class_labels, class_probs, n_bins)
        
        # Plot
        ax.bar(range(n_bins), bin_accuracies, width=0.8, alpha=0.7,
              color='steelblue', edgecolor='black')
        ax.plot(range(n_bins), bin_confidences, 'go-', markersize=6, linewidth=2)
        ax.plot([0, n_bins-1], [0, 1], 'r--', linewidth=2)
        
        ax.set_title(f'{CFG.CLASS_NAMES[class_idx]}\nECE: {ece:.4f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Confidence Bin', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved class-wise reliability to {save_path}")
    
    plt.show()


# ==========================================================
# CALIBRATION REPORT
# ==========================================================

def generate_calibration_report(y_true, y_pred_probs_before, y_pred_probs_after,
                                save_dir=None):
    """
    Generate comprehensive calibration report
    
    Args:
        y_true: True labels
        y_pred_probs_before: Probabilities before calibration
        y_pred_probs_after: Probabilities after calibration
        save_dir: Directory to save report
    
    Returns:
        Report dictionary
    """
    if save_dir is None:
        save_dir = CFG.RESULTS_DIR
    
    print("\n" + "="*70)
    print("CALIBRATION REPORT")
    print("="*70)
    
    # Overall metrics
    ece_before, mce_before = calculate_ece_mce(y_true, y_pred_probs_before)
    ece_after, mce_after = calculate_ece_mce(y_true, y_pred_probs_after)
    
    print(f"\nOverall Calibration:")
    print(f"  Before - ECE: {ece_before:.4f}, MCE: {mce_before:.4f}")
    print(f"  After  - ECE: {ece_after:.4f}, MCE: {mce_after:.4f}")
    print(f"  Improvement - ECE: {(ece_before - ece_after)/ece_before*100:.2f}%")
    
    # Class-wise metrics
    class_ece_before = calculate_class_wise_ece(y_true, y_pred_probs_before)
    class_ece_after = calculate_class_wise_ece(y_true, y_pred_probs_after)
    
    print(f"\nClass-wise ECE:")
    for class_name in CFG.CLASS_NAMES:
        ece_b = class_ece_before.get(class_name, 0)
        ece_a = class_ece_after.get(class_name, 0)
        improvement = (ece_b - ece_a) / ece_b * 100 if ece_b > 0 else 0
        print(f"  {class_name}: {ece_b:.4f} → {ece_a:.4f} ({improvement:+.2f}%)")
    
    # Plot reliability diagrams
    plot_reliability_diagram(
        y_true, y_pred_probs_before,
        title='Before Calibration',
        save_path=f"{save_dir}/reliability_before.png"
    )
    
    plot_reliability_diagram(
        y_true, y_pred_probs_after,
        title='After Calibration',
        save_path=f"{save_dir}/reliability_after.png"
    )
    
    # Class-wise reliability
    plot_class_wise_reliability(
        y_true, y_pred_probs_after,
        save_path=f"{save_dir}/reliability_classwise.png"
    )
    
    report = {
        'ece_before': ece_before,
        'mce_before': mce_before,
        'ece_after': ece_after,
        'mce_after': mce_after,
        'class_ece_before': class_ece_before,
        'class_ece_after': class_ece_after
    }
    
    return report


if __name__ == "__main__":
    print("Calibration Module")
    print("="*70)
    print("\nThis module provides:")
    print("  1. TemperatureScaling - Learn optimal temperature")
    print("  2. calculate_ece_mce() - Calibration metrics")
    print("  3. plot_reliability_diagram() - Visualization")
    print("  4. generate_calibration_report() - Comprehensive report")
