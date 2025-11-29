# ==========================================================
# EXPLAINABILITY MODULE
# Grad-CAM and Eigen-CAM for visual interpretability
# ==========================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import cv2

from aeelr_config import CFG


# ==========================================================
# GRAD-CAM
# ==========================================================

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap
    
    Args:
        model: Keras model
        img_array: Input image [1, H, W, C]
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Class index to visualize (None = predicted class)
    
    Returns:
        Heatmap array [H, W]
    """
    # Create model that outputs last conv layer and predictions
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        # Try to find a suitable layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() or 'activation' in layer.name.lower():
                last_conv_layer = layer
                print(f"Using layer: {layer.name}")
                break
    
    grad_model = Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Handle hierarchical outputs
        if isinstance(predictions, list):
            predictions = predictions[-1]  # Use KL grade output
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    # Gradient of class w.r.t. feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()


def visualize_gradcam(img_path, model, last_conv_layer_name=None, save_path=None):
    """
    Visualize Grad-CAM heatmap overlaid on original image
    
    Args:
        img_path: Path to image
        model: Keras model
        last_conv_layer_name: Name of last conv layer
        save_path: Path to save visualization
    
    Returns:
        (heatmap, pred_class, confidence)
    """
    from data_preprocessing import preprocess_pipeline
    
    if last_conv_layer_name is None:
        last_conv_layer_name = CFG.GRADCAM_LAYER
    
    # Load and preprocess image
    img = preprocess_pipeline(img_path, target_size=CFG.IMG_SIZE, return_rgb=True)
    img_array = np.expand_dims(img, axis=0)
    
    # Get prediction
    preds = model.predict(img_array, verbose=0)
    if isinstance(preds, list):
        preds = preds[-1]  # Use KL grade output
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    
    # Generate heatmap
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_class)
    
    # Load original image for overlay
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, CFG.IMG_SIZE)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, CFG.IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay - Predicted: {CFG.CLASS_NAMES[pred_class]} ({confidence:.2%})',
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ Saved Grad-CAM to {save_path}")
    
    plt.show()
    
    return heatmap, pred_class, confidence


# ==========================================================
# EIGEN-CAM
# ==========================================================

def get_eigencam_heatmap(model, img_array, last_conv_layer_name):
    """
    Generate Eigen-CAM heatmap (class-agnostic)
    
    Uses principal component of activations
    
    Args:
        model: Keras model
        img_array: Input image [1, H, W, C]
        last_conv_layer_name: Name of last conv layer
    
    Returns:
        Heatmap array [H, W]
    """
    # Get activations
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() or 'activation' in layer.name.lower():
                last_conv_layer = layer
                break
    
    activation_model = Model(inputs=model.input, outputs=last_conv_layer.output)
    activations = activation_model.predict(img_array, verbose=0)
    
    # Reshape activations [1, H, W, C] -> [H*W, C]
    batch_size, h, w, num_channels = activations.shape
    reshaped_activations = activations.reshape(h * w, num_channels)
    
    # Compute principal component
    # Use SVD to get first principal component
    _, _, vh = np.linalg.svd(reshaped_activations.T, full_matrices=False)
    principal_component = vh[0, :]  # First principal component
    
    # Reshape back to spatial dimensions
    heatmap = principal_component.reshape(h, w)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


# ==========================================================
# GRADE-WISE VISUALIZATION
# ==========================================================

def create_gradewise_visualizations(model, df, samples_per_class=None, save_dir=None):
    """
    Create Grad-CAM visualizations for representative samples per grade
    
    Args:
        model: Trained model
        df: DataFrame with filepaths and labels
        samples_per_class: Number of samples per class
        save_dir: Directory to save visualizations
    
    Returns:
        Dictionary of visualizations
    """
    if samples_per_class is None:
        samples_per_class = CFG.GRADCAM_SAMPLES_PER_CLASS
    if save_dir is None:
        save_dir = CFG.FIGURES_DIR
    
    print("\n" + "="*70)
    print("GRADE-WISE GRAD-CAM VISUALIZATION")
    print("="*70)
    
    visualizations = {}
    
    for class_idx, class_name in enumerate(CFG.CLASS_NAMES):
        print(f"\n[{class_name}] Generating visualizations...")
        
        # Get samples for this class
        class_df = df[df['labels'] == class_name]
        
        if len(class_df) == 0:
            print(f"  ⚠ No samples found for {class_name}")
            continue
        
        # Sample randomly
        n_samples = min(samples_per_class, len(class_df))
        sampled_df = class_df.sample(n=n_samples, random_state=CFG.RANDOM_SEED)
        
        # Create visualization grid
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, row) in enumerate(sampled_df.iterrows()):
            img_path = row['filepaths']
            
            # Preprocess
            from data_preprocessing import preprocess_pipeline
            img = preprocess_pipeline(img_path, return_rgb=True)
            img_array = np.expand_dims(img, axis=0)
            
            # Predict
            preds = model.predict(img_array, verbose=0)
            if isinstance(preds, list):
                preds = preds[-1]
            pred_class = np.argmax(preds[0])
            confidence = preds[0][pred_class]
            
            # Grad-CAM
            heatmap = get_gradcam_heatmap(model, img_array, CFG.GRADCAM_LAYER, pred_class)
            
            # Load original
            original = cv2.imread(img_path)
            original = cv2.resize(original, CFG.IMG_SIZE)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Heatmap overlay
            heatmap_resized = cv2.resize(heatmap, CFG.IMG_SIZE)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
            
            # Plot
            axes[i, 0].imshow(original)
            axes[i, 0].set_title('Original', fontsize=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(heatmap_resized, cmap='jet')
            axes[i, 1].set_title('Heatmap', fontsize=10)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Pred: {CFG.CLASS_NAMES[pred_class]} ({confidence:.2%})', fontsize=10)
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Grad-CAM for {class_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{save_dir}/gradcam_{class_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {save_path}")
        plt.close()
        
        visualizations[class_name] = save_path
    
    print("\n✅ Grade-wise visualizations complete!")
    return visualizations


# ==========================================================
# SANITY CHECKS
# ==========================================================

def sanity_check_shuffle_weights(model, img_array, last_conv_layer_name=None):
    """
    Sanity check: Shuffle model weights and verify CAM degrades
    
    This confirms that the CAM is actually using learned features
    
    Args:
        model: Trained model
        img_array: Input image [1, H, W, C]
        last_conv_layer_name: Name of last conv layer
    
    Returns:
        (original_heatmap, shuffled_heatmap)
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = CFG.GRADCAM_LAYER
    
    print("\n" + "="*70)
    print("SANITY CHECK: Weight Shuffling")
    print("="*70)
    
    # Get original heatmap
    original_heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    
    # Clone model and shuffle weights
    import copy
    model_shuffled = tf.keras.models.clone_model(model)
    model_shuffled.set_weights(model.get_weights())
    
    # Shuffle weights of last conv layer
    try:
        last_conv_layer = model_shuffled.get_layer(last_conv_layer_name)
        weights = last_conv_layer.get_weights()
        shuffled_weights = [np.random.permutation(w.flatten()).reshape(w.shape) for w in weights]
        last_conv_layer.set_weights(shuffled_weights)
    except:
        print("⚠ Could not shuffle weights")
        return original_heatmap, None
    
    # Get shuffled heatmap
    shuffled_heatmap = get_gradcam_heatmap(model_shuffled, img_array, last_conv_layer_name)
    
    # Compare
    correlation = np.corrcoef(original_heatmap.flatten(), shuffled_heatmap.flatten())[0, 1]
    
    print(f"\nCorrelation between original and shuffled heatmaps: {correlation:.4f}")
    print(f"Expected: Low correlation indicates CAM uses learned features")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_heatmap, cmap='jet')
    axes[0].set_title('Original Heatmap', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(shuffled_heatmap, cmap='jet')
    axes[1].set_title(f'Shuffled Heatmap\nCorr: {correlation:.4f}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{CFG.FIGURES_DIR}/sanity_check_shuffle.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    if correlation < 0.3:
        print("✅ Sanity check PASSED: Low correlation confirms CAM faithfulness")
    else:
        print("⚠ Sanity check WARNING: High correlation may indicate issues")
    
    return original_heatmap, shuffled_heatmap


# ==========================================================
# ATTENTION COMPARISON
# ==========================================================

def compare_attention_methods(model, img_path, save_path=None):
    """
    Compare Grad-CAM vs Eigen-CAM
    
    Args:
        model: Keras model
        img_path: Path to image
        save_path: Path to save comparison
    """
    from data_preprocessing import preprocess_pipeline
    
    # Preprocess
    img = preprocess_pipeline(img_path, return_rgb=True)
    img_array = np.expand_dims(img, axis=0)
    
    # Get both heatmaps
    gradcam = get_gradcam_heatmap(model, img_array, CFG.GRADCAM_LAYER)
    eigencam = get_eigencam_heatmap(model, img_array, CFG.GRADCAM_LAYER)
    
    # Load original
    original = cv2.imread(img_path)
    original = cv2.resize(original, CFG.IMG_SIZE)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.resize(gradcam, CFG.IMG_SIZE), cmap='jet')
    axes[1].set_title('Grad-CAM (Class-specific)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.resize(eigencam, CFG.IMG_SIZE), cmap='jet')
    axes[2].set_title('Eigen-CAM (Class-agnostic)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Explainability Module")
    print("="*70)
    print("\nThis module provides:")
    print("  1. get_gradcam_heatmap() - Generate Grad-CAM")
    print("  2. get_eigencam_heatmap() - Generate Eigen-CAM")
    print("  3. visualize_gradcam() - Visualize single image")
    print("  4. create_gradewise_visualizations() - Grade-wise CAMs")
    print("  5. sanity_check_shuffle_weights() - Verify CAM faithfulness")
    print("  6. compare_attention_methods() - Compare Grad-CAM vs Eigen-CAM")
