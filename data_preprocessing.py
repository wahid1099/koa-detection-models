# ==========================================================
# DATA PREPROCESSING AND AUGMENTATION
# Advanced preprocessing pipeline for knee OA X-rays
# ==========================================================

import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter, laplace
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from aeelr_config import CFG


# ==========================================================
# PREPROCESSING FUNCTIONS
# ==========================================================

def denoise_image(img, sigma=None):
    """
    Apply Gaussian denoising to reduce acquisition noise
    
    Args:
        img: Input grayscale image
        sigma: Gaussian sigma (random if None)
    
    Returns:
        Denoised image
    """
    if sigma is None:
        sigma = np.random.uniform(CFG.GAUSSIAN_SIGMA_MIN, CFG.GAUSSIAN_SIGMA_MAX)
    
    denoised = gaussian_filter(img, sigma=sigma)
    return denoised


def enhance_edges(img, weight=None):
    """
    Apply Laplacian filter to emphasize osteophytes and cortical boundaries
    
    Args:
        img: Input image
        weight: Laplacian weight (default from config)
    
    Returns:
        Edge-enhanced image
    """
    if weight is None:
        weight = CFG.LAPLACIAN_WEIGHT
    
    laplacian = laplace(img)
    enhanced = img - weight * laplacian
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def apply_clahe(img, clip_limit=None, tile_size=None):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        img: Input grayscale image
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size
    
    Returns:
        Contrast-enhanced image
    """
    if clip_limit is None:
        clip_limit = CFG.CLAHE_CLIP_LIMIT
    if tile_size is None:
        tile_size = CFG.CLAHE_TILE_SIZE
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    equalized = clahe.apply(img)
    
    return equalized


def preprocess_pipeline(img_path, target_size=None, return_rgb=True):
    """
    Complete preprocessing pipeline:
    1. Load image
    2. Gaussian denoising
    3. Laplacian edge enhancement
    4. CLAHE histogram equalization
    5. Resize with aspect ratio preservation
    6. Convert to RGB (if needed)
    7. Normalize
    
    Args:
        img_path: Path to image file
        target_size: Target size (H, W)
        return_rgb: Whether to return RGB (3-channel) image
    
    Returns:
        Preprocessed image array
    """
    if target_size is None:
        target_size = CFG.IMG_SIZE
    
    # Load grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # 1. Denoise
    denoised = denoise_image(img, sigma=CFG.GAUSSIAN_SIGMA)
    
    # 2. Edge enhancement
    enhanced = enhance_edges(denoised, weight=CFG.LAPLACIAN_WEIGHT)
    
    # 3. CLAHE
    equalized = apply_clahe(enhanced)
    
    # 4. Resize with aspect ratio preservation
    h, w = equalized.shape
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(equalized, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Pad to target size
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, target_h - new_h - pad_h,
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=0
    )
    
    # 5. Convert to RGB if needed
    if return_rgb:
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    else:
        rgb = padded
    
    # 6. Normalize using ImageNet stats
    normalized = rgb.astype(np.float32) / 255.0
    
    if return_rgb:
        # Apply ImageNet normalization
        mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(CFG.IMAGENET_STD, dtype=np.float32)
        normalized = (normalized - mean) / std
    
    return normalized


def create_preprocessed_dataset(df, output_dir, target_size=None, verbose=True):
    """
    Create preprocessed dataset from dataframe
    
    Args:
        df: DataFrame with 'filepaths' and 'labels' columns
        output_dir: Output directory for preprocessed images
        target_size: Target image size
        verbose: Print progress
    
    Returns:
        DataFrame with new filepaths
    """
    if target_size is None:
        target_size = CFG.IMG_SIZE
    
    # Clean output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    new_filepaths = []
    labels = []
    
    if verbose:
        print(f"\nPreprocessing {len(df)} images to {output_dir}...")
    
    for idx, row in df.iterrows():
        img_path = row['filepaths']
        label = row['labels']
        
        # Create class directory
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        
        # Preprocess
        processed_img = preprocess_pipeline(img_path, target_size, return_rgb=True)
        if processed_img is None:
            if verbose:
                print(f"  ⚠ Failed to load: {img_path}")
            continue
        
        # Convert back to uint8 for saving
        img_uint8 = ((processed_img * np.array(CFG.IMAGENET_STD) + np.array(CFG.IMAGENET_MEAN)) * 255.0)
        img_uint8 = np.clip(img_uint8, 0, 255).astype(np.uint8)
        
        # Save
        filename = os.path.basename(img_path)
        new_path = os.path.join(class_dir, filename)
        cv2.imwrite(new_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        
        new_filepaths.append(new_path)
        labels.append(label)
        
        if verbose and (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")
    
    if verbose:
        print(f"✅ Preprocessed {len(new_filepaths)}/{len(df)} images")
    
    return pd.DataFrame({'filepaths': new_filepaths, 'labels': labels})


# ==========================================================
# AUGMENTATION
# ==========================================================

def create_augmentation_pipeline(mode='train'):
    """
    Create class-aware augmentation pipeline
    
    Args:
        mode: 'train' or 'val'/'test'
    
    Returns:
        ImageDataGenerator
    """
    if mode == 'train':
        datagen = ImageDataGenerator(
            rotation_range=CFG.ROTATION_RANGE,
            width_shift_range=CFG.WIDTH_SHIFT_RANGE,
            height_shift_range=CFG.HEIGHT_SHIFT_RANGE,
            zoom_range=CFG.ZOOM_RANGE,
            horizontal_flip=CFG.HORIZONTAL_FLIP,
            brightness_range=CFG.BRIGHTNESS_RANGE,
            fill_mode='reflect',
            preprocessing_function=None  # Already preprocessed
        )
    else:
        # No augmentation for validation/test
        datagen = ImageDataGenerator(
            preprocessing_function=None
        )
    
    return datagen


# ==========================================================
# CROSS-VALIDATION SPLITS
# ==========================================================

def stratified_kfold_split(df, n_folds=None, random_seed=None):
    """
    Create stratified K-fold splits
    
    Args:
        df: DataFrame with 'filepaths' and 'labels'
        n_folds: Number of folds
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (train_df, val_df) tuples
    """
    if n_folds is None:
        n_folds = CFG.N_FOLDS
    if random_seed is None:
        random_seed = CFG.CV_RANDOM_SEED
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(CFG.CLASS_NAMES)}
    y = df['labels'].map(label_to_int).values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_df, val_df))
        
        print(f"Fold {fold_idx + 1}/{n_folds}: Train={len(train_df)}, Val={len(val_df)}")
    
    return folds


# ==========================================================
# CLASS BALANCING
# ==========================================================

def compute_class_weights(df, label_col='labels'):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        df: DataFrame with labels
        label_col: Column name for labels
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get label counts
    labels = df[label_col].values
    label_to_int = {label: i for i, label in enumerate(CFG.CLASS_NAMES)}
    y = np.array([label_to_int[label] for label in labels])
    
    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\nClass Weights:")
    for i, weight in class_weight_dict.items():
        print(f"  {CFG.CLASS_NAMES[i]}: {weight:.4f}")
    
    return class_weight_dict


# ==========================================================
# DATA LOADING
# ==========================================================

def build_df_from_dirs(data_dir, class_names=None):
    """
    Build DataFrame from directory structure
    
    Args:
        data_dir: Root directory with class subdirectories
        class_names: List of class names
    
    Returns:
        DataFrame with 'filepaths' and 'labels'
    """
    if class_names is None:
        class_names = CFG.CLASS_NAMES
    
    filepaths = []
    labels = []
    
    for klass in sorted(os.listdir(data_dir)):
        klass_path = os.path.join(data_dir, klass)
        if not os.path.isdir(klass_path):
            continue
        
        # Get class index
        klass_idx = int(klass)
        label = class_names[klass_idx]
        
        # Get all images
        for fname in os.listdir(klass_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepaths.append(os.path.join(klass_path, fname))
                labels.append(label)
    
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    
    print(f"\nLoaded {len(df)} images from {data_dir}")
    print("Class distribution:")
    for label in class_names:
        count = (df['labels'] == label).sum()
        print(f"  {label}: {count}")
    
    return df


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def visualize_preprocessing(img_path, save_path=None):
    """
    Visualize preprocessing steps
    
    Args:
        img_path: Path to image
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load original
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Step-by-step
    denoised = denoise_image(original)
    enhanced = enhance_edges(denoised)
    equalized = apply_clahe(enhanced)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('Gaussian Denoised', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(enhanced, cmap='gray')
    axes[1, 0].set_title('Edge Enhanced (Laplacian)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(equalized, cmap='gray')
    axes[1, 1].set_title('CLAHE Equalized', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing pipeline...")
    
    # Example usage
    test_img_path = "/kaggle/input/koa-dataset/dataset/train/0/9000016.png"
    
    if os.path.exists(test_img_path):
        # Visualize
        visualize_preprocessing(test_img_path, save_path="preprocessing_demo.png")
        
        # Test pipeline
        processed = preprocess_pipeline(test_img_path)
        print(f"\nProcessed image shape: {processed.shape}")
        print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    else:
        print(f"Test image not found: {test_img_path}")
