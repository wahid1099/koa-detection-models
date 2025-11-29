# ==========================================================
# FIXED: Hybrid CNN-ViT for Knee Osteoarthritis
# Key Fixes:
# 1. Gradient clipping to prevent explosion
# 2. Learning rate warmup + cosine decay
# 3. Normalized hierarchical loss
# 4. Fixed weight loading path
# 5. Added gradient monitoring
# ==========================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import AdamW
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, auc, 
                             precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from scipy import signal
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Configuration
# ---------------------------
CONFIG = {
    'WORK_DIR': './',
    'EPOCHS': 100,
    'BATCH_SIZE': 8,
    'IMG_SIZE': 224,
    'NUM_CLASSES': 5,
    'LEARNING_RATE': 3e-5,  # FIXED: Reduced from 1e-4
    'WEIGHT_DECAY': 1e-5,
    'PATCH_SIZE': 16,
    'NUM_PATCHES': (224 // 16) ** 2,
    'PROJECTION_DIM': 512,
    'NUM_HEADS': 12,
    'TRANSFORMER_LAYERS': 4,
    'MLP_HEAD_UNITS': [2048, 1024],
    'WARMUP_EPOCHS': 5,  # NEW: Warmup period
    'GRADIENT_CLIP_NORM': 1.0,  # NEW: Gradient clipping
}

PATHS = {
    'train': '/kaggle/input/koa-dataset/dataset/train',
    'val': '/kaggle/input/koa-dataset/dataset/val',
    'test': '/kaggle/input/koa-dataset/dataset/test'
}

CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']

# [KEEPING ALL PREPROCESSING CODE FROM ORIGINAL - Lines 30-250]
# ... (KalmanFilter1D, apply_kalman_filter, laplacian_sharpen, etc.)

class KalmanFilter1D:
    """1D Kalman filter for noise reduction"""
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
    
    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        
        return self.posteri_estimate

def apply_kalman_filter(image):
    """Apply Kalman filter to each row/column for noise reduction"""
    filtered = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        kf = KalmanFilter1D()
        for j in range(image.shape[1]):
            filtered[i, j] = kf.update(image[i, j])
    
    for j in range(image.shape[1]):
        kf = KalmanFilter1D()
        for i in range(image.shape[0]):
            filtered[i, j] = kf.update(filtered[i, j])
    
    return filtered

def laplacian_sharpen(image, kernel_size=3, strength=1.5):
    """Laplacian sharpening for edge enhancement"""
    img_float = image.astype(np.float32)
    
    laplacian_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]], dtype=np.float32)
    
    sharpened = cv2.filter2D(img_float, -1, laplacian_kernel)
    result = img_float + (sharpened - img_float) * (strength - 1.0)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def high_pass_filter(image, kernel_size=15):
    """High-pass filter for texture emphasis"""
    img_float = image.astype(np.float32)
    low_pass = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)
    high_pass = img_float - low_pass
    enhanced = img_float + (high_pass * 0.5)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def detect_knee_roi(image):
    """Detect knee joint ROI using edge detection and morphology"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    edges = cv2.Canny(enhanced, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h)
    
    h, w = image.shape[:2]
    crop_size = min(h, w) // 2
    x = (w - crop_size) // 2
    y = (h - crop_size) // 2
    return (x, y, crop_size, crop_size)

def advanced_preprocess(img_path, target_size=224):
    """Complete preprocessing pipeline"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    sharpened = laplacian_sharpen(enhanced, strength=1.3)
    high_pass = high_pass_filter(sharpened, kernel_size=15)
    
    normalized = high_pass.astype(np.float32) / 255.0
    filtered = apply_kalman_filter(normalized)
    filtered = (filtered * 255).astype(np.uint8)
    
    x, y, w, h = detect_knee_roi(filtered)
    roi = filtered[y:y+h, x:x+w]
    
    resized = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    final = (rgb.astype(np.float32) / 127.5) - 1.0
    
    return final, (x, y, w, h)

def build_dataset(data_dir, target_size=224):
    """Build dataset with advanced preprocessing"""
    filepaths, labels, rois = [], [], []
    
    for klass in sorted(os.listdir(data_dir)):
        klass_path = os.path.join(data_dir, klass)
        if not os.path.isdir(klass_path):
            continue
        
        klass_idx = int(klass)
        
        for fname in os.listdir(klass_path):
            img_path = os.path.join(klass_path, fname)
            
            result = advanced_preprocess(img_path, target_size)
            if result is not None:
                preprocessed_img, roi_coords = result
                filepaths.append(preprocessed_img)
                labels.append(klass_idx)
                rois.append(roi_coords)
    
    return np.array(filepaths), np.array(labels), rois

# ---------------------------
# Vision Transformer Components
# ---------------------------

class PatchExtractor(layers.Layer):
    """Extract patches from images"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    """Encode patches with position embeddings"""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerBlock(layers.Layer):
    """Transformer encoder block"""
    def __init__(self, projection_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(projection_dim),
            layers.Dropout(dropout),
        ])
    
    def call(self, encoded_patches, training):
        x1 = self.norm1(encoded_patches)
        attention_output = self.attn(x1, x1, training=training)
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = self.norm2(x2)
        x3 = self.mlp(x3, training=training)
        encoded_patches = layers.Add()([x3, x2])
        
        return encoded_patches

# ---------------------------
# CNN Feature Extractor
# ---------------------------

def build_cnn_backbone(input_shape=(224, 224, 3)):
    """EfficientNetB0 for local feature extraction"""
    inputs = layers.Input(shape=input_shape)
    
    backbone = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    features = backbone.output
    
    return Model(inputs, features, name='CNN_Backbone')

# ---------------------------
# Attention Fusion Module
# ---------------------------

class AttentionFusion(layers.Layer):
    """Fuse CNN and ViT features with learned attention"""
    def __init__(self, fusion_dim=1024):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        self.cnn_proj = layers.Dense(fusion_dim, activation='relu')
        self.vit_proj = layers.Dense(fusion_dim, activation='relu')
        
        self.cnn_attention = layers.Dense(1, activation='sigmoid')
        self.vit_attention = layers.Dense(1, activation='sigmoid')
        
        self.fusion_layer = layers.Dense(fusion_dim, activation='relu')
        self.dropout = layers.Dropout(0.3)
    
    def call(self, cnn_features, vit_features, training=False):
        cnn_proj = self.cnn_proj(cnn_features)
        vit_proj = self.vit_proj(vit_features)
        
        cnn_weight = self.cnn_attention(cnn_proj)
        vit_weight = self.vit_attention(vit_proj)
        
        total_weight = cnn_weight + vit_weight
        cnn_weight = cnn_weight / (total_weight + 1e-8)
        vit_weight = vit_weight / (total_weight + 1e-8)
        
        fused = cnn_weight * cnn_proj + vit_weight * vit_proj
        fused = self.fusion_layer(fused)
        fused = self.dropout(fused, training=training)
        
        return fused, cnn_weight, vit_weight

# ---------------------------
# Hierarchical Classifier
# ---------------------------

class HierarchicalClassifier(layers.Layer):
    """Hierarchical classification"""
    def __init__(self):
        super().__init__()
        
        self.binary_head = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax', name='binary_output')
        ])
        
        self.ternary_head = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax', name='ternary_output')
        ])
        
        self.fine_head = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(5, activation='softmax', name='fine_output')
        ])
    
    def call(self, features, training=False):
        binary_pred = self.binary_head(features, training=training)
        ternary_pred = self.ternary_head(features, training=training)
        fine_pred = self.fine_head(features, training=training)
        
        return binary_pred, ternary_pred, fine_pred

# ---------------------------
# Complete Hybrid Model
# ---------------------------

def build_hybrid_model(config=CONFIG):
    """Build complete hybrid CNN-ViT model"""
    img_size = config['IMG_SIZE']
    patch_size = config['PATCH_SIZE']
    num_patches = config['NUM_PATCHES']
    projection_dim = config['PROJECTION_DIM']
    num_heads = config['NUM_HEADS']
    transformer_layers = config['TRANSFORMER_LAYERS']
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Stage 1: CNN Feature Extraction
    cnn_backbone = build_cnn_backbone((img_size, img_size, 3))
    cnn_features = cnn_backbone(inputs)
    cnn_gap = layers.GlobalAveragePooling2D()(cnn_features)
    
    # Stage 2: Vision Transformer
    patches = PatchExtractor(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        encoded_patches = TransformerBlock(
            projection_dim=projection_dim,
            num_heads=num_heads,
            mlp_dim=projection_dim * 4,
            dropout=0.1
        )(encoded_patches, training=True)
    
    vit_representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    vit_gap = layers.GlobalAveragePooling1D()(vit_representation)
    
    # Stage 3: Attention Fusion
    fusion_module = AttentionFusion(fusion_dim=1024)
    fused_features, cnn_weight, vit_weight = fusion_module(cnn_gap, vit_gap)
    
    # Stage 4: Hierarchical Classification
    hierarchical_classifier = HierarchicalClassifier()
    binary_output, ternary_output, fine_output = hierarchical_classifier(fused_features)
    
    model = Model(
        inputs=inputs,
        outputs={
            'binary': binary_output,
            'ternary': ternary_output,
            'fine': fine_output
        },
        name='Hybrid_CNN_ViT_KOA'
    )
    
    return model

# ---------------------------
# FIXED: Custom Loss Functions
# ---------------------------

def hierarchical_loss(y_true_fine, y_pred_binary, y_pred_ternary, y_pred_fine, 
                      alpha=0.2, beta=0.3, gamma=0.5):  # FIXED: Adjusted weights
    """
    FIXED: Normalized hierarchical loss with proper scaling
    """
    # Convert fine labels to hierarchical labels
    y_true_binary = tf.cast(y_true_fine > 0, tf.int32)
    y_true_binary = tf.one_hot(y_true_binary, 2)
    
    y_true_ternary = tf.where(y_true_fine <= 1, 0,
                              tf.where(y_true_fine <= 3, 1, 2))
    y_true_ternary = tf.one_hot(y_true_ternary, 3)
    
    y_true_fine_onehot = tf.one_hot(y_true_fine, 5)
    
    # FIXED: Use categorical crossentropy with proper reduction
    loss_binary = tf.keras.losses.categorical_crossentropy(
        y_true_binary, y_pred_binary, from_logits=False
    )
    loss_ternary = tf.keras.losses.categorical_crossentropy(
        y_true_ternary, y_pred_ternary, from_logits=False
    )
    loss_fine = tf.keras.losses.categorical_crossentropy(
        y_true_fine_onehot, y_pred_fine, from_logits=False
    )
    
    # FIXED: Clip losses to prevent explosion
    loss_binary = tf.clip_by_value(loss_binary, 0.0, 10.0)
    loss_ternary = tf.clip_by_value(loss_ternary, 0.0, 10.0)
    loss_fine = tf.clip_by_value(loss_fine, 0.0, 10.0)
    
    # Combined loss
    total_loss = alpha * loss_binary + beta * loss_ternary + gamma * loss_fine
    
    return total_loss

# ---------------------------
# FIXED: Learning Rate Schedule
# ---------------------------

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine Decay learning rate schedule"""
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)
        
        # Cosine decay phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decay_lr = self.initial_learning_rate * cosine_decay
        
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

# ---------------------------
# FIXED: Training Loop
# ---------------------------

def train_model(model, train_data, val_data, config=CONFIG):
    """FIXED: Train the hybrid model with gradient clipping and monitoring"""
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # FIXED: Learning rate schedule
    num_train_batches = len(X_train) // config['BATCH_SIZE']
    warmup_steps = config['WARMUP_EPOCHS'] * num_train_batches
    total_steps = config['EPOCHS'] * num_train_batches
    
    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=config['LEARNING_RATE'],
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # FIXED: Optimizer with gradient clipping
    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=config['WEIGHT_DECAY'],
        clipnorm=config['GRADIENT_CLIP_NORM']  # NEW: Gradient clipping
    )
    
    train_loss_tracker = keras.metrics.Mean(name='train_loss')
    val_loss_tracker = keras.metrics.Mean(name='val_loss')
    train_acc_tracker = keras.metrics.CategoricalAccuracy(name='train_acc')
    val_acc_tracker = keras.metrics.CategoricalAccuracy(name='val_acc')
    
    # NEW: Gradient norm tracker
    grad_norm_tracker = keras.metrics.Mean(name='grad_norm')
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = hierarchical_loss(
                y, 
                predictions['binary'],
                predictions['ternary'],
                predictions['fine']
            )
            # FIXED: Ensure scalar loss
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # NEW: Monitor gradient norm
        grad_norm = tf.linalg.global_norm(gradients)
        grad_norm_tracker.update_state(grad_norm)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss_tracker.update_state(loss)
        train_acc_tracker.update_state(tf.one_hot(y, 5), predictions['fine'])
        
        return loss, grad_norm
    
    @tf.function
    def val_step(x, y):
        predictions = model(x, training=False)
        loss = hierarchical_loss(
            y,
            predictions['binary'],
            predictions['ternary'],
            predictions['fine']
        )
        # FIXED: Ensure scalar loss
        loss = tf.reduce_mean(loss)
        
        val_loss_tracker.update_state(loss)
        val_acc_tracker.update_state(tf.one_hot(y, 5), predictions['fine'])
        
        return loss
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'grad_norm': []  # NEW: Track gradient norms
    }
    
    for epoch in range(config['EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{config['EPOCHS']}")
        
        # Reset metrics
        train_loss_tracker.reset_state()
        val_loss_tracker.reset_state()
        train_acc_tracker.reset_state()
        val_acc_tracker.reset_state()
        grad_norm_tracker.reset_state()
        
        # Training
        num_batches = len(X_train) // config['BATCH_SIZE']
        for batch in range(num_batches):
            start_idx = batch * config['BATCH_SIZE']
            end_idx = start_idx + config['BATCH_SIZE']
            
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            loss, grad_norm = train_step(x_batch, y_batch)
            
            if batch % 10 == 0:
                print(f"Batch {batch}/{num_batches} - "
                      f"Loss: {train_loss_tracker.result():.4f} - "
                      f"Acc: {train_acc_tracker.result():.4f} - "
                      f"GradNorm: {grad_norm_tracker.result():.4f}", end='\r')
        
        # Validation
        num_val_batches = len(X_val) // config['BATCH_SIZE']
        for batch in range(num_val_batches):
            start_idx = batch * config['BATCH_SIZE']
            end_idx = start_idx + config['BATCH_SIZE']
            
            x_batch = X_val[start_idx:end_idx]
            y_batch = y_val[start_idx:end_idx]
            
            val_step(x_batch, y_batch)
        
        # Log results
        train_loss = train_loss_tracker.result()
        val_loss = val_loss_tracker.result()
        train_acc = train_acc_tracker.result()
        val_acc = val_acc_tracker.result()
        grad_norm = grad_norm_tracker.result()
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['grad_norm'].append(float(grad_norm))
        
        print(f"\nTrain Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        print(f"Gradient Norm: {grad_norm:.4f}")
        
        # FIXED: Check for NaN/Inf
        if tf.math.is_nan(val_loss) or tf.math.is_inf(val_loss):
            print("\n⚠️ WARNING: NaN or Inf detected in validation loss!")
            print("Stopping training to prevent further issues.")
            break
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_weights('best_hybrid_model.weights.h5')  # FIXED: Correct extension
            print("✓ Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    return model, history

# ---------------------------
# Main Execution
# ---------------------------

def main():
    """Complete pipeline execution"""
    print("\n" + "="*70)
    print("FIXED: HYBRID CNN-ViT FOR KNEE OSTEOARTHRITIS CLASSIFICATION")
    print("="*70)
    
    # Load and Preprocess Data
    print("\n[STEP 1/3] Loading and preprocessing data...")
    print("⚠️  This may take several minutes due to advanced preprocessing...")
    
    print("\nProcessing training data...")
    X_train, y_train, train_rois = build_dataset(PATHS['train'], CONFIG['IMG_SIZE'])
    
    print("Processing validation data...")
    X_val, y_val, val_rois = build_dataset(PATHS['val'], CONFIG['IMG_SIZE'])
    
    print("Processing test data...")
    X_test, y_test, test_rois = build_dataset(PATHS['test'], CONFIG['IMG_SIZE'])
    
    print(f"\n✓ Dataset loaded:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Build Model
    print("\n[STEP 2/3] Building hybrid CNN-ViT model...")
    
    model = build_hybrid_model(CONFIG)
    
    print(f"\n✓ Model built successfully!")
    print(f"  Total parameters: {model.count_params():,}")
    
    model.summary()
    
    # Train Model
    print("\n[STEP 3/3] Training model with FIXES...")
    print("✓ Gradient clipping enabled")
    print("✓ Learning rate warmup + cosine decay")
    print("✓ Normalized hierarchical loss")
    print("✓ Gradient monitoring active")
    
    trained_model, history = train_model(
        model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        config=CONFIG
    )
    
    # Load best weights - FIXED: Correct path
    trained_model.load_weights('best_hybrid_model.weights.h5')
    print("\n✓ Best model weights loaded!")
    
    # Plot training history
    print("\n📊 Plotting training history...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Gradient Norm - NEW
    axes[2].plot(history['grad_norm'], label='Gradient Norm', linewidth=2, color='red')
    axes[2].axhline(y=CONFIG['GRADIENT_CLIP_NORM'], color='black', linestyle='--', label='Clip Threshold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Gradient Norm', fontsize=12)
    axes[2].set_title('Gradient Norm Monitoring', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_fixed.png', dpi=150)
    plt.show()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Generated files:")
    print(f"  • best_hybrid_model.weights.h5")
    print(f"  • training_history_fixed.png")
    
    return trained_model, history

if __name__ == "__main__":
    model, history = main()
