# ==========================================================
# AEELR MODEL ARCHITECTURE
# EfficientNetB5 + CBAM Attention + Hierarchical Heads
# ==========================================================

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB5, ResNet50

from aeelr_config import CFG
from cbam_attention import cbam_block


# ==========================================================
# BASELINE MODEL (NO CBAM)
# ==========================================================

def build_baseline_efficientnet(input_shape=None, num_classes=None, freeze_layers=None):
    """
    Baseline EfficientNetB5 without CBAM (for ablation)
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        freeze_layers: Number of initial layers to freeze
    
    Returns:
        Keras Model
    """
    if input_shape is None:
        input_shape = (*CFG.IMG_SIZE, 3)
    if num_classes is None:
        num_classes = CFG.NUM_CLASSES
    if freeze_layers is None:
        freeze_layers = CFG.FREEZE_LAYERS
    
    # Input
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Backbone
    base = EfficientNetB5(
        include_top=False,
        weights=CFG.PRETRAINED_WEIGHTS,
        input_tensor=inputs
    )
    
    # Freeze early layers
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_layers)
    
    # Get features
    x = base.output
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(CFG.DROPOUT_RATE_1, name='dropout1')(x)
    x = layers.Dense(CFG.DENSE_UNITS, activation='relu', name='dense1')(x)
    x = layers.Dropout(CFG.DROPOUT_RATE_2, name='dropout2')(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs, output, name='EfficientNetB5_Baseline')
    
    return model


# ==========================================================
# AEELR MODEL (WITH CBAM)
# ==========================================================

def build_aeelr(input_shape=None, num_classes=None, freeze_layers=None, 
                use_hierarchical=None):
    """
    AEELR: Attention-Enhanced EfficientNet with Label Refinement
    
    Architecture:
    - EfficientNetB5 backbone (ImageNet pretrained)
    - CBAM attention after final conv block
    - Optional hierarchical multi-task heads
    
    Args:
        input_shape: Input image shape
        num_classes: Number of KL grades (5)
        freeze_layers: Number of initial layers to freeze
        use_hierarchical: Whether to use hierarchical heads
    
    Returns:
        Keras Model
    """
    if input_shape is None:
        input_shape = (*CFG.IMG_SIZE, 3)
    if num_classes is None:
        num_classes = CFG.NUM_CLASSES
    if freeze_layers is None:
        freeze_layers = CFG.FREEZE_LAYERS
    if use_hierarchical is None:
        use_hierarchical = CFG.USE_HIERARCHICAL
    
    # Input
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Backbone
    base = EfficientNetB5(
        include_top=False,
        weights=CFG.PRETRAINED_WEIGHTS,
        input_tensor=inputs
    )
    
    # Freeze early layers
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_layers)
    
    # Get features
    x = base.output
    
    # Apply CBAM attention
    x = cbam_block(
        x,
        reduction=CFG.CBAM_REDUCTION,
        kernel_size=CFG.CBAM_KERNEL_SIZE,
        name='cbam_final'
    )
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers
    x = layers.Dropout(CFG.DROPOUT_RATE_1, name='dropout1')(x)
    x = layers.Dense(CFG.DENSE_UNITS, activation='relu', name='dense1')(x)
    x = layers.Dropout(CFG.DROPOUT_RATE_2, name='dropout2')(x)
    
    if use_hierarchical:
        # Hierarchical multi-task heads
        
        # Binary head: Healthy (0) vs OA (1-4)
        binary_out = layers.Dense(
            2, 
            activation='softmax', 
            name='binary_output'
        )(x)
        
        # Ternary head: Mild (0-1) vs Moderate (2-3) vs Severe (4)
        ternary_out = layers.Dense(
            3,
            activation='softmax',
            name='ternary_output'
        )(x)
        
        # KL grade head: 0-4
        kl_out = layers.Dense(
            num_classes,
            activation='softmax',
            name='kl_output'
        )(x)
        
        model = Model(
            inputs,
            [binary_out, ternary_out, kl_out],
            name='AEELR_Hierarchical'
        )
    else:
        # Single output
        output = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        model = Model(inputs, output, name='AEELR')
    
    return model


# ==========================================================
# MULTI-SCALE CBAM (ADVANCED)
# ==========================================================

def build_aeelr_multiscale(input_shape=None, num_classes=None, freeze_layers=None):
    """
    AEELR with multi-scale CBAM insertion
    
    Inserts CBAM at multiple depths for richer attention
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        freeze_layers: Number of layers to freeze
    
    Returns:
        Keras Model
    """
    if input_shape is None:
        input_shape = (*CFG.IMG_SIZE, 3)
    if num_classes is None:
        num_classes = CFG.NUM_CLASSES
    if freeze_layers is None:
        freeze_layers = CFG.FREEZE_LAYERS
    
    # Input
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Backbone
    base = EfficientNetB5(
        include_top=False,
        weights=CFG.PRETRAINED_WEIGHTS,
        input_tensor=inputs
    )
    
    # Freeze early layers
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_layers)
    
    # Get intermediate features at different scales
    # EfficientNetB5 block names: block{1-7}{a-z}_...
    
    # Mid-level features (block 4)
    mid_features = None
    for layer in base.layers:
        if 'block4a_expand_activation' in layer.name:
            mid_features = layer.output
            break
    
    # High-level features (final)
    high_features = base.output
    
    # Apply CBAM at multiple scales
    if mid_features is not None:
        mid_cbam = cbam_block(
            mid_features,
            reduction=CFG.CBAM_REDUCTION,
            kernel_size=CFG.CBAM_KERNEL_SIZE,
            name='cbam_mid'
        )
    
    high_cbam = cbam_block(
        high_features,
        reduction=CFG.CBAM_REDUCTION,
        kernel_size=CFG.CBAM_KERNEL_SIZE,
        name='cbam_high'
    )
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='gap')(high_cbam)
    
    # Classification head
    x = layers.Dropout(CFG.DROPOUT_RATE_1, name='dropout1')(x)
    x = layers.Dense(CFG.DENSE_UNITS, activation='relu', name='dense1')(x)
    x = layers.Dropout(CFG.DROPOUT_RATE_2, name='dropout2')(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs, output, name='AEELR_MultiScale')
    
    return model


# ==========================================================
# OPTIONAL: CNN-VIT FUSION
# ==========================================================

def build_cnn_vit_fusion(input_shape=None, num_classes=None):
    """
    Optional CNN-ViT fusion architecture (time-boxed ablation)
    
    Uses ResNet50 (CNN) + EfficientNetB5 (ViT proxy) with attention fusion
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
    
    Returns:
        Keras Model
    """
    if input_shape is None:
        input_shape = (*CFG.IMG_SIZE, 3)
    if num_classes is None:
        num_classes = CFG.NUM_CLASSES
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # CNN branch (ResNet50)
    cnn_base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    cnn_features = layers.GlobalAveragePooling2D(name='cnn_gap')(cnn_base.output)
    cnn_features = layers.Dense(512, activation='relu', name='cnn_dense')(cnn_features)
    
    # ViT branch (EfficientNetB5 as proxy)
    vit_base = EfficientNetB5(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    vit_features = layers.GlobalAveragePooling2D(name='vit_gap')(vit_base.output)
    vit_features = layers.Dense(512, activation='relu', name='vit_dense')(vit_features)
    
    # Attention fusion
    concat = layers.Concatenate(name='fusion_concat')([cnn_features, vit_features])
    
    # Learn fusion weights
    fusion_weights = layers.Dense(2, activation='softmax', name='fusion_attention')(concat)
    
    # Weighted combination
    cnn_weight = layers.Lambda(lambda x: x[:, 0:1], name='cnn_weight')(fusion_weights)
    vit_weight = layers.Lambda(lambda x: x[:, 1:2], name='vit_weight')(fusion_weights)
    
    cnn_weighted = layers.Multiply(name='cnn_multiply')([cnn_features, cnn_weight])
    vit_weighted = layers.Multiply(name='vit_multiply')([vit_features, vit_weight])
    
    fused = layers.Add(name='fusion_add')([cnn_weighted, vit_weighted])
    
    # Classification head
    x = layers.Dropout(0.5, name='dropout')(fused)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs, output, name='CNN_ViT_Fusion')
    
    return model


# ==========================================================
# MODEL UTILITIES
# ==========================================================

def unfreeze_model(model, unfreeze_from=None):
    """
    Unfreeze layers for fine-tuning
    
    Args:
        model: Keras model
        unfreeze_from: Layer index to start unfreezing from
    
    Returns:
        Model with unfrozen layers
    """
    if unfreeze_from is None:
        unfreeze_from = CFG.FREEZE_LAYERS
    
    for i, layer in enumerate(model.layers):
        if i >= unfreeze_from:
            layer.trainable = True
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"✅ Unfroze {trainable_count}/{len(model.layers)} layers")
    
    return model


def print_model_summary(model, show_full=False):
    """
    Print model summary with key statistics
    
    Args:
        model: Keras model
        show_full: Whether to show full layer-by-layer summary
    """
    print("\n" + "="*70)
    print(f"MODEL: {model.name}")
    print("="*70)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    if show_full:
        print("\n" + "-"*70)
        model.summary()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test model building
    print("Testing AEELR model architectures...\n")
    
    # 1. Baseline
    print("1. Building Baseline EfficientNetB5...")
    baseline = build_baseline_efficientnet()
    print_model_summary(baseline)
    
    # 2. AEELR (single-task)
    print("2. Building AEELR (single-task)...")
    aeelr_single = build_aeelr(use_hierarchical=False)
    print_model_summary(aeelr_single)
    
    # 3. AEELR (hierarchical)
    print("3. Building AEELR (hierarchical)...")
    aeelr_hierarchical = build_aeelr(use_hierarchical=True)
    print_model_summary(aeelr_hierarchical)
    
    print("✅ All models built successfully!")
