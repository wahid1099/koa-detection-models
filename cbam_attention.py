# ==========================================================
# CBAM ATTENTION MODULE
# Convolutional Block Attention Module for feature refinement
# ==========================================================

import tensorflow as tf
from tensorflow.keras import layers, Model

from aeelr_config import CFG


def channel_attention(input_tensor, reduction=16, name='channel_attention'):
    """
    Channel Attention Module
    
    Applies attention across channels using both average and max pooling,
    followed by a shared MLP to generate channel-wise attention weights.
    
    Args:
        input_tensor: Input feature map [B, H, W, C]
        reduction: Reduction ratio for MLP
        name: Layer name prefix
    
    Returns:
        Channel-refined feature map [B, H, W, C]
    """
    channels = input_tensor.shape[-1]
    
    # Global pooling
    avg_pool = layers.GlobalAveragePooling2D(
        keepdims=True, 
        name=f'{name}_avg_pool'
    )(input_tensor)
    
    max_pool = layers.GlobalMaxPooling2D(
        keepdims=True,
        name=f'{name}_max_pool'
    )(input_tensor)
    
    # Shared MLP
    mlp_units = max(channels // reduction, 1)
    
    # Average branch
    avg_mlp = layers.Dense(mlp_units, activation='relu', name=f'{name}_mlp1_avg')(avg_pool)
    avg_mlp = layers.Dense(channels, name=f'{name}_mlp2_avg')(avg_mlp)
    
    # Max branch
    max_mlp = layers.Dense(mlp_units, activation='relu', name=f'{name}_mlp1_max')(max_pool)
    max_mlp = layers.Dense(channels, name=f'{name}_mlp2_max')(max_mlp)
    
    # Combine and activate
    channel_weights = layers.Add(name=f'{name}_add')([avg_mlp, max_mlp])
    channel_weights = layers.Activation('sigmoid', name=f'{name}_sigmoid')(channel_weights)
    
    # Apply attention
    output = layers.Multiply(name=f'{name}_multiply')([input_tensor, channel_weights])
    
    return output


def spatial_attention(input_tensor, kernel_size=7, name='spatial_attention'):
    """
    Spatial Attention Module
    
    Applies attention across spatial dimensions using pooled features
    along the channel axis.
    
    Args:
        input_tensor: Input feature map [B, H, W, C]
        kernel_size: Convolution kernel size
        name: Layer name prefix
    
    Returns:
        Spatially-refined feature map [B, H, W, C]
    """
    # Channel-wise pooling
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
        name=f'{name}_avg_pool'
    )(input_tensor)
    
    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
        name=f'{name}_max_pool'
    )(input_tensor)
    
    # Concatenate
    concat = layers.Concatenate(axis=-1, name=f'{name}_concat')([avg_pool, max_pool])
    
    # Convolution to generate spatial attention map
    spatial_weights = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding='same',
        activation='sigmoid',
        name=f'{name}_conv'
    )(concat)
    
    # Apply attention
    output = layers.Multiply(name=f'{name}_multiply')([input_tensor, spatial_weights])
    
    return output


def cbam_block(input_tensor, reduction=16, kernel_size=7, name='cbam'):
    """
    Complete CBAM Block: Channel Attention → Spatial Attention
    
    Args:
        input_tensor: Input feature map [B, H, W, C]
        reduction: Channel attention reduction ratio
        kernel_size: Spatial attention kernel size
        name: Block name prefix
    
    Returns:
        Attention-refined feature map [B, H, W, C]
    """
    # Channel attention
    x = channel_attention(input_tensor, reduction=reduction, name=f'{name}_channel')
    
    # Spatial attention
    x = spatial_attention(x, kernel_size=kernel_size, name=f'{name}_spatial')
    
    return x


def insert_cbam_in_model(base_model, layer_names, reduction=16, kernel_size=7):
    """
    Insert CBAM blocks after specified layers in a base model
    
    Args:
        base_model: Base Keras model
        layer_names: List of layer names to insert CBAM after
        reduction: CBAM channel reduction
        kernel_size: CBAM spatial kernel size
    
    Returns:
        New model with CBAM blocks inserted
    """
    # Get input
    x = base_model.input
    
    # Track outputs
    outputs = {}
    
    # Iterate through layers
    for layer in base_model.layers:
        # Get layer output
        if layer.name in outputs:
            x = layer(outputs[layer.name])
        else:
            if isinstance(layer, layers.InputLayer):
                continue
            x = layer(x)
        
        # Insert CBAM if needed
        if layer.name in layer_names:
            x = cbam_block(
                x,
                reduction=reduction,
                kernel_size=kernel_size,
                name=f'cbam_after_{layer.name}'
            )
        
        outputs[layer.name] = x
    
    # Create new model
    model = Model(inputs=base_model.input, outputs=x, name=f'{base_model.name}_CBAM')
    
    return model


# ==========================================================
# CBAM VISUALIZATION
# ==========================================================

def visualize_cbam_attention(model, img_array, cbam_layer_name):
    """
    Visualize CBAM attention weights
    
    Args:
        model: Model with CBAM layers
        img_array: Input image [1, H, W, C]
        cbam_layer_name: Name of CBAM layer to visualize
    
    Returns:
        Channel and spatial attention maps
    """
    import numpy as np
    
    # Create model to output CBAM intermediate activations
    channel_layer_name = f'{cbam_layer_name}_channel_sigmoid'
    spatial_layer_name = f'{cbam_layer_name}_spatial_conv'
    
    try:
        channel_layer = model.get_layer(channel_layer_name)
        spatial_layer = model.get_layer(spatial_layer_name)
    except:
        print(f"⚠ CBAM layers not found: {cbam_layer_name}")
        return None, None
    
    # Create extraction model
    attention_model = Model(
        inputs=model.input,
        outputs=[channel_layer.output, spatial_layer.output]
    )
    
    # Get attention maps
    channel_attn, spatial_attn = attention_model.predict(img_array, verbose=0)
    
    return channel_attn, spatial_attn


def plot_cbam_attention(channel_attn, spatial_attn, save_path=None):
    """
    Plot CBAM attention maps
    
    Args:
        channel_attn: Channel attention weights [1, 1, 1, C]
        spatial_attn: Spatial attention map [1, H, W, 1]
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Channel attention
    channel_weights = channel_attn[0, 0, 0, :]
    axes[0].bar(range(len(channel_weights)), channel_weights, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Channel Index', fontsize=12)
    axes[0].set_ylabel('Attention Weight', fontsize=12)
    axes[0].set_title('Channel Attention Weights', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Spatial attention
    spatial_map = spatial_attn[0, :, :, 0]
    im = axes[1].imshow(spatial_map, cmap='jet', aspect='auto')
    axes[1].set_title('Spatial Attention Map', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved CBAM visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test CBAM module
    print("Testing CBAM module...")
    
    # Create dummy input
    dummy_input = tf.random.normal([1, 56, 56, 256])
    
    # Test channel attention
    print("\n1. Testing Channel Attention...")
    ch_attn = channel_attention(dummy_input, reduction=16, name='test_ch')
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {ch_attn.shape}")
    
    # Test spatial attention
    print("\n2. Testing Spatial Attention...")
    sp_attn = spatial_attention(dummy_input, kernel_size=7, name='test_sp')
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {sp_attn.shape}")
    
    # Test full CBAM
    print("\n3. Testing Full CBAM Block...")
    cbam_out = cbam_block(dummy_input, reduction=16, kernel_size=7, name='test_cbam')
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {cbam_out.shape}")
    
    print("\n✅ CBAM module tests passed!")
