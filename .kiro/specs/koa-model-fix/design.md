# Design Document

## Overview

This design addresses critical training instabilities in the Hybrid CNN-ViT model for KOA classification. The primary issues are:

1. Exploding validation loss (4.8 → 21,000 → 87,000)
2. TensorFlow layout optimization errors
3. Poor validation accuracy (~3-12%)
4. Gradient instability

The solution involves gradient clipping, learning rate scheduling, proper normalization, loss rebalancing, and architectural fixes.

## Architecture

### High-Level Changes

The fixes are organized into five main areas:

1. **Gradient Stabilization**: Implement gradient clipping and monitoring
2. **Learning Rate Management**: Add warmup and decay scheduling
3. **Loss Function Rebalancing**: Normalize hierarchical loss components
4. **Data Pipeline Fixes**: Ensure numerical stability in preprocessing
5. **Model Architecture Updates**: Fix TensorFlow compatibility issues

### Component Interaction

```
Input Data → Preprocessing (Fixed) → Model (Updated) → Loss (Rebalanced) → Optimizer (Clipped) → Weights
                                                              ↓
                                                        LR Scheduler
                                                              ↓
                                                        Monitoring
```

## Components and Interfaces

### 1. Gradient Clipper

**Purpose**: Prevent gradient explosion during backpropagation

**Interface**:

```python
def clip_gradients(gradients, max_norm=1.0):
    """
    Clip gradients by global norm

    Args:
        gradients: List of gradient tensors
        max_norm: Maximum allowed gradient norm

    Returns:
        clipped_gradients: List of clipped gradient tensors
        global_norm: The computed global norm before clipping
    """
```

**Implementation Details**:

- Use `tf.clip_by_global_norm()` for consistent clipping
- Monitor and log gradient norms
- Default max_norm of 1.0 (adjustable)

### 2. Learning Rate Scheduler

**Purpose**: Implement warmup and decay for stable convergence

**Interface**:

```python
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with warmup and cosine decay

    Args:
        initial_learning_rate: Starting LR after warmup
        warmup_steps: Number of steps for linear warmup
        decay_steps: Total steps for cosine decay
        alpha: Minimum learning rate as fraction of initial
    """
```

**Implementation Details**:

- Linear warmup for first 5-10% of training
- Cosine decay after warmup
- Minimum LR of 1e-7

### 3. Hierarchical Loss Rebalancer

**Purpose**: Balance loss components to prevent dominance

**Interface**:

```python
def balanced_hierarchical_loss(y_true_fine, y_pred_binary, y_pred_ternary, y_pred_fine,
                               alpha=0.2, beta=0.3, gamma=0.5, label_smoothing=0.1):
    """
    Balanced hierarchical loss with label smoothing

    Args:
        y_true_fine: Ground truth fine-grained labels
        y_pred_binary: Binary predictions
        y_pred_ternary: Ternary predictions
        y_pred_fine: Fine-grained predictions
        alpha: Weight for binary loss (default 0.2)
        beta: Weight for ternary loss (default 0.3)
        gamma: Weight for fine loss (default 0.5)
        label_smoothing: Smoothing factor (default 0.1)

    Returns:
        total_loss: Weighted combination of losses
    """
```

**Implementation Details**:

- Rebalance weights: binary=0.2, ternary=0.3, fine=0.5
- Add label smoothing to prevent overconfidence
- Normalize each loss component by its scale

### 4. Preprocessing Stabilizer

**Purpose**: Ensure numerical stability in data preprocessing

**Changes**:

- Add bounds checking after Kalman filter
- Clip values after each preprocessing step
- Validate no NaN/Inf in final output
- Reduce Kalman filter variance parameters

**Interface**:

```python
def stable_preprocess(img_path, target_size=224):
    """
    Numerically stable preprocessing pipeline

    Returns:
        preprocessed_img: Normalized image in [-1, 1]
        roi_coords: ROI bounding box
        is_valid: Boolean indicating if preprocessing succeeded
    """
```

### 5. Model Architecture Fixes

**Purpose**: Resolve TensorFlow optimization errors

**Changes**:

1. Replace any stateless dropout with standard Dropout
2. Add explicit training flags to all dropout layers
3. Use `layers.Dropout(rate)` instead of custom implementations
4. Add batch normalization after major blocks

**Updated Transformer Block**:

```python
class StableTransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),  # Standard dropout
            layers.Dense(projection_dim),
            layers.Dropout(dropout),  # Standard dropout
        ])

    def call(self, encoded_patches, training=None):
        # Explicit training flag
        x1 = self.norm1(encoded_patches)
        attention_output = self.attn(x1, x1, training=training)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = self.norm2(x2)
        x3 = self.mlp(x3, training=training)
        encoded_patches = layers.Add()([x3, x2])

        return encoded_patches
```

### 6. Training Loop Monitor

**Purpose**: Detect and log anomalies during training

**Interface**:

```python
class TrainingMonitor:
    def __init__(self):
        self.gradient_norms = []
        self.loss_history = []

    def check_anomaly(self, loss, gradients):
        """
        Check for training anomalies

        Returns:
            is_anomaly: Boolean
            message: Diagnostic message
        """
```

**Monitored Metrics**:

- Gradient norms (warn if > 10.0)
- Loss values (warn if NaN or > 1000)
- Loss ratio (warn if val_loss / train_loss > 10)

## Data Models

### Training Configuration

```python
STABLE_CONFIG = {
    'EPOCHS': 100,
    'BATCH_SIZE': 8,
    'IMG_SIZE': 224,
    'NUM_CLASSES': 5,
    'LEARNING_RATE': 1e-4,
    'WEIGHT_DECAY': 1e-5,
    'WARMUP_EPOCHS': 5,
    'GRADIENT_CLIP_NORM': 1.0,
    'LABEL_SMOOTHING': 0.1,
    'LOSS_WEIGHTS': {
        'binary': 0.2,
        'ternary': 0.3,
        'fine': 0.5
    }
}
```

### Preprocessing Parameters

```python
STABLE_PREPROCESS_CONFIG = {
    'kalman_process_variance': 1e-6,  # Reduced from 1e-5
    'kalman_measurement_variance': 1e-2,  # Reduced from 1e-1
    'clip_min': -1.0,
    'clip_max': 1.0,
    'validate_output': True
}
```

## Correctness Properties

_A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees._

### Acceptance Criteria Testing Prework

1.1 WHEN the model trains for multiple epochs THEN the validation loss SHALL remain within 2 orders of magnitude of the training loss
Thoughts: This is a property about the relationship between two metrics across all training runs. We can test this by running training and checking the ratio at each epoch.
Testable: yes - property

1.2 WHEN gradients are computed during backpropagation THEN the system SHALL clip gradient norms to prevent explosion
Thoughts: This is about ensuring gradient clipping is applied. We can test by computing gradients and verifying the norm is below the threshold.
Testable: yes - property

1.3 WHEN loss values are computed THEN the system SHALL validate that loss values are finite before applying gradients
Thoughts: This is about validating all loss computations. We can test by generating random inputs and checking loss outputs.
Testable: yes - property

1.4 WHEN the hierarchical loss combines multiple outputs THEN the system SHALL normalize loss components to similar scales
Thoughts: This is about the relative magnitudes of loss components. We can test by computing losses and checking their ratios.
Testable: yes - property

1.5 WHEN the model processes batches THEN the system SHALL apply batch normalization after convolutional and dense layers
Thoughts: This is about model architecture structure, not runtime behavior.
Testable: no

2.1 WHEN training begins THEN the system SHALL use a warmup period for the learning rate
Thoughts: This is testing that the LR schedule implements warmup. We can check the LR values for the first N steps.
Testable: yes - example

2.2 WHEN validation loss stops improving THEN the system SHALL reduce the learning rate
Thoughts: This is about the behavior of a reduce-on-plateau scheduler. We can simulate plateaus and verify LR reduction.
Testable: yes - property

2.3 WHEN the learning rate is reduced THEN the system SHALL apply a reduction factor between 0.1 and 0.5
Thoughts: This is checking a specific constraint on LR reduction. We can test by triggering reductions and measuring the factor.
Testable: yes - property

2.4 WHEN the learning rate reaches a minimum threshold THEN the system SHALL maintain that minimum rate
Thoughts: This is testing a lower bound constraint. We can test by running many decay steps and checking the minimum.
Testable: yes - property

3.1 WHEN custom layers are created THEN the system SHALL initialize weights using appropriate initialization schemes
Thoughts: This is about checking weight initialization at layer creation time.
Testable: yes - property

3.2 WHEN dense layers are added THEN the system SHALL use He or Glorot initialization
Thoughts: This is checking specific initialization schemes for dense layers.
Testable: yes - example

3.3 WHEN the model is built THEN the system SHALL verify that no layers have zero or NaN initial weights
Thoughts: This is a validation check on the built model.
Testable: yes - property

4.1 WHEN images are preprocessed THEN the system SHALL normalize pixel values to a consistent range
Thoughts: This is about the output range of preprocessing. We can test with random images and check bounds.
Testable: yes - property

4.2 WHEN preprocessing applies filters THEN the system SHALL ensure output values remain bounded
Thoughts: This is about bounds preservation through the pipeline. We can test each filter stage.
Testable: yes - property

4.3 WHEN the Kalman filter is applied THEN the system SHALL prevent numerical overflow in variance calculations
Thoughts: This is about numerical stability in Kalman filtering. We can test with extreme inputs.
Testable: yes - property

4.4 WHEN batches are created THEN the system SHALL verify that no NaN or infinite values exist in the data
Thoughts: This is a validation property on batch data.
Testable: yes - property

5.1 WHEN the model uses dropout layers THEN the system SHALL use standard Dropout instead of stateless variants
Thoughts: This is about model architecture inspection, checking layer types.
Testable: yes - example

5.2 WHEN the model is compiled THEN the system SHALL disable problematic layout optimizations if errors occur
Thoughts: This is about error handling during compilation, which is difficult to test systematically.
Testable: no

5.3 WHEN transformer blocks are created THEN the system SHALL use compatible layer configurations
Thoughts: This is about architectural constraints, checking layer compatibility.
Testable: yes - example

5.4 WHEN the model is built THEN the system SHALL validate the computational graph for known issues
Thoughts: This is a validation check on the model graph.
Testable: yes - example

6.1 WHEN each batch completes THEN the system SHALL log loss and accuracy metrics
Thoughts: This is about logging behavior, which is more of an integration concern than a unit-testable property.
Testable: no

6.2 WHEN gradients are computed THEN the system SHALL monitor gradient norms
Thoughts: This is about monitoring behavior during training.
Testable: yes - property

6.3 WHEN anomalies are detected THEN the system SHALL log warnings with diagnostic information
Thoughts: This is about logging behavior in response to anomalies.
Testable: no

6.4 WHEN training completes an epoch THEN the system SHALL save checkpoint data
Thoughts: This is about file I/O behavior during training.
Testable: no

6.5 WHEN validation metrics are computed THEN the system SHALL compare against training metrics for divergence
Thoughts: This is about computing a ratio and checking a threshold.
Testable: yes - property

7.1 WHEN hierarchical labels are derived THEN the system SHALL correctly map fine-grained labels to coarser levels
Thoughts: This is about label transformation logic. We can test with all possible fine-grained labels.
Testable: yes - property

7.2 WHEN loss components are combined THEN the system SHALL use weights that sum to 1.0
Thoughts: This is a simple arithmetic constraint on loss weights.
Testable: yes - example

7.3 WHEN individual losses are computed THEN the system SHALL apply label smoothing to prevent overconfidence
Thoughts: This is about checking that label smoothing is applied in loss computation.
Testable: yes - property

7.4 WHEN the combined loss is calculated THEN the system SHALL ensure all components are on similar scales
Thoughts: This is about relative magnitudes of loss components.
Testable: yes - property

### Property Reflection

After reviewing all testable properties, I identify the following redundancies:

- **Property 1.4 and 7.4** both test that loss components are on similar scales - these can be combined
- **Property 4.1 and 4.2** both test output bounds - 4.2 is more general and subsumes 4.1
- **Property 1.1 and 6.5** both test the relationship between training and validation metrics - these can be combined

The remaining properties provide unique validation value and should be kept.

### Correctness Properties

**Property 1: Gradient norm clipping**
_For any_ batch of training data and computed gradients, the global gradient norm after clipping should not exceed the specified maximum norm threshold.
**Validates: Requirements 1.2**

**Property 2: Loss finiteness**
_For any_ model inputs and targets, all computed loss values (binary, ternary, fine, and combined) should be finite (not NaN or Inf).
**Validates: Requirements 1.3**

**Property 3: Loss component balance**
_For any_ batch of predictions, the ratio between the largest and smallest loss component should not exceed 100:1, ensuring components are on similar scales.
**Validates: Requirements 1.4, 7.4**

**Property 4: Learning rate warmup**
_For any_ training step during the warmup period, the learning rate should increase monotonically from near-zero to the target learning rate.
**Validates: Requirements 2.1**

**Property 5: Learning rate bounds**
_For any_ training step, the learning rate should remain within the specified minimum and maximum bounds.
**Validates: Requirements 2.4**

**Property 6: Weight initialization validity**
_For any_ newly built model, all layer weights should be finite, non-zero, and follow the expected distribution for their initialization scheme.
**Validates: Requirements 3.1, 3.3**

**Property 7: Preprocessing output bounds**
_For any_ input image, all preprocessing stages should produce outputs within the expected bounds, with no NaN or Inf values.
**Validates: Requirements 4.1, 4.2, 4.4**

**Property 8: Kalman filter stability**
_For any_ sequence of measurements, the Kalman filter variance estimates should remain positive and bounded, preventing numerical overflow.
**Validates: Requirements 4.3**

**Property 9: Training-validation divergence detection**
_For any_ completed epoch, if the validation loss exceeds the training loss by more than a factor of 10, the system should detect this as an anomaly.
**Validates: Requirements 1.1, 6.5**

**Property 10: Hierarchical label mapping correctness**
_For any_ fine-grained label in {0, 1, 2, 3, 4}, the derived binary and ternary labels should follow the correct mapping rules.
**Validates: Requirements 7.1**

**Property 11: Label smoothing application**
_For any_ one-hot encoded label with label smoothing factor ε, the maximum probability should be (1 - ε + ε/K) and minimum should be ε/K, where K is the number of classes.
**Validates: Requirements 7.3**

## Error Handling

### Gradient Explosion Detection

```python
if global_norm > 100.0:
    logger.warning(f"Very large gradient norm detected: {global_norm:.2f}")
    # Gradients are clipped, but log for monitoring
```

### Loss Anomaly Detection

```python
if tf.math.is_nan(loss) or tf.math.is_inf(loss):
    logger.error("NaN or Inf loss detected - skipping batch")
    return None  # Skip this batch
```

### Preprocessing Failure Handling

```python
try:
    preprocessed = advanced_preprocess(img_path)
    if preprocessed is None or not is_valid(preprocessed):
        logger.warning(f"Preprocessing failed for {img_path}")
        return None  # Skip this image
except Exception as e:
    logger.error(f"Preprocessing error: {e}")
    return None
```

## Testing Strategy

### Unit Testing

Unit tests will cover:

- Gradient clipping function with various norm values
- Learning rate schedule computation at different steps
- Hierarchical label mapping for all 5 classes
- Preprocessing bounds checking
- Loss weight validation

### Property-Based Testing

We will use **Hypothesis** (Python PBT library) for property-based testing. Each property-based test will run a minimum of 100 iterations.

Property-based tests will verify:

1. Gradient clipping maintains norm bounds across random gradient tensors
2. Loss computations remain finite for random valid inputs
3. Loss components stay balanced across random predictions
4. Learning rate stays within bounds across all training steps
5. Weight initialization produces valid distributions
6. Preprocessing maintains bounds for random images
7. Kalman filter remains stable for random measurement sequences
8. Hierarchical labels map correctly for all possible inputs
9. Label smoothing produces correct probability distributions

Each property-based test will be tagged with:
**Feature: koa-model-fix, Property {number}: {property_text}**

### Integration Testing

Integration tests will:

- Train for 3 epochs on a small subset and verify no explosions
- Check that all monitoring hooks are called
- Verify checkpoint saving works correctly
- Test end-to-end preprocessing pipeline

### Test Execution

Tests should be run:

- Before committing fixes
- After each major component is implemented
- As part of CI/CD pipeline

Minimum test coverage target: 80% for new/modified code.
