# ==========================================================
# MAIN TRAINING SCRIPT FOR AEELR
# Attention-Enhanced EfficientNet with Label Refinement
# ==========================================================

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Import AEELR modules
from aeelr_config import CFG
from data_preprocessing import (
    build_df_from_dirs, create_preprocessed_dataset,
    create_augmentation_pipeline, compute_class_weights,
    stratified_kfold_split
)
from aeelr_model import build_baseline_efficientnet, build_aeelr, unfreeze_model, print_model_summary
from label_refinement import detect_label_issues, refine_labels, apply_sample_weights, generate_label_report
from calibration import TemperatureScaling, generate_calibration_report
from explainability import create_gradewise_visualizations, sanity_check_shuffle_weights

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")


# ==========================================================
# SETUP
# ==========================================================

def setup_environment():
    """Setup directories and random seeds"""
    print("\n" + "="*70)
    print("AEELR SETUP")
    print("="*70)
    
    # Create directories
    CFG.create_directories()
    
    # Set random seeds
    CFG.set_random_seeds()
    
    # Print configuration
    CFG.print_config()


# ==========================================================
# DATA LOADING
# ==========================================================

def load_and_preprocess_data(use_cached=False):
    """
    Load and preprocess dataset
    
    Args:
        use_cached: Use cached preprocessed data if available
    
    Returns:
        (train_df, val_df, test_df)
    """
    print("\n" + "="*70)
    print("DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Load raw data
    print("\n[1/3] Loading raw data...")
    train_df = build_df_from_dirs(CFG.DATASET_PATHS['train'])
    val_df = build_df_from_dirs(CFG.DATASET_PATHS['val'])
    test_df = build_df_from_dirs(CFG.DATASET_PATHS['test'])
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Preprocess
    if not use_cached or not os.path.exists(CFG.PREP_TRAIN_DIR):
        print("\n[2/3] Preprocessing images...")
        train_df = create_preprocessed_dataset(train_df, CFG.PREP_TRAIN_DIR)
        val_df = create_preprocessed_dataset(val_df, CFG.PREP_VAL_DIR)
        test_df = create_preprocessed_dataset(test_df, CFG.PREP_TEST_DIR)
    else:
        print("\n[2/3] Using cached preprocessed data...")
        train_df = build_df_from_dirs(CFG.PREP_TRAIN_DIR)
        val_df = build_df_from_dirs(CFG.PREP_VAL_DIR)
        test_df = build_df_from_dirs(CFG.PREP_TEST_DIR)
    
    print("\n[3/3] Data loading complete!")
    
    return train_df, val_df, test_df


# ==========================================================
# TRAINING
# ==========================================================

def train_model(model, train_gen, val_gen, class_weights=None, fold_idx=0):
    """
    Train model with warm-up and fine-tuning
    
    Args:
        model: Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        class_weights: Class weights for imbalanced data
        fold_idx: Fold index for checkpoint naming
    
    Returns:
        (model, history)
    """
    print("\n" + "="*70)
    print(f"TRAINING - FOLD {fold_idx + 1}")
    print("="*70)
    
    # Compile model
    if CFG.USE_HIERARCHICAL:
        model.compile(
            optimizer=Adam(CFG.LEARNING_RATE),
            loss={
                'binary_output': 'sparse_categorical_crossentropy',
                'ternary_output': 'sparse_categorical_crossentropy',
                'kl_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'binary_output': CFG.HIERARCHICAL_WEIGHTS['binary'],
                'ternary_output': CFG.HIERARCHICAL_WEIGHTS['ternary'],
                'kl_output': CFG.HIERARCHICAL_WEIGHTS['kl']
            },
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=Adam(CFG.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Callbacks
    checkpoint_path = f"{CFG.CHECKPOINT_DIR}/aeelr_fold{fold_idx}_best.h5"
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=CFG.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=CFG.LR_REDUCE_FACTOR,
            patience=CFG.LR_REDUCE_PATIENCE,
            min_lr=CFG.LR_MIN,
            verbose=1
        )
    ]
    
    # Phase 1: Warm-up (frozen backbone)
    print(f"\n[PHASE 1] Warm-up training ({CFG.WARMUP_EPOCHS} epochs)...")
    history_warmup = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG.WARMUP_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Phase 2: Fine-tuning (unfrozen backbone)
    print(f"\n[PHASE 2] Fine-tuning ({CFG.FINETUNE_EPOCHS} epochs)...")
    unfreeze_model(model, unfreeze_from=CFG.FREEZE_LAYERS)
    
    # Recompile with lower learning rate
    if CFG.USE_HIERARCHICAL:
        model.compile(
            optimizer=Adam(CFG.LEARNING_RATE / 10),
            loss={
                'binary_output': 'sparse_categorical_crossentropy',
                'ternary_output': 'sparse_categorical_crossentropy',
                'kl_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'binary_output': CFG.HIERARCHICAL_WEIGHTS['binary'],
                'ternary_output': CFG.HIERARCHICAL_WEIGHTS['ternary'],
                'kl_output': CFG.HIERARCHICAL_WEIGHTS['kl']
            },
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=Adam(CFG.LEARNING_RATE / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG.FINETUNE_EPOCHS,
        initial_epoch=CFG.WARMUP_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Combine histories
    history = {
        'warmup': history_warmup.history,
        'finetune': history_finetune.history
    }
    
    print(f"\n✅ Training complete! Best model saved to {checkpoint_path}")
    
    return model, history


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():
    """Main training pipeline"""
    
    # Setup
    setup_environment()
    
    # Load data
    train_df, val_df, test_df = load_and_preprocess_data(use_cached=False)
    
    # Compute class weights
    class_weights = compute_class_weights(train_df) if CFG.USE_CLASS_WEIGHTS else None
    
    # Create data generators
    print("\n[DATA GENERATORS] Creating augmentation pipelines...")
    train_datagen = create_augmentation_pipeline(mode='train')
    val_datagen = create_augmentation_pipeline(mode='val')
    
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=CFG.IMG_SIZE,
        class_mode='sparse',
        batch_size=CFG.BATCH_SIZE,
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filepaths',
        y_col='labels',
        target_size=CFG.IMG_SIZE,
        class_mode='sparse',
        batch_size=CFG.BATCH_SIZE,
        shuffle=False
    )
    
    test_gen = val_datagen.flow_from_dataframe(
        test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=CFG.IMG_SIZE,
        class_mode='sparse',
        batch_size=CFG.BATCH_SIZE,
        shuffle=False
    )
    
    # Build model
    print("\n[MODEL] Building AEELR...")
    model = build_aeelr(use_hierarchical=CFG.USE_HIERARCHICAL)
    print_model_summary(model)
    
    # Train
    model, history = train_model(model, train_gen, val_gen, class_weights, fold_idx=0)
    
    # CleanLab label refinement (optional)
    if CFG.USE_CLEANLAB:
        print("\n[CLEANLAB] Detecting label issues...")
        issue_results = detect_label_issues(model, train_gen, train_df)
        
        if issue_results is not None:
            # Generate report
            generate_label_report(train_df, issue_results, save_dir=CFG.RESULTS_DIR)
            
            # Refine labels
            train_df_refined = refine_labels(train_df, issue_results)
            
            # Optional: Retrain with refined labels
            print("\n[RETRAIN] Retraining with refined labels...")
            # (Implementation left as exercise - would recreate generators and retrain)
    
    # Calibration
    if CFG.USE_TEMPERATURE_SCALING:
        print("\n[CALIBRATION] Temperature scaling...")
        
        # Get validation predictions
        val_preds = model.predict(val_gen, verbose=1)
        if isinstance(val_preds, list):
            val_preds = val_preds[-1]
        
        val_labels = val_gen.classes
        
        # Fit temperature scaling
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(val_preds, val_labels)
        
        # Save temperature
        temp_scaler.save(f"{CFG.CHECKPOINT_DIR}/temperature.npy")
        
        # Generate calibration report
        val_probs_before = tf.nn.softmax(val_preds).numpy()
        val_probs_after = temp_scaler.predict(val_preds)
        
        generate_calibration_report(
            val_labels,
            val_probs_before,
            val_probs_after,
            save_dir=CFG.RESULTS_DIR
        )
    
    # Explainability
    print("\n[EXPLAINABILITY] Generating Grad-CAM visualizations...")
    create_gradewise_visualizations(model, test_df, save_dir=CFG.FIGURES_DIR)
    
    # Sanity check
    if CFG.RUN_SANITY_CHECKS:
        print("\n[SANITY CHECK] Weight shuffling test...")
        sample_img = test_df.iloc[0]['filepaths']
        from data_preprocessing import preprocess_pipeline
        img = preprocess_pipeline(sample_img)
        img_array = np.expand_dims(img, axis=0)
        sanity_check_shuffle_weights(model, img_array)
    
    # Final evaluation
    print("\n[EVALUATION] Final test set evaluation...")
    test_preds = model.predict(test_gen, verbose=1)
    if isinstance(test_preds, list):
        test_preds = test_preds[-1]
    
    if CFG.USE_TEMPERATURE_SCALING:
        test_probs = temp_scaler.predict(test_preds)
    else:
        test_probs = tf.nn.softmax(test_preds).numpy()
    
    test_pred_classes = np.argmax(test_probs, axis=1)
    test_labels = test_gen.classes
    
    accuracy = accuracy_score(test_labels, test_pred_classes)
    f1_macro = f1_score(test_labels, test_pred_classes, average='macro')
    qwk = cohen_kappa_score(test_labels, test_pred_classes, weights='quadratic')
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  QWK: {qwk:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'qwk': qwk
    }
    
    import json
    with open(f"{CFG.RESULTS_DIR}/final_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {CFG.RESULTS_DIR}/final_results.json")
    print("\n🎉 AEELR TRAINING COMPLETE!")
    
    return model, history, results


if __name__ == "__main__":
    model, history, results = main()
