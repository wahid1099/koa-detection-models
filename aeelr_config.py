# ==========================================================
# AEELR CONFIGURATION
# Attention-Enhanced EfficientNet with Label Refinement
# ==========================================================

import os

class AEELRConfig:
    """Configuration for AEELR training and evaluation"""
    
    # ==================== PATHS ====================
    WORK_DIR = './'
    DATASET_PATHS = {
        'train': '/kaggle/input/koa-dataset/dataset/train',
        'val': '/kaggle/input/koa-dataset/dataset/val',
        'test': '/kaggle/input/koa-dataset/dataset/test'
    }
    
    # Preprocessed data directories
    PREP_TRAIN_DIR = os.path.join(WORK_DIR, 'prep_train_aeelr')
    PREP_VAL_DIR = os.path.join(WORK_DIR, 'prep_val_aeelr')
    PREP_TEST_DIR = os.path.join(WORK_DIR, 'prep_test_aeelr')
    
    # Output directories
    CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints_aeelr')
    RESULTS_DIR = os.path.join(WORK_DIR, 'results_aeelr')
    FIGURES_DIR = os.path.join(WORK_DIR, 'figures_aeelr')
    
    # ==================== DATA ====================
    IMG_SIZE = (456, 456)  # EfficientNetB5 optimal size
    NUM_CLASSES = 5
    CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']
    
    # Hierarchical mappings
    BINARY_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}  # Healthy vs OA
    TERNARY_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}  # Mild/Moderate/Severe
    
    # ==================== PREPROCESSING ====================
    # Gaussian denoising
    GAUSSIAN_SIGMA_MIN = 0.8
    GAUSSIAN_SIGMA_MAX = 1.2
    GAUSSIAN_SIGMA = 1.0  # Default
    
    # Laplacian edge enhancement
    LAPLACIAN_WEIGHT = 0.3
    
    # CLAHE parameters
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Normalization (ImageNet stats for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # ==================== AUGMENTATION ====================
    # Geometric augmentation
    ROTATION_RANGE = 7  # ±7 degrees
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True  # Clinically acceptable for bilateral views
    
    # Photometric augmentation
    BRIGHTNESS_RANGE = [0.9, 1.1]
    CONTRAST_RANGE = [0.9, 1.1]
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    OVERSAMPLE_MINORITY = True
    
    # ==================== MODEL ARCHITECTURE ====================
    # Backbone
    BACKBONE = 'EfficientNetB5'
    PRETRAINED_WEIGHTS = 'imagenet'
    FREEZE_LAYERS = 300  # Freeze first N layers initially
    
    # CBAM attention
    CBAM_REDUCTION = 16
    CBAM_KERNEL_SIZE = 7
    
    # Classification head
    DENSE_UNITS = 256
    DROPOUT_RATE_1 = 0.5
    DROPOUT_RATE_2 = 0.3
    
    # Hierarchical head
    USE_HIERARCHICAL = False
    HIERARCHICAL_WEIGHTS = {
        'binary': 0.2,
        'ternary': 0.3,
        'kl': 0.5
    }
    
    # ==================== TRAINING ====================
    # Optimizer
    LEARNING_RATE = 1e-4
    OPTIMIZER = 'adam'
    
    # Learning rate schedule
    LR_SCHEDULE = 'cosine'  # 'cosine' or 'reduce_on_plateau'
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 5
    LR_MIN = 1e-7
    
    # Training phases
    WARMUP_EPOCHS = 5  # Frozen backbone
    FINETUNE_EPOCHS = 20  # Unfrozen backbone
    TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS
    
    BATCH_SIZE = 16  # Adjust based on GPU memory
    
    # Regularization
    EARLY_STOPPING_PATIENCE = 7
    USE_MIXUP = False  # Optional
    MIXUP_ALPHA = 0.2
    USE_LABEL_SMOOTHING = False  # Optional
    LABEL_SMOOTHING = 0.05
    
    # ==================== CLEANLAB ====================
    USE_CLEANLAB = True
    CLEANLAB_RELABEL_TOP_PERCENT = 10  # Relabel top 5-10%
    CLEANLAB_DOWNWEIGHT_PERCENT = 15  # Down-weight next 10-15%
    CLEANLAB_MIN_CONFIDENCE = 0.3
    
    # ==================== CALIBRATION ====================
    USE_TEMPERATURE_SCALING = True
    TEMPERATURE_INIT = 1.0
    TEMPERATURE_MAX_ITER = 50
    
    # Calibration metrics
    ECE_BINS = 10
    TARGET_ECE = 0.05  # Target Expected Calibration Error
    
    # ==================== EXPLAINABILITY ====================
    # Grad-CAM settings
    GRADCAM_LAYER = 'top_activation'  # Last conv layer
    GRADCAM_SAMPLES_PER_CLASS = 5
    
    # Eigen-CAM settings
    USE_EIGENCAM = True
    
    # Sanity checks
    RUN_SANITY_CHECKS = True
    
    # ==================== EVALUATION ====================
    # Cross-validation
    N_FOLDS = 5
    CV_RANDOM_SEED = 42
    
    # Metrics
    METRICS = [
        'accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'auc_macro',
        'quadratic_weighted_kappa',
        'ece',
        'mce'
    ]
    
    # Statistical testing
    USE_STATISTICAL_TESTS = True
    SIGNIFICANCE_LEVEL = 0.05
    
    # ==================== ABLATION STUDIES ====================
    RUN_ABLATIONS = True
    ABLATION_EXPERIMENTS = [
        'baseline_vs_cbam',
        'before_vs_after_cleanlab',
        'single_vs_hierarchical',
        'augmentation_sensitivity'
    ]
    
    # Optional CNN-ViT fusion
    RUN_CNN_VIT_FUSION = False  # Time-boxed, only if GPU permits
    
    # ==================== REPRODUCIBILITY ====================
    RANDOM_SEED = 42
    TF_DETERMINISTIC = True
    
    # ==================== DEPLOYMENT ====================
    # Gradio demo
    DEMO_PORT = 7860
    DEMO_SHARE = False
    
    # Model export
    EXPORT_FORMAT = ['h5', 'savedmodel']
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        dirs = [
            cls.PREP_TRAIN_DIR,
            cls.PREP_VAL_DIR,
            cls.PREP_TEST_DIR,
            cls.CHECKPOINT_DIR,
            cls.RESULTS_DIR,
            cls.FIGURES_DIR
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"✅ Created {len(dirs)} directories")
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("AEELR CONFIGURATION")
        print("="*70)
        print(f"  Image Size: {cls.IMG_SIZE}")
        print(f"  Backbone: {cls.BACKBONE}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Total Epochs: {cls.TOTAL_EPOCHS} (Warmup: {cls.WARMUP_EPOCHS}, Finetune: {cls.FINETUNE_EPOCHS})")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Use Hierarchical: {cls.USE_HIERARCHICAL}")
        print(f"  Use CleanLab: {cls.USE_CLEANLAB}")
        print(f"  Use Temperature Scaling: {cls.USE_TEMPERATURE_SCALING}")
        print(f"  N-Fold CV: {cls.N_FOLDS}")
        print(f"  Random Seed: {cls.RANDOM_SEED}")
        print("="*70 + "\n")
    
    @classmethod
    def set_random_seeds(cls):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        import tensorflow as tf
        
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        tf.random.set_seed(cls.RANDOM_SEED)
        
        if cls.TF_DETERMINISTIC:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['PYTHONHASHSEED'] = str(cls.RANDOM_SEED)
        
        print(f"✅ Random seeds set to {cls.RANDOM_SEED}")


# Convenience alias
CFG = AEELRConfig

if __name__ == "__main__":
    # Test configuration
    CFG.print_config()
    CFG.create_directories()
    CFG.set_random_seeds()
