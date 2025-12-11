"""
===========================================
MAIN TRAINING SCRIPT
===========================================
This script orchestrates all parts of the project:
- PART 1: Data loading and preprocessing
- PART 2: DistilBERT fine-tuning
- PART 3: Model comparison (DistilBERT vs DistilRoBERTa vs ALBERT)

Parts 4-5 (Domain adaptation and Error analysis) are not implemented.
See comments at the end for implementation guidance.
"""

import os
import rootutils
# 设置根目录位置(通过自动递归往外查找名字为".project-root"的空文件实现)
root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# 设置工作目录到根目录
os.chdir(root_path)
# 使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from data_processing.data_checker import create_tiny_debug_dataset, get_texts_and_labels
from models.distilBERT import create_distilBERT, train_distilBERT
from models.model_comparison import compare_models
import torch


# ===========================================
# CONFIGURATION
# ===========================================
# Data configuration
USE_FULL_DATASET = False
USE_DEBUG_DATASET = False  # Set to True only for quick testing with 10 samples
SAMPLE_RATIO = 0.1  # Use 10% of data (360K train, 40K test) as per task requirements

# Device configuration - CHANGED: Now configurable
USE_GPU = True  # Set to True to use GPU (if available), False to force CPU
# The code will automatically check if GPU is available and use it if USE_GPU=True
# If GPU is not available, it will fall back to CPU automatically

# Training parameters
BATCH_SIZE = 16  # Can reduce to 8 if GPU memory insufficient
NUM_EPOCHS = 3
MAX_LENGTH = 100  # 100 tokens as per task, or 512 for DistilBERT max
WARMUP_STEPS = 1000

# Data paths
test_data_path = "data/test.ft.txt.bz2"
train_data_path = "data/train.ft.txt.bz2"
output_dir = "output"


# ===========================================
# DEVICE SETUP
# ===========================================
def setup_device():
    """
    Setup and verify device (GPU/CPU).
    Returns device and prints device information.
    """
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n{'='*60}")
        print("GPU DETECTED AND ENABLED")
        print(f"{'='*60}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"{'='*60}\n")
    elif USE_GPU and not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"\n{'='*60}")
        print("WARNING: GPU REQUESTED BUT NOT AVAILABLE")
        print(f"{'='*60}")
        print("USE_GPU is set to True, but CUDA is not available.")
        print("Possible reasons:")
        print("  1. PyTorch was installed without CUDA support")
        print("  2. CUDA drivers are not installed")
        print("  3. GPU is not compatible")
        print("\nFalling back to CPU. Training will be slower.")
        print("See GPU_SETUP_GUIDE.md for instructions to enable GPU.")
        print(f"{'='*60}\n")
    else:
        device = torch.device("cpu")
        print(f"\n{'='*60}")
        print("USING CPU (USE_GPU = False)")
        print(f"{'='*60}\n")

    return device


# ===========================================
# PART 1: DATA LOADING AND PREPROCESSING
# ===========================================
print("\n" + "="*60)
print("PART 1: DATA LOADING AND PREPROCESSING")
print("="*60)

if USE_DEBUG_DATASET:
    print("Using debug dataset for quick testing...")
    train_texts, train_labels = create_tiny_debug_dataset()
    test_texts, test_labels = create_tiny_debug_dataset()  # Reuse for testing
    print(f"Debug dataset: {len(train_texts)} training samples and {len(test_texts)} test samples")
else:
    print("Loading data from files...")

    # Check if data files exist
    if not os.path.exists(train_data_path):
        print(f"\n{'='*60}")
        print("ERROR: Training data file not found!")
        print(f"{'='*60}")
        print(f"Expected file: {train_data_path}")
        print("\nPlease:")
        print("1. Download the dataset from Kaggle:")
        print("   https://www.kaggle.com/bittlingmayer/amazonreviews")
        print("2. Extract train.ft.txt.bz2 and test.ft.txt.bz2")
        print("3. Place them in the data/ directory")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"Training data file not found: {train_data_path}")

    if not os.path.exists(test_data_path):
        print(f"\n{'='*60}")
        print("ERROR: Test data file not found!")
        print(f"{'='*60}")
        print(f"Expected file: {test_data_path}")
        print("\nPlease download from Kaggle (see above)")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    # Check file permissions
    try:
        # Try to open file in read mode to check permissions
        with open(train_data_path, 'rb') as f:
            pass
        print("✓ Training file is readable")
    except PermissionError as e:
        print(f"\n{'='*60}")
        print("PERMISSION ERROR: Cannot read training file")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nPossible solutions:")
        print("1. Close any programs that might be using the file (7-Zip, WinRAR, etc.)")
        print("2. Check file permissions:")
        print("   - Right-click file → Properties → Security")
        print("   - Make sure your user has 'Read' permission")
        print("3. Try running as administrator")
        print("4. Move file to a different location and update path")
        print(f"{'='*60}\n")
        raise

    try:
        with open(test_data_path, 'rb') as f:
            pass
        print("✓ Test file is readable")
    except PermissionError as e:
        print(f"\n{'='*60}")
        print("PERMISSION ERROR: Cannot read test file")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nSee solutions above")
        print(f"{'='*60}\n")
        raise

    if USE_FULL_DATASET:
        print("Loading full dataset (3.6M train, 400K test)...")
        print("WARNING: This will take 12-25 days to train. Consider using 10% sample instead.")
        train_texts, train_labels, train_size = get_texts_and_labels(
            train_data_path, total_lines=3600000,
            sample_ratio=1.0, apply_preprocessing=True
        )
        test_texts, test_labels, test_size = get_texts_and_labels(
            test_data_path, total_lines=400000,
            sample_ratio=1.0, apply_preprocessing=True
        )
    else:
        print(f"Loading {SAMPLE_RATIO*100}% sample of dataset (as per task requirements)...")
        print(f"Expected: ~360,000 training samples, ~40,000 test samples")
        if USE_GPU and torch.cuda.is_available():
            print("Using GPU - Estimated training time: 5-8 hours for all models")
        else:
            print("Using CPU - Estimated training time: 50-60 hours for all models")
        print("This may take a while. You can stop with Ctrl+C if you see no errors.")
        train_texts, train_labels, train_size = get_texts_and_labels(
            train_data_path, total_lines=3600000,
            sample_ratio=SAMPLE_RATIO, apply_preprocessing=True
        )
        test_texts, test_labels, test_size = get_texts_and_labels(
            test_data_path, total_lines=400000,
            sample_ratio=SAMPLE_RATIO, apply_preprocessing=True
        )
    print(f"Loaded {len(train_texts):,} training samples and {len(test_texts):,} test samples")

print("Data preprocessing completed (HTML removal, URL removal, lowercase, special chars)")


# ===========================================
# DEVICE VERIFICATION
# ===========================================
device = setup_device()


# ===========================================
# PART 2: DISTILBERT FINE-TUNING
# ===========================================
print("\n" + "="*60)
print("PART 2: DISTILBERT FINE-TUNING")
print("="*60)

# Create model
print("Creating DistilBERT model...")
model = create_distilBERT()

# Train model
print("Training DistilBERT...")
trained_model = train_distilBERT(
    train_texts, train_labels, model,
    test_texts=test_texts, test_labels=test_labels,
    output_dir=output_dir,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    max_length=MAX_LENGTH,
    warmup_steps=WARMUP_STEPS
)

print("DistilBERT training completed!")


# ===========================================
# PART 3: MODEL COMPARISON
# ===========================================
print("\n" + "="*60)
print("PART 3: MODEL COMPARISON")
print("="*60)
print("Comparing DistilBERT, DistilRoBERTa, and ALBERT-base...")

# Models to compare
models_to_compare = ['distilbert', 'distilroberta', 'albert']

# Run comparison
comparison_results = compare_models(
    models_to_compare,
    train_texts, train_labels,
    test_texts, test_labels,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    max_length=MAX_LENGTH,
    warmup_steps=WARMUP_STEPS,
    output_dir=output_dir
)

print("Model comparison completed!")
print(f"Results saved to {output_dir}/model_comparison.json")
print(f"\nTo visualize results, run: python src/visualize_comparison.py")


# ===========================================
# PARTS 4-5: NOT IMPLEMENTED
# ===========================================
"""
PART 4: DOMAIN ADAPTATION ANALYSIS
==================================
To implement this part, you would need to:

1. Filter dataset by product_category field:
   - Extract categories from original data (if available)
   - Create separate datasets for different categories (e.g., "electronics", "books")
   
2. Train separate models for each domain:
   - Use the same training pipeline as Part 2
   - Train one model per category
   
3. Evaluate cross-domain performance:
   - Test electronics model on books data
   - Test books model on electronics data
   - Compare in-domain vs cross-domain performance
   
4. Analyze results:
   - Calculate performance drop when testing on different domains
   - Identify which domains transfer better
   - Discuss implications for domain-specific fine-tuning

Estimated work: 4-6 hours
- Data filtering by category: 1 hour
- Multi-domain training script: 2 hours
- Cross-domain evaluation: 1-2 hours
- Analysis and reporting: 1 hour


PART 5: ERROR ANALYSIS
======================
To implement this part, you would need to:

1. Collect prediction errors:
   - Run model on test set
   - Identify false positives and false negatives
   - Store incorrect predictions with true labels
   
2. Categorize errors:
   - Semantic ambiguity (e.g., "not bad" = positive)
   - Negation handling (e.g., "not good" = negative)
   - Mixed sentiment (e.g., "good but expensive")
   - Sarcasm/irony
   - Domain-specific terms
   
3. Analyze error patterns:
   - Count errors by category
   - Find common error patterns
   - Identify which types are most common
   
4. Propose improvements:
   - Data augmentation for problematic cases
   - Model architecture changes
   - Post-processing rules
   - Ensemble methods

Estimated work: 3-5 hours
- Error collection script: 1 hour
- Error categorization: 1-2 hours
- Pattern analysis: 1 hour
- Improvement proposals: 1 hour
"""
