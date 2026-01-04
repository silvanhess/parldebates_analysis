#!/usr/bin/env python3
"""
BERT Fine-Tuning Script for Multi-Class Classification (3 classes)
- Optimized for small datasets (500 samples)
- CPU-based training
- Strong regularization to prevent overfitting
- Wandb hyperparameter sweep
- No data augmentation
"""

# ============================================================================
# SECTION 1: IMPORTS & SETUP
# ============================================================================

import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch

import logging
import wandb


# ============================================================================
# SECTION 2: DEVICE SETUP (CPU-Only)
# ============================================================================

device = torch.device("cpu")
print("="*60)
print("🖥️  Using CPU for training")
print(f"PyTorch version: {torch.__version__}")
print(f"Number of CPU threads: {torch.get_num_threads()}")
print("="*60)


# ============================================================================
# SECTION 3: CUSTOM METRIC FUNCTIONS (Multi-Class)
# ============================================================================

def f1_macro(labels, preds):
    """F1 score with macro averaging (equal weight to each class)"""
    return f1_score(labels, preds, average='macro', zero_division=0)

def f1_weighted(labels, preds):
    """F1 score with weighted averaging (accounts for class imbalance)"""
    return f1_score(labels, preds, average='weighted', zero_division=0)

def precision_macro(labels, preds):
    """Precision with macro averaging"""
    return precision_score(labels, preds, average='macro', zero_division=0)

def recall_macro(labels, preds):
    """Recall with macro averaging"""
    return recall_score(labels, preds, average='macro', zero_division=0)

def accuracy(labels, preds):
    """Accuracy score"""
    return accuracy_score(labels, preds)


# ============================================================================
# SECTION 4: DATA LOADING & PREPROCESSING
# ============================================================================

# get current directory
os.getcwd()

print("\n" + "="*60)
print("Loading training data...")
print("="*60)

# Load data
dat = pd.read_csv('training_data.csv')

# Ensure target variable is integer
dat['final_climate'] = dat['final_climate'].astype(int)

# Set paragraph_id as index
dat.set_index("paragraph_id", drop=False, inplace=True, verify_integrity=True)

print(f"Total samples loaded: {len(dat)}")
print(f"Columns: {dat.columns.tolist()}")


# ============================================================================
# SECTION 5: LABEL CREATION & STRATIFICATION
# ============================================================================

# Create numeric labels (0, 1, 2 for 3 classes)
dat["label"] = dat["final_climate"].astype("category").cat.codes

print("\nClass distribution:")
print(dat["label"].value_counts().sort_index())

# Verify we have 3 classes
n_classes = dat["label"].nunique()
assert n_classes == 3, f"Expected 3 classes, found {n_classes}"

# Create stratification variable (language + label)
dat["strata_"] = dat.set_index(['language', 'label']).index.factorize()[0]


# ============================================================================
# SECTION 6: TRAIN/TEST/VALIDATION SPLIT
# ============================================================================

print("\n" + "="*60)
print("Creating train/test/validation splits...")
print("="*60)

# First split: 75% train+val, 25% test
train_ids, test_ids = train_test_split(
    dat.index.values, 
    test_size=0.25, 
    stratify=dat.strata_.values,
    random_state=42
)

# Second split: From train+val, create 70% train, 30% val
train_ids, val_ids = train_test_split(
    train_ids, 
    test_size=0.3, 
    stratify=dat.loc[train_ids].strata_.values,
    random_state=42
)

print(f"Training samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Test samples: {len(test_ids)}")

# Calculate percentages
total = len(dat)
print("\nSplit percentages:")
print(f"  Train: {len(train_ids)/total*100:.1f}%")
print(f"  Val:   {len(val_ids)/total*100:.1f}%")
print(f"  Test:  {len(test_ids)/total*100:.1f}%")


# ============================================================================
# SECTION 7: CREATE DATAFRAMES
# ============================================================================

# Create train DataFrame
train_df = pd.DataFrame(
    zip(
        train_ids,
        dat.loc[train_ids]['original_text'].values,
        dat.loc[train_ids]['label'].values
    ),
    columns=['paragraph_id', 'text', 'labels']
)

# Create test DataFrame
test_df = pd.DataFrame(
    zip(
        test_ids,
        dat.loc[test_ids]['original_text'].values,
        dat.loc[test_ids]['label'].values
    ),
    columns=['paragraph_id', 'text', 'labels']
)

# Create validation DataFrame
val_df = pd.DataFrame(
    zip(
        val_ids,
        dat.loc[val_ids]['original_text'].values,
        dat.loc[val_ids]['label'].values
    ),
    columns=['paragraph_id', 'text', 'labels']
)

# Set paragraph_id as index
train_df.set_index("paragraph_id", drop=False, inplace=True, verify_integrity=True)
test_df.set_index("paragraph_id", drop=False, inplace=True, verify_integrity=True)
val_df.set_index("paragraph_id", drop=False, inplace=True, verify_integrity=True)

print("\nDataFrame shapes:")
print(f"  Train: {train_df.shape}")
print(f"  Test:  {test_df.shape}")
print(f"  Val:   {val_df.shape}")


# ============================================================================
# SECTION 8: LABEL ENCODING
# ============================================================================

# Initialize label encoder
label_encoder = LabelEncoder()

# FIT on training data only
train_df['labels'] = label_encoder.fit_transform(train_df['labels'])

# TRANSFORM (not fit_transform) on test and val
test_df['labels'] = label_encoder.transform(test_df['labels'])
val_df['labels'] = label_encoder.transform(val_df['labels'])

print("\nLabel encoding complete:")
print(f"  Classes: {label_encoder.classes_}")
print(f"  Train label distribution: {train_df['labels'].value_counts().sort_index().to_dict()}")


# ============================================================================
# SECTION 9: COMPUTE CLASS WEIGHTS
# ============================================================================

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['labels']),
    y=train_df['labels']
)

# Convert to PyTorch tensor
weights = torch.tensor(class_weights, dtype=torch.float)

print("\n" + "="*60)
print("Class weights (for handling imbalance):")
for i, w in enumerate(weights):
    print(f"  Class {i}: {w:.4f}")
print("="*60)


# ============================================================================
# SECTION 10: SAVE PREPROCESSED DATA
# ============================================================================

# Save to CSV for later use (optional)
train_df.to_csv('train_ft.csv', index=False)
test_df.to_csv('test_ft.csv', index=False)
val_df.to_csv('val_ft.csv', index=False)

print("\n✅ Preprocessed data saved to train_ft.csv, test_ft.csv, val_ft.csv")


# ============================================================================
# SECTION 11: WANDB HYPERPARAMETER SWEEP CONFIGURATION
# ============================================================================

sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {
        "name": "f1_macro_eval",
        "goal": "maximize"
    },
    "parameters": {
        # Training parameters
        "num_train_epochs": {"values": [2, 3]},
        "train_batch_size": {"values": [4, 8]},
        "learning_rate": {"min": 1e-5, "max": 5e-4},
        
        # STRONGER REGULARIZATION for small dataset (500 samples)
        "weight_decay": {"min": 0.05, "max": 0.2},  # Increased from 0.0-0.15
        "hidden_dropout_prob": {"min": 0.2, "max": 0.4},  # Increased from 0.1-0.3
        "attention_probs_dropout_prob": {"min": 0.2, "max": 0.4},  # Increased
        
        # Class weights toggle
        "use_class_weights": {"values": [0, 1]},
    },
}

# Create sweep (login to wandb first: wandb login)
sweep_id = wandb.sweep(sweep_config, project="roberta_multiclass_500samples")

print("\n" + "="*60)
print(f"Wandb sweep created: {sweep_id}")
print("="*60)


# ============================================================================
# SECTION 12: LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# ============================================================================
# SECTION 13: MODEL CONFIGURATION
# ============================================================================

model_type = "xlmroberta"
model_name = "xlm-roberta-base"

# Model arguments
model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.manual_seed = 42
model_args.num_train_epochs = 2
model_args.use_multiprocessing = False  # Disabled for CPU
model_args.use_multiprocessing_for_evaluation = False
model_args.learning_rate = 1e-5
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
model_args.max_seq_length = 256
model_args.labels_list = [0, 1, 2]
model_args.sliding_window = False  # REMOVED: No data augmentation
model_args.no_save = True
model_args.save_optimizer_and_scheduler = False
model_args.wandb_project = "roberta_multiclass_500samples"
model_args.fp16 = False
model_args.use_early_stopping = True  # Enable early stopping
model_args.early_stopping_patience = 2  # Stop if no improvement for 2 evals
model_args.early_stopping_delta = 0.01  # Minimum improvement threshold
model_args.early_stopping_metric = "f1_macro"
model_args.early_stopping_metric_minimize = False
model_args.weight = weights.tolist()


# ============================================================================
# SECTION 14: TRAINING FUNCTION
# ============================================================================

def train():
    """
    Training function for hyperparameter sweep
    - Creates and trains model with current sweep configuration
    - Evaluates on validation set
    - Logs metrics to wandb
    """
    
    # Initialize wandb run
    wandb.init()
    
    print("\n" + "="*60)
    print(f"Starting training run: {wandb.run.name}")
    print(f"Config: {wandb.config}")
    print("="*60)
    
    # Create classification model
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=3,
        use_cuda=False,  # CPU only
        args=model_args,
        sweep_config=wandb.config,
    )
    
    # Train the model
    print("\n📚 Starting training...")
    start_time = time.time()
    
    model.train_model(
        train_df,
        eval_df=test_df,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision=precision_macro,
        recall=recall_macro
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time/60:.2f} minutes")
    
    # Evaluate on validation set
    print("\n📊 Evaluating on validation set...")
    eval_start = time.time()
    
    result, model_outputs, wrong_predictions = model.eval_model(
        val_df,
        accuracy_eval=accuracy,
        f1_macro_eval=f1_macro,
        f1_weighted_eval=f1_weighted,
        precision_eval=precision_macro,
        recall_eval=recall_macro
    )
    
    eval_time = time.time() - eval_start
    print(f"✅ Evaluation completed in {eval_time:.2f} seconds")
    
    # Get predictions for confusion matrix
    predictions, _ = model.predict(val_df['text'].tolist())
    true_labels = val_df['labels'].values
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate per-class metrics
    print("\n" + "="*60)
    print("Validation Results:")
    print("="*60)
    print(f"Accuracy:        {result['accuracy_eval']:.4f}")
    print(f"F1 (macro):      {result['f1_macro_eval']:.4f}")
    print(f"F1 (weighted):   {result['f1_weighted_eval']:.4f}")
    print(f"Precision:       {result['precision_eval']:.4f}")
    print(f"Recall:          {result['recall_eval']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*60)
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(3):
        if cm[i, :].sum() > 0:
            class_acc = cm[i, i] / cm[i, :].sum()
        else:
            class_acc = 0
        class_accuracies.append(class_acc)
        print(f"Class {i} accuracy: {class_acc:.4f}")
    
    # Log comprehensive metrics to wandb
    wandb.log({
        # Main metrics
        'f1_macro_eval': result['f1_macro_eval'],
        'f1_weighted_eval': result['f1_weighted_eval'],
        'accuracy_eval': result['accuracy_eval'],
        'precision_eval': result['precision_eval'],
        'recall_eval': result['recall_eval'],
        
        # Confusion matrix elements (for 3x3 matrix)
        'cm_00': int(cm[0, 0]),
        'cm_01': int(cm[0, 1]),
        'cm_02': int(cm[0, 2]),
        'cm_10': int(cm[1, 0]),
        'cm_11': int(cm[1, 1]),
        'cm_12': int(cm[1, 2]),
        'cm_20': int(cm[2, 0]),
        'cm_21': int(cm[2, 1]),
        'cm_22': int(cm[2, 2]),
        
        # Per-class accuracy
        'class_0_accuracy': class_accuracies[0],
        'class_1_accuracy': class_accuracies[1],
        'class_2_accuracy': class_accuracies[2],
        
        # Timing
        'training_time_minutes': training_time / 60,
        'eval_time_seconds': eval_time,
        
        # Model info
        'device': 'cpu',
        'total_samples': len(train_df),
    })
    
    # Sync and close wandb run
    wandb.finish()
    
    print(f"\n✅ Run {wandb.run.name} completed successfully!\n")


# ============================================================================
# SECTION 15: RUN HYPERPARAMETER SWEEP
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("🚀 STARTING HYPERPARAMETER SWEEP")
    print("="*60)
    print(f"Sweep ID: {sweep_id}")
    print("Project: roberta_multiclass_500samples")
    print("Device: CPU")
    print(f"Model: {model_name}")
    print("Classes: 3")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"\nRegularization: STRONG (optimized for {len(dat)} samples)")
    print("Data augmentation: DISABLED (no sliding window)")
    print("="*60)
    
    # Ask for confirmation
    response = input("\nStart sweep? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\n🏃 Starting sweep agent...")
        print("The agent will try different combinations of:")
        print("  - Epochs: 2 or 3")
        print("  - Batch size: 4 or 8")
        print("  - Learning rate: 1e-5 to 5e-4")
        print("  - Weight decay: 0.05 to 0.2")
        print("  - Dropout: 0.2 to 0.4")
        print("  - Class weights: enabled or disabled")
        print("\nEstimated time per run: 4-7 minutes")
        print("Total sweep time (10 runs): ~40-60 minutes\n")
        
        # Run the sweep
        # The count parameter controls how many different configurations to try
        wandb.agent(sweep_id, function=train, count=10)
        
        print("\n" + "="*60)
        print("✅ HYPERPARAMETER SWEEP COMPLETED!")
        print("="*60)
        print("View results at: https://wandb.ai")
        print("Check the 'roberta_multiclass_500samples' project")
        print("="*60)
    else:
        print("\n❌ Sweep cancelled by user")


# ============================================================================
# OPTIONAL: SINGLE TRAINING RUN (Without Sweep)
# ============================================================================

def train_single_run():
    """
    Train a single model without hyperparameter sweep
    Useful for testing or final model training with best hyperparameters
    """
    
    print("\n" + "="*60)
    print("🚀 SINGLE TRAINING RUN (No Sweep)")
    print("="*60)
    
    # Initialize wandb without sweep
    wandb.init(
        project="roberta_multiclass_single",
        config={
            "model": model_name,
            "num_train_epochs": 3,
            "train_batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.1,
            "hidden_dropout_prob": 0.3,
            "attention_probs_dropout_prob": 0.3,
            "max_seq_length": 256,
        }
    )
    
    # Update model args with specific values
    model_args.num_train_epochs = 3
    model_args.train_batch_size = 8
    model_args.learning_rate = 2e-5
    model_args.weight_decay = 0.1
    model_args.hidden_dropout_prob = 0.3
    model_args.attention_probs_dropout_prob = 0.3
    
    # Create model
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=3,
        use_cuda=False,
        args=model_args,
    )
    
    # Train
    print("\n📚 Training...")
    start_time = time.time()
    
    model.train_model(
        train_df,
        eval_df=test_df,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision=precision_macro,
        recall=recall_macro
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    print("\n📊 Evaluating...")
    result, _, _ = model.eval_model(
        val_df,
        accuracy_eval=accuracy,
        f1_macro_eval=f1_macro,
        f1_weighted_eval=f1_weighted,
        precision_eval=precision_macro,
        recall_eval=recall_macro
    )
    
    # Get predictions
    predictions, _ = model.predict(val_df['text'].tolist())
    cm = confusion_matrix(val_df['labels'].values, predictions)
    
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    for key, value in result.items():
        print(f"{key}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTraining time: {training_time/60:.2f} minutes")
    print("="*60)
    
    # Save model
    model.model.save_pretrained("./final_model")
    print("\n✅ Model saved to ./final_model")
    
    wandb.finish()
    
    print("\n" + "="*60)
    print("✅ Single training run completed!")
    print("="*60)


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         HOW TO USE THIS SCRIPT                           ║
╚══════════════════════════════════════════════════════════════════════════╝

STEP 1: Install dependencies
    pip install -r requirements.txt

STEP 2: Login to Weights & Biases
    wandb login
    (You'll need a free account at https://wandb.ai)

STEP 3: Prepare your data
    - Place 'training_data.csv' in the same directory
    - Required columns: 'paragraph_id', 'original_text', 'final_climate', 'language'

STEP 4: Run the sweep
    python script.py
    (Type 'yes' when prompted)

STEP 5: Monitor training
    - Open your wandb dashboard at https://wandb.ai
    - Navigate to 'roberta_multiclass_500samples' project
    - Watch real-time metrics and comparisons

STEP 6: Select best model
    - After sweep completes, check wandb for best configuration
    - Use those hyperparameters for final training

OPTIONAL: Run single training (no sweep)
    - Uncomment the last line: train_single_run()
    - Comment out the main sweep execution block

╔══════════════════════════════════════════════════════════════════════════╗
║                         TROUBLESHOOTING                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Problem: "FileNotFoundError: training_data.csv"
Solution: Update the path in Section 4 or move CSV to script directory

Problem: "CUDA out of memory"
Solution: This script uses CPU only, should not happen

Problem: "wandb login failed"
Solution: Get API key from https://wandb.ai/authorize

Problem: Training is too slow
Solution: Reduce max_seq_length to 128 in Section 13
"""

# Uncomment to run single training instead of sweep:
# train_single_run()
