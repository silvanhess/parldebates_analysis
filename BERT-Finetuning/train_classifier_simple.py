# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    confusion_matrix,
    classification_report
)
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import wandb

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

print("Loading training data...")
dat = pd.read_csv('training_data.csv')

# Ensure target variable is binary (0, 1)
dat['final_climate'] = dat['final_climate'].astype(int)
dat["label"] = dat["final_climate"].astype("category").cat.codes

# Prepare dataframe
dat_prepared = dat[['original_text', 'label']].copy()
dat_prepared.columns = ['text', 'labels']

# Create train/test/validation splits
train_df, test_df = train_test_split(
    dat_prepared, 
    test_size=0.25, 
    stratify=dat_prepared['labels'], 
    random_state=42
)
val_df, test_df = train_test_split(
    test_df, 
    test_size=0.5, 
    stratify=test_df['labels'], 
    random_state=42
)

# ============================================================================
# CHECK CLASS DISTRIBUTION
# ============================================================================

print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print("Training set:")
print(train_df['labels'].value_counts())
print(f"\nClass 0: {(train_df['labels']==0).sum()} samples")
print(f"Class 1: {(train_df['labels']==1).sum()} samples")
print(f"Imbalance ratio: {(train_df['labels']==0).sum() / (train_df['labels']==1).sum():.2f}:1")
print("="*60)

# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['labels']),
    y=train_df['labels']
)

weights_list = class_weights.tolist()

print("\n" + "="*60)
print("CLASS WEIGHTS (to handle imbalance)")
print("="*60)
print(f"Class 0 weight: {weights_list[0]:.4f}")
print(f"Class 1 weight: {weights_list[1]:.4f}")
print("="*60)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

model_type = "xlmroberta"
model_name = "xlm-roberta-base"

model_args = ClassificationArgs()
model_args.num_train_epochs = 5  # Increased
model_args.train_batch_size = 8
model_args.eval_batch_size = 8
model_args.max_seq_length = 256
model_args.learning_rate = 3e-5  # Slightly higher
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 100  # Monitor frequently
model_args.evaluate_during_training_verbose = True
model_args.use_multiprocessing = False
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.manual_seed = 42
model_args.weight = weights_list  # ← ADD CLASS WEIGHTS

# Early stopping
model_args.use_early_stopping = True
model_args.early_stopping_patience = 3
model_args.early_stopping_metric = "eval_loss"

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    """Training function for binary classification model"""
    
    wandb.init(project="binary_classification_fixed")
    
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=2,
        use_cuda=False,
        args=model_args,
    )
    
    print("\n" + "="*60)
    print("🚀 Starting training with CLASS WEIGHTS...")
    print("="*60)
    
    model.train_model(train_df, eval_data=val_df)
    
    print("\n" + "="*60)
    print("📊 Evaluating on validation set...")
    print("="*60)
    
    # Validation evaluation
    val_predictions, _ = model.predict(val_df['text'].tolist())
    val_labels = val_df['labels'].values
    
    print("\nVALIDATION RESULTS:")
    print("="*60)
    print(classification_report(val_labels, val_predictions, 
                                target_names=['Class 0', 'Class 1']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, val_predictions))
    
    # Test evaluation
    print("\n" + "="*60)
    print("📊 Evaluating on test set...")
    print("="*60)
    
    test_predictions, _ = model.predict(test_df['text'].tolist())
    test_labels = test_df['labels'].values
    
    print("\nTEST RESULTS:")
    print("="*60)
    print(classification_report(test_labels, test_predictions, 
                                target_names=['Class 0', 'Class 1']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions)
    
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("="*60)
    
    wandb.finish()
    return model

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    trained_model = train()
    print("\n✅ Training completed!")
