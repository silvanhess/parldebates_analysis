# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split

# import simpletransformers
from simpletransformers.classification import ClassificationModel

# ============================================================================
# CLASSIFICATION WITH POLICLIM
# ============================================================================

# get current directory
os.getcwd()

#change working directory to ./BERT_Finetuning
os.chdir('./BERT_Finetuning')

# Load target data in whatever format preferred.
data = pd.read_csv('training_data.csv')

# sample 100 rows for quick testing
# with seed for reproducibility
data = data.sample(n=100, random_state=42)
print(data.head())

model = ClassificationModel(
    model_type = "xlmroberta", model_name = 'policlim'
    ) # --> this doesn't work

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("marysanford/policlim") # --> this doesn't work
model = AutoModelForSequenceClassification.from_pretrained("marysanford/policlim")

preds,output = model.predict(data['original_text'].tolist())

# ============================================================================
# DATA LOADING & PREPROCESSING
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

# Load data
dat = pd.read_csv('training_data.csv')

# ============================================================================
# LABEL CREATION & STRATIFICATION
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
# TRAIN/TEST/VALIDATION SPLIT
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
# CREATE DATAFRAMES
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
# TRAIN AND EVALUATE MODEL
# ============================================================================

## To use for further fine-tuning
from sklearn.metrics import f1_score, precision, accuracy, recall

# Load training data. Need to have text in 'text' field and corresponding labels in 'labels' field.
new_train = pd.read_csv('your_new_train_data.csv')
new_test = pd.read_csv('your_new_test_data.csv')
new_eval = pd.read_csv('your_new_eval_data.csv')

# Initialize the model with the updated arguments
model = ClassificationModel(
    model_type="xlmroberta", 
    model_name="policlim",  
    num_labels=2,                 # Number of labels for the new task
#    args=model_args,             # Update arguments (labels, hyperparameters, processing details, model evaluation preferences) as necessary
#    weight = weights,            # For class weights   
    ignore_mismatched_sizes=True, # Required if new task has labels other than 2 
    use_cuda=True
)

# Train the model
model.train_model(train_df = new_train, eval_df = new_test,
                  f1_train = f1_score(labels, preds,average=None) # You can also add your own evaluation metrics
                  )

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(val_df,
                                                            f1_eval = f1_score(labels, preds,average=None),
                                                            precision = precision(labels, preds,average=None),
                                                            recall = recall(labels, preds,average=None),
                                                            acc = accuracy_score(labels, preds,average=None)
                                                            )

print('\n\nThese are the results when testing the model on the test data set:\n')
print(result)
