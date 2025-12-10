# Processed Data Directory

This directory contains the cleaned and preprocessed data for the AMR Pattern Recognition project.

## Files Description

### Main Datasets

- **`cleaned_data.csv`**: Full cleaned dataset with all preprocessing steps applied
  - Standardized interpretation values (s, i, r)
  - Encoded resistance features
  - Calculated MAR index
  - Prepared species targets
  - Contains both original and processed columns

### Feature Matrices

Split into 70% train, 20% validation, 10% test with stratification:

- **`X_train.csv`**: Training feature matrix (70% of data)
- **`X_val.csv`**: Validation feature matrix (20% of data)
- **`X_test.csv`**: Test feature matrix (10% of data)

Each contains encoded resistance features (s=0, i=1, r=2) for all antibiotics.

### Target Variables - MAR Index

Binary classification target based on Multiple Antibiotic Resistance index:

- **`y_train_mar.csv`**: Training MAR targets
- **`y_val_mar.csv`**: Validation MAR targets
- **`y_test_mar.csv`**: Test MAR targets

Labels:
- `0`: Low MAR (MAR index ≤ 0.2)
- `1`: High MAR (MAR index > 0.2)

### Target Variables - Species Classification

Multi-class classification target for bacterial species:

- **`y_train_species.csv`**: Training species targets
- **`y_val_species.csv`**: Validation species targets
- **`y_test_species.csv`**: Test species targets

Species with fewer than 10 samples are merged into 'other' category.

### Metadata Files

- **`feature_names.json`**: List of all feature column names
  - Contains names of encoded resistance features
  - Used for feature tracking and model interpretation

- **`encoding_mappings.json`**: Encoding information and parameters
  - Resistance encoding scheme (s=0, i=1, r=2)
  - MAR threshold value (0.2)
  - Species minimum samples threshold (10)
  - Split proportions and random state
  - Number of features and antibiotics

## Data Processing Pipeline

The data was processed using the following steps:

1. **Loading**: Raw data loaded from `data/raw/rawdata.csv`
2. **Cleaning**:
   - Standardized interpretation values (s, i, r)
   - Handled special characters (*, ≥, ≤)
   - Removed rows with all antibiotic values missing
   - Standardized bacterial species names
3. **Encoding**:
   - Ordinal encoding for resistance: s=0, i=1, r=2
   - Missing values in features marked as -1 (not tested)
4. **Target Creation**:
   - MAR index calculated as: resistant_count / tested_count
   - Binary High_MAR target created with threshold 0.2
   - Species target prepared with rare species merged
5. **Splitting**:
   - Stratified split maintaining class distributions
   - 70% training, 20% validation, 10% test

## Usage Example

```python
import pandas as pd
import json

# Load training data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train_mar = pd.read_csv('data/processed/y_train_mar.csv')
y_train_species = pd.read_csv('data/processed/y_train_species.csv')

# Load feature names
with open('data/processed/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Load encoding information
with open('data/processed/encoding_mappings.json', 'r') as f:
    encoding_info = json.load(f)

print(f"Number of features: {len(feature_names)}")
print(f"Training samples: {len(X_train)}")
```

## Notes

- All splits maintain stratification on the target variable
- Feature matrix uses -1 for antibiotics that were not tested
- MAR index threshold of 0.2 (20%) follows common AMR literature standards
- Species with < 10 samples are grouped as 'other' to prevent overfitting on rare classes
- Random state is fixed at 42 for reproducibility

## File Sizes

Typical file sizes (approximate):
- `cleaned_data.csv`: ~500 KB
- Feature matrices (X_*.csv): ~50-150 KB each
- Target files (y_*.csv): ~5-15 KB each
- Metadata files (*.json): < 5 KB each

## Regenerating Processed Data

To regenerate all processed data files, run the Jupyter notebook:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

Or execute it programmatically:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_preprocessing.ipynb
```
