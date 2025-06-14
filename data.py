import numpy as np # Used for numerical operations, especially with arrays
import pandas as pd # Essential for data manipulation and analysis using DataFrames
from sklearn.impute import KNNImputer # For filling in missing numerical values
from sklearn.model_selection import train_test_split # For splitting data into training and validation sets
from sklearn.preprocessing import OneHotEncoder # For converting categorical text data into numerical format

# --- 1. Data Loading ---
# Reads the training and testing datasets from CSV files into Pandas DataFrames.
# 'train.csv' typically contains features and the target variable (SalePrice).
# 'test.csv' contains features for which you want to predict the target.
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# --- 2. Feature and Target Separation ---
# Defines the features (X) by dropping the 'SalePrice' column from the training data.
# 'SalePrice' is the target variable (y) that the model will learn to predict.
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# --- 3. Data Splitting ---
# Splits the 'X' (features) and 'y' (target) data into training and validation sets.
# X_train, y_train: Used to train the machine learning model.
# X_val, y_val: Used to evaluate the model's performance on unseen data during development.
# test_size=0.2: 20% of the data will be used for validation.
# random_state=42: Ensures the split is the same every time you run the code, for reproducibility.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Handling Missing Numerical Values (Imputation) ---
# Separates numerical columns from the training, validation, and test datasets.
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

# Initializes KNNImputer, which fills missing values by looking at the values of
# the 'k' nearest neighbors (default k=5) in the dataset.
imputer = KNNImputer()

# Fits the imputer *only* on the training data's numerical columns (`fit_transform`).
# This learns the patterns for filling missing values from the training data.
# Then, it applies the *same learned patterns* to transform the numerical columns
# of the validation and test sets (`transform`), preventing data leakage.
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# --- 5. Handling Missing Non-Numerical (Categorical) Values ---
# Iterates through each non-numerical (categorical) column.
# Fills any missing values (`NaN`) in these columns with the most frequent value (mode)
# found in that specific column *within the training data*.
# `inplace=True` modifies the DataFrame directly.
for column in non_numeric_cols:
    X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    # X_train[column] = X_train[column].fillna(X_train[column].mode()[0])

    X_val[column].fillna(X_val[column].mode()[0], inplace=True)
    test[column].fillna(test[column].mode()[0], inplace=True)

# --- 6. Encoding Categorical Features ---
# Initializes OneHotEncoder.
# `drop='first'`: Prevents multicollinearity by dropping the first category column
#                  for each feature (e.g., if 'Street' has 'Pave' and 'Grvl', only 'Street_Grvl' is kept).
# `handle_unknown='ignore'`: Crucial for deployment! If new, unseen categories appear in
#                            validation or test data, they will be ignored instead of raising an error.


"""
### Why OneHotEncoding is Required

ML models understand numbers but don’t understand meaning behind them.

### 🚫 Problem Without Encoding

Example:

```python
Feature: "Color" = ['Red', 'Green', 'Blue']
If encoded as: Red=1, Green=2, Blue=3
```

Model thinks Blue > Green > Red, which is wrong (there’s no order in colors).

### ✅ OneHotEncoding Fixes This

```python
Color_Red  Color_Green  Color_Blue  
   1           0            0  
   0           1            0  
   0           0            1  
```

### ✅ Why Needed:

* Converts categorical text → numerical ✅
* Prevents false assumptions of order or distance ❌
* Required by most ML algorithms like LinearRegression, XGBoost, etc.

### ✅ In Short:

> Encoding helps ML models treat categories as separate, not ordered numbers.

"""
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

# Fits the encoder *only* on the training data's categorical columns (`fit_transform`).
# This learns all unique categories present in the training data and creates new columns for them.
# Then, it applies the *same learned encoding* to transform the categorical columns
# of the validation and test sets (`transform`).
# The output of `transform` is a sparse matrix, which is memory-efficient for many zeros.
X_train = ohe.fit_transform(X_train)
X_val = ohe.transform(X_val)
test = ohe.transform(test)

# After this script runs, X_train, X_val, and test are all preprocessed numerical matrices
# (likely sparse matrices), ready to be fed into a machine learning model.


