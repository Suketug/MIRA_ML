import pandas as pd
import matplotlib.pyplot as plt

# Function to load your data
def load_data(path):
    return pd.read_excel(path)

# Function to analyze dimensions
def analyze_dimensions(df, name):
    print(f"Shape of {name}: {df.shape}")

# Function to analyze missing values
def analyze_missing_values(df, name):
    print(f"Missing values in {name}: {df.isnull().sum().sum()}")

# Function to analyze class distribution in Y datasets
def analyze_class_distribution(Y, title):
    plt.figure(figsize=(12, 8))
    Y.value_counts().plot(kind='bar')
    plt.title(title)
    plt.show()

# Function to analyze basic statistics in X datasets
def analyze_feature_statistics(X, name):
    print(f"Basic statistics for {name}:")
    print(X.describe())

# Paths to your datasets
X_train_path = "Data/Extracted_Data/Training_Data/X_train_dataset_reduced.xlsx"
Y_train_path = "Data/Extracted_Data/Training_Data/Y_train_dataset_reduced.xlsx"
X_val_path = "Data/Extracted_Data/Training_Data/X_val_dataset_reduced.xlsx"
Y_val_path = "Data/Extracted_Data/Training_Data/Y_val_dataset_reduced.xlsx"
X_test_path = "Data/Extracted_Data/Training_Data/X_test_dataset_reduced.xlsx"
Y_test_path = "Data/Extracted_Data/Training_Data/Y_test_dataset_reduced.xlsx"

# Load the data
X_train = load_data(X_train_path)
Y_train = load_data(Y_train_path)
X_val = load_data(X_val_path)
Y_val = load_data(Y_val_path)
X_test = load_data(X_test_path)
Y_test = load_data(Y_test_path)

# Analyze dimensions
analyze_dimensions(X_train, 'X_train')
analyze_dimensions(Y_train, 'Y_train')
analyze_dimensions(X_val, 'X_val')
analyze_dimensions(Y_val, 'Y_val')
analyze_dimensions(X_test, 'X_test')
analyze_dimensions(Y_test, 'Y_test')

# Analyze missing values
analyze_missing_values(X_train, 'X_train')
analyze_missing_values(Y_train, 'Y_train')
analyze_missing_values(X_val, 'X_val')
analyze_missing_values(Y_val, 'Y_val')
analyze_missing_values(X_test, 'X_test')
analyze_missing_values(Y_test, 'Y_test')

# Analyze class distribution
analyze_class_distribution(Y_train, 'Training Set')
analyze_class_distribution(Y_val, 'Validation Set')
analyze_class_distribution(Y_test, 'Test Set')

# Analyze feature statistics
analyze_feature_statistics(X_train, 'X_train')
analyze_feature_statistics(X_val, 'X_val')
analyze_feature_statistics(X_test, 'X_test')

# Checking class distribution in the training, validation, and test sets
print("Class distribution in Y_train:\n", Y_train['Intent_encoded'].value_counts())
print("Class distribution in Y_val:\n", Y_val['Intent_encoded'].value_counts())
print("Class distribution in Y_test:\n", Y_test['Intent_encoded'].value_counts())
