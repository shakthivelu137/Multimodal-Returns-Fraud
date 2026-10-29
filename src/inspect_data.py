import pandas as pd

# Correct path to the dataset
file_path = r"D:\Fraudulent-Return-Detection\data\return_claims_synthetic.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Check fraud distribution
if "is_fraud" in df.columns:
    print("\nFraud Distribution:")
    print(df["is_fraud"].value_counts())
else:
    print("\nColumn 'is_fraud' was not found.")