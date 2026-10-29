import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

print("Original dataset shape:", df.shape)

# 2. Remove completely duplicated rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# 3. Handle missing values in text columns
text_columns = [
    "complaint_text",
    "image_paths"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].fillna("unknown")

# 4. Handle missing numeric values
numeric_columns = [
    "cust_account_age_days",
    "delivery_pincode_risk",
    "cust_return_rate_12m"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].median())

# 5. Check whether missing values remain
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# 6. Save the cleaned dataset
df.to_csv("data/cleaned_returns.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Final dataset shape:", df.shape)