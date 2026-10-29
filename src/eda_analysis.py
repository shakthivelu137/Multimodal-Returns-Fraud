import pandas as pd

# Load the dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

print("========== DATASET ANALYSIS ==========")

# 1. Dataset size
print("\nDataset Shape:")
print(df.shape)

# 2. Fraud distribution
print("\nFraud Distribution:")
print(df["is_fraud"].value_counts())

# 3. Fraud percentage
print("\nFraud Percentage:")
print(df["is_fraud"].value_counts(normalize=True) * 100)

# 4. Fraud by product category
print("\nFraud by Product Category:")
print(df.groupby("category")["is_fraud"].mean() * 100)

# 5. Fraud by return reason
print("\nFraud by Return Reason:")
print(df.groupby("return_reason_code")["is_fraud"].mean() * 100)

# 6. Fraud by image reuse
print("\nFraud by Image Reuse:")
print(df.groupby("image_reuse_flag")["is_fraud"].mean() * 100)

# 7. Average order value
print("\nAverage Order Value:")
print(df.groupby("is_fraud")["order_value_inr"].mean())

# 8. Average days since delivery
print("\nAverage Days Since Delivery:")
print(df.groupby("is_fraud")["days_since_delivery"].mean())