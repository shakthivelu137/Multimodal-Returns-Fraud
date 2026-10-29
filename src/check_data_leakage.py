import pandas as pd

# Load the dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

print("=" * 50)
print("DATA LEAKAGE INVESTIGATION")
print("=" * 50)

# Features to investigate
features = [
    "cust_prev_fraud_flags",
    "claim_consistency_score",
    "clip_image_text_sim",
    "clip_image_listing_sim",
    "vqa_defect_visible",
    "image_reuse_flag"
]

# Calculate correlations
print("\n1. Correlation with fraud label:")

for feature in features:
    correlation = df[feature].corr(df["is_fraud"])
    print(f"{feature}: {correlation:.4f}")

# Compare averages for genuine and fraudulent returns
print("\n2. Average values by fraud class:")

for feature in features:
    print(f"\nFeature: {feature}")
    print(df.groupby("is_fraud")[feature].mean().round(4))

print("\nInvestigation completed!")
