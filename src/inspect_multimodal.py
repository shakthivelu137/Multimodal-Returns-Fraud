import pandas as pd
import os

# Load the original dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

print("=" * 45)
print("MULTIMODAL DATA INSPECTION")
print("=" * 45)

# 1. Check dataset size
print("\n1. Dataset shape:")
print(df.shape)

# 2. Inspect customer complaints
print("\n2. Customer complaint examples:")

print(
    df["complaint_text"]
    .dropna()
    .head(5)
    .to_string(index=False)
)

# 3. Inspect image paths
print("\n3. Image path examples:")

print(
    df["image_paths"]
    .dropna()
    .head(5)
    .to_string(index=False)
)

# 4. Check whether actual image files exist
print("\n4. Checking actual image files:")

image_paths = df["image_paths"].dropna().head(10)

for path in image_paths:
    print(path, "->", os.path.exists(str(path)))

# 5. Analyse image-text similarity
print("\n5. Image-text similarity:")

print(df["clip_image_text_sim"].describe())

# 6. Analyse product listing similarity
print("\n6. Product listing similarity:")

print(df["clip_image_listing_sim"].describe())

# 7. Analyse claim consistency
print("\n7. Claim consistency score:")

print(df["claim_consistency_score"].describe())

# 8. Check image reuse
print("\n8. Image reuse distribution:")

print(df["image_reuse_flag"].value_counts())

print("\nInspection completed!")