import pandas as pd

# Load the original dataset
df = pd.read_csv("data/return_claims_synthetic.csv")

print("CLAIM CONSISTENCY SCORING")
print("=" * 40)

# Check that the required columns exist
required_columns = [
    "claim_consistency_score",
    "clip_image_text_sim",
    "clip_image_listing_sim",
    "image_reuse_flag"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Create a working copy
df["ccs_score"] = (
    (1 - df["claim_consistency_score"]) * 40
    + (1 - df["clip_image_text_sim"]) * 25
    + (1 - df["clip_image_listing_sim"]) * 25
    + df["image_reuse_flag"] * 10
)

# Keep the score between 0 and 100
df["ccs_score"] = df["ccs_score"].clip(0, 100)

# Display example results
print("\nFirst 10 claim scores:")

print(
    df[
        [
            "claim_consistency_score",
            "clip_image_text_sim",
            "clip_image_listing_sim",
            "image_reuse_flag",
            "ccs_score"
        ]
    ].head(10).round(2)
)

# Display the average score
print("\nAverage CCS risk score:")
print(round(df["ccs_score"].mean(), 2))

# Save results
df.to_csv("data/returns_with_ccs.csv", index=False)

print("\nCCS scoring completed!")
print("Results saved to data/returns_with_ccs.csv")