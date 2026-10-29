import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset containing CCS scores
df = pd.read_csv("data/returns_with_ccs.csv")

print("=" * 45)
print("CCS SCORE EVALUATION")
print("=" * 45)

# Compare average CCS scores for genuine and fraudulent returns
print("\nAverage CCS score by fraud class:")

print(
    df.groupby("is_fraud")["ccs_score"]
    .agg(["mean", "median", "min", "max", "count"])
    .round(2)
)

# Calculate the correlation between CCS score and fraud label
correlation = df["ccs_score"].corr(df["is_fraud"])

print("\nCorrelation between CCS score and fraud label:")
print(round(correlation, 4))

# Visualise score distributions
plt.figure(figsize=(8, 5))

df.boxplot(
    column="ccs_score",
    by="is_fraud"
)

plt.title("CCS Risk Scores: Genuine vs Fraudulent Returns")
plt.suptitle("")
plt.xlabel("Fraud label (0 = Genuine, 1 = Fraud)")
plt.ylabel("CCS Risk Score (0-100)")
plt.tight_layout()

# Save the chart
plt.savefig("reports/ccs_score_comparison.png")

plt.show()

print("\nChart saved to reports/ccs_score_comparison.png")
print("CCS evaluation completed!")